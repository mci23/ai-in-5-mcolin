# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent JSON-backed ADK MemoryService with automatic PII redaction."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.adk.events.event import Event
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types

from app.observability.logging_utils import get_structured_logger, log_structured_event
from app.observability.pii_redactor import redact_pii

logger = get_structured_logger("weather_comparison.memory")


def _extract_text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    texts: list[str] = [part.text for part in content.parts if isinstance(part.text, str)]
    return "\n".join(texts).strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_-]{2,}", text.lower()))


class PersistentJsonMemoryService(BaseMemoryService):
    """Cross-session persistent memory service stored safely on disk as JSON.

    Every stored memory passes through ``redact_pii`` prior to persistence so
    sensitive data is scrubbed before hitting storage.
    """

    def __init__(self, store_path: str | Path) -> None:
        super().__init__()
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _read_store(self) -> list[dict[str, Any]]:
        if not self.store_path.exists():
            return []
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write_store(self, records: list[dict[str, Any]]) -> None:
        tmp_path = self.store_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp_path.replace(self.store_path)

    async def add_session_to_memory(self, session: Session) -> None:
        """Persists text events from a session into durable long-term memory."""
        await self.add_events_to_memory(
            app_name=session.app_name,
            user_id=session.user_id,
            events=session.events,
            session_id=session.id,
        )

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Extracts meaningful conversational text events and saves redacted entries."""
        memories: list[MemoryEntry] = []
        for event in events:
            raw_text = _extract_text(event.content)
            if not raw_text:
                continue
            redacted_text = redact_pii(raw_text)
            memories.append(
                MemoryEntry(
                    id=event.id or str(uuid.uuid4()),
                    author=event.author,
                    timestamp=datetime.fromtimestamp(
                        event.timestamp, tz=UTC
                    ).isoformat()
                    if getattr(event, "timestamp", None)
                    else datetime.now(UTC).isoformat(),
                    content=types.Content(
                        role=getattr(event.content, "role", "user") or "user",
                        parts=[types.Part.from_text(text=redacted_text)],
                    ),
                    custom_metadata=redact_pii(
                        dict(custom_metadata or {}, session_id=session_id or "")
                    ),
                )
            )
        if memories:
            await self.add_memory(
                app_name=app_name,
                user_id=user_id,
                memories=memories,
                custom_metadata=custom_metadata,
            )

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Stores pre-built MemoryEntry objects after scrubbing PII."""
        async with self._lock:
            records = self._read_store()
            existing_ids = {
                (r.get("app_name"), r.get("user_id"), r.get("id")) for r in records
            }
            added_count = 0
            for entry in memories:
                entry_id = entry.id or str(uuid.uuid4())
                key = (app_name, user_id, entry_id)
                if key in existing_ids:
                    continue
                redacted_text = redact_pii(_extract_text(entry.content))
                if not redacted_text:
                    continue
                merged_meta = redact_pii(
                    {**dict(custom_metadata or {}), **dict(entry.custom_metadata or {})}
                )
                records.append(
                    {
                        "id": entry_id,
                        "app_name": app_name,
                        "user_id": user_id,
                        "author": entry.author or "agent",
                        "role": getattr(entry.content, "role", "model") or "model",
                        "text": redacted_text,
                        "timestamp": entry.timestamp or datetime.now(UTC).isoformat(),
                        "custom_metadata": merged_meta,
                    }
                )
                existing_ids.add(key)
                added_count += 1

            if added_count:
                self._write_store(records)
                log_structured_event(
                    logger,
                    "memory_persisted",
                    "Persisted redacted memories to long-term store",
                    app_name=app_name,
                    user_id=user_id,
                    added_count=added_count,
                )

    async def delete_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memory_id: str | None = None,
    ) -> None:
        """Deletes matching memories for a user/app."""
        async with self._lock:
            records = self._read_store()
            filtered = [
                r
                for r in records
                if not (
                    r.get("app_name") == app_name
                    and r.get("user_id") == user_id
                    and (memory_id is None or r.get("id") == memory_id)
                )
            ]
            self._write_store(filtered)

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Searches stored memories for a user using keyword overlap + recency."""
        async with self._lock:
            records = [
                r
                for r in self._read_store()
                if r.get("app_name") == app_name and r.get("user_id") == user_id
            ]

        query_tokens = _tokenize(query)
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, rec in enumerate(records):
            text = rec.get("text", "")
            overlap = len(query_tokens & _tokenize(text)) if query_tokens else 1
            if overlap > 0 or not query_tokens:
                scored.append((overlap, idx, rec))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_entries = [
            MemoryEntry(
                id=rec.get("id"),
                author=rec.get("author"),
                timestamp=rec.get("timestamp"),
                content=types.Content(
                    role=rec.get("role", "model"),
                    parts=[types.Part.from_text(text=rec.get("text", ""))],
                ),
                custom_metadata=rec.get("custom_metadata", {}),
            )
            for _, _, rec in scored[:8]
        ]
        return SearchMemoryResponse(memories=top_entries)
