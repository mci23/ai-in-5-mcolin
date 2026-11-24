# Weather Comparison Agent

Multi-agent ADK Python capstone project for the **AI in 5 Days** course. It compares **today's current weather in Mexico City** against **historical weather on a specific day in the past year (past 365 days) in São Paulo and New York**, built to satisfy all 5 categories of the AgentOps Code Review Matrix.

## Architecture Overview

```text
User Query
   │
   ▼
weather_comparison_agent (Coordinator, Pro Model)
   ├── PreloadMemoryTool (Cross-session recall)
   ├── confirm_historical_comparison_date (HITL confirmation hook)
   ├── save_weather_preference_or_summary (Persistent session state)
   └── weather_comparison_pipeline (SequentialAgent)
          ├── 1. parallel_weather_gatherer (ParallelAgent, Flash Model)
          │      ├── mexico_city_current_agent   ──▶ Open-Meteo Forecast API
          │      ├── sao_paulo_historical_agent    ──▶ Open-Meteo Archive API
          │      └── new_york_historical_agent     ──▶ Open-Meteo Archive API
          └── 2. comparison_analyst_agent (Pro Model)
                 └── compute_weather_comparison_metrics (Deterministic deltas)
```

Cross-cutting capabilities attached at the `App` / `Runner` level:
- **Guardrails & Observability Plugin** (`app/plugins/agentops_plugin.py`): Blocks prompt injection, enforces the 365-day historical date policy, emits structured JSON logs capturing **tool intent vs. actual outcome + latency**, and scrubs PII.
- **Distributed Tracing & PII Redaction** (`app/observability/`): OpenTelemetry spans across coordinator, sub-agents, and HTTP API calls (`weather_api.open_meteo.*`), plus regex + optional Cloud DLP redaction.
- **Persistent Context & Async Memory** (`app/app_utils/services.py`, `app/memory/`): SQLite-backed `DatabaseSessionService`, `EventsCompactionConfig` token-based history compaction, `PersistentJsonMemoryService`, and non-blocking `asyncio.create_task` memory consolidation.
- **Secure Secret Management & IaC** (`app/config.py`, `deployment/terraform/`): GCP Secret Manager lookup with `.env` fallback for local prototype mode, Terraform Cloud Run + Secret Manager + Telemetry provisioning, and GitHub Actions CI/CD (`.github/workflows/`).

## Quick Start (Local Prototype Mode)

1. **Install dependencies:**
   ```bash
   uv sync --all-extras --dev
   ```
2. **Configure credentials (choose one):**
   - Copy `.env.example` to `.env`
   - **Google AI Studio:** set `GEMINI_API_KEY=...` (and remove/comment `GOOGLE_GENAI_USE_ENTERPRISE`)
   - **Vertex AI / Gemini Platform:** run `gcloud auth application-default login` and set `GOOGLE_CLOUD_PROJECT=<your-project>`
3. **Run unit tests (no LLM credentials required):**
   ```bash
   uv run pytest tests/unit
   ```
4. **Launch the interactive ADK Playground / Web UI:**
   ```bash
   agents-cli playground
   # or: uv run adk web .
   ```
5. **Run automated evaluations:**
   ```bash
   agents-cli eval run
   ```

## Example Prompts

- `"Compare today's weather in Mexico City against 2026-06-15 in São Paulo and New York."`
- `"Compare today in Mexico City with a day last summer in São Paulo and New York."` *(triggers HITL date confirmation)*
- `"Prefer Fahrenheit from now on and show the comparison again."` *(persists user preference in session state & async memory)*

## Deployment (Cloud Run + Terraform)

```bash
gcloud config set project <your-project-id>
agents-cli infra single-project --apply --project <your-project-id>
agents-cli deploy
```
