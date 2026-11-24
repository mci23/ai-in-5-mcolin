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

# Provision Secret Manager secrets and IAM bindings for secure runtime secret injection.

resource "google_project_service" "secretmanager" {
  project            = var.project_id
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false

  depends_on = [google_project_service.bootstrap]
}

resource "google_secret_manager_secret" "gemini_api_key" {
  project   = var.project_id
  secret_id = "gemini-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret" "weather_api_key" {
  project   = var.project_id
  secret_id = "weather-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_iam_member" "agent_identity_gemini_secret_access" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.agent_identity_principal
}

resource "google_secret_manager_secret_iam_member" "agent_identity_weather_secret_access" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.weather_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.agent_identity_principal
}
