#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-dani-lab-507314}"
REGION="${REGION:-europe-west8}"
SERVICE="${SERVICE:-fantamantra}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-fantamantra-sa}"
DATABASE_ID="${FIRESTORE_DATABASE:-(default)}"
APP_PIN="${APP_PIN:-}"

echo "==> Project: ${PROJECT_ID} | Region: ${REGION} | Service: ${SERVICE}"
gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  firestore.googleapis.com

SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" --display-name="FantaMantra Cloud Run"
fi

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/datastore.user" >/dev/null

if ! gcloud firestore databases describe --database="${DATABASE_ID}" >/dev/null 2>&1; then
  echo "==> Creo Firestore ${DATABASE_ID} in ${REGION}"
  gcloud firestore databases create \
    --database="${DATABASE_ID}" \
    --location="${REGION}" \
    --edition=standard \
    --type=firestore-native
else
  echo "==> Firestore ${DATABASE_ID} già presente"
fi

ENV_VARS="GCP_PROJECT_ID=${PROJECT_ID},USE_FIRESTORE=1,FIRESTORE_DATABASE=${DATABASE_ID}"
if [[ -n "${APP_PIN}" ]]; then
  ENV_VARS="${ENV_VARS},APP_PIN=${APP_PIN}"
fi

echo "==> Deploy Cloud Run"
gcloud run deploy "${SERVICE}" \
  --source . \
  --region "${REGION}" \
  --service-account "${SA_EMAIL}" \
  --allow-unauthenticated \
  --min 0 \
  --max 1 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 3600 \
  --set-env-vars "${ENV_VARS}"

echo "==> Fatto. URL:"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)'
