#!/bin/bash
# ── Deploy churn-api to Google Cloud Run ──────────────────────────────
#
# Prerequisites:
#   1. Google Cloud SDK installed:  https://cloud.google.com/sdk/install
#   2. Authenticated:               gcloud auth login
#   3. Model trained locally:       uv run kedro run
#   4. Set required env vars:
#        export GCP_PROJECT_ID=your-project-id
#        export GCS_BUCKET=your-bucket-name
#
# Usage:
#   chmod +x deploy.sh && ./deploy.sh
#
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${GCS_BUCKET:?Set GCS_BUCKET}"
IMAGE="gcr.io/${PROJECT_ID}/churn-api:latest"

echo "══════════════════════════════════════════════════════════"
echo "  Deploying churn-api to Cloud Run"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Bucket  : gs://${BUCKET}"
echo "══════════════════════════════════════════════════════════"

echo ""
echo "1/4  Creating GCS bucket (if needed)…"
gsutil ls -b "gs://${BUCKET}" 2>/dev/null \
    || gsutil mb -l "${REGION}" -p "${PROJECT_ID}" "gs://${BUCKET}"

echo ""
echo "2/4  Uploading production artifacts to GCS…"
gsutil -m cp \
    data/06_models/production_encoders.pkl \
    data/06_models/production_scalers.pkl \
    data/06_models/production_model.pkl \
    "gs://${BUCKET}/churn/models/"

echo ""
echo "3/4  Building container image with Cloud Build…"
gcloud builds submit \
    --tag "${IMAGE}" \
    --project "${PROJECT_ID}"

echo ""
echo "4/4  Deploying to Cloud Run…"
gcloud run deploy churn-api \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --set-env-vars "KEDRO_ENV=cloud,GCS_BUCKET=${BUCKET},API_KEY=${API_KEY:?Set API_KEY}" \
    --allow-unauthenticated \
    --memory 1Gi \
    --project "${PROJECT_ID}"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo ""
SERVICE_URL=$(gcloud run services describe churn-api \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format='value(status.url)')
echo "  API URL  : ${SERVICE_URL}"
echo "  Swagger  : ${SERVICE_URL}/docs"
echo "  Health   : ${SERVICE_URL}/health"
echo "══════════════════════════════════════════════════════════"
