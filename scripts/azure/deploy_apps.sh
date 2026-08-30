#!/usr/bin/env bash
# ==============================================================================
# Script: deploy_apps.sh
# Purpose: Build container images via Azure Container Registry (ACR) and deploy
#          FastAPI Backend and Next.js Frontend to Azure Container Apps (ACA).
# Resources:
#   - Resource Group: rg-handwriting-ai-playground (eastus)
#   - ACR: acrhandwritingai (acrhandwritingai.azurecr.io)
#   - ACA Environment: cae-handwriting-ai-playground
#   - Backend Container App: ca-backend-playground (Port 8000, 2.0 vCPU, 4.0Gi RAM)
#   - Frontend Container App: ca-frontend-playground (Port 3000, 0.5 vCPU, 1.0Gi RAM)
# Subscription: damians-playground-dev (bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration with override support
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-handwriting-ai-playground}"
ACR_NAME="${ACR_NAME:-acrhandwritingai}"
ACA_ENV_NAME="${ACA_ENV_NAME:-cae-handwriting-ai-playground}"
BACKEND_APP_NAME="${BACKEND_APP_NAME:-ca-backend-playground}"
FRONTEND_APP_NAME="${FRONTEND_APP_NAME:-ca-frontend-playground}"
BACKEND_IMAGE_TAG="${BACKEND_IMAGE_TAG:-latest}"
FRONTEND_IMAGE_TAG="${FRONTEND_IMAGE_TAG:-latest}"
SKIP_BUILD="${SKIP_BUILD:-false}"

echo "================================================================="
echo " Handwriting AI - Application Container Build & Azure Deployment"
echo "================================================================="
echo " Subscription ID       : ${SUBSCRIPTION_ID}"
echo " Resource Group        : ${RESOURCE_GROUP}"
echo " ACR Name              : ${ACR_NAME}"
echo " ACA Environment       : ${ACA_ENV_NAME}"
echo " Backend Container App : ${BACKEND_APP_NAME}"
echo " Frontend Container App: ${FRONTEND_APP_NAME}"
echo " Repo Root             : ${REPO_ROOT}"
echo "================================================================="

# 1. Set Active Subscription
echo "[1/6] Activating Azure subscription..."
az account set --subscription "${SUBSCRIPTION_ID}"
echo "Active subscription: $(az account show --query name -o tsv)"

# 2. Build and Push Backend Container Image via ACR
if [[ "${SKIP_BUILD}" != "true" ]]; then
  echo "[2/6] Building Backend container image in ACR (${ACR_NAME})..."
  STAGING_DIR="/tmp/handwriting_backend_build_context"
  rm -rf "${STAGING_DIR}"
  mkdir -p "${STAGING_DIR}/backend" "${STAGING_DIR}/pipeline" "${STAGING_DIR}/data/reference_handwriting"

  cp -r "${REPO_ROOT}/backend/app" "${REPO_ROOT}/backend/pyproject.toml" "${REPO_ROOT}/backend/__init__.py" "${STAGING_DIR}/backend/"
  cp -r "${REPO_ROOT}/pipeline/preprocessing" "${REPO_ROOT}/pipeline/rescorer" "${REPO_ROOT}/pipeline/__init__.py" "${STAGING_DIR}/pipeline/"
  cp -r "${REPO_ROOT}/data/reference_handwriting/vocabularies" "${STAGING_DIR}/data/reference_handwriting/"
  cp "${REPO_ROOT}/README.md" "${STAGING_DIR}/"
  cp "${REPO_ROOT}/backend/Dockerfile" "${STAGING_DIR}/Dockerfile"

  COPYFILE_DISABLE=1 az acr build \
    --registry "${ACR_NAME}" \
    --image "handwriting-backend:${BACKEND_IMAGE_TAG}" \
    --file "${STAGING_DIR}/Dockerfile" \
    "${STAGING_DIR}"
  rm -rf "${STAGING_DIR}"

  # 3. Build and Push Frontend Container Image via ACR
  echo "[3/6] Building Frontend container image in ACR (${ACR_NAME})..."
  COPYFILE_DISABLE=1 az acr build \
    --registry "${ACR_NAME}" \
    --image "handwriting-frontend:${FRONTEND_IMAGE_TAG}" \
    --file "${REPO_ROOT}/frontend/Dockerfile" \
    "${REPO_ROOT}/frontend"
else
  echo "[2/6 & 3/6] Skipping image build step (SKIP_BUILD=true)..."
fi

# Retrieve ACR credentials & server
ACR_SERVER=$(az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query loginServer -o tsv)
ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query "passwords[0].value" -o tsv)

# 4. Deploy Backend Container App
echo "[4/6] Deploying Backend Container App (${BACKEND_APP_NAME})..."
BACKEND_IMAGE="${ACR_SERVER}/handwriting-backend:${BACKEND_IMAGE_TAG}"

if az containerapp show --name "${BACKEND_APP_NAME}" --resource-group "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  REV_SUFFIX="r$(date +%s)"
  echo "Updating existing Container App ${BACKEND_APP_NAME} (revision suffix: ${REV_SUFFIX})..."
  az containerapp update \
    --name "${BACKEND_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --image "${BACKEND_IMAGE}" \
    --revision-suffix "${REV_SUFFIX}" \
    --set-env-vars DEVICE=cpu USE_MOCK_ENGINE=false VOCAB_DIR=/app/data/reference_handwriting/vocabularies
else
  echo "Creating new Container App ${BACKEND_APP_NAME}..."
  az containerapp create \
    --name "${BACKEND_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --environment "${ACA_ENV_NAME}" \
    --image "${BACKEND_IMAGE}" \
    --registry-server "${ACR_SERVER}" \
    --registry-username "${ACR_NAME}" \
    --registry-password "${ACR_PASSWORD}" \
    --target-port 8000 \
    --ingress external \
    --cpu 2.0 \
    --memory 4.0Gi \
    --min-replicas 1 \
    --max-replicas 3 \
    --env-vars DEVICE=cpu USE_MOCK_ENGINE=false VOCAB_DIR=/app/data/reference_handwriting/vocabularies \
    --tags Project=HandwritingAI Tier=Backend ManagedBy=Teamwork
fi

# Configure CORS on Backend Ingress
echo "Configuring CORS on ${BACKEND_APP_NAME}..."
az containerapp ingress cors enable \
  --name "${BACKEND_APP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --allowed-origins "*" \
  --allowed-methods "GET" "POST" "PUT" "DELETE" "OPTIONS" \
  --allowed-headers "*" \
  --allow-credentials false

# Retrieve Backend FQDN
BACKEND_FQDN=$(az containerapp show --name "${BACKEND_APP_NAME}" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" -o tsv)
BACKEND_URL="https://${BACKEND_FQDN}"
echo "Backend URL: ${BACKEND_URL}"

# 5. Deploy Frontend Container App
echo "[5/6] Deploying Frontend Container App (${FRONTEND_APP_NAME})..."
FRONTEND_IMAGE="${ACR_SERVER}/handwriting-frontend:${FRONTEND_IMAGE_TAG}"

if az containerapp show --name "${FRONTEND_APP_NAME}" --resource-group "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  FE_REV_SUFFIX="r$(date +%s)"
  echo "Updating existing Container App ${FRONTEND_APP_NAME} (revision suffix: ${FE_REV_SUFFIX})..."
  az containerapp update \
    --name "${FRONTEND_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --image "${FRONTEND_IMAGE}" \
    --revision-suffix "${FE_REV_SUFFIX}" \
    --set-env-vars BACKEND_URL="${BACKEND_URL}" NEXT_PUBLIC_BACKEND_URL="${BACKEND_URL}"
else
  echo "Creating new Container App ${FRONTEND_APP_NAME}..."
  az containerapp create \
    --name "${FRONTEND_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --environment "${ACA_ENV_NAME}" \
    --image "${FRONTEND_IMAGE}" \
    --registry-server "${ACR_SERVER}" \
    --registry-username "${ACR_NAME}" \
    --registry-password "${ACR_PASSWORD}" \
    --target-port 3000 \
    --ingress external \
    --cpu 0.5 \
    --memory 1.0Gi \
    --min-replicas 1 \
    --max-replicas 2 \
    --env-vars BACKEND_URL="${BACKEND_URL}" NEXT_PUBLIC_BACKEND_URL="${BACKEND_URL}" \
    --tags Project=HandwritingAI Tier=Frontend ManagedBy=Teamwork
fi

FRONTEND_FQDN=$(az containerapp show --name "${FRONTEND_APP_NAME}" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" -o tsv)
FRONTEND_URL="https://${FRONTEND_FQDN}"
echo "Frontend URL: ${FRONTEND_URL}"

# 6. Verify Health of Endpoints
echo "[6/6] Verifying live endpoints..."
echo "Checking Backend Health (${BACKEND_URL}/v1/health)..."
for i in {1..24}; do
  if curl -s -f "${BACKEND_URL}/v1/health" >/tmp/backend_health.json 2>/dev/null; then
    echo "Backend Health Probe: HTTP 200 OK"
    cat /tmp/backend_health.json
    echo ""
    break
  else
    echo "Waiting for backend container startup (attempt ${i}/24)..."
    sleep 5
  fi
done

echo "Checking Frontend Homepage (${FRONTEND_URL})..."
for i in {1..24}; do
  if curl -s -f -I "${FRONTEND_URL}" >/tmp/frontend_health.txt 2>/dev/null; then
    echo "Frontend Health Probe: HTTP 200 OK"
    head -n 5 /tmp/frontend_health.txt
    break
  else
    echo "Waiting for frontend container startup (attempt ${i}/24)..."
    sleep 5
  fi
done

echo ""
echo "================================================================="
echo " Azure Application Deployment Complete!"
echo "================================================================="
echo " Backend URL  : ${BACKEND_URL}"
echo " Frontend URL : ${FRONTEND_URL}"
echo "================================================================="
