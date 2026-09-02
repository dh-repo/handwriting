#!/usr/bin/env bash
# Build SHA-tagged images into the eastus2 ACR and deploy both Container Apps via Bicep.
# Requires deploy_infra.sh to have created acrhwaiplaye2 and the ACA environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-handwriting-ai-playground}"
LOCATION="${LOCATION:-eastus2}"
ACR_NAME="${ACR_NAME:-acrhwaiplaye2}"
LOG_ANALYTICS_NAME="${LOG_ANALYTICS_NAME:-law-handwriting-ai-e2}"
ACA_ENV_NAME="${ACA_ENV_NAME:-cae-handwriting-ai-playground}"
BACKEND_APP_NAME="${BACKEND_APP_NAME:-ca-backend-playground}"
FRONTEND_APP_NAME="${FRONTEND_APP_NAME:-ca-frontend-playground}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${REPO_ROOT}" rev-parse --short HEAD)}"
SKIP_BUILD="${SKIP_BUILD:-false}"

echo "================================================================="
echo " Handwriting AI - Azure app deploy"
echo "================================================================="
echo " Subscription : ${SUBSCRIPTION_ID}"
echo " ACR          : ${ACR_NAME}"
echo " Image tag    : ${IMAGE_TAG}"
echo "================================================================="

az account set --subscription "${SUBSCRIPTION_ID}"

ACR_SERVER=$(az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query loginServer -o tsv)
BACKEND_IMAGE="${ACR_SERVER}/handwriting-backend:${IMAGE_TAG}"
FRONTEND_IMAGE="${ACR_SERVER}/handwriting-frontend:${IMAGE_TAG}"

if [[ "${SKIP_BUILD}" != "true" ]]; then
  echo "Building backend image ${BACKEND_IMAGE}..."
  STAGING_DIR="/tmp/handwriting_backend_build_context"
  rm -rf "${STAGING_DIR}"
  mkdir -p "${STAGING_DIR}/backend" "${STAGING_DIR}/pipeline" "${STAGING_DIR}/data/reference_handwriting"

  cp -r "${REPO_ROOT}/backend/app" "${REPO_ROOT}/backend/pyproject.toml" "${REPO_ROOT}/backend/__init__.py" "${STAGING_DIR}/backend/"
  cp -r "${REPO_ROOT}/pipeline/preprocessing" "${REPO_ROOT}/pipeline/rescorer" "${REPO_ROOT}/pipeline/training" "${REPO_ROOT}/pipeline/__init__.py" "${STAGING_DIR}/pipeline/"
  cp -r "${REPO_ROOT}/data/reference_handwriting/vocabularies" "${STAGING_DIR}/data/reference_handwriting/"
  cp "${REPO_ROOT}/README.md" "${STAGING_DIR}/"
  cp "${REPO_ROOT}/backend/Dockerfile" "${STAGING_DIR}/Dockerfile"

  COPYFILE_DISABLE=1 az acr build \
    --registry "${ACR_NAME}" \
    --image "handwriting-backend:${IMAGE_TAG}" \
    --file "${STAGING_DIR}/Dockerfile" \
    "${STAGING_DIR}"
  rm -rf "${STAGING_DIR}"

  echo "Building frontend image ${FRONTEND_IMAGE}..."
  COPYFILE_DISABLE=1 az acr build \
    --registry "${ACR_NAME}" \
    --image "handwriting-frontend:${IMAGE_TAG}" \
    --file "${REPO_ROOT}/frontend/Dockerfile" \
    "${REPO_ROOT}/frontend"
fi

# Identity + AcrPull must exist before the image changes. Bicep cannot
# create the role and pull the new image in the same revision.
ACR_ID=$(az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query id -o tsv)
for app in "${BACKEND_APP_NAME}" "${FRONTEND_APP_NAME}"; do
  echo "Ensuring system identity and AcrPull for ${app}..."
  PRINCIPAL_ID=$(az containerapp identity assign \
    --name "${app}" \
    --resource-group "${RESOURCE_GROUP}" \
    --system-assigned \
    --query principalId -o tsv)
  az role assignment create \
    --assignee-object-id "${PRINCIPAL_ID}" \
    --assignee-principal-type ServicePrincipal \
    --role AcrPull \
    --scope "${ACR_ID}" \
    --output none 2>/dev/null || true
done

echo "Waiting 60s for AcrPull to propagate..."
sleep 60

for app in "${BACKEND_APP_NAME}" "${FRONTEND_APP_NAME}"; do
  echo "Binding ${app} to ${ACR_SERVER} via system identity..."
  az containerapp registry set \
    --name "${app}" \
    --resource-group "${RESOURCE_GROUP}" \
    --server "${ACR_SERVER}" \
    --identity system
done

AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://oai-playground-6ecfomdadeubk.openai.azure.com/}"
AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-4o}"
AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-$(az cognitiveservices account keys list -g rg-ops-copilot-playground -n oai-playground-6ecfomdadeubk --query key1 -o tsv 2>/dev/null || true)}"

az deployment group create \
  --resource-group "${RESOURCE_GROUP}" \
  --template-file "${REPO_ROOT}/infra/main.bicep" \
  --parameters \
    location="${LOCATION}" \
    acrName="${ACR_NAME}" \
    logAnalyticsName="${LOG_ANALYTICS_NAME}" \
    environmentName="${ACA_ENV_NAME}" \
    deployApps=true \
    backendImage="${BACKEND_IMAGE}" \
    frontendImage="${FRONTEND_IMAGE}" \
    azureOpenAiEndpoint="${AZURE_OPENAI_ENDPOINT}" \
    azureOpenAiDeployment="${AZURE_OPENAI_DEPLOYMENT}" \
    azureOpenAiApiKey="${AZURE_OPENAI_API_KEY}"

az acr update --name "${ACR_NAME}" --admin-enabled false >/dev/null

BACKEND_INTERNAL_URL=$(az deployment group show \
  --resource-group "${RESOURCE_GROUP}" \
  --name main \
  --query properties.outputs.backendInternalUrl.value -o tsv 2>/dev/null || true)
if [[ -z "${BACKEND_INTERNAL_URL}" ]]; then
  CAE_DOMAIN=$(az containerapp env show --name "${ACA_ENV_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.defaultDomain -o tsv)
  BACKEND_INTERNAL_URL="https://${BACKEND_APP_NAME}.internal.${CAE_DOMAIN}"
fi

FRONTEND_FQDN=$(az containerapp show --name "${FRONTEND_APP_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.configuration.ingress.fqdn -o tsv)
FRONTEND_URL="https://${FRONTEND_FQDN}"

echo "Verifying frontend ${FRONTEND_URL}..."
for i in {1..36}; do
  if curl -s -f -I "${FRONTEND_URL}" >/dev/null 2>&1; then
    echo "Frontend homepage: HTTP 200"
    break
  fi
  echo "Waiting for frontend (attempt ${i}/36)..."
  sleep 5
done

echo "Verifying frontend health proxy..."
for i in {1..36}; do
  if curl -s -f "${FRONTEND_URL}/api/health" >/tmp/frontend_health.json 2>/dev/null; then
    echo "Frontend /api/health:"
    cat /tmp/frontend_health.json
    echo ""
    break
  fi
  echo "Waiting for backend readiness via proxy (attempt ${i}/36)..."
  sleep 5
done

echo ""
echo "================================================================="
echo " Deploy complete"
echo " Frontend URL        : ${FRONTEND_URL}"
echo " Backend (internal)  : ${BACKEND_INTERNAL_URL}"
echo " Images              : ${BACKEND_IMAGE}"
echo "                       ${FRONTEND_IMAGE}"
echo "================================================================="
