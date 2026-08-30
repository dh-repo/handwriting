#!/usr/bin/env bash
# Provision colocated eastus2 foundation via Bicep: ACR, Log Analytics, ACA environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-handwriting-ai-playground}"
LOCATION="${LOCATION:-eastus2}"
ACR_NAME="${ACR_NAME:-acrhwaiplaye2}"
LOG_ANALYTICS_NAME="${LOG_ANALYTICS_NAME:-law-handwriting-ai-e2}"
ACA_ENV_NAME="${ACA_ENV_NAME:-cae-handwriting-ai-playground}"

echo "================================================================="
echo " Handwriting AI - Azure infrastructure (Bicep)"
echo "================================================================="
echo " Subscription : ${SUBSCRIPTION_ID}"
echo " Resource Group: ${RESOURCE_GROUP}"
echo " Location     : ${LOCATION}"
echo " ACR          : ${ACR_NAME}"
echo " Log Analytics: ${LOG_ANALYTICS_NAME}"
echo " ACA Env      : ${ACA_ENV_NAME}"
echo "================================================================="

az account set --subscription "${SUBSCRIPTION_ID}"

if [[ "$(az group exists --name "${RESOURCE_GROUP}")" != "true" ]]; then
  az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --tags Project=HandwritingAI Environment=Playground ManagedBy=Bicep
fi

az deployment group create \
  --resource-group "${RESOURCE_GROUP}" \
  --template-file "${REPO_ROOT}/infra/main.bicep" \
  --parameters \
    location="${LOCATION}" \
    acrName="${ACR_NAME}" \
    logAnalyticsName="${LOG_ANALYTICS_NAME}" \
    environmentName="${ACA_ENV_NAME}" \
    deployApps=false

ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query loginServer -o tsv)

echo ""
echo "================================================================="
echo " Foundation ready. Next: ./scripts/azure/deploy_apps.sh"
echo " ACR login server: ${ACR_LOGIN_SERVER}"
echo "================================================================="
