#!/usr/bin/env bash
# ==============================================================================
# Script: deploy_infra.sh
# Purpose: Provision core Azure infrastructure for Handwriting AI platform
# Resources:
#   - Resource Group: rg-handwriting-ai-playground (eastus)
#   - Azure Container Registry: acrhandwritingai (Basic SKU, Admin Enabled, eastus)
#   - Log Analytics Workspace: law-handwriting-ai-playground (eastus)
#   - Container Apps Environment: cae-handwriting-ai-playground (eastus2)
# Subscription: damians-playground-dev (bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7)
# ==============================================================================

set -euo pipefail

# Configuration with override support
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-handwriting-ai-playground}"
LOCATION="${LOCATION:-eastus}"
ACA_LOCATION="${ACA_LOCATION:-eastus2}"
ACR_NAME="${ACR_NAME:-acrhandwritingai}"
LOG_ANALYTICS_NAME="${LOG_ANALYTICS_NAME:-law-handwriting-ai-playground}"
ACA_ENV_NAME="${ACA_ENV_NAME:-cae-handwriting-ai-playground}"
TAGS="Project=HandwritingAI Environment=Playground ManagedBy=Teamwork"

echo "================================================================="
echo " Handwriting AI - Azure Infrastructure Provisioning"
echo "================================================================="
echo " Subscription ID  : ${SUBSCRIPTION_ID}"
echo " Resource Group   : ${RESOURCE_GROUP}"
echo " Default Location : ${LOCATION}"
echo " ACA Location     : ${ACA_LOCATION}"
echo " ACR Name         : ${ACR_NAME}"
echo " Log Analytics    : ${LOG_ANALYTICS_NAME}"
echo " ACA Environment  : ${ACA_ENV_NAME}"
echo "================================================================="

# 1. Ensure target subscription is active
echo "[1/5] Setting active Azure subscription..."
az account set --subscription "${SUBSCRIPTION_ID}"
CURRENT_SUB=$(az account show --query name -o tsv)
echo "Active subscription set to: ${CURRENT_SUB}"

# 2. Register required resource providers if not already registered
echo "[2/5] Checking resource provider registrations..."
for provider in Microsoft.App Microsoft.ContainerRegistry Microsoft.OperationalInsights; do
  state=$(az provider show --namespace "$provider" --query registrationState -o tsv 2>/dev/null || echo "NotRegistered")
  if [[ "$state" != "Registered" ]]; then
    echo "Registering provider ${provider}..."
    az provider register --namespace "$provider"
  else
    echo "Provider ${provider} is already registered."
  fi
done

# 3. Create Resource Group if it does not exist
echo "[3/5] Checking/Creating Resource Group: ${RESOURCE_GROUP}..."
if az group exists --name "${RESOURCE_GROUP}" | grep -q "true"; then
  echo "Resource Group ${RESOURCE_GROUP} already exists."
else
  az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --tags Project=HandwritingAI Environment=Playground ManagedBy=Teamwork
  echo "Resource Group ${RESOURCE_GROUP} created successfully."
fi
RG_ID=$(az group show --name "${RESOURCE_GROUP}" --query id -o tsv)
echo "Resource Group ID: ${RG_ID}"

# 4. Create Azure Container Registry (ACR) if it does not exist
echo "[4/5] Checking/Creating Azure Container Registry: ${ACR_NAME}..."
if az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  echo "Azure Container Registry ${ACR_NAME} already exists."
else
  az acr create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${ACR_NAME}" \
    --sku Basic \
    --admin-enabled true \
    --location "${LOCATION}" \
    --tags Project=HandwritingAI Environment=Playground ManagedBy=Teamwork
  echo "Azure Container Registry ${ACR_NAME} created successfully."
fi

ACR_ID=$(az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query id -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query loginServer -o tsv)
echo "ACR ID          : ${ACR_ID}"
echo "ACR Login Server: ${ACR_LOGIN_SERVER}"

# 5. Create Log Analytics Workspace and Container Apps Environment
echo "[5/5] Checking/Creating Log Analytics Workspace & Container Apps Environment..."

# Log Analytics Workspace
if az monitor log-analytics workspace show --resource-group "${RESOURCE_GROUP}" --workspace-name "${LOG_ANALYTICS_NAME}" >/dev/null 2>&1; then
  echo "Log Analytics Workspace ${LOG_ANALYTICS_NAME} already exists."
else
  az monitor log-analytics workspace create \
    --resource-group "${RESOURCE_GROUP}" \
    --workspace-name "${LOG_ANALYTICS_NAME}" \
    --location "${LOCATION}" \
    --tags Project=HandwritingAI Environment=Playground ManagedBy=Teamwork
  echo "Log Analytics Workspace ${LOG_ANALYTICS_NAME} created successfully."
fi

LAW_ID=$(az monitor log-analytics workspace show --resource-group "${RESOURCE_GROUP}" --workspace-name "${LOG_ANALYTICS_NAME}" --query id -o tsv)
LAW_CUSTOMER_ID=$(az monitor log-analytics workspace show --resource-group "${RESOURCE_GROUP}" --workspace-name "${LOG_ANALYTICS_NAME}" --query customerId -o tsv)
LAW_SHARED_KEY=$(az monitor log-analytics workspace get-shared-keys --resource-group "${RESOURCE_GROUP}" --workspace-name "${LOG_ANALYTICS_NAME}" --query primarySharedKey -o tsv)

echo "Log Analytics ID          : ${LAW_ID}"
echo "Log Analytics Customer ID : ${LAW_CUSTOMER_ID}"

# Container Apps Managed Environment
if az containerapp env show --name "${ACA_ENV_NAME}" --resource-group "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  echo "Container Apps Environment ${ACA_ENV_NAME} already exists."
else
  az containerapp env create \
    --name "${ACA_ENV_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${ACA_LOCATION}" \
    --logs-workspace-id "${LAW_CUSTOMER_ID}" \
    --logs-workspace-key "${LAW_SHARED_KEY}" \
    --tags Project=HandwritingAI Environment=Playground ManagedBy=Teamwork
  echo "Container Apps Environment ${ACA_ENV_NAME} created successfully."
fi

CAE_ID=$(az containerapp env show --name "${ACA_ENV_NAME}" --resource-group "${RESOURCE_GROUP}" --query id -o tsv)
CAE_DEFAULT_DOMAIN=$(az containerapp env show --name "${ACA_ENV_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.defaultDomain -o tsv)

echo "Container Apps Env ID     : ${CAE_ID}"
echo "Default Ingress Domain    : ${CAE_DEFAULT_DOMAIN}"

echo ""
echo "================================================================="
echo " Azure Infrastructure Provisioning Complete!"
echo "================================================================="
echo " Resource Group       : ${RESOURCE_GROUP}"
echo " ACR Name             : ${ACR_NAME}"
echo " ACR Login Server     : ${ACR_LOGIN_SERVER}"
echo " Log Analytics Name   : ${LOG_ANALYTICS_NAME}"
echo " ACA Environment Name : ${ACA_ENV_NAME}"
echo " Default App Domain   : ${CAE_DEFAULT_DOMAIN}"
echo "================================================================="
