# Azure Deployment Plan

> **Status:** Cut over and proven

Generated: 2026-08-29

---

## 1. Project Overview

**Goal:** Harden the handwriting playground: Bicep source of truth, colocated eastus2 ACR + Log Analytics, internal backend ingress, managed identity pulls, SHA image tags, health probes, HTTP scale rules. Keep minReplicas=1 on both apps (no cold starts).

**Path:** Modernize Existing

---

## 2. Requirements

| Attribute | Value |
|-----------|-------|
| Classification | Development / Playground |
| Scale | Small |
| Budget | Balanced (warm replicas, no scale-to-zero) |
| **Subscription** | damians-playground-dev `bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7` |
| **Location** | `eastus2` (apps already here; new ACR + LAW colocated) |

---

## 3. Components Detected

| Component | Type | Technology | Path |
|-----------|------|------------|------|
| Backend | API | FastAPI + TrOCR | `backend/` |
| Frontend | Frontend | Next.js 14 | `frontend/` |

---

## 4. Recipe Selection

**Selected:** Bicep + thin AZCLI wrappers

**Rationale:** Existing workflow is `scripts/azure/*.sh`. Bicep owns resource shape; scripts deploy and build images. No azd (user did not request a new product surface).

---

## 5. Architecture

**Stack:** Containers (Azure Container Apps, Consumption)

### Service Mapping

| Component | Azure Service | SKU |
|-----------|---------------|-----|
| Registry | Azure Container Registry `acrhwaiplaye2` | Basic, admin disabled |
| Logs | Log Analytics `law-handwriting-ai-e2` | PerGB2018 |
| Environment | Container Apps Environment `cae-handwriting-ai-playground` | Consumption |
| Backend | Container App `ca-backend-playground` | 4 vCPU / 8 Gi, min 1, max 3, internal ingress |
| Frontend | Container App `ca-frontend-playground` | 0.5 vCPU / 1 Gi, min 1, max 2, external ingress |

### Supporting Services

| Service | Purpose |
|---------|---------|
| Log Analytics | Container Apps logs (eastus2) |
| Managed Identity | AcrPull, no admin password |
| Frontend `/api/recognize` + `/api/health` | Only public path to inference |

---

## 6. Provisioning Limit Checklist

| Resource Type | Number to Deploy | Total After Deployment | Limit/Quota | Notes |
|---------------|------------------|------------------------|-------------|-------|
| Microsoft.App/managedEnvironments | 0 new (update existing) | 1 | 20 | Fetched from azure-quotas: ManagedEnvironmentCount used 1 |
| Microsoft.App/containerApps | 0 new (update existing) | 2 | 20 env-scoped | Same ManagedEnvironmentCount family; no new env |
| Microsoft.ContainerRegistry/registries | 1 new | 2 until old eastus ACR retired | No Limit | Fetched from azure-quotas |
| Microsoft.OperationalInsights/workspaces | 1 new | 2 until old eastus LAW retired | No Limit | Fetched from azure-quotas |

**Status:** ✅ All resources within limits (update-in-place plus one ACR and one LAW)

---

## 7. Execution Checklist

### Phase 1: Planning
- [x] Analyze workspace
- [x] Gather requirements
- [x] Confirm subscription and location with user
- [x] Prepare resource inventory
- [x] Fetch quotas and validate capacity
- [x] Scan codebase
- [x] Select recipe
- [x] Plan architecture
- [x] **User approved this plan** (option C + no cold starts)

### Phase 2: Execution
- [x] Generate Bicep
- [x] Rewrite deploy scripts
- [x] Add `/v1/live` and frontend health proxy
- [x] Drop browser-exposed backend URL
- [x] Update e2e + docs
- [x] Update plan status to Ready for Validation

---

## 7. Validation Proof

| Check | Command Run | Result | Timestamp |
|-------|-------------|--------|-----------|
| Auth | `az account show` | ✅ damians-playground-dev `bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7` Enabled | 2026-08-29 21:16 |
| Bicep compiles | `az bicep build --file infra/main.bicep` | ✅ BUILD_OK (BCP081 type warnings only) | 2026-08-29 21:16 |
| Template valid | `az deployment group validate` deployApps=false | ✅ provisioningState Succeeded, error null | 2026-08-29 21:16 |
| What-if | `az deployment group what-if` deployApps=false | ✅ Create acrhwaiplaye2 + law-handwriting-ai-e2; Modify env logs; Ignore existing apps | 2026-08-29 21:15 |

**Validated by:** azure-validate (Bicep recipe)
**Validation timestamp:** 2026-08-29T21:16:00-04:00

---

## 8. Files to Generate

| File | Purpose | Status |
|------|---------|--------|
| `.azure/deployment-plan.md` | This plan | ✅ |
| `infra/main.bicep` | Infrastructure | ✅ |
| `infra/modules/*.bicep` | ACR, LAW, env, apps | ✅ |
| `scripts/azure/deploy_infra.sh` | `az deployment group create` | ✅ |
| `scripts/azure/deploy_apps.sh` | SHA build + identity-before-image | ✅ |

---

## 9. Cutover Proof

Proven 2026-08-30 after deleting `acrhandwritingai` and `law-handwriting-ai-playground`.

| Check | Result |
|-------|--------|
| Frontend `GET /` | HTTP 200 |
| Frontend `GET /api/health` | healthy, TrOCR loaded, rescorer active |
| Frontend `POST /api/recognize` | HTTP 200, `X-Recognition-Provider: backend` |
| Public backend `/v1/health` and `/v1/recognize` | HTTP 404 Azure Container App Unavailable |
| Backend image | `acrhwaiplaye2.azurecr.io/handwriting-backend:5ea04ba`, internal ingress, system MI |
| Frontend image | `acrhwaiplaye2.azurecr.io/handwriting-frontend:5ea04ba`, external ingress, system MI |
| RG inventory | only eastus2: env, 2 apps, `acrhwaiplaye2`, `law-handwriting-ai-e2` |

Live frontend: https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io
