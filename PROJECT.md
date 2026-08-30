# Project: Handwriting Recognition AI Platform Azure Cloud Deployment

## Architecture
- **Cloud Provider**: Microsoft Azure (`damians-playground-dev` subscription `bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7`)
- **Region**: `eastus`
- **Resource Group**: `rg-handwriting-ai-playground`
- **Registry**: Azure Container Registry `acrhandwritingai` (`acrhandwritingai.azurecr.io`)
- **Hosting Platform**: Azure Container Apps (ACA) managed environment `cae-handwriting-ai-playground` with Log Analytics `law-handwriting-ai-playground`
- **Inference Backend**: `ca-backend-playground` (FastAPI, PyTorch TrOCR, OpenCV Sauvola binarizer, RxNorm beam rescorer, 2.0 vCPU / 4.0Gi RAM, Port 8000, external HTTPS ingress with CORS enabled)
- **Web Frontend**: `ca-frontend-playground` (Next.js 14 App Router, standalone container, 0.5 vCPU / 1.0Gi RAM, Port 3000, external HTTPS ingress, connected to backend FQDN)
- **Decommissioned**: Vercel temporary deployment artifacts and references (`temporary-sonic-ridge-qzz8f0o.vercel.app`)

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Azure Resource Group Provisioning | Create dedicated RG `rg-handwriting-ai-playground` in `damians-playground-dev` (`eastus`) | M1 | Survey Infra |
| 2 | Azure Container Registry Provisioning | Create Basic ACR `acrhandwritingai` with admin user credentials enabled | M1 | Survey Infra |
| 3 | Container Apps Environment Setup | Create Log Analytics `law-handwriting-ai-playground` and ACA environment `cae-handwriting-ai-playground` | M1 | Survey Infra |
| 4 | Backend Containerization | Build Dockerfile for FastAPI backend with `libgl1`, `libglib2.0-0`, TrOCR, Sauvola, RxNorm vocabularies, `PYTHONPATH=/app` | M2 | Survey Backend |
| 5 | Frontend Containerization | Build multi-stage Dockerfile for Next.js 14 with `output: 'standalone'` and custom security headers | M2 | Survey Frontend |
| 6 | ACR Image Build & Push | Build and push `handwriting-backend:latest` and `handwriting-frontend:latest` using `az acr build` | M2 | Survey Backend/Frontend |
| 7 | Azure Container Apps Deployment | Deploy `ca-backend-playground` and `ca-frontend-playground` with health probes, CORS, and FQDN environment variables | M2 | Survey Infra |
| 8 | Cloud Health & Model Verification | Verify `GET /v1/health` returns HTTP 200, healthy status, active model, and rescorer active on public Azure URL | M3 | Survey Backend |
| 9 | End-to-End Cloud Transcription Verification | Test real handwriting images and multi-page PDFs against Azure endpoints; verify bounding boxes and zero mock fallback | M3 | User Request / Survey |
| 10 | Vercel Teardown & Documentation Cleanup | Remove `frontend/.vercel/`, `frontend/vercel.json`, clean up references to `temporary-sonic-ridge-qzz8f0o.vercel.app`, update `README.md` | M4 | Survey Frontend |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Azure Infrastructure & Registry Provisioning | Provision RG `rg-handwriting-ai-playground`, ACR `acrhandwritingai`, Log Analytics, and ACA Environment `cae-handwriting-ai-playground` in `eastus` | None | DONE |
| 2 | M2: Backend & Frontend Containerization & Azure Deployment | Create Dockerfiles, update Next.js standalone config, build container images via ACR, deploy `ca-backend-playground` and `ca-frontend-playground` with CORS & ingress | M1 | DONE |
| 3 | M3: End-to-End Cloud Transcription Verification | Execute live cloud integration tests with real handwriting notes and multi-page PDFs; verify accuracy, bounding boxes, and zero mock fallback | M2 | DONE |
| 4 | M4: Vercel Teardown & Documentation Cleanup | Purge `.vercel/` and `vercel.json`, update `README.md` and codebase documentation to point exclusively to Azure Container Apps | M2 | DONE |

## Interface Contracts
### Frontend (`ca-frontend-playground`) ↔ Backend (`ca-backend-playground`)
- `BACKEND_URL`: Environment variable pointing to `https://<ca-backend-playground-fqdn>`
- `NEXT_PUBLIC_BACKEND_URL`: Injected public FQDN for client-side API calls
- `GET /v1/health`: Returns `HealthResponse` `{status: "healthy", loaded_models: [...], rescorer_active: true, ...}`
- `POST /v1/recognize`: Multipart form `file=@...` or JSON `{file_base64: "..."}` returning `RecognitionResponse` `{document_id, pages: [{page_number, lines: [{text, bounding_box, words: [...]}]}]}`
- `CORS`: Allowed origins `*`, allowed methods `GET, POST, PUT, DELETE, OPTIONS`, allowed headers `*`

## Code Layout
- `backend/Dockerfile`: Production multi-stage Dockerfile for FastAPI + PyTorch + OpenCV + Sauvola + RxNorm
- `frontend/Dockerfile`: Production multi-stage Dockerfile for Next.js 14 standalone
- `frontend/next.config.mjs`: Next.js config with `output: 'standalone'` and custom security headers
- `scripts/azure/deploy_infra.sh`: Infrastructure provisioning script (RG, ACR, Log Analytics, ACA)
- `scripts/azure/deploy_apps.sh`: ACR build and Container Apps deployment script
- `tests/e2e/test_azure_cloud_transcription.py`: Automated live cloud verification test suite
- `README.md`: Official project documentation referencing Microsoft Azure Container Apps
