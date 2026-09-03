# Project: Handwriting Recognition AI Platform Azure Cloud Deployment

## Architecture
- **Cloud Provider**: Microsoft Azure (`damians-playground-dev` subscription `bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7`)
- **Region**: `eastus2` for all live resources
- **Resource Group**: `rg-handwriting-ai-playground`
- **Registry**: Azure Container Registry `acrhwaiplaye2` (`acrhwaiplaye2.azurecr.io`), admin disabled, managed-identity pulls
- **Hosting Platform**: Azure Container Apps environment `cae-handwriting-ai-playground` with Log Analytics `law-handwriting-ai-e2`
- **Inference Backend**: `ca-backend-playground` (FastAPI, TrOCR, 4.0 vCPU / 8.0Gi, minReplicas=1, **internal** ingress)
- **Web Frontend**: `ca-frontend-playground` (Next.js 14, 0.5 vCPU / 1.0Gi, minReplicas=1, external ingress, `BACKEND_URL` = internal FQDN)
- **IaC**: `infra/main.bicep` is the source of truth; `scripts/azure/*.sh` are thin wrappers
- **Decommissioned**: Vercel temporary deployment; eastus ACR `acrhandwritingai` and LAW `law-handwriting-ai-playground` after cutover

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
| 11 | Feedback Ingestion API | Ingest line/word corrections, crops, and metadata via `POST /v1/feedback` | M5 | User Request / Flywheel |
| 12 | Atomic Manifest Storage | Concurrent-safe append to `data/feedback/manifest.jsonl` with POSIX `fcntl.flock(LOCK_EX)` | M5 | Flywheel Engine |
| 13 | Line Crop Persistence | Base64 PNG decode with PIL header validation to `data/feedback/crops/<id>.png` | M5 | Flywheel Storage |
| 14 | Dynamic DP Character Alignment | Character-level DP alignment extracting optical 1:1, 1:2, 2:1, 2:2 substitutions and ligatures | M6 | Rescorer Engine |
| 15 | Live Confusion Cost Recalibration | Dynamic cost discounting with clinical floor $c_{\text{min}} \ge 0.15$ and immediate in-memory re-ranking | M6 | Rescorer Engine |
| 16 | Experience Replay Sampler | `ReplayBatchSampler` enforcing exact 50:50 ratio of feedback to golden anchor lines | M7 | Training Pipeline |
| 17 | Apple Silicon MPS LoRA Micro-Tuning | PEFT LoRA ($r=16, \alpha=32$, `bf16`) fine-tuning on Apple Silicon Metal Performance Shaders | M7 | Training Pipeline |
| 18 | Clinical LASA Safety Release Gate | Strict CER regression bound ($\le 5\%$) + zero-tolerance audit on 20 bidirectional LASA drug pairs | M7 | Clinical Safety |
| 19 | Canvas Line Crop Extractor | Offscreen HTML5 `<canvas>` line crop utility with coordinate clamping and natural dimensions | M8 | Frontend Darkroom |
| 20 | Debounced Feedback Dispatch | 500ms trailing debounce, immediate Enter / quick-pick dispatch, and 4-state visual sync badges | M8 | Frontend Darkroom |
| 21 | Staging Queue & Camera Scanner | Batch document staging queue (`StagingQueue.tsx`) and camera scanner modal (`CameraScannerModal.tsx`) | M8 | Frontend Darkroom |
| 22 | Azure Blob Storage Feedback Sink | Direct stream of line crops and JSONL manifests to Azure Blob Storage with automatic offline local disk fallback | M9 | Cloud Storage |
| 23 | ONNX Runtime CPU Serving Engine | High-performance TrOCR serving engine on ONNX Runtime (`CPUExecutionProvider`, 4 threads) for 3-4x lower CPU latency | M9 | Cloud Inference |
| 24 | Ephemeral ACA Training Job IaC | Bicep infrastructure for dedicated Azure Storage Account and ephemeral Container Apps Job (`caj-lora-micro-tune`) | M9 | Azure IaC |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Azure Infrastructure & Registry Provisioning | Provision RG `rg-handwriting-ai-playground`, ACR `acrhandwritingai`, Log Analytics, and ACA Environment `cae-handwriting-ai-playground` in `eastus` | None | DONE |
| 2 | M2: Backend & Frontend Containerization & Azure Deployment | Create Dockerfiles, update Next.js standalone config, build container images via ACR, deploy `ca-backend-playground` and `ca-frontend-playground` with CORS & ingress | M1 | DONE |
| 3 | M3: End-to-End Cloud Transcription Verification | Execute live cloud integration tests with real handwriting notes and multi-page PDFs; verify accuracy, bounding boxes, and zero mock fallback | M2 | DONE |
| 4 | M4: Vercel Teardown & Documentation Cleanup | Purge `.vercel/` and `vercel.json`, update `README.md` and codebase documentation to point exclusively to Azure Container Apps | M2 | DONE |
| 5 | M5: Feedback Ingestion & Manifest Storage | Implement `POST /v1/feedback`, POSIX `flock` manifest persistence, and base64 PNG line crop extraction | M2 | DONE |
| 6 | M6: Online Self-Tuning Visual Confusion Matrix | Implement DP character alignment, dynamic cost discounting, and immediate in-memory rescorer rank-flipping | M5 | DONE |
| 7 | M7: Apple Silicon MPS LoRA Adaptation & LASA Gate | Implement `lora_micro_tune.py` on MPS, 50:50 experience replay, and zero-tolerance 20-pair LASA safety gate | M6 | DONE |
| 8 | M8: Darkroom Feedback Integration & Tier 5 Hardening | Connect `InlineEditor.tsx` debounced feedback dispatch, offscreen canvas crop extraction, and 1,665 passing tests | M7 | DONE |
| 9 | M9: Azure Cloud Optimization & Ephemeral Worker Architecture | Azure Blob Storage sink, ONNX Runtime CPU serving engine, Bicep Storage Account & ACA Job IaC | M8 | DONE |

## Interface Contracts
### Frontend (`ca-frontend-playground`) ↔ Backend (`ca-backend-playground`)
- `BACKEND_URL`: Server-only env on the frontend, pointing at the backend **internal** FQDN
- `GET /v1/live`: Process liveness (no engine load). Used by ACA startup/liveness probes
- `GET /v1/health`: Readiness + model status. Publicly reached only via frontend `GET /api/health`
- `POST /v1/recognize`: Synchronous recognition reached via frontend `POST /api/recognize`
- `POST /v1/recognize-stream`: SSE line streaming reached via frontend `POST /api/recognize-stream`
- `POST /v1/feedback`: Ingest operator corrections reached via frontend `POST /api/feedback`
- `GET /v1/feedback/stats`: Aggregated feedback ingestion and confusion statistics

- `backend/app/routes/feedback.py`: Feedback ingestion route with Azure Blob Storage sink and atomic `flock` manifest persistence
- `backend/app/onnx_engine.py`: High-performance ONNX Runtime CPU serving engine with greedy and beam search
- `backend/app/routes/recognize.py`: Recognition and streaming endpoints
- `backend/app/engine.py`: Unified inference engine with adaptive beam search, ONNX/PyTorch polymorphism, VLM fusion, and live rescorer
- `backend/app/ship_gate.py`: Re-exported clinical safety and release gate assertions
- `pipeline/rescorer/confusion_matrix.py`: Visual confusion matrix with DP character alignment and cost adaptation
- `pipeline/rescorer/beam_rescorer.py`: Multi-objective beam rescorer combining optical, lexicon, and confusion penalties
- `pipeline/training/experience_replay.py`: Experience replay dataset and batch sampler (exact 50:50 ratio)
- `pipeline/training/lora_micro_tune.py`: Apple Silicon MPS-accelerated PEFT LoRA fine-tuning engine
- `pipeline/training/ship_gate.py`: Release safety gate enforcing CER bounds and 20 bidirectional LASA pairs
- `infra/main.bicep`: Azure infrastructure source of truth (ACR, LAW, ACA env, storage account, apps, container job)
- `infra/modules/storage-account.bicep`: Azure Storage Account with `feedback-crops` and `feedback-manifests` blob containers
- `infra/modules/container-job.bicep`: Ephemeral Azure Container Apps Job for decoupled background fine-tuning
- `frontend/src/lib/cropUtils.ts`: Offscreen HTML5 canvas line crop extractor
- `frontend/src/components/InlineEditor.tsx`: Darkroom inline editor with debounced feedback dispatch
- `frontend/src/components/StagingQueue.tsx`: Multi-document staging queue for batch workloads
- `frontend/src/components/CameraScannerModal.tsx`: Live camera scanner modal with viewfinder
- `tests/unit/test_azure_storage_feedback.py`: Unit tests for Azure Blob Storage feedback sink and offline fallback
- `tests/unit/test_onnx_engine.py`: Unit tests for ONNX Runtime CPU serving engine and beam decoding
- `tests/e2e/test_flywheel_e2e.py`: 36-scenario closed-loop end-to-end integration test suite
- `tests/e2e/test_flywheel_tiers.py`: 196-scenario 4-tier flywheel test matrix (Tiers 1–4 across all 17 features)
- `tests/test_challenger_m5_adversarial.py`: Tier 5 white-box adversarial challenge test suite
- `TEST_READY.md`: E2E test suite specification, tier breakdown, and certification report
- `README.md`: Official project documentation with architecture, active learning flywheel, and quickstart
