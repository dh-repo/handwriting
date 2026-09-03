# Original User Request

## Initial Request — 2026-08-29T21:16:05Z

Deploy the complete end-to-end Handwriting Recognition AI platform to Microsoft Azure under the `damians-playground-dev` subscription, provisioning a new Resource Group, containerized GPU/CPU inference backend (Azure Container Apps / App Service / ACR), Next.js frontend, and decommissioning the temporary Vercel deployment.

Working directory: /Volumes/LaCie/GitHub/handwriting
Integrity mode: development

## Requirements

### R1. Azure Resource Group & Infrastructure Provisioning
- Create a dedicated Azure Resource Group (e.g., `rg-handwriting-ai-playground`) in subscription `damians-playground-dev` (`bc7eb14b-15b4-4425-a17d-9a4d2f5e73c7`) in an optimal region (e.g., `eastus` or `centralus`).
- Provision Azure Container Registry (ACR) and Azure Container Apps (ACA) / App Service with appropriate compute resources for hosting both the high-throughput Python FastAPI inference backend and the Next.js web frontend.
- Configure environment variables and secure service-to-service communication between the frontend and backend containers.

### R2. Backend & Frontend Containerization & Deployment
- Package and containerize the FastAPI inference backend with all PyTorch, Transformers, OpenCV, Sauvola binarization, and RxNorm beam rescorer dependencies.
- Build and containerize the Next.js frontend application with production optimizations.
- Deploy both container images to Azure Container Apps / App Service with automatic ingress, health check probes (`/v1/health`), and CORS headers enabled.

### R3. End-to-End Cloud Transcription Verification
- Execute automated end-to-end integration tests hitting the deployed Azure backend and frontend endpoints with real handwriting images and multi-page PDFs.
- Verify that transcription results match expected text, return accurate line/word bounding boxes, and execute without falling back to mock generators.

### R4. Vercel Deployment Teardown & Cleanup
- Remove temporary deployment configurations and unclaim/clean up the temporary Vercel project deployment artifacts (`temporary-sonic-ridge-qzz8f0o.vercel.app`).
- Update deployment documentation and configuration files to point exclusively to the production Azure infrastructure.

## Acceptance Criteria

### Azure Infrastructure & Deployment
- [ ] Azure CLI provisioning script creates the resource group, container registry, and container apps in `damians-playground-dev` without errors.
- [ ] Backend container health probe `GET /v1/health` returns HTTP 200 with active model status on the public Azure URL.
- [ ] Frontend Next.js container serves the interactive web application over HTTPS on the Azure endpoint.

### Recognition Quality & Integration
- [ ] End-to-end test uploads a real handwritten note to the Azure endpoint and successfully receives structured line/word bounding boxes and transcriptions.
- [ ] Zero fallback mock text is emitted on real user file uploads.

### Decommissioning
- [ ] Vercel deployment artifacts and temporary configurations are cleanly purged.

## Follow-up — 2026-09-03T14:30:47Z

Full multi-agent team (parallelize backend API, rescorer dynamic tuning, LoRA training, and frontend Darkroom integration).

Build and prove out an end-to-end self-tuning and active learning flywheel for the handwriting recognition platform locally on Apple Silicon (Mac Studio), integrating closed-loop Darkroom operator feedback, dynamic confusion matrix adaptation, and automated background LoRA micro-epochs with replay preservation.

Working directory: /Volumes/LaCie/GitHub/handwriting
Branch: feature/self-tuning-htr-flywheel
Integrity mode: development

## Requirements

### R1. Closed-Loop Operator Feedback Ingestion & Manifest Storage
The platform must capture operator corrections from the Darkroom UI (line and word edits) and persist them via a backend feedback API endpoint (`POST /v1/feedback`). Each record must store the line crop, original model prediction, operator correction, line confidence, document identifier, and timestamp in an append-only feedback manifest.

### R2. Online Self-Tuning Visual Confusion Matrix
The system must automatically update substitution and ligature costs in the visual confusion matrix using character-level dynamic programming alignment between model predictions and verified corrections. Updated confusion costs must immediately influence candidate ranking in the multi-objective beam rescorer without requiring model retraining.

### R3. Automated Background LoRA Adaptation with Experience Replay & Safety Gate
The system must support automated background fine-tuning of LoRA adapters on Apple Silicon (using PyTorch MPS). To prevent catastrophic forgetting, training batches must combine accumulated feedback samples with an anchor replay set from existing golden data. Completed adapters must pass validation checks against character error rate (CER) regression and critical clinical safety rules (zero dangerous drug substitutions) before being activated.

### R4. Frontend Darkroom Feedback Integration
The Darkroom interface (`InlineEditor`, `SplitCurtain`, and speed review queue) must dispatch verified corrections to the feedback endpoint with appropriate debouncing, optimistic UI updates, and visual confirmation indicators.

### R5. Local Proof on Apple Silicon (Mac Studio)
All feedback ingestion, confusion matrix recalibration, rescoring, and LoRA training workflows must be fully functional and verifiable locally on Apple Silicon using the local virtual environment (`.venv`).

## Verification Resources
- Existing test suites: `backend/tests/`, `pipeline/tests/`, `frontend/src/__tests__/`.
- Reference evaluation scripts and metrics in `pipeline/evaluation/metrics.py` and `backend/app/ship_gate.py`.
- Execution command for Python test suite: `.venv/bin/pytest`.

## Acceptance Criteria

### Backend Feedback & Rescorer Recalibration
- [ ] `POST /v1/feedback` validates incoming line/word correction payloads and appends them to `data/feedback/manifest.jsonl`.
- [ ] DP character alignment extracts optical confusion pairs from corrections and dynamically updates confusion matrix costs.
- [ ] Rescorer produces altered candidate rankings reflecting updated confusion penalties on subsequent inference calls.

### Continuous Adaptation & Safety
- [ ] LoRA micro-tuning script runs successfully on Apple Silicon MPS using the local `.venv`.
- [ ] Experience replay data loader successfully interweaves feedback samples with golden anchor lines.
- [ ] Automated ship gate rejects adapters that degrade benchmark CER or trigger LASA clinical confusions.

### Frontend Integration & Test Coverage
- [ ] Darkroom edits trigger feedback submissions without UI stutter or blocking user interactions.
- [ ] Automated unit and integration tests pass via `.venv/bin/pytest` and `npm test` verifying feedback ingestion, alignment, rescorer updates, and adapter gating.

