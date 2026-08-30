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
