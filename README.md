# Private line-level handwriting recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20MPS%20(Metal)-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Checkpoint](https://img.shields.io/badge/Checkpoint-trocr--large--handwritten-blue.svg)](https://huggingface.co/microsoft/trocr-large-handwritten)
[![Spec](https://img.shields.io/badge/Spec-Strategy%20v3-informational.svg)](PROJECT_STRATEGY.md)
[![Inventory](https://img.shields.io/badge/Local%20inventory-403%2C158%20real%20samples-purple.svg)](data/reference_handwriting/dataset_summary.json)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.x%20App%20Router-black.svg?logo=next.js)](https://nextjs.org/)
[![CI](https://img.shields.io/badge/CI-hygiene%20suites-lightgrey.svg)](TEST_READY.md)
[![Security](https://img.shields.io/badge/Export-CWE--1236-success.svg)](tests/e2e/test_tier5_adversarial.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

One pipeline. Three products. Local-first. The clinical gate can say no.

This repo is a private line-level handwritten text recognition (HTR) stack: a 1:1 scan-versus-transcript darkroom, a FastAPI inference daemon, and a typed lexicon after the decoder. The named checkpoint is `microsoft/trocr-large-handwritten` (TrOCR-Large, 558M: BEiT-Large encoder + 12-layer RoBERTa decoder). Official IAM cased CER for that checkpoint is 2.89%. That is not a ceiling, and it is not a prescription model.

**Spec:** [PROJECT_STRATEGY.md](PROJECT_STRATEGY.md) (v3.0.0). If this README and the spec disagree, the spec wins.

**Not claimed:** state-of-the-art doctor handwriting, signature verification, or a prescription corpus. There is no Rx pad in the training inventory. The RxNorm trie is a post-decoder spellchecker until that hole is filled. A silent `Amoxictllin` → `Ampicillin` snap is a patient-safety incident, not an OCR win.

---

## 1. One pipeline, three products

```
textpage
  → printed-text probe
  → layout + reading order
    → line / field crops (aspect preserved, page coords kept)
      → L1 TrOCR-Small/Base
        → L2 TrOCR-Large only on hard lines
          → domain pack (general / historical / Rx)
            → confusion-weighted rescore
              → LASA / low-confidence gate
                → accept | VLM referee (opt-in) | human
```

The cascade is the target architecture. Layout comes first. CLAHE / Hough / projection profiles are a fallback. 384×384 letterboxing is a TrOCR checkpoint adapter, not page geometry.

| Product | What it is | Gate |
| :--- | :--- | :--- |
| **General** | Private notes, letters, unconstrained cursive. Local MPS/MLX. Darkroom UI. | Confidence only |
| **Historical** | Manuscripts and registries (Belfort, POPP, Esposalles). Historical domain pack. | Confidence only |
| **Clinical** | Prescriptions and notes, only after a gate that can refuse. Text only. Not a medical-signature product. PHI default: off-box-never. | LASA / low-confidence → accept, escalate, or refuse |
| **Signatures** | General-purpose marks on letters, contracts, and forms. Candidate list + human review. | No auto-verify. No Dr./MD heuristic. |

Signatures are **general-purpose** (letters, contracts, forms, personal marks). Not physician credentials. HTR may propose sign-off candidates. It does not verify a hand. Medical signatures are not a product.

---

## 2. Data (local inventory, not a writer census)

[data/reference_handwriting/dataset_summary.json](data/reference_handwriting/dataset_summary.json) lists **403,158** downloaded real image+transcription rows from named Hugging Face repos. Sample counts are local inventory. Writer counts are sourced literature / official figures, not `source_key::sample_id` hashes.

| Corpus | Official / literature scale | Local samples |
| :--- | :--- | ---: |
| IAM | 657 writers, 1,539 pages, 13,353 lines | 10,373 lines + 109,971 words + 5,663 sentences + 4,097 writer-labeled words |
| IMGUR5K | 8,177 pages, 230,573 word images; ~5,000 writers in the literature | 206,687 word crops |
| Esposalles (line) | 3,827 (Teklia 2,328 / 742 / 757); 1–2 hands on the common subset | 3,827 |
| RIMES parent | ~1,300 writers | 12,104 lines (RIMES 2011) |
| POPP Generic | 80 writers | 4,794 lines |
| Belfort, German, IFT forms | writer count unknown | 32,699 + 10,844 + 2,099 |

There is **no prescription corpus** in this inventory. IAM is LOB English. IMGUR5K is internet words. RIMES is fictitious administrative letters. Esposalles is 17th-century Catalan marriage licenses.

Policy: **real-first, synthetic-disclosed, real-only eval.** TrOCR itself was pretrained on synthetic print. The older 50,500-sample / 650-writer / `prescription_item` / 3D-augmentation story is retired as a training-corpus claim. If those files still exist on disk, they are synthetic-disclosed fixtures.

---

## 3. Model and lexicon

- **Checkpoint:** `microsoft/trocr-large-handwritten`
- **Encoder:** BEiT-Large (24 layers, 1024, 16 heads)
- **Decoder:** last 12 layers of RoBERTa-Large (`decoder_layers=12`, `vocab_size=50265`)
- **Official beam:** 10
- **384×384:** processor adapter, not “physical normalization”
- **Adaptation:** LoRA on Apple Silicon. This machine is not a pretrain cluster.

On IAM, frontier VLMs (2026) sit near 1.2–1.7% CER. DTrOCR is 2.38%. TrOCR-Large is 2.89%. The honest pitch: we will not beat GPT-5 on a public page. We will beat it on privacy, cost at volume, offline, and fine-tune control — and we will refuse to silently rewrite a drug name.

The rescorer in `pipeline/rescorer/` may rank beam candidates with a typed RxNorm trie and a visual confusion matrix. June 2026-scale RxNorm is on the order of ~17.5k semantic clinical drugs, ~14.6k ingredients, plus brands, packs, and ~248k NDCs. Local JSON under `data/reference_handwriting/vocabularies/` is a development slice, not the FDA catalog. The trie must store a term type. It must not silently substitute a LASA pair.

---

## 4. Runtime split

A 558M encoder-decoder does not run on Vercel serverless. Azure Container Apps at 4 vCPU / 8 Gi is an app-shell and API host, not TrOCR-Large production.

| Runtime | Role | Where |
| :--- | :--- | :--- |
| Next.js app shell | Darkroom UI, review, CWE-1236 export | Azure ACA `ca-frontend-playground`. Vercel allowed for shell only. |
| Local MPS/MLX daemon | Private inference | FastAPI in `backend/app/`, Apple Silicon |
| Optional GPU worker | VLM referee | Dedicated box, opt-in. PHI default: off-box-never. |

Azure deploy inventory: [PROJECT.md](PROJECT.md).

---

## 5. Darkroom UI

The primary UI is a 1:1 curtain between ink and type (`frontend/src/components/SplitCurtain.tsx`). Not a chat box. Not a cinematic HUD.

Also shipped:

- SVG document viewer with pan, zoom, and bounding-box sync
- Line-level inline editor
- Confidence coloring as a review aid, not a calibration proof
- JSON, TXT, and RFC 4180 CSV export with CWE-1236 single-quote escaping of leading `=`, `+`, `-`, `@`, tab, and CR
- `SignatureInspector`: general-purpose sign-off candidates, review required. Not verification. Not medical.

---

## 6. FastAPI

Backend: `backend/app/`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/v1/health` | Service health, device, rescorer flag |
| `POST` | `/v1/recognize` | Synchronous image/PDF recognition |
| `POST` | `/v1/jobs` | Asynchronous multi-page job |
| `GET` | `/v1/jobs/{id}` | Job status |
| `GET` | `/v1/jobs/{id}/events` | SSE progress |
| `GET` | `/v1/jobs/{id}/stream` | SSE alias for the web app |

```bash
curl -X POST "http://localhost:8000/v1/recognize?beam_width=10&rescore=true" \
  -H "accept: application/json" \
  -F "file=@path/to/page.png"
```

OpenAPI: `http://localhost:8000/docs`.

Response shape is a document → pages → lines → words hierarchy with normalized boxes. Example payloads that look like filled prescriptions are **illustrative of the schema**, not evidence the model read a real Rx.

---

## 7. CI hygiene (not an HTR release gate)

Passing unit, component, and E2E suites means the plumbing works. It does not mean CER/WER, calibration, or dangerous-substitution rate were measured. Those are the v3 release gate in the spec.

| Suite | Command |
| :--- | :--- |
| Master E2E | `.venv/bin/python tests/e2e/runner.py --tier all` |
| Root tests | `PYTHONPATH=. .venv/bin/pytest tests/ -v` |
| Backend | `PYTHONPATH=. .venv/bin/pytest backend/tests/ -v` |
| Pipeline | `PYTHONPATH=. .venv/bin/pytest pipeline/tests/ -v` |
| Frontend | `npm test` (in `frontend/`) |
| Frontend lint | `npm run lint` (in `frontend/`) |
| Frontend build | `npm run build` (in `frontend/`) |

A silent hydralazine / hydroxyzine swap must be able to fail a build. That gate is specified, not claimed as shipped.

---

## 8. Quickstart

### Prerequisites

- macOS (Apple Silicon recommended for MPS), Linux (Ubuntu 22.04+), or Windows (WSL2)
- Python 3.10–3.12
- Node.js 18.x, 20.x, or 22+ and npm 9+

### 1. Environment

```bash
git clone https://github.com/your-org/handwriting.git
cd handwriting

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

### 2. Reference inventory

```bash
# Download and curate the real public corpora (local inventory; not a writer census)
.venv/bin/python pipeline/dataset/download_and_curate_multisource.py

# Generate test fixtures
.venv/bin/python tests/fixtures/generator.py --out-dir tests/fixtures
```

### 3. Local inference daemon (Apple Silicon)

```bash
.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

For Metal: `HANDWRITING_DEVICE=mps` (or `DEVICE=mps` per `backend/app/config.py`). This is the private inference runtime. It is not a serverless function.

Optional `launchd` plist pattern (edit paths to this machine):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.handwriting.backend</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Volumes/LaCie/GitHub/handwriting/.venv/bin/uvicorn</string>
        <string>backend.app.main:app</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8000</string>
        <string>--workers</string>
        <string>1</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HANDWRITING_DEVICE</key>
        <string>mps</string>
        <key>PYTORCH_MPS_HIGH_WATERMARK_RATIO</key>
        <string>0.85</string>
    </dict>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.handwriting.backend.plist
```

### 4. App shell

```bash
cd frontend
npm run dev
```

UI: `http://localhost:3000`.

---

## 9. Azure app-shell deploy

The Next.js shell and a backend container are deployed to Azure Container Apps. That is hosting for the shell and an API probe. It is not a claim that TrOCR-Large runs at production quality on 4 vCPU / 8 Gi.

- Subscription / RG / ACR / environment: [PROJECT.md](PROJECT.md)
- IaC: `infra/main.bicep`
- Scripts: `scripts/azure/deploy_infra.sh`, `scripts/azure/deploy_apps.sh`

```bash
./scripts/azure/deploy_infra.sh
./scripts/azure/deploy_apps.sh
```

| Component | URL |
| :--- | :--- |
| Web shell | https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io |
| Health proxy | https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io/api/health |

The FastAPI container is internal to the Container Apps environment. The browser never receives a backend URL.

```bash
curl -s https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io/api/health
curl -s -I https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io
```

Vercel is allowed for the app shell only. Inference never lives there.

---

## 10. Security

- **CWE-1236:** CSV/TSV/Excel export prepends `'` to cells starting with `=`, `+`, `-`, `@`, tab, or CR, then RFC 4180-quotes.
- **CWE-209:** 4xx/5xx handlers return JSON without stack traces or internal paths.
- **Hostile files:** executables and broken images/PDFs are rejected (HTTP 422).
- **PHI:** default is off-box-never. Cloud referee is opt-in.

---

## 11. License

MIT. See [LICENSE](LICENSE).
