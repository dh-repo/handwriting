# Test Infrastructure: Self-Tuning HTR Flywheel (Features F1–F17)

## 1. Executive Summary & Testing Philosophy

The **Self-Tuning and Active Learning Flywheel** establishes a closed-loop learning architecture that connects human operator corrections in the Darkroom UI directly to the handwriting recognition inference and fine-tuning pipelines locally on Apple Silicon (Mac Studio).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Frontend: Darkroom UI (Next.js)                       │
│  InlineEditor (500ms debounce / Enter / Blur / 1-5 QuickPick)                │
│  SplitCurtain (X-Ray Overlay) & DarkroomToolbar (Filters)                   │
│  Canvas Line-Crop Extractor (extractLineCropBase64 / computeCropCoordinates)│
│  Visual Confirmation Badges (debouncing -> syncing -> synced -> error)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ POST /api/feedback (Route Handler Proxy)
                                       │ POST /v1/feedback (FastAPI)
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                    Backend: Feedback Router & Ingestion                     │
│  backend/app/routes/feedback.py & schemas.py (FeedbackCorrectionPayload)     │
│  Atomic append to data/feedback/manifest.jsonl via fcntl.flock              │
│  Image crop storage to data/feedback/crops/<feedback_id>.png                │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
       (Online Recalibration: <1ms)      (Background Asynchronous Batch)
                       │                               │
┌──────────────────────▼───────────────┐ ┌─────────────▼──────────────────────┐
│    Dynamic Visual Confusion Matrix   │ │    Automated Background LoRA       │
│  pipeline/rescorer/confusion_matrix  │ │    pipeline/training/lora_micro    │
│  DP character alignment (align())    │ │    Apple Silicon MPS (PyTorch 2.13)│
│  Dynamic cost tuning (adapt_cost())  │ │    Experience Replay DataLoader    │
│  In-memory BeamRescorer singleton    │ │    (1:1 feedback : golden anchor)  │
│  Immediate candidate re-ranking!     │ └─────────────┬──────────────────────┘
└──────────────────────────────────────┘               │
                                         ┌─────────────▼──────────────────────┐
                                         │       Clinical Safety Gate         │
                                         │  CER Regression Check (<5% rel)    │
                                         │  LASA Audit (0 drug substitutions) │
                                         │  Safe Adapter Promotion / Merge    │
                                         └────────────────────────────────────┘
```

### Core Testing Principles
1. **Opaque-Box Requirement-Driven Verification**: Tests interact strictly through public interfaces, HTTP endpoints, datasets, samplers, and CLI entry points without coupling to private implementation state.
2. **Deterministic Reproducibility & State Isolation**: All tests allocate isolated temporary directories for manifests, crops, and checkpoints to guarantee zero side effects across runs.
3. **No Facade / No Mock Cheating**: Business logic (DP alignment, beam rescoring arithmetic, experience replay batch composition, PEFT LoRA parameter isolation, and LASA safety regex matching) is evaluated against authentic live tensor operations and real data structures.
4. **Hardware Acceleration Verification**: Exercises Apple Silicon MPS (`mps` device) natively with automatic graceful fallback to CPU in environments without Metal acceleration.

---

## 2. Feature Inventory & Test Coverage Matrix (F1–F17)

| Feature ID | Feature Name | Primary Source Files | Req | Tier 1 (Isolation) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workload) | Pass/Fail Acceptance Criteria |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **F1** | `POST /v1/feedback` Ingestion Endpoint | `backend/app/routes/feedback.py`<br>`backend/app/schemas.py` | R1 | 5 | 5 | ✓ | ✓ | HTTP 200 on valid correction; HTTP 422 on invalid bbox/confidence/IDs; returns generated `feedback_id`. |
| **F2** | Append-Only Manifest Storage | `backend/app/routes/feedback.py`<br>`backend/app/config.py` | R1 | 5 | 5 | ✓ | ✓ | Thread-safe POSIX `fcntl.flock(LOCK_EX)` append to `manifest.jsonl`; valid JSONL records with `fsync()`. |
| **F3** | Line Crop Persistence | `backend/app/routes/feedback.py` | R1 | 5 | 5 | ✓ | ✓ | Decodes raw base64 or Data URL to PNG at `crops/<feedback_id>.png`; HTTP 422 on corrupt base64/0 bytes. |
| **F4** | DP Character Alignment Extraction | `pipeline/rescorer/confusion_matrix.py` | R2 | 5 | 5 | ✓ | ✓ | Dynamic programming alignment detects 1:1, 2:1, 1:2, and 2:2 substitutions/contractions/expansions in $< 1\text{ms}$. |
| **F5** | Dynamic Confusion Matrix Cost Adaptation | `pipeline/rescorer/confusion_matrix.py` | R2 | 5 | 5 | ✓ | ✓ | Discounts substitution cost by $(1 - \eta)$; enforces safety floor $c_{\text{min}} \ge 0.15$; symmetric update. |
| **F6** | Live Beam Rescorer Immediate Re-ranking | `pipeline/rescorer/beam_rescorer.py`<br>`backend/app/engine.py` | R2 | 5 | 5 | ✓ | ✓ | In-memory rescorer immediately flips candidate rank on subsequent inference calls without model retraining. |
| **F7** | Dynamic Confusion Matrix Persistence | `pipeline/rescorer/confusion_matrix.py` | R2 | 5 | 5 | ✓ | ✓ | Atomic export and import to `dynamic_confusion_matrix.json`; preserves existing baseline pairs. |
| **F8** | Experience Replay Dataset & Sampler | `pipeline/training/experience_replay.py` | R3 | 5 | 5 | ✓ | ✓ | Exactly 50% feedback and 50% anchor samples per batch; round-robin sampling with replacement. |
| **F9** | Apple Silicon MPS LoRA Micro-Tuning | `pipeline/training/lora_micro_tune.py` | R3 | 5 | 5 | ✓ | ✓ | PEFT LoRA ($r=16, \alpha=32$, $q,v$ projections); trainable params $<1\%$; periodic `torch.mps.empty_cache()`. |
| **F10** | Comprehensive Ship Safety Gate | `pipeline/training/ship_gate.py`<br>`backend/app/ship_gate.py` | R3 | 5 | 5 | ✓ | ✓ | Candidate CER $\le \text{baseline} \times 1.05$; zero tolerance ($0$ violations) across 20 bidirectional LASA pairs. |
| **F11** | Adapter Merge & Promotion | `pipeline/training/lora_micro_tune.py`<br>`pipeline/training/ship_gate.py` | R3 | 5 | 5 | ✓ | ✓ | `merge_and_unload()` merges weights into base TrOCR; writes `ship_decision.json`; rejects on gate failure. |
| **F12** | Frontend Client & Route Proxy | `frontend/src/app/api/feedback/route.ts`<br>`frontend/src/lib/apiClient.ts` | R4 | 5 | 5 | ✓ | ✓ | Route proxy forwards to backend `/v1/feedback`; HTTP 400 on malformed input; simulated fallback in dev. |
| **F13** | Canvas Line Crop Extractor | `frontend/src/lib/cropUtils.ts` | R4 | 5 | 5 | ✓ | ✓ | Computes 4% padded pixel coordinates; clamps to natural image dimensions; returns base64 PNG data URL. |
| **F14** | Debounced InlineEditor Dispatch | `frontend/src/components/InlineEditor.tsx` | R4 | 5 | 5 | ✓ | ✓ | 500ms trailing-edge keystroke debounce; immediate dispatch on `Enter`, suggestion 1-5, and `onBlur`. |
| **F15** | Optimistic UI & Visual Status Badges | `frontend/src/components/InlineEditor.tsx`<br>`frontend/src/types/ocr.ts` | R4 | 5 | 5 | ✓ | ✓ | Immediate in-memory text update; 4-state lifecycle (`debouncing` $\to$ `syncing` $\to$ `synced` $\to$ `error`). |
| **F16** | Darkroom Component Mounting | `frontend/src/components/SplitCurtain.tsx`<br>`frontend/src/app/page.tsx` | R4 | 5 | 5 | ✓ | ✓ | Mounts `SplitCurtain` X-Ray slider and `DarkroomToolbar` image enhancement filters linked to active page. |
| **F17** | Local Proof on Apple Silicon | Full Test Suites (`pytest`, `npm test`) | R5 | 5 | 5 | ✓ | ✓ | Complete end-to-end integration verified locally on Mac Studio; 0 test failures; 0 unhandled promise rejections. |

---

## 3. Four-Tier Testing Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Tier 1: Feature Isolation (>=5 per feature)                 │
│  85 tests minimum: 5 nominal unit/integration tests for each feature F1-F17 │
├─────────────────────────────────────────────────────────────────────────────┤
│             Tier 2: Boundary & Corner Cases (>=5 per feature)               │
│  85 tests minimum: 5 edge/error/corruption tests for each feature F1-F17   │
├─────────────────────────────────────────────────────────────────────────────┤
│                   Tier 3: Pairwise Subsystem Combinations                   │
│  Cross-subsystem interactions: Feedback -> Alignment -> Rescorer Re-rank    │
│  Feedback -> Manifest -> Replay Loader -> LoRA Step -> Safety Gate Decision │
├─────────────────────────────────────────────────────────────────────────────┤
│                Tier 4: Real-World Workload Scenarios                        │
│  Complete Darkroom operator review sessions, prescription LASA workflows    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Feature Isolation (Nominal Paths)
Validates each feature independently under clean, expected inputs:
- **F1**: Valid line/word correction payloads ingested via `/v1/feedback`; returns HTTP 200 with generated feedback ID.
- **F2**: Appending to `manifest.jsonl` yields valid, parseable JSON lines with all 10 required metadata fields.
- **F3**: Raw base64 and Data URL payloads decoded and written as clean PNG files in `crops/`.
- **F4**: Dynamic programming alignment correctly extracts 1:1, 2:1, 1:2, and 2:2 substitution operations.
- **F5**: `adapt_from_correction()` discounts optical confusion costs symmetrically while maintaining $c_{\text{min}} \ge 0.15$.
- **F6**: Rescorer computes reduced confusion penalty, causing corrected hypothesis to gain ranking points.
- **F7**: Updated confusion matrix serialized to JSON and successfully reloaded with identical costs.
- **F8**: `ExperienceReplayDataset` and `ReplayBatchSampler` instantiate and yield properly formatted batch tensors.
- **F9**: PEFT LoRA initializes with $r=16, \alpha=32$ on target modules `["q_proj", "v_proj"]`.
- **F10**: CER check passes within 5% tolerance; LASA audit passes clean prescriptions without drug confusion.
- **F11**: Checkpoint merge produces standalone model directory; writes `ship_decision.json` with `promote: true`.
- **F12**: Next.js proxy forwards requests with appropriate headers (`X-Feedback-Provider: backend`).
- **F13**: Canvas crop coordinate computation adds 4% padding and converts normalized coordinates to integer pixels.
- **F14**: InlineEditor dispatches feedback on `Enter`, speed review quick-picks 1-5, and blur.
- **F15**: UI badge transitions through 4-stage lifecycle (`debouncing` $\to$ `syncing` $\to$ `synced` $\to$ `error`).
- **F16**: `SplitCurtain` renders with underlying handwriting image and draggable divider.
- **F17**: Local execution on Apple Silicon Mac Studio completes with zero test failures.

### Tier 2: Boundary & Corner Cases
Validates error recovery, edge conditions, malformed payloads, and resource limits:
- **F1**: Rejects empty `document_id`, whitespace-only `line_id`, out-of-bounds confidence ($<0.0$ or $>1.0$), and inverted bboxes with HTTP 422.
- **F2**: 20 concurrent threads writing simultaneously to `manifest.jsonl` produce zero corrupted or interleaved lines.
- **F3**: Rejects corrupted base64, zero-byte payloads, and non-image data with HTTP 422; gracefully handles omitted crops.
- **F4**: Aligns identical strings with zero cost; handles empty strings, non-ASCII Unicode characters, and long strings safely.
- **F5**: Prevents learning rates $< 0.0$ or $> 1.0$; prevents costs from dropping below minimum floor $c_{\text{min}} = 0.15$.
- **F6**: Handles single-hypothesis beams, empty candidate lists, and identical hypotheses without division by zero.
- **F7**: Handles missing or corrupted JSON files by logging warnings and falling back to default confusion pairs.
- **F8**: Handles empty feedback manifest, empty anchor directory, and batch size 1 without crashing.
- **F9**: Micro-tuning handles step 0, learning rate 0.0, and executes periodic `torch.mps.empty_cache()`.
- **F10**: Zero tolerance: single dangerous drug substitution (e.g., `Hydralazine` $\to$ `Hydroxyzine`) fails LASA audit. CER regression $>5\%$ strictly rejects candidate.
- **F11**: Merge refuses unapproved candidates; raises `FileNotFoundError` if adapter weights are missing.
- **F12**: Route proxy returns HTTP 400 on malformed JSON; returns HTTP 502 when backend is unreachable.
- **F13**: Zero-area bounding boxes return `null` coordinates; coordinates $>1.0$ clamped to natural dimensions.
- **F14**: Rapid bursts of keystrokes (e.g. 50 in 100ms) collapse into exactly 1 debounced API dispatch.
- **F15**: Server HTTP 500 displays `error` badge while preserving operator's typed correction in DOM.
- **F16**: Draggable divider clamped strictly to $[0, 100]\%$; image filter sliders clamped to safe ranges.
- **F17**: Test environment handles missing prior directories and executes cleanly from scratch.

### Tier 3: Pairwise Subsystem Integrations
Validates end-to-end multi-module interactions across architectural boundaries:
1. **Feedback Ingestion $\to$ DP Alignment $\to$ Live Rescorer Rank Flip (F1 + F4 + F5 + F6)**:
   - Initial inference beam has noisy candidate rank 1 and correct candidate rank 2.
   - Operator submits correction via `POST /v1/feedback`.
   - Dynamic confusion matrix adapts substitution costs.
   - Subsequent inference with same raw beam immediately elevates corrected candidate to rank 1.
2. **Feedback Ingestion $\to$ Manifest $\to$ Experience Replay Loader (F1 + F2 + F3 + F8)**:
   - Operator submits corrections with line crops.
   - Records saved to manifest and crop PNGs written to disk.
   - `ExperienceReplayDataset` reads manifest and pairs samples with anchor lines.
   - `ReplayBatchSampler` yields batches containing exactly 50% feedback and 50% anchor samples.
3. **Experience Replay $\to$ LoRA Micro-Tuning $\to$ Adapter Serialization (F8 + F9 + F11)**:
   - Replay data loader feeds mini-batches to PEFT LoRA training loop on Apple Silicon MPS.
   - Executes micro-steps with parameter isolation ($<1\%$ trainable params).
   - Serializes adapter checkpoint to disk (`adapter_model.safetensors`, `adapter_config.json`).
4. **Adapter Candidate $\to$ Safety Gate CER Check $\to$ LASA Audit $\to$ Promotion (F9 + F10 + F11)**:
   - Evaluates candidate adapter against baseline CER ($\le \text{baseline} \times 1.05$).
   - Evaluates predictions across 20 bidirectional LASA drug pairs (0 substitutions allowed).
   - Promotes adapter only if both criteria pass; merges weights into standalone model.
5. **Darkroom InlineEditor $\to$ Debounce $\to$ Route Proxy $\to$ Manifest $\to$ Confirmation Badge (F12 + F13 + F14 + F15)**:
   - Operator edits line in UI; badge shows `debouncing`.
   - 500ms trailing debounce triggers offscreen canvas crop.
   - Dispatches payload to Next.js route proxy `/api/feedback`.
   - Proxy forwards to FastAPI backend `/v1/feedback`.
   - Backend appends to manifest; UI badge transitions to `synced`.

### Tier 4: Real-World Workload Scenarios
Validates authentic operational use cases:
1. **Scenario 1: Comprehensive Operator Darkroom Review Session**:
   - Emulates a human operator reviewing a multi-page medical document.
   - Corrects ambiguous handwriting lines with cursive ligatures (`"rn"` $\to$ `"m"`).
   - Verifies debouncing, visual badge lifecycle, and immediate live rank flipping on subsequent pages.
2. **Scenario 2: Critical Clinical Prescription Workflow (LASA Protection)**:
   - Prescription contains ambiguous cursive medication line `"Hydralazine 25mg"`.
   - Model beam contains confusable drug `"Hydroxyzine 25mg"`.
   - Rescorer elevates correct medication based on clinical context.
   - Automated safety gate tests reject any candidate adapter producing dangerous LASA substitutions.
3. **Scenario 3: Closed-Loop Self-Tuning Flywheel on Apple Silicon**:
   - Full closed-loop pipeline verified end-to-end locally on Mac Studio:
     $\text{Operator Edit} \to \text{Feedback Ingestion} \to \text{DP Recalibration} \to \text{Rescorer Flip} \to \text{Replay Training} \to \text{LASA Gate} \to \text{Promotion}$.

---

## 4. Execution Commands & Test Harness Configuration

### Primary Test Suite Invocations
```bash
# 1. Execute the comprehensive Flywheel E2E integration test suite
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -v

# 2. Execute with detailed output and short traceback on failure
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -v --tb=short

# 3. Execute targeted Tier markers
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -m tier1 -v
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -m tier2 -v
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -m tier3 -v
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -m tier4 -v

# 4. Execute targeted feature tests by name
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -k "rescorer" -v
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -k "lasa" -v
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -k "feedback" -v

# 5. Run without GPU / long training tests for rapid CI checks
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -m "not gpu and not slow" -v

# 6. Run Apple Silicon MPS hardware acceleration tests
.venv/bin/pytest tests/e2e/test_flywheel_e2e.py -m gpu -v
```

### Full Multi-Track Test Commands
```bash
# Run backend feedback unit and route tests
.venv/bin/pytest backend/tests/test_feedback_route.py -v

# Run confusion matrix and rescorer unit tests
.venv/bin/pytest tests/unit/test_confusion_matrix.py tests/unit/test_beam_rescorer.py -v

# Run clinical vocabulary and ship safety gate tests
.venv/bin/pytest pipeline/tests/test_clinical_vocabularies.py pipeline/tests/test_htr_ship_gate.py -v

# Run frontend Darkroom Vitest suite (313 tests)
cd frontend && npm test

# Run master E2E runner CLI with JSON reporting
.venv/bin/python tests/e2e/runner.py --tier 3 --json-report tests/e2e/report.json
```

---

## 5. Pass/Fail Criteria & Quality Gates

1. **Test Execution & Exit Code**:
   - Every test in `tests/e2e/test_flywheel_e2e.py` must pass with exit code `0`.
   - Zero test failures, zero regressions across existing backend tests (93 tests) and frontend tests (313 tests).
2. **Dynamic Confusion Recalibration Gate**:
   - Confusion cost reduction must satisfy $c_{\text{new}} \le c_{\text{curr}} \times (1 - \eta) + 1\text{e-}4$.
   - Effective minimum cost must strictly obey $c_{\text{new}} \ge 0.15$.
   - Cost updates must be strictly symmetric: $\text{cost}(a, b) == \text{cost}(b, a)$.
3. **Live Rescorer Rank Flip Gate**:
   - Following dynamic adaptation, candidate confusion penalty must decrease by $>0.10$.
   - Rescorer score of corrected hypothesis must increase, flipping rank 2 to rank 1 in $<2\text{ms}$.
4. **Experience Replay Sampling Gate**:
   - Every mini-batch yielded by `ReplayBatchSampler` must contain exactly $\lfloor \text{batch\_size} \times \text{replay\_ratio} \rfloor$ feedback samples and $\lceil \text{batch\_size} \times (1 - \text{replay\_ratio}) \rceil$ anchor samples (50:50 ratio for batch size 8).
5. **Clinical Safety Gate (Zero-Tolerance LASA & CER Regression)**:
   - **CER Regression**: $\text{candidate\_cer} \le \text{baseline\_cer} \times 1.05$.
   - **Clinical LASA Audit**: Scanned across all 20 bidirectional Look-Alike/Sound-Alike drug pairs from `rxnorm_medications.json`. Zero tolerance: `len(violations) == 0`. ANY drug confusion results in immediate rejection (`promote: false`).
6. **Concurrency & File Locking Integrity**:
   - Multi-threaded submissions to `POST /v1/feedback` using `fcntl.flock` must produce exactly $N$ distinct valid JSON lines in `manifest.jsonl` with zero interleaved records.
