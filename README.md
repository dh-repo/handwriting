# Handwriting transcription and review

A local-first tool for general handwriting: upload an image or PDF, recognize it with TrOCR, compare the scan and transcript, correct the text, save on this device, and export.

## What the app guarantees

- Local inference is the default. Cloud recognition requires server enablement and an explicit Cloud selection. Selecting cloud sends document images to Azure; it does not authorize cloud correction storage.
- Unavailable models, broken connections, and inference failures return errors. Demo samples are labeled and excluded from correction collection. They are not accuracy demonstrations.
- Unknown confidence stays unknown and is included in review. Decoder scores are model scores, not calibrated accuracy probabilities. Human review status is separate.
- Documents, original files, page images, edits, review state, and recent revisions are saved in IndexedDB on this browser/device. Resume and delete controls are in the workspace. Browser storage can be cleared; export is the portable backup.
- Local document saving and feedback submission are separate. Failed feedback stays in the local outbox with a retry control. Stable submission IDs prevent duplicate collection. Corrections are collected locally; automatic adaptation and promotion are disabled.
- Background job polling and SSE observe the worker; they never perform inference. Jobs are single-process and memory-resident. Server restarts interrupt them; resubmit explicitly. Completed jobs expire according to `JOB_RETENTION_SECONDS`.

## Run locally

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-htr.txt
.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```sh
cd frontend
npm ci
BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://localhost:3000`. The default checkpoint is `microsoft/trocr-base-handwritten`; the first real run downloads it. `DEVICE=auto` selects available hardware. Production requests must not enable `USE_MOCK_ENGINE`; that flag is exclusively for isolated tests and labeled demos.

For cloud recognition configure `ENABLE_TURBO_MODE=true`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `TURBO_MODEL_DEPLOYMENT`. Credentials alone do not select cloud. General local inference never calls the VLM referee. Feedback remains local regardless of Azure Storage credentials.

## API changes

`POST /v1/recognize`, `/v1/recognize/stream`, and `/v1/jobs` accept `processing_mode=local|cloud`, default `local`. JSON requests use `options.processing_mode`. Legacy `turbo` is optional and must agree with an explicitly selected mode. `/v1/recognize-line` accepts local crops only; cloud callers use the document endpoint.

Results include engine/model/location and demo/incomplete metadata. Confidence fields may be `null`. Clients must not replace null with a fabricated confidence. Feedback accepts a stable `submission_id`; reusing it with another correction returns 409. Acknowledgments mean durable local collection, not training.

## Measured baseline

A real TrOCR-Base run on **100 frozen Teklia/IAM-line test lines** measured **4.79% CER**, **13.23% WER**, and **37% exact line matches**, with zero failed inferences. Four beams; MPS; 145.18 seconds for inference across the set. This is a small line-crop benchmark, not full-page accuracy, a personal handwriting evaluation, or a clinical validation.

Dataset/model revisions, content hashes, generation settings, and worst examples are recorded under `benchmarks/iam`. Raw dataset images remain local and are not committed. To reproduce, use a new directory:

```sh
.venv/bin/python -m pipeline.evaluation.reliable_benchmark freeze --output benchmarks/new-iam --limit 100
.venv/bin/python -m pipeline.evaluation.reliable_benchmark run --output benchmarks/new-iam
```

Active review seconds are recorded locally from interactions, excluding intervals over 60 seconds; this is separate from inference latency. Page segmentation and complete document quality require a separate labeled page benchmark and are not established by the line result.

## Training and manual releases

Training produces candidate artifacts only. Missing validation data cannot produce a passing release. Evaluate a standalone candidate and baseline on the same frozen holdout with `python -m pipeline.training.evaluate_release --candidate PATH --baseline PATH_OR_MODEL_ID --holdout DIR --output NEW_REPORT_DIR`. The report includes exact candidate content hashes. Clinical evaluation additionally requires `--clinical-holdout` containing real labeled images covering every directional pair; no supplied prediction tuples qualify.

Activate only after measured evaluation with `python -m pipeline.training.release_registry activate --checkpoint PATH --decision REPORT_DIR/ship_decision.json`. Loading is checked before updating `checkpoints/active.json`. Restart the daemon to load the selected checkpoint. `python -m pipeline.training.release_registry rollback` restores the previous load-validated selection; restart afterward. Automatic hot swapping is not enabled.

Specialty vocabulary, signature, and clinical training code remain experimental. The supported UI is general handwriting review.

## Verification

See [TEST_READY.md](TEST_READY.md) for current checks and limitations. Unit and mocked API tests verify mechanics; only real-model benchmark reports establish measured recognition quality. Historical passing-test totals and “zero defects” claims are retired.
