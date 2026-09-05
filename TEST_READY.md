# Verification status — 2026-09-04

This file replaces historical passing-test totals and release-readiness claims. No clinical safety certification, full-page accuracy, or automatic-learning quality is claimed.

## Verified in this repair

- Frontend: 500 tests passed in 47 files, including failure handling, local save/resume/export/deletion, review state, feedback outbox, and unknown-confidence handling. Run with one worker so timing-sensitive legacy tests are not competing with model inference.
- Backend plus focused release/evaluation/storage checks: 219 passed. This includes local/cloud option conflicts, no cloud feedback uploads, persistence failure, duplicate submission IDs, read-only polling/SSE, one worker execution with multiple subscribers, release evidence rejection, and rollback loading failure.
- TypeScript type checking and the Next.js production build passed.
- A live local TrOCR model loaded successfully on MPS. `/v1/recognize-line` returned actual recognized text and engine/model/location metadata. Conflicting local/turbo requests returned 422. The document endpoint also returned 200 with real model provenance and a page image.
- Real-model benchmark: 100 frozen Teklia/IAM-line test crops; 4.79% CER, 13.23% WER, 37% exact line match, zero failed samples, 145.18 seconds of inference. See `benchmarks/iam/baseline.json` and `manifest.json` for source revision, model revision, hashes, settings, and worst examples. This is a small line-crop result, not a page or clinical benchmark.

## Known verification limits

- Browser screenshots and desktop/narrow-width visual inspection were blocked: the browser tool could not verify an admin-enforced policy. No alternative browser mechanism was used to bypass that restriction. Automated React workflow checks passed, but visual QA remains outstanding.
- The complete historical training suite was attempted, including a bounded retry, but multiprocessing tests timed out on this Python 3.14 host. It is not reported as passing.
- The broader `tests/unit` suite has 13 vocabulary/rescoring failures. An untouched `git archive HEAD` checkout reproduced the exact same 13 failures (105 passed, 5 skipped). These experimental clinical/lexicon tests are not a green release signal. The changed local-only feedback test was updated and passes.
- No real clinical holdout was supplied, so clinical promotion is unverified and cannot qualify without measured candidate inference and complete directional coverage.
- No user's personal handwriting was evaluated. Active review timing is local interaction time, not a measured usability study.
- Cloud recognition was not tested against a live Azure account. Local mode and disabled-cloud rejection are covered by tests.

## Commands

```sh
.venv/bin/pytest backend/tests pipeline/tests/test_htr_ship_gate.py pipeline/tests/test_htr_eval_report.py pipeline/tests/test_evaluation.py tests/unit/test_azure_storage_feedback.py -q
cd frontend
npx tsc --noEmit
npm test -- --maxWorkers=1 --minWorkers=1
npm run build
```

The Python environment used Torch 2.14 and Transformers 4.57.6 on Apple Silicon. Unpinned Transformers 5.16 failed tokenizer initialization during the first benchmark attempt; `requirements-htr.txt` now pins the verified Transformers version.

Unit/API tests use controlled doubles to validate mechanics. The benchmark and live API checks use a real downloaded checkpoint. Neither substitutes for visual QA or a labeled full-document evaluation.
