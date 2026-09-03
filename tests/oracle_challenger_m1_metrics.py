"""
tests/oracle_challenger_m1_metrics.py
Empirical Challenger Oracle & Metrics Collector for Milestone 1.
Runs high-concurrency stress scenarios and outputs detailed metrics:
- Concurrency level and barrier synchronization
- Latency metrics (P50, P90, P95, P99, Max)
- Manifest integrity validation (line count, JSON parsing, field fidelity)
- Image crop verification (magic numbers, PIL header validation, non-zero bytes)
- File descriptor inspection before/after execution
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path and USE_MOCK_ENGINE is set
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["USE_MOCK_ENGINE"] = "true"

import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Tuple
import warnings

from PIL import Image
from fastapi.testclient import TestClient

from backend.app.config import Settings, get_settings, reset_settings_cache
from backend.app.engine import reset_engine
from backend.app.main import create_app

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def get_open_fd_count() -> int:
    try:
        return len(os.listdir("/dev/fd"))
    except Exception:
        return -1


def run_oracle_metrics(concurrency: int = 50) -> Dict[str, Any]:
    tmp_base = Path(f"/tmp/challenger_m1_oracle_{int(time.time())}")
    manifest_path = tmp_base / "manifest.jsonl"
    crops_dir = tmp_base / "crops"
    tmp_base.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        FEEDBACK_DIR=str(tmp_base),
        FEEDBACK_MANIFEST_PATH=str(manifest_path),
        FEEDBACK_CROPS_DIR=str(crops_dir),
        CONFUSION_LEARNING_RATE=0.10,
        USE_MOCK_ENGINE=True,
    )

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    # Clean GC and FD measure
    gc.collect()
    fd_start = get_open_fd_count()

    barrier = threading.Barrier(concurrency)
    latencies: List[float] = [0.0] * concurrency
    status_codes: List[int] = [0] * concurrency
    response_payloads: List[Dict[str, Any]] = [{}] * concurrency

    # Prepare 50 unique payloads
    input_payloads = []
    for i in range(concurrency):
        img = Image.new("RGB", (64 + (i % 32), 24 + (i % 16)), color=((i * 13) % 256, (i * 29) % 256, (i * 47) % 256))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        input_payloads.append({
            "document_id": f"oracle_doc_{i:04d}",
            "page_number": 1 + (i % 3),
            "line_id": f"line_{i:04d}",
            "word_id": f"w_{i:02d}",
            "original_prediction": f"pred_word_{i:04d}",
            "operator_correction": f"corr_word_{i:04d}",
            "confidence": round(0.40 + (i % 50) * 0.01, 2),
            "bbox": [0.05, 0.10, 0.25, 0.80],
            "line_crop_base64": b64,
        })

    def _worker(idx: int):
        payload = input_payloads[idx]
        barrier.wait(timeout=10.0)
        t0 = time.perf_counter()
        resp = client.post("/v1/feedback", json=payload)
        t1 = time.perf_counter()
        latencies[idx] = t1 - t0
        status_codes[idx] = resp.status_code
        if resp.status_code == 200:
            response_payloads[idx] = resp.json()

    t_batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_worker, i) for i in range(concurrency)]
        for f in futures:
            f.result()
    t_batch_total = time.perf_counter() - t_batch_start

    gc.collect()
    fd_end = get_open_fd_count()

    # 1. Verification of HTTP responses
    success_count = sum(1 for code in status_codes if code == 200)

    # 2. Manifest file inspection
    manifest_bytes = manifest_path.stat().st_size if manifest_path.exists() else 0
    lines = [l.strip() for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    valid_json_count = 0
    parse_errors = 0
    parsed_records = []
    seen_feedback_ids = set()

    for idx, line in enumerate(lines):
        try:
            rec = json.loads(line)
            valid_json_count += 1
            parsed_records.append(rec)
            seen_feedback_ids.add(rec["feedback_id"])
        except Exception as e:
            parse_errors += 1

    # 3. Fidelity check between submitted payloads and stored records
    stored_by_doc = {r["document_id"]: r for r in parsed_records}
    fidelity_matches = 0
    for p in input_payloads:
        doc_id = p["document_id"]
        if doc_id in stored_by_doc:
            rec = stored_by_doc[doc_id]
            if (rec["line_id"] == p["line_id"] and
                rec["original_prediction"] == p["original_prediction"] and
                rec["operator_correction"] == p["operator_correction"] and
                abs(rec["confidence"] - p["confidence"]) < 1e-5 and
                rec["bbox"] == p["bbox"]):
                fidelity_matches += 1

    # 4. Crop image inspections
    crop_files = list(crops_dir.glob("*.png"))
    valid_png_headers = 0
    non_zero_byte_files = 0
    pil_load_success = 0
    total_crop_bytes = 0

    for cf in crop_files:
        sz = cf.stat().st_size
        total_crop_bytes += sz
        if sz > 0:
            non_zero_byte_files += 1
        with open(cf, "rb") as f:
            hdr = f.read(8)
            if hdr == PNG_HEADER:
                valid_png_headers += 1
        try:
            with Image.open(cf) as im:
                im.verify()
                pil_load_success += 1
        except Exception:
            pass

    sorted_lat = sorted(latencies)
    metrics = {
        "concurrency_level": concurrency,
        "total_requests": concurrency,
        "successful_responses_200": success_count,
        "batch_wall_time_sec": round(t_batch_total, 4),
        "latency_p50_ms": round(sorted_lat[int(concurrency * 0.50)] * 1000, 2),
        "latency_p90_ms": round(sorted_lat[int(concurrency * 0.90)] * 1000, 2),
        "latency_p95_ms": round(sorted_lat[int(concurrency * 0.95)] * 1000, 2),
        "latency_p99_ms": round(sorted_lat[int(concurrency * 0.99)] * 1000, 2),
        "latency_max_ms": round(sorted_lat[-1] * 1000, 2),
        "manifest_lines_total": len(lines),
        "manifest_valid_json_count": valid_json_count,
        "manifest_parse_errors": parse_errors,
        "manifest_unique_feedback_ids": len(seen_feedback_ids),
        "manifest_fidelity_matches": fidelity_matches,
        "crop_files_found": len(crop_files),
        "crop_files_non_zero_bytes": non_zero_byte_files,
        "crop_files_valid_png_header": valid_png_headers,
        "crop_files_pil_load_success": pil_load_success,
        "crop_total_bytes": total_crop_bytes,
        "fd_before": fd_start,
        "fd_after": fd_end,
        "fd_delta": fd_end - fd_start if (fd_start > 0 and fd_end > 0) else 0,
    }
    return metrics


if __name__ == "__main__":
    print("Executing Challenger Concurrency & Integrity Oracle (50 concurrent requests)...")
    res = run_oracle_metrics(50)
    print(json.dumps(res, indent=2))
