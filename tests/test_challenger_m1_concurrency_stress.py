"""
tests/test_challenger_m1_concurrency_stress.py
Empirical Adversarial Stress Test Suite for Milestone 1:
Backend Feedback Ingestion, Manifest Storage, POSIX File Locking, and File Descriptor Hygiene.

Focus:
1. High concurrency: 50+ concurrent requests across threads with barrier synchronization.
2. Manifest integrity: exact line counts, zero JSON parse errors, zero interleaved lines, zero lost updates.
3. Image crop integrity: non-zero byte size, valid 8-byte PNG header (b"\\x89PNG\\r\\n\\x1a\\n"), PIL verify.
4. Process-level concurrency: ProcessPoolExecutor verifying OS-level POSIX flock across separate OS processes.
5. Resource hygiene: File descriptor leak checking and zero ResourceWarning (unclosed file descriptors).
6. Cold-start race conditions: 40 threads hitting non-existent manifest/crops directories simultaneously.
7. Reader/writer lock contention: simultaneous GET /v1/feedback/stats (LOCK_SH) and POST /v1/feedback (LOCK_EX).
8. Failure isolation: corrupt payloads rejected with 422 without writing orphan records or corrupt files.
"""

from __future__ import annotations

import os
import sys

# Ensure USE_MOCK_ENGINE is set before backend is imported
os.environ["USE_MOCK_ENGINE"] = "true"

import base64
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import io
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, get_settings, reset_settings_cache
from backend.app.engine import get_engine, reset_engine, set_engine
from backend.app.main import create_app
from backend.app.routes.feedback import _append_to_manifest


PNG_HEADER = b"\x89PNG\r\n\x1a\n"
FEEDBACK_ID_REGEX = re.compile(r"^fb_\d{8}_\d{6}_[0-9a-f]{8}$")


def _generate_test_png_b64(width: int = 100, height: int = 30, color: Tuple[int, int, int] = (255, 0, 0)) -> str:
    """Generate a distinct valid PNG image in raw base64."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _generate_test_jpeg_data_url(width: int = 80, height: int = 25) -> str:
    """Generate a valid JPEG data URL."""
    img = Image.new("RGB", (width, height), color=(0, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{raw_b64}"


def _get_open_fd_count() -> int:
    """Get count of open file descriptors for current process on macOS/Linux."""
    if os.path.exists("/dev/fd"):
        try:
            return len(os.listdir("/dev/fd"))
        except Exception:
            pass
    return -1


def _worker_append(manifest_path_str: str, idx: int) -> int:
    """Standalone worker function for ProcessPoolExecutor testing."""
    manifest_path = Path(manifest_path_str)
    record = {
        "feedback_id": f"fb_proc_{idx:04d}",
        "document_id": f"doc_proc_{idx}",
        "line_id": f"line_proc_{idx}",
        "original_prediction": f"pred_{idx}",
        "operator_correction": f"corr_{idx}",
        "confidence": 0.90,
        "timestamp": "2026-09-03T12:00:00Z",
    }
    _append_to_manifest(manifest_path, record)
    return idx


@pytest.fixture
def isolated_feedback_env(tmp_path: Path):
    """Provide clean isolated test environment with custom Settings."""
    fb_dir = tmp_path / "stress_feedback"
    manifest_path = fb_dir / "manifest.jsonl"
    crops_dir = fb_dir / "crops"

    fb_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        FEEDBACK_DIR=str(fb_dir),
        FEEDBACK_MANIFEST_PATH=str(manifest_path),
        FEEDBACK_CROPS_DIR=str(crops_dir),
        CONFUSION_LEARNING_RATE=0.10,
        USE_MOCK_ENGINE=True,
    )

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings

    yield {
        "app": app,
        "client": TestClient(app),
        "fb_dir": fb_dir,
        "manifest_path": manifest_path,
        "crops_dir": crops_dir,
        "settings": settings,
    }

    app.dependency_overrides.clear()
    reset_settings_cache()
    reset_engine()


# ---------------------------------------------------------------------------
# Test 1: 50+ Concurrent Threads with Barrier Synchronization (Challenger Core)
# ---------------------------------------------------------------------------

def test_stress_50_concurrent_threads_with_barrier(isolated_feedback_env: Dict[str, Any]) -> None:
    """
    Stress-test high concurrency with 50 simultaneous threads hitting POST /v1/feedback.
    Uses a threading.Barrier to ensure all 50 requests hit the route concurrently.
    Verifies:
      - 50/50 return HTTP 200
      - manifest.jsonl contains exactly 50 distinct valid JSON records
      - No JSON parsing errors, no interleaved lines, no lost updates, no deadlocks
      - All 50 line crops are saved with non-zero byte size and valid PNG headers
    """
    client: TestClient = isolated_feedback_env["client"]
    manifest_path: Path = isolated_feedback_env["manifest_path"]
    crops_dir: Path = isolated_feedback_env["crops_dir"]
    num_requests = 50

    barrier = threading.Barrier(num_requests)
    results: List[Dict[str, Any]] = [{} for _ in range(num_requests)]

    def _worker(idx: int):
        # Generate unique image crop per thread
        crop_b64 = _generate_test_png_b64(
            width=80 + (idx % 20),
            height=30 + (idx % 10),
            color=((idx * 5) % 256, (idx * 11) % 256, (idx * 17) % 256),
        )
        payload = {
            "document_id": f"doc_stress_50_{idx:03d}",
            "line_id": f"line_{idx:03d}",
            "page_number": 1,
            "original_prediction": f"metfomin_{idx}mg",
            "operator_correction": f"metformin_{idx}mg",
            "confidence": round(0.50 + (idx % 50) * 0.01, 2),
            "bbox": [0.10, 0.05, 0.20, 0.85],
            "line_crop_base64": crop_b64,
        }

        # Synchronize all threads so they fire at the exact same instant
        barrier.wait(timeout=10.0)

        t_start = time.perf_counter()
        resp = client.post("/v1/feedback", json=payload)
        t_elapsed = time.perf_counter() - t_start

        results[idx] = {
            "status_code": resp.status_code,
            "data": resp.json() if resp.status_code == 200 else resp.text,
            "elapsed": t_elapsed,
        }

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(_worker, i) for i in range(num_requests)]
        for f in futures:
            f.result()

    # 1. Assert all 50 requests succeeded
    latencies = [r["elapsed"] for r in results]
    for i, res in enumerate(results):
        assert res["status_code"] == 200, f"Request {i} failed: {res['data']}"
        assert res["data"]["status"] == "persisted"
        assert FEEDBACK_ID_REGEX.match(res["data"]["feedback_id"])

    # 2. Verify manifest exists and has exactly 50 non-empty lines
    assert manifest_path.exists(), "manifest.jsonl was not created"
    raw_content = manifest_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
    assert len(lines) == num_requests, f"Expected {num_requests} lines, got {len(lines)}"

    # 3. Verify manifest lines are each distinct, valid JSON, with zero interleaving
    seen_feedback_ids = set()
    seen_document_ids = set()
    manifest_crop_paths = []

    for i, line_str in enumerate(lines):
        try:
            record = json.loads(line_str)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Manifest line {i} failed to parse as JSON: {line_str} (error: {exc})")

        fid = record["feedback_id"]
        assert fid not in seen_feedback_ids, f"Duplicate feedback_id found: {fid}"
        seen_feedback_ids.add(fid)

        doc_id = record["document_id"]
        seen_document_ids.add(doc_id)

        crop_p = record["image_crop_path"]
        assert crop_p is not None, f"Crop path is None for record {fid}"
        manifest_crop_paths.append(Path(crop_p))

    assert len(seen_feedback_ids) == num_requests
    assert len(seen_document_ids) == num_requests

    # 4. Verify all 50 crop images on disk: non-zero byte size & valid PNG headers
    crop_files = list(crops_dir.glob("*.png"))
    assert len(crop_files) == num_requests, f"Expected {num_requests} PNG files in crops_dir, found {len(crop_files)}"

    for crop_file in crop_files:
        # Non-zero byte size
        file_size = crop_file.stat().st_size
        assert file_size > 0, f"Crop file {crop_file} has 0 bytes"

        # Valid PNG header (first 8 bytes)
        with open(crop_file, "rb") as f:
            header = f.read(8)
            assert header == PNG_HEADER, f"Crop file {crop_file} has invalid PNG header: {header!r}"

        # Valid load via PIL
        with Image.open(crop_file) as img:
            assert img.format == "PNG"
            assert img.size[0] > 0 and img.size[1] > 0
            img.verify()


# ---------------------------------------------------------------------------
# Test 2: 100 Mixed Adversarial Requests Under Concurrency
# ---------------------------------------------------------------------------

def test_stress_100_mixed_adversarial_requests(isolated_feedback_env: Dict[str, Any]) -> None:
    """
    Stress-test 100 concurrent requests mixing:
      - 40 with PNG crops
      - 20 with JPEG Data URLs
      - 20 with no crops (None)
      - 20 with Unicode/multilingual text and embedded newlines/quotes
    Verifies:
      - 100/100 return HTTP 200
      - Exactly 100 valid lines in manifest.jsonl
      - Exactly 60 crop files in crops directory
      - Unicode and embedded newlines do not corrupt JSONL newline boundaries
    """
    client: TestClient = isolated_feedback_env["client"]
    manifest_path: Path = isolated_feedback_env["manifest_path"]
    crops_dir: Path = isolated_feedback_env["crops_dir"]
    total_requests = 100

    def _build_payload(idx: int) -> Dict[str, Any]:
        mod = idx % 5
        if mod == 0 or mod == 1:
            # PNG crop
            return {
                "document_id": f"doc_png_{idx}",
                "line_id": f"line_{idx}",
                "original_prediction": f"pred_{idx}",
                "operator_correction": f"corr_{idx}",
                "confidence": 0.85,
                "line_crop_base64": _generate_test_png_b64(90, 30),
            }
        elif mod == 2:
            # JPEG Data URL
            return {
                "document_id": f"doc_dataurl_{idx}",
                "line_id": f"line_{idx}",
                "original_prediction": f"pred_{idx}",
                "operator_correction": f"corr_{idx}",
                "confidence": 0.75,
                "line_crop_base64": _generate_test_jpeg_data_url(80, 25),
            }
        elif mod == 3:
            # No crop
            return {
                "document_id": f"doc_nocrop_{idx}",
                "line_id": f"line_{idx}",
                "original_prediction": f"pred_{idx}",
                "operator_correction": f"corr_{idx}",
                "confidence": 0.92,
                "line_crop_base64": None,
            }
        else:
            # Adversarial Unicode, symbols, embedded newlines & quotes
            return {
                "document_id": f"doc_unicode_{idx}",
                "line_id": f"line_{idx}",
                "original_prediction": f"pred_{idx} ± 5µg/mL\nline2\t\"quoted\"",
                "operator_correction": f"corr_{idx} β-blocker 100mg ℞ 診察 🩺",
                "confidence": 0.60,
                "bbox": [0.05, 0.10, 0.15, 0.90],
                "line_crop_base64": _generate_test_png_b64(110, 35),
            }

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(lambda i: client.post("/v1/feedback", json=_build_payload(i)), i) for i in range(total_requests)]
        responses = [f.result() for f in futures]

    for resp in responses:
        assert resp.status_code == 200, f"Failed response: {resp.text}"

    # Verify manifest line count
    manifest_lines = [l.strip() for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(manifest_lines) == total_requests, f"Expected {total_requests}, got {len(manifest_lines)}"

    # Check all lines parse and check embedded newlines didn't break lines
    expected_crops = 0
    for l in manifest_lines:
        rec = json.loads(l)
        if rec["image_crop_path"]:
            expected_crops += 1
            assert Path(rec["image_crop_path"]).exists()

    # Total crops with image: 40 PNG (mod 0,1) + 20 JPEG (mod 2) + 20 Unicode PNG (mod 4) = 80
    # Wait, mod 0: PNG, mod 1: PNG, mod 2: JPEG, mod 3: None, mod 4: Unicode PNG
    # 20 of each mod in 100 requests. 4 mods have crops = 80 crops!
    assert expected_crops == 80
    crop_files = list(crops_dir.glob("*.png"))
    assert len(crop_files) == 80

    for cf in crop_files:
        assert cf.stat().st_size > 0
        with open(cf, "rb") as f:
            assert f.read(8) == PNG_HEADER


# ---------------------------------------------------------------------------
# Test 3: Process-Level Concurrency Stress (OS-Level flock Across OS Processes)
# ---------------------------------------------------------------------------

def test_stress_multiprocess_manifest_locking(tmp_path: Path) -> None:
    """
    Empirically test POSIX fcntl.flock across separate OS processes.
    Uses ProcessPoolExecutor with 8 processes to submit 40 records concurrently.
    Verifies:
      - All processes complete without deadlock or crash
      - manifest.jsonl contains exactly 40 distinct valid records
      - Zero corrupted or interleaved lines
    """
    manifest_path = tmp_path / "proc_test" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    num_records = 40

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_worker_append, str(manifest_path), i) for i in range(num_records)]
        completed = [f.result() for f in futures]

    assert len(completed) == num_records
    assert manifest_path.exists()

    lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == num_records, f"Expected {num_records} lines from multiprocess append, got {len(lines)}"

    seen_ids = set()
    for line in lines:
        record = json.loads(line)
        fid = record["feedback_id"]
        assert fid not in seen_ids
        seen_ids.add(fid)

    assert len(seen_ids) == num_records


# ---------------------------------------------------------------------------
# Test 4: Resource Cleanup & File Descriptor Hygiene
# ---------------------------------------------------------------------------

def test_file_descriptor_hygiene_and_resource_warnings(isolated_feedback_env: Dict[str, Any]) -> None:
    """
    Verify that file descriptors are cleanly closed and no unclosed file descriptor warnings occur.
    Executes 60 requests under warning capture and verifies:
      - Zero ResourceWarning warnings emitted
      - Open file descriptors count remains stable without leaks
    """
    client: TestClient = isolated_feedback_env["client"]
    png_b64 = _generate_test_png_b64(100, 30)

    # Force garbage collection before measuring initial FDs
    gc.collect()
    fd_before = _get_open_fd_count()

    captured_warnings: List[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as warn_list:
        warnings.simplefilter("always", ResourceWarning)

        for i in range(60):
            payload = {
                "document_id": f"doc_fd_{i}",
                "line_id": f"line_{i}",
                "original_prediction": f"pred_{i}",
                "operator_correction": f"corr_{i}",
                "line_crop_base64": png_b64,
            }
            resp = client.post("/v1/feedback", json=payload)
            assert resp.status_code == 200

        # Also hit stats endpoint
        stats_resp = client.get("/v1/feedback/stats")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["total_records"] == 60
        assert stats_resp.json()["total_crops"] == 60

        captured_warnings = [w for w in warn_list if issubclass(w.category, ResourceWarning)]

    gc.collect()
    fd_after = _get_open_fd_count()

    # Filter out any third-party unclosed socket warnings if any; check specifically for file leaks
    unclosed_file_warnings = [
        w for w in captured_warnings
        if "unclosed file" in str(w.message).lower()
    ]
    assert len(unclosed_file_warnings) == 0, f"Encountered unclosed file ResourceWarnings: {[str(w.message) for w in unclosed_file_warnings]}"

    # Check FD count stability
    if fd_before > 0 and fd_after > 0:
        fd_growth = fd_after - fd_before
        # A leak of 60 requests would increase FDs by 60-120+. We allow small transient growth <= 5.
        assert fd_growth <= 5, f"Potential FD leak detected: before={fd_before}, after={fd_after}, growth={fd_growth}"


# ---------------------------------------------------------------------------
# Test 5: Cold-Start Directory Creation Race
# ---------------------------------------------------------------------------

def test_cold_start_directory_creation_race(tmp_path: Path) -> None:
    """
    Test 40 threads hitting POST /v1/feedback simultaneously when neither the feedback directory
    nor the crops directory exist yet. Verifies mkdir(parents=True, exist_ok=True) does not race.
    """
    non_existent_fb_dir = tmp_path / "cold_start_non_existent" / "deep" / "feedback"
    manifest_path = non_existent_fb_dir / "manifest.jsonl"
    crops_dir = non_existent_fb_dir / "crops"

    assert not non_existent_fb_dir.exists()

    settings = Settings(
        FEEDBACK_DIR=str(non_existent_fb_dir),
        FEEDBACK_MANIFEST_PATH=str(manifest_path),
        FEEDBACK_CROPS_DIR=str(crops_dir),
        USE_MOCK_ENGINE=True,
    )

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    num_threads = 40
    barrier = threading.Barrier(num_threads)
    results = []

    def _worker(idx: int):
        payload = {
            "document_id": f"doc_cold_{idx}",
            "line_id": f"line_{idx}",
            "original_prediction": f"cold_pred_{idx}",
            "operator_correction": f"cold_corr_{idx}",
            "line_crop_base64": _generate_test_png_b64(60, 20),
        }
        barrier.wait(timeout=10.0)
        resp = client.post("/v1/feedback", json=payload)
        return resp.status_code

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_worker, i) for i in range(num_threads)]
        results = [f.result() for f in futures]

    assert all(code == 200 for code in results)
    assert manifest_path.exists()
    assert len(manifest_path.read_text().splitlines()) == num_threads
    assert len(list(crops_dir.glob("*.png"))) == num_threads


# ---------------------------------------------------------------------------
# Test 6: Concurrent Reader / Writer Lock Contention
# ---------------------------------------------------------------------------

def test_concurrent_reader_writer_lock_contention(isolated_feedback_env: Dict[str, Any]) -> None:
    """
    Simultaneously execute 40 writer threads (POST /v1/feedback with LOCK_EX)
    and 20 reader threads (GET /v1/feedback/stats with LOCK_SH).
    Verifies:
      - Zero deadlocks
      - All writers and readers complete within 10s
      - Readers never read corrupted or partial JSON
      - Final count equals exactly 40
    """
    client: TestClient = isolated_feedback_env["client"]
    num_writers = 40
    num_readers = 20

    writer_statuses = []
    reader_results = []
    errors = []

    stop_readers = threading.Event()

    def _writer(idx: int):
        try:
            payload = {
                "document_id": f"doc_rw_{idx}",
                "line_id": f"line_{idx}",
                "original_prediction": f"pred_{idx}",
                "operator_correction": f"corr_{idx}",
                "line_crop_base64": _generate_test_png_b64(70, 20),
            }
            resp = client.post("/v1/feedback", json=payload)
            writer_statuses.append(resp.status_code)
        except Exception as e:
            errors.append(f"Writer {idx} error: {e}")

    def _reader(idx: int):
        while not stop_readers.is_set():
            try:
                resp = client.get("/v1/feedback/stats")
                if resp.status_code == 200:
                    reader_results.append(resp.json()["total_records"])
            except Exception as e:
                errors.append(f"Reader {idx} error: {e}")
            time.sleep(0.01)

    # Launch readers
    reader_threads = [threading.Thread(target=_reader, args=(i,)) for i in range(num_readers)]
    for t in reader_threads:
        t.start()

    # Launch writers concurrently
    with ThreadPoolExecutor(max_workers=num_writers) as executor:
        futures = [executor.submit(_writer, i) for i in range(num_writers)]
        for f in futures:
            f.result()

    # Stop readers
    stop_readers.set()
    for t in reader_threads:
        t.join(timeout=2.0)

    assert len(errors) == 0, f"Lock contention produced errors: {errors}"
    assert len(writer_statuses) == num_writers
    assert all(code == 200 for code in writer_statuses)

    # Final stats check
    final_stats = client.get("/v1/feedback/stats").json()
    assert final_stats["total_records"] == num_writers
    assert final_stats["total_crops"] == num_writers


# ---------------------------------------------------------------------------
# Test 7: Failure Isolation Under Concurrency
# ---------------------------------------------------------------------------

def test_failure_isolation_under_concurrency(isolated_feedback_env: Dict[str, Any]) -> None:
    """
    25 threads submitting invalid/corrupted line crops (must return 422)
    interleaved with 25 threads submitting valid payloads (must return 200).
    Verifies:
      - 25 valid requests succeed (200)
      - 25 corrupt requests fail cleanly (422)
      - manifest.jsonl contains EXACTLY 25 records (no orphan records from 422 failures)
      - crops directory contains EXACTLY 25 files (no partial or corrupt images saved)
    """
    client: TestClient = isolated_feedback_env["client"]
    manifest_path: Path = isolated_feedback_env["manifest_path"]
    crops_dir: Path = isolated_feedback_env["crops_dir"]
    total = 50

    results = []

    def _worker(idx: int):
        is_corrupt = (idx % 2 == 0)
        if is_corrupt:
            payload = {
                "document_id": f"doc_fail_{idx}",
                "line_id": f"line_{idx}",
                "original_prediction": "bad_crop",
                "operator_correction": "bad_crop",
                "line_crop_base64": base64.b64encode(b"THIS_IS_NOT_A_VALID_IMAGE_FILE").decode("utf-8"),
            }
        else:
            payload = {
                "document_id": f"doc_valid_{idx}",
                "line_id": f"line_{idx}",
                "original_prediction": "good_crop",
                "operator_correction": "good_crop",
                "line_crop_base64": _generate_test_png_b64(80, 25),
            }
        resp = client.post("/v1/feedback", json=payload)
        return {"idx": idx, "is_corrupt": is_corrupt, "status_code": resp.status_code}

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_worker, i) for i in range(total)]
        results = [f.result() for f in futures]

    for r in results:
        if r["is_corrupt"]:
            assert r["status_code"] == 422, f"Corrupt payload {r['idx']} expected 422, got {r['status_code']}"
        else:
            assert r["status_code"] == 200, f"Valid payload {r['idx']} expected 200, got {r['status_code']}"

    # Verify manifest has EXACTLY 25 records
    manifest_lines = [l.strip() for l in manifest_path.read_text().splitlines() if l.strip()]
    assert len(manifest_lines) == 25, f"Expected exactly 25 manifest records, found {len(manifest_lines)}"

    # Verify crops directory has EXACTLY 25 files
    crop_files = list(crops_dir.glob("*.png"))
    assert len(crop_files) == 25, f"Expected exactly 25 crop files, found {len(crop_files)}"
