"""
pipeline/tests/test_profiler_empirical.py
Adversarial empirical stress tests for TrainingStepProfiler (Requirement R3 / Feature F9).
Evaluates:
1. Edge cases (0 steps, 1 step, warmup boundaries, microsecond extremes, zero division)
2. Mathematical percentile accuracy against numpy reference implementations
3. Multithreaded concurrency & thread safety stress (up to 32 threads, concurrent readers/writers)
4. Export resilience (deep non-existent paths, empty exports, large-scale event loads, schema validation)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from pipeline.training.profiler import StepTiming, TrainingStepProfiler


class TestEdgeCasesAndZeroProtection:
    """Stress-test zero-state, boundary warmup, microsecond extremes, and division by zero."""

    def test_zero_steps_metrics_and_formatting(self, tmp_path: Path) -> None:
        """Verify profiler initialized with 0 steps returns safe zero values and exports cleanly."""
        profiler = TrainingStepProfiler(warmup_steps=2)

        # Summary check
        summary = profiler.get_summary()
        assert summary["total_steps"] == 0
        assert summary["total_samples"] == 0
        assert summary["samples_per_sec"] == 0.0
        assert summary["active_steps"] == 0
        assert summary["step_latency_ms"]["mean"] == 0.0
        assert summary["step_latency_ms"]["p50"] == 0.0
        assert summary["step_latency_ms"]["p90"] == 0.0
        assert summary["step_latency_ms"]["p95"] == 0.0
        assert summary["step_latency_ms"]["p99"] == 0.0
        assert summary["breakdown_ms"]["data"] == 0.0
        assert summary["breakdown_pct"]["data"] == 0.0
        assert summary["mean_data_wait_sec"] == 0.0
        assert summary["memory_mb"]["peak"] == 0.0
        assert summary["memory_mb"]["current"] == 0.0

        # Formatted string check
        table_str = profiler.format_summary()
        assert "TRAINING STEP PROFILER TELEMETRY" in table_str
        assert "Total Steps:           0" in table_str

        # Export checks
        json_path = tmp_path / "zero_steps.json"
        csv_path = tmp_path / "zero_steps.csv"

        profiler.export_json(json_path)
        profiler.export_csv(csv_path)

        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["total_steps"] == 0
        assert loaded["step_records"] == []

        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 0

    def test_single_step_fewer_than_warmup(self) -> None:
        """Verify when step count is <= warmup_steps, all steps are used for active stats without crashing."""
        profiler = TrainingStepProfiler(warmup_steps=5)
        profiler.record_step(
            t_data=0.010,
            t_transfer=0.005,
            t_fwd=0.040,
            t_bwd=0.035,
            t_opt=0.010,
            num_samples=8,
            memory_mb=100.0,
        )

        summary = profiler.get_summary()
        assert summary["total_steps"] == 1
        assert summary["total_samples"] == 8
        assert summary["active_steps"] == 1  # Gracefully falls back to 1 active step
        assert np.isclose(summary["step_latency_ms"]["mean"], 100.0, atol=1e-2)
        assert np.isclose(summary["step_latency_ms"]["p50"], 100.0, atol=1e-2)
        assert np.isclose(summary["step_latency_ms"]["p99"], 100.0, atol=1e-2)
        assert np.isclose(summary["samples_per_sec"], 80.0, atol=1.0)

    def test_exact_warmup_steps_boundary(self) -> None:
        """Verify behavior when total steps exactly equals warmup_steps."""
        profiler = TrainingStepProfiler(warmup_steps=3)
        for _ in range(3):
            profiler.record_step(0.005, 0.005, 0.020, 0.020, 0.010, num_samples=4)

        summary = profiler.get_summary()
        assert summary["total_steps"] == 3
        # Since len <= warmup_steps, it keeps all 3 records as active
        assert summary["active_steps"] == 3
        assert np.isclose(summary["step_latency_ms"]["mean"], 60.0, atol=1e-2)

        # Adding 1 more step transitions to filtering
        profiler.record_step(0.005, 0.005, 0.020, 0.020, 0.010, num_samples=4)
        summary2 = profiler.get_summary()
        assert summary2["total_steps"] == 4
        assert summary2["active_steps"] == 1  # 4 - 3 = 1

    def test_warmup_steps_zero_and_negative(self) -> None:
        """Verify non-positive warmup_steps are sanitized properly."""
        p_zero = TrainingStepProfiler(warmup_steps=0)
        assert p_zero.warmup_steps == 0

        p_neg = TrainingStepProfiler(warmup_steps=-5)
        assert p_neg.warmup_steps == 0

    def test_extreme_microsecond_latencies(self) -> None:
        """Verify nanosecond and microsecond extreme latencies do not cause underflow or division issues."""
        profiler = TrainingStepProfiler(warmup_steps=0)

        # 1 microsecond per phase = 5 microseconds total (5e-6 s)
        profiler.record_step(
            t_data=1e-6,
            t_transfer=1e-6,
            t_fwd=1e-6,
            t_bwd=1e-6,
            t_opt=1e-6,
            num_samples=16,
        )

        summary = profiler.get_summary()
        assert summary["total_steps"] == 1
        assert summary["total_samples"] == 16
        assert summary["samples_per_sec"] > 1_000_000.0  # 16 / 5e-6 = 3.2M samples/sec
        assert summary["step_latency_ms"]["mean"] >= 0.001

    def test_sub_microsecond_nanosecond_latencies(self) -> None:
        """Verify sub-microsecond latencies down to 10ns do not crash profiler."""
        profiler = TrainingStepProfiler(warmup_steps=0)
        profiler.record_step(1e-8, 1e-8, 1e-8, 1e-8, 1e-8, num_samples=32)

        summary = profiler.get_summary()
        assert summary["total_steps"] == 1
        assert summary["samples_per_sec"] > 1e8
        assert summary["step_latency_ms"]["mean"] >= 0.0

    def test_zero_time_and_zero_samples_div_zero_protection(self) -> None:
        """Verify recording steps with zero time and zero samples never raises ZeroDivisionError."""
        profiler = TrainingStepProfiler(warmup_steps=0)

        profiler.record_step(
            t_data=0.0,
            t_transfer=0.0,
            t_fwd=0.0,
            t_bwd=0.0,
            t_opt=0.0,
            num_samples=0,
            memory_mb=0.0,
        )

        summary = profiler.get_summary()
        assert summary["total_steps"] == 1
        assert summary["total_samples"] == 0
        assert summary["samples_per_sec"] == 0.0
        assert summary["step_latency_ms"]["mean"] == 0.0
        assert summary["breakdown_pct"]["data"] == 0.0
        assert summary["breakdown_pct"]["fwd"] == 0.0

        table_str = profiler.format_summary()
        assert "Throughput:            0.00 samples/sec" in table_str

    def test_disabled_profiler_operates_as_noop(self) -> None:
        """Verify profiler with enabled=False incurs no recording overhead."""
        profiler = TrainingStepProfiler(enabled=False)
        profiler.record_data_wait(0.5)
        timing = profiler.record_step(0.01, 0.01, 0.05, 0.05, 0.01, num_samples=8)

        assert timing.num_samples == 8
        assert len(profiler.step_records) == 0
        assert len(profiler.data_wait_times) == 0
        assert profiler.total_samples == 0

        summary = profiler.get_summary()
        assert summary["total_steps"] == 0
        assert summary["samples_per_sec"] == 0.0


class TestPercentileAccuracyEmpirical:
    """Mathematically verify percentiles across various sample distributions vs numpy."""

    def test_percentiles_uniform_distribution(self) -> None:
        """Verify p50, p90, p95, p99 match numpy across 100 uniformly spaced latencies."""
        profiler = TrainingStepProfiler(warmup_steps=0)
        latencies_sec = [i * 0.001 for i in range(1, 101)]
        latencies_ms = [l * 1000.0 for l in latencies_sec]

        for lat in latencies_sec:
            profiler.record_step(lat / 5, lat / 5, lat / 5, lat / 5, lat / 5, num_samples=1)

        summary = profiler.get_summary()
        lat_stats = summary["step_latency_ms"]

        ref_mean = round(float(np.mean(latencies_ms)), 3)
        ref_p50 = round(float(np.percentile(latencies_ms, 50)), 3)
        ref_p90 = round(float(np.percentile(latencies_ms, 90)), 3)
        ref_p95 = round(float(np.percentile(latencies_ms, 95)), 3)
        ref_p99 = round(float(np.percentile(latencies_ms, 99)), 3)

        assert lat_stats["mean"] == ref_mean
        assert lat_stats["p50"] == ref_p50
        assert lat_stats["p90"] == ref_p90
        assert lat_stats["p95"] == ref_p95
        assert lat_stats["p99"] == ref_p99

    def test_percentiles_gaussian_distribution(self) -> None:
        """Verify percentiles against numpy reference on 1000 normal random latencies."""
        np.random.seed(42)
        latencies_ms = np.clip(np.random.normal(loc=50.0, scale=10.0, size=1000), 5.0, 150.0)
        latencies_sec = latencies_ms / 1000.0

        profiler = TrainingStepProfiler(warmup_steps=0)
        for lat in latencies_sec:
            profiler.record_step(lat * 0.1, lat * 0.1, lat * 0.4, lat * 0.3, lat * 0.1, num_samples=4)

        summary = profiler.get_summary()
        lat_stats = summary["step_latency_ms"]

        assert lat_stats["mean"] == round(float(np.mean(latencies_ms)), 3)
        assert lat_stats["p50"] == round(float(np.percentile(latencies_ms, 50)), 3)
        assert lat_stats["p90"] == round(float(np.percentile(latencies_ms, 90)), 3)
        assert lat_stats["p95"] == round(float(np.percentile(latencies_ms, 95)), 3)
        assert lat_stats["p99"] == round(float(np.percentile(latencies_ms, 99)), 3)

    def test_percentiles_heavy_tail_distribution(self) -> None:
        """Verify tail percentiles (p95, p99) accurately detect sudden latency spikes."""
        profiler = TrainingStepProfiler(warmup_steps=0)
        latencies_ms = [10.0] * 990 + [1000.0] * 10
        for ms in latencies_ms:
            sec = ms / 1000.0
            profiler.record_step(sec / 5, sec / 5, sec / 5, sec / 5, sec / 5, num_samples=1)

        summary = profiler.get_summary()
        lat = summary["step_latency_ms"]

        assert lat["p50"] == 10.0
        assert lat["p90"] == 10.0
        assert lat["p99"] > 10.0
        assert lat["p99"] == round(float(np.percentile(latencies_ms, 99)), 3)

    def test_percentiles_bimodal_distribution(self) -> None:
        """Verify percentiles on bimodal distribution (fast batches + periodic validation/checkpoint steps)."""
        profiler = TrainingStepProfiler(warmup_steps=0)
        # 500 fast steps of 15ms, 500 slow steps of 85ms
        latencies_ms = [15.0] * 500 + [85.0] * 500
        for ms in latencies_ms:
            sec = ms / 1000.0
            profiler.record_step(sec / 5, sec / 5, sec / 5, sec / 5, sec / 5, num_samples=2)

        summary = profiler.get_summary()
        lat = summary["step_latency_ms"]

        assert lat["mean"] == 50.0
        assert lat["p50"] == round(float(np.percentile(latencies_ms, 50)), 3)
        assert lat["p90"] == 85.0
        assert lat["p99"] == 85.0

    def test_percentiles_constant_and_small_n(self) -> None:
        """Verify percentiles on constant series and small sample sizes (N=1, N=2)."""
        # Constant series
        p_const = TrainingStepProfiler(warmup_steps=0)
        for _ in range(50):
            p_const.record_step(0.01, 0.01, 0.01, 0.01, 0.01, num_samples=2)
        s_const = p_const.get_summary()
        assert s_const["step_latency_ms"]["p50"] == 50.0
        assert s_const["step_latency_ms"]["p99"] == 50.0
        assert s_const["step_latency_ms"]["mean"] == 50.0

        # N=1
        p1 = TrainingStepProfiler(warmup_steps=0)
        p1.record_step(0.02, 0.02, 0.02, 0.02, 0.02, num_samples=4)
        s1 = p1.get_summary()
        assert s1["step_latency_ms"]["p50"] == 100.0
        assert s1["step_latency_ms"]["p99"] == 100.0

        # N=2
        p2 = TrainingStepProfiler(warmup_steps=0)
        p2.record_step(0.01, 0.0, 0.0, 0.0, 0.0, num_samples=1)
        p2.record_step(0.02, 0.0, 0.0, 0.0, 0.0, num_samples=1)
        s2 = p2.get_summary()
        assert s2["step_latency_ms"]["p50"] == round(float(np.percentile([10.0, 20.0], 50)), 3)
        assert s2["step_latency_ms"]["p99"] == round(float(np.percentile([10.0, 20.0], 99)), 3)


class TestThreadSafetyAndConcurrencyStress:
    """Stress test multithreaded recording, reading, and synchronization under high concurrency."""

    def test_concurrent_multiworker_recording(self) -> None:
        """Verify 16 concurrent threads recording 200 steps each maintain sequence and sum integrity."""
        profiler = TrainingStepProfiler(warmup_steps=0)
        num_threads = 16
        steps_per_thread = 200
        total_expected_steps = num_threads * steps_per_thread

        def worker(worker_id: int) -> None:
            for i in range(steps_per_thread):
                t_fwd = 0.010 + (random.random() * 0.005)
                profiler.record_data_wait(0.002)
                profiler.record_step(
                    t_data=0.002,
                    t_transfer=0.001,
                    t_fwd=t_fwd,
                    t_bwd=0.015,
                    t_opt=0.005,
                    num_samples=4,
                    memory_mb=128.0 + worker_id,
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = profiler.get_summary()
        assert summary["total_steps"] == total_expected_steps
        assert summary["total_samples"] == total_expected_steps * 4
        assert len(profiler.step_records) == total_expected_steps
        assert len(profiler.data_wait_times) == total_expected_steps

        # Verify step IDs are strictly 1..total_expected_steps with no duplicates
        recorded_step_ids = sorted([r["step"] for r in profiler.step_records])
        assert recorded_step_ids == list(range(1, total_expected_steps + 1))

    def test_high_concurrency_32_threads_stress(self) -> None:
        """Stress test 32 concurrent threads executing recording and context managers simultaneously."""
        profiler = TrainingStepProfiler(warmup_steps=5)
        num_threads = 32
        steps_per_thread = 100
        total_steps = num_threads * steps_per_thread

        def thread_fn(tid: int) -> None:
            for _ in range(steps_per_thread):
                with profiler.profile_data_wait() as dt:
                    time.sleep(0.00001)
                profiler.record_step(
                    t_data=dt.elapsed,
                    t_transfer=0.0001,
                    t_fwd=0.001,
                    t_bwd=0.001,
                    t_opt=0.0002,
                    num_samples=2,
                    memory_mb=64.0 + tid,
                )

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = profiler.get_summary()
        assert summary["total_steps"] == total_steps
        assert summary["total_samples"] == total_steps * 2
        assert len(profiler.data_wait_times) == total_steps

    def test_concurrent_readers_and_writers_interleaved(self, tmp_path: Path) -> None:
        """Verify simultaneous read and write operations do not deadlock or throw race exceptions."""
        profiler = TrainingStepProfiler(warmup_steps=2)
        stop_event = threading.Event()
        errors: List[Exception] = []

        def writer_task() -> None:
            try:
                for _ in range(300):
                    if stop_event.is_set():
                        break
                    profiler.record_step(0.002, 0.001, 0.010, 0.010, 0.002, num_samples=2)
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        def reader_task(idx: int) -> None:
            try:
                for _ in range(100):
                    if stop_event.is_set():
                        break
                    s = profiler.get_summary()
                    _ = profiler.format_summary()
                    if s["total_steps"] > 0 and idx % 2 == 0:
                        profiler.export_json(tmp_path / f"thread_reader_{idx}.json")
                    time.sleep(0.0003)
            except Exception as e:
                errors.append(e)

        writers = [threading.Thread(target=writer_task) for _ in range(4)]
        readers = [threading.Thread(target=reader_task, args=(i,)) for i in range(4)]

        for t in writers + readers:
            t.start()
        for t in writers + readers:
            t.join()

        assert len(errors) == 0, f"Thread errors encountered: {errors}"
        summary = profiler.get_summary()
        assert summary["total_steps"] > 0


class TestExportResilienceStress:
    """Stress-test telemetry serialization, deep nested paths, and large volume data handling."""

    def test_export_to_deep_nonexistent_directory(self, tmp_path: Path) -> None:
        """Verify export_json and export_csv automatically create deep directory structures."""
        profiler = TrainingStepProfiler(warmup_steps=0)
        profiler.record_step(0.005, 0.002, 0.020, 0.025, 0.008, num_samples=8, memory_mb=256.0)

        nested_json = tmp_path / "deep" / "nested" / "path" / "level3" / "summary.json"
        nested_csv = tmp_path / "deep" / "nested" / "path" / "level3" / "timings.csv"

        profiler.export_json(nested_json)
        profiler.export_csv(nested_csv)

        assert nested_json.exists()
        assert nested_csv.exists()

        data = json.loads(nested_json.read_text(encoding="utf-8"))
        assert data["summary"]["total_steps"] == 1
        assert len(data["step_records"]) == 1

        with open(nested_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            assert float(reader[0]["t_fwd"]) == 0.020

    def test_large_event_volume_stress(self, tmp_path: Path) -> None:
        """Stress-test profiler with 20,000 steps measuring serialization throughput and integrity."""
        profiler = TrainingStepProfiler(warmup_steps=10)
        num_steps = 20_000

        t0 = time.perf_counter()
        for i in range(num_steps):
            profiler.record_step(
                t_data=0.001,
                t_transfer=0.001,
                t_fwd=0.020,
                t_bwd=0.020,
                t_opt=0.005,
                num_samples=8,
                memory_mb=512.0,
            )
        recording_time = time.perf_counter() - t0

        # Verify recording throughput: > 20,000 steps/sec recording speed
        assert (num_steps / recording_time) > 20_000.0

        # Verify summary computation speed on 20,000 steps
        t1 = time.perf_counter()
        summary = profiler.get_summary()
        summary_time = time.perf_counter() - t1
        assert summary_time < 0.20  # Summary takes < 200ms
        assert summary["total_steps"] == num_steps
        assert summary["active_steps"] == num_steps - 10

        # Export and verify JSON
        json_file = tmp_path / "large_stress.json"
        csv_file = tmp_path / "large_stress.csv"

        t2 = time.perf_counter()
        profiler.export_json(json_file)
        json_export_time = time.perf_counter() - t2
        assert json_export_time < 2.0
        assert json_file.stat().st_size > 2_000_000

        # Export and verify CSV
        t3 = time.perf_counter()
        profiler.export_csv(csv_file)
        csv_export_time = time.perf_counter() - t3
        assert csv_export_time < 2.0
        assert csv_file.stat().st_size > 1_000_000

        # Round-trip verification
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(loaded["step_records"]) == num_steps
        assert loaded["step_records"][-1]["step"] == num_steps

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == num_steps
            assert int(reader[-1]["step"]) == num_steps
