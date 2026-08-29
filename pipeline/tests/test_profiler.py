"""
pipeline/tests/test_profiler.py
Unit and integration test suite for TrainingStepProfiler (Requirement R3 / Feature F9).
Tests microsecond breakdown metrics, latency percentiles, memory telemetry,
context managers, thread safety, JSON/CSV exports, and trainer integration.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.training.config import CurriculumConfig, CurriculumStageConfig, TrainingConfig
from pipeline.training.curriculum import MultiStageCurriculumTrainer
from pipeline.training.dataset import DummyTokenizer, create_dummy_processor
from pipeline.training.profiler import StepTiming, TrainingStepProfiler
from pipeline.training.train import TrOCRTrainer, create_tiny_mock_model


class TestStepTimingDataclass:
    """Test StepTiming initialization, sums, sanitization, and serialization."""

    def test_step_timing_default_and_computed_sum(self) -> None:
        """Verify total_step_time is automatically computed from phase latencies."""
        timing = StepTiming(
            step=1,
            t_data=0.010,
            t_transfer=0.005,
            t_fwd=0.040,
            t_bwd=0.035,
            t_opt=0.010,
            num_samples=8,
            memory_allocated_mb=128.5,
            device="mps",
        )
        assert timing.step == 1
        assert np.isclose(timing.total_step_time, 0.100, atol=1e-5)
        assert np.isclose(timing.t_step, 0.100, atol=1e-5)
        assert timing.num_samples == 8
        assert timing.memory_allocated_mb == 128.5
        assert timing.device == "mps"

    def test_step_timing_negative_sanitization(self) -> None:
        """Verify negative timing values are clamped to 0.0."""
        timing = StepTiming(
            step=2,
            t_data=-0.05,
            t_transfer=-0.01,
            t_fwd=0.02,
            t_bwd=0.02,
            t_opt=-0.01,
            num_samples=-4,
            memory_allocated_mb=-10.0,
        )
        assert timing.t_data == 0.0
        assert timing.t_transfer == 0.0
        assert timing.t_opt == 0.0
        assert timing.num_samples == 0
        assert timing.memory_allocated_mb == 0.0
        assert np.isclose(timing.total_step_time, 0.04, atol=1e-5)

    def test_step_timing_to_dict_serialization(self) -> None:
        """Verify to_dict returns serializable dictionary with all expected keys."""
        timing = StepTiming(step=5, t_fwd=0.05, num_samples=4)
        d = timing.to_dict()
        assert isinstance(d, dict)
        assert d["step"] == 5
        assert d["t_fwd"] == 0.05
        assert "timestamp" in d
        assert "total_step_time" in d
        assert "memory_allocated_mb" in d


class TestTrainingStepProfilerCore:
    """Test profiler metrics, latency percentiles, throughput, and context managers."""

    def test_profiler_record_step_and_throughput(self) -> None:
        """Verify profiler computes accurate throughput (samples/sec) and latency breakdown."""
        p = TrainingStepProfiler(warmup_steps=1)

        for _ in range(5):
            p.record_data_wait(0.002)
            p.record_step(
                t_data=0.002,
                t_transfer=0.003,
                t_fwd=0.040,
                t_bwd=0.045,
                t_opt=0.010,
                num_samples=8,
                memory_mb=256.0,
            )

        summary = p.get_summary()
        assert summary["total_steps"] == 5
        assert summary["total_samples"] == 40
        assert summary["active_steps"] == 4  # 5 minus 1 warmup
        assert summary["samples_per_sec"] > 0.0
        assert np.isclose(summary["samples_per_sec"], 8.0 / 0.100, atol=5.0)

        # Breakdown checks
        bd = summary["breakdown_ms"]
        assert bd["data"] > 0.0
        assert bd["transfer"] > 0.0
        assert bd["fwd"] > 0.0
        assert bd["bwd"] > 0.0
        assert bd["opt"] > 0.0

        # Percentage sum check
        pct = summary["breakdown_pct"]
        assert np.isclose(sum(pct.values()), 100.0, atol=0.1)

    def test_profiler_latency_percentiles(self) -> None:
        """Verify p50, p90, p95, p99 percentiles are calculated properly."""
        p = TrainingStepProfiler(warmup_steps=0)

        # Feed increasing latencies
        for i in range(1, 101):
            p.record_step(0.001 * i, 0.0, 0.0, 0.0, 0.0, num_samples=1)

        summary = p.get_summary()
        lat = summary["step_latency_ms"]
        assert lat["p50"] <= lat["p90"] <= lat["p95"] <= lat["p99"]
        assert np.isclose(lat["p50"], 50.5, atol=2.0)
        assert np.isclose(lat["p90"], 90.1, atol=2.0)

    def test_profiler_phase_timer_context_managers(self) -> None:
        """Verify context managers accurately measure phase elapsed times."""
        p = TrainingStepProfiler(warmup_steps=0)

        with p.profile_data_wait() as timer:
            time.sleep(0.01)
        assert timer.elapsed >= 0.008
        assert len(p.data_wait_times) == 1

        with p.profile_forward() as fwd_timer:
            time.sleep(0.01)
        assert fwd_timer.elapsed >= 0.008

        with p.profile_backward() as bwd_timer:
            time.sleep(0.01)
        assert bwd_timer.elapsed >= 0.008

        with p.profile_optimizer() as opt_timer:
            time.sleep(0.005)
        assert opt_timer.elapsed >= 0.004

    def test_profiler_zero_steps_no_division_error(self) -> None:
        """Verify empty profiler returns structured 0.0 metrics without crashing."""
        p = TrainingStepProfiler()
        summary = p.get_summary()
        assert summary["total_steps"] == 0
        assert summary["total_samples"] == 0
        assert summary["samples_per_sec"] == 0.0
        assert summary["step_latency_ms"]["mean"] == 0.0
        assert summary["step_latency_ms"]["p50"] == 0.0
        assert summary["breakdown_pct"]["data"] == 0.0

    def test_profiler_reset(self) -> None:
        """Verify reset clears all recorded step and wait records."""
        p = TrainingStepProfiler()
        p.record_data_wait(0.05)
        p.record_step(0.01, 0.01, 0.01, 0.01, 0.01, num_samples=4)
        assert len(p.step_records) == 1
        assert len(p.data_wait_times) == 1

        p.reset()
        assert len(p.step_records) == 0
        assert len(p.timings) == 0
        assert len(p.data_wait_times) == 0
        assert p.total_samples == 0
        assert p._current_step_num == 0

    def test_profiler_format_summary_table(self) -> None:
        """Verify format_summary returns formatted table string with expected content."""
        p = TrainingStepProfiler(warmup_steps=1)
        for _ in range(3):
            p.record_step(0.005, 0.005, 0.020, 0.020, 0.005, num_samples=4, memory_mb=150.0)

        table_str = p.format_summary()
        assert "TRAINING STEP PROFILER TELEMETRY" in table_str
        assert "Throughput:" in table_str
        assert "LATENCY PERCENTILES" in table_str
        assert "STEP LATENCY BREAKDOWN" in table_str
        assert "Data Wait (IO)" in table_str
        assert "Forward Pass" in table_str

    def test_profiler_export_json_and_csv(self, tmp_path: Path) -> None:
        """Verify export_json and export_csv write valid persistent telemetry files."""
        p = TrainingStepProfiler(warmup_steps=0)
        p.record_step(0.001, 0.002, 0.010, 0.010, 0.002, num_samples=8, memory_mb=120.0)
        p.record_step(0.001, 0.002, 0.012, 0.011, 0.002, num_samples=8, memory_mb=125.0)

        json_file = tmp_path / "profiler_summary.json"
        csv_file = tmp_path / "step_timings.csv"

        p.export_json(json_file)
        p.export_csv(csv_file)

        # Assert JSON
        assert json_file.exists()
        loaded_json = json.loads(json_file.read_text(encoding="utf-8"))
        assert "summary" in loaded_json
        assert loaded_json["summary"]["total_steps"] == 2
        assert loaded_json["summary"]["total_samples"] == 16
        assert len(loaded_json["step_records"]) == 2

        # Assert CSV
        assert csv_file.exists()
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            assert "t_fwd" in reader[0]
            assert "memory_allocated_mb" in reader[0]

    def test_profiler_thread_safety(self) -> None:
        """Verify concurrent multi-threaded step recording does not cause race conditions."""
        p = TrainingStepProfiler(warmup_steps=0)
        threads = []

        def worker() -> None:
            for _ in range(50):
                p.record_data_wait(0.001)
                p.record_step(0.001, 0.001, 0.005, 0.005, 0.001, num_samples=2)

        for _ in range(4):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        summary = p.get_summary()
        assert summary["total_steps"] == 200
        assert summary["total_samples"] == 400
        assert len(p.data_wait_times) == 200


class TestProfilerTrainerIntegration:
    """Test TrainingStepProfiler integration inside TrOCRTrainer and MultiStageCurriculumTrainer."""

    def test_trainer_profiler_enabled_by_default(self, tmp_path: Path) -> None:
        """Verify TrOCRTrainer initializes profiler when enable_step_profiling=True."""
        cfg = TrainingConfig(
            output_dir=str(tmp_path / "ckpt"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            enable_step_profiling=True,
        )
        trainer = TrOCRTrainer(config=cfg, model=create_tiny_mock_model(vocab_size=64))
        assert trainer.profiler is not None
        assert isinstance(trainer.profiler, TrainingStepProfiler)

    def test_trainer_profiler_disabled(self, tmp_path: Path) -> None:
        """Verify TrOCRTrainer sets profiler=None when enable_step_profiling=False."""
        cfg = TrainingConfig(
            output_dir=str(tmp_path / "ckpt"),
            device="cpu",
            enable_step_profiling=False,
        )
        trainer = TrOCRTrainer(config=cfg, model=create_tiny_mock_model(vocab_size=64))
        assert trainer.profiler is None

    def test_curriculum_profiler_attachment(self, tmp_path: Path) -> None:
        """Verify MultiStageCurriculumTrainer attaches profiler when configured."""
        stage = CurriculumStageConfig(
            stage_name="stage1",
            dataset_manifest="",
            num_epochs=1,
            micro_batch_size=2,
        )
        cfg = CurriculumConfig(
            root_output_dir=str(tmp_path / "curr"),
            device="cpu",
            enable_step_profiling=True,
            stages=[stage],
        )
        trainer = MultiStageCurriculumTrainer(config=cfg, model=create_tiny_mock_model(vocab_size=64))
        assert trainer.profiler is not None
        assert isinstance(trainer.profiler, TrainingStepProfiler)
