"""
pipeline/training/profiler.py
Fine-grained microsecond latency, memory, and throughput profiler for TrOCR training.
Measures data wait (T_data), device transfer (T_transfer), forward pass (T_fwd),
backward pass (T_bwd), and optimizer step (T_opt) with percentile statistics and telemetry reporting.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import io
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class StepTiming:
    """
    Granular timing and memory telemetry record for a single training step.
    All latency fields are stored in seconds.
    """
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    t_data: float = 0.0
    t_transfer: float = 0.0
    t_fwd: float = 0.0
    t_bwd: float = 0.0
    t_opt: float = 0.0
    total_step_time: float = 0.0
    t_step: float = 0.0
    num_samples: int = 0
    memory_allocated_mb: float = 0.0
    memory_peak_mb: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.t_data = max(0.0, float(self.t_data))
        self.t_transfer = max(0.0, float(self.t_transfer))
        self.t_fwd = max(0.0, float(self.t_fwd))
        self.t_bwd = max(0.0, float(self.t_bwd))
        self.t_opt = max(0.0, float(self.t_opt))
        self.num_samples = max(0, int(self.num_samples))
        self.memory_allocated_mb = max(0.0, float(self.memory_allocated_mb))
        self.memory_peak_mb = max(0.0, float(self.memory_peak_mb))

        computed_sum = self.t_data + self.t_transfer + self.t_fwd + self.t_bwd + self.t_opt
        if self.total_step_time <= 0.0:
            self.total_step_time = computed_sum
        if self.t_step <= 0.0:
            self.t_step = self.total_step_time

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _PhaseTimer:
    """Context manager for measuring execution time of individual pipeline phases."""

    def __init__(self, profiler: TrainingStepProfiler, phase_name: str) -> None:
        self.profiler = profiler
        self.phase_name = phase_name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> _PhaseTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed = max(0.0, time.perf_counter() - self.start_time)
        if self.phase_name == "data_wait":
            self.profiler.record_data_wait(self.elapsed)


class TrainingStepProfiler:
    """
    Fine-grained microsecond latency & throughput tracker for TrOCR training.
    Provides context managers for phase timings, records step breakdowns,
    computes latency percentiles (p50, p90, p95, p99), and exports JSON/CSV telemetry.
    """

    def __init__(
        self,
        warmup_steps: int = 2,
        device: Optional[Union[str, torch.device]] = None,
        enabled: bool = True,
    ) -> None:
        self.warmup_steps = max(0, int(warmup_steps))
        self.enabled = bool(enabled)
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cpu")

        self._lock = threading.Lock()
        self.step_records: List[Dict[str, Any]] = []
        self.timings: List[StepTiming] = []
        self.data_wait_times: List[float] = []
        self.total_samples: int = 0
        self._current_step_num: int = 0

    def profile_data_wait(self) -> _PhaseTimer:
        """Context manager measuring dataloader I/O wait latency."""
        return _PhaseTimer(self, "data_wait")

    def profile_transfer(self) -> _PhaseTimer:
        """Context manager measuring host-to-device tensor transfer latency."""
        return _PhaseTimer(self, "transfer")

    def profile_forward(self) -> _PhaseTimer:
        """Context manager measuring model forward pass latency."""
        return _PhaseTimer(self, "forward")

    def profile_backward(self) -> _PhaseTimer:
        """Context manager measuring loss backward pass latency."""
        return _PhaseTimer(self, "backward")

    def profile_optimizer(self) -> _PhaseTimer:
        """Context manager measuring optimizer and scaler step latency."""
        return _PhaseTimer(self, "optimizer")

    def profile_phase(self, phase_name: str) -> _PhaseTimer:
        """Generic context manager measuring named phase latency."""
        return _PhaseTimer(self, phase_name)

    def record_data_wait(self, t_sec: float) -> None:
        """Record dataloader wait latency in seconds."""
        if not self.enabled:
            return
        with self._lock:
            self.data_wait_times.append(max(0.0, float(t_sec)))

    def get_memory_allocated_mb(self) -> float:
        """Query current allocated memory on device in megabytes."""
        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                return float(torch.cuda.memory_allocated(self.device) / (1024 * 1024))
            elif self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                return float(torch.mps.current_allocated_memory() / (1024 * 1024))
        except Exception:
            pass
        return 0.0

    def record_step(
        self,
        t_data: float,
        t_transfer: float,
        t_fwd: float,
        t_bwd: float,
        t_opt: float,
        num_samples: int,
        memory_mb: Optional[float] = None,
        device_str: Optional[str] = None,
    ) -> StepTiming:
        """
        Record fine-grained microsecond latency breakdown for a completed training step.
        """
        if not self.enabled:
            return StepTiming(num_samples=num_samples)

        with self._lock:
            self._current_step_num += 1
            step_idx = self._current_step_num

            dev_str = device_str or str(self.device)
            current_mem = memory_mb if memory_mb is not None else self.get_memory_allocated_mb()
            peak_mem = current_mem

            timing = StepTiming(
                step=step_idx,
                timestamp=time.time(),
                t_data=t_data,
                t_transfer=t_transfer,
                t_fwd=t_fwd,
                t_bwd=t_bwd,
                t_opt=t_opt,
                num_samples=num_samples,
                memory_allocated_mb=current_mem,
                memory_peak_mb=peak_mem,
                device=dev_str,
            )

            rec = {
                "step": timing.step,
                "timestamp": timing.timestamp,
                "t_data": timing.t_data,
                "t_transfer": timing.t_transfer,
                "t_fwd": timing.t_fwd,
                "t_bwd": timing.t_bwd,
                "t_opt": timing.t_opt,
                "t_step": timing.t_step,
                "total_step_time": timing.total_step_time,
                "num_samples": timing.num_samples,
                "memory_allocated_mb": timing.memory_allocated_mb,
                "memory_peak_mb": timing.memory_peak_mb,
                "device": timing.device,
            }

            self.timings.append(timing)
            self.step_records.append(rec)
            self.total_samples += int(num_samples)
            return timing

    def get_summary(self) -> Dict[str, Any]:
        """
        Calculate summary telemetry statistics including samples/sec throughput,
        latency percentiles (p50, p90, p95, p99), and percentage breakdown.
        """
        with self._lock:
            if not self.step_records:
                return {
                    "total_steps": 0,
                    "total_samples": 0,
                    "samples_per_sec": 0.0,
                    "step_latency_ms": {
                        "mean": 0.0,
                        "p50": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                        "p99": 0.0,
                    },
                    "breakdown_ms": {
                        "data": 0.0,
                        "transfer": 0.0,
                        "fwd": 0.0,
                        "bwd": 0.0,
                        "opt": 0.0,
                    },
                    "breakdown_pct": {
                        "data": 0.0,
                        "transfer": 0.0,
                        "fwd": 0.0,
                        "bwd": 0.0,
                        "opt": 0.0,
                    },
                    "mean_data_wait_sec": float(np.mean(self.data_wait_times)) if self.data_wait_times else 0.0,
                    "memory_mb": {
                        "peak": 0.0,
                        "current": 0.0,
                    },
                    "warmup_steps": self.warmup_steps,
                    "active_steps": 0,
                }

            # Filter out warmup steps if sufficient records exist
            active_records = (
                self.step_records[self.warmup_steps:]
                if len(self.step_records) > self.warmup_steps
                else self.step_records
            )

            total_time_sec = sum(r["t_step"] for r in active_records)
            total_samples = sum(r["num_samples"] for r in active_records)
            samples_per_sec = (total_samples / total_time_sec) if total_time_sec > 1e-9 else 0.0

            step_latencies_ms = [r["t_step"] * 1000.0 for r in active_records]
            data_ms = [r["t_data"] * 1000.0 for r in active_records]
            transfer_ms = [r["t_transfer"] * 1000.0 for r in active_records]
            fwd_ms = [r["t_fwd"] * 1000.0 for r in active_records]
            bwd_ms = [r["t_bwd"] * 1000.0 for r in active_records]
            opt_ms = [r["t_opt"] * 1000.0 for r in active_records]

            mean_step_ms = float(np.mean(step_latencies_ms)) if step_latencies_ms else 0.0
            tot_ms = mean_step_ms if mean_step_ms > 1e-9 else 1.0

            mean_data = float(np.mean(data_ms)) if data_ms else 0.0
            mean_trans = float(np.mean(transfer_ms)) if transfer_ms else 0.0
            mean_fwd = float(np.mean(fwd_ms)) if fwd_ms else 0.0
            mean_bwd = float(np.mean(bwd_ms)) if bwd_ms else 0.0
            mean_opt = float(np.mean(opt_ms)) if opt_ms else 0.0

            mem_list = [r.get("memory_allocated_mb", 0.0) for r in self.step_records]
            peak_mem = float(max(mem_list)) if mem_list else 0.0
            curr_mem = float(mem_list[-1]) if mem_list else 0.0

            # Percentage calculation normalized to 100%
            sum_components = mean_data + mean_trans + mean_fwd + mean_bwd + mean_opt
            denom = sum_components if sum_components > 1e-9 else tot_ms
            pct_data = (mean_data / denom) * 100.0
            pct_trans = (mean_trans / denom) * 100.0
            pct_fwd = (mean_fwd / denom) * 100.0
            pct_bwd = (mean_bwd / denom) * 100.0
            pct_opt = (mean_opt / denom) * 100.0

            active_data_wait = (
                self.data_wait_times[self.warmup_steps:]
                if len(self.data_wait_times) > self.warmup_steps
                else self.data_wait_times
            )
            mean_wait = float(np.mean(active_data_wait)) if active_data_wait else 0.0

            return {
                "total_steps": len(self.step_records),
                "total_samples": self.total_samples,
                "samples_per_sec": float(samples_per_sec),
                "step_latency_ms": {
                    "mean": round(mean_step_ms, 3),
                    "p50": round(float(np.percentile(step_latencies_ms, 50)), 3),
                    "p90": round(float(np.percentile(step_latencies_ms, 90)), 3),
                    "p95": round(float(np.percentile(step_latencies_ms, 95)), 3),
                    "p99": round(float(np.percentile(step_latencies_ms, 99)), 3),
                },
                "breakdown_ms": {
                    "data": round(mean_data, 3),
                    "transfer": round(mean_trans, 3),
                    "fwd": round(mean_fwd, 3),
                    "bwd": round(mean_bwd, 3),
                    "opt": round(mean_opt, 3),
                },
                "breakdown_pct": {
                    "data": round(pct_data, 2),
                    "transfer": round(pct_trans, 2),
                    "fwd": round(pct_fwd, 2),
                    "bwd": round(pct_bwd, 2),
                    "opt": round(pct_opt, 2),
                },
                "mean_data_wait_sec": mean_wait,
                "memory_mb": {
                    "peak": round(peak_mem, 2),
                    "current": round(curr_mem, 2),
                },
                "warmup_steps": self.warmup_steps,
                "active_steps": len(active_records),
            }

    def format_summary(self) -> str:
        """Format summary telemetry into a readable ASCII table."""
        summary = self.get_summary()
        lat = summary["step_latency_ms"]
        bd_ms = summary["breakdown_ms"]
        bd_pct = summary["breakdown_pct"]
        mem = summary["memory_mb"]

        lines = [
            "=" * 78,
            "                   TRAINING STEP PROFILER TELEMETRY",
            "=" * 78,
            f" Total Steps:           {summary['total_steps']:<8} (Warmup: {summary['warmup_steps']}, Active: {summary['active_steps']})",
            f" Total Samples:         {summary['total_samples']:<8}",
            f" Throughput:            {summary['samples_per_sec']:.2f} samples/sec",
            "-" * 78,
            " LATENCY PERCENTILES (ms)",
            f"  Mean:                 {lat['mean']:>8.2f} ms",
            f"  p50:                  {lat['p50']:>8.2f} ms",
            f"  p90:                  {lat['p90']:>8.2f} ms",
            f"  p95:                  {lat['p95']:>8.2f} ms",
            f"  p99:                  {lat['p99']:>8.2f} ms",
            "-" * 78,
            " STEP LATENCY BREAKDOWN",
            f"  {'Phase':<22} {'Latency (ms)':>14} {'Percentage':>14}",
            f"  {'Data Wait (IO)':<22} {bd_ms['data']:>11.2f} ms {bd_pct['data']:>13.2f}%",
            f"  {'Device Transfer':<22} {bd_ms['transfer']:>11.2f} ms {bd_pct['transfer']:>13.2f}%",
            f"  {'Forward Pass':<22} {bd_ms['fwd']:>11.2f} ms {bd_pct['fwd']:>13.2f}%",
            f"  {'Backward Pass':<22} {bd_ms['bwd']:>11.2f} ms {bd_pct['bwd']:>13.2f}%",
            f"  {'Optimizer Step':<22} {bd_ms['opt']:>11.2f} ms {bd_pct['opt']:>13.2f}%",
            "-" * 78,
            " HARDWARE & MEMORY",
            f"  Device:               {str(self.device)}",
            f"  Mean Data Wait:       {summary['mean_data_wait_sec'] * 1000.0:.2f} ms",
            f"  Current Allocated:    {mem['current']:.2f} MB",
            f"  Peak Allocated:       {mem['peak']:.2f} MB",
            "=" * 78,
        ]
        return "\n".join(lines)

    def export_json(self, filepath: Union[str, Path], indent: int = 2) -> str:
        """Export telemetry summary and step records to a JSON file."""
        summary = self.get_summary()
        payload = {
            "summary": summary,
            "step_records": self.step_records,
        }
        json_str = json.dumps(payload, indent=indent)
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_str, encoding="utf-8")
        return json_str

    def export_csv(self, filepath: Union[str, Path]) -> str:
        """Export granular step timings to a CSV file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "step",
            "timestamp",
            "t_data",
            "t_transfer",
            "t_fwd",
            "t_bwd",
            "t_opt",
            "t_step",
            "num_samples",
            "memory_allocated_mb",
            "device",
        ]

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in self.step_records:
                writer.writerow(record)

        return p.read_text(encoding="utf-8")

    def reset(self) -> None:
        """Reset all accumulated timing and memory records."""
        with self._lock:
            self.step_records.clear()
            self.timings.clear()
            self.data_wait_times.clear()
            self.total_samples = 0
            self._current_step_num = 0
