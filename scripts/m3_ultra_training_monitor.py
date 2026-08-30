#!/usr/bin/env python3
"""
scripts/m3_ultra_training_monitor.py
Real-time diagnostic monitor and telemetry logger for Apple M3 Ultra training.
Tracks:
- CPU utilization across all 28 cores
- Unified Memory (RAM) allocation
- MPS GPU training progress, steps, throughput (samples/sec)
- Loss curve convergence and ETA
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "trocr-large-m3-saturated"
LOSSES_CSV = OUTPUT_DIR / "losses.csv"
STATE_JSON = OUTPUT_DIR / "training_state.json"
MONITOR_LOG = OUTPUT_DIR / "training_monitor.log"

def get_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_per_core = psutil.cpu_percent(percpu=True)
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    return {
        "overall_cpu_pct": cpu_percent,
        "active_cores": sum(1 for c in cpu_per_core if c > 15.0),
        "total_cores": len(cpu_per_core),
        "ram_used_gb": round(used_gb, 2),
        "ram_total_gb": round(total_gb, 2),
        "ram_pct": mem.percent
    }

def read_latest_loss():
    step_csv = OUTPUT_DIR / "step_losses.csv"
    if step_csv.exists():
        try:
            lines = step_csv.read_text().strip().split("\n")
            if len(lines) > 1:
                header = lines[0].split(",")
                last_row = lines[-1].split(",")
                data = dict(zip(header, last_row))
                return {
                    "step": int(data.get("step", 0)),
                    "loss": float(data.get("train_loss", data.get("loss", 0.0))),
                    "lr": float(data.get("learning_rate", data.get("lr", 0.0))),
                    "epoch": float(data.get("epoch", 1)),
                    "total_logged_steps": len(lines) - 1
                }
        except Exception:
            pass

    if not LOSSES_CSV.exists():
        return None
    try:
        lines = LOSSES_CSV.read_text().strip().split("\n")
        if len(lines) <= 1:
            return None
        header = lines[0].split(",")
        last_row = lines[-1].split(",")
        data = dict(zip(header, last_row))
        return {
            "step": int(data.get("step", 0)),
            "loss": float(data.get("train_loss", data.get("loss", 0.0))),
            "lr": float(data.get("learning_rate", data.get("lr", 0.0))),
            "epoch": float(data.get("epoch", 1)),
            "total_logged_steps": len(lines) - 1
        }
    except Exception:
        return None

def compute_throughput_and_eta(latest_step, start_time, start_step, batch_size=64, total_steps=14000):
    elapsed = max(1.0, time.time() - start_time)
    steps_done = max(1, latest_step - start_step)
    steps_per_sec = steps_done / elapsed
    samples_per_sec = steps_per_sec * batch_size
    remaining_steps = max(0, total_steps - latest_step)
    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
    eta_mins = eta_seconds / 60.0
    return {
        "samples_per_sec": round(samples_per_sec, 2),
        "steps_per_sec": round(steps_per_sec, 2),
        "eta_mins": round(eta_mins, 1),
        "elapsed_mins": round(elapsed / 60.0, 1)
    }

def print_telemetry_snapshot():
    metrics = get_system_metrics()
    loss_info = read_latest_loss()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    print(f"=== [M3 ULTRA TELEMETRY HEARTBEAT: {timestamp}] ===")
    print(f"• CPU Utilization: {metrics['overall_cpu_pct']}% across {metrics['active_cores']}/{metrics['total_cores']} active cores")
    print(f"• Unified Memory:  {metrics['ram_used_gb']} GB / {metrics['ram_total_gb']} GB ({metrics['ram_pct']}%)")
    if loss_info:
        print(f"• Training Step:   Step {loss_info['step']} | Epoch {loss_info['epoch']:.2f}")
        print(f"• Current Loss:    {loss_info['loss']:.4f} | LR: {loss_info['lr']:.2e}")
    else:
        print("• Training Step:   Initializing worker pool / DataLoader...")
    print("=" * 60)

if __name__ == "__main__":
    print_telemetry_snapshot()
