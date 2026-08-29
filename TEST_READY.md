# Test Ready: TrOCR Handwriting Training Acceleration

## Overview
This document publishes the complete specification, execution instructions, tier breakdown, and verification criteria for the TrOCR Handwriting Training Acceleration test suite (Milestone M0: E2E Testing Track).

All tests are implemented with genuine tensor operations, model forward/backward graphs, and profiling telemetry adhering to opaque-box testing standards.

---

## Test Inventory & Feature Coverage Mapping

| # | Feature | Requirement | Tier 1 (Isolation) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workload) |
|---|---------|:-----------:|:------------------:|:-----------------:|:-----------------:|:-----------------:|
| F1 | `AsyncDevicePrefetcher` | R1 | 7 tests | 3 tests | ✓ (P01, P02, P06, P08) | ✓ (S01, S02) |
| F2 | `MMapOCRDataset` & Memory Mapping | R1 | 5 tests | 2 tests | ✓ (P01, P07, P08) | ✓ (S01) |
| F3 | Multi-Worker Multiprocessing | R1 | 5 tests | 2 tests | ✓ (P01, P04, P08) | ✓ (S01) |
| F4 | Curriculum Multi-Worker Support | R1 | 5 tests | 1 test | ✓ (P04) | ✓ (S03) |
| F5 | PyTorch SDPA Attention Kernel | R2 | 5 tests | 2 tests | ✓ (P02, P04, P08) | ✓ (S04) |
| F6 | MPS Native `GradScaler` & FP16 AMP | R2 | 5 tests | 1 test | ✓ (P02, P03, P05, P08) | ✓ (S01, S04, S06) |
| F7 | Micro-Batch & Grad Accumulation | R2 | 5 tests | 2 tests | ✓ (P05) | ✓ (S01) |
| F8 | Lazy MPS Cache Management | R2 | 5 tests | 1 test | ✓ (P06) | ✓ (S06) |
| F9 | `TrainingStepProfiler` Micro-Breakdown | R3 | 5 tests | 1 test | ✓ (P03, P07, P08) | ✓ (S01, S05) |
| F10 | Standalone Benchmark CLI | R3 | 5 tests | 1 test | ✓ (P08) | ✓ (S05) |
| F11 | Metal GPU Watermark Telemetry | R3 | 5 tests | 1 test | ✓ (P03, P06, P08) | ✓ (S06) |
| F12 | Loss Convergence & Stability | All | 5 tests | 1 test | ✓ (P02, P05, P08) | ✓ (S01, S03, S04, S06) |

---

## Test Architecture & Suite Layout

1. **`tests/e2e/test_training_acceleration.py`** (93 tests):
   - **Tier 1 (Feature Isolation)**: Isolated validation of prefetcher queuing, mmap slicing, worker configuration, SDPA attention, GradScaler scaling, micro-batch accumulation, cache triggers, profiler metrics, CLI options, watermark telemetry, and loss convergence.
   - **Tier 2 (Boundary & Error Handling)**: Zero-length inputs, single-sample batches, queue timeout/cancellation, out-of-bounds indexing, worker prefetch invariants, sequence length 1 SDPA, Inf/NaN gradient detection, remainder micro-batches, and masked label loss handling.
   - **Tier 3 (Pairwise Subsystem Integration)**: Cross-module interactions including Prefetcher + DataLoader + TrOCRTrainer + SDPA + GradScaler + Profiler + Telemetry.
   - **Tier 4 (Real-World Workload Scenarios)**: Throughput speedup (>6.5 samples/sec), zero dataloader wait time ($T_{\text{data}} \to 0$), multi-stage curriculum execution, SDPA numerical fidelity, and long-run memory stability.

2. **`pipeline/tests/test_acceleration_suite.py`** (21 tests):
   - Low-level subsystem mechanics, memory-mapping array manipulation, thread safety, mathematical scaled dot product attention equivalence, GradScaler backoff arithmetic, and profiler serialization.

---

## Test Execution Commands

### 1. Run Complete Acceleration E2E Suite
```bash
./.venv/bin/pytest tests/e2e/test_training_acceleration.py -v
```

### 2. Run Acceleration Subsystem Suite
```bash
./.venv/bin/pytest pipeline/tests/test_acceleration_suite.py -v
```

### 3. Run Specific Tiers
```bash
# Tier 1: Feature Isolation
./.venv/bin/pytest tests/e2e/test_training_acceleration.py -m tier1 -v

# Tier 2: Boundary & Error Handling
./.venv/bin/pytest tests/e2e/test_training_acceleration.py -m tier2 -v

# Tier 3: Pairwise Subsystem Integration
./.venv/bin/pytest tests/e2e/test_training_acceleration.py -m tier3 -v

# Tier 4: Real-World Workload Scenarios
./.venv/bin/pytest tests/e2e/test_training_acceleration.py -m tier4 -v
```

### 4. Run Full Project Regression Suites
```bash
# All E2E Tests (448 tests)
./.venv/bin/pytest tests/e2e/ -q

# All Pipeline Tests (196 tests)
./.venv/bin/pytest pipeline/tests/ -q

# All Backend API Tests (50 tests)
./.venv/bin/pytest backend/tests/ -q
```

---

## Pass/Fail Verification Criteria
1. **Throughput Acceleration**: End-to-end training throughput strictly exceeds baseline (>6.5 samples/sec).
2. **Zero Data Starvation**: Mean data wait latency per step ($T_{\text{data}}$) is $< 5.0\text{ ms}$ under asynchronous device prefetching.
3. **Numerical Convergence & Stability**: Zero `NaN` or `Inf` in losses, parameter weights, and gradients. Loss decreases monotonically on synthetic handwriting batches.
4. **Memory Stability**: Memory watermark variance remains bounded ($< 5\%$ variance across 50 iterations).
5. **Zero Regressions**: 100% pass rate across all existing unit, backend, and end-to-end tests.
