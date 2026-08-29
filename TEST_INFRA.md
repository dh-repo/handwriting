# E2E Test Infra: TrOCR Training Acceleration

## Test Philosophy
- Opaque-box, requirement-driven, empirical verification of training throughput, latency, memory stability, and convergence.
- Zero dependency on internal shortcuts; all tests verify live tensor execution, model training loops, and benchmark CLI invocations.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial + Real-World Workload Testing.

## Feature Inventory & Test Coverage Mapping
| # | Feature | Requirement | Tier 1 (Isolation) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workload) |
|---|---------|-------------|:------------------:|:-----------------:|:-----------------:|:-----------------:|
| 1 | `AsyncDevicePrefetcher` | R1 | 5 | 5 | ✓ | ✓ |
| 2 | `MMapOCRDataset` & Memory Mapping | R1 | 5 | 5 | ✓ | ✓ |
| 3 | Dynamic Worker Multiprocessing | R1 | 5 | 5 | ✓ | ✓ |
| 4 | Curriculum Multi-Worker Execution | R1 | 5 | 5 | ✓ | ✓ |
| 5 | SDPA Attention Kernel Fusion | R2 | 5 | 5 | ✓ | ✓ |
| 6 | MPS `GradScaler` & FP16 AMP | R2 | 5 | 5 | ✓ | ✓ |
| 7 | Micro-Batch & Gradient Accumulation | R2 | 5 | 5 | ✓ | ✓ |
| 8 | MPS Cache Management & Defragmentation | R2 | 5 | 5 | ✓ | ✓ |
| 9 | `TrainingStepProfiler` Micro-Breakdown | R3 | 5 | 5 | ✓ | ✓ |
| 10 | Standalone Benchmark CLI | R3 | 5 | 5 | ✓ | ✓ |
| 11 | Metal GPU Memory Watermark Telemetry | R3 | 5 | 5 | ✓ | ✓ |
| 12 | Loss Convergence & Stability Verification | All | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Test Runner**: `tests/e2e/test_training_acceleration.py` and `pytest pipeline/tests/`
- **Pass/Fail Criteria**:
  1. Training throughput strictly exceeds baseline (>6.5 samples/sec).
  2. Data wait time per step is zero or near-zero ($< 1.0\text{ ms}$) under async prefetching.
  3. All backward steps execute without `NaN` / `Inf` loss gradients.
  4. Memory growth remains bounded across iterations without leaks.
  5. All 1,209 existing unit/backend/e2e tests continue to pass (zero regressions).

## Coverage Thresholds
- Tier 1: $\ge 5$ test cases per feature (60 tests minimum across 12 features)
- Tier 2: $\ge 5$ boundary/error test cases per feature (60 tests minimum)
- Tier 3: Pairwise subsystem integration tests (DataLoader + Prefetcher + SDPA + GradScaler + Profiler)
- Tier 4: Real-world training workload benchmark tests (full training step verification on CPU & MPS)
