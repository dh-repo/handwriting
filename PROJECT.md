# Project: TrOCR Handwriting Training Acceleration

## Architecture
The TrOCR training acceleration project optimizes end-to-end training throughput across data ingestion, kernel computation, and hardware memory management for Apple Silicon Metal (MPS) and CUDA.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  TRAINING PIPELINE ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
    ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
    ▼                                                                                   ▼
┌───────────────────────────────────────────┐                       ┌───────────────────────────────────────────┐
│ 1. Data Ingestion & Prefetching (R1)      │                       │ 2. Compute Kernel & Memory Engine (R2)   │
├───────────────────────────────────────────┤                       ├───────────────────────────────────────────┤
│ • MMapOCRDataset / Pre-Cached Storage     │                       │ • PyTorch SDPA (Scaled Dot-Product Attn)  │
│ • Dynamic Multi-Worker DataLoader         │                       │ • PyTorch 2.x Native MPS GradScaler       │
│ • Persistent Workers & Prefetch Buffers   │                       │ • FP16 Mixed Precision AMP                │
│ • AsyncDevicePrefetcher (Double/Triple)   │                       │ • Micro-Batch Sizing & Grad Accumulation  │
│ • Pre-tokenized Manifest Byte Indexing    │                       │ • Lazy MPS Allocator Memory Defrag        │
└─────────────────────┬─────────────────────┘                       └─────────────────────┬─────────────────────┘
                      │                                                                   │
                      └───────────────────────────────┬───────────────────────────────────┘
                                                      │
                                                      ▼
                                    ┌───────────────────────────────────┐
                                    │ 3. Profiling & Telemetry (R3)     │
                                    ├───────────────────────────────────┤
                                    │ • StepProfiler (T_data, T_fwd, ..)│
                                    │ • Samples/Sec Throughput Tracker  │
                                    │ • Metal GPU Watermark Telemetry   │
                                    │ • Standalone Benchmark CLI        │
                                    │ • Loss Convergence Verification   │
                                    └───────────────────────────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| F1 | `AsyncDevicePrefetcher` | Threaded background queue pre-staging batches directly to MPS/CUDA device | M1 | R1 | DONE |
| F2 | `MMapOCRDataset` | Zero-copy memory-mapped tensor dataset for high-throughput image reading | M1 | R1 | DONE |
| F3 | Multi-Worker Multiprocessing | Dynamic worker scaling with `persistent_workers=True` and `prefetch_factor>=2` | M1 | R1 | DONE |
| F4 | Curriculum Multi-Worker Support | Configurable workers in `MultiStageCurriculumTrainer` replacing hardcoded 0 | M1 | R1 | DONE |
| F5 | PyTorch SDPA Attention | Scaled Dot-Product Attention kernel fusion (`_attn_implementation="sdpa"`) | M2 | R2 | DONE |
| F6 | MPS Native `GradScaler` | PyTorch 2.x `torch.amp.GradScaler('mps')` for stable FP16 backprop | M2 | R2 | DONE |
| F7 | Micro-Batch & Grad Accumulation | Optimized micro-batching (8-16) and gradient accumulation (4) | M2 | R2 | DONE |
| F8 | Lazy MPS Cache Management | Dynamic cache defragmentation avoiding 4.5ms synchronization stalls | M2 | R2 | DONE |
| F9 | Granular `StepProfiler` | Microsecond timing breakdowns ($T_{\text{data}}, T_{\text{transfer}}, T_{\text{fwd}}, T_{\text{bwd}}, T_{\text{opt}}$) | M3 | R3 | DONE |
| F10 | Acceleration Benchmark CLI | Standalone CLI comparing baseline vs accelerated pipeline throughput | M3 | R3 | DONE |
| F11 | Metal GPU Telemetry | Memory allocation and watermark telemetry tracking across iterations | M3 | R3 | DONE |
| F12 | E2E & Adversarial Hardening | 100% E2E test verification, numerical convergence, and adversarial stress | M4 (Final) | E2E | DONE |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | E2E Testing Track | Test harness, synthetic/real benchmark suites, Tiers 1-4 coverage, `TEST_READY.md` | none | DONE |
| M1 | Data Ingestion & Prefetching (R1) | `AsyncDevicePrefetcher`, `MMapOCRDataset`, multi-worker scaling, curriculum worker unblocking | none | DONE |
| M2 | Compute Kernel & Memory (R2) | SDPA kernel integration, MPS `GradScaler`, micro-batching, lazy cache eviction | M1 | DONE |
| M3 | Profiling & Telemetry (R3) | `TrainingStepProfiler`, benchmark harness `benchmark_throughput.py`, telemetry logging | M1, M2 | DONE |
| M4 | Final Milestone & Hardening | Pass 100% E2E tests (Tiers 1-4), verify >6.5 samples/sec, Tier 5 Adversarial Hardening | M0, M1, M2, M3 | DONE |

## Code Layout
- `pipeline/training/config.py`: Training & curriculum configurations (workers, prefetch, SDPA, AMP, batch size, cache steps).
- `pipeline/training/dataset.py`: `OCRDataset`, `MMapOCRDataset`, `OCRDataCollator`, tensor transforms.
- `pipeline/training/prefetcher.py`: `AsyncDevicePrefetcher` asynchronous device queue & background workers.
- `pipeline/training/train.py`: `TrOCRTrainer`, SDPA model loading, `GradScaler` forward/backward execution, device transfers.
- `pipeline/training/curriculum.py`: `MultiStageCurriculumTrainer` multi-worker execution, SDPA, curriculum telemetry.
- `pipeline/training/profiler.py`: `TrainingStepProfiler` fine-grained microsecond latency & throughput tracker.
- `pipeline/training/benchmark_throughput.py`: Standalone CLI benchmark harness for baseline vs accelerated runs.
- `pipeline/tests/`: Unit and subsystem test suite.
- `tests/e2e/`: End-to-end integration and acceleration verification test suites.

## Interface Contracts
### `AsyncDevicePrefetcher` ↔ `TrOCRTrainer` / `MultiStageCurriculumTrainer`
- `AsyncDevicePrefetcher(loader: DataLoader, device: torch.device, mixed_precision: str = "fp16", queue_size: int = 3)`
- Implements `__iter__()` and `__next__() -> Dict[str, torch.Tensor]` yielding pre-staged batches already located on `device`.
- Implements `close()` and context manager support `__enter__` / `__exit__`.

### `TrainingConfig` ↔ `TrOCRTrainer`
- `attn_implementation: str = "sdpa"`
- `num_workers: int = 4` (dynamically defaulted based on CPU count)
- `persistent_workers: bool = True`
- `prefetch_factor: int = 4`
- `empty_cache_steps: int = 100`
- `enable_step_profiling: bool = True`

### `TrainingStepProfiler` ↔ `TrOCRTrainer`
- `profiler.record_data_wait(t_sec: float)`
- `profiler.record_step(t_data, t_transfer, t_fwd, t_bwd, t_opt, num_samples)`
- `profiler.get_summary() -> Dict[str, Any]` (samples_per_sec, step_latency_ms, percentiles p50/p90/p99, memory_mb)
