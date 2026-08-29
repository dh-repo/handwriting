"""
pipeline/evaluation/__init__.py
Clean exports for evaluation subsystem, CER/WER metrics, and latency profiling.
"""

from pipeline.evaluation.benchmark_ablation import (
    AblationBenchmarkRunner,
    MultiTierAblationReport,
    TierBenchmarkResult,
    calculate_pnda,
)
from pipeline.evaluation.evaluate import (
    EvaluationReport,
    evaluate_checkpoint,
    evaluate_model,
    load_model_and_processor,
)
from pipeline.evaluation.metrics import (
    ErrorBreakdown,
    LatencyMetrics,
    LatencyTracker,
    MetricResult,
    NormalizationConfig,
    ThroughputMetrics,
    ThroughputProfiler,
    compute_cer,
    compute_metrics,
    compute_wer,
    normalize_text,
)

__all__ = [
    "compute_cer",
    "compute_wer",
    "compute_metrics",
    "normalize_text",
    "NormalizationConfig",
    "ErrorBreakdown",
    "MetricResult",
    "LatencyMetrics",
    "ThroughputMetrics",
    "LatencyTracker",
    "ThroughputProfiler",
    "EvaluationReport",
    "evaluate_model",
    "load_model_and_processor",
    "evaluate_checkpoint",
    "TierBenchmarkResult",
    "MultiTierAblationReport",
    "AblationBenchmarkRunner",
    "calculate_pnda",
]
