"""Deployment automation components for canary validation."""

from .canary_analyzer import (
    canary_analyzer,
    CanaryAnalyzer,
    CanaryMetrics,
    fetch_canary_metrics,
)

from .rollback_policy import (
    rollback_policy,
    RollbackPolicy,
    RollbackThresholds,
    ComparisonResult,
    evaluate_canary,
    RollbackDecision,
)

__all__ = [
    # Canary analyzer
    "canary_analyzer",
    "CanaryAnalyzer",
    "CanaryMetrics",
    "fetch_canary_metrics",
    # Rollback policy
    "rollback_policy",
    "RollbackPolicy",
    "RollbackThresholds",
    "ComparisonResult",
    "evaluate_canary",
    "RollbackDecision",
]
