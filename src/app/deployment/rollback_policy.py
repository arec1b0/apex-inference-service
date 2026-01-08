"""Rollback policy evaluation for canary deployments.

Implements automated decision logic for determining whether to proceed
with canary promotion, rollback automatically, or require manual review.
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Optional
from enum import Enum

from loguru import logger

from .canary_analyzer import CanaryMetrics, RollbackDecision


@dataclass
class RollbackThresholds:
    """Thresholds for rollback decision making.
    
    Configurable limits for when to trigger different rollback actions.
    All values are multipliers relative to baseline metrics.
    """
    latency_p99_threshold: float = 1.5  # 50% increase triggers auto-rollback
    error_rate_threshold: float = 2.0   # 2x increase triggers auto-rollback
    confidence_threshold: float = 0.8   # Below 80% of baseline triggers auto-rollback
    proceed_tolerance: float = 1.1      # Within 10% is acceptable for promotion
    
    # Additional thresholds for manual review
    latency_p95_warning: float = 1.3    # 30% increase needs review
    error_rate_warning: float = 1.5     # 50% increase needs review
    
    def __post_init__(self):
        """Validate thresholds are reasonable."""
        if self.latency_p99_threshold <= 1.0:
            raise ValueError("Latency threshold must be > 1.0")
        if self.error_rate_threshold <= 1.0:
            raise ValueError("Error rate threshold must be > 1.0")
        if not 0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if self.proceed_tolerance <= 1.0:
            raise ValueError("Proceed tolerance must be > 1.0")


@dataclass
class ComparisonResult:
    """Result of comparing two metric sets."""
    metric_name: str
    baseline_value: float
    canary_value: float
    ratio: float  # canary / baseline
    change_percent: float  # Percentage change
    status: str  # "improved", "degraded", "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "canary_value": self.canary_value,
            "ratio": self.ratio,
            "change_percent": self.change_percent,
            "status": self.status,
        }


class RollbackPolicy:
    """Evaluates canary metrics against rollback policies.
    
    Implements business logic for automated rollback decisions based
    on performance regressions and improvements.
    """
    
    def __init__(self, thresholds: Optional[RollbackThresholds] = None):
        """Initialize rollback policy.
        
        Args:
            thresholds: Custom thresholds for decision making
        """
        self.thresholds = thresholds or RollbackThresholds()
    
    def evaluate(
        self,
        v1_metrics: CanaryMetrics,
        v2_metrics: CanaryMetrics
    ) -> Tuple[RollbackDecision, Dict[str, Any]]:
        """Evaluate rollback decision based on metrics comparison.
        
        Args:
            v1_metrics: Baseline (production) metrics
            v2_metrics: Canary (new version) metrics
            
        Returns:
            Tuple of (decision, reasoning_dict)
        """
        logger.info(
            f"Evaluating rollback decision: {v1_metrics.version} -> {v2_metrics.version}"
        )
        
        # Compare all metrics
        comparisons = self._compare_metrics(v1_metrics, v2_metrics)
        
        # Check for automatic rollback conditions
        auto_rollback_reasons = self._check_auto_rollback_conditions(comparisons)
        if auto_rollback_reasons:
            decision = RollbackDecision.ROLLBACK_AUTO
            reasoning = {
                "decision": decision.value,
                "primary_reason": "Critical performance regression",
                "triggered_conditions": auto_rollback_reasons,
                "comparisons": [c.to_dict() for c in comparisons],
                "recommendation": "Immediate rollback recommended"
            }
            logger.warning(f"Auto-rollback triggered: {auto_rollback_reasons}")
            return decision, reasoning
        
        # Check if all metrics are good for promotion
        if self._check_proceed_conditions(comparisons):
            decision = RollbackDecision.PROCEED
            reasoning = {
                "decision": decision.value,
                "primary_reason": "All metrics within acceptable range",
                "triggered_conditions": [],
                "comparisons": [c.to_dict() for c in comparisons],
                "recommendation": "Proceed with canary promotion"
            }
            logger.info("Canary validation passed - proceed with promotion")
            return decision, reasoning
        
        # Otherwise, require manual review
        decision = RollbackDecision.ROLLBACK_MANUAL
        reasoning = {
            "decision": decision.value,
            "primary_reason": "Mixed performance metrics need review",
            "triggered_conditions": self._get_warning_conditions(comparisons),
            "comparisons": [c.to_dict() for c in comparisons],
            "recommendation": "Manual review required before decision"
        }
        logger.warning("Manual review required for canary decision")
        return decision, reasoning
    
    def _compare_metrics(
        self,
        v1_metrics: CanaryMetrics,
        v2_metrics: CanaryMetrics
    ) -> List[ComparisonResult]:
        """Compare metrics between versions.
        
        Args:
            v1_metrics: Baseline metrics
            v2_metrics: Canary metrics
            
        Returns:
            List of comparison results
        """
        comparisons = []
        
        # Compare latencies (lower is better)
        comparisons.append(self._compare_single_metric(
            "latency_p99",
            v1_metrics.latency_p99,
            v2_metrics.latency_p99,
            lower_is_better=True
        ))
        
        comparisons.append(self._compare_single_metric(
            "latency_p95",
            v1_metrics.latency_p95,
            v2_metrics.latency_p95,
            lower_is_better=True
        ))
        
        comparisons.append(self._compare_single_metric(
            "latency_p50",
            v1_metrics.latency_p50,
            v2_metrics.latency_p50,
            lower_is_better=True
        ))
        
        # Compare error rate (lower is better)
        comparisons.append(self._compare_single_metric(
            "error_rate",
            v1_metrics.error_rate,
            v2_metrics.error_rate,
            lower_is_better=True
        ))
        
        # Compare model confidence (higher is better)
        comparisons.append(self._compare_single_metric(
            "model_confidence",
            v1_metrics.model_confidence,
            v2_metrics.model_confidence,
            lower_is_better=False
        ))
        
        # Compare request rate (higher is better, but not critical)
        comparisons.append(self._compare_single_metric(
            "request_rate",
            v1_metrics.request_rate,
            v2_metrics.request_rate,
            lower_is_better=False,
            critical=False
        ))
        
        return comparisons
    
    def _compare_single_metric(
        self,
        name: str,
        baseline: float,
        canary: float,
        lower_is_better: bool,
        critical: bool = True
    ) -> ComparisonResult:
        """Compare a single metric between versions.
        
        Args:
            name: Metric name
            baseline: Baseline value
            canary: Canary value
            lower_is_better: Whether lower values are better
            critical: Whether this is a critical metric
            
        Returns:
            ComparisonResult with analysis
        """
        # Handle edge cases
        if baseline == 0:
            if canary == 0:
                ratio = 1.0
                change_percent = 0.0
                status = "stable"
            else:
                ratio = float('inf') if lower_is_better else float('inf')
                change_percent = float('inf')
                status = "degraded" if lower_is_better else "improved"
        else:
            ratio = canary / baseline
            change_percent = (ratio - 1) * 100
            
            if lower_is_better:
                if ratio < 0.9:
                    status = "improved"
                elif ratio > 1.1:
                    status = "degraded"
                else:
                    status = "stable"
            else:
                if ratio > 1.1:
                    status = "improved"
                elif ratio < 0.9:
                    status = "degraded"
                else:
                    status = "stable"
        
        return ComparisonResult(
            metric_name=name,
            baseline_value=baseline,
            canary_value=canary,
            ratio=ratio,
            change_percent=change_percent,
            status=status
        )
    
    def _check_auto_rollback_conditions(
        self,
        comparisons: List[ComparisonResult]
    ) -> List[str]:
        """Check conditions that trigger automatic rollback.
        
        Args:
            comparisons: Metric comparisons
            
        Returns:
            List of triggered condition descriptions
        """
        triggered = []
        
        for comp in comparisons:
            if comp.metric_name == "latency_p99":
                if comp.ratio > self.thresholds.latency_p99_threshold:
                    triggered.append(
                        f"P99 latency increased by {comp.change_percent:.1f}% "
                        f"(threshold: {(self.thresholds.latency_p99_threshold - 1) * 100:.0f}%)"
                    )
            
            elif comp.metric_name == "error_rate":
                if comp.ratio > self.thresholds.error_rate_threshold:
                    triggered.append(
                        f"Error rate increased by {comp.change_percent:.1f}% "
                        f"(threshold: {(self.thresholds.error_rate_threshold - 1) * 100:.0f}%)"
                    )
            
            elif comp.metric_name == "model_confidence":
                # For confidence, lower ratio is bad
                if comp.ratio < self.thresholds.confidence_threshold:
                    triggered.append(
                        f"Model confidence decreased by {abs(comp.change_percent):.1f}% "
                        f"(threshold: {(1 - self.thresholds.confidence_threshold) * 100:.0f}%)"
                    )
        
        return triggered
    
    def _check_proceed_conditions(
        self,
        comparisons: List[ComparisonResult]
    ) -> bool:
        """Check if all metrics are within proceed tolerance.
        
        Args:
            comparisons: Metric comparisons
            
        Returns:
            True if can proceed with promotion
        """
        for comp in comparisons:
            # Skip non-critical metrics for proceed decision
            if comp.metric_name == "request_rate":
                continue
            
            # Check if metric is within tolerance
            if comp.metric_name in ["latency_p99", "latency_p95", "latency_p50", "error_rate"]:
                if comp.ratio > self.thresholds.proceed_tolerance:
                    return False
            elif comp.metric_name == "model_confidence":
                if comp.ratio < (1 / self.thresholds.proceed_tolerance):
                    return False
        
        return True
    
    def _get_warning_conditions(
        self,
        comparisons: List[ComparisonResult]
    ) -> List[str]:
        """Get conditions that triggered manual review.
        
        Args:
            comparisons: Metric comparisons
            
        Returns:
            List of warning condition descriptions
        """
        warnings = []
        
        for comp in comparisons:
            if comp.metric_name == "latency_p95":
                if comp.ratio > self.thresholds.latency_p95_warning:
                    warnings.append(
                        f"P95 latency increased by {comp.change_percent:.1f}%"
                    )
            
            elif comp.metric_name == "error_rate":
                if comp.ratio > self.thresholds.error_rate_warning:
                    warnings.append(
                        f"Error rate increased by {comp.change_percent:.1f}%"
                    )
            
            # Add warnings for any degraded metrics
            if comp.status == "degraded" and comp.metric_name not in ["request_rate"]:
                warnings.append(f"{comp.metric_name} shows degradation")
        
        return warnings or ["General performance variation detected"]


# Global instance with default thresholds
rollback_policy = RollbackPolicy()


# Convenience function for evaluation
async def evaluate_canary(
    production_version: str,
    canary_version: str,
    duration_minutes: int = 10,
    thresholds: Optional[RollbackThresholds] = None
) -> Tuple[RollbackDecision, Dict[str, Any]]:
    """Evaluate canary deployment with automatic metric fetching.
    
    Args:
        production_version: Production deployment version
        canary_version: Canary deployment version
        duration_minutes: Analysis window
        thresholds: Custom rollback thresholds
        
    Returns:
        Tuple of (decision, reasoning)
    """
    from .canary_analyzer import fetch_canary_metrics
    
    # Fetch metrics
    v1_metrics, v2_metrics = await fetch_canary_metrics(
        production_version,
        canary_version,
        duration_minutes
    )
    
    # Evaluate with policy
    policy = RollbackPolicy(thresholds)
    return policy.evaluate(v1_metrics, v2_metrics)
