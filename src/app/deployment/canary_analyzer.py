"""Automated canary validation and rollback analysis.

Analyzes metrics between production and canary deployments to make
automated rollback decisions based on performance regressions.

Example:
    >>> from src.app.deployment.canary_analyzer import canary_analyzer
    >>> 
    >>> # Fetch metrics for comparison
    >>> v1_metrics = await canary_analyzer.fetch_metrics("v1.2.0")
    >>> v2_metrics = await canary_analyzer.fetch_metrics("v1.3.0-canary")
    >>> 
    >>> # Evaluate rollback decision
    >>> decision, reasoning = rollback_policy.evaluate(v1_metrics, v2_metrics)
    >>> 
    >>> if decision == RollbackDecision.ROLLBACK_AUTO:
    >>>     await trigger_rollback()
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Optional, Tuple, Any, List
import warnings

import httpx
from prometheus_client import CollectorRegistry, Gauge
from loguru import logger


@dataclass
class CanaryMetrics:
    """Metrics collected for canary analysis.
    
    Contains key performance indicators for evaluating deployment health.
    All timing values are in milliseconds, rates as percentages.
    """
    latency_p50: float = 0.0  # 50th percentile latency in ms
    latency_p95: float = 0.0  # 95th percentile latency in ms
    latency_p99: float = 0.0  # 99th percentile latency in ms
    error_rate: float = 0.0   # Error rate as percentage (0-100)
    model_confidence: float = 0.0  # Average model confidence (0-1)
    request_rate: float = 0.0  # Requests per second
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = ""
    sample_count: int = 0  # Number of requests analyzed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "error_rate": self.error_rate,
            "model_confidence": self.model_confidence,
            "request_rate": self.request_rate,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "sample_count": self.sample_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanaryMetrics":
        """Create from dictionary."""
        return cls(
            latency_p50=data["latency_p50"],
            latency_p95=data["latency_p95"],
            latency_p99=data["latency_p99"],
            error_rate=data["error_rate"],
            model_confidence=data["model_confidence"],
            request_rate=data["request_rate"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data["version"],
            sample_count=data["sample_count"],
        )


class RollbackDecision(Enum):
    """Automated rollback decision types."""
    PROCEED = "proceed"  # All metrics good, promote canary
    ROLLBACK_AUTO = "rollback_auto"  # Critical regression, auto-rollback
    ROLLBACK_MANUAL = "rollback_manual"  # Needs human review


class CanaryAnalyzer:
    """Fetches and analyzes metrics for canary validation.
    
    Queries Prometheus to gather performance metrics for specific
    deployment versions over a time window.
    """
    
    def __init__(
        self,
        prometheus_url: str = "http://prometheus:9090",
        timeout: int = 30,
        default_duration_minutes: int = 10
    ):
        """Initialize canary analyzer.
        
        Args:
            prometheus_url: Prometheus server URL
            timeout: Request timeout in seconds
            default_duration_minutes: Default analysis window
        """
        self.prometheus_url = prometheus_url.rstrip("/")
        self.timeout = timeout
        self.default_duration_minutes = default_duration_minutes
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Prometheus queries for different metrics
        self.queries = {
            "latency_p50": 'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{{version="{version}"}}[5m])) * 1000',
            "latency_p95": 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{version="{version}"}}[5m])) * 1000',
            "latency_p99": 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{version="{version}"}}[5m])) * 1000',
            "error_rate": '(rate(http_requests_total{{version="{version}",status=~"5.."}}[5m]) / rate(http_requests_total{{version="{version}"}}[5m])) * 100',
            "model_confidence": 'avg(model_prediction_confidence{{version="{version}"}})',
            "request_rate": 'rate(http_requests_total{{version="{version}"}}[5m])',
            "sample_count": 'sum(increase(http_requests_total{{version="{version}"}}[5m]))',
        }
    
    async def fetch_metrics(self, version: str, duration_minutes: Optional[int] = None) -> CanaryMetrics:
        """Fetch metrics for a specific version.
        
        Args:
            version: Deployment version to query
            duration_minutes: Time window for analysis (optional)
            
        Returns:
            CanaryMetrics with fetched values
            
        Raises:
            httpx.HTTPError: If Prometheus query fails
            ValueError: If metrics are invalid
        """
        duration = duration_minutes or self.default_duration_minutes
        
        logger.info(f"Fetching canary metrics for version {version} over {duration} minutes")
        
        metrics = {}
        
        # Query each metric
        for metric_name, query_template in self.queries.items():
            query = query_template.format(version=version)
            
            try:
                value = await self._query_prometheus(query)
                metrics[metric_name] = value
                
                logger.debug(f"{metric_name} for {version}: {value}")
                
            except Exception as e:
                logger.error(f"Failed to fetch {metric_name} for {version}: {e}")
                # Use default values for failed metrics
                metrics[metric_name] = self._get_default_value(metric_name)
        
        # Create and validate metrics object
        canary_metrics = CanaryMetrics(
            latency_p50=metrics["latency_p50"],
            latency_p95=metrics["latency_p95"],
            latency_p99=metrics["latency_p99"],
            error_rate=metrics["error_rate"],
            model_confidence=metrics["model_confidence"],
            request_rate=metrics["request_rate"],
            version=version,
            sample_count=int(metrics["sample_count"]),
        )
        
        # Validate metrics
        self._validate_metrics(canary_metrics)
        
        return canary_metrics
    
    async def _query_prometheus(self, query: str) -> float:
        """Execute Prometheus query and return value.
        
        Args:
            query: PromQL query string
            
        Returns:
            Metric value as float
            
        Raises:
            httpx.HTTPError: If query fails
        """
        url = f"{self.prometheus_url}/api/v1/query"
        params = {"query": query}
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data["status"] != "success":
            raise ValueError(f"Prometheus query failed: {data}")
        
        result = data["data"]["result"]
        
        if not result:
            warnings.warn(f"No data returned for query: {query}")
            return 0.0
        
        # Extract value from result
        value_str = result[0]["value"][1]
        
        try:
            return float(value_str)
        except (ValueError, TypeError):
            warnings.warn(f"Invalid metric value: {value_str}")
            return 0.0
    
    def _get_default_value(self, metric_name: str) -> float:
        """Get default value for failed metric queries.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Default value based on metric type
        """
        defaults = {
            "latency_p50": 100.0,  # 100ms default
            "latency_p95": 500.0,  # 500ms default
            "latency_p99": 1000.0,  # 1s default
            "error_rate": 0.0,  # 0% errors
            "model_confidence": 0.5,  # 50% confidence
            "request_rate": 0.0,  # No requests
            "sample_count": 0.0,  # No samples
        }
        
        return defaults.get(metric_name, 0.0)
    
    def _validate_metrics(self, metrics: CanaryMetrics) -> None:
        """Validate fetched metrics are reasonable.
        
        Args:
            metrics: CanaryMetrics to validate
            
        Raises:
            ValueError: If metrics are invalid
        """
        # Check for negative values where not allowed
        if metrics.latency_p50 < 0 or metrics.latency_p95 < 0 or metrics.latency_p99 < 0:
            raise ValueError("Latency values cannot be negative")
        
        if metrics.error_rate < 0 or metrics.error_rate > 100:
            raise ValueError("Error rate must be between 0 and 100")
        
        if metrics.model_confidence < 0 or metrics.model_confidence > 1:
            raise ValueError("Model confidence must be between 0 and 1")
        
        if metrics.sample_count < 0:
            raise ValueError("Sample count cannot be negative")
        
        # Warn about suspicious values
        if metrics.sample_count < 10:
            warnings.warn(f"Low sample count: {metrics.sample_count}")
        
        if metrics.latency_p99 > 10000:  # 10 seconds
            warnings.warn(f"Very high P99 latency: {metrics.latency_p99}ms")
        
        if metrics.error_rate > 50:  # 50% errors
            warnings.warn(f"High error rate: {metrics.error_rate}%")
    
    async def compare_versions(
        self,
        version1: str,
        version2: str,
        duration_minutes: Optional[int] = None
    ) -> Tuple[CanaryMetrics, CanaryMetrics]:
        """Fetch metrics for two versions simultaneously.
        
        Args:
            version1: First version (typically production)
            version2: Second version (typically canary)
            duration_minutes: Time window for analysis
            
        Returns:
            Tuple of metrics for both versions
        """
        # Fetch metrics concurrently
        tasks = [
            self.fetch_metrics(version1, duration_minutes),
            self.fetch_metrics(version2, duration_minutes)
        ]
        
        metrics1, metrics2 = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        if isinstance(metrics1, Exception):
            logger.error(f"Failed to fetch metrics for {version1}: {metrics1}")
            raise metrics1
        
        if isinstance(metrics2, Exception):
            logger.error(f"Failed to fetch metrics for {version2}: {metrics2}")
            raise metrics2
        
        return metrics1, metrics2
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Global instance for application use
canary_analyzer = CanaryAnalyzer()


# Convenience function for canary validation
async def fetch_canary_metrics(
    production_version: str,
    canary_version: str,
    duration_minutes: int = 10
) -> Tuple[CanaryMetrics, CanaryMetrics]:
    """Fetch metrics for production and canary versions.
    
    Args:
        production_version: Production deployment version
        canary_version: Canary deployment version
        duration_minutes: Analysis window in minutes
        
    Returns:
        Tuple of (production_metrics, canary_metrics)
    """
    async with CanaryAnalyzer() as analyzer:
        return await analyzer.compare_versions(
            production_version,
            canary_version,
            duration_minutes
        )
