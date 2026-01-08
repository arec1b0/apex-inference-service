"""Unit tests for canary validation and rollback system.

Tests cover metrics fetching, policy evaluation, and decision logic
for automated canary deployment validation.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from src.app.deployment.canary_analyzer import (
    CanaryMetrics,
    CanaryAnalyzer,
    fetch_canary_metrics
)
from src.app.deployment.rollback_policy import (
    RollbackPolicy,
    RollbackThresholds,
    ComparisonResult,
    rollback_policy,
    evaluate_canary,
    RollbackDecision
)


class TestCanaryMetrics:
    """Test CanaryMetrics dataclass functionality."""
    
    def test_metrics_creation(self):
        """Test creating metrics with default values."""
        metrics = CanaryMetrics()
        
        assert metrics.latency_p50 == 0.0
        assert metrics.latency_p95 == 0.0
        assert metrics.latency_p99 == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.model_confidence == 0.0
        assert metrics.request_rate == 0.0
        assert metrics.version == ""
        assert metrics.sample_count == 0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metrics_with_values(self):
        """Test creating metrics with specific values."""
        timestamp = datetime.now(timezone.utc)
        metrics = CanaryMetrics(
            latency_p50=100.0,
            latency_p95=500.0,
            latency_p99=1000.0,
            error_rate=1.5,
            model_confidence=0.85,
            request_rate=100.0,
            version="v1.2.0",
            sample_count=1000,
            timestamp=timestamp
        )
        
        assert metrics.latency_p50 == 100.0
        assert metrics.latency_p95 == 500.0
        assert metrics.latency_p99 == 1000.0
        assert metrics.error_rate == 1.5
        assert metrics.model_confidence == 0.85
        assert metrics.request_rate == 100.0
        assert metrics.version == "v1.2.0"
        assert metrics.sample_count == 1000
        assert metrics.timestamp == timestamp
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=1.5,
            version="v1.2.0"
        )
        
        data_dict = metrics.to_dict()
        
        assert data_dict["latency_p99"] == 1000.0
        assert data_dict["error_rate"] == 1.5
        assert data_dict["version"] == "v1.2.0"
        assert "timestamp" in data_dict
        assert isinstance(data_dict["timestamp"], str)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data_dict = {
            "latency_p50": 100.0,
            "latency_p95": 500.0,
            "latency_p99": 1000.0,
            "error_rate": 1.5,
            "model_confidence": 0.85,
            "request_rate": 100.0,
            "timestamp": "2023-01-01T00:00:00+00:00",
            "version": "v1.2.0",
            "sample_count": 1000
        }
        
        metrics = CanaryMetrics.from_dict(data_dict)
        
        assert metrics.latency_p50 == 100.0
        assert metrics.latency_p95 == 500.0
        assert metrics.latency_p99 == 1000.0
        assert metrics.error_rate == 1.5
        assert metrics.model_confidence == 0.85
        assert metrics.version == "v1.2.0"
        assert metrics.sample_count == 1000


class TestCanaryAnalyzer:
    """Test canary metrics fetching and analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer for testing."""
        return CanaryAnalyzer(
            prometheus_url="http://test-prometheus:9090",
            timeout=10
        )
    
    @pytest.mark.asyncio
    async def test_query_prometheus_success(self, analyzer):
        """Test successful Prometheus query."""
        mock_response = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "value": ["1234567890", "123.45"]
                    }
                ]
            }
        }
        
        with patch.object(analyzer.client, 'get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()
            
            value = await analyzer._query_prometheus("test_query")
            
            assert value == 123.45
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_prometheus_no_data(self, analyzer):
        """Test Prometheus query with no data."""
        mock_response = {
            "status": "success",
            "data": {"result": []}
        }
        
        with patch.object(analyzer.client, 'get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()
            
            with pytest.warns(UserWarning):
                value = await analyzer._query_prometheus("test_query")
            
            assert value == 0.0
    
    @pytest.mark.asyncio
    async def test_query_prometheus_error(self, analyzer):
        """Test Prometheus query error handling."""
        with patch.object(analyzer.client, 'get') as mock_get:
            mock_get.return_value.raise_for_status.side_effect = Exception("Connection error")
            
            with pytest.raises(Exception):
                await analyzer._query_prometheus("test_query")
    
    @pytest.mark.asyncio
    async def test_fetch_metrics_success(self, analyzer):
        """Test successful metrics fetching."""
        # Mock all metric queries
        mock_values = {
            "latency_p50": 100.0,
            "latency_p95": 500.0,
            "latency_p99": 1000.0,
            "error_rate": 1.5,
            "model_confidence": 0.85,
            "request_rate": 100.0,
            "sample_count": 1000.0
        }
        
        with patch.object(analyzer, '_query_prometheus') as mock_query:
            mock_query.side_effect = lambda q: mock_values.get(q.split("{")[0].split("_")[-1], 0.0)
            
            metrics = await analyzer.fetch_metrics("v1.2.0")
            
            assert metrics.version == "v1.2.0"
            assert metrics.latency_p50 == 100.0
            assert metrics.latency_p95 == 500.0
            assert metrics.latency_p99 == 1000.0
            assert metrics.error_rate == 1.5
            assert metrics.model_confidence == 0.85
            assert metrics.request_rate == 100.0
            assert metrics.sample_count == 1000
    
    @pytest.mark.asyncio
    async def test_fetch_metrics_with_defaults(self, analyzer):
        """Test metrics fetching with default values on failure."""
        with patch.object(analyzer, '_query_prometheus') as mock_query:
            mock_query.side_effect = Exception("Query failed")
            
            with patch.object(analyzer, '_get_default_value') as mock_default:
                mock_default.return_value = 999.0
                
                metrics = await analyzer.fetch_metrics("v1.2.0")
                
                # Should use default values
                assert metrics.latency_p50 == 999.0
                assert metrics.version == "v1.2.0"
    
    def test_validate_metrics_valid(self, analyzer):
        """Test validation of valid metrics."""
        metrics = CanaryMetrics(
            latency_p50=100.0,
            latency_p95=500.0,
            latency_p99=1000.0,
            error_rate=1.5,
            model_confidence=0.85,
            sample_count=1000
        )
        
        # Should not raise
        analyzer._validate_metrics(metrics)
    
    def test_validate_metrics_invalid(self, analyzer):
        """Test validation of invalid metrics."""
        # Negative latency
        metrics = CanaryMetrics(latency_p50=-10.0)
        with pytest.raises(ValueError, match="cannot be negative"):
            analyzer._validate_metrics(metrics)
        
        # Invalid error rate
        metrics = CanaryMetrics(error_rate=150.0)
        with pytest.raises(ValueError, match="between 0 and 100"):
            analyzer._validate_metrics(metrics)
        
        # Invalid confidence
        metrics = CanaryMetrics(model_confidence=1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            analyzer._validate_metrics(metrics)
    
    @pytest.mark.asyncio
    async def test_compare_versions(self, analyzer):
        """Test comparing two versions."""
        with patch.object(analyzer, 'fetch_metrics') as mock_fetch:
            mock_fetch.side_effect = [
                CanaryMetrics(version="v1.0.0", latency_p99=1000.0),
                CanaryMetrics(version="v1.1.0", latency_p99=900.0)
            ]
            
            v1_metrics, v2_metrics = await analyzer.compare_versions("v1.0.0", "v1.1.0")
            
            assert v1_metrics.version == "v1.0.0"
            assert v2_metrics.version == "v1.1.0"
            assert v1_metrics.latency_p99 == 1000.0
            assert v2_metrics.latency_p99 == 900.0
    
    @pytest.mark.asyncio
    async def test_close(self, analyzer):
        """Test closing HTTP client."""
        with patch.object(analyzer.client, 'aclose') as mock_close:
            await analyzer.close()
            mock_close.assert_called_once()


class TestRollbackPolicy:
    """Test rollback policy evaluation logic."""
    
    @pytest.fixture
    def policy(self):
        """Create policy for testing."""
        return RollbackPolicy()
    
    @pytest.fixture
    def thresholds(self):
        """Create custom thresholds for testing."""
        return RollbackThresholds(
            latency_p99_threshold=1.5,
            error_rate_threshold=2.0,
            confidence_threshold=0.8,
            proceed_tolerance=1.1
        )
    
    def test_all_metrics_improve(self, policy):
        """Test decision when all metrics improve."""
        v1_metrics = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=2.0,
            model_confidence=0.80,
            version="v1.0.0"
        )
        
        v2_metrics = CanaryMetrics(
            latency_p99=800.0,  # 20% improvement
            error_rate=1.0,     # 50% improvement
            model_confidence=0.90,  # 12.5% improvement
            version="v1.1.0"
        )
        
        decision, reasoning = policy.evaluate(v1_metrics, v2_metrics)
        
        assert decision == RollbackDecision.PROCEED
        assert reasoning["primary_reason"] == "All metrics within acceptable range"
        assert len(reasoning["triggered_conditions"]) == 0
    
    def test_latency_regresses_50pct(self, policy):
        """Test auto-rollback when latency regresses 50%."""
        v1_metrics = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=1.0,
            model_confidence=0.85,
            version="v1.0.0"
        )
        
        v2_metrics = CanaryMetrics(
            latency_p99=1600.0,  # 60% increase (above 50% threshold)
            error_rate=1.0,
            model_confidence=0.85,
            version="v1.1.0"
        )
        
        decision, reasoning = policy.evaluate(v1_metrics, v2_metrics)
        
        assert decision == RollbackDecision.ROLLBACK_AUTO
        assert "latency" in reasoning["triggered_conditions"][0]
        assert reasoning["recommendation"] == "Immediate rollback recommended"
    
    def test_error_rate_doubles(self, policy):
        """Test auto-rollback when error rate doubles."""
        v1_metrics = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=1.0,
            model_confidence=0.85,
            version="v1.0.0"
        )
        
        v2_metrics = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=2.5,  # 150% increase (above 2x threshold)
            model_confidence=0.85,
            version="v1.1.0"
        )
        
        decision, reasoning = policy.evaluate(v1_metrics, v2_metrics)
        
        assert decision == RollbackDecision.ROLLBACK_AUTO
        assert "error rate" in reasoning["triggered_conditions"][0]
    
    def test_confidence_drops_20pct(self, policy):
        """Test auto-rollback when confidence drops 20%."""
        v1_metrics = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=1.0,
            model_confidence=0.90,  # High confidence baseline
            version="v1.0.0"
        )
        
        v2_metrics = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=1.0,
            model_confidence=0.70,  # 22% drop (below 80% threshold)
            version="v1.1.0"
        )
        
        decision, reasoning = policy.evaluate(v1_metrics, v2_metrics)
        
        assert decision == RollbackDecision.ROLLBACK_AUTO
        assert "confidence" in reasoning["triggered_conditions"][0]
    
    def test_mixed_metrics_within_threshold(self, policy):
        """Test proceed decision with mixed but acceptable metrics."""
        v1_metrics = CanaryMetrics(
            latency_p99=1000.0,
            latency_p95=500.0,
            error_rate=1.0,
            model_confidence=0.85,
            version="v1.0.0"
        )
        
        v2_metrics = CanaryMetrics(
            latency_p99=1050.0,  # 5% increase (within 10% tolerance)
            latency_p95=600.0,  # 20% increase (triggers warning but not rollback)
            error_rate=1.05,    # 5% increase (within tolerance)
            model_confidence=0.83,  # 2.3% drop (within tolerance)
            version="v1.1.0"
        )
        
        decision, reasoning = policy.evaluate(v1_metrics, v2_metrics)
        
        assert decision == RollbackDecision.PROCEED
        assert reasoning["primary_reason"] == "All metrics within acceptable range"
    
    def test_manual_review_needed(self, policy):
        """Test manual review decision with mixed metrics."""
        v1_metrics = CanaryMetrics(
            latency_p99=1000.0,
            latency_p95=500.0,
            error_rate=1.0,
            model_confidence=0.85,
            version="v1.0.0"
        )
        
        v2_metrics = CanaryMetrics(
            latency_p99=1200.0,  # 20% increase (needs review)
            error_rate=1.8,     # 80% increase (needs review)
            model_confidence=0.90,  # Improvement
            version="v1.1.0"
        )
        
        decision, reasoning = policy.evaluate(v1_metrics, v2_metrics)
        
        assert decision == RollbackDecision.ROLLBACK_MANUAL
        assert reasoning["primary_reason"] == "Mixed performance metrics need review"
        assert len(reasoning["triggered_conditions"]) > 0
    
    def test_compare_single_metric_latency(self, policy):
        """Test single metric comparison for latency."""
        result = policy._compare_single_metric(
            "latency_p99",
            1000.0,  # baseline
            1500.0,  # canary (worse)
            lower_is_better=True
        )
        
        assert result.metric_name == "latency_p99"
        assert result.baseline_value == 1000.0
        assert result.canary_value == 1500.0
        assert result.ratio == 1.5
        assert result.change_percent == 50.0
        assert result.status == "degraded"
    
    def test_compare_single_metric_confidence(self, policy):
        """Test single metric comparison for confidence."""
        result = policy._compare_single_metric(
            "model_confidence",
            0.80,   # baseline
            0.90,   # canary (better)
            lower_is_better=False
        )
        
        assert result.metric_name == "model_confidence"
        assert result.ratio == 1.125
        assert result.change_percent == 12.5
        assert result.status == "improved"
    
    def test_custom_thresholds(self, thresholds):
        """Test policy with custom thresholds."""
        policy = RollbackPolicy(thresholds)
        
        v1_metrics = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=1.0,
            model_confidence=0.85,
            version="v1.0.0"
        )
        
        v2_metrics = CanaryMetrics(
            latency_p99=1400.0,  # 40% increase (below 50% threshold)
            error_rate=1.0,
            model_confidence=0.85,
            version="v1.1.0"
        )
        
        decision, _ = policy.evaluate(v1_metrics, v2_metrics)
        
        # Should not trigger auto-rollback with custom threshold
        assert decision != RollbackDecision.ROLLBACK_AUTO


class TestRollbackThresholds:
    """Test rollback thresholds validation."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = RollbackThresholds()
        
        assert thresholds.latency_p99_threshold == 1.5
        assert thresholds.error_rate_threshold == 2.0
        assert thresholds.confidence_threshold == 0.8
        assert thresholds.proceed_tolerance == 1.1
    
    def test_invalid_thresholds(self):
        """Test validation of invalid thresholds."""
        # Latency threshold <= 1.0
        with pytest.raises(ValueError, match="must be > 1.0"):
            RollbackThresholds(latency_p99_threshold=1.0)
        
        # Error rate threshold <= 1.0
        with pytest.raises(ValueError, match="must be > 1.0"):
            RollbackThresholds(error_rate_threshold=0.5)
        
        # Confidence threshold out of range
        with pytest.raises(ValueError, match="between 0 and 1"):
            RollbackThresholds(confidence_threshold=1.5)
        
        # Proceed tolerance <= 1.0
        with pytest.raises(ValueError, match="must be > 1.0"):
            RollbackThresholds(proceed_tolerance=0.9)


class TestIntegration:
    """Integration tests for canary validation system."""
    
    @pytest.mark.asyncio
    async def test_evaluate_canary_function(self):
        """Test the convenience evaluate_canary function."""
        mock_v1 = CanaryMetrics(
            latency_p99=1000.0,
            error_rate=1.0,
            model_confidence=0.85,
            version="v1.0.0"
        )
        
        mock_v2 = CanaryMetrics(
            latency_p99=900.0,
            error_rate=0.8,
            model_confidence=0.90,
            version="v1.1.0"
        )
        
        with patch('src.app.deployment.rollback_policy.fetch_canary_metrics') as mock_fetch:
            mock_fetch.return_value = (mock_v1, mock_v2)
            
            decision, reasoning = await evaluate_canary("v1.0.0", "v1.1.0")
            
            assert decision == RollbackDecision.PROCEED
            assert "decision" in reasoning
            assert "comparisons" in reasoning
    
    @pytest.mark.asyncio
    async def test_fetch_canary_metrics_function(self):
        """Test the convenience fetch_canary_metrics function."""
        mock_v1 = CanaryMetrics(version="v1.0.0")
        mock_v2 = CanaryMetrics(version="v1.1.0")
        
        with patch('src.app.deployment.canary_analyzer.CanaryAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.compare_versions.return_value = (mock_v1, mock_v2)
            mock_analyzer.__aenter__.return_value = mock_analyzer
            mock_analyzer.__aexit__.return_value = None
            mock_analyzer_class.return_value = mock_analyzer
            
            v1, v2 = await fetch_canary_metrics("v1.0.0", "v1.1.0", 15)
            
            assert v1.version == "v1.0.0"
            assert v2.version == "v1.1.0"
            mock_analyzer.compare_versions.assert_called_once_with(
                "v1.0.0", "v1.1.0", 15
            )


# Mock for testing
class Mock:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def raise_for_status(self):
        pass
