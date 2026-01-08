# Canary Validation & Rollback Integration Guide

## Overview
The automated canary validation system provides production-grade deployment safety by comparing metrics between production and canary deployments before promoting new versions.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   CI/CD     │    │    Helm      │    │ Prometheus  │
│ Pipeline    │───▶│ Deployment   │───▶│   Metrics    │
└─────────────┘    └──────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌──────────────┐    ┌─────────────┐
                   │ Canary       │    │ Alert       │
                   │ Validator    │───▶│ System      │
                   └──────────────┘    └─────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │ Rollback/    │
                   │ Promote      │
                   └──────────────┘
```

## Key Components

### 1. CanaryMetrics
Dataclass containing performance metrics:
- Latency percentiles (P50, P95, P99)
- Error rate percentage
- Model confidence score
- Request rate
- Sample count and timestamp

### 2. CanaryAnalyzer
Fetches metrics from Prometheus:
- Configurable Prometheus queries
- Automatic metric validation
- Concurrent version comparison
- Error handling with defaults

### 3. RollbackPolicy
Evaluates metrics against thresholds:
- Automatic rollback on critical regressions
- Proceed decision for improvements
- Manual review for ambiguous cases
- Configurable thresholds

## Decision Logic

### Automatic Rollback Triggers:
- P99 latency > 150% of baseline
- Error rate > 200% of baseline
- Model confidence < 80% of baseline

### Proceed Conditions:
- All critical metrics within 110% of baseline
- No significant regressions

### Manual Review:
- Mixed performance (some better, some worse)
- Non-critical regressions
- Insufficient data

## Integration Steps

### 1. Prometheus Metrics Setup

Ensure your application exports the required metrics:

```python
from prometheus_client import Histogram, Counter, Gauge

# Request latency histogram
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['version'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Request counter
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['version', 'status']
)

# Model confidence gauge
MODEL_CONFIDENCE = Gauge(
    'model_prediction_confidence',
    'Model prediction confidence',
    ['version']
)
```

### 2. Helm Chart Configuration

Add canary support to your Helm chart:

```yaml
# values.yaml
canary:
  enabled: false
  replicas: 1
  weight: 10  # Percentage of traffic

image:
  repository: ghcr.io/your-org/apex-inference
  tag: latest

# values-canary.yaml
canary:
  enabled: true
  replicas: 1
  weight: 10

service:
  annotations:
    # Enable Istio canary (if using Istio)
    "getambassador.io/config": |
      ---
      apiVersion: ambassador/v2
      kind: Mapping
      name: apex-inference-mapping
      hostname: "*"
      prefix: /
      service: apex-inference-canary:8000
      weight: 10
```

### 3. Deployment Script

Create the main validation script:

```python
#!/usr/bin/env python3
"""Deploy with canary validation."""

import asyncio
import os
from src.app.deployment import CanaryValidator, HelmManager, AlertManager

async def main():
    # Configuration
    config = {
        "version": os.getenv("VERSION", "latest"),
        "production_version": os.getenv("PRODUCTION_VERSION", "v1.0.0"),
        "values_file": os.getenv("VALUES_FILE", "values-canary.yaml"),
        "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
        "prometheus_url": os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    }
    
    # Initialize components
    helm = HelmManager()
    alerts = AlertManager(config["slack_webhook"])
    validator = CanaryValidator(helm, alerts, config["prometheus_url"])
    
    # Run validation
    success = await validator.validate_canary(
        version=config["version"],
        production_version=config["production_version"],
        values_file=config["values_file"],
        wait_time_minutes=15
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
```

### 4. GitHub Actions Workflow

See `.github/workflows/deploy-canary.yml` for complete workflow.

### 5. Environment Variables

Configure the following secrets in GitHub:

```bash
# Required
KUBE_CONFIG_PROD          # Base64-encoded kubeconfig for production
KUBE_CONFIG_STAGING       # Base64-encoded kubeconfig for staging
GITHUB_TOKEN             # For container registry access

# Optional
SLACK_WEBHOOK_URL        # For notifications
PROMETHEUS_URL           # Custom Prometheus URL
```

## Configuration Options

### Rollback Thresholds

Customize thresholds based on your service requirements:

```python
from src.app.deployment import RollbackThresholds, RollbackPolicy

# Custom thresholds for latency-sensitive service
thresholds = RollbackThresholds(
    latency_p99_threshold=1.3,  # 30% increase triggers rollback
    error_rate_threshold=1.5,   # 50% increase triggers rollback
    confidence_threshold=0.85,  # 85% of baseline required
    proceed_tolerance=1.05      # 5% tolerance for promotion
)

policy = RollbackPolicy(thresholds)
```

### Prometheus Queries

Customize queries for your metrics:

```python
custom_queries = {
    "latency_p99": 'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{version="%s"}[5m])) by (le)) * 1000',
    "error_rate": '(sum(rate(http_requests_total{version="%s",status=~"5.."}[5m])) / sum(rate(http_requests_total{version="%s"}[5m]))) * 100',
    # Add custom queries
}

analyzer = CanaryAnalyzer(prometheus_url="http://prometheus:9090")
analyzer.queries.update(custom_queries)
```

## Monitoring and Alerting

### 1. Canary Metrics Dashboard

Create a Grafana dashboard showing:
- Real-time metric comparison
- Historical canary performance
- Rollback frequency
- Time to promotion

### 2. Alert Rules

Configure PrometheusAlertManager rules:

```yaml
# canary-alerts.yml
groups:
  - name: canary
    rules:
      - alert: CanaryRollback
        expr: increase(canary_rollback_total[1h]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Automatic rollback triggered"
      
      - alert: CanaryValidationFailure
        expr: canary_validation_success == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Canary validation failing"
```

### 3. Slack Integration

Set up Slack alerts with detailed information:

```python
# Enhanced alert message
message = f"""
🚨 Automatic rollback triggered for {version}

Metrics:
- P99 Latency: {v2_metrics.latency_p99}ms (was {v1_metrics.latency_p99}ms)
- Error Rate: {v2_metrics.error_rate}% (was {v1_metrics.error_rate}%)
- Confidence: {v2_metrics.model_confidence:.2f} (was {v1_metrics.model_confidence:.2f})

Action: Investigation required
Dashboard: {dashboard_url}
"""
```

## Best Practices

### 1. Traffic Splitting
- Start with 5-10% traffic to canary
- Increase gradually if metrics look good
- Use feature flags for additional control

### 2. Metric Selection
- Focus on business-critical metrics
- Include latency, error rate, and quality metrics
- Avoid noisy metrics that fluctuate normally

### 3. Timing Considerations
- Allow sufficient time for metrics to stabilize
- Account for warm-up periods
- Consider time-of-day patterns

### 4. Rollback Strategy
- Always have a quick rollback path
- Test rollback procedures regularly
- Document rollback reasons

### 5. Testing
- Test canary validation in staging first
- Simulate various failure scenarios
- Validate alert configurations

## Troubleshooting

### Issue: Insufficient Metrics
**Symptoms**: Validation fails with "insufficient data"

**Solutions**:
- Increase wait time before validation
- Check Prometheus query syntax
- Verify version labels are correct
- Ensure canary is receiving traffic

### Issue: False Positive Rollbacks
**Symptoms**: Rollback triggered without actual issues

**Solutions**:
- Adjust thresholds for your service
- Implement metric smoothing
- Add minimum sample requirements
- Exclude non-critical metrics

### Issue: Canary Not Receiving Traffic
**Symptoms**: No metrics for canary version

**Solutions**:
- Verify service configuration
- Check ingress/ambassador settings
- Validate traffic splitting rules
- Ensure pods are healthy

### Issue: Slow Validation
**Symptoms**: Validation takes too long

**Solutions**:
- Optimize Prometheus queries
- Use metric pre-aggregation
- Implement caching
- Reduce analysis window

## Advanced Features

### 1. Progressive Canary
```python
# Gradually increase canary traffic
async def progressive_canary(validator, version):
    traffic_weights = [5, 10, 25, 50]
    
    for weight in traffic_weights:
        await update_canary_weight(weight)
        success = await validator.validate_canary(
            version=version,
            wait_time_minutes=5
        )
        
        if not success:
            return False
    
    return True
```

### 2. A/B Testing Integration
```python
# Compare business metrics alongside technical metrics
async def validate_with_ab_test(validator, version):
    # Get technical metrics
    tech_decision, _ = await validator.validate_canary(version)
    
    # Get business metrics
    business_metrics = await get_business_metrics(version)
    
    # Combine decisions
    if tech_decision == RollbackDecision.PROCEED and business_metrics.passes:
        return RollbackDecision.PROCEED
    
    return RollbackDecision.ROLLBACK_MANUAL
```

### 3. Multi-Region Canary
```python
# Validate across multiple regions
async def multi_region_canary(validator, version, regions):
    results = []
    
    for region in regions:
        region_validator = CanaryValidator(
            prometheus_url=f"http://prometheus-{region}.internal"
        )
        result = await region_validator.validate_canary(version)
        results.append(result)
    
    # Require success in all regions
    return all(r.success for r in results)
```

## Production Checklist

- [ ] Prometheus metrics properly labeled with version
- [ ] Helm chart supports canary deployments
- [ ] Traffic splitting configured
- [ ] Alert channels set up (Slack, PagerDuty)
- [ ] Rollback procedures tested
- [ ] Monitoring dashboards created
- [ ] CI/CD pipeline integrated
- [ ] Environment variables configured
- [ ] Thresholds tuned for your service
- [ ] Documentation updated
- [ ] Team trained on procedures
- [ ] Incident response plan ready
