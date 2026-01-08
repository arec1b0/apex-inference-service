# Data Drift Detection Integration Guide

## Overview
The data drift detection system monitors changes in input feature distributions compared to reference data from model training time. It uses statistical tests (KS statistic, PSI) to detect distribution shifts that may indicate model performance degradation.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Production    │    │   Feature Store  │    │  Drift Monitor  │
│   Predictions   │───▶│   (Database)     │───▶│  (Batch Job)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │  Alert System   │
                                              │  (Slack/Email)  │
                                              └─────────────────┘
```

## Integration Steps

### 1. Reference Distribution Generation

Generate reference distribution during model deployment:

```python
# In your model deployment script
from src.app.monitoring.drift_detector import ReferenceDistribution
import numpy as np

# Load training data
X_train, _, _, _ = load_training_data()
feature_names = ["age", "income", "credit_score", "loan_amount"]

# Create reference distribution
reference = ReferenceDistribution.from_data(X_train, feature_names)

# Save with model version
model_version = "v1.2.0"
reference.save(f"model_artifacts/{model_version}/reference_distribution.json")

print(f"Reference distribution saved for {model_version}")
```

### 2. Feature Storage Setup

Store prediction features for drift analysis:

#### Option A: PostgreSQL
```python
# Add to your prediction endpoint
async def store_prediction_features(request_id: str, features: np.ndarray):
    """Store features in PostgreSQL for drift analysis."""
    async with db.pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO prediction_logs (request_id, feature_vector, created_at)
            VALUES ($1, $2, NOW())
        """, request_id, features.tolist())
```

#### Option B: Redis (for high throughput)
```python
# Store recent features in Redis list
async def store_features_redis(features: np.ndarray):
    """Store features in Redis with TTL."""
    feature_json = json.dumps(features.tolist())
    await redis.lpush("recent_features", feature_json)
    await redis.ltrim("recent_features", 0, 9999)  # Keep last 10k
    await redis.expire("recent_features", 86400)  # 24 hour TTL
```

#### Option C: Time Series Database (InfluxDB)
```python
# Store as time series data
async def store_features_influx(features: np.ndarray, feature_names: List[str]):
    """Store features in InfluxDB."""
    points = []
    for i, name in enumerate(feature_names):
        points.append({
            "measurement": "model_features",
            "tags": {"feature_name": name},
            "fields": {"value": float(features[i])},
            "time": datetime.utcnow()
        })
    
    write_client.write_points(points)
```

### 3. Feature Retrieval Implementation

Implement the feature retrieval function for the drift job:

```python
async def get_recent_prediction_features(window_hours: int = 1) -> np.ndarray:
    """Fetch features from the last N hours."""
    
    # PostgreSQL example
    async with db.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT feature_vector 
            FROM prediction_logs 
            WHERE created_at > NOW() - INTERVAL '{window_hours} hours'
            LIMIT 10000
        """.format(window_hours=window_hours))
        
        if not rows:
            return np.array([]).reshape(0, n_features)
        
        features = np.array([row['feature_vector'] for row in rows])
        return features
```

### 4. Scheduler Integration

Set up APScheduler in your main application:

```python
# main.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.app.monitoring.drift_detector import drift_check_job

async def start_scheduler():
    """Initialize and start the drift monitoring scheduler."""
    scheduler = AsyncIOScheduler()
    
    # Schedule drift check every 2 hours
    scheduler.add_job(
        drift_check_job,
        trigger="interval",
        hours=2,
        args=[
            get_recent_prediction_features,
            "model_artifacts/v1.2.0/reference_distribution.json"
        ],
        id="drift_monitor",
        name="Data drift detection",
        max_instances=1,
        misfire_grace_time=300
    )
    
    scheduler.start()
    logger.info("Drift monitoring scheduler started")

# Call this in your startup
@app.on_event("startup")
async def startup_event():
    await start_scheduler()
```

### 5. Alert Configuration

Set up alerts when drift is detected:

```python
# In your drift monitoring module
async def send_drift_alert(drift_result: Dict[str, Any]):
    """Send alert when significant drift is detected."""
    
    # Slack alert
    if slack_webhook_url:
        message = {
            "text": "🚨 Data Drift Detected!",
            "attachments": [{
                "color": "danger",
                "fields": [
                    {"title": "Affected Features", 
                     "value": ", ".join(drift_result["drifted_features"].keys())},
                    {"title": "Max Drift Score", 
                     "value": f"{drift_result['max_drift_score']:.3f}"},
                    {"title": "Sample Count", 
                     "value": str(drift_result["sample_count"])},
                    {"title": "Timestamp", 
                     "value": drift_result["timestamp"]}
                ]
            }]
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(slack_webhook_url, json=message)
    
    # Email alert
    if alert_emails:
        await send_email(
            to=alert_emails,
            subject=f"Data Drift Alert - {drift_result['max_drift_score']:.3f}",
            template="drift_alert.html",
            context=drift_result
        )
    
    # Trigger retraining pipeline
    if drift_result["max_drift_score"] > 0.5:
        logger.warning("High drift detected - triggering retraining")
        await trigger_retraining_pipeline()

# Modify drift_check_job to include alerts
async def drift_check_job_with_alerts(get_features_func, reference_path: str):
    """Drift check job with alert integration."""
    result = await drift_check_job(get_features_func, reference_path)
    
    if result.get("drift_detected"):
        await send_drift_alert(result)
    
    return result
```

### 6. Dashboard Integration

Add drift metrics to your monitoring dashboard:

```python
# metrics endpoint
@app.get("/metrics/drift")
async def get_drift_metrics():
    """Get current drift metrics for dashboard."""
    
    # Get latest drift scores from Prometheus
    drift_metrics = {
        "feature_scores": {},
        "last_check": None,
        "drift_detected": False
    }
    
    # Query Prometheus for drift scores
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://prometheus:9090/api/v1/query",
            params={"query": "feature_drift_score"}
        )
        
        if response.status_code == 200:
            data = response.json()
            for result in data["data"]["result"]:
                feature = result["metric"]["feature_name"]
                score = float(result["value"][1])
                drift_metrics["feature_scores"][feature] = score
                
                if score > 0.3:
                    drift_metrics["drift_detected"] = True
    
    return drift_metrics
```

## Configuration

### Environment Variables
```bash
# Drift detection settings
DRIFT_THRESHOLD=0.3
DRIFT_CHECK_INTERVAL_HOURS=2
DRIFT_MIN_SAMPLES=100

# Alert settings
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
ALERT_EMAILS=admin@company.com,ml-team@company.com

# Feature storage
FEATURE_STORE_TYPE=postgresql  # postgresql, redis, influxdb
FEATURE_RETENTION_HOURS=24
```

### Model Versioning
```python
# Store reference distribution with each model version
model_artifacts/
├── v1.0.0/
│   ├── model.pkl
│   └── reference_distribution.json
├── v1.1.0/
│   ├── model.pkl
│   └── reference_distribution.json
└── v1.2.0/
    ├── model.pkl
    └── reference_distribution.json
```

## Best Practices

### 1. Reference Data Quality
- Use representative training data
- Include all feature transformations
- Store feature names and preprocessing steps
- Version with model artifacts

### 2. Sampling Strategy
- Minimum 100 samples for drift detection
- Prefer recent data (last 1-6 hours)
- Stratify by key segments if needed
- Handle missing values appropriately

### 3. Threshold Tuning
- Start with 0.3 threshold
- Adjust based on false positive rate
- Consider per-feature thresholds
- Account for seasonal patterns

### 4. Performance Optimization
- Use efficient feature storage
- Batch feature retrieval
- Cache reference distributions
- Parallelize drift calculations

### 5. Monitoring the Monitor
- Track drift check execution time
- Monitor feature storage size
- Alert on check failures
- Log drift score trends

## Troubleshooting

### Issue: High false positive rate
**Solution**: 
- Increase drift threshold
- Use longer time windows
- Account for known patterns (hourly/daily)
- Implement smoothing

### Issue: Missing features in reference
**Solution**:
- Verify feature order consistency
- Check preprocessing pipeline
- Update reference distribution
- Add feature mapping

### Issue: Insufficient samples
**Solution**:
- Extend time window
- Lower minimum sample threshold
- Check feature storage retention
- Verify data pipeline

### Issue: Slow drift checks
**Solution**:
- Optimize database queries
- Use feature sampling
- Implement incremental updates
- Consider approximate methods

## Advanced Features

### 1. Multivariate Drift Detection
```python
# Add to DriftDetector class
def multivariate_drift(self, reference: np.ndarray, current: np.ndarray) -> float:
    """Detect drift using multivariate methods."""
    # Maximum Mean Discrepancy
    from sklearn.metrics import pairwise_distances
    
    mmd = compute_mmd(reference, current)
    return min(mmd / 10.0, 1.0)  # Normalize to 0-1
```

### 2. Concept Drift Detection
```python
# Monitor prediction confidence drift
async def check_confidence_drift():
    """Detect drift in prediction confidence scores."""
    recent_confidences = await get_recent_confidences()
    reference_confidences = load_reference_confidences()
    
    drift_score = detector.ks_statistic(
        reference_confidences,
        recent_confidences
    )
    
    return drift_score
```

### 3. Automated Retraining
```python
# Trigger retraining pipeline
async def trigger_retraining_pipeline():
    """Trigger automated model retraining."""
    
    # Create new training dataset with recent data
    recent_data = await get_recent_labeled_data()
    
    # Start retraining job
    job_id = await mlflow.start_run(
        experiment_name="automated_retraining",
        tags={"trigger": "data_drift"}
    )
    
    # Queue training job
    await training_queue.enqueue(
        "retrain_model",
        dataset=recent_data,
        run_id=job_id
    )
    
    logger.info(f"Retraining triggered: job_id={job_id}")
```

## Testing

### Unit Tests
```bash
# Run all drift detection tests
pytest tests/monitoring/test_drift_detector.py -v

# Test specific components
pytest tests/monitoring/test_drift_detector.py::TestDriftDetector -v
pytest tests/monitoring/test_drift_detector.py::TestReferenceDistribution -v
```

### Integration Tests
```python
# Test with real data
async def test_drift_detection_integration():
    """Test drift detection with production-like data."""
    
    # Load real training data
    X_train = load_production_training_data()
    reference = ReferenceDistribution.from_data(X_train, feature_names)
    
    # Simulate production drift
    X_prod = simulate_production_drift(X_train, drift_factor=0.5)
    
    # Detect drift
    detector = DriftDetector()
    scores = detector.detect(X_prod, reference)
    
    # Verify drift detected
    assert max(scores.values()) > 0.3
```

### Load Testing
```python
# Test performance with large datasets
async def test_drift_performance():
    """Test drift detection performance."""
    
    # Generate large dataset
    X_large = np.random.randn(100000, 100)
    
    # Time detection
    start_time = time.time()
    scores = detector.detect(X_large, reference)
    duration = time.time() - start_time
    
    # Should complete within reasonable time
    assert duration < 10.0
    print(f"Processed 100k samples in {duration:.2f}s")
```

## Production Checklist

- [ ] Reference distribution generated and versioned
- [ ] Feature storage implemented and tested
- [ ] Scheduler configured with appropriate interval
- [ ] Alert channels set up and verified
- [ ] Dashboard metrics configured
- [ ] Error handling and retry logic
- [ ] Monitoring for the monitoring system
- [ ] Documentation updated
- [ ] Team trained on drift alerts
- [ ] Retraining pipeline integrated
- [ ] Performance benchmarks met
- [ ] Security review completed
