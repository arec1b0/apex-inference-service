"""Example demonstrating data drift detection integration.

This script shows how to set up and run drift detection monitoring.
Run with: python examples/drift_detection_example.py
"""
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.app.monitoring.drift_detector import (
    ReferenceDistribution,
    DriftDetector,
    DriftMonitor,
    drift_check_job
)


# Mock feature storage for demonstration
class MockFeatureStore:
    """Mock storage for recent prediction features."""
    
    def __init__(self):
        self.features = []
        self.timestamps = []
    
    def add_features(self, features: np.ndarray):
        """Add a batch of features with timestamp."""
        self.features.append(features)
        self.timestamps.append(datetime.now(timezone.utc))
    
    async def get_recent_features(self, window_hours: int = 1) -> np.ndarray:
        """Get features from the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        
        recent_features = []
        for features, timestamp in zip(self.features, self.timestamps):
            if timestamp > cutoff:
                recent_features.append(features)
        
        if recent_features:
            return np.vstack(recent_features)
        return np.array([]).reshape(0, 3)  # Empty array with correct shape


async def simulate_production_data():
    """Simulate production data with gradual drift."""
    feature_store = MockFeatureStore()
    
    # Feature names
    feature_names = ["temperature", "humidity", "pressure"]
    
    # Generate initial training data (reference)
    print("📊 Generating reference distribution...")
    np.random.seed(42)
    training_data = np.random.multivariate_normal(
        mean=[20.0, 50.0, 1013.25],
        cov=[[4.0, 0.5, 1.0],
             [0.5, 25.0, 2.0],
             [1.0, 2.0, 100.0]],
        size=1000
    )
    
    # Create and save reference distribution
    reference = ReferenceDistribution.from_data(training_data, feature_names)
    reference_path = Path("reference_distribution.json")
    reference.save(reference_path)
    print(f"✅ Reference distribution saved to {reference_path}")
    
    # Simulate production data over time
    print("\n🔄 Simulating production data over time...")
    
    for hour in range(24):  # Simulate 24 hours
        # Gradually introduce drift in temperature
        temp_drift = hour * 0.5  # Temperature increases by 0.5°C per hour
        
        # Generate batch of production data
        batch_size = np.random.randint(50, 150)
        production_data = np.random.multivariate_normal(
            mean=[20.0 + temp_drift, 50.0, 1013.25],
            cov=[[4.0, 0.5, 1.0],
                 [0.5, 25.0, 2.0],
                 [1.0, 2.0, 100.0]],
            size=batch_size
        )
        
        # Add to feature store
        feature_store.add_features(production_data)
        
        # Run drift check every 4 hours
        if hour % 4 == 0 and hour > 0:
            print(f"\n⏰ Hour {hour}: Running drift check...")
            
            # Create feature retrieval function
            async def get_features():
                return feature_store.get_recent_features(window_hours=4)
            
            # Run drift check job
            result = await drift_check_job(get_features, str(reference_path))
            
            # Display results
            if result["status"] == "success":
                print(f"  Status: ✅ Success")
                print(f"  Samples analyzed: {result['sample_count']}")
                print(f"  Max drift score: {result['max_drift_score']:.3f}")
                
                if result["drift_detected"]:
                    print(f"  🚨 DRIFT DETECTED in {len(result['drifted_features'])} features:")
                    for feature, score in result["drifted_features"].items():
                        print(f"    - {feature}: {score:.3f}")
                else:
                    print(f"  ✅ No significant drift detected")
                    
                # Show drift scores for all features
                print(f"  All drift scores:")
                for feature, score in result["drift_scores"].items():
                    status = "⚠️ " if score > 0.3 else "✅ "
                    print(f"    {status}{feature}: {score:.3f}")
            else:
                print(f"  ❌ Error: {result.get('message', 'Unknown error')}")
        
        # Small delay to make simulation realistic
        await asyncio.sleep(0.1)
    
    print("\n📈 Simulation complete!")
    
    # Final analysis
    print("\n📊 Final drift analysis:")
    all_features = feature_store.get_recent_features(window_hours=24)
    
    if len(all_features) > 0:
        # Load reference and create detector
        reference = ReferenceDistribution.from_file(reference_path)
        detector = DriftDetector(drift_threshold=0.3)
        
        # Detect drift in all data
        final_scores = detector.detect(all_features, reference)
        
        print("\nFinal drift scores (24 hours):")
        for feature, score in final_scores.items():
            if score > 0.3:
                print(f"  🚨 {feature}: {score:.3f} (SIGNIFICANT DRIFT)")
            else:
                print(f"  ✅ {feature}: {score:.3f}")
    
    # Cleanup
    if reference_path.exists():
        reference_path.unlink()


async def demonstrate_drift_detection_methods():
    """Demonstrate different drift detection methods."""
    print("\n" + "="*60)
    print("🔬 Drift Detection Methods Comparison")
    print("="*60)
    
    # Create reference data
    np.random.seed(42)
    reference_data = np.random.normal(0, 1, 1000)
    
    # Create test scenarios
    scenarios = {
        "No drift": np.random.normal(0, 1, 1000),
        "Mean shift": np.random.normal(2, 1, 1000),  # Shifted mean
        "Variance change": np.random.normal(0, 2, 1000),  # Higher variance
        "Both shift": np.random.normal(2, 2, 1000),  # Both mean and variance
    }
    
    detector = DriftDetector()
    
    print("\nComparing KS statistic vs PSI score:")
    print("-" * 50)
    print(f"{'Scenario':<15} {'KS Score':<10} {'PSI Score':<10} {'Combined':<10}")
    print("-" * 50)
    
    for scenario_name, current_data in scenarios.items():
        ks_score = detector.ks_statistic(reference_data, current_data.tolist())
        psi_score = detector.psi_score(reference_data, current_data.tolist())
        combined = 0.6 * ks_score + 0.4 * psi_score
        
        print(f"{scenario_name:<15} {ks_score:<10.3f} {psi_score:<10.3f} {combined:<10.3f}")


async def setup_scheduler_example():
    """Example of setting up APScheduler for drift monitoring."""
    print("\n" + "="*60)
    print("⏰ Scheduler Setup Example")
    print("="*60)
    
    scheduler_code = '''
# Add to your main application:
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.app.monitoring.drift_detector import drift_check_job

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Feature retrieval function (implement based on your storage)
async def get_recent_features():
    """Fetch features from your database/storage."""
    # Example: PostgreSQL
    # async with db.pool.acquire() as conn:
    #     rows = await conn.fetch("""
    #         SELECT feature_vector FROM prediction_logs
    #         WHERE created_at > NOW() - INTERVAL '1 hour'
    #     """)
    #     return np.array([row['feature_vector'] for row in rows])
    
    # Example: Redis
    # features = await redis.lrange("recent_features", 0, -1)
    # return np.array([json.loads(f) for f in features])
    
    pass

# Schedule drift check to run every hour
scheduler.add_job(
    drift_check_job,
    trigger="interval",
    hours=1,
    args=[get_recent_features, "reference_distribution.json"],
    id="drift_check_job",
    name="Data drift detection",
    max_instances=1,  # Prevent overlapping runs
    misfire_grace_time=300,  # 5 minutes grace time
)

# Start scheduler
scheduler.start()
'''
    
    print("Add this code to your application:")
    print(scheduler_code)
    
    print("\n📌 Key configuration options:")
    print("- Run every 1-6 hours depending on data volume")
    print("- Use max_instances=1 to prevent overlapping jobs")
    print("- Set misfire_grace_time for system maintenance windows")
    print("- Store reference distribution with model version")


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Data Drift Detection Example")
    print("=" * 60)
    
    # Run main simulation
    asyncio.run(simulate_production_data())
    
    # Demonstrate detection methods
    asyncio.run(demonstrate_drift_detection_methods())
    
    # Show scheduler setup
    asyncio.run(setup_scheduler_example())
    
    print("\n✅ Example completed!")
    print("\nNext steps:")
    print("1. Integrate with your feature storage/database")
    print("2. Set up APScheduler in your main application")
    print("3. Configure alerts for drift detection")
    print("4. Add drift metrics to your monitoring dashboard")
    print("5. Implement automated retraining triggers")
