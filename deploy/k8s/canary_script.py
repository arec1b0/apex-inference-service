import time
import subprocess
import requests
import sys
from typing import Optional

# Configuration
PROMETHEUS_URL = "http://localhost:9090"
INGRESS_NAME = "apex-inference-v2-apex-inference-service"
NAMESPACE = "default"
STEP_WEIGHT = 10
Step_INTERVAL = 30  # Seconds between checks
MAX_WEIGHT = 100

# Thresholds
MAX_ERROR_RATE = 0.01  # 1%
MAX_LATENCY_P90 = 0.5  # 500ms

def get_prometheus_metric(query: str) -> float:
    """Fetch a single scalar value from Prometheus."""
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success" and data["data"]["result"]:
            # Return the value from the first result
            return float(data["data"]["result"][0]["value"][1])
        return 0.0
    except Exception as e:
        print(f"Warning: Failed to fetch metric: {e}")
        return 0.0

def check_health() -> bool:
    """Returns True if metrics are healthy, False otherwise."""
    print("  [*] Checking metrics...")
    
    # 1. Error Rate (non-200 codes / total) - rate over last 1m
    query_errors = 'sum(rate(http_requests_total{status_code!~"2.."}[1m])) / sum(rate(http_requests_total[1m]))'
    error_rate = get_prometheus_metric(query_errors)
    
    # 2. Latency P90
    query_latency = 'histogram_quantile(0.9, sum(rate(http_request_duration_seconds_bucket[1m])) by (le))'
    latency = get_prometheus_metric(query_latency)
    
    print(f"  [-] Error Rate: {error_rate:.2%} (Threshold: {MAX_ERROR_RATE:.2%})")
    print(f"  [-] P90 Latency: {latency:.3f}s (Threshold: {MAX_LATENCY_P90}s)")
    
    if error_rate > MAX_ERROR_RATE:
        print("  [!] CRITICAL: Error rate too high!")
        return False
        
    if latency > MAX_LATENCY_P90:
        print("  [!] CRITICAL: Latency too high!")
        return False
        
    return True

def set_canary_weight(weight: int):
    """Patch the Ingress to shift traffic."""
    print(f"--> Setting Canary weight to {weight}%")
    cmd = [
        "kubectl", "annotate", "ingress", INGRESS_NAME,
        f"nginx.ingress.kubernetes.io/canary-weight={weight}",
        "--overwrite",
        "-n", NAMESPACE
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

def rollback():
    print("\n[!!!] INITIATING ROLLBACK [!!!]")
    set_canary_weight(0)
    print("Rollback complete. Traffic shifted back to stable.")
    sys.exit(1)

def run_rollout():
    print(f"Starting Canary Rollout for {INGRESS_NAME}...")
    current_weight = 0
    
    while current_weight < MAX_WEIGHT:
        # 1. Increment Weight
        current_weight += STEP_WEIGHT
        if current_weight > MAX_WEIGHT: 
            current_weight = MAX_WEIGHT
            
        set_canary_weight(current_weight)
        
        # 2. Wait for traffic to generate metrics
        print(f"Waiting {Step_INTERVAL}s for metrics to stabilize...")
        time.sleep(Step_INTERVAL)
        
        # 3. Check Health
        if not check_health():
            rollback()
            
    print("\n[+] Rollout Successful! Canary is now taking 100% traffic.")
    # In a real CI, you would now trigger the 'promote' job to replace v1 with v2

if __name__ == "__main__":
    # Ensure dependencies are met
    try:
        run_rollout()
    except KeyboardInterrupt:
        rollback()