import time
import sys
import requests
import random
from typing import Dict

# This script simulates the Logic required to shift traffic weights
# In a real scenario, this would interact with K8s API (Ingress) or Istio VirtualService

class CanaryManager:
    def __init__(self, service_url: str, prom_url: str):
        self.service_url = service_url
        self.prom_url = prom_url
        self.current_weight = 5
        self.max_weight = 100
        self.step = 10
    
    def check_health(self) -> bool:
        """Query Prometheus for error rate of the canary version."""
        # Simulated check
        print(f"Checking metrics for weight {self.current_weight}%...")
        
        # Simulate a metric check (e.g., error rate < 1%)
        # In reality: requests.get(f"{self.prom_url}/api/v1/query?query=...")
        error_rate = random.uniform(0, 0.02)
        
        if error_rate > 0.015:
            print(f"Health check FAILED. Error rate: {error_rate:.2%}")
            return False
        
        print(f"Health check PASSED. Error rate: {error_rate:.2%}")
        return True

    def update_weight(self, weight: int):
        """Call K8s API to patch Ingress resource."""
        print(f"--> Patching Ingress: Set canary-weight to {weight}")
        # subprocess.run(["kubectl", "annotate", "ingress", "...", f"nginx.ingress.kubernetes.io/canary-weight={weight}", "--overwrite"])
        self.current_weight = weight
        time.sleep(2) # Wait for propagation

    def run_rollout(self):
        print("Starting Canary Rollout...")
        
        while self.current_weight < self.max_weight:
            if not self.check_health():
                print("!!! DEGRADING DETECTED. INITIATING ROLLBACK !!!")
                self.rollback()
                sys.exit(1)
            
            new_weight = min(self.current_weight + self.step, self.max_weight)
            self.update_weight(new_weight)
            time.sleep(1) # Wait for stabilization
            
        print("Rollout Complete. Promoting version.")

    def rollback(self):
        self.update_weight(0)
        print("Rollback complete.")

if __name__ == "__main__":
    # Example usage
    manager = CanaryManager("http://localhost:8000", "http://prometheus:9090")
    manager.run_rollout()