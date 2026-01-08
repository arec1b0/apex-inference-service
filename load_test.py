import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost/api/v1/predict"
HEADERS = {"Content-Type": "application/json"}

def send_request():
    try:
        # Randomize input to vary model predictions
        features = [
            random.uniform(4.0, 7.0),
            random.uniform(2.0, 4.0),
            random.uniform(1.0, 6.0),
            random.uniform(0.1, 2.5)
        ]
        
        # 5% chance to send malformed data (triggers 422 errors for testing)
        if random.random() < 0.05:
            payload = {"id": "bad-req"}
        else:
            payload = {"id": "load-test", "features": features}

        resp = requests.post(URL, json=payload, headers=HEADERS, timeout=1)
        print(f".", end="", flush=True)
    except Exception:
        print("x", end="", flush=True)

def run_load():
    print("Starting load generator (Press Ctrl+C to stop)...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        while True:
            executor.submit(send_request)
            time.sleep(0.1) # 10 RPS

if __name__ == "__main__":
    run_load()