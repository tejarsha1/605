import base64
import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import json
import os
# ===== CONFIG =====
SYSTEM_NAME = "multi_container_concurrent"
URL = "http://localhost:8000/predict"
IMAGE_PATH = "../../sample.png"
NUM_REQUESTS = 1000
CONCURRENCY = 50
TIMEOUT = 10

# ===== LOAD IMAGE =====
with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {"image_base64": image_base64}

# ===== REQUEST FUNCTION =====
def send_request():
    start = time.time()
    try:
        response = requests.post(URL, json=payload, timeout=TIMEOUT)
        end = time.time()
        latency = end - start

        if response.status_code == 200:
            return True, latency, response.status_code
        else:
            return False, latency, response.status_code

    except Exception:
        end = time.time()
        return False, end - start, "error"

# ===== RUN LOAD TEST =====
latencies = []
success = 0
failed = 0
status_codes = {}

start_total = time.time()

with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    futures = [executor.submit(send_request) for _ in range(NUM_REQUESTS)]

    for future in as_completed(futures):
        ok, latency, status = future.result()
        status_codes[status] = status_codes.get(status, 0) + 1

        if ok:
            success += 1
            latencies.append(latency)
        else:
            failed += 1

end_total = time.time()

# ===== METRICS =====
total_time = end_total - start_total
throughput = success / total_time if total_time > 0 else 0
success_rate = (success / NUM_REQUESTS) * 100
error_rate = (failed / NUM_REQUESTS) * 100

# ===== PRINT RESULTS =====
print("\n===== CONCURRENT LOAD TEST RESULTS =====")
print("System:", SYSTEM_NAME)
print("Total requests:", NUM_REQUESTS)
print("Concurrency:", CONCURRENCY)
print("Successful:", success)
print("Failed:", failed)
print("Success rate:", round(success_rate, 2), "%")
print("Error rate:", round(error_rate, 2), "%")
print("Status codes:", status_codes)
print("Total duration:", round(total_time, 4), "sec")
print("Throughput:", round(throughput, 2), "req/sec")

# ===== LATENCY METRICS + GRAPHS =====
if latencies:
    lat = np.array(latencies)

    avg_latency = np.mean(lat)
    median_latency = np.median(lat)
    min_latency = np.min(lat)
    max_latency = np.max(lat)
    p50 = np.percentile(lat, 50)
    p75 = np.percentile(lat, 75)
    p90 = np.percentile(lat, 90)
    p95 = np.percentile(lat, 95)
    p99 = np.percentile(lat, 99)
    std_dev = np.std(lat)

    print("\n--- Latency Metrics ---")
    print("Average latency:", round(avg_latency, 4), "sec")
    print("Median latency:", round(median_latency, 4), "sec")
    print("Min latency:", round(min_latency, 4), "sec")
    print("Max latency:", round(max_latency, 4), "sec")
    print("P50 latency:", round(p50, 4), "sec")
    print("P75 latency:", round(p75, 4), "sec")
    print("P90 latency:", round(p90, 4), "sec")
    print("P95 latency:", round(p95, 4), "sec")
    print("P99 latency:", round(p99, 4), "sec")
    print("Std deviation:", round(std_dev, 4))

    # ===== GRAPH 1: LATENCY SUMMARY =====
    labels = ["Avg", "Median", "P95", "P99"]
    values = [avg_latency, median_latency, p95, p99]

    plt.figure()
    plt.bar(labels, values)
    plt.title(f"{SYSTEM_NAME} Latency Summary")
    plt.ylabel("Latency (sec)")
    plt.savefig(f"{SYSTEM_NAME}_latency_summary.png")

    # ===== GRAPH 2: THROUGHPUT =====
    plt.figure()
    plt.bar([SYSTEM_NAME], [throughput])
    plt.title(f"{SYSTEM_NAME} Throughput")
    plt.ylabel("req/sec")
    plt.savefig(f"{SYSTEM_NAME}_throughput.png")

    print("\nGraphs saved:")
    print(f"✔ {SYSTEM_NAME}_latency_summary.png")
    print(f"✔ {SYSTEM_NAME}_throughput.png")

else:
    print("No successful requests.")

# ===== SAVE RESULTS FOR FINAL COMPARISON GRAPH =====
if latencies:
    result = {
        "system": SYSTEM_NAME,
        "test_type": "concurrent",
        "total_requests": NUM_REQUESTS,
        "concurrency": CONCURRENCY,
        "successful": success,
        "failed": failed,
        "success_rate": round(success_rate, 2),
        "error_rate": round(error_rate, 2),
        "throughput": round(throughput, 4),
        "avg_latency": round(float(avg_latency), 4),
        "median_latency": round(float(median_latency), 4),
        "p95_latency": round(float(p95), 4),
        "p99_latency": round(float(p99), 4),
    }

    results_file = "../../results.json"

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[SYSTEM_NAME] = result

    with open(results_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✔ Saved metrics to {results_file}")