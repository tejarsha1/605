import base64
import time
import csv
import requests
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
SYSTEM_NAME = "monolithic"
URL = "http://localhost:8010/predict"
IMAGE_PATH = "sample.png"
NUM_REQUESTS = 1000
TIMEOUT = 10

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {"image_base64": image_base64}

latencies = []
success = 0
failed = 0
status_codes = {}
errors = []

start_total = time.time()

for i in range(NUM_REQUESTS):
    start = time.time()

    try:
        response = requests.post(URL, json=payload, timeout=TIMEOUT)
        end = time.time()
        latency = end - start

        status_codes[response.status_code] = status_codes.get(response.status_code, 0) + 1

        if response.status_code == 200:
            latencies.append(latency)
            success += 1
        else:
            failed += 1
            errors.append(f"Request {i+1}: Status {response.status_code}")

    except Exception as e:
        failed += 1
        errors.append(f"Request {i+1}: {str(e)}")

end_total = time.time()

total_time = end_total - start_total
throughput = success / total_time if total_time > 0 else 0
success_rate = (success / NUM_REQUESTS) * 100
error_rate = (failed / NUM_REQUESTS) * 100

print("\n===== RESULTS =====")
print("System:", SYSTEM_NAME)
print("Total requests:", NUM_REQUESTS)
print("Successful:", success)
print("Failed:", failed)
print("Success rate:", round(success_rate, 2), "%")
print("Error rate:", round(error_rate, 2), "%")
print("Status codes:", status_codes)
print("Total duration:", round(total_time, 4), "sec")
print("Throughput:", round(throughput, 2), "req/sec")

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
    variance = np.var(lat)
    first_request = lat[0]

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
    print("Variance:", round(variance, 6))
    print("First request:", round(first_request, 4), "sec")

    if len(lat) > 1:
        warm_lat = lat[1:]
        print("\n--- Warmed-up Metrics Excluding First Request ---")
        print("Warm avg latency:", round(np.mean(warm_lat), 4), "sec")
        print("Warm P95 latency:", round(np.percentile(warm_lat, 95), 4), "sec")
        print("Warm P99 latency:", round(np.percentile(warm_lat, 99), 4), "sec")

    # Save metrics to CSV
    csv_file = f"{SYSTEM_NAME}_metrics.csv"
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["System", SYSTEM_NAME])
        writer.writerow(["Endpoint", URL])
        writer.writerow(["Total Requests", NUM_REQUESTS])
        writer.writerow(["Successful Requests", success])
        writer.writerow(["Failed Requests", failed])
        writer.writerow(["Success Rate (%)", round(success_rate, 2)])
        writer.writerow(["Error Rate (%)", round(error_rate, 2)])
        writer.writerow(["Status Codes", status_codes])
        writer.writerow(["Total Duration (sec)", round(total_time, 4)])
        writer.writerow(["Throughput (req/sec)", round(throughput, 2)])
        writer.writerow(["Average Latency (sec)", round(avg_latency, 4)])
        writer.writerow(["Median Latency (sec)", round(median_latency, 4)])
        writer.writerow(["Min Latency (sec)", round(min_latency, 4)])
        writer.writerow(["Max Latency (sec)", round(max_latency, 4)])
        writer.writerow(["P50 Latency (sec)", round(p50, 4)])
        writer.writerow(["P75 Latency (sec)", round(p75, 4)])
        writer.writerow(["P90 Latency (sec)", round(p90, 4)])
        writer.writerow(["P95 Latency (sec)", round(p95, 4)])
        writer.writerow(["P99 Latency (sec)", round(p99, 4)])
        writer.writerow(["Std Deviation", round(std_dev, 4)])
        writer.writerow(["Variance", round(variance, 6)])
        writer.writerow(["First Request Latency (sec)", round(first_request, 4)])

    # Histogram
    plt.figure()
    plt.hist(lat, bins=30)
    plt.title(f"Latency Distribution - {SYSTEM_NAME}")
    plt.xlabel("Latency (sec)")
    plt.ylabel("Frequency")
    plt.savefig(f"{SYSTEM_NAME}_latency_histogram.png")

    # CDF
    sorted_lat = np.sort(lat)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)

    plt.figure()
    plt.plot(sorted_lat, cdf)
    plt.title(f"CDF of Latency - {SYSTEM_NAME}")
    plt.xlabel("Latency (sec)")
    plt.ylabel("CDF")
    plt.savefig(f"{SYSTEM_NAME}_latency_cdf.png")

    print("\nFiles saved:")
    print(f"✔ {SYSTEM_NAME}_metrics.csv")
    print(f"✔ {SYSTEM_NAME}_latency_histogram.png")
    print(f"✔ {SYSTEM_NAME}_latency_cdf.png")

else:
    print("\nNo successful requests.")