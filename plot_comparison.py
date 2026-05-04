import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open("results.json", "r") as f:
    data = json.load(f)

labels = ["Sequential", "Concurrent"]

# Throughput
mono = [
    data["monolithic_sequential"]["throughput"],
    data["monolithic_concurrent"]["throughput"]
]

multi = [
    data["multi_container_sequential"]["throughput"],
    data["multi_container_concurrent"]["throughput"]
]

x = np.arange(len(labels))
width = 0.35

# ===== Throughput Graph =====
plt.figure()
plt.bar(x - width/2, mono, width, label="Monolithic")
plt.bar(x + width/2, multi, width, label="Multi-container")

plt.xlabel("Workload Type")
plt.ylabel("Throughput (req/sec)")
plt.title("Throughput Comparison")
plt.xticks(x, labels)
plt.legend()

plt.savefig("final_throughput_comparison.png")
plt.show()

# ===== Latency Graph =====
mono_lat = [
    data["monolithic_sequential"]["avg_latency"],
    data["monolithic_concurrent"]["avg_latency"]
]

multi_lat = [
    data["multi_container_sequential"]["avg_latency"],
    data["multi_container_concurrent"]["avg_latency"]
]

plt.figure()
plt.bar(x - width/2, mono_lat, width, label="Monolithic")
plt.bar(x + width/2, multi_lat, width, label="Multi-container")

plt.xlabel("Workload Type")
plt.ylabel("Latency (sec)")
plt.title("Latency Comparison")
plt.xticks(x, labels)
plt.legend()

plt.savefig("final_latency_comparison.png")
plt.show()