# Evaluating Monolithic vs Multi-Container Deployment of Machine Learning Systems

## Overview

This project compares two deployment architectures for a CNN-based image classification system:

- **Monolithic Deployment** — preprocessing and inference run in a single container.
- **Multi-Container Deployment** — preprocessing and inference are separated into independent services, with scalable inference replicas.

The goal is to evaluate how deployment architecture affects:

- Latency
- Throughput
- Availability
- Scalability

---

## Tech Stack

- Python
- FastAPI
- NumPy
- PyTorch
- Docker
- Docker Compose
- AWS EC2

---

## Project Structure

```text
Project/
├── Dockerfile-Monolith
├── app.py
├── load_test.py
├── concurrent_load_test.py
├── requirements.txt
├── results.json
└── multi_container/
    ├── docker-compose.yml
    ├── preprocessing_service/
    └── inference_service/
        ├── load_test.py
        └── concurrent_load_test.py
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Docker
- Docker Compose

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Monolithic Architecture

### Build Docker Image

```bash
docker build -f Dockerfile-Monolith -t cnn-monolith .
```

### Run Container

```bash
docker run -d -p 8010:8000 --name cnn-monolith-run cnn-monolith
```

### Verify API

Open in browser:

```text
http://localhost:8010/docs
```

---

## Test Monolithic Architecture

### Sequential Test

```bash
python load_test.py
```

### Concurrent Test

```bash
python concurrent_load_test.py
```

---

## Stop Monolithic Container

```bash
docker stop cnn-monolith-run
docker rm cnn-monolith-run
```

---

## Run Multi-Container Architecture

### Navigate to Multi-Container Folder

```bash
cd multi_container
```

### Start Services with Scaled Inference Replicas

```bash
docker compose up --build --scale inference=3
```

### Verify API

Open in browser:

```text
http://localhost:8000/docs
```

---

## Test Multi-Container Architecture

Open a new terminal and navigate to the preprocessing service:

```bash
cd multi_container/preprocessing_service
```

### Sequential Test

```bash
python load_test.py
```

### Concurrent Test

```bash
python concurrent_load_test.py
```

---

## Stop Multi-Container Services

```bash
docker compose down
```

---

## Evaluation Metrics

The system is evaluated using:

- Average latency
- Median latency
- P95 latency
- P99 latency
- Throughput measured in requests per second
- Success rate
- Error rate
- Latency distribution using histogram and CDF plots

---

## Key Results

### Sequential Load

- Monolithic architecture performs better under sequential load.
- It achieves lower latency, approximately 20 ms.
- It provides higher throughput because there is no inter-service communication overhead.

### Concurrent Load

- Multi-container architecture performs better under concurrent load.
- It achieves higher throughput when multiple users send requests at the same time.
- It provides lower median latency under load.
- It scales better because inference replicas can process requests in parallel.

---

## Conclusion

Monolithic architecture is efficient for low-load scenarios because it has a simpler design and minimal overhead.

Multi-container architecture introduces some communication overhead between services, but it scales better under concurrent workloads. This makes it more suitable for production environments where availability, scalability, and parallel request handling are important.

---

## Future Improvements

- Add load balancing for inference services.
- Deploy using Kubernetes.
- Add monitoring tools such as Prometheus and Grafana.
- Optimize model inference using batching or GPU acceleration.

---

## Contributors

- Rithvik Kommareddy
- Tejarsha Chappidi
- Nishanth Reddy Adidela
