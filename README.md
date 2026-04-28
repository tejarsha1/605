# CNN Deployment Evaluation: Monolithic vs Multi-Container

## Overview

This project compares two deployment architectures for a CNN-based image classification system:

* Monolithic Deployment – All components (preprocessing and inference) run in a single container
* Multi-Container Deployment – Preprocessing and inference are separated into independent services with scalable inference replicas

The goal is to evaluate how deployment architecture affects:

* Latency
* Throughput
* Availability
* Scalability

---

## Tech Stack

* Python (FastAPI, NumPy)
* PyTorch (CNN model)
* Docker and Docker Compose
* AWS EC2

---

## Project Structure

```
Project/
├── Dockerfile-Monolith
├── app.py
├── load_test.py
├── concurrent_load_test.py
├── multi_container/
│   ├── docker-compose.yml
│   ├── preprocessing_service/
│   └── inference_service/
```

---

## Setup Instructions

### Prerequisites

* Python 3.10 or higher
* Docker
* Docker Compose

Install dependencies:

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

### Verify

Open in browser:

```
http://localhost:8010/docs
```

---

## Run Tests

### Sequential Test

```bash
python load_test.py
```

### Concurrent Test

```bash
python concurrent_load_test.py
```

---

## Stop Monolith

```bash
docker stop cnn-monolith-run
docker rm cnn-monolith-run
```

---

## Run Multi-Container Architecture

### Navigate

```bash
cd multi_container
```

### Start Services (with scaling)

```bash
docker compose up --build --scale inference=3
```

### Verify

```
http://localhost:8000/docs
```

---

## Run Tests

Open a new terminal:

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

## Stop Multi-Container

```bash
docker compose down
```

---

## Evaluation Metrics

The system is evaluated using:

* Average latency
* Median latency
* P95 and P99 latency
* Throughput (requests per second)
* Success rate and error rate
* Latency distribution (Histogram and CDF)

---

## Key Results

### Sequential Load

* Monolithic performs better
* Lower latency (~20 ms)
* Higher throughput

### Concurrent Load (50 users)

* Multi-container performs better
* Higher throughput
* Lower median latency
* Better scalability

---

## Conclusion

Monolithic architecture performs well under low-load scenarios due to minimal overhead and simpler design.

Multi-container architecture introduces slight overhead in simple cases but scales better under concurrent workloads, making it more suitable for production environments.

---

## Future Improvements

* Add load balancing for inference services
* Deploy using Kubernetes
* Add monitoring tools such as Prometheus and Grafana
* Optimize model inference performance

---

## Contributors

* Rithvik Kommareddy
* Tejarsha Chappidi
* Nishanth Reddy Adidela
