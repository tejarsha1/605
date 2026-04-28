# CNN ML Deployment - MLOps Project

## Overview
This project compares two deployment architectures for a CNN-based ML model:
1. Monolithic deployment
2. Multi-container deployment

## Tech Stack
- Python
- FastAPI
- PyTorch
- Docker
- Docker Compose

## Project Structure
- Monolithic: Single container serving model
- Multi-container:
  - Preprocessing service
  - Inference service

## How to Run 
change these in load_test.py
### Monolithic
docker build -t cnn-monolith -f Dockerfile-Monolith .
docker run -p 8010:8000 cnn-monolith

### Multi-container
cd multi_container
docker compose up --build

## Load Testing

Run:
python load_test.py

Metrics measured:
- Latency (avg, median, P95, P99)
- Throughput
- Success rate
- Error rate
- Stability (std deviation)
- Latency distribution (histogram, CDF)

## Goal
Evaluate performance, scalability, and reliability of ML deployments using containerization.