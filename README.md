# ML Model Drift Detection Service
A lightweight, scalable microservice that detects data drift in real-time for machine learning models deployed in production. It compares live prediction data against baseline training data to ensure model relevance over time.

Built with FastAPI, Prometheus, Docker, and Kubernetes.

## Overview
This service monitors incoming feature distributions and alerts if they deviate significantly from your model's original training distribution. It helps maintain model performance in production by:

Detecting feature drift via PSI and Chi-square tests

Logging structured metrics

Exporting Prometheus-compatible monitoring data

Running securely in containerized/Kubernetes environments

## Installation & Setup
Clone the repository:

git clone https://github.com/your-org/drift-detector.git
cd drift-detector

Install dependencies:

pip install -r requirements.txt

Start the service:

python drift_detector.py

Or run with Docker:

docker build -t drift-detector .
docker run -p 8080:8080 drift-detector