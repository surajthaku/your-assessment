# System Architecture – Model Drift Detection Service
This document outlines the architecture and design rationale of the Machine Learning Model Drift Detection Service.

## High-Level Architecture
The system is designed as a stateless, containerized microservice that detects data drift in real-time for machine learning models deployed in production. It exposes RESTful APIs, calculates drift metrics, exports Prometheus-compatible metrics, and scales with Kubernetes.


client ➝ API Gateway ➝ Drift Detection Service ➝ Metrics Exporter
⬑──── Baseline Data (JSON or Redis)

## Component Details
Component	Description
FastAPI Service	Hosts REST API endpoints for drift detection, health check, and metrics
Drift Detection Logic	Calculates PSI for numerical and chi-square test for categorical features
Prometheus Client	Collects and exposes metrics like PSI scores and drift alert counts
Baseline Data Store	JSON file (local) or Redis (future enhancement) storing training distribution
Docker Container	Optimized, secure container with multi-stage build
Kubernetes Deployment	Orchestrates pods, health checks, autoscaling, and service exposure

## Data Flow
Incoming JSON request (via POST /monitor/predict) is validated and parsed.

Features are compared to baseline distribution using:

PSI (Population Stability Index) for numerical columns

Chi-square test for categorical columns

Drift scores are computed and evaluated against configured thresholds.

Metrics are:

Logged (JSON-structured logs)

Exposed to Prometheus via /monitor/metrics

Response is returned with status: normal, warning, or critical.



## Scalability & Performance
Stateless service: can scale horizontally across pods

Health probes: liveness & readiness ensure high availability

HorizontalPodAutoscaler (HPA): optional for dynamic load scaling

JSON input/output is lightweight; suitable for low-latency responses

## Monitoring Strategy
Prometheus metrics exposed via /monitor/metrics

drift_warning_count

drift_critical_count

current_psi_score

JSON structured logs can be aggregated using tools like:

Grafana Loki, ELK Stack, or Fluent Bit

Kubernetes probes:

Liveness: ensures container is responsive

Readiness: ensures container is ready to receive traffic


