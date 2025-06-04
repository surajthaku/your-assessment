# ML Model Drift Detection Service

A scalable monitoring service to detect data drift in production ML models. This service compares incoming prediction data to baseline training data and alerts when feature distributions deviate beyond configurable thresholds.

Built with FastAPI, Prometheus, and Docker — and ready for deployment in Kubernetes environments.

# Features

 Real-time drift detection via REST API

 PSI (Population Stability Index) for numerical features

 Chi-square test for categorical features

 Configurable warning and critical thresholds

 Metrics exported for Prometheus scraping

 Containerized with Docker

 Deployable via Kubernetes with resource limits and scaling

 Unit-tested core logic