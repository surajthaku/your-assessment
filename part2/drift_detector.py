# drift_detector.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import json
import logging
import prometheus_client
from prometheus_client import Counter, Gauge, generate_latest
import uvicorn
import os
from scipy.stats import entropy, chi2_contingency

# Load baseline data
with open("baseline_data.json") as f:
    BASELINE = pd.DataFrame(json.load(f))

# Prometheus metrics
drift_warnings = Counter("drift_warning_count", "Number of drift warnings")
drift_critical = Counter("drift_critical_count", "Number of critical drift alerts")
current_psi = Gauge("current_psi_score", "Latest PSI score")

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("drift-detector")

# FastAPI
app = FastAPI()

# Configurable thresholds
WARNING_THRESHOLD = 0.2
CRITICAL_THRESHOLD = 0.5

class PredictRequest(BaseModel):
    features: Dict[str, Any]
    model_version: str
    timestamp: str

def calculate_psi(expected, actual, buckets=10):
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()

    quantiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    psi_values = [
        (a - e) * np.log((a + 1e-6) / (e + 1e-6))
        for e, a in zip(expected_percents, actual_percents)
    ]
    return sum(psi_values)

def calculate_categorical_drift(expected, actual):
    contingency_table = pd.crosstab(pd.Series(expected, name='baseline'),
                                    pd.Series(actual, name='current'))
    stat, p, _, _ = chi2_contingency(contingency_table, correction=False)
    return p

@app.post("/monitor/predict")
async def monitor_prediction(payload: PredictRequest):
    incoming = pd.DataFrame([payload.features])
    drift_scores = {}
    warning_flag = False
    critical_flag = False

    for col in incoming.columns:
        if col not in BASELINE.columns:
            continue

        if pd.api.types.is_numeric_dtype(BASELINE[col]):
            psi = calculate_psi(BASELINE[col], incoming[col])
            drift_scores[col] = psi
            if psi > CRITICAL_THRESHOLD:
                drift_critical.inc()
                critical_flag = True
            elif psi > WARNING_THRESHOLD:
                drift_warnings.inc()
                warning_flag = True
            current_psi.set(psi)

        else:  # categorical
            p_value = calculate_categorical_drift(BASELINE[col], incoming[col])
            drift_scores[col] = 1 - p_value  # closer to 1 = more drift
            if (1 - p_value) > CRITICAL_THRESHOLD:
                drift_critical.inc()
                critical_flag = True
            elif (1 - p_value) > WARNING_THRESHOLD:
                drift_warnings.inc()
                warning_flag = True

    logger.info(json.dumps({
        "timestamp": payload.timestamp,
        "model_version": payload.model_version,
        "drift_scores": drift_scores
    }))

    return {
        "status": "critical" if critical_flag else "warning" if warning_flag else "normal",
        "drift_scores": drift_scores
    }

@app.get("/monitor/health")
async def health_check():
    return {
        "status": "healthy",
        "drift_detection_status": "running"
    }

@app.get("/monitor/metrics")
async def metrics():
    return generate_latest(prometheus_client.REGISTRY)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
