import os
import sys
import argparse
import pickle
import logging
import requests
import pandas as pd
import psycopg2
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ----------------------------- Configuration -----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------- Helper Functions -----------------------------

def get_env_variable(key, required=True, default=None):
    value = os.environ.get(key, default)
    if required and value is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_test_data(path='test_data.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test data file '{path}' not found.")
    df = pd.read_csv(path)
    if 'target' not in df.columns:
        raise ValueError("Test data must contain a 'target' column.")
    return df

def evaluate_model(model, test_data):
    X = test_data.drop('target', axis=1)
    y = test_data['target']
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y, predictions),
        'precision': precision_score(y, predictions),
        'recall': recall_score(y, predictions),
        'probabilities': probabilities
    }

    logger.info(f"Model Performance: Accuracy={metrics['accuracy']:.3f}, "
                f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    return metrics

# ----------------------------- Core Logic -----------------------------

def deploy_to_api(env, model_path, metrics):
    api_key = get_env_variable('API_KEY')
    version = os.environ.get('GITHUB_SHA', 'unknown')[:8]
    api_url = f"http://ml-api.company.com/{env}/deploy"

    if metrics['accuracy'] < 0.75:
        logger.warning("Model accuracy too low for deployment.")
        sys.exit(1)

    payload = {
        'model_path': model_path,
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'version': version,
        'environment': env,
        'deployed_at': datetime.utcnow().isoformat()
    }

    try:
        response = requests.post(api_url, json=payload, headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }, timeout=30)

        response.raise_for_status()
        deployment_id = response.json().get('deployment_id')
        if not deployment_id:
            raise ValueError("Missing 'deployment_id' in response.")
        logger.info(f"Model deployed successfully with ID: {deployment_id}")
        return deployment_id

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

def update_deployment_database(deployment_id, env, metrics):
    db_url = get_env_variable('DATABASE_URL')
    version = os.environ.get('GITHUB_SHA', 'unknown')[:8]
    deployed_at = datetime.utcnow()

    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO model_deployments 
            (deployment_id, model_version, environment, accuracy, precision, recall, deployed_at, status) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            deployment_id,
            version,
            env,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            deployed_at,
            'active'
        ))

        cursor.execute("""
            UPDATE model_registry 
            SET current_deployment_id = %s, last_updated = %s 
            WHERE environment = %s
        """, (
            deployment_id,
            deployed_at,
            env
        ))

        conn.commit()
        logger.info("Database updated successfully.")

    except Exception as e:
        logger.error(f"Failed to update database: {e}")
        sys.exit(1)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def send_slack_notification(deployment_id, env, metrics):
    webhook_url = os.environ.get('SLACK_WEBHOOK')
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK not set; skipping Slack notification.")
        return

    message = {
        "text": f"🚀 Model deployed successfully!",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Model Deployment*\n"
                        f"Environment: {env}\n"
                        f"Deployment ID: {deployment_id}\n"
                        f"Accuracy: {metrics['accuracy']:.3f}\n"
                        f"Precision: {metrics['precision']:.3f}\n"
                        f"Recall: {metrics['recall']:.3f}"
                    )
                }
            }
        ]
    }

    try:
        requests.post(webhook_url, json=message)
        logger.info("Slack notification sent.")
    except Exception as e:
        logger.warning(f"Failed to send Slack notification: {e}")

# ----------------------------- Entry Point -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='production', choices=['staging', 'production'])
    args = parser.parse_args()
    env = args.env

    logger.info(f"Starting model deployment for environment: {env}")

    model_path = get_env_variable('MODEL_PATH')
    model = load_model(model_path)
    test_data = load_test_data()
    metrics = evaluate_model(model, test_data)

    deployment_id = deploy_to_api(env, model_path, metrics)
    update_deployment_database(deployment_id, env, metrics)
    send_slack_notification(deployment_id, env, metrics)

    logger.info("Deployment process completed successfully.")

if __name__ == "__main__":
    main()
