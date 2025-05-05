import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

# Bucket names
DATA_BUCKET = "data"
MODEL_BUCKET = "models"

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "random-forest-model"

# Model parameters
MODEL_NAME = "random-forest-classifier"
MODEL_VERSION = "1.0.0"

# File paths
ARTIFACTS_DIR = "/tmp/artifacts"
ONNX_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.onnx")
PARAMS_JSON_PATH = os.path.join(ARTIFACTS_DIR, "params.json")
METRICS_JSON_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

# Create artifacts directory if it doesn't exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)