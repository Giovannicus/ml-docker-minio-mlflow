import os
import json
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import skl2onnx
import onnx

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODEL_NAME,
    MODEL_VERSION,
    ONNX_MODEL_PATH,
    PARAMS_JSON_PATH,
    METRICS_JSON_PATH
)
from src.data.minio_client import MinioClient

class MLflowManager:
    """Utility class for managing MLflow experiments and models"""
    
    def __init__(self):
        """Initialize MLflow settings"""
        self.minio_client = MinioClient()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.experiment = self._get_or_create_experiment()
        
    def _get_or_create_experiment(self):
        """Get or create the experiment"""
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            experiment = mlflow.get_experiment(experiment_id)
        return experiment
    
    def log_model(self, model, X_train, hyperparameters, metrics, model_name=MODEL_NAME):
        """
        Log the model and its metadata to MLflow
        
        Args:
            model: The trained model
            X_train: Training features (for signature)
            hyperparameters (dict): Model hyperparameters
            metrics (dict): Model evaluation metrics
            model_name (str): Name for the registered model
        
        Returns:
            str: The run ID
        """
        with mlflow.start_run(experiment_id=self.experiment.experiment_id) as run:
            # Log parameters
            mlflow.log_params(hyperparameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Infer model signature
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Log the model
            mlflow.sklearn.log_model(
                model, 
                "model", 
                signature=signature,
                registered_model_name=model_name
            )
            
            # Export to ONNX format
            self._export_to_onnx(model, X_train)
            
            # Save parameters and metrics as JSON files
            self._save_json_artifacts(hyperparameters, metrics)
            
            # Upload ONNX model and JSON files to MinIO
            self._upload_artifacts_to_minio(run.info.run_id)
            
            return run.info.run_id
    
    def _export_to_onnx(self, model, X_train):
        """Export the model to ONNX format"""
        # Define the input type based on the data
        initial_type = [('float_input', skl2onnx.common.data_types.FloatTensorType([None, X_train.shape[1]]))]
        
        # Convert to ONNX format
        onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
        
        # Save the model
        os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)
        with open(ONNX_MODEL_PATH, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Log the ONNX model to MLflow
        mlflow.log_artifact(ONNX_MODEL_PATH, "onnx")
    
    def _save_json_artifacts(self, hyperparameters, metrics):
        """Save hyperparameters and metrics as JSON files"""
        # Save parameters
        os.makedirs(os.path.dirname(PARAMS_JSON_PATH), exist_ok=True)
        with open(PARAMS_JSON_PATH, "w") as f:
            json.dump(hyperparameters, f, indent=2)
        
        # Save metrics
        os.makedirs(os.path.dirname(METRICS_JSON_PATH), exist_ok=True)
        with open(METRICS_JSON_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Log JSON files to MLflow
        mlflow.log_artifact(PARAMS_JSON_PATH)
        mlflow.log_artifact(METRICS_JSON_PATH)
    
    def _upload_artifacts_to_minio(self, run_id):
        """Upload artifacts to MinIO"""
        # Upload ONNX model
        self.minio_client.upload_file(
            ONNX_MODEL_PATH,
            "models",
            f"{MODEL_NAME}/{MODEL_VERSION}/{run_id}/model.onnx"
        )
        
        # Upload parameters JSON
        self.minio_client.upload_file(
            PARAMS_JSON_PATH,
            "models",
            f"{MODEL_NAME}/{MODEL_VERSION}/{run_id}/params.json"
        )
        
        # Upload metrics JSON
        self.minio_client.upload_file(
            METRICS_JSON_PATH,
            "models",
            f"{MODEL_NAME}/{MODEL_VERSION}/{run_id}/metrics.json"
        )
    
    def load_model(self, model_name=MODEL_NAME, version=None):
        """
        Load a model from MLflow
        
        Args:
            model_name (str): The name of the registered model
            version (str or int, optional): The model version to load
            
        Returns:
            The loaded model
        """
        if version is not None:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None