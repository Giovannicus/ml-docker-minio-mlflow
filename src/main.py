import json
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi.responses import StreamingResponse

from data.minio_client import MinioClient
from data.data_processor import DataProcessor
from model.mlflow_utils import MLflowManager
from model.train import train_model
from config import DATA_BUCKET, MODEL_NAME, MODEL_BUCKET

from sklearn.datasets import load_iris
from datetime import datetime, timedelta
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from io import BytesIO


app = FastAPI(
    title="ML Model API",
    description="API for data ingestion and model predictions",
    version="1.0.0"
)

# Initialize clients
minio_client = MinioClient()
data_processor = DataProcessor()
#mlflow_manager = MLflowManager()

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class TimeRangeFilter(BaseModel):
    time_column: str
    start_time: str
    end_time: str

class DataUploadResponse(BaseModel):
    message: str
    object_name: str
    record_count: int

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    model_version: str

@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {"message": "ML Model API is running", "status": "ok"}

@app.get("/models")
async def list_models():
    """
    List available models in MLflow
    """
    # This would be implemented using MLflow's client API
    # For now, just return a placeholder response
    return {
        "models": [
            {
                "name": MODEL_NAME,
                "versions": ["1", "2", "3"],
                "latest_version": "3"
            }
        ]
    }

from pymongo import MongoClient

# Connessione a MongoDB
client_mongo = MongoClient(
        "mongodb://diamond:eXpr1viAKCMct@52.20.211.97:27117/diamond?tls=false&authSource=diamond", 
        serverSelectionTimeoutMS=5000)
db = client_mongo["diamond"]
collection = db["telemetries"]

# Esempio di uso in un endpoint FastAPI
@app.get("/get-telemetries")
async def get_telemetries():
    # Estrai i dati dalla collezione
    data = list(collection.find({}))
    return {"data": str(data)}

@app.get("/view-data")
async def view_data():
    try:
        # Estrai i dati (limita a 100 documenti per evitare sovraccarichi)
        cursor = collection.find({}).limit(100)
        
        # Converti in lista di dizionari (più sicuro)
        data_list = []
        for doc in cursor:
            # Converti ObjectId in stringhe
            doc["_id"] = str(doc["_id"])
            data_list.append(doc)
        
        # Restituisci i dati come risposta
        return {"status": "success", "count": len(data_list), "data": data_list}
    
    except Exception as e:
        # Gestione degli errori
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

from fastapi import FastAPI, HTTPException, Body
from typing import Optional
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime, timedelta
import uuid

app = FastAPI()

# Variabile globale per mantenere il dataset
iris_df = None

@app.post("/upload-iris")
async def upload_iris():
    global iris_df
    iris = load_iris(as_frame=True)
    df = iris.frame

    iris_df = df.copy()

    return {
        "message": "Iris dataset caricato con successo",
        "record_count": len(iris_df),
        "first_rows": iris_df.head().to_dict(orient="records")
    }

@app.post("/train")
async def train_new_model(
    target_column: str = Body(..., embed=True),
):
    """
    Allena un modello sul dataset Iris precedentemente caricato.
    """
    global iris_df

    if iris_df is None:
        raise HTTPException(
            status_code=400,
            detail="Iris dataset non ancora caricato. Esegui prima /upload-iris."
        )

    try:
        df = iris_df.copy()

        # Verifica che la colonna target esista
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found in data"
            )

        # Separazione features e target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Conversione in ONNX
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Salva il modello in memoria
        onnx_bytes = onnx_model.SerializeToString()
        run_id = str(uuid.uuid4())
        model_object_name = f"model_{run_id}.onnx"

        # Salva su MinIO
        # Verifica se il bucket esiste prima di tentare di salvare
        #if not minio_client.bucket_exists(MODEL_BUCKET):
        #    raise HTTPException(status_code=404, detail=f"Bucket '{MODEL_BUCKET}' not found")

        success = save_bytes_to_minio(MODEL_BUCKET, model_object_name, onnx_bytes)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save model to MinIO")

        # Predizione
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "message": "Model trained and saved as ONNX in MinIO",
            "run_id": run_id,
            "model_object": model_object_name,
            "metrics": {
                "accuracy": accuracy
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


from minio import Minio
from minio.error import S3Error

from io import BytesIO
from fastapi import HTTPException

def save_bytes_to_minio(bucket_name, object_name, data_bytes):
    try:
        # Se il bucket non esiste, crealo
        minio_client._ensure_buckets_exist()
        print(f"Bucket '{bucket_name}' created.")
        
        # Salva il modello come oggetto nel bucket MinIO
        minio_client.client.put_object(bucket_name, object_name, BytesIO(data_bytes), len(data_bytes), content_type="application/octet-stream")

        print(f"Model saved successfully to {bucket_name}/{object_name}")
        return True
    except Exception as e:
        print(f"Error saving model to MinIO: {str(e)}")  # Log dell'errore
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")

@app.get("/help")
async def Help():
    # Ottieni la lista di metodi
    minio_methods = dir(MinioClient)
    methods_info = {}

    for method in minio_methods:
        method_obj = getattr(MinioClient, method, None)
        if callable(method_obj):
            methods_info[method] = method_obj.__doc__  # Ottieni la docstring del metodo

    return {"methods_info": methods_info}



@app.get("/download-model")
async def download_model(bucket_name,object_name):
    try:
        # Verifica se il bucket esiste
        #if not minio_client._ensure_buckets_exist():
        #    raise HTTPException(status_code=404, detail="Bucket not found")

        # Recupera il modello da MinIO
        response = minio_client.client.get_object(bucket_name, object_name)
        model_data = response.read()


        # Restituisci il file come risposta HTTP
        return StreamingResponse(BytesIO(model_data), media_type="application/octet-stream")
    
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)