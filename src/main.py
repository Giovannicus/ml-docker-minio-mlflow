import json
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

from data.minio_client import MinioClient
from data.data_processor import DataProcessor
from model.mlflow_utils import MLflowManager
from model.train import train_model
from config import DATA_BUCKET, MODEL_NAME

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

@app.post("/upload-data", response_model=DataUploadResponse)
async def upload_historical_data(
    file: UploadFile = File(...),
    time_filter: Optional[TimeRangeFilter] = None
):
    """
    Upload historical data in JSON format to MinIO
    Optionally filter by time range before saving
    """
    try:
        # Read JSON data from uploaded file
        contents = await file.read()
        json_data = json.loads(contents)
        
        # Convert to DataFrame
        df = pd.DataFrame(json_data)
        
        # Apply time filtering if provided
        if time_filter:
            if time_filter.time_column not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Time column '{time_filter.time_column}' not found in data"
                )
            
            df = data_processor.minio_client.filter_data_by_time_range(
                df, 
                time_filter.time_column,
                time_filter.start_time,
                time_filter.end_time
            )
        
        # Save to MinIO with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = f"historical_data_{timestamp}.json"
        
        success = minio_client.save_dataframe_to_minio(
            df, 
            DATA_BUCKET, 
            object_name, 
            format="json"
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save data to storage"
            )
        
        return DataUploadResponse(
            message="Data uploaded successfully",
            object_name=object_name,
            record_count=len(df)
        )
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using the latest model version
    """
    try:
        # Convert input features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Load the latest model
        model = mlflow_manager.load_model()
        
        if model is None:
            raise HTTPException(
                status_code=404,
                detail="Model not found. Please train a model first."
            )
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get confidence if available (for classifiers)
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(features_df)[0]
                confidence = float(max(probas))
            except:
                pass
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version="latest"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/train")
async def train_new_model(
    data_object: str = Body(..., embed=True),
    target_column: str = Body(..., embed=True),
    time_column: Optional[str] = Body(None, embed=True),
    start_time: Optional[str] = Body(None, embed=True),
    end_time: Optional[str] = Body(None, embed=True)
):
    """
    Train a new model using data from MinIO
    """
    try:
        model, metrics, run_id = train_model(
            data_object,
            target_column,
            time_column=time_column,
            start_time=start_time,
            end_time=end_time
        )
        
        if model is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to train model. Check logs for details."
            )
        
        return {
            "message": "Model trained successfully",
            "run_id": run_id,
            "metrics": metrics
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training error: {str(e)}"
        )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)