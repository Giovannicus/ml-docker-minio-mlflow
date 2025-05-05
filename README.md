# ML Docker Project with MinIO and MLflow

This project provides a Docker-based environment for machine learning workflows, integrating MinIO for data storage and MLflow for model tracking and registry. It includes a FastAPI application for data ingestion and model predictions.

## Features

- **Data Storage**: Use MinIO for storing JSON data, processed dataframes, and model artifacts
- **Time-based Filtering**: Filter data by time intervals during loading
- **Model Training**: Train Random Forest models on the data in MinIO
- **Model Versioning**: Register models in MLflow with proper versioning
- **ONNX Export**: Export models to ONNX format for better interoperability
- **API Interface**: Expose endpoints for data upload and predictions

## Project Structure

```
ml-docker-project/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration settings
│   ├── data/
│   │   ├── __init__.py
│   │   ├── minio_client.py        # MinIO client utilities
│   │   └── data_processor.py      # Data loading and processing functions
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py               # Model training script
│   │   └── mlflow_utils.py        # MLflow utilities
│   └── api/
│       ├── __init__.py
│       └── main.py                # FastAPI application
└── README.md
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ml-docker-project.git
   cd ml-docker-project
   ```

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

This will start:
- The FastAPI application on http://localhost:8000
- MinIO server on http://localhost:9000 (console on http://localhost:9001)
- MLflow server on http://localhost:5000

### Usage

#### 1. Upload Data

Upload JSON data to MinIO using the API:

```bash
curl -X POST "http://localhost:8000/upload-data" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.json" \
  -F 'time_filter={"time_column": "timestamp", "start_time": "2023-01-01", "end_time": "2023-12-31"}'
```

#### 2. Train a Model

Train a model using the uploaded data:

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data_object": "historical_data_20230501_120000.json",
    "target_column": "target",
    "time_column": "timestamp",
    "start_time": "2023-01-01",
    "end_time": "2023-12-31"
  }'
```

#### 3. Make Predictions

Use the trained model to make predictions:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature1": 1.0,
      "feature2": 2.0,
      "feature3": "category_a"
    }
  }'
```

## Accessing Services

- **FastAPI Documentation**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001 (login with `minioadmin:minioadmin`)
- **MLflow UI**: http://localhost:5000

## Customization

You can customize the project by:

1. Modifying the model parameters and algorithms in `src/model/train.py`
2. Adjusting the MinIO and MLflow configuration in `src/config.py`
3. Adding more API endpoints in `src/api/main.py`

## Artifacts and Model Versioning

The project creates the following artifacts:

- **Model File**: Stored as ONNX format in MinIO
- **Parameters**: Saved as `params.json` in MinIO
- **Metrics**: Saved as `metrics.json` in MinIO

These artifacts are versioned using MLflow and can be retrieved using the MLflow API or UI.

## Technical Notes

- The project uses FastAPI for API development
- MinIO is used as an S3-compatible object storage
- MLflow manages model tracking and versioning
- Docker ensures consistent environment for development and deployment

## License

[MIT License](LICENSE)