import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.data_processor import DataProcessor
from model.mlflow_utils import MLflowManager

def train_model(data_object_name, target_column, time_column=None, start_time=None, end_time=None):
    """
    Train a Random Forest model using data from MinIO
    
    Args:
        data_object_name (str): Name of the data object in MinIO
        target_column (str): Target column for prediction
        time_column (str, optional): Column with timestamp data
        start_time (str, optional): Start time for filtering data
        end_time (str, optional): End time for filtering data
    
    Returns:
        tuple: (model, metrics, run_id)
    """
    # Load and preprocess data
    data_processor = DataProcessor()
    df = data_processor.load_data_from_minio(
        data_object_name,
        time_column=time_column,
        start_time=start_time,
        end_time=end_time
    )
    
    if df is None or target_column not in df.columns:
        print(f"Missing data or target column '{target_column}' not found")
        return None, None, None
    
    # Preprocess the data
    X, y = data_processor.preprocess_data(df, target_column=target_column)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model hyperparameters
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }
    
    # Train the model
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    
    print(f"Model training complete with metrics: {metrics}")
    
    # Log model to MLflow and MinIO
    mlflow_manager = MLflowManager()
    run_id = mlflow_manager.log_model(model, X_train, hyperparameters, metrics)
    
    print(f"Model logged to MLflow with run_id: {run_id}")
    
    # Save processed features back to MinIO for later use
    processed_data_name = f"processed_{data_object_name}"
    data_processor.save_processed_data(X, processed_data_name)
    
    return model, metrics, run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Random Forest model')
    parser.add_argument('--data', required=True, help='Data object name in MinIO')
    parser.add_argument('--target', required=True, help='Target column for prediction')
    parser.add_argument('--time_col', help='Time column for filtering')
    parser.add_argument('--start_time', help='Start time for filtering')
    parser.add_argument('--end_time', help='End time for filtering')
    
    args = parser.parse_args()
    
    train_model(
        args.data,
        args.target,
        time_column=args.time_col,
        start_time=args.start_time,
        end_time=args.end_time
    )