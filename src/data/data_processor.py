import os
import json
import pandas as pd
from datetime import datetime, timedelta

from data.minio_client import MinioClient
from config import DATA_BUCKET

class DataProcessor:
    """Utility class for processing data from MinIO"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.minio_client = MinioClient()
    
    def load_data_from_minio(self, object_name, time_column=None, start_time=None, end_time=None):
        """
        Load data from MinIO and optionally filter by time range
        
        Args:
            object_name (str): The name of the object in MinIO
            time_column (str, optional): The column containing timestamp
            start_time (str or datetime, optional): Start of time range
            end_time (str or datetime, optional): End of time range
            
        Returns:
            pandas.DataFrame: The loaded and filtered DataFrame
        """
        df = self.minio_client.load_json_to_dataframe(DATA_BUCKET, object_name)
        if df is None:
            print(f"Failed to load data from {object_name}")
            return None
            
        print(f"Loaded {len(df)} records from {object_name}")
        
        # Filter by time range if specified
        if all([time_column, start_time, end_time]) and time_column in df.columns:
            df = self.minio_client.filter_data_by_time_range(df, time_column, start_time, end_time)
            print(f"Filtered to {len(df)} records between {start_time} and {end_time}")
            
        return df
    
    def save_processed_data(self, df, object_name, format="parquet"):
        """
        Save processed DataFrame to MinIO
        
        Args:
            df (pandas.DataFrame): The DataFrame to save
            object_name (str): The name to save the object as
            format (str): The format to save as (parquet, csv, json)
            
        Returns:
            bool: True if successful, False otherwise
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add timestamp to filename but keep extension
        base_name, ext = os.path.splitext(object_name)
        if not ext:
            ext = f".{format}"
        versioned_name = f"{base_name}_{timestamp}{ext}"
        
        return self.minio_client.save_dataframe_to_minio(
            df, 
            DATA_BUCKET, 
            versioned_name, 
            format=format
        )
    
    def preprocess_data(self, df, target_column=None):
        """
        Perform basic preprocessing on the DataFrame
        
        Args:
            df (pandas.DataFrame): The DataFrame to preprocess
            target_column (str, optional): The target column for ML
            
        Returns:
            tuple: (X, y) if target_column provided, otherwise just X
        """
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
                
            elif df[col].dtype.kind in 'biufc':  # numeric columns
                df[col] = df[col].fillna(df[col].median())
        
        # Convert categorical features
        cat_columns = df.select_dtypes(include=['object']).columns
        for col in cat_columns:
            if df[col].nunique() < 100:  # Only encode if not too many categories
                df[col] = df[col].astype('category').cat.codes
        
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            return X, y
        else:
            return df