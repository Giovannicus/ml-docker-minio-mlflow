import io
import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from datetime import datetime

from src.config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    DATA_BUCKET,
    MODEL_BUCKET
)

class MinioClient:
    """Utility class for interacting with MinIO storage"""
    
    def __init__(self):
        """Initialize MinIO client with configured credentials"""
        self.client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        self._ensure_buckets_exist()
    
    def _ensure_buckets_exist(self):
        """Ensure that required buckets exist, create them if they don't"""
        for bucket in [DATA_BUCKET, MODEL_BUCKET]:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                print(f"Created bucket: {bucket}")
    
    def list_objects(self, bucket_name, prefix=""):
        """List objects in a bucket with optional prefix"""
        try:
            return list(self.client.list_objects(bucket_name, prefix=prefix, recursive=True))
        except S3Error as e:
            print(f"Error listing objects: {e}")
            return []
    
    def load_json_to_dataframe(self, bucket_name, object_name):
        """Load JSON data from MinIO as pandas DataFrame"""
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = json.load(response)
            response.close()
            response.release_conn()
            return pd.DataFrame(data)
        except S3Error as e:
            print(f"Error loading JSON: {e}")
            return None
    
    def filter_data_by_time_range(self, df, time_column, start_time, end_time):
        """Filter DataFrame by time range"""
        if not all([time_column, start_time, end_time]) or time_column not in df.columns:
            return df
        
        # Convert string dates to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
        # Ensure time column is in datetime format
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Filter by time range
        return df[(df[time_column] >= start_time) & (df[time_column] <= end_time)]
    
    def save_dataframe_to_minio(self, df, bucket_name, object_name, format="parquet"):
        """Save DataFrame to MinIO in specified format"""
        try:
            buffer = io.BytesIO()
            
            if format.lower() == "csv":
                df.to_csv(buffer, index=False)
            elif format.lower() == "parquet":
                df.to_parquet(buffer, index=False)
            elif format.lower() == "json":
                buffer.write(df.to_json(orient="records").encode("utf-8"))
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            buffer.seek(0)
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type=f"application/{format}"
            )
            buffer.close()
            print(f"Successfully saved {object_name} to {bucket_name}")
            return True
        except Exception as e:
            print(f"Error saving DataFrame: {e}")
            return False
    
    def upload_file(self, file_path, bucket_name, object_name):
        """Upload a file to MinIO"""
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return False
    
    def download_file(self, bucket_name, object_name, file_path):
        """Download a file from MinIO"""
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            print(f"Successfully downloaded {bucket_name}/{object_name} to {file_path}")
            return True
        except S3Error as e:
            print(f"Error downloading file: {e}")
            return False