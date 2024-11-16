from pathlib import Path
import numpy as np
import os
from typing import List, Tuple
import logging
from google.cloud import storage
from google.api_core import exceptions
import tempfile
import io

class DataAggregator:
    """Handles combining and saving final processed data to Google Cloud Storage"""
    
    def __init__(self, bucket_name: str):
        self.temp_dir = Path("temp_chunks")  # Changed to match processor
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _check_blob_exists(self, blob_path: str) -> bool:
        """Check if a blob exists in GCS"""
        blob = self.bucket.blob(blob_path)
        try:
            blob.reload()
            return True
        except exceptions.NotFound:
            return False

    def _list_available_chunks(self) -> List[str]:
        """List all available chunks in the temp directory"""
        blobs = self.bucket.list_blobs(prefix=str(self.temp_dir))
        return [blob.name for blob in blobs]

    def _download_from_gcs(self, blob_path: str) -> np.ndarray:
        """Downloads and loads a numpy array from GCS"""
        try:
            # Convert Path object to string if necessary
            blob_path_str = str(blob_path) if isinstance(blob_path, Path) else blob_path
            
            # Check if blob exists
            if not self._check_blob_exists(blob_path_str):
                available_files = self._list_available_chunks()
                self.logger.error(f"File {blob_path_str} not found in bucket {self.bucket_name}")
                self.logger.info(f"Available files in temp directory: {available_files}")
                raise FileNotFoundError(f"File {blob_path_str} not found in bucket {self.bucket_name}")
            
            blob = self.bucket.blob(blob_path_str)
            with tempfile.NamedTemporaryFile() as temp_file:
                blob.download_to_filename(temp_file.name)
                return np.load(temp_file.name)
        except exceptions.NotFound as e:
            self.logger.error(f"File {blob_path_str} not found: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error downloading {blob_path_str}: {str(e)}")
            raise

    def combine_chunks(self, chunk_paths: List[Tuple[Path, Path]]) -> Tuple[np.ndarray, np.ndarray]:
        """Combines all chunks from GCS into final arrays"""
        self.logger.info("Starting to combine chunks from GCS")
        try:
            # Log the chunks we're trying to process
            self.logger.info(f"Attempting to process chunks: {chunk_paths}")
            
            # Verify all chunks exist before processing
            for x_path, y_path in chunk_paths:
                if not self._check_blob_exists(str(x_path)):
                    raise FileNotFoundError(f"X chunk not found: {x_path}")
                if not self._check_blob_exists(str(y_path)):
                    raise FileNotFoundError(f"Y chunk not found: {y_path}")
            
            X_list = [self._download_from_gcs(str(x_path)) for x_path, _ in chunk_paths]
            y_list = [self._download_from_gcs(str(y_path)) for _, y_path in chunk_paths]
            
            if not X_list or not y_list:
                raise ValueError("No data chunks were successfully loaded")
            
            self.logger.info(f"Successfully loaded {len(X_list)} X chunks and {len(y_list)} y chunks")
            return np.concatenate(X_list), np.concatenate(y_list)
        except Exception as e:
            self.logger.error(f"Error during chunk combination: {str(e)}")
            raise

    def cleanup_chunks(self, chunk_paths: List[Tuple[Path, Path]]):
        """Removes temporary chunk files from GCS"""
        self.logger.info("Starting cleanup of chunk files from GCS")
        for x_path, y_path in chunk_paths:
            try:
                x_path_str = str(x_path)
                y_path_str = str(y_path)
                
                if self._check_blob_exists(x_path_str):
                    self.bucket.blob(x_path_str).delete()
                if self._check_blob_exists(y_path_str):
                    self.bucket.blob(y_path_str).delete()
                    
                self.logger.debug(f"Cleaned previous chunk files: {x_path_str}, {y_path_str}")
            except Exception as e:
                self.logger.error(f"Error cleaning chunk files {x_path} and {y_path}: {str(e)}")

    def save_final(self, X: np.ndarray, y: np.ndarray, prefix: str = "data/preprocessed/facial_expression") -> bool:
        """Saves final processed arrays to GCS"""
        self.logger.info("Saving final data arrays to GCS")
        try:
            # Ensure prefix is string
            prefix_str = str(prefix) if isinstance(prefix, Path) else prefix
            
            # Save X array
            x_blob = self.bucket.blob(f"{prefix_str}/X.npy")
            with tempfile.NamedTemporaryFile() as temp_file:
                np.save(temp_file.name, X)
                x_blob.upload_from_filename(temp_file.name)
                
            # Save y array
            y_blob = self.bucket.blob(f"{prefix_str}/y.npy")
            with tempfile.NamedTemporaryFile() as temp_file:
                np.save(temp_file.name, y)
                y_blob.upload_from_filename(temp_file.name)
                
            self.logger.info(f"X and Y saved successfully to gs://{self.bucket_name}/{prefix_str}/")
            return True
        except Exception as e:
            self.logger.error(f"Error saving final data to GCS: {str(e)}")
            return False