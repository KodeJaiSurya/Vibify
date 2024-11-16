from pathlib import Path
from typing import List, Tuple
import logging
from google.cloud import storage
import tempfile
import os

from .emotion_data_processor import DataProcessor
from .emotion_data_aggregator import DataAggregator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)

class GCSDataDownloader:
    """Handles downloading data from Google Cloud Storage"""
    
    def __init__(self, bucket_name: str):
        """
        Initialize GCS downloader
        
        Args:
            bucket_name: Name of the GCS bucket
        """
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def download_from_gcs(self, blob_path: str) -> str:
        """
        Download file from GCS to local temporary storage
        
        Args:
            blob_path: Path to file in GCS bucket
            
        Returns:
            str: Path to downloaded file
        """
        logger.info(f"Downloading {blob_path} from GCS bucket {self.bucket_name}")
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, os.path.basename(blob_path))
        
        # Download blob
        blob = self.bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        
        logger.info(f"Successfully downloaded file to {local_path}")
        return local_path

def download_emotion_data(bucket_name: str, blob_path: str) -> str:
    """
    Task to download emotion data from GCS
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to file within the bucket
        
    Returns:
        str: Path to downloaded file
    """
    logger.info(f"Starting download task for emotion data from GCS bucket {bucket_name}")
    downloader = GCSDataDownloader(bucket_name)
    
    try:
        file_path = downloader.download_from_gcs(blob_path)
        logger.info("Emotion data downloaded successfully from GCS.")
        return file_path
    except Exception as e:
        logger.error(f"Error in download_emotion_data: {e}")
        raise

def process_emotion_data(file_path: str, bucket_name: str, chunk_size: int = 1000) -> List[Tuple[Path, Path]]:
    """Task to process emotion data in chunks"""
    logger.info("Starting processing task for emotion data")
    processor = DataProcessor(bucket_name, chunk_size=chunk_size)
    try:
        chunk_paths = processor.process_all_chunks(file_path)
        logger.info("Emotion data processed into chunks successfully.")
        return chunk_paths
    except Exception as e:
        logger.error(f"Error in process_emotion_data: {e}")
        raise

def aggregate_filtered_data(chunk_paths: List[Tuple[Path, Path]], bucket_name: str) -> bool:
    """Task to combine chunks and save final data"""
    logger.info("Starting aggregation task for emotion data")
    aggregator = DataAggregator(bucket_name)
    try:
        X, y = aggregator.combine_chunks(chunk_paths)
        success = aggregator.save_final(X, y)
        aggregator.cleanup_chunks(chunk_paths)
        if success:
            logger.info("Emotion data aggregation and saving completed successfully.")
        else:
            logger.warning("Emotion data aggregation completed but saving failed.")
        return success
    except Exception as e:
        logger.error(f"Error in aggregate_filtered_data: {e}")
        raise