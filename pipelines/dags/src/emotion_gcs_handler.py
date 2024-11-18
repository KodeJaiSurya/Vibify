from google.cloud import storage
from pathlib import Path
import logging
import tempfile

class GCSHandler:
    """Handles all Google Cloud Storage operations"""
    
    def __init__(self, bucket_name: str = bucket_name):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def download_to_temp(self, gcs_path: str) -> Path:
        """Download a file from GCS to a temporary location"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_path = Path(temp_file.name)
            
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(str(temp_path))
            
            self.logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} to temporary file")
            return temp_path
        except Exception as e:
            self.logger.error(f"Error downloading from GCS: {e}")
            raise
    
    def upload_file(self, local_path: Path, gcs_path: str) -> bool:
        """Upload a file to GCS"""
        try:
            blob = self.bucket.blob(gcs_path)
            print(str(local_path))
            blob.upload_from_filename(str(local_path))
            self.logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading to GCS: {e}")
            return False