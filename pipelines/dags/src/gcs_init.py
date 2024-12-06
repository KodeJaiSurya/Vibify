from google.cloud import storage
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_gcs_folder(bucket, folder_path: str) -> None:
    """Create a folder path in GCS using placeholder objects."""
    if not folder_path.endswith('/'):
        folder_path += '/'
    
    placeholder = bucket.blob(folder_path + '.placeholder')
    if not placeholder.exists():
        placeholder.upload_from_string('')
        logger.info(f"Created folder: gs://{bucket.name}/{folder_path}")

def initialize_bucket_structure(bucket_name: str) -> None:
    """Initialize the required folder structure in the GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        required_folders = [
            'models/',
            'data/preprocessed/spotify/',
            'data/preprocessed/facial_expression/',
            'data/raw/spotify/',
            'data/raw/facial_expression/',
            'src/'
        ]
        
        for folder in required_folders:
            ensure_gcs_folder(bucket, folder)
            
        logger.info(f"Successfully initialized folder structure in bucket: {bucket_name}")
        
    except Exception as e:
        logger.error(f"Error initializing bucket structure: {str(e)}")
        raise

def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        print("Usage: python gcs_initialization.py <bucket_name>")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    try:
        initialize_bucket_structure(bucket_name)
    except Exception as e:
        logger.error(f"Failed to initialize GCS structure: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()