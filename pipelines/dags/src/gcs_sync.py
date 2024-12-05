from google.cloud import storage
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sync_src_to_gcs(bucket_name: str, src_dir: str) -> None:
    """
    Sync all files from local src directory to GCS src folder.
    
    Args:
        bucket_name: Name of the GCS bucket
        src_dir: Path to source directory
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Clear existing files in src folder
        logger.info(f"Clearing existing files in gs://{bucket_name}/src/")
        blobs = bucket.list_blobs(prefix='src/')
        for blob in blobs:
            if not blob.name.endswith('.placeholder'):
                blob.delete()
                logger.info(f"Deleted: gs://{bucket_name}/{blob.name}")
        
        # Upload all files from src directory
        for root, _, files in os.walk(src_dir):
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, src_dir)
                    gcs_path = f"src/{relative_path}"
                    
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_filename(local_path)
                    logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
                    
        logger.info(f"Successfully synced src directory to gs://{bucket_name}/src/")
        
    except Exception as e:
        logger.error(f"Error syncing src directory: {str(e)}")
        raise

def main():
    """Main entry point for the script."""
    if len(sys.argv) != 3:
        print("Usage: python gcs_sync_src.py <bucket_name> <src_dir>")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    src_dir = sys.argv[2]
    
    try:
        sync_src_to_gcs(bucket_name, src_dir)
    except Exception as e:
        logger.error(f"Failed to sync src directory: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
