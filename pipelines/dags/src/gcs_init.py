from google.cloud import storage
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_gcs_folder(bucket, folder_path):
    """
    Create a folder path in GCS using placeholder objects.
    
    Args:
        bucket: GCS bucket object
        folder_path (str): Path to create
    """
    if not folder_path.endswith('/'):
        folder_path += '/'
    
    placeholder = bucket.blob(folder_path + '.placeholder')
    if not placeholder.exists():
        placeholder.upload_from_string('')
        logger.info(f"Created folder: gs://{bucket.name}/{folder_path}")

def upload_src_files(bucket, src_dir):
    """
    Upload all files from local src directory to GCS src folder.
    
    Args:
        bucket: GCS bucket object
        src_dir (str): Local src directory path
    """
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):  # Upload only Python files
                local_path = os.path.join(root, file)
                # Get relative path from src directory
                relative_path = os.path.relpath(local_path, src_dir)
                gcs_path = f"src/{relative_path}"
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                logger.info(f"Uploaded {local_path} to gs://{bucket.name}/{gcs_path}")

def initialize_bucket_structure(bucket_name: str, src_dir: str = 'src'):
    """
    Initialize the required folder structure in the GCS bucket and copy src files.
    
    Args:
        bucket_name (str): Name of the GCS bucket
        src_dir (str): Path to local src directory
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Define all required folder paths
        required_folders = [
            'models/',
            'data/preprocessed/spotify/',
            'data/preprocessed/emotions/',
            'data/raw/spotify/',
            'data/raw/emotions/',
            'src/'
        ]
        
        # Create each folder
        for folder in required_folders:
            ensure_gcs_folder(bucket, folder)
            
        logger.info(f"Successfully initialized folder structure in bucket: {bucket_name}")
        
        # Upload src files
        if os.path.exists(src_dir):
            logger.info(f"Uploading files from {src_dir} to gs://{bucket_name}/src/")
            upload_src_files(bucket, src_dir)
        else:
            logger.warning(f"Source directory {src_dir} not found")
        
    except Exception as e:
        logger.error(f"Error initializing bucket structure: {str(e)}")
        raise

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gcs_initialization.py <bucket_name> [src_dir]")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    src_dir = sys.argv[2] if len(sys.argv) > 2 else 'src'
    initialize_bucket_structure(bucket_name, src_dir)
