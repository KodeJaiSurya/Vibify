from google.cloud import storage
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_gcs_folder(bucket, folder_path: str) -> None:
    """
    Create a folder path in GCS using placeholder objects.
    
    Args:
        bucket: GCS bucket object
        folder_path: Path to create in GCS
    """
    if not folder_path.endswith('/'):
        folder_path += '/'
    
    placeholder = bucket.blob(folder_path + '.placeholder')
    if not placeholder.exists():
        placeholder.upload_from_string('')
        logger.info(f"Created folder: gs://{bucket.name}/{folder_path}")

def upload_src_files(bucket, repo_root: str) -> None:
    """
    Upload all Python files from the repository's src directory to GCS.
    
    Args:
        bucket: GCS bucket object
        repo_root: Root path of the repository
    """
    src_dir = os.path.join(repo_root, 'src')
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found at {src_dir}")
        
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                local_path = os.path.join(root, file)
                # Get path relative to src directory
                relative_path = os.path.relpath(local_path, src_dir)
                gcs_path = f"src/{relative_path}"
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                logger.info(f"Uploaded {local_path} to gs://{bucket.name}/{gcs_path}")

def initialize_bucket_structure(bucket_name: str, repo_root: str) -> None:
    """
    Initialize the required folder structure in the GCS bucket and copy src files.
    
    Args:
        bucket_name: Name of the GCS bucket
        repo_root: Root path of the repository
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Define required folder structure
        required_folders = [
            'models/',
            'data/preprocessed/spotify/',
            'data/preprocessed/facial_expression/',
            'data/raw/spotify/',
            'data/raw/facial_expression/',
            'src/'
        ]
        
        # Create each folder
        for folder in required_folders:
            ensure_gcs_folder(bucket, folder)
            
        logger.info(f"Successfully initialized folder structure in bucket: {bucket_name}")
        
        # Upload src files from repository
        logger.info(f"Uploading source files from repository to gs://{bucket_name}/src/")
        upload_src_files(bucket, repo_root)
        
    except Exception as e:
        logger.error(f"Error initializing bucket structure: {str(e)}")
        raise

def get_repository_root() -> str:
    """
    Find the root directory of the repository by looking for .git folder.
    
    Returns:
        str: Path to repository root
    """
    current = Path.cwd()
    while current != current.parent:
        if (current / '.git').exists():
            return str(current)
        current = current.parent
    raise FileNotFoundError("Not inside a git repository")

def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        print("Usage: python gcs_initialization.py <bucket_name>")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    try:
        repo_root = get_repository_root()
        initialize_bucket_structure(bucket_name, repo_root)
    except Exception as e:
        logger.error(f"Failed to initialize GCS structure: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()