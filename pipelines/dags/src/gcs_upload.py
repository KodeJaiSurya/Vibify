from google.cloud import storage
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_to_gcs(bucket_name: str, source_dir: str, destination_path: str, specific_file: str = ''):
    """
    Upload files to Google Cloud Storage with replacement of existing files.
    
    Args:
        bucket_name (str): Name of the GCS bucket
        source_dir (str): Local directory containing files to upload
        destination_path (str): Destination path in GCS bucket
        specific_file (str, optional): Specific file to upload. If empty, uploads all files.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Delete existing files
        logger.info(f"Deleting existing files in gs://{bucket_name}/{destination_path}")
        blobs = bucket.list_blobs(prefix=destination_path)
        for blob in blobs:
            blob.delete()
            logger.info(f"Deleted: gs://{bucket_name}/{blob.name}")
        
        # Upload new files
        total_files = 0
        total_size = 0
        
        # If specific file is specified, only upload that file
        if specific_file:
            files_to_process = [os.path.join(source_dir, specific_file)]
        else:
            files_to_process = []
            for root, _, files in os.walk(source_dir):
                for file in files:
                    files_to_process.append(os.path.join(root, file))

        for local_path in files_to_process:
            if not os.path.exists(local_path):
                logger.warning(f"File not found: {local_path}")
                continue

            file = os.path.basename(local_path)
            if file.startswith('.'):
                continue
                
            relative_path = os.path.basename(local_path)
            blob_path = f"{destination_path}/{relative_path}"
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            
            size = os.path.getsize(local_path) / (1024 * 1024)  # Size in MB
            total_size += size
            total_files += 1
            
            logger.info(f"Uploaded {local_path} ({size:.2f} MB) to gs://{bucket_name}/{blob_path}")
        
        logger.info(f"Upload complete. Total files: {total_files}, Total size: {total_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}")
        raise

if __name__ == '__main__':
    import sys
    bucket_name = sys.argv[1]
    source_dir = sys.argv[2]
    destination_path = sys.argv[3]
    specific_file = sys.argv[4] if len(sys.argv) > 4 else ''
    upload_to_gcs(bucket_name, source_dir, destination_path, specific_file)
