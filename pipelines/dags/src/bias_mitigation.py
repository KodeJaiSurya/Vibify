from google.cloud import storage
import os
import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_from_gcs(bucket_name: str, source_blob: str, destination_file: str):
    """
    Download a file from Google Cloud Storage.
    
    Args:
        bucket_name (str): Name of the GCS bucket
        source_blob (str): Path to the source file in GCS
        destination_file (str): Local destination path
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        blob.download_to_filename(destination_file)
        logger.info(f"Downloaded {source_blob}")
    except Exception as e:
        logger.error(f"Error downloading bias script: {str(e)}")
        raise

def process_and_upload_mitigated_data(bucket_name: str, source_dir: str, specific_file: str = ''):
    """
    Process files for bias mitigation and upload results to GCS.
    
    Args:
        bucket_name (str): Name of the GCS bucket
        source_dir (str): Local directory containing files to process
        specific_file (str, optional): Specific file to process. If empty, processes all files.
    """
    try:
        # Download the bias mitigation script
        download_from_gcs(bucket_name, 'src/biasMitigation.py', './biasMitigation.py')
        
        # Import the bias mitigation module
        spec = importlib.util.spec_from_file_location("biasMitigation", "./biasMitigation.py")
        bias_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bias_module)
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Determine which files to process
        if specific_file:
            files_to_process = [os.path.join(source_dir, specific_file)]
        else:
            files_to_process = []
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if not file.startswith('.'):
                        files_to_process.append(os.path.join(root, file))
        
        for file_path in files_to_process:
            try:
                logger.info(f"Running bias mitigation on {file_path}")
                
                # Run bias mitigation
                mitigated_data = bias_module.mitigate_bias(file_path)
                
                # Save mitigated data with same filename
                filename = os.path.basename(file_path)
                mitigated_local_path = os.path.join('./data', filename)
                
                # Save mitigated data based on its type
                if isinstance(mitigated_data, (str, bytes)):
                    with open(mitigated_local_path, 'wb') as f:
                        f.write(mitigated_data if isinstance(mitigated_data, bytes) else mitigated_data.encode())
                else:
                    mitigated_data.to_csv(mitigated_local_path, index=False)
                
                # Upload to GCS with same filename in data/raw directory
                bias_blob_path = f"data/raw/{filename}"
                bias_blob = bucket.blob(bias_blob_path)
                bias_blob.upload_from_filename(mitigated_local_path)
                
                logger.info(f"Uploaded mitigated data to gs://{bucket_name}/{bias_blob_path}")
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
    except Exception as e:
        logger.error(f"Error in bias mitigation process: {str(e)}")
        raise
    finally:
        # Clean up the bias mitigation script
        if os.path.exists('./biasMitigation.py'):
            os.remove('./biasMitigation.py')

if __name__ == '__main__':
    import sys
"""     bucket_name = sys.argv[1]
    source_dir = sys.argv[2]
    specific_file = sys.argv[3] if len(sys.argv) > 3 else ''
    process_and_upload_mitigated_data(bucket_name, source_dir, specific_file) """