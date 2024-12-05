from google.cloud import storage
from google.cloud import composer
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dag_folder(project_id, location, environment_id):
    """Get the DAGs folder path for the Composer environment."""
    client = composer.CloudComposerClient()
    environment_name = f"projects/{project_id}/locations/{location}/environments/{environment_id}"
    response = client.get_environment(name=environment_name)
    
    # Extract the GCS bucket from the DAGs folder URI
    config = response.config
    dag_gcs_uri = config.dag_gcs_prefix
    
    return dag_gcs_uri

def deploy_dag():
    """Deploy the DAG file from source bucket to Composer environment."""
    try:
        project_id = os.environ['PROJECT_ID']
        location = os.environ['COMPOSER_LOCATION']
        environment_id = os.environ['COMPOSER_ENVIRONMENT']
        source_bucket_name = os.environ['GCS_BUCKET']

        # Get the Composer DAGs folder
        dag_folder = get_dag_folder(project_id, location, environment_id)
        
        # Initialize Storage client
        storage_client = storage.Client()
        
        # Get source bucket and DAG file
        source_bucket = storage_client.bucket(source_bucket_name)
        source_blob = source_bucket.blob('src/airflow_data.py')
        
        # Get destination bucket from DAG folder URI
        dest_bucket_name = dag_folder.split('gs://')[-1].split('/')[0]
        dest_bucket = storage_client.bucket(dest_bucket_name)
        
        # Copy DAG file
        destination_blob_name = 'dags/airflow_data.py'
        
        # Copy the file
        logger.info(f"Copying DAG file to {dag_folder}")
        blob_copy = source_bucket.copy_blob(
            source_blob, dest_bucket, destination_blob_name
        )
        
        # Wait for the DAG to be detected (usually takes a few minutes)
        logger.info("Waiting for DAG to be detected...")
        time.sleep(120)  # Wait 2 minutes
        
        logger.info("Successfully deployed DAG file")

    except Exception as e:
        logger.error(f"Error deploying DAG: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_dag()
