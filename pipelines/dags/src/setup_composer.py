from airflow.providers.google.cloud import composer
from google.cloud.orchestration.airflow.service.v1 import Environments
import logging
import os
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def wait_for_operation(operation, client):
    """Wait for a long-running operation to complete."""
    while not operation.done:
        logger.info("Waiting for operation to complete...")
        time.sleep(60)  # Check every minute
        operation = client.transport.operations_client.get_operation(operation.name)
    return operation

def get_airflow_endpoint(environment_name):
    """Get the Airflow web UI endpoint."""
    client = composer.CloudComposerClient()
    response = client.get_environment(name=environment_name)
    return response.config.airflow_uri

def set_airflow_variables(airflow_endpoint, variables):
    """Set Airflow variables via the REST API."""
    api_url = f"{airflow_endpoint}/api/v1/variables"
    
    # Get the access token for authentication
    import google.auth
    import google.auth.transport.requests
    
    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json"
    }
    
    for key, value in variables.items():
        data = {
            "key": key,
            "value": value
        }
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            logger.info(f"Successfully set variable: {key}")
        else:
            logger.error(f"Failed to set variable {key}: {response.text}")

def setup_composer_environment():
    """Create or update Cloud Composer environment."""
    try:
        project_id = os.environ['PROJECT_ID']
        location = os.environ['COMPOSER_LOCATION']
        environment_id = os.environ['COMPOSER_ENVIRONMENT']
        image_version = os.environ['COMPOSER_IMAGE_VERSION']
        bucket_name = os.environ['GCS_BUCKET']

        # Initialize Composer client
        client = environments_v1.EnvironmentsClient()
        parent = f"projects/{project_id}/locations/{location}"
        
        # Check if environment exists
        environment_path = f"{parent}/environments/{environment_id}"
        try:
            existing_env = client.get_environment(name=environment_path)
            logger.info(f"Environment {environment_id} already exists")
        except Exception:
            logger.info(f"Creating new environment: {environment_id}")
            # Environment configuration
            env_config = {
                "node_count": 3,
                "software_config": {
                    "image_version": image_version,
                    "python_version": "3",
                },
                "web_server_network_access_control": {
                    "allowed_ip_ranges": [{"value": "0.0.0.0/0", "description": "Allow all"}]
                }
            }

            # Create environment
            operation = client.create_environment(
                request={
                    "parent": parent,
                    "environment_id": environment_id,
                    "environment": Environment(
                        name=environment_path,
                        config=env_config
                    )
                }
            )

            # Wait for environment creation to complete
            logger.info("Creating Cloud Composer environment (this may take 45-60 minutes)...")
            operation = wait_for_operation(operation, client)
            
            if operation.error:
                logger.error(f"Failed to create environment: {operation.error}")
                raise Exception(operation.error)

        # Wait for the environment to be fully ready
        time.sleep(60)  # Wait a minute after creation/verification
        
        # Get Airflow web UI endpoint
        airflow_endpoint = get_airflow_endpoint(environment_path)
        
        # Set Airflow variables
        variables = {
            "GCS_BUCKET_NAME": bucket_name
        }
        
        logger.info("Setting Airflow variables...")
        set_airflow_variables(airflow_endpoint, variables)
        logger.info(f"Successfully set up environment {environment_id} with variables")

    except Exception as e:
        logger.error(f"Error setting up Cloud Composer: {str(e)}")
        raise

if __name__ == "__main__":
    setup_composer_environment()