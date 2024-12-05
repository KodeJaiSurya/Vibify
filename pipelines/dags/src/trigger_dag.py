from google.cloud import composer
from google.cloud.composer.airflow import trigger_dag
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def trigger_composer_dag():
    """
    Trigger DAG in Cloud Composer environment
    """
    try:
        project_id = os.environ['PROJECT_ID']
        location = os.environ['COMPOSER_LOCATION']
        composer_env = os.environ['COMPOSER_ENVIRONMENT']
        
        # Initialize Composer client
        client = composer.CloudComposerClient()
        
        # Get environment name
        environment_name = client.environment_path(
            project_id, 
            location, 
            composer_env
        )
        
        # Create a unique run ID based on timestamp
        run_id = f"pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Trigger the DAG
        trigger_dag.trigger_dag(
            environment_name=environment_name,
            dag_id='data_processing_pipeline',
            conf={
                'run_id': run_id,
                'triggered_by': 'github_actions'
            }
        )
        
        logger.info(f"Successfully triggered DAG with run_id: {run_id}")
        
    except Exception as e:
        logger.error(f"Failed to trigger DAG: {str(e)}")
        raise

if __name__ == "__main__":
    trigger_composer_dag()
