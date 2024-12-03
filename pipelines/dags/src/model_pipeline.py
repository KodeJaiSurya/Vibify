from google.cloud import aiplatform
from google.cloud import storage
from typing import List

def init_vertex_ai(
    project_id: str = "vibe-team",
    location: str = "us-east1",
    staging_bucket: str = None
) -> None:
    """Initialize Vertex AI with the specified project and location."""
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=staging_bucket
    )

def get_data_from_gcs(
    bucket_name: str,
    blob_names: List[str],
    local_paths: List[str]
) -> None:
    """Download data from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for blob_name, local_path in zip(blob_names, local_paths):
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)

class CustomTrainer:
    def __init__(
        self,
        project_id: str,
        location: str,
        training_container_uri: str,
        model_serving_container_uri: str
    ):
        self.project_id = project_id
        self.location = location
        self.training_container_uri = training_container_uri
        self.model_serving_container_uri = model_serving_container_uri
    
    def create_custom_job(
        self,
        display_name: str,
        machine_type: str = "n1-standard-4",
        accelerator_type: str = None,
        accelerator_count: int = None
    ) -> aiplatform.CustomJob:
        """Create a custom training job specification."""
        
        worker_pool_specs = [{
            "machine_spec": {
                "machine_type": machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": self.training_container_uri,
                "command": [],
                "args": [],
            },
        }]
        
        if accelerator_type and accelerator_count:
            worker_pool_specs[0]["machine_spec"]["accelerator_type"] = accelerator_type
            worker_pool_specs[0]["machine_spec"]["accelerator_count"] = accelerator_count
        
        return aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
            base_output_dir=f"gs://{self.project_id}-bucket/model_output"
        )
    
    def create_custom_model(
        self,
        display_name: str,
        model_path: str
    ) -> aiplatform.Model:
        """Create a custom model for deployment."""
        return aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=model_path,
            serving_container_image_uri=self.model_serving_container_uri,
        )

def main():
    # Configuration
    PROJECT_ID = "vibe-team"
    LOCATION = "us-east1"
    BUCKET_NAME = " us-east1-vibeoncloud-89c595e0-bucket "
    
    # Initialize Vertex AI
    init_vertex_ai(PROJECT_ID, LOCATION)
    
    # Create trainer instance
    trainer = CustomTrainer(
        project_id=PROJECT_ID,
        location=LOCATION,
        training_container_uri="gcr.io/{PROJECT_ID}/emotion-training:latest",
        model_serving_container_uri="gcr.io/{PROJECT_ID}/emotion-serving:latest"
    )
    
    # Create and run custom job
    custom_job = trainer.create_custom_job(
        display_name="emotion_model_training",
        machine_type="n1-standard-4"
    )
    
    custom_job.run()
    
    # Create custom model
    model = trainer.create_custom_model(
        display_name="emotion_model_v1",
        model_path=f"gs://{BUCKET_NAME}/model_output/model"
    )
    
    # Deploy model
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )

if __name__ == "__main__":
    main()