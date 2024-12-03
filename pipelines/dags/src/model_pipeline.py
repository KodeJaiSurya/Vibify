from google.cloud import aiplatform
from google.cloud import storage
from typing import List, Dict
import mlflow
import logging

def init_vertex_ai_and_mlflow(
    project_id: str = "vibe-team",
    location: str = "us-east1",
    staging_bucket: str = None,
    experiment_name: str = "emotion-model-training"
) -> None:
    """Initialize both Vertex AI and MLflow with the specified configurations."""
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=staging_bucket
    )
    
    # Configure MLflow
    mlflow.set_tracking_uri(f"gs://{staging_bucket}/mlflow")
    mlflow.set_experiment(experiment_name)
    
    logging.info("Initialized Vertex AI and MLflow")

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
        accelerator_count: int = None,
        hyperparameters: Dict = None
    ) -> aiplatform.CustomJob:
        """Create a custom training job specification with MLflow tracking."""
        
        # Start MLflow run
        mlflow.start_run(run_name=display_name)
        
        # Log hyperparameters
        if hyperparameters:
            mlflow.log_params(hyperparameters)
        
        worker_pool_specs = [{
            "machine_spec": {
                "machine_type": machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": self.training_container_uri,
                "command": [],
                "args": [
                    f"--mlflow_tracking_uri={mlflow.get_tracking_uri()}",
                    f"--mlflow_experiment_id={mlflow.active_run().info.experiment_id}",
                    f"--mlflow_run_id={mlflow.active_run().info.run_id}"
                ],
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
        model_path: str,
        metrics: Dict = None
    ) -> aiplatform.Model:
        """Create a custom model for deployment with MLflow logging."""
        # Log metrics to MLflow
        if metrics:
            mlflow.log_metrics(metrics)
        
        # Log model to MLflow
        mlflow.tensorflow.log_model(
            tf_saved_model_dir=model_path,
            artifact_path="model"
        )
        
        # Create Vertex AI model
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=model_path,
            serving_container_image_uri=self.model_serving_container_uri,
        )
        
        # Log model URI to MLflow
        mlflow.log_param("model_uri", model.uri)
        
        return model
    
    def run_training_pipeline(
        self,
        display_name: str,
        hyperparameters: Dict,
        machine_type: str = "n1-standard-4"
    ) -> aiplatform.Model:
        """Run complete training pipeline with MLflow tracking."""
        try:
            with mlflow.start_run(run_name=display_name) as run:
                # Log training configuration
                mlflow.log_params({
                    "machine_type": machine_type,
                    "training_container": self.training_container_uri,
                    **hyperparameters
                })
                
                # Create and run custom job
                custom_job = self.create_custom_job(
                    display_name=display_name,
                    machine_type=machine_type,
                    hyperparameters=hyperparameters
                )
                
                job_response = custom_job.run()
                
                # Log training metrics
                metrics = job_response.metrics()
                mlflow.log_metrics(metrics)
                
                # Create model
                model = self.create_custom_model(
                    display_name=f"{display_name}_model",
                    model_path=job_response.output_dir,
                    metrics=metrics
                )
                
                # Deploy model
                endpoint = model.deploy(
                    machine_type="n1-standard-4",
                    min_replica_count=1,
                    max_replica_count=3
                )
                
                # Log deployment info
                mlflow.log_param("endpoint_name", endpoint.resource_name)
                
                return model
                
        except Exception as e:
            logging.error(f"Training pipeline failed: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

def main():
    # Configuration
    PROJECT_ID = "vibe-team"
    LOCATION = "us-east1"
    BUCKET_NAME = BUCKET_NAME
    
    # Initialize both services
    init_vertex_ai_and_mlflow(
        project_id=PROJECT_ID,
        location=LOCATION,
        staging_bucket=BUCKET_NAME,
        experiment_name="emotion-model-training"
    )
    
    # Define hyperparameters
    hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam"
    }
    
    # Create trainer instance
    trainer = CustomTrainer(
        project_id=PROJECT_ID,
        location=LOCATION,
        training_container_uri=f"gcr.io/{PROJECT_ID}/emotion-training:latest",
        model_serving_container_uri=f"gcr.io/{PROJECT_ID}/emotion-serving:latest"
    )
    
    # Run complete training pipeline
    model = trainer.run_training_pipeline(
        display_name="emotion_model_training",
        hyperparameters=hyperparameters
    )
    
    logging.info(f"Training completed. Model URI: {model.uri}")

if __name__ == "__main__":
    main()