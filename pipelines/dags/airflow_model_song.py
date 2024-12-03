import os
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("REGION", "us-east1")
PROJECT_ID = os.getenv("PROJECT_ID", "vibe-team")
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", "gs://BUCKET_NAME/models")
BUCKET = os.getenv("AIP_MODEL_DIR", "BUCKET_NAME")
DISPLAY_NAME = 'song-mood-classification'
SERVICE_ACCOUNT_EMAIL = os.getenv("SERVICE_ACCOUNT_EMAIL")

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)
    
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=DISPLAY_NAME,
        python_package_gcs_uri=f"gs://{BUCKET}/trainer/task.py",
        python_module_name="task",
        container_uri=container_uri,
        staging_bucket=BUCKET
    )
    
    model = job.run(
        model_display_name=DISPLAY_NAME,
        base_output_dir=BASE_OUTPUT_DIR,
        service_account=SERVICE_ACCOUNT_EMAIL,
        args=[
            "--input_data=gs://BUCKET_NAME/data/preprocessed/spotify/genres_v2.csv",
            f"--output_dir={BASE_OUTPUT_DIR}"
        ]
    )
    
    endpoint = model.deploy(
        deployed_model_display_name=DISPLAY_NAME,
        sync=True,
        service_account=SERVICE_ACCOUNT_EMAIL,
        machine_type="n1-standard-4"
    )
    
    return endpoint

if __name__ == '__main__':
    endpoint = main()