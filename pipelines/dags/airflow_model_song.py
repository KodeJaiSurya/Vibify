from airflow import DAG
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import CreateCustomTrainingJobOperator
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta
import os
from src.song_model_pipeline import save_final

PROJECT_ID = "vibe-team"
REGION = "us-east1"
CONTAINER_URI = CONTAINER_URI
BUCKET_NAME = BUCKET_NAME
STAGING_BUCKET = f"gs://{BUCKET_NAME}/vertex-staging"

DAG_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_PATH = os.path.join(DAG_DIR, "src", "trainer", "task.py")
CREDENTIALS_PATH = os.path.join(DAG_DIR, "config", "vibe-team-86db5da891f4.json")

conf.set('core', 'enable_xcom_pickling', 'True')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

default_args = {
    'owner': 'Team_Vibe',
    'start_date': datetime(2024, 11, 2),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': True,
}

dag = DAG(
    dag_id='model_pipeline_song_vertex',
    default_args=default_args,
    description='Song Model DAG with Vertex AI Training',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'vertex-ai', 'song-model']
)

vertex_training_task = CreateCustomTrainingJobOperator(
    task_id="vertex_train_model",
    project_id=PROJECT_ID,
    region=REGION,
    display_name="song_mood_clustering_{{ds}}",
    container_uri=CONTAINER_URI,
    python_package_gcs_uri=None,  # Remove this if not using package-based training
    python_module=None,  # Remove this if not using package-based training
    script_path=TRAINER_PATH,
    staging_bucket=STAGING_BUCKET,
    machine_type="n1-standard-4",
    args=[
        f"--input_data=gs://{BUCKET_NAME}/data/preprocessed/spotify/genres_v2.csv",
        f"--output_dir=gs://{BUCKET_NAME}/models/{{ds}}",
    ],
    dag=dag,
)

def _save_final(execution_date, bucket_name, **context):
    model_outputs = context['task_instance'].xcom_pull(task_ids='vertex_train_model')
    return save_final(model_outputs=model_outputs, execution_date=execution_date, bucket_name=bucket_name)

save_final_task = PythonOperator(
    task_id='save_final_task',
    python_callable=_save_final,
    op_kwargs={
        'execution_date': '{{ds}}',
        'bucket_name': BUCKET_NAME,
    },
    provide_context=True,
    retries=3,
    retry_delay=timedelta(minutes=2),
    dag=dag
)

vertex_training_task >> save_final_task

if __name__ == '__main__':
    dag.cli()