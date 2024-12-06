from airflow import DAG
import os
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta
from airflow.models import Variable
from google.cloud import storage

# Import modules from GCS
def import_modules_from_gcs():
    """Import required modules from GCS bucket."""
    try:
        bucket_name = Variable.get("GCS_BUCKET_NAME")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        modules = {
            'song_data_pipeline': 'src/song_data_pipeline.py',
            'emotion_data_pipeline': 'src/emotion_data_pipeline.py'
        }
        
        for module_name, path in modules.items():
            blob = bucket.blob(path)
            module_content = blob.download_as_string().decode('utf-8')
            with open(f'/home/airflow/gcs/data/{module_name}.py', 'w') as f:
                f.write(module_content)
        
        # Now import the modules normally
        from song_data_pipeline import load_song_data, data_cleaning, scale_features, save_features
        from emotion_data_pipeline import emotion_gcs_handler, process_emotion_data, aggregate_emotion_data
        
        # Make them global so tasks can access them
        globals().update(locals())
        return True
        
    except Exception as e:
        print(f"Error importing modules: {str(e)}")
        raise

# Enable XCom pickling
conf.set('core', 'enable_xcom_pickling', 'True')

# Get bucket name from Airflow variables
BUCKET_NAME = Variable.get("GCS_BUCKET_NAME")

default_args = {
    'owner': 'Team_Vibe',
    'start_date': datetime(2024, 11, 15),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'provide_context': True,
}

with DAG(
    dag_id='data_pipeline',
    default_args=default_args,
    description='Data Pipeline for Song and Emotion Processing',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
) as dag:
    
    # Initial setup task
    setup_task = PythonOperator(
        task_id='setup_imports',
        python_callable=import_modules_from_gcs,
    )
    
    # Pipeline for song recommendation
    load_song_data_task = PythonOperator(
        task_id='load_song_data_task',
        python_callable=load_song_data,
        op_args=[BUCKET_NAME, 'data/raw/spotify/genres_v2.csv'],
    )
    
    clean_song_data_task = PythonOperator(
        task_id='clean_song_data_task',
        python_callable=data_cleaning,
    )
    
    scale_song_data_task = PythonOperator(
        task_id='scale_song_data_task',
        python_callable=scale_features,
    )
    
    save_song_data_task = PythonOperator(
        task_id='save_song_data_task',
        python_callable=save_features,
        op_args=[None, BUCKET_NAME, 'data/preprocessed/spotify/genres_v2.csv'],
    )
    
    def get_chunk_paths(**context):
        """Helper function to get chunk paths from XCom"""
        try:
            task_instance = context['task_instance']
            chunk_paths = task_instance.xcom_pull(task_ids='process_emotion_data_task')
            if not chunk_paths:
                raise ValueError("No chunk paths found in XCom")
            return chunk_paths
        except Exception as e:
            print(f"Error getting chunk paths: {str(e)}")
            raise
    
    # Emotion Pipeline tasks
    load_emotion_data_task = PythonOperator(
        task_id='load_emotion_data_task',
        python_callable=emotion_gcs_handler,
    )
    
    process_emotion_data_task = PythonOperator(
        task_id='process_emotion_data_task',
        python_callable=process_emotion_data,
    )
    
    aggregate_emotion_data_task = PythonOperator(
        task_id='aggregate_emotion_data_task',
        python_callable=aggregate_emotion_data,
        op_args=[get_chunk_paths],
    )
    
    # Set task dependencies
    setup_task >> [load_song_data_task, load_emotion_data_task]
    load_song_data_task >> clean_song_data_task >> scale_song_data_task >> save_song_data_task
    load_emotion_data_task >> process_emotion_data_task >> aggregate_emotion_data_task