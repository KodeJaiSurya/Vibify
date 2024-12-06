from airflow import DAG
import os
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta
from airflow import Variable

from src.song_data_pipeline import load_song_data, data_cleaning, scale_features, save_features
from src.emotion_data_pipeline import init_gcs_handler, process_emotion_data, aggregate_emotion_data

conf.set('core', 'enable_xcom_pickling', 'True')
bucket_name = Variable.get("BUCKET_NAME")

default_args = {
    'owner': 'Team_Vibe',
    'start_date': datetime(2024, 11, 15),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

dag = DAG(
    dag_id ='data_pipeline',
    default_args=default_args,
    description='Data DAG',
    schedule_interval=None,
    catchup=False,
 )

# Pipeline for song recommendation
load_song_data_task = PythonOperator(
    task_id='load_song_data_task',
    python_callable=load_song_data,
    dag=dag,
    op_args=[bucket_name, 'data/raw/spotify/genres_v2.csv'],
)

clean_song_data_task = PythonOperator(
    task_id='clean_song_data_task',
    python_callable=data_cleaning,
    dag=dag,
    op_args=[load_song_data_task.output],
)

scale_song_data_task = PythonOperator(
    task_id='scale_song_data_task',
    python_callable=scale_features,
    dag=dag,
    op_args=[clean_song_data_task.output],
)

save_song_data_task = PythonOperator(
    task_id='save_song_data_task',
    python_callable=save_features,
    dag=dag,
    op_args=[scale_song_data_task.output, bucket_name, 'data/preprocessed/spotify/genres_v2.csv'],
)

def get_chunk_paths(**context):
    """Helper function to get chunk paths from XCom"""
    task_instance = context['task_instance']
    chunk_paths = task_instance.xcom_pull(task_ids='process_emotion_data_task')
    return chunk_paths

# Emotion Pipeline tasks
load_emotion_data_task = PythonOperator(
    task_id='load_emotion_data_task',
    python_callable=init_gcs_handler,
    dag=dag,
    provide_context=True,
)

process_emotion_data_task = PythonOperator(
    task_id='process_emotion_data_task',
    python_callable=process_emotion_data,
    dag=dag,
    provide_context=True,
)

aggregate_emotion_data_task = PythonOperator(
    task_id='aggregate_emotion_data_task',
    python_callable=aggregate_emotion_data,
    dag=dag,
    provide_context=True,
    op_args=[get_chunk_paths],  # Pass the function to get chunk paths
)

# Set task dependencies
load_song_data_task >> clean_song_data_task >> scale_song_data_task >> save_song_data_task
load_emotion_data_task >> process_emotion_data_task >> aggregate_emotion_data_task

if __name__ =='__main__':
    dag.cli()
