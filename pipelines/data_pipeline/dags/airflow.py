from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta

from src.song_data_pipeline import load_song_data, data_cleaning, scale_features, save_features
from src.emotion_data_pipeline import download_emotion_data, process_emotion_data, aggregate_filtered_data

conf.set('core', 'enable_xcom_pickling', 'True')

default_args = {
    'owner': 'Team_Vibe',
    'start_date': datetime(2024, 11, 2),
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
    op_args=["1zckGHmd_tJfyMqePfol0L-lIScstOCh9"],
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
    op_args=[scale_song_data_task.output],
)

# Pipeline for emotions with chunking
download_emotion_data_task = PythonOperator(
    task_id='download_emotion_data_task',
    python_callable=download_emotion_data,
    dag=dag,
    op_args=["1uMcSfJBWTgh_gqIxmB8MRYpmMV_O4PPU"],
)

process_emotion_data_task = PythonOperator(
    task_id='process_emotion_data_task',
    python_callable=process_emotion_data,
    dag=dag,
    op_args=[download_emotion_data_task.output, 1000],
)

aggregate_filtered_data_task = PythonOperator(
    task_id="aggregate_filtered_data_task",
    python_callable=aggregate_filtered_data,
    op_args=[process_emotion_data_task.output],
    dag=dag
)

#setting task dependencies
load_song_data_task >> clean_song_data_task >> scale_song_data_task >> save_song_data_task
download_emotion_data_task >> process_emotion_data_task >> aggregate_filtered_data_task

if __name__ =='__main__':
    dag.cli()
