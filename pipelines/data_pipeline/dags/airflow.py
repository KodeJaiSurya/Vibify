from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta

from src.song_recommendation import load_song_data, data_cleaning, scale_features, save_features
from src.data_preprocessing import download_emotion_data, load_emotion_data, filter_emotions, map_emotions, preprocess_pixels, preprocess_labels, aggregate_filtered_data

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

# Pipeline for emotions
download_emotion_data_task = PythonOperator(
    task_id='download_emotion_data_task',
    python_callable=download_emotion_data,
    dag=dag,
    op_args=["1Zkc0a2Ovf2S7vr0akRWKdHPGp9I5QdUT"],
)

load_emotion_data_task = PythonOperator(
    task_id='load_emotion_data_task',
    python_callable=load_emotion_data,
    dag=dag,
    op_args=[download_emotion_data_task.output],
)

filter_data_task = PythonOperator(
    task_id ='filter_data_task',
    python_callable=filter_emotions,
    op_args=[load_emotion_data_task.output],
    dag=dag
)

map_data_task = PythonOperator(
    task_id ='map_data_task',
    python_callable=map_emotions,
    op_args=[filter_data_task.output],
    dag=dag
)

preprocess_pixels_task = PythonOperator(
    task_id ='preprocess_pixels_task',
    python_callable=preprocess_pixels,
    op_args=[map_data_task.output,48,48],
    dag=dag
)

preprocess_labels_task = PythonOperator(
    task_id="preprocess_labels_task",
    python_callable= preprocess_labels,
    op_args=[map_data_task.output],
    dag = dag
)

aggregate_filtered_data_task = PythonOperator(
    task_id= "aggregate_filtered_data_task",
    python_callable=aggregate_filtered_data,
    op_args=[preprocess_pixels_task.output,preprocess_labels_task.output],
    dag=dag
)

#setting task dependencies
load_song_data_task >> clean_song_data_task >> scale_song_data_task >> save_song_data_task
load_emotion_data_task >> filter_data_task >> map_data_task >> [preprocess_pixels_task, preprocess_labels_task] >> aggregate_filtered_data_task


if __name__ =='__main__':
    dag.cli()
