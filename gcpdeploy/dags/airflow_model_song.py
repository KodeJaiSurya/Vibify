from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta
from src.song_model_pipeline import apply_pca, apply_kmeans, assign_mood, save_final
from airflow_data import scale_song_data_task


conf.set('core', 'enable_xcom_pickling', 'True')

default_args = {
    'owner': 'Team_Vibe',
    'start_date': datetime(2024, 11, 2),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

dag = DAG(
    dag_id ='model_pipeline_song',
    default_args=default_args,
    description='Song Model DAG',
    schedule_interval=None,
    catchup=False,
 )

apply_pca_task = PythonOperator(
    task_id="apply_pca_task",
    python_callable=apply_pca,
    dag=dag,
    op_args=[scale_song_data_task.output],
 )

apply_kmeans_task = PythonOperator(
    task_id='apply_kmeans_song_data_task',
    python_callable=apply_kmeans,
    dag=dag,
    op_args=[scale_song_data_task.output, apply_pca_task.output],
)

assign_mood_song_task = PythonOperator(
    task_id='assign_mood_song_task',
    python_callable=assign_mood,
    dag=dag,
    op_args=[apply_kmeans_task.output],
)

save_final_task = PythonOperator(
    task_id='save_final_task',
    python_callable=save_final,
    dag=dag,
    op_args=[assign_mood_song_task.output],
)
 

apply_pca_task >> apply_kmeans_task >> assign_mood_song_task >> save_final_task

if __name__ =='__main__':
    dag.cli()