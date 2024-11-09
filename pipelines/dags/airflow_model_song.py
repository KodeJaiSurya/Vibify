from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta


from src.song_model_pipeline import apply_pca, visualize_clusters,apply_kmeans, getCluster_Mood


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
    description='Data DAG',
    schedule_interval=None,
    catchup=False,
 )

 apply_pca_task = PythonOperator(
    task_id="apply_pca_task",
    python_callable=load_data,
    dag=dag,
    op_args=[],
 )


 

