from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta

from src.song_recommendation import load_data, data_cleaning, scale_features, apply_kmeans, getCluster_Mood

conf.set('core','enable_xcom-pickling','True')

default_args = {
    'owner': 'your_name',
    'start_date': datetime(2024, 11, 2),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

dag = DAG(
    dag_id ='songs_preprocessing',
    default_args=default_args,
    description='DAG for Songs Preprocessing',
    schedule_interval=None,
    catchup=False,
 )

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
    op_args=["1zckGHmd_tJfyMqePfol0L-lIScstOCh9"],
)

clean_data_task = PythonOperator(
    task_id='clean_data_task',
    python_callable=data_cleaning,
    dag=dag,
    op_args=[load_data_task.output],
)

scale_data_task = PythonOperator(
    task_id='scale_data_task',
    python_callable=scale_features,
    dag=dag,
    op_args=[clean_data_task.output],
)

cluster_data_task = PythonOperator(
    task_id='cluster_data_task',
    python_callable=apply_kmeans,
    dag=dag,
    op_args=[scale_data_task.output],
)

get_Cluster_task = PythonOperator(
    task_id='get_Cluster_task',
    python_callable=getCluster_Mood,
    dag=dag,
    op_args=[cluster_data_task.output],
)

#setting task dependencies
load_data_task >> clean_data_task >> scale_data_task >> cluster_data_task >> get_Cluster_task

if __name__ =='__main__':
    dag.cli()
