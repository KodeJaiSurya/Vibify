from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configurations as conf
from datetime import datetime, timedelta

from src.data_preprocessing import load_data

conf.set('core','enable_xcom-pickling','True')

default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

dag = DAG(
    dag_id ='fer2013_preprocessing',
    default_args=default_args,
    schedule_interval=None
 )

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

#setting task dependencies
load_data_task

if __name__ =='__main__':
    dag.cli()
