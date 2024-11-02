from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configurations as conf

from src.data_preprocessing import preprocess_data



conf.set('core','enable_xcom-pickling','True')

default_args={
    'owner':'airflow',
    retries : 1
}


dag = DAG(
    dag_id ='fer2013_preprocessing',
    default_args=default_args,
    schedule_interval=None
 )
    
preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        dag=dag

    )

#setting task dependencies
preprocess_task

if __name__ =='__main__':
    dag.cli()
