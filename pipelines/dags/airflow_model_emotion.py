from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta

from src.emotion_model_pipeline import load_data, create_model, compile_model, train_model, save_model

conf.set('core', 'enable_xcom_pickling', 'True')

default_args = {
    'owner': 'Team_Vibe',
    'start_date': datetime(2024, 11, 2),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='model_pipeline_emotion',
    default_args=default_args,
    description='Emotion Model Pipeline DAG',
    schedule_interval=None,
    catchup=False,
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
    op_args=['pipelines/dags/data/preprocessed/X.npy', 'pipelines/dags/data/preprocessed/y.npy', 0.2],
)

create_model_task = PythonOperator(
    task_id="create_model_task",
    python_callable=create_model,
    dag=dag,
    op_args=[(48, 48)],  # Input shape for your data
)

compile_model_task = PythonOperator(
    task_id="model_compile_task",
    python_callable=compile_model,
    dag=dag,
    op_args=[
        create_model_task.output,  # Use XCom to pull the model created in the previous task
    ],
)

train_model_task = PythonOperator(
    task_id="train_model_task",
    python_callable=train_model,
    dag=dag,
    op_args=[
        compile_model_task.xcom_pull(task_ids='model_compile_task'),
        load_data_task.xcom_pull(task_ids='load_data_task', key='X_train'),
        load_data_task.xcom_pull(task_ids='load_data_task', key='y_train'),
        30,  # Number of epochs
        64,  # Batch size
        0.2,  # Validation split
    ],
)

save_model_task = PythonOperator(
    task_id="save_model_task",
    python_callable=save_model,
    dag=dag,
    op_args=[
        train_model_task.xcom_pull(task_ids='train_model_task', key='model'),
    ],
)

# Set task dependencies
load_data_task >> create_model_task >> compile_model_task >> train_model_task >> save_model_task

if __name__ == '__main__':
    dag.cli()
