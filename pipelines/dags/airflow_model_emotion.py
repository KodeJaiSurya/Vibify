from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from datetime import datetime, timedelta


from src.emotion_model_pipeline import load_data ,create_model, compile_model, train_model, save_model


conf.set('core', 'enable_xcom_pickling', 'True')

default_args = {
    'owner': 'Team_Vibe',
    'start_date': datetime(2024, 11, 2),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}


dag = DAG(
    dag_id ='model_pipeline_emotion',
    default_args=default_args,
    description='Data DAG',
    schedule_interval=None,
    catchup=False,
 )


load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
    op_args=['pipelines/data_pipeline/dags/data/preprocessed/X.npy','pipelines/data_pipeline/dags/data/preprocessed/y.npy', 0.2],
)

create_model_task = PythonOperator(
    task_id="create_model_task",
    python_callable=create_model,
    dag=dag,
    op_args=[(48,48)],
)

compile_model_task = PythonOperator(
    task_id = "model_compile_task",
    python_callable= compile_model,
    dag=dag,
    op_args=[create_model_task.output],
) 

train_model_task = PythonOperator(
    task_id = "train_model_task",
    python_callable=train_model,
    dag= dag,
    op_args=[model_compile_task.output, load_data_task.output['X_train'], load_data_task['y_train'],30,64,0.2],
)

save_model_task = PythonOperator(
    task_id="save_model_task",
    python_callable=save_model,
    dag = dag,
    op_args=[train_model_task.output['model']],
)
