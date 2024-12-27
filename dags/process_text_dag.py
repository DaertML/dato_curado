from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from text_processing_utils import read_file, filter_lines, save_lines

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 12, 26),
    'retries': 1,
}

dag = DAG(
    'process_text_dag',
    default_args=default_args,
    description='A DAG to process text files',
    schedule_interval='@daily',
)

def process_text(**kwargs):
    input_file = kwargs['input_file']
    lines = read_file(input_file)
    filtered_lines = filter_lines(lines)
    save_lines(filtered_lines)

process_task = PythonOperator(
    task_id='process_text',
    python_callable=process_text,
    op_kwargs={'input_file': '/tinystories.txt'},
    dag=dag,
)

process_task
