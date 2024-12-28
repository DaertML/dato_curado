from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from text_processing_utils import read_file, filter_lines, save_lines, create_dataset

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

    batch_size = 1000  # Define your batch size
    start_time = datetime.now()
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]

        dataset = Dataset.from_dict({'text': batch})
        filtered_lines = filter_lines(dataset['text'])

        print("Filtered " + str(len(batch) - len(filtered_lines)))
        save_lines(filtered_lines)
        print("Finished Batch " + str(i))

    end_time = datetime.now()
    print("Took " + str(end_time - start_time))

process_task = PythonOperator(
    task_id='process_text',
    python_callable=process_text,
    op_kwargs={'input_file': '/tinystories.txt'},
    dag=dag,
)

process_task
