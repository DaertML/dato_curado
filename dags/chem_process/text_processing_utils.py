#from langchain_community.document_loaders import PyPDFLoader
from groq import Groq


GROQ_API_KEY = "YOURAPIKEY"

from dotenv import load_dotenv
load_dotenv()
from minio.commonconfig import REPLACE, CopySource
import re
import os
import json
import io
from ollama import chat
from ollama import ChatResponse
import pandas as pd
from transformers import pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

from minio import Minio
from minio.error import S3Error

prompt_judge_llm = """
You are an AI assistant. Your task is to evaluate if two given answers are correct; they are answers to chemistry questions.
Such responses may use different notations to express the response, they may also provide the solutions to a certain problem in different orders.
In such cases and many others that you may find, if the result is equivalent you should consider them as correct.
Also, have into mind that chemistry problems are hard to get right to a high precision, so consider solutions that are too close to be equivalent.
Only output CORRECT or MISTAKE for the given solutions, only focus on the generated solution, do not pay attention to intermediate results.
Here is an example:
Answer A: 107.41
Answer B: 104.25
Response: CORRECT
"""

minio_client = Minio(
        "localhost:9000",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
)

def read_file_from_minio():
    bucket_name = os.getenv("MINIO_BUCKET_NAME")
    file_path = os.getenv("MINIO_FILE_PATH")

    try:
        response = minio_client.get_object(bucket_name, file_path)
        lines = response.read().decode('utf-8').splitlines()
        response.close()
        response.release_conn()
        return lines
    except S3Error as e:
        print(f"Error occurred: {e}")
        return None

def move_files_to_processed():
    source_bucket = "data"
    destination_bucket = "processed"

    try:
        # List all objects in the source bucket
        objects = minio_client.list_objects(source_bucket, recursive=True)
        for obj in objects:
            # Copy each object to the destination bucket
            minio_client.copy_object(
              destination_bucket,
              obj.object_name,
              CopySource(
                source_bucket,
                obj.object_name,
              ),
            )
            print(f"Copied {obj.object_name} to {destination_bucket}")

            # Remove the object from the source bucket
            minio_client.remove_object(source_bucket, obj.object_name)
            print(f"Removed {obj.object_name} from {source_bucket}")

    except S3Error as e:
        print(f"Error occurred: {e}")

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def save_lines(solutions,file):
    with open(file, 'w') as f:
        for dictionary in solutions:
            json.dump(dictionary, f)
            f.write('\n')

def save_lines_to_minio(lines):
    # Run the language filter on all lines at once
    language_results = detect_language(lines)
    
    # Create a dictionary to store lines by language and topic
    language_folders = {}
    
    for line, result in zip(lines, language_results):
        print(result)
        language = result['label']  # Directly access the label
        topic = detect_topic(line)[0]['label']  # Get the label of the most likely prediction
        
        if language not in language_folders:
            language_folders[language] = {}
        if topic not in language_folders[language]:
            language_folders[language][topic] = []
        
        language_folders[language][topic].append(line)
    
    # Save the lines to MinIO based on language and topic
    bucket_name = os.getenv("MINIO_OUT_BUCKET_NAME")
    for language, topics in language_folders.items():
        for topic, lines in topics.items():
            file_path = f"{language}/{topic}/output.txt"
            file_content = "\n".join(lines)
            file_data = io.BytesIO(file_content.encode('utf-8'))
            
            try:
                minio_client.put_object(
                    bucket_name,
                    file_path,
                    file_data,
                    length=len(file_content),
                    content_type="text/plain"
                )
                print(f"Saved {file_path} to MinIO bucket {bucket_name}")
            except S3Error as e:
                print(f"Error occurred: {e}")

def run_ollama_w_sys(model,prompt,question):
    response = chat(model=model, messages=[
        {
            'role': 'system',
            'content': prompt,
        },
        {
            'role': 'user',
            'content': question
        }
    ])

    return response.message.content

def run_ollama(model,question):
    response = chat(model=model, messages=[
        {
            'role': 'user',
            'content': question
        }
    ])

    return response.message.content

def run_groq(model,prompt,question):
    client = Groq(
        api_key=GROQ_API_KEY,
    )
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                model=model,
            )
            return chat_completion.choices[0].message.content
        except:
            continue

def csv_to_prompt(filename,column_names):
    try:
        df = pd.read_csv(filename)
        missing_columns = [col for col in column_names if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns were not found in the CSV: {', '.join(missing_columns)}")
        selected_data = df[column_names].fillna('').astype(str)
        result_list = [' '.join(row) for index, row in selected_data.iterrows()]
        return result_list

    except Exception as e:
        return f"An error occurred: {str(e)}"

def gen_synth_dataset(questions):
    dataset = []
    for i, question in enumerate(questions):
        gen_answer = run_ollama('hf.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:IQ1_S',question)
        print("Q: ",question)
        print("A: ",gen_answer)
        res = {"question": question, "answer": gen_answer}
        dataset.append(res)
    return dataset

def gen_synth_dataset_groq(questions):
    dataset = []
    for i, question in enumerate(questions):
        gen_answer = run_groq('meta-llama/llama-4-maverick-17b-128e-instruct',"",question)
        print("Q: ",question)
        print("A: ",gen_answer)
        res = {"question": question, "answer": gen_answer}
        dataset.append(res)
    return dataset

def add_solution_process_judge_llm(questions,answers):
    dataset = []
    for i, question in enumerate(questions):
        gen_answer = run_ollama('qwen2.5:32b',prompt_math_process,question)
        print("Q: ",question)
        print("A: ",gen_answer)
        print("T: ",answers[i])
        request = "Answer A: "+gen_answer+"\nAnswer B: "+answers[i]+"\nResponse: "

        judge_response = run_ollama('qwen2.5:32b',prompt_judge_llm,request)
        print("JUDGE:",judge_response)
        if "CORRECT" in judge_response:
            res = {"question": question, "answer": gen_answer}
            dataset.append(res)
        else:
            print("WRONGGG!!!")
        print("======================================")

    return dataset

def create_dataset(lines, batch_number):
    dataset = Dataset.from_dict({'text': lines})
    dataset.save_to_disk(f'./output/dataset_batch_{batch_number}')
