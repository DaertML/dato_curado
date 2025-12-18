from langchain_community.document_loaders import PyPDFLoader
from groq import Groq
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
import concurrent.futures
from typing import List, Dict, Any
import asyncio

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

def save_lines(solutions, file):
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

# Batch processing functions
def run_ollama_batch(model: str, prompt: str, questions: List[str], batch_size: int = 10) -> List[str]:
    """Run Ollama in batch mode"""
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
            future_to_question = {
                executor.submit(_run_ollama_single, model, prompt, question): question 
                for question in batch_questions
            }
            
            for future in concurrent.futures.as_completed(future_to_question):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as exc:
                    print(f"Ollama request generated an exception: {exc}")
                    batch_results.append("")  # Append empty string on failure
        
        results.extend(batch_results)
        print(f"Processed batch {i//batch_size + 1}/{(len(questions)-1)//batch_size + 1}")
    
    return results

def _run_ollama_single(model: str, prompt: str, question: str) -> str:
    """Single Ollama request - used for parallel execution"""
    try:
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
    except Exception as e:
        print(f"Error in Ollama request: {e}")
        return ""

def run_groq_batch(model: str, questions: List[str], batch_size: int = 10) -> List[str]:
    """Run Groq in batch mode"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
            future_to_question = {
                executor.submit(_run_groq_single, client, model, question): question 
                for question in batch_questions
            }
            
            for future in concurrent.futures.as_completed(future_to_question):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as exc:
                    print(f"Groq request generated an exception: {exc}")
                    batch_results.append("")  # Append empty string on failure
        
        results.extend(batch_results)
        print(f"Processed batch {i//batch_size + 1}/{(len(questions)-1)//batch_size + 1}")
    
    return results

def _run_groq_single(client: Groq, model: str, question: str) -> str:
    """Single Groq request - used for parallel execution"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": question}],
                model=model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Groq request failed after {max_retries} attempts: {e}")
                return ""
            continue

def csv_to_prompt(filename: str, column_names: List[str]) -> List[str]:
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

def gen_synth_dataset_batch(questions: List[str], batch_size: int = 10) -> List[Dict[str, str]]:
    """Generate synthetic dataset in batch mode"""
    print(f"Generating answers for {len(questions)} questions...")
    gen_answers = run_groq_batch('meta-llama/llama-4-maverick-17b-128e-instruct', questions, batch_size)
    
    dataset = []
    for i, (question, gen_answer) in enumerate(zip(questions, gen_answers)):
        print(f"Q {i+1}: {question[:100]}...")  # Print first 100 chars
        print(f"A {i+1}: {gen_answer[:100]}...")  # Print first 100 chars
        res = {"question": question, "answer": gen_answer}
        dataset.append(res)
    
    return dataset

def add_solution_process_judge_llm_batch(questions: List[str], answers: List[str], batch_size: int = 5) -> List[Dict[str, str]]:
    """Process questions with judge LLM in batch mode"""
    print(f"Generating solutions for {len(questions)} questions...")
    gen_answers = run_ollama_batch('qwen2.5:32b', prompt_math_process, questions, batch_size)
    
    # Prepare judge requests
    judge_requests = []
    for i, (gen_answer, true_answer) in enumerate(zip(gen_answers, answers)):
        print(f"Q {i+1}: {questions[i][:100]}...")
        print(f"A {i+1}: {gen_answer[:100]}...")
        print(f"T {i+1}: {true_answer[:100]}...")
        
        request = f"Answer A: {gen_answer}\nAnswer B: {true_answer}\nResponse: "
        judge_requests.append(request)
    
    print("Running judge LLM...")
    judge_responses = run_ollama_batch('qwen2.5:32b', prompt_judge_llm, judge_requests, batch_size)
    
    dataset = []
    for i, (question, gen_answer, judge_response) in enumerate(zip(questions, gen_answers, judge_responses)):
        print(f"JUDGE {i+1}: {judge_response}")
        if "CORRECT" in judge_response:
            res = {"question": question, "answer": gen_answer}
            dataset.append(res)
        else:
            print(f"WRONG at index {i+1}!")
        print("=" * 50)
    
    return dataset

def create_dataset(lines, batch_number):
    dataset = Dataset.from_dict({'text': lines})
    dataset.save_to_disk(f'./output/dataset_batch_{batch_number}')