#from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()
from minio.commonconfig import REPLACE, CopySource
import re
import os
import json
import io
from ollama import chat
from ollama import ChatResponse

from transformers import pipeline
from datasets import Dataset

from minio import Minio
from minio.error import S3Error

#OLLAMA_MODEL = "qwen2.5:32b"
#OLLAMA_MODEL = "wesjos/Qwen3-4B-math"
#OLLAMA_MODEL = "hopephoto/Qwen3-4B-Thinking-2507_q8"
OLLAMA_MODEL = "qwen3:8b"

OLLAMA_JUDGE_MODEL = "qwen2.5:32b"
OLLAMA_JUDGE_MODEL = "qwen3:8b"


prompt_math_process = """
You are an AI assistant. Your task is to answer questions concisely and accurately. Use LaTeX format for any mathematical expressions. Provide the answer in a clear and easily parsable format.
Make sure to follow the same format, and determine the answer inside \boxed{}. If the response is a single numeric value, only give the numeric value inside \boxed{}
Here is an example:
Question: What is the derivative of \( f(x) = x^2 + 3x + 5 \)?
Steps:
To find the derivative of the function \( f(x) = x^2 + 3x + 5 \), we apply the power rule to each term. The power rule states that the derivative of \( x^n \) is \( nx^{n-1} \).

1. For the term \( x^2 \), the derivative is \( 2x \) because \( 2x^{2-1} = 2x \).
2. For the term \( 3x \), the derivative is \( 3 \) because \( 3x^{1-1} = 3 \).
3. The constant term \( 5 \) has a derivative of \( 0 \) because the derivative of a constant is always \( 0 \).

Combining these results, the derivative of the function \( f(x) = x^2 + 3x + 5 \) is \( f'(x) = 2x + 3 \).

Answer: \boxed{f'(x) = 2x + 3}
"""

prompt_judge_llm = """
You are an AI assistant. Your task is to evaluate if two given answers are correct; they are answers to mathematical problems.
Such responses may use different notations like ^ or ** to express the power operation, they may also provide the solutions to 
a certain problem in different orders, or solutions that skip the product symbol between scalars and variables (3a == 3*a).
In such cases and many others that you may find, if the result is equivalent you should
consider them as correct.
Only output CORRECT or MISTAKE for the given solutions, only focus on the generated solution, do not pay attention to intermediate results.
Here is an example:
Answer A: -5v^2(v+61)
Answer B: -5*v**2*(v+61)
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

def run_ollama(model,prompt,question):
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

def extract_answer(response):
    match = re.search(r"\\boxed\{(.+)\}", response)
    if match:
        return match.group(1)
    return None

def extract_answer_regex(gen_answer):
    # This pattern looks for boxed answers or any sequence of numbers separated by commas
    match = re.search(r'\boxed\{(.+?)\}|(\d+(?:,\s*\d+)*)', gen_answer)
    if match:
        return match.group(1) or match.group(2).replace(' ', '')
    else:
        return None

def normalize_expression(expr):
    # Remove spaces, commas and convert to lower case for a consistent comparison
    return re.sub(r'[,\s]', '', expr.lower())

def add_solution_process_eval_expr(questions, answers):
    dataset = []
    for i, question in enumerate(questions):
        gen_answer = run_ollama(OLLAMA_MODEL, prompt_math_process, question)
        print("Q: ", question)
        print("A: ", gen_answer)
        print("T: ", answers[i])
        answer = extract_answer_regex(gen_answer)
        
        # Normalize both generated and target answers for comparison
        norm_gen_answer = normalize_expression(answer) if answer else None
        norm_target_answer = normalize_expression(answers[i])

        print(f"COMPARING {norm_gen_answer} {norm_target_answer}")
        if norm_gen_answer is not None and (norm_gen_answer == norm_target_answer):
            print("CORRECT! Adding it...")
            res = {"question": question, "answer": gen_answer}
            dataset.append(res)
        else:
            print("WRONGGG!!!")
        print("======================================")

    return dataset

def add_solution_process(questions,answers):
    dataset = []
    for i, question in enumerate(questions):
        gen_answer = run_ollama(OLLAMA_MODEL,prompt_math_process,question)
        print("Q: ",question)
        print("A: ",gen_answer)
        print("T: ",answers[i])
        answer = extract_answer(gen_answer)
        print("COMPARING",answer,answers[i])
        if answer != None and (answer == answers[i].strip() or answers[i] in answer):
            print("CORRECT! Adding it...")
            res = {"question": question, "answer": gen_answer}
            dataset.append(res)
        else:
            print("WRONGGG!!!")
        print("======================================")

    return dataset

def add_solution_process_judge_llm(questions,answers):
    dataset = []
    for i, question in enumerate(questions):
        gen_answer = run_ollama(OLLAMA_MODEL,prompt_math_process,question)
        print("Q: ",question)
        print("A: ",gen_answer)
        print("T: ",answers[i])
        request = "Answer A: "+gen_answer+"\nAnswer B: "+answers[i]+"\nResponse: "

        judge_response = run_ollama(OLLAMA_JUDGE_MODEL,prompt_judge_llm,request)
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
