import os
import json
from ollama import chat
from dotenv import load_dotenv

load_dotenv()

# --- Ollama Model Configuration ---
OLLAMA_MODEL = "qwen3:8b" 
OLLAMA_JUDGE_MODEL = "qwen3:8b" 

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

def run_ollama(model: str, system_prompt: str, user_question: str):
    """Wrapper function to call the Ollama chat API."""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': user_question})
    
    # We use a standard dictionary access for the response for wider compatibility
    response = chat(model=model, messages=messages)
    return response['message']['content']

def read_jsonl_file(file_path: str) -> list:
    """Reads a JSONL file and returns a list of dictionaries."""
    lines = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {file_path}. Returning empty list.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")
    return lines

def save_jsonl_lines(data: list, file_path: str):
    """Saves a list of dictionaries to a JSONL file."""
    print(f"Saving {len(data)} records to {file_path}...")
    with open(file_path, 'w') as f:
        for record in data:
            json.dump(record, f)
            f.write('\n')

def add_solution_process_judge_llm(questions: list, answers: list, generated_answers: list):
    """Simulates the LLM Judge process for RAG comparison."""
    dataset = []
    for i, question in enumerate(questions):
        gen_answer = generated_answers[i]
        target_answer = answers[i]
        
        request = "Answer A: " + gen_answer + "\nAnswer B: " + target_answer + "\nResponse: "

        judge_response = run_ollama(OLLAMA_JUDGE_MODEL, prompt_judge_llm, request)
        print("JUDGE:", judge_response)
        
        if "CORRECT" in judge_response:
            res = {"question": question, "answer": gen_answer} 
            dataset.append(res)
        
    return dataset

# All other original functions not directly used by the RAG pipeline are omitted for clarity 
# but can be added back if needed (e.g., read_file, save_lines, minio_functions).