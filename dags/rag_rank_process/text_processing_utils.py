import os
import json
from ollama import chat
from dotenv import load_dotenv

load_dotenv()

def run_ollama(model: str, system_prompt: str, user_question: str):
    """Wrapper function to call the Ollama chat API."""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': user_question})
    
    # We use a standard dictionary access for the response for wider compatibility
    # No streaming is used, as per general constraints.
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
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w') as f:
        for record in data:
            # We use separators=(',', ':') to save space and remove unnecessary whitespace
            json.dump(record, f, separators=(',', ':'))
            f.write('\n')

# Note: The specific judge functions from the prior Q/A task are removed 
# for a cleaner, more general-purpose utility file.