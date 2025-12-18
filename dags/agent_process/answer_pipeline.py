import json
import os
import sys

# Add the directory of the current script to the path to import the processor module
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

# Import necessary components and constants
try:
    from ollama_task_processor import process_task_with_ollama, OLLAMA_MODEL, INPUT_FILE, OUTPUT_FILE
    # Import save_jsonl_data from the new utils module
    from utils import *
except ImportError as e:
    print(f"Error: Could not import necessary modules. Ensure ollama_task_processor.py and ollama_utils.py are in the same directory. Details: {e}")
    sys.exit(1)


def run_ollama_pipeline():
    """
    Main function to run the data processing pipeline.
    Reads tasks, processes them with the Ollama model, and saves results.
    """
    print(f"Starting Ollama Tool Task Pipeline (using ollama.chat)...")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Input File: {INPUT_FILE}, Output File: {OUTPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found. Please ensure it exists.")
        return

    tasks = []
    try:
        # Read the JSONL input file
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        print(f"Successfully loaded {len(tasks)} tasks.")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    results = []
    
    # Process each task
    for i, task in enumerate(tasks):
        print(f"\n--- Processing Task {i+1}/{len(tasks)}: {task['question'][:70]}... ---")
        
        # Call the external function to handle the LLM interaction
        result_record = process_task_with_ollama(task)
        results.append(result_record)
        
    # Write the results to the new JSONL output file using the utility function
    save_jsonl_data(results, OUTPUT_FILE)


if __name__ == "__main__":
    # Ensure 'ollama' is available before starting, as it's a hard requirement now
    try:
        import ollama
    except ImportError:
        print("FATAL ERROR: The 'ollama' library is not installed.")
        print("Please run: pip install ollama")
        sys.exit(1)
        
    run_ollama_pipeline()