import json
import os
from triton_validation_bridge import TritonExecutionBridge # Renamed import
from triton_llm_utils import generate_and_check_triton_solution

CORRECT_FILE = "correct_triton_results.jsonl"
FAILED_FILE = "failed_triton_results.jsonl"

# --- Utility Functions (load_jsonl_with_escape_fix, append_to_file) remain unchanged ---
# ... (omitted for brevity, assume content is the same as previous response) ...

def load_jsonl_with_escape_fix(file_path):
    # ... (function body as before) ...
    questions = []
    print(f"Attempting to load questions from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue

                try:
                    data = json.loads(clean_line)
                    questions.append(data)
                
                except json.JSONDecodeError as e:
                    if "Invalid \\escape" in str(e):
                        fixed_line = clean_line.replace('\\', '\\\\')
                        try:
                            data = json.loads(fixed_line)
                            questions.append(data)
                            print(f"✅ Line {i}: Successfully loaded after fixing escape characters.")
                        except json.JSONDecodeError as fixed_e:
                            print(f"❌ Line {i}: Failed to load even after fixing escapes. Error: {fixed_e}")
                    else:
                        print(f"❌ Line {i}: Error decoding JSON: {e}")
                        
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return []

    print(f"\nSuccessfully loaded {len(questions)} questions.")
    return questions

def append_to_file(filepath: str, data: dict):
    """Appends a single JSON dictionary as a JSONL line to the specified file."""
    with open(filepath, 'a') as f:
        f.write(json.dumps(data) + '\n')


def process_triton_problems(input_filepath: str):
    """
    Main pipeline to process questions, generate Triton code, check validity,
    and save results to separate files.
    """
    
    # 1. Load questions from JSONL file
    questions = load_jsonl_with_escape_fix(input_filepath)
    if not questions:
        print("No questions loaded. Exiting.")
        return

    # 2. Initialize the REAL Execution Bridge
    validator = TritonExecutionBridge()

    print(f"\nProcessing {len(questions)} problems...")

    # Clear previous results for a clean run
    if os.path.exists(CORRECT_FILE):
        os.remove(CORRECT_FILE)
    if os.path.exists(FAILED_FILE):
        os.remove(FAILED_FILE)

    # 3. Process each question
    for i, question in enumerate(questions):
        print(f"\n===== Processing Problem {i+1}/{len(questions)} =====")
        
        # This function encapsulates LLM generation and validation
        result = generate_and_check_triton_solution(validator, question)
        
        # 4. Save results to the appropriate file
        if result['valid']:
            append_to_file(CORRECT_FILE, result)
            print(f"\nFINAL VERDICT: **CORRECT** -> Result appended to {CORRECT_FILE}")
        else:
            append_to_file(FAILED_FILE, result)
            print(f"\nFINAL VERDICT: **MISTAKE** -> Result appended to {FAILED_FILE}")
        
        print("-" * 50)

    print("\n\n--- PIPELINE FINISHED ---")
    print(f"Verification results saved to {CORRECT_FILE} and {FAILED_FILE}.")
    
if __name__ == "__main__":
    input_file_path = 'input_questions.jsonl'
    process_triton_problems(input_file_path)