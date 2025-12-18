import json
import os
from lean_prover_bridge import LeanProverBridge
from lean_llm_utils import check_lean_solution

CORRECT_FILE = "correct_results.jsonl"
FAILED_FILE = "failed_results.jsonl"

def load_jsonl_with_escape_fix(file_path):
    """
    Loads a JSONL file, attempting to fix common 'Invalid \escape' errors 
    caused by improperly escaped LaTeX or other backslash characters.
    """
    questions = []
    print(f"Attempting to load questions from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                # JSONL lines must be stripped of whitespace, as JSON does not allow 
                # non-control characters outside of strings/values.
                clean_line = line.strip()
                if not clean_line:
                    continue

                try:
                    # Attempt normal load first
                    data = json.loads(clean_line)
                    questions.append(data)
                
                except json.JSONDecodeError as e:
                    # If an error occurs, check if it's the expected Invalid \escape
                    if "Invalid \\escape" in str(e):
                        # The error means Python's json decoder found a backslash
                        # followed by a character that is NOT a valid JSON escape 
                        # (like \n, \t, \", or \\). In your case, it's likely \f, \c, \v, etc.
                        
                        # The fix is to re-escape all single backslashes in the string.
                        # We use .replace('\\', '\\\\') to change \ to \\.
                        # Note: In Python string literals, we must use double backslashes
                        # for a single backslash: '\\' represents the character '\'.
                        # Therefore, we replace '\\' with '\\\\' (which is '\' with ' \\').
                        
                        # Apply the fix (re-escaping single backslashes into double backslashes)
                        fixed_line = clean_line.replace('\\', '\\\\')
                        
                        try:
                            data = json.loads(fixed_line)
                            questions.append(data)
                            print(f"✅ Line {i}: Successfully loaded after fixing escape characters.")
                        except json.JSONDecodeError as fixed_e:
                            print(f"❌ Line {i}: Failed to load even after fixing escapes. Error: {fixed_e}")
                            print(f"   Original line: {clean_line[:100]}...")
                            print(f"   Fixed line: {fixed_line[:100]}...")
                    else:
                        # Handle other JSON errors
                        print(f"❌ Line {i}: Error decoding JSON: {e}")
                        print(f"   Line content: {clean_line[:100]}...")
                        
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return []

    print(f"\nSuccessfully loaded {len(questions)} questions.")
    return questions

def load_questions_from_jsonl(filepath: str) -> list[str]:
    """Reads questions from a JSONL file, expecting a 'question' field in each line."""
    questions = []
    print(f"Attempting to load questions from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "question" in data:
                        questions.append(data["question"])
                    else:
                        print(f"Skipping line: 'question' field missing in {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Input file '{filepath}' not found. Please create it first.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{filepath}': {e}")
    return questions

def append_to_file(filepath: str, data: dict):
    """Appends a single JSON dictionary as a JSONL line to the specified file."""
    # Use 'a' mode for append
    with open(filepath, 'a') as f:
        f.write(json.dumps(data) + '\n')

def process_lean_problems(input_filepath: str):
    """
    Main pipeline to process questions, generate Lean code, check validity,
    and save results to separate files.
    """
    
    # 1. Load questions from JSONL file
    questions = load_jsonl_with_escape_fix(input_filepath)
    if not questions:
        print("No questions loaded. Exiting.")
        return

    # 2. Initialize Prover Bridge
    prover = LeanProverBridge()

    print(f"\nProcessing {len(questions)} problems...")

    # 3. Process each question
    for i, question in enumerate(questions):
        print(f"\n===== Processing Problem {i+1}/{len(questions)} =====")
        result = check_lean_solution(prover, question)
        
        # 4. Save results to the appropriate file
        if result['valid']:
            append_to_file(CORRECT_FILE, result)
            print(f"\nFINAL VERDICT: CORRECT -> Result appended to {CORRECT_FILE}")
        else:
            append_to_file(FAILED_FILE, result)
            print(f"\nFINAL VERDICT: MISTAKE -> Result appended to {FAILED_FILE}")
        
        print("-" * 50)

    print("\n\n--- PIPELINE FINISHED ---")
    print(f"Verification results saved to {CORRECT_FILE} and {FAILED_FILE}.")
    
if __name__ == "__main__":
    input_file_path = 'input_questions.jsonl'
    process_lean_problems(input_file_path)