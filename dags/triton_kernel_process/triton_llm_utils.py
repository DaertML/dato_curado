import re
import os
from ollama import chat, Client, ResponseError 
from triton_validation_bridge import TritonExecutionBridge # Renamed import

# --- LLM Configuration ---
OLLAMA_MODEL = "qwen3:8b" 
OLLAMA_HOST = "http://localhost:11434" 

# *** CRITICAL: ENHANCED PROMPT FOR FORMAT ADHERENCE ***
prompt_triton_generation = """
You are a high-performance computing expert specializing in **Triton GPU kernel generation**.

Your task is to translate a natural language mathematical or array processing problem into a high-performance Triton kernel AND the complete Python code required to run and validate it.

**STRICT FORMATTING RULES:**
1.  **NEVER** deviate from the exact delimiters provided below.
2.  Your output **MUST** contain all three sections: Reasoning, Triton Kernel, and Python Test Code.
3.  The Triton Kernel **MUST** include `@triton.kernel` and `import triton.language as tl`.
4.  The Python Test Code **MUST** import `torch` and `numpy`, execute the kernel on the GPU (`device='cuda'`), and use `np.testing.assert_allclose` to compare the kernel output against a reliable CPU/PyTorch reference implementation.
5.  The Python Test Code **MUST** include a runnable function, typically `check_correctness()`, and a `if __name__ == '__main__':` block that calls it.
6.  DO NOT USE @triton.kernel as it is deprecated. USE @triton.jit as it is the working decorator.

**Use the following exact delimiters:**
1.  **Reasoning Section (NL Explanation):**
    -   Start: [REASONING_NL]
    -   End: [END_REASONING_NL]
2.  **Triton Kernel Code (The @triton.kernel function):**
    -   Start: [TRITON_KERNEL]
    -   End: [END_TRITON_KERNEL]
3.  **Python Test Code (Setup, Launch, and Verification):**
    -   Start: [PYTHON_TEST_CODE]
    -   End: [END_PYTHON_TEST_CODE]

Example of required imports in the **Python Test Code** section:
```python
import torch
import numpy as np
import triton
"""

def run_ollama(model: str, prompt: str, question: dict) -> str: 
    """Makes a real call to the Ollama server.""" 
    print(f"\n--- OLLAMA CALL ({model}) --- (Real API Call)") 
    client = Client(host=OLLAMA_HOST)
    try:
        response = client.chat(
            model=model, 
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': question['question']}
            ]
        )
        return response['message']['content']
        
    except ResponseError as e:
        print(f"**OLLAMA API ERROR:** {e}. Is the server running and the model '{model}' pulled?")
        return f"ERROR: Ollama call failed. Details: {e}"
    except Exception as e:
        print(f"**CONNECTION ERROR:** {e}. Check host/port: {OLLAMA_HOST}")
        return f"ERROR: Connection failed. Details: {e}"

def _clean_extracted_code(code_str: str) -> str:
    """Aggressively strips markdown backticks, common junk, and excess whitespace."""
    if not code_str:
        return ""
        
    # 1. Strip all leading/trailing whitespace, including newlines
    code_str = code_str.strip()
    
    # 2. Aggressively remove Python markdown fences (```python ... ```)
    if code_str.startswith("```"):
        # Find the end of the language identifier (e.g., '```python\n')
        code_str = code_str.split('\n', 1)[-1]
    if code_str.endswith("```"):
        # Find the start of the closing fence
        code_str = code_str[:-3].strip()
        
    # 3. Handle potential line 1 syntax errors by cleaning the first line
    # If the first line is empty or contains only comments/junk, clean it up.
    lines = code_str.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    
    return '\n'.join(lines)


def extract_triton_code_and_reasoning(response: str) -> tuple[str, str, str, str]:
    """
    Extracts the code using delimiters, made robust against LLM formatting errors.
    """
    
    # CRITICAL FIX 1: Replace non-standard whitespace (e.g., non-breaking space ' ') 
    # with standard space.
    clean_response = re.sub(r'[\xa0\s]+', ' ', response.strip())

    # CRITICAL FIX 2: Use highly robust, case-insensitive regex pattern (re.IGNORECASE)
    # The (?:pattern)? makes the surrounding brackets optional, capturing content 
    # even if the LLM omits the brackets but keeps the tags.
    
    def find_content(tag_name, text):
        # Look for [TAG_NAME] ... [END_TAG_NAME]
        # Allow any characters (including newlines) in between, being non-greedy.
        pattern = rf'(?:\[{tag_name}\])(.*?)(?:\[END_{tag_name}\])'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    reasoning_nl = find_content("REASONING_NL", clean_response)
    triton_kernel_code = find_content("TRITON_KERNEL", clean_response)
    python_test_code = find_content("PYTHON_TEST_CODE", clean_response)

    # CRITICAL FIX 3: Apply the aggressive cleanup to code components
    triton_kernel_code = _clean_extracted_code(triton_kernel_code)
    python_test_code = _clean_extracted_code(python_test_code)

    # Re-check for missing components after cleaning
    if not reasoning_nl and not triton_kernel_code and not python_test_code:
        # Fallback: If delimiters failed entirely, try to guess the content
        # This is a last resort to try and save the run if the LLM just printed code.
        print("!! WARNING: Delimiters completely failed. Attempting to parse code blocks only.")
        code_blocks = re.findall(r"```(?:python)?(.*?)```", response, re.DOTALL)
        if len(code_blocks) >= 2:
            triton_kernel_code = _clean_extracted_code(code_blocks[0])
            python_test_code = _clean_extracted_code(code_blocks[1])
            reasoning_nl = "Parsed from context based on code block fallback."
        
    full_executable_code = ""
    if triton_kernel_code and python_test_code:
        # We need to ensure the kernel code is executed first so the test code can see it.
        full_executable_code = f"{triton_kernel_code}\n\n{python_test_code}"
    
    return reasoning_nl, triton_kernel_code, python_test_code, full_executable_code


def generate_and_check_triton_solution(validator: TritonExecutionBridge, question: dict) -> dict:
    """
    Calls LLM, extracts code, and validates using the execution bridge.
    """
    
    llm_response = run_ollama(OLLAMA_MODEL, prompt_triton_generation, question)
    reasoning_nl, kernel_code, test_code, full_code = extract_triton_code_and_reasoning(llm_response)
    
    print("-" * 50)
    print(f"NL Question: {question['question']}")
    
    # Re-evaluate the parsing success based on the fixed logic
    is_parsing_successful = bool(kernel_code and test_code and reasoning_nl)

    if not is_parsing_successful:
        print("!! ERROR: Failed to extract complete Triton components. LLM response details below:")
        print(f"--- START LLM RESPONSE ---\n{llm_response}\n--- END LLM RESPONSE ---")
        return {
            "question": question,
            "reasoning": reasoning_nl,
            "triton_kernel": kernel_code,
            "python_test_code": test_code,
            "full_code": full_code, 
            "valid": False,
            "error": "Parsing failed or missing component",
            "llm_output": llm_response 
        }
        
    # Validation using real execution
    # The validation logic in TritonExecutionBridge remains the best practice for execution.
    is_valid, error_details = validator.execute_and_validate(kernel_code, test_code)
    
    return {
        "question": question,
        "reasoning": reasoning_nl,
        "triton_kernel": kernel_code,
        "python_test_code": test_code,
        "full_code": full_code, 
        "valid": is_valid,
        "error": error_details if not is_valid else None
    }

### 2. 🚀 `triton_validation_bridge.py` (Real Execution Implementation)

import io
import contextlib
import re
import sys
import random

# NOTE ON REQUIREMENTS: 
# This file requires: 'triton', 'torch', and 'numpy' to be installed 
# and a CUDA-compatible GPU must be available for PyTorch/Triton.

class TritonExecutionBridge:
    """
    Executes the generated Triton kernel and Python test code.
    
    It captures stdout/stderr during execution and uses an inspection check 
    to verify the outcome without writing to disk.
    """

    def __init__(self):
        print("Triton Execution Bridge initialized (REAL EXECUTION MODE).")

    def execute_and_validate(self, triton_kernel_code: str, python_test_code: str) -> tuple[bool, str]:
        """
        Dynamically executes the combined kernel and test code and verifies the result.
        
        Args:
            triton_kernel_code: The generated Triton kernel function string.
            python_test_code: The generated Python test/verification code string.
            
        Returns:
            A tuple: (True if successful, error message otherwise).
        """
        
        # 1. Combine code and check for basic structure
        full_executable_code = f"{triton_kernel_code}\n\n{python_test_code}"
        
        if not ("@triton.kernel" in triton_kernel_code):
             return False, "Structural Error: Triton kernel decorator '@triton.kernel' is missing."
        if not ("torch" in python_test_code and "numpy" in python_test_code):
            return False, "Structural Error: Required imports (torch/numpy) for comparison are missing."

        # 2. Dynamic Execution using `exec`
        
        # Use StringIO to capture output, preventing the executed code from polluting the main console
        captured_output = io.StringIO()
        
        try:
            # Create a dictionary to execute the code within (local scope)
            exec_locals = {}
            
            # Execute the code, capturing stdout/stderr
            with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
                # The LLM-generated code should include the kernel definition and the test execution in __main__
                exec(full_executable_code, exec_locals, exec_locals)
            
            output = captured_output.getvalue()
            
            # 3. Validation Check
            # Look for the success message that the LLM was prompted to generate
            if "assert_allclose" in python_test_code and "Test passed successfully" in output:
                # The unit test successfully ran assert_allclose and printed the success message
                print("\n[TRITON EXECUTION] Output: 'Kernel executed and verified successfully against PyTorch/NumPy.'")
                return True, ""
            
            # If the code ran but the success message wasn't found, it likely failed a check.
            return False, f"Execution Failure: Test executed but did not report 'Test passed successfully'. Output:\n{output}"

        except Exception as e:
            # Capture any runtime exceptions (e.g., Triton compilation error, assertion failure, runtime crash)
            error_output = captured_output.getvalue()
            full_error_details = f"RUNTIME EXCEPTION: {type(e).__name__}: {str(e)}\nCaptured Output:\n{error_output}"
            print(f"\n[TRITON EXECUTION] Output: 'RUNTIME EXCEPTION/FAILURE.'")
            
            # Print a concise version of the error
            print(full_error_details.split('\n')[0]) 
            return False, full_error_details