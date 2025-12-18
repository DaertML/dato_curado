import re
from ollama import chat # Real ollama chat import
from python_executor_bridge import PythonExecutorBridge

# --- LLM Configuration ---
OLLAMA_MODEL = "qwen3:8b" # Keep the same model as in the reference

prompt_code_generation = """
You are a general programming assistant. Your task is to translate a natural language problem into a complete Python solution and an accompanying test case.

Your output MUST contain a reasoning section, the Python solution code (function/class), and the test code, all enclosed in the specified delimiters.

CRITICAL INSTRUCTION: The solution code MUST be complete and standalone (i.e., import necessary libraries). The test code must use standard Python assertion mechanisms (e.g., 'assert') to verify the solution.

Use the following exact delimiters:
1. The **Reasoning Section**: A natural language explanation of the solution approach and test strategy.
    - Start of Reasoning: [REASONING_NL]
    - End of Reasoning: [END_REASONING_NL]
2. The **Solution Code**: The complete Python function(s) or class(es) that solve the problem.
    - Start of Solution: [SOLUTION_CODE]
    - End of Solution: [END_SOLUTION_CODE]
3. The **Test Code**: The Python code that calls and validates the solution using 'assert' statements.
    - Start of Test: [TEST_CODE]
    - End of Test: [END_TEST_CODE]

Example:
Question: Write a function that returns the square of a given number.
Output:
[REASONING_NL]
The function will accept one argument, 'n', and return 'n * n'. The test will verify this with a few simple inputs like 2 and -5, expecting 4 and 25 respectively.
[END_REASONING_NL]

[SOLUTION_CODE]
def square(n):
    return n * n
[END_SOLUTION_CODE]

[TEST_CODE]
assert square(2) == 4, "Test Case 1 Failed: square(2) != 4"
assert square(-5) == 25, "Test Case 2 Failed: square(-5) != 25"
assert square(0) == 0, "Test Case 3 Failed: square(0) != 0"
print("All tests passed for square function.")
[END_TEST_CODE]
"""

# --- Core Utilities ---

def run_ollama(model, prompt, question):
    """Makes a real call to the Ollama server."""
    print(f"\n--- OLLAMA CALL ({model}) ---")
    # The 'question' object is a dict, so we extract the actual question string
    question_str = question.get('question', 'No question found.')
    
    response = chat(model=model, messages=[
        {
            'role': 'system',
            'content': prompt,
        },
        {
            'role': 'user',
            'content': question_str
        }
    ])
    return response['message']['content']

def extract_code_and_reasoning(response: str) -> tuple[str, str, str, str]:
    """
    Extracts the solution code, test code, and reasoning from the LLM response.
    """
    
    reasoning_match = re.search(r"\[REASONING_NL\]\s*(.*?)\s*\[END_REASONING_NL\]", response, re.DOTALL)
    solution_match = re.search(r"\[SOLUTION_CODE\]\s*(.*?)\s*\[END_SOLUTION_CODE\]", response, re.DOTALL)
    test_match = re.search(r"\[TEST_CODE\]\s*(.*?)\s*\[END_TEST_CODE\]", response, re.DOTALL)
    
    # Extract the content, stripping outer whitespace
    reasoning_nl = reasoning_match.group(1).strip() if reasoning_match else ""
    solution_code = solution_match.group(1).strip() if solution_match else ""
    test_code = test_match.group(1).strip() if test_match else ""

    # Keep track of the original question string from the input data
    # (Note: The combined code is created directly in the bridge/executor)
    
    return reasoning_nl, solution_code, test_code

def check_python_solution(executor: PythonExecutorBridge, question_data: dict) -> dict:
    """
    1. Calls the LLM to generate the Python solution, test, and reasoning.
    2. Calls the Python Executor (simulated) to verify the code by running the test.
    3. Returns the comprehensive result.
    """
    
    nl_question = question_data.get('question', 'N/A')
    
    # 1. Generate Python code and reasoning from natural language
    llm_response = run_ollama(OLLAMA_MODEL, prompt_code_generation, nl_question)
    reasoning_nl, solution_code, test_code = extract_code_and_reasoning(llm_response)
    
    print("-" * 50)
    print(f"NL Question: {nl_question[:80]}...")
    
    if not solution_code or not test_code or not reasoning_nl:
        print("!! ERROR: Failed to extract complete code components (Solution, Test, or Reasoning) from LLM.")
        return {
            "question": nl_question,
            "reasoning": reasoning_nl,
            "solution_code": solution_code,
            "test_code": test_code,
            "valid": False,
            "error": "Parsing failed or missing component",
            "llm_output": llm_response
        }
        
    # 2. Check validity using the Executor Bridge (simulated execution)
    is_valid, execution_output = executor.execute_code_and_test(solution_code, test_code)
    
    # 3. Return the comprehensive result
    return {
        "question": nl_question,
        "reasoning": reasoning_nl, 
        "solution_code": solution_code,
        "test_code": test_code,
        "valid": is_valid,
        "execution_output": execution_output # Include output/error for context
    }