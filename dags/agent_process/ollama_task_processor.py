import json
import os
from typing import Dict, Any, List
from utils import *

# --- Configuration Constants (Updated for Tool Execution Model) ---
# !!! IMPORTANT !!!
# Ensure you have a tool-calling capable model pulled (e.g., 'ollama pull llama3:8b-instruct-tool-use-latest')
OLLAMA_MODEL = "qwen3:8b" 

# File paths
INPUT_FILE = "synthetic_tool_tasks.jsonl"
OUTPUT_FILE = "ollama_tool_task_results.jsonl"

# Import the new tool execution utility function
try:
    from ollama_utils import run_ollama_tool_execution
except ImportError:
    # This should be handled by the main pipeline script, but kept as a reminder
    pass 

def process_task_with_ollama(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls the Ollama chat API with a single task using the utility function,
    and processes the raw response to extract the answer and tool calls.
    The 'answer' field is now a parsable JSON string if tool calls are detected.
    """
    question = task.get("question", "")
    available_tools = task.get("available_tools", [])
    
    # --- CRITICAL FIX 1: Ensure available_tools is a Python List ---
    if isinstance(available_tools, str):
        try:
            available_tools = json.loads(available_tools)
            print("  > Correction: Successfully parsed 'available_tools' string into a list.")
        except json.JSONDecodeError:
            print(f"ERROR: Could not JSON parse string value for 'available_tools'. Treating as no tools.")
            available_tools = []
    
    if not isinstance(available_tools, list):
        print(f"Warning: Input data for task '{question[:50]}...' contains malformed 'available_tools' value ({type(available_tools)}). Treating as no tools available for this task.")
        available_tools = [] # Final safeguard

    print(f"  > Executing task with Ollama ({OLLAMA_MODEL})...")

    # 1. Call the Ollama utility function 
    full_ollama_response = run_ollama_tool_execution(OLLAMA_MODEL, question, available_tools)

    if full_ollama_response is None:
        return {
            "question": question,
            "available_tools": available_tools,
            "answer": "ERROR: Ollama execution failed or returned no response.",
            "model_response_type": "error"
        }

    # The actual content is nested in the 'message' key of the response object from ollama.chat()
    result_message = full_ollama_response.get('message', {})
    
    # 2. Extract Tool Calls and Answer
    tool_calls = result_message.get('tool_calls') 
    response_text = result_message.get('content', '') # Use empty string if no content
    
    if tool_calls:
        # --- CRITICAL FIX 3: Structure output as parsable JSON for the 'answer' field ---
        structured_tool_calls = []
        for i, call in enumerate(tool_calls):
            function_data = call.get('function')
            
            if not function_data or not hasattr(function_data, 'name') or not hasattr(function_data, 'arguments'):
                print(f"Warning: Found malformed or unexpected tool call object structure in LLM response: {call}. Skipping.")
                continue
                
            # Access name and arguments as attributes of the custom Function object
            func_name = function_data.name
            args = function_data.arguments
            
            # Ensure arguments are a dictionary (they usually are, but defensive check)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    print(f"Warning: Arguments for {func_name} were a non-parsable string. Using raw string.")
                    args = {"raw_arguments": args}

            structured_tool_calls.append({
                "id": f"call_{i}",  # Add a simple ID for tracking
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": args
                }
            })
        
        # The final answer is the JSON string containing the tool calls and reasoning
        # This makes the answer field fully parsable for downstream systems.
        final_response_object = {
            "model_reasoning": response_text.strip(),
            "tool_calls": structured_tool_calls
        }
        
        final_answer = json.dumps(final_response_object, indent=2)
        model_response_type = "tool_calls_json"
        
    else:
        # Standard text response
        final_answer = response_text.strip()
        model_response_type = "text_generation"


    # 3. Prepare the output record
    return {
        "question": question,
        "available_tools": available_tools,
        "answer": final_answer,
        "model_response_type": model_response_type
    }