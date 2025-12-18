import os
import json
# Import the ollama library for direct Python API calls
# Requires: pip install ollama
from ollama import chat, Client 
from typing import Dict, Any, List, Optional

# LLM model for synthetic question generation (user-provided constant)
PROMPT_GENERATION_MODEL = "qwen3:8b" 

# System prompt for the LLM to generate a question based on available tools (user-provided constant)
PROMPT_GENERATION = """
You are an AI assistant specialized in creating synthetic tasks for an AI agent.
Your task is to generate a realistic, complex "question" that requires the use of the provided "available_tools" to be fully accomplished.
The generated question should be a single, coherent task suitable for an agent.
Only output the raw, synthetically generated question as plain text. Do not include any prefixes, titles, or formatting like quotes or XML tags.

Available Tools:
{}
"""

def read_tools_json(file_path):
    """
    Reads and loads the content of a JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return None

def run_ollama_generation(model, tools_json_content):
    """
    Runs Ollama to generate a synthetic question based on the provided tools.
    """
    # Convert tools content to a formatted JSON string for inclusion in the prompt
    tools_str = json.dumps(tools_json_content, indent=2)
    
    # Construct the full prompt with the tools
    full_prompt = PROMPT_GENERATION.format(tools_str)

    user_message = "Generate the synthetic task now."
    
    try:
        response = chat(model=model, messages=[
            {
                'role': 'system',
                'content': full_prompt,
            },
            {
                'role': 'user',
                'content': user_message
            }
        ])

        return response.message.content.strip()
    except Exception as e:
        print(f"Error during Ollama inference: {e}")
        return None

def save_jsonl_data(data_list, file_path):
    """
    Saves a list of dictionaries to a JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for data_point in data_list:
            # Write each dictionary as a single JSON line
            json.dump(data_point, f, ensure_ascii=False)
            f.write('\n')
    print(f"Successfully saved {len(data_list)} data points to {file_path}")

# --- NEW FUNCTIONS FOR TOOL EXECUTION ---

def format_tools_for_ollama_chat(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts a list of function descriptions into the 'tools' format
    expected by the ollama.chat() payload.
    """
    ollama_tools = []
    
    # Defensive check: if it's not a list, we return an empty list immediately.
    if not isinstance(tools, (list, tuple)):
        return []

    for tool in tools:
        # Check if the item is a dictionary before using .get()
        if not isinstance(tool, dict):
            continue
            
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters")
            }
        })
    return ollama_tools

def run_ollama_tool_execution(model: str, question: str, available_tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Calls the Ollama chat API to execute a question, potentially using the provided tools.
    """
    # 1. Format the tools
    # Since the caller (processor) guarantees 'available_tools' is a list, 
    # we just need to ensure the inner items are structured correctly.
    ollama_tools = format_tools_for_ollama_chat(available_tools)
    
    # 2. Build the messages payload
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant specialized in executing tasks. Use the available tools when necessary to answer the user\'s question.'},
        {'role': 'user', 'content': question}
    ]

    try:
        # Use ollama.chat for tool execution. Pass the formatted tools separately.
        response = chat(
            model=model, 
            messages=messages,
            tools=ollama_tools # Ollama handles tool injection
        )
        
        # Return the complete response object from ollama.chat
        return response 
        
    except Exception as e:
        print(f"Error during Ollama tool execution: {e}")
        return None