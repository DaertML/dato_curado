# data_utils.py
import json
import re
import random
import os
from typing import List, Dict, Any, Optional
import ollama

# --- Configuration ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_GENERATION = "llama3.2"  # Example model for generating tools and queries
OLLAMA_MODEL_TRACING = "llama3.2"     # Example model for trace simulation

# Initialize Ollama Client (assumes service is running)
try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
except Exception as e:
    # Use standard error handling for production code warning
    import sys
    print(f"ERROR: Could not initialize Ollama client. Ensure Ollama service is running at {OLLAMA_HOST}. Error: {e}", file=sys.stderr)
    ollama_client = None

# --- LLM Prompts ---
# System prompt for dynamic tool generation - FIXED by escaping literal curly braces
TOOL_GEN_INSTRUCTIONS = """
You are an expert AI tool designer. Your task is to invent 2 to 3 distinct, highly relevant tools based on the provided user context.

Your output MUST be a single, valid JSON array containing objects for each tool. Each tool object must have the following structure:
{{
  "tool_name": "snake_case_name",
  "description": "A clear, concise description of the tool's function.",
  "args": {{
    "arg1": "description of arg1",
    "arg2": "description of arg2"
  }}
}}

Context: {context}
"""

def run_ollama_chat(
    model: str, 
    messages: List[Dict[str, str]]
) -> str:
    """
    Production implementation using the Ollama chat API.
    """
    if not ollama_client:
        raise ConnectionError("Ollama client is not initialized. Cannot run production code.")

    actual_model = model if model else OLLAMA_MODEL_TRACING
    
    try:
        response = ollama_client.chat(
            model=actual_model, 
            messages=messages, 
            options={'temperature': 1.0} 
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error during Ollama chat call with model {actual_model}: {e}")
        # Structured error message for clarity
        return f"LLM_ERROR: Failed to generate content using {actual_model}. Check connection and model availability."


def generate_tool_metadata_from_ollama(context: str) -> List[str]:
    """
    Uses Ollama to generate tool definitions dynamically based on a given context.
    Returns a list of tool metadata strings (JSON dumps).
    """
    # FIX APPLIED: Tool generation prompt is correctly formatted here.
    system_prompt = TOOL_GEN_INSTRUCTIONS.format(context=context)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate the tools now."}
    ]
    
    raw_response = run_ollama_chat(OLLAMA_MODEL_GENERATION, messages)
    
    # If the LLM returned an error string, propagate it
    if raw_response.startswith("LLM_ERROR"):
        print(f"Skipping tool generation for context: {context[:50]}. Reason: LLM Error.")
        return []

    # Try to extract and parse the JSON array from the response
    try:
        match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', raw_response.strip())
        if match:
            tools_data = json.loads(match.group(0))
            if isinstance(tools_data, list):
                return [json.dumps(tool) for tool in tools_data]
        
        # Fallback: attempt to parse the entire response as JSON array
        tools_data = json.loads(raw_response)
        if isinstance(tools_data, list):
            return [json.dumps(tool) for tool in tools_data]

    except (json.JSONDecodeError, AttributeError):
        print(f"Warning: Failed to parse tool definitions for context: {context[:50]}. Raw response: {raw_response[:100]}")
        return []

    return []

# The rest of the utility functions are unchanged as they were not the source of the error.

def get_tool_name(tool_metadata: str) -> str:
    """Extracts the tool name from its metadata string."""
    try:
        data = json.loads(tool_metadata)
        return data.get("tool_name", "unknown_tool")
    except json.JSONDecodeError:
        return "unknown_tool"

def num_tools_available(input_choice: str) -> int:
    """Determines the number of tools to include in the system prompt."""
    if input_choice == 'none': return 0
    if input_choice == 'single': return 1
    if input_choice == 'few': return random.randint(2, 4)
    if input_choice == 'many': return random.randint(5, 10)
    return 0

def generate_system_prompt(
    tool_name: Optional[str], 
    num_tools: int, 
    filepath: str = 'prompts/system.md',
    tool_metadata_list: Optional[List[str]] = None 
) -> str:
    """
    Generates the system prompt by loading content and inserting generated tools.
    """
    if 'system_cot.md' in filepath:
        template_content = """<instructions>
You are an AI assistant with access to a set of tools. Your goal is to accurately and helpfully respond to user queries.
**CRITICAL: You MUST output the tool call JSON for EVERY query, even if you already explained the intent.**
**Workflow:** 1. Brief Intent Analysis. 2. Tool Selection. 3. EXECUTE TOOL. 
</instructions><tools>{tools}</tools>"""
    else: # Default to system.md
        template_content = """<instructions>
You are an AI assistant with access to a set of tools. Your goal is to accurately and helpfully respond to user queries.
**Workflow:** 1. Understand Query. 2. Decide Tool Use. 3. Select & Argue Tool. 4. Execute Tool. 5. Process Tool Results. 6. Formulate Final Response.
</instructions><tools>{tools}</tools>"""
    
    mock_tools_dict = {}
    
    if num_tools > 0 and tool_metadata_list:
        if num_tools >= len(tool_metadata_list):
            selected_metadata = tool_metadata_list
        else:
            selected_metadata = random.sample(tool_metadata_list, num_tools)
        
        for metadata in selected_metadata:
            try:
                tool_data = json.loads(metadata)
                mock_tools_dict[tool_data['tool_name']] = {
                    'description': tool_data.get('description', 'No description.'),
                    'args': tool_data.get('args', {})
                }
            except json.JSONDecodeError:
                continue

    tools_json = json.dumps(mock_tools_dict, indent=2)
    return template_content.format(tools=tools_json)

def parse_tool_call(response: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parses the tool call JSON from the LLM's response."""
    match = re.search(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>', response)
    if match:
        try:
            tool_call_json = match.group(1).strip()
            tool_call_json = tool_call_json.replace("'", "\"")
            call_data = json.loads(tool_call_json)
            tool_name = call_data.get('tool_name')
            tool_args = call_data.get('args', {})
            return tool_name, tool_args
        except json.JSONDecodeError:
            pass
    return None, None

def call_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Mocks the actual execution of a tool."""
    return f"Result of {tool_name} execution with parameters: {tool_args}"

def format_tool_result(result: str) -> str:
    """Formats the raw tool result into the expected LLM input format."""
    return f"<tool_result>\n{result}\n</tool_result>"

def add_answer_tags(trace_messages: List[Dict[str, str]], index: int) -> List[Dict[str, str]]:
    """Adds <final_answer> tags to the assistant's final response in the trace."""
    if trace_messages and trace_messages[-1]['role'] == 'assistant':
        final_content = trace_messages[-1]['content']
        if not final_content.strip().startswith('<final_answer>'):
            trace_messages[-1]['content'] = f"<final_answer>\n{final_content}\n</final_answer>"
    return trace_messages

def save_traces_to_jsonl(traces: List[Dict[str, Any]], filepath: str):
    """Saves a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for dictionary in traces:
            f.write(json.dumps(dictionary) + '\n')

def read_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {filepath}: {e}")
    return data