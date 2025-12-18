import ast
import json
import importlib.util
from types import ModuleType
from typing import Dict, Any, List, Type
from pydantic import BaseModel # Must import Pydantic to access .schema() later
import re
import os

# New Constant for the output file
TOOLS_OUTPUT_FILE = 'extracted_tools.json'
AGENT_FILE_PATH = 'agent2.py'

# --- 1. Pydantic Schema Generation (Unchanged) ---

def get_pydantic_schema_from_model(model_class: Type[Any]) -> Dict[str, Any]:
    """
    Generates the JSON schema for a Pydantic model, handling both v1 and v2.
    """
    try:
        import pydantic
    except ImportError:
        raise ImportError("Pydantic is required to extract the tool schemas.")
        
    if pydantic.VERSION.startswith("2."):
        # Pydantic v2
        return model_class.model_json_schema()
    else:
        # Pydantic v1
        return model_class.schema()

# --- 2. Safe Pydantic Model Loading (Unchanged) ---

def load_pydantic_models_safely(file_path: str, model_names: List[str]) -> Dict[str, Type[BaseModel]]:
    """
    Loads only the Pydantic BaseModel classes from the file's code string 
    into a temporary module without executing the network/API setup code.
    """
    model_map = {}
    
    with open(file_path, 'r') as f:
        code_content = f.read()

    tree = ast.parse(code_content)
    safe_code_parts = []
    
    for node in tree.body:
        if (isinstance(node, ast.Import) or 
            isinstance(node, ast.ImportFrom) or
            (isinstance(node, ast.ClassDef) and node.name in model_names)):
            
            try:
                safe_code_parts.append(ast.unparse(node))
            except AttributeError:
                # Fallback placeholder for older Python versions
                print("Warning: ast.unparse not available (Python < 3.9). Safe code execution may be compromised.")
                return {} # Return empty to prevent errors

    safe_code = "\n".join(safe_code_parts)
    temp_module = ModuleType("temp_agent_module")
    
    try:
        exec(safe_code, temp_module.__dict__)
    except Exception as e:
        print(f"⚠️ Warning: Error executing Pydantic model definitions: {e}")
        return {}

    for name in model_names:
        if hasattr(temp_module, name):
            model_map[name] = getattr(temp_module, name)
            
    return model_map

# --- 3. AST Analysis for Tool Extraction (Unchanged logic) ---

def extract_tools_from_decorated_functions(agent_code_file: str) -> List[Dict[str, Any]]:
    """
    Analyzes the Python file using AST to find functions decorated with @agent.tool
    and extracts their information and Pydantic schemas.
    """
    with open(agent_code_file, 'r') as f:
        code_content = f.read()
        tree = ast.parse(code_content)

    tool_function_infos = []
    pydantic_model_names = set()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            is_tool = any(
                isinstance(decorator, ast.Attribute) and 
                decorator.attr == 'tool' and 
                isinstance(decorator.value, (ast.Name, ast.Call))
                for decorator in node.decorator_list
            )

            if is_tool:
                tool_info = {
                    "name": node.name,
                    "description": ast.get_docstring(node) or "No description provided.",
                    "input_model_name": None
                }
                
                args = node.args.args or node.args.posonlyargs
                if args and args[0].annotation:
                    first_arg_annotation = args[0].annotation
                    
                    if isinstance(first_arg_annotation, ast.Subscript):
                        slice_value = first_arg_annotation.slice
                        if isinstance(slice_value, ast.Name):
                            tool_info["input_model_name"] = slice_value.id
                            pydantic_model_names.add(slice_value.id)
                        elif isinstance(slice_value, ast.Attribute) and isinstance(slice_value.value, ast.Name):
                            tool_info["input_model_name"] = slice_value.attr
                            pydantic_model_names.add(slice_value.attr)
                        elif isinstance(slice_value, ast.Tuple) and slice_value.elts and isinstance(slice_value.elts[0], ast.Name):
                             tool_info["input_model_name"] = slice_value.elts[0].id
                             pydantic_model_names.add(slice_value.elts[0].id)
                
                tool_function_infos.append(tool_info)


    model_map = load_pydantic_models_safely(agent_code_file, list(pydantic_model_names))
    final_tools_schema = []
    
    for info in tool_function_infos:
        input_model_name = info["input_model_name"]
        final_tool = {
            "name": info["name"],
            "description": info["description"],
            "parameters": {}
        }
        
        if input_model_name in model_map:
            try:
                input_model_class = model_map[input_model_name]
                final_tool["parameters"] = get_pydantic_schema_from_model(input_model_class)
            except Exception as e:
                print(f"❌ Error generating schema for tool '{info['name']}' from model '{input_model_name}': {e}")
                
        elif input_model_name:
            print(f"⚠️ Model '{input_model_name}' required by tool '{info['name']}' was not safely loaded. Parameters defaulted to empty.")
        
        if final_tool["parameters"].get("properties") is None or not final_tool["parameters"].get("properties"):
             final_tool["parameters"] = {"type": "object", "properties": {}, "required": []}


        final_tools_schema.append(final_tool)

    return final_tools_schema

# --- New Function to Save to JSON ---

def save_tools_to_json(tools_list: List[Dict[str, Any]], output_file: str):
    """Saves the extracted list of tools to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(tools_list, f, indent=2)
        print(f"\n✅ Successfully saved {len(tools_list)} tools to **{output_file}**.")
    except Exception as e:
        print(f"❌ Error saving tools to JSON file: {e}")

# --- Main Execution (for testing) ---

if __name__ == "__main__":
    try:    
        print(f"Reading tools from **{AGENT_FILE_PATH}**...")
        
        extracted_tools = extract_tools_from_decorated_functions(AGENT_FILE_PATH)
        
        if extracted_tools:
            save_tools_to_json(extracted_tools, TOOLS_OUTPUT_FILE)
        else:
            print("\n--- ❌ Failed to extract any tools. ---")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")