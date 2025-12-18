from utils import read_tools_json, run_ollama_generation, save_jsonl_data
from datetime import datetime
import json

# --- Configuration ---
INPUT_TOOLS_FILE = 'extracted_tools.json'  # Input JSON file containing the tools
OUTPUT_JSONL_FILE = 'synthetic_tool_tasks.jsonl'  # Output JSONL file
GENERATION_COUNT = 10  # Number of synthetic questions to generate
OLLAMA_MODEL = "qwen3:8b" # Must match the model in utils.py

def generate_synthetic_data(tools_data):
    """
    Generates synthetic questions using the LLM and the provided tools data.

    Args:
        tools_data (dict): The loaded content of the tools JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary is a data point
              for the JSONL file.
    """
    generated_data = []
    
    # The tools JSON content is the same for every data point
    tools_json_field = json.dumps(tools_data)

    for i in range(GENERATION_COUNT):
        print(f"--- Generating Task {i+1}/{GENERATION_COUNT} ---")
        
        # 1. Generate the synthetic question using Ollama
        synthetic_question = run_ollama_generation(OLLAMA_MODEL, tools_data)

        if synthetic_question:
            # 2. Create the final data point dictionary
            data_point = {
                "question": synthetic_question,
                "available_tools": tools_json_field
            }
            
            generated_data.append(data_point)
            print(f"Generated Question: {synthetic_question[:80]}...")
        else:
            print("Generation failed, skipping this iteration.")
        
    return generated_data

def main_pipeline():
    """
    Main function to run the synthetic data generation pipeline.
    """
    start_time = datetime.now()
    
    # 1. Read the input tools JSON file
    print(f"⏳ Reading tools from: {INPUT_TOOLS_FILE}")
    tools_data = read_tools_json(INPUT_TOOLS_FILE)

    if tools_data is None:
        print("❌ Pipeline failed to start due to error in reading tools file.")
        return

    # 2. Generate the synthetic data points
    print(f"🧠 Starting synthetic question generation (x{GENERATION_COUNT}) with LLM...")
    final_data = generate_synthetic_data(tools_data)

    # 3. Save the results to the output JSONL file
    if final_data:
        print("💾 Saving final results...")
        save_jsonl_data(final_data, OUTPUT_JSONL_FILE)
    else:
        print("⚠️ No data generated. Skipping file save.")

    end_time = datetime.now()
    print("--- Pipeline Complete ---")
    print(f"Total time taken: {end_time - start_time}")


if __name__ == "__main__":
    main_pipeline()