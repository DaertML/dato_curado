# data_pipeline.py
import pandas as pd
import os
import random
import json
from datasets import DatasetDict, Dataset
from data_utils import (
    run_ollama_chat, generate_tool_metadata_from_ollama, get_tool_name, 
    num_tools_available, generate_system_prompt, parse_tool_call, 
    call_tool, format_tool_result, add_answer_tags, save_traces_to_jsonl, read_jsonl_file,
    OLLAMA_MODEL_GENERATION, OLLAMA_MODEL_TRACING
)

# Configuration from the original script
N = 200
category_list = ["General Knowledge", "Lifestyle advice", "Creative Writing", "Definitions", "Explanations", "Comparison", "Step-by-step instructions", "Writing feedback", "Venting", "code generation"]
seed_list = ["cooking", "fitness", "business", "education", "hobbies",]
difficulty_list = ["beginner level", "intermediate level"]
typos_list = ["", "Please include formatting errors and typos."]
num_configs = len(category_list) * len(typos_list) * len(difficulty_list) * len(seed_list)
num_queries_per_call = max(1, int(N / num_configs)) # Ensure at least 1 query per config
num_tool_queries_per_type = 5 

# Cache for dynamically generated tool metadata
ALL_TOOL_METADATA_CACHE = []

# --- Step 0: Dynamic Tool Generation for ALL Contexts ---

def generate_dynamic_tool_metadata():
    """Generates tool definitions using LLM for every context combination."""
    global ALL_TOOL_METADATA_CACHE
    print("--- Generating Dynamic Tool Metadata using LLM ---")
    
    # Context for no_tool queries
    ALL_TOOL_METADATA_CACHE.append(
        generate_tool_metadata_from_ollama("General purpose assistant tasks that require no external tools like defining concepts or writing code.")
    )

    # Context for tool queries (using combinations)
    for category in category_list:
        for seed in seed_list:
            context = f"Category: {category}, Focus: {seed}. Need a tool to help with this domain."
            ALL_TOOL_METADATA_CACHE.extend(generate_tool_metadata_from_ollama(context))
    
    # Flatten the list and remove empty entries
    ALL_TOOL_METADATA_CACHE = [item for sublist in ALL_TOOL_METADATA_CACHE for item in sublist if item]
    
    print(f"Generated a total of {len(ALL_TOOL_METADATA_CACHE)} unique tool metadata definitions.")


# --- Step 1: Query Generation (Unchanged logic, but uses real LLM) ---

def generate_no_tool_queries() -> pd.DataFrame:
    """Generates queries that do not require a tool."""
    query_dict = {"query": [], "query_type": [], "tool_needed": [], "tool_name": []}
    no_tool_instructions = "You are a query generator. Create one simple, human-like query that can be easily solved without the use of any tools. The context provided is to inspire the query topic."
    user_prompt_template = lambda category, seed, difficulty, typo: f"Category: {category} \nSeed: {seed} \nDifficulty: {difficulty} \n{typo}"

    for category in category_list:
        for typo in typos_list:
            for difficulty in difficulty_list:
                for seed in seed_list:
                    user_message = user_prompt_template(category, seed, difficulty, typo)
                    messages = [{"role": "system", "content": no_tool_instructions}, {"role": "user", "content": user_message}]
                    query_list = [run_ollama_chat(OLLAMA_MODEL_GENERATION, messages) for _ in range(num_queries_per_call)]
            
                    query_dict['query'].extend(query_list)
                    query_dict['query_type'].extend(['no_tool'] * len(query_list))
                    query_dict['tool_needed'].extend([False] * len(query_list))
                    query_dict['tool_name'].extend([None] * len(query_list))
    
    return pd.DataFrame(query_dict)

def generate_tool_queries() -> pd.DataFrame:
    """Generates queries that require a tool (easy and hard variants), using the dynamically generated tools."""
    global ALL_TOOL_METADATA_CACHE
    
    # Ensure there are tools to generate queries for
    if not ALL_TOOL_METADATA_CACHE:
        print("Error: No tool metadata available for tool query generation.")
        return pd.DataFrame()
        
    query_dict = {"query": [], "query_type": [], "tool_needed": [], "tool_name": []}
    easy_instructions = 'Generate a simple, human-like query that can only be resolved using the tool described by the user.'
    hard_instructions = 'Generate a simple, human-like query that can only be resolved using the tool described by the user and a little thinking. Please include formatting errors and typos.'

    # Use the dynamically generated tools as input
    for tool_metadata in ALL_TOOL_METADATA_CACHE:
        tool_name = get_tool_name(tool_metadata)

        # Easy queries
        messages_easy = [{"role": "system", "content": easy_instructions}, {"role": "user", "content": tool_metadata}]
        easy_query_list = [run_ollama_chat(OLLAMA_MODEL_GENERATION, messages_easy) for _ in range(num_tool_queries_per_type)]

        # Hard queries
        messages_hard = [{"role": "system", "content": hard_instructions}, {"role": "user", "content": tool_metadata}]
        hard_query_list = [run_ollama_chat(OLLAMA_MODEL_GENERATION, messages_hard) for _ in range(num_tool_queries_per_type)]

        query_dict['query'].extend(easy_query_list + hard_query_list)
        query_dict['query_type'].extend((['easy'] * len(easy_query_list)) + (['hard'] * len(hard_query_list)))
        query_dict['tool_needed'].extend([True] * (len(easy_query_list) + len(hard_query_list)))
        query_dict['tool_name'].extend([tool_name] * (len(easy_query_list) + len(hard_query_list)))
    
    return pd.DataFrame(query_dict)


# --- Step 2 & 3: Trace Execution and Data Finalization ---

def execute_no_tool_traces(df_notool: pd.DataFrame) -> tuple[list, list]:
    """Simulates LLM response for no-tool queries."""
    trace_list = []
    num_tools_list = []
    
    print("--- Executing No-Tool Traces ---")
    
    # NOTE: The original code uses a system prompt that includes tools for no-tool queries, 
    # forcing the model to decide not to use them. We use the full global tool list here.
    
    for _, row in df_notool.iterrows():
        num_tools_input = random.choice(['single', 'few', 'many'])
        num_tools = num_tools_available(num_tools_input)
        system_prompt = generate_system_prompt(
            tool_name=None, 
            num_tools=num_tools, 
            tool_metadata_list=ALL_TOOL_METADATA_CACHE
        )
        query = row['query']

        message_list = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        
        response = run_ollama_chat(OLLAMA_MODEL_TRACING, message_list)
        message_list.append({"role": "assistant", "content": response})

        trace_list.append(message_list)
        num_tools_list.append(num_tools)
        
    return trace_list, num_tools_list

def execute_tool_traces(df_tool: pd.DataFrame, system_prompt_path: str) -> tuple[list, list]:
    """Simulates LLM response for tool queries (call, result, final answer)."""
    
    trace_list = []
    num_tools_list = []
    
    print(f"--- Executing Tool Traces for {df_tool['query_type'].iloc[0]} (System Prompt: {system_prompt_path}) ---")
    
    for _, row in df_tool.iterrows():
        query = row['query']
        tool_name_required = row['tool_name']
        
        # 1. Generate Tool Call 
        # For the tool call generation step, we include only the required tool (num_tools=1)
        # to ensure the LLM focuses on generating the correct call.
        system_prompt_call = generate_system_prompt(
            tool_name=tool_name_required, 
            num_tools=1, 
            filepath=system_prompt_path, 
            tool_metadata_list=ALL_TOOL_METADATA_CACHE
        )
        messages_call = [{"role": "system", "content": system_prompt_call}, {"role": "user", "content": query}]
        tool_call_response = run_ollama_chat(OLLAMA_MODEL_GENERATION, messages_call)

        # 2. Parse Tool Call and Execute Tool
        tool_name, tool_args = parse_tool_call(tool_call_response)

        tool_result = ""
        try:
            if tool_name and tool_args is not None:
                result = call_tool(tool_name, tool_args)
                tool_result = format_tool_result(result)
            else:
                raise Exception("Could not parse tool call.")
        except Exception as e:
            tool_result = format_tool_result(f"Error executing tool {tool_name}: {e}")

        # 3. Generate Final Response
        num_tools_input = random.choice(['none', 'few', 'many'])
        num_tools = num_tools_available(num_tools_input)
        
        # The final prompt includes the required tool plus others based on num_tools_input
        system_prompt_final = generate_system_prompt(
            tool_name=tool_name_required, 
            num_tools=num_tools, 
            tool_metadata_list=ALL_TOOL_METADATA_CACHE
        )

        message_list = [
            {"role": "system", "content": system_prompt_final},
            {"role": "user", "content": query},
            {"role": "assistant", "content": tool_call_response},
            {"role": "user", "content": tool_result},
        ]

        final_response = run_ollama_chat(OLLAMA_MODEL_TRACING, message_list)
        message_list.append({"role": "assistant", "content": final_response})

        trace_list.append(message_list)
        num_tools_list.append(num_tools)
        
    return trace_list, num_tools_list


def main_data_pipeline(traces_jsonl_path: str = 'data/traces.jsonl'):
    """Main function to run the production data generation and tracing process."""
    
    # 0. Generate all needed tool definitions dynamically
    generate_dynamic_tool_metadata()

    # 1. Generate Query Dataset
    df_no_tool = generate_no_tool_queries()
    df_tool = generate_tool_queries()
    
    df_queries = pd.concat([df_no_tool, df_tool], ignore_index=True)
    df_queries['query_type'] = pd.Categorical(df_queries['query_type'], categories=['no_tool', 'easy', 'hard'], ordered=True)
    df_queries = df_queries.sort_values('query_type').reset_index(drop=True)

    # 2. Execute Traces
    df_notool = df_queries[df_queries['query_type'] == 'no_tool'].copy()
    df_easy = df_queries[df_queries['query_type'] == 'easy'].copy()
    df_hard = df_queries[df_queries['query_type'] == 'hard'].copy()

    trace_notool, num_tools_notool = execute_no_tool_traces(df_notool)
    trace_easy, num_tools_easy = execute_tool_traces(df_easy, 'prompts/system.md')
    trace_hard, num_tools_hard = execute_tool_traces(df_hard, 'prompts/system_cot.md')

    # 3. Finalize Data and Write to JSONL
    df_final = pd.concat([df_notool, df_easy, df_hard]).sort_values('query_type').reset_index(drop=True)
    
    all_traces = trace_notool + trace_easy + trace_hard
    all_num_tools = num_tools_notool + num_tools_easy + num_tools_hard
    
    final_data = df_final.to_dict('records')
    
    for i, row_dict in enumerate(final_data):
        row_dict['trace'] = all_traces[i]
        row_dict['num_tools_available'] = all_num_tools[i]
        row_dict['trace'] = add_answer_tags(row_dict['trace'], i)
    
    save_traces_to_jsonl(final_data, traces_jsonl_path)
    print(f"\n--- Data generation complete. Traces saved to {traces_jsonl_path} ---")


def split_dataset(traces_jsonl_path: str = 'data/traces.jsonl') -> DatasetDict:
    """Loads the final traces JSONL and splits it into train/validation/test DatasetDict."""
    
    data_list = read_jsonl_file(traces_jsonl_path)
    if not data_list:
        return DatasetDict({})

    df = pd.DataFrame(data_list)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_frac = 0.8
    valid_frac = 0.1
    n_samples = len(df_shuffled)
    train_size = int(train_frac * n_samples)
    valid_size = int(valid_frac * n_samples) + 1 

    df_train = df_shuffled[:train_size]
    df_valid = df_shuffled[train_size:train_size + valid_size]
    df_test = df_shuffled[train_size + valid_size:]

    print("\n--- Dataset Split ---")
    print(f"Training set: {len(df_train)} samples")
    print(f"Validation set: {len(df_valid)} samples")
    print(f"Test set: {len(df_test)} samples")

    train_ds = Dataset.from_pandas(df_train)
    valid_ds = Dataset.from_pandas(df_valid)
    test_ds = Dataset.from_pandas(df_test)

    dataset_dict = DatasetDict({
        'train': train_ds,
        'validation': valid_ds,
        'test': test_ds
    })
    
    return dataset_dict


if __name__ == "__main__":
    
    # NOTE: Set the OLLAMA_HOST environment variable if running on a remote server.
    # e.g., export OLLAMA_HOST="http://192.168.1.100:11434"
    # Ensure the models are pulled: ollama pull qwen2:7b 
    
    traces_file_path = 'data/traces.jsonl'
    
    main_data_pipeline(traces_file_path)
    
    dataset_dict = split_dataset(traces_file_path)
    
    print("\nFinal DatasetDict object:")
    print(dataset_dict)