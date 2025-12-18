import json
import os
import random
from typing import Dict, Any, List, Optional
from utils import run_ollama_tool_execution, save_jsonl_data

# --- Configuration Constants ---
# !!! IMPORTANT !!!
# Ensure you have a tool-calling capable model pulled (e.g., 'ollama pull llama3:8b-instruct-tool-use-latest')
OLLAMA_MODEL = "qwen3:8b" # Use a capable model
MAX_INTERACTION_TURNS = 5 # Maximum number of turns (N)
INPUT_TASK_FILE = "synthetic_tool_tasks.jsonl" # File created by task_pipeline.py
OUTPUT_FILE = "ollama_interactive_results.jsonl"

# --- LLM-as-User Prompt (Reflector/Feedback Generator) ---
# This is the critical prompt to emulate the user's next turn.
# It is specifically instructed to create scenarios like mistakes and feedback.
REFLECTOR_SYSTEM_PROMPT = """
You are an AI assistant specialized in simulating a user's role in a multi-turn conversation.
Your goal is to generate the *next message* a user would send to an AI agent, based on the *previous turn's exchange*.

You will be given the full conversation history so far (messages) and the Agent's most recent action/response.
Your generated message must advance the conversation, either by:
1.  **Providing the output of a requested tool call** (if the Agent requested one).
2.  **Giving feedback on the Agent's answer**, perhaps noting a mistake, clarifying a detail, or asking a follow-up question.
3.  **If the conversation seems complete**, generate a final confirmation or a new, simple question.

**CRITICAL INSTRUCTION:** For the first turn, or randomly (about 30% of the time), generate a message that either:
-   **Requires the Agent to use a tool incorrectly** (e.g., asking for weather in an invalid place) or;
-   **Gives ambiguous instructions** that necessitate clarification from the Agent, or;
-   **Asks a multi-step question** that forces the Agent to use multiple tools or turns.

Only output the raw, synthetically generated user message as plain text. Do not include any prefixes.
"""

def run_ollama_reflector_turn(history: List[Dict[str, str]], last_agent_response: Dict[str, Any]) -> Optional[str]:
    """
    Uses Ollama to generate the next user message/feedback based on history.
    """
    
    # 1. Format the history and last response for the Reflector's context
    context_str = "\n--- CONVERSATION HISTORY ---\n"
    for msg in history:
        context_str += f"[{msg['role'].upper()}]: {msg['content']}\n"
    
    context_str += "\n--- AGENT'S LAST RESPONSE (Action/Answer) ---\n"
    context_str += json.dumps(last_agent_response, indent=2)

    # 2. Build the message payload for the Reflector LLM
    messages = [
        {'role': 'system', 'content': REFLECTOR_SYSTEM_PROMPT},
        {'role': 'user', 'content': f"Here is the current conversation context and the Agent's last response. Generate the next user message now:\n\n{context_str}"}
    ]
    
    print("  > Reflector (LLM-as-User) is generating next message...")
    
    try:
        response = run_ollama_tool_execution(
            model=OLLAMA_MODEL, 
            question=messages[1]['content'], # Pass the user message content
            available_tools=[] # Reflector does not need tools
        )
        return response.message.content.strip() if response else None
    except Exception as e:
        print(f"Error during Reflector LLM inference: {e}")
        return None

def run_interactive_session(task: Dict[str, Any], session_id: int) -> Dict[str, Any]:
    """
    Runs the multi-turn interaction loop for a single task.
    """
    question = task.get("question", "")
    # Ensure available_tools is a list
    available_tools = task.get("available_tools", [])
    if isinstance(available_tools, str):
        try:
            available_tools = json.loads(available_tools)
        except json.JSONDecodeError:
            available_tools = []
    
    # Randomly determine the number of turns (N) for this session
    num_turns = random.randint(1, MAX_INTERACTION_TURNS) 
    print(f"  > Session {session_id} configured for {num_turns} turns.")

    # The conversation history for the Agent (messages format for ollama.chat)
    conversation_history: List[Dict[str, str]] = []
    
    # The full record of the exchange for saving (for the final output JSONL)
    session_record = [] 
    
    # Start with the initial question as the first user message
    next_user_message = question
    
    for i in range(1, num_turns + 1):
        print(f"\n  --- Turn {i}/{num_turns} (Agent Action) ---")

        # 1. Agent receives the user message and history
        # Add the new user message to the conversation history
        conversation_history.append({'role': 'user', 'content': next_user_message})
        
        # Call the Agent LLM with the full history and tools
        # We must re-implement the core logic of run_ollama_tool_execution here 
        # to correctly pass the history/messages array for multi-turn.
        ollama_tools = []
        if available_tools:
            ollama_tools = [
                {"type": "function", "function": {"name": t.get("name"), "description": t.get("description"), "parameters": t.get("parameters")}}
                for t in available_tools if isinstance(t, dict)
            ]
        
        # Build the final message list for ollama.chat, including system prompt
        messages_with_system = [
            {'role': 'system', 'content': 'You are a helpful assistant specialized in executing tasks. Use the available tools when necessary to answer the user\'s question.'}
        ] + conversation_history

        try:
            full_ollama_response = chat(
                model=OLLAMA_MODEL, 
                messages=messages_with_system,
                tools=ollama_tools # Ollama handles tool injection
            )
        except Exception as e:
            print(f"Error during Agent LLM inference (Turn {i}): {e}")
            break # Exit the loop on error

        # The actual content is nested in the 'message' key
        result_message = full_ollama_response.get('message', {})
        
        # 2. Extract Response and Prepare for Next Turn
        tool_calls = result_message.get('tool_calls') 
        response_text = result_message.get('content', '').strip()

        # Add the Agent's response to history for the next turn
        conversation_history.append({'role': 'assistant', 'content': response_text, 'tool_calls': tool_calls})
        
        # Record the turn for final output
        turn_record = {
            "turn": i,
            "user_message": next_user_message,
            "agent_response": response_text,
            "agent_tool_calls": json.loads(json.dumps(tool_calls, default=lambda o: getattr(o, '__dict__', str(o)))) if tool_calls else None,
            "is_final_turn": i == num_turns
        }
        session_record.append(turn_record)
        print(f"    Agent Response: {response_text[:80]}...")
        if tool_calls:
            print(f"    Agent Tool Calls: {len(tool_calls)}")

        # 3. Reflector (LLM-as-User) generates the next message (if not the last turn)
        if i < num_turns:
            # Prepare the Agent's response structure for the Reflector prompt
            agent_output_for_reflector = {
                "turn": i,
                "response_text": response_text,
                "tool_calls_present": bool(tool_calls)
            }
            
            # The Reflector generates the next user message
            next_user_message = run_ollama_reflector_turn(conversation_history, agent_output_for_reflector)
            
            if not next_user_message:
                print("  > Reflector failed to generate next message. Ending session early.")
                break
            
            print(f"  > NEXT USER MESSAGE: {next_user_message[:80]}...")
        
    # 4. Final compilation of the task record
    final_record = {
        "original_question": task.get("question"),
        "available_tools": available_tools,
        "total_turns": i,
        "max_turns_configured": num_turns,
        "interaction_history": session_record
    }

    return final_record


def main_interactive_pipeline():
    """
    Main function to run the interactive, multi-turn data processing pipeline.
    """
    print(f"🚀 Starting Interactive Ollama Tool Task Pipeline (Multi-Turn Simulation)")
    print(f"Model: {OLLAMA_MODEL}, Max Turns: {MAX_INTERACTION_TURNS}")

    if not os.path.exists(INPUT_TASK_FILE):
        print(f"Error: Input file '{INPUT_TASK_FILE}' not found. Run 'task_pipeline.py' first.")
        return

    tasks = []
    try:
        # Read the JSONL input file
        with open(INPUT_TASK_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        print(f"Successfully loaded {len(tasks)} tasks for interactive processing.")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    interactive_results = []
    
    # Process each task interactively
    for i, task in enumerate(tasks):
        print(f"\n=======================================================")
        print(f"** Running Session {i+1}/{len(tasks)} **")
        print(f"Initial Question: {task['question'][:70]}...")
        print(f"=======================================================")
        
        result_record = run_interactive_session(task, i + 1)
        interactive_results.append(result_record)
        
    # Write the results to the new JSONL output file
    save_jsonl_data(interactive_results, OUTPUT_FILE)


if __name__ == "__main__":
    from ollama import chat # Import the required ollama function
    main_interactive_pipeline()