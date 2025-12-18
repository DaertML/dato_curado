import os
import json
from text_processing_utils import run_ollama, save_jsonl_lines, read_jsonl_file
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OLLAMA_MODEL_ANSWER = "qwen3:8b" 
INPUT_FILE_QA_CONTEXT = 'synthetic_qa_context.jsonl'
OUTPUT_FILE_FINAL_DATA = 'synthetic_qa_final.jsonl'

PROMPT_ANSWER_GEN = """
You are an expert Q&A AI. Your task is to generate a concise and accurate answer for the given question, relying *strictly* on the provided context.

Do not use any external knowledge. If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
---
{context}
---

Question: {question}

Answer:
"""

def generate_answers(input_file: str):
    """Reads Q/C pairs, generates answers, and saves the final Q/C/A data."""
    
    # Read the data from the first pipeline
    qa_context_data = read_jsonl_file(input_file)
    print(f"Loaded {len(qa_context_data)} question/context pairs from {input_file}.")
    
    final_qa_data = []

    for i, item in enumerate(qa_context_data):
        question = item['question']
        context = item['context']
        
        print(f"Processing item {i+1}/{len(qa_context_data)}...")
        
        # Format the answer generation prompt
        answer_prompt = PROMPT_ANSWER_GEN.format(context=context, question=question)
        
        try:
            # Generate the answer using Ollama
            answer = run_ollama(OLLAMA_MODEL_ANSWER, "", answer_prompt).strip()
            
            # Create the final data structure
            final_qa_data.append({
                "context": context,
                "question": question,
                "answer": answer
            })
            print(f"  Q: {question}")
            print(f"  A: {answer[:80]}")

        except Exception as e:
            print(f"Error generating answer for item {i}: {e}. Skipping.")
            continue

    # Save the final results
    save_jsonl_lines(final_qa_data, OUTPUT_FILE_FINAL_DATA)
    print(f"\n✅ Finished. Saved {len(final_qa_data)} final Q/C/A records to {OUTPUT_FILE_FINAL_DATA}")


if __name__ == "__main__":
    generate_answers(INPUT_FILE_QA_CONTEXT)