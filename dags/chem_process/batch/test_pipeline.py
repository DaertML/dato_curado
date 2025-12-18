import os
from transformers import pipeline
from datasets import Dataset
from text_processing_utils import (
    read_file, save_lines, move_files_to_processed, 
    add_solution_process_judge_llm_batch, csv_to_prompt, gen_synth_dataset_batch
)
from datetime import datetime
import pandas as pd

MAX_LINES = 100000

def process_batch(batch):
    batch['text'] = [text for text in batch['text']]
    return batch

def process_text_batch(input_file, start_index=0, batch_size=50):
    """Process text in batches for better efficiency"""
    lines = csv_to_prompt(input_file, ["SMILES", "Question"])
    
    # Skip already processed lines
    lines_to_process = lines[start_index:]
    
    start_time = datetime.now()
    
    # Process in larger batches
    for i in range(0, len(lines_to_process), batch_size):
        batch_end = min(i + batch_size, len(lines_to_process))
        batch = lines_to_process[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1} (lines {start_index + i} to {start_index + batch_end - 1})")
        
        # Use batch processing
        processed_lines = gen_synth_dataset_batch(batch, batch_size=min(batch_size, 10))
        
        # Save results
        save_lines(processed_lines, f"../ChEMBL_QA_sol_{start_index + i}.jsonl")
        print(f"Finished Batch {start_index + i}")

    end_time = datetime.now()
    print(f"Total processing time: {end_time - start_time}")

def process_text_judge_batch(questions_file, answers_file, start_index=0, batch_size=20):
    """Process with judge LLM in batch mode"""
    # Load questions and answers
    questions = csv_to_prompt(questions_file, ["SMILES", "Question"])
    answers_df = pd.read_csv(answers_file)
    answers = answers_df["Answer"].fillna('').astype(str).tolist()
    
    # Ensure same length
    min_length = min(len(questions), len(answers))
    questions = questions[:min_length]
    answers = answers[:min_length]
    
    # Skip already processed lines
    questions_to_process = questions[start_index:]
    answers_to_process = answers[start_index:]
    
    start_time = datetime.now()
    
    for i in range(0, len(questions_to_process), batch_size):
        batch_end = min(i + batch_size, len(questions_to_process))
        batch_questions = questions_to_process[i:batch_end]
        batch_answers = answers_to_process[i:batch_end]
        
        print(f"Processing judge batch {i//batch_size + 1} (items {start_index + i} to {start_index + batch_end - 1})")
        
        # Use batch processing with judge
        processed_data = add_solution_process_judge_llm_batch(
            batch_questions, batch_answers, batch_size=min(batch_size, 5)
        )
        
        # Save results
        save_lines(processed_data, f"../ChEMBL_QA_judged_{start_index + i}.jsonl")
        print(f"Finished Judge Batch {start_index + i}")

    end_time = datetime.now()
    print(f"Total judge processing time: {end_time - start_time}")

if __name__ == "__main__":
    input_file = '../ChEMBL_QA.csv'
    
    # Choose processing mode
    processing_mode = "synthetic"  # Change to "judge" for judge mode
    
    if processing_mode == "synthetic":
        # For synthetic dataset generation
        process_text_batch(input_file, start_index=28490, batch_size=50)
    elif processing_mode == "judge":
        # For judge processing (if you have answers file)
        answers_file = '../ChEMBL_QA_answers.csv'  # Update with your answers file
        process_text_judge_batch(input_file, answers_file, start_index=0, batch_size=20)

