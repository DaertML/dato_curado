import os
from transformers import pipeline
from datasets import Dataset
from text_processing_utils import read_file, save_lines, move_files_to_processed,add_solution_process_judge_llm,csv_to_prompt,gen_synth_dataset
from datetime import datetime
import pandas as pd

MAX_LINES = 100000


def process_batch(batch):
    batch['text'] = [text for text in batch['text']]
    return batch

def process_text(input_file):
    lines = csv_to_prompt(input_file,["SMILES","Question"])

    batch_size = 10  # Define your batch size
    start_time = datetime.now()
    for i in range(34240, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        print(batch)
        dataset = Dataset.from_dict({'questions': batch})
        processed_lines = gen_synth_dataset(dataset['questions'])
        save_lines(processed_lines,f"../ChEMBL_QA_sol{i}.jsonl")
        print("Finished Batch " + str(i))

    end_time = datetime.now()
    print("Took " + str(end_time - start_time))

if __name__ == "__main__":
    input_file = '../ChEMBL_QA.csv'  # Update with your input file path
    process_text(input_file)