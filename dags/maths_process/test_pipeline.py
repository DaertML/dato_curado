import os
from transformers import pipeline
from datasets import Dataset
from text_processing_utils import read_file, add_solution_process, save_lines, move_files_to_processed,add_solution_process_judge_llm, add_solution_process_eval_expr
from datetime import datetime

MAX_LINES = 100000
final_result_file = "../algebra__polynomial_roots.jsonl"
lines_final_result_file_content = open(final_result_file,"r").read()


def process_batch(batch):
    batch['text'] = [text for text in batch['text']]
    return batch

def process_text(input_file):
    lines = read_file(input_file)

    batch_size = 10  # Define your batch size
    start_time = datetime.now()
    start_line = 2180
    for i in range(start_line, len(lines), batch_size):
        try:
            batch = lines[i:i + batch_size*2]
            questions = []
            answers = []
            skipped = False
            for j, line in enumerate(batch):
                print(line)
                if skipped or line.strip() in lines_final_result_file_content:
                    skipped = True
                    print("⚠️ SKIPPING...")
                    continue
                if j %2 == 0:
                    questions.append(line)
                else:
                    answers.append(line)
                skipped = False
            print(len(questions),len(answers))
            dataset = Dataset.from_dict({'questions': questions, 'answers': answers})
            processed_lines = add_solution_process_judge_llm(dataset['questions'],dataset['answers'])

            if len(processed_lines) != 0:
                save_lines(processed_lines,f"../algebra__polynomial_roots_sol{i}.txt")
            print("Finished Batch " + str(i))
        except:
            continue
    end_time = datetime.now()
    print("Took " + str(end_time - start_time))


if __name__ == "__main__":
    input_file = '../algebra__polynomial_roots.txt'  # Update with your input file path
    process_text(input_file)
