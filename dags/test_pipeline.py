import os
from transformers import pipeline
from datasets import Dataset
from text_processing_utils import read_file, filter_lines, save_lines
import datetime

MAX_LINES = 100000


def process_batch(batch):
    batch['text'] = [text for text in batch['text']]
    return batch

def process_text(input_file):
    lines = read_file(input_file)[:MAX_LINES]

    batch_size = 10000  # Define your batch size
    start_time = datetime.datetime.now()
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]

        dataset = Dataset.from_dict({'text': batch})
        #dataset = dataset.map(process_batch, batched=True, batch_size=10)
        filtered_lines = filter_lines(dataset['text'])

        print("Filtered " + str(len(batch) - len(filtered_lines)))
        save_lines(filtered_lines)
        print("Finished Batch " + str(i))

    end_time = datetime.datetime.now()
    print("Took " + str(end_time - start_time))

if __name__ == "__main__":
    input_file = '../tinystories.txt'  # Update with your input file path
    process_text(input_file)
