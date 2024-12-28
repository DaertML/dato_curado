import os
from transformers import pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

# Load models once
toxic_filter = pipeline('text-classification', model='unitary/toxic-bert', batch_size=100, device='cuda')
lang_filter = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection', batch_size=100, device='cuda')
topic_filter = pipeline('text-classification', model='OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract', batch_size=100, device='cuda')

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def filter_lines(lines):
    # Run the toxic filter on all lines at once
    results = toxic_filter(lines, truncation=True)
    filtered_lines = []
    for line, result in zip(lines, results):
        #if len(line) < 150:
        #    continue
        if result['label'] in ['toxic', 'hate'] and result['score'] > 0.8:
            continue
        else:
            filtered_lines.append(line)
    return filtered_lines

def save_lines(lines):
    # Run the language filter on all lines at once
    language_results = detect_language(lines)
    
    # Create a dictionary to store lines by language and topic
    language_folders = {}
    
    for line, result in zip(lines, language_results):
        print(result)
        language = result['label']  # Directly access the label
        topic = detect_topic(line)[0]['label']  # Get the label of the most likely prediction
        
        if language not in language_folders:
            language_folders[language] = {}
        if topic not in language_folders[language]:
            language_folders[language][topic] = []
        
        language_folders[language][topic].append(line)
    
    # Save the lines to files based on language and topic
    for language, topics in language_folders.items():
        language_path = os.path.join('./output', language)
        os.makedirs(language_path, exist_ok=True)
        for topic, lines in topics.items():
            topic_path = os.path.join(language_path, topic)
            os.makedirs(topic_path, exist_ok=True)
            with open(os.path.join(topic_path, 'output.txt'), 'w') as file:
                file.writelines(lines)

def detect_language(text):
    results = lang_filter(text)
    return results  # Return the list of predictions

def detect_topic(text):
    results = topic_filter(text)
    return results  # Return the list of predictions

def create_dataset(lines, batch_number):
    dataset = Dataset.from_dict({'text': lines})
    dataset.save_to_disk(f'./output/dataset_batch_{batch_number}')
