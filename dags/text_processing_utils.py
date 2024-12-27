import os
from transformers import pipeline

toxic_filter = pipeline('text-classification', model='unitary/toxic-bert')
lang_filter = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def filter_lines(lines):
    filtered_lines = []
    for line in lines:
        if len(line) < 150:
            continue
        results = toxic_filter(line)
        if any(label['label'] in ['toxic', 'hate'] and label['score'] > 0.8 for label in results):
            continue
        filtered_lines.append(line)
    return filtered_lines

def save_lines(lines):
    language_folders = {}
    for line in lines:
        language = detect_language(line)
        topic = detect_topic(line)
        if language not in language_folders:
            language_folders[language] = {}
        if topic not in language_folders[language]:
            language_folders[language][topic] = []
        language_folders[language][topic].append(line)
    
    for language, topics in language_folders.items():
        language_path = os.path.join('/path/to/output', language)
        os.makedirs(language_path, exist_ok=True)
        for topic, lines in topics.items():
            topic_path = os.path.join(language_path, topic)
            os.makedirs(topic_path, exist_ok=True)
            with open(os.path.join(topic_path, 'output.txt'), 'w') as file:
                file.writelines(lines)

def detect_language(text):
    results = lang_filter(text)
    # Get the most likely prediction
    prediction = max(results, key=lambda x: x['score'])
    return prediction['label']

def detect_topic(text):
    # Implement topic detection logic
    return 'general'
