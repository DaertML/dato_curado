import reasoning_gym
from datasets import Dataset
import json

# Create the original dataset
data = reasoning_gym.create_dataset('leg_counting', size=1000, seed=42)

# Transform the data into HuggingFace format
transformed_data = []

for entry in data:
    question = entry['question']
    answer = str(entry['answer'])
    
    # Generate multiple valid answer formats
    solutions = [
        answer,  # Plain number: "286"
        f"{answer} legs",  # With "legs": "286 legs"
        f"{answer}.0",  # Decimal format: "286.0"
        f"{answer}.0 legs",  # Decimal with "legs": "286.0 legs"
        answer,  # Duplicate for common format
    ]
    
    # Remove duplicates while preserving order
    solutions = list(dict.fromkeys(solutions))
    
    transformed_data.append({
        'question': question,
        'solutions': solutions
    })

# Create HuggingFace dataset
hf_dataset = Dataset.from_dict({
    'question': [item['question'] for item in transformed_data],
    'solutions': [item['solutions'] for item in transformed_data]
})

# Split into train (80%) and test (20%)
train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Save as parquet files
train_dataset.to_parquet('train.parquet')
test_dataset.to_parquet('test.parquet')

print(f"Train dataset saved to train.parquet ({len(train_dataset)} examples)")
print(f"Test dataset saved to test.parquet ({len(test_dataset)} examples)")
print(f"Total examples: {len(hf_dataset)}")

print("\nFirst train example:")
print(f"Question: {train_dataset[0]['question']}")
print(f"Solutions: {train_dataset[0]['solutions']}")

print("\nFirst test example:")
print(f"Question: {test_dataset[0]['question']}")
print(f"Solutions: {test_dataset[0]['solutions']}")

# Optional: Also save full dataset to disk
output_path = './leg_counting_dataset'
hf_dataset.save_to_disk(output_path)
print(f"\nFull dataset also saved to {output_path}")
