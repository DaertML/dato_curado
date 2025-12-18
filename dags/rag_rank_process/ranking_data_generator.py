import os
import json
import random
from pypdf import PdfReader
from text_processing_utils import run_ollama, save_jsonl_lines
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OLLAMA_MODEL_QGEN = "qwen3:8b" 
PDF_PATH = 'metasploit.pdf' 
OUTPUT_FILE_UNRANKED_DATA = 'synthetic_ranking_unranked.jsonl'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

PROMPT_QUESTION_GEN = """
You are an expert Question Generator. Your task is to generate a single, clear, and complex question based *only* on the provided text context.

The question must be fully answerable by the context.
Do not use phrases like 'based on the provided context' or 'according to the text'.
Output only the question string, without any additional text or introductory phrases.

Context:
---
{context}
---

Question:
"""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF using pypdf."""
    print(f"Extracting text from PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        return full_text
    except FileNotFoundError:
        print(f"⚠️ Error: PDF file not found at {pdf_path}. Returning dummy content.")
        # Provide sufficient length for chunking to work
        return "The capital of France is Paris and it is famous for the Eiffel Tower. The speed of light in a vacuum is approximately 299,792,458 meters per second. This value is a fundamental constant of nature and defines the metric system's unit of length. Modern physics relies heavily on this constant for defining relationships between space and time, particularly in special relativity. Understanding this speed is key to satellite communication and deep space exploration. This marks the end of the second major paragraph about speed of light. Now, let's discuss historical aspects of measurement, like early attempts to use eclipses. The measurement of the speed of light has evolved significantly over centuries, moving from astronomical methods to highly precise laboratory techniques, which confirmed the constant nature of 'c'. This is the third chunk of text for placeholder purposes." * 5
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def recursive_character_text_splitter(text: str, chunk_size: int, chunk_overlap: int, separators: list) -> list:
    """
    Manually implements a recursive character text splitter with overlap.
    (Implementation is simplified for brevity, based on previous prompt's constraints)
    """
    if not text:
        return []
    
    chunks = []
    
    def split_recursively(current_text, current_separators):
        if len(current_text) <= chunk_size:
            chunks.append(current_text)
            return

        if not current_separators:
            # Fallback to simple character splitting if no separators remain
            for i in range(0, len(current_text), chunk_size - chunk_overlap):
                chunks.append(current_text[i:i + chunk_size])
            return

        separator = current_separators[0]
        remaining_separators = current_separators[1:]
        
        parts = current_text.split(separator)
        current_chunk = ""
        
        for part in parts:
            new_addition = (separator if current_chunk else "") + part
            
            if len(current_chunk) + len(new_addition) > chunk_size:
                if current_chunk:
                    # If current_chunk is already large, split it further
                    if len(current_chunk) > chunk_size:
                        split_recursively(current_chunk, remaining_separators)
                    else:
                        chunks.append(current_chunk)
                    
                    # Start new chunk with the current part
                    current_chunk = part
                else:
                    # The part itself is too big, recurse on it
                    split_recursively(part, remaining_separators)
                    current_chunk = "" # Reset after recursion
            else:
                current_chunk += new_addition
        
        if current_chunk:
            if len(current_chunk) > chunk_size:
                split_recursively(current_chunk, remaining_separators)
            else:
                chunks.append(current_chunk)

    split_recursively(text, separators)
    
    # Simple overlap application on final chunks (as implemented in previous response)
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and chunk_overlap > 0 and (CHUNK_SIZE - CHUNK_OVERLAP) > 0:
            # Simple overlap application (re-add overlap from the previous chunk)
            overlap_index = min(CHUNK_OVERLAP, len(chunks[i-1]))
            overlap_text = chunks[i-1][-overlap_index:]
            final_chunks.append(overlap_text + chunk)
        else:
            final_chunks.append(chunk)

    return [c.strip() for c in final_chunks if c.strip() and len(c.strip()) > 100]


def generate_ranking_data(pdf_path: str):
    """Generates questions and selects excerpts of varying relevance."""
    
    # 1. Extract Text
    full_text = extract_text_from_pdf(pdf_path)
    
    # 2. Split Text Manually
    separators = ["\n\n", "\n", ".", " "] # Priority of separators
    contexts = recursive_character_text_splitter(full_text, CHUNK_SIZE, CHUNK_OVERLAP, separators)
    
    if len(contexts) < 10:
        print(f"🛑 Not enough chunks ({len(contexts)}). Need at least 10 for varied relevance selection. Exiting.")
        return

    print(f"Total text split into {len(contexts)} usable chunks.")
    
    unranked_data = []
    # Iterate over a middle subset to ensure we can select adjacent and distant chunks
    indices_to_process = range(5, len(contexts) - 5, 2) # Process every second chunk in the middle
    
    for i in indices_to_process:
        print(f"\nProcessing chunk {i+1}/{len(contexts)}...")
        
        # 3. Select Excerpts for Relevance Mix
        target_context = contexts[i]
        
        # Highly Relevant (Score ~10): The context itself
        excerpt_high_rel = target_context
        
        # Somewhat Relevant (Score ~5-7): Adjacent chunk (one index before or after)
        # Using i-1 to ensure some proximity in content
        excerpt_medium_rel = contexts[i - 1] 
        
        # Irrelevant (Score ~0-2): A chunk far away (e.g., first or last chunk)
        irrelevant_index = random.choice([0, len(contexts) - 1])
        excerpt_low_rel = contexts[irrelevant_index]
        
        # Compile excerpts, ensuring no duplicates and shuffling the order for the LLM prompt
        excerpts_list = [
            {"text": excerpt_high_rel, "type": "high"},
            {"text": excerpt_medium_rel, "type": "medium"},
            {"text": excerpt_low_rel, "type": "low"}
        ]
        
        # Remove duplicates, although they are selected systematically to avoid this
        unique_excerpts = []
        seen_texts = set()
        for item in excerpts_list:
            if item['text'] not in seen_texts:
                unique_excerpts.append(item)
                seen_texts.add(item['text'])
        
        random.shuffle(unique_excerpts)

        # 4. Generate the Question
        question_prompt = PROMPT_QUESTION_GEN.format(context=target_context)
        
        try:
            question = run_ollama(OLLAMA_MODEL_QGEN, "", question_prompt).strip()
            question = question.split('\n')[0].strip()
            if not question.endswith('?'):
                question += '?'

            print(f"  Q: {question}")
            
            # 5. Create the data structure
            unranked_data.append({
                "question": question,
                # Store the original text and its relevance type for validation (optional, but helpful)
                "excerpts": [{"text": e['text'], "original_type": e['type']} for e in unique_excerpts]
            })
            
        except Exception as e:
            print(f"Error generating question for chunk {i}: {e}. Skipping.")
            continue
            
    # Save the results
    save_jsonl_lines(unranked_data, OUTPUT_FILE_UNRANKED_DATA)
    print(f"\n✅ Finished. Saved {len(unranked_data)} unranked data points to {OUTPUT_FILE_UNRANKED_DATA}")


if __name__ == "__main__":
    generate_ranking_data(PDF_PATH)