import os
import json
from pypdf import PdfReader
from text_processing_utils import run_ollama, save_jsonl_lines
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OLLAMA_MODEL_QGEN = "qwen3:8b" 
PDF_PATH = 'metasploit.pdf' 
OUTPUT_FILE_QA_CONTEXT = 'synthetic_qa_context.jsonl'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

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
        return "The capital of France is Paris. The speed of light in a vacuum is approximately 299,792,458 meters per second."
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def recursive_character_text_splitter(text: str, chunk_size: int, chunk_overlap: int, separators: list) -> list:
    """
    Manually implements a recursive character text splitter.
    It attempts to split the text based on a hierarchy of separators 
    to create chunks that respect semantic boundaries.
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
        
        # Split text by the current separator
        parts = current_text.split(separator)
        
        # Process each part
        current_chunk = ""
        for part in parts:
            if current_chunk and (len(current_chunk) + len(part) + len(separator)) > chunk_size:
                # Chunk is getting too big, recurse on the current chunk or append it
                if len(current_chunk) > chunk_size:
                    split_recursively(current_chunk, remaining_separators)
                else:
                    chunks.append(current_chunk)
                current_chunk = part
            else:
                # Add part to the current chunk
                if current_chunk:
                    current_chunk += separator
                current_chunk += part
        
        # Process the final accumulated chunk
        if current_chunk:
            if len(current_chunk) > chunk_size:
                split_recursively(current_chunk, remaining_separators)
            else:
                chunks.append(current_chunk)


    # Start the recursive splitting
    # Note: Using "\n\n" (paragraph) and "\n" (newline) as primary separators
    split_recursively(text, separators)
    
    # Final cleanup to apply overlap (simple sliding window on the resulting chunks)
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and chunk_overlap > 0:
            # Add overlap from the previous chunk's end
            overlap_text = chunks[i-1][-(chunk_overlap):]
            final_chunks.append(overlap_text + chunk)
        else:
            final_chunks.append(chunk)

    return [c.strip() for c in final_chunks if c.strip()]


def generate_questions_and_contexts(pdf_path: str):
    """Generates questions from PDF chunks using Ollama and saves to JSONL."""
    
    # 1. Extract Text
    full_text = extract_text_from_pdf(pdf_path)
    
    # 2. Split Text Manually
    separators = ["\n\n", "\n", ".", " "] # Priority of separators
    contexts = recursive_character_text_splitter(full_text, CHUNK_SIZE, CHUNK_OVERLAP, separators)
    
    print(f"Total text split into {len(contexts)} usable chunks.")
    
    qa_context_data = []

    for i, context in enumerate(contexts):
        print(f"\nProcessing chunk {i+1}/{len(contexts)}...")
        
        # Format the question generation prompt
        question_prompt = PROMPT_QUESTION_GEN.format(context=context)
        
        try:
            # Generate the question using Ollama
            # The system prompt is passed as an empty string "" since the prompt includes all instructions
            question = run_ollama(OLLAMA_MODEL_QGEN, "", question_prompt).strip()
            
            # Simple cleanup
            question = question.split('\n')[0].strip()
            if not question.endswith('?'):
                question += '?'

            # Create the data structure
            qa_context_data.append({
                "question": question,
                "context": context
            })
            print(f"  Q: {question}")
            
        except Exception as e:
            print(f"Error generating question for chunk {i}: {e}. Skipping.")
            continue
            
    # Save the results
    save_jsonl_lines(qa_context_data, OUTPUT_FILE_QA_CONTEXT)
    print(f"\n✅ Finished. Saved {len(qa_context_data)} Q/C pairs to {OUTPUT_FILE_QA_CONTEXT}")


if __name__ == "__main__":
    generate_questions_and_contexts(PDF_PATH)