import os
import torch
import tempfile
import sys
from pathlib import Path
import shutil
from PIL import Image
import io
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
# --- PDF RENDERING LIBRARY ---
from pdf2image import convert_from_path 

# Hugging Face Imports
from transformers import AutoModel, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = '/media/pc/easystore1/hf_llms/deepseek-ocr' 
PROMPT_DEFAULT = "<image>\n<|grounding|>Convert the document to markdown. "

# Global model objects
tokenizer = None
model = None
device = 'cpu'

# Directory configuration
SOURCE_DIR = "ocr_source"
OUTPUT_DIR = "ocr_output"
PROCESSED_DIR = "ocr_processed"

# Batch processing configuration
BATCH_SIZE = 4  # Number of pages to process concurrently
MAX_WORKERS = 4  # Maximum number of parallel threads

# --- SYNCHRONOUS MODEL LOADING ---

def load_model_and_tokenizer():
    """
    Synchronously loads the tokenizer and model.
    """
    global tokenizer, model, device
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        print(f"Loading tokenizer: {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=True)
        print(f"Loading model: {MODEL_ID}...")
        
        load_kwargs = {
            "trust_remote_code": True, 
            "use_safetensors": True,
        }
        
        if device.startswith('cuda'):
            load_kwargs["_attn_implementation"] = 'flash_attention_2'
            
        model = AutoModel.from_pretrained(MODEL_ID, **load_kwargs)
        
        dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
        model = model.eval().to(device).to(dtype)
        
        print(f"Model loaded successfully to {device} with dtype {dtype}.")
        
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        tokenizer, model = None, None
        sys.exit(1)

# --- PDF RENDERING UTILITY ---

def pdf_to_images(pdf_path: str) -> Tuple[List[Path], Path]:
    """
    Renders all pages of a PDF file into temporary PNG images.
    Returns a tuple of (list of image paths, temp directory path).
    """
    temp_dir = Path(tempfile.mkdtemp())
    try:
        print(f"Rendering PDF pages from '{pdf_path}' to temporary images...")
        
        images = convert_from_path(pdf_path, dpi=300)
        
        image_paths = []
        for i, pil_image in enumerate(images):
            temp_image_path = temp_dir / f"page_{i+1:04d}.png"
            pil_image.save(temp_image_path, 'PNG')
            image_paths.append(temp_image_path)
            
        print(f"Successfully rendered {len(image_paths)} pages.")
        return image_paths, temp_dir

    except Exception as e:
        print(f"Error rendering PDF '{pdf_path}': {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [], None

# --- SINGLE PAGE INFERENCE ---

def process_single_page(
    image_path: Path,
    page_num: int,
    total_pages: int,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool
) -> Tuple[int, str]:
    """
    Process a single page and return (page_number, markdown_text).
    """
    page_info = f" (Page {page_num}/{total_pages})" if total_pages > 1 else ""
    print(f"\n--- Processing {image_path.name}{page_info} ---")

    if model is None or tokenizer is None:
        return (page_num, f"ERROR: Model not loaded for page {page_num}")

    try:
        output_path = tempfile.gettempdir()
        f = io.StringIO()
        
        with redirect_stdout(f):
            result_text = model.infer(
                tokenizer, 
                prompt=prompt, 
                image_file=str(image_path), 
                output_path=output_path,
                base_size=base_size, 
                image_size=image_size, 
                crop_mode=crop_mode, 
                save_results=False, 
                test_compress=False 
            )
        
        captured_output = f.getvalue().strip()
        
        if result_text is None and captured_output:
            result_to_return = captured_output
        elif result_text is not None and isinstance(result_text, str):
            result_to_return = result_text
        else:
            if captured_output:
                result_to_return = f"INFERENCE_FAILED: Model returned None, captured:\n{captured_output}"
            else:
                result_to_return = f"INFERENCE_FAILED: No output for {image_path.name}"
        
        print(f"✅ Completed page {page_num}")
        return (page_num, result_to_return)
    
    except Exception as e:
        error_message = f"INFERENCE ERROR on page {page_num}: {e}"
        print(f"❌ {error_message}")
        return (page_num, error_message)

# --- BATCH INFERENCE WITH STREAMING OUTPUT ---

def run_inference_with_streaming(
    image_file: str, 
    prompt: str, 
    base_size: int = 1024, 
    image_size: int = 640, 
    crop_mode: bool = True,
    output_path: Path = None
):
    """
    Runs inference on all pages with batch processing and streams results to file immediately.
    """
    
    if model is None or tokenizer is None:
        print("Error: Model or tokenizer is not loaded.")
        return False

    source_path = Path(image_file)
    if not source_path.exists():
        print(f"Error: File not found at '{image_file}'.")
        return False

    file_extension = source_path.suffix.lower()
    temp_dir = None
    
    # Determine if we're processing a PDF or single image
    if file_extension == '.pdf':
        image_paths, temp_dir = pdf_to_images(str(source_path))
        if not image_paths:
            print("Skipping PDF file due to rendering failure.")
            return False
    else:
        image_paths = [source_path]

    total_pages = len(image_paths)
    print(f"\n📄 Processing {total_pages} page(s) from {source_path.name}")
    
    # Open output file for streaming writes
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            # Process pages in batches
            for batch_start in range(0, total_pages, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_pages)
                batch_pages = image_paths[batch_start:batch_end]
                
                print(f"\n🔄 Processing batch: pages {batch_start + 1} to {batch_end}")
                
                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch_pages))) as executor:
                    # Submit all pages in the batch
                    future_to_page = {
                        executor.submit(
                            process_single_page,
                            img_path,
                            batch_start + idx + 1,
                            total_pages,
                            prompt,
                            base_size,
                            image_size,
                            crop_mode
                        ): (batch_start + idx + 1, img_path)
                        for idx, img_path in enumerate(batch_pages)
                    }
                    
                    # Collect results as they complete
                    results = []
                    for future in as_completed(future_to_page):
                        page_num, img_path = future_to_page[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            print(f"❌ Exception processing page {page_num}: {e}")
                            results.append((page_num, f"EXCEPTION: {e}"))
                    
                    # Sort results by page number to maintain order
                    results.sort(key=lambda x: x[0])
                    
                    # Write results immediately to file
                    for page_num, markdown_text in results:
                        if batch_start > 0 or page_num > 1:
                            output_file.write("\n\n--- PAGE BREAK ---\n\n")
                        output_file.write(markdown_text)
                        output_file.flush()  # Ensure immediate write to disk
                        print(f"💾 Written page {page_num} to {output_path.name}")
        
        print(f"\n✅ All pages written to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error writing to output file: {e}")
        return False
    
    finally:
        # Cleanup temporary PDF images
        if temp_dir and temp_dir.exists():
            print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Warning: Failed to clean up temp directory: {e}")
    
    return True

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    print("=" * 60)
    print("🚀 DeepSeek-OCR with Streaming & Batch Processing")
    print("=" * 60)
    
    # 1. Ensure all necessary directories exist
    for directory in [SOURCE_DIR, OUTPUT_DIR, PROCESSED_DIR]:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Checked directory: {directory}")
    
    # 2. Load the model and tokenizer
    load_model_and_tokenizer()

    prompt_to_use = PROMPT_DEFAULT
    base_size_param = 1024
    image_size_param = 640
    crop_mode_param = True 

    # 3. Process files from the source directory
    source_files = sorted([f for f in os.listdir(SOURCE_DIR) if Path(SOURCE_DIR, f).is_file()])
    
    if not source_files:
        print("\n⚠️ No files found in source directory.")
    
    for item_name in source_files:
        source_path = Path(SOURCE_DIR) / item_name
        
        print("\n" + "=" * 60)
        print(f"📝 Processing File: {item_name}")
        print("=" * 60)
        
        if model and tokenizer:
            # Determine output file path
            base_name = source_path.stem
            output_file_name = f"{base_name}.md"
            output_path = Path(OUTPUT_DIR) / output_file_name
            
            # 4. Run inference with streaming output
            success = run_inference_with_streaming(
                image_file=str(source_path),
                prompt=prompt_to_use,
                base_size=base_size_param,
                image_size=image_size_param,
                crop_mode=crop_mode_param,
                output_path=output_path
            )

            if success:
                # 5. Move the source file to the processed folder
                try:
                    destination_path = Path(PROCESSED_DIR) / item_name
                    shutil.move(source_path, destination_path)
                    print(f"✅ Moved source file to: {destination_path}")
                except Exception as e:
                    print(f"❌ ERROR moving file {item_name}: {e}")
            else:
                print(f"⚠️ Skipping file operations for {item_name} due to processing failure.")
        else:
            print("❌ Script terminated due to model loading failure.")
            break
    
    print("\n" + "=" * 60)
    print("🎉 Processing Complete!")
    print("=" * 60)