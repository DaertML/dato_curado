import os
import torch
import tempfile
import sys
from pathlib import Path
import shutil
from PIL import Image
import io
from contextlib import redirect_stdout
# --- PDF RENDERING LIBRARY ---
# You must install this: pip install pdf2image Pillow
from pdf2image import convert_from_path 

# Hugging Face Imports
from transformers import AutoModel, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = '/media/pc/easystore1/hf_llms/deepseek-ocr' 
# Hardcoded image file name for local script test, matching the user's example
IMAGE_FILE_DEFAULT = 'output.png' 
# Hardcoded default prompt, matching one of the user's examples
PROMPT_DEFAULT = "<image>\n<|grounding|>Convert the document to markdown. "

# Global model objects
tokenizer = None
model = None
device = 'cpu'

# Directory configuration
SOURCE_DIR = "ocr_source"
OUTPUT_DIR = "ocr_output"
PROCESSED_DIR = "ocr_processed"

# --- SYNCHRONOUS MODEL LOADING ---

def load_model_and_tokenizer():
    """
    Synchronously loads the tokenizer and model, adapting the logic 
    from the previous async lifespan manager.
    """
    global tokenizer, model, device
    try:
        # 1. Determine Device
        # Note: If you are using Windows/WSL, you might need to adjust the CUDA check
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # 2. Load Tokenizer and Model
        print(f"Loading tokenizer: {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=True)
        print(f"Loading model: {MODEL_ID}...")
        
        load_kwargs = {
            "trust_remote_code": True, 
            "use_safetensors": True,
        }
        
        # Enable flash attention if CUDA is available, as requested in user's initial example
        if device.startswith('cuda'):
            load_kwargs["_attn_implementation"] = 'flash_attention_2'
            
        model = AutoModel.from_pretrained(MODEL_ID, **load_kwargs)
        #model.to('cuda')
        
        # 3. Move model to device and set dtype (bfloat16 for CUDA, float32 otherwise)
        dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
        
        model = model.eval().to(device).to(dtype)
        
        print(f"Model loaded successfully to {device} with dtype {dtype}.")
        
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        tokenizer, model = None, None
        sys.exit(1) # Exit the script if loading fails

# --- PDF RENDERING UTILITY ---

def pdf_to_images(pdf_path: str) -> list[Path]:
    """
    Renders all pages of a PDF file into temporary PNG images.
    Returns a list of Path objects for the temporary image files.
    """
    temp_dir = Path(tempfile.mkdtemp())
    try:
        print(f"Rendering PDF pages from '{pdf_path}' to temporary images...")
        
        # Use convert_from_path to get a list of PIL Image objects
        # dpi=300 is a good standard for OCR quality
        images = convert_from_path(pdf_path, dpi=300)
        
        image_paths = []
        for i, pil_image in enumerate(images):
            # Save the PIL Image to a temporary PNG file
            temp_image_path = temp_dir / f"page_{i+1}.png"
            pil_image.save(temp_image_path, 'PNG')
            image_paths.append(temp_image_path)
            
        print(f"Successfully rendered {len(image_paths)} pages.")
        return image_paths

    except Exception as e:
        print(f"Error rendering PDF '{pdf_path}': {e}")
        # Clean up the temp dir if an error occurs
        for file in temp_dir.glob('*'):
            os.remove(file)
        os.rmdir(temp_dir)
        return []

# --- SYNCHRONOUS INFERENCE FUNCTION ---

def run_inference(
    image_file: str, 
    prompt: str, 
    base_size: int = 1024, 
    image_size: int = 640, 
    crop_mode: bool = True
):
    """
    Runs DeepSeek-OCR VLM inference, capturing standard output 
    if the function returns None.
    """
    
    if model is None or tokenizer is None:
        print("Error: Model or tokenizer is not loaded. Cannot run inference.")
        return None

    source_path = Path(image_file)
    if not source_path.exists():
        print(f"Error: File not found at '{image_file}'. Cannot run inference.")
        return None

    # --- PDF RENDERING LOGIC (omitted for brevity, remains the same) ---
    file_extension = source_path.suffix.lower()
    
    if file_extension == '.pdf':
        image_paths = pdf_to_images(str(source_path))
        if not image_paths:
            print("Skipping PDF file due to rendering failure.")
            return None
    else:
        image_paths = [source_path]

    all_results = []
    
    # --- LOOP OVER IMAGE PAGES ---
    for i, image_path in enumerate(image_paths):
        page_info = f" (Page {i+1}/{len(image_paths)})" if file_extension == '.pdf' else ""
        print(f"\n--- Running Inference on {image_path.name}{page_info} ---")
        print(f"Prompt: {prompt}")

        result_to_append = ""
        
        try:
            output_path = tempfile.gettempdir()
            
            # 1. Prepare to capture stdout
            f = io.StringIO()
            
            with redirect_stdout(f):
                # 2. Call model.infer while redirecting all print statements to 'f'
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
            
            # 3. Get the captured output
            captured_output = f.getvalue().strip()
            
            # 4. Determine the final result to append
            if result_text is None and captured_output:
                # Use the captured output if model returns None but printed text
                result_to_append = captured_output
            elif result_text is not None and isinstance(result_text, str):
                # Use the returned value if it's a valid string
                result_to_append = result_text
            else:
                # Handle total failure or empty output
                if captured_output:
                    # If model returned None/empty but there was noise in stdout
                    result_to_append = f"INFERENCE_FAILED: Model returned None, captured noise:\n{captured_output}"
                else:
                    result_to_append = f"INFERENCE_FAILED: Model returned None and captured no output for {image_path.name}"
            
        
        except Exception as e:
            error_message = f"INFERENCE ERROR: {e}"
            print(f"Error during VLM inference for {image_path.name}: {error_message}")
            result_to_append = error_message
        
        # Now append the determined result
        all_results.append(result_to_append)

        # Print what is actually being saved (for terminal verification)
        print(f"\n--- Model Response Appended for {image_path.name} ---")
        print(result_to_append)
        print("---------------------------------------------")
            
    # --- CLEANUP TEMPORARY PDF IMAGES (omitted for brevity, remains the same) ---
    if file_extension == '.pdf' and image_paths:
        temp_dir = image_paths[0].parent
        print(f"\nCleaning up temporary image directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory {temp_dir}: {e}")
        
    return "\n\n--- PAGE BREAK ---\n\n".join(all_results)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    print("--- Starting Local DeepSeek-OCR Script ---")
    
    # 1. Ensure all necessary directories exist 📁
    for directory in [SOURCE_DIR, OUTPUT_DIR, PROCESSED_DIR]:
        Path(directory).mkdir(exist_ok=True)
        print(f"Checked directory: {directory}")
    
    # 2. Load the model and tokenizer
    load_model_and_tokenizer()

    prompt_to_use = PROMPT_DEFAULT
    
    # Example Parameters 
    base_size_param = 1024
    image_size_param = 640
    crop_mode_param = True 

    # 3. Process files from the source directory
    for item_name in os.listdir(SOURCE_DIR):
        source_path = Path(SOURCE_DIR) / item_name
        
        # Only process files
        if source_path.is_file():
            print(f"\n### Processing File: {item_name} ###")
            
            if model and tokenizer:
                # 4. Run the inference
                markdown_output = run_inference(
                    image_file=str(source_path), # Pass as string
                    prompt=prompt_to_use,
                    base_size=base_size_param,
                    image_size=image_size_param,
                    crop_mode=crop_mode_param
                )

                if markdown_output:
                    # Determine the base name for the output file (remove original extension)
                    base_name = source_path.stem
                    output_file_name = f"{base_name}.md"
                    output_path = Path(OUTPUT_DIR) / output_file_name
                    
                    # 5. Save the output to the ocr_output folder 💾
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(markdown_output)
                        print(f"✅ Successfully saved output to: {output_path}")
                        
                        # 6. Move the source file to the ocr_processed folder ➡️
                        destination_path = Path(PROCESSED_DIR) / item_name
                        shutil.move(source_path, destination_path)
                        print(f"✅ Successfully moved source file to: {destination_path}")
                        
                    except Exception as e:
                        print(f"❌ ERROR in file operations for {item_name}: {e}")
                else:
                    print(f"⚠️ Skipping file operations for {item_name} because no output was returned.")
            else:
                print("Script terminated due to model loading failure.")