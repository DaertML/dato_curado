from dotenv import load_dotenv
load_dotenv()

import os
import io
from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource

# For document loading (PDF/Image)
# NOTE: Using standard libraries here, as external loaders need installation
from PIL import Image

# For OCR pipeline
from transformers import pipeline

# --- Configuration (using environment variables for MINIO) ---
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

# DeepSeek-OCR model (placeholder, use the correct model name)
OCR_MODEL = "deepseek-ai/DeepSeek-OCR-base" 

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def initialize_ocr_pipeline():
    """Initializes the OCR model pipeline using Hugging Face."""
    # The actual implementation of a deepseek OCR pipeline might be more complex
    # (e.g., a custom processing function), but this demonstrates the concept.
    try:
        # A standard text-recognition/visual-question-answering pipeline
        ocr_pipeline = pipeline(
            "image-to-text", 
            model=OCR_MODEL, 
            device=0 # Use 'cpu' or specify a GPU index
        )
        return ocr_pipeline
    except Exception as e:
        print(f"Error initializing OCR pipeline: {e}")
        return None

def run_ocr_on_image(image_path_or_file, ocr_pipeline):
    """Runs the DeepSeek OCR pipeline on a single image and returns Markdown text."""
    try:
        # Load the image
        if isinstance(image_path_or_file, str):
            image = Image.open(image_path_or_file)
        else: # Assumes it's a file-like object (e.g., from MinIO)
            image = Image.open(image_path_or_file)
        
        print("Running OCR inference...")
        
        # Run inference (This is a simplified call; DeepSeek OCR often needs specific prompts)
        # Assuming the pipeline is configured for document analysis and outputting text.
        ocr_result = ocr_pipeline(image)[0]['generated_text']
        
        # Format the result as simple Markdown
        markdown_output = f"# Document Content\n\n{ocr_result}\n\n---\n\n"
        
        return markdown_output
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None

def read_file_from_minio(bucket_name, file_path):
    """Reads a file (image or PDF) from MinIO."""
    try:
        response = minio_client.get_object(bucket_name, file_path)
        # We return the BytesIO object which is a file-like object
        file_data = io.BytesIO(response.read())
        response.close()
        response.release_conn()
        return file_data
    except S3Error as e:
        print(f"Error reading {file_path} from MinIO: {e}")
        return None

def save_markdown_to_minio(markdown_content, source_file_path):
    """Saves the OCR output as a Markdown file in the 'processed' bucket."""
    
    # Example: 'data/document.pdf' -> 'processed/document.md'
    filename = os.path.basename(source_file_path)
    # Replace the extension with .md
    markdown_filename = os.path.splitext(filename)[0] + ".md"
    
    destination_bucket = os.getenv("MINIO_PROCESSED_BUCKET_NAME", "processed-ocr")
    file_path = f"markdown_output/{markdown_filename}"
    file_content = markdown_content.encode('utf-8')
    file_data = io.BytesIO(file_content)
    
    try:
        minio_client.put_object(
            destination_bucket,
            file_path,
            file_data,
            length=len(file_content),
            content_type="text/markdown"
        )
        print(f"Saved {file_path} to MinIO bucket {destination_bucket}")
        return True
    except S3Error as e:
        print(f"Error occurred while saving to MinIO: {e}")
        return False

# Re-using your MinIO file movement idea for cleanup
def move_source_file_to_archive(source_bucket, source_file_path):
    """Moves the original file to an archive bucket after processing."""
    
    archive_bucket = os.getenv("MINIO_ARCHIVE_BUCKET_NAME", "archive")
    
    try:
        # 1. Copy to archive
        minio_client.copy_object(
          archive_bucket,
          source_file_path,
          CopySource(
            source_bucket,
            source_file_path,
          ),
        )
        print(f"Copied {source_file_path} to {archive_bucket}")

        # 2. Remove from source
        minio_client.remove_object(source_bucket, source_file_path)
        print(f"Removed {source_file_path} from {source_bucket}")

    except S3Error as e:
        print(f"Error occurred during file move: {e}")