# ocr_pipeline.py

import os
from ocr_utils import (
    initialize_ocr_pipeline, 
    run_ocr_on_image, 
    read_file_from_minio, 
    save_markdown_to_minio, 
    move_source_file_to_archive,
    minio_client
)
from datetime import datetime

SOURCE_BUCKET = os.getenv("MINIO_SOURCE_BUCKET_NAME", "raw-documents")

def process_minio_files():
    """Reads all files from the source bucket, runs OCR, and saves the result."""
    
    # 1. Initialize the OCR model once
    ocr_pipeline = initialize_ocr_pipeline()
    if not ocr_pipeline:
        print("Cannot proceed without OCR pipeline.")
        return

    start_time = datetime.now()
    processed_count = 0
    
    try:
        # List all objects (PDFs, JPEGs, PNGs, etc.)
        objects = minio_client.list_objects(SOURCE_BUCKET, recursive=True)
        
        for obj in objects:
            if obj.is_dir:
                continue

            file_path = obj.object_name
            print(f"--- Processing: {file_path} ---")

            # 2. Read the file content
            file_data_bytes = read_file_from_minio(SOURCE_BUCKET, file_path)
            
            if file_data_bytes:
                # 3. Run the OCR process
                # NOTE: For PDF, you would first need a library (like PyMuPDF/Pillow) 
                # to convert pages into images before running run_ocr_on_image. 
                # This example assumes a single image file for simplicity.
                markdown_result = run_ocr_on_image(file_data_bytes, ocr_pipeline)

                if markdown_result:
                    # 4. Save the Markdown output
                    if save_markdown_to_minio(markdown_result, file_path):
                        # 5. Archive the original file
                        move_source_file_to_archive(SOURCE_BUCKET, file_path)
                        processed_count += 1
            
            print("------------------------------------")
            
    except Exception as e:
        print(f"An error occurred during file listing/iteration: {e}")
        
    end_time = datetime.now()
    print(f"Finished processing {processed_count} files.")
    print(f"Total time taken: {end_time - start_time}")


if __name__ == "__main__":
    process_minio_files()