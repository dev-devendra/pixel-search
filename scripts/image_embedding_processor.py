import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torchvision import transforms
import os
import uuid
from datetime import datetime
import time
from PIL import Image
from pinecone import Pinecone
from google.cloud import storage
from dotenv import load_dotenv
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

# Load environment variables from .env file
load_dotenv()

REGION = 'us-central1'
FILE_TYPE = 'image'
DIMENSION = 512  # CLIP output dimension for ViT-B/32

# Initialize CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()

def process_image(image_file, bucket_name, prefix, model, processor, index, file_path, image_index, total_images, max_retries=5):
    gcs_uri = f'gs://{bucket_name}/{prefix}/{image_file}'
    file_path = f'{bucket_name}/{prefix}/'

    attempt = 0
    while attempt < max_retries:
        try:
            # Load image from GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"{prefix}/{image_file}")
            image_data = blob.download_as_bytes()

            # Open image using PIL
            image = Image.open(BytesIO(image_data)).convert("RGB")

            # Preprocess image and generate embeddings using CLIP
            inputs = processor(images=image, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = model.get_image_features(**inputs)

            embeddings = outputs.squeeze().detach().cpu().numpy()  # Convert embeddings to numpy array

            print(f"Received embeddings for: {image_file} ({image_index}/{total_images})")

            # Prepare metadata
            date_added = datetime.now().isoformat()
            embedding_id = str(uuid.uuid4())

            vector = [
                {
                    'id': embedding_id,
                    'values': embeddings.tolist(),
                    'metadata': {
                        'date_added': date_added,
                        'file_type': FILE_TYPE,
                        'gcs_file_path': file_path,
                        'gcs_file_name': image_file,
                    }
                }
            ]
            index.upsert(vectors=vector)  # Upsert to Pinecone
            print(f"Processed and upserted: {image_file} ({image_index}/{total_images})")
            break  # Exit loop if successful
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")
            attempt += 1
            if attempt < max_retries:
                wait_time = 5 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process file {image_file} after {max_retries} attempts.")

def main(gc_project_id, gcs_bucket_name, gcs_folder_name, pinecone_index_name):
    api_key = os.getenv('PINECONE_API_KEY')  # Pinecone API key
    file_path = f'{gcs_bucket_name}/{gcs_folder_name}/'  # GCS bucket path

    google_credentials_base64 = os.getenv('GOOGLE_CREDENTIALS_BASE64')
    credentials_path = '/tmp/google-credentials.json'

    if google_credentials_base64:
        google_credentials = base64.b64decode(google_credentials_base64).decode('utf-8')
        with open(credentials_path, 'w') as f:
            f.write(google_credentials)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Initialize Pinecone
    pc = Pinecone(api_key=api_key, source_tag="pinecone:stl_sample_app")
    index = pc.Index(pinecone_index_name)

    # Initialize Google Cloud Storage Client
    client = storage.Client()
    blobs = client.list_blobs(gcs_bucket_name, prefix=gcs_folder_name)
    # Filter image formats: jpg, jpeg, png, bmp, webp
    image_files = [blob.name.replace(gcs_folder_name + '/', '') for blob in blobs if blob.name.lower().endswith(('jpeg', 'jpg', 'png', 'bmp', 'webp'))]

    total_images = len(image_files)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_image, 
                image_file, 
                gcs_bucket_name, 
                gcs_folder_name, 
                model, 
                processor, 
                index, 
                file_path, 
                i + 1, 
                total_images
            ) for i, image_file in enumerate(image_files)
        ]
        for future in as_completed(futures):
            future.result()  # This will re-raise any exceptions that occurred during processing

if __name__ == '__main__':
    gc_project_id = os.getenv("GC_PROJECT_ID")
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    gcs_folder_name = os.getenv("GCS_FOLDER_NAME")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    main(gc_project_id, gcs_bucket_name, gcs_folder_name, pinecone_index_name)