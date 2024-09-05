"""
Video Embedding Generator and Upserter

This script processes videos from a Google Cloud Storage (GCS) bucket, generates embeddings using Vertex AI, and upserts them to a Pinecone index.

Requirements:
- Python 3.7+
- Google Cloud SDK
- Pinecone account
- Vertex AI API enabled in your Google Cloud project

Required Python packages:
- vertexai
- google-cloud-storage
- pinecone-client

Usage:
1. Set up environment variables (see below)
2. Run the script with the required arguments

Example:
python video_embedding_processor.py -p your-gc-project-id -b your-gcs-bucket-name -f your-gcs-folder-name -i your-pinecone-index-name
"""

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
import cv2  # OpenCV for video frame extraction
from transformers import CLIPProcessor, CLIPModel

# Load environment variables from .env file
load_dotenv()

REGION = 'us-central1'
FILE_TYPE = 'video'
MAX_RETRIES = 5
SUPPORTED_VIDEO_FORMATS = ('mov', 'mp4', 'avi', 'flv', 'mkv', 'mpeg', 'mpg', 'webm', 'wmv')

# Video embedding settings
INTERVAL_SEC = 15
START_OFFSET_SEC = 0
END_OFFSET_SEC = 120

# Initialize CLIP model and processor
model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()

class VideoSegmentConfig:
    def __init__(self, start_offset_sec=None, end_offset_sec=None, interval_sec=None):
        self.start_offset_sec = start_offset_sec
        self.end_offset_sec = end_offset_sec
        self.interval_sec = interval_sec

def setup_google_credentials():
    """Set up Google Cloud credentials from base64-encoded environment variable."""
    google_credentials_base64 = os.getenv('GOOGLE_CREDENTIALS_BASE64')
    if google_credentials_base64:
        credentials_path = '/tmp/google-credentials.json'
        google_credentials = base64.b64decode(google_credentials_base64).decode('utf-8')
        with open(credentials_path, 'w') as f:
            f.write(google_credentials)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    else:
        print("Warning: GOOGLE_CREDENTIALS_BASE64 environment variable not set.")

def extract_frames_from_video(video_path, interval_sec, start_offset_sec, end_offset_sec):
    """Extract frames from a video file at specified intervals."""
    frames = []
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    interval_frames = int(fps * interval_sec)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(fps * start_offset_sec)
    end_frame = min(int(fps * end_offset_sec), total_frames)

    current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % interval_frames == 0:
            frames.append(frame)
        current_frame += 1

    cap.release()
    return frames

def process_video(video_file, bucket_name, prefix, model, processor, index, file_path, video_index, total_videos):
    """Process a single video file, generate embeddings, and upsert to Pinecone."""
    gcs_uri = f'gs://{bucket_name}/{prefix}/{video_file}'

    for attempt in range(MAX_RETRIES):
        try:
            # Load video from GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"{prefix}/{video_file}")
            video_data = blob.download_as_bytes()

            # Save video data temporarily to disk to be processed by OpenCV
            temp_video_path = f'/tmp/{uuid.uuid4()}.mp4'
            with open(temp_video_path, 'wb') as f:
                f.write(video_data)

            # Extract frames from video
            frames = extract_frames_from_video(temp_video_path, INTERVAL_SEC, START_OFFSET_SEC, END_OFFSET_SEC)

            # Process each frame using CLIP model to generate embeddings
            for i, frame in enumerate(frames):
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
                inputs = processor(images=image, return_tensors="pt", padding=True)

                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)

                embeddings = outputs.squeeze().detach().cpu().numpy().tolist()

                # Prepare metadata and upsert to Pinecone
                vector = [{
                    'id': str(uuid.uuid4()),
                    'values': embeddings,
                    'metadata': {
                        'date_added': datetime.now().isoformat(),
                        'file_type': FILE_TYPE,
                        'gcs_file_path': file_path,
                        'gcs_file_name': video_file,
                        'segment': i,
                        'start_offset_sec': START_OFFSET_SEC + i * INTERVAL_SEC,
                        'end_offset_sec': min(START_OFFSET_SEC + (i + 1) * INTERVAL_SEC, END_OFFSET_SEC),
                        'interval_sec': INTERVAL_SEC,
                    }
                }]
                index.upsert(vector)

            print(f"Processed and upserted: {video_file} ({video_index}/{total_videos})")
            return  # Exit function if successful
        except Exception as e:
            print(f"Error processing file {video_file}: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = 5 ** (attempt + 1)  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process file {video_file} after {MAX_RETRIES} attempts.")

def main(gc_project_id, gcs_bucket_name, gcs_folder_name, pinecone_index_name):
    """Main function to process videos from GCS and upsert embeddings to Pinecone."""
    setup_google_credentials()

    # Initialize Pinecone
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")
    pc = Pinecone(api_key=api_key, source_tag="pinecone:stl_sample_app")
    index = pc.Index(pinecone_index_name)

    # List video files in GCS bucket
    client = storage.Client()
    blobs = client.list_blobs(gcs_bucket_name, prefix=gcs_folder_name)
    video_files = [
        blob.name.replace(f"{gcs_folder_name}/", "")
        for blob in blobs
        if blob.name.lower().endswith(SUPPORTED_VIDEO_FORMATS)
    ]

    # Process videos in parallel
    total_videos = len(video_files)
    file_path = f'{gcs_bucket_name}/{gcs_folder_name}/'
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_video,
                video_file,
                gcs_bucket_name,
                gcs_folder_name,
                model,
                processor,
                index,
                file_path,
                i + 1,
                total_videos
            )
            for i, video_file in enumerate(video_files)
        ]
        for future in as_completed(futures):
            future.result()  # This will re-raise any exceptions that occurred during processing

if __name__ == '__main__':

    gc_project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    gcs_bucket_name = os.getenv("GOOGLE_CLOUD_STORAGE_VIDEO_BUCKET_NAME")
    gcs_folder_name = "videos"
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    main(gc_project_id, gcs_bucket_name, gcs_folder_name, pinecone_index_name)

"""
Setup Instructions:

1. Environment Variables:
   Set the following environment variables before running the script:

   a. GOOGLE_CREDENTIALS_BASE64
      Base64-encoded Google Cloud service account key JSON.
      To set this:
      - Get your service account key JSON file
      - Encode it to base64:
        $ base64 -i path/to/your/service-account-key.json | tr -d '\n'
      - Set the environment variable:
        $ export GOOGLE_CREDENTIALS_BASE64="<base64-encoded-string>"

   b. PINECONE_API_KEY
      Your Pinecone API key.
      $ export PINECONE_API_KEY="your-pinecone-api-key"

2. Install required Python packages:
   $ pip install vertexai google-cloud-storage pinecone-client

3. Run the script:
   $ python video_embedding_processor.py -p your-gc-project-id -b your-gcs-bucket-name -f your-gcs-folder-name -i your-pinecone-index-name

   Replace the placeholders with your actual values:
   - your-gc-project-id: Your Google Cloud project ID
   - your-gcs-bucket-name: The name of your GCS bucket containing the videos
   - your-gcs-folder-name: The folder name within the bucket where videos are stored
   - your-pinecone-index-name: The name of your Pinecone index

Example command:
$ python video_embedding_processor.py -p my-gcp-project -b my-video-bucket -f processed-videos -i my-pinecone-index

Notes:
- Ensure that your Google Cloud service account has the necessary permissions to access the GCS bucket and use Vertex AI.
- The script supports the following video formats: AVI, FLV, MKV, MOV, MP4, MPEG, MPG, WEBM, and WMV.
- The script uses exponential backoff for retrying failed operations, with a maximum of 5 attempts per video.
- Video embedding settings (INTERVAL_SEC, START_OFFSET_SEC, END_OFFSET_SEC) can be adjusted at the top of the script.
"""