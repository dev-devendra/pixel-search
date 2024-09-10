import os
import base64
from datetime import datetime, timedelta
import uuid
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import torch
import io
from PIL import Image
import cv2 
import numpy as np

load_dotenv()

# Video embedding settings
INTERVAL_SEC = 15
START_OFFSET_SEC = 0
END_OFFSET_SEC = 120

# Initialize CLIP model and processor
model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()
class Settings:
    def __init__(self):
        # Google services
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
        self.location = os.getenv('GOOGLE_CLOUD_PROJECT_LOCATION')
        self.gcs_bucket_name = os.getenv('GOOGLE_CLOUD_STORAGE_BUCKET_NAME')
        self.google_credentials_base64 = os.getenv('GOOGLE_CREDENTIALS_BASE64')
        self.credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or '/tmp/google-credentials.json'
        self.access_token = None
        self.token_expiry = None
        self.credentials = None

        # Pinecone services
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
        self.k = int(os.getenv('PINECONE_TOP_K')) 
    
    def get_credentials(self):
        if self.credentials:
            return self.credentials

        try:
            # Case 1: Using GOOGLE_CREDENTIALS_BASE64
            if self.google_credentials_base64:
                print("Loading Google credentials from GOOGLE_CREDENTIALS_BASE64")
                google_credentials = base64.b64decode(self.google_credentials_base64).decode('utf-8')
                with open(self.credentials_path, 'w') as f:
                    f.write(google_credentials)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
                credential_source = "GOOGLE_CREDENTIALS_BASE64"

            # Case 2: Using existing service account JSON file
            elif os.path.exists(self.credentials_path):
                credential_source = "GOOGLE_APPLICATION_CREDENTIALS file"

            # Case 3: No credentials available
            else:
                raise ValueError("Google credentials not found. Please set GOOGLE_CREDENTIALS_BASE64 or ensure GOOGLE_APPLICATION_CREDENTIALS points to a valid file.")

            # Load credentials from the file (works for both Case 1 and Case 2)
            self.credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            print(f"Successfully loaded Google credentials from {credential_source}")
            return self.credentials

        except Exception as e:
            error_message = (
                f"Failed to load Google service account credentials: {str(e)}\n"
                f"Attempted to load credentials from: {credential_source}\n"
                "Please ensure you have set up a Google Cloud service account correctly.\n"
                "For instructions on setting up a service account, visit:\n"
                "https://cloud.google.com/docs/authentication/getting-started"
            )
            raise ValueError(error_message) from e

    def get_access_token(self):
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.access_token

        try:
            credentials = self.get_credentials()
            credentials.refresh(Request())
            self.access_token = credentials.token
            self.token_expiry = datetime.now() + timedelta(hours=1)
            print("Access token refreshed", self.access_token)
            return self.access_token
        except Exception as e:
            print(f"Error getting access token: {str(e)}")
            return None
        
    def extract_frames_from_video(self, video_path, interval_sec, start_offset_sec, end_offset_sec):
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

    def get_clip_embedding(self, content, data_type):
        with torch.no_grad():
            if data_type == 'image':
                # Load and preprocess the image using CLIP processor
                image = Image.open(io.BytesIO(content)).convert("RGB")
                inputs = processor(images=image, return_tensors="pt", padding=True)
                outputs = model.get_image_features(**inputs)
                embeddings = outputs.squeeze().cpu().numpy().tolist()
            
            elif data_type == 'video':
                # Decode the base64 string to raw bytes
                try:
                    video_bytes = base64.b64decode(content)  # Decode to bytes directly
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 content: {e}")

                temp_video_path = f'/tmp/{uuid.uuid4()}.mp4'
                try:
                    with open(temp_video_path, 'wb') as f:
                        f.write(video_bytes)
                except Exception as e:
                    print(f"Failed to write video to temporary file: {e}")
                    raise
                
                # Extract frames from the video
                frames = self.extract_frames_from_video(temp_video_path, INTERVAL_SEC, START_OFFSET_SEC, END_OFFSET_SEC)
                
                # Check if frames were extracted
                if not frames:
                    os.remove(temp_video_path)  # Clean up even if extraction fails
                    raise ValueError("No frames extracted from the video within the specified time range.")
                
                # Generate embeddings for each extracted frame
                frame_embeddings = []
                for frame in frames:
                    # Convert frame to RGB PIL Image format
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
                    inputs = processor(images=image, return_tensors="pt", padding=True)
                    outputs = model.get_image_features(**inputs)
                    frame_embedding = outputs.squeeze().cpu().numpy()
                    frame_embeddings.append(frame_embedding)
                
                # Use the mean of all frame embeddings to represent the video
                video_embedding = np.mean(frame_embeddings, axis=0).tolist()
                
                # Clean up temporary video file
                os.remove(temp_video_path)

            elif data_type == 'text':
                # Process the text input
                inputs = processor(text=content, return_tensors="pt", padding=True, truncation=True)
                outputs = model.get_text_features(**inputs)
                embeddings = outputs.squeeze().cpu().numpy().tolist()

            else:
                raise ValueError("Unsupported data type. Use 'image', 'video', or 'text'.")

            return video_embedding if data_type == 'video' else embeddings
settings = Settings()
