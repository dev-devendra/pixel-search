import os
import base64
from datetime import datetime, timedelta
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import torch
import io
from PIL import Image

load_dotenv()

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
        print("value of k",self.k)
    
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

    def get_clip_embedding(self,content, data_type):
   
        with torch.no_grad():
            if data_type == 'image':
                # Load and preprocess the image using CLIP processor
                image = Image.open(io.BytesIO(content)).convert("RGB")
                inputs = processor(images=image, return_tensors="pt", padding=True)
                outputs = model.get_image_features(**inputs)
            elif data_type == 'text':
                # Process the text input
                inputs = processor(text=content, return_tensors="pt", padding=True, truncation=True)
                outputs = model.get_text_features(**inputs)
            else:
                raise ValueError("Unsupported data type. Use 'image' or 'text'.")
            
            # Convert the outputs to list format
            embeddings = outputs.squeeze().cpu().numpy().tolist()
            return embeddings
settings = Settings()
