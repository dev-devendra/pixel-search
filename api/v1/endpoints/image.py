import base64
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.config import settings
from api import deps
from PIL import Image


router = APIRouter()

@router.post("/search/image")
async def query_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()

        # Open the image to determine its format
        with Image.open(io.BytesIO(contents)) as img:
            file_format = img.format.lower()
        
        # Check for supported image formats
        if file_format not in ['bmp', 'gif', 'jpeg', 'png', 'jpg']:
            raise HTTPException(status_code=400, detail="We only support BMP, GIF, JPG, JPEG, and PNG for images. Please upload a valid image file.")
        
        # Load and preprocess the image using CLIP processor
        embeddings = settings.get_clip_embedding(contents, 'image')


        # Query Pinecone DB using the generated embeddings
        query_response = deps.index.query(
            vector=embeddings,
            top_k=settings.k,
            include_metadata=True
        )

        matches = query_response['matches']
        results = [{
            "score": match['score'],
            "metadata": {
                "gcs_file_name": match['metadata'].get('gcs_file_name'),
                "gcs_file_path": match['metadata'].get('gcs_file_path'),
                "gcs_public_url": f"https://storage.googleapis.com/{match['metadata'].get('gcs_file_path')}{match['metadata'].get('gcs_file_name')}",
                "file_type": match['metadata'].get('file_type'),
                "segment": match['metadata'].get('segment'),
                "start_offset_sec": match['metadata'].get('start_offset_sec'),
                "end_offset_sec": match['metadata'].get('end_offset_sec'),
                "interval_sec": match['metadata'].get('interval_sec'),
                "tags": match['metadata'].get('tags'),
            }
        } for match in matches]
        
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
