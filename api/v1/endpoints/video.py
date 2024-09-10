import os
import base64
import requests
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.config import settings
from api import deps

router = APIRouter()

@router.post("/search/video")
async def query_video(file: UploadFile = File(...)):
    try:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        with open(file_path, "rb") as video_file:
            base64_encoded_data = base64.b64encode(video_file.read())
            base64_video = base64_encoded_data.decode('utf-8')
        
        
        # Extract the first embedding from the response
        embedding_data = settings.get_clip_embedding(base64_video,'video')
        print("embedding_data", embedding_data)
 
        query_response = deps.index.query(
            vector=embedding_data,
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
        
        os.remove(file_path)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))