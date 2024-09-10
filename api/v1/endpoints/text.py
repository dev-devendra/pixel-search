import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.config import settings
from api import deps

router = APIRouter()

class TextQuery(BaseModel):
    query: str

@router.post("/search/text")
async def query_text(query: TextQuery):
    try:
        if not query.query:
            raise HTTPException(status_code=400, detail="The query text cannot be empty")

        # Convert PyTorch tensor to Python list
        embeddings = settings.get_clip_embedding(query.query, "text")

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