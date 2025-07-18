import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.github_sync import github_sync
from app.chroma_manager import chroma_manager
from app.config import settings
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import uvicorn

app = FastAPI(title="Linera RAG Service")

class QueryRequest(BaseModel):
    text: str
    top_k: int = 5

class QueryResponseItem(BaseModel):
    document: str
    metadata: dict
    score: float

class QueryResponse(BaseModel):
    results: list[QueryResponseItem]

@app.on_event("startup")
async def startup_event():
    # Initialize repositories and index on startup
    await github_sync.update()
    
    # Schedule regular updates
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        github_sync.update,
        'interval',
        hours=settings.UPDATE_INTERVAL_HOURS
    )
    scheduler.start()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        results = chroma_manager.query_index(request.text, request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )