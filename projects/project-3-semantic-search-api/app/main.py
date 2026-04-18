from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.search import semantic_search
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Search API", version="1.0")

# Request model
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# Response model
class GameResult(BaseModel):
    name: str
    similarity: float

class SearchResponse(BaseModel):
    query: str
    results: List[GameResult]
    count: int


@app.get("/health")
def health():
    """Check the status of the API"""
    return {"status": "ok", "version": "1.0"}


@app.post("/search")
def search(request: SearchRequest):
    """Search for games based on query"""
    
    logger.info(f"Search requested: query='{request.query}', top_k={request.top_k}")
    
    # Validate input - return 400 Bad Request
    if not request.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.top_k <= 0:
        logger.warning(f"Invalid top_k: {request.top_k}")
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")
    
    if request.top_k > 100:
        logger.warning(f"top_k too large: {request.top_k}, max value is 100")
        request.top_k = 100
    
    try:
        logger.info("Performing semantic search...")
        results = semantic_search(request.query, top_k=request.top_k)
        
        # Transform results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "name": r['game'],
                "similarity": float(r['score'])
            })
        
        logger.info(f"Search completed: found {len(results)} results")
        
        return {
            "query": request.query,
            "results": formatted_results,
            "count": len(results)
        }
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")