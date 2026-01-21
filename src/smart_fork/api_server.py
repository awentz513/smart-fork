"""REST API server for Smart Fork search and indexing."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .search_service import SearchService, SessionSearchResult
from .session_registry import SessionRegistry
from .background_indexer import BackgroundIndexer
from .session_parser import SessionParser


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response validation

class SearchRequest(BaseModel):
    """Request model for chunk search."""
    query: str = Field(..., description="Natural language search query")
    k_chunks: int = Field(200, description="Number of chunks to retrieve from vector DB", ge=1, le=1000)
    top_n_sessions: int = Field(5, description="Number of top sessions to return", ge=1, le=50)
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")


class SearchResponse(BaseModel):
    """Response model for chunk search."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    execution_time_ms: float


class IndexRequest(BaseModel):
    """Request model for session indexing."""
    session_file: str = Field(..., description="Path to session JSONL file")
    force_reindex: bool = Field(False, description="Force reindexing even if already indexed")


class IndexResponse(BaseModel):
    """Response model for session indexing."""
    session_id: str
    status: str
    chunks_added: int
    messages_processed: int
    execution_time_ms: float


class SessionResponse(BaseModel):
    """Response model for session metadata."""
    session_id: str
    metadata: Dict[str, Any]


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    vector_db_stats: Dict[str, Any]
    registry_stats: Dict[str, Any]
    indexer_stats: Dict[str, Any]
    search_stats: Dict[str, Any]


# Create FastAPI application
app = FastAPI(
    title="Smart Fork API",
    description="Local REST API for Smart Fork session search and indexing",
    version="0.1.0",
    docs_url=None,  # Disable docs in production
    redoc_url=None  # Disable redoc in production
)


# Global service instances (will be initialized on startup)
search_service: Optional[SearchService] = None
session_registry: Optional[SessionRegistry] = None
background_indexer: Optional[BackgroundIndexer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global search_service, session_registry, background_indexer

    logger.info("Initializing Smart Fork API services...")

    try:
        # Initialize session registry
        session_registry = SessionRegistry()
        logger.info("Session registry initialized")

        # Initialize search service
        search_service = SearchService(session_registry=session_registry)
        logger.info("Search service initialized")

        # Get Claude directory to monitor
        claude_dir = Path.home() / ".claude"

        # Initialize background indexer
        session_parser = SessionParser()
        background_indexer = BackgroundIndexer(
            claude_dir=claude_dir,
            vector_db=search_service.vector_db,
            session_registry=session_registry,
            embedding_service=search_service.embedding_service,
            chunking_service=search_service.chunking_service,
            session_parser=session_parser,
            debounce_seconds=5.0,
            checkpoint_interval=15
        )
        logger.info("Background indexer initialized")

        # Start background indexer to begin monitoring
        background_indexer.start()
        logger.info(f"Background indexer started, monitoring {claude_dir}")

        logger.info("Smart Fork API services started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on shutdown."""
    global background_indexer

    logger.info("Shutting down Smart Fork API services...")

    if background_indexer:
        background_indexer.stop()
        logger.info("Background indexer stopped")

    logger.info("Smart Fork API services shut down successfully")


@app.post("/chunks/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_chunks(request: SearchRequest) -> SearchResponse:
    """
    Search for relevant session chunks using semantic search.

    Args:
        request: Search parameters including query, k_chunks, top_n_sessions

    Returns:
        SearchResponse with ranked session results
    """
    if not search_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not initialized"
        )

    try:
        start_time = datetime.now()

        # Perform search
        results = search_service.search(
            query=request.query,
            k_chunks=request.k_chunks,
            top_n_sessions=request.top_n_sessions,
            metadata_filter=request.metadata_filter
        )

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Convert results to dict format using the to_dict method
        results_dict = [r.to_dict() for r in results]

        return SearchResponse(
            query=request.query,
            results=results_dict,
            total_results=len(results),
            execution_time_ms=execution_time_ms
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/sessions/index", response_model=IndexResponse, status_code=status.HTTP_200_OK)
async def index_session(request: IndexRequest) -> IndexResponse:
    """
    Index a session file into the vector database.

    Args:
        request: Indexing parameters including session_file path

    Returns:
        IndexResponse with indexing results
    """
    if not background_indexer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background indexer not initialized"
        )

    try:
        start_time = datetime.now()

        # Validate file exists
        session_path = Path(request.session_file).expanduser()
        if not session_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session file not found: {request.session_file}"
            )

        # Get session ID from filename (without .jsonl extension)
        session_id = session_path.stem

        # Check if already indexed (unless force_reindex)
        if not request.force_reindex and session_registry:
            existing_session = session_registry.get_session(session_id)
            if existing_session:
                logger.info(f"Session {session_id} already indexed, skipping")
                return IndexResponse(
                    session_id=session_id,
                    status="already_indexed",
                    chunks_added=0,
                    messages_processed=0,
                    execution_time_ms=0
                )

        # Index the file
        logger.info(f"Indexing session file: {session_path}")
        stats_before = background_indexer.get_stats()

        success = background_indexer.index_file(str(session_path))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to index session: {session_id}"
            )

        stats_after = background_indexer.get_stats()

        # Calculate changes
        chunks_added = stats_after["chunks_added"] - stats_before["chunks_added"]
        messages_processed = stats_after["files_indexed"] - stats_before["files_indexed"]

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return IndexResponse(
            session_id=session_id,
            status="indexed",
            chunks_added=chunks_added,
            messages_processed=messages_processed,
            execution_time_ms=execution_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )


@app.get("/sessions/{session_id}", response_model=SessionResponse, status_code=status.HTTP_200_OK)
async def get_session(session_id: str) -> SessionResponse:
    """
    Get metadata for a specific session.

    Args:
        session_id: Session identifier

    Returns:
        SessionResponse with session metadata
    """
    if not session_registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session registry not initialized"
        )

    try:
        metadata = session_registry.get_session(session_id)

        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Convert SessionMetadata to dict
        return SessionResponse(
            session_id=session_id,
            metadata=metadata.to_dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, status_code=status.HTTP_200_OK)
async def get_stats() -> StatsResponse:
    """
    Get system statistics including vector DB, registry, indexer, and search stats.

    Returns:
        StatsResponse with all system statistics
    """
    try:
        stats = {
            "vector_db_stats": {},
            "registry_stats": {},
            "indexer_stats": {},
            "search_stats": {}
        }

        # Get vector DB stats
        if search_service and search_service.vector_db:
            stats["vector_db_stats"] = search_service.vector_db.get_stats()

        # Get registry stats
        if session_registry:
            stats["registry_stats"] = session_registry.get_stats()

        # Get indexer stats
        if background_indexer:
            stats["indexer_stats"] = background_indexer.get_stats()

        # Get search stats
        if search_service:
            stats["search_stats"] = search_service.get_stats()

        return StatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Health status of all services
    """
    return {
        "status": "healthy",
        "services": {
            "search_service": search_service is not None,
            "session_registry": session_registry is not None,
            "background_indexer": background_indexer is not None
        },
        "timestamp": datetime.now().isoformat()
    }


def start_server(host: str = "127.0.0.1", port: int = 8741) -> None:
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        port: Port to bind to (default: 8741)
    """
    logger.info(f"Starting Smart Fork API server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    start_server()
