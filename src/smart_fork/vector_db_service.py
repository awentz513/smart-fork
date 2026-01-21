"""
ChromaDB vector database service for storing and searching session chunks.

This module provides a wrapper around ChromaDB for persistent storage of
session chunks with embeddings, supporting CRUD operations and similarity search.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings


@dataclass
class ChunkSearchResult:
    """Represents a search result from the vector database."""
    chunk_id: str
    session_id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    chunk_index: int


class VectorDBService:
    """
    Service for managing session chunks in ChromaDB vector database.

    Provides methods for storing, searching, and managing session chunks
    with their embeddings in a persistent ChromaDB collection.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the VectorDBService.

        Args:
            persist_directory: Directory for persistent storage.
                             Defaults to ~/.smart-fork/vector_db/
        """
        if persist_directory is None:
            home = os.path.expanduser("~")
            persist_directory = os.path.join(home, ".smart-fork", "vector_db")

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        self.persist_directory = persist_directory

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create the main collection
        self.collection = self.client.get_or_create_collection(
            name="session_chunks",
            metadata={"description": "Claude Code session chunks with embeddings"}
        )

    def add_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        chunk_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add chunks to the vector database.

        Args:
            chunks: List of text chunks to add
            embeddings: List of embedding vectors (must match chunks length)
            metadata: List of metadata dicts (must match chunks length)
            chunk_ids: Optional list of custom IDs. If None, auto-generated.

        Returns:
            List of chunk IDs that were added

        Raises:
            ValueError: If lengths don't match or inputs are invalid
        """
        if not chunks:
            return []

        # Validate inputs
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        if len(chunks) != len(metadata):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match metadata count ({len(metadata)})"
            )

        # Generate IDs if not provided
        if chunk_ids is None:
            # Use session_id and chunk_index to create unique IDs
            chunk_ids = []
            for i, meta in enumerate(metadata):
                session_id = meta.get("session_id", "unknown")
                chunk_index = meta.get("chunk_index", i)
                chunk_ids.append(f"{session_id}_chunk_{chunk_index}")

        if len(chunk_ids) != len(chunks):
            raise ValueError(
                f"Chunk IDs count ({len(chunk_ids)}) must match chunks count ({len(chunks)})"
            )

        # Convert metadata to chromadb format (all values must be strings, ints, floats, or bools)
        processed_metadata = []
        for meta in metadata:
            processed = {}
            for key, value in meta.items():
                # Convert to supported types
                if isinstance(value, (str, int, float, bool)):
                    processed[key] = value
                elif value is None:
                    processed[key] = ""
                elif isinstance(value, list):
                    # Serialize lists to JSON string (e.g., memory_types)
                    processed[key] = json.dumps(value)
                else:
                    # Convert other types to string
                    processed[key] = str(value)
            processed_metadata.append(processed)

        # Add to collection
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=processed_metadata
        )

        return chunk_ids

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize metadata from ChromaDB format back to Python types.

        Specifically handles JSON-serialized lists (e.g., memory_types).

        Args:
            metadata: Raw metadata from ChromaDB

        Returns:
            Deserialized metadata dictionary
        """
        deserialized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                # Try to parse as JSON (for lists that were serialized)
                if value.startswith('[') and value.endswith(']'):
                    try:
                        deserialized[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # Not valid JSON, keep as string
                        deserialized[key] = value
                else:
                    deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized

    def search_chunks(
        self,
        query_embedding: List[float],
        k: int = 200,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ChunkSearchResult]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return (default 200)
            filter_metadata: Optional metadata filters (e.g., {"session_id": "abc123"})

        Returns:
            List of ChunkSearchResult objects, sorted by similarity (highest first)
        """
        if k <= 0:
            return []

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to ChunkSearchResult objects
        search_results = []

        if results["ids"] and len(results["ids"]) > 0:
            chunk_ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(chunk_ids)):
                # Convert distance to similarity (ChromaDB uses L2 distance)
                # For normalized embeddings, similarity = 1 - (distance^2 / 2)
                # But we'll use a simpler conversion: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distances[i])

                metadata = metadatas[i] if metadatas else {}
                metadata = self._deserialize_metadata(metadata)

                search_results.append(ChunkSearchResult(
                    chunk_id=chunk_ids[i],
                    session_id=metadata.get("session_id", "unknown"),
                    content=documents[i],
                    metadata=metadata,
                    similarity=similarity,
                    chunk_index=int(metadata.get("chunk_index", 0))
                ))

        return search_results

    def delete_session_chunks(self, session_id: str) -> int:
        """
        Delete all chunks for a specific session.

        Args:
            session_id: Session ID to delete chunks for

        Returns:
            Number of chunks deleted
        """
        # Query to find all chunks for this session
        results = self.collection.get(
            where={"session_id": session_id},
            include=[]
        )

        if results["ids"]:
            chunk_ids = results["ids"]
            self.collection.delete(ids=chunk_ids)
            return len(chunk_ids)

        return 0

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkSearchResult]:
        """
        Get a specific chunk by ID.

        Args:
            chunk_id: Chunk ID to retrieve

        Returns:
            ChunkSearchResult if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )

            if results["ids"] and len(results["ids"]) > 0:
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                metadata = self._deserialize_metadata(metadata)

                return ChunkSearchResult(
                    chunk_id=results["ids"][0],
                    session_id=metadata.get("session_id", "unknown"),
                    content=results["documents"][0],
                    metadata=metadata,
                    similarity=1.0,  # Perfect match for direct retrieval
                    chunk_index=int(metadata.get("chunk_index", 0))
                )
        except Exception:
            pass

        return None

    def get_session_chunks(self, session_id: str) -> List[ChunkSearchResult]:
        """
        Get all chunks for a specific session.

        Args:
            session_id: Session ID to retrieve chunks for

        Returns:
            List of ChunkSearchResult objects, sorted by chunk_index
        """
        results = self.collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                metadata = self._deserialize_metadata(metadata)

                chunks.append(ChunkSearchResult(
                    chunk_id=results["ids"][i],
                    session_id=metadata.get("session_id", "unknown"),
                    content=results["documents"][i],
                    metadata=metadata,
                    similarity=1.0,
                    chunk_index=int(metadata.get("chunk_index", 0))
                ))

        # Sort by chunk_index
        chunks.sort(key=lambda x: x.chunk_index)
        return chunks

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics (total_chunks, collection_name, etc.)
        """
        count = self.collection.count()

        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "persist_directory": self.persist_directory
        }

    def reset(self):
        """
        Reset the database (DELETE ALL DATA).

        WARNING: This permanently deletes all chunks and embeddings.
        Use only for testing or when you want to rebuild the index.
        """
        self.client.delete_collection("session_chunks")
        self.collection = self.client.get_or_create_collection(
            name="session_chunks",
            metadata={"description": "Claude Code session chunks with embeddings"}
        )
