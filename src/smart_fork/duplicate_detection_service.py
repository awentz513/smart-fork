"""
Duplicate Detection Service for identifying similar sessions.

This module computes session-level embeddings and finds similar session pairs
to help users identify potential duplicates or related sessions.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .vector_db_service import VectorDBService
from .session_registry import SessionRegistry

logger = logging.getLogger(__name__)


@dataclass
class SimilarSession:
    """Represents a similar session with similarity score."""
    session_id: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'similarity': self.similarity,
            'metadata': self.metadata
        }


class DuplicateDetectionService:
    """
    Service for detecting duplicate or similar sessions.

    Features:
    - Compute session-level embeddings (average of chunk embeddings)
    - Find similar session pairs above threshold
    - Get similar sessions for a given session
    - Flag potential duplicates in search results
    """

    def __init__(
        self,
        vector_db_service: VectorDBService,
        session_registry: SessionRegistry,
        similarity_threshold: float = 0.85,
        min_chunks_for_comparison: int = 3
    ):
        """
        Initialize the DuplicateDetectionService.

        Args:
            vector_db_service: VectorDBService for accessing chunk embeddings
            session_registry: SessionRegistry for accessing session metadata
            similarity_threshold: Minimum similarity score (0-1) to consider sessions similar (default: 0.85)
            min_chunks_for_comparison: Minimum number of chunks a session must have for comparison (default: 3)
        """
        self.vector_db_service = vector_db_service
        self.session_registry = session_registry
        self.similarity_threshold = similarity_threshold
        self.min_chunks_for_comparison = min_chunks_for_comparison

        logger.info(
            f"Initialized DuplicateDetectionService "
            f"(threshold={similarity_threshold}, min_chunks={min_chunks_for_comparison})"
        )

    def compute_session_embedding(self, session_id: str) -> Optional[np.ndarray]:
        """
        Compute session-level embedding by averaging chunk embeddings.

        Args:
            session_id: Session ID to compute embedding for

        Returns:
            Session embedding as numpy array, or None if session has no chunks
        """
        # Get all chunks for this session
        chunks = self.vector_db_service.get_session_chunks(session_id)

        if not chunks or len(chunks) < self.min_chunks_for_comparison:
            logger.debug(f"Session {session_id} has too few chunks ({len(chunks) if chunks else 0}) for comparison")
            return None

        # Get the embeddings from ChromaDB
        try:
            # Query ChromaDB to get chunk embeddings
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            results = self.vector_db_service.collection.get(
                ids=chunk_ids,
                include=["embeddings"]
            )

            if not results["embeddings"]:
                logger.warning(f"No embeddings found for session {session_id}")
                return None

            embeddings = results["embeddings"]

            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)

            # Normalize the result (for cosine similarity)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            return avg_embedding

        except Exception as e:
            logger.error(f"Error computing embedding for session {session_id}: {e}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        # Both embeddings should already be normalized
        similarity = np.dot(embedding1, embedding2)

        # Clamp to [0, 1] range (due to floating point precision)
        similarity = max(0.0, min(1.0, float(similarity)))

        return similarity

    def get_similar_sessions(
        self,
        session_id: str,
        top_k: int = 5,
        include_metadata: bool = True
    ) -> List[SimilarSession]:
        """
        Find sessions similar to the given session.

        Args:
            session_id: Session ID to find similar sessions for
            top_k: Number of similar sessions to return (default: 5)
            include_metadata: Whether to include session metadata (default: True)

        Returns:
            List of SimilarSession objects, sorted by similarity (highest first)
        """
        # Compute embedding for the query session
        query_embedding = self.compute_session_embedding(session_id)

        if query_embedding is None:
            logger.warning(f"Cannot compute embedding for session {session_id}")
            return []

        # Get all sessions from registry
        all_sessions = self.session_registry.list_sessions()

        similar_sessions = []

        for other_session_id in all_sessions:
            # Skip the query session itself
            if other_session_id == session_id:
                continue

            # Compute embedding for this session
            other_embedding = self.compute_session_embedding(other_session_id)

            if other_embedding is None:
                continue

            # Compute similarity
            similarity = self.compute_similarity(query_embedding, other_embedding)

            # Only include if above threshold
            if similarity >= self.similarity_threshold:
                metadata = None
                if include_metadata:
                    metadata_obj = self.session_registry.get_session(other_session_id)
                    metadata = metadata_obj.to_dict() if metadata_obj else None

                similar_sessions.append(SimilarSession(
                    session_id=other_session_id,
                    similarity=similarity,
                    metadata=metadata
                ))

        # Sort by similarity (highest first) and take top_k
        similar_sessions.sort(key=lambda x: x.similarity, reverse=True)

        return similar_sessions[:top_k]

    def find_all_duplicate_pairs(
        self,
        batch_size: int = 100,
        progress_callback: Optional[Any] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find all similar session pairs in the database.

        This is a batch operation that compares all sessions pairwise.

        Args:
            batch_size: Number of sessions to process at once (default: 100)
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            List of tuples (session_id1, session_id2, similarity) for pairs above threshold
        """
        # Get all sessions
        all_sessions = self.session_registry.list_sessions()
        total_sessions = len(all_sessions)

        logger.info(f"Finding duplicate pairs among {total_sessions} sessions")

        # Compute embeddings for all sessions
        session_embeddings: Dict[str, np.ndarray] = {}

        for i, session_id in enumerate(all_sessions):
            if progress_callback and i % 10 == 0:
                progress_callback(i, total_sessions)

            embedding = self.compute_session_embedding(session_id)
            if embedding is not None:
                session_embeddings[session_id] = embedding

        logger.info(f"Computed embeddings for {len(session_embeddings)} sessions")

        # Compare all pairs
        duplicate_pairs = []
        session_ids = list(session_embeddings.keys())

        for i in range(len(session_ids)):
            for j in range(i + 1, len(session_ids)):
                session_id1 = session_ids[i]
                session_id2 = session_ids[j]

                similarity = self.compute_similarity(
                    session_embeddings[session_id1],
                    session_embeddings[session_id2]
                )

                if similarity >= self.similarity_threshold:
                    duplicate_pairs.append((session_id1, session_id2, similarity))

        # Sort by similarity (highest first)
        duplicate_pairs.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"Found {len(duplicate_pairs)} duplicate pairs above threshold {self.similarity_threshold}")

        return duplicate_pairs

    def flag_duplicates_in_results(
        self,
        search_results: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Flag potential duplicates in search results.

        Adds a 'similar_to' field to each result that contains similar sessions
        from within the result set.

        Args:
            search_results: List of search result objects

        Returns:
            List of result dictionaries with duplicate flags
        """
        if len(search_results) < 2:
            # No duplicates possible with less than 2 results
            return [
                {
                    'session_id': r.session_id,
                    'score': r.score.to_dict() if hasattr(r.score, 'to_dict') else r.score,
                    'similar_to': []
                }
                for r in search_results
            ]

        # Compute embeddings for all results
        result_embeddings: Dict[str, np.ndarray] = {}

        for result in search_results:
            session_id = result.session_id
            embedding = self.compute_session_embedding(session_id)
            if embedding is not None:
                result_embeddings[session_id] = embedding

        # Find similar pairs within results
        similar_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for i, result1 in enumerate(search_results):
            session_id1 = result1.session_id

            if session_id1 not in result_embeddings:
                continue

            for j, result2 in enumerate(search_results):
                if i >= j:  # Skip self and already compared pairs
                    continue

                session_id2 = result2.session_id

                if session_id2 not in result_embeddings:
                    continue

                # Compute similarity
                similarity = self.compute_similarity(
                    result_embeddings[session_id1],
                    result_embeddings[session_id2]
                )

                if similarity >= self.similarity_threshold:
                    # Add to both sessions' similar lists
                    similar_map[session_id1].append({
                        'session_id': session_id2,
                        'similarity': float(similarity)
                    })
                    similar_map[session_id2].append({
                        'session_id': session_id1,
                        'similarity': float(similarity)
                    })

        # Build result list with duplicate flags
        flagged_results = []

        for result in search_results:
            session_id = result.session_id
            similar_sessions = similar_map.get(session_id, [])

            # Sort similar sessions by similarity (highest first)
            similar_sessions.sort(key=lambda x: x['similarity'], reverse=True)

            flagged_results.append({
                'session_id': session_id,
                'score': result.score.to_dict() if hasattr(result.score, 'to_dict') else result.score,
                'similar_to': similar_sessions
            })

        return flagged_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about duplicate detection.

        Returns:
            Dictionary with statistics
        """
        all_sessions = self.session_registry.list_sessions()
        total_sessions = len(all_sessions)

        # Count sessions with enough chunks
        sessions_with_enough_chunks = 0
        for session_id in all_sessions:
            chunks = self.vector_db_service.get_session_chunks(session_id)
            if chunks and len(chunks) >= self.min_chunks_for_comparison:
                sessions_with_enough_chunks += 1

        return {
            'total_sessions': total_sessions,
            'sessions_eligible_for_comparison': sessions_with_enough_chunks,
            'similarity_threshold': self.similarity_threshold,
            'min_chunks_required': self.min_chunks_for_comparison
        }
