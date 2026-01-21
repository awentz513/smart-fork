"""
Search service for finding relevant sessions using semantic search and composite scoring.

This module orchestrates the embedding generation, vector search, and composite scoring
to rank and return the most relevant sessions for a given query.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from .embedding_service import EmbeddingService
from .vector_db_service import VectorDBService, ChunkSearchResult
from .scoring_service import ScoringService, SessionScore
from .session_registry import SessionRegistry, SessionMetadata
from .cache_service import CacheService

logger = logging.getLogger(__name__)


@dataclass
class SessionSearchResult:
    """Represents a ranked search result for a session."""
    session_id: str
    score: SessionScore
    metadata: Optional[SessionMetadata]
    preview: str
    matched_chunks: List[ChunkSearchResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'score': self.score.to_dict(),
            'metadata': self.metadata.to_dict() if self.metadata else None,
            'preview': self.preview,
            'num_matched_chunks': len(self.matched_chunks),
        }


class SearchService:
    """
    Service for searching and ranking sessions by relevance.

    Orchestrates:
    1. Query embedding generation
    2. K-nearest neighbors vector search (k=200 chunks)
    3. Grouping chunks by session
    4. Composite score calculation per session
    5. Ranking and returning top N sessions
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_db_service: VectorDBService,
        scoring_service: ScoringService,
        session_registry: SessionRegistry,
        k_chunks: int = 200,
        top_n_sessions: int = 5,
        preview_length: int = 200,
        cache_service: Optional[CacheService] = None,
        enable_cache: bool = True
    ):
        """
        Initialize the SearchService.

        Args:
            embedding_service: Service for generating query embeddings
            vector_db_service: Service for vector similarity search
            scoring_service: Service for calculating composite scores
            session_registry: Registry for session metadata
            k_chunks: Number of chunks to retrieve in k-NN search (default 200)
            top_n_sessions: Number of top sessions to return (default 5)
            preview_length: Length of preview text in characters (default 200)
            cache_service: Optional CacheService for caching embeddings and results
            enable_cache: Whether to use caching (default True)
        """
        self.embedding_service = embedding_service
        self.vector_db_service = vector_db_service
        self.scoring_service = scoring_service
        self.session_registry = session_registry
        self.k_chunks = k_chunks
        self.top_n_sessions = top_n_sessions
        self.preview_length = preview_length

        # Initialize cache if enabled
        self.enable_cache = enable_cache
        if enable_cache and cache_service is None:
            self.cache_service = CacheService()
            logger.info("Initialized default CacheService")
        else:
            self.cache_service = cache_service

        if not enable_cache:
            logger.info("Caching disabled")

        logger.info(
            f"Initialized SearchService (k={k_chunks}, top_n={top_n_sessions}, cache={'enabled' if enable_cache else 'disabled'})"
        )

    def search(
        self,
        query: str,
        top_n: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SessionSearchResult]:
        """
        Search for relevant sessions using semantic search and ranking.

        Args:
            query: Natural language search query
            top_n: Number of top sessions to return (overrides default)
            filter_metadata: Optional metadata filters for search

        Returns:
            List of SessionSearchResult objects, ranked by relevance score
        """
        if top_n is None:
            top_n = self.top_n_sessions

        logger.info(f"Searching for query: '{query[:50]}...' (top_n={top_n})")

        # Try to get cached results first
        if self.enable_cache and self.cache_service:
            cached_results = self.cache_service.get_search_results(query, filter_metadata)
            if cached_results is not None:
                logger.info(f"Returning {len(cached_results)} cached search results")
                return cached_results[:top_n]  # Respect top_n parameter

        # Step 1: Generate query embedding (with caching)
        logger.debug("Generating query embedding...")
        self.embedding_service.load_model()

        # Try to get cached embedding
        query_embedding = None
        if self.enable_cache and self.cache_service:
            query_embedding = self.cache_service.get_query_embedding(query)
            if query_embedding:
                logger.debug("Using cached query embedding")

        # Generate new embedding if not cached
        if query_embedding is None:
            query_embedding = self.embedding_service.embed_single(query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            # Cache the embedding
            if self.enable_cache and self.cache_service:
                self.cache_service.put_query_embedding(query, query_embedding)

        # Step 2: Perform k-NN search
        logger.debug(f"Performing k-NN search (k={self.k_chunks})...")
        search_results = self.vector_db_service.search_chunks(
            query_embedding=query_embedding,
            k=self.k_chunks,
            filter_metadata=filter_metadata
        )

        if not search_results:
            logger.info("No matching chunks found")
            return []

        logger.info(f"Found {len(search_results)} matching chunks")

        # Step 3: Group chunks by session_id
        logger.debug("Grouping chunks by session...")
        session_chunks = self._group_chunks_by_session(search_results)

        logger.info(f"Grouped into {len(session_chunks)} sessions")

        # Step 4: Calculate composite scores for each session
        logger.debug("Calculating composite scores...")
        session_scores = self._calculate_session_scores(session_chunks)

        # Step 5: Rank sessions and return top N
        logger.debug(f"Ranking sessions and selecting top {top_n}...")
        ranked_scores = self.scoring_service.rank_sessions(session_scores)
        top_sessions = ranked_scores[:top_n]

        # Step 6: Build search results with metadata and previews
        results = []
        for score in top_sessions:
            session_metadata = self.session_registry.get_session(score.session_id)
            chunks = session_chunks[score.session_id]
            preview = self._generate_preview(chunks)

            results.append(SessionSearchResult(
                session_id=score.session_id,
                score=score,
                metadata=session_metadata,
                preview=preview,
                matched_chunks=chunks
            ))

        logger.info(
            f"Returning {len(results)} results "
            f"(scores: {[f'{r.score.final_score:.3f}' for r in results]})"
        )

        # Cache the results
        if self.enable_cache and self.cache_service:
            self.cache_service.put_search_results(query, results, filter_metadata)
            logger.debug("Search results cached")

        return results

    def _group_chunks_by_session(
        self,
        chunks: List[ChunkSearchResult]
    ) -> Dict[str, List[ChunkSearchResult]]:
        """
        Group chunks by session_id.

        Args:
            chunks: List of chunk search results

        Returns:
            Dictionary mapping session_id to list of chunks
        """
        grouped = defaultdict(list)

        for chunk in chunks:
            grouped[chunk.session_id].append(chunk)

        # Sort chunks within each session by chunk_index for consistent ordering
        for session_id in grouped:
            grouped[session_id].sort(key=lambda c: c.chunk_index)

        return dict(grouped)

    def _calculate_session_scores(
        self,
        session_chunks: Dict[str, List[ChunkSearchResult]]
    ) -> List[SessionScore]:
        """
        Calculate composite scores for each session.

        Args:
            session_chunks: Dictionary mapping session_id to chunks

        Returns:
            List of SessionScore objects
        """
        scores = []

        for session_id, chunks in session_chunks.items():
            # Extract similarity scores
            chunk_similarities = [chunk.similarity for chunk in chunks]

            # Get session metadata for total chunk count and timestamps
            session_metadata = self.session_registry.get_session(session_id)

            total_chunks = session_metadata.chunk_count if session_metadata else len(chunks)
            last_modified = session_metadata.last_modified if session_metadata else None

            # Extract memory types from chunk metadata
            memory_types = []
            for chunk in chunks:
                if 'memory_types' in chunk.metadata and chunk.metadata['memory_types']:
                    memory_types.extend(chunk.metadata['memory_types'])

            # Calculate composite score
            score = self.scoring_service.calculate_session_score(
                session_id=session_id,
                chunk_similarities=chunk_similarities,
                total_chunks_in_session=total_chunks,
                session_last_modified=last_modified,
                memory_types=memory_types if memory_types else None
            )

            scores.append(score)

        return scores

    def _generate_preview(self, chunks: List[ChunkSearchResult]) -> str:
        """
        Generate a preview snippet from the most relevant chunks.

        Args:
            chunks: List of chunk search results for a session

        Returns:
            Preview text (truncated to preview_length)
        """
        if not chunks:
            return ""

        # Use the highest-scoring chunk for the preview
        best_chunk = max(chunks, key=lambda c: c.similarity)

        preview = best_chunk.content.strip()

        # Truncate if necessary
        if len(preview) > self.preview_length:
            preview = preview[:self.preview_length].rsplit(' ', 1)[0] + "..."

        return preview

    def get_session_preview(
        self,
        session_id: str,
        length: int = 500,
        claude_dir: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a preview of a session's content.

        Args:
            session_id: ID of the session to preview
            length: Maximum number of characters to return (default 500)
            claude_dir: Optional Claude directory path (default: ~/.claude)

        Returns:
            Dictionary containing:
            - session_id: The session identifier
            - preview: First N characters of concatenated message content
            - message_count: Total number of messages in the session
            - date_range: Dict with 'start' and 'end' timestamps
            - metadata: Session metadata if available

            Returns None if session not found or cannot be read.
        """
        # Get session metadata from registry
        metadata = self.session_registry.get_session(session_id)
        if not metadata:
            logger.warning(f"Session not found in registry: {session_id}")
            return None

        # Find session file path
        from .fork_generator import ForkGenerator
        from .session_parser import SessionParser

        fork_gen = ForkGenerator(claude_sessions_dir=claude_dir or "~/.claude")
        file_path = fork_gen.find_session_path(session_id, project=metadata.project)

        if not file_path:
            logger.warning(f"Session file path not found for: {session_id}")
            return None

        # Parse session file to get full content
        parser = SessionParser(strict=False)

        try:
            session_data = parser.parse_file(file_path)
        except Exception as e:
            logger.error(f"Failed to parse session file {file_path}: {e}")
            return None

        if not session_data.messages:
            logger.warning(f"Session {session_id} has no messages")
            return None

        # Concatenate message content for preview
        message_texts = []
        for msg in session_data.messages:
            # Format: "role: content"
            message_texts.append(f"{msg.role}: {msg.content}")

        full_text = "\n\n".join(message_texts)

        # Truncate to requested length
        preview = full_text[:length]
        if len(full_text) > length:
            preview = preview.rsplit(' ', 1)[0] + "..."

        # Get date range
        timestamps = [msg.timestamp for msg in session_data.messages if msg.timestamp]
        date_range = None
        if timestamps:
            date_range = {
                'start': min(timestamps).isoformat(),
                'end': max(timestamps).isoformat()
            }
        elif session_data.created_at and session_data.last_modified:
            date_range = {
                'start': session_data.created_at.isoformat(),
                'end': session_data.last_modified.isoformat()
            }

        return {
            'session_id': session_id,
            'preview': preview,
            'message_count': len(session_data.messages),
            'date_range': date_range,
            'metadata': metadata.to_dict() if metadata else None
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get search service statistics.

        Returns:
            Dictionary with service configuration and stats
        """
        db_stats = self.vector_db_service.get_stats()
        registry_stats = self.session_registry.get_stats()

        stats = {
            'k_chunks': self.k_chunks,
            'top_n_sessions': self.top_n_sessions,
            'preview_length': self.preview_length,
            'cache_enabled': self.enable_cache,
            'vector_db': db_stats,
            'registry': registry_stats,
        }

        # Add cache statistics if caching is enabled
        if self.enable_cache and self.cache_service:
            stats['cache'] = self.cache_service.get_stats()

        return stats
