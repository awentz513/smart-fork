"""
Composite scoring service for ranking session relevance.

This module implements the composite scoring algorithm that combines multiple
factors to rank sessions by relevance: similarity scores, chunk ratios, recency,
chain quality, and memory type boosts.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import math


@dataclass
class SessionScore:
    """Represents the composite score for a session."""
    session_id: str
    final_score: float
    best_similarity: float
    avg_similarity: float
    chunk_ratio: float
    recency_score: float
    chain_quality: float
    memory_boost: float
    preference_boost: float
    num_chunks_matched: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'final_score': self.final_score,
            'best_similarity': self.best_similarity,
            'avg_similarity': self.avg_similarity,
            'chunk_ratio': self.chunk_ratio,
            'recency_score': self.recency_score,
            'chain_quality': self.chain_quality,
            'memory_boost': self.memory_boost,
            'preference_boost': self.preference_boost,
            'num_chunks_matched': self.num_chunks_matched,
        }


class ScoringService:
    """
    Service for calculating composite relevance scores for sessions.

    Implements the scoring formula:
    Final Score = (best_similarity × 0.40)
                + (avg_similarity × 0.20)
                + (chunk_ratio × 0.05)
                + (recency × 0.25)
                + (chain_quality × 0.10)
                + memory_boost
                + preference_boost

    With memory type boosts:
    - PATTERN: +5%
    - WORKING_SOLUTION: +8%
    - WAITING: +2%

    With preference learning boost:
    - Based on user fork history and selection patterns
    - Up to +0.10 for frequently forked sessions
    - Weighted by recency and position quality
    """

    # Scoring weights
    WEIGHT_BEST_SIMILARITY = 0.40
    WEIGHT_AVG_SIMILARITY = 0.20
    WEIGHT_CHUNK_RATIO = 0.05
    WEIGHT_RECENCY = 0.25
    WEIGHT_CHAIN_QUALITY = 0.10

    # Memory type boosts (additive percentages)
    BOOST_PATTERN = 0.05
    BOOST_WORKING_SOLUTION = 0.08
    BOOST_WAITING = 0.02

    # Recency decay constant (30 days in seconds)
    RECENCY_DECAY_CONSTANT = 30 * 24 * 60 * 60  # 30 days

    def __init__(self, chain_quality_placeholder: float = 0.5):
        """
        Initialize the ScoringService.

        Args:
            chain_quality_placeholder: Placeholder value for chain quality (0.0-1.0).
                                      Default is 0.5 as specified in the PRD.
        """
        self.chain_quality_placeholder = chain_quality_placeholder

    def calculate_session_score(
        self,
        session_id: str,
        chunk_similarities: List[float],
        total_chunks_in_session: int,
        session_last_modified: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        preference_boost: float = 0.0,
        current_time: Optional[datetime] = None
    ) -> SessionScore:
        """
        Calculate composite score for a single session.

        Args:
            session_id: The session identifier
            chunk_similarities: List of similarity scores for matched chunks
            total_chunks_in_session: Total number of chunks in the session
            session_last_modified: ISO timestamp of session's last modification
            memory_types: List of memory types found in matched chunks
            preference_boost: Preference learning boost (from PreferenceService)
            current_time: Current time for recency calculation (defaults to now)

        Returns:
            SessionScore object with all calculated components
        """
        if not chunk_similarities:
            # No chunks matched - return zero score
            return SessionScore(
                session_id=session_id,
                final_score=0.0,
                best_similarity=0.0,
                avg_similarity=0.0,
                chunk_ratio=0.0,
                recency_score=0.0,
                chain_quality=self.chain_quality_placeholder,
                memory_boost=0.0,
                preference_boost=0.0,
                num_chunks_matched=0
            )

        # Calculate best similarity (max of all matched chunks)
        best_similarity = max(chunk_similarities)

        # Calculate average similarity
        avg_similarity = sum(chunk_similarities) / len(chunk_similarities)

        # Calculate chunk ratio (matched chunks / total chunks)
        chunk_ratio = len(chunk_similarities) / total_chunks_in_session if total_chunks_in_session > 0 else 0.0

        # Calculate recency score
        recency_score = self._calculate_recency_score(session_last_modified, current_time)

        # Use chain quality placeholder
        chain_quality = self.chain_quality_placeholder

        # Calculate memory boost
        memory_boost = self._calculate_memory_boost(memory_types)

        # Calculate base score (before boosts)
        base_score = (
            (best_similarity * self.WEIGHT_BEST_SIMILARITY) +
            (avg_similarity * self.WEIGHT_AVG_SIMILARITY) +
            (chunk_ratio * self.WEIGHT_CHUNK_RATIO) +
            (recency_score * self.WEIGHT_RECENCY) +
            (chain_quality * self.WEIGHT_CHAIN_QUALITY)
        )

        # Apply boosts (additive)
        final_score = base_score + memory_boost + preference_boost

        # Ensure score stays in valid range [0.0, 1.0+boosts]
        final_score = max(0.0, final_score)

        return SessionScore(
            session_id=session_id,
            final_score=final_score,
            best_similarity=best_similarity,
            avg_similarity=avg_similarity,
            chunk_ratio=chunk_ratio,
            recency_score=recency_score,
            chain_quality=chain_quality,
            memory_boost=memory_boost,
            preference_boost=preference_boost,
            num_chunks_matched=len(chunk_similarities)
        )

    def _calculate_recency_score(
        self,
        session_last_modified: Optional[str],
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate recency score using exponential decay.

        Uses the formula: recency = exp(-age_in_seconds / DECAY_CONSTANT)
        where DECAY_CONSTANT = 30 days

        Args:
            session_last_modified: ISO timestamp string
            current_time: Current time (defaults to now)

        Returns:
            Recency score between 0.0 and 1.0
        """
        if not session_last_modified:
            # No timestamp - assume very old (minimum recency)
            return 0.0

        if current_time is None:
            current_time = datetime.now()

        try:
            # Parse ISO format timestamp
            last_modified = datetime.fromisoformat(session_last_modified.replace('Z', '+00:00'))

            # Calculate age in seconds
            age_seconds = (current_time - last_modified).total_seconds()

            # Handle negative age (future timestamps - shouldn't happen but be safe)
            if age_seconds < 0:
                age_seconds = 0

            # Calculate exponential decay
            recency = math.exp(-age_seconds / self.RECENCY_DECAY_CONSTANT)

            return recency

        except (ValueError, AttributeError):
            # Failed to parse timestamp - return 0
            return 0.0

    def _calculate_memory_boost(self, memory_types: Optional[List[str]]) -> float:
        """
        Calculate memory type boost score.

        Memory types are additive:
        - PATTERN: +5%
        - WORKING_SOLUTION: +8%
        - WAITING: +2%

        Args:
            memory_types: List of memory type strings

        Returns:
            Total boost value (additive)
        """
        if not memory_types:
            return 0.0

        boost = 0.0
        memory_types_set = set(memory_types)

        if 'PATTERN' in memory_types_set:
            boost += self.BOOST_PATTERN

        if 'WORKING_SOLUTION' in memory_types_set:
            boost += self.BOOST_WORKING_SOLUTION

        if 'WAITING' in memory_types_set:
            boost += self.BOOST_WAITING

        return boost

    def rank_sessions(
        self,
        session_scores: List[SessionScore],
        top_k: int = 5
    ) -> List[SessionScore]:
        """
        Rank sessions by final score and return top K.

        Args:
            session_scores: List of SessionScore objects
            top_k: Number of top sessions to return

        Returns:
            List of top K SessionScore objects sorted by final_score (descending)
        """
        # Sort by final score (descending)
        sorted_scores = sorted(session_scores, key=lambda x: x.final_score, reverse=True)

        # Return top K
        return sorted_scores[:top_k]
