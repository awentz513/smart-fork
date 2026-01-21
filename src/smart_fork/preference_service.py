"""Preference learning service for Smart Fork Detection.

This module learns from user fork selections to improve future search rankings.
It tracks:
- Which result position users select (1-5, or -1 for custom)
- Which sessions users fork frequently
- Query-session preference patterns

The preference score is integrated into the composite scoring algorithm to boost
sessions that users have forked before or prefer in similar contexts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading
import math

logger = logging.getLogger(__name__)


@dataclass
class PreferenceEntry:
    """Represents a single user selection event."""
    session_id: str
    query: str
    position: int  # 1-5 for displayed results, -1 for custom entry
    timestamp: str  # ISO format

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreferenceEntry':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PreferenceScore:
    """Calculated preference score for a session."""
    session_id: str
    preference_boost: float  # Boost value to add to composite score
    fork_count: int  # Number of times user has forked this session
    avg_position: float  # Average position when selected (lower is better)
    recency_weight: float  # How recent the last fork was

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PreferenceService:
    """
    Service for learning from user fork selections.

    Implements preference learning that:
    1. Records which result positions users select
    2. Tracks session fork frequency
    3. Calculates preference boost for scoring pipeline
    4. Applies recency weighting to recent preferences

    Preference boost calculation:
    - Base boost: 0.02 per fork (up to 5 forks = 0.10 max)
    - Position quality bonus: higher for selections at position 1
    - Recency weight: exponential decay over 90 days

    Thread-safe implementation with file-based persistence.
    """

    # Preference weights
    BOOST_PER_FORK = 0.02  # Boost per fork occurrence
    MAX_FORK_BOOST = 0.10  # Maximum boost from fork count (5 forks)
    POSITION_WEIGHT = 0.01  # Bonus for being selected at top positions
    RECENCY_DECAY_DAYS = 90  # Exponential decay over 90 days

    def __init__(
        self,
        preference_file: Optional[str] = None,
        max_entries: int = 1000
    ):
        """
        Initialize preference service.

        Args:
            preference_file: Path to preference file (default: ~/.smart-fork/preferences.json)
            max_entries: Maximum number of entries to keep (default: 1000)
        """
        if preference_file is None:
            home = Path.home()
            storage_dir = home / ".smart-fork"
            storage_dir.mkdir(parents=True, exist_ok=True)
            preference_file = str(storage_dir / "preferences.json")

        self.preference_file = Path(preference_file)
        self.max_entries = max_entries
        self.lock = threading.Lock()

        # Create empty file if it doesn't exist
        if not self.preference_file.exists():
            self._save([])
            logger.info(f"Created preference file: {self.preference_file}")

    def record_selection(
        self,
        session_id: str,
        query: str,
        position: int = -1
    ) -> None:
        """
        Record a user selection event.

        Args:
            session_id: The session that was selected/forked
            query: The search query that led to this selection
            position: Result position (1-5 for displayed, -1 for custom)
        """
        with self.lock:
            timestamp = datetime.utcnow().isoformat() + "Z"
            entry = PreferenceEntry(
                session_id=session_id,
                query=query,
                position=position,
                timestamp=timestamp
            )

            # Load existing preferences
            preferences = self._load()

            # Add new entry at the beginning (most recent first)
            preferences.insert(0, entry)

            # Trim to max_entries
            if len(preferences) > self.max_entries:
                preferences = preferences[:self.max_entries]

            # Save back to file
            self._save(preferences)
            logger.info(f"Recorded selection: session={session_id}, query={query[:30]}..., position={position}")

    def calculate_preference_boost(
        self,
        session_id: str,
        query: Optional[str] = None,
        current_time: Optional[datetime] = None
    ) -> PreferenceScore:
        """
        Calculate preference boost for a session.

        Args:
            session_id: The session to calculate boost for
            query: Optional query context for query-aware boosting
            current_time: Current time for recency calculation (defaults to now)

        Returns:
            PreferenceScore with boost value and breakdown
        """
        with self.lock:
            preferences = self._load()

            # Filter to this session
            session_prefs = [p for p in preferences if p.session_id == session_id]

            if not session_prefs:
                # No history for this session - no boost
                return PreferenceScore(
                    session_id=session_id,
                    preference_boost=0.0,
                    fork_count=0,
                    avg_position=0.0,
                    recency_weight=0.0
                )

            # Calculate fork count
            fork_count = len(session_prefs)

            # Calculate average position (ignore -1 custom entries)
            valid_positions = [p.position for p in session_prefs if p.position > 0]
            avg_position = sum(valid_positions) / len(valid_positions) if valid_positions else 0.0

            # Calculate recency weight from most recent fork
            if current_time is None:
                current_time = datetime.utcnow()

            most_recent = session_prefs[0]  # Already sorted by timestamp descending
            recency_weight = self._calculate_recency_weight(most_recent.timestamp, current_time)

            # Calculate base boost from fork count (capped)
            fork_boost = min(fork_count * self.BOOST_PER_FORK, self.MAX_FORK_BOOST)

            # Calculate position quality bonus (lower position = higher bonus)
            # Position 1 = +0.01, Position 2 = +0.008, etc.
            position_bonus = 0.0
            if avg_position > 0:
                position_bonus = self.POSITION_WEIGHT * (6 - avg_position) / 5.0

            # Apply recency weight to the combined boost
            preference_boost = (fork_boost + position_bonus) * recency_weight

            logger.debug(
                f"Preference boost for {session_id}: "
                f"fork_count={fork_count}, avg_pos={avg_position:.1f}, "
                f"recency={recency_weight:.2f}, boost={preference_boost:.4f}"
            )

            return PreferenceScore(
                session_id=session_id,
                preference_boost=preference_boost,
                fork_count=fork_count,
                avg_position=avg_position,
                recency_weight=recency_weight
            )

    def calculate_preference_boosts(
        self,
        session_ids: List[str],
        query: Optional[str] = None,
        current_time: Optional[datetime] = None
    ) -> Dict[str, PreferenceScore]:
        """
        Calculate preference boosts for multiple sessions (batch operation).

        Args:
            session_ids: List of session IDs to calculate boosts for
            query: Optional query context
            current_time: Current time for recency calculation

        Returns:
            Dictionary mapping session_id -> PreferenceScore
        """
        return {
            session_id: self.calculate_preference_boost(session_id, query, current_time)
            for session_id in session_ids
        }

    def get_most_forked_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequently forked sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session statistics (session_id, fork_count, last_forked)
        """
        with self.lock:
            preferences = self._load()

            # Count forks per session
            session_counts: Dict[str, List[PreferenceEntry]] = {}
            for pref in preferences:
                if pref.session_id not in session_counts:
                    session_counts[pref.session_id] = []
                session_counts[pref.session_id].append(pref)

            # Build results
            results = []
            for session_id, prefs in session_counts.items():
                results.append({
                    'session_id': session_id,
                    'fork_count': len(prefs),
                    'last_forked': prefs[0].timestamp,  # Most recent
                    'avg_position': sum(p.position for p in prefs if p.position > 0) / len([p for p in prefs if p.position > 0]) if any(p.position > 0 for p in prefs) else 0.0
                })

            # Sort by fork count descending
            results.sort(key=lambda x: x['fork_count'], reverse=True)

            return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about preference learning.

        Returns:
            Dictionary with stats like total_selections, unique_sessions, etc.
        """
        with self.lock:
            preferences = self._load()

            if not preferences:
                return {
                    'total_selections': 0,
                    'unique_sessions': 0,
                    'position_distribution': {},
                    'most_recent_selection': None
                }

            session_ids = {p.session_id for p in preferences}
            position_counts = {}
            for pref in preferences:
                pos = pref.position
                position_counts[pos] = position_counts.get(pos, 0) + 1

            return {
                'total_selections': len(preferences),
                'unique_sessions': len(session_ids),
                'position_distribution': position_counts,
                'most_recent_selection': preferences[0].timestamp if preferences else None
            }

    def clear(self) -> None:
        """Clear all preference data (use with caution)."""
        with self.lock:
            self._save([])
            logger.info("Cleared preference data")

    def _calculate_recency_weight(
        self,
        timestamp: str,
        current_time: datetime
    ) -> float:
        """
        Calculate recency weight using exponential decay.

        Uses the formula: weight = exp(-age_in_days / RECENCY_DECAY_DAYS)

        Args:
            timestamp: ISO format timestamp string
            current_time: Current time

        Returns:
            Recency weight between 0.0 and 1.0
        """
        try:
            last_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            # Make current_time timezone-aware if it isn't already
            if current_time.tzinfo is None:
                # Assume UTC if no timezone info
                from datetime import timezone
                current_time = current_time.replace(tzinfo=timezone.utc)

            age_seconds = (current_time - last_time).total_seconds()
            age_days = age_seconds / (24 * 60 * 60)

            # Exponential decay
            weight = math.exp(-age_days / self.RECENCY_DECAY_DAYS)
            return weight

        except (ValueError, AttributeError):
            # Failed to parse timestamp - return minimum weight
            return 0.0

    def _load(self) -> List[PreferenceEntry]:
        """Load preferences from file."""
        try:
            if not self.preference_file.exists():
                return []

            with open(self.preference_file, 'r') as f:
                data = json.load(f)
                return [PreferenceEntry.from_dict(entry) for entry in data]
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
            return []

    def _save(self, preferences: List[PreferenceEntry]) -> None:
        """Save preferences to file."""
        try:
            with open(self.preference_file, 'w') as f:
                data = [entry.to_dict() for entry in preferences]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
