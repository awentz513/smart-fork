"""Fork history tracking service for Smart Fork Detection.

This module tracks when users fork from sessions, storing:
- session_id: Which session was forked
- timestamp: When the fork occurred
- query: What query led to this fork
- position: Which result position was selected (1-5, or -1 for custom)

This enables:
1. Quick access to recently forked sessions
2. Learning user preferences over time
3. Understanding which queries lead to successful forks
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class ForkHistoryEntry:
    """Represents a single fork event."""
    session_id: str
    timestamp: str  # ISO format
    query: str
    position: int  # 1-5 for displayed results, -1 for custom entry

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForkHistoryEntry':
        """Create from dictionary."""
        return cls(**data)


class ForkHistoryService:
    """
    Service for tracking fork history.

    Thread-safe implementation with file-based persistence.
    """

    def __init__(self, history_file: Optional[str] = None, max_entries: int = 100):
        """
        Initialize fork history service.

        Args:
            history_file: Path to history file (default: ~/.smart-fork/fork_history.json)
            max_entries: Maximum number of entries to keep (default: 100)
        """
        if history_file is None:
            home = Path.home()
            storage_dir = home / ".smart-fork"
            storage_dir.mkdir(parents=True, exist_ok=True)
            history_file = str(storage_dir / "fork_history.json")

        self.history_file = Path(history_file)
        self.max_entries = max_entries
        self.lock = threading.Lock()

        # Create empty file if it doesn't exist
        if not self.history_file.exists():
            self._save([])
            logger.info(f"Created fork history file: {self.history_file}")

    def record_fork(
        self,
        session_id: str,
        query: str,
        position: int = -1
    ) -> None:
        """
        Record a fork event.

        Args:
            session_id: The session that was forked
            query: The search query that led to this fork
            position: Result position (1-5 for displayed, -1 for custom)
        """
        with self.lock:
            timestamp = datetime.utcnow().isoformat() + "Z"
            entry = ForkHistoryEntry(
                session_id=session_id,
                timestamp=timestamp,
                query=query,
                position=position
            )

            # Load existing history
            history = self._load()

            # Add new entry at the beginning (most recent first)
            history.insert(0, entry)

            # Trim to max_entries
            if len(history) > self.max_entries:
                history = history[:self.max_entries]

            # Save back to file
            self._save(history)
            logger.info(f"Recorded fork: session={session_id}, query={query[:50]}...")

    def get_recent_forks(self, limit: int = 10) -> List[ForkHistoryEntry]:
        """
        Get the most recent fork events.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of fork history entries (most recent first)
        """
        with self.lock:
            history = self._load()
            return history[:limit]

    def get_forks_for_session(self, session_id: str) -> List[ForkHistoryEntry]:
        """
        Get all fork events for a specific session.

        Args:
            session_id: The session to query

        Returns:
            List of fork history entries for this session
        """
        with self.lock:
            history = self._load()
            return [entry for entry in history if entry.session_id == session_id]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about fork history.

        Returns:
            Dictionary with stats like total_forks, unique_sessions, etc.
        """
        with self.lock:
            history = self._load()

            if not history:
                return {
                    'total_forks': 0,
                    'unique_sessions': 0,
                    'position_distribution': {}
                }

            session_ids = {entry.session_id for entry in history}
            position_counts = {}
            for entry in history:
                pos = entry.position
                position_counts[pos] = position_counts.get(pos, 0) + 1

            return {
                'total_forks': len(history),
                'unique_sessions': len(session_ids),
                'position_distribution': position_counts,
                'most_recent': history[0].timestamp if history else None
            }

    def clear(self) -> None:
        """Clear all fork history (use with caution)."""
        with self.lock:
            self._save([])
            logger.info("Cleared fork history")

    def _load(self) -> List[ForkHistoryEntry]:
        """Load history from file."""
        try:
            if not self.history_file.exists():
                return []

            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return [ForkHistoryEntry.from_dict(entry) for entry in data]
        except Exception as e:
            logger.error(f"Error loading fork history: {e}")
            return []

    def _save(self, history: List[ForkHistoryEntry]) -> None:
        """Save history to file."""
        try:
            with open(self.history_file, 'w') as f:
                data = [entry.to_dict() for entry in history]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving fork history: {e}")
