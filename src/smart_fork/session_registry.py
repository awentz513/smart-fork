"""
Session registry manager for tracking indexed Claude Code sessions.

This module provides a JSON-based registry for tracking session metadata,
including project information, timestamps, chunk counts, and synchronization status.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class SessionMetadata:
    """Represents metadata for a single session."""
    session_id: str
    project: Optional[str] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    last_synced: Optional[str] = None
    chunk_count: int = 0
    message_count: int = 0
    tags: List[str] = None
    summary: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """Create SessionMetadata from dictionary."""
        return cls(**data)


class SessionRegistry:
    """
    Manages a registry of indexed sessions with metadata.

    Provides thread-safe methods for tracking session information including
    project names, timestamps, chunk counts, and tags in a JSON file.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the SessionRegistry.

        Args:
            registry_path: Path to the registry JSON file.
                         Defaults to ~/.smart-fork/session-registry.json
        """
        if registry_path is None:
            home = os.path.expanduser("~")
            registry_path = os.path.join(home, ".smart-fork", "session-registry.json")

        self.registry_path = registry_path
        self._lock = threading.Lock()

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

        # Load existing registry or create new one
        self._sessions: Dict[str, SessionMetadata] = {}
        self._load()

    def _load(self):
        """Load registry from JSON file."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._sessions = {
                        session_id: SessionMetadata.from_dict(session_data)
                        for session_id, session_data in data.get('sessions', {}).items()
                    }
            except (json.JSONDecodeError, IOError) as e:
                # If registry is corrupted, start fresh
                self._sessions = {}

    def _save(self):
        """Save registry to JSON file."""
        data = {
            'sessions': {
                session_id: metadata.to_dict()
                for session_id, metadata in self._sessions.items()
            },
            'last_updated': datetime.utcnow().isoformat()
        }

        # Write to temporary file first, then rename for atomic write
        temp_path = self.registry_path + '.tmp'
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.registry_path)
        except IOError:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Get session metadata by ID.

        Args:
            session_id: The session identifier

        Returns:
            SessionMetadata if found, None otherwise
        """
        with self._lock:
            return self._sessions.get(session_id)

    def add_session(self, session_id: str, metadata: Optional[SessionMetadata] = None) -> SessionMetadata:
        """
        Add a new session to the registry.

        Args:
            session_id: The session identifier
            metadata: Optional SessionMetadata. If None, creates default metadata.

        Returns:
            The added SessionMetadata
        """
        with self._lock:
            if metadata is None:
                metadata = SessionMetadata(session_id=session_id)
            else:
                metadata.session_id = session_id

            self._sessions[session_id] = metadata
            self._save()
            return metadata

    def update_session(self, session_id: str, **kwargs) -> Optional[SessionMetadata]:
        """
        Update session metadata.

        Args:
            session_id: The session identifier
            **kwargs: Fields to update (project, created_at, last_modified, etc.)

        Returns:
            Updated SessionMetadata if session exists, None otherwise
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            # Update fields
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)

            self._save()
            return session

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the registry.

        Args:
            session_id: The session identifier

        Returns:
            True if session was deleted, False if it didn't exist
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._save()
                return True
            return False

    def list_sessions(
        self,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[SessionMetadata]:
        """
        List all sessions, optionally filtered by project or tags.

        Args:
            project: Filter by project name
            tags: Filter by tags (sessions must have at least one matching tag)

        Returns:
            List of SessionMetadata matching the filters
        """
        with self._lock:
            sessions = list(self._sessions.values())

            if project is not None:
                sessions = [s for s in sessions if s.project == project]

            if tags is not None:
                sessions = [
                    s for s in sessions
                    if any(tag in s.tags for tag in tags)
                ]

            return sessions

    def get_all_sessions(self) -> Dict[str, SessionMetadata]:
        """
        Get all sessions as a dictionary.

        Returns:
            Dictionary mapping session_id to SessionMetadata
        """
        with self._lock:
            return dict(self._sessions)

    def set_last_synced(self, session_id: str, timestamp: Optional[str] = None) -> bool:
        """
        Update the last_synced timestamp for a session.

        Args:
            session_id: The session identifier
            timestamp: ISO format timestamp. If None, uses current UTC time.

        Returns:
            True if successful, False if session doesn't exist
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        result = self.update_session(session_id, last_synced=timestamp)
        return result is not None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics about the registry
        """
        with self._lock:
            total_sessions = len(self._sessions)
            total_chunks = sum(s.chunk_count for s in self._sessions.values())
            total_messages = sum(s.message_count for s in self._sessions.values())

            projects = set()
            for session in self._sessions.values():
                if session.project:
                    projects.add(session.project)

            return {
                'total_sessions': total_sessions,
                'total_chunks': total_chunks,
                'total_messages': total_messages,
                'total_projects': len(projects),
                'projects': sorted(projects)
            }

    def clear(self):
        """
        Clear all sessions from the registry.

        WARNING: This will delete all session metadata.
        """
        with self._lock:
            self._sessions = {}
            self._save()
