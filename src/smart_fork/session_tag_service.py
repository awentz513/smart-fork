"""
Session tagging service for organizing and categorizing sessions.

This module provides functionality to add, remove, and manage tags on sessions,
enabling better organization and filtering of session data.
"""

import logging
from typing import List, Optional, Set, Dict, Any
from .session_registry import SessionRegistry

logger = logging.getLogger(__name__)


class SessionTagService:
    """
    Service for managing session tags.

    Provides methods to add, remove, list, and suggest tags for sessions.
    Works in conjunction with SessionRegistry to persist tag data.
    """

    def __init__(self, session_registry: SessionRegistry):
        """
        Initialize the SessionTagService.

        Args:
            session_registry: SessionRegistry instance for accessing session metadata
        """
        self.session_registry = session_registry

    def add_tag(self, session_id: str, tag: str) -> bool:
        """
        Add a tag to a session.

        Args:
            session_id: The session identifier
            tag: Tag to add (will be normalized to lowercase)

        Returns:
            True if tag was added, False if session doesn't exist or tag already present
        """
        # Normalize tag to lowercase
        tag = tag.strip().lower()

        if not tag:
            logger.warning("Attempted to add empty tag")
            return False

        # Get current session metadata
        session = self.session_registry.get_session(session_id)
        if session is None:
            logger.warning(f"Session {session_id} not found")
            return False

        # Check if tag already exists
        if tag in session.tags:
            logger.info(f"Tag '{tag}' already exists on session {session_id}")
            return False

        # Add tag
        updated_tags = session.tags + [tag]
        result = self.session_registry.update_session(session_id, tags=updated_tags)

        if result:
            logger.info(f"Added tag '{tag}' to session {session_id}")
            return True
        return False

    def remove_tag(self, session_id: str, tag: str) -> bool:
        """
        Remove a tag from a session.

        Args:
            session_id: The session identifier
            tag: Tag to remove (case-insensitive)

        Returns:
            True if tag was removed, False if session doesn't exist or tag not present
        """
        # Normalize tag to lowercase
        tag = tag.strip().lower()

        # Get current session metadata
        session = self.session_registry.get_session(session_id)
        if session is None:
            logger.warning(f"Session {session_id} not found")
            return False

        # Check if tag exists
        if tag not in session.tags:
            logger.info(f"Tag '{tag}' not found on session {session_id}")
            return False

        # Remove tag
        updated_tags = [t for t in session.tags if t != tag]
        result = self.session_registry.update_session(session_id, tags=updated_tags)

        if result:
            logger.info(f"Removed tag '{tag}' from session {session_id}")
            return True
        return False

    def get_session_tags(self, session_id: str) -> Optional[List[str]]:
        """
        Get all tags for a session.

        Args:
            session_id: The session identifier

        Returns:
            List of tags if session exists, None otherwise
        """
        session = self.session_registry.get_session(session_id)
        if session is None:
            return None
        return session.tags

    def list_all_tags(self) -> List[Dict[str, Any]]:
        """
        List all unique tags across all sessions with usage counts.

        Returns:
            List of dicts with 'tag' and 'count' keys, sorted by count descending
        """
        tag_counts: Dict[str, int] = {}

        all_sessions = self.session_registry.get_all_sessions()
        for session in all_sessions.values():
            for tag in session.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Convert to list of dicts and sort by count descending
        tag_list = [
            {"tag": tag, "count": count}
            for tag, count in tag_counts.items()
        ]
        tag_list.sort(key=lambda x: (-x["count"], x["tag"]))

        return tag_list

    def find_sessions_by_tag(self, tag: str) -> List[str]:
        """
        Find all session IDs that have a specific tag.

        Args:
            tag: Tag to search for (case-insensitive)

        Returns:
            List of session IDs with the tag
        """
        tag = tag.strip().lower()
        sessions = self.session_registry.list_sessions(tags=[tag])
        return [s.session_id for s in sessions]

    def find_sessions_by_tags(self, tags: List[str], match_all: bool = False) -> List[str]:
        """
        Find sessions that match given tags.

        Args:
            tags: List of tags to search for (case-insensitive)
            match_all: If True, session must have all tags. If False, any tag matches.

        Returns:
            List of session IDs matching the criteria
        """
        if not tags:
            return []

        # Normalize tags
        normalized_tags = [t.strip().lower() for t in tags]

        all_sessions = self.session_registry.get_all_sessions()
        matching_sessions = []

        for session_id, session in all_sessions.items():
            session_tags_set = set(session.tags)

            if match_all:
                # All tags must be present
                if all(tag in session_tags_set for tag in normalized_tags):
                    matching_sessions.append(session_id)
            else:
                # Any tag present
                if any(tag in session_tags_set for tag in normalized_tags):
                    matching_sessions.append(session_id)

        return matching_sessions

    def suggest_tags(self, session_id: str, max_suggestions: int = 5) -> List[str]:
        """
        Suggest tags for a session based on existing tag patterns.

        This is a simple implementation that suggests the most common tags
        that aren't already on the session.

        Args:
            session_id: The session identifier
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggested tag names
        """
        # Get current session tags
        current_tags = self.get_session_tags(session_id)
        if current_tags is None:
            return []

        current_tags_set = set(current_tags)

        # Get all tags with counts
        all_tags = self.list_all_tags()

        # Filter out tags already on this session
        suggestions = [
            tag_info["tag"]
            for tag_info in all_tags
            if tag_info["tag"] not in current_tags_set
        ]

        # Return top N suggestions
        return suggestions[:max_suggestions]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tagging statistics.

        Returns:
            Dictionary with statistics about tag usage
        """
        all_tags = self.list_all_tags()
        all_sessions = self.session_registry.get_all_sessions()

        total_sessions = len(all_sessions)
        tagged_sessions = sum(1 for s in all_sessions.values() if s.tags)
        total_tags = sum(len(s.tags) for s in all_sessions.values())
        unique_tags = len(all_tags)

        # Most used tags (top 5)
        top_tags = all_tags[:5] if all_tags else []

        return {
            "total_sessions": total_sessions,
            "tagged_sessions": tagged_sessions,
            "untagged_sessions": total_sessions - tagged_sessions,
            "total_tags": total_tags,
            "unique_tags": unique_tags,
            "avg_tags_per_session": total_tags / total_sessions if total_sessions > 0 else 0,
            "top_tags": top_tags
        }
