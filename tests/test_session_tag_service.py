"""
Tests for SessionTagService.
"""

import pytest
import tempfile
import os
from pathlib import Path

from smart_fork.session_registry import SessionRegistry, SessionMetadata
from smart_fork.session_tag_service import SessionTagService


@pytest.fixture
def temp_registry_path():
    """Create a temporary registry file."""
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def session_registry(temp_registry_path):
    """Create a SessionRegistry with test data."""
    registry = SessionRegistry(registry_path=temp_registry_path)

    # Add some test sessions
    registry.add_session("session-001", SessionMetadata(
        session_id="session-001",
        project="test-project-1",
        tags=["python", "testing"]
    ))

    registry.add_session("session-002", SessionMetadata(
        session_id="session-002",
        project="test-project-2",
        tags=["javascript", "react"]
    ))

    registry.add_session("session-003", SessionMetadata(
        session_id="session-003",
        project="test-project-1",
        tags=[]  # No tags
    ))

    return registry


@pytest.fixture
def tag_service(session_registry):
    """Create a SessionTagService."""
    return SessionTagService(session_registry)


class TestSessionTagService:
    """Tests for SessionTagService."""

    def test_add_tag_success(self, tag_service):
        """Test adding a tag to a session."""
        result = tag_service.add_tag("session-003", "bug-fix")
        assert result is True

        tags = tag_service.get_session_tags("session-003")
        assert "bug-fix" in tags

    def test_add_tag_normalizes_case(self, tag_service):
        """Test that tags are normalized to lowercase."""
        result = tag_service.add_tag("session-003", "BUG-FIX")
        assert result is True

        tags = tag_service.get_session_tags("session-003")
        assert "bug-fix" in tags
        assert "BUG-FIX" not in tags

    def test_add_tag_duplicate(self, tag_service):
        """Test adding a duplicate tag returns False."""
        tag_service.add_tag("session-001", "testing")  # Already exists
        result = tag_service.add_tag("session-001", "testing")
        assert result is False

    def test_add_tag_empty_string(self, tag_service):
        """Test adding empty tag returns False."""
        result = tag_service.add_tag("session-001", "")
        assert result is False

    def test_add_tag_whitespace_only(self, tag_service):
        """Test adding whitespace-only tag returns False."""
        result = tag_service.add_tag("session-001", "   ")
        assert result is False

    def test_add_tag_nonexistent_session(self, tag_service):
        """Test adding tag to nonexistent session returns False."""
        result = tag_service.add_tag("session-999", "test")
        assert result is False

    def test_remove_tag_success(self, tag_service):
        """Test removing a tag from a session."""
        result = tag_service.remove_tag("session-001", "python")
        assert result is True

        tags = tag_service.get_session_tags("session-001")
        assert "python" not in tags
        assert "testing" in tags  # Other tag still present

    def test_remove_tag_case_insensitive(self, tag_service):
        """Test removing tag is case-insensitive."""
        result = tag_service.remove_tag("session-001", "PYTHON")
        assert result is True

        tags = tag_service.get_session_tags("session-001")
        assert "python" not in tags

    def test_remove_tag_not_found(self, tag_service):
        """Test removing non-existent tag returns False."""
        result = tag_service.remove_tag("session-001", "nonexistent")
        assert result is False

    def test_remove_tag_nonexistent_session(self, tag_service):
        """Test removing tag from nonexistent session returns False."""
        result = tag_service.remove_tag("session-999", "test")
        assert result is False

    def test_get_session_tags_success(self, tag_service):
        """Test getting tags for a session."""
        tags = tag_service.get_session_tags("session-001")
        assert tags == ["python", "testing"]

    def test_get_session_tags_empty(self, tag_service):
        """Test getting tags for session with no tags."""
        tags = tag_service.get_session_tags("session-003")
        assert tags == []

    def test_get_session_tags_nonexistent(self, tag_service):
        """Test getting tags for nonexistent session returns None."""
        tags = tag_service.get_session_tags("session-999")
        assert tags is None

    def test_list_all_tags(self, tag_service):
        """Test listing all tags with counts."""
        all_tags = tag_service.list_all_tags()

        # Should have 4 unique tags
        assert len(all_tags) == 4

        # Check format
        assert all(isinstance(t, dict) for t in all_tags)
        assert all("tag" in t and "count" in t for t in all_tags)

        # Check sorted by count descending
        tag_dict = {t["tag"]: t["count"] for t in all_tags}
        assert tag_dict["python"] == 1
        assert tag_dict["testing"] == 1
        assert tag_dict["javascript"] == 1
        assert tag_dict["react"] == 1

    def test_list_all_tags_with_multiple_uses(self, tag_service):
        """Test listing tags when a tag is used on multiple sessions."""
        # Add "python" tag to session-002
        tag_service.add_tag("session-002", "python")

        all_tags = tag_service.list_all_tags()
        tag_dict = {t["tag"]: t["count"] for t in all_tags}

        # Python should now have count of 2
        assert tag_dict["python"] == 2

        # Check sorted by count - python should be first
        assert all_tags[0]["tag"] == "python"
        assert all_tags[0]["count"] == 2

    def test_find_sessions_by_tag(self, tag_service):
        """Test finding sessions by tag."""
        sessions = tag_service.find_sessions_by_tag("python")
        assert sessions == ["session-001"]

    def test_find_sessions_by_tag_case_insensitive(self, tag_service):
        """Test finding sessions by tag is case-insensitive."""
        sessions = tag_service.find_sessions_by_tag("PYTHON")
        assert sessions == ["session-001"]

    def test_find_sessions_by_tag_not_found(self, tag_service):
        """Test finding sessions by non-existent tag returns empty list."""
        sessions = tag_service.find_sessions_by_tag("nonexistent")
        assert sessions == []

    def test_find_sessions_by_tags_any_match(self, tag_service):
        """Test finding sessions matching any of the given tags."""
        sessions = tag_service.find_sessions_by_tags(["python", "react"], match_all=False)
        assert set(sessions) == {"session-001", "session-002"}

    def test_find_sessions_by_tags_all_match(self, tag_service):
        """Test finding sessions matching all of the given tags."""
        # Add "testing" to session-002
        tag_service.add_tag("session-002", "testing")

        # Now find sessions with both "testing" and "react"
        sessions = tag_service.find_sessions_by_tags(["testing", "react"], match_all=True)
        assert sessions == ["session-002"]

    def test_find_sessions_by_tags_empty_list(self, tag_service):
        """Test finding sessions with empty tag list returns empty."""
        sessions = tag_service.find_sessions_by_tags([], match_all=False)
        assert sessions == []

    def test_suggest_tags(self, tag_service):
        """Test suggesting tags for a session."""
        # Session-003 has no tags, should suggest from common tags
        suggestions = tag_service.suggest_tags("session-003")

        # Should suggest tags from other sessions
        assert len(suggestions) <= 5
        assert all(tag in ["python", "testing", "javascript", "react"] for tag in suggestions)

    def test_suggest_tags_excludes_current(self, tag_service):
        """Test suggestions exclude tags already on the session."""
        # Session-001 has "python" and "testing"
        suggestions = tag_service.suggest_tags("session-001")

        # Suggestions should not include existing tags
        assert "python" not in suggestions
        assert "testing" not in suggestions

    def test_suggest_tags_max_suggestions(self, tag_service):
        """Test max_suggestions parameter limits results."""
        # Add many tags to other sessions
        for i in range(10):
            tag_service.add_tag("session-002", f"tag-{i}")

        suggestions = tag_service.suggest_tags("session-001", max_suggestions=3)
        assert len(suggestions) <= 3

    def test_suggest_tags_nonexistent_session(self, tag_service):
        """Test suggesting tags for nonexistent session returns empty."""
        suggestions = tag_service.suggest_tags("session-999")
        assert suggestions == []

    def test_get_stats(self, tag_service):
        """Test getting tagging statistics."""
        stats = tag_service.get_stats()

        assert stats["total_sessions"] == 3
        assert stats["tagged_sessions"] == 2
        assert stats["untagged_sessions"] == 1
        assert stats["total_tags"] == 4
        assert stats["unique_tags"] == 4
        assert stats["avg_tags_per_session"] == pytest.approx(4/3)
        assert len(stats["top_tags"]) <= 5

    def test_get_stats_no_sessions(self, temp_registry_path):
        """Test stats with empty registry."""
        registry = SessionRegistry(registry_path=temp_registry_path)
        tag_service = SessionTagService(registry)

        stats = tag_service.get_stats()

        assert stats["total_sessions"] == 0
        assert stats["tagged_sessions"] == 0
        assert stats["untagged_sessions"] == 0
        assert stats["total_tags"] == 0
        assert stats["unique_tags"] == 0
        assert stats["avg_tags_per_session"] == 0
        assert stats["top_tags"] == []

    def test_add_multiple_tags_to_session(self, tag_service):
        """Test adding multiple tags to a session."""
        tag_service.add_tag("session-003", "bug-fix")
        tag_service.add_tag("session-003", "urgent")
        tag_service.add_tag("session-003", "production")

        tags = tag_service.get_session_tags("session-003")
        assert len(tags) == 3
        assert "bug-fix" in tags
        assert "urgent" in tags
        assert "production" in tags

    def test_tags_persist_across_service_instances(self, session_registry, tag_service):
        """Test that tag changes persist when creating new service instance."""
        # Add tag with first service
        tag_service.add_tag("session-003", "persistent")

        # Create new service instance with same registry
        new_service = SessionTagService(session_registry)

        # Tag should still be there
        tags = new_service.get_session_tags("session-003")
        assert "persistent" in tags

    def test_whitespace_trimming(self, tag_service):
        """Test that whitespace is trimmed from tags."""
        tag_service.add_tag("session-003", "  whitespace-test  ")

        tags = tag_service.get_session_tags("session-003")
        assert "whitespace-test" in tags
        assert "  whitespace-test  " not in tags
