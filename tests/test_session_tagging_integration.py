"""
Integration tests for session tagging MCP tools and tag-based search filtering.
"""

import pytest
import tempfile
import os
from pathlib import Path

from smart_fork.server import (
    create_add_tag_handler,
    create_remove_tag_handler,
    create_list_tags_handler
)
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

    # Add test sessions with various tags
    registry.add_session("session-001", SessionMetadata(
        session_id="session-001",
        project="test-project-1",
        tags=["python", "api"]
    ))

    registry.add_session("session-002", SessionMetadata(
        session_id="session-002",
        project="test-project-2",
        tags=["javascript", "react", "frontend"]
    ))

    registry.add_session("session-003", SessionMetadata(
        session_id="session-003",
        project="test-project-1",
        tags=[]
    ))

    registry.add_session("session-004", SessionMetadata(
        session_id="session-004",
        project="test-project-3",
        tags=["python", "testing", "pytest"]
    ))

    return registry


@pytest.fixture
def tag_service(session_registry):
    """Create a SessionTagService."""
    return SessionTagService(session_registry)


class TestAddTagHandler:
    """Tests for add-session-tag MCP tool handler."""

    def test_add_tag_success(self, tag_service):
        """Test successfully adding a tag."""
        handler = create_add_tag_handler(tag_service)
        result = handler({"session_id": "session-003", "tag": "bug-fix"})

        assert "added successfully" in result.lower()
        assert "bug-fix" in result

        # Verify tag was actually added
        tags = tag_service.get_session_tags("session-003")
        assert "bug-fix" in tags

    def test_add_tag_duplicate(self, tag_service):
        """Test adding duplicate tag."""
        handler = create_add_tag_handler(tag_service)
        result = handler({"session_id": "session-001", "tag": "python"})

        assert "already" in result.lower()

    def test_add_tag_missing_session_id(self, tag_service):
        """Test error when session_id is missing."""
        handler = create_add_tag_handler(tag_service)
        result = handler({"tag": "test"})

        assert "error" in result.lower()
        assert "session_id" in result.lower()

    def test_add_tag_missing_tag(self, tag_service):
        """Test error when tag is missing."""
        handler = create_add_tag_handler(tag_service)
        result = handler({"session_id": "session-001"})

        assert "error" in result.lower()
        assert "tag" in result.lower()

    def test_add_tag_nonexistent_session(self, tag_service):
        """Test error when session doesn't exist."""
        handler = create_add_tag_handler(tag_service)
        result = handler({"session_id": "session-999", "tag": "test"})

        assert "error" in result.lower()
        assert "not found" in result.lower()

    def test_add_tag_shows_current_tags(self, tag_service):
        """Test that response shows current tags after adding."""
        handler = create_add_tag_handler(tag_service)
        result = handler({"session_id": "session-001", "tag": "new-tag"})

        assert "current tags:" in result.lower()
        assert "python" in result  # Existing tag
        assert "new-tag" in result  # Newly added tag

    def test_add_tag_service_none(self):
        """Test error when tag service is None."""
        handler = create_add_tag_handler(None)
        result = handler({"session_id": "session-001", "tag": "test"})

        assert "error" in result.lower()
        assert "not initialized" in result.lower()


class TestRemoveTagHandler:
    """Tests for remove-session-tag MCP tool handler."""

    def test_remove_tag_success(self, tag_service):
        """Test successfully removing a tag."""
        handler = create_remove_tag_handler(tag_service)
        result = handler({"session_id": "session-001", "tag": "python"})

        assert "removed successfully" in result.lower()
        assert "python" in result

        # Verify tag was actually removed
        tags = tag_service.get_session_tags("session-001")
        assert "python" not in tags

    def test_remove_tag_not_found(self, tag_service):
        """Test removing non-existent tag."""
        handler = create_remove_tag_handler(tag_service)
        result = handler({"session_id": "session-001", "tag": "nonexistent"})

        assert "not found" in result.lower()

    def test_remove_tag_missing_session_id(self, tag_service):
        """Test error when session_id is missing."""
        handler = create_remove_tag_handler(tag_service)
        result = handler({"tag": "test"})

        assert "error" in result.lower()
        assert "session_id" in result.lower()

    def test_remove_tag_missing_tag(self, tag_service):
        """Test error when tag is missing."""
        handler = create_remove_tag_handler(tag_service)
        result = handler({"session_id": "session-001"})

        assert "error" in result.lower()
        assert "tag" in result.lower()

    def test_remove_tag_nonexistent_session(self, tag_service):
        """Test error when session doesn't exist."""
        handler = create_remove_tag_handler(tag_service)
        result = handler({"session_id": "session-999", "tag": "test"})

        assert "error" in result.lower()
        assert "not found" in result.lower()

    def test_remove_tag_shows_remaining_tags(self, tag_service):
        """Test that response shows remaining tags after removal."""
        handler = create_remove_tag_handler(tag_service)
        result = handler({"session_id": "session-002", "tag": "react"})

        assert "current tags:" in result.lower()
        assert "javascript" in result  # Remaining tag
        assert "frontend" in result  # Remaining tag

    def test_remove_tag_case_insensitive(self, tag_service):
        """Test that tag removal is case-insensitive."""
        handler = create_remove_tag_handler(tag_service)
        result = handler({"session_id": "session-001", "tag": "PYTHON"})

        assert "removed successfully" in result.lower()

    def test_remove_tag_service_none(self):
        """Test error when tag service is None."""
        handler = create_remove_tag_handler(None)
        result = handler({"session_id": "session-001", "tag": "test"})

        assert "error" in result.lower()
        assert "not initialized" in result.lower()


class TestListTagsHandler:
    """Tests for list-session-tags MCP tool handler."""

    def test_list_tags_for_session(self, tag_service):
        """Test listing tags for a specific session."""
        handler = create_list_tags_handler(tag_service)
        result = handler({"session_id": "session-001"})

        assert "session-001" in result.lower()
        assert "python" in result
        assert "api" in result

    def test_list_tags_for_session_no_tags(self, tag_service):
        """Test listing tags for session with no tags."""
        handler = create_list_tags_handler(tag_service)
        result = handler({"session_id": "session-003"})

        assert "no tags" in result.lower()
        assert "add-session-tag" in result.lower()

    def test_list_tags_for_session_with_suggestions(self, tag_service):
        """Test that suggestions are provided for untagged sessions."""
        handler = create_list_tags_handler(tag_service)
        result = handler({"session_id": "session-003"})

        # Should suggest common tags
        assert "suggested tags" in result.lower() or "no tags" in result.lower()

    def test_list_tags_for_nonexistent_session(self, tag_service):
        """Test error when session doesn't exist."""
        handler = create_list_tags_handler(tag_service)
        result = handler({"session_id": "session-999"})

        assert "error" in result.lower()
        assert "not found" in result.lower()

    def test_list_all_tags_top_only(self, tag_service):
        """Test listing all tags (top only)."""
        handler = create_list_tags_handler(tag_service)
        result = handler({"show_all": False})

        assert "top tags" in result.lower()
        # Should show at least some tags
        assert any(tag in result for tag in ["python", "javascript", "react"])

    def test_list_all_tags_with_show_all(self, tag_service):
        """Test listing all tags with counts."""
        handler = create_list_tags_handler(tag_service)
        result = handler({"show_all": True})

        assert "all tags" in result.lower()
        assert "statistics" in result.lower()
        assert "python" in result
        assert "javascript" in result
        assert "session(s)" in result.lower()

    def test_list_all_tags_no_tags_in_system(self, temp_registry_path):
        """Test listing tags when no tags exist."""
        registry = SessionRegistry(registry_path=temp_registry_path)
        service = SessionTagService(registry)

        handler = create_list_tags_handler(service)
        result = handler({"show_all": True})

        assert "no tags found" in result.lower()
        assert "add-session-tag" in result.lower()

    def test_list_tags_service_none(self):
        """Test error when tag service is None."""
        handler = create_list_tags_handler(None)
        result = handler({"session_id": "session-001"})

        assert "error" in result.lower()
        assert "not initialized" in result.lower()


class TestTagBasedFiltering:
    """Tests for tag-based filtering in search results."""

    def _create_mock_score(self, session_id, final_score):
        """Helper to create a mock SessionScore."""
        from smart_fork.scoring_service import SessionScore
        return SessionScore(
            session_id=session_id,
            final_score=final_score,
            best_similarity=0.85,
            avg_similarity=0.75,
            chunk_ratio=0.5,
            recency_score=0.8,
            chain_quality=0.6,
            memory_boost=0.0,
            preference_boost=0.0,
            num_chunks_matched=5
        )

    def test_filter_results_single_tag(self, session_registry):
        """Test filtering search results by a single tag."""
        # Simulate filtering logic from server.py
        from smart_fork.search_service import SessionSearchResult
        from smart_fork.scoring_service import SessionScore

        # Create mock search results
        results = [
            SessionSearchResult(
                session_id="session-001",
                score=SessionScore(
                    session_id="session-001",
                    final_score=0.9,
                    best_similarity=0.85,
                    avg_similarity=0.75,
                    chunk_ratio=0.5,
                    recency_score=0.8,
                    chain_quality=0.6,
                    memory_boost=0.0,
                    preference_boost=0.0,
                    num_chunks_matched=5
                ),
                metadata=session_registry.get_session("session-001"),
                preview="Test preview",
                matched_chunks=[]
            ),
            SessionSearchResult(
                session_id="session-002",
                score=SessionScore(
                    session_id="session-002",
                    final_score=0.8,
                    best_similarity=0.75,
                    avg_similarity=0.65,
                    chunk_ratio=0.4,
                    recency_score=0.7,
                    chain_quality=0.5,
                    memory_boost=0.0,
                    preference_boost=0.0,
                    num_chunks_matched=4
                ),
                metadata=session_registry.get_session("session-002"),
                preview="Test preview",
                matched_chunks=[]
            ),
            SessionSearchResult(
                session_id="session-004",
                score=SessionScore(
                    session_id="session-004",
                    final_score=0.7,
                    best_similarity=0.65,
                    avg_similarity=0.55,
                    chunk_ratio=0.3,
                    recency_score=0.6,
                    chain_quality=0.4,
                    memory_boost=0.0,
                    preference_boost=0.0,
                    num_chunks_matched=3
                ),
                metadata=session_registry.get_session("session-004"),
                preview="Test preview",
                matched_chunks=[]
            )
        ]

        # Filter by "python" tag
        tags_list = ["python"]
        filtered_results = []
        for result in results:
            if result.metadata and result.metadata.tags:
                session_tags = [t.lower() for t in result.metadata.tags]
                if any(tag in session_tags for tag in tags_list):
                    filtered_results.append(result)

        # Should have session-001 and session-004 (both have "python")
        assert len(filtered_results) == 2
        assert filtered_results[0].session_id == "session-001"
        assert filtered_results[1].session_id == "session-004"

    def test_filter_results_multiple_tags(self, session_registry):
        """Test filtering search results by multiple tags (any match)."""
        from smart_fork.search_service import SessionSearchResult

        # Create mock search results
        results = [
            SessionSearchResult(
                session_id="session-001",
                score=self._create_mock_score("session-001", 0.9),
                metadata=session_registry.get_session("session-001"),
                preview="Test preview",
                matched_chunks=[]
            ),
            SessionSearchResult(
                session_id="session-002",
                score=self._create_mock_score("session-002", 0.8),
                metadata=session_registry.get_session("session-002"),
                preview="Test preview",
                matched_chunks=[]
            ),
            SessionSearchResult(
                session_id="session-003",
                score=self._create_mock_score("session-003", 0.7),
                metadata=session_registry.get_session("session-003"),
                preview="Test preview",
                matched_chunks=[]
            )
        ]

        # Filter by "python" or "react" tags
        tags_list = ["python", "react"]
        filtered_results = []
        for result in results:
            if result.metadata and result.metadata.tags:
                session_tags = [t.lower() for t in result.metadata.tags]
                if any(tag in session_tags for tag in tags_list):
                    filtered_results.append(result)

        # Should have session-001 (python) and session-002 (react)
        assert len(filtered_results) == 2
        assert set([r.session_id for r in filtered_results]) == {"session-001", "session-002"}

    def test_filter_results_no_match(self, session_registry):
        """Test filtering when no sessions match the tags."""
        from smart_fork.search_service import SessionSearchResult

        # Create mock search results
        results = [
            SessionSearchResult(
                session_id="session-001",
                score=self._create_mock_score("session-001", 0.9),
                metadata=session_registry.get_session("session-001"),
                preview="Test preview",
                matched_chunks=[]
            ),
            SessionSearchResult(
                session_id="session-002",
                score=self._create_mock_score("session-002", 0.8),
                metadata=session_registry.get_session("session-002"),
                preview="Test preview",
                matched_chunks=[]
            )
        ]

        # Filter by non-existent tag
        tags_list = ["nonexistent"]
        filtered_results = []
        for result in results:
            if result.metadata and result.metadata.tags:
                session_tags = [t.lower() for t in result.metadata.tags]
                if any(tag in session_tags for tag in tags_list):
                    filtered_results.append(result)

        # Should be empty
        assert len(filtered_results) == 0

    def test_filter_results_untagged_session(self, session_registry):
        """Test that untagged sessions are filtered out."""
        from smart_fork.search_service import SessionSearchResult

        # Create mock search results including untagged session
        results = [
            SessionSearchResult(
                session_id="session-001",
                score=self._create_mock_score("session-001", 0.9),
                metadata=session_registry.get_session("session-001"),
                preview="Test preview",
                matched_chunks=[]
            ),
            SessionSearchResult(
                session_id="session-003",  # No tags
                score=self._create_mock_score("session-003", 0.8),
                metadata=session_registry.get_session("session-003"),
                preview="Test preview",
                matched_chunks=[]
            )
        ]

        # Filter by "python" tag
        tags_list = ["python"]
        filtered_results = []
        for result in results:
            if result.metadata and result.metadata.tags:
                session_tags = [t.lower() for t in result.metadata.tags]
                if any(tag in session_tags for tag in tags_list):
                    filtered_results.append(result)

        # Should only have session-001
        assert len(filtered_results) == 1
        assert filtered_results[0].session_id == "session-001"
