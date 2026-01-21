"""
Integration tests for session summarization.
"""

import pytest
import tempfile
import os
from pathlib import Path
import json

from smart_fork.background_indexer import BackgroundIndexer
from smart_fork.session_registry import SessionRegistry, SessionMetadata
from smart_fork.vector_db_service import VectorDBService
from smart_fork.embedding_service import EmbeddingService
from smart_fork.chunking_service import ChunkingService
from smart_fork.session_parser import SessionParser
from smart_fork.session_summary_service import SessionSummaryService


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as vector_db_dir, \
         tempfile.TemporaryDirectory() as claude_dir, \
         tempfile.TemporaryDirectory() as storage_dir:

        registry_path = os.path.join(storage_dir, "session-registry.json")

        yield {
            'vector_db_dir': vector_db_dir,
            'claude_dir': claude_dir,
            'registry_path': registry_path,
            'storage_dir': storage_dir
        }


@pytest.fixture
def services(temp_dirs):
    """Initialize all required services."""
    vector_db = VectorDBService(persist_directory=temp_dirs['vector_db_dir'])
    session_registry = SessionRegistry(registry_path=temp_dirs['registry_path'])
    # Use temp directory for embedding cache
    cache_dir = os.path.join(temp_dirs['storage_dir'], 'embedding_cache')
    embedding_service = EmbeddingService(cache_dir=cache_dir)
    chunking_service = ChunkingService()
    session_parser = SessionParser()
    summary_service = SessionSummaryService()

    return {
        'vector_db': vector_db,
        'session_registry': session_registry,
        'embedding_service': embedding_service,
        'chunking_service': chunking_service,
        'session_parser': session_parser,
        'summary_service': summary_service
    }


@pytest.fixture
def background_indexer(temp_dirs, services):
    """Create a BackgroundIndexer with summary service."""
    indexer = BackgroundIndexer(
        claude_dir=Path(temp_dirs['claude_dir']),
        vector_db=services['vector_db'],
        session_registry=services['session_registry'],
        embedding_service=services['embedding_service'],
        chunking_service=services['chunking_service'],
        session_parser=services['session_parser'],
        summary_service=services['summary_service']
    )
    return indexer


def create_test_session_file(claude_dir: str, session_id: str, messages: list):
    """Helper to create a test session file."""
    session_file = os.path.join(claude_dir, f"{session_id}.jsonl")

    with open(session_file, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(json.dumps(msg) + '\n')

    return Path(session_file)


class TestSessionSummaryIntegration:
    """Integration tests for session summarization."""

    def test_summary_generated_during_indexing(self, temp_dirs, background_indexer, services):
        """Test that summary is generated when indexing a session."""
        # Create a test session file
        messages = [
            {
                "type": "user",
                "content": "I need help building a REST API with FastAPI for user authentication.",
                "timestamp": "2026-01-21T10:00:00Z"
            },
            {
                "type": "assistant",
                "content": "I can help you build a REST API with FastAPI for authentication.",
                "timestamp": "2026-01-21T10:00:05Z"
            },
            {
                "type": "user",
                "content": "The API should support JWT tokens and refresh token functionality.",
                "timestamp": "2026-01-21T10:00:30Z"
            }
        ]

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-session-001", messages)

        # Index the session
        background_indexer.index_file(session_file, force=True)

        # Check that session was indexed with summary
        session = services['session_registry'].get_session("test-session-001")

        assert session is not None
        assert session.summary is not None
        assert len(session.summary) > 0
        # Summary should contain key terms
        assert any(term in session.summary.lower() for term in ["api", "fastapi", "authentication", "jwt"])

    def test_summary_updated_on_reindex(self, temp_dirs, background_indexer, services):
        """Test that summary is updated when session is re-indexed."""
        # Create initial session
        messages_v1 = [
            {
                "type": "user",
                "content": "I need help with basic Python syntax for loops and conditionals.",
                "timestamp": "2026-01-21T10:00:00Z"
            }
        ]

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-session-002", messages_v1)
        background_indexer.index_file(session_file, force=True)

        session_v1 = services['session_registry'].get_session("test-session-002")
        summary_v1 = session_v1.summary

        # Update session with more content
        messages_v2 = messages_v1 + [
            {
                "type": "assistant",
                "content": "Here's how to use loops and conditionals in Python effectively.",
                "timestamp": "2026-01-21T10:00:05Z"
            },
            {
                "type": "user",
                "content": "Now I need help with async programming and concurrent execution patterns.",
                "timestamp": "2026-01-21T10:01:00Z"
            }
        ]

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-session-002", messages_v2)
        background_indexer.index_file(session_file, force=True)

        session_v2 = services['session_registry'].get_session("test-session-002")
        summary_v2 = session_v2.summary

        # Summary should be updated
        assert summary_v2 != summary_v1
        # New summary should reflect async content
        assert "async" in summary_v2.lower() or "concurrent" in summary_v2.lower()

    def test_summary_handles_code_heavy_session(self, temp_dirs, background_indexer, services):
        """Test that summarization handles code-heavy sessions gracefully."""
        messages = [
            {
                "type": "user",
                "content": "I need to implement a binary search tree in Python.",
                "timestamp": "2026-01-21T10:00:00Z"
            },
            {
                "type": "assistant",
                "content": "class TreeNode:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None",
                "timestamp": "2026-01-21T10:00:05Z"
            },
            {
                "type": "user",
                "content": "The tree should support insertion, deletion, and search operations efficiently.",
                "timestamp": "2026-01-21T10:00:30Z"
            }
        ]

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-session-003", messages)
        background_indexer.index_file(session_file, force=True)

        session = services['session_registry'].get_session("test-session-003")

        # Should have a summary
        assert session.summary is not None
        # Summary should prefer natural language over code
        assert "binary search tree" in session.summary.lower() or "tree" in session.summary.lower()
        # Should not include raw code syntax
        assert "TreeNode" not in session.summary or "def __init__" not in session.summary

    def test_empty_session_summary(self, temp_dirs, background_indexer, services):
        """Test summarization of empty or minimal sessions."""
        messages = [
            {
                "type": "user",
                "content": "Hi",
                "timestamp": "2026-01-21T10:00:00Z"
            }
        ]

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-session-004", messages)
        background_indexer.index_file(session_file, force=True)

        session = services['session_registry'].get_session("test-session-004")

        # Should handle gracefully, may have minimal or no summary
        assert session.summary is not None

    def test_long_session_summary(self, temp_dirs, background_indexer, services):
        """Test summarization of a long session with many messages."""
        messages = []
        topics = ["database", "API", "frontend", "testing", "deployment"]

        for i, topic in enumerate(topics * 5):  # 25 messages
            messages.append({
                "type": "user",
                "content": f"I have a question about {topic} implementation in my project.",
                "timestamp": f"2026-01-21T10:{i:02d}:00Z"
            })

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-session-005", messages)
        background_indexer.index_file(session_file, force=True)

        session = services['session_registry'].get_session("test-session-005")

        assert session.summary is not None
        assert len(session.summary) > 0
        # Should be concise despite long session
        assert len(session.summary) < 1000  # Reasonable length

    def test_multiple_sessions_indexed(self, temp_dirs, background_indexer, services):
        """Test that multiple sessions are indexed with summaries."""
        sessions_data = [
            ("session-a", [
                {"type": "user", "content": "Help me with React hooks and state management.", "timestamp": "2026-01-21T10:00:00Z"}
            ]),
            ("session-b", [
                {"type": "user", "content": "I need assistance with Docker container orchestration.", "timestamp": "2026-01-21T11:00:00Z"}
            ]),
            ("session-c", [
                {"type": "user", "content": "How do I optimize SQL query performance in PostgreSQL.", "timestamp": "2026-01-21T12:00:00Z"}
            ])
        ]

        for session_id, messages in sessions_data:
            session_file = create_test_session_file(temp_dirs['claude_dir'], session_id, messages)
            background_indexer.index_file(session_file, force=True)

        # Check all sessions have summaries
        for session_id, _ in sessions_data:
            session = services['session_registry'].get_session(session_id)
            assert session is not None
            assert session.summary is not None
            assert len(session.summary) > 0

    def test_summary_service_failure_handled(self, temp_dirs, services):
        """Test that indexing continues if summary generation fails."""
        # Create a mock summary service that raises an exception
        class FailingSummaryService:
            def generate_summary(self, messages, session_id):
                raise ValueError("Summary generation failed")

        failing_indexer = BackgroundIndexer(
            claude_dir=Path(temp_dirs['claude_dir']),
            vector_db=services['vector_db'],
            session_registry=services['session_registry'],
            embedding_service=services['embedding_service'],
            chunking_service=services['chunking_service'],
            session_parser=services['session_parser'],
            summary_service=FailingSummaryService()
        )

        messages = [
            {
                "type": "user",
                "content": "This is a test message for summary failure handling.",
                "timestamp": "2026-01-21T10:00:00Z"
            }
        ]

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-session-fail", messages)

        # Should not raise exception, indexing should continue
        failing_indexer.index_file(session_file, force=True)

        session = services['session_registry'].get_session("test-session-fail")
        assert session is not None
        # Summary may be None due to failure
        assert session.summary is None

    def test_summary_in_session_metadata(self, temp_dirs, background_indexer, services):
        """Test that summary is properly stored in session metadata."""
        messages = [
            {
                "type": "user",
                "content": "I need to implement OAuth2 authentication flow in my web application.",
                "timestamp": "2026-01-21T10:00:00Z"
            }
        ]

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-session-meta", messages)
        background_indexer.index_file(session_file, force=True)

        # Get session from registry
        session = services['session_registry'].get_session("test-session-meta")

        # Verify all metadata fields
        assert session.session_id == "test-session-meta"
        assert session.message_count >= 1
        assert session.chunk_count >= 1
        assert session.summary is not None
        assert isinstance(session.summary, str)
        assert len(session.summary) > 0

    def test_summary_persists_across_registry_reload(self, temp_dirs, services):
        """Test that summaries persist when registry is reloaded."""
        messages = [
            {
                "type": "user",
                "content": "How do I set up continuous integration with GitHub Actions for my project.",
                "timestamp": "2026-01-21T10:00:00Z"
            }
        ]

        # Create first indexer and index a session
        indexer1 = BackgroundIndexer(
            claude_dir=Path(temp_dirs['claude_dir']),
            vector_db=services['vector_db'],
            session_registry=services['session_registry'],
            embedding_service=services['embedding_service'],
            chunking_service=services['chunking_service'],
            session_parser=services['session_parser'],
            summary_service=SessionSummaryService()
        )

        session_file = create_test_session_file(temp_dirs['claude_dir'], "test-persist", messages)
        indexer1.index_file(session_file, force=True)

        original_summary = services['session_registry'].get_session("test-persist").summary

        # Create new registry instance (simulates reload)
        new_registry = SessionRegistry(registry_path=temp_dirs['registry_path'])
        reloaded_session = new_registry.get_session("test-persist")

        # Summary should persist
        assert reloaded_session is not None
        assert reloaded_session.summary == original_summary
