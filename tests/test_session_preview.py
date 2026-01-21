"""Tests for session preview functionality."""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, ANY

from smart_fork.search_service import SearchService
from smart_fork.session_registry import SessionMetadata
from smart_fork.session_parser import SessionData, SessionMessage


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    embedding_service = Mock()
    vector_db_service = Mock()
    scoring_service = Mock()
    session_registry = Mock()

    return {
        'embedding_service': embedding_service,
        'vector_db_service': vector_db_service,
        'scoring_service': scoring_service,
        'session_registry': session_registry
    }


@pytest.fixture
def search_service(mock_services):
    """Create a SearchService instance with mocked dependencies."""
    return SearchService(
        embedding_service=mock_services['embedding_service'],
        vector_db_service=mock_services['vector_db_service'],
        scoring_service=mock_services['scoring_service'],
        session_registry=mock_services['session_registry'],
        preview_length=200
    )


@pytest.fixture
def sample_session_data():
    """Create sample session data for testing."""
    messages = [
        SessionMessage(
            role="user",
            content="How do I implement authentication in FastAPI?",
            timestamp=datetime(2024, 1, 15, 10, 0, 0)
        ),
        SessionMessage(
            role="assistant",
            content="To implement authentication in FastAPI, you can use OAuth2 with JWT tokens. Here's a basic example...",
            timestamp=datetime(2024, 1, 15, 10, 1, 0)
        ),
        SessionMessage(
            role="user",
            content="Can you show me how to protect routes?",
            timestamp=datetime(2024, 1, 15, 10, 2, 0)
        ),
        SessionMessage(
            role="assistant",
            content="Sure! You can use dependency injection to protect routes. Here's how...",
            timestamp=datetime(2024, 1, 15, 10, 3, 0)
        )
    ]

    return SessionData(
        session_id="test-session-123",
        messages=messages,
        file_path=Path("/fake/path/test-session-123.jsonl"),
        created_at=datetime(2024, 1, 15, 10, 0, 0),
        last_modified=datetime(2024, 1, 15, 10, 3, 0)
    )


@pytest.fixture
def sample_metadata():
    """Create sample session metadata."""
    return SessionMetadata(
        session_id="test-session-123",
        project="test-project",
        chunk_count=10,
        message_count=4,
        created_at="2024-01-15T10:00:00",
        last_modified="2024-01-15T10:03:00"
    )


class TestGetSessionPreview:
    """Test suite for get_session_preview functionality."""

    def test_get_session_preview_success(
        self,
        search_service,
        mock_services,
        sample_metadata,
        sample_session_data
    ):
        """Test successful session preview retrieval."""
        # Setup mocks
        mock_services['session_registry'].get_session.return_value = sample_metadata

        with patch('smart_fork.fork_generator.ForkGenerator') as MockForkGen, \
             patch('smart_fork.search_service.SessionParser') as MockParser:
            # Mock ForkGenerator
            mock_fork_gen = MockForkGen.return_value
            mock_fork_gen.find_session_path.return_value = "/fake/path/test-session-123.jsonl"

            # Mock SessionParser
            mock_parser = MockParser.return_value
            mock_parser.parse_file.return_value = sample_session_data

            # Get preview
            result = search_service.get_session_preview("test-session-123", length=200)

            # Verify result
            assert result is not None
            assert result['session_id'] == "test-session-123"
            assert 'preview' in result
            assert result['message_count'] == 4
            assert result['date_range'] is not None
            assert result['date_range']['start'] == "2024-01-15T10:00:00"
            assert result['date_range']['end'] == "2024-01-15T10:03:00"

            # Verify preview content contains message data
            preview = result['preview']
            assert "user:" in preview
            assert "assistant:" in preview
            assert "authentication" in preview.lower()

    def test_get_session_preview_truncation(
        self,
        search_service,
        mock_services,
        sample_metadata,
        sample_session_data
    ):
        """Test that preview is truncated to specified length."""
        # Setup mocks
        mock_services['session_registry'].get_session.return_value = sample_metadata

        with patch('smart_fork.fork_generator.ForkGenerator') as MockForkGen, \
             patch('smart_fork.search_service.SessionParser') as MockParser:
            mock_fork_gen = MockForkGen.return_value
            mock_fork_gen.find_session_path.return_value = "/fake/path/test-session-123.jsonl"

            mock_parser = MockParser.return_value
            mock_parser.parse_file.return_value = sample_session_data

            # Get short preview
            result = search_service.get_session_preview("test-session-123", length=50)

            # Verify truncation
            assert result is not None
            preview = result['preview']
            assert len(preview) <= 53  # 50 + "..." (with some buffer for word boundaries)
            assert preview.endswith("...")

    def test_get_session_preview_session_not_found(
        self,
        search_service,
        mock_services
    ):
        """Test behavior when session is not found in registry."""
        # Setup mock to return None
        mock_services['session_registry'].get_session.return_value = None

        # Get preview
        result = search_service.get_session_preview("nonexistent-session")

        # Verify None returned
        assert result is None

    def test_get_session_preview_parse_error(
        self,
        search_service,
        mock_services,
        sample_metadata
    ):
        """Test behavior when session file cannot be parsed."""
        # Setup mocks
        mock_services['session_registry'].get_session.return_value = sample_metadata

        with patch('smart_fork.fork_generator.ForkGenerator') as MockForkGen, \
             patch('smart_fork.search_service.SessionParser') as MockParser:
            mock_fork_gen = MockForkGen.return_value
            mock_fork_gen.find_session_path.return_value = "/fake/path/test-session-123.jsonl"

            mock_parser = MockParser.return_value
            mock_parser.parse_file.side_effect = Exception("Parse error")

            # Get preview
            result = search_service.get_session_preview("test-session-123")

            # Verify None returned on error
            assert result is None

    def test_get_session_preview_empty_messages(
        self,
        search_service,
        mock_services,
        sample_metadata
    ):
        """Test behavior when session has no messages."""
        # Setup mocks
        mock_services['session_registry'].get_session.return_value = sample_metadata

        # Create session data with no messages
        empty_session = SessionData(
            session_id="test-session-123",
            messages=[],
            file_path=Path("/fake/path/test-session-123.jsonl"),
            created_at=datetime(2024, 1, 15, 10, 0, 0),
            last_modified=datetime(2024, 1, 15, 10, 0, 0)
        )

        with patch('smart_fork.fork_generator.ForkGenerator') as MockForkGen, \
             patch('smart_fork.search_service.SessionParser') as MockParser:
            mock_fork_gen = MockForkGen.return_value
            mock_fork_gen.find_session_path.return_value = "/fake/path/test-session-123.jsonl"

            mock_parser = MockParser.return_value
            mock_parser.parse_file.return_value = empty_session

            # Get preview
            result = search_service.get_session_preview("test-session-123")

            # Verify None returned
            assert result is None

    def test_get_session_preview_no_timestamps(
        self,
        search_service,
        mock_services,
        sample_metadata
    ):
        """Test preview with messages that have no timestamps."""
        # Setup mocks
        mock_services['session_registry'].get_session.return_value = sample_metadata

        # Create session data without timestamps
        messages = [
            SessionMessage(role="user", content="Question 1"),
            SessionMessage(role="assistant", content="Answer 1")
        ]

        session_data = SessionData(
            session_id="test-session-123",
            messages=messages,
            file_path=Path("/fake/path/test-session-123.jsonl"),
            created_at=datetime(2024, 1, 15, 10, 0, 0),
            last_modified=datetime(2024, 1, 15, 10, 3, 0)
        )

        with patch('smart_fork.fork_generator.ForkGenerator') as MockForkGen, \
             patch('smart_fork.search_service.SessionParser') as MockParser:
            mock_fork_gen = MockForkGen.return_value
            mock_fork_gen.find_session_path.return_value = "/fake/path/test-session-123.jsonl"

            mock_parser = MockParser.return_value
            mock_parser.parse_file.return_value = session_data

            # Get preview
            result = search_service.get_session_preview("test-session-123")

            # Verify result uses session timestamps
            assert result is not None
            assert result['date_range'] is not None
            assert result['date_range']['start'] == "2024-01-15T10:00:00"
            assert result['date_range']['end'] == "2024-01-15T10:03:00"

    def test_get_session_preview_custom_length(
        self,
        search_service,
        mock_services,
        sample_metadata,
        sample_session_data
    ):
        """Test preview with custom length parameter."""
        # Setup mocks
        mock_services['session_registry'].get_session.return_value = sample_metadata

        with patch('smart_fork.fork_generator.ForkGenerator') as MockForkGen, \
             patch('smart_fork.search_service.SessionParser') as MockParser:
            mock_fork_gen = MockForkGen.return_value
            mock_fork_gen.find_session_path.return_value = "/fake/path/test-session-123.jsonl"

            mock_parser = MockParser.return_value
            mock_parser.parse_file.return_value = sample_session_data

            # Get preview with custom length
            result = search_service.get_session_preview("test-session-123", length=1000)

            # Verify result
            assert result is not None
            preview = result['preview']
            # Should include more content with longer length
            assert "FastAPI" in preview or "authentication" in preview.lower()

    def test_get_session_preview_metadata_included(
        self,
        search_service,
        mock_services,
        sample_metadata,
        sample_session_data
    ):
        """Test that session metadata is included in preview result."""
        # Setup mocks
        mock_services['session_registry'].get_session.return_value = sample_metadata

        with patch('smart_fork.fork_generator.ForkGenerator') as MockForkGen, \
             patch('smart_fork.search_service.SessionParser') as MockParser:
            mock_fork_gen = MockForkGen.return_value
            mock_fork_gen.find_session_path.return_value = "/fake/path/test-session-123.jsonl"

            mock_parser = MockParser.return_value
            mock_parser.parse_file.return_value = sample_session_data

            # Get preview
            result = search_service.get_session_preview("test-session-123")

            # Verify metadata is included
            assert result is not None
            assert 'metadata' in result
            metadata = result['metadata']
            assert metadata is not None
            assert metadata['session_id'] == "test-session-123"
            assert metadata['chunk_count'] == 10
            assert metadata['message_count'] == 4


class TestMCPSessionPreviewHandler:
    """Test suite for MCP session preview handler."""

    def test_session_preview_handler_success(self):
        """Test successful MCP handler invocation."""
        from smart_fork.server import create_session_preview_handler

        # Create mock search service
        mock_search = Mock()
        mock_search.get_session_preview.return_value = {
            'session_id': 'test-123',
            'preview': 'This is a preview...',
            'message_count': 5,
            'date_range': {
                'start': '2024-01-15T10:00:00',
                'end': '2024-01-15T11:00:00'
            }
        }

        handler = create_session_preview_handler(mock_search)

        # Call handler
        result = handler({"session_id": "test-123", "length": 500})

        # Verify result format
        assert "Session Preview: test-123" in result
        assert "Messages: 5" in result
        assert "Date Range:" in result
        assert "This is a preview..." in result
        mock_search.get_session_preview.assert_called_once_with("test-123", 500)

    def test_session_preview_handler_no_session_id(self):
        """Test handler with missing session_id."""
        from smart_fork.server import create_session_preview_handler

        mock_search = Mock()
        handler = create_session_preview_handler(mock_search)

        # Call handler without session_id
        result = handler({})

        # Verify error message
        assert "Error:" in result
        assert "session_id" in result

    def test_session_preview_handler_service_not_initialized(self):
        """Test handler when search service is not initialized."""
        from smart_fork.server import create_session_preview_handler

        handler = create_session_preview_handler(None)

        # Call handler
        result = handler({"session_id": "test-123"})

        # Verify error message
        assert "Error:" in result
        assert "not initialized" in result

    def test_session_preview_handler_session_not_found(self):
        """Test handler when session is not found."""
        from smart_fork.server import create_session_preview_handler

        mock_search = Mock()
        mock_search.get_session_preview.return_value = None

        handler = create_session_preview_handler(mock_search)

        # Call handler
        result = handler({"session_id": "nonexistent"})

        # Verify error message
        assert "Error:" in result
        assert "not found" in result

    def test_session_preview_handler_exception(self):
        """Test handler with unexpected exception."""
        from smart_fork.server import create_session_preview_handler

        mock_search = Mock()
        mock_search.get_session_preview.side_effect = Exception("Unexpected error")

        handler = create_session_preview_handler(mock_search)

        # Call handler
        result = handler({"session_id": "test-123"})

        # Verify error message
        assert "Error:" in result
        assert "Unexpected error" in result

    def test_session_preview_handler_default_length(self):
        """Test handler with default length parameter."""
        from smart_fork.server import create_session_preview_handler

        mock_search = Mock()
        mock_search.get_session_preview.return_value = {
            'session_id': 'test-123',
            'preview': 'Preview text',
            'message_count': 3,
            'date_range': None
        }

        handler = create_session_preview_handler(mock_search)

        # Call handler without specifying length
        result = handler({"session_id": "test-123"})

        # Verify default length (500) was used
        mock_search.get_session_preview.assert_called_once_with("test-123", 500)
        assert "Session Preview: test-123" in result
