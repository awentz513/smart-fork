"""
End-to-end tests for the /fork-detect workflow.

This test suite simulates the complete user journey from invoking /fork-detect
to generating fork commands, covering all UI options and error states.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import Mock, MagicMock, patch

import pytest

from smart_fork.server import MCPServer, create_server, initialize_services
from smart_fork.search_service import SearchService, SessionSearchResult
from smart_fork.selection_ui import SelectionUI
from smart_fork.fork_generator import ForkGenerator
from smart_fork.session_parser import SessionMessage
from smart_fork.chunking_service import ChunkingService
from smart_fork.embedding_service import EmbeddingService
from smart_fork.vector_db_service import VectorDBService
from smart_fork.scoring_service import ScoringService, SessionScore
from smart_fork.session_registry import SessionRegistry


class TestForkDetectEndToEnd:
    """End-to-end tests for the complete /fork-detect workflow."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_search_service(self):
        """Create a mock search service with realistic test data."""
        service = Mock(spec=SearchService)

        # Create realistic search results
        results = [
            SessionSearchResult(
                session_id="session-123",
                score=SessionScore(
                    best_similarity=0.92,
                    avg_similarity=0.85,
                    chunk_ratio=0.75,
                    recency=0.95,
                    chain_quality=0.5,
                    memory_boost=0.08,
                    final_score=0.876
                ),
                metadata={
                    "project": "my-project",
                    "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
                    "message_count": 45,
                    "chunk_count": 12,
                    "tags": ["authentication", "bug-fix"]
                },
                preview="Implemented JWT authentication with refresh tokens. Working solution tested."
            ),
            SessionSearchResult(
                session_id="session-456",
                score=SessionScore(
                    best_similarity=0.88,
                    avg_similarity=0.80,
                    chunk_ratio=0.65,
                    recency=0.90,
                    chain_quality=0.5,
                    memory_boost=0.05,
                    final_score=0.824
                ),
                metadata={
                    "project": "my-project",
                    "created_at": (datetime.now() - timedelta(days=5)).isoformat(),
                    "message_count": 32,
                    "chunk_count": 9,
                    "tags": ["api", "design-pattern"]
                },
                preview="Designed REST API architecture using repository pattern. Approach documented."
            ),
            SessionSearchResult(
                session_id="session-789",
                score=SessionScore(
                    best_similarity=0.82,
                    avg_similarity=0.75,
                    chunk_ratio=0.55,
                    recency=0.85,
                    chain_quality=0.5,
                    memory_boost=0.02,
                    final_score=0.762
                ),
                metadata={
                    "project": "other-project",
                    "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
                    "message_count": 28,
                    "chunk_count": 8,
                    "tags": ["database", "waiting"]
                },
                preview="Database migration pending. Waiting for review before proceeding."
            )
        ]

        service.search.return_value = results
        return service

    @pytest.fixture
    def mcp_server(self, mock_search_service):
        """Create an MCP server with mock search service."""
        server = create_server(search_service=mock_search_service)
        return server

    def test_invoke_fork_detect_with_query(self, mcp_server, mock_search_service):
        """Test invoking /fork-detect with a valid query."""
        # Simulate user invoking /fork-detect
        query = "authentication bug fix"

        # Call the fork-detect tool
        result = mcp_server.call_tool("fork-detect", {"query": query})

        # Verify search service was called with the query
        mock_search_service.search.assert_called_once_with(query)

        # Verify result contains formatted search results
        assert result is not None
        assert "session-123" in result
        assert "session-456" in result
        assert "session-789" in result
        assert "87.6%" in result  # First result score
        assert "JWT authentication" in result

    def test_fork_detect_displays_results(self, mcp_server, mock_search_service):
        """Test that /fork-detect displays results in the correct format."""
        query = "API design patterns"
        result = mcp_server.call_tool("fork-detect", {"query": query})

        # Verify result contains all required elements
        assert "Session ID:" in result
        assert "Score:" in result
        assert "Project:" in result
        assert "Created:" in result
        assert "Messages:" in result
        assert "Chunks:" in result
        assert "Tags:" in result
        assert "Preview:" in result

        # Verify score breakdown is included
        assert "Best Similarity:" in result
        assert "Avg Similarity:" in result
        assert "Chunk Ratio:" in result
        assert "Recency:" in result
        assert "Chain Quality:" in result
        assert "Memory Boost:" in result

    def test_fork_detect_empty_query(self, mcp_server):
        """Test /fork-detect with an empty query."""
        result = mcp_server.call_tool("fork-detect", {"query": ""})

        # Verify error message is returned
        assert "error" in result.lower() or "query" in result.lower()

    def test_fork_detect_no_results(self, mcp_server, mock_search_service):
        """Test /fork-detect when search returns no results."""
        # Configure mock to return empty results
        mock_search_service.search.return_value = []

        query = "nonexistent topic"
        result = mcp_server.call_tool("fork-detect", {"query": query})

        # Verify helpful message is returned
        assert "no" in result.lower() or "found" in result.lower()

    def test_fork_detect_with_missing_service(self):
        """Test /fork-detect when search service is not available."""
        # Create server without search service
        server = create_server(search_service=None)

        result = server.call_tool("fork-detect", {"query": "test"})

        # Verify error message is returned
        assert "error" in result.lower() or "not available" in result.lower()

    def test_selection_ui_creates_five_options(self, mock_search_service):
        """Test that selection UI creates exactly 5 options."""
        results = mock_search_service.search("test")
        ui = SelectionUI()

        options = ui.create_options(results)

        # Verify exactly 5 options: top 3 + 'None' + 'Type something'
        assert len(options) == 5

        # Verify top 3 are from search results
        assert options[0].session_id == "session-123"
        assert options[1].session_id == "session-456"
        assert options[2].session_id == "session-789"

        # Verify 'None - start fresh' option
        assert options[3].label == "None - start fresh"
        assert options[3].action == "start_fresh"

        # Verify 'Type something else' option
        assert options[4].label == "Type something else"
        assert options[4].action == "refine"

    def test_selection_ui_marks_recommended(self, mock_search_service):
        """Test that highest-scoring result is marked as recommended."""
        results = mock_search_service.search("test")
        ui = SelectionUI()

        selection_prompt = ui.format_selection_prompt(results)

        # Verify first result is marked as recommended
        assert "Recommended" in selection_prompt or "⭐" in selection_prompt

    def test_selection_ui_handles_fork_action(self, mock_search_service):
        """Test selecting a result and handling fork action."""
        results = mock_search_service.search("test")
        ui = SelectionUI()

        # Simulate user selecting first option
        action, data = ui.handle_selection(0, results)

        # Verify fork action is returned
        assert action == "fork"
        assert data["session_id"] == "session-123"

    def test_selection_ui_handles_start_fresh_action(self, mock_search_service):
        """Test selecting 'None - start fresh' option."""
        results = mock_search_service.search("test")
        ui = SelectionUI()

        # Simulate user selecting 'None - start fresh' (index 3)
        action, data = ui.handle_selection(3, results)

        # Verify start_fresh action is returned
        assert action == "start_fresh"
        assert data == {}

    def test_selection_ui_handles_refine_action(self, mock_search_service):
        """Test selecting 'Type something else' option."""
        results = mock_search_service.search("test")
        ui = SelectionUI()

        # Simulate user selecting 'Type something else' (index 4)
        action, data = ui.handle_selection(4, results)

        # Verify refine action is returned
        assert action == "refine"
        assert data == {}

    def test_fork_generator_creates_commands(self, temp_storage):
        """Test that fork generator creates both terminal and in-session commands."""
        generator = ForkGenerator(
            claude_sessions_dir=temp_storage,
            registry=Mock(spec=SessionRegistry)
        )

        session_id = "session-123"
        metadata = {
            "project": "my-project",
            "created_at": datetime.now().isoformat(),
            "message_count": 45,
            "chunk_count": 12
        }

        # Create a dummy session file
        session_file = Path(temp_storage) / f"{session_id}.jsonl"
        session_file.write_text('{"role": "user", "content": "test"}\n')

        # Generate fork command
        fork_cmd = generator.generate_fork_command(session_id, metadata)

        # Verify both commands are generated
        assert fork_cmd.terminal_command is not None
        assert fork_cmd.in_session_command is not None
        assert f"--resume {session_id}" in fork_cmd.terminal_command
        assert f"/fork {session_id}" in fork_cmd.in_session_command

    def test_fork_generator_formats_output(self, temp_storage):
        """Test that fork generator formats output for display."""
        generator = ForkGenerator(
            claude_sessions_dir=temp_storage,
            registry=Mock(spec=SessionRegistry)
        )

        session_id = "session-123"
        metadata = {
            "project": "my-project",
            "created_at": datetime.now().isoformat(),
            "message_count": 45,
            "chunk_count": 12
        }

        # Create a dummy session file
        session_file = Path(temp_storage) / f"{session_id}.jsonl"
        session_file.write_text('{"role": "user", "content": "test"}\n')

        # Generate and format
        output = generator.generate_and_format(session_id, metadata)

        # Verify output contains metadata
        assert "Session ID:" in output
        assert session_id in output
        assert "my-project" in output

        # Verify output contains commands
        assert "New Terminal:" in output or "Terminal Command:" in output
        assert "In-Session:" in output or "In-Session Command:" in output

    def test_error_handling_invalid_selection(self, mock_search_service):
        """Test error handling for invalid selection index."""
        results = mock_search_service.search("test")
        ui = SelectionUI()

        # Simulate invalid selection
        action, data = ui.handle_selection(10, results)  # Out of range

        # Verify error action is returned
        assert action == "error"

    def test_error_handling_search_exception(self):
        """Test error handling when search raises an exception."""
        # Create mock that raises exception
        service = Mock(spec=SearchService)
        service.search.side_effect = Exception("Search failed")

        server = create_server(search_service=service)
        result = server.call_tool("fork-detect", {"query": "test"})

        # Verify error is handled gracefully
        assert result is not None
        assert "error" in result.lower() or "failed" in result.lower()

    def test_complete_workflow_fork_session(self, mcp_server, mock_search_service, temp_storage):
        """Test the complete workflow: query -> results -> selection -> fork command."""
        # Step 1: User invokes /fork-detect with query
        query = "authentication bug fix"
        search_result = mcp_server.call_tool("fork-detect", {"query": query})

        # Verify search was performed
        mock_search_service.search.assert_called_once()
        assert "session-123" in search_result

        # Step 2: User sees results and makes a selection
        results = mock_search_service.search.return_value
        ui = SelectionUI()
        action, data = ui.handle_selection(0, results)  # Select first result

        # Verify fork action
        assert action == "fork"
        assert data["session_id"] == "session-123"

        # Step 3: Generate fork command
        generator = ForkGenerator(
            claude_sessions_dir=temp_storage,
            registry=Mock(spec=SessionRegistry)
        )

        # Create a dummy session file
        session_file = Path(temp_storage) / f"{data['session_id']}.jsonl"
        session_file.write_text('{"role": "user", "content": "test"}\n')

        fork_cmd = generator.generate_fork_command(
            data["session_id"],
            results[0].metadata
        )

        # Verify fork command was generated
        assert fork_cmd.terminal_command is not None
        assert fork_cmd.in_session_command is not None
        assert data["session_id"] in fork_cmd.terminal_command

    def test_complete_workflow_start_fresh(self, mcp_server, mock_search_service):
        """Test the complete workflow: query -> results -> start fresh."""
        # Step 1: User invokes /fork-detect with query
        query = "new feature"
        mcp_server.call_tool("fork-detect", {"query": query})

        # Step 2: User decides to start fresh instead
        results = mock_search_service.search.return_value
        ui = SelectionUI()
        action, data = ui.handle_selection(3, results)  # Select 'None - start fresh'

        # Verify start_fresh action
        assert action == "start_fresh"
        assert data == {}

    def test_complete_workflow_refine_search(self, mcp_server, mock_search_service):
        """Test the complete workflow: query -> results -> refine search."""
        # Step 1: User invokes /fork-detect with query
        query = "database optimization"
        mcp_server.call_tool("fork-detect", {"query": query})

        # Step 2: User decides to refine search
        results = mock_search_service.search.return_value
        ui = SelectionUI()
        action, data = ui.handle_selection(4, results)  # Select 'Type something else'

        # Verify refine action
        assert action == "refine"
        assert data == {}

        # Step 3: User provides new query
        refined_query = "PostgreSQL query performance"
        mcp_server.call_tool("fork-detect", {"query": refined_query})

        # Verify search was called again
        assert mock_search_service.search.call_count == 2


class TestForkDetectEdgeCases:
    """Edge case tests for the /fork-detect workflow."""

    def test_very_long_query(self):
        """Test handling of very long queries."""
        service = Mock(spec=SearchService)
        service.search.return_value = []

        server = create_server(search_service=service)

        # Create a very long query (1000+ characters)
        long_query = "authentication " * 100
        result = server.call_tool("fork-detect", {"query": long_query})

        # Verify query is handled without errors
        assert result is not None
        service.search.assert_called_once()

    def test_special_characters_in_query(self):
        """Test handling of special characters in queries."""
        service = Mock(spec=SearchService)
        service.search.return_value = []

        server = create_server(search_service=service)

        # Query with special characters
        query = "API @authentication #bug-fix $config <test> & validation"
        result = server.call_tool("fork-detect", {"query": query})

        # Verify query is handled without errors
        assert result is not None
        service.search.assert_called_once()

    def test_unicode_in_query(self):
        """Test handling of unicode characters in queries."""
        service = Mock(spec=SearchService)
        service.search.return_value = []

        server = create_server(search_service=service)

        # Query with unicode characters
        query = "データベース optimization 🚀 émoji test"
        result = server.call_tool("fork-detect", {"query": query})

        # Verify query is handled without errors
        assert result is not None
        service.search.assert_called_once()

    def test_session_file_not_found(self, temp_storage):
        """Test fork generator when session file doesn't exist."""
        generator = ForkGenerator(
            claude_sessions_dir=temp_storage,
            registry=Mock(spec=SessionRegistry)
        )

        # Try to generate command for non-existent session
        session_id = "nonexistent-session"
        metadata = {"project": "test"}

        fork_cmd = generator.generate_fork_command(session_id, metadata)

        # Verify command is still generated (path will be None)
        assert fork_cmd.terminal_command is not None
        assert fork_cmd.in_session_command is not None

    def test_selection_with_empty_results(self):
        """Test selection UI with empty search results."""
        ui = SelectionUI()
        results = []

        options = ui.create_options(results)

        # Verify still creates 5 options (0 results + None + Type something + 2 placeholders)
        assert len(options) == 5

        # Verify last two are always None and Type something
        assert options[3].action == "start_fresh"
        assert options[4].action == "refine"

    def test_selection_with_single_result(self, temp_storage):
        """Test selection UI with only one search result."""
        ui = SelectionUI()
        results = [
            SessionSearchResult(
                session_id="session-only",
                score=SessionScore(
                    best_similarity=0.9,
                    avg_similarity=0.8,
                    chunk_ratio=0.7,
                    recency=0.95,
                    chain_quality=0.5,
                    memory_boost=0.0,
                    final_score=0.85
                ),
                metadata={
                    "project": "test",
                    "created_at": datetime.now().isoformat(),
                    "message_count": 10,
                    "chunk_count": 3,
                    "tags": []
                },
                preview="Only result"
            )
        ]

        options = ui.create_options(results)

        # Verify creates 5 options (1 result + 2 placeholders + None + Type something)
        assert len(options) == 5
        assert options[0].session_id == "session-only"


class TestForkDetectIntegration:
    """Integration tests with real components (mocked external dependencies only)."""

    @pytest.fixture
    def integration_storage(self):
        """Create temporary storage for integration tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock(spec=EmbeddingService)
        # Return a realistic 768-dimensional embedding
        service.embed_single.return_value = [0.1] * 768
        service.embed_texts.return_value = [[0.1] * 768]
        return service

    def test_integration_with_real_components(self, integration_storage, mock_embedding_service):
        """Test integration with real SearchService, ScoringService, etc."""
        # Create real components with mocked embedding
        vector_db = VectorDBService(storage_dir=integration_storage)
        scoring = ScoringService()
        registry = SessionRegistry(storage_dir=integration_storage)

        # Create search service with real components
        search_service = SearchService(
            embedding_service=mock_embedding_service,
            vector_db_service=vector_db,
            scoring_service=scoring,
            session_registry=registry
        )

        # Create MCP server
        server = create_server(search_service=search_service)

        # Invoke /fork-detect
        result = server.call_tool("fork-detect", {"query": "test query"})

        # Verify result is returned (even if empty)
        assert result is not None
        assert isinstance(result, str)


def test_fork_detect_tool_registered():
    """Test that /fork-detect tool is properly registered in MCP server."""
    server = create_server(search_service=None)

    # Verify tool is registered
    tools = server.list_tools()
    assert "fork-detect" in tools

    # Verify tool has correct schema
    tool_info = tools["fork-detect"]
    assert "query" in str(tool_info)


def test_fork_detect_tool_schema():
    """Test that /fork-detect tool has correct input schema."""
    server = create_server(search_service=None)
    tools = server.list_tools()

    tool_info = tools["fork-detect"]

    # Verify schema includes query parameter
    assert "query" in str(tool_info)
    # Verify query is required
    assert "string" in str(tool_info).lower()
