"""Integration tests for MCP tool flow.

Tests the full end-to-end workflow of MCP tool invocation, simulating
how a real MCP client (like Claude Code) would interact with the server.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

from smart_fork.embedding_service import EmbeddingService
from smart_fork.fork_generator import ForkGenerator
from smart_fork.scoring_service import ScoringService, SessionScore
from smart_fork.search_service import SearchService, SessionSearchResult
from smart_fork.selection_ui import SelectionUI
from smart_fork.server import (
    MCPServer,
    create_fork_detect_handler,
    create_server,
    format_search_results_with_selection,
)
from smart_fork.session_registry import SessionMetadata, SessionRegistry
from smart_fork.vector_db_service import VectorDBService


class MockMCPClient:
    """Mock MCP client for testing."""

    def __init__(self, server: MCPServer):
        """Initialize mock client with server instance."""
        self.server = server
        self.request_id = 0

    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a request to the server and return the response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        return self.server.handle_request(request)

    def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a notification to the server (no response expected)."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        self.server.handle_request(request)


class TestMCPClientIntegration:
    """Test MCP tool flow from client perspective."""

    def test_client_initialization_flow(self):
        """Test complete client initialization sequence."""
        server = create_server()
        client = MockMCPClient(server)

        # Step 1: Client sends initialize request
        response = client.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })

        assert response is not None
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert response["result"]["serverInfo"]["name"] == "smart-fork"
        assert "capabilities" in response["result"]

        # Step 2: Client sends initialized notification
        client.send_notification("notifications/initialized")

        # Step 3: Client requests tool list
        response = client.send_request("tools/list")

        assert response is not None
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) == 1
        assert response["result"]["tools"][0]["name"] == "fork-detect"

    def test_fork_detect_tool_invocation(self):
        """Test invoking fork-detect tool via MCP protocol."""
        server = create_server()
        client = MockMCPClient(server)

        # Initialize connection
        client.send_request("initialize")
        client.send_notification("notifications/initialized")

        # Invoke fork-detect tool
        response = client.send_request("tools/call", {
            "name": "fork-detect",
            "arguments": {"query": "implement authentication"}
        })

        assert response is not None
        assert "result" in response
        assert "content" in response["result"]
        assert len(response["result"]["content"]) > 0
        assert response["result"]["content"][0]["type"] == "text"
        assert "implement authentication" in response["result"]["content"][0]["text"]

    def test_fork_detect_tool_empty_query_error(self):
        """Test fork-detect tool with empty query returns error message."""
        server = create_server()
        client = MockMCPClient(server)

        client.send_request("initialize")
        client.send_notification("notifications/initialized")

        # Invoke with empty query
        response = client.send_request("tools/call", {
            "name": "fork-detect",
            "arguments": {}
        })

        assert response is not None
        assert "result" in response
        assert "content" in response["result"]
        text = response["result"]["content"][0]["text"]
        assert "Error" in text or "provide a query" in text

    def test_fork_detect_tool_invalid_arguments(self):
        """Test fork-detect tool with invalid arguments."""
        server = create_server()
        client = MockMCPClient(server)

        client.send_request("initialize")
        client.send_notification("notifications/initialized")

        # Invoke with invalid arguments (missing required query)
        response = client.send_request("tools/call", {
            "name": "fork-detect",
            "arguments": {"invalid_param": "value"}
        })

        assert response is not None
        # Should still return a result (error message in content)
        assert "result" in response
        assert "content" in response["result"]

    def test_unknown_tool_error(self):
        """Test that invoking unknown tool returns proper error."""
        server = create_server()
        client = MockMCPClient(server)

        client.send_request("initialize")
        client.send_notification("notifications/initialized")

        # Invoke non-existent tool
        response = client.send_request("tools/call", {
            "name": "non-existent-tool",
            "arguments": {}
        })

        assert response is not None
        assert "error" in response
        assert "Unknown tool" in response["error"]["message"]
        assert response["error"]["code"] == -32603

    def test_unknown_method_error(self):
        """Test that unknown method returns proper error."""
        server = create_server()
        client = MockMCPClient(server)

        response = client.send_request("unknown/method")

        assert response is not None
        assert "error" in response
        assert "Unknown method" in response["error"]["message"]
        assert response["error"]["code"] == -32603


class TestSearchSelectForkWorkflow:
    """Test complete search-select-fork workflow."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory for test database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_search_service(self, temp_db_dir):
        """Create mock search service with sample data."""
        # Create mock dependencies
        embedding_service = Mock(spec=EmbeddingService)
        vector_db_service = Mock(spec=VectorDBService)
        session_registry = Mock(spec=SessionRegistry)
        scoring_service = Mock(spec=ScoringService)

        # Create search service with correct parameter names
        search_service = SearchService(
            embedding_service=embedding_service,
            vector_db_service=vector_db_service,
            session_registry=session_registry,
            scoring_service=scoring_service
        )

        # Mock search results - use SessionSearchResult objects
        mock_results = [
            SessionSearchResult(
                session_id='session1',
                score=SessionScore(
                    session_id='session1',
                    final_score=0.95,
                    best_similarity=0.95,
                    avg_similarity=0.90,
                    chunk_ratio=0.80,
                    recency_score=0.85,
                    chain_quality=0.75,
                    memory_boost=0.0,
                    num_chunks_matched=5
                ),
                metadata=SessionMetadata(
                    session_id='session1',
                    created_at='2024-01-01T00:00:00',
                    message_count=10,
                    chunk_count=5
                ),
                preview='Implementing user authentication with JWT',
                matched_chunks=[]
            ),
            SessionSearchResult(
                session_id='session2',
                score=SessionScore(
                    session_id='session2',
                    final_score=0.85,
                    best_similarity=0.85,
                    avg_similarity=0.80,
                    chunk_ratio=0.70,
                    recency_score=0.75,
                    chain_quality=0.65,
                    memory_boost=0.0,
                    num_chunks_matched=4
                ),
                metadata=SessionMetadata(
                    session_id='session2',
                    created_at='2024-01-02T00:00:00',
                    message_count=8,
                    chunk_count=4
                ),
                preview='OAuth authentication flow',
                matched_chunks=[]
            ),
            SessionSearchResult(
                session_id='session3',
                score=SessionScore(
                    session_id='session3',
                    final_score=0.75,
                    best_similarity=0.75,
                    avg_similarity=0.70,
                    chunk_ratio=0.60,
                    recency_score=0.70,
                    chain_quality=0.55,
                    memory_boost=0.0,
                    num_chunks_matched=6
                ),
                metadata=SessionMetadata(
                    session_id='session3',
                    created_at='2024-01-03T00:00:00',
                    message_count=12,
                    chunk_count=6
                ),
                preview='Setting up authentication middleware',
                matched_chunks=[]
            )
        ]

        # Patch search method
        with patch.object(search_service, 'search', return_value=mock_results):
            yield search_service

    def test_full_search_select_workflow(self, mock_search_service, temp_db_dir):
        """Test complete workflow from search to fork command generation."""
        # Create session files
        for i in range(1, 4):
            session_file = temp_db_dir / f'session{i}.jsonl'
            session_file.write_text('{"role": "user", "content": "test"}\n')

        # Create handler with mock search service
        handler = create_fork_detect_handler(
            search_service=mock_search_service,
            claude_dir=str(temp_db_dir)
        )

        # Execute search
        result = handler({"query": "implement authentication"})

        # Verify result format
        assert isinstance(result, str)
        assert "Fork Detection" in result
        assert "implement authentication" in result

        # Verify result contains selection options
        assert "1." in result or "Option 1" in result or "RECOMMENDED" in result

        # Verify result contains fork commands
        assert "fork" in result.lower() or "claude" in result.lower()

    def test_search_with_no_results(self, temp_db_dir):
        """Test workflow when search returns no results."""
        # Create empty search service with correct parameters
        embedding_service = Mock(spec=EmbeddingService)
        vector_db_service = Mock(spec=VectorDBService)
        session_registry = Mock(spec=SessionRegistry)
        scoring_service = Mock(spec=ScoringService)

        search_service = SearchService(
            embedding_service=embedding_service,
            vector_db_service=vector_db_service,
            session_registry=session_registry,
            scoring_service=scoring_service
        )

        # Mock empty search results
        with patch.object(search_service, 'search', return_value=[]):
            handler = create_fork_detect_handler(
                search_service=search_service,
                claude_dir=str(temp_db_dir),
                session_registry=session_registry
            )

            result = handler({"query": "non-existent topic"})

            assert isinstance(result, str)
            assert "No Results Found" in result or "No relevant sessions" in result
            assert "non-existent topic" in result
            assert "Suggested Actions" in result

    def test_format_search_results_with_selection(self, temp_db_dir):
        """Test formatting search results with selection UI."""
        # Create mock results using SessionSearchResult objects
        results = [
            SessionSearchResult(
                session_id='session1',
                score=SessionScore(
                    session_id='session1',
                    final_score=0.95,
                    best_similarity=0.95,
                    avg_similarity=0.90,
                    chunk_ratio=0.80,
                    recency_score=0.85,
                    chain_quality=0.75,
                    memory_boost=0.0,
                    num_chunks_matched=5
                ),
                metadata=SessionMetadata(
                    session_id='session1',
                    created_at='2024-01-01T00:00:00',
                    message_count=10,
                    chunk_count=5
                ),
                preview='Authentication implementation',
                matched_chunks=[]
            )
        ]

        # Create session file
        session_file = temp_db_dir / 'session1.jsonl'
        session_file.write_text('{"role": "user", "content": "test"}\n')

        # Format results
        formatted = format_search_results_with_selection(
            query="authentication",
            results=results,
            claude_dir=str(temp_db_dir)
        )

        assert isinstance(formatted, str)
        assert "authentication" in formatted
        assert "Fork Detection" in formatted

    def test_format_search_results_empty(self, temp_db_dir):
        """Test formatting with empty results."""
        formatted = format_search_results_with_selection(
            query="test query",
            results=[],
            claude_dir=str(temp_db_dir)
        )

        assert isinstance(formatted, str)
        assert "No Results Found" in formatted or "No relevant sessions" in formatted
        assert "test query" in formatted


class TestMCPResponseFormat:
    """Test MCP response format compliance."""

    def test_response_has_jsonrpc_version(self):
        """Test all responses include JSON-RPC version."""
        server = create_server()
        client = MockMCPClient(server)

        # Test various methods
        methods = ["initialize", "tools/list"]
        for method in methods:
            response = client.send_request(method)
            assert response is not None
            assert "jsonrpc" in response
            assert response["jsonrpc"] == "2.0"

    def test_response_has_matching_id(self):
        """Test responses include matching request ID."""
        server = create_server()
        client = MockMCPClient(server)

        response1 = client.send_request("initialize")
        assert response1["id"] == 1

        response2 = client.send_request("tools/list")
        assert response2["id"] == 2

        response3 = client.send_request("tools/call", {
            "name": "fork-detect",
            "arguments": {"query": "test"}
        })
        assert response3["id"] == 3

    def test_success_response_has_result(self):
        """Test successful responses have result field."""
        server = create_server()
        client = MockMCPClient(server)

        response = client.send_request("initialize")
        assert "result" in response
        assert "error" not in response

    def test_error_response_has_error(self):
        """Test error responses have error field with code and message."""
        server = create_server()
        client = MockMCPClient(server)

        response = client.send_request("unknown/method")
        assert "error" in response
        assert "result" not in response
        assert "code" in response["error"]
        assert "message" in response["error"]
        assert isinstance(response["error"]["code"], int)
        assert isinstance(response["error"]["message"], str)

    def test_tools_list_response_format(self):
        """Test tools/list response matches MCP spec."""
        server = create_server()
        client = MockMCPClient(server)

        response = client.send_request("tools/list")
        assert "result" in response
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)

        # Check tool format
        for tool in response["result"]["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert isinstance(tool["inputSchema"], dict)

    def test_tools_call_response_format(self):
        """Test tools/call response matches MCP spec."""
        server = create_server()
        client = MockMCPClient(server)

        response = client.send_request("tools/call", {
            "name": "fork-detect",
            "arguments": {"query": "test"}
        })

        assert "result" in response
        assert "content" in response["result"]
        assert isinstance(response["result"]["content"], list)

        # Check content format
        for content_item in response["result"]["content"]:
            assert "type" in content_item
            assert content_item["type"] == "text"
            assert "text" in content_item
            assert isinstance(content_item["text"], str)


class TestErrorHandling:
    """Test error handling in MCP tool flow."""

    def test_service_not_initialized_error(self):
        """Test error handling when search service is not initialized."""
        handler = create_fork_detect_handler(search_service=None)
        result = handler({"query": "test"})

        assert "Service Not Initialized" in result
        assert "test" in result
        assert "Suggested Actions" in result or "Common Causes" in result

    def test_search_service_exception_handling(self):
        """Test graceful handling of search service exceptions."""
        # Create mock search service that raises exception
        mock_search_service = Mock(spec=SearchService)
        mock_search_service.search.side_effect = Exception("Database connection failed")

        handler = create_fork_detect_handler(search_service=mock_search_service)
        result = handler({"query": "test"})

        # Should return error message, not raise exception
        assert isinstance(result, str)
        assert "Error" in result or "Database connection failed" in result

    def test_malformed_request_handling(self):
        """Test handling of malformed requests."""
        server = create_server()

        # Missing jsonrpc field
        response = server.handle_request({
            "id": 1,
            "method": "initialize"
        })
        # Should still handle gracefully
        assert response is not None

        # Missing method field
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2
        })
        assert response is not None
        assert "error" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
