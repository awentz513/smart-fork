"""Integration tests for fork history MCP tools."""

import tempfile
from pathlib import Path

import pytest

from smart_fork.fork_history_service import ForkHistoryService
from smart_fork.server import (
    create_record_fork_handler,
    create_fork_history_handler
)


class TestRecordForkHandler:
    """Tests for record-fork MCP tool handler."""

    @pytest.fixture
    def temp_history_file(self):
        """Create a temporary history file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def service(self, temp_history_file):
        """Create a ForkHistoryService instance."""
        return ForkHistoryService(history_file=temp_history_file)

    @pytest.fixture
    def handler(self, service):
        """Create record-fork handler."""
        return create_record_fork_handler(service)

    def test_record_fork_success(self, handler, service):
        """Test successful fork recording."""
        result = handler({
            'session_id': 'session-001',
            'query': 'test query',
            'position': 1
        })

        assert 'recorded successfully' in result.lower()
        assert 'session-001' in result

        # Verify it was actually recorded
        recent = service.get_recent_forks(limit=1)
        assert len(recent) == 1
        assert recent[0].session_id == 'session-001'

    def test_record_fork_missing_session_id(self, handler):
        """Test error when session_id is missing."""
        result = handler({
            'query': 'test query',
            'position': 1
        })

        assert 'error' in result.lower()
        assert 'session_id' in result.lower()

    def test_record_fork_default_position(self, handler, service):
        """Test that position defaults to -1 when not provided."""
        result = handler({
            'session_id': 'session-001',
            'query': 'test query'
        })

        assert 'recorded successfully' in result.lower()

        recent = service.get_recent_forks(limit=1)
        assert recent[0].position == -1

    def test_record_fork_without_service(self):
        """Test error when service is not initialized."""
        handler = create_record_fork_handler(None)
        result = handler({
            'session_id': 'session-001',
            'query': 'test query'
        })

        assert 'error' in result.lower()
        assert 'not initialized' in result.lower()


class TestForkHistoryHandler:
    """Tests for get-fork-history MCP tool handler."""

    @pytest.fixture
    def temp_history_file(self):
        """Create a temporary history file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def service(self, temp_history_file):
        """Create a ForkHistoryService instance."""
        return ForkHistoryService(history_file=temp_history_file)

    @pytest.fixture
    def handler(self, service):
        """Create get-fork-history handler."""
        return create_fork_history_handler(service)

    def test_get_history_empty(self, handler):
        """Test getting history when empty."""
        result = handler({})

        assert 'no history yet' in result.lower()
        assert 'haven\'t forked' in result.lower()

    def test_get_history_with_data(self, handler, service):
        """Test getting history with fork data."""
        # Record some forks
        service.record_fork('session-001', 'query 1', 1)
        service.record_fork('session-002', 'query 2', 2)
        service.record_fork('session-003', 'query 3', -1)

        result = handler({'limit': 10})

        # Check that all sessions are mentioned
        assert 'session-001' in result
        assert 'session-002' in result
        assert 'session-003' in result

        # Check stats are included
        assert 'total forks' in result.lower()
        assert 'unique sessions' in result.lower()
        assert '3' in result  # Total forks count

    def test_get_history_with_limit(self, handler, service):
        """Test getting history with limit parameter."""
        # Record 5 forks
        for i in range(5):
            service.record_fork(f'session-{i:03d}', f'query {i}', 1)

        result = handler({'limit': 2})

        # Should only show last 2
        assert 'session-004' in result
        assert 'session-003' in result
        assert 'session-002' not in result

    def test_get_history_shows_positions(self, handler, service):
        """Test that result positions are displayed."""
        service.record_fork('session-001', 'query 1', 1)
        service.record_fork('session-002', 'query 2', 3)
        service.record_fork('session-003', 'query 3', -1)

        result = handler({'limit': 10})

        # Check position indicators
        assert '#1' in result  # Position 1
        assert '#3' in result  # Position 3
        assert 'custom' in result  # Position -1

    def test_get_history_truncates_long_queries(self, handler, service):
        """Test that long queries are truncated in display."""
        long_query = 'a' * 100
        service.record_fork('session-001', long_query, 1)

        result = handler({'limit': 1})

        # Should be truncated with ellipsis
        assert '...' in result
        # Should not show full query
        assert long_query not in result

    def test_get_history_without_service(self):
        """Test error when service is not initialized."""
        handler = create_fork_history_handler(None)
        result = handler({})

        assert 'error' in result.lower()
        assert 'not initialized' in result.lower()

    def test_get_history_includes_usage_hints(self, handler, service):
        """Test that result includes helpful usage hints."""
        service.record_fork('session-001', 'query 1', 1)

        result = handler({'limit': 10})

        # Check for hints
        assert 'get-session-preview' in result.lower()
        assert 'session id' in result.lower()


class TestEndToEndIntegration:
    """End-to-end tests for fork history workflow."""

    @pytest.fixture
    def temp_history_file(self):
        """Create a temporary history file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def service(self, temp_history_file):
        """Create a ForkHistoryService instance."""
        return ForkHistoryService(history_file=temp_history_file)

    @pytest.fixture
    def record_handler(self, service):
        """Create record-fork handler."""
        return create_record_fork_handler(service)

    @pytest.fixture
    def history_handler(self, service):
        """Create get-fork-history handler."""
        return create_fork_history_handler(service)

    def test_record_and_retrieve_workflow(self, record_handler, history_handler):
        """Test complete workflow of recording and retrieving forks."""
        # 1. Start with empty history
        result = history_handler({})
        assert 'no history yet' in result.lower()

        # 2. Record a fork
        result = record_handler({
            'session_id': 'session-001',
            'query': 'authentication',
            'position': 1
        })
        assert 'recorded successfully' in result.lower()

        # 3. Retrieve history
        result = history_handler({'limit': 10})
        assert 'session-001' in result
        assert 'authentication' in result
        assert '#1' in result

        # 4. Record more forks
        record_handler({
            'session_id': 'session-002',
            'query': 'database query',
            'position': 2
        })
        record_handler({
            'session_id': 'session-003',
            'query': 'custom entry',
            'position': -1
        })

        # 5. Retrieve updated history
        result = history_handler({'limit': 10})
        assert 'total forks recorded: 3' in result.lower()
        assert 'unique sessions forked: 3' in result.lower()

        # Most recent should be listed first
        lines = result.split('\n')
        session_003_idx = next(i for i, line in enumerate(lines) if 'session-003' in line)
        session_001_idx = next(i for i, line in enumerate(lines) if 'session-001' in line)
        assert session_003_idx < session_001_idx  # session-003 appears before session-001

    def test_repeated_forks_same_session(self, record_handler, history_handler, service):
        """Test forking from the same session multiple times."""
        # Fork from same session 3 times
        for i in range(3):
            record_handler({
                'session_id': 'session-001',
                'query': f'query {i}',
                'position': i + 1
            })

        # Check history
        result = history_handler({'limit': 10})
        assert 'total forks recorded: 3' in result.lower()
        assert 'unique sessions forked: 1' in result.lower()

        # Check service stats directly
        stats = service.get_stats()
        assert stats['total_forks'] == 3
        assert stats['unique_sessions'] == 1
        assert stats['position_distribution'] == {1: 1, 2: 1, 3: 1}
