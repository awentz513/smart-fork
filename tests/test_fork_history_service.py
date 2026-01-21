"""Tests for ForkHistoryService."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from smart_fork.fork_history_service import ForkHistoryService, ForkHistoryEntry


class TestForkHistoryEntry:
    """Tests for ForkHistoryEntry dataclass."""

    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = ForkHistoryEntry(
            session_id="test-session-001",
            timestamp="2026-01-21T10:00:00Z",
            query="test query",
            position=1
        )

        result = entry.to_dict()

        assert result == {
            'session_id': "test-session-001",
            'timestamp': "2026-01-21T10:00:00Z",
            'query': "test query",
            'position': 1
        }

    def test_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            'session_id': "test-session-001",
            'timestamp': "2026-01-21T10:00:00Z",
            'query': "test query",
            'position': 2
        }

        entry = ForkHistoryEntry.from_dict(data)

        assert entry.session_id == "test-session-001"
        assert entry.timestamp == "2026-01-21T10:00:00Z"
        assert entry.query == "test query"
        assert entry.position == 2


class TestForkHistoryService:
    """Tests for ForkHistoryService."""

    @pytest.fixture
    def temp_history_file(self):
        """Create a temporary history file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def service(self, temp_history_file):
        """Create a ForkHistoryService instance with temp file."""
        return ForkHistoryService(history_file=temp_history_file, max_entries=10)

    def test_initialization_creates_empty_file(self, temp_history_file):
        """Test that initialization creates an empty history file."""
        # Remove the temp file so we can test initialization
        Path(temp_history_file).unlink(missing_ok=True)

        service = ForkHistoryService(history_file=temp_history_file)

        assert Path(temp_history_file).exists()
        with open(temp_history_file) as f:
            data = json.load(f)
            assert data == []

    def test_record_fork_basic(self, service, temp_history_file):
        """Test recording a basic fork event."""
        service.record_fork(
            session_id="session-001",
            query="test query",
            position=1
        )

        # Verify file was written
        with open(temp_history_file) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]['session_id'] == "session-001"
            assert data[0]['query'] == "test query"
            assert data[0]['position'] == 1
            assert 'timestamp' in data[0]

    def test_record_fork_multiple(self, service):
        """Test recording multiple fork events."""
        service.record_fork("session-001", "query 1", 1)
        service.record_fork("session-002", "query 2", 2)
        service.record_fork("session-003", "query 3", -1)

        recent = service.get_recent_forks(limit=10)

        assert len(recent) == 3
        # Most recent should be first
        assert recent[0].session_id == "session-003"
        assert recent[1].session_id == "session-002"
        assert recent[2].session_id == "session-001"

    def test_record_fork_respects_max_entries(self, temp_history_file):
        """Test that max_entries limit is enforced."""
        service = ForkHistoryService(history_file=temp_history_file, max_entries=5)

        # Record 10 forks
        for i in range(10):
            service.record_fork(f"session-{i:03d}", f"query {i}", i % 5 + 1)

        recent = service.get_recent_forks(limit=100)

        # Should only keep last 5
        assert len(recent) == 5
        # Most recent should be session-009
        assert recent[0].session_id == "session-009"
        assert recent[4].session_id == "session-005"

    def test_get_recent_forks_with_limit(self, service):
        """Test getting recent forks with limit."""
        # Record 10 forks
        for i in range(10):
            service.record_fork(f"session-{i:03d}", f"query {i}", 1)

        recent = service.get_recent_forks(limit=3)

        assert len(recent) == 3
        assert recent[0].session_id == "session-009"
        assert recent[1].session_id == "session-008"
        assert recent[2].session_id == "session-007"

    def test_get_recent_forks_empty(self, service):
        """Test getting recent forks when history is empty."""
        recent = service.get_recent_forks(limit=10)
        assert recent == []

    def test_get_forks_for_session(self, service):
        """Test getting forks for a specific session."""
        # Record forks from multiple sessions
        service.record_fork("session-001", "query 1", 1)
        service.record_fork("session-002", "query 2", 2)
        service.record_fork("session-001", "query 3", 3)
        service.record_fork("session-003", "query 4", 1)
        service.record_fork("session-001", "query 5", -1)

        forks = service.get_forks_for_session("session-001")

        assert len(forks) == 3
        # All should be from session-001
        for fork in forks:
            assert fork.session_id == "session-001"

        # Check order (most recent first)
        assert forks[0].query == "query 5"
        assert forks[1].query == "query 3"
        assert forks[2].query == "query 1"

    def test_get_forks_for_session_not_found(self, service):
        """Test getting forks for a session that doesn't exist."""
        service.record_fork("session-001", "query 1", 1)

        forks = service.get_forks_for_session("nonexistent-session")
        assert forks == []

    def test_get_stats_empty(self, service):
        """Test getting stats when history is empty."""
        stats = service.get_stats()

        assert stats['total_forks'] == 0
        assert stats['unique_sessions'] == 0
        assert stats['position_distribution'] == {}

    def test_get_stats_with_data(self, service):
        """Test getting stats with fork history."""
        # Record various forks
        service.record_fork("session-001", "query 1", 1)
        service.record_fork("session-001", "query 2", 1)
        service.record_fork("session-002", "query 3", 2)
        service.record_fork("session-003", "query 4", 3)
        service.record_fork("session-002", "query 5", -1)

        stats = service.get_stats()

        assert stats['total_forks'] == 5
        assert stats['unique_sessions'] == 3
        assert stats['position_distribution'] == {1: 2, 2: 1, 3: 1, -1: 1}
        assert 'most_recent' in stats

    def test_clear(self, service):
        """Test clearing fork history."""
        # Record some forks
        service.record_fork("session-001", "query 1", 1)
        service.record_fork("session-002", "query 2", 2)

        # Verify history exists
        assert len(service.get_recent_forks()) == 2

        # Clear history
        service.clear()

        # Verify history is empty
        assert len(service.get_recent_forks()) == 0
        stats = service.get_stats()
        assert stats['total_forks'] == 0

    def test_thread_safety(self, service):
        """Test thread-safe concurrent access."""
        import threading

        def record_forks(start_idx):
            for i in range(10):
                service.record_fork(
                    f"session-{start_idx}-{i:03d}",
                    f"query {start_idx}-{i}",
                    1
                )

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=record_forks, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify all forks were recorded
        stats = service.get_stats()
        # Should have min(50, max_entries) forks (max_entries=10 in fixture)
        assert stats['total_forks'] == 10  # Limited by max_entries

    def test_timestamp_format(self, service):
        """Test that timestamps are in ISO format with Z suffix."""
        service.record_fork("session-001", "query 1", 1)

        recent = service.get_recent_forks(limit=1)
        timestamp = recent[0].timestamp

        # Should end with Z
        assert timestamp.endswith('Z')

        # Should be parseable as ISO format
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        assert isinstance(dt, datetime)

    def test_custom_position(self, service):
        """Test recording fork with custom position (-1)."""
        service.record_fork("session-001", "custom entry", -1)

        recent = service.get_recent_forks(limit=1)
        assert recent[0].position == -1

    def test_persistence_across_instances(self, temp_history_file):
        """Test that history persists across service instances."""
        # Create first instance and record forks
        service1 = ForkHistoryService(history_file=temp_history_file)
        service1.record_fork("session-001", "query 1", 1)
        service1.record_fork("session-002", "query 2", 2)

        # Create second instance with same file
        service2 = ForkHistoryService(history_file=temp_history_file)
        recent = service2.get_recent_forks()

        # Should see forks from service1
        assert len(recent) == 2
        assert recent[0].session_id == "session-002"
        assert recent[1].session_id == "session-001"
