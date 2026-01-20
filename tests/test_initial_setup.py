"""
Unit tests for InitialSetup class.
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

from smart_fork.initial_setup import (
    InitialSetup,
    SetupProgress,
    SetupState
)


class TestSetupProgress:
    """Tests for SetupProgress dataclass."""

    def test_setup_progress_creation(self):
        """Test creating a SetupProgress instance."""
        progress = SetupProgress(
            total_files=100,
            processed_files=50,
            current_file="session_123.jsonl",
            total_chunks=1000,
            elapsed_time=120.0,
            estimated_remaining=120.0
        )

        assert progress.total_files == 100
        assert progress.processed_files == 50
        assert progress.current_file == "session_123.jsonl"
        assert progress.total_chunks == 1000
        assert progress.elapsed_time == 120.0
        assert progress.estimated_remaining == 120.0
        assert progress.is_complete is False
        assert progress.error is None

    def test_setup_progress_complete(self):
        """Test SetupProgress with is_complete flag."""
        progress = SetupProgress(
            total_files=100,
            processed_files=100,
            current_file="",
            total_chunks=2000,
            elapsed_time=300.0,
            estimated_remaining=0.0,
            is_complete=True
        )

        assert progress.is_complete is True

    def test_setup_progress_with_error(self):
        """Test SetupProgress with error."""
        progress = SetupProgress(
            total_files=100,
            processed_files=50,
            current_file="session_bad.jsonl",
            total_chunks=1000,
            elapsed_time=120.0,
            estimated_remaining=120.0,
            error="Failed to parse session"
        )

        assert progress.error == "Failed to parse session"


class TestSetupState:
    """Tests for SetupState dataclass."""

    def test_setup_state_creation(self):
        """Test creating a SetupState instance."""
        state = SetupState(
            total_files=100,
            processed_files=["file1.jsonl", "file2.jsonl"],
            started_at=time.time(),
            last_updated=time.time()
        )

        assert state.total_files == 100
        assert len(state.processed_files) == 2
        assert "file1.jsonl" in state.processed_files

    def test_setup_state_to_dict(self):
        """Test converting SetupState to dictionary."""
        started = time.time()
        state = SetupState(
            total_files=50,
            processed_files=["file1.jsonl"],
            started_at=started,
            last_updated=started
        )

        data = state.to_dict()
        assert data['total_files'] == 50
        assert data['processed_files'] == ["file1.jsonl"]
        assert data['started_at'] == started
        assert data['last_updated'] == started

    def test_setup_state_from_dict(self):
        """Test creating SetupState from dictionary."""
        started = time.time()
        data = {
            'total_files': 75,
            'processed_files': ["file1.jsonl", "file2.jsonl"],
            'started_at': started,
            'last_updated': started
        }

        state = SetupState.from_dict(data)
        assert state.total_files == 75
        assert len(state.processed_files) == 2
        assert state.started_at == started

    def test_setup_state_roundtrip(self):
        """Test SetupState serialization round-trip."""
        started = time.time()
        original = SetupState(
            total_files=100,
            processed_files=["a.jsonl", "b.jsonl", "c.jsonl"],
            started_at=started,
            last_updated=started + 10
        )

        data = original.to_dict()
        restored = SetupState.from_dict(data)

        assert restored.total_files == original.total_files
        assert restored.processed_files == original.processed_files
        assert restored.started_at == original.started_at
        assert restored.last_updated == original.last_updated


class TestInitialSetupInit:
    """Tests for InitialSetup initialization."""

    def test_init_default_paths(self):
        """Test initialization with default paths."""
        setup = InitialSetup()

        assert setup.storage_dir == Path("~/.smart-fork").expanduser()
        assert setup.claude_dir == Path("~/.claude").expanduser()
        assert setup.progress_callback is None

    def test_init_custom_paths(self):
        """Test initialization with custom paths."""
        setup = InitialSetup(
            storage_dir="/tmp/smart-fork-test",
            claude_dir="/tmp/claude-test"
        )

        assert setup.storage_dir == Path("/tmp/smart-fork-test")
        assert setup.claude_dir == Path("/tmp/claude-test")

    def test_init_with_callback(self):
        """Test initialization with progress callback."""
        callback = Mock()
        setup = InitialSetup(progress_callback=callback)

        assert setup.progress_callback is callback

    def test_init_services_none(self):
        """Test that services are initially None."""
        setup = InitialSetup()

        assert setup.embedding_service is None
        assert setup.vector_db_service is None
        assert setup.session_registry is None


class TestInitialSetupFirstRun:
    """Tests for first-run detection."""

    def test_is_first_run_when_dir_missing(self, tmp_path):
        """Test is_first_run returns True when directory doesn't exist."""
        storage_dir = tmp_path / "nonexistent"
        setup = InitialSetup(storage_dir=str(storage_dir))

        assert setup.is_first_run() is True

    def test_is_first_run_when_dir_exists(self, tmp_path):
        """Test is_first_run returns False when directory exists."""
        storage_dir = tmp_path / "existing"
        storage_dir.mkdir()
        setup = InitialSetup(storage_dir=str(storage_dir))

        assert setup.is_first_run() is False

    def test_has_incomplete_setup_when_missing(self, tmp_path):
        """Test has_incomplete_setup returns False when no state file."""
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()
        setup = InitialSetup(storage_dir=str(storage_dir))

        assert setup.has_incomplete_setup() is False

    def test_has_incomplete_setup_when_exists(self, tmp_path):
        """Test has_incomplete_setup returns True when state file exists."""
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()
        state_file = storage_dir / "setup_state.json"
        state_file.write_text("{}")

        setup = InitialSetup(storage_dir=str(storage_dir))
        assert setup.has_incomplete_setup() is True


class TestInitialSetupSessionFiles:
    """Tests for finding session files."""

    def test_find_session_files_empty_dir(self, tmp_path):
        """Test finding session files in empty directory."""
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()

        setup = InitialSetup(claude_dir=str(claude_dir))
        files = setup._find_session_files()

        assert len(files) == 0

    def test_find_session_files_nonexistent_dir(self, tmp_path):
        """Test finding session files when directory doesn't exist."""
        claude_dir = tmp_path / "nonexistent"

        setup = InitialSetup(claude_dir=str(claude_dir))
        files = setup._find_session_files()

        assert len(files) == 0

    def test_find_session_files_with_sessions(self, tmp_path):
        """Test finding session files."""
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()

        # Create some session files
        (claude_dir / "session1.jsonl").write_text("x" * 200)
        (claude_dir / "session2.jsonl").write_text("x" * 200)

        # Create a file that's too small (should be ignored)
        (claude_dir / "small.jsonl").write_text("x")

        # Create a non-jsonl file (should be ignored)
        (claude_dir / "other.txt").write_text("x" * 200)

        setup = InitialSetup(claude_dir=str(claude_dir))
        files = setup._find_session_files()

        assert len(files) == 2
        assert all(f.suffix == ".jsonl" for f in files)

    def test_find_session_files_recursive(self, tmp_path):
        """Test finding session files recursively."""
        claude_dir = tmp_path / "claude"
        projects_dir = claude_dir / "projects" / "myproject"
        projects_dir.mkdir(parents=True)

        # Create sessions at different levels
        (claude_dir / "session1.jsonl").write_text("x" * 200)
        (projects_dir / "session2.jsonl").write_text("x" * 200)

        setup = InitialSetup(claude_dir=str(claude_dir))
        files = setup._find_session_files()

        assert len(files) == 2


class TestInitialSetupState:
    """Tests for state management."""

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state."""
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()

        setup = InitialSetup(storage_dir=str(storage_dir))

        # Create state
        started = time.time()
        state = SetupState(
            total_files=100,
            processed_files=["file1.jsonl", "file2.jsonl"],
            started_at=started,
            last_updated=started + 10
        )

        # Save state
        setup._save_state(state)

        # Load state
        loaded = setup._load_state()

        assert loaded is not None
        assert loaded.total_files == 100
        assert len(loaded.processed_files) == 2
        assert loaded.started_at == started

    def test_load_state_missing_file(self, tmp_path):
        """Test loading state when file doesn't exist."""
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()

        setup = InitialSetup(storage_dir=str(storage_dir))
        loaded = setup._load_state()

        assert loaded is None

    def test_load_state_invalid_json(self, tmp_path):
        """Test loading state with invalid JSON."""
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()
        state_file = storage_dir / "setup_state.json"
        state_file.write_text("invalid json{")

        setup = InitialSetup(storage_dir=str(storage_dir))
        loaded = setup._load_state()

        assert loaded is None

    def test_delete_state(self, tmp_path):
        """Test deleting state file."""
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()
        state_file = storage_dir / "setup_state.json"
        state_file.write_text("{}")

        setup = InitialSetup(storage_dir=str(storage_dir))
        assert state_file.exists()

        setup._delete_state()
        assert not state_file.exists()


class TestInitialSetupExtractProject:
    """Tests for project extraction."""

    def test_extract_project_from_projects_path(self):
        """Test extracting project from path with 'projects' directory."""
        setup = InitialSetup()
        file_path = Path("/home/user/.claude/projects/myproject/sessions/session1.jsonl")

        project = setup._extract_project(file_path)
        assert project == "myproject"

    def test_extract_project_without_projects_path(self):
        """Test extracting project from path without 'projects' directory."""
        setup = InitialSetup()
        file_path = Path("/home/user/.claude/session1.jsonl")

        project = setup._extract_project(file_path)
        assert project == "unknown"

    def test_extract_project_edge_case(self):
        """Test extracting project from path where 'projects' is last."""
        setup = InitialSetup()
        file_path = Path("/home/user/.claude/projects")

        project = setup._extract_project(file_path)
        assert project == "unknown"


class TestInitialSetupEstimateTime:
    """Tests for time estimation."""

    def test_estimate_remaining_time_zero_processed(self):
        """Test estimating time when no files processed."""
        setup = InitialSetup()
        remaining = setup._estimate_remaining_time(0, 100, 0.0)

        assert remaining == 0.0

    def test_estimate_remaining_time_half_complete(self):
        """Test estimating time when half complete."""
        setup = InitialSetup()
        remaining = setup._estimate_remaining_time(50, 100, 100.0)

        assert remaining == 100.0  # Should take another 100s

    def test_estimate_remaining_time_nearly_complete(self):
        """Test estimating time when nearly complete."""
        setup = InitialSetup()
        remaining = setup._estimate_remaining_time(99, 100, 99.0)

        assert remaining == pytest.approx(1.0, rel=0.01)


class TestInitialSetupProgressNotification:
    """Tests for progress notification."""

    def test_notify_progress_with_callback(self):
        """Test that progress notification calls callback."""
        callback = Mock()
        setup = InitialSetup(progress_callback=callback)

        setup._notify_progress(
            total=100,
            processed=50,
            current_file="session.jsonl",
            total_chunks=1000,
            start_time=time.time() - 60
        )

        callback.assert_called_once()
        progress = callback.call_args[0][0]
        assert isinstance(progress, SetupProgress)
        assert progress.total_files == 100
        assert progress.processed_files == 50

    def test_notify_progress_without_callback(self):
        """Test that progress notification without callback doesn't crash."""
        setup = InitialSetup(progress_callback=None)

        # Should not raise exception
        setup._notify_progress(
            total=100,
            processed=50,
            current_file="session.jsonl",
            total_chunks=1000,
            start_time=time.time()
        )

    def test_notify_progress_complete(self):
        """Test progress notification when complete."""
        callback = Mock()
        setup = InitialSetup(progress_callback=callback)

        setup._notify_progress(
            total=100,
            processed=100,
            current_file="",
            total_chunks=2000,
            start_time=time.time() - 300,
            is_complete=True
        )

        progress = callback.call_args[0][0]
        assert progress.is_complete is True


class TestInitialSetupInterruption:
    """Tests for graceful interruption."""

    def test_interrupt_sets_flag(self):
        """Test that interrupt() sets the _interrupted flag."""
        setup = InitialSetup()
        assert setup._interrupted is False

        setup.interrupt()
        assert setup._interrupted is True


class TestInitialSetupIntegration:
    """Integration tests for setup process."""

    @patch('smart_fork.initial_setup.EmbeddingService')
    @patch('smart_fork.initial_setup.VectorDBService')
    @patch('smart_fork.initial_setup.SessionRegistry')
    def test_initialize_services(
        self,
        mock_registry,
        mock_vector_db,
        mock_embedding,
        tmp_path
    ):
        """Test service initialization."""
        storage_dir = tmp_path / "storage"
        setup = InitialSetup(storage_dir=str(storage_dir))

        setup._initialize_services()

        assert storage_dir.exists()
        mock_embedding.assert_called_once()
        mock_vector_db.assert_called_once()
        mock_registry.assert_called_once()

    @patch('smart_fork.initial_setup.SessionParser')
    @patch('smart_fork.initial_setup.ChunkingService')
    def test_process_session_file_mock(
        self,
        mock_chunking,
        mock_parser,
        tmp_path
    ):
        """Test processing a session file with mocks."""
        storage_dir = tmp_path / "storage"
        setup = InitialSetup(storage_dir=str(storage_dir))

        # Mock services
        setup.embedding_service = Mock()
        setup.vector_db_service = Mock()
        setup.session_registry = Mock()

        # Mock session data
        mock_session_data = Mock()
        mock_session_data.messages = [
            Mock(timestamp="2024-01-01T00:00:00Z", content="test")
        ]
        setup.session_parser.parse_file.return_value = mock_session_data

        # Mock chunks
        mock_chunk = Mock()
        mock_chunk.text = "test chunk"
        mock_chunk.start_index = 0
        mock_chunk.end_index = 1
        mock_chunk.message_indices = [0]
        setup.chunking_service.chunk_messages.return_value = [mock_chunk]

        # Mock embeddings
        setup.embedding_service.embed_texts.return_value = [[0.1] * 768]

        # Create test file
        test_file = tmp_path / "test_session.jsonl"
        test_file.write_text('{"role": "user", "content": "test"}')

        # Process file
        result = setup._process_session_file(test_file)

        assert result['success'] is True
        assert result['chunks'] == 1
        setup.vector_db_service.add_chunks.assert_called_once()
        setup.session_registry.add_session.assert_called_once()

    def test_run_setup_no_files(self, tmp_path):
        """Test running setup with no session files."""
        storage_dir = tmp_path / "storage"
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()

        setup = InitialSetup(
            storage_dir=str(storage_dir),
            claude_dir=str(claude_dir)
        )

        result = setup.run_setup()

        assert result['success'] is True
        assert result['files_processed'] == 0
        assert 'No session files found' in result['message']

    @patch('smart_fork.initial_setup.EmbeddingService')
    @patch('smart_fork.initial_setup.VectorDBService')
    @patch('smart_fork.initial_setup.SessionRegistry')
    def test_run_setup_with_interruption(
        self,
        mock_registry,
        mock_vector_db,
        mock_embedding,
        tmp_path
    ):
        """Test that setup handles interruption gracefully."""
        storage_dir = tmp_path / "storage"
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()

        # Create test files
        (claude_dir / "session1.jsonl").write_text("x" * 200)
        (claude_dir / "session2.jsonl").write_text("x" * 200)

        setup = InitialSetup(
            storage_dir=str(storage_dir),
            claude_dir=str(claude_dir)
        )

        # Interrupt immediately
        setup._interrupted = True

        result = setup.run_setup()

        assert result['success'] is False
        assert result.get('interrupted') is True
        assert 'can be resumed' in result['message']
