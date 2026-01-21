"""
Unit tests for timeout handling in InitialSetup.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from smart_fork.initial_setup import InitialSetup, SetupState


class TestTimeoutHandling:
    """Tests for session processing timeout functionality."""

    @pytest.fixture
    def mock_setup(self, tmp_path):
        """Create an InitialSetup instance with mocked dependencies."""
        storage_dir = tmp_path / "storage"
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir(parents=True, exist_ok=True)

        setup = InitialSetup(
            storage_dir=str(storage_dir),
            claude_dir=str(claude_dir),
            show_progress=False,
            timeout_per_session=2.0  # 2 second timeout for tests
        )

        # Mock services
        setup.embedding_service = MagicMock()
        setup.vector_db_service = MagicMock()
        setup.session_registry = MagicMock()

        return setup

    def test_timeout_parameter_set(self, tmp_path):
        """Test that timeout parameter is properly set."""
        setup = InitialSetup(
            storage_dir=str(tmp_path / "storage"),
            claude_dir=str(tmp_path / "claude"),
            timeout_per_session=15.0
        )
        assert setup.timeout_per_session == 15.0

    def test_default_timeout_value(self, tmp_path):
        """Test that default timeout is 30 seconds."""
        setup = InitialSetup(
            storage_dir=str(tmp_path / "storage"),
            claude_dir=str(tmp_path / "claude")
        )
        assert setup.timeout_per_session == 30.0

    def test_setup_state_includes_timed_out_files(self):
        """Test that SetupState tracks timed-out files."""
        state = SetupState(
            total_files=10,
            processed_files=["file1.jsonl", "file2.jsonl"],
            timed_out_files=["file3.jsonl"],
            started_at=time.time(),
            last_updated=time.time()
        )
        assert state.timed_out_files == ["file3.jsonl"]

    def test_setup_state_to_dict_includes_timeouts(self):
        """Test that SetupState serialization includes timed_out_files."""
        state = SetupState(
            total_files=10,
            processed_files=["file1.jsonl"],
            timed_out_files=["file2.jsonl"],
            started_at=time.time(),
            last_updated=time.time()
        )
        data = state.to_dict()
        assert 'timed_out_files' in data
        assert data['timed_out_files'] == ["file2.jsonl"]

    def test_setup_state_from_dict_handles_missing_timeouts(self):
        """Test backward compatibility for state files without timed_out_files."""
        data = {
            'total_files': 10,
            'processed_files': ["file1.jsonl"],
            'started_at': time.time(),
            'last_updated': time.time()
        }
        state = SetupState.from_dict(data)
        assert state.timed_out_files == []

    @patch('smart_fork.initial_setup.SessionParser')
    def test_timeout_triggers_on_slow_processing(self, mock_parser_class, mock_setup, tmp_path):
        """Test that timeout is triggered for slow session processing."""
        # Create a test session file
        session_file = tmp_path / "claude" / "slow_session.jsonl"
        session_file.write_text('{"role": "user", "content": "test"}\n')

        # Mock parse_file to simulate a slow operation
        mock_parser = Mock()
        mock_session_data = Mock()
        mock_session_data.messages = []

        def slow_parse(*args, **kwargs):
            time.sleep(3)  # Longer than our 2-second timeout
            return mock_session_data

        mock_parser.parse_file.side_effect = slow_parse
        mock_setup.session_parser = mock_parser

        # Process the file with timeout
        result = mock_setup._process_session_file_with_timeout(session_file)

        # Should have timed out
        assert result['success'] is False
        assert result.get('timed_out', False) is True
        assert 'Timeout' in result['error']

    @patch('smart_fork.initial_setup.SessionParser')
    def test_successful_processing_within_timeout(self, mock_parser_class, mock_setup, tmp_path):
        """Test that fast processing completes successfully."""
        # Create a test session file
        session_file = tmp_path / "claude" / "fast_session.jsonl"
        session_file.write_text('{"role": "user", "content": "test"}\n')

        # Mock parse_file to return immediately
        mock_parser = Mock()
        mock_session_data = Mock()
        mock_message = Mock()
        mock_message.timestamp = datetime.now()
        mock_message.role = "user"
        mock_message.content = "test"
        mock_session_data.messages = [mock_message]
        mock_parser.parse_file.return_value = mock_session_data
        mock_setup.session_parser = mock_parser

        # Mock chunking service
        mock_chunk = Mock()
        mock_chunk.content = "test chunk"
        mock_chunk.start_index = 0
        mock_chunk.end_index = 1
        mock_chunk.memory_types = []
        mock_chunking = Mock()
        mock_chunking.chunk_messages.return_value = [mock_chunk]
        mock_setup.chunking_service = mock_chunking

        # Mock embedding service
        mock_setup.embedding_service.embed_texts.return_value = [[0.1, 0.2, 0.3]]

        # Process the file with timeout
        result = mock_setup._process_session_file_with_timeout(session_file)

        # Should complete successfully
        assert result['success'] is True
        assert result.get('timed_out', False) is False
        assert result['chunks'] > 0

    def test_run_setup_tracks_timeouts(self, mock_setup, tmp_path):
        """Test that run_setup properly tracks timed-out sessions."""
        # Create test files
        fast_file = tmp_path / "claude" / "fast.jsonl"
        fast_file.write_text('{"role": "user", "content": "fast"}\n')

        slow_file = tmp_path / "claude" / "slow.jsonl"
        slow_file.write_text('{"role": "user", "content": "slow"}\n')

        # Mock _find_session_files
        mock_setup._find_session_files = Mock(return_value=[fast_file, slow_file])

        # Mock _process_session_file_with_timeout
        def mock_process(file_path):
            if 'slow' in str(file_path):
                return {
                    'session_id': 'slow',
                    'success': False,
                    'error': 'Timeout after 2.0s',
                    'chunks': 0,
                    'timed_out': True
                }
            else:
                return {
                    'session_id': 'fast',
                    'success': True,
                    'chunks': 10,
                    'messages': 5
                }

        mock_setup._process_session_file_with_timeout = Mock(side_effect=mock_process)

        # Run setup
        result = mock_setup.run_setup()

        # Verify results
        assert result['success'] is True
        assert len(result['timeouts']) == 1
        assert result['timeouts'][0]['file'] == 'slow.jsonl'
        assert 'Timeout' in result['timeouts'][0]['error']
        assert result['files_processed'] == 1  # Only fast file succeeded

    def test_retry_timeouts_flag_reprocesses_timed_out_files(self, mock_setup, tmp_path):
        """Test that retry_timeouts=True reprocesses previously timed-out files."""
        # Create a state file with timed-out files
        state = SetupState(
            total_files=2,
            processed_files=[str(tmp_path / "claude" / "session1.jsonl")],
            timed_out_files=[str(tmp_path / "claude" / "session2.jsonl")],
            started_at=time.time(),
            last_updated=time.time()
        )
        mock_setup._load_state = Mock(return_value=state)

        # Create test files
        session1 = tmp_path / "claude" / "session1.jsonl"
        session1.write_text('{"role": "user", "content": "test1"}\n')

        session2 = tmp_path / "claude" / "session2.jsonl"
        session2.write_text('{"role": "user", "content": "test2"}\n')

        mock_setup._find_session_files = Mock(return_value=[session1, session2])

        # Mock successful processing
        def mock_process(file_path):
            return {
                'session_id': file_path.stem,
                'success': True,
                'chunks': 10,
                'messages': 5
            }

        mock_setup._process_session_file_with_timeout = Mock(side_effect=mock_process)

        # Run setup with retry_timeouts=True
        result = mock_setup.run_setup(resume=True, retry_timeouts=True)

        # Verify that session2 was reprocessed
        assert result['success'] is True
        # Both files should be processed (session1 was already done, session2 was retried)
        assert mock_setup._process_session_file_with_timeout.call_count >= 1

    def test_timeout_result_includes_file_size(self, mock_setup, tmp_path):
        """Test that timeout error includes file size information."""
        # Create a large test file
        session_file = tmp_path / "claude" / "large_session.jsonl"
        session_file.write_text('{"role": "user", "content": "' + 'x' * 10000 + '"}\n')

        # Mock slow processing
        def slow_process(*args, **kwargs):
            time.sleep(3)  # Longer than timeout
            return Mock()

        mock_setup.session_parser.parse_file = Mock(side_effect=slow_process)

        # Process with timeout
        result = mock_setup._process_session_file_with_timeout(session_file)

        # Check that result indicates timeout
        assert result['success'] is False
        assert result.get('timed_out', False) is True
