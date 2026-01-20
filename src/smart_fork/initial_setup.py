"""
Initial database setup flow for Smart Fork.

This module handles the first-run experience, scanning existing Claude Code sessions
and building the initial vector database and session registry.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict

from .session_parser import SessionParser
from .chunking_service import ChunkingService
from .embedding_service import EmbeddingService
from .vector_db_service import VectorDBService
from .session_registry import SessionRegistry

logger = logging.getLogger(__name__)


@dataclass
class SetupProgress:
    """Progress information for setup process."""
    total_files: int
    processed_files: int
    current_file: str
    total_chunks: int
    elapsed_time: float
    estimated_remaining: float
    is_complete: bool = False
    error: Optional[str] = None


@dataclass
class SetupState:
    """Persistent state for resuming interrupted setup."""
    total_files: int
    processed_files: List[str]
    started_at: float
    last_updated: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SetupState':
        """Create from dictionary."""
        return SetupState(**data)


class InitialSetup:
    """
    Manages the initial database setup flow.

    Responsibilities:
    - Detect first-run (no ~/.smart-fork/ directory)
    - Scan ~/.claude/ for all existing session files
    - Display progress with estimated time remaining
    - Support graceful interruption and resume
    - Create session registry on completion
    """

    def __init__(
        self,
        storage_dir: str = "~/.smart-fork",
        claude_dir: str = "~/.claude",
        progress_callback: Optional[Callable[[SetupProgress], None]] = None
    ):
        """
        Initialize the setup manager.

        Args:
            storage_dir: Directory for Smart Fork data
            claude_dir: Directory containing Claude Code sessions
            progress_callback: Optional callback for progress updates
        """
        self.storage_dir = Path(storage_dir).expanduser()
        self.claude_dir = Path(claude_dir).expanduser()
        self.progress_callback = progress_callback

        self.state_file = self.storage_dir / "setup_state.json"
        self.session_parser = SessionParser()
        self.chunking_service = ChunkingService()

        # Services will be initialized during setup
        self.embedding_service: Optional[EmbeddingService] = None
        self.vector_db_service: Optional[VectorDBService] = None
        self.session_registry: Optional[SessionRegistry] = None

        self._interrupted = False

    def is_first_run(self) -> bool:
        """
        Check if this is the first run (no storage directory exists).

        Returns:
            True if this is the first run
        """
        return not self.storage_dir.exists()

    def has_incomplete_setup(self) -> bool:
        """
        Check if there's an incomplete setup that can be resumed.

        Returns:
            True if there's a setup state file
        """
        return self.state_file.exists()

    def _find_session_files(self) -> List[Path]:
        """
        Find all session JSONL files in the Claude directory.

        Returns:
            List of session file paths
        """
        if not self.claude_dir.exists():
            logger.warning(f"Claude directory not found: {self.claude_dir}")
            return []

        session_files = []

        # Search for .jsonl files recursively
        for jsonl_file in self.claude_dir.rglob("*.jsonl"):
            # Skip files that are too small (likely empty or invalid)
            if jsonl_file.stat().st_size > 100:
                session_files.append(jsonl_file)

        logger.info(f"Found {len(session_files)} session files")
        return sorted(session_files)

    def _load_state(self) -> Optional[SetupState]:
        """
        Load setup state from disk.

        Returns:
            SetupState if found, None otherwise
        """
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            return SetupState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load setup state: {e}")
            return None

    def _save_state(self, state: SetupState) -> None:
        """
        Save setup state to disk.

        Args:
            state: State to save
        """
        try:
            # Ensure directory exists
            self.storage_dir.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic rename
            temp_file.rename(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save setup state: {e}")

    def _delete_state(self) -> None:
        """Delete the setup state file."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
        except Exception as e:
            logger.error(f"Failed to delete setup state: {e}")

    def _initialize_services(self) -> None:
        """Initialize all required services."""
        logger.info("Initializing services...")

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.embedding_service = EmbeddingService()
        self.vector_db_service = VectorDBService(
            persist_directory=str(self.storage_dir / "vector_db")
        )
        self.session_registry = SessionRegistry(
            storage_path=str(self.storage_dir / "session-registry.json")
        )

    def _process_session_file(
        self,
        file_path: Path,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single session file.

        Args:
            file_path: Path to session file
            session_id: Optional session ID (derived from filename if not provided)

        Returns:
            Dictionary with processing results
        """
        if session_id is None:
            session_id = file_path.stem

        try:
            # Parse session
            session_data = self.session_parser.parse_file(str(file_path))

            if not session_data.messages:
                logger.warning(f"No messages in session {session_id}")
                return {
                    'session_id': session_id,
                    'success': False,
                    'error': 'No messages found',
                    'chunks': 0
                }

            # Chunk messages
            chunks = self.chunking_service.chunk_messages(session_data.messages)

            if not chunks:
                logger.warning(f"No chunks generated for session {session_id}")
                return {
                    'session_id': session_id,
                    'success': False,
                    'error': 'No chunks generated',
                    'chunks': 0
                }

            # Extract text from chunks
            chunk_texts = [chunk.text for chunk in chunks]

            # Generate embeddings
            embeddings = self.embedding_service.embed_texts(chunk_texts)

            # Prepare chunks for vector DB
            chunk_records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_records.append({
                    'id': f"{session_id}_{i}",
                    'text': chunk.text,
                    'embedding': embedding,
                    'metadata': {
                        'session_id': session_id,
                        'chunk_index': i,
                        'start_index': chunk.start_index,
                        'end_index': chunk.end_index,
                        'message_count': len(chunk.message_indices)
                    }
                })

            # Add to vector database
            self.vector_db_service.add_chunks(chunk_records)

            # Add to session registry
            project = self._extract_project(file_path)
            created_at = session_data.messages[0].timestamp if session_data.messages else None

            self.session_registry.add_session(
                session_id=session_id,
                project=project,
                created_at=created_at,
                chunk_count=len(chunks),
                message_count=len(session_data.messages),
                tags=[]
            )

            return {
                'session_id': session_id,
                'success': True,
                'chunks': len(chunks),
                'messages': len(session_data.messages)
            }

        except Exception as e:
            logger.error(f"Error processing session {session_id}: {e}")
            return {
                'session_id': session_id,
                'success': False,
                'error': str(e),
                'chunks': 0
            }

    def _extract_project(self, file_path: Path) -> str:
        """
        Extract project name from file path.

        Args:
            file_path: Path to session file

        Returns:
            Project name
        """
        # Try to get project from parent directory name
        # Typical structure: ~/.claude/projects/{project_name}/sessions/{session_id}.jsonl
        parts = file_path.parts
        if 'projects' in parts:
            idx = parts.index('projects')
            if idx + 1 < len(parts):
                return parts[idx + 1]

        # Default to "unknown"
        return "unknown"

    def _estimate_remaining_time(
        self,
        processed: int,
        total: int,
        elapsed: float
    ) -> float:
        """
        Estimate remaining time based on current progress.

        Args:
            processed: Number of files processed
            total: Total number of files
            elapsed: Elapsed time in seconds

        Returns:
            Estimated remaining time in seconds
        """
        if processed == 0:
            return 0.0

        avg_time_per_file = elapsed / processed
        remaining_files = total - processed
        return avg_time_per_file * remaining_files

    def _notify_progress(
        self,
        total: int,
        processed: int,
        current_file: str,
        total_chunks: int,
        start_time: float,
        is_complete: bool = False,
        error: Optional[str] = None
    ) -> None:
        """
        Notify progress callback.

        Args:
            total: Total number of files
            processed: Number of files processed
            current_file: Current file being processed
            total_chunks: Total chunks processed
            start_time: Start time of setup
            is_complete: Whether setup is complete
            error: Optional error message
        """
        if self.progress_callback is None:
            return

        elapsed = time.time() - start_time
        estimated_remaining = self._estimate_remaining_time(processed, total, elapsed)

        progress = SetupProgress(
            total_files=total,
            processed_files=processed,
            current_file=current_file,
            total_chunks=total_chunks,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
            is_complete=is_complete,
            error=error
        )

        self.progress_callback(progress)

    def interrupt(self) -> None:
        """Signal that setup should be interrupted gracefully."""
        self._interrupted = True
        logger.info("Setup interrupted by user")

    def run_setup(self, resume: bool = False) -> Dict[str, Any]:
        """
        Run the initial setup process.

        Args:
            resume: Whether to resume from previous state

        Returns:
            Dictionary with setup results
        """
        # Find all session files
        all_files = self._find_session_files()

        if not all_files:
            logger.warning("No session files found")
            return {
                'success': True,
                'files_processed': 0,
                'total_chunks': 0,
                'errors': [],
                'message': 'No session files found'
            }

        # Load or create state
        state = None
        if resume:
            state = self._load_state()

        if state is None:
            # New setup
            state = SetupState(
                total_files=len(all_files),
                processed_files=[],
                started_at=time.time(),
                last_updated=time.time()
            )

        # Initialize services
        self._initialize_services()

        # Process files
        start_time = state.started_at
        total_chunks = 0
        errors = []

        for file_path in all_files:
            # Check if already processed
            file_str = str(file_path)
            if file_str in state.processed_files:
                continue

            # Check for interruption
            if self._interrupted:
                logger.info("Setup interrupted, saving state...")
                self._save_state(state)
                return {
                    'success': False,
                    'files_processed': len(state.processed_files),
                    'total_chunks': total_chunks,
                    'errors': errors,
                    'interrupted': True,
                    'message': 'Setup interrupted, can be resumed later'
                }

            # Notify progress
            self._notify_progress(
                total=len(all_files),
                processed=len(state.processed_files),
                current_file=file_path.name,
                total_chunks=total_chunks,
                start_time=start_time
            )

            # Process file
            result = self._process_session_file(file_path)

            if result['success']:
                total_chunks += result['chunks']
                state.processed_files.append(file_str)
            else:
                errors.append({
                    'file': file_path.name,
                    'error': result.get('error', 'Unknown error')
                })

            # Update state
            state.last_updated = time.time()
            self._save_state(state)

        # Setup complete
        self._delete_state()

        # Final progress notification
        self._notify_progress(
            total=len(all_files),
            processed=len(state.processed_files),
            current_file="",
            total_chunks=total_chunks,
            start_time=start_time,
            is_complete=True
        )

        return {
            'success': True,
            'files_processed': len(state.processed_files),
            'total_chunks': total_chunks,
            'errors': errors,
            'elapsed_time': time.time() - start_time
        }
