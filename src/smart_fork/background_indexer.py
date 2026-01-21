"""
Background indexing service for Claude Code session files.

Monitors ~/.claude/ directory for session file changes and indexes them
in the background with debouncing and checkpoint support.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Set, Callable
from threading import Thread, Lock, Event
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    FileModifiedEvent = None
    FileCreatedEvent = None

from .session_parser import SessionParser
from .chunking_service import ChunkingService
from .embedding_service import EmbeddingService
from .vector_db_service import VectorDBService
from .session_registry import SessionRegistry, SessionMetadata
from .session_summary_service import SessionSummaryService


logger = logging.getLogger(__name__)


@dataclass
class IndexingTask:
    """Represents a file indexing task."""

    file_path: Path
    last_modified: float
    message_count: int = 0
    last_indexed_count: int = 0

    def needs_indexing(self) -> bool:
        """Check if this task needs indexing."""
        return self.message_count > self.last_indexed_count


class SessionFileHandler(FileSystemEventHandler):
    """Handles file system events for session files."""

    def __init__(self, callback: Callable[[Path], None]):
        """
        Initialize the file handler.

        Args:
            callback: Function to call when a file is modified/created
        """
        super().__init__()
        self.callback = callback

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and str(event.src_path).endswith('.jsonl'):
            self.callback(Path(event.src_path))

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and str(event.src_path).endswith('.jsonl'):
            self.callback(Path(event.src_path))


class BackgroundIndexer:
    """
    Background service for monitoring and indexing Claude Code sessions.

    Features:
    - File system monitoring with watchdog
    - Debouncing (5-second delay after last modification)
    - Background thread pool processing
    - Checkpoint indexing (every 10-20 messages)
    - Graceful handling of rapid successive changes
    """

    def __init__(
        self,
        claude_dir: Path,
        vector_db: VectorDBService,
        session_registry: SessionRegistry,
        embedding_service: EmbeddingService,
        chunking_service: ChunkingService,
        session_parser: SessionParser,
        debounce_seconds: float = 5.0,
        checkpoint_interval: int = 15,
        max_workers: int = 2,
        summary_service: Optional[SessionSummaryService] = None
    ):
        """
        Initialize the background indexer.

        Args:
            claude_dir: Directory to monitor for session files (~/.claude/)
            vector_db: Vector database service
            session_registry: Session registry service
            embedding_service: Embedding service
            chunking_service: Chunking service
            session_parser: Session parser
            debounce_seconds: Delay after last modification before indexing
            checkpoint_interval: Number of messages between checkpoints
            max_workers: Maximum number of worker threads
            summary_service: Optional session summary service
        """
        self.claude_dir = Path(claude_dir)
        self.vector_db = vector_db
        self.session_registry = session_registry
        self.embedding_service = embedding_service
        self.chunking_service = chunking_service
        self.session_parser = session_parser
        self.debounce_seconds = debounce_seconds
        self.checkpoint_interval = checkpoint_interval
        self.max_workers = max_workers
        self.summary_service = summary_service or SessionSummaryService()

        # State management
        self._pending_tasks: Dict[str, IndexingTask] = {}
        self._tasks_lock = Lock()
        self._running = False
        self._stop_event = Event()
        self._observer = None
        self._monitor_thread = None
        self._executor = None

        # Statistics
        self._stats = {
            'files_indexed': 0,
            'chunks_added': 0,
            'errors': 0,
            'last_index_time': None
        }
        self._stats_lock = Lock()

    def start(self):
        """Start the background indexer."""
        if self._running:
            logger.warning("Background indexer already running")
            return

        logger.info(f"Starting background indexer monitoring {self.claude_dir}")
        self._running = True
        self._stop_event.clear()

        # Start thread pool
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Start file monitor if watchdog is available
        if Observer is not None:
            try:
                self._observer = Observer()
                event_handler = SessionFileHandler(self._on_file_changed)
                self._observer.schedule(event_handler, str(self.claude_dir), recursive=True)
                self._observer.start()
                logger.info("File system monitoring started")
            except Exception as e:
                logger.error(f"Failed to start file system monitoring: {e}")
                self._observer = None
        else:
            logger.warning("watchdog not available, file monitoring disabled")

        # Start background monitor thread
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        """Stop the background indexer."""
        if not self._running:
            return

        logger.info("Stopping background indexer")
        self._running = False
        self._stop_event.set()

        # Stop file observer
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        # Wait for monitor thread
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=10)
            self._monitor_thread = None

        # Shutdown executor
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

        logger.info("Background indexer stopped")

    def _on_file_changed(self, file_path: Path):
        """
        Handle file change event.

        Args:
            file_path: Path to the changed file
        """
        if not file_path.exists():
            return

        try:
            # Get file stats
            stat = file_path.stat()
            last_modified = stat.st_mtime

            # Count messages in file
            message_count = self._count_messages(file_path)

            with self._tasks_lock:
                session_id = file_path.stem

                # Check if we need to update the task
                if session_id in self._pending_tasks:
                    existing = self._pending_tasks[session_id]
                    # Only update if file is actually newer and has more messages
                    if last_modified > existing.last_modified or message_count > existing.message_count:
                        existing.last_modified = last_modified
                        existing.message_count = message_count
                        logger.debug(f"Updated task for {session_id}: {message_count} messages")
                else:
                    # Get last indexed count from registry
                    last_indexed_count = 0
                    session_meta = self.session_registry.get_session(session_id)
                    if session_meta:
                        last_indexed_count = session_meta.message_count

                    # Create new task
                    task = IndexingTask(
                        file_path=file_path,
                        last_modified=last_modified,
                        message_count=message_count,
                        last_indexed_count=last_indexed_count
                    )

                    # Only add if there's work to do
                    if task.needs_indexing():
                        self._pending_tasks[session_id] = task
                        logger.debug(f"Added task for {session_id}: {message_count} messages")

        except Exception as e:
            logger.error(f"Error handling file change for {file_path}: {e}")

    def _count_messages(self, file_path: Path) -> int:
        """
        Count messages in a session file.

        Args:
            file_path: Path to the session file

        Returns:
            Number of messages in the file
        """
        try:
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        count += 1
            return count
        except Exception as e:
            logger.error(f"Error counting messages in {file_path}: {e}")
            return 0

    def _monitor_loop(self):
        """Background monitoring loop for processing pending tasks."""
        logger.info("Monitor loop started")

        while self._running and not self._stop_event.is_set():
            try:
                # Check for tasks ready to process
                ready_tasks = []
                current_time = time.time()

                with self._tasks_lock:
                    for session_id, task in list(self._pending_tasks.items()):
                        # Check if task has been idle long enough (debouncing)
                        if current_time - task.last_modified >= self.debounce_seconds:
                            ready_tasks.append((session_id, task))
                            del self._pending_tasks[session_id]

                # Process ready tasks
                for session_id, task in ready_tasks:
                    if not self._running:
                        break

                    logger.info(f"Processing {session_id} ({task.message_count} messages)")

                    # Submit to thread pool
                    if self._executor is not None:
                        future = self._executor.submit(self._index_session, task)
                        # Don't block waiting for result

                # Sleep briefly before next check
                self._stop_event.wait(timeout=1.0)

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(1)

        logger.info("Monitor loop stopped")

    def _index_session(self, task: IndexingTask):
        """
        Index a session file.

        Args:
            task: Indexing task to process
        """
        session_id = task.file_path.stem

        try:
            # Parse session file
            session_data = self.session_parser.parse_file(task.file_path)

            if not session_data or not session_data.messages:
                logger.warning(f"No messages found in {session_id}")
                return

            # Check if this is a checkpoint update
            new_messages = session_data.messages[task.last_indexed_count:]

            if len(new_messages) < self.checkpoint_interval and task.last_indexed_count > 0:
                # Not enough new messages for checkpoint indexing
                logger.debug(f"Skipping {session_id}: only {len(new_messages)} new messages")
                return

            # Chunk the messages
            chunks = self.chunking_service.chunk_messages(session_data.messages)

            if not chunks:
                logger.warning(f"No chunks generated for {session_id}")
                return

            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)

            # Delete old chunks for this session
            self.vector_db.delete_session_chunks(session_id)

            # Add new chunks with embeddings
            chunk_texts = []
            chunk_metadata = []
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_ids.append(f"{session_id}_chunk_{i}")
                chunk_texts.append(chunk.content)
                metadata = {
                    'session_id': session_id,
                    'chunk_index': i,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'token_count': chunk.token_count
                }
                # Add memory_types if present
                if chunk.memory_types:
                    metadata['memory_types'] = chunk.memory_types
                chunk_metadata.append(metadata)

            self.vector_db.add_chunks(
                chunks=chunk_texts,
                embeddings=embeddings,
                metadata=chunk_metadata,
                chunk_ids=chunk_ids
            )

            # Generate session summary
            summary_text = None
            try:
                summary = self.summary_service.generate_summary(
                    session_data.messages,
                    session_id
                )
                summary_text = summary.summary
                logger.debug(f"Generated summary for {session_id}: {len(summary_text)} chars")
            except Exception as e:
                logger.warning(f"Failed to generate summary for {session_id}: {e}")

            # Update session registry
            project = task.file_path.parent.name if task.file_path.parent.name != '.claude' else 'default'

            created_at = session_data.created_at or datetime.now()
            # Convert datetime to ISO string if needed
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()

            session_metadata = SessionMetadata(
                session_id=session_id,
                project=project,
                created_at=created_at,
                message_count=len(session_data.messages),
                chunk_count=len(chunks),
                tags=[],
                summary=summary_text
            )
            self.session_registry.add_session(session_id, session_metadata)

            # Update statistics
            with self._stats_lock:
                self._stats['files_indexed'] += 1
                self._stats['chunks_added'] += len(chunks)
                self._stats['last_index_time'] = datetime.now()

            logger.info(f"Indexed {session_id}: {len(chunks)} chunks from {len(session_data.messages)} messages")

        except Exception as e:
            logger.error(f"Error indexing {session_id}: {e}")
            with self._stats_lock:
                self._stats['errors'] += 1

    def index_file(self, file_path: Path, force: bool = False):
        """
        Manually trigger indexing of a specific file.

        Args:
            file_path: Path to the session file
            force: If True, index even if already up-to-date
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        session_id = file_path.stem
        message_count = self._count_messages(file_path)

        # Get last indexed count
        last_indexed_count = 0
        if not force:
            session_meta = self.session_registry.get_session(session_id)
            if session_meta:
                last_indexed_count = session_meta.message_count

        task = IndexingTask(
            file_path=file_path,
            last_modified=file_path.stat().st_mtime,
            message_count=message_count,
            last_indexed_count=last_indexed_count
        )

        if force or task.needs_indexing():
            self._index_session(task)
        else:
            logger.info(f"Session {session_id} already up-to-date")

    def scan_directory(self, force: bool = False):
        """
        Scan the Claude directory and index all session files.

        Args:
            force: If True, re-index all files
        """
        if not self.claude_dir.exists():
            logger.warning(f"Claude directory not found: {self.claude_dir}")
            return

        session_files = list(self.claude_dir.rglob('*.jsonl'))
        logger.info(f"Found {len(session_files)} session files")

        for file_path in session_files:
            try:
                self.index_file(file_path, force=force)
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")

    def get_stats(self) -> Dict:
        """
        Get indexer statistics.

        Returns:
            Dictionary with statistics
        """
        with self._stats_lock:
            return self._stats.copy()

    def is_running(self) -> bool:
        """Check if the indexer is running."""
        return self._running

    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        with self._tasks_lock:
            return len(self._pending_tasks)
