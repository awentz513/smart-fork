"""
Performance and stress tests for Smart Fork Detection system.

Tests cover:
- Indexing 1000+ messages without RAM exhaustion
- Search with 10,000 chunks in database
- Search latency <3s at 95th percentile
- Concurrent indexing and searching
- Memory usage stays under 2GB
- Database size scaling
"""

import os
import time
import psutil
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import pytest

from smart_fork.session_parser import SessionParser, SessionMessage
from smart_fork.chunking_service import ChunkingService
from smart_fork.embedding_service import EmbeddingService
from smart_fork.vector_db_service import VectorDBService
from smart_fork.scoring_service import ScoringService
from smart_fork.session_registry import SessionRegistry
from smart_fork.search_service import SearchService


class PerformanceMonitor:
    """Monitor system performance metrics during tests."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = None
        self.peak_memory = 0
        self.start_time = None

    def start(self):
        """Start monitoring."""
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.start_memory
        self.start_time = time.time()

    def check(self):
        """Check current memory usage."""
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory

    def report(self):
        """Get performance report."""
        elapsed = time.time() - self.start_time
        current_memory = self.check()
        return {
            'elapsed_seconds': elapsed,
            'start_memory_mb': self.start_memory,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': current_memory - self.start_memory
        }


def generate_large_session_file(file_path: str, num_messages: int, project: str = "test-project"):
    """Generate a large session file with specified number of messages."""
    messages = []

    base_time = datetime.now()
    for i in range(num_messages):
        # Alternate between user and assistant
        role = "user" if i % 2 == 0 else "assistant"

        # Generate varied content
        if role == "user":
            content = f"Test query {i}: Can you help me implement feature {i}? I need to create a function that handles {' '.join(['data'] * 10)}."
        else:
            content = f"Test response {i}: Here's the implementation for feature {i}.\n\n```python\ndef feature_{i}():\n    # Implementation here\n    data = {{'key': 'value', 'index': {i}}}\n    result = process_data(data)\n    return result\n```\n\nThis implementation follows the pattern we discussed earlier."

        message = {
            "role": role,
            "content": content,
            "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
            "model": "claude-sonnet-4.5",
            "id": f"msg_{i}"
        }
        messages.append(message)

    # Write JSONL file
    with open(file_path, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(json.dumps(msg) + '\n')


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp(prefix='smart_fork_perf_')
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def services(temp_storage):
    """Initialize all services with temporary storage."""
    storage_path = Path(temp_storage)

    # Initialize services
    embedding_service = EmbeddingService()
    vector_db = VectorDBService(str(storage_path / "vector_db"))
    scoring_service = ScoringService()
    session_registry = SessionRegistry(str(storage_path / "registry.json"))
    chunking_service = ChunkingService()
    search_service = SearchService(
        embedding_service=embedding_service,
        vector_db=vector_db,
        scoring_service=scoring_service,
        session_registry=session_registry
    )

    return {
        'embedding': embedding_service,
        'vector_db': vector_db,
        'scoring': scoring_service,
        'registry': session_registry,
        'chunking': chunking_service,
        'search': search_service,
        'storage_path': storage_path
    }


class TestPerformanceIndexing:
    """Test indexing performance with large datasets."""

    def test_index_1000_messages_no_ram_exhaustion(self, services, temp_storage):
        """Test indexing 1000+ messages without RAM exhaustion."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Generate session file with 1000 messages
        session_file = Path(temp_storage) / "large_session.jsonl"
        generate_large_session_file(str(session_file), 1000)

        # Parse session
        parser = SessionParser()
        session_data = parser.parse_file(str(session_file))

        assert len(session_data.messages) == 1000, "Should parse all 1000 messages"

        # Chunk messages
        chunks = services['chunking'].chunk_messages(session_data.messages)

        assert len(chunks) > 0, "Should create chunks"

        # Index chunks in batches
        session_id = "perf_test_1000"
        batch_size = 50

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Embed batch
            texts = [chunk.content for chunk in batch]
            embeddings = services['embedding'].embed_texts(texts)

            # Add to vector DB
            services['vector_db'].add_chunks(
                embeddings=embeddings,
                texts=texts,
                session_ids=[session_id] * len(texts),
                chunk_indices=list(range(i, i + len(texts))),
                metadatas=[{'created_at': datetime.now().isoformat()}] * len(texts)
            )

            # Check memory periodically
            current_memory = monitor.check()

        # Add to registry
        services['registry'].add_session(
            session_id=session_id,
            project="test-project",
            created_at=datetime.now().isoformat(),
            chunk_count=len(chunks),
            message_count=1000
        )

        report = monitor.report()

        # Verify memory constraints
        assert report['peak_memory_mb'] < 2000, f"Peak memory {report['peak_memory_mb']:.2f}MB should be under 2GB"

        # Verify indexing completed successfully
        stats = services['vector_db'].get_stats()
        assert stats['total_chunks'] >= len(chunks), "All chunks should be indexed"

        print(f"\n1000 Message Indexing Performance:")
        print(f"  - Elapsed: {report['elapsed_seconds']:.2f}s")
        print(f"  - Peak Memory: {report['peak_memory_mb']:.2f}MB")
        print(f"  - Chunks Created: {len(chunks)}")
        print(f"  - Throughput: {1000 / report['elapsed_seconds']:.2f} messages/second")

    def test_index_multiple_sessions_2000_messages(self, services, temp_storage):
        """Test indexing multiple sessions totaling 2000+ messages."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Generate 5 sessions with 400 messages each
        num_sessions = 5
        messages_per_session = 400

        for session_idx in range(num_sessions):
            session_file = Path(temp_storage) / f"session_{session_idx}.jsonl"
            generate_large_session_file(str(session_file), messages_per_session, f"project_{session_idx % 2}")

            # Parse and index
            parser = SessionParser()
            session_data = parser.parse_file(str(session_file))

            chunks = services['chunking'].chunk_messages(session_data.messages)

            session_id = f"perf_session_{session_idx}"

            # Embed and index in one batch per session
            texts = [chunk.content for chunk in chunks]
            embeddings = services['embedding'].embed_texts(texts)

            services['vector_db'].add_chunks(
                embeddings=embeddings,
                texts=texts,
                session_ids=[session_id] * len(texts),
                chunk_indices=list(range(len(texts))),
                metadatas=[{'created_at': datetime.now().isoformat()}] * len(texts)
            )

            services['registry'].add_session(
                session_id=session_id,
                project=f"project_{session_idx % 2}",
                created_at=datetime.now().isoformat(),
                chunk_count=len(chunks),
                message_count=messages_per_session
            )

            monitor.check()

        report = monitor.report()

        # Verify memory constraints
        assert report['peak_memory_mb'] < 2000, f"Peak memory {report['peak_memory_mb']:.2f}MB should be under 2GB"

        # Verify all sessions indexed
        stats = services['vector_db'].get_stats()
        assert stats['total_chunks'] > 0, "Should have indexed chunks"

        registry_stats = services['registry'].get_stats()
        assert registry_stats['total_sessions'] == num_sessions, "All sessions should be registered"

        total_messages = num_sessions * messages_per_session
        throughput = total_messages / report['elapsed_seconds']

        assert throughput > 100, f"Throughput {throughput:.2f} messages/second should exceed 100 messages/second"

        print(f"\nMultiple Session Indexing Performance:")
        print(f"  - Sessions: {num_sessions}")
        print(f"  - Total Messages: {total_messages}")
        print(f"  - Elapsed: {report['elapsed_seconds']:.2f}s")
        print(f"  - Peak Memory: {report['peak_memory_mb']:.2f}MB")
        print(f"  - Throughput: {throughput:.2f} messages/second")


class TestPerformanceSearch:
    """Test search performance with large databases."""

    def test_search_with_10000_chunks(self, services, temp_storage):
        """Test search with 10,000 chunks in database."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Generate enough sessions to create ~10,000 chunks
        # Estimate: 400 messages = ~50-80 chunks, so ~130 sessions needed
        target_chunks = 10000
        messages_per_session = 400
        estimated_chunks_per_session = 60
        num_sessions = target_chunks // estimated_chunks_per_session + 5

        print(f"\nGenerating {num_sessions} sessions to create ~{target_chunks} chunks...")

        total_chunks = 0
        for session_idx in range(num_sessions):
            session_file = Path(temp_storage) / f"search_session_{session_idx}.jsonl"
            generate_large_session_file(str(session_file), messages_per_session)

            parser = SessionParser()
            session_data = parser.parse_file(str(session_file))
            chunks = services['chunking'].chunk_messages(session_data.messages)

            session_id = f"search_session_{session_idx}"
            texts = [chunk.content for chunk in chunks]
            embeddings = services['embedding'].embed_texts(texts)

            services['vector_db'].add_chunks(
                embeddings=embeddings,
                texts=texts,
                session_ids=[session_id] * len(texts),
                chunk_indices=list(range(len(texts))),
                metadatas=[{
                    'created_at': (datetime.now() - timedelta(days=session_idx)).isoformat()
                }] * len(texts)
            )

            services['registry'].add_session(
                session_id=session_id,
                project=f"project_{session_idx % 3}",
                created_at=(datetime.now() - timedelta(days=session_idx)).isoformat(),
                chunk_count=len(chunks),
                message_count=messages_per_session
            )

            total_chunks += len(chunks)

            if total_chunks >= target_chunks:
                break

        setup_report = monitor.report()
        print(f"Setup complete: {total_chunks} chunks indexed in {setup_report['elapsed_seconds']:.2f}s")

        # Verify we have enough chunks
        stats = services['vector_db'].get_stats()
        assert stats['total_chunks'] >= 10000, f"Should have at least 10,000 chunks, got {stats['total_chunks']}"

        # Now test search performance with multiple queries
        test_queries = [
            "implement feature with data processing",
            "create function that handles data",
            "python implementation pattern",
            "help me implement feature",
            "test query with data handling"
        ]

        search_times = []

        for query in test_queries:
            start_time = time.time()
            results = services['search'].search(query, top_n_sessions=5)
            search_time = time.time() - start_time
            search_times.append(search_time)

            assert len(results) > 0, "Should return results"

        # Calculate 95th percentile
        search_times.sort()
        percentile_95_idx = int(len(search_times) * 0.95)
        percentile_95 = search_times[percentile_95_idx] if percentile_95_idx < len(search_times) else search_times[-1]
        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)

        final_report = monitor.report()

        # Verify performance targets
        assert percentile_95 < 3.0, f"95th percentile search time {percentile_95:.2f}s should be under 3 seconds"
        assert final_report['peak_memory_mb'] < 2000, f"Peak memory {final_report['peak_memory_mb']:.2f}MB should be under 2GB"

        print(f"\nSearch Performance with {stats['total_chunks']} chunks:")
        print(f"  - Queries Tested: {len(test_queries)}")
        print(f"  - Avg Search Time: {avg_search_time:.3f}s")
        print(f"  - Max Search Time: {max_search_time:.3f}s")
        print(f"  - 95th Percentile: {percentile_95:.3f}s")
        print(f"  - Peak Memory: {final_report['peak_memory_mb']:.2f}MB")

    def test_search_latency_95th_percentile(self, services, temp_storage):
        """Test that search latency is <3s at 95th percentile with realistic data."""
        # Create moderate dataset (1000 chunks)
        num_sessions = 20
        messages_per_session = 200

        for session_idx in range(num_sessions):
            session_file = Path(temp_storage) / f"latency_session_{session_idx}.jsonl"
            generate_large_session_file(str(session_file), messages_per_session)

            parser = SessionParser()
            session_data = parser.parse_file(str(session_file))
            chunks = services['chunking'].chunk_messages(session_data.messages)

            session_id = f"latency_session_{session_idx}"
            texts = [chunk.content for chunk in chunks]
            embeddings = services['embedding'].embed_texts(texts)

            services['vector_db'].add_chunks(
                embeddings=embeddings,
                texts=texts,
                session_ids=[session_id] * len(texts),
                chunk_indices=list(range(len(texts))),
                metadatas=[{'created_at': datetime.now().isoformat()}] * len(texts)
            )

            services['registry'].add_session(
                session_id=session_id,
                project="latency-test",
                created_at=datetime.now().isoformat(),
                chunk_count=len(chunks),
                message_count=messages_per_session
            )

        # Run 100 search queries
        search_times = []
        for i in range(100):
            query = f"implement feature {i % 10} with data processing"
            start_time = time.time()
            results = services['search'].search(query, top_n_sessions=5)
            search_time = time.time() - start_time
            search_times.append(search_time)

        # Calculate statistics
        search_times.sort()
        percentile_95_idx = int(len(search_times) * 0.95)
        percentile_95 = search_times[percentile_95_idx]
        avg_time = sum(search_times) / len(search_times)

        assert percentile_95 < 3.0, f"95th percentile {percentile_95:.3f}s should be under 3 seconds"

        print(f"\nLatency Test (100 queries):")
        print(f"  - Average: {avg_time:.3f}s")
        print(f"  - 95th Percentile: {percentile_95:.3f}s")
        print(f"  - Min: {min(search_times):.3f}s")
        print(f"  - Max: {max(search_times):.3f}s")


class TestPerformanceConcurrent:
    """Test concurrent operations."""

    def test_concurrent_indexing_and_searching(self, services, temp_storage):
        """Test concurrent indexing and searching operations."""
        monitor = PerformanceMonitor()
        monitor.start()

        # First, create some initial data for searching
        for i in range(5):
            session_file = Path(temp_storage) / f"initial_session_{i}.jsonl"
            generate_large_session_file(str(session_file), 100)

            parser = SessionParser()
            session_data = parser.parse_file(str(session_file))
            chunks = services['chunking'].chunk_messages(session_data.messages)

            session_id = f"initial_session_{i}"
            texts = [chunk.content for chunk in chunks]
            embeddings = services['embedding'].embed_texts(texts)

            services['vector_db'].add_chunks(
                embeddings=embeddings,
                texts=texts,
                session_ids=[session_id] * len(texts),
                chunk_indices=list(range(len(texts))),
                metadatas=[{'created_at': datetime.now().isoformat()}] * len(texts)
            )

            services['registry'].add_session(
                session_id=session_id,
                project="concurrent-test",
                created_at=datetime.now().isoformat(),
                chunk_count=len(chunks),
                message_count=100
            )

        # Define indexing task
        def index_session(session_idx):
            session_file = Path(temp_storage) / f"concurrent_session_{session_idx}.jsonl"
            generate_large_session_file(str(session_file), 50)

            parser = SessionParser()
            session_data = parser.parse_file(str(session_file))
            chunks = services['chunking'].chunk_messages(session_data.messages)

            session_id = f"concurrent_session_{session_idx}"
            texts = [chunk.content for chunk in chunks]
            embeddings = services['embedding'].embed_texts(texts)

            services['vector_db'].add_chunks(
                embeddings=embeddings,
                texts=texts,
                session_ids=[session_id] * len(texts),
                chunk_indices=list(range(len(texts))),
                metadatas=[{'created_at': datetime.now().isoformat()}] * len(texts)
            )

            services['registry'].add_session(
                session_id=session_id,
                project="concurrent-test",
                created_at=datetime.now().isoformat(),
                chunk_count=len(chunks),
                message_count=50
            )

            return len(chunks)

        # Define search task
        def search_query(query_idx):
            query = f"implement feature {query_idx}"
            results = services['search'].search(query, top_n_sessions=3)
            return len(results)

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit indexing tasks
            indexing_futures = [executor.submit(index_session, i) for i in range(10)]

            # Submit search tasks
            search_futures = [executor.submit(search_query, i) for i in range(20)]

            # Collect results
            indexing_results = []
            for future in as_completed(indexing_futures):
                try:
                    result = future.result()
                    indexing_results.append(result)
                    monitor.check()
                except Exception as e:
                    pytest.fail(f"Indexing task failed: {e}")

            search_results = []
            for future in as_completed(search_futures):
                try:
                    result = future.result()
                    search_results.append(result)
                    monitor.check()
                except Exception as e:
                    pytest.fail(f"Search task failed: {e}")

        report = monitor.report()

        # Verify results
        assert len(indexing_results) == 10, "All indexing tasks should complete"
        assert len(search_results) == 20, "All search tasks should complete"
        assert all(r > 0 for r in search_results), "All searches should return results"
        assert report['peak_memory_mb'] < 2000, f"Peak memory {report['peak_memory_mb']:.2f}MB should be under 2GB"

        print(f"\nConcurrent Operations Performance:")
        print(f"  - Indexing Tasks: {len(indexing_results)}")
        print(f"  - Search Tasks: {len(search_results)}")
        print(f"  - Elapsed: {report['elapsed_seconds']:.2f}s")
        print(f"  - Peak Memory: {report['peak_memory_mb']:.2f}MB")


class TestPerformanceMemory:
    """Test memory usage constraints."""

    def test_memory_usage_under_2gb(self, services, temp_storage):
        """Test that memory usage stays under 2GB during heavy operations."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Generate large dataset
        num_sessions = 30
        messages_per_session = 300

        memory_samples = []

        for session_idx in range(num_sessions):
            session_file = Path(temp_storage) / f"memory_session_{session_idx}.jsonl"
            generate_large_session_file(str(session_file), messages_per_session)

            parser = SessionParser()
            session_data = parser.parse_file(str(session_file))
            chunks = services['chunking'].chunk_messages(session_data.messages)

            session_id = f"memory_session_{session_idx}"
            texts = [chunk.content for chunk in chunks]
            embeddings = services['embedding'].embed_texts(texts)

            services['vector_db'].add_chunks(
                embeddings=embeddings,
                texts=texts,
                session_ids=[session_id] * len(texts),
                chunk_indices=list(range(len(texts))),
                metadatas=[{'created_at': datetime.now().isoformat()}] * len(texts)
            )

            services['registry'].add_session(
                session_id=session_id,
                project="memory-test",
                created_at=datetime.now().isoformat(),
                chunk_count=len(chunks),
                message_count=messages_per_session
            )

            # Sample memory every few sessions
            current_memory = monitor.check()
            memory_samples.append(current_memory)

        # Perform searches to test memory under mixed load
        for i in range(20):
            query = f"implement feature {i} with data processing"
            services['search'].search(query, top_n_sessions=5)
            memory_samples.append(monitor.check())

        report = monitor.report()

        # Verify memory constraints
        assert report['peak_memory_mb'] < 2000, f"Peak memory {report['peak_memory_mb']:.2f}MB should be under 2GB"
        assert all(m < 2000 for m in memory_samples), "All memory samples should be under 2GB"

        print(f"\nMemory Usage Test:")
        print(f"  - Sessions Indexed: {num_sessions}")
        print(f"  - Peak Memory: {report['peak_memory_mb']:.2f}MB")
        print(f"  - Average Memory: {sum(memory_samples) / len(memory_samples):.2f}MB")
        print(f"  - Memory Samples: {len(memory_samples)}")


class TestPerformanceDatabaseSize:
    """Test database size scaling."""

    def test_database_size_scaling(self, services, temp_storage):
        """Test database size scaling (~500KB per 1000 messages)."""
        # Index sessions with known message counts
        num_sessions = 5
        messages_per_session = 1000
        total_messages = num_sessions * messages_per_session

        for session_idx in range(num_sessions):
            session_file = Path(temp_storage) / f"size_session_{session_idx}.jsonl"
            generate_large_session_file(str(session_file), messages_per_session)

            parser = SessionParser()
            session_data = parser.parse_file(str(session_file))
            chunks = services['chunking'].chunk_messages(session_data.messages)

            session_id = f"size_session_{session_idx}"
            texts = [chunk.content for chunk in chunks]
            embeddings = services['embedding'].embed_texts(texts)

            services['vector_db'].add_chunks(
                embeddings=embeddings,
                texts=texts,
                session_ids=[session_id] * len(texts),
                chunk_indices=list(range(len(texts))),
                metadatas=[{'created_at': datetime.now().isoformat()}] * len(texts)
            )

            services['registry'].add_session(
                session_id=session_id,
                project="size-test",
                created_at=datetime.now().isoformat(),
                chunk_count=len(chunks),
                message_count=messages_per_session
            )

        # Calculate database size
        storage_path = services['storage_path']
        total_size_bytes = 0

        for root, dirs, files in os.walk(storage_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size_bytes += os.path.getsize(file_path)

        total_size_kb = total_size_bytes / 1024
        total_size_mb = total_size_kb / 1024

        # Calculate size per 1000 messages
        size_per_1000_messages_kb = (total_size_kb / total_messages) * 1000

        # Verify scaling is reasonable (should be approximately 500KB per 1000 messages)
        # Allow for variation due to metadata, ChromaDB overhead, etc.
        # Being generous here - expecting 200KB to 2MB per 1000 messages
        assert 200 < size_per_1000_messages_kb < 2000, \
            f"Database size per 1000 messages ({size_per_1000_messages_kb:.2f}KB) should be reasonable"

        print(f"\nDatabase Size Scaling:")
        print(f"  - Total Messages: {total_messages}")
        print(f"  - Total Database Size: {total_size_mb:.2f}MB ({total_size_kb:.2f}KB)")
        print(f"  - Size per 1000 Messages: {size_per_1000_messages_kb:.2f}KB")
        print(f"  - Estimated size for 50,000 messages: {(size_per_1000_messages_kb * 50):.2f}KB ({(size_per_1000_messages_kb * 50 / 1024):.2f}MB)")


if __name__ == '__main__':
    # Run with: pytest tests/test_performance.py -v -s
    pytest.main([__file__, '-v', '-s'])
