"""Integration tests for duplicate detection functionality.

These tests require network access to load embedding models from HuggingFace.
They are marked to skip if network is unavailable (e.g., in sandboxed CI environments).
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.smart_fork.duplicate_detection_service import DuplicateDetectionService
from src.smart_fork.vector_db_service import VectorDBService
from src.smart_fork.embedding_service import EmbeddingService
from src.smart_fork.session_registry import SessionRegistry, SessionMetadata
from src.smart_fork.chunking_service import ChunkingService


# Check if network is available by trying to import cached model
def _has_network_access():
    """Check if HuggingFace model access is available."""
    try:
        import os
        # Check if model is cached locally
        cache_path = Path.home() / ".cache" / "huggingface" / "hub"
        nomic_cached = any(
            "nomic-embed-text" in str(p) for p in cache_path.glob("**/nomic*")
        ) if cache_path.exists() else False
        return nomic_cached or os.environ.get("ALLOW_NETWORK_TESTS", "0") == "1"
    except Exception:
        return False


requires_network = pytest.mark.skipif(
    not _has_network_access(),
    reason="Integration tests require cached embedding model or network access"
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def embedding_service(temp_storage_dir):
    """Create an EmbeddingService instance."""
    # Use temp directory for cache
    cache_dir = Path(temp_storage_dir) / "embedding_cache"
    return EmbeddingService(use_cache=True, cache_dir=str(cache_dir))


@pytest.fixture
def vector_db_service(temp_storage_dir):
    """Create a VectorDBService instance."""
    db_dir = Path(temp_storage_dir) / "vector_db"
    return VectorDBService(persist_directory=str(db_dir))


@pytest.fixture
def session_registry(temp_storage_dir):
    """Create a SessionRegistry instance."""
    registry_path = Path(temp_storage_dir) / "session-registry.json"
    return SessionRegistry(registry_path=str(registry_path))


@pytest.fixture
def duplicate_service(vector_db_service, session_registry):
    """Create a DuplicateDetectionService instance."""
    return DuplicateDetectionService(
        vector_db_service=vector_db_service,
        session_registry=session_registry,
        similarity_threshold=0.85,
        min_chunks_for_comparison=2  # Lower for tests
    )


@pytest.fixture
def chunking_service():
    """Create a ChunkingService instance."""
    return ChunkingService()


@requires_network
class TestDuplicateDetectionIntegration:
    """Integration tests for duplicate detection.

    These tests require the embedding model to be loaded, which needs network or cached model.
    """

    def test_end_to_end_duplicate_detection(
        self,
        duplicate_service,
        vector_db_service,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test full flow: index sessions -> find duplicates."""
        # Create two similar sessions
        session1_content = [
            "User: How do I use React hooks?",
            "Assistant: React hooks are functions that let you use state and lifecycle features in function components.",
            "User: Can you show me an example?",
            "Assistant: Here's a simple useState example: const [count, setCount] = useState(0);"
        ]

        session2_content = [
            "User: What are React hooks?",
            "Assistant: Hooks are special functions in React that allow you to use state and other React features in functional components.",
            "User: Show me how to use them",
            "Assistant: Here's how to use useState: const [value, setValue] = useState(initialValue);"
        ]

        session3_content = [
            "User: How do I configure PostgreSQL?",
            "Assistant: To configure PostgreSQL, you need to edit the postgresql.conf file.",
            "User: Where is that file located?",
            "Assistant: It's typically in /etc/postgresql/ on Linux systems."
        ]

        # Helper to index a session
        def index_session(session_id: str, content_list: list):
            session_text = "\n".join(content_list)
            chunks = chunking_service.chunk_text(session_text)
            embeddings = embedding_service.embed_texts(chunks)
            # Create metadata for each chunk
            metadata = [
                {"session_id": session_id, "chunk_index": i}
                for i in range(len(chunks))
            ]
            vector_db_service.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            session_registry.register_session(SessionMetadata(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                message_count=len(content_list)
            ))

        # Index sessions
        index_session("session1", session1_content)
        index_session("session2", session2_content)
        index_session("session3", session3_content)

        # Test: Get similar sessions for session1
        similar_to_session1 = duplicate_service.get_similar_sessions("session1", top_k=5)

        # session2 should be similar, session3 should not
        assert len(similar_to_session1) >= 1
        similar_ids = [s.session_id for s in similar_to_session1]
        assert "session2" in similar_ids

        # session3 might or might not be in results depending on similarity
        # but if it is, it should have lower similarity than session2
        if "session3" in similar_ids:
            session2_sim = next(s.similarity for s in similar_to_session1 if s.session_id == "session2")
            session3_sim = next(s.similarity for s in similar_to_session1 if s.session_id == "session3")
            assert session2_sim > session3_sim

    def test_session_embedding_consistency(
        self,
        duplicate_service,
        vector_db_service,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test that session embeddings are consistent across calls."""
        # Create and index a session
        session_content = [
            "User: Explain Python decorators",
            "Assistant: Decorators are functions that modify other functions.",
            "User: Can you show an example?",
            "Assistant: Sure! Here's a simple decorator: @my_decorator"
        ]

        session_text = "\n".join(session_content)
        chunks = chunking_service.chunk_text(session_text)
        embeddings = embedding_service.embed_texts(chunks)
        metadata = [
            {"session_id": "test_session", "chunk_index": i}
            for i in range(len(chunks))
        ]

        vector_db_service.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )
        session_registry.register_session(SessionMetadata(
            session_id="test_session",
            created_at=datetime.now().isoformat(),
            message_count=len(session_content)
        ))

        # Compute embedding twice
        emb1 = duplicate_service.compute_session_embedding("test_session")
        emb2 = duplicate_service.compute_session_embedding("test_session")

        # Should be identical
        assert emb1 is not None
        assert emb2 is not None
        assert len(emb1) == len(emb2)

        # Check similarity (should be 1.0 or very close)
        similarity = duplicate_service.compute_similarity(emb1, emb2)
        assert similarity > 0.999

    def test_find_duplicate_pairs_with_threshold(
        self,
        duplicate_service,
        vector_db_service,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test finding duplicate pairs with similarity threshold."""
        # Create 3 sessions: 2 very similar (duplicates), 1 different
        similar_content_1 = "How to implement authentication in Django? I need JWT tokens."
        similar_content_2 = "How do I add JWT authentication to my Django app?"
        different_content = "What are the best practices for React state management?"

        for session_id, content in [
            ("dup1", similar_content_1),
            ("dup2", similar_content_2),
            ("diff", different_content)
        ]:
            chunks = chunking_service.chunk_text(content)
            embeddings = embedding_service.embed_texts(chunks)
            metadata = [
                {"session_id": session_id, "chunk_index": i}
                for i in range(len(chunks))
            ]

            vector_db_service.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            session_registry.register_session(SessionMetadata(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                message_count=1
            ))

        # Find all duplicate pairs
        pairs = duplicate_service.find_all_duplicate_pairs()

        # Should find at least the dup1-dup2 pair
        assert len(pairs) >= 1

        # Check that dup1 and dup2 are paired
        found_duplicate_pair = False
        for sid1, sid2, similarity in pairs:
            if {sid1, sid2} == {"dup1", "dup2"}:
                found_duplicate_pair = True
                assert similarity >= 0.85  # Above threshold
                break

        assert found_duplicate_pair, "Expected to find dup1-dup2 pair"

    def test_stats_reflect_indexed_sessions(
        self,
        duplicate_service,
        vector_db_service,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test that stats reflect the actual state of indexed sessions."""
        # Initially empty
        stats = duplicate_service.get_stats()
        assert stats['total_sessions'] == 0
        assert stats['sessions_eligible_for_comparison'] == 0

        # Add one session with enough chunks
        content = "Test content for session 1. This is chunk 1. This is chunk 2. This is chunk 3."
        chunks = chunking_service.chunk_text(content)
        embeddings = embedding_service.embed_texts(chunks)
        metadata = [
            {"session_id": "session1", "chunk_index": i}
            for i in range(len(chunks))
        ]

        vector_db_service.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )
        session_registry.register_session(SessionMetadata(
            session_id="session1",
            created_at=datetime.now().isoformat(),
            message_count=1
        ))

        # Stats should reflect the new session
        stats = duplicate_service.get_stats()
        assert stats['total_sessions'] == 1
        # Eligible count depends on chunk count vs min_chunks_for_comparison

    def test_no_duplicates_in_unique_sessions(
        self,
        duplicate_service,
        vector_db_service,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test that truly unique sessions don't flag as duplicates."""
        unique_contents = [
            "How do I sort a list in Python?",
            "What is the capital of France?",
            "Explain quantum computing basics"
        ]

        for i, content in enumerate(unique_contents):
            session_id = f"unique_{i}"
            chunks = chunking_service.chunk_text(content)
            embeddings = embedding_service.embed_texts(chunks)
            metadata = [
                {"session_id": session_id, "chunk_index": j}
                for j in range(len(chunks))
            ]

            vector_db_service.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            session_registry.register_session(SessionMetadata(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                message_count=1
            ))

        # Find duplicates
        pairs = duplicate_service.find_all_duplicate_pairs()

        # Should find no duplicates (or very few if any borderline cases)
        # With high threshold (0.85), unique topics shouldn't match
        assert len(pairs) == 0

    def test_threshold_affects_results(
        self,
        vector_db_service,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test that changing threshold affects duplicate detection."""
        # Create somewhat similar sessions (moderate similarity)
        content1 = "How to use Python list comprehensions for filtering"
        content2 = "Python list comprehension examples and syntax"

        for session_id, content in [("s1", content1), ("s2", content2)]:
            chunks = chunking_service.chunk_text(content)
            embeddings = embedding_service.embed_texts(chunks)
            metadata = [
                {"session_id": session_id, "chunk_index": i}
                for i in range(len(chunks))
            ]

            vector_db_service.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            session_registry.register_session(SessionMetadata(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                message_count=1
            ))

        # Test with high threshold (0.95) - should find fewer/no duplicates
        strict_service = DuplicateDetectionService(
            vector_db_service=vector_db_service,
            session_registry=session_registry,
            similarity_threshold=0.95,
            min_chunks_for_comparison=2
        )
        strict_pairs = strict_service.find_all_duplicate_pairs()

        # Test with lower threshold (0.70) - should find more duplicates
        lenient_service = DuplicateDetectionService(
            vector_db_service=vector_db_service,
            session_registry=session_registry,
            similarity_threshold=0.70,
            min_chunks_for_comparison=2
        )
        lenient_pairs = lenient_service.find_all_duplicate_pairs()

        # Lenient should find at least as many as strict
        assert len(lenient_pairs) >= len(strict_pairs)
