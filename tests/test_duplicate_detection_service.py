"""Unit tests for DuplicateDetectionService."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.smart_fork.duplicate_detection_service import (
    DuplicateDetectionService,
    SimilarSession
)
from src.smart_fork.vector_db_service import ChunkSearchResult
from src.smart_fork.session_registry import SessionMetadata


@pytest.fixture
def mock_vector_db():
    """Create a mock VectorDBService."""
    db = Mock()
    db.collection = Mock()
    return db


@pytest.fixture
def mock_session_registry():
    """Create a mock SessionRegistry."""
    registry = Mock()
    return registry


@pytest.fixture
def duplicate_service(mock_vector_db, mock_session_registry):
    """Create a DuplicateDetectionService with mocked dependencies."""
    return DuplicateDetectionService(
        vector_db_service=mock_vector_db,
        session_registry=mock_session_registry,
        similarity_threshold=0.85,
        min_chunks_for_comparison=3
    )


def create_mock_chunk(chunk_id: str, session_id: str, content: str, chunk_index: int = 0):
    """Helper to create a mock chunk."""
    return ChunkSearchResult(
        chunk_id=chunk_id,
        session_id=session_id,
        content=content,
        metadata={'session_id': session_id, 'chunk_index': chunk_index},
        similarity=1.0,
        chunk_index=chunk_index
    )


class TestDuplicateDetectionService:
    """Test suite for DuplicateDetectionService."""

    def test_initialization(self, duplicate_service):
        """Test service initialization."""
        assert duplicate_service.similarity_threshold == 0.85
        assert duplicate_service.min_chunks_for_comparison == 3

    def test_compute_session_embedding_success(self, duplicate_service, mock_vector_db):
        """Test computing session embedding with valid chunks."""
        # Mock chunks
        chunks = [
            create_mock_chunk(f"test_chunk_{i}", "session1", f"content {i}", i)
            for i in range(5)
        ]
        mock_vector_db.get_session_chunks.return_value = chunks

        # Mock embeddings from ChromaDB
        mock_embeddings = [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
            [0.5, 0.6, 0.7]
        ]
        mock_vector_db.collection.get.return_value = {
            "embeddings": mock_embeddings
        }

        # Compute embedding
        embedding = duplicate_service.compute_session_embedding("session1")

        # Verify result
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 3

        # Verify it's normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.001

    def test_compute_session_embedding_too_few_chunks(self, duplicate_service, mock_vector_db):
        """Test that sessions with too few chunks return None."""
        # Mock only 2 chunks (below min_chunks_for_comparison=3)
        chunks = [
            create_mock_chunk(f"test_chunk_{i}", "session1", f"content {i}", i)
            for i in range(2)
        ]
        mock_vector_db.get_session_chunks.return_value = chunks

        # Compute embedding
        embedding = duplicate_service.compute_session_embedding("session1")

        # Should return None
        assert embedding is None

    def test_compute_session_embedding_no_chunks(self, duplicate_service, mock_vector_db):
        """Test that sessions with no chunks return None."""
        mock_vector_db.get_session_chunks.return_value = []

        embedding = duplicate_service.compute_session_embedding("session1")

        assert embedding is None

    def test_compute_similarity(self, duplicate_service):
        """Test computing similarity between embeddings."""
        # Create normalized embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])  # Identical
        emb3 = np.array([0.0, 1.0, 0.0])  # Orthogonal

        # Test identical embeddings
        sim_identical = duplicate_service.compute_similarity(emb1, emb2)
        assert abs(sim_identical - 1.0) < 0.001

        # Test orthogonal embeddings
        sim_orthogonal = duplicate_service.compute_similarity(emb1, emb3)
        assert abs(sim_orthogonal - 0.0) < 0.001

    def test_get_similar_sessions_success(self, duplicate_service, mock_vector_db, mock_session_registry):
        """Test getting similar sessions."""
        # Mock embeddings for query session
        query_chunks = [
            create_mock_chunk(f"query_chunk_{i}", "query_session", f"content {i}", i)
            for i in range(5)
        ]
        mock_vector_db.get_session_chunks.side_effect = lambda sid: {
            "query_session": query_chunks,
            "similar1": [create_mock_chunk(f"s1_chunk_{i}", "similar1", f"content {i}", i) for i in range(5)],
            "similar2": [create_mock_chunk(f"s2_chunk_{i}", "similar2", f"content {i}", i) for i in range(5)],
            "different": [create_mock_chunk(f"diff_chunk_{i}", "different", f"other {i}", i) for i in range(5)]
        }.get(sid, [])

        # Mock embeddings from ChromaDB - normalized vectors
        import numpy as np
        query_emb = np.array([0.7, 0.7, 0.0])
        query_emb = (query_emb / np.linalg.norm(query_emb)).tolist()

        similar_emb1 = np.array([0.71, 0.71, 0.0])
        similar_emb1 = (similar_emb1 / np.linalg.norm(similar_emb1)).tolist()  # Very similar

        similar_emb2 = np.array([0.68, 0.72, 0.1])
        similar_emb2 = (similar_emb2 / np.linalg.norm(similar_emb2)).tolist()  # Similar

        different_emb = np.array([0.0, 0.0, 1.0])
        different_emb = (different_emb / np.linalg.norm(different_emb)).tolist()  # Very different

        def get_embeddings_side_effect(ids, include):
            if not ids:
                return {"embeddings": []}
            # Extract session prefix from chunk_id
            first_id = ids[0]
            # Chunk IDs look like "query_session_chunk_0", "similar1_chunk_0", etc.
            if '_chunk_' in first_id:
                session_prefix = first_id.split('_chunk_')[0]
                # Handle underscores in session name (e.g., query_session -> query)
                if session_prefix == "query_session":
                    session_prefix = "query"
            else:
                # Fallback - just use first part
                session_prefix = first_id.split('_')[0]

            emb_map = {
                "query": query_emb,
                "similar1": similar_emb1,
                "similar2": similar_emb2,
                "different": different_emb
            }
            # Return the same embedding for all chunks of a session
            emb = emb_map.get(session_prefix, query_emb)
            return {"embeddings": [emb] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        # Mock registry
        mock_session_registry.list_sessions.return_value = ["query_session", "similar1", "similar2", "different"]

        def get_session_side_effect(sid):
            return SessionMetadata(
                session_id=sid,
                created_at=datetime.now().isoformat(),
                message_count=10
            )

        mock_session_registry.get_session.side_effect = get_session_side_effect

        # Get similar sessions
        similar_sessions = duplicate_service.get_similar_sessions("query_session", top_k=3)

        # Should find sessions (at least 2 above threshold)
        assert len(similar_sessions) >= 2
        assert all(isinstance(s, SimilarSession) for s in similar_sessions)
        # Results should be sorted by similarity (highest first)
        for i in range(len(similar_sessions) - 1):
            assert similar_sessions[i].similarity >= similar_sessions[i+1].similarity

    def test_get_similar_sessions_no_query_embedding(self, duplicate_service, mock_vector_db):
        """Test getting similar sessions when query session has no embedding."""
        mock_vector_db.get_session_chunks.return_value = []  # No chunks

        similar_sessions = duplicate_service.get_similar_sessions("session1")

        assert similar_sessions == []

    def test_find_all_duplicate_pairs(self, duplicate_service, mock_vector_db, mock_session_registry):
        """Test finding all duplicate pairs."""
        # Mock 3 sessions: 2 similar, 1 different
        sessions = ["session1", "session2", "session3"]
        mock_session_registry.list_sessions.return_value = sessions

        # Mock chunks for each session
        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        # Mock embeddings
        emb1 = [0.7, 0.7, 0.0]
        emb2 = [0.71, 0.71, 0.0]  # Very similar to emb1
        emb3 = [0.0, 0.0, 1.0]    # Different

        def get_embeddings_side_effect(ids, include):
            if not ids:
                return {"embeddings": []}
            session_id = ids[0].split('_chunk_')[0]
            emb_map = {"session1": emb1, "session2": emb2, "session3": emb3}
            return {"embeddings": [emb_map[session_id]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        # Find duplicate pairs
        pairs = duplicate_service.find_all_duplicate_pairs()

        # Should find 1 pair (session1, session2)
        assert len(pairs) == 1
        assert pairs[0][0] in ["session1", "session2"]
        assert pairs[0][1] in ["session1", "session2"]
        assert pairs[0][2] > 0.85  # Above threshold

    def test_flag_duplicates_in_results(self, duplicate_service, mock_vector_db):
        """Test flagging duplicates in search results."""
        # Create mock search results
        result1 = Mock()
        result1.session_id = "session1"
        result1.score = Mock()
        result1.score.to_dict.return_value = {"total": 0.9}

        result2 = Mock()
        result2.session_id = "session2"
        result2.score = Mock()
        result2.score.to_dict.return_value = {"total": 0.8}

        results = [result1, result2]

        # Mock embeddings
        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        emb1 = [0.7, 0.7, 0.0]
        emb2 = [0.71, 0.71, 0.0]  # Very similar

        def get_embeddings_side_effect(ids, include):
            if not ids:
                return {"embeddings": []}
            session_id = ids[0].split('_chunk_')[0]
            emb_map = {"session1": emb1, "session2": emb2}
            return {"embeddings": [emb_map[session_id]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        # Flag duplicates
        flagged = duplicate_service.flag_duplicates_in_results(results)

        # Verify structure
        assert len(flagged) == 2
        assert all('similar_to' in r for r in flagged)

        # Both should flag each other as similar
        assert len(flagged[0]['similar_to']) == 1
        assert len(flagged[1]['similar_to']) == 1
        assert flagged[0]['similar_to'][0]['session_id'] == "session2"
        assert flagged[1]['similar_to'][0]['session_id'] == "session1"

    def test_flag_duplicates_single_result(self, duplicate_service):
        """Test flagging duplicates with only one result."""
        result = Mock()
        result.session_id = "session1"
        result.score = Mock()
        result.score.to_dict.return_value = {"total": 0.9}

        flagged = duplicate_service.flag_duplicates_in_results([result])

        assert len(flagged) == 1
        assert flagged[0]['similar_to'] == []

    def test_get_stats(self, duplicate_service, mock_vector_db, mock_session_registry):
        """Test getting service statistics."""
        # Mock sessions
        mock_session_registry.list_sessions.return_value = ["session1", "session2", "session3"]

        # Mock chunks - only 2 sessions have enough chunks
        def get_chunks_side_effect(sid):
            if sid in ["session1", "session2"]:
                return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]
            else:
                return [create_mock_chunk(f"{sid}_chunk_0", sid, "content 0", 0)]  # Only 1 chunk

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        # Get stats
        stats = duplicate_service.get_stats()

        assert stats['total_sessions'] == 3
        assert stats['sessions_eligible_for_comparison'] == 2
        assert stats['similarity_threshold'] == 0.85
        assert stats['min_chunks_required'] == 3

    def test_similar_session_to_dict(self):
        """Test SimilarSession to_dict method."""
        similar = SimilarSession(
            session_id="test_session",
            similarity=0.92,
            metadata={"created_at": "2026-01-20", "num_messages": 15}
        )

        result = similar.to_dict()

        assert result['session_id'] == "test_session"
        assert result['similarity'] == 0.92
        assert result['metadata']['num_messages'] == 15


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_database(self, duplicate_service, mock_vector_db, mock_session_registry):
        """Test behavior with empty database."""
        mock_session_registry.list_sessions.return_value = []

        pairs = duplicate_service.find_all_duplicate_pairs()

        assert pairs == []

    def test_all_sessions_below_threshold(self, duplicate_service, mock_vector_db, mock_session_registry):
        """Test when no sessions are similar enough."""
        sessions = ["session1", "session2"]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        # Very different embeddings
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]

        def get_embeddings_side_effect(ids, include):
            if not ids:
                return {"embeddings": []}
            session_id = ids[0].split('_chunk_')[0]
            emb_map = {"session1": emb1, "session2": emb2}
            return {"embeddings": [emb_map[session_id]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        similar = duplicate_service.get_similar_sessions("session1")

        assert similar == []

    def test_embedding_computation_error(self, duplicate_service, mock_vector_db):
        """Test handling of embedding computation errors."""
        mock_vector_db.get_session_chunks.return_value = [
            create_mock_chunk(f"chunk_{i}", "session1", f"content {i}", i)
            for i in range(5)
        ]
        mock_vector_db.collection.get.side_effect = Exception("ChromaDB error")

        embedding = duplicate_service.compute_session_embedding("session1")

        assert embedding is None
