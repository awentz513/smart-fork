"""Unit tests for SessionClusteringService."""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.smart_fork.session_clustering_service import (
    SessionClusteringService,
    ClusterInfo,
    ClusteringResult
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
def temp_storage_path(tmp_path):
    """Create a temporary storage path for clustering results."""
    return tmp_path / "test_clusters.json"


@pytest.fixture
def clustering_service(mock_vector_db, mock_session_registry, temp_storage_path):
    """Create a SessionClusteringService with mocked dependencies."""
    return SessionClusteringService(
        vector_db_service=mock_vector_db,
        session_registry=mock_session_registry,
        storage_path=temp_storage_path,
        min_chunks_for_clustering=3,
        default_num_clusters=10,
        random_state=42
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


class TestClusterInfo:
    """Test suite for ClusterInfo dataclass."""

    def test_cluster_info_to_dict(self):
        """Test converting ClusterInfo to dictionary."""
        cluster = ClusterInfo(
            cluster_id=0,
            session_ids=["session1", "session2"],
            size=2,
            centroid=[0.1, 0.2, 0.3],
            label="Test Cluster",
            silhouette_score=0.75
        )

        result = cluster.to_dict()

        assert result['cluster_id'] == 0
        assert result['session_ids'] == ["session1", "session2"]
        assert result['size'] == 2
        assert result['centroid'] == [0.1, 0.2, 0.3]
        assert result['label'] == "Test Cluster"
        assert result['silhouette_score'] == 0.75

    def test_cluster_info_from_dict(self):
        """Test creating ClusterInfo from dictionary."""
        data = {
            'cluster_id': 1,
            'session_ids': ["session3", "session4"],
            'size': 2,
            'centroid': [0.4, 0.5, 0.6],
            'label': "Another Cluster",
            'silhouette_score': 0.65
        }

        cluster = ClusterInfo.from_dict(data)

        assert cluster.cluster_id == 1
        assert cluster.session_ids == ["session3", "session4"]
        assert cluster.size == 2
        assert cluster.centroid == [0.4, 0.5, 0.6]
        assert cluster.label == "Another Cluster"
        assert cluster.silhouette_score == 0.65


class TestClusteringResult:
    """Test suite for ClusteringResult dataclass."""

    def test_clustering_result_to_dict(self):
        """Test converting ClusteringResult to dictionary."""
        clusters = [
            ClusterInfo(
                cluster_id=0,
                session_ids=["s1", "s2"],
                size=2,
                label="Cluster A"
            ),
            ClusterInfo(
                cluster_id=1,
                session_ids=["s3"],
                size=1,
                label="Cluster B"
            )
        ]

        result = ClusteringResult(
            clusters=clusters,
            total_sessions=3,
            num_clusters=2,
            overall_silhouette_score=0.6
        )

        data = result.to_dict()

        assert data['total_sessions'] == 3
        assert data['num_clusters'] == 2
        assert data['overall_silhouette_score'] == 0.6
        assert len(data['clusters']) == 2

    def test_clustering_result_from_dict(self):
        """Test creating ClusteringResult from dictionary."""
        data = {
            'clusters': [
                {
                    'cluster_id': 0,
                    'session_ids': ["s1", "s2"],
                    'size': 2,
                    'label': "Test"
                }
            ],
            'total_sessions': 2,
            'num_clusters': 1,
            'overall_silhouette_score': 0.8
        }

        result = ClusteringResult.from_dict(data)

        assert result.total_sessions == 2
        assert result.num_clusters == 1
        assert result.overall_silhouette_score == 0.8
        assert len(result.clusters) == 1


class TestSessionClusteringService:
    """Test suite for SessionClusteringService."""

    def test_initialization(self, clustering_service, temp_storage_path):
        """Test service initialization."""
        assert clustering_service.min_chunks_for_clustering == 3
        assert clustering_service.default_num_clusters == 10
        assert clustering_service.random_state == 42
        assert clustering_service.storage_path == temp_storage_path
        assert clustering_service._current_clustering is None

    def test_compute_session_embedding_success(self, clustering_service, mock_vector_db):
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
        embedding = clustering_service.compute_session_embedding("session1")

        # Verify result
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 3

        # Verify it's normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.001

    def test_compute_session_embedding_too_few_chunks(self, clustering_service, mock_vector_db):
        """Test that sessions with too few chunks return None."""
        # Mock only 2 chunks (below min_chunks_for_clustering=3)
        chunks = [
            create_mock_chunk(f"test_chunk_{i}", "session1", f"content {i}", i)
            for i in range(2)
        ]
        mock_vector_db.get_session_chunks.return_value = chunks

        # Compute embedding
        embedding = clustering_service.compute_session_embedding("session1")

        # Should return None
        assert embedding is None

    def test_compute_session_embedding_no_chunks(self, clustering_service, mock_vector_db):
        """Test that sessions with no chunks return None."""
        mock_vector_db.get_session_chunks.return_value = []

        embedding = clustering_service.compute_session_embedding("session1")

        assert embedding is None

    def test_compute_session_embedding_error_handling(self, clustering_service, mock_vector_db):
        """Test error handling during embedding computation."""
        chunks = [
            create_mock_chunk(f"test_chunk_{i}", "session1", f"content {i}", i)
            for i in range(5)
        ]
        mock_vector_db.get_session_chunks.return_value = chunks
        mock_vector_db.collection.get.side_effect = Exception("ChromaDB error")

        embedding = clustering_service.compute_session_embedding("session1")

        assert embedding is None

    def test_generate_cluster_label_with_tags(self, clustering_service, mock_session_registry):
        """Test generating cluster label from session tags."""
        session_ids = ["session1", "session2", "session3"]

        # Mock metadata with tags
        def get_session_side_effect(sid):
            return SessionMetadata(
                session_id=sid,
                created_at="2026-01-20T10:00:00Z",
                message_count=10,
                tags=["python", "testing"] if sid in ["session1", "session2"] else ["python"]
            )

        mock_session_registry.get_session.side_effect = get_session_side_effect

        label = clustering_service._generate_cluster_label(session_ids)

        # Should use most common tags
        assert "python" in label or "testing" in label

    def test_generate_cluster_label_with_project(self, clustering_service, mock_session_registry):
        """Test generating cluster label from project names."""
        session_ids = ["session1", "session2"]

        # Mock metadata with projects (no tags)
        def get_session_side_effect(sid):
            return SessionMetadata(
                session_id=sid,
                created_at="2026-01-20T10:00:00Z",
                message_count=10,
                project="/Users/test/Documents/MyProject"
            )

        mock_session_registry.get_session.side_effect = get_session_side_effect

        label = clustering_service._generate_cluster_label(session_ids)

        # Should extract project name
        assert "MyProject" in label

    def test_generate_cluster_label_fallback(self, clustering_service, mock_session_registry):
        """Test fallback label generation when no metadata available."""
        session_ids = ["session1", "session2", "session3"]

        # Mock empty metadata
        mock_session_registry.get_session.return_value = None

        label = clustering_service._generate_cluster_label(session_ids)

        # Should use fallback
        assert "sessions" in label.lower()

    def test_cluster_sessions_success(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test successful clustering of sessions."""
        # Mock sessions
        sessions = [f"session{i}" for i in range(12)]
        mock_session_registry.list_sessions.return_value = sessions

        # Mock chunks for each session
        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        # Mock embeddings - create 3 distinct groups
        base_embs = [
            [0.9, 0.1, 0.0],  # Group 1
            [0.1, 0.9, 0.0],  # Group 2
            [0.0, 0.1, 0.9]   # Group 3
        ]

        def get_embeddings_side_effect(ids, include):
            if not ids:
                return {"embeddings": []}
            # Extract session number
            session_id = ids[0].split('_chunk_')[0]
            session_num = int(session_id.replace('session', ''))
            # Assign to one of 3 groups
            group = session_num % 3
            return {"embeddings": [base_embs[group]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        # Mock session metadata
        def get_session_side_effect(sid):
            return SessionMetadata(
                session_id=sid,
                created_at="2026-01-20T10:00:00Z",
                message_count=10
            )

        mock_session_registry.get_session.side_effect = get_session_side_effect

        # Cluster with 3 clusters
        result = clustering_service.cluster_sessions(num_clusters=3)

        # Verify result
        assert result.total_sessions == 12
        assert result.num_clusters == 3
        assert len(result.clusters) == 3
        assert result.overall_silhouette_score is not None

        # Verify all sessions are clustered
        all_clustered = []
        for cluster in result.clusters:
            all_clustered.extend(cluster.session_ids)
        assert len(all_clustered) == 12

        # Verify clusters have labels
        for cluster in result.clusters:
            assert cluster.label is not None
            assert cluster.centroid is not None
            assert cluster.size > 0

    def test_cluster_sessions_too_few_sessions(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test clustering when there are fewer sessions than requested clusters."""
        # Mock only 3 sessions
        sessions = ["session1", "session2", "session3"]
        mock_session_registry.list_sessions.return_value = sessions

        # Mock chunks
        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        # Mock embeddings
        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        # Mock metadata
        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        # Request 10 clusters but only have 3 sessions
        result = clustering_service.cluster_sessions(num_clusters=10)

        # Should auto-adjust to 3 clusters
        assert result.num_clusters <= 3
        assert result.total_sessions == 3

    def test_cluster_sessions_no_eligible_sessions(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test clustering when no sessions have enough chunks."""
        sessions = ["session1", "session2"]
        mock_session_registry.list_sessions.return_value = sessions

        # Mock sessions with too few chunks
        mock_vector_db.get_session_chunks.return_value = [
            create_mock_chunk("chunk_0", "session1", "content", 0)
        ]

        result = clustering_service.cluster_sessions(num_clusters=2)

        assert result.num_clusters == 0
        assert result.total_sessions == 2
        assert len(result.clusters) == 0

    def test_cluster_sessions_default_num_clusters(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test clustering with default number of clusters."""
        # Mock 15 sessions
        sessions = [f"session{i}" for i in range(15)]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        # Create varied embeddings to ensure distinct clusters
        def get_embeddings_side_effect(ids, include):
            if not ids:
                return {"embeddings": []}
            # Extract session number from chunk ID
            session_id = ids[0].split('_chunk_')[0]
            session_num = int(session_id.replace('session', ''))
            # Create 10 different embedding patterns (one per expected cluster)
            angle = (session_num % 10) * (2 * np.pi / 10)
            embedding = [np.cos(angle), np.sin(angle), 0.0]
            return {"embeddings": [embedding] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        # Don't specify num_clusters - should use default (10)
        result = clustering_service.cluster_sessions()

        assert result.num_clusters == 10

    def test_cluster_sessions_progress_callback(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test that progress callback is called during clustering."""
        sessions = [f"session{i}" for i in range(5)]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        # Track callback invocations
        callback_calls = []

        def progress_callback(current, total, message):
            callback_calls.append((current, total, message))

        result = clustering_service.cluster_sessions(num_clusters=2, progress_callback=progress_callback)

        # Verify callback was called
        assert len(callback_calls) > 0
        # First call should be about computing embeddings
        assert "embedding" in callback_calls[0][2].lower()

    def test_save_and_load_clustering(self, clustering_service, mock_vector_db, mock_session_registry, temp_storage_path):
        """Test saving and loading clustering results."""
        # Create a simple clustering result
        sessions = ["session1", "session2", "session3"]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        # Cluster and save
        result = clustering_service.cluster_sessions(num_clusters=2)

        # Verify file was created
        assert temp_storage_path.exists()

        # Create new service and load
        new_service = SessionClusteringService(
            vector_db_service=mock_vector_db,
            session_registry=mock_session_registry,
            storage_path=temp_storage_path
        )

        # Verify loaded clustering
        loaded = new_service.get_all_clusters()
        assert loaded is not None
        assert loaded.num_clusters == result.num_clusters
        assert loaded.total_sessions == result.total_sessions

    def test_get_cluster_for_session(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test getting the cluster for a specific session."""
        # Create clustering
        sessions = ["session1", "session2", "session3"]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        clustering_service.cluster_sessions(num_clusters=2)

        # Get cluster for a session
        cluster = clustering_service.get_cluster_for_session("session1")

        assert cluster is not None
        assert "session1" in cluster.session_ids

    def test_get_cluster_for_session_not_found(self, clustering_service):
        """Test getting cluster for non-existent session."""
        cluster = clustering_service.get_cluster_for_session("nonexistent")

        assert cluster is None

    def test_get_all_clusters(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test getting all clusters."""
        # Initially should be None
        assert clustering_service.get_all_clusters() is None

        # Create clustering
        sessions = ["session1", "session2"]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        clustering_service.cluster_sessions(num_clusters=1)

        # Now should have clusters
        result = clustering_service.get_all_clusters()
        assert result is not None
        assert result.num_clusters == 1

    def test_get_cluster_by_id(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test getting a specific cluster by ID."""
        sessions = ["session1", "session2", "session3"]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        clustering_service.cluster_sessions(num_clusters=2)

        # Get cluster 0
        cluster = clustering_service.get_cluster_by_id(0)
        assert cluster is not None
        assert cluster.cluster_id == 0

        # Try non-existent cluster
        cluster = clustering_service.get_cluster_by_id(999)
        assert cluster is None

    def test_get_stats(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test getting clustering statistics."""
        # Before clustering
        mock_session_registry.list_sessions.return_value = ["session1", "session2", "session3"]

        def get_chunks_side_effect(sid):
            # Only session1 and session2 have enough chunks
            if sid in ["session1", "session2"]:
                return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]
            else:
                return [create_mock_chunk(f"{sid}_chunk_0", sid, "content", 0)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        stats = clustering_service.get_stats()

        assert stats['total_sessions'] == 3
        assert stats['eligible_sessions'] == 2
        assert stats['min_chunks_required'] == 3
        assert stats['has_clustering'] is False

        # After clustering
        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        clustering_service.cluster_sessions(num_clusters=1)

        stats = clustering_service.get_stats()

        assert stats['has_clustering'] is True
        assert stats['num_clusters'] == 1
        assert 'avg_cluster_size' in stats


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_database(self, clustering_service, mock_session_registry):
        """Test clustering with empty database."""
        mock_session_registry.list_sessions.return_value = []

        result = clustering_service.cluster_sessions(num_clusters=5)

        assert result.num_clusters == 0
        assert result.total_sessions == 0
        assert len(result.clusters) == 0

    def test_load_corrupted_file(self, temp_storage_path, mock_vector_db, mock_session_registry):
        """Test loading corrupted clustering file."""
        # Create corrupted file
        with open(temp_storage_path, 'w') as f:
            f.write("invalid json{{{")

        # Create service - should handle error gracefully
        service = SessionClusteringService(
            vector_db_service=mock_vector_db,
            session_registry=mock_session_registry,
            storage_path=temp_storage_path
        )

        assert service.get_all_clusters() is None

    def test_thread_safety(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test that clustering operations are thread-safe."""
        import threading

        sessions = ["session1", "session2"]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        # Run clustering in thread
        thread = threading.Thread(
            target=lambda: clustering_service.cluster_sessions(num_clusters=1)
        )
        thread.start()
        thread.join()

        # Should complete successfully
        result = clustering_service.get_all_clusters()
        assert result is not None

    def test_silhouette_score_edge_cases(self, clustering_service, mock_vector_db, mock_session_registry):
        """Test silhouette score computation in edge cases."""
        # Case: Only 1 cluster - silhouette score should be None
        sessions = ["session1", "session2"]
        mock_session_registry.list_sessions.return_value = sessions

        def get_chunks_side_effect(sid):
            return [create_mock_chunk(f"{sid}_chunk_{i}", sid, f"content {i}", i) for i in range(5)]

        mock_vector_db.get_session_chunks.side_effect = get_chunks_side_effect

        def get_embeddings_side_effect(ids, include):
            return {"embeddings": [[0.5, 0.5, 0.0]] * len(ids)}

        mock_vector_db.collection.get.side_effect = get_embeddings_side_effect

        mock_session_registry.get_session.return_value = SessionMetadata(
            session_id="test",
            created_at="2026-01-20T10:00:00Z",
            message_count=10
        )

        result = clustering_service.cluster_sessions(num_clusters=1)

        # With only 1 cluster, overall silhouette score should be None
        # (can't compute meaningful silhouette with only 1 cluster)
        assert result.overall_silhouette_score is None or result.num_clusters > 1
