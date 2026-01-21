"""Integration tests for SessionClusteringService with real dependencies."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.smart_fork.session_clustering_service import SessionClusteringService
from src.smart_fork.vector_db_service import VectorDBService
from src.smart_fork.session_registry import SessionRegistry, SessionMetadata
from src.smart_fork.embedding_service import EmbeddingService
from src.smart_fork.chunking_service import ChunkingService


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def vector_db(temp_dir):
    """Create a real VectorDBService for testing."""
    db_path = temp_dir / "vector_db"
    return VectorDBService(persist_directory=str(db_path))


@pytest.fixture
def session_registry(temp_dir):
    """Create a real SessionRegistry for testing."""
    registry_path = temp_dir / "registry.json"
    return SessionRegistry(registry_path=str(registry_path))


@pytest.fixture
def embedding_service(temp_dir):
    """Create a real EmbeddingService for testing."""
    cache_dir = temp_dir / "embedding_cache"
    return EmbeddingService(cache_dir=str(cache_dir))


@pytest.fixture
def chunking_service():
    """Create a real ChunkingService for testing."""
    return ChunkingService()


@pytest.fixture
def clustering_service(vector_db, session_registry, temp_dir):
    """Create a SessionClusteringService with real dependencies."""
    clusters_path = temp_dir / "clusters.json"
    return SessionClusteringService(
        vector_db_service=vector_db,
        session_registry=session_registry,
        storage_path=clusters_path,
        min_chunks_for_clustering=2,  # Lower for testing
        default_num_clusters=3,
        random_state=42
    )


def add_test_session(
    session_id: str,
    content: str,
    vector_db: VectorDBService,
    session_registry: SessionRegistry,
    embedding_service: EmbeddingService,
    chunking_service: ChunkingService,
    tags: list = None,
    project: str = None
):
    """Helper to add a test session with embeddings."""
    # Create chunks
    chunks = chunking_service.chunk_text(content)

    # Generate embeddings
    embeddings = embedding_service.embed_texts(chunks)

    # Create chunk IDs and metadata
    chunk_ids = [f"{session_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            'session_id': session_id,
            'chunk_index': i
        }
        for i in range(len(chunks))
    ]

    # Add to vector DB
    vector_db.add_chunks(
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        contents=chunks,
        metadatas=metadatas
    )

    # Register session
    metadata = SessionMetadata(
        session_id=session_id,
        created_at=datetime.now().isoformat(),
        message_count=1,
        chunk_count=len(chunks),
        tags=tags,
        project=project
    )
    session_registry.add_session(session_id, metadata)


class TestClusteringIntegration:
    """Integration tests for clustering with real services."""

    def test_cluster_related_sessions(
        self,
        clustering_service,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test clustering sessions with similar content into groups."""
        # Add sessions about Python
        add_test_session(
            "python1",
            "How to use Python list comprehensions and lambda functions for data processing",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service,
            tags=["python", "tutorial"]
        )
        add_test_session(
            "python2",
            "Python decorators and context managers for clean code",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service,
            tags=["python", "advanced"]
        )

        # Add sessions about JavaScript
        add_test_session(
            "js1",
            "JavaScript async/await patterns and promise handling in modern web development",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service,
            tags=["javascript", "async"]
        )
        add_test_session(
            "js2",
            "React hooks and component lifecycle in functional components",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service,
            tags=["javascript", "react"]
        )

        # Add sessions about databases
        add_test_session(
            "db1",
            "SQL query optimization and indexing strategies for PostgreSQL",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service,
            tags=["database", "sql"]
        )
        add_test_session(
            "db2",
            "MongoDB aggregation pipeline and schema design patterns",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service,
            tags=["database", "nosql"]
        )

        # Cluster into 3 groups
        result = clustering_service.cluster_sessions(num_clusters=3)

        # Verify clustering
        assert result.num_clusters == 3
        assert result.total_sessions == 6
        assert len(result.clusters) == 3

        # All sessions should be clustered
        all_clustered = []
        for cluster in result.clusters:
            all_clustered.extend(cluster.session_ids)
        assert len(all_clustered) == 6
        assert set(all_clustered) == {"python1", "python2", "js1", "js2", "db1", "db2"}

        # Verify clusters have labels
        for cluster in result.clusters:
            assert cluster.label is not None
            assert cluster.size > 0
            assert len(cluster.session_ids) == cluster.size

        # Verify silhouette score is reasonable
        if result.overall_silhouette_score is not None:
            assert -1.0 <= result.overall_silhouette_score <= 1.0

    def test_get_cluster_for_session(
        self,
        clustering_service,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test retrieving the cluster a session belongs to."""
        # Add test sessions
        add_test_session(
            "session1",
            "Python programming tutorial for beginners",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )
        add_test_session(
            "session2",
            "JavaScript web development fundamentals",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )

        # Cluster
        clustering_service.cluster_sessions(num_clusters=2)

        # Get cluster for session
        cluster = clustering_service.get_cluster_for_session("session1")

        assert cluster is not None
        assert "session1" in cluster.session_ids
        assert cluster.cluster_id in [0, 1]

    def test_persistence_across_instances(
        self,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service,
        temp_dir
    ):
        """Test that clustering results persist across service instances."""
        clusters_path = temp_dir / "clusters.json"

        # Create first service and cluster
        service1 = SessionClusteringService(
            vector_db_service=vector_db,
            session_registry=session_registry,
            storage_path=clusters_path,
            min_chunks_for_clustering=2
        )

        # Add sessions
        add_test_session(
            "session1",
            "Test content about Python programming",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )
        add_test_session(
            "session2",
            "Test content about JavaScript development",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )

        result1 = service1.cluster_sessions(num_clusters=2)

        # Create second service - should load existing clusters
        service2 = SessionClusteringService(
            vector_db_service=vector_db,
            session_registry=session_registry,
            storage_path=clusters_path,
            min_chunks_for_clustering=2
        )

        result2 = service2.get_all_clusters()

        # Results should match
        assert result2 is not None
        assert result2.num_clusters == result1.num_clusters
        assert result2.total_sessions == result1.total_sessions

    def test_cluster_label_generation(
        self,
        clustering_service,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test that cluster labels are generated from session metadata."""
        # Add sessions with tags
        add_test_session(
            "tagged1",
            "Python tutorial content",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service,
            tags=["python", "tutorial"]
        )
        add_test_session(
            "tagged2",
            "Python advanced content",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service,
            tags=["python", "advanced"]
        )

        result = clustering_service.cluster_sessions(num_clusters=1)

        # Label should contain common tag
        cluster = result.clusters[0]
        assert "python" in cluster.label.lower()

    def test_stats_after_clustering(
        self,
        clustering_service,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test statistics after clustering."""
        # Add sessions
        for i in range(5):
            add_test_session(
                f"session{i}",
                f"Test content about topic {i % 2}",
                vector_db,
                session_registry,
                embedding_service,
                chunking_service
            )

        # Get stats before clustering
        stats_before = clustering_service.get_stats()
        assert stats_before['has_clustering'] is False

        # Cluster
        clustering_service.cluster_sessions(num_clusters=2)

        # Get stats after clustering
        stats_after = clustering_service.get_stats()
        assert stats_after['has_clustering'] is True
        assert stats_after['num_clusters'] == 2
        assert 'avg_cluster_size' in stats_after
        assert stats_after['clustered_sessions'] == 5

    def test_reclustering_updates_results(
        self,
        clustering_service,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test that reclustering updates the stored results."""
        # Add initial sessions
        add_test_session(
            "session1",
            "Python content",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )
        add_test_session(
            "session2",
            "JavaScript content",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )

        # First clustering
        result1 = clustering_service.cluster_sessions(num_clusters=1)
        assert result1.num_clusters == 1

        # Add more sessions
        add_test_session(
            "session3",
            "Python advanced content",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )
        add_test_session(
            "session4",
            "JavaScript advanced content",
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )

        # Recluster with more clusters
        result2 = clustering_service.cluster_sessions(num_clusters=2)
        assert result2.num_clusters == 2
        assert result2.total_sessions == 4

        # Verify updated results are accessible
        current = clustering_service.get_all_clusters()
        assert current.num_clusters == 2
        assert current.total_sessions == 4

    def test_get_cluster_by_id(
        self,
        clustering_service,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test retrieving specific cluster by ID."""
        # Add sessions
        for i in range(4):
            add_test_session(
                f"session{i}",
                f"Content about topic {i % 2}",
                vector_db,
                session_registry,
                embedding_service,
                chunking_service
            )

        # Cluster
        result = clustering_service.cluster_sessions(num_clusters=2)

        # Get each cluster by ID
        for cluster_id in range(2):
            cluster = clustering_service.get_cluster_by_id(cluster_id)
            assert cluster is not None
            assert cluster.cluster_id == cluster_id
            assert len(cluster.session_ids) > 0

        # Try invalid cluster ID
        invalid_cluster = clustering_service.get_cluster_by_id(999)
        assert invalid_cluster is None

    def test_empty_database_clustering(self, clustering_service):
        """Test clustering with no sessions."""
        result = clustering_service.cluster_sessions(num_clusters=5)

        assert result.num_clusters == 0
        assert result.total_sessions == 0
        assert len(result.clusters) == 0

    def test_sessions_below_chunk_threshold(
        self,
        clustering_service,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test that sessions with too few chunks are excluded."""
        # Add a session with very short content (will create < 2 chunks)
        add_test_session(
            "short_session",
            "Hi",  # Very short
            vector_db,
            session_registry,
            embedding_service,
            chunking_service
        )

        # Try to cluster
        result = clustering_service.cluster_sessions(num_clusters=1)

        # Should have no eligible sessions
        assert result.num_clusters == 0

    def test_cluster_quality_metrics(
        self,
        clustering_service,
        vector_db,
        session_registry,
        embedding_service,
        chunking_service
    ):
        """Test that clustering quality metrics are computed."""
        # Add distinct sessions for good clustering
        topics = [
            "Python list comprehensions and generators",
            "Python decorators and metaclasses",
            "JavaScript promises and async await",
            "React hooks and component state",
            "SQL joins and subqueries",
            "NoSQL document design patterns"
        ]

        for i, topic in enumerate(topics):
            add_test_session(
                f"session{i}",
                topic,
                vector_db,
                session_registry,
                embedding_service,
                chunking_service
            )

        result = clustering_service.cluster_sessions(num_clusters=3)

        # Should have quality metrics
        assert result.overall_silhouette_score is not None
        # Silhouette score should be in valid range
        assert -1.0 <= result.overall_silhouette_score <= 1.0
