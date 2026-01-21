"""
Session Clustering Service for automatic topic-based organization.

This module uses k-means clustering on session-level embeddings to group
related sessions together, enabling automatic topic discovery and organization.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import threading

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .vector_db_service import VectorDBService
from .session_registry import SessionRegistry

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Information about a session cluster."""
    cluster_id: int
    session_ids: List[str]
    size: int
    centroid: Optional[List[float]] = None  # Serializable version of numpy array
    label: Optional[str] = None
    silhouette_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'session_ids': self.session_ids,
            'size': self.size,
            'centroid': self.centroid,
            'label': self.label,
            'silhouette_score': self.silhouette_score
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ClusterInfo':
        """Create ClusterInfo from dictionary."""
        return ClusterInfo(
            cluster_id=data['cluster_id'],
            session_ids=data['session_ids'],
            size=data['size'],
            centroid=data.get('centroid'),
            label=data.get('label'),
            silhouette_score=data.get('silhouette_score')
        )


@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    clusters: List[ClusterInfo]
    total_sessions: int
    num_clusters: int
    overall_silhouette_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'clusters': [c.to_dict() for c in self.clusters],
            'total_sessions': self.total_sessions,
            'num_clusters': self.num_clusters,
            'overall_silhouette_score': self.overall_silhouette_score
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ClusteringResult':
        """Create ClusteringResult from dictionary."""
        return ClusteringResult(
            clusters=[ClusterInfo.from_dict(c) for c in data['clusters']],
            total_sessions=data['total_sessions'],
            num_clusters=data['num_clusters'],
            overall_silhouette_score=data.get('overall_silhouette_score')
        )


class SessionClusteringService:
    """
    Service for automatic clustering of sessions by topic.

    Features:
    - k-means clustering on session-level embeddings
    - Automatic cluster label generation from session content
    - Cluster quality metrics (silhouette score)
    - Persistent storage of clustering results
    - Update clusters when new sessions are indexed
    """

    def __init__(
        self,
        vector_db_service: VectorDBService,
        session_registry: SessionRegistry,
        storage_path: Optional[Path] = None,
        min_chunks_for_clustering: int = 3,
        default_num_clusters: int = 10,
        random_state: int = 42
    ):
        """
        Initialize the SessionClusteringService.

        Args:
            vector_db_service: VectorDBService for accessing chunk embeddings
            session_registry: SessionRegistry for accessing session metadata
            storage_path: Path to store clustering results (default: ~/.smart-fork/clusters.json)
            min_chunks_for_clustering: Minimum chunks required for a session to be clustered (default: 3)
            default_num_clusters: Default number of clusters to create (default: 10)
            random_state: Random state for reproducible clustering (default: 42)
        """
        self.vector_db_service = vector_db_service
        self.session_registry = session_registry
        self.min_chunks_for_clustering = min_chunks_for_clustering
        self.default_num_clusters = default_num_clusters
        self.random_state = random_state

        # Set storage path
        if storage_path is None:
            storage_path = Path.home() / ".smart-fork" / "clusters.json"
        self.storage_path = storage_path

        # Thread safety
        self._lock = threading.Lock()

        # Current clustering result (cached)
        self._current_clustering: Optional[ClusteringResult] = None

        # Load existing clusters if available
        self._load()

        logger.info(
            f"Initialized SessionClusteringService "
            f"(default_clusters={default_num_clusters}, min_chunks={min_chunks_for_clustering})"
        )

    def _load(self) -> None:
        """Load clustering results from disk."""
        if not self.storage_path.exists():
            logger.debug(f"No existing clusters file at {self.storage_path}")
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self._current_clustering = ClusteringResult.from_dict(data)
            logger.info(f"Loaded clustering with {self._current_clustering.num_clusters} clusters")

        except Exception as e:
            logger.error(f"Error loading clusters from {self.storage_path}: {e}")
            self._current_clustering = None

    def _save(self) -> None:
        """Save clustering results to disk."""
        if self._current_clustering is None:
            logger.debug("No clustering to save")
            return

        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first, then rename (atomic operation)
            temp_path = self.storage_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self._current_clustering.to_dict(), f, indent=2)

            temp_path.replace(self.storage_path)
            logger.info(f"Saved clustering to {self.storage_path}")

        except Exception as e:
            logger.error(f"Error saving clusters to {self.storage_path}: {e}")

    def compute_session_embedding(self, session_id: str) -> Optional[np.ndarray]:
        """
        Compute session-level embedding by averaging chunk embeddings.

        This reuses the same approach as DuplicateDetectionService.

        Args:
            session_id: Session ID to compute embedding for

        Returns:
            Session embedding as numpy array, or None if session has too few chunks
        """
        # Get all chunks for this session
        chunks = self.vector_db_service.get_session_chunks(session_id)

        if not chunks or len(chunks) < self.min_chunks_for_clustering:
            logger.debug(
                f"Session {session_id} has too few chunks "
                f"({len(chunks) if chunks else 0}) for clustering"
            )
            return None

        # Get the embeddings from ChromaDB
        try:
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            results = self.vector_db_service.collection.get(
                ids=chunk_ids,
                include=["embeddings"]
            )

            if not results["embeddings"]:
                logger.warning(f"No embeddings found for session {session_id}")
                return None

            embeddings = results["embeddings"]

            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)

            # Normalize the result
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            return avg_embedding

        except Exception as e:
            logger.error(f"Error computing embedding for session {session_id}: {e}")
            return None

    def _generate_cluster_label(
        self,
        session_ids: List[str],
        max_words: int = 3
    ) -> str:
        """
        Generate a descriptive label for a cluster based on session content.

        This is a simple implementation that uses common words from session projects.
        A more sophisticated version could use TF-IDF or extract key terms from content.

        Args:
            session_ids: List of session IDs in the cluster
            max_words: Maximum number of words in the label (default: 3)

        Returns:
            Generated label string
        """
        # Collect metadata
        projects = []
        tags = []

        for session_id in session_ids:
            metadata = self.session_registry.get_session(session_id)
            if metadata:
                if metadata.project:
                    projects.append(metadata.project)
                if metadata.tags:
                    tags.extend(metadata.tags)

        # Find most common project or tag
        if tags:
            # Prefer tags if available
            tag_counts = Counter(tags)
            most_common = tag_counts.most_common(max_words)
            return " / ".join([tag for tag, _ in most_common])

        if projects:
            # Use most common project
            project_counts = Counter(projects)
            most_common = project_counts.most_common(1)[0][0]
            # Extract last part of path if it's a file path
            project_name = Path(most_common).name if most_common else most_common
            return project_name

        # Fallback to generic label
        return f"Topic {len(session_ids)} sessions"

    def cluster_sessions(
        self,
        num_clusters: Optional[int] = None,
        progress_callback: Optional[Any] = None
    ) -> ClusteringResult:
        """
        Cluster all eligible sessions using k-means.

        Args:
            num_clusters: Number of clusters to create (default: self.default_num_clusters)
            progress_callback: Optional callback function(current, total, message) for progress updates

        Returns:
            ClusteringResult with cluster information
        """
        with self._lock:
            if num_clusters is None:
                num_clusters = self.default_num_clusters

            # Get all sessions
            all_sessions = self.session_registry.list_sessions()
            total_sessions = len(all_sessions)

            logger.info(f"Clustering {total_sessions} sessions into {num_clusters} clusters")

            if progress_callback:
                progress_callback(0, total_sessions, "Computing session embeddings")

            # Compute embeddings for all eligible sessions
            session_embeddings: Dict[str, np.ndarray] = {}

            for i, session_id in enumerate(all_sessions):
                if progress_callback and i % 10 == 0:
                    progress_callback(i, total_sessions, "Computing session embeddings")

                embedding = self.compute_session_embedding(session_id)
                if embedding is not None:
                    session_embeddings[session_id] = embedding

            eligible_sessions = len(session_embeddings)
            logger.info(f"Computed embeddings for {eligible_sessions} eligible sessions")

            if eligible_sessions < num_clusters:
                logger.warning(
                    f"Only {eligible_sessions} eligible sessions, "
                    f"reducing num_clusters from {num_clusters} to {max(1, eligible_sessions)}"
                )
                num_clusters = max(1, eligible_sessions)

            if eligible_sessions == 0:
                logger.warning("No sessions eligible for clustering")
                return ClusteringResult(
                    clusters=[],
                    total_sessions=total_sessions,
                    num_clusters=0,
                    overall_silhouette_score=None
                )

            # Prepare data for clustering
            session_ids_list = list(session_embeddings.keys())
            embeddings_matrix = np.array([session_embeddings[sid] for sid in session_ids_list])

            if progress_callback:
                progress_callback(
                    eligible_sessions,
                    total_sessions,
                    f"Running k-means clustering (k={num_clusters})"
                )

            # Run k-means clustering
            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )

            cluster_labels = kmeans.fit_predict(embeddings_matrix)
            centroids = kmeans.cluster_centers_

            # Compute silhouette score (quality metric)
            # Silhouette score requires at least 2 distinct clusters and more samples than clusters
            num_distinct_clusters = len(np.unique(cluster_labels))
            if num_distinct_clusters >= 2 and eligible_sessions > num_distinct_clusters:
                overall_silhouette = silhouette_score(embeddings_matrix, cluster_labels)
            else:
                overall_silhouette = None

            logger.info(f"Clustering complete. Silhouette score: {overall_silhouette}")

            # Build cluster information
            clusters = []

            for cluster_id in range(num_clusters):
                # Get sessions in this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_session_ids = [
                    session_ids_list[i]
                    for i in range(len(session_ids_list))
                    if cluster_mask[i]
                ]

                if not cluster_session_ids:
                    continue

                # Generate label for this cluster
                label = self._generate_cluster_label(cluster_session_ids)

                # Compute per-cluster silhouette score
                # Note: Per-cluster silhouette is same as overall (using all data)
                # Set to None for now, or use overall_silhouette if available
                cluster_silhouette = overall_silhouette

                # Create cluster info
                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    session_ids=cluster_session_ids,
                    size=len(cluster_session_ids),
                    centroid=centroids[cluster_id].tolist(),  # Convert numpy to list
                    label=label,
                    silhouette_score=cluster_silhouette
                )

                clusters.append(cluster_info)

            # Create result
            result = ClusteringResult(
                clusters=clusters,
                total_sessions=total_sessions,
                num_clusters=len(clusters),
                overall_silhouette_score=overall_silhouette
            )

            # Cache and save
            self._current_clustering = result
            self._save()

            if progress_callback:
                progress_callback(
                    total_sessions,
                    total_sessions,
                    f"Clustering complete ({len(clusters)} clusters)"
                )

            return result

    def get_cluster_for_session(self, session_id: str) -> Optional[ClusterInfo]:
        """
        Get the cluster that a session belongs to.

        Args:
            session_id: Session ID to look up

        Returns:
            ClusterInfo if session is in a cluster, None otherwise
        """
        with self._lock:
            if self._current_clustering is None:
                logger.debug("No clustering available")
                return None

            for cluster in self._current_clustering.clusters:
                if session_id in cluster.session_ids:
                    return cluster

            return None

    def get_all_clusters(self) -> Optional[ClusteringResult]:
        """
        Get all current clusters.

        Returns:
            Current ClusteringResult, or None if no clustering has been performed
        """
        with self._lock:
            return self._current_clustering

    def get_cluster_by_id(self, cluster_id: int) -> Optional[ClusterInfo]:
        """
        Get a specific cluster by ID.

        Args:
            cluster_id: Cluster ID to retrieve

        Returns:
            ClusterInfo if found, None otherwise
        """
        with self._lock:
            if self._current_clustering is None:
                return None

            for cluster in self._current_clustering.clusters:
                if cluster.cluster_id == cluster_id:
                    return cluster

            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about clustering.

        Returns:
            Dictionary with clustering statistics
        """
        with self._lock:
            all_sessions = self.session_registry.list_sessions()
            total_sessions = len(all_sessions)

            # Count eligible sessions
            eligible_sessions = 0
            for session_id in all_sessions:
                chunks = self.vector_db_service.get_session_chunks(session_id)
                if chunks and len(chunks) >= self.min_chunks_for_clustering:
                    eligible_sessions += 1

            stats = {
                'total_sessions': total_sessions,
                'eligible_sessions': eligible_sessions,
                'min_chunks_required': self.min_chunks_for_clustering,
                'has_clustering': self._current_clustering is not None
            }

            if self._current_clustering:
                stats['num_clusters'] = self._current_clustering.num_clusters
                stats['clustered_sessions'] = self._current_clustering.total_sessions
                stats['overall_silhouette_score'] = self._current_clustering.overall_silhouette_score

                # Cluster size distribution
                cluster_sizes = [c.size for c in self._current_clustering.clusters]
                if cluster_sizes:
                    stats['avg_cluster_size'] = sum(cluster_sizes) / len(cluster_sizes)
                    stats['min_cluster_size'] = min(cluster_sizes)
                    stats['max_cluster_size'] = max(cluster_sizes)

            return stats
