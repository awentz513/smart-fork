"""Tests for preference learning integration with scoring and search."""

import pytest
import tempfile
import os
from datetime import datetime

from smart_fork.preference_service import PreferenceService
from smart_fork.scoring_service import ScoringService, SessionScore
from smart_fork.search_service import SearchService
from smart_fork.embedding_service import EmbeddingService
from smart_fork.vector_db_service import VectorDBService
from smart_fork.session_registry import SessionRegistry
from smart_fork.cache_service import CacheService


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    import tempfile
    import shutil

    temp_root = tempfile.mkdtemp()
    preference_file = os.path.join(temp_root, "preferences.json")
    vector_db_dir = os.path.join(temp_root, "vector_db")
    registry_file = os.path.join(temp_root, "registry.json")

    yield {
        'root': temp_root,
        'preference_file': preference_file,
        'vector_db_dir': vector_db_dir,
        'registry_file': registry_file
    }

    # Cleanup
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)


@pytest.fixture
def preference_service(temp_dirs):
    """Create preference service."""
    return PreferenceService(preference_file=temp_dirs['preference_file'])


@pytest.fixture
def scoring_service():
    """Create scoring service."""
    return ScoringService()


def test_scoring_with_preference_boost(scoring_service):
    """Test that preference boost is added to final score."""
    # Calculate score without preference boost
    score_no_pref = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.9, 0.8],
        total_chunks_in_session=10,
        preference_boost=0.0
    )

    # Calculate score with preference boost
    score_with_pref = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.9, 0.8],
        total_chunks_in_session=10,
        preference_boost=0.05
    )

    # Final score should be higher with preference boost
    assert score_with_pref.final_score > score_no_pref.final_score
    assert score_with_pref.final_score == score_no_pref.final_score + 0.05
    assert score_with_pref.preference_boost == 0.05
    assert score_no_pref.preference_boost == 0.0


def test_scoring_with_large_preference_boost(scoring_service):
    """Test scoring with maximum preference boost."""
    # Max boost from 5 forks
    max_boost = 0.10

    score = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.8],
        total_chunks_in_session=10,
        preference_boost=max_boost
    )

    assert score.preference_boost == max_boost
    # Final score is weighted: (0.8*0.4 + 0.8*0.2 + 0.1*0.05 + 0*0.25 + 0.5*0.1) + 0.10 = 0.535 + 0.10 = 0.635
    assert score.final_score > 0.6  # Base weighted score + boost


def test_preference_boost_in_session_score_dict(scoring_service):
    """Test that preference boost is included in SessionScore.to_dict()."""
    score = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.9],
        total_chunks_in_session=10,
        preference_boost=0.05
    )

    score_dict = score.to_dict()
    assert 'preference_boost' in score_dict
    assert score_dict['preference_boost'] == 0.05


def test_search_service_with_preference_disabled(temp_dirs):
    """Test that search works with preferences disabled."""
    embedding_service = EmbeddingService(use_cache=False)
    vector_db_service = VectorDBService(
        persist_directory=temp_dirs['vector_db_dir']
    )
    scoring_service = ScoringService()
    session_registry = SessionRegistry(registry_path=temp_dirs['registry_file'])

    # Create search service with preferences disabled
    search_service = SearchService(
        embedding_service=embedding_service,
        vector_db_service=vector_db_service,
        scoring_service=scoring_service,
        session_registry=session_registry,
        enable_preferences=False,
        enable_cache=False
    )

    # Should initialize without error
    assert not search_service.enable_preferences
    assert search_service.preference_service is None


def test_search_service_with_preference_enabled(temp_dirs, preference_service):
    """Test that search service initializes with preferences enabled."""
    embedding_service = EmbeddingService(use_cache=False)
    vector_db_service = VectorDBService(
        persist_directory=temp_dirs['vector_db_dir']
    )
    scoring_service = ScoringService()
    session_registry = SessionRegistry(registry_path=temp_dirs['registry_file'])

    # Create search service with preferences enabled
    search_service = SearchService(
        embedding_service=embedding_service,
        vector_db_service=vector_db_service,
        scoring_service=scoring_service,
        session_registry=session_registry,
        preference_service=preference_service,
        enable_preferences=True,
        enable_cache=False
    )

    assert search_service.enable_preferences
    assert search_service.preference_service is not None


def test_preference_learning_workflow(preference_service, scoring_service):
    """Test complete workflow: record selection -> calculate boost -> apply to scoring."""
    # Step 1: Simulate user forking a session multiple times
    for i in range(3):
        preference_service.record_selection("session-001", "test query", 1)

    # Step 2: Calculate preference boost
    pref_score = preference_service.calculate_preference_boost("session-001")
    assert pref_score.fork_count == 3
    assert pref_score.preference_boost > 0

    # Step 3: Apply boost to scoring
    score = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.7],
        total_chunks_in_session=10,
        preference_boost=pref_score.preference_boost
    )

    # Score should be boosted
    assert score.preference_boost == pref_score.preference_boost
    # Composite score is weighted, not a simple sum
    assert score.final_score > 0.5  # Base score + boost > 0.5


def test_preference_affects_ranking():
    """Test that preference boost affects session ranking."""
    scoring_service = ScoringService()

    # Create two sessions with same similarity but different preference boost
    score1 = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.8],
        total_chunks_in_session=10,
        preference_boost=0.0
    )

    score2 = scoring_service.calculate_session_score(
        session_id="session-002",
        chunk_similarities=[0.8],
        total_chunks_in_session=10,
        preference_boost=0.05
    )

    # Rank them
    ranked = scoring_service.rank_sessions([score1, score2])

    # Session with preference boost should rank higher
    assert ranked[0].session_id == "session-002"
    assert ranked[1].session_id == "session-001"


def test_multiple_sessions_with_different_preferences(preference_service):
    """Test batch preference calculation for multiple sessions."""
    # Record different fork patterns
    for i in range(5):  # Session 1: forked 5 times
        preference_service.record_selection("session-001", "query", 1)

    for i in range(2):  # Session 2: forked 2 times
        preference_service.record_selection("session-002", "query", 2)

    # No forks for session-003

    # Calculate boosts
    boosts = preference_service.calculate_preference_boosts([
        "session-001",
        "session-002",
        "session-003"
    ])

    # Session 1 should have highest boost
    assert boosts["session-001"].preference_boost > boosts["session-002"].preference_boost
    assert boosts["session-002"].preference_boost > boosts["session-003"].preference_boost
    assert boosts["session-003"].preference_boost == 0.0


def test_preference_boost_with_memory_boost(scoring_service):
    """Test that preference boost and memory boost work together."""
    score = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.8],
        total_chunks_in_session=10,
        memory_types=['PATTERN'],
        preference_boost=0.05
    )

    # Should have both boosts
    assert score.memory_boost == 0.05  # PATTERN boost
    assert score.preference_boost == 0.05
    # Final score includes both boosts added to weighted base
    # Weighted base: (0.8*0.4 + 0.8*0.2 + 0.1*0.05 + 0*0.25 + 0.5*0.1) = 0.535
    # Plus boosts: 0.535 + 0.05 + 0.05 = 0.635
    assert score.final_score > 0.6  # Base score + boosts


def test_empty_preferences_file_handling(temp_dirs):
    """Test handling of empty preferences file."""
    # Create empty file
    with open(temp_dirs['preference_file'], 'w') as f:
        f.write('[]')

    service = PreferenceService(preference_file=temp_dirs['preference_file'])
    stats = service.get_stats()

    assert stats['total_selections'] == 0
    assert stats['unique_sessions'] == 0


def test_corrupted_preferences_file_handling(temp_dirs):
    """Test handling of corrupted preferences file."""
    # Create corrupted file
    with open(temp_dirs['preference_file'], 'w') as f:
        f.write('{ invalid json }')

    service = PreferenceService(preference_file=temp_dirs['preference_file'])

    # Should handle gracefully and return empty stats
    stats = service.get_stats()
    assert stats['total_selections'] == 0


def test_backward_compatibility_scoring(scoring_service):
    """Test that old code without preference_boost parameter still works."""
    # Call without preference_boost parameter (should default to 0.0)
    score = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.8],
        total_chunks_in_session=10
    )

    assert score.preference_boost == 0.0
    assert score.final_score > 0


def test_preference_boost_serialization(scoring_service):
    """Test that SessionScore with preference_boost can be serialized."""
    score = scoring_service.calculate_session_score(
        session_id="session-001",
        chunk_similarities=[0.8],
        total_chunks_in_session=10,
        preference_boost=0.05
    )

    # Convert to dict and back
    score_dict = score.to_dict()
    import json
    json_str = json.dumps(score_dict)
    loaded_dict = json.loads(json_str)

    assert loaded_dict['preference_boost'] == 0.05
    assert loaded_dict['session_id'] == "session-001"
