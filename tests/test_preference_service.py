"""Tests for preference service."""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

from smart_fork.preference_service import (
    PreferenceService,
    PreferenceEntry,
    PreferenceScore
)


@pytest.fixture
def temp_preference_file():
    """Create a temporary preference file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def preference_service(temp_preference_file):
    """Create a preference service with temporary storage."""
    return PreferenceService(preference_file=temp_preference_file)


def test_initialization(temp_preference_file):
    """Test service initialization."""
    service = PreferenceService(preference_file=temp_preference_file)
    assert os.path.exists(temp_preference_file)
    stats = service.get_stats()
    assert stats['total_selections'] == 0
    assert stats['unique_sessions'] == 0


def test_record_selection(preference_service):
    """Test recording a selection."""
    preference_service.record_selection(
        session_id="session-001",
        query="test query",
        position=1
    )

    stats = preference_service.get_stats()
    assert stats['total_selections'] == 1
    assert stats['unique_sessions'] == 1
    assert stats['position_distribution'] == {1: 1}


def test_record_multiple_selections(preference_service):
    """Test recording multiple selections."""
    # Record 3 selections
    preference_service.record_selection("session-001", "query 1", 1)
    preference_service.record_selection("session-002", "query 2", 2)
    preference_service.record_selection("session-001", "query 3", 3)

    stats = preference_service.get_stats()
    assert stats['total_selections'] == 3
    assert stats['unique_sessions'] == 2
    assert stats['position_distribution'] == {1: 1, 2: 1, 3: 1}


def test_calculate_preference_boost_no_history(preference_service):
    """Test calculating boost for session with no history."""
    score = preference_service.calculate_preference_boost("session-999")

    assert score.session_id == "session-999"
    assert score.preference_boost == 0.0
    assert score.fork_count == 0
    assert score.avg_position == 0.0
    assert score.recency_weight == 0.0


def test_calculate_preference_boost_single_fork(preference_service):
    """Test calculating boost for session with one fork."""
    preference_service.record_selection("session-001", "test query", 1)

    score = preference_service.calculate_preference_boost("session-001")

    assert score.session_id == "session-001"
    assert score.fork_count == 1
    assert score.avg_position == 1.0
    assert score.recency_weight > 0.9  # Should be very recent
    assert score.preference_boost > 0.0


def test_calculate_preference_boost_multiple_forks(preference_service):
    """Test calculating boost for frequently forked session."""
    # Fork the same session 5 times
    for i in range(5):
        preference_service.record_selection("session-001", f"query {i}", 1)

    score = preference_service.calculate_preference_boost("session-001")

    assert score.fork_count == 5
    assert score.avg_position == 1.0
    # Should hit max boost cap
    base_boost = 5 * 0.02  # 0.10 (at cap)
    position_bonus = 0.01 * (6 - 1) / 5.0  # 0.01
    expected_min_boost = (base_boost + position_bonus) * 0.9  # with recency
    assert score.preference_boost >= expected_min_boost


def test_position_quality_impact(preference_service):
    """Test that position affects boost calculation."""
    # Record selection at position 1 (best)
    preference_service.record_selection("session-001", "query", 1)

    # Record selection at position 5 (worst)
    preference_service.record_selection("session-002", "query", 5)

    score1 = preference_service.calculate_preference_boost("session-001")
    score2 = preference_service.calculate_preference_boost("session-002")

    # Position 1 should get higher boost than position 5
    assert score1.preference_boost > score2.preference_boost
    assert score1.avg_position < score2.avg_position


def test_recency_weight_decay(preference_service, temp_preference_file):
    """Test that recency decays over time."""
    # Record a selection
    preference_service.record_selection("session-001", "query", 1)

    # Get current time and future time (30 days later)
    current_time = datetime.utcnow()
    future_time = current_time + timedelta(days=30)

    # Calculate boost at current time
    score_now = preference_service.calculate_preference_boost(
        "session-001",
        current_time=current_time
    )

    # Calculate boost 30 days later
    score_future = preference_service.calculate_preference_boost(
        "session-001",
        current_time=future_time
    )

    # Boost should decay over time
    assert score_future.preference_boost < score_now.preference_boost
    assert score_future.recency_weight < score_now.recency_weight


def test_custom_position_excluded_from_average(preference_service):
    """Test that custom entries (position=-1) don't affect average."""
    # Mix of positioned and custom selections
    preference_service.record_selection("session-001", "query", 1)
    preference_service.record_selection("session-001", "query", 2)
    preference_service.record_selection("session-001", "query", -1)  # Custom

    score = preference_service.calculate_preference_boost("session-001")

    # Average should only consider positions 1 and 2
    assert score.avg_position == 1.5
    assert score.fork_count == 3  # All 3 count toward fork count


def test_calculate_preference_boosts_batch(preference_service):
    """Test batch calculation of preference boosts."""
    # Record selections for multiple sessions
    preference_service.record_selection("session-001", "query", 1)
    preference_service.record_selection("session-002", "query", 2)
    preference_service.record_selection("session-003", "query", 1)

    # Calculate boosts for all sessions
    session_ids = ["session-001", "session-002", "session-003", "session-999"]
    boosts = preference_service.calculate_preference_boosts(session_ids)

    assert len(boosts) == 4
    assert boosts["session-001"].preference_boost > 0
    assert boosts["session-002"].preference_boost > 0
    assert boosts["session-003"].preference_boost > 0
    assert boosts["session-999"].preference_boost == 0.0  # No history


def test_get_most_forked_sessions(preference_service):
    """Test getting most frequently forked sessions."""
    # Fork different sessions different amounts
    for i in range(5):
        preference_service.record_selection("session-001", "query", 1)

    for i in range(3):
        preference_service.record_selection("session-002", "query", 2)

    preference_service.record_selection("session-003", "query", 3)

    most_forked = preference_service.get_most_forked_sessions(limit=3)

    assert len(most_forked) == 3
    assert most_forked[0]['session_id'] == "session-001"
    assert most_forked[0]['fork_count'] == 5
    assert most_forked[1]['session_id'] == "session-002"
    assert most_forked[1]['fork_count'] == 3
    assert most_forked[2]['session_id'] == "session-003"
    assert most_forked[2]['fork_count'] == 1


def test_persistence(temp_preference_file):
    """Test that preferences are persisted to disk."""
    # Create service and record selection
    service1 = PreferenceService(preference_file=temp_preference_file)
    service1.record_selection("session-001", "test query", 1)

    # Create new service instance (simulating restart)
    service2 = PreferenceService(preference_file=temp_preference_file)
    stats = service2.get_stats()

    assert stats['total_selections'] == 1
    assert stats['unique_sessions'] == 1

    score = service2.calculate_preference_boost("session-001")
    assert score.fork_count == 1


def test_max_entries_pruning(temp_preference_file):
    """Test that old entries are pruned when exceeding max."""
    service = PreferenceService(preference_file=temp_preference_file, max_entries=10)

    # Record 15 selections (more than max)
    for i in range(15):
        service.record_selection(f"session-{i:03d}", "query", 1)

    stats = service.get_stats()
    # Should only keep 10 most recent
    assert stats['total_selections'] == 10


def test_clear(preference_service):
    """Test clearing all preference data."""
    # Record some data
    preference_service.record_selection("session-001", "query", 1)
    preference_service.record_selection("session-002", "query", 2)

    # Clear
    preference_service.clear()

    stats = preference_service.get_stats()
    assert stats['total_selections'] == 0
    assert stats['unique_sessions'] == 0


def test_thread_safety_simulation(preference_service):
    """Test basic thread safety by rapid sequential access."""
    # Simulate concurrent access with rapid calls
    for i in range(100):
        preference_service.record_selection(f"session-{i % 10}", "query", 1)

    stats = preference_service.get_stats()
    assert stats['total_selections'] == 100


def test_preference_entry_serialization():
    """Test PreferenceEntry to_dict and from_dict."""
    entry = PreferenceEntry(
        session_id="session-001",
        query="test query",
        position=1,
        timestamp="2026-01-21T12:00:00Z"
    )

    # Test to_dict
    data = entry.to_dict()
    assert data['session_id'] == "session-001"
    assert data['query'] == "test query"
    assert data['position'] == 1

    # Test from_dict
    entry2 = PreferenceEntry.from_dict(data)
    assert entry2.session_id == entry.session_id
    assert entry2.query == entry.query
    assert entry2.position == entry.position
    assert entry2.timestamp == entry.timestamp


def test_preference_score_to_dict():
    """Test PreferenceScore to_dict."""
    score = PreferenceScore(
        session_id="session-001",
        preference_boost=0.05,
        fork_count=3,
        avg_position=1.5,
        recency_weight=0.95
    )

    data = score.to_dict()
    assert data['session_id'] == "session-001"
    assert data['preference_boost'] == 0.05
    assert data['fork_count'] == 3
    assert data['avg_position'] == 1.5
    assert data['recency_weight'] == 0.95


def test_invalid_timestamp_handling(preference_service, temp_preference_file):
    """Test handling of invalid timestamps in data."""
    # Manually create entry with invalid timestamp
    preferences = [{
        'session_id': 'session-001',
        'query': 'test',
        'position': 1,
        'timestamp': 'invalid-timestamp'
    }]

    with open(temp_preference_file, 'w') as f:
        json.dump(preferences, f)

    # Should not crash, should return minimum recency weight
    score = preference_service.calculate_preference_boost("session-001")
    assert score.recency_weight == 0.0
