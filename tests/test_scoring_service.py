"""
Unit tests for the ScoringService class.

Tests all components of the composite scoring algorithm including:
- Best similarity calculation
- Average similarity calculation
- Chunk ratio calculation
- Recency decay calculation
- Memory type boosting
- Final score composition
- Session ranking
"""

import pytest
from datetime import datetime, timedelta
import math
from smart_fork.scoring_service import ScoringService, SessionScore


class TestScoringServiceInit:
    """Test ScoringService initialization."""

    def test_default_chain_quality(self):
        """Test default chain quality placeholder is 0.5."""
        service = ScoringService()
        assert service.chain_quality_placeholder == 0.5

    def test_custom_chain_quality(self):
        """Test custom chain quality placeholder."""
        service = ScoringService(chain_quality_placeholder=0.7)
        assert service.chain_quality_placeholder == 0.7


class TestBestSimilarityCalculation:
    """Test best_similarity component calculation."""

    def test_best_similarity_single_chunk(self):
        """Test best similarity with single chunk."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.85],
            total_chunks_in_session=10
        )
        assert score.best_similarity == 0.85

    def test_best_similarity_multiple_chunks(self):
        """Test best similarity picks the maximum."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.6, 0.9, 0.7, 0.85],
            total_chunks_in_session=20
        )
        assert score.best_similarity == 0.9

    def test_best_similarity_empty_chunks(self):
        """Test best similarity with no chunks."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[],
            total_chunks_in_session=10
        )
        assert score.best_similarity == 0.0


class TestAvgSimilarityCalculation:
    """Test avg_similarity component calculation."""

    def test_avg_similarity_single_chunk(self):
        """Test average similarity with single chunk."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.75],
            total_chunks_in_session=10
        )
        assert score.avg_similarity == 0.75

    def test_avg_similarity_multiple_chunks(self):
        """Test average similarity calculation."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.6, 0.8, 0.7, 0.9],
            total_chunks_in_session=20
        )
        expected_avg = (0.6 + 0.8 + 0.7 + 0.9) / 4
        assert abs(score.avg_similarity - expected_avg) < 0.001

    def test_avg_similarity_empty_chunks(self):
        """Test average similarity with no chunks."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[],
            total_chunks_in_session=10
        )
        assert score.avg_similarity == 0.0


class TestChunkRatioCalculation:
    """Test chunk_ratio component calculation."""

    def test_chunk_ratio_all_matched(self):
        """Test chunk ratio when all chunks matched."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8, 0.7, 0.9, 0.85, 0.75],
            total_chunks_in_session=5
        )
        assert score.chunk_ratio == 1.0

    def test_chunk_ratio_partial_match(self):
        """Test chunk ratio with partial matches."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8, 0.7, 0.9],
            total_chunks_in_session=10
        )
        assert abs(score.chunk_ratio - 0.3) < 0.001

    def test_chunk_ratio_zero_total(self):
        """Test chunk ratio when total chunks is zero."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=0
        )
        assert score.chunk_ratio == 0.0


class TestRecencyScoreCalculation:
    """Test recency_score component calculation."""

    def test_recency_score_recent_session(self):
        """Test recency score for very recent session (should be close to 1.0)."""
        service = ScoringService()
        current_time = datetime.now()
        last_modified = current_time - timedelta(hours=1)

        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            session_last_modified=last_modified.isoformat(),
            current_time=current_time
        )

        # 1 hour old should have recency close to 1.0
        assert score.recency_score > 0.99

    def test_recency_score_30_days_old(self):
        """Test recency score at exactly 30 days (decay constant)."""
        service = ScoringService()
        current_time = datetime.now()
        last_modified = current_time - timedelta(days=30)

        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            session_last_modified=last_modified.isoformat(),
            current_time=current_time
        )

        # At decay constant, recency should be exp(-1) ≈ 0.368
        expected = math.exp(-1)
        assert abs(score.recency_score - expected) < 0.01

    def test_recency_score_old_session(self):
        """Test recency score for very old session."""
        service = ScoringService()
        current_time = datetime.now()
        last_modified = current_time - timedelta(days=365)

        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            session_last_modified=last_modified.isoformat(),
            current_time=current_time
        )

        # 365 days old should have very low recency
        assert score.recency_score < 0.01

    def test_recency_score_no_timestamp(self):
        """Test recency score when timestamp is None."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            session_last_modified=None
        )
        assert score.recency_score == 0.0

    def test_recency_score_invalid_timestamp(self):
        """Test recency score with invalid timestamp."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            session_last_modified="invalid-timestamp"
        )
        assert score.recency_score == 0.0

    def test_recency_score_future_timestamp(self):
        """Test recency score with future timestamp (edge case)."""
        service = ScoringService()
        current_time = datetime.now()
        future_time = current_time + timedelta(days=1)

        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            session_last_modified=future_time.isoformat(),
            current_time=current_time
        )

        # Future timestamp should be treated as age=0, so recency=1.0
        assert score.recency_score == 1.0


class TestMemoryBoostCalculation:
    """Test memory type boost calculation."""

    def test_memory_boost_pattern(self):
        """Test PATTERN memory boost (+5%)."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            memory_types=['PATTERN']
        )
        assert score.memory_boost == 0.05

    def test_memory_boost_working_solution(self):
        """Test WORKING_SOLUTION memory boost (+8%)."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            memory_types=['WORKING_SOLUTION']
        )
        assert score.memory_boost == 0.08

    def test_memory_boost_waiting(self):
        """Test WAITING memory boost (+2%)."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            memory_types=['WAITING']
        )
        assert score.memory_boost == 0.02

    def test_memory_boost_multiple_types(self):
        """Test multiple memory types are additive."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            memory_types=['PATTERN', 'WORKING_SOLUTION', 'WAITING']
        )
        expected_boost = 0.05 + 0.08 + 0.02
        assert abs(score.memory_boost - expected_boost) < 0.001

    def test_memory_boost_duplicate_types(self):
        """Test duplicate memory types only count once."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            memory_types=['PATTERN', 'PATTERN', 'PATTERN']
        )
        assert score.memory_boost == 0.05

    def test_memory_boost_no_types(self):
        """Test no memory types results in zero boost."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            memory_types=None
        )
        assert score.memory_boost == 0.0

    def test_memory_boost_unknown_types(self):
        """Test unknown memory types are ignored."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.8],
            total_chunks_in_session=10,
            memory_types=['UNKNOWN', 'INVALID']
        )
        assert score.memory_boost == 0.0


class TestFinalScoreComposition:
    """Test final score calculation with all components."""

    def test_final_score_formula_weights(self):
        """Test that final score uses correct weights."""
        service = ScoringService()
        current_time = datetime.now()
        last_modified = current_time - timedelta(hours=1)

        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.9, 0.8, 0.7],  # best=0.9, avg=0.8
            total_chunks_in_session=10,  # ratio=0.3
            session_last_modified=last_modified.isoformat(),
            current_time=current_time,
            memory_types=None
        )

        # Calculate expected base score
        expected_base = (
            (0.9 * 0.40) +   # best_similarity
            (0.8 * 0.20) +   # avg_similarity
            (0.3 * 0.05) +   # chunk_ratio
            (score.recency_score * 0.25) +  # recency (close to 1.0)
            (0.5 * 0.10)     # chain_quality placeholder
        )

        # Final score should match base score (no memory boost)
        assert abs(score.final_score - expected_base) < 0.01

    def test_final_score_with_memory_boost(self):
        """Test final score includes memory boost additively."""
        service = ScoringService()
        current_time = datetime.now()
        last_modified = current_time - timedelta(hours=1)

        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[0.9, 0.8],
            total_chunks_in_session=10,
            session_last_modified=last_modified.isoformat(),
            current_time=current_time,
            memory_types=['WORKING_SOLUTION']  # +8%
        )

        # Base score + memory boost
        expected_base = (
            (0.9 * 0.40) +
            (0.85 * 0.20) +
            (0.2 * 0.05) +
            (score.recency_score * 0.25) +
            (0.5 * 0.10)
        )
        expected_final = expected_base + 0.08

        assert abs(score.final_score - expected_final) < 0.01

    def test_final_score_empty_chunks(self):
        """Test final score with no matched chunks is zero."""
        service = ScoringService()
        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[],
            total_chunks_in_session=10
        )
        assert score.final_score == 0.0

    def test_final_score_perfect_match(self):
        """Test final score with perfect similarity and recent session."""
        service = ScoringService()
        current_time = datetime.now()
        last_modified = current_time - timedelta(minutes=1)

        score = service.calculate_session_score(
            session_id="test-1",
            chunk_similarities=[1.0] * 10,
            total_chunks_in_session=10,
            session_last_modified=last_modified.isoformat(),
            current_time=current_time,
            memory_types=['PATTERN', 'WORKING_SOLUTION', 'WAITING']
        )

        # Should be close to maximum possible score
        # Base: 1.0*0.4 + 1.0*0.2 + 1.0*0.05 + ~1.0*0.25 + 0.5*0.1 = ~0.95
        # Boost: 0.05 + 0.08 + 0.02 = 0.15
        # Total: ~1.10
        assert score.final_score > 1.0


class TestSessionScoreDataclass:
    """Test SessionScore dataclass."""

    def test_session_score_to_dict(self):
        """Test SessionScore.to_dict() conversion."""
        score = SessionScore(
            session_id="test-1",
            final_score=0.85,
            best_similarity=0.9,
            avg_similarity=0.8,
            chunk_ratio=0.5,
            recency_score=0.7,
            chain_quality=0.5,
            memory_boost=0.05,
            preference_boost=0.02,
            num_chunks_matched=5
        )

        result = score.to_dict()
        assert result['session_id'] == "test-1"
        assert result['final_score'] == 0.85
        assert result['best_similarity'] == 0.9
        assert result['avg_similarity'] == 0.8
        assert result['chunk_ratio'] == 0.5
        assert result['recency_score'] == 0.7
        assert result['chain_quality'] == 0.5
        assert result['memory_boost'] == 0.05
        assert result['preference_boost'] == 0.02
        assert result['num_chunks_matched'] == 5


class TestSessionRanking:
    """Test session ranking functionality."""

    def test_rank_sessions_sorts_by_score(self):
        """Test that sessions are sorted by final score."""
        service = ScoringService()
        scores = [
            SessionScore("s1", 0.5, 0.6, 0.5, 0.3, 0.4, 0.5, 0.0, 0.0, 3),
            SessionScore("s2", 0.9, 0.95, 0.9, 0.7, 0.8, 0.5, 0.05, 0.0, 8),
            SessionScore("s3", 0.7, 0.8, 0.7, 0.5, 0.6, 0.5, 0.0, 0.0, 5),
        ]

        ranked = service.rank_sessions(scores, top_k=3)

        assert ranked[0].session_id == "s2"
        assert ranked[1].session_id == "s3"
        assert ranked[2].session_id == "s1"

    def test_rank_sessions_top_k(self):
        """Test that only top K sessions are returned."""
        service = ScoringService()
        scores = [
            SessionScore(f"s{i}", float(i) / 10, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1)
            for i in range(10)
        ]

        ranked = service.rank_sessions(scores, top_k=3)
        assert len(ranked) == 3

    def test_rank_sessions_fewer_than_k(self):
        """Test ranking when fewer sessions than top_k."""
        service = ScoringService()
        scores = [
            SessionScore("s1", 0.8, 0.8, 0.8, 0.5, 0.7, 0.5, 0.0, 0.0, 5),
            SessionScore("s2", 0.6, 0.7, 0.6, 0.4, 0.5, 0.5, 0.0, 0.0, 4),
        ]

        ranked = service.rank_sessions(scores, top_k=5)
        assert len(ranked) == 2

    def test_rank_sessions_empty_list(self):
        """Test ranking with empty list."""
        service = ScoringService()
        ranked = service.rank_sessions([], top_k=5)
        assert len(ranked) == 0
