"""Tests for CacheService."""

import time
import pytest
from src.smart_fork.cache_service import CacheService, LRUCacheWithTTL, CacheStats


class TestLRUCacheWithTTL:
    """Tests for LRUCacheWithTTL."""

    def test_basic_get_put(self):
        """Test basic get and put operations."""
        cache = LRUCacheWithTTL(max_size=10, ttl_seconds=60)

        # Put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Get non-existent key
        assert cache.get("key2") is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCacheWithTTL(max_size=3, ttl_seconds=60)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Add one more - should evict key1 (least recently used)
        cache.put("key4", "value4")

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_access_updates_order(self):
        """Test that accessing an item updates its position."""
        cache = LRUCacheWithTTL(max_size=3, ttl_seconds=60)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4 - should evict key2 (now least recently used)
        cache.put("key4", "value4")

        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = LRUCacheWithTTL(max_size=10, ttl_seconds=0.1)  # 100ms TTL

        cache.put("key1", "value1")

        # Should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        assert cache.get("key1") is None

    def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = LRUCacheWithTTL(max_size=10, ttl_seconds=60)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Update
        cache.put("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_clear(self):
        """Test clearing the cache."""
        cache = LRUCacheWithTTL(max_size=10, ttl_seconds=60)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.size() == 2

        cache.clear()

        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats_tracking(self):
        """Test that cache statistics are tracked correctly."""
        cache = LRUCacheWithTTL(max_size=10, ttl_seconds=60)

        # Initial state
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

        # Miss
        cache.get("key1")
        stats = cache.get_stats()
        assert stats.misses == 1

        # Put and hit
        cache.put("key1", "value1")
        cache.get("key1")
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1

        # Eviction
        cache = LRUCacheWithTTL(max_size=2, ttl_seconds=60)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1

        stats = cache.get_stats()
        assert stats.evictions == 1


class TestCacheService:
    """Tests for CacheService."""

    def test_query_embedding_cache(self):
        """Test query embedding caching."""
        cache = CacheService(
            embedding_cache_size=10,
            embedding_ttl_seconds=60,
            result_cache_size=10,
            result_ttl_seconds=60
        )

        embedding = [0.1, 0.2, 0.3]

        # Cache miss
        assert cache.get_query_embedding("test query") is None

        # Store
        cache.put_query_embedding("test query", embedding)

        # Cache hit
        assert cache.get_query_embedding("test query") == embedding

    def test_query_embedding_normalization(self):
        """Test that query strings are normalized for caching."""
        cache = CacheService()

        embedding = [0.1, 0.2, 0.3]

        # Store with one format
        cache.put_query_embedding("Test Query", embedding)

        # Retrieve with different case and whitespace
        assert cache.get_query_embedding("test query") == embedding
        assert cache.get_query_embedding("  TEST QUERY  ") == embedding

    def test_search_results_cache(self):
        """Test search results caching."""
        cache = CacheService(
            embedding_cache_size=10,
            embedding_ttl_seconds=60,
            result_cache_size=10,
            result_ttl_seconds=60
        )

        results = [{"session_id": "test1", "score": 0.9}]

        # Cache miss
        assert cache.get_search_results("test query") is None

        # Store
        cache.put_search_results("test query", results)

        # Cache hit
        assert cache.get_search_results("test query") == results

    def test_search_results_with_filters(self):
        """Test that filters are part of cache key."""
        cache = CacheService()

        results1 = [{"session_id": "test1"}]
        results2 = [{"session_id": "test2"}]

        # Store with no filters
        cache.put_search_results("test query", results1)

        # Store with filters
        filters = {"project": "myproject"}
        cache.put_search_results("test query", results2, filter_metadata=filters)

        # Should get different results
        assert cache.get_search_results("test query") == results1
        assert cache.get_search_results("test query", filter_metadata=filters) == results2

    def test_invalidate_results(self):
        """Test invalidating result cache only."""
        cache = CacheService()

        embedding = [0.1, 0.2, 0.3]
        results = [{"session_id": "test1"}]

        cache.put_query_embedding("query", embedding)
        cache.put_search_results("query", results)

        # Invalidate results only
        cache.invalidate_results()

        # Embedding should still be cached
        assert cache.get_query_embedding("query") == embedding
        # Results should be gone
        assert cache.get_search_results("query") is None

    def test_invalidate_all(self):
        """Test invalidating all caches."""
        cache = CacheService()

        embedding = [0.1, 0.2, 0.3]
        results = [{"session_id": "test1"}]

        cache.put_query_embedding("query", embedding)
        cache.put_search_results("query", results)

        # Invalidate everything
        cache.invalidate_all()

        # Both should be gone
        assert cache.get_query_embedding("query") is None
        assert cache.get_search_results("query") is None

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = CacheService(
            embedding_cache_size=100,
            embedding_ttl_seconds=300,
            result_cache_size=50,
            result_ttl_seconds=300
        )

        # Add some items
        cache.put_query_embedding("query1", [0.1, 0.2])
        cache.put_query_embedding("query2", [0.3, 0.4])
        cache.put_search_results("query1", [{"id": "1"}])

        stats = cache.get_stats()

        # Check structure
        assert "embedding_cache" in stats
        assert "result_cache" in stats

        # Check embedding cache
        assert stats["embedding_cache"]["size"] == 2
        assert stats["embedding_cache"]["max_size"] == 100
        assert stats["embedding_cache"]["ttl_seconds"] == 300

        # Check result cache
        assert stats["result_cache"]["size"] == 1
        assert stats["result_cache"]["max_size"] == 50
        assert stats["result_cache"]["ttl_seconds"] == 300

        # Hit some items to verify stats
        cache.get_query_embedding("query1")  # Hit
        cache.get_query_embedding("query3")  # Miss

        stats = cache.get_stats()
        embedding_stats = stats["embedding_cache"]["stats"]
        assert embedding_stats["hits"] == 1
        assert embedding_stats["misses"] == 1


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_calculation(self):
        """Test hit rate percentage calculation."""
        stats = CacheStats(hits=75, misses=25)

        assert stats.total_requests == 100
        assert stats.hit_rate == 75.0

    def test_hit_rate_zero_requests(self):
        """Test hit rate with zero requests."""
        stats = CacheStats()

        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CacheStats(hits=50, misses=50, evictions=10)

        result = stats.to_dict()

        assert result["hits"] == 50
        assert result["misses"] == 50
        assert result["evictions"] == 10
        assert result["total_requests"] == 100
        assert result["hit_rate"] == "50.00%"
