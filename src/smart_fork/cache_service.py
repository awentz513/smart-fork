"""
Cache service for storing query embeddings and search results.

This module provides LRU caching with TTL support to improve performance
by avoiding redundant embedding computations and database searches.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'total_requests': self.total_requests,
            'hit_rate': f"{self.hit_rate:.2f}%"
        }


@dataclass
class CacheEntry:
    """A cache entry with value and expiration time."""
    value: Any
    expires_at: float

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() > self.expires_at


class LRUCacheWithTTL:
    """
    LRU cache with time-to-live (TTL) support.

    Features:
    - Least Recently Used (LRU) eviction policy
    - Time-to-live (TTL) expiration per entry
    - Thread-safe operations
    - Cache statistics tracking
    """

    def __init__(self, max_size: int, ttl_seconds: float):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries to store
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()

        logger.info(f"Initialized LRU cache (max_size={max_size}, ttl={ttl_seconds}s)")

    def _compute_key(self, key: Any) -> str:
        """
        Compute a string key from any hashable value.

        Args:
            key: Key to hash (typically a string or tuple)

        Returns:
            String key suitable for cache lookup
        """
        if isinstance(key, str):
            return key
        return str(key)

    def get(self, key: Any) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        cache_key = self._compute_key(key)

        if cache_key not in self.cache:
            self.stats.misses += 1
            logger.debug(f"Cache miss: {cache_key[:50]}")
            return None

        entry = self.cache[cache_key]

        # Check if entry has expired
        if entry.is_expired():
            logger.debug(f"Cache entry expired: {cache_key[:50]}")
            del self.cache[cache_key]
            self.stats.misses += 1
            return None

        # Move to end (mark as recently used)
        self.cache.move_to_end(cache_key)
        self.stats.hits += 1
        logger.debug(f"Cache hit: {cache_key[:50]}")

        return entry.value

    def put(self, key: Any, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        cache_key = self._compute_key(key)
        expires_at = time.time() + self.ttl_seconds

        # If key already exists, update it and move to end
        if cache_key in self.cache:
            self.cache[cache_key] = CacheEntry(value=value, expires_at=expires_at)
            self.cache.move_to_end(cache_key)
            logger.debug(f"Cache updated: {cache_key[:50]}")
            return

        # Check if we need to evict
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            evicted_key, _ = self.cache.popitem(last=False)
            self.stats.evictions += 1
            logger.debug(f"Cache evicted LRU: {evicted_key[:50]}")

        # Add new entry
        self.cache[cache_key] = CacheEntry(value=value, expires_at=expires_at)
        logger.debug(f"Cache stored: {cache_key[:50]}")

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared ({count} entries removed)")

    def size(self) -> int:
        """Get current number of entries in cache."""
        return len(self.cache)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


class CacheService:
    """
    Service for caching query embeddings and search results.

    Provides two separate caches:
    1. Embedding cache: Stores query text -> embedding vectors
    2. Result cache: Stores (query text, filter params) -> search results
    """

    def __init__(
        self,
        embedding_cache_size: int = 100,
        embedding_ttl_seconds: float = 300,  # 5 minutes
        result_cache_size: int = 50,
        result_ttl_seconds: float = 300,  # 5 minutes
    ):
        """
        Initialize the cache service.

        Args:
            embedding_cache_size: Max number of query embeddings to cache
            embedding_ttl_seconds: TTL for embedding cache entries
            result_cache_size: Max number of search results to cache
            result_ttl_seconds: TTL for result cache entries
        """
        self.embedding_cache = LRUCacheWithTTL(
            max_size=embedding_cache_size,
            ttl_seconds=embedding_ttl_seconds
        )

        self.result_cache = LRUCacheWithTTL(
            max_size=result_cache_size,
            ttl_seconds=result_ttl_seconds
        )

        logger.info(
            f"Initialized CacheService "
            f"(embedding={embedding_cache_size}/{embedding_ttl_seconds}s, "
            f"result={result_cache_size}/{result_ttl_seconds}s)"
        )

    def get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Get cached embedding for a query.

        Args:
            query: Query text

        Returns:
            Cached embedding vector if found, None otherwise
        """
        # Normalize query (lowercase, strip whitespace) for cache key
        normalized_query = query.lower().strip()
        return self.embedding_cache.get(normalized_query)

    def put_query_embedding(self, query: str, embedding: List[float]) -> None:
        """
        Store query embedding in cache.

        Args:
            query: Query text
            embedding: Embedding vector to cache
        """
        normalized_query = query.lower().strip()
        self.embedding_cache.put(normalized_query, embedding)

    def get_search_results(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Any]]:
        """
        Get cached search results.

        Args:
            query: Query text
            filter_metadata: Optional metadata filters

        Returns:
            Cached search results if found, None otherwise
        """
        cache_key = self._make_result_cache_key(query, filter_metadata)
        return self.result_cache.get(cache_key)

    def put_search_results(
        self,
        query: str,
        results: List[Any],
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store search results in cache.

        Args:
            query: Query text
            results: Search results to cache
            filter_metadata: Optional metadata filters
        """
        cache_key = self._make_result_cache_key(query, filter_metadata)
        self.result_cache.put(cache_key, results)

    def _make_result_cache_key(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """
        Create a cache key for search results.

        Args:
            query: Query text
            filter_metadata: Optional metadata filters

        Returns:
            Cache key combining query and filters
        """
        normalized_query = query.lower().strip()

        # If no filters, just use the query
        if not filter_metadata:
            return normalized_query

        # Create deterministic string from filters
        filter_str = str(sorted(filter_metadata.items()))

        # Hash the combined key to keep it short
        combined = f"{normalized_query}|{filter_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def invalidate_all(self) -> None:
        """
        Invalidate all cached data.

        This should be called when the database is updated
        to ensure cache consistency.
        """
        self.embedding_cache.clear()
        self.result_cache.clear()
        logger.info("All caches invalidated")

    def invalidate_results(self) -> None:
        """
        Invalidate only search result cache.

        Keep embedding cache intact since embeddings don't change
        when database is updated (only results change).
        """
        self.result_cache.clear()
        logger.info("Result cache invalidated")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with statistics for both caches
        """
        return {
            'embedding_cache': {
                'size': self.embedding_cache.size(),
                'max_size': self.embedding_cache.max_size,
                'ttl_seconds': self.embedding_cache.ttl_seconds,
                'stats': self.embedding_cache.get_stats().to_dict()
            },
            'result_cache': {
                'size': self.result_cache.size(),
                'max_size': self.result_cache.max_size,
                'ttl_seconds': self.result_cache.ttl_seconds,
                'stats': self.result_cache.get_stats().to_dict()
            }
        }
