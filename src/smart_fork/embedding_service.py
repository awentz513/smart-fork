"""Embedding service for generating vector embeddings using Nomic model."""

import gc
import logging
import time
from typing import List, Union, Optional

import psutil
import torch
from sentence_transformers import SentenceTransformer

from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using nomic-embed-text-v1.5 model.

    Features:
    - 768-dimensional embeddings
    - Adaptive batch sizing based on available RAM
    - Memory monitoring and garbage collection
    - Efficient batch processing
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        min_batch_size: int = 8,
        max_batch_size: int = 128,
        memory_threshold_mb: int = 500,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        throttle_seconds: float = 0.1,
        use_mps: bool = True,
    ):
        """Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use
            min_batch_size: Minimum batch size for processing
            max_batch_size: Maximum batch size for processing
            memory_threshold_mb: Memory threshold in MB for adaptive batching
            use_cache: Whether to use embedding cache (default: True)
            cache_dir: Optional cache directory (defaults to ~/.smart-fork/embedding_cache)
            throttle_seconds: Sleep time between batches to reduce CPU usage (default: 0.1)
            use_mps: Whether to use MPS (Metal) acceleration on Apple Silicon (default: True)
        """
        self.model_name = model_name
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold_mb = memory_threshold_mb
        self.model = None
        self.embedding_dimension = 768  # nomic-embed-text-v1.5 dimension
        self.throttle_seconds = throttle_seconds
        self.use_mps = use_mps
        self.device = "cpu"  # Will be updated when model loads

        # Initialize embedding cache
        self.use_cache = use_cache
        self.cache: Optional[EmbeddingCache] = None
        if use_cache:
            self.cache = EmbeddingCache(cache_dir=cache_dir)

        logger.info(
            f"Initializing EmbeddingService with model: {model_name} "
            f"(cache={'enabled' if use_cache else 'disabled'}, throttle={throttle_seconds}s)"
        )

    def load_model(self) -> None:
        """Load the embedding model into memory."""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model: {self.model_name}")
        try:
            # Detect best available device
            if self.use_mps and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("MPS (Metal) acceleration available - using GPU")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("CUDA acceleration available - using GPU")
            else:
                self.device = "cpu"
                logger.info("No GPU acceleration available - using CPU")

            # trust_remote_code is required for nomic models
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                device=self.device
            )
            logger.info(
                f"Model loaded successfully on {self.device}. "
                f"Embedding dimension: {self.embedding_dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB.

        Returns:
            Available memory in megabytes
        """
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        return available_mb

    def calculate_batch_size(self) -> int:
        """Calculate optimal batch size based on available RAM.

        Returns:
            Optimal batch size between min_batch_size and max_batch_size
        """
        available_mb = self.get_available_memory_mb()

        # If we have plenty of memory, use max batch size
        if available_mb > 2 * self.memory_threshold_mb:
            return self.max_batch_size

        # If memory is tight, use min batch size
        if available_mb < self.memory_threshold_mb:
            return self.min_batch_size

        # Scale linearly between min and max based on available memory
        ratio = (available_mb - self.memory_threshold_mb) / self.memory_threshold_mb
        batch_size = int(self.min_batch_size + ratio * (self.max_batch_size - self.min_batch_size))

        # Clamp to min/max bounds
        return max(self.min_batch_size, min(self.max_batch_size, batch_size))

    def embed_texts(self, texts: Union[str, List[str]], batch_size: int = None) -> List[List[float]]:
        """Generate embeddings for one or more texts.

        Args:
            texts: Single text string or list of text strings
            batch_size: Optional batch size override. If None, uses adaptive sizing.

        Returns:
            List of embedding vectors (each vector is 768 dimensions)
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Try cache lookup if enabled
        if self.use_cache and self.cache is not None:
            cached_embeddings, miss_indices = self.cache.get_batch(texts)

            # If all embeddings are cached, return immediately
            if not miss_indices:
                logger.info(f"All {len(texts)} embeddings retrieved from cache (100% hit rate)")
                return [emb for emb in cached_embeddings if emb is not None]

            # Log cache performance
            hit_count = len(texts) - len(miss_indices)
            hit_rate = (hit_count / len(texts)) * 100
            logger.info(
                f"Cache: {hit_count}/{len(texts)} hits ({hit_rate:.1f}%), "
                f"computing {len(miss_indices)} new embeddings"
            )

            # Only compute embeddings for cache misses
            texts_to_compute = [texts[i] for i in miss_indices]
        else:
            # No cache, compute all embeddings
            texts_to_compute = texts
            miss_indices = list(range(len(texts)))
            cached_embeddings = [None] * len(texts)

        # Determine batch size
        if batch_size is None:
            batch_size = self.calculate_batch_size()

        logger.info(f"Generating embeddings for {len(texts_to_compute)} texts with batch size {batch_size}")

        new_embeddings = []
        total_batches = (len(texts_to_compute) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(texts_to_compute), batch_size):
            batch_texts = texts_to_compute[batch_idx:batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1

            logger.debug(f"Processing batch {current_batch_num}/{total_batches} ({len(batch_texts)} texts)")

            # Generate embeddings for this batch
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )

            # Convert numpy arrays to lists
            new_embeddings.extend([embedding.tolist() for embedding in batch_embeddings])

            # Memory management and throttling after each batch
            if current_batch_num < total_batches:
                gc.collect()

                # Log memory usage
                available_mb = self.get_available_memory_mb()
                logger.debug(f"Available memory after batch {current_batch_num}: {available_mb:.1f} MB")

                # Throttle to prevent 100% CPU usage
                if self.throttle_seconds > 0:
                    time.sleep(self.throttle_seconds)

        # Store new embeddings in cache
        if self.use_cache and self.cache is not None:
            self.cache.put_batch(texts_to_compute, new_embeddings)
            logger.debug(f"Cached {len(new_embeddings)} new embeddings")

        # Merge cached and newly computed embeddings
        final_embeddings = []
        new_embedding_iter = iter(new_embeddings)

        for i in range(len(texts)):
            if cached_embeddings[i] is not None:
                # Use cached embedding
                final_embeddings.append(cached_embeddings[i])
            else:
                # Use newly computed embedding
                final_embeddings.append(next(new_embedding_iter))

        logger.info(f"Generated {len(final_embeddings)} embeddings successfully")
        return final_embeddings

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (768 dimensions)
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service.

        Returns:
            Embedding dimension (768 for nomic-embed-text-v1.5)
        """
        return self.embedding_dimension

    def unload_model(self) -> None:
        """Unload the model from memory to free resources."""
        if self.model is not None:
            logger.info("Unloading model from memory")
            self.model = None
            gc.collect()
            logger.info("Model unloaded successfully")

    def flush_cache(self) -> None:
        """Flush embedding cache to disk."""
        if self.cache is not None:
            self.cache.flush()

    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics.

        Returns:
            Dictionary with cache statistics, or empty dict if cache disabled
        """
        if self.cache is not None:
            return self.cache.get_stats().to_dict()
        return {}
