"""Tests for EmbeddingService."""

import gc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smart_fork.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        service = EmbeddingService(use_cache=False)
        assert service.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert service.min_batch_size == 8
        assert service.max_batch_size == 128
        assert service.memory_threshold_mb == 500
        assert service.embedding_dimension == 768
        assert service.model is None
        assert service.use_cache is False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        service = EmbeddingService(
            model_name="custom/model",
            min_batch_size=4,
            max_batch_size=64,
            memory_threshold_mb=1000,
            use_cache=False,
        )
        assert service.model_name == "custom/model"
        assert service.min_batch_size == 4
        assert service.max_batch_size == 64
        assert service.memory_threshold_mb == 1000

    @patch("psutil.virtual_memory")
    def test_get_available_memory_mb(self, mock_memory):
        """Test getting available memory in MB."""
        # Mock 4GB available
        mock_memory.return_value = MagicMock(available=4 * 1024 * 1024 * 1024)

        service = EmbeddingService(use_cache=False)
        available = service.get_available_memory_mb()

        assert available == 4096.0

    @patch("psutil.virtual_memory")
    def test_calculate_batch_size_plenty_memory(self, mock_memory):
        """Test batch size calculation with plenty of memory."""
        # Mock 2GB available (more than 2x threshold of 500MB)
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)

        service = EmbeddingService(use_cache=False)
        batch_size = service.calculate_batch_size()

        assert batch_size == service.max_batch_size

    @patch("psutil.virtual_memory")
    def test_calculate_batch_size_low_memory(self, mock_memory):
        """Test batch size calculation with low memory."""
        # Mock 300MB available (less than threshold of 500MB)
        mock_memory.return_value = MagicMock(available=300 * 1024 * 1024)

        service = EmbeddingService(use_cache=False)
        batch_size = service.calculate_batch_size()

        assert batch_size == service.min_batch_size

    @patch("psutil.virtual_memory")
    def test_calculate_batch_size_medium_memory(self, mock_memory):
        """Test batch size calculation with medium memory."""
        # Mock 750MB available (1.5x threshold)
        mock_memory.return_value = MagicMock(available=750 * 1024 * 1024)

        service = EmbeddingService(use_cache=False)
        batch_size = service.calculate_batch_size()

        # Should be between min and max
        assert service.min_batch_size < batch_size < service.max_batch_size

    @patch("smart_fork.embedding_service.torch")
    @patch("smart_fork.embedding_service.SentenceTransformer")
    def test_load_model(self, mock_transformer, mock_torch):
        """Test model loading."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        # Mock torch to simulate no GPU available
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        service = EmbeddingService(use_cache=False)
        service.load_model()

        assert service.model == mock_model
        mock_transformer.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="cpu"
        )

    @patch("smart_fork.embedding_service.torch")
    @patch("smart_fork.embedding_service.SentenceTransformer")
    def test_load_model_with_mps(self, mock_transformer, mock_torch):
        """Test model loading with MPS (Metal) acceleration."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        # Mock torch to simulate MPS available
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False

        service = EmbeddingService(use_cache=False, use_mps=True)
        service.load_model()

        assert service.device == "mps"
        mock_transformer.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="mps"
        )

    @patch("smart_fork.embedding_service.torch")
    @patch("smart_fork.embedding_service.SentenceTransformer")
    def test_load_model_with_cuda(self, mock_transformer, mock_torch):
        """Test model loading with CUDA acceleration."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        # Mock torch to simulate CUDA available but not MPS
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True

        service = EmbeddingService(use_cache=False)
        service.load_model()

        assert service.device == "cuda"
        mock_transformer.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="cuda"
        )

    @patch("smart_fork.embedding_service.torch")
    @patch("smart_fork.embedding_service.SentenceTransformer")
    def test_load_model_mps_disabled(self, mock_transformer, mock_torch):
        """Test model loading with MPS explicitly disabled."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        # Mock torch to simulate MPS available
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False

        # Explicitly disable MPS
        service = EmbeddingService(use_cache=False, use_mps=False)
        service.load_model()

        # Should fall back to CPU since MPS is disabled
        assert service.device == "cpu"
        mock_transformer.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="cpu"
        )

    @patch("smart_fork.embedding_service.SentenceTransformer")
    def test_load_model_only_once(self, mock_transformer):
        """Test that model is only loaded once."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        service = EmbeddingService(use_cache=False)
        service.load_model()
        service.load_model()  # Second call

        # Should only be called once
        assert mock_transformer.call_count == 1

    @patch("smart_fork.embedding_service.SentenceTransformer")
    def test_load_model_error(self, mock_transformer):
        """Test model loading error handling."""
        mock_transformer.side_effect = Exception("Model not found")

        service = EmbeddingService(use_cache=False)
        with pytest.raises(Exception, match="Model not found"):
            service.load_model()

    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("psutil.virtual_memory")
    def test_embed_texts_single_string(self, mock_memory, mock_transformer):
        """Test embedding a single string."""
        # Setup mocks
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_transformer.return_value = mock_model

        service = EmbeddingService(use_cache=False)
        embeddings = service.embed_texts("test text")

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        mock_model.encode.assert_called_once()

    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("psutil.virtual_memory")
    def test_embed_texts_list(self, mock_memory, mock_transformer):
        """Test embedding a list of texts."""
        # Setup mocks
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768, [0.3] * 768])
        mock_transformer.return_value = mock_model

        service = EmbeddingService(use_cache=False)
        texts = ["text 1", "text 2", "text 3"]
        embeddings = service.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)

    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("psutil.virtual_memory")
    def test_embed_texts_empty_list(self, mock_memory, mock_transformer):
        """Test embedding empty list."""
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        service = EmbeddingService(use_cache=False)
        embeddings = service.embed_texts([])

        assert embeddings == []
        mock_model.encode.assert_not_called()

    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("psutil.virtual_memory")
    @patch("gc.collect")
    def test_embed_texts_batching(self, mock_gc, mock_memory, mock_transformer):
        """Test batching with multiple batches."""
        # Setup mocks
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)
        mock_model = MagicMock()

        # Return different embeddings for each batch
        def encode_side_effect(texts, **kwargs):
            return np.array([[0.1] * 768 for _ in texts])

        mock_model.encode.side_effect = encode_side_effect
        mock_transformer.return_value = mock_model

        service = EmbeddingService(use_cache=False)
        # Create 20 texts with batch size of 8
        texts = [f"text {i}" for i in range(20)]
        embeddings = service.embed_texts(texts, batch_size=8)

        assert len(embeddings) == 20
        # Should have 3 batches (8 + 8 + 4)
        assert mock_model.encode.call_count == 3
        # Garbage collection should be called between batches (2 times)
        assert mock_gc.call_count >= 2

    @patch("smart_fork.embedding_service.time.sleep")
    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("psutil.virtual_memory")
    @patch("gc.collect")
    def test_embed_texts_throttling(self, mock_gc, mock_memory, mock_transformer, mock_sleep):
        """Test that throttling sleeps between batches."""
        # Setup mocks
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)
        mock_model = MagicMock()

        def encode_side_effect(texts, **kwargs):
            return np.array([[0.1] * 768 for _ in texts])

        mock_model.encode.side_effect = encode_side_effect
        mock_transformer.return_value = mock_model

        # Use throttle_seconds=0.2 for testing
        service = EmbeddingService(use_cache=False, throttle_seconds=0.2)
        # Create 20 texts with batch size of 8 (3 batches)
        texts = [f"text {i}" for i in range(20)]
        embeddings = service.embed_texts(texts, batch_size=8)

        assert len(embeddings) == 20
        # Should have 3 batches (8 + 8 + 4)
        assert mock_model.encode.call_count == 3
        # Sleep should be called between batches (2 times, not after the last batch)
        assert mock_sleep.call_count == 2
        # Each sleep call should use the configured throttle_seconds
        mock_sleep.assert_called_with(0.2)

    @patch("smart_fork.embedding_service.time.sleep")
    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("psutil.virtual_memory")
    @patch("gc.collect")
    def test_embed_texts_no_throttling_when_zero(self, mock_gc, mock_memory, mock_transformer, mock_sleep):
        """Test that no throttling occurs when throttle_seconds=0."""
        # Setup mocks
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)
        mock_model = MagicMock()

        def encode_side_effect(texts, **kwargs):
            return np.array([[0.1] * 768 for _ in texts])

        mock_model.encode.side_effect = encode_side_effect
        mock_transformer.return_value = mock_model

        # Disable throttling
        service = EmbeddingService(use_cache=False, throttle_seconds=0)
        texts = [f"text {i}" for i in range(20)]
        embeddings = service.embed_texts(texts, batch_size=8)

        assert len(embeddings) == 20
        # Sleep should NOT be called when throttle_seconds=0
        mock_sleep.assert_not_called()

    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("psutil.virtual_memory")
    def test_embed_texts_custom_batch_size(self, mock_memory, mock_transformer):
        """Test embedding with custom batch size."""
        # Setup mocks
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 768 for _ in range(5)])
        mock_transformer.return_value = mock_model

        service = EmbeddingService(use_cache=False)
        texts = [f"text {i}" for i in range(5)]
        embeddings = service.embed_texts(texts, batch_size=5)

        assert len(embeddings) == 5
        # Should process in single batch
        assert mock_model.encode.call_count == 1

    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("psutil.virtual_memory")
    def test_embed_single(self, mock_memory, mock_transformer):
        """Test embedding a single text with convenience method."""
        # Setup mocks
        mock_memory.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5] * 768])
        mock_transformer.return_value = mock_model

        service = EmbeddingService(use_cache=False)
        embedding = service.embed_single("test text")

        assert len(embedding) == 768
        assert embedding[0] == 0.5

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        service = EmbeddingService(use_cache=False)
        assert service.get_embedding_dimension() == 768

    @patch("smart_fork.embedding_service.SentenceTransformer")
    @patch("gc.collect")
    def test_unload_model(self, mock_gc, mock_transformer):
        """Test unloading model from memory."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        service = EmbeddingService(use_cache=False)
        service.load_model()
        assert service.model is not None

        service.unload_model()
        assert service.model is None
        mock_gc.assert_called_once()

    def test_unload_model_when_not_loaded(self):
        """Test unloading when model is not loaded."""
        service = EmbeddingService(use_cache=False)
        service.unload_model()

        # Model should remain None, no error raised
        assert service.model is None


class TestEmbeddingServiceIntegration:
    """Integration tests for EmbeddingService (requires actual model download)."""

    @pytest.mark.skip(reason="Requires model download - run manually for integration testing")
    def test_real_embedding_generation(self):
        """Test real embedding generation with actual model."""
        service = EmbeddingService(use_cache=False)
        service.load_model()

        text = "This is a test sentence for embedding generation."
        embedding = service.embed_single(text)

        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

        service.unload_model()

    @pytest.mark.skip(reason="Requires model download - run manually for integration testing")
    def test_real_batch_embedding(self):
        """Test real batch embedding with actual model."""
        service = EmbeddingService(use_cache=False)
        service.load_model()

        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]
        embeddings = service.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
        # Check that embeddings are different
        assert embeddings[0] != embeddings[1]

        service.unload_model()

    @pytest.mark.skip(reason="Requires model download - run manually for integration testing")
    def test_real_semantic_similarity(self):
        """Test that semantically similar texts have similar embeddings."""
        service = EmbeddingService(use_cache=False)
        service.load_model()

        texts = [
            "The cat sat on the mat.",
            "A feline rested on the rug.",  # Similar meaning
            "Python is a programming language.",  # Different meaning
        ]
        embeddings = service.embed_texts(texts)

        # Calculate cosine similarity (embeddings are already normalized)
        def cosine_similarity(a, b):
            return sum(x * y for x, y in zip(a, b))

        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])

        # Similar sentences should have higher similarity
        assert sim_01 > sim_02

        service.unload_model()
