"""Configuration management for Smart Fork MCP Server.

This module provides configuration loading, validation, and default values
for the Smart Fork system.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    dimension: int = 768
    batch_size: int = 32
    max_batch_size: int = 128
    min_batch_size: int = 8
    throttle_seconds: float = 0.1  # Sleep between batches to reduce CPU usage
    use_mps: bool = True  # Use Metal acceleration on Apple Silicon


@dataclass
class SearchConfig:
    """Configuration for search parameters."""
    k_chunks: int = 200
    top_n_sessions: int = 5
    preview_length: int = 200
    similarity_threshold: float = 0.3
    recency_weight: float = 0.25


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    embedding_cache_size: int = 100
    embedding_ttl_seconds: float = 300.0  # 5 minutes
    result_cache_size: int = 50
    result_ttl_seconds: float = 300.0  # 5 minutes


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    target_tokens: int = 750
    overlap_tokens: int = 150
    max_tokens: int = 1000


@dataclass
class IndexingConfig:
    """Configuration for background indexing."""
    debounce_delay: float = 5.0
    checkpoint_interval: int = 15
    enabled: bool = True


@dataclass
class ArchiveConfig:
    """Configuration for session archiving."""
    archive_threshold_days: int = 365
    enabled: bool = True


@dataclass
class ServerConfig:
    """Configuration for REST API server."""
    host: str = "127.0.0.1"
    port: int = 8741


@dataclass
class MemoryConfig:
    """Configuration for memory limits."""
    max_memory_mb: int = 2000
    gc_between_batches: bool = True


@dataclass
class Config:
    """Main configuration container."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    storage_dir: str = "~/.smart-fork"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "embedding": asdict(self.embedding),
            "search": asdict(self.search),
            "cache": asdict(self.cache),
            "chunking": asdict(self.chunking),
            "indexing": asdict(self.indexing),
            "server": asdict(self.server),
            "memory": asdict(self.memory),
            "archive": asdict(self.archive),
            "storage_dir": self.storage_dir
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = Config()

        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        if "search" in data:
            config.search = SearchConfig(**data["search"])
        if "cache" in data:
            config.cache = CacheConfig(**data["cache"])
        if "chunking" in data:
            config.chunking = ChunkingConfig(**data["chunking"])
        if "indexing" in data:
            config.indexing = IndexingConfig(**data["indexing"])
        if "server" in data:
            config.server = ServerConfig(**data["server"])
        if "memory" in data:
            config.memory = MemoryConfig(**data["memory"])
        if "archive" in data:
            config.archive = ArchiveConfig(**data["archive"])
        if "storage_dir" in data:
            config.storage_dir = data["storage_dir"]

        return config


class ConfigManager:
    """Manages configuration loading, saving, and validation."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize ConfigManager.

        Args:
            config_path: Path to config file. If None, uses ~/.smart-fork/config.json
        """
        if config_path is None:
            storage_dir = Path.home() / ".smart-fork"
            config_path = storage_dir / "config.json"

        self.config_path = Path(config_path)
        self._config: Optional[Config] = None
        self._lock = threading.RLock()  # RLock to allow reentrant calls (get -> load)

    def load(self) -> Config:
        """Load configuration from file or create default.

        Returns:
            Config object
        """
        with self._lock:
            if not self.config_path.exists():
                logger.info(f"Config file not found at {self.config_path}, using defaults")
                self._config = Config()
                return self._config

            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self._config = Config.from_dict(data)
                logger.info(f"Loaded config from {self.config_path}")
                return self._config

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in config file: {e}")
                logger.warning("Using default configuration")
                self._config = Config()
                return self._config

            except Exception as e:
                logger.error(f"Error loading config: {e}")
                logger.warning("Using default configuration")
                self._config = Config()
                return self._config

    def save(self, config: Optional[Config] = None) -> None:
        """Save configuration to file.

        Args:
            config: Config object to save. If None, saves current config.
        """
        with self._lock:
            if config is not None:
                self._config = config

            if self._config is None:
                raise ValueError("No configuration to save")

            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file then rename for atomicity
            temp_path = self.config_path.with_suffix('.json.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config.to_dict(), f, indent=2)

                temp_path.replace(self.config_path)
                logger.info(f"Saved config to {self.config_path}")

            except Exception as e:
                logger.error(f"Error saving config: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                raise

    def get(self) -> Config:
        """Get current configuration.

        Returns:
            Config object
        """
        with self._lock:
            if self._config is None:
                return self.load()
            return self._config

    def update(self, **kwargs) -> None:
        """Update configuration values.

        Args:
            **kwargs: Configuration values to update
        """
        with self._lock:
            if self._config is None:
                self._config = self.load()

            # Update nested configs
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    if isinstance(value, dict):
                        # Update nested config object
                        nested_config = getattr(self._config, key)
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(nested_config, nested_key, nested_value)
                    else:
                        setattr(self._config, key, value)

    def validate(self) -> bool:
        """Validate current configuration.

        Returns:
            True if valid, False otherwise
        """
        with self._lock:
            if self._config is None:
                self._config = self.load()

            try:
                # Validate embedding config
                if self._config.embedding.dimension <= 0:
                    logger.error("Invalid embedding dimension")
                    return False
                if self._config.embedding.batch_size <= 0:
                    logger.error("Invalid batch size")
                    return False
                if self._config.embedding.min_batch_size > self._config.embedding.max_batch_size:
                    logger.error("min_batch_size cannot be greater than max_batch_size")
                    return False

                # Validate search config
                if self._config.search.k_chunks <= 0:
                    logger.error("Invalid k_chunks")
                    return False
                if self._config.search.top_n_sessions <= 0:
                    logger.error("Invalid top_n_sessions")
                    return False
                if not 0.0 <= self._config.search.similarity_threshold <= 1.0:
                    logger.error("similarity_threshold must be between 0 and 1")
                    return False
                if not 0.0 <= self._config.search.recency_weight <= 1.0:
                    logger.error("recency_weight must be between 0 and 1")
                    return False

                # Validate cache config
                if self._config.cache.embedding_cache_size <= 0:
                    logger.error("Invalid embedding_cache_size")
                    return False
                if self._config.cache.result_cache_size <= 0:
                    logger.error("Invalid result_cache_size")
                    return False
                if self._config.cache.embedding_ttl_seconds <= 0:
                    logger.error("Invalid embedding_ttl_seconds")
                    return False
                if self._config.cache.result_ttl_seconds <= 0:
                    logger.error("Invalid result_ttl_seconds")
                    return False

                # Validate chunking config
                if self._config.chunking.target_tokens <= 0:
                    logger.error("Invalid target_tokens")
                    return False
                if self._config.chunking.overlap_tokens < 0:
                    logger.error("Invalid overlap_tokens")
                    return False
                if self._config.chunking.max_tokens <= 0:
                    logger.error("Invalid max_tokens")
                    return False
                if self._config.chunking.target_tokens > self._config.chunking.max_tokens:
                    logger.error("target_tokens cannot be greater than max_tokens")
                    return False

                # Validate indexing config
                if self._config.indexing.debounce_delay < 0:
                    logger.error("Invalid debounce_delay")
                    return False
                if self._config.indexing.checkpoint_interval <= 0:
                    logger.error("Invalid checkpoint_interval")
                    return False

                # Validate server config
                if not 1024 <= self._config.server.port <= 65535:
                    logger.error("Port must be between 1024 and 65535")
                    return False

                # Validate memory config
                if self._config.memory.max_memory_mb <= 0:
                    logger.error("Invalid max_memory_mb")
                    return False

                # Validate archive config
                if self._config.archive.archive_threshold_days <= 0:
                    logger.error("Invalid archive_threshold_days")
                    return False

                return True

            except Exception as e:
                logger.error(f"Validation error: {e}")
                return False

    def reset(self) -> None:
        """Reset configuration to defaults."""
        with self._lock:
            self._config = Config()
            logger.info("Reset configuration to defaults")


def load_config(config_path: Optional[str] = None) -> Config:
    """Convenience function to load configuration.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Config object
    """
    manager = ConfigManager(config_path)
    return manager.load()


def save_config(config: Config, config_path: Optional[str] = None) -> None:
    """Convenience function to save configuration.

    Args:
        config: Config object to save
        config_path: Path to config file. If None, uses default location.
    """
    manager = ConfigManager(config_path)
    manager.save(config)
