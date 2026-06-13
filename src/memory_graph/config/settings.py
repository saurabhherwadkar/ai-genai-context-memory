"""Application settings module using Pydantic Settings with YAML configuration."""

import os  # Operating system interface for environment variables
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Any  # Type hint for generic dictionary values

import yaml  # YAML configuration file parser
from pydantic import BaseModel, Field  # Data validation base class and field metadata
from pydantic_settings import BaseSettings  # Settings management with env var support


# Determine the project root directory for config file resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Navigate up to project root
CONFIG_DIR = PROJECT_ROOT / "config"  # Path to configuration directory


class ServerSettings(BaseModel):
    """HTTP server configuration settings."""

    host: str = Field(default="0.0.0.0", description="Server bind address")  # Network interface to bind
    port: int = Field(default=8000, description="Server listen port")  # TCP port number
    workers: int = Field(default=4, description="Number of worker processes")  # Uvicorn worker count


class DatabaseSettings(BaseModel):
    """SQLite database configuration settings."""

    path: str = Field(default="./data/memory_graph.db", description="Database file path")  # SQLite file location
    journal_mode: str = Field(default="WAL", description="SQLite journal mode")  # Write-Ahead Logging for concurrency
    busy_timeout_ms: int = Field(default=5000, description="Busy timeout in milliseconds")  # Wait time for locked DB


class GraphSettings(BaseModel):
    """NetworkX graph configuration settings."""

    max_nodes: int = Field(default=10000, description="Maximum nodes in graph")  # Upper bound for memory consumption
    max_edges_per_node: int = Field(default=50, description="Maximum edges per node")  # Limit edge fan-out
    sync_interval_seconds: int = Field(default=300, description="Graph sync interval")  # Periodic consistency check


class EmbeddingSettings(BaseModel):
    """Sentence transformer embedding configuration."""

    model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")  # HuggingFace model ID
    batch_size: int = Field(default=32, description="Batch size for encoding")  # Texts processed per batch
    cache_size: int = Field(default=1000, description="LRU cache size")  # Maximum cached embeddings
    similarity_threshold: float = Field(default=0.92, description="Duplicate detection threshold")  # Cosine cutoff


class ExtractionSettings(BaseModel):
    """Memory extraction pipeline configuration."""

    enabled: bool = Field(default=True, description="Enable extraction pipeline")  # Toggle extraction on/off
    concurrency: int = Field(default=2, description="Max concurrent extraction tasks")  # Parallel extraction workers
    retry_attempts: int = Field(default=3, description="Retry count on failure")  # Number of retry attempts
    retry_delay_seconds: int = Field(default=5, description="Delay between retries")  # Backoff delay in seconds
    provider: str = Field(default="openai", description="LLM provider for extraction")  # Provider to use
    model: str = Field(default="gpt-4o-mini", description="Model for extraction")  # Specific model identifier
    max_memories_per_conversation: int = Field(default=5, description="Max memories per conversation")  # Limit extraction


class RankingWeights(BaseModel):
    """Weights for memory ranking algorithm."""

    semantic_similarity: float = Field(default=0.35, description="Semantic similarity weight")  # Embedding closeness
    recency: float = Field(default=0.20, description="Recency weight")  # Time-based decay factor
    access_frequency: float = Field(default=0.10, description="Access frequency weight")  # Usage popularity
    confidence: float = Field(default=0.15, description="Confidence weight")  # Certainty score factor
    intensity: float = Field(default=0.10, description="Intensity weight")  # Memory strength factor
    graph_centrality: float = Field(default=0.10, description="Graph centrality weight")  # Network importance


class ContextSettings(BaseModel):
    """Context building and injection configuration."""

    max_token_budget: int = Field(default=2000, description="Max tokens for context")  # Token injection limit
    format: str = Field(default="markdown", description="Output format")  # Format: markdown, xml, plain
    max_memories_injected: int = Field(default=20, description="Max memories in context")  # Hard cap on injected items
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)  # Scoring weight configuration


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    base_url: str = Field(description="Provider API base URL")  # HTTP endpoint for the provider
    api_key_env: str = Field(description="Environment variable name for API key")  # Env var holding the secret


class ProxySettings(BaseModel):
    """LLM proxy configuration settings."""

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)  # Named provider configurations
    timeout_seconds: int = Field(default=120, description="Request timeout")  # HTTP timeout for upstream calls
    max_request_size_bytes: int = Field(default=1048576, description="Max request body size")  # 1MB default limit


class RateLimitSettings(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(default=60, description="Requests per minute limit")  # Sustained rate cap
    burst_size: int = Field(default=10, description="Burst size allowance")  # Short burst tolerance


class InputSettings(BaseModel):
    """Input validation configuration."""

    max_content_length: int = Field(default=50000, description="Max content character length")  # Content size limit
    max_tags: int = Field(default=20, description="Max tags per memory")  # Tag count limit


class SecuritySettings(BaseModel):
    """Security configuration combining rate limits and input validation."""

    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)  # Rate limiting rules
    input: InputSettings = Field(default_factory=InputSettings)  # Input validation rules


class LoggingSettings(BaseModel):
    """Logging configuration settings."""

    level: str = Field(default="INFO", description="Log level")  # Minimum log severity
    format: str = Field(default="json", description="Log format")  # Output format: json or text
    file: str = Field(default="./logs/memory_graph.log", description="Log file path")  # Log output file
    max_file_size_mb: int = Field(default=100, description="Max log file size in MB")  # Rotation threshold
    backup_count: int = Field(default=5, description="Number of backup log files")  # Rotated file retention


class AppSettings(BaseSettings):
    """Root application settings aggregating all configuration sections."""

    server: ServerSettings = Field(default_factory=ServerSettings)  # Server configuration section
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # Database configuration section
    graph: GraphSettings = Field(default_factory=GraphSettings)  # Graph configuration section
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)  # Embeddings configuration section
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)  # Extraction configuration section
    context: ContextSettings = Field(default_factory=ContextSettings)  # Context configuration section
    proxy: ProxySettings = Field(default_factory=ProxySettings)  # Proxy configuration section
    security: SecuritySettings = Field(default_factory=SecuritySettings)  # Security configuration section
    logging: LoggingSettings = Field(default_factory=LoggingSettings)  # Logging configuration section

    model_config = {"env_prefix": "MEMORY_GRAPH__", "env_nested_delimiter": "__"}  # Environment variable mapping


def _load_yaml_config(env_name: str) -> dict[str, Any]:
    """Load and merge YAML configuration files for the given environment.

    Loads the base application.yaml and merges environment-specific overrides on top.
    """
    merged_config: dict[str, Any] = {}  # Initialize empty merged configuration

    # Load the base configuration file
    base_config_path = CONFIG_DIR / "application.yaml"  # Path to default config
    if base_config_path.exists():  # Check if base config file exists
        with open(base_config_path, "r", encoding="utf-8") as config_file:  # Open with explicit encoding
            base_data = yaml.safe_load(config_file)  # Parse YAML safely to prevent code execution
            if base_data:  # Guard against empty YAML files
                merged_config = base_data  # Start with base configuration

    # Load environment-specific override file
    env_config_path = CONFIG_DIR / f"application-{env_name}.yaml"  # Build env-specific path
    if env_config_path.exists():  # Check if override file exists
        with open(env_config_path, "r", encoding="utf-8") as config_file:  # Open with explicit encoding
            env_data = yaml.safe_load(config_file)  # Parse environment overrides
            if env_data:  # Guard against empty override files
                merged_config = _deep_merge(merged_config, env_data)  # Deep merge overrides onto base

    return merged_config  # Return the fully merged configuration


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Nested dictionaries are merged; scalar values from override replace base values.
    """
    result = base.copy()  # Create a copy to avoid mutating the original base dict
    for key, value in override.items():  # Iterate over all override entries
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):  # Both are dicts
            result[key] = _deep_merge(result[key], value)  # Recursively merge nested dictionaries
        else:  # Scalar value or new key
            result[key] = value  # Override replaces base value directly
    return result  # Return the merged dictionary


def load_settings() -> AppSettings:
    """Load application settings from YAML config and environment variables.

    Resolution order: base YAML -> env YAML override -> environment variables.
    """
    env_name = os.getenv("APP_ENV", "dev")  # Read environment name, default to development
    yaml_config = _load_yaml_config(env_name)  # Load merged YAML configuration
    settings = AppSettings(**yaml_config)  # Construct settings with YAML values and env var overrides
    return settings  # Return the fully resolved settings instance


# Module-level singleton settings instance for application-wide access
_settings_instance: AppSettings | None = None  # Cached settings singleton


def get_settings() -> AppSettings:
    """Get the singleton application settings instance.

    Lazily initializes settings on first access and caches for subsequent calls.
    """
    global _settings_instance  # Reference the module-level singleton variable
    if _settings_instance is None:  # First access, need to initialize
        _settings_instance = load_settings()  # Load and cache the settings
    return _settings_instance  # Return the cached instance


def reset_settings() -> None:
    """Reset the settings singleton, forcing reload on next access.

    Primarily used in testing to reload configuration between test cases.
    """
    global _settings_instance  # Reference the module-level singleton variable
    _settings_instance = None  # Clear the cached instance to force reload
