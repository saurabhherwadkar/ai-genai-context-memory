"""Secrets management for loading API keys from environment variables."""

import os  # Operating system interface for environment variable access

import structlog  # Structured logging

# Module logger for secrets operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class SecretsManager:
    """Manages loading and access of sensitive configuration from environment variables.

    Provides a centralized access point for API keys and other secrets
    without hardcoding values or exposing them in configuration files.
    """

    def __init__(self) -> None:
        """Initialize the secrets manager."""
        self._secrets_cache: dict[str, str] = {}  # Cache loaded secrets in memory
        logger.info("secrets_manager_initialized")  # Log initialization

    def get_api_key(self, env_var_name: str) -> str | None:
        """Retrieve an API key from the specified environment variable.

        Returns None if the environment variable is not set.
        Caches the value after first load for performance.
        """
        # Check cache first to avoid repeated os.getenv calls
        if env_var_name in self._secrets_cache:  # Already loaded
            return self._secrets_cache[env_var_name]  # Return cached value

        # Load from environment
        value = os.getenv(env_var_name)  # Read environment variable

        if value is None:  # Environment variable not set
            logger.warning("api_key_not_found", env_var=env_var_name)  # Log missing key
            return None  # Return None to indicate not configured

        if not value.strip():  # Environment variable is empty string
            logger.warning("api_key_empty", env_var=env_var_name)  # Log empty key
            return None  # Treat empty as not set

        # Cache the loaded secret
        self._secrets_cache[env_var_name] = value.strip()  # Store trimmed value
        logger.debug("api_key_loaded", env_var=env_var_name)  # Log successful load (no value!)
        return self._secrets_cache[env_var_name]  # Return the secret value

    def has_api_key(self, env_var_name: str) -> bool:
        """Check whether an API key is available without returning its value."""
        return self.get_api_key(env_var_name) is not None  # Delegate to get method

    def clear_cache(self) -> None:
        """Clear all cached secrets from memory.

        Useful for forcing reload or for security-sensitive cleanup.
        """
        self._secrets_cache.clear()  # Remove all cached entries
        logger.info("secrets_cache_cleared")  # Log cache clear
