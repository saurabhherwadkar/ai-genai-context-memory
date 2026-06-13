"""Registry for managing LLM provider adapters."""

import structlog  # Structured logging

from memory_graph.config.settings import ProxySettings  # Proxy configuration
from memory_graph.proxy.providers.anthropic_provider import AnthropicProvider  # Anthropic adapter
from memory_graph.proxy.providers.base_provider import BaseProvider  # Abstract interface
from memory_graph.proxy.providers.generic_provider import GenericProvider  # Generic adapter
from memory_graph.proxy.providers.openai_provider import OpenAIProvider  # OpenAI adapter
from memory_graph.security.secrets_manager import SecretsManager  # API key loading

# Module logger for provider registry operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class ProviderRegistry:
    """Manages registration and lookup of LLM provider adapters.

    Initializes provider adapters from configuration and provides
    name-based lookup for the proxy service.
    """

    def __init__(self, settings: ProxySettings, secrets_manager: SecretsManager) -> None:
        """Initialize the registry and register configured providers."""
        self._providers: dict[str, BaseProvider] = {}  # Name -> provider mapping
        self._settings = settings  # Proxy configuration
        self._secrets = secrets_manager  # Secrets access
        self._register_configured_providers()  # Auto-register from config
        logger.info("provider_registry_initialized", count=len(self._providers))  # Log init

    def get_provider(self, name: str) -> BaseProvider | None:
        """Look up a registered provider by name.

        Returns None if no provider is registered with the given name.
        """
        provider = self._providers.get(name)  # Look up by name
        if provider is None:  # Not found
            logger.warning("provider_not_found", name=name)  # Log miss
        return provider  # Return provider or None

    def register_provider(self, name: str, provider: BaseProvider) -> None:
        """Register a new provider adapter under the given name."""
        self._providers[name] = provider  # Store in registry
        logger.info("provider_registered", name=name)  # Log registration

    def list_providers(self) -> list[str]:
        """Get a list of all registered provider names."""
        return list(self._providers.keys())  # Return name list

    def has_provider(self, name: str) -> bool:
        """Check whether a provider is registered with the given name."""
        return name in self._providers  # Check registry

    def _register_configured_providers(self) -> None:
        """Register all providers defined in the configuration."""
        for provider_name, config in self._settings.providers.items():  # Iterate config entries
            api_key = self._secrets.get_api_key(config.api_key_env)  # Load API key

            if api_key is None:  # API key not available
                logger.warning(  # Log missing key
                    "provider_skipped_no_api_key",
                    provider=provider_name,
                    env_var=config.api_key_env,
                )
                continue  # Skip this provider

            # Create the appropriate adapter based on provider name
            provider = self._create_provider(  # Factory method
                name=provider_name,
                base_url=config.base_url,
                api_key=api_key,
                timeout=self._settings.timeout_seconds,
            )

            if provider:  # Successfully created
                self._providers[provider_name] = provider  # Register it
                logger.info("provider_auto_registered", name=provider_name)  # Log success

    def _create_provider(
        self,
        name: str,
        base_url: str,
        api_key: str,
        timeout: int,
    ) -> BaseProvider | None:
        """Create a provider adapter instance based on the provider name."""
        if name == "openai":  # OpenAI provider
            return OpenAIProvider(base_url=base_url, api_key=api_key, timeout=timeout)

        if name == "anthropic":  # Anthropic provider
            return AnthropicProvider(base_url=base_url, api_key=api_key, timeout=timeout)

        # Default to generic OpenAI-compatible provider
        return GenericProvider(name=name, base_url=base_url, api_key=api_key, timeout=timeout)
