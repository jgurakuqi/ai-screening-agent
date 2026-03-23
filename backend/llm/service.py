"""LLM service with provider-chain fallback.

Wraps one or more :class:`LLMProvider` instances and tries each in order,
returning a sentinel empty string when all providers are exhausted.
"""

from backend.logging_config import logger
from backend.llm.base import LLMProvider, LLMProviderExhausted

# Sentinel returned when all providers are exhausted
LLM_BUSY_RESPONSE = ""


class LLMService:
    """Coordinate LLM providers with ordered fallback behavior."""

    def __init__(self) -> None:
        self.providers: list[LLMProvider] = []

    def add_provider(self, provider: LLMProvider, priority: int | None = None) -> None:
        """Register an LLM provider in the fallback chain.

        Args:
            provider: Provider instance to add.
            priority: Zero-based insertion index. When omitted, the provider is
                appended to the end of the fallback order.
        """
        if priority is None:
            self.providers.append(provider)
        else:
            self.providers.insert(priority, provider)

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str | dict:
        """Generate a completion using the first provider that succeeds.

        Args:
            messages: Chat-completion message payload.
            temperature: Sampling temperature shared across providers.
            max_tokens: Upper bound for generated completion tokens.
            tools: Optional tool/function definitions for function calling.
            tool_choice: Controls whether the model must call a tool.

        Returns:
            Raw assistant text (or parsed tool-call dict) from the first
            successful provider, or ``LLM_BUSY_RESPONSE`` if every provider
            fails or is exhausted.
        """
        for provider in self.providers:
            try:
                return await provider.generate(
                    messages, temperature, max_tokens,
                    tools=tools, tool_choice=tool_choice,
                )
            except LLMProviderExhausted as e:
                logger.warning(
                    "LLM provider {} exhausted: {}", provider.name(), e
                )
                continue
            except Exception as e:
                logger.warning(
                    "LLM provider {} failed unexpectedly: {}",
                    provider.name(),
                    e,
                )
                continue

        logger.error("All LLM providers failed")
        return LLM_BUSY_RESPONSE
