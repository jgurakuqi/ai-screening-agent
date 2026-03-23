"""Interfaces shared by all LLM providers."""

from abc import ABC, abstractmethod


class LLMProviderExhausted(Exception):
    """Raised when a provider has tried all its models/retries and failed."""


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Concrete subclasses must implement :meth:`generate` (the completion call)
    and :meth:`name` (used in log messages).
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str | dict:
        """Generate a completion from a list of chat messages.

        Args:
            messages: Chat-completion messages in ``{"role", "content"}``
                format.
            temperature: Sampling temperature to pass to the provider.
            max_tokens: Upper bound for generated completion tokens.
            tools: Optional list of tool/function definitions for function
                calling.
            tool_choice: Controls whether the model must call a tool.
                ``"required"`` forces a tool call.

        Returns:
            Raw assistant text when no tools are used, or a parsed dict of
            the tool call arguments when function calling is active.

        Raises:
            LLMProviderExhausted: When all internal retries/models are used up.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the provider label used in logs and diagnostics.

        Returns:
            Short human-readable provider identifier.
        """
