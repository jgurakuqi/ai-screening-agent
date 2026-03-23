"""OpenRouter LLM provider with multi-model fallback and retry logic.

Cycles through a prioritized list of models, retrying on transient errors
(rate limits, server errors) with exponential back-off and jitter.  Models
that return empty content are placed on a configurable cooldown.
"""

import asyncio
import random
import time

from openai import APIError, AsyncOpenAI, RateLimitError

from backend.llm.base import LLMProvider, LLMProviderExhausted
from backend.logging_config import logger


class OpenRouterProvider(LLMProvider):
    """LLM provider using OpenRouter's API with multi-model fallback."""

    def __init__(
        self,
        api_key: str,
        models: list[str],
        max_retries_per_model: int = 3,
        retry_base_delay: float = 2.0,
        rate_limit_base_delay: float = 12.0,
        rate_limit_max_wait: float = 60.0,
        pass_base_delay: float = 15.0,
        pass_max_wait: float = 300.0,
        max_full_passes: int = 0,
        jitter_fraction: float = 0.3,
        empty_content_cooldown: int = 600,
    ) -> None:
        """Initialize the OpenRouter client and retry policy.

        Args:
            api_key: OpenRouter API key used for authenticated requests.
            models: Ordered model identifiers to try during fallback.
            max_retries_per_model: Retry attempts before moving to the next
                model.
            retry_base_delay: Base delay for general exponential backoff.
            rate_limit_base_delay: Base delay used after rate-limit failures.
            rate_limit_max_wait: Maximum delay allowed for rate-limit backoff.
            pass_base_delay: Base delay before retrying the full model list.
            pass_max_wait: Maximum delay allowed between full passes.
            max_full_passes: Number of full model-list passes to allow. ``0``
                means retry indefinitely.
            jitter_fraction: Symmetric random jitter applied to sleep delays.
            empty_content_cooldown: Seconds to cool down models that return an
                empty completion.
        """
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._models = [m for m in models if m]
        self._max_retries = max(1, max_retries_per_model)
        self._retry_base_delay = retry_base_delay
        self._rate_limit_base_delay = rate_limit_base_delay
        self._rate_limit_max_wait = rate_limit_max_wait
        self._pass_base_delay = pass_base_delay
        self._pass_max_wait = pass_max_wait
        self._max_full_passes = max(0, max_full_passes)
        self._jitter_fraction = jitter_fraction
        self._empty_content_cooldown = empty_content_cooldown
        self._empty_content_cooldown_until: dict[str, float] = {}

    def name(self) -> str:
        """Return the provider identifier used by the service layer.

        Returns:
            Stable provider name for logging and fallback reporting.
        """
        return "openrouter"

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        """Generate a chat completion using ordered model fallback.

        Args:
            messages: Chat-completion messages to send to OpenRouter.
            temperature: Sampling temperature for the request.
            max_tokens: Upper bound for completion tokens.

        Returns:
            Assistant text from the first successful model response.

        Raises:
            LLMProviderExhausted: If all configured models are exhausted across
                the allowed full passes.
        """
        last_error: Exception | None = None
        call_start = time.monotonic()
        pass_number = 0

        logger.debug(
            "OpenRouter call started | temp={} models={}",
            temperature,
            self._models,
        )

        while True:
            pass_number += 1
            models_to_try = self._get_models_to_try()
            if models_to_try:
                logger.debug(
                    "OpenRouter pass {}/{} with models={}",
                    pass_number,
                    self._max_full_passes if self._max_full_passes else "inf",
                    models_to_try,
                )
            else:
                logger.warning(
                    "OpenRouter pass {}/{} has no models ready (all cooling down)",
                    pass_number,
                    self._max_full_passes if self._max_full_passes else "inf",
                )

            for i, model in enumerate(models_to_try):
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                for attempt in range(1, self._max_retries + 1):
                    attempt_start = time.monotonic()
                    try:
                        logger.trace(
                            "Calling model={} attempt={}/{} pass={}",
                            model,
                            attempt,
                            self._max_retries,
                            pass_number,
                        )
                        response = await self._client.chat.completions.create(**kwargs)
                        choice = response.choices[0]
                        provider_model = getattr(response, "model", model)
                        content = self._extract_message_text(choice.message).strip()
                        elapsed = time.monotonic() - attempt_start

                        if content:
                            total_elapsed = time.monotonic() - call_start
                            logger.info(
                                "LLM response received | model={} provider={} "
                                "attempt_ms={:.0f} total_ms={:.0f} len={}",
                                model,
                                provider_model,
                                elapsed * 1000,
                                total_elapsed * 1000,
                                len(content),
                            )
                            logger.debug(
                                "LLM raw response (first 300 chars): {}",
                                content[:300],
                            )
                            return content

                        finish_reason = getattr(choice, "finish_reason", None)
                        self._mark_model_empty_content(model)
                        logger.warning(
                            "Model {} resolved to {} but returned empty content "
                            "(finish_reason={}) after {:.0f}ms - cooling down for {}s",
                            model,
                            provider_model,
                            finish_reason,
                            elapsed * 1000,
                            self._empty_content_cooldown,
                        )
                        break
                    except (RateLimitError, APIError) as e:
                        last_error = e
                        elapsed = time.monotonic() - attempt_start
                        retryable = self._is_retryable_error(e)
                        is_429 = self._is_rate_limit_error(e)

                        if is_429 and i < len(models_to_try) - 1:
                            logger.warning(
                                "Model {} rate-limited (429) after {:.0f}ms, "
                                "skipping to next model. Error: {}",
                                model,
                                elapsed * 1000,
                                e,
                            )
                            break

                        if retryable and attempt < self._max_retries:
                            delay = self._compute_retry_delay(e, attempt)
                            logger.warning(
                                "Model {} attempt {}/{} failed after {:.0f}ms ({}): {}. "
                                "Retrying in {:.1f}s",
                                model,
                                attempt,
                                self._max_retries,
                                elapsed * 1000,
                                "429 rate-limited" if is_429 else "retryable",
                                e,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue

                        if i < len(models_to_try) - 1:
                            logger.warning(
                                "Model {} failed after {:.0f}ms, trying next fallback: {}",
                                model,
                                elapsed * 1000,
                                e,
                            )
                            break

                        logger.warning(
                            "Model {} exhausted on pass {} after {:.0f}ms: {}",
                            model,
                            pass_number,
                            elapsed * 1000,
                            e,
                        )
                        break
                    except Exception as e:
                        last_error = e
                        elapsed = time.monotonic() - attempt_start
                        if i < len(models_to_try) - 1:
                            logger.warning(
                                "Model {} unexpected error after {:.0f}ms, "
                                "trying next fallback: {}",
                                model,
                                elapsed * 1000,
                                e,
                            )
                            break
                        logger.warning(
                            "Model {} unexpected error on pass {} after {:.0f}ms: {}",
                            model,
                            pass_number,
                            elapsed * 1000,
                            e,
                        )
                        break

            if self._max_full_passes and pass_number >= self._max_full_passes:
                total_elapsed = time.monotonic() - call_start
                msg = (
                    f"Last error: {last_error}"
                    if last_error
                    else "All models returned empty content"
                )
                raise LLMProviderExhausted(
                    f"OpenRouter exhausted after {total_elapsed:.0f}ms across "
                    f"{pass_number} pass(es). {msg}"
                )

            delay = self._compute_pass_delay(pass_number, last_error)
            next_ready_delay = self._next_ready_delay()
            if next_ready_delay is not None:
                delay = max(delay, next_ready_delay)

            logger.warning(
                "OpenRouter pass {} exhausted with no usable response. "
                "Waiting {:.1f}s before retrying the full model list.",
                pass_number,
                delay,
            )
            await asyncio.sleep(delay)

    def _get_models_to_try(self) -> list[str]:
        """Return the model list excluding those currently on empty-content cooldown."""
        base_models = list(dict.fromkeys(self._models))
        now = time.monotonic()
        ready_models = []

        for model in base_models:
            cooldown_until = self._empty_content_cooldown_until.get(model, 0.0)
            if cooldown_until > now:
                logger.info(
                    "Skipping model {} for {:.1f}s (empty-content cooldown)",
                    model,
                    cooldown_until - now,
                )
                continue
            ready_models.append(model)

        logger.debug("Models to try: {}", ready_models)
        return ready_models

    def _mark_model_empty_content(self, model: str) -> None:
        """Place a model on cooldown after it returned empty content."""
        if self._empty_content_cooldown <= 0:
            return
        self._empty_content_cooldown_until[model] = (
            time.monotonic() + self._empty_content_cooldown
        )

    @staticmethod
    def _extract_message_text(message) -> str:
        """Normalize assistant content across plain-string and rich-content payloads."""
        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(text, dict) and isinstance(text.get("value"), str):
                    parts.append(text["value"])
                    continue

            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue

            value = getattr(text, "value", None)
            if isinstance(value, str):
                parts.append(value)

        return "".join(parts)

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Return True if the error is transient and worth retrying."""
        if isinstance(error, RateLimitError):
            return True
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            return True
        return status_code in (408, 409, 429) or status_code >= 500

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Return True if the error is specifically a 429 rate-limit."""
        if isinstance(error, RateLimitError):
            return True
        return getattr(error, "status_code", None) == 429

    @staticmethod
    def _get_retry_after(error: Exception) -> float | None:
        """Extract the ``Retry-After`` header value from an error response, if present."""
        response = getattr(error, "response", None)
        if response is None:
            return None
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            return None

    def _add_jitter(self, delay: float) -> float:
        """Apply symmetric random jitter to a delay value."""
        jitter_range = delay * self._jitter_fraction
        return delay + random.uniform(-jitter_range, jitter_range)

    def _next_ready_delay(self) -> float | None:
        """Return seconds until the next cooled-down model becomes available, or ``None``."""
        now = time.monotonic()
        ready_in = [
            cooldown_until - now
            for cooldown_until in self._empty_content_cooldown_until.values()
            if cooldown_until > now
        ]
        if not ready_in:
            return None
        return max(0.0, min(ready_in))

    def _compute_retry_delay(self, error: Exception, attempt: int) -> float:
        """Compute exponential back-off delay for a single-model retry."""
        retry_after = self._get_retry_after(error)
        if retry_after is not None and retry_after > 0:
            delay = min(retry_after, self._rate_limit_max_wait)
            return self._add_jitter(delay)

        if self._is_rate_limit_error(error):
            base = self._rate_limit_base_delay
        else:
            base = self._retry_base_delay

        delay = base * (2 ** (attempt - 1))
        delay = min(delay, self._rate_limit_max_wait)
        return self._add_jitter(delay)

    def _compute_pass_delay(self, pass_number: int, error: Exception | None) -> float:
        """Compute delay between full passes through the entire model list."""
        if error is not None and self._is_rate_limit_error(error):
            base = max(self._pass_base_delay, self._rate_limit_base_delay)
            wait_cap = max(self._pass_max_wait, self._rate_limit_max_wait)
        else:
            base = self._pass_base_delay
            wait_cap = self._pass_max_wait

        delay = base * (2 ** (pass_number - 1))
        delay = min(delay, wait_cap)
        return self._add_jitter(delay)
