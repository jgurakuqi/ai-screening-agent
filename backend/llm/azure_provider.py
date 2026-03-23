"""Azure OpenAI LLM provider with multi-model fallback and retry logic.

Uses Azure AI Foundry deployments via the OpenAI SDK's AzureOpenAI client.
Cycles through a prioritized list of deployment names, retrying on transient
errors with exponential back-off and jitter.  Supports function calling via
the ``tools`` / ``tool_choice`` parameters.
"""

import asyncio
import json
import random
import time

from openai import APIError, AsyncAzureOpenAI, RateLimitError

from backend.llm.base import LLMProvider, LLMProviderExhausted
from backend.logging_config import logger


class AzureProvider(LLMProvider):
    """LLM provider using Azure OpenAI Service with multi-model fallback."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str = "2024-12-01-preview",
        models: list[str] | None = None,
        max_retries_per_model: int = 3,
        retry_base_delay: float = 2.0,
        rate_limit_base_delay: float = 12.0,
        rate_limit_max_wait: float = 60.0,
        max_full_passes: int = 3,
        jitter_fraction: float = 0.3,
    ) -> None:
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self._models = [m for m in (models or []) if m]
        self._max_retries = max(1, max_retries_per_model)
        self._retry_base_delay = retry_base_delay
        self._rate_limit_base_delay = rate_limit_base_delay
        self._rate_limit_max_wait = rate_limit_max_wait
        self._max_full_passes = max(1, max_full_passes)
        self._jitter_fraction = jitter_fraction

    def name(self) -> str:
        return "azure"

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str | dict:
        last_error: Exception | None = None
        call_start = time.monotonic()
        use_tools = bool(tools)

        for pass_number in range(1, self._max_full_passes + 1):
            logger.debug(
                "Azure pass {}/{} with models={}",
                pass_number,
                self._max_full_passes,
                self._models,
            )

            for i, model in enumerate(self._models):
                for attempt in range(1, self._max_retries + 1):
                    attempt_start = time.monotonic()
                    try:
                        logger.trace(
                            "Azure calling model={} attempt={}/{} pass={} tools={}",
                            model, attempt, self._max_retries, pass_number, use_tools,
                        )

                        kwargs: dict = dict(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_completion_tokens=max_tokens,
                        )
                        if tools:
                            kwargs["tools"] = tools
                        if tool_choice is not None:
                            kwargs["tool_choice"] = tool_choice

                        response = await self._client.chat.completions.create(**kwargs)
                        message = response.choices[0].message
                        elapsed = time.monotonic() - attempt_start

                        # Function calling mode: parse tool_calls
                        if use_tools and message.tool_calls:
                            tool_call = message.tool_calls[0]
                            arguments = json.loads(tool_call.function.arguments)
                            total_elapsed = time.monotonic() - call_start
                            logger.info(
                                "Azure tool call response | model={} fn={} attempt_ms={:.0f} "
                                "total_ms={:.0f}",
                                model, tool_call.function.name,
                                elapsed * 1000, total_elapsed * 1000,
                            )
                            return arguments

                        # Content mode: return text
                        content = (message.content or "").strip()

                        if content:
                            total_elapsed = time.monotonic() - call_start
                            logger.info(
                                "Azure LLM response | model={} attempt_ms={:.0f} "
                                "total_ms={:.0f} len={}",
                                model, elapsed * 1000, total_elapsed * 1000, len(content),
                            )
                            return content

                        logger.warning(
                            "Azure model {} returned empty content after {:.0f}ms",
                            model, elapsed * 1000,
                        )
                        break  # move to next model

                    except (RateLimitError, APIError) as e:
                        last_error = e
                        elapsed = time.monotonic() - attempt_start
                        is_retryable = self._is_retryable(e)

                        if is_retryable and attempt < self._max_retries:
                            delay = self._compute_delay(e, attempt)
                            logger.warning(
                                "Azure model {} attempt {}/{} failed after {:.0f}ms: {}. "
                                "Retrying in {:.1f}s",
                                model, attempt, self._max_retries, elapsed * 1000, e, delay,
                            )
                            await asyncio.sleep(delay)
                            continue

                        logger.warning(
                            "Azure model {} failed after {:.0f}ms, moving on: {}",
                            model, elapsed * 1000, e,
                        )
                        break

                    except Exception as e:
                        last_error = e
                        elapsed = time.monotonic() - attempt_start
                        logger.warning(
                            "Azure model {} unexpected error after {:.0f}ms: {}",
                            model, elapsed * 1000, e,
                        )
                        break

            if pass_number < self._max_full_passes:
                delay = self._add_jitter(self._retry_base_delay * (2 ** (pass_number - 1)))
                logger.warning(
                    "Azure pass {} exhausted, waiting {:.1f}s before retry",
                    pass_number, delay,
                )
                await asyncio.sleep(delay)

        total_elapsed = time.monotonic() - call_start
        msg = f"Last error: {last_error}" if last_error else "All models returned empty"
        raise LLMProviderExhausted(
            f"Azure exhausted after {total_elapsed:.0f}ms across "
            f"{self._max_full_passes} pass(es). {msg}"
        )

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            return True
        return status_code in (408, 429) or status_code >= 500

    def _compute_delay(self, error: Exception, attempt: int) -> float:
        if isinstance(error, RateLimitError):
            base = self._rate_limit_base_delay
        else:
            base = self._retry_base_delay
        delay = base * (2 ** (attempt - 1))
        delay = min(delay, self._rate_limit_max_wait)
        return self._add_jitter(delay)

    def _add_jitter(self, delay: float) -> float:
        jitter_range = delay * self._jitter_fraction
        return delay + random.uniform(-jitter_range, jitter_range)
