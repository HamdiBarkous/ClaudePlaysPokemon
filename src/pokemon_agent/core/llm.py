"""LLM factory for creating language model instances."""

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from pokemon_agent.core.settings import ThinkingLevel, get_settings

logger = logging.getLogger(__name__)


def _hit_token_limit(result: ChatResult) -> bool:
    """Check if the response was truncated by hitting the max output token limit."""
    for gen in result.generations:
        if not isinstance(gen, ChatGeneration):
            continue
        finish = (
            gen.generation_info.get("finish_reason")
            if gen.generation_info
            else None
        ) or gen.message.response_metadata.get("finish_reason")
        if finish == "length":
            return True
    return False


class RobustChatOpenAI(ChatOpenAI):
    """ChatOpenAI with retry, loop detection, and OpenRouter reasoning preservation.

    This handles cases where:
    1. API returns HTTP 200 but response body is truncated/malformed JSON
    2. API returns HTTP 200 but response contains an error (e.g., OpenRouter 500)
    3. Model enters infinite generation loop and hits max_tokens (finish_reason=length)
    4. OpenRouter returns reasoning in non-standard fields that ChatOpenAI drops

    Reasoning preservation:
    - Extracts reasoning/reasoning_details from raw responses into AIMessage.additional_kwargs
    - Re-serializes them back when building requests (for multi-turn continuity)
    """

    # Max retries when the model hits the token limit (infinite loop detection).
    # Set to 0 to disable loop detection retries.
    loop_retry_limit: int = 0

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        # Malformed JSON response
        if isinstance(error, json.JSONDecodeError):
            return True

        # OpenRouter/proxy returns 200 OK but with error in body
        # e.g., ValueError: {'message': 'Internal Server Error', 'code': 500}
        if isinstance(error, ValueError):
            error_str = str(error)
            if "400" in error_str or "500" in error_str or "502" in error_str or "503" in error_str:
                return True
            if "Internal Server Error" in error_str or "Provider returned error" in error_str:
                return True

        return False

    def _extract_reasoning(self, raw_json: dict) -> dict[str, Any]:
        """Extract reasoning fields from raw OpenRouter response JSON."""
        reasoning_data: dict[str, Any] = {}
        if raw_json.get("choices"):
            raw_msg = raw_json["choices"][0].get("message", {})
            if reasoning := raw_msg.get("reasoning"):
                reasoning_data["reasoning_content"] = reasoning
            if reasoning_details := raw_msg.get("reasoning_details"):
                reasoning_data["reasoning_details"] = reasoning_details
        return reasoning_data

    def _inject_reasoning(self, result: ChatResult, reasoning_data: dict[str, Any]) -> None:
        """Inject reasoning data into the AIMessage's additional_kwargs."""
        if reasoning_data and result.generations:
            result.generations[0].message.additional_kwargs.update(reasoning_data)

    def _has_pydantic_response_format(self, **kwargs: Any) -> bool:
        """Check if the request uses a Pydantic model as response_format (structured output)."""
        from pydantic import BaseModel

        rf = kwargs.get("response_format") or self.model_kwargs.get("response_format")
        return isinstance(rf, type) and issubclass(rf, BaseModel)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override to preserve reasoning, retry on errors, and detect loops."""
        # Structured output (Pydantic response_format) requires parse() not create(),
        # so fall back to default ChatOpenAI path — no reasoning to preserve anyway.
        if self._has_pydantic_response_format(**kwargs):
            return super()._generate(messages, stop, run_manager, **kwargs)

        max_attempts = max(3, self.loop_retry_limit + 1)
        last_error = None

        for attempt in range(max_attempts):
            try:
                # Make raw request to preserve reasoning fields
                payload = self._get_request_payload(messages, stop=stop, **kwargs)
                raw_http = self.client.with_raw_response.create(**payload)

                reasoning_data = self._extract_reasoning(json.loads(raw_http.text))

                # Let ChatOpenAI parse the response normally
                response = raw_http.parse()
                result = self._create_chat_result(response, None)

                self._inject_reasoning(result, reasoning_data)
            except (json.JSONDecodeError, ValueError) as e:
                if not self._is_retryable_error(e):
                    raise
                last_error = e
                if attempt < max_attempts - 1:
                    wait_time = 3 ** (attempt + 1)
                    time.sleep(wait_time)
                    continue
                raise

            # Check for infinite generation loop (finish_reason=length)
            if self.loop_retry_limit > 0 and _hit_token_limit(result):
                if attempt < self.loop_retry_limit:
                    logger.warning(
                        "Infinite generation loop detected (finish_reason=length), "
                        "retrying (%d/%d)",
                        attempt + 1,
                        self.loop_retry_limit,
                    )
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                logger.warning(
                    "Infinite generation loop persists after %d retries, returning last result",
                    self.loop_retry_limit,
                )
            return result

        raise last_error  # type: ignore

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async override to preserve reasoning and detect loops."""
        if self._has_pydantic_response_format(**kwargs):
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

        max_attempts = max(3, self.loop_retry_limit + 1)

        for attempt in range(max_attempts):
            try:
                payload = self._get_request_payload(messages, stop=stop, **kwargs)
                raw_http = await self.async_client.with_raw_response.create(**payload)

                reasoning_data = self._extract_reasoning(json.loads(raw_http.text))

                response = raw_http.parse()
                result = self._create_chat_result(response, None)

                self._inject_reasoning(result, reasoning_data)
            except (json.JSONDecodeError, ValueError) as e:
                if not self._is_retryable_error(e):
                    raise
                if attempt < max_attempts - 1:
                    wait_time = 3 ** (attempt + 1)
                    await asyncio.sleep(wait_time)
                    continue
                raise

            if self.loop_retry_limit > 0 and _hit_token_limit(result):
                if attempt < self.loop_retry_limit:
                    logger.warning(
                        "Infinite generation loop detected (finish_reason=length), "
                        "retrying (%d/%d)",
                        attempt + 1,
                        self.loop_retry_limit,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.warning(
                    "Infinite generation loop persists after %d retries, returning last result",
                    self.loop_retry_limit,
                )
            return result

        return result  # type: ignore

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Override to re-inject reasoning into outgoing messages for multi-turn continuity."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Re-serialize reasoning from AIMessage.additional_kwargs back into
        # the message dicts so OpenRouter gets the encrypted reasoning blob.
        # Match by index (both lists are in the same order) to avoid ambiguity
        # when multiple assistant messages have identical content (e.g. empty
        # string with tool calls in ReAct loops).
        if isinstance(input_, list):
            msg_dicts = payload.get("messages", [])
            orig_idx = 0
            for msg_dict in msg_dicts:
                if msg_dict.get("role") != "assistant":
                    continue
                # Advance to the next AIMessage in the original list
                while orig_idx < len(input_) and not isinstance(input_[orig_idx], AIMessage):
                    orig_idx += 1
                if orig_idx >= len(input_):
                    break
                orig_msg = input_[orig_idx]
                if "reasoning_content" in orig_msg.additional_kwargs:
                    msg_dict["reasoning"] = orig_msg.additional_kwargs["reasoning_content"]
                if "reasoning_details" in orig_msg.additional_kwargs:
                    msg_dict["reasoning_details"] = orig_msg.additional_kwargs["reasoning_details"]
                orig_idx += 1

        return payload


_global_rate_limiter: InMemoryRateLimiter | None = None
_rate_limiter_initialized = False


def _get_rate_limiter() -> InMemoryRateLimiter | None:
    """Get a shared global rate limiter instance based on settings."""
    global _global_rate_limiter, _rate_limiter_initialized
    if _rate_limiter_initialized:
        return _global_rate_limiter
    settings = get_settings()
    if settings.max_requests_per_second > 0:
        _global_rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.max_requests_per_second
        )
    _rate_limiter_initialized = True
    return _global_rate_limiter


def get_llm(
    model_name: str | None = None,
    thinking: ThinkingLevel | None = None,
    **kwargs,
) -> BaseChatModel:
    """Create an LLM instance via OpenRouter.

    Args:
        model_name: OpenRouter model ID (e.g., "google/gemini-3.1-flash-lite").
                   Defaults to settings.game_model.
        thinking: Thinking level for reasoning control. Defaults to settings.thinking.
                 "none" disables thinking; "minimal"/"low"/"medium"/"high" set the level.
        **kwargs: Additional arguments passed to the LLM constructor.

    Returns:
        Configured RobustChatOpenAI instance.
    """
    settings = get_settings()
    rate_limiter = _get_rate_limiter()

    resolved_model = model_name or settings.game_model
    resolved_thinking = thinking if thinking is not None else settings.thinking

    extra_body: dict = {"provider": {"sort": "latency"}}
    if resolved_thinking == "none":
        extra_body["reasoning"] = {"enabled": False}
    else:
        extra_body["reasoning"] = {"effort": resolved_thinking}

    return RobustChatOpenAI(
        model=resolved_model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        max_retries=3,
        request_timeout=120.0,  # type: ignore
        extra_body=extra_body,
        rate_limiter=rate_limiter,
        default_headers={
            "HTTP-Referer": "https://hamdibarkous.com",
            "X-Title": "Claude Plays Pokemon",
        },
        **kwargs,
    )
