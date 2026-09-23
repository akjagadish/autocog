"""LLM client abstraction.

One protocol (LLMClient) with provider-specific implementations. Gemini is
the default. MockClient is used in unit tests.
"""
from __future__ import annotations

import json as _json
import random as _random
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
import openai
from pydantic import BaseModel

from pathlib import Path

from src.config import LLMConfig, load_config
from src.logger import info as _log_info


Message = dict[str, str]


@dataclass
class ChatResult:
    text: str
    parsed: Any = None
    usage: dict = field(default_factory=dict)


# --- transient-error retry --------------------------------------------------
#
# Long Della runs occasionally hit network-layer errors that the provider
# SDKs' built-in retry policies don't cover (e.g. `httpx.RemoteProtocolError`:
# "Server disconnected without sending a response"). When that bubbles up
# the orchestrator's `run_round` aborts the whole multi-hour SBATCH job.
#
# We wrap each provider's network call in a small retry layer that ONLY
# catches connection-level transients — never schema/parse failures or
# auth errors, which deserve to surface fast.

_T = TypeVar("_T")
_TRANSIENT_HTTP_EXC_TYPES: tuple[type[BaseException], ...] | None = None


def _transient_http_exc_types() -> tuple[type[BaseException], ...]:
    """Resolve transient httpx/httpcore exception types lazily.

    Both libraries are pulled in transitively by google-genai / anthropic /
    openai but we still guard the imports so a missing dep doesn't crash
    module load.
    """
    global _TRANSIENT_HTTP_EXC_TYPES
    if _TRANSIENT_HTTP_EXC_TYPES is not None:
        return _TRANSIENT_HTTP_EXC_TYPES
    found: list[type[BaseException]] = []
    try:
        import httpx  # type: ignore

        found.extend(
            [
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
            ]
        )
    except Exception:
        pass
    try:
        import httpcore  # type: ignore

        found.append(httpcore.RemoteProtocolError)
    except Exception:
        pass
    # The OpenAI SDK wraps httpx connection/timeout errors in its own
    # APIConnectionError (APITimeoutError is a subclass), so the httpx
    # classes above never reach us on that path.
    found.append(openai.APIConnectionError)
    _TRANSIENT_HTTP_EXC_TYPES = tuple(found)
    return _TRANSIENT_HTTP_EXC_TYPES


def _call_with_retry(
    fn: Callable[..., _T],
    *args,
    max_attempts: int = 18,
    base_delay: float = 2.0,
    max_delay: float = 300.0,
    label: str = "LLM call",
    **kwargs,
) -> _T:
    """Run `fn(*args, **kwargs)`, retrying on transient HTTP/connection errors.

    Exponential backoff with jitter; gives up after `max_attempts` attempts
    and re-raises the last exception. Non-transient exceptions propagate
    immediately on the first attempt.

    Backoff is **per-invocation, not global**: the `attempt` counter lives
    inside this function, so a fresh call always starts at `base_delay`
    even if a previous call hit transients and escalated to a longer
    sleep. That keeps long Della runs responsive — once the network
    recovers, the next blip waits 2s rather than picking up where the
    earlier failure left off.

    Defaults are tuned for multi-hour SBATCH jobs: 18 attempts with a
    300s cap give ~53 minutes of total tolerance for a Gemini/Anthropic
    outage before aborting the run. The previous 10×120s settings
    (~8 min) proved insufficient on 2026-07-04, when a Gemini outage
    outlasted the window and killed a ~12h H200 Centaur run at round
    22/25 (job 10658911).
    """
    transient = _transient_http_exc_types()
    if not transient:
        return fn(*args, **kwargs)
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except transient as e:
            last_exc = e
            if attempt == max_attempts:
                _log_info(
                    f"{label}: transient error {type(e).__name__}: {e} — "
                    f"giving up after {attempt} attempts."
                )
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += _random.uniform(0.0, base_delay)
            _log_info(
                f"{label}: transient error {type(e).__name__}: {e} — "
                f"retrying in {delay:.1f}s (attempt {attempt}/{max_attempts})."
            )
            _time.sleep(delay)
    # Unreachable: loop either returns or raises. Kept for type-checker.
    raise last_exc if last_exc is not None else RuntimeError("retry loop fell through")


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[Message],
        system: str,
        response_schema: type | None = None,
    ) -> ChatResult: ...


def _to_contents(messages: list[Message]) -> list[types.Content]:
    """Translate our {role, content} messages to google-genai Content objects."""
    role_map = {"user": "user", "assistant": "model"}
    out = []
    for m in messages:
        role = role_map.get(m["role"])
        if role is None:
            raise ValueError(f"unknown role: {m['role']!r}")
        out.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
    return out


class GeminiClient:
    """Gemini backend via google-genai."""

    def __init__(
        self,
        model: str,
        client: genai.Client,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking_budget: int = 4096,
    ):
        self.model = model
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget

    def chat(
        self,
        messages: list[Message],
        system: str,
        response_schema: type | None = None,
    ) -> ChatResult:
        config_args: dict = {
            "system_instruction": system,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        if self.model.lower().startswith("gemini-2.") or self.model.lower().startswith("gemini-3."):
            config_args["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )
        if response_schema is not None:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = response_schema

        resp = _call_with_retry(
            self.client.models.generate_content,
            model=self.model,
            contents=_to_contents(messages),
            config=types.GenerateContentConfig(**config_args),
            label=f"GeminiClient.chat({self.model})",
        )

        usage = {}
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage = {
                "prompt_token_count": getattr(meta, "prompt_token_count", None),
                "candidates_token_count": getattr(meta, "candidates_token_count", None),
                "total_token_count": getattr(meta, "total_token_count", None),
            }

        # raise error if parsing failed when a schema was provided
        parsed = getattr(resp, "parsed", None)
        if response_schema is not None and parsed is None:
            # The SDK sometimes returns parsed=None even when resp.text is
            # valid JSON — e.g. Gemini emits a float in exponent form
            # (`0.40000000000000013e0`), which stdlib json accepts but the
            # SDK's own parser rejects. Recover by parsing the text ourselves
            # and validating against the schema. Genuinely broken/truncated
            # text (MAX_TOKENS cutoff) fails json.loads and still raises.
            # Malformed JSON is a parse failure (named below); valid JSON
            # that breaks the schema's own invariants raises pydantic's
            # ValidationError unchanged so AutoCog.propose_round can retry.
            try:
                payload = _json.loads(resp.text)
            except Exception:
                finish = None
                cands = getattr(resp, "candidates", None) or []
                if cands:
                    finish = getattr(cands[0], "finish_reason", None)
                raise RuntimeError(
                    f"structured output parse failed (finish_reason={finish}, usage={usage}).\n"
                    f"raw text:\n{resp.text!r}"
                )
            parsed = response_schema.model_validate(payload)
        return ChatResult(text=resp.text, parsed=parsed, usage=usage)


# Anthropic models that reject sampling parameters (`temperature`, `top_p`,
# `top_k`) with a 400 ("`temperature` is deprecated for this model"): the
# adaptive/always-on-thinking generation. Older models (Sonnet 4.6, Haiku 4.5,
# ...) still accept them. Verified against the live API on 2026-09-05:
# claude-sonnet-5, claude-opus-5, claude-opus-4-8, claude-opus-4-7 and
# claude-fable-5-1 reject it; claude-sonnet-4-6, claude-opus-4-6 and
# claude-haiku-4-5 accept it. (claude-mythos-* shares Fable's API surface.)
_ANTHROPIC_NO_SAMPLING_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-",
    "claude-mythos-",
)


# On Anthropic, thinking tokens count toward `max_tokens` (Gemini budgets
# thinking separately via thinking_config). Adaptive thinking on claude-opus-5
# used a median of ~25k output tokens per theory-generation call, so the YAML
# default of 32768 starved the answer: 39/72 calls returned zero text at
# stop_reason='max_tokens' (2026-09-05). Request the model output ceiling —
# the SDK streams, so a large cap has no timeout cost and is only billed when
# used. 128k is the ceiling for every current Opus/Sonnet/Fable model; Haiku
# 4.5's is 64k.
_ANTHROPIC_MAX_OUTPUT_TOKENS = 128_000
_ANTHROPIC_HAIKU_MAX_OUTPUT_TOKENS = 64_000

# A stop_reason of "max_tokens" at the model ceiling is a sampling accident,
# not a budget problem: under grammar-constrained structured output
# claude-opus-5 once looped on `\n` escapes inside a JSON string until the
# cap (2026-09-05; legitimate schema-bound answers peak at ~35k). Resample a
# bounded number of times before giving up.
_ANTHROPIC_MAX_TOKENS_RESAMPLES = 2


def _anthropic_max_output_tokens(model: str) -> int:
    if model.startswith("claude-haiku"):
        return _ANTHROPIC_HAIKU_MAX_OUTPUT_TOKENS
    return _ANTHROPIC_MAX_OUTPUT_TOKENS


def _anthropic_json_schema(schema: type[BaseModel]) -> dict:
    """JSON schema for `output_config.format`, in the subset the API accepts.

    The SDK's `transform_schema` (the same helper `messages.parse()` uses)
    adds `additionalProperties: false` and drops keywords the structured-
    output grammar rejects (numeric bounds, defaults, titles).
    """
    try:
        from anthropic.lib._parse._transform import transform_schema
    except ImportError:  # private SDK module; degrade to the raw schema
        raw = schema.model_json_schema()
        raw.setdefault("additionalProperties", False)
        return raw
    return transform_schema(schema.model_json_schema())


def _anthropic_temperature_kwarg(model: str, temperature: float) -> dict:
    """Omit `temperature` for Anthropic models that reject it (see above)."""
    if model.startswith(_ANTHROPIC_NO_SAMPLING_PREFIXES):
        return {}
    return {"temperature": temperature}


class AnthropicClient:
    """Anthropic (Claude) backend via the anthropic SDK."""

    def __init__(
        self,
        model: str,
        client: anthropic.Anthropic,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: list[Message],
        system: str,
        response_schema: type | None = None,
    ) -> ChatResult:
        api_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        # Enforce the schema natively (mirrors Gemini's `response_schema`).
        # Prose-only instructions are not enough — claude-sonnet-5 answered
        # the experiment-proposal prompt in pure markdown without this. We
        # send the schema DICT (not the pydantic class) so the SDK does not
        # parse the stream itself: a truncated/invalid response then reaches
        # our own error path below (raw text + usage) instead of dying as a
        # bare pydantic ValidationError inside get_final_message().
        schema_kwargs: dict = {}
        if response_schema is not None and issubclass(response_schema, BaseModel):
            schema_kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": _anthropic_json_schema(response_schema),
                }
            }

        def _do_stream():
            with self.client.messages.stream(
                model=self.model,
                system=system,
                messages=api_messages,
                max_tokens=self.max_tokens,
                **_anthropic_temperature_kwarg(self.model, self.temperature),
                **schema_kwargs,
            ) as stream:
                return stream.get_final_message()

        label = f"AnthropicClient.chat({self.model})"
        for resample in range(_ANTHROPIC_MAX_TOKENS_RESAMPLES + 1):
            resp = _call_with_retry(_do_stream, label=label)
            if resp.stop_reason != "max_tokens":
                break
            if resample < _ANTHROPIC_MAX_TOKENS_RESAMPLES:
                _log_info(
                    f"{label}: stop_reason=max_tokens at output_tokens="
                    f"{resp.usage.output_tokens} — resampling "
                    f"({resample + 1}/{_ANTHROPIC_MAX_TOKENS_RESAMPLES})."
                )
        else:
            raise RuntimeError(
                f"{self.model} hit max_tokens={self.max_tokens} on "
                f"{_ANTHROPIC_MAX_TOKENS_RESAMPLES + 1} consecutive samples "
                f"(last output_tokens={resp.usage.output_tokens}); giving up."
            )

        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        # Safety-classifier refusals come back as HTTP 200 with no text block.
        # Name the refusal explicitly rather than surfacing it as a confusing
        # empty-JSON parse error. Callers' per-attempt retry loops (improver,
        # theory_generator) catch and log it like any other failed generation
        # and retry, so the log shows what happened instead of a bare parse error.
        if resp.stop_reason == "refusal":
            raise RuntimeError(
                f"{self.model} refused the request "
                f"(stop_details={getattr(resp, 'stop_details', None)!r}, "
                f"usage={usage})"
            )

        text = "".join(
            block.text for block in resp.content if block.type == "text"
        )
        if not text:
            # No text block at all (max_tokens stops are already handled
            # above, so this is an unexpected shape). Name it so the callers'
            # retry logs say so instead of a bare JSONDecodeError.
            raise RuntimeError(
                f"{self.model} returned no text block "
                f"(stop_reason={resp.stop_reason!r}, usage={usage})"
            )

        parsed = None
        if schema_kwargs:
            json_text = _extract_json(text)
            # Malformed JSON is a parse failure (named below); valid JSON
            # that breaks the schema's own invariants raises pydantic's
            # ValidationError unchanged so AutoCog.propose_round can retry.
            try:
                payload = _json.loads(json_text)
            except Exception as exc:
                raise RuntimeError(
                    f"structured output parse failed (stop_reason="
                    f"{resp.stop_reason}, usage={usage}).\n"
                    f"raw text:\n{text!r}"
                ) from exc
            parsed = response_schema.model_validate(payload)

        return ChatResult(text=text, parsed=parsed, usage=usage)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM output that may contain prose, code fences, or both."""
    import re
    import json

    def _parses(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False

    # 1. Fenced blocks: prefer ```json first, then any fence. Skip fences
    #    whose body doesn't parse (e.g. prose blocks that happen to be fenced).
    for pattern in (r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        for m in re.finditer(pattern, text, re.DOTALL):
            candidate = m.group(1).strip()
            if _parses(candidate):
                return candidate

    # 2. Balanced-brace scan — tolerates missing/truncated closing fence.
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                if depth == 0:
                    candidate = text[i:j+1]
                    if _parses(candidate):
                        return candidate
                    break
        i += 1
    return text.strip()


_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

# Bad samples worth another draw, bounded: an empty body at finish_reason
# "stop" (z-ai/glm-5.3 via OpenRouter, 2026-09-18), a backend that died
# mid-generation ("error"), and a "length" stop. The last looked
# deterministic but is not: all five schema-call cap hits in the 2026-09-19
# battery (deepseek-v4-pro @max, glm-5.3 @high) succeeded on the fresh
# sample the relaunch took. A length resample re-bills up to the cap, so
# keep the bound small.
_OPENAI_EMPTY_RESAMPLES = 2


def _completion_token_kwarg(model: str, max_tokens: int) -> dict:
    """Return the right token-limit kwarg for an OpenAI chat-completion call.

    Reasoning models reject `max_tokens` and require `max_completion_tokens`.
    Older models (gpt-4o, gpt-4o-mini) still take `max_tokens`. Ported from
    antagonistic_collab/pipeline/llm.py.
    """
    if model.startswith(_REASONING_PREFIXES):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _temperature_kwarg(model: str, temperature: float) -> dict:
    """OpenAI reasoning models only accept the default temperature (1.0).

    Omit the kwarg entirely so the API uses its default, rather than sending
    1.0 which could drift if OpenAI changes defaults.
    """
    if model.startswith(_REASONING_PREFIXES):
        return {}
    return {"temperature": temperature}


class OpenAIClient:
    """OpenAI-protocol backend. Works for plain OpenAI, Princeton/Portkey and
    OpenRouter. `extra_body` is merged into every request body for gateway-
    specific fields the SDK has no kwarg for (e.g. OpenRouter routing)."""

    def __init__(
        self,
        model: str,
        client,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        extra_body: dict | None = None,
    ):
        self.model = model
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body

    def chat(
        self,
        messages: list[Message],
        system: str,
        response_schema: type | None = None,
    ) -> ChatResult:
        api_messages: list[dict] = [{"role": "system", "content": system}]
        api_messages.extend(
            {"role": m["role"], "content": m["content"]} for m in messages
        )
        kwargs = {
            "model": self.model,
            "messages": api_messages,
            **_temperature_kwarg(self.model, self.temperature),
            **_completion_token_kwarg(self.model, self.max_tokens),
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        structured = response_schema is not None and issubclass(
            response_schema, BaseModel
        )
        if response_schema is not None and not structured:
            raise RuntimeError(
                f"response_schema must be a pydantic BaseModel, got {response_schema!r}"
            )
        label = f"OpenAIClient.chat({self.model})"
        call = self.client.chat.completions.create
        if structured:
            call = self.client.chat.completions.parse
            kwargs["response_format"] = response_schema

        for attempt in range(1 + _OPENAI_EMPTY_RESAMPLES):
            try:
                resp = _call_with_retry(call, label=label, **kwargs)
            except openai.LengthFinishReasonError as exc:
                # `.parse` raises this itself on a cap hit, before we see
                # the message. Same treatment as finish_reason="length".
                completion = getattr(exc, "completion", None)
                usage = _openai_usage(completion) if completion is not None else {}
                problem = "hit max_tokens (finish_reason='length')"
                if attempt < _OPENAI_EMPTY_RESAMPLES:
                    _log_info(
                        f"{label}: {problem} on attempt {attempt + 1} "
                        f"(usage={usage}); resampling ({attempt + 1}/{_OPENAI_EMPTY_RESAMPLES})"
                    )
                continue
            usage = _openai_usage(resp)
            # OpenRouter answers HTTP 200 with choices=[] and a top-level
            # `error` when the backend fails mid-request: a bad sample.
            if not resp.choices:
                problem = f"no choices (error={getattr(resp, 'error', None)!r})"
            else:
                choice = resp.choices[0]
                msg = choice.message
                text = msg.content or ""
                if getattr(msg, "refusal", None):
                    raise RuntimeError(f"{label}: model refused: {msg.refusal!r}")
                if text.strip():
                    break
                # "stop": a bad sample; "error": OpenRouter's marker for a
                # backend that died mid-generation; "length": a stochastic
                # runaway (see _OPENAI_EMPTY_RESAMPLES). Anything else
                # (e.g. content_filter) is not worth another draw.
                if choice.finish_reason not in (None, "stop", "error", "length"):
                    raise RuntimeError(
                        f"{label}: empty response body with "
                        f"finish_reason={choice.finish_reason!r} (usage={usage})"
                    )
                problem = f"empty response body (finish_reason={choice.finish_reason!r})"
            if attempt < _OPENAI_EMPTY_RESAMPLES:
                _log_info(
                    f"{label}: {problem} on attempt {attempt + 1}; resampling "
                    f"({attempt + 1}/{_OPENAI_EMPTY_RESAMPLES})"
                )
        else:
            raise RuntimeError(
                f"{label}: {problem} on {1 + _OPENAI_EMPTY_RESAMPLES} "
                f"consecutive samples (usage={usage})"
            )

        parsed = None
        if structured:
            parsed = msg.parsed
            if parsed is None:
                raise RuntimeError(
                    f"structured output parse failed for schema "
                    f"{response_schema.__name__}.\nraw text:\n{text!r}"
                )
        return ChatResult(text=text, parsed=parsed, usage=usage)


def _openai_usage(resp) -> dict:
    """Token counts plus, when the endpoint reports them, the reasoning-token
    share and the backend that served the call (OpenRouter sets `provider`).
    Both end up in each prompt log's `## Usage` block; summarize_compute.py
    sums numeric keys containing 'token' per key and ignores the rest. Note
    `reasoning_tokens` is a SUBSET of `output_tokens`, not additional to it.
    `usage` itself is optional on the wire; an absent one yields {}."""
    if resp.usage is None:
        return {}
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    details = getattr(resp.usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    if reasoning_tokens is not None:
        usage["reasoning_tokens"] = reasoning_tokens
    provider = getattr(resp, "provider", None)
    if provider:
        usage["provider"] = provider
    return usage


class MockClient:
    """Deterministic canned-response client for unit tests."""

    def __init__(self, canned: list[str] | None = None):
        self.canned = canned or []
        self.idx = 0

    def chat(
        self,
        messages: list[Message],
        system: str,
        response_schema: type | None = None,
    ) -> ChatResult:
        if self.idx >= len(self.canned):
            raise RuntimeError("MockClient exhausted canned responses")
        text = self.canned[self.idx]
        self.idx += 1
        parsed = None
        if response_schema is not None and issubclass(response_schema, BaseModel):
            parsed = response_schema.model_validate_json(text)
        return ChatResult(text=text, parsed=parsed, usage={})


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The OpenAI-protocol client does not stream, so one request must outlive
# the whole generation: a max-effort answer near the 131k cap took ~20 min
# (AtlasCloud, 2026-09-18) and deepseek-v4-pro-0813 tripped the SDK's
# default 600 s read timeout mid-battery (2026-09-19).
OPENROUTER_TIMEOUT_S = 3600

# OpenRouter fans one model id out to ~20-30 third-party backends that differ
# in weight precision (fp4/fp8/unlisted) and reasoning defaults: the same
# round-0 prompt to z-ai/glm-5.3 got 0, 22.6k and 24.8k reasoning tokens on
# three backends and an empty body on a fourth (2026-09-18). The run, not
# the backend, must fix these, so every request carries:
#   * require_parameters — only backends that implement every parameter we
#     send (in particular the JSON-schema `response_format`; a backend
#     without it would silently return prose);
#   * quantizations=["fp8"] — never fp4 weights, and the same precision on
#     every call of a run;
#   * reasoning.effort="max" — the top of `supported_efforts` for both
#     deepseek-v4-pro-0813 and glm-5.3 (per GET /api/v1/models).
OPENROUTER_DEFAULT_REASONING_EFFORT = "max"


def openrouter_extra_body(reasoning_effort: str | None) -> dict:
    """Request body for OpenRouter at the given effort (None = default).
    Some models cannot finish at `max` — z-ai/glm-5.3 spent 128k reasoning
    tokens on the free-text theory-generation prompt without ever answering
    (three backends, 2026-09-18) but answered at `high` — so the level is a
    per-run setting (`LLMConfig.reasoning_effort`, CLI --reasoning_effort)."""
    return {
        "provider": {"require_parameters": True, "quantizations": ["fp8"]},
        "reasoning": {
            "effort": reasoning_effort or OPENROUTER_DEFAULT_REASONING_EFFORT
        },
    }


# As on Anthropic, reasoning tokens count toward `max_tokens` on OpenRouter,
# so the YAML budget (32768, sized for Gemini's answer-only output) starves
# reasoning models: z-ai/glm-5.3 spent 24k reasoning + 8.5k answer tokens on
# the first experiment-proposal call and died at the cap (2026-09-18).
# Request the ceiling instead; it is only billed when used. 131072 is the
# max_completion_tokens OpenRouter lists for glm-5.3 (deepseek-v4-pro: 393k).
# Models with a lower ceiling still work: OpenRouter clamps an over-ceiling
# max_tokens rather than rejecting it, even with require_parameters (checked
# live on z-ai/glm-5 @128k and deepseek/deepseek-v3.2 @64k, 2026-09-18).
OPENROUTER_MAX_OUTPUT_TOKENS = 131_072


def model_path_tag(model: str) -> str:
    """Model id made safe for a single path component: OpenRouter ids are
    `vendor/model`, and the run directory embeds the model name."""
    return model.replace("/", "-")


def client_from_config(
    config_path: str | Path, llm: LLMConfig | None = None
) -> LLMClient:
    """Client for an agent's `from_config`. `llm` (e.g. built from a CLI
    --llm_provider/--llm_model) overrides the YAML llm section so one run
    uses a single provider everywhere; the YAML is only read when needed."""
    if llm is None:
        llm = load_config(Path(config_path)).llm
    return make_client(llm)


def _openai_gateway_client(
    cfg: LLMConfig,
    *,
    env_var: str,
    base_url: str,
    hint: str,
    extra_body: dict | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> "OpenAIClient":
    """OpenAIClient for an OpenAI-protocol gateway (Portkey, OpenRouter) that
    needs its own key and base URL. `max_tokens` overrides cfg.max_tokens;
    `timeout` (seconds) overrides the SDK's 600 s read timeout."""
    load_dotenv()
    import os

    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(
            f"Set {env_var} to use provider={cfg.provider} ({hint})."
        )
    sdk_kwargs = {} if timeout is None else {"timeout": timeout}
    client = openai.OpenAI(api_key=api_key, base_url=base_url, **sdk_kwargs)
    return OpenAIClient(
        model=cfg.model,
        client=client,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens if max_tokens is None else max_tokens,
        extra_body=extra_body,
    )


def make_client(cfg: LLMConfig) -> LLMClient:
    """Construct an LLMClient from the llm section of RunConfig."""
    if cfg.provider == "gemini":
        load_dotenv()
        client = genai.Client()
        return GeminiClient(
            model=cfg.model,
            client=client,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            thinking_budget=cfg.thinking_budget,
        )
    if cfg.provider == "anthropic":
        load_dotenv()
        client = anthropic.Anthropic()
        return AnthropicClient(
            model=cfg.model,
            client=client,
            temperature=cfg.temperature,
            # cfg.max_tokens is an output-only budget sized for Gemini; see
            # _ANTHROPIC_MAX_OUTPUT_TOKENS for why it is not used here.
            max_tokens=_anthropic_max_output_tokens(cfg.model),
        )
    if cfg.provider == "openai":
        load_dotenv()

        # Plain OpenAI endpoint; the SDK reads OPENAI_API_KEY from the env.
        return OpenAIClient(
            model=cfg.model,
            client=openai.OpenAI(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    if cfg.provider == "princeton":
        return _openai_gateway_client(
            cfg,
            env_var="AI_SANDBOX_KEY",
            base_url="https://api.portkey.ai/v1",
            hint="get one from Princeton's AI Sandbox portal",
        )
    if cfg.provider == "openrouter":
        return _openai_gateway_client(
            cfg,
            env_var="OPENROUTER_API_KEY",
            base_url=OPENROUTER_BASE_URL,
            hint="https://openrouter.ai/keys",
            extra_body=openrouter_extra_body(cfg.reasoning_effort),
            max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
            timeout=OPENROUTER_TIMEOUT_S,
        )
    if cfg.provider == "mock":
        return MockClient()
    raise NotImplementedError(f"provider {cfg.provider!r} not supported yet")
