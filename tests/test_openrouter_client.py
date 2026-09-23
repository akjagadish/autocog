"""Tests for the OpenRouter provider path in src/llm.py.

Motivation: re-running the recovery battery with open-weight frontier models
(deepseek/deepseek-v4-pro-0813, z-ai/glm-5.3). OpenRouter speaks the OpenAI
chat-completions protocol, so `provider="openrouter"` reuses `OpenAIClient`
with (a) the OpenRouter base URL and key, (b) `provider.require_parameters`
so OpenRouter only routes to backends that honour `response_format` (the
structured-output schema), and (c) a path-safe tag for model ids that
contain a slash, which the run-directory name embeds.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.config import LLMConfig
from types import SimpleNamespace

from src.llm import (
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_OUTPUT_TOKENS,
    OPENROUTER_TIMEOUT_S,
    OpenAIClient,
    make_client,
    model_path_tag,
    openrouter_extra_body,
)

REQUIRE_PARAMS = {"provider": {"require_parameters": True}}


class TestConfig:
    def test_openrouter_provider_accepted(self):
        cfg = LLMConfig(provider="openrouter", model="z-ai/glm-5.3")
        assert cfg.provider == "openrouter"
        assert cfg.model == "z-ai/glm-5.3"


class TestMakeClientOpenRouter:
    def test_builds_openai_client_against_openrouter(self):
        cfg = LLMConfig(provider="openrouter", model="deepseek/deepseek-v4-pro-0813")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}), patch(
            "openai.OpenAI"
        ) as mock_ctor:
            mock_ctor.return_value = MagicMock()
            client = make_client(cfg)

        assert isinstance(client, OpenAIClient)
        assert client.model == "deepseek/deepseek-v4-pro-0813"
        assert client.temperature == cfg.temperature
        # Reasoning tokens count toward max_tokens on OpenRouter, so the YAML
        # budget (32768, sized for Gemini's answer-only output) is replaced by
        # the model output ceiling: z-ai/glm-5.3 spent 24k reasoning + 8.5k
        # answer tokens on the first experiment-proposal call and hit the cap.
        assert client.max_tokens == OPENROUTER_MAX_OUTPUT_TOKENS == 131_072
        assert client.max_tokens > cfg.max_tokens
        assert client.extra_body == openrouter_extra_body(None)
        # Non-streaming; a max-effort answer near the 131k cap takes ~20 min
        # (AtlasCloud, 2026-09-18), well past the SDK's 600 s read timeout.
        assert OPENROUTER_TIMEOUT_S == 3600
        mock_ctor.assert_called_once_with(
            api_key="or-test", base_url=OPENROUTER_BASE_URL, timeout=OPENROUTER_TIMEOUT_S
        )

    def test_request_body_pins_reasoning_and_quantization(self):
        """OpenRouter fans one model id out to ~30 backends with different
        reasoning defaults and fp4/fp8 weights (z-ai/glm-5.3 replayed the same
        prompt with 0, 22.6k and 24.8k reasoning tokens on three backends,
        2026-09-18). The run, not the backend, must fix both. `max` is the top
        of `supported_efforts` for both chosen models."""
        assert openrouter_extra_body(None) == {
            "provider": {"require_parameters": True, "quantizations": ["fp8"]},
            "reasoning": {"effort": "max"},
        }

    def test_reasoning_effort_config_overrides_default(self):
        """z-ai/glm-5.3 at effort=max never leaves its thinking block on the
        free-text theory-generation prompt (128k reasoning tokens, empty
        answer, on three backends, 2026-09-18) but answers at effort=high.
        The level is therefore a per-run config knob; None keeps `max`."""
        assert LLMConfig(provider="openrouter", model="z-ai/glm-5.3").reasoning_effort is None
        assert openrouter_extra_body("high") == {
            "provider": {"require_parameters": True, "quantizations": ["fp8"]},
            "reasoning": {"effort": "high"},
        }
        cfg = LLMConfig(provider="openrouter", model="z-ai/glm-5.3", reasoning_effort="high")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}), patch(
            "openai.OpenAI"
        ) as mock_ctor:
            mock_ctor.return_value = MagicMock()
            client = make_client(cfg)
        assert client.extra_body["reasoning"] == {"effort": "high"}

    def test_princeton_branch_unchanged_by_shared_helper(self):
        """princeton now goes through the same gateway helper; its key, base
        URL, YAML token budget and absent extra_body must be as before."""
        cfg = LLMConfig(provider="princeton", model="gpt-4o-mini")
        with patch.dict(os.environ, {"AI_SANDBOX_KEY": "sb-test"}), patch(
            "openai.OpenAI"
        ) as mock_ctor:
            mock_ctor.return_value = MagicMock()
            client = make_client(cfg)
        assert client.max_tokens == cfg.max_tokens
        assert client.extra_body is None
        mock_ctor.assert_called_once_with(
            api_key="sb-test", base_url="https://api.portkey.ai/v1"
        )

    def test_missing_key_raises(self):
        cfg = LLMConfig(provider="openrouter", model="z-ai/glm-5.3")
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with patch.dict(os.environ, env, clear=True), patch(
            "src.llm.load_dotenv"
        ), pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            make_client(cfg)


class _Answer(BaseModel):
    x: int


def _resp(
    parsed=None,
    content="hi",
    provider=None,
    reasoning_tokens=None,
    finish_reason="stop",
    refusal=None,
    usage=True,
    choices=True,
    error=None,
):
    """One chat-completion response with only the fields the client reads."""
    details = (
        None if reasoning_tokens is None
        else SimpleNamespace(reasoning_tokens=reasoning_tokens)
    )
    usage_obj = SimpleNamespace(
        prompt_tokens=3, completion_tokens=2, completion_tokens_details=details
    ) if usage else None
    msg = SimpleNamespace(parsed=parsed, content=content, refusal=refusal)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice] if choices else [], usage=usage_obj)
    if provider is not None:
        resp.provider = provider
    if error is not None:
        resp.error = error
    return resp


def _mock_openai(*responses):
    """openai.OpenAI stand-in whose .parse/.create return `responses` in turn."""
    sdk = MagicMock()
    sdk.chat.completions.parse.side_effect = list(responses)
    sdk.chat.completions.create.side_effect = list(responses)
    return sdk


class TestExtraBodyForwarding:
    def test_extra_body_reaches_parse_and_create(self):
        sdk = _mock_openai(
            _resp(parsed=_Answer(x=1), content='{"x": 1}'), _resp()
        )
        client = OpenAIClient(
            model="z-ai/glm-5.3", client=sdk, extra_body=REQUIRE_PARAMS
        )
        msgs = [{"role": "user", "content": "go"}]

        out = client.chat(msgs, system="sys", response_schema=_Answer)
        assert out.parsed == _Answer(x=1)
        assert sdk.chat.completions.parse.call_args.kwargs["extra_body"] == REQUIRE_PARAMS

        client.chat(msgs, system="sys")
        assert sdk.chat.completions.create.call_args.kwargs["extra_body"] == REQUIRE_PARAMS

    def test_no_extra_body_by_default(self):
        """Plain OpenAI / Princeton calls must not grow an `extra_body` kwarg."""
        sdk = _mock_openai(_resp())
        client = OpenAIClient(model="gpt-4o-mini", client=sdk)
        client.chat([{"role": "user", "content": "go"}], system="sys")
        assert "extra_body" not in sdk.chat.completions.create.call_args.kwargs


class TestUsageProvenance:
    """Which backend served a call and how much it reasoned go into `usage`,
    hence into each prompt log's `## Usage` block. summarize_compute.py sums
    every numeric key containing 'token' and ignores the rest."""

    def test_provider_and_reasoning_tokens_recorded(self):
        sdk = _mock_openai(_resp(provider="Baidu", reasoning_tokens=24849))
        out = OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(
            [{"role": "user", "content": "go"}], system="sys"
        )
        assert out.usage == {
            "input_tokens": 3,
            "output_tokens": 2,
            "reasoning_tokens": 24849,
            "provider": "Baidu",
        }

    def test_plain_openai_usage_unchanged(self):
        sdk = _mock_openai(_resp())
        out = OpenAIClient(model="gpt-4o-mini", client=sdk).chat(
            [{"role": "user", "content": "go"}], system="sys"
        )
        assert out.usage == {"input_tokens": 3, "output_tokens": 2}


class TestEmptyContentResample:
    """z-ai/glm-5.3 returned finish_reason=stop with an empty body on the
    first pipeline call (2026-09-18) and crashed the run. Mirror the
    Anthropic client's bounded resample instead of dying on one bad sample."""

    MSGS = [{"role": "user", "content": "go"}]

    def test_structured_resamples_then_succeeds(self):
        sdk = _mock_openai(
            _resp(parsed=None, content=""),
            _resp(parsed=_Answer(x=2), content='{"x": 2}'),
        )
        out = OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(
            self.MSGS, system="sys", response_schema=_Answer
        )
        assert out.parsed == _Answer(x=2)
        assert sdk.chat.completions.parse.call_count == 2

    def test_free_text_resamples_then_succeeds(self):
        sdk = _mock_openai(_resp(content=""), _resp(content="theory"))
        out = OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(
            self.MSGS, system="sys"
        )
        assert out.text == "theory"
        assert sdk.chat.completions.create.call_count == 2

    def test_gives_up_after_bounded_resamples(self):
        sdk = _mock_openai(*[_resp(content="")] * 5)
        with pytest.raises(RuntimeError, match="empty"):
            OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(
                self.MSGS, system="sys"
            )
        assert sdk.chat.completions.create.call_count == 3  # 1 + 2 resamples

    def test_refusal_raises_without_resampling(self):
        """A refusal is a model decision, not a bad sample: raise once, name it."""
        sdk = _mock_openai(
            _resp(parsed=None, content=None, refusal="I can't help with that"),
            _resp(),
        )
        with pytest.raises(RuntimeError, match="refus"):
            OpenAIClient(model="gpt-6-astra", client=sdk).chat(
                self.MSGS, system="sys", response_schema=_Answer
            )
        assert sdk.chat.completions.parse.call_count == 1

    def test_length_stop_is_resampled(self):
        """A cap hit looked deterministic but is not: every one of five
        schema-call cap hits in the 2026-09-19 battery (DeepSeek @max, GLM
        @high) succeeded on the fresh sample the watchdog relaunch took.
        Resample in-process, bounded, instead of crashing the run."""
        sdk = _mock_openai(_resp(content="", finish_reason="length"), _resp(content="ok"))
        out = OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(self.MSGS, system="sys")
        assert out.text == "ok"
        assert sdk.chat.completions.create.call_count == 2

    def test_sdk_length_error_on_schema_path_is_resampled(self):
        """`.parse` raises openai.LengthFinishReasonError itself on a cap
        hit, before our loop sees the message; catch it as a bad sample."""
        import openai

        completion = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=5, completion_tokens=131072, completion_tokens_details=None))
        sdk = _mock_openai(
            openai.LengthFinishReasonError(completion=completion),
            _resp(parsed=_Answer(x=3), content='{"x": 3}'),
        )
        out = OpenAIClient(model="deepseek/deepseek-v4-pro-0813", client=sdk).chat(
            self.MSGS, system="sys", response_schema=_Answer
        )
        assert out.parsed == _Answer(x=3)
        assert sdk.chat.completions.parse.call_count == 2

    def test_repeated_length_gives_up_with_reason(self):
        import openai

        completion = SimpleNamespace(usage=None)
        sdk = _mock_openai(*[openai.LengthFinishReasonError(completion=completion)] * 5)
        with pytest.raises(RuntimeError, match="length"):
            OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(
                self.MSGS, system="sys", response_schema=_Answer
            )
        assert sdk.chat.completions.parse.call_count == 3

    def test_upstream_error_finish_reason_is_resampled(self):
        """OpenRouter reports a backend that died mid-generation as
        finish_reason='error' with an empty body (NextBit, deepseek-v4-pro,
        2026-09-19). That is transient, unlike 'length'."""
        sdk = _mock_openai(_resp(content="", finish_reason="error"), _resp(content="ok"))
        out = OpenAIClient(model="deepseek/deepseek-v4-pro-0813", client=sdk).chat(
            self.MSGS, system="sys"
        )
        assert out.text == "ok"
        assert sdk.chat.completions.create.call_count == 2

    def test_missing_finish_reason_is_resampled(self):
        """Some backends omit finish_reason on a bad sample; treat as 'stop'."""
        sdk = _mock_openai(_resp(content="", finish_reason=None), _resp(content="ok"))
        out = OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(self.MSGS, system="sys")
        assert out.text == "ok"
        assert sdk.chat.completions.create.call_count == 2

    def test_no_choices_error_body_is_resampled(self):
        """OpenRouter answers 200 with choices=[] and a top-level `error` when
        the backend fails mid-request; that is a bad sample, not a crash."""
        sdk = _mock_openai(
            _resp(choices=False, usage=False, error={"code": 502, "message": "upstream"}),
            _resp(content="fine"),
        )
        out = OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(
            self.MSGS, system="sys"
        )
        assert out.text == "fine"
        assert sdk.chat.completions.create.call_count == 2

    def test_missing_usage_does_not_crash(self):
        sdk = _mock_openai(_resp(content="ok", usage=False))
        out = OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(
            self.MSGS, system="sys"
        )
        assert out.text == "ok"
        assert out.usage == {}

    def test_non_empty_parse_failure_is_not_resampled(self):
        """Invalid-but-non-empty JSON is a model error the caller sees once."""
        sdk = _mock_openai(_resp(parsed=None, content="not json"), _resp())
        with pytest.raises(RuntimeError, match="parse failed"):
            OpenAIClient(model="z-ai/glm-5.3", client=sdk).chat(
                self.MSGS, system="sys", response_schema=_Answer
            )
        assert sdk.chat.completions.parse.call_count == 1


class TestModelPathTag:
    def test_slash_becomes_dash(self):
        assert model_path_tag("deepseek/deepseek-v4-pro-0813") == "deepseek-deepseek-v4-pro-0813"
        assert model_path_tag("z-ai/glm-5.3") == "z-ai-glm-5.3"

    def test_plain_ids_unchanged(self):
        assert model_path_tag("gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"
        assert model_path_tag("claude-opus-5") == "claude-opus-5"
