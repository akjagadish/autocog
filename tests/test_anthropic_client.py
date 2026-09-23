"""Tests for the AnthropicClient in src/llm.py."""
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llm import AnthropicClient, ChatResult, _extract_json, make_client
from src.config import LLMConfig


class DummySchema(BaseModel):
    answer: str
    score: float


def _fake_message(text: str, input_tokens: int = 10, output_tokens: int = 20):
    """Build a mock anthropic Message object."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    msg = MagicMock()
    msg.content = [block]
    msg.usage = usage
    return msg


def _mock_sdk_with_stream(text: str, **msg_kwargs):
    """Build a mock SDK where .messages.stream() returns a context manager
    whose .get_final_message() yields a fake Message."""
    mock_sdk = MagicMock()
    fake_msg = _fake_message(text, **msg_kwargs)
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    stream_ctx.get_final_message.return_value = fake_msg
    mock_sdk.messages.stream.return_value = stream_ctx
    return mock_sdk


class TestAnthropicClient:
    def test_chat_returns_chat_result(self):
        mock_sdk = _mock_sdk_with_stream("hello world")

        client = AnthropicClient(
            model="claude-haiku-4-5-20251001",
            client=mock_sdk,
        )
        result = client.chat(
            messages=[{"role": "user", "content": "hi"}],
            system="you are helpful",
        )

        assert isinstance(result, ChatResult)
        assert result.text == "hello world"
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 20

    def test_chat_passes_system_and_messages(self):
        mock_sdk = _mock_sdk_with_stream("ok")

        client = AnthropicClient(
            model="claude-haiku-4-5-20251001",
            client=mock_sdk,
            temperature=0.5,
            max_tokens=2048,
        )
        client.chat(
            messages=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "partial"},
                {"role": "user", "content": "followup"},
            ],
            system="be concise",
        )

        call_kwargs = mock_sdk.messages.stream.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["system"] == "be concise"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 2048
        assert len(call_kwargs["messages"]) == 3

    def test_chat_with_response_schema_parses_json(self):
        payload = json.dumps({"answer": "yes", "score": 0.95})
        mock_sdk = _mock_sdk_with_stream(payload)

        client = AnthropicClient(
            model="claude-haiku-4-5-20251001",
            client=mock_sdk,
        )
        result = client.chat(
            messages=[{"role": "user", "content": "evaluate"}],
            system="judge",
            response_schema=DummySchema,
        )

        assert isinstance(result.parsed, DummySchema)
        assert result.parsed.answer == "yes"
        assert result.parsed.score == 0.95

    def test_chat_with_schema_strips_code_fences(self):
        payload = json.dumps({"answer": "yes", "score": 0.95})
        fenced = f"```json\n{payload}\n```\n\nSome trailing commentary."
        mock_sdk = _mock_sdk_with_stream(fenced)

        client = AnthropicClient(
            model="claude-haiku-4-5-20251001",
            client=mock_sdk,
        )
        result = client.chat(
            messages=[{"role": "user", "content": "evaluate"}],
            system="judge",
            response_schema=DummySchema,
        )

        assert isinstance(result.parsed, DummySchema)
        assert result.parsed.answer == "yes"
        assert result.parsed.score == 0.95

    def test_chat_with_schema_lets_validation_errors_propagate(self):
        """Valid JSON that breaks the schema's own invariants is a model
        error the caller may retry (AutoCog.propose_round re-proposes on
        pydantic ValidationError). claude-opus-5 returned an experiment with
        empty validities (2026-09-20); wrapping it in RuntimeError hid it
        from that retry and crashed the run."""
        from pydantic import BaseModel, ValidationError, field_validator

        class Strict(BaseModel):
            score: float

            @field_validator("score")
            @classmethod
            def _bounded(cls, v):
                if not 0 <= v <= 1:
                    raise ValueError("score must be in [0, 1]")
                return v

        mock_sdk = _mock_sdk_with_stream(json.dumps({"score": 5.0}))
        client = AnthropicClient(model="claude-opus-5", client=mock_sdk)
        with pytest.raises(ValidationError, match="score must be in"):
            client.chat(
                messages=[{"role": "user", "content": "evaluate"}],
                system="judge",
                response_schema=Strict,
            )

    def test_chat_with_schema_raises_on_invalid_json(self):
        mock_sdk = _mock_sdk_with_stream("not json at all")

        client = AnthropicClient(
            model="claude-haiku-4-5-20251001",
            client=mock_sdk,
        )
        with pytest.raises(RuntimeError, match="structured output parse failed"):
            client.chat(
                messages=[{"role": "user", "content": "evaluate"}],
                system="judge",
                response_schema=DummySchema,
            )


class TestExtractJson:
    def test_plain_json_passthrough(self):
        raw = '{"answer": "yes"}'
        assert _extract_json(raw) == raw

    def test_strips_json_code_fence(self):
        raw = '```json\n{"answer": "yes"}\n```'
        assert _extract_json(raw) == '{"answer": "yes"}'

    def test_strips_plain_code_fence(self):
        raw = '```\n{"answer": "yes"}\n```'
        assert _extract_json(raw) == '{"answer": "yes"}'

    def test_strips_trailing_commentary(self):
        raw = '```json\n{"answer": "yes"}\n```\n\nHere is some explanation.'
        assert _extract_json(raw) == '{"answer": "yes"}'

    def test_bare_json_after_prose(self):
        raw = '## Analysis\n\nSome long explanation.\n\n{"answer": "yes", "score": 0.5}'
        assert _extract_json(raw) == '{"answer": "yes", "score": 0.5}'

    def test_skips_spurious_braces_in_prose(self):
        raw = (
            'The structure {not json} is interesting.\n\n'
            '{"answer": "yes", "score": 0.5}'
        )
        assert _extract_json(raw) == '{"answer": "yes", "score": 0.5}'

    def test_prefers_json_fence_over_earlier_prose_fence(self):
        """Earlier ``` prose ``` blocks must not shadow a later ```json block."""
        raw = (
            "Here is a sketch:\n\n"
            "```\n"
            "Dim0 = perfect rule: 0->Cat0\n"
            "```\n\n"
            "Now the JSON:\n\n"
            '```json\n{"answer": "yes", "score": 0.5}\n```'
        )
        assert _extract_json(raw) == '{"answer": "yes", "score": 0.5}'

    def test_falls_through_to_brace_scan_when_fence_truncated(self):
        """If the closing fence is truncated to 2 backticks, fall back to brace scan."""
        raw = 'Prose.\n\n```json\n{"answer": "yes", "score": 0.5}\n``'
        assert _extract_json(raw) == '{"answer": "yes", "score": 0.5}'


class TestMakeClientAnthropic:
    @patch("src.llm.load_dotenv")
    @patch("src.llm.anthropic")
    def test_make_client_creates_anthropic_client(self, mock_anthropic_mod, mock_dotenv):
        mock_anthropic_mod.Anthropic.return_value = MagicMock()
        cfg = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        client = make_client(cfg)
        assert isinstance(client, AnthropicClient)
        mock_dotenv.assert_called_once()


class TestAnthropicSamplingParams:
    """Newer Anthropic models (adaptive/always-on thinking generation) reject
    `temperature` with a 400 (`temperature` is deprecated for this model).
    Verified live on 2026-09-05: claude-sonnet-5 and claude-fable-5-1 reject
    it, claude-sonnet-4-6 accepts it."""

    @pytest.mark.parametrize(
        "model",
        ["claude-sonnet-5", "claude-fable-5-1", "claude-opus-5", "claude-opus-4-8"],
    )
    def test_omits_temperature_for_models_that_reject_it(self, model):
        mock_sdk = _mock_sdk_with_stream("ok")
        client = AnthropicClient(model=model, client=mock_sdk, temperature=0.7)
        client.chat(messages=[{"role": "user", "content": "q"}], system="s")
        call_kwargs = mock_sdk.messages.stream.call_args[1]
        assert "temperature" not in call_kwargs
        assert call_kwargs["model"] == model

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-haiku-4-5"])
    def test_keeps_temperature_for_models_that_accept_it(self, model):
        mock_sdk = _mock_sdk_with_stream("ok")
        client = AnthropicClient(model=model, client=mock_sdk, temperature=0.7)
        client.chat(messages=[{"role": "user", "content": "q"}], system="s")
        assert mock_sdk.messages.stream.call_args[1]["temperature"] == 0.7


class TestAnthropicRefusal:
    def test_refusal_stop_reason_raises_instead_of_returning_empty_text(self):
        """A safety-classifier refusal returns HTTP 200 with
        stop_reason='refusal' and no text block. Surfacing it as a clear error
        beats a downstream 'structured output parse failed' on empty text."""
        mock_sdk = _mock_sdk_with_stream("")
        msg = mock_sdk.messages.stream.return_value.get_final_message.return_value
        msg.content = []
        msg.stop_reason = "refusal"
        msg.stop_details = MagicMock(category="frontier_llm", explanation="x")
        client = AnthropicClient(model="claude-fable-5-1", client=mock_sdk)
        with pytest.raises(RuntimeError, match="refus"):
            client.chat(messages=[{"role": "user", "content": "q"}], system="s")


class TestAnthropicStructuredOutput:
    """Mirror the Gemini path: when a `response_schema` is given, the Anthropic
    call must enforce it natively (`output_config.format` JSON schema), not
    rely on the prose prompt. Without this, claude-sonnet-5 answered the
    experiment-proposal prompt in pure markdown (dry run, 2026-09-05) and the
    JSON extraction failed."""

    def test_passes_schema_as_output_format(self):
        payload = json.dumps({"answer": "yes", "score": 0.5})
        mock_sdk = _mock_sdk_with_stream(payload)
        client = AnthropicClient(model="claude-sonnet-5", client=mock_sdk)
        result = client.chat(
            messages=[{"role": "user", "content": "q"}],
            system="s",
            response_schema=DummySchema,
        )
        fmt = mock_sdk.messages.stream.call_args[1]["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert set(fmt["schema"]["properties"]) == {"answer", "score"}
        assert fmt["schema"]["additionalProperties"] is False
        assert result.parsed == DummySchema(answer="yes", score=0.5)

    def test_no_output_format_without_schema(self):
        mock_sdk = _mock_sdk_with_stream("free text")
        client = AnthropicClient(model="claude-sonnet-5", client=mock_sdk)
        client.chat(messages=[{"role": "user", "content": "q"}], system="s")
        assert "output_config" not in mock_sdk.messages.stream.call_args[1]


class TestAnthropicOutputBudget:
    """On Anthropic, thinking tokens count toward `max_tokens` (Gemini budgets
    thinking separately). With the YAML default of 32768, claude-opus-5 spent
    the whole budget thinking on 39/72 theory-generation calls (2026-09-05,
    stop_reason='max_tokens', zero text blocks), each retried 10x at ~$0.80.
    Successful calls had a median of 24.8k output tokens."""

    @patch("src.llm.load_dotenv")
    @patch("src.llm.anthropic")
    def test_make_client_requests_the_model_output_ceiling(self, mock_mod, _dotenv):
        mock_mod.Anthropic.return_value = MagicMock()
        cfg = LLMConfig(provider="anthropic", model="claude-opus-5", max_tokens=32768)
        client = make_client(cfg)
        assert client.max_tokens == 128_000

    def test_empty_text_at_max_tokens_raises_named_error(self):
        mock_sdk = _mock_sdk_with_stream("")
        msg = mock_sdk.messages.stream.return_value.get_final_message.return_value
        msg.content = []  # thinking only, no text block
        msg.stop_reason = "max_tokens"
        client = AnthropicClient(model="claude-opus-5", client=mock_sdk)
        with pytest.raises(RuntimeError, match="max_tokens"):
            client.chat(messages=[{"role": "user", "content": "q"}], system="s")

    @patch("src.llm.load_dotenv")
    @patch("src.llm.anthropic")
    def test_haiku_gets_its_lower_ceiling(self, mock_mod, _dotenv):
        mock_mod.Anthropic.return_value = MagicMock()
        client = make_client(LLMConfig(provider="anthropic", model="claude-haiku-4-5"))
        assert client.max_tokens == 64_000


class TestAnthropicMaxTokensResample:
    """Under grammar-constrained structured output, claude-opus-5 once fell into
    a repetition loop (`\\n` escapes inside a JSON string) until the 128k cap
    (2026-09-05, feedback critique, $3.20, run crashed). A max_tokens stop is a
    sampling accident, so resample before giving up."""

    def _sdk_with_sequence(self, messages):
        mock_sdk = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.get_final_message.side_effect = messages
        mock_sdk.messages.stream.return_value = ctx
        return mock_sdk

    def _truncated(self):
        m = _fake_message('{"answer": "ye')
        m.stop_reason = "max_tokens"
        return m

    def _good(self):
        m = _fake_message(json.dumps({"answer": "yes", "score": 0.5}))
        m.stop_reason = "end_turn"
        return m

    def test_resamples_after_max_tokens_and_returns_good_answer(self):
        mock_sdk = self._sdk_with_sequence([self._truncated(), self._good()])
        client = AnthropicClient(model="claude-opus-5", client=mock_sdk)
        result = client.chat(
            messages=[{"role": "user", "content": "q"}], system="s",
            response_schema=DummySchema,
        )
        assert result.parsed == DummySchema(answer="yes", score=0.5)
        assert mock_sdk.messages.stream.call_count == 2

    def test_gives_up_after_bounded_resamples(self):
        mock_sdk = self._sdk_with_sequence([self._truncated()] * 5)
        client = AnthropicClient(model="claude-opus-5", client=mock_sdk)
        with pytest.raises(RuntimeError, match="max_tokens"):
            client.chat(
                messages=[{"role": "user", "content": "q"}], system="s",
                response_schema=DummySchema,
            )
        assert mock_sdk.messages.stream.call_count == 3  # 1 + 2 resamples
