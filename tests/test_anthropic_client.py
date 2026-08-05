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
