"""Tests for GeminiClient's structured-output parse fallback.

Gemini's google-genai SDK occasionally returns `resp.parsed is None` even
when `resp.text` is perfectly valid JSON. The trigger we hit in practice:
the model emits a float in exponent form (e.g. `0.40000000000000013e0`),
which is valid JSON (stdlib `json.loads` accepts it) but the SDK's own
parser rejects, leaving `parsed=None`. Without a fallback this aborts the
whole run.

These tests pin the contract: when a schema is requested and the SDK fails
to parse its own valid-JSON text, the client recovers by parsing the text
itself and validating against the schema. Genuinely broken/truncated text
(e.g. a MAX_TOKENS cutoff) must still raise.
"""
from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from src.llm import GeminiClient


class _Experiment(BaseModel):
    validities: list[float]


def _fake_client(resp):
    """A stand-in for genai.Client whose generate_content returns `resp`."""
    return SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: resp)
    )


def test_parse_fallback_recovers_valid_json_when_sdk_returns_none():
    # Valid JSON the SDK's parser chokes on: a float in `e0` exponent form.
    raw = '{"validities": [0.9, 0.8, 0.7, 0.6, 0.5, 0.40000000000000013e0]}'
    resp = SimpleNamespace(
        parsed=None,
        text=raw,
        usage_metadata=None,
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    client = GeminiClient(model="gemini-3.1-pro-preview", client=_fake_client(resp))

    result = client.chat(messages=[{"role": "user", "content": "hi"}],
                         system="sys", response_schema=_Experiment)

    assert isinstance(result.parsed, _Experiment)
    assert result.parsed.validities[-1] == 0.40000000000000013


def test_truncated_text_still_raises():
    # A MAX_TOKENS cutoff yields incomplete JSON that cannot be recovered.
    resp = SimpleNamespace(
        parsed=None,
        text='{"validities": [0.9, 0.8, 0.41000000000',  # truncated
        usage_metadata=None,
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )
    client = GeminiClient(model="gemini-3.1-pro-preview", client=_fake_client(resp))

    try:
        client.chat(messages=[{"role": "user", "content": "hi"}],
                    system="sys", response_schema=_Experiment)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on unrecoverable truncated text")
