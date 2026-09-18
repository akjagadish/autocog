"""`Promptable.from_llm_text` must tolerate a prose preamble inside the fence.

Observed on 2026-09-05 (claude-sonnet-5, theory_generator free-text path):
the model returned

    ```json
    Just increase ALPHA_FIXED slightly from 0.15 to 0.25 ...

    {"theory": ..., "predict": ..., "policy": ..., "parameters": {...}}
    ```

The JSON was complete and valid, but a bare `json.loads` on the fence body
rejected it, wasting an LLM call per occurrence (2 of ~11 attempts in one
round). The client-side `_extract_json` already handles this shape.
"""
import json

import pytest

from src.theory import Theory

_PAYLOAD = {
    "theory": "People use one cue.",
    "predict": "def predict(parameters, stimulus, history):\n    return np.array([0.5, 0.5])",
    "policy": "def policy(probs):\n    return int(np.random.choice(2, p=probs))",
    "parameters": {"beta": "[0.1, 20.0]"},
    "rationale": "test",
}


def _assert_roundtrip(text: str) -> None:
    th = Theory.from_llm_text(text)
    assert th.description == _PAYLOAD["theory"]
    assert th.parameters == _PAYLOAD["parameters"]
    assert th.predict_source == _PAYLOAD["predict"]


def test_plain_fenced_json_still_parses():
    _assert_roundtrip("```json\n" + json.dumps(_PAYLOAD, indent=2) + "\n```")


def test_bare_json_still_parses():
    _assert_roundtrip(json.dumps(_PAYLOAD))


def test_prose_preamble_inside_fence():
    text = (
        "```json\n"
        "Just increase ALPHA_FIXED slightly from 0.15 to 0.25 as suggested.\n\n"
        + json.dumps(_PAYLOAD, indent=2)
        + "\n```"
    )
    _assert_roundtrip(text)


def test_prose_before_fence_with_braces_in_prose():
    text = (
        "Here is the {revised} model:\n\n```json\n"
        + json.dumps(_PAYLOAD)
        + "\n```\nTrailing notes."
    )
    _assert_roundtrip(text)


def test_no_json_at_all_still_raises():
    with pytest.raises(Exception):
        Theory.from_llm_text("## Theory\n\nProse only, no object here.")
