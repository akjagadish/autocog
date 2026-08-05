"""
Resolve parameter templates (env context, expressions) and draw one uniform
sample per leaf. Search space: ``[lo, hi]`` (interval) or ``{a, b, …}`` (finite
set), then ``interval`` / ``choices`` dicts → ``random.uniform`` / ``random.choice``.
"""

from __future__ import annotations

import ast
import json
import math
import random
from typing import Any

import numpy as np


def parameter_context_from_env(env: Any) -> dict[str, Any]:
    fn = getattr(env, "parameter_context", None)
    if callable(fn):
        return dict(fn())
    return {}



def llm_interval_or_set_shorthand(s: str) -> dict[str, Any] | None:
    """Whole string only: ``[lo, hi]`` → interval, ``{a, b, …}`` → choices."""
    t = s.strip()
    if len(t) < 2 or t[0] not in "[{":
        return None
    try:
        v = ast.literal_eval(t)
    except (ValueError, SyntaxError):
        return None
    if t[0] == "[":
        return {"interval": v} if isinstance(v, list) and len(v) == 2 else None
    return {"choices": sorted(v)} if isinstance(v, set) else None


def resolve_parameters(params: Any, context: dict[str, Any]) -> Any:
    """Recursively materialize ``params`` using ``context`` (see module doc)."""
    eval_ns = {"__builtins__": {}}
    eval_ctx: dict[str, Any] = {"np": np, "math": math, **context}

    def _eval_str(s: str) -> Any:
        t = s.strip()
        # Strip angle brackets that LLMs sometimes add around variable names
        if t.startswith("<") and t.endswith(">") and t.count("<") == 1:
            t = t[1:-1].strip()
        llm = llm_interval_or_set_shorthand(t)
        if llm is not None:
            return llm
        try:
            return ast.literal_eval(t)
        except (ValueError, SyntaxError):
            pass
        try:
            return eval(t, eval_ns, eval_ctx)
        except Exception:
            return s

    def _resolve(v: Any) -> Any:
        if isinstance(v, str):
            return _eval_str(v)
        if isinstance(v, dict):
            return {k: _resolve(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_resolve(x) for x in v]
        return v

    return _resolve(params)


def draw_parameter_samples(params: Any) -> Any:
    """
    One uniform draw per search-space leaf: ``interval``, ``choices``;
    recurse into nested dicts/lists.
    """
    if isinstance(params, tuple) and len(params) == 2:
        lo, hi = float(params[0]), float(params[1])
        if hi < lo:
            lo, hi = hi, lo
        if lo == hi:
            return lo
        return random.uniform(lo, hi)
    if isinstance(params, dict):
        if set(params.keys()) == {"interval"}:
            iv = params["interval"]
            if isinstance(iv, list) and len(iv) == 2:
                lo, hi = float(iv[0]), float(iv[1])
                if hi < lo:
                    lo, hi = hi, lo
                if lo == hi:
                    return lo
                return random.uniform(lo, hi)
        if set(params.keys()) == {"choices"}:
            ch = params["choices"]
            if isinstance(ch, list) and len(ch) == 0:
                raise ValueError("choices must be a non-empty list")
            if isinstance(ch, list):
                return random.choice(ch)
        return {k: draw_parameter_samples(v) for k, v in params.items()}
    if isinstance(params, list):
        return [draw_parameter_samples(v) for v in params]
    return params


def sample_parameters(
    raw: dict[str, Any],
    *,
    env: Any | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve ``raw`` with :func:`resolve_parameters`, then :func:`draw_parameter_samples`.

    Builds evaluation context from ``context`` and, when ``env`` is given,
    from :func:`parameter_context_from_env` (explicit ``context`` wins on key clashes).
    """
    ctx = dict(context) if context is not None else {}
    if env is not None:
        ctx = {**parameter_context_from_env(env), **ctx}
    resolved = resolve_parameters(raw, ctx)
    if not isinstance(resolved, dict):
        raise TypeError(
            f"resolved parameters must be a dict, got {type(resolved).__name__}"
        )
    out = draw_parameter_samples(resolved)
    if not isinstance(out, dict):
        raise TypeError(
            f"sampled parameters must be a dict, got {type(out).__name__}"
        )
    return out
