"""Tests for the transient-error retry layer in `src.llm._call_with_retry`.

Long Della runs can't afford to die on a single `httpx.RemoteProtocolError`.
These tests pin the contract: transients are retried with escalating
backoff, non-transients propagate immediately, and (most important for
multi-hour runs) the backoff resets between independent invocations so
a recovered run doesn't keep paying for an earlier blip.
"""
from __future__ import annotations

import httpx

from src.llm import _call_with_retry


def test_retry_returns_after_transient_failures():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.RemoteProtocolError("disconnected")
        return "ok"

    out = _call_with_retry(flaky, base_delay=0.0, max_delay=0.0, label="t-flaky")
    assert out == "ok"
    assert attempts["n"] == 3


def test_retry_gives_up_after_max_attempts():
    attempts = {"n": 0}

    def always_fail():
        attempts["n"] += 1
        raise httpx.RemoteProtocolError("boom")

    try:
        _call_with_retry(
            always_fail,
            max_attempts=3,
            base_delay=0.0,
            max_delay=0.0,
            label="t-always",
        )
    except httpx.RemoteProtocolError:
        pass
    else:
        raise AssertionError("expected RemoteProtocolError to propagate")
    assert attempts["n"] == 3


def test_non_transient_propagates_immediately():
    attempts = {"n": 0}

    def schema_error():
        attempts["n"] += 1
        raise ValueError("bad schema")

    try:
        _call_with_retry(schema_error, base_delay=0.0, max_delay=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError to propagate")
    assert attempts["n"] == 1, "non-transient error must not be retried"


def test_default_tolerance_survives_multi_minute_outage(monkeypatch):
    """A multi-hour SBATCH job must not die on a brief Gemini outage.

    Concretely: with default settings, the retry layer should keep trying
    for at least ~5 minutes of cumulative backoff before giving up. That's
    the property that distinguishes "resilient to a network blip" from
    "kills an 8-hour run because the server hiccupped for 60 seconds."
    """
    import src.llm as _llm

    sleeps: list[float] = []
    monkeypatch.setattr(_llm._time, "sleep", lambda d: sleeps.append(d))
    monkeypatch.setattr(_llm._random, "uniform", lambda a, b: 0.0)

    attempts = {"n": 0}

    def always_fail():
        attempts["n"] += 1
        raise httpx.RemoteProtocolError("server gone")

    try:
        _call_with_retry(always_fail, label="t-tolerance")
    except httpx.RemoteProtocolError:
        pass
    else:
        raise AssertionError("expected RemoteProtocolError to propagate")

    total = sum(sleeps)
    assert total >= 300.0, (
        f"default retry budget too small for long runs: {total:.1f}s "
        f"across {len(sleeps)} sleeps ({sleeps})"
    )


def test_backoff_resets_between_independent_calls(monkeypatch):
    """After a call recovers, the NEXT call's first delay must be `base_delay`
    again — not whatever the previous call escalated to. This is the property
    that keeps a long-running Della job responsive after a network blip."""
    import src.llm as _llm

    sleeps: list[float] = []
    monkeypatch.setattr(_llm._time, "sleep", lambda d: sleeps.append(d))
    # Strip jitter so the asserted delay is exactly base_delay * 2**(attempt-1).
    monkeypatch.setattr(_llm._random, "uniform", lambda a, b: 0.0)

    # Call A: fails 3 times, succeeds on the 4th. Sleeps escalate: 2, 4, 8.
    a_attempts = {"n": 0}

    def a_flaky():
        a_attempts["n"] += 1
        if a_attempts["n"] < 4:
            raise httpx.RemoteProtocolError("a-down")
        return "a-ok"

    assert _call_with_retry(a_flaky, base_delay=2.0, max_delay=60.0) == "a-ok"
    assert sleeps == [2.0, 4.0, 8.0], sleeps

    # Call B: a brand-new invocation. If backoff state leaked, the first
    # sleep would be 16.0 (continuing the escalation). The contract is that
    # it resets to base_delay (2.0).
    sleeps.clear()
    b_attempts = {"n": 0}

    def b_flaky():
        b_attempts["n"] += 1
        if b_attempts["n"] < 2:
            raise httpx.RemoteProtocolError("b-down")
        return "b-ok"

    assert _call_with_retry(b_flaky, base_delay=2.0, max_delay=60.0) == "b-ok"
    assert sleeps == [2.0], (
        f"expected fresh call to start at base_delay=2.0, got {sleeps}"
    )


def test_default_backoff_tolerates_long_provider_outage():
    """Pin the worst-case outage the *defaults* can ride out.

    On 2026-07-04 a ~12h Della Centaur run (job 10658911) died at round 22/25
    because Gemini returned `RemoteProtocolError` for longer than the retry
    window: 10 attempts capped at 120s ≈ 8.5 minutes of tolerance. A provider
    outage measured in tens of minutes is routine, so the defaults must keep
    a multi-hour GPU job alive through at least a 45-minute outage.
    """
    import inspect

    sig = inspect.signature(_call_with_retry)
    max_attempts = sig.parameters["max_attempts"].default
    base_delay = sig.parameters["base_delay"].default
    max_delay = sig.parameters["max_delay"].default

    # Sleeps happen after attempts 1..max_attempts-1 (the last attempt raises).
    total_tolerance = sum(
        min(max_delay, base_delay * 2 ** (attempt - 1))
        for attempt in range(1, max_attempts)
    )
    assert total_tolerance >= 45 * 60, (
        f"defaults tolerate only {total_tolerance / 60:.1f} min of provider "
        f"outage; need >= 45 min so a transient API failure can't kill a "
        f"12-hour H200 run"
    )
