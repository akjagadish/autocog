"""Every shipped entry point must import cleanly on a fresh checkout.

This is the regression test for "the repo is runnable": an entry point that
raises at import time (stale module path, renamed YAML, hard-coded absolute
path to somebody's laptop) is broken for every user before it parses a single
CLI flag. Importing is a weak check of *behaviour* but a strong check of
*wiring*, and wiring is exactly what rots when a project is reorganised.

Two complementary tests keep the claim honest:

``test_entry_point_imports``
    Every entry point NOT listed in ``UNAVAILABLE`` must import. This is the
    set we promise is runnable.

``test_unavailable_entry_point_still_unavailable``
    Every entry point listed in ``UNAVAILABLE`` must STILL fail, and fail for
    the documented reason. Without this, the exclusion list silently becomes a
    junk drawer: a script could start failing for a brand-new reason, or start
    working, and nobody would notice. Pinning the reason turns each exclusion
    into a claim that has to stay true.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Each entry point is imported in its OWN interpreter, from the repo root, with
# nothing added to sys.path. That is deliberate and load-bearing:
#
#   * `tests/conftest.py` puts the repo root on sys.path for the suite. If we
#     imported in-process, every script would inherit that and a script that
#     forgot its own `sys.path` bootstrap would still pass here while failing
#     for a real user running `python scripts/foo.py`.
#   * Several scripts prepend to `sys.path` at import time. In-process, the
#     first one to run would silently satisfy the imports of every later one,
#     making results order-dependent.
#
# The runner loads the file under a non-`__main__` module name, so an
# `if __name__ == "__main__":` block does not execute. It reports the outcome
# on the last line of stdout so the parent can tell a clean import from a
# swallowed one.
#
# argv is set to `[<script>, "--help"]`. Several entry points build their
# argparse parser at module scope, so importing them with no arguments exits 2
# ("the following arguments are required") even though the module is perfectly
# fine. With `--help` those exit 0 instead, which keeps the success signal
# unambiguous: a module that still exits non-zero — say `sys.exit("missing
# data file")` — is genuinely broken rather than merely argument-hungry.
_RUNNER = """
import importlib.util, os, sys
path = sys.argv[1]
sys.argv = [path, "--help"]
# Reproduce `python <script>` exactly: the interpreter puts the SCRIPT's own
# directory on sys.path[0], not the working directory. Using `python -c` would
# instead put the cwd there, which silently hands the repo root to every script
# under scripts/ and hides the ones that never bootstrap sys.path themselves.
sys.path[0] = os.path.dirname(os.path.abspath(path))
spec = importlib.util.spec_from_file_location("entry_point_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules["entry_point_under_test"] = module
try:
    spec.loader.exec_module(module)
except SystemExit as exc:
    print("\\n__RESULT__ SYSTEMEXIT " + repr(exc.code))
except BaseException as exc:
    print("\\n__RESULT__ RAISED " + type(exc).__name__ + ": " + str(exc).replace(chr(10), " "))
else:
    print("\\n__RESULT__ OK")
"""

# Entry points that cannot import in a plain checkout, each mapped to a
# substring that must appear in the raised exception. Keep the reason specific
# enough that a *different* failure fails the test.
UNAVAILABLE: dict[str, str] = {
    # Centaur (Llama-3.1-Centaur-70B) loader left over from the pre-autocog
    # tree; `unsloth` is a GPU fine-tuning dependency that is not in
    # requirements.txt and nothing in the repo imports this module.
    "agents.py": "unsloth",
    # Reads the online experiment design through sweetbean; vendored under
    # vendor/sweetbean but only importable once installed.
    "scripts/preregistration/download_data.py": "sweetbean",
}


def _entry_points() -> list[str]:
    """Repo-relative paths of every runnable Python file we ship."""
    paths = (
        sorted(ROOT.glob("main*.py"))
        + sorted(ROOT.glob("agents.py"))
        + sorted((ROOT / "scripts").glob("*.py"))
        + sorted((ROOT / "scripts" / "preregistration").glob("*.py"))
    )
    return [str(p.relative_to(ROOT)) for p in paths]


ENTRY_POINTS = _entry_points()
IMPORTABLE = [p for p in ENTRY_POINTS if p not in UNAVAILABLE]


def _probe(rel_path: str) -> str:
    """Import one entry point in a clean subprocess; return its outcome line.

    ``"OK"`` on a clean import. ``"SYSTEMEXIT <code>"`` when the module exits
    at import time — only code ``0``/``None`` counts as success, so a module
    that bails with ``sys.exit("missing data file")`` is a failure rather than
    a pass. ``"RAISED <Type>: <msg>"`` for anything else.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(ROOT / rel_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__ "):
            return line.removeprefix("__RESULT__ ").strip()
    return f"NO_RESULT (exit {proc.returncode}): {proc.stderr.strip()[-400:]}"


@pytest.fixture(scope="session")
def probes() -> dict[str, str]:
    """Probe every entry point once, concurrently, and share across tests."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_probe, ENTRY_POINTS))
    return dict(zip(ENTRY_POINTS, outcomes))


def _is_success(outcome: str) -> bool:
    return outcome == "OK" or outcome in ("SYSTEMEXIT 0", "SYSTEMEXIT None")


def test_entry_points_discovered() -> None:
    """Guard the guard: an empty or tiny glob would make the suite vacuous."""
    assert len(ENTRY_POINTS) > 50, f"only found {len(ENTRY_POINTS)} entry points"
    assert "main_ablation_binary.py" in ENTRY_POINTS
    assert "scripts/recovery_correlation.py" in ENTRY_POINTS


@pytest.mark.parametrize("rel_path", IMPORTABLE)
def test_entry_point_imports(rel_path: str, probes: dict[str, str]) -> None:
    outcome = probes[rel_path]
    assert _is_success(outcome), f"`python {rel_path}` fails at import: {outcome}"


@pytest.mark.parametrize("rel_path", sorted(UNAVAILABLE))
def test_unavailable_entry_point_still_unavailable(
    rel_path: str, probes: dict[str, str]
) -> None:
    """A documented-broken script must still be broken, for its stated reason.

    If this fails with an OK outcome, the script has been fixed — delete its
    ``UNAVAILABLE`` entry so it joins the importable set. If it fails on the
    substring, the reason drifted and the note above it is now a lie.
    """
    assert (ROOT / rel_path).exists(), f"{rel_path} is listed but does not exist"
    outcome = probes[rel_path]
    assert not _is_success(outcome), (
        f"{rel_path} now imports cleanly — remove it from UNAVAILABLE"
    )
    reason = UNAVAILABLE[rel_path]
    assert reason in outcome, (
        f"{rel_path} failed with {outcome!r}, "
        f"which does not mention the documented reason {reason!r}"
    )
