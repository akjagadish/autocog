"""The test suite must not modify the committed `results/` tree.

`results/` is the paper's record: every figure and number reported in the
manuscript is regenerated from it. A test that writes there — usually by
forgetting one output flag and inheriting a default that points inside
`results/` — silently rewrites that record. The damage is easy to miss because
the test still passes and matplotlib re-renders look identical at a glance,
while the bytes differ (an embedded timestamp plus randomized element ids).

This test walks the suite's own source looking for that mistake rather than
running it: it checks that no test hands an analysis script a path under
`results/` to write to, and that the two known offenders keep their fixes.

It is deliberately a static check. Detecting mutation dynamically would mean
running the whole suite twice under a git-clean tree, which is far too slow to
belong inside the suite it is checking; `scripts/smoke_analyses.sh` covers the
dynamic case for the analyses themselves.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# Flags whose value is an output path. If a test passes one of these a literal
# under results/, that test writes into the paper's record.
OUTPUT_FLAGS = (
    "--out", "--out-dir", "--out_dir", "--csv", "--mse-out", "--autocorr-out",
    "--eval-out", "--exp-out", "--out-path", "--out_path", "--out-name",
)


def _test_sources() -> list[tuple[Path, str]]:
    return [
        (p, p.read_text())
        for p in sorted(TESTS.glob("test_*.py"))
        if p.name != Path(__file__).name
    ]


def test_no_test_passes_an_output_flag_a_path_under_results() -> None:
    """An output flag must never be given a literal `results/...` destination."""
    offenders: list[str] = []
    flag_alt = "|".join(re.escape(f) for f in OUTPUT_FLAGS)
    # e.g.  "--autocorr-out", "results/recovery/x.png"
    pattern = re.compile(
        rf'["\']({flag_alt})["\']\s*,\s*(?:str\()?\s*["\']results/',
    )
    for path, src in _test_sources():
        for lineno, line in enumerate(src.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these tests direct an analysis script's output into the committed "
        "results/ tree:\n  " + "\n  ".join(offenders)
        + "\n(point them at pytest's tmp_path instead)"
    )


def test_recovery_correlation_end_to_end_redirects_every_output() -> None:
    """Regression: `--autocorr-out` defaults into `results/recovery/`.

    `test_main_end_to_end_writes_csv_and_png` passed --csv/--out/--mse-out to
    tmp_path but omitted --autocorr-out, so each run overwrote the committed
    results/recovery/recovery_autocorr.{png,svg}.
    """
    src = (TESTS / "test_recovery_correlation.py").read_text()
    body = src[src.index("def test_main_end_to_end_writes_csv_and_png"):]
    body = body[: body.index("\ndef ")]
    for flag in ("--csv", "--out", "--mse-out", "--autocorr-out"):
        assert flag in body, f"{flag} is not redirected; it will default into results/"
    assert "tmp_path" in body


def test_preregistration_driver_test_restores_the_bundle() -> None:
    """Regression: the bundle's plot modules hardcode their own output/ dir.

    `render_layout` therefore always writes into the committed bundle. The test
    must snapshot and restore it.
    """
    src = (TESTS / "test_plot_preregistration_figures.py").read_text()
    body = src[src.index("def test_driver_renders_a_layout_with_png_svg_pdf"):]
    body = body[: body.index("\ndef ")]
    assert "finally:" in body, "restore must be in a finally block"
    assert "copytree" in body, "expected a snapshot of the bundle output dir"


def test_the_guard_covers_the_flags_scripts_actually_use() -> None:
    """Keep OUTPUT_FLAGS in step with the flags the analyses expose.

    If an analysis grows a new output flag that this list does not know about,
    the check above goes quietly blind to it.
    """
    declared: set[str] = set()
    for path in (ROOT / "scripts").glob("*.py"):
        src = path.read_text()
        # "out" as a --/-/_ delimited token, so --layout is not a false positive.
        for m in re.finditer(
            r'add_argument\(\s*["\'](--(?:[a-z_-]*[-_])?out(?:[-_][a-z_-]*)?)["\']', src
        ):
            declared.add(m.group(1))
    unknown = declared - set(OUTPUT_FLAGS)
    assert not unknown, (
        f"scripts expose output flags this check does not track: {sorted(unknown)}"
        " — add them to OUTPUT_FLAGS"
    )
