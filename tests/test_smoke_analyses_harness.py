"""The smoke harness must stay in sync with the scripts it claims to cover.

`scripts/smoke_analyses.sh` is the answer to "can someone still run every
analysis behind a result in the paper?". Its failure mode is silent decay: a new
analysis script gets added, nobody wires it in, and the harness keeps reporting
all-green over a shrinking fraction of the codebase.

These tests don't execute the analyses — that is the harness's own job, and it
takes minutes. They check the two properties that make its all-green meaningful:
every shipped analysis script is either exercised or explicitly skipped, and the
harness genuinely protects the committed `results/` tree.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "smoke_analyses.sh"

# Scripts that are libraries or are named in a documented `skip` line rather
# than invoked. Each entry needs a reason, so this cannot become a junk drawer.
NOT_INVOKED: dict[str, str] = {
    "figure_style.py": "shared style module, imported not run",
    "judge_similarity.py": "needs a live LLM judge; covered by a documented skip",
    "aggregate_hilibig_battery.py": "consumes uncommitted per-run battery outputs; documented skip",
    "plot_hilibig_battery_aggregate.py": "same battery outputs; documented skip",
    "plot_hilibig_battery_correlation.py": "same battery outputs; documented skip",
    "smoke_analyses.sh": "the harness itself",
}


def _harness_text() -> str:
    return HARNESS.read_text()


def test_harness_exists_and_is_valid_bash() -> None:
    assert HARNESS.is_file(), f"{HARNESS} is missing"
    proc = subprocess.run(["bash", "-n", str(HARNESS)], capture_output=True, text=True)
    assert proc.returncode == 0, f"bash syntax error:\n{proc.stderr}"


def test_every_analysis_script_is_covered_or_documented() -> None:
    """No analysis script may be silently absent from the harness."""
    text = _harness_text()
    uncovered: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        name = path.name
        if name in NOT_INVOKED:
            continue
        if f"scripts/{name}" not in text:
            uncovered.append(name)
    assert not uncovered, (
        "these analysis scripts are neither run nor documented-as-skipped in "
        "scripts/smoke_analyses.sh:\n  " + "\n  ".join(uncovered)
        + "\n(add a `run` line, or a `skip` line plus a NOT_INVOKED entry here)"
    )


def test_not_invoked_entries_are_all_real_files() -> None:
    """Stale exclusions are as bad as missing ones."""
    for name in NOT_INVOKED:
        target = ROOT / "scripts" / name
        assert target.is_file(), f"NOT_INVOKED lists {name}, which does not exist"


def test_harness_declares_every_group_it_documents() -> None:
    """The usage line's group list must match the actual `group` calls."""
    text = _harness_text()
    documented = set(re.search(r"\(fig1\|([a-z0-9|]+)\)", text).group(0)
                     .strip("()").split("|"))
    declared = set(re.findall(r"^group (\w+)$", text, re.M))
    assert documented == declared, (
        f"usage line advertises {sorted(documented)} but the script defines "
        f"{sorted(declared)}"
    )


def test_harness_protects_the_committed_results_tree() -> None:
    """The non-destructive guard must be wired up, not just described.

    `results/` is the paper's record. The harness snapshots git's view of it and
    reverts what the run touched, via an EXIT trap — if that trap is ever
    dropped, a smoke run silently overwrites reported results.
    """
    text = _harness_text()
    assert "trap restore_results EXIT" in text, "the restore trap is not installed"
    assert "git status --porcelain results/" in text, "no before-snapshot of results/"
    assert "git checkout --" in text, "guard cannot revert modified files"
    # The stats table is the one output that would otherwise be clobbered even
    # with the trap in place if the run were killed uncleanly.
    assert "AUTOCOG_STATS_OUT" in text, "stats.py output is not redirected"


def test_guard_snapshot_survives_the_output_dir_cleanup() -> None:
    """Regression: the snapshot must not live under $OUT.

    On a clean run the script does `rm -rf "$OUT"` before the EXIT trap fires.
    When the before-snapshot lived at "$OUT/.results_before.txt" it was already
    gone by then, so `restore_results` found nothing to compare against and
    returned early — silently leaving every overwritten file in `results/`
    modified. The guard looked installed and did nothing.
    """
    text = _harness_text()
    snap_line = next(
        line for line in text.splitlines() if line.startswith("SNAP_BEFORE=")
    )
    assert "$OUT" not in snap_line, (
        f"the before-snapshot must not be stored under $OUT (it is deleted "
        f"before the EXIT trap runs): {snap_line}"
    )
    assert "mktemp" in snap_line, "expected the snapshot in its own temp file"
    # ...and the same for the after-snapshot taken inside restore_results.
    after_lines = [
        line for line in text.splitlines()
        if "autocog_smoke_after" in line or 'after="' in line
    ]
    assert after_lines, "restore_results does not create an after-snapshot"
    assert all("$OUT" not in line for line in after_lines), (
        "the after-snapshot must not be stored under $OUT either"
    )


def test_stats_script_honours_the_output_override() -> None:
    """`AUTOCOG_STATS_OUT` must actually be read by scripts/stats.py."""
    src = (ROOT / "scripts" / "stats.py").read_text()
    assert 'os.environ.get("AUTOCOG_STATS_OUT")' in src
    # and the default must still be the committed location
    assert '"results" / "stats"' in src


@pytest.mark.parametrize("group", ["fig1", "fig3", "fig4", "fig5", "si", "loop"])
def test_each_group_has_at_least_one_step(group: str) -> None:
    """A group that runs nothing would report a vacuous pass."""
    text = _harness_text()
    body = text.split(f"group {group}\n", 1)
    assert len(body) == 2, f"group {group} is not declared"
    # up to the next group declaration or the report section
    rest = re.split(r"^# -+ report", body[1], maxsplit=1, flags=re.M)[0]
    # Steps may be indented (the loop group gates them behind an if/else on
    # SMOKE_LLM), so allow leading whitespace.
    assert re.search(r"^\s*(run|skip) ", rest, re.M), f"group {group} has no steps"
