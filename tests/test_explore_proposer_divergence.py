"""Tests for scripts/explore_proposer_divergence.py (proposer-divergence metrics).

Each test pins one correctness property identified in code review:
  1. malformed LLM designs are skipped, never crash the analysis
  2. proposers are enumerated from theories.json labels, not directory-name
     pattern matching (a regex bug silently dropped pi_2_1 dirs once)
  3. each proposal's baseline is independent of other proposals (per-proposal
     seeding — no shared RNG stream)
  4. results are exactly reproducible across invocations (seeded simulations)
  5. parse failures are reported explicitly, with per-proposal records that
     cannot misalign
  6. max_round caps the analysis to the first N rounds (matched-budget
     comparisons against longer runs)
  7. partial round dirs (missing/invalid theories.json) are skipped and
     reported, not fatal
  8. _reshuffle_trials stays in lockstep with the experiment constructor's
     trial preparation (drift detection)
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.explore_proposer_divergence import (
    _parse_design,
    _reshuffle_trials,
    analyse,
)
from src.theory import Theory

THEORIES_DIR = Path("theories/heuristic_decision_making")

VALID_DESIGN = {
    "validities": [0.9, 0.7, 0.6, 0.55],
    "trial_a_ratings": [[1, 0, 0, 0]] * 4,
    "trial_b_ratings": [[0, 1, 1, 1]] * 4,
}
# 3 validities but 4-feature ratings: the experiment constructor must reject it
MALFORMED_DESIGN = {
    "validities": [0.9, 0.7, 0.6],
    "trial_a_ratings": [[1, 0, 0, 0]] * 4,
    "trial_b_ratings": [[0, 1, 1, 1]] * 4,
}


def _attempt_md(design: dict) -> str:
    return (
        "# experiment_attempt_00\n\n## System Prompt\n\nx\n\n"
        "## User Prompt\n\ny\n\n## Response\n\n"
        f"```json\n{json.dumps(design, indent=2)}\n```\n\n"
        '## Usage\n\n```json\n{"input_tokens": 1}\n```\n'
    )


def _write_attempt(round_dir: Path, label: str, design: dict) -> Path:
    prompts = round_dir / label / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    f = prompts / "experiment_attempt_00.md"
    f.write_text(_attempt_md(design))
    return f


def _write_round(
    run_dir: Path, round_name: str, *, labels, designs, write_theories=True
) -> Path:
    round_dir = run_dir / "rounds" / round_name
    round_dir.mkdir(parents=True, exist_ok=True)
    if write_theories:
        theories = [
            Theory.from_yaml(THEORIES_DIR / "ttb_sampling.yaml"),
            Theory.from_yaml(THEORIES_DIR / "tallying_sampling.yaml"),
        ]
        (round_dir / "theories.json").write_text(json.dumps({
            "round_idx": int(round_name.split("_")[-1]),
            "verdict": "new_theory",
            "starting_theories": [
                {"slot": i + 1, "label": lab, "killed": False,
                 "theory": json.loads(th.model_dump_json())}
                for i, (lab, th) in enumerate(zip(labels, theories))
            ],
            "replacement": None,
        }))
    for lab, design in zip(labels, designs):
        _write_attempt(round_dir, lab, design)
    return round_dir


def _make_run_dir(
    tmp_path: Path, *, labels=("pi_1", "pi_2_1"), designs=(VALID_DESIGN, VALID_DESIGN)
) -> Path:
    """One-round run dir whose slot labels include an improver-style label."""
    run_dir = tmp_path / "run"
    _write_round(run_dir, "round_000", labels=labels, designs=designs)
    return run_dir


def _parsed(out: dict) -> list[dict]:
    return [p for p in out["proposals"] if p["parsed"]]


def test_parse_design_valid_roundtrip(tmp_path):
    f = _write_attempt(tmp_path, "pi_1", VALID_DESIGN)
    exp = _parse_design(f)
    assert exp is not None
    assert list(exp.validities) == VALID_DESIGN["validities"]
    assert [list(r) for r in exp.trial_a_ratings] == VALID_DESIGN["trial_a_ratings"]


def test_parse_design_malformed_returns_none_not_raise(tmp_path):
    f = _write_attempt(tmp_path, "pi_1", MALFORMED_DESIGN)
    assert _parse_design(f) is None  # must not raise


def test_proposers_enumerated_from_theories_json_not_dirnames(tmp_path):
    run_dir = _make_run_dir(tmp_path)  # labels pi_1 and pi_2_1
    # impostor dir matching the pi_* shape but NOT a slot label of this round
    _write_attempt(run_dir / "rounds" / "round_000", "pi_99", VALID_DESIGN)
    out = analyse(run_dir, n_runs=30)
    assert out["n_designs_parsed"] == 2  # pi_2_1 included, pi_99 excluded
    assert sorted(p["label"] for p in out["proposals"]) == [
        "round_000/pi_1", "round_000/pi_2_1",
    ]


def test_baseline_independent_of_other_proposals(tmp_path):
    intact = _make_run_dir(tmp_path / "a")
    broken = _make_run_dir(tmp_path / "b", designs=(MALFORMED_DESIGN, VALID_DESIGN))
    out_intact = analyse(intact, n_runs=30)
    out_broken = analyse(broken, n_runs=30)
    assert out_intact["n_designs_parsed"] == 2
    assert out_broken["n_designs_parsed"] == 1
    # pi_2_1's proposal is identical in both run dirs; its raw/baseline/excess
    # must not depend on whether pi_1's design parsed. Exact equality: the
    # per-proposal seed depends only on (round, label, base_seed).
    rec_intact = [p for p in _parsed(out_intact) if p["label"].endswith("pi_2_1")][0]
    rec_broken = _parsed(out_broken)[0]
    assert rec_broken["label"] == "round_000/pi_2_1"
    assert rec_intact["raw_jsd"] == rec_broken["raw_jsd"]
    assert rec_intact["baseline_jsd"] == rec_broken["baseline_jsd"]
    assert rec_intact["excess_jsd"] == rec_broken["excess_jsd"]


def test_results_exactly_reproducible(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    out1 = analyse(run_dir, n_runs=30)
    out2 = analyse(run_dir, n_runs=30)
    assert out1["n_designs_parsed"] == 2  # guard: not vacuously comparing []
    assert out1["proposals"] == out2["proposals"]


def test_parse_failures_reported_and_records_consistent(tmp_path):
    run_dir = _make_run_dir(tmp_path, designs=(MALFORMED_DESIGN, VALID_DESIGN))
    out = analyse(run_dir, n_runs=30)
    assert out["n_parse_failures"] == 1
    assert out["n_designs_parsed"] == 1
    assert out["n_proposals"] == 2
    # per-proposal records: the unparsed one is present with null metrics
    unparsed = [p for p in out["proposals"] if not p["parsed"]]
    assert len(unparsed) == 1
    assert unparsed[0]["label"] == "round_000/pi_1"
    assert unparsed[0]["raw_jsd"] is None
    parsed = _parsed(out)
    assert len(parsed) == 1
    assert None not in (parsed[0]["raw_jsd"], parsed[0]["baseline_jsd"],
                        parsed[0]["excess_jsd"])


def test_max_round_caps_analysis(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    _write_round(run_dir, "round_001", labels=("pi_1", "pi_3"),
                 designs=(VALID_DESIGN, VALID_DESIGN))
    out_all = analyse(run_dir, n_runs=30)
    out_capped = analyse(run_dir, n_runs=30, max_round=1)
    assert out_all["n_proposals"] == 4
    assert out_capped["n_proposals"] == 2
    assert all(p["label"].startswith("round_000/") for p in out_capped["proposals"])


def test_partial_round_skipped_and_reported(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    # crashed round: prompts written, theories.json never written
    _write_round(run_dir, "round_001", labels=("pi_1", "pi_3"),
                 designs=(VALID_DESIGN, VALID_DESIGN), write_theories=False)
    out = analyse(run_dir, n_runs=30)
    assert out["skipped_rounds"] == ["round_001"]
    assert out["n_proposals"] == 2  # round_000 still fully analysed


def test_reshuffle_matches_constructor_trial_preparation(tmp_path):
    """Drift detection: _reshuffle_trials must rebuild exactly the multiset of
    trials the constructor prepared (same pairs, same repeat count) — only the
    order may differ. Catches drift in the levels construction (e.g. a future
    constructor dedup/normalization the helper doesn't mirror). Coverage
    boundary: _n_repeats itself cannot drift — the helper reads the
    constructor-set value instead of recomputing it."""
    f = _write_attempt(tmp_path, "pi_1", VALID_DESIGN)
    exp = _parse_design(f)
    constructor_trials = sorted(exp._trials)
    _reshuffle_trials(exp, seed=123)
    assert sorted(exp._trials) == constructor_trials
    assert len(exp._trials) == len(constructor_trials)
