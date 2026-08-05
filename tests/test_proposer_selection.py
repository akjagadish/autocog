"""Tests for scripts/proposer_selection.py — the shared, non-oracle selection
logic used by every proposer-comparison analysis (outcome, speed, behavior).

Pins the rules the reported numbers depend on:
  * survivors at round r = un-killed starting theories + the replacement
  * scores come from that round's post-admit leaderboard block (post-data
    fallback), parsed by label
  * best = highest-scored survivor; labels absent from the block rank last
  * first-appearance round of a label (M3)
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.proposer_selection import (
    best_surviving_at,
    rounds_to_first_appearance,
    scores_at,
    survivors_at,
)
from src.theory import Theory

THEORIES_DIR = Path("theories/heuristic_decision_making")

LEADERBOARD = """\
## round 0 — post-admit (pi_3)

```
[round 0 | post-admit (pi_3)]
  # 1  pi_3  1.000  |████|  (new)
  # 2  pi_1  0.500  |██··|  Δ-0.100
  # 3  pi_2  0.000  |····|  Δ ·
```

## round 1 — post-data

```
[round 1 | post-data]
  # 1  pi_1  0.900  |████|
  # 2  pi_3  0.400  |██··|
```
"""


def _theory_dump() -> dict:
    return json.loads(
        Theory.from_yaml(THEORIES_DIR / "ttb_sampling.yaml").model_dump_json()
    )


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    th = _theory_dump()
    r0 = run / "rounds" / "round_000"
    r0.mkdir(parents=True)
    (r0 / "theories.json").write_text(json.dumps({
        "starting_theories": [
            {"slot": 1, "label": "pi_1", "killed": False, "theory": th},
            {"slot": 2, "label": "pi_2", "killed": True, "theory": th},
        ],
        "replacement": {"label": "pi_3", "slot": 2, "theory": th},
    }))
    r1 = run / "rounds" / "round_001"
    r1.mkdir(parents=True)
    (r1 / "theories.json").write_text(json.dumps({
        "starting_theories": [
            {"slot": 1, "label": "pi_1", "killed": False, "theory": th},
            {"slot": 2, "label": "pi_3", "killed": False, "theory": th},
        ],
        "replacement": None,
    }))
    (run / "leaderboard.md").write_text(LEADERBOARD)
    return run


def test_survivors_excludes_killed_includes_replacement(tmp_path):
    run = _make_run(tmp_path)
    assert sorted(survivors_at(run, 0)) == ["pi_1", "pi_3"]  # pi_2 killed
    assert sorted(survivors_at(run, 1)) == ["pi_1", "pi_3"]


def test_scores_parse_post_admit_then_post_data_fallback(tmp_path):
    run = _make_run(tmp_path)
    assert scores_at(run, 0) == {"pi_3": 1.0, "pi_1": 0.5, "pi_2": 0.0}
    # round 1 has no post-admit block — must fall back to post-data
    assert scores_at(run, 1) == {"pi_1": 0.9, "pi_3": 0.4}


def test_best_surviving_uses_round_scores(tmp_path):
    run = _make_run(tmp_path)
    label0, _ = best_surviving_at(run, 0)
    label1, _ = best_surviving_at(run, 1)
    assert label0 == "pi_3"  # 1.0 beats pi_1's 0.5; killed pi_2 ineligible
    assert label1 == "pi_1"  # ranking flipped at round 1


def test_label_absent_from_block_ranks_last(tmp_path):
    run = _make_run(tmp_path)
    # rewrite round-1 block without pi_3 — pi_3 survives but has no score row
    (run / "leaderboard.md").write_text(LEADERBOARD.replace(
        "  # 2  pi_3  0.400  |██··|\n", ""))
    label, theory = best_surviving_at(run, 1)
    assert label == "pi_1"
    assert theory is not None


def test_rounds_to_first_appearance(tmp_path):
    run = _make_run(tmp_path)
    assert rounds_to_first_appearance(run, "pi_1") == 0  # seed
    assert rounds_to_first_appearance(run, "pi_3") == 0  # round-0 replacement
    assert rounds_to_first_appearance(run, "pi_9") is None
