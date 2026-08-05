"""Shared, non-oracle theory selection for the proposer-comparison analyses.

Every reported outcome/speed number selects "the best theory the pipeline
itself would pick at round r" — never the theory that happens to score best
against the (unknown at run time) ground truth:

  * survivors at round r  = un-killed starting theories of that round, plus
    the round's replacement (rounds/round_{r:03d}/theories.json);
  * scores at round r     = the round-r post-admit leaderboard block
    (post-data fallback for rounds that end without an admission), parsed
    from leaderboard.md — verified loss-ordered and written AFTER backfill;
  * best surviving        = highest-scored survivor; a surviving label absent
    from the block ranks last (score -1).

Used by scripts/proposer_outcome_speed.py, scripts/proposer_speed_trace.py,
and scripts/proposer_perseveration_behavior.py. Behavior pinned by
tests/test_proposer_selection.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.theory import Theory

_ROW_RE = re.compile(r"#\s*\d+\s+(\S+)\s+([\d.]+)")


def survivors_at(run_dir: Path, r: int) -> dict[str, Theory]:
    """Un-killed starting theories + replacement of round `r`, by label."""
    d = json.loads(
        (Path(run_dir) / "rounds" / f"round_{r:03d}" / "theories.json").read_text()
    )
    out = {
        s["label"]: Theory.model_validate(s["theory"])
        for s in d["starting_theories"]
        if not s.get("killed", False)
    }
    if d.get("replacement"):
        out[d["replacement"]["label"]] = Theory.model_validate(
            d["replacement"]["theory"]
        )
    return out


def scores_at(run_dir: Path, r: int) -> dict[str, float]:
    """Label -> leaderboard score from the round-`r` post-admit block
    (post-data fallback). Empty dict if neither block exists."""
    txt = (Path(run_dir) / "leaderboard.md").read_text()
    for tag in ("post-admit", "post-data"):
        m = re.search(rf"## round {r} — {tag}.*?```(.*?)```", txt, re.DOTALL)
        if m:
            return {lab: float(s) for lab, s in _ROW_RE.findall(m.group(1))}
    return {}


def best_surviving_at(run_dir: Path, r: int) -> tuple[str, Theory]:
    """(label, theory) of the highest-scored survivor at round `r`."""
    surv = survivors_at(run_dir, r)
    sc = scores_at(run_dir, r)
    label = max(surv, key=lambda l: sc.get(l, -1.0))
    return label, surv[label]


def rounds_to_first_appearance(
    run_dir: Path, label: str, *, max_round: int | None = None
) -> int | None:
    """First round index (0-based) at which `label` entered the pool, scanning
    rounds 0..max_round-1 (all rounds when None). None if never found."""
    rounds = sorted((Path(run_dir) / "rounds").glob("round_*"))[:max_round]
    for i, rd in enumerate(rounds):
        d = json.loads((rd / "theories.json").read_text())
        labs = [t["label"] for t in d["starting_theories"]]
        if d.get("replacement"):
            labs.append(d["replacement"]["label"])
        if label in labs:
            return i
    return None
