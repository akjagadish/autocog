"""Convert Prolific HDM JSONL observations to a Hilbig-2014-shaped exp.txt CSV.

Each line in the input JSONL has:
    subject_id, option_a_ratings, option_b_ratings, response

The Prolific HDM task used `rating_max=5` and validities
[0.95, 0.85, 0.75, 0.65] (per `observations/state.json`), both differing
from Hilbig 2014's binary, [0.9 0.8 0.7 0.6] design. This script preserves
the raw 1–5 rating values and emits the actual Prolific validities;
downstream consumers (e.g. `eval_hilbig.py`) can binarize or override
validities if they need to map onto Hilbig's exact design.

Output columns mirror `results/heuristic_decision_making/hilbig2014/exp1.txt`:
    , participant, task, trial, choice, reward, stimulus_0, stimulus_1,
    validities, item_type, tier

Usage:
    python scripts/jsonl_to_hilbig_exp.py \\
        results/.../observations/data/round_*.jsonl \\
        --out results/.../prolific_exp.txt
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _fmt_int_array(xs: list[int]) -> str:
    return "[" + " ".join(str(int(x)) for x in xs) + "]"


def _fmt_validities(vs: list[float]) -> str:
    return "[" + " ".join(f"{v:g}" for v in vs) + "]"


def convert(
    inputs: list[Path],
    out_path: Path,
    validities: list[float],
) -> int:
    """Write a Hilbig-shaped CSV from one or more Prolific HDM JSONLs.
    Returns the number of trial rows written."""
    rows: list[dict] = []
    pair_index: dict[tuple, int] = {}
    # Per-participant trial counter spans all input files so concatenating
    # multiple JSONLs for one participant gives a contiguous trial sequence.
    trial_by_subj: dict[int, int] = defaultdict(int)

    for path in inputs:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                subj = int(obj["subject_id"])
                a = list(obj["option_a_ratings"])
                b = list(obj["option_b_ratings"])
                key = (tuple(a), tuple(b))
                if key not in pair_index:
                    pair_index[key] = len(pair_index) + 1
                trial = trial_by_subj[subj]
                trial_by_subj[subj] = trial + 1
                rows.append({
                    "participant": subj,
                    "task": 0,
                    "trial": trial,
                    "choice": int(obj["response"]),
                    "reward": 0,
                    "stimulus_0": _fmt_int_array(a),
                    "stimulus_1": _fmt_int_array(b),
                    "validities": _fmt_validities(validities),
                    "item_type": pair_index[key],
                    "tier": 0.0,
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "", "participant", "task", "trial", "choice", "reward",
            "stimulus_0", "stimulus_1", "validities", "item_type", "tier",
        ])
        for i, r in enumerate(rows):
            w.writerow([
                i, r["participant"], r["task"], r["trial"], r["choice"],
                r["reward"], r["stimulus_0"], r["stimulus_1"],
                r["validities"], r["item_type"], r["tier"],
            ])
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "inputs", type=Path, nargs="+",
        help="One or more Prolific HDM observation JSONL files.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--validities", type=float, nargs="+",
        default=[0.95, 0.85, 0.75, 0.65],
        help=(
            "Cue validities (one float per feature). Length must match the "
            "stimulus feature count. The Prolific run varied n_features "
            "across rounds (4 or 5), so callers must pass the actual "
            "per-round vector from observations/state.json."
        ),
    )
    a = p.parse_args(argv)
    n = convert(a.inputs, a.out, a.validities)
    print(f"wrote {a.out} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
