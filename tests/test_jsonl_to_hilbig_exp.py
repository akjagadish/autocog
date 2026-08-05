"""Round-trip test for `scripts.jsonl_to_hilbig_exp.convert`.

Writes a tiny JSONL mimicking the Prolific HDM observation format
(`subject_id`, `option_a_ratings`, `option_b_ratings`, `response`),
runs the converter, and re-parses the output with `eval_hilbig`'s own
array parser to confirm the file is consumable by the existing
Hilbig-eval pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.eval_hilbig import _parse_array_string
from scripts.jsonl_to_hilbig_exp import convert


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_convert_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "round_000_obs_00.jsonl"
    rows = [
        # subject 0, two trials with two distinct stimulus pairs
        {"subject_id": 0,
         "option_a_ratings": [5, 4, 1, 1],
         "option_b_ratings": [5, 3, 2, 2],
         "response": 0},
        {"subject_id": 0,
         "option_a_ratings": [3, 2, 4, 4],
         "option_b_ratings": [3, 3, 1, 1],
         "response": 1},
        # subject 1 sees the same first pair (so item_type for that
        # pair must match across participants)
        {"subject_id": 1,
         "option_a_ratings": [5, 4, 1, 1],
         "option_b_ratings": [5, 3, 2, 2],
         "response": 1},
    ]
    _write_jsonl(src, rows)

    out = tmp_path / "exp.txt"
    n = convert([src], out, validities=[0.95, 0.85, 0.75, 0.65])
    assert n == 3

    df = pd.read_csv(out)
    # Schema must match hilbig2014/exp1.txt exactly so eval_hilbig can
    # consume it without changes.
    assert list(df.columns)[:11] == [
        "Unnamed: 0", "participant", "task", "trial", "choice", "reward",
        "stimulus_0", "stimulus_1", "validities", "item_type", "tier",
    ]
    assert df["participant"].tolist() == [0, 0, 1]
    assert df["trial"].tolist() == [0, 1, 0]  # per-participant counter
    assert df["choice"].tolist() == [0, 1, 1]

    # Stimulus arrays survive the bracketed-string round trip.
    assert _parse_array_string(df["stimulus_0"].iloc[0]) == (5, 4, 1, 1)
    assert _parse_array_string(df["stimulus_1"].iloc[1]) == (3, 3, 1, 1)

    # Validities are written in Hilbig's space-separated bracketed form.
    assert df["validities"].iloc[0] == "[0.95 0.85 0.75 0.65]"

    # Same (option_a, option_b) pair across participants gets the same
    # item_type; distinct pairs get distinct item_types.
    assert df.loc[0, "item_type"] == df.loc[2, "item_type"]
    assert df.loc[0, "item_type"] != df.loc[1, "item_type"]

    # Constant columns.
    assert (df["task"] == 0).all()
    assert (df["reward"] == 0).all()
    assert (df["tier"] == 0.0).all()
