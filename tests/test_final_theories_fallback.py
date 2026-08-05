"""`final_theories_for_run` must use the last COMPLETE round.

An interrupted autocog run can leave trailing round dirs that were created but
never got a `theories.json` written (e.g. a run that did its 5 rounds, then a
resume attempt created round_005/006/007 stubs and crashed). The surfaced
theories should come from the last round that actually has a readable
`theories.json`, not crash/skip the whole run because a newer stub exists.
"""

from __future__ import annotations

import json

from scripts.summarize_final_theories import final_theories_for_run


def _write_round(run_dir, name, theories=None):
    rd = run_dir / "rounds" / name
    rd.mkdir(parents=True)
    if theories is not None:
        (rd / "theories.json").write_text(json.dumps(theories))
    return rd


def _theories(label_a, label_b):
    return {
        "starting_theories": [
            {"slot": 1, "label": label_a, "theory": {"description": label_a}},
            {"slot": 2, "label": label_b, "theory": {"description": label_b}},
        ]
    }


def test_falls_back_to_last_round_with_theories_json(tmp_path):
    run = tmp_path / "run1"
    _write_round(run, "round_000", _theories("A0", "B0"))
    _write_round(run, "round_004", _theories("A4", "B4"))   # last complete
    _write_round(run, "round_005")                          # crashed stub
    _write_round(run, "round_006")                          # crashed stub

    got = final_theories_for_run(run)
    labels = sorted(t["label"] for t in got)
    assert labels == ["A4", "B4"]


def test_uses_last_round_when_complete(tmp_path):
    run = tmp_path / "run2"
    _write_round(run, "round_000", _theories("A0", "B0"))
    _write_round(run, "round_001", _theories("A1", "B1"))

    got = final_theories_for_run(run)
    labels = sorted(t["label"] for t in got)
    assert labels == ["A1", "B1"]
