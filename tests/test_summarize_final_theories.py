import json
from pathlib import Path

from scripts.summarize_final_theories import final_theories_for_run, summarize_meta_dir


def _write_run(run_dir: Path, rounds: list[dict]) -> None:
    (run_dir / "rounds").mkdir(parents=True)
    for i, payload in enumerate(rounds):
        rd = run_dir / "rounds" / f"round_{i:03d}"
        rd.mkdir()
        (rd / "theories.json").write_text(json.dumps(payload))


def _theory(desc: str, src: str) -> dict:
    return {"description": desc, "predict_source": src}


def test_final_theories_applies_replacement(tmp_path: Path) -> None:
    run = tmp_path / "run_x"
    r0 = {
        "round_idx": 0,
        "starting_theories": [
            {"slot": 1, "label": "pi_1", "killed": True, "theory": _theory("d1", "s1")},
            {"slot": 2, "label": "pi_2", "killed": False, "theory": _theory("d2", "s2")},
        ],
        "replacement": {
            "slot": 1, "label": "pi_3", "theory": _theory("d3", "s3"),
        },
    }
    _write_run(run, [r0])

    result = final_theories_for_run(run)
    assert [t["slot"] for t in result] == [1, 2]
    assert result[0]["label"] == "pi_3"
    assert result[0]["theory"]["description"] == "d3"
    assert result[1]["label"] == "pi_2"


def test_final_theories_uses_last_round(tmp_path: Path) -> None:
    run = tmp_path / "run_y"
    r0 = {
        "round_idx": 0,
        "starting_theories": [
            {"slot": 1, "label": "pi_a", "killed": False, "theory": _theory("da", "sa")},
            {"slot": 2, "label": "pi_b", "killed": False, "theory": _theory("db", "sb")},
        ],
    }
    r1 = {
        "round_idx": 1,
        "starting_theories": [
            {"slot": 1, "label": "pi_a", "killed": False, "theory": _theory("da2", "sa2")},
            {"slot": 2, "label": "pi_b", "killed": False, "theory": _theory("db2", "sb2")},
        ],
    }
    _write_run(run, [r0, r1])

    result = final_theories_for_run(run)
    assert [t["theory"]["description"] for t in result] == ["da2", "db2"]


def test_summarize_meta_dir_writes_file(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    run1 = meta / "task_run1"
    run2 = meta / "task_run2"
    _write_run(run1, [{
        "round_idx": 0,
        "starting_theories": [
            {"slot": 1, "label": "pi_1", "killed": False, "theory": _theory("alpha", "code_a")},
            {"slot": 2, "label": "pi_2", "killed": False, "theory": _theory("beta", "code_b")},
        ],
    }])
    _write_run(run2, [{
        "round_idx": 0,
        "starting_theories": [
            {"slot": 1, "label": "pi_3", "killed": False, "theory": _theory("gamma", "code_c")},
            {"slot": 2, "label": "pi_4", "killed": False, "theory": _theory("delta", "code_d")},
        ],
    }])

    out = summarize_meta_dir(meta)
    assert out == meta / "final_theories.md"
    text = out.read_text()
    assert "task_run1" in text
    assert "task_run2" in text
    assert "alpha" in text and "beta" in text
    assert "gamma" in text and "delta" in text
    assert "code_a" in text
