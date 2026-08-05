"""Tests for the standalone scalar-similarity LLM judge.

The similarity judge reads a surfaced theory and the ground-truth theory and
emits a single mechanism-similarity score in [0, 1]. Two input modes:
  - "description": only the prose descriptions are shown to the LLM.
  - "joint":       description AND predict() source are shown, paired.
"""

import csv
import json
from pathlib import Path

import pydantic
import pytest

from src.llm import MockClient

from scripts.judge_similarity import GroundTruthSpec, SurfacedTheory
from scripts.judge_similarity import (
    CSV_FIELDNAMES,
    SimilarityVerdict,
    judge_run_group,
    judge_similarity_once,
    judge_theory_similarity,
    similarity_user_prompt,
)


def test_similarity_verdict_unit_interval() -> None:
    assert SimilarityVerdict(similarity=0.42, rationale="variant").similarity == 0.42
    assert SimilarityVerdict(similarity=0.0, rationale="").similarity == 0.0
    assert SimilarityVerdict(similarity=1.0, rationale="").similarity == 1.0
    with pytest.raises(pydantic.ValidationError):
        SimilarityVerdict(similarity=1.5, rationale="x")
    with pytest.raises(pydantic.ValidationError):
        SimilarityVerdict(similarity=-0.1, rationale="x")


def test_user_prompt_description_mode_excludes_predict() -> None:
    surfaced = SurfacedTheory(
        label="pi_5", description="SURF_DESC", predict_source="SURF_CODE"
    )
    gt = GroundTruthSpec(
        name="tallying", description="GT_DESC", predict_source="GT_CODE"
    )
    prompt = similarity_user_prompt(surfaced, gt, input_mode="description")

    assert "SURF_DESC" in prompt
    assert "GT_DESC" in prompt
    # description mode must NOT leak the predict() source of either theory.
    assert "SURF_CODE" not in prompt
    assert "GT_CODE" not in prompt


def test_user_prompt_joint_mode_includes_predict() -> None:
    surfaced = SurfacedTheory(
        label="pi_5", description="SURF_DESC", predict_source="SURF_CODE"
    )
    gt = GroundTruthSpec(
        name="tallying", description="GT_DESC", predict_source="GT_CODE"
    )
    prompt = similarity_user_prompt(surfaced, gt, input_mode="joint")

    assert "SURF_DESC" in prompt
    assert "GT_DESC" in prompt
    assert "SURF_CODE" in prompt
    assert "GT_CODE" in prompt


def test_judge_similarity_once_parses_response() -> None:
    canned = SimilarityVerdict(similarity=0.7, rationale="variant").model_dump_json()
    client = MockClient(canned=[canned])
    surfaced = SurfacedTheory(label="pi_5", description="d", predict_source="s")
    gt = GroundTruthSpec(name="tallying", description="gtd", predict_source="gts")

    verdict = judge_similarity_once(
        client=client, surfaced=surfaced, ground_truth=gt, input_mode="description"
    )
    assert verdict.similarity == 0.7
    assert verdict.rationale == "variant"


def test_judge_theory_similarity_averages_votes() -> None:
    """n_votes scalar samples reduce to their exact arithmetic mean.

    [0.8, 0.6] -> mean 0.7. The votes string preserves the raw samples in
    order, and the rationale is taken from the first sample.
    """
    canned = [
        SimilarityVerdict(similarity=0.8, rationale="first").model_dump_json(),
        SimilarityVerdict(similarity=0.6, rationale="second").model_dump_json(),
    ]
    client = MockClient(canned=canned)
    surfaced = SurfacedTheory(label="pi_5", description="d", predict_source="s")
    gt = GroundTruthSpec(name="tallying", description="gtd", predict_source="gts")

    out = judge_theory_similarity(
        client=client,
        surfaced=surfaced,
        ground_truth=gt,
        input_mode="description",
        n_votes=2,
    )
    assert out["similarity"] == "0.700"
    assert out["similarity_votes"] == "0.800;0.600"
    assert out["similarity_rationale"] == "first"


def test_judge_theory_similarity_rejects_zero_votes() -> None:
    """n_votes < 1 has no defined mean; the function must reject it loudly
    rather than raising a cryptic ZeroDivisionError on sum([])/len([])."""
    client = MockClient(canned=[])
    surfaced = SurfacedTheory(label="pi_5", description="d", predict_source="s")
    gt = GroundTruthSpec(name="tallying", description="gtd", predict_source="gts")
    with pytest.raises(ValueError):
        judge_theory_similarity(
            client=client,
            surfaced=surfaced,
            ground_truth=gt,
            input_mode="description",
            n_votes=0,
        )


def _write_full_run(
    run_dir: Path,
    pi1: tuple[str, str],
    pi2: tuple[str, str],
    surfaced: list[tuple[int, str, str, str]],
) -> None:
    rounds = run_dir / "rounds"
    (rounds / "round_000").mkdir(parents=True)
    (rounds / "round_000" / "theories.json").write_text(json.dumps({
        "round_idx": 0,
        "starting_theories": [
            {"slot": 1, "label": "pi_1", "killed": False,
             "theory": {"description": pi1[0], "predict_source": pi1[1]}},
            {"slot": 2, "label": "pi_2", "killed": False,
             "theory": {"description": pi2[0], "predict_source": pi2[1]}},
        ],
    }))
    (rounds / "round_001").mkdir(parents=True)
    (rounds / "round_001" / "theories.json").write_text(json.dumps({
        "round_idx": 1,
        "starting_theories": [
            {"slot": s, "label": label, "killed": False,
             "theory": {"description": desc, "predict_source": pred}}
            for s, label, desc, pred in surfaced
        ],
    }))


def test_judge_run_group_writes_rows_per_run_per_slot(tmp_path: Path) -> None:
    runs_dir = tmp_path / "noise=0.0"
    runs_dir.mkdir()
    for i in (1, 2):
        _write_full_run(
            runs_dir / f"run{i}",
            pi1=("seed1 desc", "seed1 src"),
            pi2=("seed2 desc", "seed2 src"),
            surfaced=[
                (1, f"pi_a{i}", "surf1 desc", "surf1 src"),
                (2, f"pi_b{i}", "surf2 desc", "surf2 src"),
            ],
        )

    # 2 runs x 2 slots x 1 vote = 4 calls.
    canned = [
        SimilarityVerdict(similarity=0.5, rationale="r").model_dump_json()
        for _ in range(4)
    ]
    client = MockClient(canned=canned)
    gt = GroundTruthSpec(name="tallying", description="gtd", predict_source="gts")

    out = judge_run_group(
        runs_dir=runs_dir,
        domain="heuristic_decision_making",
        ground_truth=gt,
        client=client,
        input_mode="description",
        n_votes=1,
        skip_existing=False,
        out_name="judge_similarity_desc.csv",
    )
    assert out == runs_dir / "judge_similarity_desc.csv"

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4
    assert {r["run"] for r in rows} == {"1", "2"}
    for r in rows:
        assert r["domain"] == "heuristic_decision_making"
        assert r["run_group"] == "noise=0.0"
        assert r["input_mode"] == "description"
        assert r["similarity"] == "0.500"


def test_judge_run_group_skip_existing_does_not_call_llm(tmp_path: Path) -> None:
    runs_dir = tmp_path / "noise=0.0"
    runs_dir.mkdir()
    for i in (1, 2):
        _write_full_run(
            runs_dir / f"run{i}",
            pi1=("seed1", "src1"), pi2=("seed2", "src2"),
            surfaced=[(1, f"pi_a{i}", "d", "s"), (2, f"pi_b{i}", "d", "s")],
        )

    out_csv = runs_dir / "judge_similarity_desc.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for run_idx in ("1", "2"):
            for slot in ("1", "2"):
                row = {k: "" for k in CSV_FIELDNAMES}
                row.update({
                    "domain": "heuristic_decision_making",
                    "run_group": "noise=0.0",
                    "run": run_idx,
                    "run_dir": f"run{run_idx}",
                    "slot": slot,
                    "label": f"prev_{run_idx}_{slot}",
                    "input_mode": "description",
                    "similarity": "0.999",
                })
                writer.writerow(row)

    client = MockClient(canned=[])  # any LLM call raises

    judge_run_group(
        runs_dir=runs_dir,
        domain="heuristic_decision_making",
        ground_truth=GroundTruthSpec(name="t", description="d", predict_source="s"),
        client=client,
        input_mode="description",
        n_votes=1,
        skip_existing=True,
        out_name="judge_similarity_desc.csv",
    )
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 4
    for r in rows:
        assert r["similarity"] == "0.999"


def test_skip_existing_is_per_input_mode(tmp_path: Path) -> None:
    """skip-existing must key on input_mode: a CSV holding 'description' rows
    must NOT cause a later 'joint' pass into the same file to skip its work."""
    runs_dir = tmp_path / "noise=0.0"
    runs_dir.mkdir()
    _write_full_run(
        runs_dir / "run1",
        pi1=("seed1", "src1"), pi2=("seed2", "src2"),
        surfaced=[(1, "pi_a", "d", "s"), (2, "pi_b", "d", "s")],
    )

    out_csv = runs_dir / "shared.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for slot in ("1", "2"):
            row = {k: "" for k in CSV_FIELDNAMES}
            row.update({
                "domain": "heuristic_decision_making", "run_group": "noise=0.0",
                "run": "1", "run_dir": "run1", "slot": slot,
                "label": f"pi_{slot}", "input_mode": "description",
                "similarity": "0.111",
            })
            writer.writerow(row)

    # 1 run x 2 slots x 1 vote = 2 joint calls expected (not skipped).
    canned = [
        SimilarityVerdict(similarity=0.9, rationale="r").model_dump_json()
        for _ in range(2)
    ]
    client = MockClient(canned=canned)

    judge_run_group(
        runs_dir=runs_dir,
        domain="heuristic_decision_making",
        ground_truth=GroundTruthSpec(name="t", description="d", predict_source="s"),
        client=client,
        input_mode="joint",
        n_votes=1,
        skip_existing=True,
        out_name="shared.csv",
    )

    rows = list(csv.DictReader(out_csv.open()))
    joint_rows = [r for r in rows if r["input_mode"] == "joint"]
    desc_rows = [r for r in rows if r["input_mode"] == "description"]
    assert len(joint_rows) == 2  # joint work was done, not skipped
    assert len(desc_rows) == 2   # prior description rows preserved


def test_main_invokes_judge_run_group(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "noise=0.0"
    runs_dir.mkdir()
    for i in (1, 2, 3):
        _write_full_run(
            runs_dir / f"run{i}",
            pi1=("seed1", "src1"), pi2=("seed2", "src2"),
            surfaced=[(1, f"pi_a{i}", "d", "s"), (2, f"pi_b{i}", "d", "s")],
        )

    # Ground truth lookup: point theories_root at a tmp dir with a yaml.
    theories_root = tmp_path / "theories" / "heuristic_decision_making"
    theories_root.mkdir(parents=True)
    (theories_root / "tallying.yaml").write_text(
        "name: Tallying\ntheory: |\n  count wins\npredict: |\n  def predict(p, s, h):\n      return 0\n"
    )

    canned = [
        SimilarityVerdict(similarity=0.5, rationale="r").model_dump_json()
        for _ in range(6)  # 3 runs * 2 slots * 1 vote
    ]
    fake_client = MockClient(canned=canned)
    monkeypatch.setattr(
        "scripts.judge_similarity.make_client", lambda cfg: fake_client
    )
    monkeypatch.setattr("scripts.judge_similarity.ROOT", tmp_path)

    from scripts.judge_similarity import main

    main([
        "--runs-dir", str(runs_dir),
        "--domain", "heuristic_decision_making",
        "--ground-truth", "tallying",
        "--input-mode", "description",
        "--out-name", "judge_similarity_desc.csv",
    ])

    out = runs_dir / "judge_similarity_desc.csv"
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6
