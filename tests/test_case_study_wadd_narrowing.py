"""Tests for the round-0 WADD case-study figure's data extraction.

The figure tells one story — how the LLM narrows from the TTB+Tallying seeds to
WADD in round 0. These tests cover the **binary** task variant
(`dmb_ground_truth_wadd_sampling`): features are 0/1, so there are no
magnitudes, and the arbiter picks WADD because behavior is intermediate — a
*weighted compensatory* rule integrating the cues' *specific validities*
(TTB = non-compensatory; Tallying = unweighted compensatory).

`case_study_wadd_narrowing.py` also handles a cardinal variant
(`hdm_ground_truth_wadd`, features are 1..rating_max ratings), but that data is
not part of this repository, so its `rating_max` / magnitude-insight /
"cardinal task" subtitle branches are **not covered here**.

Only the extraction / text-selection logic is unit-tested here; the matplotlib
drawing is iterated visually. Every number/string below is hand-verified ground
truth read off the run's round_000 artifacts.
"""
from pathlib import Path

import pytest

from scripts.case_study_wadd_narrowing import (
    count_subjects,
    key_insight_sentence,
    parse_arbitration,
    parse_experiment,
    parse_wadd_from_theories,
    run_subtitle,
    stage1_taglines,
)

BINARY_RUN = Path(
    "results/recovery/"
    "wadd_sampling/noise=0.0/"
    "dmb_ground_truth_wadd_sampling_noise=0.0_gemini-3.1-pro-preview_run3"
)


# --- experiment parsing: binary features carry no rating_max ------------------
def test_parse_experiment_binary_is_binary_and_has_no_rating_max():
    exp1 = parse_experiment(BINARY_RUN, "pi_1")
    exp2 = parse_experiment(BINARY_RUN, "pi_2")
    assert exp1["trial_a"].shape == (8, 5)
    assert exp2["trial_a"].shape == (12, 5)
    assert exp1["rating_max"] is None
    # Binary features only ever take the values 0 and 1.
    assert set(exp1["trial_a"].flatten().tolist()) <= {0, 1}
    assert exp1["validities"] == [0.95, 0.85, 0.75, 0.65, 0.55]


# --- taglines are read from the data, not hardcoded ---------------------------
def test_stage1_taglines_binary_report_real_dims_and_omit_rating_max():
    exp1 = parse_experiment(BINARY_RUN, "pi_1")
    exp2 = parse_experiment(BINARY_RUN, "pi_2")
    assert stage1_taglines(exp1, exp2) == (
        "Exp 1  ·  proposed by TTB  ·  5 feats × 8 trials",
        "Exp 2  ·  proposed by Tallying  ·  5 feats × 12 trials",
    )


# --- arbiter predictions (the bracket plot anchors) ---------------------------
def test_parse_arbitration_binary_predictions_and_observations():
    arb = parse_arbitration(BINARY_RUN)
    e1, e2 = arb["experiments"][1], arb["experiments"][2]
    assert e1["pi_1"][0] == pytest.approx(0.8717)
    assert e1["pi_2"][0] == pytest.approx(0.1389)
    assert e1["observed"][0] == pytest.approx(0.3450)
    assert e2["pi_1"][0] == pytest.approx(0.1425)
    assert e2["pi_2"][0] == pytest.approx(0.8603)
    assert e2["observed"][0] == pytest.approx(0.6887)
    # The observation is strictly between the two theory predictions on both
    # experiments — the "in-between" gap that motivates WADD.
    for e in (e1, e2):
        lo, hi = sorted((e["pi_1"][0], e["pi_2"][0]))
        assert lo < e["observed"][0] < hi


# --- key-insight sentence selection: the heart of the narrative ---------------
def test_key_insight_sentence_binary_is_the_weighted_validity_sentence():
    arb = parse_arbitration(BINARY_RUN)
    assert key_insight_sentence(arb["response"]["interpretation"]) == (
        "Instead, subjects likely use a weighted compensatory strategy where "
        "both the number of cues and their specific validities are integrated."
    )


def test_key_insight_sentence_does_not_match_unweighted_compensatory():
    # "unweighted compensatory" contains the substring "weighted compensatory";
    # the selector must not latch onto that earlier, wrong sentence.
    interp = (
        "Neither a purely unweighted compensatory rule captures the data. "
        "Instead, subjects likely use a weighted compensatory strategy where "
        "both the number of cues and their specific validities are integrated."
    )
    assert key_insight_sentence(interp).startswith("Instead,")


# --- WADD theory + subject count ---------------------------------------------
def test_parse_wadd_binary_description_names_wadd():
    wadd = parse_wadd_from_theories(BINARY_RUN)
    assert wadd["description"].startswith("Weighted Additive (WADD) Theory:")
    assert "cue validity" in wadd["description"].lower()


def test_count_subjects_matches_the_observation_files():
    assert count_subjects(BINARY_RUN) == 25


# --- subtitle labels the run truthfully (no more hardcoded "run 1") -----------
def test_run_subtitle_binary_names_task_gt_noise_run():
    assert run_subtitle(BINARY_RUN) == (
        "binary task  ·  ground truth = WADD  ·  noise = 0.0  ·  run3"
    )


