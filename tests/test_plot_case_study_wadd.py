# tests/test_plot_case_study_wadd.py
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_case_study_wadd import (
    DEFAULT_RUN_DIR, sem, ttb_choice, tally_choice, disagreement_mask,
    reconstruct_metric, load_observations, assert_validity_order,
)
from scripts.case_study_wadd_narrowing import parse_arbitration, parse_experiment

RUN = _ROOT / DEFAULT_RUN_DIR


def test_sem_analytic():
    # SEM = sqrt(var / n); var=0.0120, n=25 -> sqrt(0.00048)=0.0219089023...
    assert sem(0.0120, 25) == pytest.approx(0.0219089023, abs=1e-9)
    assert sem(0.0, 25) == 0.0


def test_theory_helpers_handbuilt():
    # A higher on first cue (col 0) -> TTB picks A(0). B wins 2 of 3 -> Tally picks B(1).
    A = np.array([[1, 0, 0]]); B = np.array([[0, 1, 1]])
    assert int(ttb_choice(A, B)[0]) == 0
    assert int(tally_choice(A, B)[0]) == 1
    assert bool(disagreement_mask(A, B)[0]) is True
    # Tie in feature wins -> tally returns -1, not a disagreement.
    A2 = np.array([[1, 0]]); B2 = np.array([[0, 1]])
    assert int(tally_choice(A2, B2)[0]) == -1
    assert bool(disagreement_mask(A2, B2)[0]) is False


def test_reconstruct_metric_matches_arbiter_observed():
    # The schematic's number MUST equal the arbiter's observed value.
    arb = parse_arbitration(RUN)
    obs1 = load_observations(RUN, 1)
    obs2 = load_observations(RUN, 2)
    assert reconstruct_metric(obs1, "ttb") == pytest.approx(
        arb["experiments"][1]["observed"][0], abs=1e-3)
    assert reconstruct_metric(obs2, "tally") == pytest.approx(
        arb["experiments"][2]["observed"][0], abs=1e-3)
    # And the exact verified values.
    assert reconstruct_metric(obs1, "ttb") == pytest.approx(0.3450, abs=1e-3)
    assert reconstruct_metric(obs2, "tally") == pytest.approx(0.6887, abs=1e-3)


def test_disagreement_counts_match_design():
    e1 = parse_experiment(RUN, "pi_1")
    e2 = parse_experiment(RUN, "pi_2")
    assert int(disagreement_mask(e1["trial_a"], e1["trial_b"]).sum()) == 6
    assert int(disagreement_mask(e2["trial_a"], e2["trial_b"]).sum()) == 8


def test_validity_guard():
    assert_validity_order([0.95, 0.85, 0.55])  # descending: ok
    with pytest.raises(ValueError):
        assert_validity_order([0.5, 0.9, 0.7])


def test_reconstruct_metric_rejects_unknown_target():
    # Validation must fire at entry, even on degenerate data with no
    # disagreement rows (option_a == option_b), where the 0.5 fallback would
    # otherwise short-circuit the check before it is reached.
    degenerate = [{"option_a_ratings": [1, 0], "option_b_ratings": [1, 0], "response": 0}]
    with pytest.raises(ValueError):
        reconstruct_metric(degenerate, "wadd")


def test_render_stage1_returns_figure():
    from matplotlib.figure import Figure
    from scripts.plot_case_study_wadd import render_stage1
    fig = render_stage1(parse_experiment(RUN, "pi_1"))
    assert isinstance(fig, Figure)
    # pi_1: 8 trials x 2 options x 5 cue tiles = 80 tiles + 2 option-sides x 5
    # validity bars = 10 -> exactly 90 patches (exact count, per the CLAUDE.md
    # analytic-test rule; a weak >= bound would miss a dropped-bar regression).
    n_rects = sum(1 for a in fig.axes for p in a.patches)
    assert n_rects == 90


@pytest.mark.parametrize("layout", ["numberline", "dots", "forest"])
@pytest.mark.parametrize("target,exp_k", [("ttb", 1), ("tally", 2)])
def test_render_stage3_layouts(target, exp_k, layout):
    # Both targets are exercised so _METRIC_LABEL['tally'] is covered too.
    from matplotlib.figure import Figure
    from scripts.plot_case_study_wadd import render_stage3
    from scripts.case_study_wadd_narrowing import parse_arbitration, count_subjects
    arb = parse_arbitration(RUN)
    fig = render_stage3(arb["experiments"][exp_k], target=target,
                        n_subjects=count_subjects(RUN), layout=layout)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize("variant", ["schematic", "formula", "axis"])
@pytest.mark.parametrize("target,exp_k", [("ttb", 1), ("tally", 2)])
def test_render_stage2_variants(target, exp_k, variant):
    from matplotlib.figure import Figure
    from scripts.plot_case_study_wadd import render_stage2, load_observations
    from scripts.case_study_wadd_narrowing import parse_experiment
    pi = "pi_1" if exp_k == 1 else "pi_2"
    fig = render_stage2(parse_experiment(RUN, pi), load_observations(RUN, exp_k),
                        target=target, variant=variant)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize("target,exp_k", [("ttb", 1), ("tally", 2)])
def test_stage2_schematic_fractions_average_to_metric(target, exp_k):
    # Truthfulness invariant: the per-trial fractions shown in the schematic must
    # average to the labeled metric == reconstruct_metric == the arbiter observed
    # value. Guards against the schematic drifting from the number it reports.
    from scripts.plot_case_study_wadd import (
        _per_trial_target_fraction, reconstruct_metric, load_observations,
    )
    from scripts.case_study_wadd_narrowing import parse_experiment
    pi = "pi_1" if exp_k == 1 else "pi_2"
    exp = parse_experiment(RUN, pi)
    obs = load_observations(RUN, exp_k)
    fracs = [frac for _, _, frac in _per_trial_target_fraction(exp, obs, target)]
    assert np.mean(fracs) == pytest.approx(reconstruct_metric(obs, target), abs=1e-9)


def test_render_stage4_5_return_figures():
    from matplotlib.figure import Figure
    from scripts.plot_case_study_wadd import render_stage4, render_stage5
    from scripts.case_study_wadd_narrowing import parse_arbitration, parse_wadd_from_theories
    arb = parse_arbitration(RUN)
    assert isinstance(render_stage4(arb["response"]), Figure)
    assert isinstance(render_stage5(parse_wadd_from_theories(RUN)), Figure)


def test_main_writes_all_figures(tmp_path):
    from scripts.plot_case_study_wadd import main
    written = main(run_dir=RUN, out_dir=tmp_path)
    names = {p.stem for p in written}
    expected = {
        "stage1_design_exp1", "stage1_design_exp2",
        "stage1_choices_exp1", "stage1_choices_exp2",
        "stage2_metric_exp1_schematic", "stage2_metric_exp1_formula", "stage2_metric_exp1_axis",
        "stage2_metric_exp2_schematic", "stage2_metric_exp2_formula", "stage2_metric_exp2_axis",
        "stage3_gap_exp1_numberline", "stage3_gap_exp1_dots", "stage3_gap_exp1_forest",
        "stage3_gap_exp2_numberline", "stage3_gap_exp2_dots", "stage3_gap_exp2_forest",
        "stage4_arbiter", "stage5_wadd",
    }
    assert expected <= names
    assert all(p.exists() for p in written)
    # SVG + PNG per figure -> 18 figures * 2 = 36 files.
    assert len([p for p in written if p.suffix in (".svg", ".png")]) == 36


def test_render_stage3_numberline_marker_positions():
    # Analytic check: the three markers sit at exactly the TTB / Tallying /
    # Observed metric values, so a TTB<->Tally swap or mislabel is caught.
    from scripts.plot_case_study_wadd import render_stage3
    from scripts.case_study_wadd_narrowing import parse_arbitration, count_subjects
    res = parse_arbitration(RUN)["experiments"][1]
    fig = render_stage3(res, target="ttb",
                        n_subjects=count_subjects(RUN), layout="numberline")
    ax = fig.axes[0]
    xs = sorted(c.lines[0].get_xdata()[0] for c in ax.containers)
    expected = sorted([res["pi_1"][0], res["pi_2"][0], res["observed"][0]])
    assert xs == pytest.approx(expected)


def test_wadd_choice_argmax_and_ties():
    # WADD = argmax of the validity-weighted cue sum. Hand-built trial where
    # WADD dissociates from BOTH heuristics: A wins the highest-validity cue and
    # the majority of cues, yet B's weighted sum is larger (B takes the 2nd/3rd
    # highest validities). Then a tie (equal weighted sums) -> -1.
    from scripts.plot_case_study_wadd import wadd_choice
    A = np.array([[1, 0, 0, 1, 1]]); B = np.array([[0, 1, 1, 0, 0]])
    v = [0.9, 0.85, 0.8, 0.1, 0.1]
    assert int(ttb_choice(A, B)[0]) == 0       # TTB -> A (first discriminating cue)
    assert int(tally_choice(A, B)[0]) == 0     # Tally -> A (3 wins vs 2)
    assert int(wadd_choice(A, B, v)[0]) == 1   # WADD -> B (1.65 > 1.10)
    assert int(wadd_choice(np.array([[1, 0]]), np.array([[0, 1]]), [0.5, 0.5])[0]) == -1


def test_render_stage1_choices_returns_figure():
    # Uses the SAME designed option pairs as render_stage1 (parse_experiment),
    # so rows align with the stage-1 design figure. 8 trials x 3 models = 24 cells.
    from matplotlib.figure import Figure
    from scripts.plot_case_study_wadd import render_stage1_choices
    fig = render_stage1_choices(parse_experiment(RUN, "pi_1"))
    assert isinstance(fig, Figure)
    n_cells = sum(1 for a in fig.axes for p in a.patches)
    assert n_cells == 24
