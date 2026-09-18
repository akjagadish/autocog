"""Analytic checks for the model-comparison table behind
scripts/plot_recovery_model_comparison.py (Gemini vs Claude recovery bars).

Long-format rows are aggregated exactly as the paper's Figure 3C: theories
are first pooled WITHIN a run-dir (mean), then mean +/- SEM ACROSS run-dirs
(SEM = std(ddof=1)/sqrt(n_runs)). Numbers below are hand-computed.
"""
import argparse
import pandas as pd
import pytest

from scripts.plot_recovery_model_comparison import build_comparison_table


def _long(model_rows):
    return pd.DataFrame(model_rows, columns=["family", "noise", "run_dir", "role", "theory_label", "mse", "pearson_r"])


@pytest.fixture
def two_model_inputs(tmp_path):
    # Model A: run1 surfaced mse {0.1, 0.3} -> 0.2 ; run2 surfaced {0.4} -> 0.4
    #          across runs: mean 0.3, std(ddof=1)=0.1414.., SEM=0.1
    #          best-per-run (min mse): run1 0.1, run2 0.4 -> mean 0.25, SEM 0.15
    #          seed: run1 {0.5}, run2 {0.5} -> mean 0.5, SEM 0
    a = _long([
        ("single_cue", 0.0, "r1", "seed", "pi_1", 0.5, 0.0),
        ("single_cue", 0.0, "r1", "surfaced", "pi_3", 0.1, 0.9),
        ("single_cue", 0.0, "r1", "surfaced", "pi_4", 0.3, 0.5),
        ("single_cue", 0.0, "r2", "seed", "pi_1", 0.5, 0.0),
        ("single_cue", 0.0, "r2", "surfaced", "pi_5", 0.4, 0.2),
    ])
    # Model B: single run, surfaced {0.2} ; SEM must be 0 with one run.
    b = _long([
        ("single_cue", 0.0, "r1", "seed", "pi_1", 0.6, 0.0),
        ("single_cue", 0.0, "r1", "surfaced", "pi_3", 0.2, 0.7),
    ])
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False); b.to_csv(pb, index=False)
    return {"Model A": pa, "Model B": pb}


def _cell(table, model, role, col):
    row = table[(table.model == model) & (table.role == role) & (table.family == "single_cue")]
    assert len(row) == 1, row
    return float(row[col].iloc[0])


def test_mse_means_and_sems_are_exact(two_model_inputs):
    t = build_comparison_table(two_model_inputs, metric="mse")
    assert _cell(t, "Model A", "surfaced", "mean") == pytest.approx(0.3)
    assert _cell(t, "Model A", "surfaced", "sem") == pytest.approx(0.1)
    assert _cell(t, "Model A", "surfaced (best per run)", "mean") == pytest.approx(0.25)
    assert _cell(t, "Model A", "surfaced (best per run)", "sem") == pytest.approx(0.15)
    assert _cell(t, "Model A", "seed", "mean") == pytest.approx(0.5)
    assert _cell(t, "Model A", "seed", "sem") == pytest.approx(0.0)
    assert _cell(t, "Model B", "surfaced", "mean") == pytest.approx(0.2)
    assert _cell(t, "Model B", "surfaced", "sem") == pytest.approx(0.0)
    assert _cell(t, "Model A", "surfaced", "n_runs") == 2
    assert _cell(t, "Model B", "surfaced", "n_runs") == 1


def test_best_flips_direction_for_correlation(two_model_inputs):
    # For pearson_r "best" is the MAX per run: run1 0.9, run2 0.2 -> mean 0.55
    t = build_comparison_table(two_model_inputs, metric="pearson_r")
    assert _cell(t, "Model A", "surfaced (best per run)", "mean") == pytest.approx(0.55)


def test_model_order_follows_input_order(two_model_inputs):
    t = build_comparison_table(two_model_inputs, metric="mse")
    assert list(t.model.cat.categories) == ["Model A", "Model B"]


def test_preexisting_best_rows_are_not_double_counted(tmp_path):
    # If the CSV already carries 'surfaced (best per run)' rows (as
    # recovery_correlation --csv can), they must be recomputed, not stacked:
    # n_runs stays 2 and the SEM stays 0.15 (see two_model_inputs).
    rows = _long([
        ("single_cue", 0.0, "r1", "surfaced", "pi_3", 0.1, 0.9),
        ("single_cue", 0.0, "r1", "surfaced", "pi_4", 0.3, 0.5),
        ("single_cue", 0.0, "r1", "surfaced (best per run)", "pi_3", 0.1, 0.9),
        ("single_cue", 0.0, "r2", "surfaced", "pi_5", 0.4, 0.2),
        ("single_cue", 0.0, "r2", "surfaced (best per run)", "pi_5", 0.4, 0.2),
    ])
    p = tmp_path / "a.csv"; rows.to_csv(p, index=False)
    t = build_comparison_table({"A": p}, metric="mse")
    assert _cell(t, "A", "surfaced (best per run)", "n_runs") == 2
    assert _cell(t, "A", "surfaced (best per run)", "sem") == pytest.approx(0.15)


def test_noise_level_is_selected_not_mixed(tmp_path):
    rows = _long([
        ("single_cue", 0.0, "r1", "surfaced", "pi_3", 0.1, 0.9),
        ("single_cue", 0.5, "r1", "surfaced", "pi_3", 0.9, 0.1),
    ])
    p = tmp_path / "a.csv"; rows.to_csv(p, index=False)
    t = build_comparison_table({"A": p}, metric="mse", noise=0.0)
    assert _cell(t, "A", "surfaced", "mean") == pytest.approx(0.1)
    assert set(t.noise.unique()) == {0.0}


def test_duplicate_input_labels_are_rejected():
    from scripts.plot_recovery_model_comparison import _parse_inputs
    with pytest.raises(argparse.ArgumentTypeError, match="duplicate"):
        _parse_inputs(["A=x.csv", "A=y.csv"])
