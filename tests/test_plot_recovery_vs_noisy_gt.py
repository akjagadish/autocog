"""Tests for scripts/plot_recovery_vs_noisy_gt.py."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_main_writes_mse_and_correlation_figures(tmp_path):
    """End-to-end: the driver builds the long table for a canonical sampling
    family and writes both the MSE and correlation figures (svg + png)."""
    from scripts.plot_recovery_vs_noisy_gt import main

    rc = main([
        "--families", "ttb_sampling",
        "--noises", "0.0", "1.0",
        "--n-draws", "5",
        "--seed", "0",
        "--out-dir", str(tmp_path),
        "--csv", str(tmp_path / "long.csv"),
    ])
    assert rc == 0
    for stem in ("recovery_vs_noisy_gt_mse",
                 "recovery_vs_noisy_gt_correlation"):
        png = tmp_path / f"{stem}.png"
        assert png.is_file() and png.stat().st_size > 0
    assert (tmp_path / "long.csv").is_file()


def test_reference_lines_are_gt_floor_and_gt_clean_not_bars():
    """The figure shows seed / surfaced / best as bars and TWO reference LINES:
    the flat gt floor and the rising gt-vs-noisy-gt baseline. Lines and bars
    must be disjoint (plot_grid forbids overlap)."""
    from scripts.plot_recovery_vs_noisy_gt import BAR_ROLES, BASELINE_ROLES

    assert set(BAR_ROLES) == {"seed", "surfaced", "surfaced (best per run)"}
    assert set(BASELINE_ROLES) == {"gt", "gt_clean"}   # floor + baseline lines
    assert set(BAR_ROLES) & set(BASELINE_ROLES) == set()


def test_from_csv_replots_without_resimulation(tmp_path):
    """`--from-csv` must re-plot straight from a long table (no run-dirs, no
    re-simulation) and still write both figures."""
    import pandas as pd
    from scripts.plot_recovery_vs_noisy_gt import main

    # minimal long table: one family, two noise levels, the needed roles.
    rows = []
    for nz in (0.0, 0.5):
        for role in ("seed", "surfaced", "gt", "gt_clean"):
            rows.append(dict(family="ttb_sampling", noise=nz, run_dir="r0",
                             role=role, theory_label=role, n_stimuli=24,
                             pearson_r=0.8, mse=0.05, lag1_autocorr=0.0))
    csv = tmp_path / "long.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    rc = main(["--from-csv", str(csv), "--out-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "recovery_vs_noisy_gt_mse.png").is_file()
    assert (tmp_path / "recovery_vs_noisy_gt_correlation.png").is_file()
