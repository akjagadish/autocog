"""Tests for scripts/plot_recovery_vs_clean_gt.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_bars_only_no_reference_line():
    """The clean-reference figure shows only the recovery bars — no gt reference
    line (no BASELINE_ROLES module constant)."""
    import scripts.plot_recovery_vs_clean_gt as mod

    assert set(mod.BAR_ROLES) == {"seed", "surfaced", "surfaced (best per run)"}
    assert not hasattr(mod, "BASELINE_ROLES")    # reference line removed


def test_select_noises_filters_levels():
    """`select_noises` keeps only the requested noise levels (None/empty = all),
    so a --from-csv re-plot can drop the degenerate ε=1 column."""
    import pandas as pd
    from scripts.plot_recovery_vs_clean_gt import select_noises

    df = pd.DataFrame({"noise": [0.0, 0.5, 1.0, 0.5], "mse": [1, 2, 3, 4]})
    assert sorted(select_noises(df, [0.0, 0.5])["noise"].unique()) == [0.0, 0.5]
    assert len(select_noises(df, None)) == len(df)        # None -> unchanged
    assert len(select_noises(df, [])) == len(df)          # empty -> unchanged


def test_main_writes_both_figures(tmp_path):
    """End-to-end: build against the clean reference and write the MSE and
    correlation figures (svg + png)."""
    from scripts.plot_recovery_vs_clean_gt import main

    rc = main([
        "--families", "ttb_sampling",
        "--noises", "0.0", "1.0",
        "--n-draws", "5",
        "--seed", "0",
        "--out-dir", str(tmp_path),
        "--csv", str(tmp_path / "long.csv"),
    ])
    assert rc == 0
    for stem in ("recovery_vs_clean_gt_mse", "recovery_vs_clean_gt_correlation"):
        assert (tmp_path / f"{stem}.png").is_file()
    assert (tmp_path / "long.csv").is_file()


def test_from_csv_replots_without_resimulation(tmp_path):
    """`--from-csv` re-plots from a long table (no run-dirs, no re-sim)."""
    from scripts.plot_recovery_vs_clean_gt import main

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
    assert (tmp_path / "recovery_vs_clean_gt_mse.png").is_file()
