"""Tests for scripts/plot_recovery_clean_vs_noisy_gt.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLING_ROOT = REPO_ROOT / "results" / "recovery"


def test_build_clean_and_noisy_tags_both_references_and_differs_only_at_noise():
    """`build_clean_and_noisy` scores each role against BOTH references. The two
    references coincide at ε=0 (gt@0 == gt@0), so a role's mean MSE is IDENTICAL
    across references there; at ε=1 the noisy reference (gt@1) diverges from the
    clean one, so the role's MSE differs."""
    from scripts.plot_recovery_clean_vs_noisy_gt import build_clean_and_noisy
    from scripts.recovery_correlation import unique_stimulus_pairs

    pairs = unique_stimulus_pairs()
    df = build_clean_and_noisy(
        results_root=SAMPLING_ROOT, families=["ttb_sampling"],
        noises=[0.0, 1.0], stimulus_pairs=pairs, n_draws=120, base_seed=0,
    )
    assert set(df["reference"].unique()) == {"clean", "noisy"}
    seed = df[df.role == "seed"].groupby(["noise", "reference"])["mse"].mean()
    # ε=0: clean and noisy references are the same target -> identical MSE.
    assert seed.loc[(0.0, "clean")] == seed.loc[(0.0, "noisy")]
    # ε=1: noisy reference (gt@1) diverges -> the two differ.
    assert abs(seed.loc[(1.0, "clean")] - seed.loc[(1.0, "noisy")]) > 0.01


def test_gt_is_a_paired_bar_role():
    """The ground truth is shown as a paired-bar role (gt vs clean / gt vs noisy
    gt), alongside the recovery roles."""
    from scripts.plot_recovery_clean_vs_noisy_gt import PLOT_ROLES

    assert PLOT_ROLES[-1] == "gt"
    assert {"seed", "surfaced", "surfaced (best per run)", "gt"} == set(PLOT_ROLES)


def test_main_writes_both_figures(tmp_path):
    """End-to-end: build + paired-bar MSE and correlation figures (svg + png)."""
    from scripts.plot_recovery_clean_vs_noisy_gt import main

    rc = main([
        "--families", "ttb_sampling",
        "--noises", "0.0", "1.0",
        "--n-draws", "5",
        "--seed", "0",
        "--out-dir", str(tmp_path),
        "--csv", str(tmp_path / "merged.csv"),
    ])
    assert rc == 0
    for stem in ("recovery_clean_vs_noisy_gt_mse",
                 "recovery_clean_vs_noisy_gt_correlation"):
        assert (tmp_path / f"{stem}.png").is_file()
    # merged table carries the reference tag for re-plotting.
    merged = pd.read_csv(tmp_path / "merged.csv")
    assert set(merged["reference"].unique()) == {"clean", "noisy"}


def test_from_csv_replots_without_resimulation(tmp_path):
    """`--from-csv` re-plots straight from a merged long table (no run-dirs)."""
    from scripts.plot_recovery_clean_vs_noisy_gt import main

    rows = []
    for ref in ("clean", "noisy"):
        for nz in (0.0, 0.5):
            for role in ("seed", "surfaced"):
                rows.append(dict(family="ttb_sampling", noise=nz, run_dir="r0",
                                 role=role, theory_label=role, n_stimuli=24,
                                 pearson_r=0.8, mse=0.05, lag1_autocorr=0.0,
                                 reference=ref))
    csv = tmp_path / "merged.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    rc = main(["--from-csv", str(csv), "--out-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "recovery_clean_vs_noisy_gt_mse.png").is_file()
