"""Tests for re-plotting eval_hilbig figures from existing CSVs."""
import matplotlib

matplotlib.use("Agg")
import pandas as pd

from scripts.plot_eval_hilbig_figures import replot


def _write_csvs(eval_dir):
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        dict(model="pi_1", role="base", n_stimuli=2,
             mae=0.20, mse=0.05, pearson_r=0.78),
        dict(model="pi_2", role="base", n_stimuli=2,
             mae=0.22, mse=0.06, pearson_r=0.80),
        dict(model="pi_4", role="surfaced", n_stimuli=2,
             mae=0.18, mse=0.03, pearson_r=0.95),
    ])
    rows = []
    for m, role in (("pi_1", "base"), ("pi_2", "base"), ("pi_4", "surfaced")):
        for k in range(2):
            rows.append(dict(model=m, role=role, option_a="[1, 1]",
                             option_b="[0, 0]", p_b_human=0.3 + 0.2 * k,
                             p_b_model=0.3 + 0.2 * k + 0.05, n_trials_human=10))
    per_stim = pd.DataFrame(rows)
    summary.to_csv(eval_dir / "eval_hilbig_summary.csv", index=False)
    per_stim.to_csv(eval_dir / "eval_hilbig_per_stimulus.csv", index=False)


def test_replot_writes_all_figures(tmp_path):
    """Re-plot must emit the scatter grid, one scatter per model, and a bar
    chart per metric (mae/mse/pearson_r), each as SVG + PNG. With an
    unparseable run_dir the colours fall back to CYCLE but files still write."""
    eval_dir = tmp_path / "analysis" / "eval_hilbig"
    _write_csvs(eval_dir)
    run_dir = tmp_path  # no rounds/ -> lineage + name maps fall back

    written = replot(eval_dir, run_dir)

    expected = [
        "eval_hilbig_scatter.png",
        "eval_hilbig_scatter_pi_1.png",
        "eval_hilbig_scatter_pi_2.png",
        "eval_hilbig_scatter_pi_4.png",
        "eval_hilbig_bars_mae.png",
        "eval_hilbig_bars_mse.png",
        "eval_hilbig_bars_pearson_r.png",
    ]
    for name in expected:
        png = eval_dir / name
        assert png.exists(), f"missing {name}"
        assert png.with_suffix(".svg").exists(), f"missing svg for {name}"
    assert len(written) == len(expected)
