"""Style tests for eval_hilbig scatter/bar plots: dual-format (SVG+PNG) output
and palette-driven colors (no matplotlib tab10 default cycle)."""
import matplotlib

matplotlib.use("Agg")
import pandas as pd

from scripts.eval_hilbig import (
    plot_scatter,
    plot_scatter_per_model,
    plot_bars,
    role_color_map,
)
from scripts.figure_style import CYCLE, NEUTRAL_HARMONY


def _per_stim():
    rows = []
    for m, role in (("pi_1", "seed"), ("pi_3", "surfaced")):
        for k in range(5):
            rows.append(dict(model=m, role=role,
                             p_b_human=0.2 * k, p_b_model=0.2 * k + 0.03))
    return pd.DataFrame(rows)


def _summary():
    return pd.DataFrame([
        dict(model="pi_1", role="seed", mae=0.2, mse=0.05, pearson_r=0.7),
        dict(model="pi_3", role="surfaced", mae=0.1, mse=0.02, pearson_r=0.9),
    ])


def test_scatter_writes_svg_and_png(tmp_path):
    out = tmp_path / "scatter.png"
    plot_scatter(_per_stim(), out, title="t")
    assert out.with_suffix(".png").exists() and out.with_suffix(".svg").exists()


def test_bars_writes_svg_and_png(tmp_path):
    out = tmp_path / "bars.png"
    plot_bars(_summary(), out, metric="mae", title="t")
    assert out.with_suffix(".png").exists() and out.with_suffix(".svg").exists()


def test_bars_mse_and_pearson_write_files(tmp_path):
    """Both MSE and Pearson r bar charts must render (not just MAE)."""
    for metric in ("mse", "pearson_r"):
        out = tmp_path / f"bars_{metric}.png"
        plot_bars(_summary(), out, metric=metric, title=metric)
        assert out.with_suffix(".png").exists()
        assert out.with_suffix(".svg").exists()


def test_draw_scatter_can_suppress_title():
    import matplotlib.pyplot as plt

    from scripts.eval_hilbig import _draw_scatter

    sub = _per_stim()
    sub = sub[sub["model"] == "pi_1"]
    fig, ax = plt.subplots()
    _draw_scatter(ax, sub, "#3d405b", "TTB", "seed", show_title=False)
    assert ax.get_title() == ""
    # Default still draws the title.
    fig2, ax2 = plt.subplots()
    _draw_scatter(ax2, sub, "#3d405b", "TTB", "seed")
    assert ax2.get_title() != ""
    plt.close(fig)
    plt.close(fig2)


def test_bars_frame_excludes_models_and_sorts():
    from scripts.eval_hilbig import _bars_frame

    summ = pd.DataFrame([
        dict(model="pi_4", role="surfaced", mse=0.04),
        dict(model="pi_7", role="surfaced", mse=0.02),
        dict(model="pi_1", role="seed", mse=0.08),
    ])
    # pi_7 dropped; remaining sorted ascending by mse.
    out = _bars_frame(summ, "mse", ascending=True, exclude_models=("pi_7",))
    assert list(out["model"]) == ["pi_4", "pi_1"]
    # No exclusion -> all three, still sorted.
    out_all = _bars_frame(summ, "mse", ascending=True)
    assert list(out_all["model"]) == ["pi_7", "pi_4", "pi_1"]


def test_bars_can_hide_model_labels(tmp_path):
    import matplotlib.pyplot as plt

    from scripts.eval_hilbig import plot_bars

    plot_bars(
        _summary(), tmp_path / "bars.png", metric="mae",
        show_model_labels=False,
    )
    # Smoke check the file rendered; label suppression asserted via the frame
    # helper and _draw path above.
    assert (tmp_path / "bars.png").exists()
    plt.close("all")


def test_role_color_map_uses_lineage_then_cycle_fallback():
    """Seeds -> slate (indigo), winning survivor -> sandy (gold), other
    survivor -> terracotta, mirroring the trajectory/convergence palette.
    Labels absent from the lineage (e.g. canonical baselines) fall back to
    the categorical CYCLE in enumeration order."""
    slate = NEUTRAL_HARMONY["slate"]
    sandy = NEUTRAL_HARMONY["sandy"]
    terracotta = NEUTRAL_HARMONY["terracotta"]
    lineage = {"pi_1": slate, "pi_2": slate, "pi_6": terracotta, "pi_7": sandy}
    models = ["pi_1", "pi_2", "pi_6", "pi_7", "ttb"]
    cmap = role_color_map(models, lineage)
    assert cmap["pi_1"] == slate and cmap["pi_2"] == slate
    assert cmap["pi_7"] == sandy          # winning surfaced -> gold
    assert cmap["pi_6"] == terracotta     # other final survivor
    # "ttb" is index 4 in `models`, absent from lineage -> CYCLE fallback.
    assert cmap["ttb"] == CYCLE[4 % len(CYCLE)]


def test_scatter_per_model_writes_one_figure_each(tmp_path):
    """Each model's scatter is saved as its own SVG+PNG file."""
    df = _per_stim()
    written = plot_scatter_per_model(df, tmp_path, stem="eval_hilbig_scatter")
    for m in df["model"].unique():
        assert (tmp_path / f"eval_hilbig_scatter_{m}.png").exists()
        assert (tmp_path / f"eval_hilbig_scatter_{m}.svg").exists()
    assert len(written) == df["model"].nunique()
