"""Tests for standardizing the preregistration flagship figure bundle to the
human-dataset look, plus the folder-parameterized scripts/ driver.

Strong analytic checks where possible: exact canonical colours (sandy/gray/
slate/sage), exact rcParams font sizes, the per-rival H1 bar colours read off
the rendered patches, and that big suptitles are gone while per-panel titles
stay. Plus a smoke check that save_figure now writes PNG/SVG/PDF.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import same_color

import pytest

from scripts.figure_style import FONTSIZE, GRAY, NEUTRAL_HARMONY

BUNDLE = Path(
    "results/human_decision_making_cardinal/"
    "ttb+tallying/preregistration_visualization"
)
SANDY = NEUTRAL_HARMONY["sandy"]
SLATE = NEUTRAL_HARMONY["slate"]
SAGE = NEUTRAL_HARMONY["sage"]

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="bundle absent")

# The bundle imports its peers absolutely (`from style import ...`); put it on
# the path once. No repo module shares these names (verified).
if str(BUNDLE.resolve()) not in sys.path:
    sys.path.insert(0, str(BUNDLE.resolve()))


# --- canonical palette (exact) ---------------------------------------------
def test_bundle_colors_match_canonical_tallying_palette():
    import style

    assert style.CONCAVE == SANDY          # Diminishing Returns WADD (pi_7)
    assert style.STEEP == SANDY            # steep region / steep option
    assert style.WADD == GRAY              # WADD (pi_3 transient)
    assert style.FLAT == GRAY              # flat region / flat option
    assert style.TALLYING == SLATE         # Tallying (pi_2 seed)
    assert style.TTB == SAGE               # TTB (pi_1 seed)


def test_rival_colors_map_each_rival_to_its_canonical_colour():
    import style

    assert style.RIVAL_COLORS == {
        "wadd": GRAY, "tallying": SLATE, "ttb": SAGE,
    }


def test_neutral_harmony_does_not_drift_from_repo_palette():
    import style

    assert style.NEUTRAL_HARMONY == NEUTRAL_HARMONY


def test_gain_arrow_contrasts_with_the_region_washes():
    # The "big gain" arrow sits inside the steep-region wash; it must not be the
    # same colour as that wash (or the steep/flat region colours) or it vanishes.
    import style

    assert style.GAIN_ARROW != style.STEEP
    assert style.GAIN_ARROW != style.STEEP_FILL
    assert style.GAIN_ARROW != style.FLAT
    assert style.GAIN_ARROW != style.FLAT_FILL


# --- fonts / axes / output format ------------------------------------------
def test_apply_style_sets_axis_and_tick_fontsizes():
    """`apply_style` must publish the bundle's own declared sizes.

    This bundle deliberately enlarges axis/tick labels for legibility (18/16)
    instead of inheriting `scripts/figure_style.FONTSIZE` (14) — see the comment
    on `FONTS["size_axis_label"]`. Asserting against `style.FONTS` tracks that
    decision; the previous `== FONTSIZE` anchor silently went stale because this
    whole module was skipping on a pre-rename bundle path.
    """
    import style

    style.apply_style()
    assert plt.rcParams["axes.labelsize"] == style.FONTS["size_axis_label"]
    assert plt.rcParams["xtick.labelsize"] == style.FONTS["size_tick"]
    assert plt.rcParams["ytick.labelsize"] == style.FONTS["size_tick"]
    # Axis labels stay 2pt above tick labels, as in the original 14/12 pairing.
    assert style.FONTS["size_axis_label"] == style.FONTS["size_tick"] + 2


def test_save_figure_writes_png_svg_and_pdf(tmp_path):
    import style

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    style.save_figure(fig, "probe", out_dir=tmp_path)
    plt.close(fig)
    for ext in ("png", "svg", "pdf"):
        assert (tmp_path / f"probe.{ext}").exists(), ext


# --- per-rival H1 bar colours (read off the rendered patches) --------------
def test_h1_result_bars_use_per_rival_canonical_colours():
    import plot_flagship

    fig, ax = plt.subplots()
    plot_flagship.draw_result_h1(ax)
    # Bars are added in (dr, rival) pairs for wadd, tallying, ttb in order.
    bars = ax.patches
    assert len(bars) == 6
    dr_colours = [bars[i].get_facecolor() for i in (0, 2, 4)]
    rival_colours = [bars[i].get_facecolor() for i in (1, 3, 5)]
    for c in dr_colours:
        assert same_color(c, SANDY)
    assert same_color(rival_colours[0], GRAY)    # vs WADD
    assert same_color(rival_colours[1], SLATE)   # vs Tallying
    assert same_color(rival_colours[2], SAGE)    # vs TTB
    plt.close(fig)


def test_h2_result_steep_is_sandy_flat_is_gray():
    import plot_flagship

    fig, ax = plt.subplots()
    plot_flagship.draw_result_h2(ax)
    bars = ax.patches
    assert len(bars) == 2
    assert same_color(bars[0].get_facecolor(), SANDY)   # chose steep option
    assert same_color(bars[1].get_facecolor(), GRAY)    # chose flat option
    plt.close(fig)


# --- titles: big suptitle gone, per-panel titles kept ----------------------
def test_flagship_drops_suptitle_but_keeps_panel_titles():
    import plot_flagship

    fig = plot_flagship.build_flagship_figure()
    assert fig._suptitle is None
    assert any(ax.get_title() for ax in fig.axes)
    plt.close(fig)


# --- driver wiring ----------------------------------------------------------
def test_driver_renders_a_layout_with_png_svg_pdf(tmp_path):
    """The driver must render a layout to png+svg+pdf.

    The bundle's plot modules hardcode their own `output/` directory and are
    imported by name, so `render_layout` always writes into the real bundle no
    matter which path we hand it (earlier tests in this file have already
    imported those modules, and importlib returns the cached copies). Since
    `results/` is the paper's record, snapshot whatever the render touches and
    put it back — a re-render is byte-different (matplotlib stamps a timestamp
    and random element ids) even when it is visually identical.
    """
    import shutil

    from scripts.plot_preregistration_figures import render_layout

    out_dir = BUNDLE / "output"
    backup = tmp_path / "output_backup"
    if out_dir.is_dir():
        shutil.copytree(out_dir, backup)
    try:
        paths = render_layout(BUNDLE, "flagship")
        exts = {p.suffix for p in paths}
        assert {".png", ".svg", ".pdf"} <= exts
        for p in paths:
            assert p.exists()
    finally:
        if backup.is_dir():
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.copytree(backup, out_dir)


def test_render_bundle_lists_all_entry_points():
    from scripts.plot_preregistration_figures import LAYOUTS

    assert set(LAYOUTS) == {
        "flagship", "columns", "rows", "single_row",
        "single_row_h1_first", "panels",
    }
