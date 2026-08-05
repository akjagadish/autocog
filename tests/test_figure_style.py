"""Tests for the shared publication figure style module (scripts/figure_style.py)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from scripts.figure_style import (
    FONTSIZE,
    NEUTRAL_HARMONY,
    ROLE_COLOR,
    FAMILY_COLOR,
    save_figure,
    style_axes,
)


def test_palette_is_the_neutral_harmony_hexes():
    expected = {
        "cream": "#f4f1de",
        "terracotta": "#e07a5f",
        "slate": "#3d405b",
        "sage": "#81b29a",
        "sandy": "#f2cc8f",
    }
    assert NEUTRAL_HARMONY == expected


def test_semantic_maps_draw_only_from_palette_or_neutrals():
    allowed = set(NEUTRAL_HARMONY.values()) | {"#999999", "#111111"}
    for mapping in (ROLE_COLOR, FAMILY_COLOR):
        assert mapping, "semantic color map must not be empty"
        assert set(mapping.values()) <= allowed


def test_save_figure_writes_svg_and_png(tmp_path: Path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = save_figure(fig, tmp_path / "myfig")
    plt.close(fig)
    assert sorted(p.suffix for p in written) == [".png", ".svg"]
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_style_axes_despines_and_sets_fontsize():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    style_axes(ax, xlabel="x", ylabel="y")
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.xaxis.label.get_fontsize() == FONTSIZE
    assert ax.yaxis.label.get_fontsize() == FONTSIZE
    plt.close(fig)


def test_fontsize_constant():
    assert FONTSIZE == 14
