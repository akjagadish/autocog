"""Tests for the similarity-vs-noise plot's data layer.

The matplotlib rendering is not unit-tested; the testable core is the
aggregation from per-(run, slot) similarity rows to a per-cell mean and SEM,
and the pooling across families. Those are checked against hand-computed values.
"""

import csv
import math
from pathlib import Path

import pytest

import numpy as np

from scripts.plot_similarity import (
    CellStat,
    collect,
    collect_scores,
    filter_max_noise,
    make_bar_swarm_figure,
    pool_across_families,
    pool_raw_by_noise,
    read_scores,
    summarize,
    _jitter,
    _parse_noise,
)


def test_make_bar_swarm_has_one_bar_per_noise_level(tmp_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")

    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.8, 1.0], "description")
    _write_csv(tmp_path / "wadd_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.5], "description")
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.5" / "judge_similarity_desc.csv",
               [0.4, 0.6], "description")
    raw = collect_scores(tmp_path, families=["ttb", "tallying", "wadd"])

    fig = make_bar_swarm_figure(raw, mode="description")
    ax = fig.axes[0]
    # One bar (Rectangle patch) per noise level present: 0.0 and 0.5.
    assert len(ax.patches) == 2
    # Bars are at categorical positions in ascending-noise order, and their
    # heights are the POOLED means: noise 0.0 -> mean([0.8,1.0,0.5]) = 0.7667,
    # noise 0.5 -> mean([0.4,0.6]) = 0.5.
    heights = [p.get_height() for p in ax.patches]
    assert heights[0] == pytest.approx((0.8 + 1.0 + 0.5) / 3)
    assert heights[1] == pytest.approx(0.5)
    # The pooled-mean entry is not shown in a legend on this figure.
    assert ax.get_legend() is None


def test_filter_max_noise_drops_higher_noise_levels() -> None:
    scores = {
        ("ttb", 0.0, "description"): [0.9],
        ("ttb", 0.75, "description"): [0.5],
        ("ttb", 1.0, "description"): [0.1],
        ("wadd", 1.0, "joint"): [0.2],
    }
    kept = filter_max_noise(scores, max_noise=0.75)
    assert set(kept) == {("ttb", 0.0, "description"), ("ttb", 0.75, "description")}
    # None keeps everything unchanged.
    assert filter_max_noise(scores, max_noise=None) == scores


def test_pool_raw_by_noise_concatenates_per_noise(tmp_path: Path) -> None:
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.8, 1.0], "description")
    _write_csv(tmp_path / "wadd_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.6, 0.4], "description")
    _write_csv(tmp_path / "tallying_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.0], "description")
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_joint.csv",
               [0.99], "joint")

    raw = collect_scores(tmp_path, families=["ttb", "tallying", "wadd"])
    pooled = pool_raw_by_noise(raw, mode="description")

    assert sorted(pooled[0.0]) == pytest.approx([0.0, 0.4, 0.6, 0.8, 1.0])
    assert 0.99 not in pooled[0.0]  # joint mode excluded


def test_jitter_is_seeded_and_bounded() -> None:
    a = _jitter(5, np.random.default_rng(0), width=0.01)
    b = _jitter(5, np.random.default_rng(0), width=0.01)
    assert np.allclose(a, b)            # same seed -> identical (reproducible)
    assert a.shape == (5,)
    assert np.all(np.abs(a) <= 0.01)    # within the requested half-width


def test_parse_noise_reads_float_suffix() -> None:
    assert _parse_noise(Path("noise=0.75")) == 0.75
    assert _parse_noise(Path("noise=0.0")) == 0.0
    assert _parse_noise(Path("noise=1.0")) == 1.0


def test_summarize_mean_and_sem_against_hand_values() -> None:
    # scores = [0.2, 0.4, 0.6]: mean = 0.4 exactly.
    # sample std (ddof=1) = 0.2, so SEM = 0.2 / sqrt(3).
    stat = summarize([0.2, 0.4, 0.6])
    assert isinstance(stat, CellStat)
    assert stat.n == 3
    assert stat.mean == pytest.approx(0.4)
    assert stat.sem == pytest.approx(0.2 / math.sqrt(3))


def test_summarize_single_value_has_zero_sem() -> None:
    # n = 1: SEM is undefined; collapse to zero spread.
    stat = summarize([0.73])
    assert stat.n == 1
    assert stat.mean == pytest.approx(0.73)
    assert stat.sem == 0.0


def test_summarize_identical_scores_has_zero_sem() -> None:
    # Zero variance: SEM is exactly 0 (and the scipy tstd warning is avoided).
    stat = summarize([0.5, 0.5, 0.5])
    assert stat.n == 3
    assert stat.mean == pytest.approx(0.5)
    assert stat.sem == 0.0


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarize([])


def _write_csv(path: Path, scores: list[float], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "slot", "input_mode", "similarity"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, s in enumerate(scores):
            w.writerow({"run": str(i), "slot": "1",
                        "input_mode": mode, "similarity": f"{s:.3f}"})


def test_read_scores_reads_similarity_column(tmp_path: Path) -> None:
    p = tmp_path / "judge_similarity_desc.csv"
    _write_csv(p, [0.1, 0.9, 0.5], "description")
    assert read_scores(p) == pytest.approx([0.1, 0.9, 0.5])


def test_collect_keys_by_family_noise_mode(tmp_path: Path) -> None:
    # Build root/ttb_sampling/noise=0.0/{desc,joint}.csv and one wadd cell.
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.8, 1.0], "description")
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_joint.csv",
               [0.6, 0.8], "joint")
    _write_csv(tmp_path / "wadd_sampling" / "noise=0.5" / "judge_similarity_desc.csv",
               [0.5, 0.5], "description")

    stats = collect(tmp_path, families=["ttb", "tallying", "wadd"])

    # tallying has no files -> absent, not an error.
    assert ("ttb", 0.0, "description") in stats
    assert ("ttb", 0.0, "joint") in stats
    assert ("wadd", 0.5, "description") in stats
    assert not any(k[0] == "tallying" for k in stats)

    assert stats[("ttb", 0.0, "description")].mean == pytest.approx(0.9)
    assert stats[("ttb", 0.0, "joint")].mean == pytest.approx(0.7)
    assert stats[("wadd", 0.5, "description")].mean == pytest.approx(0.5)


def test_collect_scores_returns_raw_lists(tmp_path: Path) -> None:
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.8, 1.0], "description")
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_joint.csv",
               [0.6, 0.8], "joint")

    raw = collect_scores(tmp_path, families=["ttb", "tallying", "wadd"])

    assert raw[("ttb", 0.0, "description")] == pytest.approx([0.8, 1.0])
    assert raw[("ttb", 0.0, "joint")] == pytest.approx([0.6, 0.8])
    assert not any(k[0] == "tallying" for k in raw)


def test_pool_across_families_pools_per_noise(tmp_path: Path) -> None:
    # Three families at noise 0.0, description mode; a joint file that must be
    # ignored when pooling the 'description' mode.
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.8, 1.0], "description")
    _write_csv(tmp_path / "wadd_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.6, 0.4], "description")
    _write_csv(tmp_path / "tallying_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.0], "description")
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_joint.csv",
               [0.99, 0.99], "joint")

    raw = collect_scores(tmp_path, families=["ttb", "tallying", "wadd"])
    pooled = pool_across_families(raw, mode="description")

    # Pooled description scores at noise 0.0: [0.8,1.0,0.6,0.4,0.0] -> mean 0.56, n 5.
    assert pooled[0.0].n == 5
    assert pooled[0.0].mean == pytest.approx(0.56)
    # The joint file did not leak into the description pool.
    assert pooled[0.0].mean != pytest.approx((0.8 + 1.0 + 0.6 + 0.4 + 0.0 + 0.99 + 0.99) / 7)


def test_collect_skips_non_numeric_noise_dir(tmp_path: Path) -> None:
    """A stray noise=<non-float> dir must be skipped, not crash the walk —
    matching how missing-CSV cells are simply absent."""
    _write_csv(tmp_path / "ttb_sampling" / "noise=0.0" / "judge_similarity_desc.csv",
               [0.8, 1.0], "description")
    # A sibling dir with a non-parseable noise suffix.
    (tmp_path / "ttb_sampling" / "noise=baseline").mkdir(parents=True)

    stats = collect(tmp_path, families=["ttb"])

    assert ("ttb", 0.0, "description") in stats
    assert not any(isinstance(k[1], str) for k in stats)
