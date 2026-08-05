"""Standalone PNAS figures for the round-0 WADD case study.

Decomposes the round-0 pipeline of the synthetic WADD ground-truth run into
per-stage, per-experiment figures (stages 1-5), restyled with
scripts/figure_style.py. Parsers are reused from case_study_wadd_narrowing.py;
only the data helpers here are unit-tested (renderers are iterated visually).

Run:  python scripts/plot_case_study_wadd.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.case_study_wadd_narrowing import (  # noqa: E402
    parse_experiment, parse_arbitration,
    parse_wadd_from_theories, count_subjects,
    key_insight_sentence, _first_sentence, _wrap, _trim,
)
from scripts.figure_style import (  # noqa: E402
    FONTSIZE, GRAY, INK, NEUTRAL_HARMONY,
    save_figure, style_axes,
)

DEFAULT_RUN_DIR = _REPO_ROOT / (
    "results/heuristic_decision_making/synthetic_corrected_theories_binary_sampling/"
    "wadd_sampling/noise=0.0/"
    "dmb_ground_truth_wadd_sampling_noise=0.0_gemini-3.1-pro-preview_run3"
)

# Case-study palette (PNAS): TTB in gold, Tallying in indigo, and WADD (plus the
# WADD-generated observed data) in peach/terracotta.
TTB_COLOR = NEUTRAL_HARMONY["sandy"]        # gold
TALLY_COLOR = NEUTRAL_HARMONY["slate"]      # indigo
WADD_COLOR = NEUTRAL_HARMONY["terracotta"]  # peach
OBS_COLOR = NEUTRAL_HARMONY["terracotta"]   # peach
# Light neutral fill for the inner pull-quote boxes on the stage 4/5 cards.
BOX_TINT = "#ececef"


# -- data helpers (mirror the pipeline metric source exactly) ----------------

def sem(var: float, n: int) -> float:
    """Standard error of the mean from a between-subject variance."""
    return float(np.sqrt(var / n)) if var > 0 and n > 0 else 0.0


def ttb_choice(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row TTB choice: first COLUMN-ORDER discriminating cue (0=A, 1=B).

    Mirrors the metric source: `first_diff_idx = np.argmax(diff != 0)`. Only
    consulted on disagreement rows, where a discriminating cue exists.
    """
    diff = np.asarray(A) - np.asarray(B)
    first_diff = np.argmax(diff != 0, axis=1)
    chosen = diff[np.arange(len(diff)), first_diff]
    # `== 1` (not `> 0`) mirrors the pipeline metric_source verbatim so this
    # reconstruction reproduces the arbiter's numbers. For binary 0/1 cues the
    # first discriminating diff is always +/-1, so == 1 and > 0 coincide here.
    return np.where(chosen == 1, 0, 1)


def tally_choice(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row Tallying choice: more feature wins (0=A, 1=B, -1=tie)."""
    diff = np.asarray(A) - np.asarray(B)
    a_wins = np.sum(diff == 1, axis=1)
    b_wins = np.sum(diff == -1, axis=1)
    out = np.full(len(diff), -1)
    out[b_wins > a_wins] = 1
    out[a_wins > b_wins] = 0
    return out


def wadd_choice(A: np.ndarray, B: np.ndarray, validities) -> np.ndarray:
    """Per-row WADD choice: argmax of the validity-weighted cue sum (0=A, 1=B, -1=tie).

    Each option's score is sum_i validity_i * cue_i; the option with the larger
    weighted sum is chosen. This is the schematic, deterministic WADD pick (no
    softmax/lapse) used to compare the three theories' predictions on a trial.
    """
    v = np.asarray(validities, dtype=float)
    score_a = np.asarray(A, dtype=float) @ v
    score_b = np.asarray(B, dtype=float) @ v
    out = np.full(len(score_a), -1)
    out[score_a > score_b] = 0
    out[score_b > score_a] = 1
    return out


def disagreement_mask(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rows where TTB and Tallying make strictly opposing (non-tie) choices."""
    diff = np.asarray(A) - np.asarray(B)
    has_diff = (diff != 0).any(axis=1)
    ttb = ttb_choice(A, B)
    tally = tally_choice(A, B)
    return has_diff & (tally != -1) & (ttb != tally)


def reconstruct_metric(obs: list[dict], target: str) -> float:
    """Fraction of disagreement rows whose `response` matches `target`.

    `target` is "ttb" or "tally". Equals the arbiter's observed metric value.
    Returns 0.5 when no disagreement rows exist (mirrors the metric source's empty-set fallback).
    """
    # Validate at entry so an unknown target raises regardless of the data
    # (the empty-mask fallback below must not short-circuit the check).
    if target not in ("ttb", "tally"):
        raise ValueError(f"target must be 'ttb' or 'tally', got {target!r}")
    A = np.array([o["option_a_ratings"] for o in obs])
    B = np.array([o["option_b_ratings"] for o in obs])
    resp = np.array([o["response"] for o in obs])
    mask = disagreement_mask(A, B)
    if mask.sum() == 0:
        return 0.5
    pred = ttb_choice(A, B) if target == "ttb" else tally_choice(A, B)
    return float(np.mean(resp[mask] == pred[mask]))


def load_observations(run_dir: Path, exp_idx: int) -> list[dict]:
    """Round-0 observations for experiment `exp_idx` (1->obs_00, 2->obs_01)."""
    path = Path(run_dir) / "observations" / "data" / f"round_000_obs_0{exp_idx - 1}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_validity_order(validities) -> None:
    """Guard: column-order TTB == validity-order TTB only when validities desc."""
    v = list(validities)
    if v != sorted(v, reverse=True):
        raise ValueError(
            "validities must be in descending column order for the metric's "
            f"column-order TTB to equal validity-order TTB; got {v}"
        )


# -- Stage 1: experiment design renderer -------------------------------------

def _text_on(facecolor: str) -> str:
    """Return readable ink or white label color for a given fill color."""
    r, g, b = to_rgb(facecolor)
    return "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"


def _text_safe(color: str) -> str:
    """Theory color for text on a WHITE background, falling back to ink when the
    color is too light to read on white (e.g. gold) so labels never wash out."""
    r, g, b = to_rgb(color)
    return color if (0.299 * r + 0.587 * g + 0.114 * b) <= 0.6 else INK


def _option_color(which: int, ttb_pick: int, tally_pick: int, is_disagree: bool) -> str:
    """Block fill: gray on agreement; else the color of the theory that picked it."""
    if not is_disagree:
        return GRAY
    return TTB_COLOR if ttb_pick == which else TALLY_COLOR


def _draw_option_blocks(ax, a_row, b_row, *, y, cell, gap, ttb_pick, tally_pick,
                        is_disagree, n_feat, value_fontsize=FONTSIZE - 4):
    """Two side-by-side cue-tile blocks (Product A | Product B) at height y."""
    block_w = n_feat * cell
    for which, (x0, opt) in enumerate([(0.0, a_row), (block_w + gap, b_row)]):
        fc = _option_color(which, ttb_pick, tally_pick, is_disagree)
        for j in range(n_feat):
            ax.add_patch(Rectangle((x0 + j * cell, y), cell, 1.0,
                                   facecolor=fc, edgecolor="white", linewidth=1.0))
            ax.text(x0 + j * cell + cell / 2, y + 0.5, str(int(opt[j])),
                    ha="center", va="center", fontsize=value_fontsize,
                    color=_text_on(fc), fontweight="bold")
    return block_w


def render_stage1(exp: dict):
    """Validity bars + per-trial [Product A][Product B] blocks colored by choice."""
    A = exp["trial_a"]; B = exp["trial_b"]
    vals = np.asarray(exp["validities"], dtype=float)
    assert_validity_order(vals)
    n_trials, n_feat = A.shape
    ttb = ttb_choice(A, B); tally = tally_choice(A, B); dis = disagreement_mask(A, B)

    cell, gap = 1.0, 1.2
    block_w = n_feat * cell
    b_x0 = block_w + gap

    fig, ax = plt.subplots(figsize=(8.5, 0.62 * n_trials + 2.4))
    ax.set_axis_off()

    # Validity bars + feature labels on top.
    bar_base = n_trials + 0.45
    bar_h = 0.85
    for x0 in (0.0, b_x0):
        for j in range(n_feat):
            h = bar_h * vals[j] / vals.max()
            ax.add_patch(Rectangle((x0 + j * cell, bar_base), cell, h,
                                   facecolor=INK, edgecolor="white", linewidth=1.2))
            ax.text(x0 + j * cell + cell / 2, bar_base + h + 0.06, f"{vals[j]:.2f}",
                    ha="center", va="bottom", fontsize=FONTSIZE - 2, color=INK)
            ax.text(x0 + j * cell + cell / 2, bar_base - 0.12, f"f{j + 1}",
                    ha="center", va="top", fontsize=FONTSIZE - 7, color=INK)

    # Trial rows.
    for t in range(n_trials):
        y = n_trials - 1 - t
        _draw_option_blocks(ax, A[t], B[t], y=y, cell=cell, gap=gap,
                            ttb_pick=int(ttb[t]), tally_pick=int(tally[t]),
                            is_disagree=bool(dis[t]), n_feat=n_feat,
                            value_fontsize=FONTSIZE + 2)
        ax.text(-0.35, y + 0.5, f"t{t + 1}", ha="right", va="center",
                fontsize=FONTSIZE - 6, color=INK)

    # Column-group titles, placed above the validity bars so they clear the
    # per-feature (f1..fN) labels that sit just below the bars.
    label_y = bar_base + bar_h + 0.45
    ax.text(block_w / 2, label_y, "Product A", ha="center", va="bottom",
            fontsize=FONTSIZE + 2, fontweight="bold")
    ax.text(b_x0 + block_w / 2, label_y, "Product B", ha="center", va="bottom",
            fontsize=FONTSIZE + 2, fontweight="bold")

    # Legend.
    handles = [
        Patch(facecolor=TTB_COLOR, label="TTB's choice"),
        Patch(facecolor=TALLY_COLOR, label="Tallying's choice"),
        Patch(facecolor=GRAY, label="agreement"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
              fontsize=FONTSIZE - 5, bbox_to_anchor=(0.5, -0.06))

    ax.set_xlim(-1.4, b_x0 + block_w + 0.4)
    ax.set_ylim(-1.0, bar_base + bar_h + 1.1)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def render_stage1_choices(exp: dict):
    """Schematic of each theory's deterministic pick (A/B) per designed trial.

    Pairs with render_stage1 (SAME designed option pairs, SAME row order with t1
    at top): one column each for TTB, Tallying, and WADD, cell filled with the
    model's family color and labeled with its predicted choice (A/B, or - on a
    tie). No stimulus blocks here — those live in the stage-1 design figure.
    """
    A = exp["trial_a"]; B = exp["trial_b"]
    vals = exp["validities"]
    assert_validity_order(vals)  # TTB column uses column-order TTB (see render_stage1)
    n_trials = A.shape[0]
    columns = [
        ("TTB", ttb_choice(A, B), TTB_COLOR),
        ("Tallying", tally_choice(A, B), TALLY_COLOR),
        ("WADD", wadd_choice(A, B, vals), WADD_COLOR),
    ]
    letter = {0: "A", 1: "B", -1: "—"}
    col_w = 1.8  # column width — wide enough for the 'Tallying' header at large font

    fig, ax = plt.subplots(figsize=(6.0, 0.55 * n_trials + 2.0))
    ax.set_axis_off()

    for col, (name, choice, color) in enumerate(columns):
        x = col * col_w
        ax.text(x + col_w / 2, n_trials + 0.25, name, ha="center", va="bottom",
                fontsize=FONTSIZE + 2, fontweight="bold", color=_text_safe(color))
        for t in range(n_trials):
            y = n_trials - 1 - t
            ax.add_patch(Rectangle((x, y), col_w, 1.0, facecolor=color,
                                   edgecolor="white", linewidth=1.5))
            ax.text(x + col_w / 2, y + 0.5, letter[int(choice[t])], ha="center",
                    va="center", fontsize=FONTSIZE + 1, fontweight="bold", color=_text_on(color))

    for t in range(n_trials):
        y = n_trials - 1 - t
        ax.text(-0.4, y + 0.5, f"t{t + 1}", ha="right", va="center",
                fontsize=FONTSIZE - 5, color=INK)

    ax.set_xlim(-1.3, 3 * col_w + 0.2)
    ax.set_ylim(-0.4, n_trials + 1.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# -- Stage 2: metric renderer -------------------------------------------------

def _per_trial_target_fraction(exp, obs, target):
    """For each DESIGN disagreement trial, fraction of obs rows on that exact
    option-pair whose response matched the target theory's choice."""
    A = exp["trial_a"]; B = exp["trial_b"]
    dis = disagreement_mask(A, B)
    oa = np.array([o["option_a_ratings"] for o in obs])
    ob = np.array([o["option_b_ratings"] for o in obs])
    resp = np.array([o["response"] for o in obs])
    rows = []
    for t in np.where(dis)[0]:
        pick = (ttb_choice(A[t:t+1], B[t:t+1]) if target == "ttb"
                else tally_choice(A[t:t+1], B[t:t+1]))[0]
        sel = np.all(oa == A[t], axis=1) & np.all(ob == B[t], axis=1)
        frac = float(np.mean(resp[sel] == pick)) if sel.any() else float("nan")
        rows.append((t, int(pick), frac))
    return rows


def render_stage2(exp, obs, *, target, variant):
    """Stage 2 figure: metric definition in three variants.

    variant="schematic" — per-trial fraction of subjects choosing the target
        theory's option, annotated on each disagreement trial block; the mean
        must equal reconstruct_metric(obs, target) exactly.
    variant="formula"   — the algebraic definition of the metric.
    variant="axis"      — a [0,1] number line showing the observed value.
    """
    other = "Tallying" if target == "ttb" else "TTB"
    this = "TTB" if target == "ttb" else "Tallying"
    metric_val = reconstruct_metric(obs, target)

    if variant == "schematic":
        A = exp["trial_a"]; B = exp["trial_b"]
        n_feat = A.shape[1]
        ttb = ttb_choice(A, B); tally = tally_choice(A, B)
        rows = _per_trial_target_fraction(exp, obs, target)
        cell, gap = 1.0, 1.0
        block_w = n_feat * cell; b_x0 = block_w + gap
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 2.0))
        ax.set_axis_off()
        for i, (t, pick, frac) in enumerate(rows):
            y = len(rows) - 1 - i
            _draw_option_blocks(ax, A[t], B[t], y=y, cell=cell, gap=gap,
                                ttb_pick=int(ttb[t]), tally_pick=int(tally[t]),
                                is_disagree=True, n_feat=n_feat)
            ax.text(b_x0 + block_w + 0.6, y + 0.5,
                    f"chose {this}: {frac:.2f}", va="center",
                    fontsize=FONTSIZE - 5, color=TTB_COLOR if target == "ttb" else TALLY_COLOR)
        ax.text(b_x0 + block_w + 0.6, -0.9,
                f"mean = metric = {metric_val:.3f}", va="center",
                fontsize=FONTSIZE - 3, fontweight="bold", color=INK)
        ax.set_xlim(-0.4, b_x0 + block_w + 4.0)
        ax.set_ylim(-1.4, len(rows) + 0.4)
        ax.set_aspect("equal")

    elif variant == "formula":
        fig, ax = plt.subplots(figsize=(7.5, 3.2)); ax.set_axis_off()
        ax.text(0.5, 0.74,
                r"$m = \dfrac{1}{|D|}\sum_{t\in D}\mathbb{1}\!\left[\,c_t = "
                + (r"\mathrm{TTB}_t" if target == "ttb" else r"\mathrm{Tally}_t") + r"\,\right]$",
                ha="center", va="center", fontsize=FONTSIZE + 4, color=INK)
        ax.text(0.5, 0.40, r"$D$ = trials where TTB and Tallying disagree;  "
                r"$c_t$ = subject's choice", ha="center", fontsize=FONTSIZE - 3)
        ax.text(0.5, 0.20, f"m = 1 → all-{this}-like      m = 0 → all-{other}-like",
                ha="center", fontsize=FONTSIZE - 3, color=GRAY)

    elif variant == "axis":
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.hlines(0, 0, 1, color=INK, lw=1.2)
        ax.scatter([metric_val], [0], marker="D", s=130, color=OBS_COLOR,
                   edgecolor="white", zorder=5)
        ax.text(metric_val, 0.12, f"observed = {metric_val:.3f}", ha="center",
                fontsize=FONTSIZE - 4, fontweight="bold")
        ax.text(0, -0.18, f"0\n{other}-like", ha="center", va="top", fontsize=FONTSIZE - 5)
        ax.text(1, -0.18, f"1\n{this}-like", ha="center", va="top", fontsize=FONTSIZE - 5)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.4, 0.3); ax.axis("off")

    else:
        raise ValueError(f"unknown variant: {variant}")

    fig.tight_layout()
    return fig


# -- Stage 3: predictions vs observed renderer --------------------------------

_METRIC_LABEL = {
    "ttb": "TTB-match rate (disagreement trials)",
    "tally": "Tallying-match rate (disagreement trials)",
}


def render_stage3(exp_results, *, target, n_subjects, layout):
    """TTB, Tallying, and Observed on the metric in [0,1], with SEM."""
    if target not in _METRIC_LABEL:
        raise ValueError(f"target must be 'ttb' or 'tally', got {target!r}")
    ttb_m, ttb_v = exp_results["pi_1"]
    tal_m, tal_v = exp_results["pi_2"]
    obs_m, obs_v = exp_results["observed"]
    points = [("TTB", ttb_m, sem(ttb_v, n_subjects), TTB_COLOR, "o"),
              ("Tallying", tal_m, sem(tal_v, n_subjects), TALLY_COLOR, "s"),
              ("Observed", obs_m, sem(obs_v, n_subjects), OBS_COLOR, "D")]
    xlabel = _METRIC_LABEL[target]

    if layout == "numberline":
        fig, ax = plt.subplots(figsize=(9, 3.0))
        for name, m, e, c, mk in points:
            ax.errorbar(m, 0, xerr=e, fmt=mk, color=c, markersize=15,
                        capsize=5, markeredgecolor="white", zorder=5, label=name)
            ax.text(m, 0.085, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=FONTSIZE + 1, color=_text_safe(c), fontweight="bold")
        ax.set_yticks([]); ax.set_ylim(-0.2, 0.32); ax.set_xlim(-0.02, 1.02)
        ax.spines["left"].set_visible(False)
        style_axes(ax, xlabel=xlabel)
        ax.xaxis.label.set_size(FONTSIZE + 4)
        ax.tick_params(axis="x", labelsize=FONTSIZE + 1)

    elif layout == "dots":
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for i, (name, m, e, c, mk) in enumerate(points):
            ax.errorbar(i, m, yerr=e, fmt=mk, color=c, markersize=13,
                        capsize=5, markeredgecolor="white")
            ax.text(i + 0.12, m, f"{m:.3f}", va="center",
                    fontsize=FONTSIZE - 5, color=_text_safe(c), fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels([p[0] for p in points])
        ax.set_xlim(-0.5, 2.7); ax.set_ylim(-0.02, 1.02)
        style_axes(ax, ylabel=xlabel)

    elif layout == "forest":
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axvline(obs_m, color=OBS_COLOR, lw=1.4, zorder=1)
        obs_sem = points[2][2]  # already computed above; avoid recomputing sem
        ax.axvspan(obs_m - obs_sem, obs_m + obs_sem,
                   color=GRAY, alpha=0.3, zorder=0)
        for i, (name, m, e, c, mk) in enumerate([points[0], points[1]]):
            ax.errorbar(m, i, xerr=e, fmt=mk, color=c, markersize=13,
                        capsize=5, markeredgecolor="white", zorder=5)
            ax.text(m, i + 0.22, f"{m:.3f}", ha="center",
                    fontsize=FONTSIZE - 5, color=_text_safe(c), fontweight="bold")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["TTB", "Tallying"])
        ax.set_ylim(-0.6, 1.8); ax.set_xlim(-0.02, 1.02)
        ax.text(obs_m, 1.6, "observed", ha="center", fontsize=FONTSIZE - 5,
                color=OBS_COLOR)
        style_axes(ax, xlabel=xlabel)
    else:
        raise ValueError(f"unknown layout: {layout}")

    fig.tight_layout()
    return fig


# -- Stage 4: arbiter rationale renderer --------------------------------------

def _quote_card(ax, *, edge):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.88,
                 boxstyle="round,pad=0.02,rounding_size=0.02",
                 transform=ax.transAxes, facecolor="white",
                 edgecolor=edge, linewidth=1.2))
    return ax


def render_stage4(arbiter_resp):
    """Arbiter rationale card: interpretation + pull-quote + verdict chip."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    _quote_card(ax, edge=INK)
    interp = arbiter_resp.get("interpretation", "")
    ax.text(0.5, 0.80, _wrap(_first_sentence(interp, n=1), 88), transform=ax.transAxes,
            ha="center", va="top", fontsize=FONTSIZE - 3, color=INK)
    ax.text(0.5, 0.45, "“" + _wrap(key_insight_sentence(interp), 70) + "”",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONTSIZE - 2, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BOX_TINT,
                      edgecolor=INK, linewidth=0.8))
    verdict = arbiter_resp.get("verdict", "?")
    ax.text(0.5, 0.13, f"verdict: {verdict}", transform=ax.transAxes,
            ha="center", va="center", fontsize=FONTSIZE - 4, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.4",
                                     facecolor=INK, edgecolor="none"))
    fig.tight_layout()
    return fig


# -- Stage 5: WADD description renderer --------------------------------------

def render_stage5(wadd):
    """WADD description card: description + equation + rationale + pull-quote."""
    fig, ax = plt.subplots(figsize=(9, 4))
    _quote_card(ax, edge=INK)
    ax.text(0.5, 0.85, _wrap(_trim(_first_sentence(wadd["description"], n=1), 300), 92),
            transform=ax.transAxes, ha="center", va="top", fontsize=FONTSIZE - 3, color=INK)
    ax.text(0.5, 0.55,
            r"$s_A = \sum_i v_i\, x_{A,i}$        $P(A) \propto \exp(\beta\, s_A)$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONTSIZE + 2, color=WADD_COLOR)
    ax.text(0.5, 0.30, _wrap(_trim(_first_sentence(wadd["rationale"], n=2), 320), 92),
            transform=ax.transAxes, ha="center", va="top", fontsize=FONTSIZE - 4, color=INK)
    ax.text(0.5, 0.10,
            "“validity-weighted sum of cue values → compensatory choice”",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONTSIZE - 3, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor=BOX_TINT, edgecolor=INK, linewidth=0.8))
    fig.tight_layout()
    return fig


# -- CLI / main ---------------------------------------------------------------

def main(run_dir: Path = DEFAULT_RUN_DIR, out_dir: Path | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "analysis" / "case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    exps = {1: parse_experiment(run_dir, "pi_1"), 2: parse_experiment(run_dir, "pi_2")}
    obs = {1: load_observations(run_dir, 1), 2: load_observations(run_dir, 2)}
    # Round-0 convention: experiment 1 is proposed by the TTB seed (pi_1) and
    # scored with the TTB-match metric; experiment 2 by the Tallying seed (pi_2).
    # Holds for this single-run case study; a reordered run would need this
    # derived from each experiment's "proposed by pi_N" header.
    targets = {1: "ttb", 2: "tally"}
    arb = parse_arbitration(run_dir)
    wadd = parse_wadd_from_theories(run_dir)
    n_sub = count_subjects(run_dir)

    written: list[Path] = []

    def emit(fig, name):
        nonlocal written
        written += save_figure(fig, out_dir / name)
        plt.close(fig)

    for k in (1, 2):
        emit(render_stage1(exps[k]), f"stage1_design_exp{k}")
        emit(render_stage1_choices(exps[k]), f"stage1_choices_exp{k}")
        for variant in ("schematic", "formula", "axis"):
            emit(render_stage2(exps[k], obs[k], target=targets[k], variant=variant),
                 f"stage2_metric_exp{k}_{variant}")
        for layout in ("numberline", "dots", "forest"):
            emit(render_stage3(arb["experiments"][k], target=targets[k],
                               n_subjects=n_sub, layout=layout),
                 f"stage3_gap_exp{k}_{layout}")

    emit(render_stage4(arb["response"]), "stage4_arbiter")
    emit(render_stage5(wadd), "stage5_wadd")

    print("wrote:", *(str(p) for p in written), sep="\n  ")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(run_dir=args.run_dir, out_dir=args.out_dir)
