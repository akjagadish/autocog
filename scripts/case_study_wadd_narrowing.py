"""Case study figure: how the LLM narrows from TTB+Tallying to WADD in round 0.

Reads existing round_000 artifacts from a run directory and renders a 5-stage
pipeline figure. Each stage is also saved as a standalone slide-sized PNG so the
narrative can be built up across multiple slides.

Stages (two parallel lanes in 1-3 merging at 4):
  1. experiment design            (dissociation matrix + validity bar)
  2. metric proposal              (code excerpt + plain-english tagline)
  3. predictions + observation    (bracket plot on [0, 1]) -- the anchor
  4. arbiter                      (rationale quote, highlighted phrase)
  5. theory proposer (WADD)       (description + equation + rationale)

Default run:
    results/wadd/noise=0.0/hdm_ground_truth_wadd_noise=0.0_gemini-3.1-pro-preview_run1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


# -- palette -----------------------------------------------------------------
# 4-color theory palette + near-black for text/observed.
COLOR_TTB = "#B7894C"       # warm tan
COLOR_TALLY = "#4F76B0"     # muted blue
COLOR_WADD = "#7C4B27"      # dark brown
COLOR_OBS = "#000000"       # pure black
COLOR_TEXT = "#111111"      # body text
# Neutrals (non-semantic chrome; not part of the 4-color palette).
COLOR_TIE = "#D0D0D0"
COLOR_BRACKET = "#E3E3E3"
COLOR_CODE_BG = "#F4F4F4"
COLOR_PANEL_BG = "#FAFAFA"
# Pull-quote tint: very light tan derived from #B7894C (20% blend with white).
COLOR_HIGHLIGHT_BG = "#F2E9DA"
COLOR_HIGHLIGHT_EDGE = "#B7894C"
# Blue analogue: 20% blend of #4F76B0 with white, for Stage 5.
COLOR_HIGHLIGHT_BG_BLUE = "#DCE4EF"
COLOR_HIGHLIGHT_EDGE_BLUE = "#4F76B0"


# -- parsers -----------------------------------------------------------------

JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _load_response_json(path: Path) -> dict[str, Any]:
    """Return the first ```json {...}``` block under the `## Response` header."""
    text = path.read_text()
    # Narrow to the part after "## Response" to avoid matching example blocks.
    if "## Response" in text:
        text = text.split("## Response", 1)[1]
    match = JSON_BLOCK.search(text)
    if match is None:
        raise ValueError(f"No JSON response block found in {path}")
    return json.loads(match.group(1))


def parse_experiment(run_dir: Path, pi: str) -> dict[str, Any]:
    """Parse experiment design and rationale for a given seed theory ('pi_1' or 'pi_2').

    `rating_max` is present only in the cardinal task (1..rating_max ratings);
    the binary task (0/1 features) omits it, so it is parsed as None there.
    """
    path = run_dir / "rounds" / "round_000" / pi / "prompts" / "experiment_attempt_00.md"
    resp = _load_response_json(path)
    rating_max = resp.get("rating_max")
    return {
        "validities": list(resp["validities"]),
        "rating_max": int(rating_max) if rating_max is not None else None,
        "trial_a": np.array(resp["trial_a_ratings"], dtype=int),
        "trial_b": np.array(resp["trial_b_ratings"], dtype=int),
        "rationale": resp["rationale"].strip(),
    }


def count_subjects(run_dir: Path) -> int:
    """Number of distinct subjects in the round-0 observations.

    Used for the observed-value SEM (`sqrt(between-subject var / n_subjects)`).
    Reads the first round-0 observation file and counts unique `subject_id`s.
    """
    obs_path = run_dir / "observations" / "data" / "round_000_obs_00.jsonl"
    subjects = {
        json.loads(line)["subject_id"]
        for line in obs_path.read_text().splitlines()
        if line.strip()
    }
    return len(subjects)


def run_subtitle(run_dir: Path) -> str:
    """Truthful one-line subtitle (task / ground truth / noise / run) from the path.

    The binary task lives under the `results/recovery` directory; everything else
    is the cardinal-rating task. Ground truth, noise, and run index are read from
    the run directory name.
    """
    name = run_dir.name
    task = "binary" if "results/recovery" in str(run_dir) else "cardinal"
    gt = re.search(r"ground_truth_([a-z]+)", name)
    noise = re.search(r"noise=([0-9.]+)", name)
    run = re.search(r"(run\d+)", name)
    return (
        f"{task} task  ·  ground truth = {gt.group(1).upper() if gt else '?'}"
        f"  ·  noise = {noise.group(1) if noise else '?'}"
        f"  ·  {run.group(1) if run else '?'}"
    )


def parse_metric(run_dir: Path, pi: str) -> dict[str, Any]:
    """Parse metric source and rationale for a given seed theory."""
    path = run_dir / "rounds" / "round_000" / pi / "prompts" / "metric_exp00_attempt_00.md"
    resp = _load_response_json(path)
    return {
        "source": resp["metric_source"],
        "rationale": resp["rationale"].strip(),
    }


RESULT_LINE = re.compile(
    r"-\s+Predicted under pi_(\d+)\s+\(simulated\):\s+([0-9.]+)\s+\(var=([0-9.]+)\)"
)
OBSERVED_LINE = re.compile(
    r"-\s+Observed on real data:\s+([0-9.]+)\s+\(var=([0-9.]+)\)"
)
EXPERIMENT_HEADER = re.compile(r"^##\s+EXPERIMENT\s+(\d+)\s+", re.MULTILINE)


def parse_arbitration(run_dir: Path) -> dict[str, Any]:
    """Return per-experiment predictions/observations and the arbiter's JSON response."""
    path = run_dir / "rounds" / "round_000" / "arbiter" / "prompts" / "arbitration.md"
    text = path.read_text()

    # Split into per-experiment blocks using the headers.
    headers = [(m.start(), int(m.group(1))) for m in EXPERIMENT_HEADER.finditer(text)]
    if not headers:
        raise ValueError(f"No experiment headers found in {path}")
    # Upper bound for last experiment: the '## PERFORMANCE' or '## RESPONSE' section.
    end_markers = [m.start() for m in re.finditer(r"^##\s+(PERFORMANCE|RESPONSE)\s", text, re.MULTILINE)]
    end_pos = min(end_markers) if end_markers else len(text)

    experiments: dict[int, dict[str, Any]] = {}
    for i, (start, idx) in enumerate(headers):
        stop = headers[i + 1][0] if i + 1 < len(headers) else end_pos
        block = text[start:stop]
        results: dict[str, tuple[float, float]] = {}
        for m in RESULT_LINE.finditer(block):
            pi = f"pi_{m.group(1)}"
            results[pi] = (float(m.group(2)), float(m.group(3)))
        obs_m = OBSERVED_LINE.search(block)
        if obs_m is None:
            raise ValueError(f"No observed line in experiment {idx} block of {path}")
        results["observed"] = (float(obs_m.group(1)), float(obs_m.group(2)))
        experiments[idx] = results

    # Arbiter response JSON.
    resp = _load_response_json(path)

    return {"experiments": experiments, "response": resp}


def parse_wadd_from_theories(run_dir: Path) -> dict[str, Any]:
    """Extract the WADD (pi_3) description and rationale from round_000/theories.md."""
    path = run_dir / "rounds" / "round_000" / "theories.md"
    text = path.read_text()

    # Find the pi_3 replacement section.
    m = re.search(r"### `pi_3` → slot 1 \(via `new_theory`\)(.+?)(?:\n###\s|\Z)", text, re.DOTALL)
    if m is None:
        raise ValueError(f"Could not locate pi_3 section in {path}")
    block = m.group(1)
    desc_m = re.search(r"\*\*Description:\*\*\s+(.+?)\n\n", block, re.DOTALL)
    rat_m = re.search(r"\*\*Rationale:\*\*\s+(.+?)\n\n", block, re.DOTALL)
    if desc_m is None or rat_m is None:
        raise ValueError(f"Could not parse description/rationale for pi_3 in {path}")
    return {
        "description": desc_m.group(1).strip(),
        "rationale": rat_m.group(1).strip(),
    }


# -- helpers -----------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Naively split prose into sentences on '. '/'! '/'? ' boundaries."""
    return re.split(r"(?<=[.!?])\s+", text)


def _first_sentence(text: str, *, n: int = 1) -> str:
    """Return the first `n` sentences from `text`, naively split on '. '."""
    return " ".join(_split_sentences(text)[:n]).strip()


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)


# Priority-ordered phrases that mark the arbiter's decisive insight — the reason
# WADD is proposed. The cardinal task hinges on magnitude sensitivity; the binary
# task hinges on validity-weighted (vs unweighted) compensation. We match
# "weighted compensatory strategy" (with the noun) rather than the bare
# "weighted compensatory" so the selector never latches onto an earlier
# "...unweighted compensatory rule..." clause, which contains it as a substring.
KEY_INSIGHT_TRIGGERS = (
    "cardinal magnitude",            # cardinal task
    "specific validit",             # binary task: integrates specific validities
    "weighted compensatory strategy",
)


def key_insight_sentence(interpretation: str) -> str:
    """The single sentence carrying the arbiter's decisive reason for WADD.

    Scans the interpretation for the highest-priority trigger phrase and returns
    the sentence containing it, verbatim. Falls back to the last sentence (the
    arbiter's conclusion) when no trigger matches.
    """
    sentences = [s.strip() for s in _split_sentences(interpretation.strip()) if s.strip()]
    for trigger in KEY_INSIGHT_TRIGGERS:
        for sentence in sentences:
            if trigger in sentence.lower():
                return sentence
    return sentences[-1] if sentences else interpretation.strip()


# -- renderers ---------------------------------------------------------------


def render_stage1_lane(ax, exp: dict[str, Any], *, lane_title: str,
                       show_values: bool = False) -> None:
    """Dissociation matrix: rows=trials, cols=features, cells=feature-wise winner.

    Two extra annotation columns on the right show TTB's predicted winner (first
    discriminating cue) and Tally's predicted winner (majority).
    """
    a = exp["trial_a"]
    b = exp["trial_b"]
    n_trials, n_feat = a.shape

    # Feature-wise winner (0 = A wins, 1 = B wins, 2 = tie).
    winner = np.full_like(a, 2, dtype=int)
    winner[a > b] = 0
    winner[b > a] = 1

    # TTB prediction per trial: first discriminating cue in validity order.
    val_order = np.argsort(-np.asarray(exp["validities"], dtype=float), kind="stable")
    ttb_pred = np.zeros(n_trials, dtype=int)
    for t in range(n_trials):
        pred = 2
        for j in val_order:
            if a[t, j] > b[t, j]:
                pred = 0
                break
            if b[t, j] > a[t, j]:
                pred = 1
                break
        ttb_pred[t] = pred

    # Tally prediction per trial: majority of feature wins.
    tally_pred = np.zeros(n_trials, dtype=int)
    for t in range(n_trials):
        a_wins = int(np.sum(a[t] > b[t]))
        b_wins = int(np.sum(b[t] > a[t]))
        if a_wins > b_wins:
            tally_pred[t] = 0
        elif b_wins > a_wins:
            tally_pred[t] = 1
        else:
            tally_pred[t] = 2

    color_map = {0: COLOR_TTB, 1: COLOR_TALLY, 2: COLOR_TIE}

    # Layout: we use plain axis coords.
    # Columns: validity bar row is one "pseudo-row" above the matrix.
    # Cell positions: col index 0..n_feat-1 for features, then a gap,
    # then 2 annotation cols (TTB pred, Tally pred).
    x_gap = 0.4
    cell_w = 1.0
    x_of = lambda c: c * cell_w

    # Validity bars above the matrix; numeric value labels ABOVE the bars
    # so the cells → gap → bars → labels stack has no visual crowding of t1.
    vals = np.asarray(exp["validities"], dtype=float)
    max_val = max(1.0, float(vals.max()) if vals.size else 1.0)
    bar_base = n_trials + 0.12
    bar_max_h = 0.55
    for j in range(n_feat):
        h = bar_max_h * (vals[j] / max_val)
        # Full cell-width bars -- align exactly with the feature columns below,
        # with white edge matching the matrix cell gap.
        ax.add_patch(mpatches.Rectangle(
            (x_of(j), bar_base),
            cell_w,
            h,
            facecolor=COLOR_TEXT,
            edgecolor="white",
            linewidth=2.0,
        ))
        ax.text(x_of(j) + cell_w / 2, bar_base + h + 0.08,
                f"{vals[j]:.2f}", ha="center", va="bottom", fontsize=8,
                color=COLOR_TEXT)

    # Feature-wise cells. Color encodes the per-feature winner. When
    # show_values=True, the raw ratings for option A and B are also
    # printed inside each tile (white text) as "a | b".
    for t in range(n_trials):
        # row y: top row is trial 1, so place trial t at y = n_trials - 1 - t
        y = n_trials - 1 - t
        for j in range(n_feat):
            ax.add_patch(mpatches.Rectangle(
                (x_of(j), y),
                cell_w, 1.0,
                facecolor=color_map[int(winner[t, j])],
                edgecolor="white", linewidth=0.8,
            ))
            if show_values:
                ax.text(x_of(j) + cell_w / 2, y + 0.5,
                        f"{a[t, j]} | {b[t, j]}", ha="center", va="center",
                        fontsize=9, color="white", weight="bold")

        # Annotation columns: TTB pred, Tally pred.
        base = x_of(n_feat) + x_gap
        for k, pred in enumerate([ttb_pred[t], tally_pred[t]]):
            ax.add_patch(mpatches.Rectangle(
                (base + k * cell_w, y),
                cell_w, 1.0,
                facecolor=color_map[int(pred)],
                edgecolor="white", linewidth=0.8,
            ))
            label = {0: "A", 1: "B", 2: "—"}[int(pred)]
            ax.text(base + k * cell_w + cell_w / 2, y + 0.5,
                    label, ha="center", va="center", fontsize=8, color="white", weight="bold")

    # Column labels (below matrix).
    for j in range(n_feat):
        ax.text(x_of(j) + cell_w / 2, -0.35, f"f{j+1}",
                ha="center", va="center", fontsize=9, color=COLOR_TEXT)
    base = x_of(n_feat) + x_gap
    ax.text(base + cell_w / 2, -0.35, "TTB", ha="center", va="center",
            fontsize=9, weight="bold", color=COLOR_TTB)
    ax.text(base + 3 * cell_w / 2, -0.35, "Tally", ha="center", va="center",
            fontsize=9, weight="bold", color=COLOR_TALLY)

    # Trial row labels (left).
    for t in range(n_trials):
        y = n_trials - 1 - t
        ax.text(-0.3, y + 0.5, f"t{t+1}", ha="right", va="center",
                fontsize=9, color=COLOR_TEXT)

    # Lane title above the matrix.
    ax.set_title(lane_title, fontsize=11, weight="bold", pad=12, color=COLOR_TEXT)

    # Axes cosmetics -- leave room above the matrix for bars + value labels.
    ax.set_xlim(-0.8, x_of(n_feat) + x_gap + 2 * cell_w + 0.2)
    ax.set_ylim(-0.9, n_trials + 1.20)
    ax.set_aspect("equal")
    ax.set_axis_off()


def render_stage2_lane(ax, metric: dict[str, Any], *, lane_title: str, tagline: str) -> None:
    """Metric code excerpt + plain-english tagline.

    No rationale footer -- the presenter narrates that part. The code box
    is sized to fill most of the axis so the code itself is readable at
    slide distance.
    """
    ax.set_axis_off()
    # Header (lane title + plain-english tagline).
    ax.text(0.02, 0.98, lane_title, transform=ax.transAxes,
            fontsize=12, weight="bold", va="top", color=COLOR_TEXT)
    ax.text(0.02, 0.89, tagline, transform=ax.transAxes,
            fontsize=10.5, va="top", color=COLOR_TEXT)

    # Full metric body: drop only `def ...:` and `import`/`from` lines.
    src = metric["source"]
    lines = src.replace("\\n", "\n").splitlines()
    core = [ln for ln in lines
            if ln.strip() and not ln.strip().startswith(("def ", "import ", "from "))]
    common = min((len(ln) - len(ln.lstrip()) for ln in core if ln.strip()), default=0)
    core = [ln[common:] for ln in core]
    excerpt = "\n".join(core)
    n_lines = len(core)

    # Fit-to-content: box height = n_lines * line_height + padding.
    code_font_size = 9.5
    line_height = 0.036  # axes-fraction per line at this font size
    pad_top = 0.028
    pad_bottom = 0.022
    box_top = 0.82
    box_height = n_lines * line_height + pad_top + pad_bottom
    box_bottom = box_top - box_height
    box = mpatches.FancyBboxPatch(
        (0.02, box_bottom), 0.96, box_height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=COLOR_CODE_BG, edgecolor="#DDDDDD", linewidth=0.6,
    )
    ax.add_patch(box)
    ax.text(0.04, box_top - pad_top, excerpt,
            transform=ax.transAxes, fontsize=code_font_size,
            family=["Courier New", "Courier", "monospace"],
            va="top", color=COLOR_TEXT)


def render_stage3_lane(ax, exp_results: dict[str, tuple[float, float]],
                       *, lane_title: str, metric_name: str,
                       n_subjects: int) -> None:
    """Horizontal bracket plot on [0, 1]: TTB, Tally, Observed markers + bracket band."""
    ttb = exp_results["pi_1"][0]
    tally = exp_results["pi_2"][0]
    obs, obs_var = exp_results["observed"]
    # SEM of the observed point estimate from the between-subject variance.
    obs_sem = np.sqrt(obs_var / n_subjects) if obs_var > 0 else 0.0

    lo, hi = (tally, ttb) if tally <= ttb else (ttb, tally)

    # Bracket band.
    ax.axvspan(lo, hi, facecolor=COLOR_BRACKET, alpha=0.45, zorder=1)

    # Axis line.
    ax.hlines(0, 0, 1, colors="#888888", linewidth=1.2, zorder=2)
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ax.vlines(tick, -0.03, 0.03, colors="#888888", linewidth=0.8)
        ax.text(tick, -0.11, f"{tick:.2f}", ha="center", va="top", fontsize=8, color="#555555")

    # Markers.
    ax.scatter([ttb], [0], marker="o", s=180, color=COLOR_TTB,
               zorder=5, edgecolor="white", linewidth=1.2, label="TTB")
    ax.scatter([tally], [0], marker="s", s=160, color=COLOR_TALLY,
               zorder=5, edgecolor="white", linewidth=1.2, label="Tallying")
    ax.scatter([obs], [0], marker="D", s=140, color=COLOR_OBS,
               zorder=6, edgecolor="white", linewidth=1.2, label="Observed")

    # SEM bar on observed (if nonzero).
    if obs_sem > 0:
        ax.hlines(0, obs - obs_sem, obs + obs_sem, colors=COLOR_OBS, linewidth=2.0, zorder=5)

    # Value labels above markers.
    ax.text(ttb, 0.16, f"{ttb:.3f}", ha="center", va="bottom",
            fontsize=9, color=COLOR_TTB, weight="bold")
    ax.text(tally, 0.16, f"{tally:.3f}", ha="center", va="bottom",
            fontsize=9, color=COLOR_TALLY, weight="bold")
    ax.text(obs, -0.30, f"observed = {obs:.3f}", ha="center", va="top",
            fontsize=9.5, color=COLOR_OBS, weight="bold")

    # Theory labels below axis.
    ax.text(ttb, 0.32, "TTB prediction", ha="center", va="bottom",
            fontsize=8.5, color=COLOR_TTB)
    ax.text(tally, 0.32, "Tally prediction", ha="center", va="bottom",
            fontsize=8.5, color=COLOR_TALLY)

    # Annotation: report the gap from each theory corner so the "in-between"
    # story is honest regardless of which theory happens to be numerically closer.
    gap_ttb = abs(obs - ttb)
    gap_tally = abs(obs - tally)
    arrow_dy = -0.55
    ax.annotate(
        f"gap from TTB = {gap_ttb:.3f}   |   gap from Tally = {gap_tally:.3f}",
        xy=(obs, -0.45), xytext=(0.5, arrow_dy - 0.05),
        ha="center", va="top", fontsize=8.5, style="italic", color="#333333",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
    )

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.85, 0.65)
    ax.set_title(f"{lane_title}  —  {metric_name}", fontsize=10.5, weight="bold", pad=12)
    ax.set_axis_off()


def _pull_quote(ax, x: float, y: float, text: str, *,
                facecolor: str = COLOR_HIGHLIGHT_BG,
                edgecolor: str = COLOR_HIGHLIGHT_EDGE,
                fontsize: float = 11.0) -> None:
    """Render a standalone emphasized phrase as a bbox-backed pull quote."""
    ax.text(x, y, text, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=fontsize, weight="bold", color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.55",
                      facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8))


def render_stage4_arbiter(ax, arbiter_resp: dict[str, Any]) -> None:
    """Arbiter rationale + pull quote with the decisive insight (verbatim).

    The pull quote is the arbiter's own decisive sentence (selected by
    `key_insight_sentence`), so it tells the *true* per-task story: cardinal-
    magnitude sensitivity for the rating task, validity-weighted compensation
    for the binary task — rather than a hardcoded phrase from one task.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel background: neutral off-white with tan edge.
    panel = mpatches.FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.90,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor=COLOR_PANEL_BG, edgecolor=COLOR_TTB, linewidth=1.0,
    )
    ax.add_patch(panel)

    ax.text(0.5, 0.93, "Stage 4 — Arbiter reasons over both observations",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, weight="bold", color=COLOR_TEXT)

    # Context (first sentence) on top; the decisive insight as the pull quote.
    interp = arbiter_resp.get("interpretation", "")
    ax.text(0.5, 0.80, _wrap(_first_sentence(interp, n=1), 95),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=COLOR_TEXT)

    _pull_quote(
        ax, 0.5, 0.34,
        "“" + _wrap(key_insight_sentence(interp), 72) + "”",
        fontsize=10.5,
    )

    # Verdict chip: tan (ties visually to the TTB slot being replaced).
    verdict = arbiter_resp.get("verdict", "?")
    target = arbiter_resp.get("target_theory_idx", "?")
    chip_text = f"verdict: {verdict}  →  pi_{target}"
    chip = mpatches.FancyBboxPatch(
        (0.37, 0.09), 0.26, 0.08,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        transform=ax.transAxes,
        facecolor=COLOR_TTB, edgecolor="none",
    )
    ax.add_patch(chip)
    ax.text(0.5, 0.13, chip_text, transform=ax.transAxes,
            ha="center", va="center", fontsize=10, weight="bold", color="white")


def render_stage5_wadd(ax, wadd: dict[str, Any]) -> None:
    """WADD description + equation + proposer rationale + pull quote."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel background: neutral off-white with blue edge (balances Stage 4's tan).
    panel = mpatches.FancyBboxPatch(
        (0.02, 0.03), 0.96, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor=COLOR_PANEL_BG, edgecolor=COLOR_TALLY, linewidth=1.0,
    )
    ax.add_patch(panel)

    ax.text(0.5, 0.94, "Stage 5 — Theory proposer translates the gap into WADD",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, weight="bold", color=COLOR_TEXT)

    # Description, wrapped.
    desc = _trim(_first_sentence(wadd["description"], n=1), 300)
    ax.text(0.5, 0.83, _wrap(desc, 100), transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color=COLOR_TEXT)

    # Equation block.
    ax.text(0.5, 0.52,
            r"$s_A = \sum_i v_i \cdot x_{A,i}$     "
            r"$s_B = \sum_i v_i \cdot x_{B,i}$     "
            r"$P(A) \propto \exp(\beta\, s_A)$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color=COLOR_TALLY)

    # Rationale, wrapped.
    rat = _trim(_first_sentence(wadd["rationale"], n=2), 340)
    ax.text(0.5, 0.38, _wrap(rat, 100), transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5, color=COLOR_TEXT)

    # Pull quote below (blue tint). Validity-weighted sum is the mechanism in
    # both tasks (cardinal ratings and binary 0/1 cue values alike).
    _pull_quote(
        ax, 0.5, 0.10,
        "“validity-weighted sum of cue values → compensatory decision-making”",
        facecolor=COLOR_HIGHLIGHT_BG_BLUE,
        edgecolor=COLOR_HIGHLIGHT_EDGE_BLUE,
        fontsize=11,
    )


# -- figure builders ---------------------------------------------------------


def stage1_taglines(exp1: dict[str, Any], exp2: dict[str, Any]) -> tuple[str, str]:
    """One tagline per experiment, read from the parsed designs.

    Reports the proposer (Exp 1 from the TTB seed, Exp 2 from the Tallying seed),
    the design dimensions, and `rating_max` only when the task defines it — the
    binary 0/1 task does not, so its taglines simply omit it.
    """
    def tagline(idx: int, proposer: str, exp: dict[str, Any]) -> str:
        n_trials, n_feat = exp["trial_a"].shape
        parts = [f"Exp {idx}", f"proposed by {proposer}",
                 f"{n_feat} feats × {n_trials} trials"]
        if exp["rating_max"] is not None:
            parts.append(f"rating_max={exp['rating_max']}")
        return "  ·  ".join(parts)

    return tagline(1, "TTB", exp1), tagline(2, "Tallying", exp2)


def _stage2_taglines() -> tuple[str, str]:
    # Both seed metrics are match-rates computed ONLY on the trials where TTB and
    # Tallying make strictly opposing predictions (the disagreement trials) — that
    # restriction is load-bearing and must be stated, not dropped.
    return ("Metric: fraction matching TTB on TTB-vs-Tallying disagreement trials.",
            "Metric: fraction matching Tallying on those disagreement trials.")


def _stage3_titles() -> tuple[str, str]:
    return ("Exp 1 — TTB-match rate", "Exp 2 — Tally-match rate")


STAGE1_CAPTION = (
    "Cell color = per-feature winner (A = tan, B = blue, tie = gray).  "
    "Right columns = each theory's predicted winner per trial."
)


def render_stage0_figure(exp: dict[str, Any], trial_idx: int = 0,
                         *, show_wadd: bool = False):
    """Precursor figure: one trial of the decision-making task.

    Shows validities along the top, then Option A (tan) and Option B (blue)
    as side-by-side rows of feature-rating tiles. This establishes the
    color convention (A = tan, B = blue) before Stage 1 merges them into a
    single 'per-feature winner' row.

    When show_wadd=True, adds a third row between the two options showing
    WADD's prediction, alongside the existing TTB and Tally prediction tiles.
    """
    a = exp["trial_a"][trial_idx]
    b = exp["trial_b"][trial_idx]
    vals = np.asarray(exp["validities"], dtype=float)
    n_feat = len(a)

    fig_h = 6.4 if show_wadd else 5.2
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_axis_off()

    cell_w = 1.0
    x_of = lambda j: j * cell_w

    # y coordinates (from bottom to top). When show_wadd, expand the gap
    # between A and B to slot a WADD tile between them.
    if show_wadd:
        y_opt_b = 0.0
        y_wadd = 1.2
        y_opt_a = 2.4
        bar_base = 3.8
    else:
        y_opt_b = 0.0
        y_wadd = None  # unused
        y_opt_a = 1.2
        bar_base = 2.6
    bar_max_h = 0.60
    max_val = max(1.0, float(vals.max()) if vals.size else 1.0)

    # Validity bars (top row, black, aligned with feature columns).
    for j in range(n_feat):
        h = bar_max_h * (vals[j] / max_val)
        ax.add_patch(mpatches.Rectangle(
            (x_of(j), bar_base),
            cell_w, h,
            facecolor=COLOR_TEXT, edgecolor="white", linewidth=2.0,
        ))
        ax.text(x_of(j) + cell_w / 2, bar_base + h + 0.08,
                f"{vals[j]:.2f}", ha="center", va="bottom",
                fontsize=10, color=COLOR_TEXT)

    # Option A row (tan tiles with rating value inside).
    for j in range(n_feat):
        ax.add_patch(mpatches.Rectangle(
            (x_of(j), y_opt_a),
            cell_w, 1.0,
            facecolor=COLOR_TTB, edgecolor="white", linewidth=2.0,
        ))
        ax.text(x_of(j) + cell_w / 2, y_opt_a + 0.5,
                str(int(a[j])), ha="center", va="center",
                fontsize=18, weight="bold", color="white")

    # Option B row (blue tiles with rating value inside).
    for j in range(n_feat):
        ax.add_patch(mpatches.Rectangle(
            (x_of(j), y_opt_b),
            cell_w, 1.0,
            facecolor=COLOR_TALLY, edgecolor="white", linewidth=2.0,
        ))
        ax.text(x_of(j) + cell_w / 2, y_opt_b + 0.5,
                str(int(b[j])), ha="center", va="center",
                fontsize=18, weight="bold", color="white")

    # Row labels (left).
    ax.text(-0.25, y_opt_a + 0.5, "Option A", ha="right", va="center",
            fontsize=12, weight="bold", color=COLOR_TTB)
    ax.text(-0.25, y_opt_b + 0.5, "Option B", ha="right", va="center",
            fontsize=12, weight="bold", color=COLOR_TALLY)

    # Heuristic-prediction tiles on the right side of each option row.
    # Tally tile sits on the Option A row; TTB tile sits on the Option B row.
    # Tile color = winning option's color; tile label = 'A' or 'B'.
    heuristic_gap = 0.6
    h_x = n_feat + heuristic_gap

    # TTB: first discriminating cue in validity order.
    val_order = np.argsort(-vals, kind="stable")
    ttb_winner = None
    for j in val_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
    # Tally: majority of feature-wise wins.
    a_wins = int(np.sum(a > b))
    b_wins = int(np.sum(b > a))
    tally_winner = 0 if a_wins > b_wins else (1 if b_wins > a_wins else None)

    def _heuristic_tile(y: float, winner, name: str) -> None:
        if winner == 0:
            face, letter = COLOR_TTB, "A"
        elif winner == 1:
            face, letter = COLOR_TALLY, "B"
        else:
            face, letter = COLOR_TIE, "—"
        ax.add_patch(mpatches.Rectangle(
            (h_x, y), cell_w, 1.0,
            facecolor=face, edgecolor="white", linewidth=2.0,
        ))
        ax.text(h_x + cell_w / 2, y + 0.5, letter,
                ha="center", va="center",
                fontsize=16, weight="bold", color="white")
        # Name label on the right side of the tile.
        # Color-match the name to the winning option so the tile and label agree.
        ax.text(h_x + cell_w + 0.25, y + 0.5, name,
                ha="left", va="center",
                fontsize=12, weight="bold", color=face)

    _heuristic_tile(y_opt_a, tally_winner, "Tally")
    _heuristic_tile(y_opt_b, ttb_winner, "TTB")

    # WADD: weighted sum of rating * validity.
    if show_wadd:
        score_a = float(np.dot(a, vals))
        score_b = float(np.dot(b, vals))
        wadd_winner = 0 if score_a > score_b else (1 if score_b > score_a else None)
        _heuristic_tile(y_wadd, wadd_winner, "WADD")

    # Feature labels at the bottom (only under the feature columns).
    for j in range(n_feat):
        ax.text(x_of(j) + cell_w / 2, y_opt_b - 0.35, f"f{j+1}",
                ha="center", va="center", fontsize=10, color=COLOR_TEXT)

    # "validity per feature" label centered above the bars.
    ax.text(n_feat / 2.0, bar_base + bar_max_h + 0.50,
            "validity per feature", ha="center", va="bottom",
            fontsize=10, color=COLOR_TEXT)

    # Axes limits -- leave room for the heuristic tiles + right-side labels.
    ax.set_xlim(-2.2, h_x + cell_w + 1.8)
    ax.set_ylim(y_opt_b - 0.9, bar_base + bar_max_h + 1.10)
    ax.set_aspect("equal")

    # Title + caption.
    fig.suptitle("A single trial of the decision-making task",
                 fontsize=14, weight="bold", y=0.97, color=COLOR_TEXT)
    fig.text(0.5, 0.04,
             "Subject sees both options and picks one. "
             "Expert validities are shown before the task begins.",
             ha="center", va="bottom", fontsize=10, color=COLOR_TEXT)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.04, right=0.98)
    return fig


def render_stage1_figure(exp1, exp2, *, show_values: bool = False):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"wspace": 0.18})
    fig.suptitle("Stage 1 — Experiment design",
                 fontsize=14, weight="bold", y=0.97, color=COLOR_TEXT)
    tl, tr = stage1_taglines(exp1, exp2)
    render_stage1_lane(ax_l, exp1, lane_title=tl, show_values=show_values)
    render_stage1_lane(ax_r, exp2, lane_title=tr, show_values=show_values)
    fig.text(0.5, 0.04, STAGE1_CAPTION, ha="center", va="bottom",
             fontsize=9, color=COLOR_TEXT)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.06, right=0.96)

    # "validity per feature" label at a shared figure y, computed from the
    # top of the validity-value text in each panel (figure coords). Using
    # the MAX of the two tops guarantees the label clears the numbers in
    # both panels while staying below the lane titles.
    fig.canvas.draw()
    tops: list[float] = []
    for ax, n_trials in ((ax_l, exp1["trial_a"].shape[0]),
                         (ax_r, exp2["trial_a"].shape[0])):
        # Data coord y of the value text: bar_base + bar_max_h*val + 0.08.
        # We use the max val (~1.0) to be safe.
        top_data_y = n_trials + 0.12 + 0.55 + 0.30
        _, top_fig_y = fig.transFigure.inverted().transform(
            ax.transData.transform((0, top_data_y))
        )
        tops.append(top_fig_y)
    label_y = max(tops)
    # Center the label over the feature columns only (not over the gap +
    # TTB/Tally annotation columns on the right of each panel).
    n_feat_l = exp1["trial_a"].shape[1]
    feat_mid_data_x = n_feat_l / 2.0  # cell_w = 1.0 in data coords
    x_mid, _ = fig.transFigure.inverted().transform(
        ax_l.transData.transform((feat_mid_data_x, 0))
    )
    fig.text(x_mid, label_y, "validity per feature",
             ha="center", va="bottom", fontsize=9, color=COLOR_TEXT)
    return fig


def render_stage2_figure(metric1, metric2):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.08})
    fig.suptitle("Stage 2 — Metric proposal",
                 fontsize=14, weight="bold", y=0.97, color=COLOR_TEXT)
    tl, tr = _stage2_taglines()
    render_stage2_lane(ax_l, metric1, lane_title="Exp 1 metric", tagline=tl)
    render_stage2_lane(ax_r, metric2, lane_title="Exp 2 metric", tagline=tr)
    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.04, right=0.98)
    return fig


def render_stage3_figure(exp_results: dict[int, dict[str, tuple[float, float]]],
                         *, n_subjects: int):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"wspace": 0.12})
    fig.suptitle("Stage 3 — Predictions bracket the observation on both experiments",
                 fontsize=14, weight="bold", y=0.995)
    titles = _stage3_titles()
    render_stage3_lane(ax_l, exp_results[1], lane_title=titles[0],
                       metric_name="TTB-match", n_subjects=n_subjects)
    render_stage3_lane(ax_r, exp_results[2], lane_title=titles[1],
                       metric_name="Tally-match", n_subjects=n_subjects)
    # Shared legend.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_TTB,
                   markersize=11, label="TTB prediction"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_TALLY,
                   markersize=10, label="Tally prediction"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_OBS,
                   markersize=9, label="Observed (human)"),
        mpatches.Patch(facecolor=COLOR_BRACKET, alpha=0.5, label="bracket band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    return fig


def render_stage4_figure(arbiter_resp):
    fig, ax = plt.subplots(figsize=(13, 4))
    render_stage4_arbiter(ax, arbiter_resp)
    return fig


def render_stage5_figure(wadd):
    fig, ax = plt.subplots(figsize=(13, 5))
    render_stage5_wadd(ax, wadd)
    return fig


def render_composite(exp1, exp2, metric1, metric2, exp_results, arbiter_resp, wadd,
                     *, subtitle: str = "", n_subjects: int):
    fig = plt.figure(figsize=(10, 20))
    gs = fig.add_gridspec(
        nrows=5, ncols=1,
        height_ratios=[1.0, 1.0, 1.6, 1.0, 1.2],
        hspace=0.45,
    )
    fig.suptitle("Case study: how the LLM narrows in on WADD\n" + subtitle,
                 fontsize=15, weight="bold", y=0.995)

    # Stage 1 — two lanes.
    gs1 = gs[0].subgridspec(1, 2, wspace=0.08)
    ax1a = fig.add_subplot(gs1[0, 0])
    ax1b = fig.add_subplot(gs1[0, 1])
    tl1, tr1 = stage1_taglines(exp1, exp2)
    render_stage1_lane(ax1a, exp1, lane_title=tl1)
    render_stage1_lane(ax1b, exp2, lane_title=tr1)
    fig.text(0.5, ax1a.get_position().y1 + 0.013, "Stage 1 — Experiment design",
             ha="center", va="bottom", fontsize=13, weight="bold", color="#222222")
    fig.text(0.5, ax1a.get_position().y0 - 0.015, STAGE1_CAPTION,
             ha="center", va="top", fontsize=7.5, color="#666666")

    # Stage 2 — two lanes.
    gs2 = gs[1].subgridspec(1, 2, wspace=0.08)
    ax2a = fig.add_subplot(gs2[0, 0])
    ax2b = fig.add_subplot(gs2[0, 1])
    tl2, tr2 = _stage2_taglines()
    render_stage2_lane(ax2a, metric1, lane_title="Exp 1 metric", tagline=tl2)
    render_stage2_lane(ax2b, metric2, lane_title="Exp 2 metric", tagline=tr2)
    fig.text(0.5, ax2a.get_position().y1 + 0.013, "Stage 2 — Metric proposal",
             ha="center", va="bottom", fontsize=13, weight="bold", color="#222222")

    # Stage 3 — two lanes (anchor).
    gs3 = gs[2].subgridspec(1, 2, wspace=0.12)
    ax3a = fig.add_subplot(gs3[0, 0])
    ax3b = fig.add_subplot(gs3[0, 1])
    titles = _stage3_titles()
    render_stage3_lane(ax3a, exp_results[1], lane_title=titles[0],
                       metric_name="TTB-match", n_subjects=n_subjects)
    render_stage3_lane(ax3b, exp_results[2], lane_title=titles[1],
                       metric_name="Tally-match", n_subjects=n_subjects)
    fig.text(0.5, ax3a.get_position().y1 + 0.013,
             "Stage 3 — Predictions bracket the observation (the gap)",
             ha="center", va="bottom", fontsize=13, weight="bold", color="#222222")

    # Stage 4 — single merged panel.
    ax4 = fig.add_subplot(gs[3])
    render_stage4_arbiter(ax4, arbiter_resp)

    # Stage 5 — single panel.
    ax5 = fig.add_subplot(gs[4])
    render_stage5_wadd(ax5, wadd)

    # Flow arrows: draw vertical arrows between stages via figure coords.
    fig.canvas.draw()  # ensure positions are finalized
    def _arrow_between(ax_top, ax_bottom, label: str) -> None:
        pos_top = ax_top.get_position()
        pos_bot = ax_bottom.get_position()
        x = 0.5
        y_top = pos_bot.y1 + (pos_top.y0 - pos_bot.y1) * 0.85
        y_bot = pos_bot.y1 + (pos_top.y0 - pos_bot.y1) * 0.15
        arr = FancyArrowPatch(
            (x, y_top), (x, y_bot),
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=15,
            color="#888888", linewidth=1.2,
        )
        fig.add_artist(arr)
        fig.text(x + 0.015, (y_top + y_bot) / 2, label,
                 ha="left", va="center", fontsize=9, color="#666666", style="italic")

    _arrow_between(ax1a, ax2a, "→ metric")
    _arrow_between(ax2a, ax3a, "→ predictions + data")
    _arrow_between(ax3a, ax4, "→ rationale")
    _arrow_between(ax4, ax5, "→ theory")

    return fig


# -- cartoon pipeline --------------------------------------------------------
# Compact single-row schematic of the full pipeline for a roadmap/overview
# slide. Six nodes (Seeds → Design → Metric → Gap → Arbiter → Theory) with
# tiny icons + arrows.


def _cartoon_seeds(ax, *, cx, cy):
    r = 1.6
    ax.add_patch(mpatches.Circle((cx - 1.9, cy), r,
                                 facecolor=COLOR_TTB, edgecolor="white", lw=1.2))
    ax.add_patch(mpatches.Circle((cx + 1.9, cy), r,
                                 facecolor=COLOR_TALLY, edgecolor="white", lw=1.2))
    ax.text(cx - 1.9, cy, "TTB", ha="center", va="center",
            fontsize=7.5, color="white", weight="bold")
    ax.text(cx + 1.9, cy, "Tally", ha="center", va="center",
            fontsize=7, color="white", weight="bold")


def _cartoon_design(ax, *, cx, cy):
    n_rows, n_cols = 3, 4
    cell = 0.95
    left = cx - (n_cols * cell) / 2
    bot = cy - (n_rows * cell) / 2
    pattern = [
        [COLOR_TTB, COLOR_TALLY, COLOR_TTB, COLOR_TALLY],
        [COLOR_TALLY, COLOR_TTB, COLOR_TALLY, COLOR_TTB],
        [COLOR_TTB, COLOR_TTB, COLOR_TALLY, COLOR_TALLY],
    ]
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(mpatches.Rectangle(
                (left + c * cell, bot + (n_rows - 1 - r) * cell),
                cell, cell,
                facecolor=pattern[r][c], edgecolor="white", lw=1.0,
            ))


def _cartoon_metric(ax, *, cx, cy):
    w, h = 6.4, 3.2
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        facecolor=COLOR_CODE_BG, edgecolor="#CCCCCC", lw=0.8,
    ))
    ax.text(cx, cy, r"f(data) $\to$ %",
            ha="center", va="center", fontsize=10,
            family=["Courier New", "Courier", "monospace"], color=COLOR_TEXT)


def _cartoon_gap(ax, *, cx, cy, exp_results):
    e2 = exp_results[2]
    ttb = e2["pi_1"][0]
    tally = e2["pi_2"][0]
    obs = e2["observed"][0]
    lo_x, hi_x = cx - 3.0, cx + 3.0
    def xp(v):
        return lo_x + v * (hi_x - lo_x)
    lo_v, hi_v = sorted([ttb, tally])
    ax.add_patch(mpatches.Rectangle(
        (xp(lo_v), cy - 0.45), xp(hi_v) - xp(lo_v), 0.9,
        facecolor=COLOR_BRACKET, edgecolor="none",
    ))
    ax.hlines(cy, lo_x, hi_x, colors="#888", linewidth=1.0)
    ax.scatter([xp(ttb)], [cy], marker="o", s=45,
               color=COLOR_TTB, edgecolor="white", lw=0.8, zorder=5)
    ax.scatter([xp(tally)], [cy], marker="s", s=42,
               color=COLOR_TALLY, edgecolor="white", lw=0.8, zorder=5)
    ax.scatter([xp(obs)], [cy], marker="D", s=38,
               color=COLOR_OBS, edgecolor="white", lw=0.8, zorder=6)


def _cartoon_arbiter(ax, *, cx, cy):
    w, h = 6.4, 3.4
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.25,rounding_size=0.6",
        facecolor=COLOR_HIGHLIGHT_BG, edgecolor=COLOR_HIGHLIGHT_EDGE, lw=1.0,
    ))
    # Generic across tasks: cardinal WADD weights by magnitude×validity, binary
    # WADD by validity alone — both are "weighted compensation".
    ax.text(cx, cy, "weighted\ncompensation",
            ha="center", va="center", fontsize=8.5,
            color=COLOR_TEXT, style="italic")


def _cartoon_wadd(ax, *, cx, cy):
    w, h = 6.4, 3.4
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.25,rounding_size=0.6",
        facecolor=COLOR_WADD, edgecolor="none",
    ))
    ax.text(cx, cy + 0.55, "WADD",
            ha="center", va="center", fontsize=10.5, weight="bold", color="white")
    ax.text(cx, cy - 0.75, r"$\sum_i\, v_i\, x_i$",
            ha="center", va="center", fontsize=10.5, color="white")


def render_cartoon_figure(exp_results):
    fig, ax = plt.subplots(figsize=(15, 3.25))
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.set_axis_off()

    node_x = [5, 15, 25, 35, 45, 55]
    node_labels = ["Seeds", "Design", "Metric", "Gap", "Arbiter", "Theory"]
    icon_cy = 7.8
    label_y = 2.4

    _cartoon_seeds(ax, cx=node_x[0], cy=icon_cy)
    _cartoon_design(ax, cx=node_x[1], cy=icon_cy)
    _cartoon_metric(ax, cx=node_x[2], cy=icon_cy)
    _cartoon_gap(ax, cx=node_x[3], cy=icon_cy, exp_results=exp_results)
    _cartoon_arbiter(ax, cx=node_x[4], cy=icon_cy)
    _cartoon_wadd(ax, cx=node_x[5], cy=icon_cy)

    for x, label in zip(node_x, node_labels):
        ax.text(x, label_y, label, ha="center", va="top",
                fontsize=11, weight="bold", color=COLOR_TEXT)

    for i in range(len(node_x) - 1):
        ax.annotate("",
                    xy=(node_x[i + 1] - 4.0, icon_cy),
                    xytext=(node_x[i] + 4.0, icon_cy),
                    arrowprops=dict(arrowstyle="->", color="#888888",
                                    lw=1.2, shrinkA=0, shrinkB=0))

    fig.suptitle("Pipeline: seed heuristics $\\to$ discovered theory",
                 fontsize=13, weight="bold", y=0.96, color=COLOR_TEXT)
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.02, right=0.98)
    return fig


# -- main --------------------------------------------------------------------

DEFAULT_RUN_DIR = Path(
    "results/recovery/"
    "wadd_sampling/noise=0.0/"
    "dmb_ground_truth_wadd_sampling_noise=0.0_gemini-3.1-pro-preview_run3"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                   help="Path to the run directory containing rounds/round_000/")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <run-dir>/analysis/case_study/)")
    args = p.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"error: run-dir not found: {run_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (run_dir / "analysis" / "case_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse all artifacts.
    exp1 = parse_experiment(run_dir, "pi_1")
    exp2 = parse_experiment(run_dir, "pi_2")
    metric1 = parse_metric(run_dir, "pi_1")
    metric2 = parse_metric(run_dir, "pi_2")
    arb = parse_arbitration(run_dir)
    wadd = parse_wadd_from_theories(run_dir)
    n_subjects = count_subjects(run_dir)

    # Per-stage figures.
    figs = {
        "stage0_task_trial": render_stage0_figure(exp1, trial_idx=0),
        "stage0_task_trial_with_wadd": render_stage0_figure(exp1, trial_idx=0, show_wadd=True),
        "stage1_design": render_stage1_figure(exp1, exp2),
        "stage1_design_values": render_stage1_figure(exp1, exp2, show_values=True),
        "stage2_metric": render_stage2_figure(metric1, metric2),
        "stage3_gap": render_stage3_figure(arb["experiments"], n_subjects=n_subjects),
        "stage4_arbiter": render_stage4_figure(arb["response"]),
        "stage5_wadd": render_stage5_figure(wadd),
        "cartoon_pipeline": render_cartoon_figure(arb["experiments"]),
    }
    for name, fig in figs.items():
        out = out_dir / f"{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    # Composite (portrait).
    fig = render_composite(exp1, exp2, metric1, metric2, arb["experiments"],
                           arb["response"], wadd,
                           subtitle=run_subtitle(run_dir), n_subjects=n_subjects)
    out_png = out_dir / "case_study_full.png"
    out_pdf = out_dir / "case_study_full.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
