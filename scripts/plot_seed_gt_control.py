"""Plot the seed-with-ground-truth control: does a seeded GT persist?

For each ground truth we ran one soft-control run (slot 1 seeded with the GT
itself, slot 2 = tallying_sampling) and recorded a per-round leaderboard. This
script reads each run's `leaderboard.md`, extracts the end-of-round (post-admit)
score of the seeded GT (`pi_1`) and of the best competing theory, and plots both
trajectories — one panel per ground truth.

The story: the GT-seed line stays at the top every round (it is never killed),
while the best-adversary line either climbs to it (easy-to-rediscover GTs) or
stalls below (hard, non-canonical GTs).

Run:  python scripts/plot_seed_gt_control.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import (  # noqa: E402
    FONTSIZE,
    ROLE_COLOR,
    save_figure,
    style_axes,
)

# A leaderboard score row: "  # 1  pi_1  0.980  |####|  Δ-0.020".
_SCORE_RE = re.compile(r"#\s*\d+\s+(\S+)\s+([0-9]+\.[0-9]+)\s+\|")
# The block header that starts an end-of-round (post-admit) table.
_POST_ADMIT_RE = re.compile(r"\[round \d+ \| post-admit")

SEED_LABEL = "pi_1"  # slot 1 always carries the seeded ground truth
SEED_COLOR = ROLE_COLOR["seed"]        # slate
ADVERSARY_COLOR = ROLE_COLOR["surfaced"]  # terracotta

# Ground truths plotted, in order, with display titles.
GROUND_TRUTHS: list[tuple[str, str]] = [
    ("ttb_sampling", "TTB"),
    ("anti_majority", "Anti-majority"),
    ("cue_parity", "Cue-parity"),
]
RESULTS_ROOT = Path(
    "results/heuristic_decision_making/seed_gt_control/condition_baseline"
)


def parse_post_admit_scores(text: str) -> list[dict[str, float]]:
    """Per-round end-of-round scores from a `leaderboard.md`.

    Returns one `{label: score}` dict per *post-admit* block, in round order.
    The intermediate *post-data* blocks are ignored so each entry is the
    leaderboard state at the end of a round (after the new theory is admitted).
    """
    rounds: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in text.splitlines():
        if _POST_ADMIT_RE.search(line):
            current = {}
            rounds.append(current)
            continue
        if current is None:
            continue
        match = _SCORE_RE.search(line)
        if match:
            current[match.group(1)] = float(match.group(2))
        elif line.strip().startswith("```"):
            current = None  # table closed; ignore until the next post-admit
    return rounds


def seed_and_best_adversary(
    rounds: list[dict[str, float]], *, seed_label: str = SEED_LABEL
) -> tuple[list[float], list[float]]:
    """Split per-round scores into (seed trajectory, best-adversary trajectory).

    The best adversary each round is the highest-scoring theory that is NOT the
    seed — i.e. the strongest model the discovery loop produced so far.
    """
    seed: list[float] = []
    best_adversary: list[float] = []
    for scores in rounds:
        seed.append(scores.get(seed_label, float("nan")))
        others = [s for label, s in scores.items() if label != seed_label]
        best_adversary.append(max(others) if others else float("nan"))
    return seed, best_adversary


def _find_leaderboard(ground_truth: str) -> Path:
    matches = sorted((RESULTS_ROOT / ground_truth).glob("**/leaderboard.md"))
    if not matches:
        raise FileNotFoundError(
            f"no leaderboard.md under {RESULTS_ROOT / ground_truth}"
        )
    return matches[0]


def main() -> None:
    n = len(GROUND_TRUTHS)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
    axes = list(axes) if n > 1 else [axes]
    for ax, (ground_truth, title) in zip(axes, GROUND_TRUTHS):
        rounds = parse_post_admit_scores(_find_leaderboard(ground_truth).read_text())
        seed, best_adversary = seed_and_best_adversary(rounds)
        x = list(range(len(seed)))
        ax.plot(
            x, seed, color=SEED_COLOR, lw=3, marker="o", markersize=8,
            label="GT seed (slot 1)",
        )
        ax.plot(
            x, best_adversary, color=ADVERSARY_COLOR, lw=3, marker="x",
            markersize=8, label="Best adversary",
        )
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(x)
        style_axes(ax, xlabel="Round", title=title)
    axes[0].set_ylabel("Leaderboard score", fontsize=FONTSIZE)
    axes[0].legend(frameon=False, fontsize=FONTSIZE - 2, loc="lower right")
    fig.tight_layout()
    written = save_figure(
        fig,
        "results/heuristic_decision_making/seed_gt_control/seed_persistence_trajectory",
    )
    print("wrote:", *(str(p) for p in written), sep="\n  ")


if __name__ == "__main__":
    main()
