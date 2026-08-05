"""Plot the exact leaderboard scores (from leaderboard.md) across rounds.

Parses `<run_dir>/leaderboard.md`, extracts each `round N - post-data` block,
and plots one line per theory of score-vs-round. These are the identical
numbers the `theory_scores()` function emits (`1 - L2_norm / max_L2_norm`),
so no re-simulation is needed.

Only `post-data` blocks are used (pre-admission), giving a clean
one-point-per-round trajectory.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import FONTSIZE, NEUTRAL_HARMONY, save_figure, style_axes


POST_DATA_HEADER = re.compile(r"^\[round\s+(\d+)\s*\|\s*post-data\]\s*$")
ROW = re.compile(r"^\s*#\s*\d+\s+(\S+)\s+([0-9]+\.[0-9]+)\s+\|")


def parse_leaderboard(path: Path) -> dict[int, dict[str, float]]:
    """Return `{round_idx: {label: score}}` from post-data blocks only."""
    rounds: dict[int, dict[str, float]] = {}
    current_round: int | None = None
    in_block = False

    with path.open() as f:
        for line in f:
            stripped = line.rstrip("\n")

            m = POST_DATA_HEADER.match(stripped.strip())
            if m:
                current_round = int(m.group(1))
                rounds.setdefault(current_round, {})
                in_block = True
                continue

            if stripped.startswith("```"):
                in_block = False
                continue

            if current_round is None or not in_block:
                continue

            r = ROW.match(stripped)
            if r:
                label = r.group(1)
                score = float(r.group(2))
                # If the same round appears more than once (e.g. the round_008
                # double-posting in this run), keep the first occurrence.
                rounds[current_round].setdefault(label, score)

    return rounds


def write_csv(rounds: dict[int, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "theory_label", "score"])
        for r in sorted(rounds):
            for label in sorted(rounds[r]):
                w.writerow([r, label, rounds[r][label]])


def plot(rounds: dict[int, dict[str, float]], path: Path, *, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_labels = sorted({L for d in rounds.values() for L in d})
    sorted_rounds = sorted(rounds)

    series: dict[str, list[tuple[int, float]]] = {L: [] for L in all_labels}
    for r in sorted_rounds:
        for L in all_labels:
            if L in rounds[r]:
                series[L].append((r, rounds[r][L]))

    final_round = sorted_rounds[-1]
    last_score = {
        L: next((s for (rd, s) in reversed(series[L]) if rd == final_round), -1.0)
        for L in all_labels
    }
    ordered = sorted(all_labels, key=lambda L: last_score[L], reverse=True)

    # Color each model by the round it first appeared in: slate for early
    # arrivals, blending to terracotta for those that emerged in later rounds.
    from matplotlib.colors import LinearSegmentedColormap
    first_round = {L: series[L][0][0] for L in all_labels}
    fmin = min(first_round.values())
    span = max(max(first_round.values()) - fmin, 1)
    cmap = LinearSegmentedColormap.from_list(
        "arrival", [NEUTRAL_HARMONY["slate"], NEUTRAL_HARMONY["terracotta"]])
    def color_for(L: str):
        t = (first_round[L] - fmin) / span  # 0.0 = earliest, 1.0 = latest
        return cmap(t)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for L in ordered:
        xs = [r for (r, _) in series[L]]
        ys = [s for (_, s) in series[L]]
        ax.plot(xs, ys, marker="o", markersize=8, lw=3, color=color_for(L),
                label=f"{L}  ({last_score[L]:.3f})")

    ax.set_ylim(-0.02, 1.02)
    style_axes(ax, xlabel="round",
               ylabel="leaderboard score\n(1 - ||d|| / max ||d||)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
              fontsize=FONTSIZE - 6, title="theory  (final score)",
              title_fontsize=FONTSIZE - 6)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Path to results/<run>/ directory containing leaderboard.md")
    p.add_argument("--out-name", default="leaderboard_trajectory",
                   help="Basename for <run>/analysis/<out-name>.{png,csv}")
    args = p.parse_args(argv)

    lb_path = args.run_dir / "leaderboard.md"
    if not lb_path.is_file():
        print(f"error: {lb_path} not found", file=sys.stderr)
        return 1

    rounds = parse_leaderboard(lb_path)
    if not rounds:
        print(f"error: no post-data blocks parsed from {lb_path}", file=sys.stderr)
        return 1

    out_dir = args.run_dir / "analysis"
    csv_path = out_dir / f"{args.out_name}.csv"
    png_path = out_dir / f"{args.out_name}.png"

    write_csv(rounds, csv_path)
    plot(rounds, png_path, title=f"Leaderboard score per round - {args.run_dir.name}")

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())