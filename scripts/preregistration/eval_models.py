"""Fit of {pi_7, WADD, Tallying, TTB} to the preregistered-study
humans, pooled across BOTH experiments (exp1 + exp2) and all five validity
vectors, in stimulus choice-proportion space.

This mirrors ``scripts/eval_human.py`` / ``scripts/eval_pi4_vs_pi7_on_humans.py``:
every model is replayed (mean P(choose B) over ``--n_subjects`` parameter draws)
on each stimulus the humans saw, then scored against the pooled human "choose B"
proportion via MAE, MSE, and Pearson r. The models:

  pi_7              — "Diminishing Returns WADD" (concave-utility WADD) surfaced
                      by the heuristic_decision_making human run (round 4).
  wadd / tallying / ttb — the canonical paper heuristics (fixed baselines).

A "stimulus" here is a (validity-vector, option_a, option_b) triple. The five
vectors are different tasks (different validities / cue orderings), so each model
is replayed per-vector with that vector's validities pinned, and the per-stimulus
fits are then POOLED across vectors AND experiments for the reported metrics
("all data taken together"). exp1 (model-discrimination) and exp2 (steep / offset)
are separate between-subjects groups, so a given stimulus contributes its choose-B
proportion from whichever experiment's participants saw it.

Outputs (under ``--out``, default ``validation_online/data/analysis/``):
  eval_models_per_stimulus.csv  — long format (model, experiment, vector, stimulus, p_b_human, p_b_model)
  eval_models_summary.csv       — per-model MAE / MSE / Pearson r vs pooled humans
  eval_models_metrics.png       — 1x3 bars: MSE | MAE | Pearson r, one bar per model
  eval_models_scatter.png       — per-model human-vs-model P(B) scatter, coloured by experiment

Usage:
  .venv/bin/python validation_online/eval_models.py
  .venv/bin/python validation_online/eval_models.py --n_subjects 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_human as eh  # noqa: E402  (model replay + canonical-theory loading)
from src.theory import Theory  # noqa: E402

# Surfaced theories live in their run-dirs; pin the same sources the other
# eval scripts use so the numbers line up.
PI7_SOURCE = (REPO_ROOT / "results" / "human_decision_making_cardinal"
              / "ttb+tallying"
              / "rounds" / "round_004" / "theories.json")

# Display order (left-to-right in every figure) and pretty labels.
MODEL_ORDER = ["pi_7", "wadd", "tallying", "ttb"]
PRETTY = {"pi_7": "pi_7\n(surfaced)",
          "wadd": "WADD", "tallying": "Tallying", "ttb": "TTB"}


def _load_replacement(source: Path, expect: str) -> Theory:
    repl = json.loads(source.read_text())["replacement"]
    assert repl["label"] == expect, f"expected {expect}, got {repl['label']}"
    return Theory.model_validate(repl["theory"])


def load_models() -> dict[str, Theory]:
    """The four models to score, keyed by the label used everywhere downstream."""
    models: dict[str, Theory] = {
        "pi_7": _load_replacement(PI7_SOURCE, "pi_7"),
    }
    canon = eh.load_canonical_theories()  # ttb / wadd / tallying / ew / ...
    for m in ("wadd", "tallying", "ttb"):
        models[m] = canon[m]
    return models


def load_pooled_trials(exp_dirs: list[Path]) -> pd.DataFrame:
    """Concatenate each experiment's trials.csv into one trial-level table with
    parsed option tuples (one row per completed choice trial)."""
    frames = []
    for d in exp_dirs:
        p = d / "trials.csv"
        if not p.exists():
            print(f"[eval_models] note: {p} missing, skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        if "experiment" not in df.columns:
            df["experiment"] = d.name  # fall back to the dir name (exp1/exp2)
        frames.append(df)
    if not frames:
        raise SystemExit("[eval_models] no trials.csv found under any experiment dir.")
    df = pd.concat(frames, ignore_index=True)
    df["a"] = df["option_a"].apply(lambda s: tuple(json.loads(s)))
    df["b"] = df["option_b"].apply(lambda s: tuple(json.loads(s)))
    df["response"] = df["response"].astype(int)
    df["vector"] = df["vector"].astype(str)
    return df


def human_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled choose-B proportion per (experiment, vector, option_a, option_b).
    Pooled = every trial across all participants and repeats for that stimulus."""
    rows = []
    keys = ["experiment", "vector", "a", "b"]
    for (exp, vec, a, b), g in df.groupby(keys, sort=True):
        rows.append({
            "experiment": exp, "vector": vec,
            "a": a, "b": b,
            "component": g["component"].iloc[0] if "component" in g.columns else None,
            "n_trials": int(len(g)),
            "n_participants": int(g["participant"].nunique()),
            "p_b_human": float(g["response"].mean()),
        })
    return pd.DataFrame(rows)


def score_models(human: pd.DataFrame, models: dict[str, Theory], design: dict,
                 *, n_subjects: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay each model per validity vector (its own validities pinned) on that
    vector's stimuli, then pool every stimulus across vectors + experiments to
    score MAE / MSE / Pearson r. Returns (per_stimulus_df, summary_df)."""
    rating_max = int(design["rating_max"])
    vec_validities = {v: list(design["vectors"][v]["validities"]) for v in design["vectors"]}

    per_stim_rows: list[dict] = []
    for label, theory in models.items():
        for vec, gv in human.groupby("vector"):
            pairs = list(zip(gv["a"], gv["b"]))
            try:
                preds = eh.model_proportions_for_theory(
                    theory, stimulus_pairs=pairs,
                    validities=vec_validities[vec], rating_max=rating_max,
                    n_subjects=n_subjects, base_seed=seed)
            except Exception as e:  # incompatible design -> skip this model
                print(f"  [skip] {label} on vector {vec}: {e}", file=sys.stderr)
                continue
            for (_, row), (a, b) in zip(gv.iterrows(), pairs):
                per_stim_rows.append({
                    "model": label, "experiment": row["experiment"],
                    "vector": vec, "component": row["component"],
                    "option_a": str(list(a)), "option_b": str(list(b)),
                    "n_trials_human": int(row["n_trials"]),
                    "p_b_human": float(row["p_b_human"]),
                    "p_b_model": float(preds[(a, b)])})
    per_stim = pd.DataFrame(per_stim_rows)

    summary_rows = []
    for label, g in per_stim.groupby("model"):
        h = g["p_b_human"].to_numpy(float)
        m = g["p_b_model"].to_numpy(float)
        diff = m - h
        r = (float(np.corrcoef(m, h)[0, 1])
             if m.std() > 0 and h.std() > 0 else float("nan"))
        summary_rows.append({
            "model": label, "n_stimuli": int(len(g)),
            "mae": float(np.mean(np.abs(diff))),
            "mse": float(np.mean(diff * diff)),
            "pearson_r": r})
    summary = pd.DataFrame(summary_rows)
    # Stable, requested display order (any unexpected models go last).
    summary["__o"] = summary["model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else len(MODEL_ORDER))
    summary = summary.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    return per_stim, summary


def _color_map(models: list[str]):
    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    return {m: cmap(i % 10) for i, m in enumerate(models)}


def plot_metrics(summary: pd.DataFrame, path: Path, *, title: str) -> None:
    """1x3 bar panels — MSE | MAE | Pearson r — one bar per model. MSE/MAE: lower
    is better (start at 0); Pearson r: higher is better."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = summary["model"].tolist()
    cmap = _color_map(models)
    colors = [cmap[m] for m in models]
    labels = [PRETTY.get(m, m) for m in models]
    x = np.arange(len(models))

    panels = [("mse", "MSE  (lower better)"),
              ("mae", "MAE  (lower better)"),
              ("pearson_r", "Pearson r  (higher better)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), squeeze=False)
    for ax, (col, ttl) in zip(axes[0], panels):
        vals = summary[col].to_numpy(float)
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=11)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == "pearson_r":
            ax.axhline(0, color="0.6", linewidth=0.8)
            ax.set_ylim(min(0.0, float(np.nanmin(vals)) - 0.05), 1.0)
        else:
            ax.set_ylim(0, float(np.nanmax(vals)) * 1.18)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7,
                        rotation=0 if col != "pearson_r" else 0)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(per_stim: pd.DataFrame, summary: pd.DataFrame, path: Path,
                 *, title: str) -> None:
    """One subplot per model: human P(B) vs model P(B), coloured by experiment,
    with the y=x line and the model's pooled MAE / r in the subplot title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = summary["model"].tolist()
    exps = sorted(per_stim["experiment"].astype(str).unique())
    ecmap = {e: matplotlib.colormaps["Set1"](i) for i, e in enumerate(exps)}
    n = len(models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    srow = summary.set_index("model")
    for i, mdl in enumerate(models):
        ax = axes[i // cols][i % cols]
        sub = per_stim[per_stim["model"] == mdl]
        for e in exps:
            s = sub[sub["experiment"].astype(str) == e]
            ax.scatter(s["p_b_human"], s["p_b_model"], s=34, alpha=0.8,
                       color=ecmap[e], edgecolor="black", linewidth=0.3, label=e)
        ax.plot([0, 1], [0, 1], ":", color="0.6", linewidth=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Human P(choose B)")
        ax.set_ylabel("Model P(choose B)")
        row = srow.loc[mdl]
        ax.set_title(f"{PRETTY.get(mdl, mdl).replace(chr(10), ' ')}  "
                     f"(MAE={row['mae']:.3f}, r={row['pearson_r']:.2f})", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9, title="experiment")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=str(HERE / "data"),
                    help="dir holding exp1/ and exp2/ (default: validation_online/data)")
    ap.add_argument("--experiments", nargs="+", default=["exp1", "exp2"],
                    help="experiment subdirs to pool (default: exp1 exp2)")
    ap.add_argument("--n_subjects", type=int, default=200,
                    help="parameter draws per model when replaying P(B)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output dir (default: <data_root>/analysis)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out) if args.out else data_root / "analysis"
    exp_dirs = [data_root / e for e in args.experiments]

    with open(HERE / "battery.json", encoding="utf-8") as f:
        design = json.load(f)

    df = load_pooled_trials(exp_dirs)
    human = human_proportions(df)
    print(f"[eval_models] pooled {len(df)} trials, "
          f"{df['participant'].nunique()} participants, "
          f"{len(human)} unique stimuli across "
          f"{sorted(human['experiment'].astype(str).unique())} "
          f"and vectors {sorted(human['vector'].unique())}.")

    models = load_models()
    print(f"[eval_models] models: {list(models)} "
          f"(n_subjects={args.n_subjects}, seed={args.seed})")

    per_stim, summary = score_models(human, models, design,
                                     n_subjects=args.n_subjects, seed=args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_stim.to_csv(out_dir / "eval_models_per_stimulus.csv", index=False)
    summary.to_csv(out_dir / "eval_models_summary.csv", index=False)

    exps_present = sorted(human["experiment"].astype(str).unique())
    scope = ("+".join(exps_present) + " pooled" if len(exps_present) > 1
             else f"{exps_present[0]} only")
    title = (f"Model fit to validation_online humans — {scope} "
             f"({len(human)} stimuli, {df['participant'].nunique()} participants)")
    try:
        plot_metrics(summary, out_dir / "eval_models_metrics.png", title=title)
        plot_scatter(per_stim, summary, out_dir / "eval_models_scatter.png", title=title)
        plot_msg = f"+ PNGs under {out_dir}"
    except ImportError as e:
        plot_msg = f"(plots skipped: {e})"

    pd.set_option("display.width", 200)
    print(f"\n=== per-model fit ({scope}, lower MAE/MSE / higher r = better) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote eval_models_summary.csv, eval_models_per_stimulus.csv {plot_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
