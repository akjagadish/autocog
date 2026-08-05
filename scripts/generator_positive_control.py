"""Skeptic's test #2: positive control for the data-vs-generator pipeline.

If the pipeline is sound it must correctly identify the generator of the
families we are NOT suspicious about. Tallying is the decisive control: it
ignores validities entirely (counts feature-wise wins), so the validity bug
cannot touch it -- tallying data should match canonical tallying. WADD ranks by
validity-weighted sums (a DISTINCT ranking from tally/TTB), so it controls
whether the machinery can tell rankings apart at all.

For each family's noise=0.0 runs, take one experiment, pool the on-disk choices
into empirical P(B), and correlate against every candidate model simulated the
SAME generation-faithful way (argmax(predict), ε=0, validities = the run's own).
A sound pipeline puts each family's OWN canonical model on the diagonal (highest
r, lowest MSE). That it also flags canonical TTB as the odd-one-out for the TTB
family then cannot be an artifact.

Output: results/.../generator_positive_control.png (rows = metric, x = data
family, bars = candidate model).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

_REPO = Path("/Users/aj9225/Local/autopi")
sys.path.insert(0, str(_REPO))
from src.theory import Theory  # noqa: E402

YAML_DIR = _REPO / "theories" / "heuristic_decision_making"
SYN = _REPO / "results/heuristic_decision_making/synthetic"
OUT_PNG = SYN / "generator_positive_control.png"
EMPTY_HISTORY = {"stimulus": [], "label": [],
                 "previous_stimuli": [], "previous_labels": []}
FAMILIES = ("tallying", "wadd", "ttb")
N_SUBJ = 400


def load_experiment(run_dir):
    st = json.loads((run_dir / "observations" / "state.json").read_text())
    exp = st["rounds"][0]["observations"][0]["experiment"]
    data_file = next((run_dir / "observations" / "data").glob(
        "round_000_obs_00.jsonl"))
    return exp, data_file


def empirical_pb(data_file):
    counts: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    for line in Path(data_file).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        a = tuple(int(x) for x in rec["option_a_ratings"])
        b = tuple(int(x) for x in rec["option_b_ratings"])
        key = (a, b)
        if key not in counts:
            counts[key] = [0, 0]; order.append(key)
        counts[key][0] += int(rec["response"]); counts[key][1] += 1
    return order, np.array([counts[k][0] / counts[k][1] for k in order])


def sim_pb(theory, pairs, *, validities, rating_max, seed=0):
    random.seed(seed)
    ctx = {"n_features": len(validities), "n_labels": 2,
           "validities": list(validities), "rating_max": int(rating_max)}
    acc = np.zeros(len(pairs))
    for _ in range(N_SUBJ):
        params = theory.sample_parameters(ctx)
        for j, (a, b) in enumerate(pairs):
            probs = np.asarray(theory.predict(
                params, np.array([a, b], dtype=float), EMPTY_HISTORY))
            acc[j] += int(np.argmax(probs))
    return acc / N_SUBJ


def pearson(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mse(x, y):
    return float(np.mean((np.asarray(x) - np.asarray(y)) ** 2))


def candidate_models():
    rand_ttb = Theory.from_yaml(YAML_DIR / "ttb.yaml")
    rand_ttb.parameters["validities"] = "[(0.0, 1.0)] * n_features"
    return {
        "canon tallying": Theory.from_yaml(YAML_DIR / "tallying.yaml"),
        "canon WADD": Theory.from_yaml(YAML_DIR / "wadd.yaml"),
        "canon TTB": Theory.from_yaml(YAML_DIR / "ttb.yaml"),
        "randomized-cue TTB": rand_ttb,
    }


CAND_ORDER = ("canon tallying", "canon WADD", "canon TTB", "randomized-cue TTB")
CAND_COLOR = {"canon tallying": "#4C72B0", "canon WADD": "#DD8452",
              "canon TTB": "#C44E52", "randomized-cue TTB": "#8172B2"}


def main():
    models = candidate_models()
    agg_r: dict[str, dict[str, list[float]]] = {}
    agg_m: dict[str, dict[str, list[float]]] = {}
    for fam in FAMILIES:
        fam_dir = SYN / fam / "noise=0.0"
        runs = sorted(fam_dir.glob(f"hdm_ground_truth_{fam}_*_run*"))
        print(f"\n=== data family: {fam}  ({len(runs)} runs)")
        for run in runs:
            exp, data_file = load_experiment(run)
            validities = [float(v) for v in exp["validities"]]
            rating_max = int(exp["rating_max"])
            pairs, data = empirical_pb(data_file)
            print(f"    {run.name[-4:]} val={validities}")
            for name in CAND_ORDER:
                vec = sim_pb(models[name], pairs, validities=validities,
                             rating_max=rating_max)
                r, m = pearson(vec, data), mse(vec, data)
                agg_r.setdefault(fam, {}).setdefault(name, []).append(r)
                agg_m.setdefault(fam, {}).setdefault(name, []).append(m)
                print(f"        data vs {name:20s} r={r:+.3f}  mse={m:.4f}")
    plot(agg_r, agg_m, OUT_PNG)


def _mean_sem(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan, 0.0
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, s


def plot(agg_r, agg_m, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(FAMILIES))
    width = 0.8 / len(CAND_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, agg, ylab, ttl in [
        (axes[0], agg_r, "Pearson r (empirical data vs model)", "Correlation"),
        (axes[1], agg_m, "MSE (lower=better)", "MSE"),
    ]:
        for i, name in enumerate(CAND_ORDER):
            means = np.array([_mean_sem(agg[f].get(name, []))[0]
                              for f in FAMILIES], dtype=float)
            sems = np.array([_mean_sem(agg[f].get(name, []))[1]
                             for f in FAMILIES], dtype=float)
            ax.bar(x - 0.4 + width * (i + 0.5), means, width,
                   yerr=sems, capsize=2, color=CAND_COLOR[name],
                   edgecolor="black", linewidth=0.3,
                   label=name if ax is axes[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f} data" for f in FAMILIES])
        ax.set_xlabel("data-generating family")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].axhline(0.0, color="0.4", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Positive control: each family's own canonical model should "
                 "win on its own data (mean ± SEM across runs)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
