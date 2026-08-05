"""Skeptic's test #1: distinguish randomized-cue TTB from its behavioural twin
(probability-matched Tallying) and from canonical TTB using WITHIN-SUBJECT
structure, which per-stimulus proportions throw away.

50 subjects x 19 repeats/stimulus gives the leverage. Two axes, no permutation
overfitting:

  D (within-subject determinism) = mean over (subject, stimulus) of
      max(p, 1-p) on that subject's repeats.  Deterministic generator -> ~1.0;
      a per-trial coin (prob-matched tally) -> < 1.0.
  G (pooled gradedness) = mean over stimuli of min(p_pool, 1-p_pool).
      Binary 0/1 proportions (canonical TTB) -> ~0; graded -> > 0.

Fingerprints:  canonical (D~1, G~0) | randomized-cue (D~1, G>0) |
               prob-matched tally (D<1, G>0).
Only randomized-cue TTB sits at HIGH D and HIGH G.

Posterior-predictive: we compute (D, G) on the REAL data and on data SIMULATED
from each candidate with the run's exact design (same stimuli, validities,
n_subjects, repeats, action noise) and see which candidate the data lands on.

Confirmatory lexicographic check: collapse each subject to its majority choice
vector and ask whether ONE cue permutation explains it (TTB signature) and
whether the best permutation VARIES across subjects (randomized-cue) or is
shared (canonical).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
SYN = _REPO / "results/synthetic_cardinal"
OUT_PNG = SYN / "within_subject_generator_test.png"


# ----------------------------- data loading ------------------------------- #
def load_records(run_dir, round_idx=0, obs_idx=0):
    f = next((run_dir / "observations" / "data").glob(
        f"round_{round_idx:03d}_obs_{obs_idx:02d}.jsonl"))
    recs = []
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs.append((r["subject_id"],
                         tuple(int(x) for x in r["option_a_ratings"]),
                         tuple(int(x) for x in r["option_b_ratings"]),
                         int(r["response"])))
    return recs


def experiment_meta(run_dir, round_idx=0, obs_idx=0):
    st = json.loads((run_dir / "observations" / "state.json").read_text())
    exp = st["rounds"][round_idx]["observations"][obs_idx]["experiment"]
    return [float(v) for v in exp["validities"]], int(exp["rating_max"])


# --------------------------- the two statistics --------------------------- #
def determinism_gradedness(recs):
    """D, G from a list of (subject, a, b, response) records."""
    by_ss: dict[tuple, list[int]] = defaultdict(list)   # (subject, stim) -> responses
    by_stim: dict[tuple, list[int]] = defaultdict(list)  # stim -> all responses
    for s, a, b, y in recs:
        by_ss[(s, (a, b))].append(y)
        by_stim[(a, b)].append(y)
    det = [max(np.mean(v), 1 - np.mean(v)) for v in by_ss.values()]
    grad = [min(np.mean(v), 1 - np.mean(v)) for v in by_stim.values()]
    return float(np.mean(det)), float(np.mean(grad))


# ----------------------------- candidate sims ----------------------------- #
def _ttb_choice(a, b, order):
    """First cue in `order` that discriminates decides; b>a -> 1 (B), else 0.
    Full tie -> 0 (option A), matching argmax of equal scores."""
    for c in order:
        if a[c] != b[c]:
            return 1 if b[c] > a[c] else 0
    return 0


def _tally_fraction(a, b):
    disc = [(ai, bi) for ai, bi in zip(a, b) if ai != bi]
    if not disc:
        return 0.5
    return sum(bi > ai for ai, bi in disc) / len(disc)


def simulate(kind, stimuli, *, validities, n_subj, reps, eps, rng):
    """Return records [(subject, a, b, response)] for a candidate generator.
      kind='canonical'    : every subject uses descending-validity order.
      kind='randomized'   : every subject draws a fresh random cue order.
      kind='tally_pm'     : every trial ~ Bernoulli(tally fraction).
    eps-greedy action noise: with prob eps the choice is a uniform coin."""
    n_features = len(validities)
    desc = list(np.argsort(-np.asarray(validities), kind="stable"))
    recs = []
    for s in range(n_subj):
        if kind == "randomized":
            order = list(rng.permutation(n_features))
        else:
            order = desc
        for (a, b) in stimuli:
            for _ in range(reps):
                if eps > 0 and rng.random() < eps:
                    y = int(rng.integers(0, 2))
                elif kind == "tally_pm":
                    y = int(rng.random() < _tally_fraction(a, b))
                else:
                    y = _ttb_choice(a, b, order)
                recs.append((s, a, b, y))
    return recs


# -------------------------- lexicographic check --------------------------- #
def lexicographic_stats(recs, n_features):
    """Collapse each subject to its majority choice vector over unique stimuli;
    find the single cue permutation that best reproduces it. Return mean best
    match-rate and the number of DISTINCT best permutations across subjects."""
    stim_order: list[tuple] = []
    seen = set()
    by_ss: dict[tuple, list[int]] = defaultdict(list)
    subjects: list[int] = []
    for s, a, b, y in recs:
        if s not in seen:
            seen.add(s); subjects.append(s)
        if (a, b) not in stim_order and (a, b) not in set(stim_order):
            pass
        by_ss[(s, (a, b))].append(y)
    stims = sorted({(a, b) for _, a, b, _ in recs})
    perms = list(permutations(range(n_features)))
    pred_cache = {p: np.array([_ttb_choice(a, b, p) for (a, b) in stims])
                  for p in perms}
    best_matches, best_perms = [], []
    for s in subjects:
        cvec = np.array([1 if np.mean(by_ss[(s, st)]) > 0.5 else 0 for st in stims])
        scores = [(np.mean(pred_cache[p] == cvec), p) for p in perms]
        m, p = max(scores, key=lambda t: t[0])
        best_matches.append(m); best_perms.append(p)
    return float(np.mean(best_matches)), len(set(best_perms)), len(subjects)


# --------------------------------- driver --------------------------------- #
CANDS = [("canonical", "#C44E52"), ("randomized", "#8172B2"),
         ("tally_pm", "#55A868")]
CAND_LABEL = {"canonical": "canonical TTB", "randomized": "randomized-cue TTB",
              "tally_pm": "prob-matched tally", "data": "REAL DATA"}


def analyse(run_dir, eps, seed=0):
    validities, _ = experiment_meta(run_dir)
    n_features = len(validities)
    recs = load_records(run_dir)
    n_subj = len({s for s, *_ in recs})
    stims = sorted({(a, b) for _, a, b, _ in recs})
    reps = len(recs) // (n_subj * len(stims))

    out = {}
    out["data"] = determinism_gradedness(recs)
    lm, lp, ns = lexicographic_stats(recs, n_features)
    out["data_lex"] = (lm, lp, ns)

    rng = np.random.default_rng(seed)
    for kind, _ in CANDS:
        sim = simulate(kind, stims, validities=validities, n_subj=n_subj,
                       reps=reps, eps=eps, rng=rng)
        out[kind] = determinism_gradedness(sim)
        out[kind + "_lex"] = lexicographic_stats(sim, n_features)
    return out, n_features, reps, n_subj


def main():
    levels = [("noise=0.0", 0.0), ("noise=0.3", 0.3)]
    results = {}
    for lvl, eps in levels:
        for run in sorted((SYN / "ttb" / lvl).glob("hdm_ground_truth_ttb_*_run*")):
            out, nf, reps, ns = analyse(run, eps)
            results[(lvl, run.name[-4:])] = out
            d, g = out["data"]
            lm, lp, nsub = out["data_lex"]
            print(f"\n=== {lvl} {run.name[-4:]}  (n_features={nf}, "
                  f"{ns} subj x {reps} reps)")
            print(f"    {'generator':22s} {'D(within-subj)':>14s} "
                  f"{'G(graded)':>10s} {'lex-match':>10s} {'#perms':>7s}")
            for key in ("data", "canonical", "randomized", "tally_pm"):
                dd, gg = out[key]
                lmm, lpp, _ = out[key + "_lex"] if key != "data" else out["data_lex"]
                print(f"    {CAND_LABEL[key]:22s} {dd:14.3f} {gg:10.3f} "
                      f"{lmm:10.3f} {lpp:7d}")
    plot(results, OUT_PNG)


def plot(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = ["noise=0.0", "noise=0.3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, lvl in zip(axes, levels):
        keys = [k for k in results if k[0] == lvl]
        # candidate clouds
        for kind, color in CANDS:
            xs = [results[k][kind][0] for k in keys]
            ys = [results[k][kind][1] for k in keys]
            ax.scatter(xs, ys, s=70, color=color, edgecolor="black",
                       linewidth=0.4, alpha=0.85, label=CAND_LABEL[kind], zorder=3)
        # real data (black stars)
        xs = [results[k]["data"][0] for k in keys]
        ys = [results[k]["data"][1] for k in keys]
        ax.scatter(xs, ys, s=240, marker="*", color="black",
                   edgecolor="white", linewidth=0.6, label="REAL DATA", zorder=5)
        ax.set_title(lvl)
        ax.set_xlabel("D  =  within-subject determinism")
        ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("G  =  pooled-proportion gradedness")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Within-subject test: only randomized-cue TTB sits at "
                 "HIGH determinism AND HIGH gradedness", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
