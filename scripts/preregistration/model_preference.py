"""Exploratory: per-subject model preference + within-subject consistency for an
experiment's collected data (uses the model-prediction + repeat structure that
download_data.py writes into trials.csv).

For each participant we compute their MATCH RATE to each model = proportion of
that participant's trials on which their choice equals the model's predicted
option, over the trials where that model makes a (non-tie) prediction. The
"preferred" model is the participant's argmax match rate (ties reported).

WITHIN-SUBJECT VARIANCE: every stimulus is shown several times per participant,
so for each (participant, stimulus) we have a choice proportion p and a binary-
choice variance p*(1-p) (0 = perfectly consistent across repeats, 0.25 = 50/50 /
maximally noisy). A participant's within-subject variance is the mean of that
over their stimuli. We also test whether it differs across the preferred-model
groups (Kruskal-Wallis).

Usage:  .venv/bin/python validation_online/model_preference.py --experiment exp1
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("concave", "wadd", "tallying", "ttb")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["exp1", "exp2"], default="exp1")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    path = args.data or os.path.join(HERE, "data", args.experiment, "trials.csv")
    df = pd.read_csv(path)
    n_sub = df["participant"].nunique()
    print(f"{args.experiment}: {len(df)} trials, {n_sub} participants "
          f"(model-discrimination trials only have model predictions)\n")

    # restrict to trials that carry model predictions (Exp 1 md trials)
    md = df[df["concave_pred"].notna()].copy()

    # ---- per-subject match rate to each model -------------------------------
    rows = []
    for pid, g in md.groupby("participant"):
        row = {"participant": pid}
        for m in MODELS:
            pred = g[f"{m}_pred"]
            ok = pred.notna()
            row[f"match_{m}"] = float((g["response"][ok] == pred[ok]).mean()) if ok.any() else np.nan
            row[f"n_{m}"] = int(ok.sum())
        rows.append(row)
    sub = pd.DataFrame(rows)

    # preferred model = argmax match rate (flag ties)
    def prefer(r):
        vals = {m: r[f"match_{m}"] for m in MODELS}
        mx = max(vals.values())
        best = [m for m, v in vals.items() if abs(v - mx) < 1e-9]
        return best[0] if len(best) == 1 else "tie:" + "/".join(best)
    sub["preferred"] = sub.apply(prefer, axis=1)

    print("=== mean match rate to each model (across subjects) ===")
    for m in MODELS:
        print(f"  {m:9s}: {sub[f'match_{m}'].mean():.3f}  (SD {sub[f'match_{m}'].std():.3f})")

    print("\n=== how many subjects PREFER each model (argmax match rate) ===")
    vc = sub["preferred"].value_counts()
    for k in list(MODELS) + [k for k in vc.index if k.startswith("tie")]:
        if k in vc.index:
            print(f"  {k:9s}: {int(vc[k])} / {n_sub}")

    # ---- within-subject variance (repeat consistency) -----------------------
    g = (md.groupby(["participant", "vector", "option_a", "option_b"])["response"]
           .agg(["mean", "count"]).reset_index())
    g["within_var"] = g["mean"] * (1.0 - g["mean"])     # binary-choice variance per stimulus
    wsv = g.groupby("participant")["within_var"].mean().rename("within_subj_var")
    sub = sub.merge(wsv, on="participant")
    reps = int(g["count"].median())
    print(f"\n=== within-subject variance (repeat consistency, ~{reps} reps/stimulus) ===")
    print(f"  mean p(1-p) within stimulus across subjects: {sub['within_subj_var'].mean():.3f} "
          f"(SD {sub['within_subj_var'].std():.3f}); 0=perfectly consistent, 0.25=random")
    # implied per-stimulus choice consistency (how often they repeat the same choice)
    print(f"  mean within-stimulus choice proportion |p-0.5| (higher=more decisive): "
          f"{(g['mean'].sub(0.5).abs()).mean():.3f}")

    print("\n=== does within-subject variance differ by preferred model? ===")
    grp = {m: sub.loc[sub["preferred"] == m, "within_subj_var"].to_numpy()
           for m in MODELS if (sub["preferred"] == m).any()}
    for m, v in grp.items():
        print(f"  prefers {m:9s} (n={len(v):2d}): within-subj var mean={v.mean():.3f} SD={v.std():.3f}")
    if len(grp) >= 2:
        H, p = stats.kruskal(*grp.values())
        print(f"  Kruskal-Wallis across preferred-model groups: H={H:.3f}, p={p:.3f} "
              f"({'differs' if p < 0.05 else 'no significant difference'})")

    print("\n=== between-subject variance of each model's match rate ===")
    for m in MODELS:
        print(f"  {m:9s}: var={sub[f'match_{m}'].var():.4f}  SD={sub[f'match_{m}'].std():.3f}")
    # do the four match-rate distributions differ in spread? Levene on match rates
    W, pv = stats.levene(*[sub[f"match_{m}"].dropna() for m in MODELS])
    print(f"  Levene test (equal variance of match rates across models): "
          f"W={W:.3f}, p={pv:.3f} ({'variances differ' if pv < 0.05 else 'no sig. difference'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
