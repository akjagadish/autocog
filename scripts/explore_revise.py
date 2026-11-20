"""How each discovery LLM explores experiments and revises theories.

Reads a `compare_experiments.py` output directory (made with `--models-dir`)
and each run's round history, and asks two questions per LLM:

1. **Exploring experiments.** In discrimination-profile space: how varied a
   run's experiments are, how far each new experiment is from the run's
   earlier ones (novelty), and how far they sit from the Hilbig & Moshagen
   design. Targeting compares each experiment with a typical experiment of
   the same family (1 = typical): how much it separates the two theories
   being compared that round (`rival_sep`), and how badly the theory that
   ends up killed misses the ground truth on it (`killed_exposed`).

2. **Revising theories.** Every round kills one of the two alive theories
   and writes a replacement into its slot (`rounds/round_*/theories.json`),
   so each revision has a parent (killed) and a child (replacement). Over
   the family's proposed experiments, with T = RMS distance from the
   ground truth's predictions (transfer error):
     step_behaviour  RMS difference between killed and replacement
     step_code       code distance between them (if available)
     killed_the_worse  T(killed) > T(survivor)
     progress        T(killed) - T(replacement)
     efficiency      progress / step_behaviour (1 = straight toward truth)
     new_beats_best  replacement better than both alive theories
   Per run: gain = best T among the alive theories before round 0 minus
   after the last round; path_efficiency = gain / summed step sizes.

Outputs (under `--out`): revision_steps.csv, experiment_targeting.csv,
experiment_novelty.csv, run_summary.csv, exploration_by_llm.csv,
revision_by_llm.csv, and printed tests (binomial against chance, Spearman
of run gain against exploration and targeting).

Run from the repo root:

  python scripts/explore_revise.py --ce-dir <compare_experiments out-dir> \\
      --max-rounds 5 --out <out-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.compare_experiments import euclidean_distances  # noqa: E402
from scripts.compare_models import _round_dirs, llm_label, parse_run_dir  # noqa: E402


# ---------------------------------------------------------------------------
# Lineage.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Round:
    round: int
    alive: tuple[str, ...]
    killed: str
    survivor: str
    replacement: str
    seeds: frozenset[str]


def load_lineage(run_dir: Path, *, max_rounds: int | None = None) -> list[Round]:
    """Each round's alive theories, the one killed and its replacement."""
    rounds = _round_dirs(run_dir)
    if max_rounds is not None:
        rounds = rounds[:max_rounds]
    out: list[Round] = []
    seeds: frozenset[str] = frozenset()
    for i, rd in enumerate(rounds):
        d = json.loads((rd / "theories.json").read_text())
        alive = tuple(s["label"] for s in d["starting_theories"])
        if i == 0:
            seeds = frozenset(alive)
        killed = [s["label"] for s in d["starting_theories"] if s.get("killed")]
        repl = d.get("replacement")
        if len(killed) != 1 or repl is None:
            continue
        survivor = [a for a in alive if a != killed[0]]
        out.append(Round(i, alive, killed[0], survivor[0] if survivor else "",
                         repl["label"], seeds))
    return out


def model_id(info: dict[str, Any], label: str, *, seed: bool, known: set[str],
             kept_seed: dict[tuple[str, str], str]) -> str | None:
    """A run's theory as a `compare_experiments` model id. Seeds were
    de-duplicated across runs, so a seed maps to its kept copy."""
    m = f"{info['llm']}/{info['family']}/run{info['replicate']}/{label}"
    if m in known:
        return m
    if seed:
        return kept_seed.get((info["family"], label))
    return None


# ---------------------------------------------------------------------------
# Distances on the joint prediction matrix.
# ---------------------------------------------------------------------------


def rms(P: list[np.ndarray], i: int, j: int, exps: list[int]) -> float:
    """RMS difference between theories (rows) i and j over experiments,
    each experiment weighted equally."""
    return float(np.sqrt(np.mean([np.nanmean((P[e][i] - P[e][j]) ** 2) for e in exps])))


def revision_step(P: list[np.ndarray], row: dict[str, int], exps: list[int], *,
                  killed: str, survivor: str, new: str, gt: str) -> dict[str, Any]:
    k, s, n, g = row[killed], row[survivor], row[new], row[gt]
    tk, ts, tn = rms(P, k, g, exps), rms(P, s, g, exps), rms(P, n, g, exps)
    step = rms(P, k, n, exps)
    return {"step_behaviour": step, "T_killed": tk, "T_survivor": ts, "T_new": tn,
            "killed_the_worse": tk > ts, "progress": tk - tn,
            "efficiency": (tk - tn) / step if step > 0 else np.nan,
            "new_beats_best": tn < min(tk, ts),
            "best_before": min(tk, ts), "best_after": min(ts, tn)}


def targeting(P: list[np.ndarray], row: dict[str, int], e: int, family_exps: list[int], *,
              alive: tuple[str, ...], killed: str, gt: str) -> dict[str, float]:
    """Experiment e against the family's typical experiment (ratio)."""
    a, b, k, g = row[alive[0]], row[alive[1]], row[killed], row[gt]
    sep = [rms(P, a, b, [x]) for x in family_exps]
    exp_k = [rms(P, k, g, [x]) for x in family_exps]
    return {"rival_sep": rms(P, a, b, [e]) / np.mean(sep),
            "killed_exposed": rms(P, k, g, [e]) / np.mean(exp_k)}


def novelty(D: np.ndarray, rounds: np.ndarray) -> np.ndarray:
    """Distance from each experiment to the nearest one from an earlier
    round of the same run (NaN in the first round)."""
    out = np.full(len(rounds), np.nan)
    for i, r in enumerate(rounds):
        prev = np.where(rounds < r)[0]
        if len(prev):
            out[i] = D[i, prev].min()
    return out


def run_summary(steps: pd.DataFrame) -> pd.DataFrame:
    s = steps.sort_values("round").groupby("run").agg(
        llm=("llm", "first"), family=("family", "first"),
        best_start=("best_before", "first"), best_end=("best_after", "last"),
        path=("step_behaviour", "sum"), total_progress=("progress", "sum"))
    s["gain"] = s["best_start"] - s["best_end"]
    s["path_efficiency"] = s["gain"] / s["path"]
    return s.reset_index()


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ce-dir", type=Path, required=True,
                    help="compare_experiments.py output (run with --models-dir).")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="Match the --max-rounds used for compare_experiments.")
    ap.add_argument("--results-root", type=Path, default=_REPO_ROOT / "results")
    ap.add_argument("--min-gain", type=float, default=0.005,
                    help="A run counts as improved when its gain exceeds this.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    th = pd.read_csv(args.ce_dir / "theories.csv")
    ex = pd.read_csv(args.ce_dir / "experiments.csv")
    ex = ex[ex["llm"] != "reference"].reset_index(drop=True)
    npz = np.load(args.ce_dir / "joint_predictions.npz")
    t_ids = [str(t) for t in npz["theory_ids"]]
    e_ids = [str(e) for e in npz["exp_ids"]]
    if e_ids != ex["exp_id"].tolist():
        raise SystemExit("experiments.csv and joint_predictions.npz disagree on experiments")
    P = [npz[f"e{i}"] for i in range(len(e_ids))]
    row = {t: i for i, t in enumerate(t_ids)}
    cpath = args.ce_dir / "theory_distances_code.csv"
    code = pd.read_csv(cpath, index_col=0) if cpath.is_file() else None
    kept_seed = {(r.family, r.label): r.model for r in th[th["generation"] == 0].itertuples()}
    known = set(row)
    fam_exps = {f: g.index.tolist() for f, g in ex.groupby("family")}

    steps, targ = [], []
    for run in sorted(ex["run"].unique()):
        rd = args.results_root / run
        info = parse_run_dir(rd)
        fam, gt = info["family"], f"ground_truth/{info['family']}"
        if gt not in row:
            print(f"  skipping {run}: no ground truth (human run?)")
            continue
        fe = fam_exps[fam]
        for rnd in load_lineage(rd, max_rounds=args.max_rounds):
            ids = {lab: model_id(info, lab, seed=lab in rnd.seeds, known=known,
                                 kept_seed=kept_seed)
                   for lab in (*rnd.alive, rnd.replacement)}
            if None in ids.values() or len(rnd.alive) != 2:
                print(f"  skipping {run} round {rnd.round}: theory not in the joint space")
                continue
            k, s, n = ids[rnd.killed], ids[rnd.survivor], ids[rnd.replacement]
            step = {"run": run, "llm": llm_label(info["llm"]), "family": fam,
                    "round": rnd.round, "killed": k, "survivor": s, "replacement": n}
            step |= revision_step(P, row, fe, killed=k, survivor=s, new=n, gt=gt)
            if code is not None and {k, s, n} <= set(code.index):
                step["step_code"] = float(code.at[k, n])
                step["new_vs_survivor_code"] = float(code.at[s, n])
            steps.append(step)
            alive = tuple(ids[a] for a in rnd.alive)
            for e in [e for e in fe if ex.at[e, "run"] == run and ex.at[e, "round"] == rnd.round]:
                targ.append({"exp_id": e_ids[e], "run": run, "llm": llm_label(info["llm"]),
                             "family": fam, "round": rnd.round,
                             **targeting(P, row, e, fe, alive=alive, killed=k, gt=gt)})
    st, tg = pd.DataFrame(steps), pd.DataFrame(targ)

    prof = ex[[c for c in ex.columns if c.startswith("disc_")]].to_numpy(float)
    D = euclidean_distances(prof)
    nov = []
    for run, g in ex.groupby("run"):
        idx = g.index.to_numpy()
        v = novelty(D[np.ix_(idx, idx)], g["round"].to_numpy())
        nov += [{"exp_id": ex.at[i, "exp_id"], "run": run, "llm": llm_label(ex.at[i, "llm"]),
                 "round": int(ex.at[i, "round"]), "novelty": x} for i, x in zip(idx, v)]
    nov_df = pd.DataFrame(nov)
    div = pd.read_csv(args.ce_dir / "run_diversity.csv")
    div["llm"] = div["llm"].map(llm_label)

    explore = pd.concat([
        div.groupby("llm")["profile_diversity"].mean().rename("within_run_spread"),
        nov_df.groupby("llm")["novelty"].mean().rename("novelty"),
        div.groupby("llm")["profile_dist_to_hilbig"].mean().rename("dist_to_hilbig"),
        tg.groupby("llm")["rival_sep"].median().rename("rival_sep"),
        tg.groupby("llm")["killed_exposed"].median().rename("killed_exposed"),
    ], axis=1).sort_values("within_run_spread", ascending=False)

    runs = run_summary(st)
    g = st.groupby("llm")
    rg = runs.groupby("llm")
    revise = pd.DataFrame({
        "n_revisions": g.size(),
        "step_behaviour": g["step_behaviour"].median(),
        "step_code": g["step_code"].median() if "step_code" in st else np.nan,
        "killed_the_worse": g["killed_the_worse"].mean(),
        "improves_on_killed": g["progress"].apply(lambda x: (x > 0).mean()),
        "new_beats_best": g["new_beats_best"].mean(),
        "efficiency": g["efficiency"].median(),
        "gain": rg["gain"].median(),
        "runs_improved": rg["gain"].apply(lambda x: int((x > args.min_gain).sum())),
        "n_runs": rg.size(),
    }).sort_values("gain", ascending=False)

    st.to_csv(args.out / "revision_steps.csv", index=False)
    tg.to_csv(args.out / "experiment_targeting.csv", index=False)
    nov_df.to_csv(args.out / "experiment_novelty.csv", index=False)
    runs.to_csv(args.out / "run_summary.csv", index=False)
    explore.to_csv(args.out / "exploration_by_llm.csv")
    revise.to_csv(args.out / "revision_by_llm.csv")

    print("Exploring experiments:")
    print(explore.round(2).to_string())
    print("\nRevising theories:")
    print(revise.round(2).to_string())
    n = len(st)
    kw = int(st["killed_the_worse"].sum())
    imp = int((st["progress"] > 0).sum())
    print(f"\nKilled the worse theory: {kw}/{n} (binomial p = {binomtest(kw, n).pvalue:.2g})")
    print(f"Replacement better than the killed theory: {imp}/{n} "
          f"(binomial p = {binomtest(imp, n).pvalue:.2g})")
    print(f"Median efficiency: {st['efficiency'].median():.2f}; step size vs progress: "
          f"Spearman {spearmanr(st['step_behaviour'], st['progress']).statistic:.2f}")
    for flag, grp in st.groupby("killed_the_worse"):
        print(f"  killed the {'worse' if flag else 'better'} theory -> replacement improves "
              f"{(grp['progress'] > 0).mean():.0%}")
    per_run = runs.set_index("run").join(tg.groupby("run")[["rival_sep", "killed_exposed"]].median())
    per_run = per_run.join(div.set_index("run")["profile_diversity"])
    for col in ("rival_sep", "killed_exposed", "profile_diversity"):
        r = spearmanr(per_run[col], per_run["gain"], nan_policy="omit")
        print(f"Run gain vs {col}: Spearman {r.statistic:.2f} (p = {r.pvalue:.3f}, "
              f"n = {per_run[col].notna().sum()} runs)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
