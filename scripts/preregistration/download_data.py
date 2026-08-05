"""Assemble the analysis dataset from the collected raw observations.

The study is two SEPARATE experiments; this script assembles ONE experiment's
dataset per invocation (``--experiment exp1|exp2``). For the chosen experiment,
``run_experiment.py`` writes two files under ``data/<experiment>/``:
  raw_observations.json  — the runner's raw per-participant observations
  conditions.json        — {slot_key: validity-vector id} assignment map

This script parses those into a single trial-level table,
``data/<experiment>/trials.csv``, with one row per completed choice trial and
columns:
  participant, vector, validities, experiment, component, condition,
  option_a, option_b, response (0=A, 1=B), rt (ms),
  concave_pred, wadd_pred, tallying_pred, ttb_pred, compare, cell, shift, d
where `validities` is the displayed accuracy vector, `component`/`condition`
say what the trial tests (md / steep_vs_flat / level_shift_*), and `*_pred` are
each model's predicted option (0=A, 1=B) from the frozen battery (`concave_pred`
is the curvature-consistent option on every trial; the heuristic predictions and
`compare` are filled for the Experiment-1 model-discrimination trials).

Inclusion follows the registration: a participant is included if and only if the
study was completed, defined as the full set of choice trials being recorded
(the experiment's trials_per_participant from battery.json). Participants with
incomplete data (technical failure or non-completion) are excluded. No
response-pattern, response-time, or performance exclusions are applied.

Usage:
  .venv/bin/python validation_online/download_data.py --experiment exp1
  .venv/bin/python validation_online/download_data.py --experiment exp2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from sweetbean.util.data_process import parse_autora_observations  # noqa: E402

def _as_int_list(v) -> str:
    return json.dumps([int(x) for x in v])


def _stimulus_meta(design: dict) -> dict:
    """Map (vector, option_a_json, option_b_json) -> the frozen per-trial
    condition + model predictions from battery.json, so every collected trial
    can be annotated with what it tests and what each model predicts.

      component     : md / steep / offset
      condition     : concave_vs_<m> | steep_vs_flat | level_shift_low/high
      concave_pred  : the curvature-consistent option (0=A, 1=B): md -> concave
                      grid choice, steep -> the steep option, offset -> the
                      low-range target option
      wadd_pred/tallying_pred/ttb_pred, compare : md only (the pairwise contrast)
      cell, shift   : offset only;  d : steep only (the gain size)
    """
    meta: dict = {}
    for vid, spec in design["vectors"].items():
        for s in spec["md"]:
            meta[(vid, json.dumps(s["a"]), json.dumps(s["b"]))] = {
                "component": "md", "condition": f"concave_vs_{s['compare']}",
                "concave_pred": s["concave"], "wadd_pred": s["wadd"],
                "tallying_pred": s["tallying"], "ttb_pred": s["ttb"],
                "compare": s["compare"]}
        for s in spec["steep"]:
            meta[(vid, json.dumps(s["a"]), json.dumps(s["b"]))] = {
                "component": "steep", "condition": "steep_vs_flat",
                "concave_pred": s["steep_opt"], "d": s["d"]}
        for s in spec["offset"]:
            meta[(vid, json.dumps(s["a"]), json.dumps(s["b"]))] = {
                "component": "offset", "condition": f"level_shift_{s['cell']}",
                "concave_pred": s["target"], "cell": s["cell"], "shift": s["shift"]}
    return meta


_META_FIELDS = ("component", "condition", "concave_pred", "wadd_pred",
                "tallying_pred", "ttb_pred", "compare", "cell", "shift", "d")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["exp1", "exp2"], required=True,
                    help="which experiment's data to assemble (sets data/<exp>/ "
                         "paths and the expected per-participant trial count)")
    ap.add_argument("--raw", default=None,
                    help="default: data/<experiment>/raw_observations.json")
    ap.add_argument("--conditions", default=None,
                    help="default: data/<experiment>/conditions.json")
    ap.add_argument("--out", default=None,
                    help="default: data/<experiment>/trials.csv")
    ap.add_argument("--append", action="store_true",
                    help="merge this batch into an existing trials.csv instead of "
                         "overwriting it, adding only participants not already "
                         "present (dedup by participant id). Lets several runs "
                         "accumulate toward the balanced per-vector target.")
    args = ap.parse_args()

    exp_dir = os.path.join(HERE, "data", args.experiment)
    raw_path = args.raw or os.path.join(exp_dir, "raw_observations.json")
    cond_path = args.conditions or os.path.join(exp_dir, "conditions.json")
    out_path = args.out or os.path.join(exp_dir, "trials.csv")

    with open(os.path.join(HERE, "battery.json"), encoding="utf-8") as f:
        design = json.load(f)
    exp_meta = design["experiments"][args.experiment]
    expected_trials = exp_meta["trials_per_participant"]
    target_per_vec = exp_meta.get("n_per_vector")
    all_vids = list(design["vectors"])

    with open(raw_path, encoding="utf-8") as f:
        observations = json.load(f)
    with open(cond_path, encoding="utf-8") as f:
        slot_to_vector = json.load(f)             # {slot_key (str): vector id}

    trials = parse_autora_observations(observations)

    # Keep the binary A/B choice trials (bean_response is the decoded key press).
    resp = trials["bean_response"].astype("string").str.lower()
    mask = resp.isin(["a", "b"])
    choice = trials.loc[mask].copy()
    choice["response"] = (resp[mask] == "b").astype(int)
    choice["option_a"] = choice["bean_option_a"].apply(_as_int_list)
    choice["option_b"] = choice["bean_option_b"].apply(_as_int_list)

    slot = choice["slot_key"].astype(str)
    choice["vector"] = slot.map(slot_to_vector)
    choice["participant"] = (choice["prolific_pid"].astype(str)
                             if "prolific_pid" in choice.columns else slot)
    # response time (ms) and the displayed accuracy/validity vector for the trial.
    choice["rt"] = choice["bean_rt"] if "bean_rt" in choice.columns else choice.get("rt")
    vid_to_vals = {vid: json.dumps(design["vectors"][vid]["validities"])
                   for vid in all_vids}
    choice["validities"] = choice["vector"].map(vid_to_vals)

    # Per-trial condition + model predictions, joined from the frozen battery.
    meta = _stimulus_meta(design)
    keys = list(zip(choice["vector"], choice["option_a"], choice["option_b"]))
    for field in _META_FIELDS:
        choice[field] = [meta.get(k, {}).get(field) for k in keys]

    counts = choice.groupby("participant").size()
    completed = set(counts[counts >= expected_trials].index)
    excluded = sorted(set(counts.index) - completed)

    kept = choice[choice["participant"].isin(completed)].copy()
    if kept["vector"].isna().any():
        missing = sorted(kept.loc[kept["vector"].isna(), "slot_key"].astype(str).unique())
        raise ValueError(f"slot_key(s) without a validity-vector assignment: {missing}")

    kept["experiment"] = args.experiment          # self-describing dataset
    out = kept[["participant", "vector", "validities", "experiment",
                "component", "condition", "option_a", "option_b",
                "response", "rt",
                "concave_pred", "wadd_pred", "tallying_pred", "ttb_pred",
                "compare", "cell", "shift", "d"]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_added = out["participant"].nunique()
    if args.append and os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        already = set(prev["participant"].astype(str))
        fresh = out[~out["participant"].astype(str).isin(already)]
        n_added = fresh["participant"].nunique()
        final = pd.concat([prev, fresh], ignore_index=True)
    else:
        final = out
    final.to_csv(out_path, index=False)

    print(f"[{args.experiment}] expected {expected_trials} trials/participant.")
    print(f"This batch: {len(completed)} completed, {len(excluded)} excluded "
          f"(incomplete); {n_added} new participant(s) "
          f"{'appended' if args.append else 'written'}.")
    by_vec = final.groupby("vector")["participant"].nunique().to_dict()
    total = int(final["participant"].nunique())
    if target_per_vec:
        tot_target = target_per_vec * len(all_vids)
        print(f"Cumulative participants per validity vector (target "
              f"{target_per_vec} each, {tot_target} total) -> {out_path}:")
        for vid in all_vids:
            n = int(by_vec.get(vid, 0))
            print(f"  {vid}: {n}/{target_per_vec}{'  DONE' if n >= target_per_vec else ''}")
        print(f"  TOTAL: {total}/{tot_target}"
              + ("  ALL CONDITIONS FILLED" if total >= tot_target
                 and all(by_vec.get(v, 0) >= target_per_vec for v in all_vids)
                 else ""))
    else:
        print(f"Cumulative participants per validity vector -> {out_path}:")
        for vid in all_vids:
            print(f"  {vid}: {int(by_vec.get(vid, 0))}")
        print(f"  TOTAL: {total}")


if __name__ == "__main__":
    main()
