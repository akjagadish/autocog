"""Launch ONE of the two concave value-function experiments on Prolific
(recruitment) + Firebase (hosting and data) and collect its data.

The study is two SEPARATE, between-subjects experiments; this script launches
exactly one per invocation, selected with ``--experiment``:
  exp1  Model comparison  — each participant sees only the 4 model-discrimination
        stimuli per vector, replicated to 96 trials (reps md=24).
  exp2  Value curvature   — each participant sees only the 4 steep-vs-flat + 4
        level-shift/offset stimuli per vector, replicated to 96 trials
        (reps steep=8, offset=16; the subtle offset effect keeps more trials).

Both experiments use the same five validity vectors (the strictly-descending,
all-distinct 4-tuples of {.5,.6,.7,.8,.9}); each participant is assigned one
vector and the vectors are counterbalanced across that experiment's participants
(n_per_vector each). A fresh experiment is built per participant, so trial order
and left/right option side are randomized per participant. The stimulus set,
per-experiment reps, and per-experiment sample size (n_per_vector / n_total,
re-derived from the preregistered power simulation in build.py) are all fixed in
``battery.json``.

This creates a real, paid Prolific study and recruits participants for the
chosen experiment. On completion it writes, under ``data/<experiment>/``:
  raw_observations.json  — the runner's raw observations
  conditions.json        — {slot_key: validity-vector id}
which download_data.py turns into that experiment's analysis dataset.

Usage:
  # build only, no upload
  .venv/bin/python scripts/preregistration/run_experiment.py --experiment exp1 --dry_run
  # full preregistered Prolific run (exp1=50, exp2=100; balanced across vectors)
  .venv/bin/python scripts/preregistration/run_experiment.py --experiment exp1
  # small no-Prolific test: 2 self-served sessions on Firebase only
  .venv/bin/python scripts/preregistration/run_experiment.py --experiment exp1 \
      --backend firebase --n_sessions 2

Convenience launchers (scripts/preregistration/): run_online_firebase_exp1.sh /
run_online_firebase_exp2.sh (no-Prolific tests) and run_online_prolific_exp1.sh /
run_online_prolific_exp2.sh (full Prolific runs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)

from src.heuristic_decision_making.experiment import (  # noqa: E402
    HeuristicDecisionMakingExperiment)

DEFAULT_STUDY_URL = "https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"
DEFAULT_DURATION_MIN = 7
DEFAULT_MIN_RT_MS = 1500


def _current_per_vector(exp: str) -> dict[str, int]:
    """Completed participants per validity vector in the accumulated
    data/<exp>/trials.csv (empty if no file yet). Used by --fill."""
    import csv
    path = os.path.join(HERE, "data", exp, "trials.csv")
    if not os.path.exists(path):
        return {}
    seen: dict[str, set] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.setdefault(row["vector"], set()).add(row["participant"])
    return {k: len(v) for k, v in seen.items()}


def build_trial_lists(spec: dict, components: list[str],
                      multiplicity: dict[str, int]) -> tuple[list, list]:
    """Trial-pair lists for one validity vector for a SINGLE experiment, with
    per-component list multiplicity. Only this experiment's `components` are
    emitted; the experiment class then replicates the list to MAX_TRIALS = 96, so
    per-stimulus reps = multiplicity[c] * (96 / list_len)."""
    a_list, b_list = [], []
    for component in components:
        for s in spec[component]:
            for _ in range(multiplicity[component]):
                a_list.append(list(s["a"]))
                b_list.append(list(s["b"]))
    return a_list, b_list


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["exp1", "exp2"], required=True,
                    help="which experiment to launch: exp1 (model comparison) "
                         "or exp2 (value curvature)")
    ap.add_argument("--per_vector", type=int, default=None,
                    help="participants per validity vector; default = the "
                         "experiment's preregistered n_per_vector in battery.json")
    ap.add_argument("--n_sessions", type=int, default=None,
                    help="total number of sessions to build, assigned round-robin "
                         "across the 5 validity vectors (overrides --per_vector). "
                         "Use for small no-Prolific smoke tests, e.g. --n_sessions 2.")
    ap.add_argument("--backend", choices=["firebase", "firebase_prolific"],
                    default="firebase_prolific")
    ap.add_argument("--study_url", default=DEFAULT_STUDY_URL)
    ap.add_argument("--completion_code", default=None,
                    help="defaults to completion_code in prolific_settings.json")
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION_MIN,
                    help="study timeout in minutes (also drives Prolific reward)")
    ap.add_argument("--min_rt_ms", type=int, default=DEFAULT_MIN_RT_MS)
    ap.add_argument("--firebase_credentials_path",
                    default=os.path.join(REPO_ROOT, "firebase_credentials.json"))
    ap.add_argument("--prolific_settings_path",
                    default=os.path.join(REPO_ROOT, "prolific_settings.json"))
    ap.add_argument("--consent_yaml_path", default=os.path.join(REPO_ROOT, "consent.yaml"))
    ap.add_argument("--fill", action="store_true",
                    help="recruit only for validity vectors still under their "
                         "target n_per_vector, reading current counts from "
                         "data/<exp>/trials.csv (use with download_data --append "
                         "to top up a balanced sample across several runs). "
                         "--n_sessions then caps how many to recruit this batch.")
    ap.add_argument("--dry_run", action="store_true",
                    help="build and validate all experiments without uploading")
    args = ap.parse_args()

    with open(os.path.join(HERE, "battery.json"), encoding="utf-8") as f:
        design = json.load(f)
    exp_spec = design["experiments"][args.experiment]
    components = exp_spec["components"]
    multiplicity = exp_spec["multiplicity"]
    vids = list(design["vectors"])
    # Build plan = the ordered list of validity vectors, one entry per session.
    #  --fill: recruit only for vectors still below target n_per_vector (reading
    #          current counts from the accumulated trials.csv), round-robin over
    #          the under-filled vectors, capped by --n_sessions if given.
    #  --n_sessions (alone): total cap, round-robin across all vectors (smoke test).
    #  otherwise: per_vector (default = preregistered n_per_vector) per vector.
    if args.fill:
        target = exp_spec["n_per_vector"]
        if target is None:
            raise SystemExit(f"{args.experiment}: no n_per_vector in battery.json.")
        current = _current_per_vector(args.experiment)
        remaining = {vid: max(0, target - current.get(vid, 0)) for vid in vids}
        print(f"[{args.experiment}] fill toward n_per_vector={target}. Current "
              f"per-vector: { {v: current.get(v, 0) for v in vids} }; still needed: "
              f"{ {v: remaining[v] for v in vids} }.")
        plan = []
        cap = args.n_sessions
        while sum(remaining.values()) > 0 and (cap is None or len(plan) < cap):
            # neediest vectors first (stable by vector order on ties), so small
            # batches go to the emptiest conditions and stay balanced as they fill
            order = sorted(vids, key=lambda v: (-remaining[v], vids.index(v)))
            progressed = False
            for vid in order:
                if remaining[vid] > 0:
                    plan.append(vid); remaining[vid] -= 1; progressed = True
                    if cap is not None and len(plan) >= cap:
                        break
            if not progressed:
                break
        if not plan:
            print(f"[{args.experiment}] all vectors already at target; "
                  f"nothing to recruit.")
            return
    elif args.n_sessions is not None:
        if args.n_sessions < 1:
            raise SystemExit("--n_sessions must be >= 1.")
        plan = [vids[i % len(vids)] for i in range(args.n_sessions)]
    else:
        per_vector = args.per_vector or exp_spec["n_per_vector"]
        if per_vector is None:
            raise SystemExit(f"{args.experiment}: no n_per_vector in battery.json; "
                             f"re-run build.py or pass --per_vector / --n_sessions.")
        plan = [vid for vid in vids for _ in range(per_vector)]
    consent_config: dict = {}
    if os.path.exists(args.consent_yaml_path):
        with open(args.consent_yaml_path, encoding="utf-8") as f:
            consent_config = yaml.safe_load(f) or {}

    completion_code = args.completion_code or ""
    if not completion_code and os.path.exists(args.prolific_settings_path):
        with open(args.prolific_settings_path, encoding="utf-8") as f:
            completion_code = (json.load(f) or {}).get("completion_code", "")

    # Build one fresh experiment per session (independent trial-order and
    # option-side randomization). Only this experiment's components are included.
    trial_lists = {vid: build_trial_lists(design["vectors"][vid],
                                          components, multiplicity) for vid in vids}
    payload_experiments: list = []
    payload_conditions: list[dict] = []
    slot_to_vector: dict[str, str] = {}
    for slot, vid in enumerate(plan):
        spec = design["vectors"][vid]
        a_list, b_list = trial_lists[vid]
        exp = HeuristicDecisionMakingExperiment(
            validities=spec["validities"],
            rating_max=design["rating_max"],
            trial_a_ratings=a_list,
            trial_b_ratings=b_list,
            rationale=(f"Concave value-function study, {args.experiment} "
                       f"({exp_spec['label']}); components {components} "
                       f"under validity vector {vid}."),
        )
        js_exp, _timelines = exp.build_online_experiment(
            consent_config=consent_config, min_rt_ms=args.min_rt_ms)
        payload_experiments.append(js_exp)
        payload_conditions.append({"vector": vid, "slot_in_vector": slot})
        slot_to_vector[str(slot)] = vid

    n = len(payload_experiments)
    trials_pp = exp_spec["trials_per_participant"]
    per_vec_counts = {vid: plan.count(vid) for vid in vids}
    print(f"[{args.experiment}: {exp_spec['label']}] built {n} session(s) "
          f"(per-vector {per_vec_counts}), components={components}, "
          f"reps={exp_spec['reps']}, rating_max={design['rating_max']}, "
          f"{trials_pp} trials each.")
    if args.dry_run:
        print("dry_run: not uploading. Per-vector:")
        for vid in vids:
            a, _ = trial_lists[vid]
            print(f"  {vid} validities={design['vectors'][vid]['validities']}: "
                  f"{len(a)} unique-list entries -> {trials_pp} trials "
                  f"(x{per_vec_counts[vid]} session(s))")
        return

    from src.online_workflow import run_online

    observations = run_online(
        payload_experiments,
        payload_conditions,
        credentials_path=args.firebase_credentials_path,
        is_prolific=(args.backend == "firebase_prolific"),
        study_completion_time=args.duration,
        prolific_settings_path=args.prolific_settings_path,
        study_url=args.study_url.strip(),
        completion_code=completion_code,
        share_template=False,                 # heterogeneous experiments
    )

    data_dir = os.path.join(HERE, "data", args.experiment)
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "raw_observations.json"), "w", encoding="utf-8") as f:
        json.dump(observations, f, default=str)
    with open(os.path.join(data_dir, "conditions.json"), "w", encoding="utf-8") as f:
        json.dump(slot_to_vector, f, indent=1)
    print(f"Collected {len(observations)} sessions. Wrote "
          f"data/{args.experiment}/raw_observations.json and conditions.json. "
          f"Next: download_data.py --experiment {args.experiment}, then "
          f"analyse.py --experiment {args.experiment}.")


if __name__ == "__main__":
    main()
