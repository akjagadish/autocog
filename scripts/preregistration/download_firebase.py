"""Download collected sessions for one experiment straight from Firestore and
write them as ``data/<experiment>/raw_observations.json`` + ``conditions.json``,
ready for ``download_data.py`` to turn into trials.csv.

Why this exists: ``run_experiment.py`` overwrites raw_observations.json on every
launch, so earlier batches survive only in Firestore. The runner accumulates
EVERY submission (across all studies that ever used this Firebase project) under

    /autora/autora_out/data_all/<slot>_<epoch_ms>   -> {observation, condition}

This script reads that collection, keeps only the submissions that (a) are
complete (>= the experiment's trials_per_participant choice trials) and (b)
match THIS experiment's frozen battery (every option pair is one of this
experiment's stimuli for the doc's validity vector — which also excludes other
experiments and old studies), takes the most recent ``--last N`` of them, and
writes them out keyed by their unique Firestore doc id (slot_epoch_ms) as the
participant id, with the validity vector read from each doc's condition.

Usage:
  .venv/bin/python validation_online/download_firebase.py --experiment exp1 --last 49
  # then:
  .venv/bin/python validation_online/download_data.py --experiment exp1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

import analyse as A  # noqa: E402  (build_lookup + load_design)


def _ts(doc_id: str) -> int:
    try:
        return int(doc_id.split("_")[-1])
    except ValueError:
        return -1


def _choice_trials(observation):
    """Return the a/b choice trials (minimal fields) from one observation."""
    obs = observation
    if isinstance(obs, str):
        obs = json.loads(obs)
    trials = obs.get("trials", []) if isinstance(obs, dict) else []
    out = []
    for t in trials:
        r = str(t.get("bean_response", "")).lower()
        if r in ("a", "b"):
            out.append({"bean_response": r,
                        "bean_option_a": t.get("bean_option_a"),
                        "bean_option_b": t.get("bean_option_b"),
                        "bean_rt": t.get("bean_rt")})
    return out


def _vector_of(condition) -> str | None:
    """Pull the validity-vector id out of a data_all doc's condition field."""
    if not isinstance(condition, dict):
        return None
    inner = condition.get("condition", condition)
    ser = inner.get("serialized") if isinstance(inner, dict) else None
    if ser:
        try:
            return json.loads(ser).get("vector")
        except (json.JSONDecodeError, AttributeError):
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["exp1", "exp2"], required=True)
    ap.add_argument("--last", type=int, required=True,
                    help="keep the most recent N matching+complete submissions")
    ap.add_argument("--credentials",
                    default=os.path.join(REPO_ROOT, "firebase_credentials.json"))
    ap.add_argument("--out_dir", default=None,
                    help="default: data/<experiment>/")
    args = ap.parse_args()

    design = A.load_design()
    lookup = A.build_lookup(design)
    exp_components = set(design["experiments"][args.experiment]["components"])
    need = design["experiments"][args.experiment]["trials_per_participant"]

    import firebase_admin
    from firebase_admin import credentials, firestore
    cred = credentials.Certificate(json.load(open(args.credentials)))
    try:
        app = firebase_admin.initialize_app(cred, name="dl")
    except ValueError:
        app = firebase_admin.get_app("dl")
    db = firestore.client(app)
    col = (db.collection("autora").document("autora_out").collection("data_all"))

    doc_ids = sorted((d.id for d in col.list_documents()), key=_ts, reverse=True)
    print(f"data_all has {len(doc_ids)} docs; scanning newest-first for "
          f"{args.experiment} (components={sorted(exp_components)}) ...")

    kept: list[dict] = []
    conditions: dict[str, str] = {}
    scanned = 0
    for did in doc_ids:
        if len(kept) >= args.last:
            break
        scanned += 1
        data = col.document(did).get().to_dict() or {}
        vec = _vector_of(data.get("condition"))
        if vec is None:
            continue
        trials = _choice_trials(data.get("observation"))
        if len(trials) < need:
            continue
        # every option pair must be one of THIS experiment's stimuli for `vec`
        ok = True
        for t in trials:
            a, b = t["bean_option_a"], t["bean_option_b"]
            if a is None or b is None:
                ok = False; break
            rec = lookup.get((vec, tuple(int(x) for x in a),
                              tuple(int(x) for x in b)))
            if rec is None or rec["component"] not in exp_components:
                ok = False; break
        if not ok:
            continue
        kept.append({"obs": {"trials": trials},
                     "prolific_pid": did, "slot_key": did})
        conditions[did] = vec

    if len(kept) < args.last:
        print(f"WARNING: only found {len(kept)} matching complete {args.experiment} "
              f"submissions (asked for {args.last}).", file=sys.stderr)

    out_dir = args.out_dir or os.path.join(HERE, "data", args.experiment)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "raw_observations.json"), "w") as f:
        json.dump(kept, f)
    with open(os.path.join(out_dir, "conditions.json"), "w") as f:
        json.dump(conditions, f, indent=1)

    import collections
    per_vec = collections.Counter(conditions.values())
    print(f"scanned {scanned} docs; kept {len(kept)} {args.experiment} "
          f"submissions -> {out_dir}/raw_observations.json + conditions.json")
    print(f"per validity vector: {dict(sorted(per_vec.items()))}")
    print(f"next: .venv/bin/python validation_online/download_data.py "
          f"--experiment {args.experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
