"""Aggregate per-run `hilibig_battery.py` outputs into one long CSV.

Walks all run directories matching

    results/<family>/<noise>/hdm_ground_truth_<family>_<noise>_<llm>_run<id>/

reads each run's

    analysis/hilibig_battery/distance_to_ground_truth.csv
    analysis/hilibig_battery/choice_agreement.csv

and emits one long-format CSV with `ground_truth`, `noise`, `run_id`,
`llm`, and the per-run metrics side by side. One row per (run × model).

Default output: `results/hilibig_battery_aggregate.csv`.

CLI:
    python scripts/aggregate_hilibig_battery.py \\
        [--results-root results] \\
        [--families ttb wadd tallying] \\
        [--out results/hilibig_battery_aggregate.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# Parse LLM + run_id from the run-dir basename. Everything between the
# known prefix `hdm_ground_truth_<family>_<noise>_` and the trailing
# `_run<id>` is the LLM identifier — captured non-greedily so a run id
# that happens to contain digits doesn't eat the llm tag.
_RUN_DIR_RE = re.compile(
    r"^hdm_ground_truth_(?P<family>[^_]+)_(?P<noise>noise=[0-9.]+)_"
    r"(?P<llm>.+)_run(?P<run_id>\d+)$"
)


@dataclass(frozen=True)
class RunKey:
    family: str
    noise: str   # e.g. "noise=0.3"
    llm: str
    run_id: int


def iter_run_dirs(
    results_root: Path, families: list[str],
) -> list[tuple[RunKey, Path]]:
    """Enumerate run-dirs as (RunKey, path), sorted deterministically."""
    found: list[tuple[RunKey, Path]] = []
    for family in families:
        family_root = results_root / family
        if not family_root.is_dir():
            continue
        for noise_dir in sorted(family_root.glob("noise=*")):
            if not noise_dir.is_dir():
                continue
            for run_dir in sorted(noise_dir.glob(f"hdm_ground_truth_{family}_*_run*")):
                if not run_dir.is_dir():
                    continue
                m = _RUN_DIR_RE.match(run_dir.name)
                if not m:
                    print(
                        f"[aggregate] skipping unparseable dir: {run_dir}",
                        file=sys.stderr,
                    )
                    continue
                if m["family"] != family:
                    # Belt-and-suspenders: shouldn't happen given the glob.
                    continue
                found.append((
                    RunKey(
                        family=m["family"],
                        noise=m["noise"],
                        llm=m["llm"],
                        run_id=int(m["run_id"]),
                    ),
                    run_dir,
                ))
    return found


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _coerce_float(s: str) -> float | str:
    """Convert a CSV cell to float; leave it as-is if that fails
    (so semicolon-joined per-subject arrays pass through unchanged)."""
    try:
        return float(s)
    except ValueError:
        return s


def collect_one_run(
    run_dir: Path,
) -> list[dict[str, object]] | None:
    """Merge `distance_to_ground_truth.csv` + `choice_agreement.csv` from a
    single run. Returns a list of per-model rows, or None if either file
    is missing (in which case we skip the run with a warning)."""
    dist_path = run_dir / "analysis" / "hilibig_battery" / "distance_to_ground_truth.csv"
    agree_path = run_dir / "analysis" / "hilibig_battery" / "choice_agreement.csv"

    if not dist_path.is_file():
        print(f"[aggregate] missing {dist_path} — skipping run", file=sys.stderr)
        return None

    dist_rows = _read_csv_rows(dist_path)
    agree_rows = (
        {r["model"]: r for r in _read_csv_rows(agree_path)}
        if agree_path.is_file()
        else {}
    )
    if not agree_path.is_file():
        print(
            f"[aggregate] note: {agree_path} not found — agreement columns "
            f"will be blank for this run",
            file=sys.stderr,
        )

    merged: list[dict[str, object]] = []
    for r in dist_rows:
        llm_label = r["model"]
        a = agree_rows.get(llm_label, {})
        merged.append({
            # The aggregator rewrites `model` to be positional (see
            # slot-assignment loop in `aggregate`) so it's stable across
            # runs even when the LLM's surfaced labels differ
            # (pi_3 here, pi_5 there, pi_7 elsewhere). The LLM's original
            # label is preserved as `llm_label` for provenance.
            "llm_label": llm_label,
            "role": r["role"],
            "n_subjects": int(r["n_subjects"]),
            "mae_mean": _coerce_float(r["mae_mean"]),
            "mae_sem": _coerce_float(r["mae_sem"]),
            "mse_mean": _coerce_float(r["mse_mean"]),
            "mse_sem": _coerce_float(r["mse_sem"]),
            "agreement_mean": _coerce_float(a.get("agreement_mean", "")),
            "agreement_sem": _coerce_float(a.get("agreement_sem", "")),
            "mae_subjects": r.get("mae_subjects", ""),
            "mse_subjects": r.get("mse_subjects", ""),
            "agreement_subjects": a.get("agreement_subjects", ""),
        })
    return merged


OUTPUT_COLUMNS: list[str] = [
    "ground_truth", "noise", "run_id", "llm",
    "role", "slot", "model", "llm_label", "n_subjects",
    "mae_mean", "mae_sem", "mse_mean", "mse_sem",
    "agreement_mean", "agreement_sem",
    "mae_subjects", "mse_subjects", "agreement_subjects",
]


def aggregate(
    results_root: Path, families: list[str], out_path: Path,
) -> tuple[int, int]:
    """Walk the tree and write the aggregated CSV.

    Returns (n_rows_written, n_runs_included).
    """
    runs = iter_run_dirs(results_root, families)
    if not runs:
        print(
            f"[aggregate] no run-dirs found under {results_root} "
            f"for families={families}",
            file=sys.stderr,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_runs = 0
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for key, run_dir in runs:
            rows = collect_one_run(run_dir)
            if rows is None:
                continue
            n_runs += 1
            noise_val = key.noise.removeprefix("noise=")
            # `slot` is the 1-based ordinal of each row within (run, role), in
            # the order the per-run CSV listed them. `model = pi_{slot}`
            # gives a stable positional label across runs even when the LLM
            # rotates its pi_N labels (base is always pi_1 / pi_2; surfaced
            # may be pi_3 here, pi_5 there — we re-index those to pi_1 / pi_2
            # within the surfaced role). Callers who need the original LLM
            # label can read `llm_label`.
            slot_counter: dict[str, int] = {}
            for r in rows:
                role = r["role"]
                slot_counter[role] = slot_counter.get(role, 0) + 1
                slot = slot_counter[role]
                w.writerow({
                    "ground_truth": key.family,
                    "noise": noise_val,
                    "run_id": key.run_id,
                    "llm": key.llm,
                    "slot": slot,
                    "model": f"pi_{slot}",
                    **r,
                })
                n_rows += 1
    return n_rows, n_runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate per-run hilibig_battery outputs into one long CSV "
            "keyed by (ground_truth, noise, run_id, model)."
        )
    )
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument(
        "--families", nargs="+",
        default=["ttb", "wadd", "tallying"],
        help="Ground-truth families to include.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("results/hilibig_battery_aggregate.csv"),
    )
    args = p.parse_args(argv)

    n_rows, n_runs = aggregate(args.results_root, args.families, args.out)
    print(
        f"[aggregate] wrote {args.out}: {n_rows} rows across {n_runs} runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
