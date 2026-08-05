"""Quantify how often the ground-truth heuristic is discovered across runs.

Directory layout assumed:
    results/{heuristic}/noise={x}/{run_dir}/rounds/round_NNN/theories.json

For each run we look at the final round's two surfaced theories (applying any
replacement to starting_theories by slot). An LLM judge is then asked, for
each surfaced theory, whether it implements the same algorithm as the
ground-truth heuristic (loaded from theories/heuristic_decision_making/*.yaml).
Results are written as CSV — one row per (run, slot).

Usage
-----
Analyze one heuristic/noise cell:
    uv run python scripts/analyze_ground_truth_discovery.py \\
        results/tallying/noise=0.3 --ground-truth tallying --noise 0.3

Analyze every cell under results/ in one go:
    uv run python scripts/analyze_ground_truth_discovery.py --sweep results/
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_final_theories import final_theories_for_run  # noqa: E402
from src.config import LLMConfig  # noqa: E402
from src.llm import LLMClient, make_client  # noqa: E402

THEORIES_DIR = ROOT / "theories" / "heuristic_decision_making"
MIN_RUNS_PER_CELL = 3


def _load_ground_truths() -> dict[str, dict[str, str]]:
    specs: dict[str, dict[str, str]] = {}
    for name in ("tallying", "ttb", "wadd"):
        path = THEORIES_DIR / f"{name}.yaml"
        data = yaml.safe_load(path.read_text())
        specs[name] = {
            "name": data.get("name", name),
            "description": data["theory"].strip(),
            "predict": data["predict"].strip(),
        }
    return specs


GROUND_TRUTHS = _load_ground_truths()


class JudgeVerdict(BaseModel):
    matches: bool
    family_match: bool
    rationale: str


_FAMILY_DEFINITIONS = (
    "Heuristic families (for the looser `family_match` signal):\n"
    "- **Tallying family**: models that combine cues by *counting or summing "
    "feature-wise comparisons with unit-ish weights* — e.g. Tallying (Dawes), "
    "Equal-Weight, Majority Voting, Lexicographic Tallying, any variant where "
    "each cue contributes equally (or nearly equally) to an additive score.\n"
    "- **TTB family**: models that apply a *lexicographic single-cue stopping "
    "rule* — consult cues in some order (validity-sorted, arbitrary, learned), "
    "stop at the first discriminating cue, ignore the rest. Any one-reason "
    "decision rule belongs here.\n"
    "- **WADD family**: models that compute a *weighted sum of cue values* "
    "where the weights are non-uniform and reflect some notion of cue "
    "importance (validities, regression weights, learned weights). Linear "
    "compensatory integration with heterogeneous weights.\n"
)


_JUDGE_SYSTEM = (
    "You are a careful evaluator of cognitive-model implementations. "
    "You will produce two signals per surfaced model:\n"
    "1. `matches` — whether the surfaced model is *behaviorally equivalent* "
    "to the ground-truth heuristic. Judge by algorithmic content (the "
    "decision rule and how cues are combined), not by cosmetic naming. "
    "Ignore softmax temperatures, lapse rates, and parameter ranges. If the "
    "surfaced model adds extra structure (e.g. lexicographic framing on top "
    "of tallying) but collapses to the same choice function, call it a "
    "match. If it uses a fundamentally different rule, call it a non-match.\n"
    "2. `family_match` — whether the surfaced model belongs to the same "
    "*algorithmic family* as the ground truth, regardless of exact "
    "behavioral equivalence. A tallying-family variant with unusual weights "
    "is a family match but not an exact match. `matches=True` implies "
    "`family_match=True`.\n\n"
    + _FAMILY_DEFINITIONS
    + "\nKeep `rationale` to one short sentence (<= 40 words) covering both "
    "signals."
)


def _judge_prompt(
    ground_truth_name: str,
    ground_truth_spec: dict[str, str],
    surfaced_description: str,
    surfaced_predict: str,
) -> str:
    return (
        f"# Ground-truth heuristic: {ground_truth_spec['name']} "
        f"(internal label: {ground_truth_name})\n\n"
        f"**Description:**\n{ground_truth_spec['description']}\n\n"
        f"**predict():**\n```python\n{ground_truth_spec['predict']}\n```\n\n"
        f"---\n\n"
        f"# Surfaced model\n\n"
        f"**Description:**\n{surfaced_description.strip()}\n\n"
        f"**predict():**\n```python\n{surfaced_predict.strip()}\n```\n\n"
        f"---\n\n"
        f"Return strict JSON matching the schema "
        f"{{\"matches\": bool, \"family_match\": bool, \"rationale\": str}}. "
        f"`matches` is true iff the surfaced model implements the same "
        f"decision rule as the ground-truth heuristic (behavioral "
        f"equivalence). `family_match` is true iff the surfaced model "
        f"belongs to the same algorithmic family as the ground truth "
        f"(tallying / ttb / wadd family per the definitions above); it must "
        f"be true whenever `matches` is true. `rationale` is one short "
        f"sentence (<= 40 words)."
    )


def judge_theory(
    *,
    client: LLMClient,
    ground_truth_name: str,
    ground_truth_spec: dict[str, str],
    surfaced_description: str,
    surfaced_predict: str,
) -> JudgeVerdict:
    prompt = _judge_prompt(
        ground_truth_name,
        ground_truth_spec,
        surfaced_description,
        surfaced_predict,
    )
    result = client.chat(
        messages=[{"role": "user", "content": prompt}],
        system=_JUDGE_SYSTEM,
        response_schema=JudgeVerdict,
    )
    if isinstance(result.parsed, JudgeVerdict):
        return result.parsed
    return JudgeVerdict.model_validate_json(result.text)


_NOISE_RE = re.compile(r"noise=([0-9.]+)")
_RUN_RE = re.compile(r"_run(\d+)$")


def _parse_run_index(run_dir: Path) -> str:
    m = _RUN_RE.search(run_dir.name)
    return m.group(1) if m else run_dir.name


def analyze_meta_dir(
    *,
    meta_dir: Path,
    ground_truth: str,
    noise: str,
    client: LLMClient,
) -> Path:
    if ground_truth not in GROUND_TRUTHS:
        raise ValueError(
            f"unknown ground_truth {ground_truth!r}; "
            f"known: {sorted(GROUND_TRUTHS)}"
        )
    spec = GROUND_TRUTHS[ground_truth]

    runs = sorted(
        d for d in meta_dir.iterdir() if d.is_dir() and (d / "rounds").is_dir()
    )

    rows: list[dict[str, str]] = []
    for run in runs:
        run_idx = _parse_run_index(run)
        try:
            theories = final_theories_for_run(run)
        except (FileNotFoundError, IndexError, ValueError, KeyError) as exc:
            print(
                f"  SKIP {run.name}: incomplete run — {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        if not theories:
            print(f"  SKIP {run.name}: no final theories found", flush=True)
            continue
        per_slot_rows: list[dict[str, str]] = []
        any_match = False
        any_family_match = False
        for t in theories:
            theory_body = t.get("theory", {})
            verdict = judge_theory(
                client=client,
                ground_truth_name=ground_truth,
                ground_truth_spec=spec,
                surfaced_description=theory_body.get("description", ""),
                surfaced_predict=theory_body.get("predict_source", ""),
            )
            any_match = any_match or verdict.matches
            any_family_match = any_family_match or verdict.family_match
            per_slot_rows.append({
                "ground_truth": ground_truth,
                "noise": noise,
                "run": run_idx,
                "run_dir": run.name,
                "slot": str(t.get("slot", "")),
                "label": str(t.get("label", "")),
                "matches": str(verdict.matches),
                "family_match": str(verdict.family_match),
                "rationale": verdict.rationale,
            })
        for r in per_slot_rows:
            r["run_discovered"] = str(any_match)
            r["run_family_match"] = str(any_family_match)
        rows.extend(per_slot_rows)

    out = meta_dir / "ground_truth_discovery.csv"
    fieldnames = [
        "ground_truth", "noise", "run", "run_dir",
        "slot", "label",
        "matches", "family_match", "rationale",
        "run_discovered", "run_family_match",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _sweep(root: Path, client: LLMClient, skip_existing: bool = False) -> list[Path]:
    """Walk results/{heuristic}/noise=*/ under `root` and analyze each cell.

    When ``skip_existing`` is True, a cell whose ``ground_truth_discovery.csv``
    already exists is not re-analyzed but is still included in the returned
    outputs so the summary aggregate can pick it up.
    """
    outputs: list[Path] = []
    for gt_dir in sorted(root.iterdir()):
        if not gt_dir.is_dir() or gt_dir.name not in GROUND_TRUTHS:
            continue
        for noise_dir in sorted(gt_dir.iterdir()):
            if not noise_dir.is_dir():
                continue
            m = _NOISE_RE.match(noise_dir.name)
            if not m:
                continue
            existing = noise_dir / "ground_truth_discovery.csv"
            if skip_existing and existing.exists():
                print(
                    f"SKIP {gt_dir.name} / {noise_dir.name}: "
                    f"already analyzed ({existing.name} exists)",
                    flush=True,
                )
                outputs.append(existing)
                continue
            run_dirs = [
                d for d in noise_dir.iterdir() if d.is_dir() and (d / "rounds").is_dir()
            ]
            if len(run_dirs) < MIN_RUNS_PER_CELL:
                print(
                    f"SKIP {gt_dir.name} / {noise_dir.name}: "
                    f"only {len(run_dirs)} run(s), need >= {MIN_RUNS_PER_CELL}",
                    flush=True,
                )
                continue
            print(f"analyzing {gt_dir.name} / {noise_dir.name} ...")
            out = analyze_meta_dir(
                meta_dir=noise_dir,
                ground_truth=gt_dir.name,
                noise=m.group(1),
                client=client,
            )
            outputs.append(out)
            print(f"  wrote {out}")
    return outputs


def _aggregate(outputs: list[Path], dest: Path) -> None:
    """Produce a per-(gt,noise) summary with exact + family discovery rate."""
    from collections import defaultdict
    buckets: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    for csv_path in outputs:
        rows = list(csv.DictReader(csv_path.open()))
        seen_runs: set[str] = set()
        for r in rows:
            key = (r["ground_truth"], r["noise"])
            if r["run"] in seen_runs:
                continue
            seen_runs.add(r["run"])
            buckets[key].append((
                r["run_discovered"] == "True",
                r.get("run_family_match", "False") == "True",
            ))

    fieldnames = [
        "ground_truth", "noise", "n_runs",
        "n_discovered", "discovery_rate",
        "n_family_discovered", "family_discovery_rate",
    ]
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (gt, noise), outcomes in sorted(buckets.items()):
            n = len(outcomes)
            k = sum(exact for exact, _ in outcomes)
            kf = sum(fam for _, fam in outcomes)
            writer.writerow({
                "ground_truth": gt,
                "noise": noise,
                "n_runs": n,
                "n_discovered": k,
                "discovery_rate": f"{(k / n):.3f}" if n else "",
                "n_family_discovered": kf,
                "family_discovery_rate": f"{(kf / n):.3f}" if n else "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meta_dir",
        type=Path,
        nargs="?",
        help="Directory like results/tallying/noise=0.3/. Not needed with --sweep.",
    )
    parser.add_argument("--ground-truth", choices=sorted(GROUND_TRUTHS))
    parser.add_argument("--noise", help="Noise level label written into the CSV (e.g. 0.3).")
    parser.add_argument(
        "--sweep",
        type=Path,
        help="Walk results/{tallying,ttb,wadd}/noise=*/ under this root and "
        "analyze every cell.",
    )
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In --sweep, skip any cell whose ground_truth_discovery.csv "
        "already exists (it still feeds into the summary aggregate).",
    )
    args = parser.parse_args()

    client = make_client(LLMConfig(provider=args.provider, model=args.model))

    if args.sweep:
        outputs = _sweep(args.sweep.resolve(), client, skip_existing=args.skip_existing)
        summary = args.sweep.resolve() / "ground_truth_discovery_summary.csv"
        _aggregate(outputs, summary)
        print(f"wrote summary: {summary}")
        return

    if not args.meta_dir or not args.ground_truth or not args.noise:
        parser.error("meta_dir, --ground-truth and --noise are required unless --sweep is used")
    out = analyze_meta_dir(
        meta_dir=args.meta_dir.resolve(),
        ground_truth=args.ground_truth,
        noise=args.noise,
        client=client,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
