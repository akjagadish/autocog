"""Domain-general LLM-as-judge for surfaced theories.

Runs six independent LLM judgements per surfaced theory in a run group:

    1. gt_family_match              (binary, requires --ground-truth)
    2. gt_algorithm_match              (binary, requires --ground-truth)
    3. theory_novelty_vs_seeds      (scalar)
    4. algo_novelty_vs_seeds        (scalar)
    5. joint_novelty_vs_seeds       (scalar)
    6. novelty_vs_domain_literature (scalar)

Output: <runs-dir>/judge_results.csv (one row per (run, slot)).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_final_theories import final_theories_for_run  # noqa: E402
from src.config import LLMConfig  # noqa: E402
from src.llm import LLMClient, make_client  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
import yaml  # noqa: E402


def majority_vote(decisions: list[bool]) -> tuple[bool, str]:
    """Collapse N binary judge samples into (majority, "k/N" tally).

    `k` is the count of True votes. The majority is True only on a STRICT
    majority (`2*k > N`); an even split resolves to False — the conservative
    call for a "does the surfaced theory match the ground truth" judgement.
    `n_votes=1` passes the single sample straight through.
    """
    n = len(decisions)
    if n == 0:
        raise ValueError("majority_vote requires at least one decision")
    k = sum(1 for d in decisions if d)
    return (2 * k > n, f"{k}/{n}")


class BinaryJudgeVerdict(BaseModel):
    matches: bool
    rationale: str


class NoveltyJudgeVerdict(BaseModel):
    novelty: float = Field(ge=0.0, le=1.0)
    rationale: str


class SeedTheory(BaseModel):
    label: str
    description: str
    predict_source: str


class SeedTheories(BaseModel):
    pi_1: SeedTheory
    pi_2: SeedTheory


def extract_seeds(run_dir: Path) -> SeedTheories:
    """Read pi_1 and pi_2 from round_000/theories.json of a single run."""
    theories_path = run_dir / "rounds" / "round_000" / "theories.json"
    if not theories_path.exists():
        raise FileNotFoundError(
            f"missing seed theories: {theories_path} (every run must have "
            "a round_000/theories.json with two starting_theories)"
        )
    data = json.loads(theories_path.read_text())
    starts = {t["slot"]: t for t in data.get("starting_theories", [])}
    if 1 not in starts or 2 not in starts:
        raise ValueError(
            f"{theories_path} starting_theories must contain slots 1 and 2"
        )

    def _seed(slot: int) -> SeedTheory:
        s = starts[slot]
        body = s.get("theory", {})
        return SeedTheory(
            label=s["label"],
            description=body.get("description", ""),
            predict_source=body.get("predict_source", ""),
        )

    return SeedTheories(pi_1=_seed(1), pi_2=_seed(2))


def validate_seed_invariant(run_dirs: list[Path]) -> SeedTheories:
    """Ensure every run in the group has identical pi_1 and pi_2.

    Returns the shared SeedTheories. Raises ValueError on mismatch with a
    message naming the offending run.
    """
    if not run_dirs:
        raise ValueError("validate_seed_invariant called with empty run list")
    first = extract_seeds(run_dirs[0])
    for run in run_dirs[1:]:
        seeds = extract_seeds(run)
        if seeds.pi_1 != first.pi_1 or seeds.pi_2 != first.pi_2:
            raise ValueError(
                f"seeds differ between {run_dirs[0].name} and {run.name}: "
                f"all runs in a group must share pi_1 and pi_2"
            )
    return first


JUDGEMENT_NAMES: tuple[str, ...] = (
    "gt_family_match",
    "gt_algorithm_match",
    "theory_novelty_vs_seeds",
    "algo_novelty_vs_seeds",
    "joint_novelty_vs_seeds",
    "novelty_vs_domain_literature",
)


class GroundTruthSpec(BaseModel):
    name: str
    description: str
    predict_source: str


class SurfacedTheory(BaseModel):
    label: str
    description: str
    predict_source: str


_BINARY_SYSTEM = {
    "gt_family_match": (
        "You are a renowned cognitive scientist evaluating cognitive models in a "
        "BINARY-cue decision task: every cue/feature value is binary (0 or 1), and "
        "models are compared by the CHOICES they produce over these binary stimuli. "
        "Decide whether the surfaced model belongs to the SAME FAMILY as the "
        "ground-truth heuristic, judged by its decision BEHAVIOR on binary cues. "
        "It is a match if, on binary stimuli, the surfaced model realizes the "
        "ground-truth's characteristic decision rule. This INCLUDES (a) constrained "
        "or special cases of the family, and more general models restricted to this "
        "behavior, and (b) formulations that are behaviorally identical on binary "
        "cues even if they differ in the general (non-binary) case. Crucially, "
        "because cues are 0/1, distinctions that VANISH under binary features do NOT "
        "make a different family. For example: summing cue values (Equal-Weight) is "
        "identical to counting feature-wise wins (Tallying) when cues are 0/1, so "
        "'retains magnitudes vs discards magnitudes' is vacuous here and is a match; "
        "and a validity-weighted-additive model with constrained weights (e.g. "
        "log-odds or rank-based) is still the WADD family even if it cannot express "
        "every free weight vector. Do NOT require the surfaced model to encompass or "
        "be able to express every parameterization of the ground-truth. Mark a "
        "NON-match only when the surfaced model implements a genuinely different "
        "decision rule that yields DIFFERENT choices on binary stimuli (e.g. a "
        "one-reason lexicographic rule vs a compensatory count, a least-valid-cue "
        "rule, a non-monotone/parity rule, or a contrarian/anti-majority rule). "
        "Reply with strict JSON matching {\"matches\": bool, \"rationale\": str}; "
        "rationale is one short sentence (<= 50 words)."
    ),
    # "gt_theory_match": (
    #     "You are a reknowned cognitive scientist evaluating cognitive models. Decide whether the surfaced "
    #     "model is behaviorally equivalent to the ground-truth model — i.e. "
    #     "it would produce the same output on the same input, ignoring "
    #     "parameter values, softmax temperatures, and lapse rates. Reply "
    #     "with strict JSON matching {\"matches\": bool, \"rationale\": str}; "
    #     "rationale is one short sentence (<= 50 words)."
    # ),
    "gt_algorithm_match": (
        "You are a reknowned cognitive scientist evaluating cognitive models. Decide whether the surfaced "
        "model is algorithmically equivalent to the ground-truth model. "
        " Algorithmic equivalence means the surfaced and ground-truth model "
        "follow the same algorithmic structure and would produce the same output on the same input. "
        " It does have to folow the same implementation scheme as the ground-truth but must to ble able to implement the same algorithm."
        "Reply with strict JSON matching {\"matches\": bool, \"rationale\": str}; "
        "rationale is one short sentence (<= 50 words)."
    ),

}


_NOVELTY_ANCHOR = (
    "Score in [0, 1]: 0 = essentially identical in mechanism, 0.5 = a "
    "recognizable variant or hybrid with non-trivial deviation, 1 = a "
    "mechanism not present in the reference set. Be calibrated, not "
    "generous: most surfaced theories are minor variants. Use the full "
    "range; do not cluster near 0.5. Reply with strict JSON matching "
    "{\"novelty\": float, \"rationale\": str}; rationale is one short "
    "sentence (<= 50 words)."
)


_NOVELTY_SYSTEM = {
    "theory_novelty_vs_seeds": (
        "You are evaluating cognitive models. Score how novel the surfaced "
        "theory's *description* is relative to the two seed theories' "
        "descriptions. " + _NOVELTY_ANCHOR
    ),
    "algo_novelty_vs_seeds": (
        "You are evaluating cognitive models. Score how novel the surfaced "
        "theory's *predict() implementation* is relative to the two seed "
        "theories' predict() implementations. " + _NOVELTY_ANCHOR
    ),
    "joint_novelty_vs_seeds": (
        "You are evaluating cognitive models. Score how novel the surfaced "
        "theory is relative to the two seed theories, considering "
        "*description and predict() implementation jointly*. " + _NOVELTY_ANCHOR
    ),
    "novelty_vs_domain_literature": (
        "You are evaluating cognitive models. Score how novel the surfaced "
        "theory is relative to extant theories in the field of {domain}. "
        "Use only your own knowledge of the published literature in this "
        "domain. " + _NOVELTY_ANCHOR
    ),
}


def _gt_user_prompt(surfaced: SurfacedTheory, gt: GroundTruthSpec) -> str:
    return (
        f"# Ground-truth model: {gt.name}\n\n"
        f"**Description:**\n{gt.description.strip()}\n\n"
        f"**predict():**\n```python\n{gt.predict_source.strip()}\n```\n\n"
        f"---\n\n"
        f"# Surfaced model: {surfaced.label}\n\n"
        f"**Description:**\n{surfaced.description.strip()}\n\n"
        f"**predict():**\n```python\n{surfaced.predict_source.strip()}\n```\n"
    )


def _seeds_block(seeds: SeedTheories, mode: str) -> str:
    """Render seeds as theory-only, predict-only, or both."""
    parts: list[str] = []
    for label, seed in (("pi_1", seeds.pi_1), ("pi_2", seeds.pi_2)):
        parts.append(f"## Seed {label}")
        if mode in ("description", "joint"):
            parts.append(f"**Description:**\n{seed.description.strip()}")
        if mode in ("predict", "joint"):
            parts.append(
                f"**predict():**\n```python\n{seed.predict_source.strip()}\n```"
            )
    return "\n\n".join(parts)


def _surfaced_block(surfaced: SurfacedTheory, mode: str) -> str:
    parts = [f"# Surfaced theory: {surfaced.label}"]
    if mode in ("description", "joint"):
        parts.append(f"**Description:**\n{surfaced.description.strip()}")
    if mode in ("predict", "joint"):
        parts.append(
            f"**predict():**\n```python\n{surfaced.predict_source.strip()}\n```"
        )
    return "\n\n".join(parts)


def _novelty_user_prompt(
    judgement: str,
    domain: str,
    surfaced: SurfacedTheory,
    seeds: SeedTheories | None,
) -> str:
    if judgement == "theory_novelty_vs_seeds":
        assert seeds is not None
        return (
            f"{_seeds_block(seeds, 'description')}\n\n---\n\n"
            f"{_surfaced_block(surfaced, 'description')}"
        )
    if judgement == "algo_novelty_vs_seeds":
        assert seeds is not None
        return (
            f"{_seeds_block(seeds, 'predict')}\n\n---\n\n"
            f"{_surfaced_block(surfaced, 'predict')}"
        )
    if judgement == "joint_novelty_vs_seeds":
        assert seeds is not None
        return (
            f"{_seeds_block(seeds, 'joint')}\n\n---\n\n"
            f"{_surfaced_block(surfaced, 'joint')}"
        )
    if judgement == "novelty_vs_domain_literature":
        return (
            f"Domain: {domain}\n\n---\n\n"
            f"{_surfaced_block(surfaced, 'joint')}"
        )
    raise ValueError(f"unknown novelty judgement {judgement!r}")


def judge_binary(
    *,
    client: LLMClient,
    judgement: str,
    domain: str,
    surfaced: SurfacedTheory,
    ground_truth: GroundTruthSpec | None,
    seeds: SeedTheories | None,
) -> BinaryJudgeVerdict:
    if judgement not in _BINARY_SYSTEM:
        raise ValueError(f"not a binary judgement: {judgement!r}")
    if ground_truth is None:
        raise ValueError(f"{judgement} requires ground truth")
    system = _BINARY_SYSTEM[judgement]
    user = _gt_user_prompt(surfaced, ground_truth)
    result = client.chat(
        messages=[{"role": "user", "content": user}],
        system=system,
        response_schema=BinaryJudgeVerdict,
    )
    if isinstance(result.parsed, BinaryJudgeVerdict):
        return result.parsed
    return BinaryJudgeVerdict.model_validate_json(result.text)


def judge_novelty(
    *,
    client: LLMClient,
    judgement: str,
    domain: str,
    surfaced: SurfacedTheory,
    seeds: SeedTheories | None,
    ground_truth: GroundTruthSpec | None,
) -> NoveltyJudgeVerdict:
    if judgement not in _NOVELTY_SYSTEM:
        raise ValueError(f"not a novelty judgement: {judgement!r}")
    if judgement.endswith("_vs_seeds") and seeds is None:
        raise ValueError(f"{judgement} requires seeds")
    system = _NOVELTY_SYSTEM[judgement]
    if "{domain}" in system:
        system = system.replace("{domain}", domain)
    user = _novelty_user_prompt(judgement, domain, surfaced, seeds)
    result = client.chat(
        messages=[{"role": "user", "content": user}],
        system=system,
        response_schema=NoveltyJudgeVerdict,
    )
    if isinstance(result.parsed, NoveltyJudgeVerdict):
        return result.parsed
    return NoveltyJudgeVerdict.model_validate_json(result.text)


def judge_surfaced_theory(
    *,
    client: LLMClient,
    domain: str,
    surfaced: SurfacedTheory,
    seeds: SeedTheories,
    ground_truth: GroundTruthSpec | None,
    n_votes: int = 1,
    gt_only: bool = False,
) -> dict[str, str]:
    """Run all applicable judgements and return CSV-ready string fields.

    Each binary GT judgement is sampled `n_votes` times and reduced via
    `majority_vote`; the stored `<name>` is the majority boolean, `<name>_votes`
    is the "k/N" tally, and `<name>_rationale` is taken from the first sample on
    the winning side (or the first sample if no side won). `gt_only=True` skips
    the four novelty judgements (this analysis only needs family/algorithm).
    """
    out: dict[str, str] = {name: "" for name in JUDGEMENT_NAMES}
    for name in JUDGEMENT_NAMES:
        out[f"{name}_rationale"] = ""
    out["gt_family_match_votes"] = ""
    out["gt_algorithm_match_votes"] = ""

    if ground_truth is not None:
        for name in ("gt_family_match", "gt_algorithm_match"):
            samples = [
                judge_binary(
                    client=client,
                    judgement=name,
                    domain=domain,
                    surfaced=surfaced,
                    ground_truth=ground_truth,
                    seeds=seeds,
                )
                for _ in range(n_votes)
            ]
            decision, tally = majority_vote([s.matches for s in samples])
            winning = next(
                (s for s in samples if s.matches is decision), samples[0]
            )
            out[name] = str(decision)
            out[f"{name}_votes"] = tally
            out[f"{name}_rationale"] = winning.rationale

    if not gt_only:
        for name in (
            "theory_novelty_vs_seeds",
            "algo_novelty_vs_seeds",
            "joint_novelty_vs_seeds",
            "novelty_vs_domain_literature",
        ):
            v = judge_novelty(
                client=client,
                judgement=name,
                domain=domain,
                surfaced=surfaced,
                seeds=seeds,
                ground_truth=ground_truth,
            )
            out[name] = f"{v.novelty:.3f}"
            out[f"{name}_rationale"] = v.rationale

    return out


CSV_FIELDNAMES: list[str] = [
    "domain", "run_group", "run", "run_dir", "slot", "label",
    "gt_family_match", "gt_family_match_votes", "gt_family_match_rationale",
    "gt_algorithm_match", "gt_algorithm_match_votes", "gt_algorithm_match_rationale",
    "theory_novelty_vs_seeds", "theory_novelty_vs_seeds_rationale",
    "algo_novelty_vs_seeds", "algo_novelty_vs_seeds_rationale",
    "joint_novelty_vs_seeds", "joint_novelty_vs_seeds_rationale",
    "novelty_vs_domain_literature", "novelty_vs_domain_literature_rationale",
]


_RUN_RE = re.compile(r"_?run(\d+)$")


def _parse_run_index(run_dir: Path) -> str:
    m = _RUN_RE.search(run_dir.name)
    return m.group(1) if m else run_dir.name


def load_ground_truth(
    *,
    theories_root: Path,
    domain: str,
    name: str,
) -> GroundTruthSpec:
    path = theories_root / domain / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"ground truth not found: {path}")
    data = yaml.safe_load(path.read_text())
    return GroundTruthSpec(
        name=data.get("name", name),
        description=data["theory"].strip(),
        predict_source=data["predict"].strip(),
    )


def judge_run_group(
    *,
    runs_dir: Path,
    domain: str,
    ground_truth: GroundTruthSpec | None,
    client: LLMClient,
    skip_existing: bool,
    n_votes: int = 1,
    gt_only: bool = False,
    out_name: str = "judge_results.csv",
) -> Path:
    run_dirs = sorted(
        d for d in runs_dir.iterdir() if d.is_dir() and (d / "rounds").is_dir()
    )
    if not run_dirs:
        raise ValueError(f"no run subdirectories under {runs_dir}")

    seeds = validate_seed_invariant(run_dirs)

    out_path = runs_dir / out_name
    existing_rows: list[dict[str, str]] = []
    existing_keys: set[tuple[str, str]] = set()
    if skip_existing and out_path.exists():
        existing_rows = list(csv.DictReader(out_path.open()))
        existing_keys = {(r["run"], r["slot"]) for r in existing_rows}

    rows: list[dict[str, str]] = list(existing_rows)
    for run in run_dirs:
        run_idx = _parse_run_index(run)
        try:
            theories = final_theories_for_run(run)
        except (FileNotFoundError, IndexError, ValueError, KeyError) as exc:
            print(
                f"  SKIP {run.name}: incomplete run -- {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        if not theories:
            print(f"  SKIP {run.name}: no final theories found", flush=True)
            continue
        for t in theories:
            slot = str(t.get("slot", ""))
            if (run_idx, slot) in existing_keys:
                print(f"  skip-existing run={run_idx} slot={slot}", flush=True)
                continue
            body = t.get("theory", {})
            surfaced = SurfacedTheory(
                label=str(t.get("label", "")),
                description=body.get("description", ""),
                predict_source=body.get("predict_source", ""),
            )
            verdicts = judge_surfaced_theory(
                client=client,
                domain=domain,
                surfaced=surfaced,
                seeds=seeds,
                ground_truth=ground_truth,
                n_votes=n_votes,
                gt_only=gt_only,
            )
            row = {k: "" for k in CSV_FIELDNAMES}
            row.update(verdicts)
            row.update({
                "domain": domain,
                "run_group": runs_dir.name,
                "run": run_idx,
                "run_dir": run.name,
                "slot": slot,
                "label": surfaced.label,
            })
            rows.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True,
                        help="Directory whose subdirectories are runs sharing "
                        "the same seed theories at round_000.")
    parser.add_argument("--domain", required=True,
                        help="Domain name, e.g. heuristic_decision_making, "
                        "category_learning. Used in prompts and CSV.")
    parser.add_argument("--ground-truth", default=None,
                        help="Optional. If set, looks up theories/<domain>/"
                        "<ground-truth>.yaml and runs gt_family_match + "
                        "gt_algorithm_match in addition to the four novelty "
                        "judgements.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (run, slot) rows already present in "
                        "judge_results.csv from a prior run.")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--n-votes", type=int, default=1,
        help="Sample each binary GT judgement this many times and store the "
        "majority verdict + a 'k/N' tally. Default 1 = single judgement.",
    )
    parser.add_argument(
        "--gt-only", action="store_true",
        help="Run only the two GT judgements (family/algorithm), skipping the "
        "four novelty judgements.",
    )
    parser.add_argument(
        "--out-name", default="judge_results.csv",
        help="Filename written under --runs-dir. Use a distinct name to avoid "
        "overwriting an existing judge_results.csv.",
    )
    args = parser.parse_args(argv)

    if args.n_votes < 1:
        parser.error(f"--n-votes must be >= 1; got {args.n_votes}")

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        parser.error(f"runs-dir does not exist: {runs_dir}")

    gt: GroundTruthSpec | None = None
    if args.ground_truth:
        gt = load_ground_truth(
            theories_root=ROOT / "theories",
            domain=args.domain,
            name=args.ground_truth,
        )

    client = make_client(LLMConfig(provider=args.provider, model=args.model))
    out = judge_run_group(
        runs_dir=runs_dir,
        domain=args.domain,
        ground_truth=gt,
        client=client,
        skip_existing=args.skip_existing,
        n_votes=args.n_votes,
        gt_only=args.gt_only,
        out_name=args.out_name,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
