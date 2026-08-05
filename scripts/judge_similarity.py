"""Scalar mechanism-similarity LLM judge for surfaced theories.

One job: compare a surfaced theory against the ground-truth theory and return a
single similarity score in [0, 1] rating how similar their underlying decision
MECHANISM is. Two input modes control what the judge sees per theory:

    --input-mode description   only the prose descriptions
    --input-mode joint         description AND predict() source, paired

Run it twice (once per mode, with distinct --out-name) to compare how much the
code changes the score. This is the only LLM-judge analysis reported in the
paper (mechanism similarity, Figure 3B); the earlier binary family/algorithm
match rubric has been removed.

Output: <runs-dir>/<out-name> (one row per (run, slot)).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from scripts.summarize_final_theories import final_theories_for_run  # noqa: E402
from src.config import LLMConfig  # noqa: E402
from src.llm import LLMClient, make_client  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

INPUT_MODES: tuple[str, ...] = ("description", "joint")


# The two theory records and the ground-truth loader below used to live in
# `judge_runs.py`, which ran the binary family/algorithm-match rubric. That
# analysis is no longer reported — the paper reports mechanism similarity only
# — so `judge_runs.py` was removed and these three definitions moved here
# unchanged, leaving this script self-contained.
class GroundTruthSpec(BaseModel):
    name: str
    description: str
    predict_source: str


class SurfacedTheory(BaseModel):
    label: str
    description: str
    predict_source: str


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


class SimilarityVerdict(BaseModel):
    similarity: float = Field(ge=0.0, le=1.0)
    rationale: str


SIMILARITY_SYSTEM = (
    "You are a cognitive scientist comparing two cognitive models in a "
    "decision-making task. Rate how similar their UNDERLYING DECISION "
    "MECHANISM is -- the rule each model uses to turn inputs into a choice -- "
    "ignoring wording, terminology, framing, and parameter values. Score in "
    "[0, 1]: 1.0 = the same decision rule (they would make the same choices "
    "for the same reason); 0.5 = a recognizable variant or partial overlap "
    "(shares core structure but deviates non-trivially); 0.0 = an unrelated "
    "mechanism (a genuinely different rule). Use the full range and "
    "intermediate values; be calibrated, not generous. Reply with strict JSON "
    "matching {\"similarity\": float, \"rationale\": str}; rationale is one "
    "short sentence (<= 50 words)."
)


def _theory_block(title: str, name: str, description: str, predict_source: str,
                  input_mode: str) -> str:
    parts = [f"# {title}: {name}", f"**Description:**\n{description.strip()}"]
    if input_mode == "joint":
        parts.append(f"**predict():**\n```python\n{predict_source.strip()}\n```")
    return "\n\n".join(parts)


def similarity_user_prompt(
    surfaced: SurfacedTheory, gt: GroundTruthSpec, input_mode: str
) -> str:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unknown input_mode {input_mode!r}; expected one of {INPUT_MODES}")
    gt_block = _theory_block(
        "Ground-truth theory", gt.name, gt.description, gt.predict_source, input_mode
    )
    surfaced_block = _theory_block(
        "Surfaced theory", surfaced.label, surfaced.description,
        surfaced.predict_source, input_mode,
    )
    return f"{gt_block}\n\n---\n\n{surfaced_block}"


def judge_similarity_once(
    *,
    client: LLMClient,
    surfaced: SurfacedTheory,
    ground_truth: GroundTruthSpec,
    input_mode: str,
) -> SimilarityVerdict:
    user = similarity_user_prompt(surfaced, ground_truth, input_mode)
    result = client.chat(
        messages=[{"role": "user", "content": user}],
        system=SIMILARITY_SYSTEM,
        response_schema=SimilarityVerdict,
    )
    if isinstance(result.parsed, SimilarityVerdict):
        return result.parsed
    return SimilarityVerdict.model_validate_json(result.text)


def judge_theory_similarity(
    *,
    client: LLMClient,
    surfaced: SurfacedTheory,
    ground_truth: GroundTruthSpec,
    input_mode: str,
    n_votes: int = 1,
) -> dict[str, str]:
    """Sample the score `n_votes` times and reduce to the arithmetic mean.

    Stores the mean as `similarity`, the raw samples (in order) as a
    ";"-joined `similarity_votes` string, and the first sample's rationale as
    `similarity_rationale`.
    """
    if n_votes < 1:
        raise ValueError(f"n_votes must be >= 1; got {n_votes}")
    samples = [
        judge_similarity_once(
            client=client,
            surfaced=surfaced,
            ground_truth=ground_truth,
            input_mode=input_mode,
        )
        for _ in range(n_votes)
    ]
    scores = [s.similarity for s in samples]
    mean = sum(scores) / len(scores)
    return {
        "similarity": f"{mean:.3f}",
        "similarity_votes": ";".join(f"{s:.3f}" for s in scores),
        "similarity_rationale": samples[0].rationale,
    }


CSV_FIELDNAMES: list[str] = [
    "domain", "run_group", "run", "run_dir", "slot", "label", "input_mode",
    "similarity", "similarity_votes", "similarity_rationale",
]


_RUN_RE = re.compile(r"_?run(\d+)$")


def _parse_run_index(run_dir: Path) -> str:
    m = _RUN_RE.search(run_dir.name)
    return m.group(1) if m else run_dir.name


def judge_run_group(
    *,
    runs_dir: Path,
    domain: str,
    ground_truth: GroundTruthSpec,
    client: LLMClient,
    input_mode: str,
    n_votes: int = 1,
    skip_existing: bool = False,
    out_name: str = "judge_similarity.csv",
) -> Path:
    run_dirs = sorted(
        d for d in runs_dir.iterdir() if d.is_dir() and (d / "rounds").is_dir()
    )
    if not run_dirs:
        raise ValueError(f"no run subdirectories under {runs_dir}")

    out_path = runs_dir / out_name
    existing_rows: list[dict[str, str]] = []
    existing_keys: set[tuple[str, str]] = set()
    if skip_existing and out_path.exists():
        existing_rows = list(csv.DictReader(out_path.open()))
        # Key on input_mode too: the same (run, slot) judged under a different
        # mode is distinct work, so a 'description' row must not skip a 'joint'
        # pass into the same out-name (and vice versa).
        existing_keys = {
            (r["run"], r["slot"], r.get("input_mode", "")) for r in existing_rows
        }

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
            if (run_idx, slot, input_mode) in existing_keys:
                print(f"  skip-existing run={run_idx} slot={slot} mode={input_mode}", flush=True)
                continue
            body = t.get("theory", {})
            surfaced = SurfacedTheory(
                label=str(t.get("label", "")),
                description=body.get("description", ""),
                predict_source=body.get("predict_source", ""),
            )
            verdict = judge_theory_similarity(
                client=client,
                surfaced=surfaced,
                ground_truth=ground_truth,
                input_mode=input_mode,
                n_votes=n_votes,
            )
            row = {k: "" for k in CSV_FIELDNAMES}
            row.update(verdict)
            row.update({
                "domain": domain,
                "run_group": runs_dir.name,
                "run": run_idx,
                "run_dir": run.name,
                "slot": slot,
                "label": surfaced.label,
                "input_mode": input_mode,
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
                        help="Directory whose subdirectories are runs to judge.")
    parser.add_argument("--domain", required=True,
                        help="Domain name, e.g. heuristic_decision_making.")
    parser.add_argument("--ground-truth", required=True,
                        help="Looks up theories/<domain>/<ground-truth>.yaml.")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="description",
                        help="'description' shows only prose; 'joint' also shows "
                        "predict() source for each theory.")
    parser.add_argument("--n-votes", type=int, default=1,
                        help="Sample the score this many times and store the mean "
                        "plus a ';'-joined tally of the raw samples.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (run, slot) rows already present in --out-name.")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--out-name", default="judge_similarity.csv",
                        help="Filename written under --runs-dir.")
    args = parser.parse_args(argv)

    if args.n_votes < 1:
        parser.error(f"--n-votes must be >= 1; got {args.n_votes}")

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        parser.error(f"runs-dir does not exist: {runs_dir}")

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
        input_mode=args.input_mode,
        n_votes=args.n_votes,
        skip_existing=args.skip_existing,
        out_name=args.out_name,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
