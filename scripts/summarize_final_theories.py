"""Summarize the last two surfaced theories (and their predict source) per run.

A "run" is any subdirectory of the meta-directory that contains a `rounds/`
folder. The final surfaced theories for a run come from the last round's
`theories.json`: the two slots in `starting_theories`, with any `replacement`
overriding the slot it targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def final_theories_for_run(run_dir: Path) -> list[dict]:
    rounds_dir = run_dir / "rounds"
    round_dirs = sorted(
        d for d in rounds_dir.iterdir() if d.is_dir() and d.name.startswith("round_")
    )
    # Use the last COMPLETE round: an interrupted run can leave trailing round
    # dirs that never got a `theories.json` written. Scan newest->oldest and
    # take the first round whose theories.json exists, rather than crashing on
    # a half-written stub.
    last = next(
        (d for d in reversed(round_dirs) if (d / "theories.json").is_file()),
        None,
    )
    if last is None:
        raise FileNotFoundError(
            f"no round with a theories.json under {rounds_dir}"
        )
    data = json.loads((last / "theories.json").read_text())

    by_slot: dict[int, dict] = {t["slot"]: t for t in data.get("starting_theories", [])}
    replacement = data.get("replacement")
    if replacement and "slot" in replacement:
        by_slot[replacement["slot"]] = replacement

    return [by_slot[s] for s in sorted(by_slot)]


def _format_run(run_dir: Path, index: int) -> list[str]:
    theories = final_theories_for_run(run_dir)
    lines = [f"## run {index}: {run_dir.name}", ""]
    for t in theories:
        label = t.get("label", "?")
        slot = t.get("slot", "?")
        theory = t.get("theory", {})
        desc = theory.get("description", "").strip()
        predict = theory.get("predict_source", "").rstrip()
        lines.append(f"### surfaced theory {slot} — `{label}`")
        lines.append("")
        lines.append(desc)
        lines.append("")
        lines.append("```python")
        lines.append(predict)
        lines.append("```")
        lines.append("")
    return lines


def summarize_meta_dir(meta_dir: Path) -> Path:
    runs = sorted(
        d for d in meta_dir.iterdir() if d.is_dir() and (d / "rounds").is_dir()
    )
    lines = [f"# Final surfaced theories — {meta_dir.name}", ""]
    for i, run in enumerate(runs, start=1):
        lines.extend(_format_run(run, i))

    out = meta_dir / "final_theories.md"
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meta_dir", type=Path, help="Directory containing run subdirectories.")
    args = parser.parse_args()
    out = summarize_meta_dir(args.meta_dir.resolve())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
