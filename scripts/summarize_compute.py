"""Sum LLM calls/tokens for a run directory from its prompt logs.

Usage: python scripts/summarize_compute.py <run_dir> [<run_dir> ...]
Prints one JSON line per run dir. Token keys are summed from each log's
`## Usage` block, tolerating provider-specific key names (any key
containing 'token')."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

USAGE_RE = re.compile(r"## Usage\s+```json\s+(\{.*?\})\s+```", re.DOTALL)


def summarize_run(run_dir: Path) -> dict:
    totals: dict[str, int] = defaultdict(int)
    n_calls = 0
    for md in Path(run_dir).rglob("prompts/*.md"):
        match = USAGE_RE.search(md.read_text())
        if not match:
            continue
        n_calls += 1
        for k, v in json.loads(match.group(1)).items():
            if "token" in k and isinstance(v, (int, float)):
                totals[k] += int(v)
    return {"run_dir": str(run_dir), "n_llm_calls": n_calls, **totals}


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(summarize_run(Path(d))))
