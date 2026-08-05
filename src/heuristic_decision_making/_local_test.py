"""Local one-subject smoke test for HeuristicDecisionMakingExperiment.

Two-step workflow:

1. MODE = "build" — write `experiment.html` into this folder. Open it in a
   browser, complete the task, and the data downloads as DATA_DOWNLOAD_NAME
   into your browser's default download folder.
2. MODE = "load" — load the downloaded data file (DATA_FILE) and print the
   canonical DataFrame produced by HeuristicDecisionMakingExperiment._observations_to_df.

Not part of the test suite; intended to be run by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.heuristic_decision_making.experiment import HeuristicDecisionMakingExperiment

# === edit me ================================================================

MODE = "load"  # "build" or "load"

HTML_PATH = Path(__file__).with_name("experiment.html")
DATA_DOWNLOAD_NAME = "heuristic_decision_making_data.json"
DATA_FILE = Path("~/Downloads/heuristic_decision_making_data.json").expanduser()

EXPERIMENT_KWARGS: dict = dict(
    trial_pairs=[
        ([1, 1, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 1, 1], [1, 1, 1, 0]),
        ([1, 1, 1, 0], [0, 0, 0, 1]),
    ],
    rationale="Local smoke test: 4 pairs that dissociate TTB (uses feature 1) from EQW (sums features).",
)

# ============================================================================


def build() -> None:
    exp = HeuristicDecisionMakingExperiment(**EXPERIMENT_KWARGS)
    js, _ = exp.build_online_experiment()
    js.to_html(str(HTML_PATH), path_local_download=DATA_DOWNLOAD_NAME)
    print(f"Wrote {HTML_PATH}")
    print(
        f"Open the HTML in a browser, complete the task, and the data will "
        f"download as {DATA_DOWNLOAD_NAME!r} (typically into ~/Downloads/). "
        f"Then flip MODE to 'load' and re-run."
    )


def load() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(
            f"Data file not found: {DATA_FILE}. Run MODE='build' first, "
            f"complete the experiment in a browser, and move the downloaded "
            f"file to {DATA_FILE} (or update DATA_FILE)."
        )
    raw = json.loads(DATA_FILE.read_text())
    trials = raw if isinstance(raw, list) else raw.get("trials", [])
    wrapped = json.dumps({"trials": trials})
    df = HeuristicDecisionMakingExperiment._observations_to_df([wrapped])
    print(df.to_string())
    print(f"\nLoaded {len(df)} rows from {DATA_FILE}")


if __name__ == "__main__":
    {"build": build, "load": load}[MODE]()
