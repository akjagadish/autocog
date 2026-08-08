#!/usr/bin/env bash
#
# Per-subject model preference + within-subject consistency for an experiment.
#
#   ./run_preference.sh            # exp1 (default)
#   ./run_preference.sh exp2
#
# Reports, from the current data/<exp>/trials.csv: each subject's match rate to
# concave / WADD / tallying / TTB, how many subjects prefer each (argmax),
# within-subject variance (consistency across stimulus repeats) and whether it
# differs by preferred model, and the between-subject spread of the match rates.

set -euo pipefail
cd "$(dirname "$0")"             # repo root

EXP="${1:-exp1}"
uv run python scripts/preregistration/model_preference.py --experiment "$EXP"
