# autopi

**Automated cognitive scientist — an adversarial theory-debate framework.** The researcher
provides two seed theories (and their executable model instantiations) for a cognitive
domain. Each theory is owned by an LLM agent that steelmans the opponent, proposes
experiments + computable metrics where its model should win, validates them on simulated
data, runs them on (real or simulated) humans, interprets the results, and an arbiter
picks a winner and guides synthesis of a new theory. See [pi.md](pi.md) for the full
design and [docs/](docs/) for prompts and concerns.

> This is **research code**. The priority is accuracy and readability over performance.
> See [.claude/CLAUDE.md](.claude/CLAUDE.md) for working conventions (TDD, analytical
> tests, "never commit on my behalf", etc.).

---

## 1. Get the code (with submodule)

`vendor/sweetbean` is a git submodule. Clone with it, or initialise after the fact:

```bash
git clone --recurse-submodules <repo-url>
# or, in an existing clone:
git submodule update --init --recursive
```

## 2. Environment

The project is managed with **`uv`** (Python `>=3.11`; the committed `.venv` uses 3.12).
A prebuilt `.venv/` may already be present. To build/reproduce it from scratch:

```bash
uv sync                              # core deps from pyproject.toml + uv.lock
uv pip install -r requirements.txt   # adds anthropic, openai, torch, autora* (git)
```

`uv sync` alone is **not enough** — `requirements.txt` carries the LLM SDKs and the
AutoRA stack (and pulls in `matplotlib`/`scikit-learn` transitively). The `git+https`
deps in `requirements.txt` (`auto-prompt`, `sweetbean`, `autora*`) track their `@main`
branch; re-run `uv pip install -r requirements.txt` to pull newer upstream commits.

Run everything through the venv: `uv run python ...` or `source .venv/bin/activate`.

> **Note on `pip install -e .`:** it does *not* give you an importable `autopi` package.
> The code lives in `src/` and is imported as `src.*`, so commands are run **from the
> repository root** (the root is on `sys.path`). The `autopi*`/`domains*` packages named
> in `pyproject.toml` do not exist yet — that declaration is aspirational.

## 3. Secrets (API keys)

LLM calls read keys from a `.env` file at the repo root (loaded via `python-dotenv`).
It is git-ignored, so it never ships with the repo — create it yourself:

```bash
GEMINI_API_KEY=...      # default provider (google-genai); google-genai also accepts GOOGLE_API_KEY
ANTHROPIC_API_KEY=...   # provider=anthropic
OPENAI_API_KEY=...      # provider=openai / princeton sandbox
# AI_SANDBOX_KEY=...     # only for provider=princeton (Princeton AI Sandbox)
```

**No key needed for a dry run:** every entry point accepts `--llm_provider mock`, which
uses a canned client and makes the full pipeline runnable offline for free. Use this to
verify the harness before spending tokens. Supported providers: `gemini` (default),
`anthropic`, `princeton`, `mock` (see [src/llm.py](src/llm.py)).

## 4. Run the tests

Tests live in `tests/` and add the repo root to `sys.path` via
[tests/conftest.py](tests/conftest.py), so run `pytest` **from the repo root**:

```bash
uv run pytest -q                              # full suite
uv run pytest -q tests/test_jsd.py            # one file
uv run pytest -q -k jsd                        # by keyword
uv run pytest -q --continue-on-collection-errors   # skip the stale files (see below)
```

Most tests are offline unit tests; some integration tests (`test_propose_live.py`,
`test_anthropic_client.py`, ...) need API keys / network. **Known issue:** 13 test files
(e.g. `test_gcm.py`, `test_domains.py`, `test_sample_parameters.py`) fail at collection
because they import `domains`/`autopi` packages that don't exist in the current tree —
they belong to an unfinished refactor. Use `--continue-on-collection-errors` to run the
~382 collectable tests.

## 5. Run the framework

Entry points are the root `main*.py` scripts (argparse-driven, run from root):

| Script | Domain / purpose |
| --- | --- |
| `main.py` | category learning debate (GCM / RULEX / SUSTAIN) |
| `main_ablation_binary.py` | decision-making (binary cues) — ablations & controls |
| `main_decision_making_binary.py` | decision-making (binary cues), full pipeline |
| `main_heuristic_decision_making.py` | heuristic decision making |
| `main_*_centaur.py` / `main_*_online.py` | Centaur-simulated / live online (Firebase+Prolific) variants |

Examples (start with `mock` to check the harness for free):

```bash
# Category learning, free dry run
uv run python main.py --ground_truth gcm --n_rounds 1 --llm_provider mock

# Decision-making ablation, one condition, free dry run
uv run python main_ablation_binary.py \
  --condition baseline --ground_truth ttb_sampling \
  --n_rounds 1 --llm_provider mock

# Real run (billed): default provider is gemini / gemini-3.1-pro-preview
uv run python main_ablation_binary.py --condition baseline --ground_truth ttb_sampling --n_rounds 5
```

Key `main_ablation_binary.py` flags: `--condition {baseline,jsd_metric,neutral_proposer,blind_design}`,
`--ground_truth` (canonical `*_sampling` theories + non-canonical stress-test baselines),
`--n_rounds`, `--gt_epsilon`/`--gt_seed` (ground-truth action noise), `--design_seed`
(blind_design), `--run_id`, `--out_path`. Run any script with `-h` for the full list.

### Batch wrappers (`run_*.sh`)

`run_ablation_smoke.sh`, `run_blind_design.sh`, `run_online_*.sh`, etc. sweep ground
truths × seeds, write one log per run under `logs/`, and call a summary/scoring script
at the end (e.g. `score_blind_design.sh` → `scripts/recovery_correlation.py`).

> **Gotcha for fresh clones / cloud agents:** these scripts begin with
> `source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate`, a machine-specific
> venv path. Edit that line to `source .venv/bin/activate` (or your venv) before running
> them anywhere else. Most accept `LLM_PROVIDER=mock` for a free harness check.

## 6. Repository map

```text
main*.py                 entry points (run from repo root)
run_*.sh / score_*.sh    batch sweep + scoring wrappers
src/                     the framework (imported as src.*)
  pi.py                    AutoPi orchestrator (debate loop)
  theory.py, theory_generator.py, improver.py, arbiter.py
  llm.py                   provider clients (gemini/anthropic/princeton/mock)
  controls.py, ablations.py, jsd.py   ablation / control variants & metrics
  experiment.py, observation.py, metric.py, feedback.py
  category_learning/  decision_making_binary_features/  heuristic_decision_making/
  prompts/                 prompt templates
theories/                seed theory YAMLs (category_learning/, heuristic_decision_making/)
configs/                 run configs (default.yaml, category_learning.yaml, mock.yaml)
scripts/                 analysis & plotting (plot_*.py, summarize_compute.py, recovery_*.py)
tests/                   pytest suite (conftest puts root on sys.path)
results/  logs/          run outputs and per-run logs (git-ignored)
docs/  pi.md             design notes, base prompts, framework concerns
vendor/sweetbean         git submodule
```

## 7. Conventions

- **Run from the repo root** so `src.*` and the seed YAMLs resolve.
- **Don't commit on the maintainer's behalf** — propose changes for review (see
  [.claude/CLAUDE.md](.claude/CLAUDE.md)).
- **TDD with analytical tests**: prefer tests against a known closed-form value over
  ordering/sanity bounds.
- Python style: `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE`
  constants. Plotting entry points are named `plot_*.py`.
- **Check the logs** under `logs/` after any batch run before trusting results.
