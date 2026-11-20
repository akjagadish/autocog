"""Compare the models an autocog run generated: complexity and predictions.

For every theory a run entertained (or just the surfaced ones), this script
reports three families of measures:

1. **Parameter complexity** — how many free parameters the theory declares.
   A parameter is *free* when, after resolving task context (validities,
   n_features, ...), it is still an interval `[lo, hi]` with lo != hi
   (continuous) or a set `{a, b, ...}` with more than one element
   (discrete). Context-pinned entries such as `validities: "validities"`
   are fixed, not free. `n_free_unused` counts free parameters whose name
   never appears as a string key in `predict_source` — declared but dead.

2. **Description length** — of the `predict` code (characters, logical
   lines, AST nodes, branch points, gzip-compressed bytes as a crude
   Kolmogorov proxy) and, separately, of the prose `description` (words,
   characters). Comment lines and docstrings are excluded from the
   line / AST counts so commenting style does not count as complexity.

3. **Prediction similarity** — each theory's P(choose B) on a common
   stimulus grid, under two parameter regimes:

     fit    maximum-likelihood parameters on the Hilbig & Moshagen (2014)
            Exp 1 stimulus pairs (binary cues, validities [0.9, 0.8, 0.7,
            0.6]). `--fit-target human` fits the pooled human choices (79
            participants); `--fit-target ground_truth` fits each recovery
            run's ground-truth model instead (its prior-predictive P(B) per
            pair, as binomial pseudo-counts — i.e. minimum KL), which is the
            right target for synthetic-recovery runs. Random search over the
            declared ranges, then bounded Nelder-Mead from the best starts;
            discrete parameters are enumerated.
     prior  the prior-predictive mean over `--n-prior-draws` uniform draws
            from the declared ranges — what the autocog loop itself
            simulates.

   Pairwise similarity between two theories' prediction vectors is
   reported as Pearson r, SSE, and MAE. The fit also yields NLL / AIC / BIC
   against the human data, and the prior draws yield `prior_pred_sd`
   (mean over stimuli of the SD of P(B) across draws) — a functional
   flexibility measure that complements the raw parameter count.

   Pairs are compared within a ground-truth family only (fits to different
   targets are not commensurable); canonical heuristics are compared with
   everything. `--all-pairs` lifts the restriction.

   As a fourth, prediction-free similarity, `code_ncd` is the normalized
   compression distance between two theories' predict sources (0 = same
   code, ~1 = nothing shared).

Predictions use an empty history, as `eval_hilbig.py` does. Theories that
read `history` are flagged `uses_history` — their predictions here are their
first-trial behaviour only.

Outputs (under `--out`, default `<run-dir>/analysis/compare_models/` for a
single run):

  models.csv           one row per model: complexity, fit, flexibility
  fit_params.json      best-fit parameter values per model
  predictions.csv      long: model, stimulus, p_b_fit, p_b_prior, p_b_human
  pairwise.csv         long: model_a, model_b, {fit,prior}_{pearson_r,sse,mae}, code_ncd
  group_summary.csv    per (family, llm): mean complexity / fit measures
  group_pairwise.csv   per (family, llm_a, llm_b): mean pairwise similarity
                       (llm_a == llm_b rows are within-LLM similarity)
  targets.csv          the fit target P(B) per (family, stimulus pair)
  similarity_*.png/svg heatmaps; complexity.png/svg bar panels
  llm_<llm>.png/svg    per discovery LLM: surfaced theories' predictions vs
                       the target, and fit / free parameters by cycle
  llm_trajectories.*   median fit MSE / free parameters / code lines by
                       cycle, one line per LLM, one column per family
  best_per_llm.*       the best-fitting non-seed theory of each LLM per
                       family (surfaced preferred): predictions vs target,
                       fit MSE, free parameters, code lines (+ .csv)

Redraw the per-LLM figures from saved outputs without refitting:

  python scripts/compare_models.py --replot <out-dir> --fit-target ground_truth

Run from the repo root:

  python scripts/compare_models.py \\
      --run-dir results/human_decision_making_binary/ttb+wadd --include-canonical

  # every discovery LLM on the single_cue recovery family
  python scripts/compare_models.py --fit-target ground_truth --workers 8 \
      --run-dir results/recovery*/single_cue/noise=0.0/*_run* \
      --out results/recovery_anthropic/analysis/compare_models/single_cue
"""

from __future__ import annotations

import argparse
import ast
import gzip
import io
import itertools
import json
import random
import re
import signal
import sys
import threading
import time
import tokenize
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sample_parameters import (  # noqa: E402
    draw_parameter_samples,
    resolve_parameters,
)
from src.theory import Theory  # noqa: E402
from scripts.eval_hilbig import (  # noqa: E402
    CANONICAL_YAML_DIR,
    HUMAN_DATA_DEFAULT,
    HUMAN_RATING_MAX,
    HUMAN_VALIDITIES,
    load_canonical_theories,
    _round_dirs,
    load_human_choices,
)

CANONICAL_NAMES: tuple[str, ...] = ("ttb", "wadd", "tallying", "ew")
P_CLIP: float = 1e-6
MAX_DISCRETE_COMBOS: int = 64


# ---------------------------------------------------------------------------
# Loading models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    run: str
    label: str
    role: str
    theory: Theory
    llm: str = ""
    family: str = ""
    noise: float | None = None
    replicate: str = ""
    generation: int | None = None


_RUN_NAME_RE = re.compile(
    r"ground_truth_(?P<family>.+?)_noise=(?P<noise>[\d.]+)_(?P<llm>.+)_run(?P<rep>[^_/]+)$"
)


def parse_run_dir(run_dir: Path) -> dict[str, Any]:
    """(family, noise, llm, replicate) from a recovery run-dir name; falls
    back to run_meta.json (human runs) and the directory name."""
    m = _RUN_NAME_RE.search(run_dir.name)
    if m:
        return {"family": m["family"], "noise": float(m["noise"]),
                "llm": m["llm"], "replicate": m["rep"]}
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return {"family": run_dir.name, "noise": None,
            "llm": meta.get("llm_model", "unknown"), "replicate": meta.get("run_id", "")}


def _run_name(run_dir: Path) -> str:
    try:
        return str(run_dir.resolve().relative_to(_REPO_ROOT / "results"))
    except ValueError:
        return run_dir.name


def run_theories(
    run_dir: Path, *, max_rounds: int | None = None,
) -> list[tuple[str, str, int, Theory]]:
    """Every theory a run entertained in its first `max_rounds` rounds, as
    (label, role, generation, theory). Mirrors
    `eval_hilbig.resolve_all_explored_theories`, plus truncation:

      role        base (round-0 seed, later killed) | intermediate (proposed,
                  later killed) | surfaced (alive after the last kept round)
      generation  0 for seeds; r+1 for the replacement written in round r
    """
    rounds = _round_dirs(run_dir)
    if max_rounds is not None:
        rounds = rounds[:max_rounds]
    theories: dict[str, tuple[int, Theory]] = {}
    data = {}
    for i, r in enumerate(rounds):
        data = json.loads((r / "theories.json").read_text())
        for s in data.get("starting_theories", []):
            theories.setdefault(s["label"], (i, Theory.model_validate(s["theory"])))
        repl = data.get("replacement")
        if repl is not None:
            theories.setdefault(repl["label"], (i + 1, Theory.model_validate(repl["theory"])))
    first = json.loads((rounds[0] / "theories.json").read_text())
    base = {s["label"] for s in first.get("starting_theories", [])}
    surfaced = {s["label"] for s in data.get("starting_theories", []) if not s.get("killed")}
    if data.get("replacement") is not None:
        surfaced.add(data["replacement"]["label"])
    out = []
    for lab, (gen, th) in theories.items():
        role = "surfaced" if lab in surfaced else "base" if lab in base else "intermediate"
        out.append((lab, role, 0 if lab in base else gen, th))
    out.sort(key=lambda t: _natural_key(t[0]))
    return out


def load_models(
    run_dirs: list[Path], *, which: str, include_canonical: bool,
    max_rounds: int | None = None,
) -> list[ModelEntry]:
    """Theories from each run (`which` = all | surfaced), plus the canonical
    heuristics if asked. `model_id` is the bare label for a single run and
    `<llm>/<family>/run<N>/<label>` when several runs are pooled."""
    out: list[ModelEntry] = []
    prefix = len(run_dirs) > 1
    for rd in run_dirs:
        run = _run_name(rd)
        info = parse_run_dir(rd)
        try:
            rows = run_theories(rd, max_rounds=max_rounds)
        except FileNotFoundError as e:
            print(f"  skipping {run}: {e}")
            continue
        short = f"{info['llm']}/{info['family']}/run{info['replicate']}"
        for lab, role, gen, th in rows:
            if which == "surfaced" and role != "surfaced":
                continue
            mid = f"{short}/{lab}" if prefix else lab
            out.append(ModelEntry(mid, run, lab, role, th, generation=gen, **info))
    if include_canonical:
        canon = load_canonical_theories()
        for name in CANONICAL_NAMES:
            if name in canon:
                out.append(ModelEntry(name, "canonical", name, "canonical", canon[name],
                                      llm="canonical", family="canonical"))
    return out


def _natural_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ---------------------------------------------------------------------------
# 1. Parameter complexity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeParam:
    name: str
    kind: str                   # "continuous" | "discrete"
    bounds: tuple[float, float] | None = None
    choices: tuple[Any, ...] | None = None


def task_context(
    validities: list[float] = HUMAN_VALIDITIES,
) -> dict[str, Any]:
    return {
        "n_features": len(validities),
        "n_labels": 2,
        "validities": list(validities),
        "rating_max": HUMAN_RATING_MAX,
    }


def parameter_space(
    theory: Theory, context: dict[str, Any],
) -> tuple[dict[str, Any], list[FreeParam]]:
    """Split a theory's declared parameters into (fixed values, free params).

    Only top-level keys can be free; a nested interval/set is drawn once
    and held fixed (none of the committed theories nest them)."""
    resolved = resolve_parameters(theory.parameters, context)
    fixed: dict[str, Any] = {}
    free: list[FreeParam] = []
    for name, v in resolved.items():
        if isinstance(v, dict) and set(v) == {"interval"}:
            lo, hi = sorted(float(x) for x in v["interval"])
            if lo != hi:
                free.append(FreeParam(name, "continuous", bounds=(lo, hi)))
                continue
            fixed[name] = lo
        elif isinstance(v, dict) and set(v) == {"choices"}:
            ch = tuple(v["choices"])
            if len(ch) > 1:
                free.append(FreeParam(name, "discrete", choices=ch))
                continue
            fixed[name] = ch[0]
        else:
            fixed[name] = draw_parameter_samples(v)
    # Context keys pinned regardless of how the theory declared them.
    for key in ("validities", "rating_max"):
        if key in fixed:
            fixed[key] = context[key]
    return fixed, free


def _param_used(name: str, source: str) -> bool:
    return bool(re.search(rf"""["']{re.escape(name)}["']""", source))


def parameter_metrics(theory: Theory, context: dict[str, Any]) -> dict[str, Any]:
    fixed, free = parameter_space(theory, context)
    n_cont = sum(p.kind == "continuous" for p in free)
    n_disc = sum(p.kind == "discrete" for p in free)
    return {
        "n_params_declared": len(theory.parameters),
        "n_free_params": len(free),
        "n_free_continuous": n_cont,
        "n_free_discrete": n_disc,
        "n_free_unused": sum(not _param_used(p.name, theory.predict_source) for p in free),
        "free_params": ";".join(p.name for p in free),
    }


# ---------------------------------------------------------------------------
# 2. Description length.
# ---------------------------------------------------------------------------

_BRANCH_NODES = (ast.If, ast.IfExp, ast.For, ast.While, ast.comprehension,
                 ast.BoolOp, ast.Try)


def _strip_comments_and_docstrings(source: str) -> str:
    """Source with comments and docstrings removed (round-trips via AST)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def code_metrics(source: str) -> dict[str, Any]:
    """Length measures of a predict source. `code_lines` / `code_ast_nodes` /
    `code_branches` ignore comments and docstrings; `code_tokens` counts
    lexical tokens excluding comments, newlines and indentation."""
    clean = _strip_comments_and_docstrings(source)
    tree = ast.parse(clean)
    toks = [
        t for t in tokenize.generate_tokens(io.StringIO(clean).readline)
        if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                          tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
    ]
    return {
        "code_chars_raw": len(source),
        "code_lines": sum(1 for ln in clean.splitlines() if ln.strip()),
        "code_tokens": len(toks),
        "code_ast_nodes": sum(1 for _ in ast.walk(tree)),
        "code_branches": sum(isinstance(n, _BRANCH_NODES) for n in ast.walk(tree)),
        "code_gzip_bytes": len(gzip.compress(clean.encode(), mtime=0)),
        "uses_history": _uses_history(tree),
    }


def _uses_history(tree: ast.AST) -> bool:
    """True when the predict body reads its `history` argument."""
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "predict":
            args = [a.arg for a in fn.args.args]
            if len(args) < 3:
                return False
            hname = args[2]
            return any(isinstance(n, ast.Name) and n.id == hname
                       for stmt in fn.body for n in ast.walk(stmt))
    return False


def description_metrics(text: str) -> dict[str, Any]:
    return {"desc_words": len(text.split()), "desc_chars": len(text)}


def ncd(x: str, y: str) -> float:
    """Normalized compression distance (Cilibrasi & Vitanyi 2005), gzip."""
    c = lambda s: len(gzip.compress(s.encode(), compresslevel=9, mtime=0))  # noqa: E731
    cx, cy, cxy = c(x), c(y), c(x + "\n" + y)
    return (cxy - min(cx, cy)) / max(cx, cy)


# ---------------------------------------------------------------------------
# 3. Predictions: fitting + prior predictive.
# ---------------------------------------------------------------------------

Pair = tuple[tuple[int, ...], tuple[int, ...]]


def _empty_history() -> dict[str, list[Any]]:
    return {"stimulus": [], "label": [], "previous_stimuli": [], "previous_labels": []}


def predict_p_b(theory: Theory, params: dict[str, Any], pairs: list[Pair]) -> np.ndarray:
    out = np.empty(len(pairs))
    for j, (a, b) in enumerate(pairs):
        stim = np.asarray([a, b], dtype=float)
        p = np.asarray(theory.predict(dict(params), stim, _empty_history()), dtype=float)
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            raise ValueError(f"bad prediction {p!r}")
        out[j] = p[1] / p.sum()
    return out


def binomial_nll(p_b: np.ndarray, n: np.ndarray, k: np.ndarray) -> float:
    p = np.clip(p_b, P_CLIP, 1 - P_CLIP)
    return float(-(k * np.log(p) + (n - k) * np.log1p(-p)).sum())


class _OutOfTime(BaseException):
    """Raised when a model's time budget runs out. A BaseException so the
    broad `except Exception` guards around LLM-written code do not eat it."""


class _Budget:
    """Wall-clock budget. `check()` is the soft deadline, polled between
    predict calls; a SIGALRM at 2x + 5 s is the hard backstop for a single
    predict call that never returns (main thread only). Leaving the `with`
    block swallows `_OutOfTime`; `expired` records whether it fired."""

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.end = None if seconds is None else time.monotonic() + seconds
        self.expired = False
        self._armed = False

    def check(self) -> None:
        if self.end is not None and time.monotonic() > self.end:
            raise _OutOfTime

    def __enter__(self) -> "_Budget":
        if (self.seconds is not None and hasattr(signal, "setitimer")
                and threading.current_thread() is threading.main_thread()):
            def _raise(*_):
                raise _OutOfTime
            self._old = signal.signal(signal.SIGALRM, _raise)
            signal.setitimer(signal.ITIMER_REAL, 2 * self.seconds + 5)
            self._armed = True
        return self

    def __exit__(self, et, ev, tb) -> bool:
        if self._armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
        if et is not None and issubclass(et, _OutOfTime):
            self.expired = True
            return True
        return False


@dataclass
class FitResult:
    params: dict[str, Any]
    nll: float
    n_free: int
    n_obs: int
    error: str | None = None
    real_counts: bool = True     # False for pseudo-count (ground-truth) targets
    truncated: bool = False      # time budget ran out; best-so-far reported

    @property
    def aic(self) -> float:
        return 2 * self.n_free + 2 * self.nll if self.real_counts else np.nan

    @property
    def bic(self) -> float:
        if not self.real_counts:
            return np.nan
        return self.n_free * np.log(self.n_obs) + 2 * self.nll


def fit_theory(
    theory: Theory,
    pairs: list[Pair],
    n: np.ndarray,
    k: np.ndarray,
    context: dict[str, Any],
    *,
    n_random: int = 200,
    n_starts: int = 3,
    seed: int = 0,
    time_budget: float | None = None,
) -> FitResult:
    """Maximum-likelihood fit of a theory's free parameters to binomial
    choice counts (`k` of `n` chose B on each pair). With `time_budget`
    (seconds) the search stops when it runs out and the best point seen so
    far is returned with `truncated=True`."""
    from scipy.optimize import minimize

    fixed, free = parameter_space(theory, context)
    cont = [p for p in free if p.kind == "continuous"]
    disc = [p for p in free if p.kind == "discrete"]
    n_obs = int(n.sum())
    rng = np.random.default_rng(seed)

    combos = list(itertools.product(*(p.choices for p in disc))) if disc else [()]
    if len(combos) > MAX_DISCRETE_COMBOS:
        idx = rng.choice(len(combos), MAX_DISCRETE_COMBOS, replace=False)
        combos = [combos[i] for i in sorted(idx)]

    lo = np.array([p.bounds[0] for p in cont])
    hi = np.array([p.bounds[1] for p in cont])

    def assemble(x: np.ndarray, combo: tuple) -> dict[str, Any]:
        params = dict(fixed)
        params.update({p.name: float(v) for p, v in zip(cont, x)})
        params.update({p.name: v for p, v in zip(disc, combo)})
        return params

    best: dict[str, Any] = {"x": None, "combo": None, "nll": np.inf}
    budget = _Budget(time_budget)

    def objective(x: np.ndarray, combo: tuple) -> float:
        budget.check()
        x = np.clip(x, lo, hi) if cont else x
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                v = binomial_nll(predict_p_b(theory, assemble(x, combo), pairs), n, k)
        except Exception:  # noqa: BLE001 — LLM-written code can raise anything
            v = np.inf
        if v < best["nll"]:
            best.update(x=np.array(x, dtype=float), combo=combo, nll=v)
        return v

    with budget:
        for combo in combos:
            if not cont:
                objective(np.array([]), combo)
                continue
            draws = lo + rng.random((n_random, len(cont))) * (hi - lo)
            vals = np.array([objective(x, combo) for x in draws])
            for i in np.argsort(vals)[:n_starts]:
                if np.isfinite(vals[i]):
                    minimize(objective, draws[i], args=(combo,), method="Nelder-Mead",
                             bounds=list(zip(lo, hi)),
                             options={"maxiter": 200 * len(cont), "xatol": 1e-4,
                                      "fatol": 1e-6})

    if best["x"] is None or not np.isfinite(best["nll"]):
        err = "time budget exhausted" if budget.expired else "no finite likelihood"
        with _Budget(time_budget):
            try:
                predict_p_b(theory, assemble(lo if cont else np.array([]), combos[0]), pairs)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        return FitResult({}, np.nan, len(free), n_obs, error=err, truncated=budget.expired)
    return FitResult(assemble(best["x"], best["combo"]), float(best["nll"]), len(free),
                     n_obs, truncated=budget.expired)


def prior_predictive(
    theory: Theory, pairs: list[Pair], context: dict[str, Any],
    *, n_draws: int, seed: int, time_budget: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of P(B) per pair over `n_draws` uniform parameter draws
    (fewer if `time_budget` runs out first)."""
    fixed, free = parameter_space(theory, context)
    random.seed(seed)
    rows = []
    with _Budget(time_budget) as budget:
        for _ in range(n_draws):
            budget.check()
            params = dict(fixed)
            for p in free:
                params[p.name] = (random.uniform(*p.bounds) if p.kind == "continuous"
                                  else random.choice(p.choices))
            try:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    rows.append(predict_p_b(theory, params, pairs))
            except Exception:  # noqa: BLE001
                pass
            if not free:
                break
    if not rows:
        nan = np.full(len(pairs), np.nan)
        return nan, nan
    arr = np.vstack(rows)
    return arr.mean(axis=0), arr.std(axis=0)


# ---------------------------------------------------------------------------
# Stimulus grids and human data.
# ---------------------------------------------------------------------------


def human_counts(path: Path) -> pd.DataFrame:
    df = load_human_choices(path)
    g = df.groupby(["option_a", "option_b"], sort=True)["choice"]
    out = g.agg(n="size", k="sum").reset_index()
    out["p_b_human"] = out["k"] / out["n"]
    return out


def full_grid(n_features: int) -> list[Pair]:
    """All ordered pairs of distinct binary cue profiles."""
    profiles = list(itertools.product((0, 1), repeat=n_features))
    return [(a, b) for a in profiles for b in profiles if a != b]


# ---------------------------------------------------------------------------
# Per-model worker (process pool friendly: takes plain dicts).
# ---------------------------------------------------------------------------


def _evaluate_one(job: dict[str, Any]) -> dict[str, Any]:
    t0 = time.monotonic()
    theory = Theory.model_validate(job["theory"])
    ctx = job["context"]
    budget = job.get("time_budget")
    row: dict[str, Any] = {"model": job["model_id"]}
    row.update(parameter_metrics(theory, ctx))
    row.update(code_metrics(theory.predict_source))
    row.update(description_metrics(theory.description))

    fit_pairs, grid = job["fit_pairs"], job["grid_pairs"]
    n, k = np.asarray(job["n"], dtype=float), np.asarray(job["k"], dtype=float)
    fit = fit_theory(theory, fit_pairs, n, k, ctx, n_random=job["n_random"],
                     n_starts=job["n_starts"], seed=job["seed"], time_budget=budget)
    fit.real_counts = job["real_counts"]

    fit_mse, p_fit = np.nan, np.full(len(grid), np.nan)
    if fit.error is None:
        with _Budget(budget), np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_mse = float(np.mean((predict_p_b(theory, fit.params, fit_pairs) - k / n) ** 2))
            p_fit = predict_p_b(theory, fit.params, grid)
    row.update({"nll": fit.nll, "aic": fit.aic, "bic": fit.bic, "fit_mse": fit_mse,
                "n_obs": fit.n_obs, "fit_error": fit.error, "fit_truncated": fit.truncated})

    p_prior, sd_prior = prior_predictive(theory, grid, ctx, n_draws=job["n_prior_draws"],
                                         seed=job["seed"], time_budget=budget)
    row["prior_pred_sd"] = float(np.nanmean(sd_prior)) if np.isfinite(sd_prior).any() else np.nan
    row["eval_seconds"] = round(time.monotonic() - t0, 2)
    return {"row": row, "params": _jsonable(fit.params),
            "p_fit": np.asarray(p_fit).tolist(), "p_prior": p_prior.tolist()}


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, v in d.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        out[key] = v
    return out


# ---------------------------------------------------------------------------
# Pairwise similarity.
# ---------------------------------------------------------------------------


def pairwise_prediction_similarity(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Pearson r, SSE, MAE between two prediction vectors (NaN-safe)."""
    m = np.isfinite(p) & np.isfinite(q)
    if not m.any():
        return {"pearson_r": np.nan, "sse": np.nan, "mae": np.nan}
    p, q = p[m], q[m]
    r = np.nan if p.std() == 0 or q.std() == 0 else float(np.corrcoef(p, q)[0, 1])
    d = p - q
    return {"pearson_r": r, "sse": float(d @ d), "mae": float(np.abs(d).mean())}


def comparable(a: ModelEntry, b: ModelEntry) -> bool:
    """Same ground-truth family, or either one is a canonical reference."""
    return a.family == b.family or "canonical" in (a.family, b.family)


def pairwise_table(
    models: list[ModelEntry], preds: dict[str, dict[str, np.ndarray]],
    *, all_pairs: bool = False,
) -> pd.DataFrame:
    clean = {m.model_id: _strip_comments_and_docstrings(m.theory.predict_source)
             for m in models}
    rows = []
    for a, b in itertools.combinations(models, 2):
        if not (all_pairs or comparable(a, b)):
            continue
        row: dict[str, Any] = {"model_a": a.model_id, "model_b": b.model_id}
        for mode in ("fit", "prior"):
            s = pairwise_prediction_similarity(preds[mode][a.model_id], preds[mode][b.model_id])
            row.update({f"{mode}_{k}": v for k, v in s.items()})
        row["code_ncd"] = ncd(clean[a.model_id], clean[b.model_id])
        rows.append(row)
    return pd.DataFrame(rows)


def square(pw: pd.DataFrame, ids: list[str], col: str, diag: float) -> pd.DataFrame:
    pos = {m: i for i, m in enumerate(ids)}
    arr = np.full((len(ids), len(ids)), np.nan)
    ia, ib = pw["model_a"].map(pos).to_numpy(), pw["model_b"].map(pos).to_numpy()
    arr[ia, ib] = arr[ib, ia] = pw[col].to_numpy(dtype=float)
    np.fill_diagonal(arr, diag)
    return pd.DataFrame(arr, index=ids, columns=ids)


SUMMARY_COLS: tuple[str, ...] = (
    "n_free_params", "n_free_unused", "code_lines", "code_tokens", "code_ast_nodes",
    "code_branches", "code_gzip_bytes", "desc_words", "fit_mse", "nll", "bic",
    "prior_pred_sd",
)


def group_summary(models_df: pd.DataFrame) -> pd.DataFrame:
    """Mean (and SEM) of each per-model measure by (family, llm, role)."""
    g = models_df.groupby(["family", "llm"])
    out = g.size().rename("n_models").to_frame()
    out["n_runs"] = g["run"].nunique()
    for col in SUMMARY_COLS:
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_sem"] = g[col].sem()
    return out.reset_index()


def group_pairwise(pw: pd.DataFrame, models_df: pd.DataFrame) -> pd.DataFrame:
    """Mean pairwise similarity per (family, llm_1, llm_2, same_run), with the
    LLM pair unordered. llm_1 == llm_2 rows are within-LLM similarity, split
    into pairs from the same run (lineage) and from different replicate runs.
    Seed theories (generation 0) are dropped: they are identical in every run,
    so they say nothing about the discovery LLM and inflate every cell."""
    info = models_df.set_index("model")[["llm", "family", "run", "generation"]]
    d = pw.join(info.add_suffix("_a"), on="model_a").join(info.add_suffix("_b"), on="model_b")
    d = d[(d["family_a"] == d["family_b"])
          & (d["generation_a"] > 0) & (d["generation_b"] > 0)].copy()
    swap = d["llm_a"] > d["llm_b"]
    d["llm_1"] = np.where(swap, d["llm_b"], d["llm_a"])
    d["llm_2"] = np.where(swap, d["llm_a"], d["llm_b"])
    d["same_run"] = d["run_a"] == d["run_b"]
    cols = [c for c in d.columns if c.startswith(("fit_", "prior_")) or c == "code_ncd"]
    out = (d.groupby(["family_a", "llm_1", "llm_2", "same_run"])[cols]
           .mean().join(d.groupby(["family_a", "llm_1", "llm_2", "same_run"]).size()
                        .rename("n_pairs")))
    return out.reset_index().rename(columns={"family_a": "family"})


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------


def plot_heatmap(mat: pd.DataFrame, *, title: str, cmap: str, vmin: float | None,
                 vmax: float | None, out_base: Path) -> None:
    import matplotlib.pyplot as plt
    from scripts.figure_style import FONTSIZE, save_figure

    n = len(mat)
    size = max(5.0, 0.38 * n + 2)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    im = ax.imshow(mat.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    if n <= 40:
        ax.set_xticks(range(n), mat.columns, rotation=90, fontsize=FONTSIZE - 4)
        ax.set_yticks(range(n), mat.index, fontsize=FONTSIZE - 4)
        if n <= 16:
            for i in range(n):
                for j in range(n):
                    v = mat.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    else:
        ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=FONTSIZE)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


def plot_complexity(models_df: pd.DataFrame, out_base: Path) -> None:
    import matplotlib.pyplot as plt
    from scripts.figure_style import ROLE_COLOR, GRAY, save_figure, style_axes

    panels = [("n_free_params", "free parameters"), ("code_ast_nodes", "predict AST nodes"),
              ("desc_words", "description words"), ("fit_mse", "best-fit MSE to target")]
    n = len(models_df)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), max(3.5, 0.28 * n + 1.5)),
                             sharey=True)
    colors = [ROLE_COLOR.get(r, GRAY) for r in models_df["role"]]
    y = np.arange(n)
    for ax, (col, lab) in zip(axes, panels):
        ax.barh(y, models_df[col], color=colors)
        style_axes(ax, xlabel=lab)
    axes[0].set_yticks(y, models_df["model"] if n <= 60 else [""] * n)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-LLM figures.
# ---------------------------------------------------------------------------

# Display names, in the fixed order that also fixes each LLM's color (the
# color follows the LLM, so it stays put when a figure omits some LLMs).
LLM_LABELS: dict[str, str] = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "claude-opus-5": "Opus 5",
    "gpt-5.6-sol": "GPT-5.6",
    "deepseek-deepseek-v4-pro-0813": "DeepSeek V4 Pro",
    "z-ai-glm-5.3": "GLM-5.3",
}
FAMILY_LABELS: dict[str, str] = {
    "single_cue": "Single cue", "anti_majority": "Anti-majority",
    "take_the_worst": "Take-the-worst",
}
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def llm_label(llm: str) -> str:
    return LLM_LABELS.get(llm, llm)


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


def _ordered(values: Any, known: dict[str, str]) -> list[str]:
    present = set(values)
    return [v for v in known if v in present] + sorted(present - set(known))


def llm_styles(llms: Any) -> dict[str, tuple[str, str]]:
    """(color, marker) per LLM. Known LLMs take the house categorical cycle in
    LLM_LABELS order; unknown ones follow in name order."""
    from scripts.figure_style import CYCLE

    import matplotlib

    extra = [matplotlib.colors.to_hex(c) for c in matplotlib.colormaps["tab10"].colors]
    slots = list(LLM_LABELS) + sorted(set(llms) - set(LLM_LABELS))
    palette = list(CYCLE) + extra
    return {llm: (palette[i % len(palette)], _MARKERS[i % len(_MARKERS)])
            for i, llm in enumerate(slots) if llm in set(llms)}


def best_per_llm(models_df: pd.DataFrame) -> pd.DataFrame:
    """The best-fitting (lowest fit_mse) model per (family, llm), excluding
    the generation-0 seeds, which every run shares. Surfaced models are
    preferred — they are what the run reports; if a run family has no
    surfaced non-seed model, the best non-seed model of any role is used."""
    d = models_df[(models_df["generation"] > 0) & models_df["fit_mse"].notna()
                  & (models_df["family"] != "canonical")]
    rows = []
    for _, g in d.groupby(["family", "llm"], sort=False):
        pool = g[g["role"] == "surfaced"]
        pool = g if pool.empty else pool
        rows.append(pool.loc[pool["fit_mse"].idxmin()])
    return pd.DataFrame(rows).reset_index(drop=True)


def llm_trajectory(models_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Median (and IQR) of `col` across runs per (family, llm, generation).
    Generation g >= 1 has one model per run (that cycle's replacement);
    generation 0 pools the seeds."""
    d = models_df[models_df[col].notna() & (models_df["family"] != "canonical")]
    g = d.groupby(["family", "llm", "generation"])[col]
    return g.agg(median="median", q25=lambda s: s.quantile(0.25),
                 q75=lambda s: s.quantile(0.75), n="size").reset_index()


def load_saved_outputs(
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """models.csv, predictions.csv and (if present) targets.csv from a
    previous run, keeping the binary stimulus codes ("0001") as strings."""
    codes = {"option_a": str, "option_b": str}
    models = pd.read_csv(out_dir / "models.csv")
    preds = pd.read_csv(out_dir / "predictions.csv", dtype=codes)
    tpath = out_dir / "targets.csv"
    targets = pd.read_csv(tpath, dtype=codes) if tpath.is_file() else None
    return models, preds, targets


def _slug(s: str) -> str:
    """Filename-safe, and dot-free: save_figure swaps the suffix, so a dot in
    "gemini-3.1-pro" would truncate the name."""
    return re.sub(r"[^\w-]+", "-", s)


def _curves(preds: pd.DataFrame, models: list[str],
            order: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    lut = preds.set_index(["model", "option_a", "option_b"])["p_b_fit"]
    lut = lut[~lut.index.duplicated()]
    out = {}
    for m in models:
        idx = pd.MultiIndex.from_tuples([(m, a, b) for a, b in order])
        out[m] = lut.reindex(idx).to_numpy(dtype=float)
    return out


def _target_orders(targets: pd.DataFrame) -> dict[str, tuple[list[tuple[str, str]], np.ndarray]]:
    """Per family: stimulus pairs sorted by the target P(B), and that target."""
    out = {}
    for fam, g in targets.groupby("family"):
        g = g.sort_values(["p_b_target", "option_a", "option_b"])
        out[fam] = (list(zip(g["option_a"], g["option_b"])), g["p_b_target"].to_numpy(float))
    return out


_TRAJ_PANELS = (("fit_mse", "Best-fit MSE"), ("n_free_params", "Free parameters"),
                ("code_lines", "Code lines"))


def _plot_llm_detail(models_df: pd.DataFrame, preds: pd.DataFrame,
                     orders: dict[str, Any], llm: str, style: tuple[str, str],
                     target_name: str, out_base: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.figure_style import FONTSIZE, INK, save_figure, style_axes

    color, marker = style
    d = models_df[models_df["llm"] == llm]
    fams = [f for f in _ordered(d["family"], FAMILY_LABELS) if f in orders]
    panels = (("pred", "P(choose B)"),) + _TRAJ_PANELS[:2]
    fig, axes = plt.subplots(len(fams), len(panels), figsize=(14, 3.4 * len(fams) + 0.9),
                             squeeze=False)
    for r, fam in enumerate(fams):
        df = d[d["family"] == fam]
        order, target = orders[fam]
        x = np.arange(len(order))
        ax = axes[r, 0]
        surf = df[(df["role"] == "surfaced") & (df["generation"] > 0)]
        for p in _curves(preds, surf["model"].tolist(), order).values():
            ax.plot(x, p, color=color, lw=1.2, alpha=0.6)
        ax.plot(x, target, color=INK, lw=2.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([])
        style_axes(ax, ylabel=f"{family_label(fam)}\nP(choose B)",
                   xlabel="stimulus pair (sorted by target)" if r == len(fams) - 1 else None,
                   title=f"Surfaced theories ({df['run'].nunique()} runs)" if r == 0 else None)
        for c, (col, lab) in enumerate(panels[1:], start=1):
            ax = axes[r, c]
            seeds = df[df["generation"] == 0]
            ax.scatter(seeds["generation"], seeds[col], facecolor="white", edgecolor=INK,
                       linewidth=1.2, s=40, zorder=3)
            for _, run in df[df["generation"] > 0].groupby("run"):
                run = run.sort_values("generation")
                ax.plot(run["generation"], run[col], color=color, lw=1, alpha=0.5,
                        marker=marker, ms=4)
            med = df[df["generation"] > 0].groupby("generation")[col].median()
            ax.plot(med.index, med.values, color=color, lw=2.5, marker=marker, ms=8,
                    markeredgecolor=INK, markeredgewidth=0.6, zorder=4)
            ax.set_xticks(range(0, int(df["generation"].max()) + 1))
            ax.set_ylim(bottom=0)
            style_axes(ax, title=lab if r == 0 else None,
                       xlabel="cycle" if r == len(fams) - 1 else None)
    handles = [
        Line2D([], [], color=INK, lw=2.5, label=target_name),
        Line2D([], [], color=color, lw=1.2, alpha=0.6, label="One theory / one run"),
        Line2D([], [], color=color, lw=2.5, marker=marker, ms=8, markeredgecolor=INK,
               markeredgewidth=0.6, label="Median across runs"),
        Line2D([], [], lw=0, marker="o", ms=6, markerfacecolor="white", markeredgecolor=INK,
               markeredgewidth=1.2, label="Seeds"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=FONTSIZE - 2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 1.0, llm_label(llm), fontsize=FONTSIZE + 2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    paths = save_figure(fig, out_base)
    plt.close(fig)
    return paths


def _plot_trajectories(models_df: pd.DataFrame, styles: dict[str, tuple[str, str]],
                       out_base: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.figure_style import FONTSIZE, INK, save_figure, style_axes

    d = models_df[models_df["family"] != "canonical"]
    fams = _ordered(d["family"], FAMILY_LABELS)
    llms = _ordered(d["llm"], LLM_LABELS)
    fig, axes = plt.subplots(len(_TRAJ_PANELS), len(fams), figsize=(4.6 * len(fams), 10.5),
                             squeeze=False, sharex=True)
    for r, (col, lab) in enumerate(_TRAJ_PANELS):
        t = llm_trajectory(d, col)
        for c, fam in enumerate(fams):
            ax = axes[r, c]
            for llm in llms:
                s = t[(t["family"] == fam) & (t["llm"] == llm)].sort_values("generation")
                if s.empty:
                    continue
                color, marker = styles[llm]
                ax.plot(s["generation"], s["median"], color=color, lw=2, marker=marker,
                        ms=7, markeredgecolor=INK, markeredgewidth=0.5)
            ax.set_ylim(bottom=0)
            style_axes(ax, title=family_label(fam) if r == 0 else None,
                       ylabel=f"{lab}\n(median across runs)" if c == 0 else None,
                       xlabel="cycle (0 = seeds)" if r == len(_TRAJ_PANELS) - 1 else None)
    handles = [Line2D([], [], color=styles[l][0], marker=styles[l][1], lw=2, ms=7,
                      markeredgecolor=INK, markeredgewidth=0.5, label=llm_label(l))
               for l in llms]
    fig.legend(handles=handles, loc="upper center", ncol=len(llms), frameon=False,
               fontsize=FONTSIZE - 2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    paths = save_figure(fig, out_base)
    plt.close(fig)
    return paths


def _fmt_mse(v: float) -> str:
    return "<0.001" if v < 0.0005 else f"{v:.3f}"


_BEST_BARS = (("fit_mse", "Best-fit MSE", _fmt_mse),
              ("n_free_params", "Free parameters", "{:.0f}".format),
              ("code_lines", "Code lines", "{:.0f}".format))


def _plot_best(best: pd.DataFrame, preds: pd.DataFrame, orders: dict[str, Any],
               styles: dict[str, tuple[str, str]], target_name: str,
               out_base: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.figure_style import FONTSIZE, INK, save_figure, style_axes

    fams = [f for f in _ordered(best["family"], FAMILY_LABELS) if f in orders]
    llms = _ordered(best["llm"], LLM_LABELS)
    nrow = 1 + len(_BEST_BARS)
    fig, axes = plt.subplots(nrow, len(fams), figsize=(5 * len(fams), 3.6 + 2.0 * len(_BEST_BARS)),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [1.8] + [1] * len(_BEST_BARS)})
    y = np.arange(len(llms))
    for c, fam in enumerate(fams):
        b = best[best["family"] == fam].set_index("llm")
        order, target = orders[fam]
        x = np.arange(len(order))
        ax = axes[0, c]
        ax.plot(x, target, color=INK, lw=3, zorder=1)
        curves = _curves(preds, b["model"].tolist(), order)
        for llm in llms:
            if llm in b.index:
                color, marker = styles[llm]
                ax.plot(x, curves[b.loc[llm, "model"]], color=color, lw=1.5, marker=marker,
                        ms=5, markeredgecolor=INK, markeredgewidth=0.4, zorder=2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([])
        style_axes(ax, title=family_label(fam), xlabel="stimulus pair (sorted by target)",
                   ylabel="P(choose B)" if c == 0 else None)
        for r, (col, lab, fmt) in enumerate(_BEST_BARS, start=1):
            ax = axes[r, c]
            vals = [b.loc[l, col] if l in b.index else np.nan for l in llms]
            ax.barh(y, vals, height=0.62, color=[styles[l][0] for l in llms],
                    edgecolor=INK, linewidth=0.4)
            for yi, v in zip(y, vals):
                if np.isfinite(v):
                    ax.text(v, yi, " " + fmt(v), va="center", ha="left",
                            fontsize=FONTSIZE - 4, color=INK)
                else:
                    ax.text(0, yi, " no runs", va="center", ha="left",
                            fontsize=FONTSIZE - 4, color="#666666")
            ax.set_yticks(y, [llm_label(l) for l in llms] if c == 0 else [""] * len(llms))
            ax.set_ylim(len(llms) - 0.5, -0.5)          # top-down, every LLM keeps its row
            ax.margins(x=0.22)
            style_axes(ax, xlabel=lab)
    # Same x scale per metric row, so families compare at a glance.
    for r in range(1, nrow):
        hi = max(a.get_xlim()[1] for a in axes[r])
        for a in axes[r]:
            a.set_xlim(0, hi)
    handles = [Line2D([], [], color=INK, lw=3, label=target_name)] + [
        Line2D([], [], color=styles[l][0], marker=styles[l][1], lw=1.5, ms=6,
               markeredgecolor=INK, markeredgewidth=0.4, label=llm_label(l)) for l in llms]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=FONTSIZE - 2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    paths = save_figure(fig, out_base)
    plt.close(fig)
    return paths


def plot_llm_figures(models_df: pd.DataFrame, preds: pd.DataFrame,
                     targets: pd.DataFrame, out: Path, *,
                     target_name: str = "Ground truth") -> list[Path]:
    """One detail figure per discovery LLM, a cross-LLM trajectory figure,
    and a best-model-per-LLM comparison (plus its table)."""
    d = models_df[models_df["family"] != "canonical"]
    if d.empty:
        return []
    orders = _target_orders(targets)
    styles = llm_styles(d["llm"].unique())
    written: list[Path] = []
    for llm in _ordered(d["llm"], LLM_LABELS):
        written += _plot_llm_detail(d, preds, orders, llm, styles[llm], target_name,
                                    out / f"llm_{_slug(llm)}")
    written += _plot_trajectories(d, styles, out / "llm_trajectories")
    best = best_per_llm(d)
    keep = ["family", "llm", "model", "role", "generation", "fit_mse", "n_free_params",
            "code_lines", "desc_words", "prior_pred_sd"]
    best[[c for c in keep if c in best.columns]].to_csv(out / "best_per_llm.csv", index=False)
    written.append(out / "best_per_llm.csv")
    written += _plot_best(best, preds, orders, styles, target_name, out / "best_per_llm")
    return written


# ---------------------------------------------------------------------------
# Fit targets.
# ---------------------------------------------------------------------------


def ground_truth_p_b(
    family: str, pairs: list[Pair], context: dict[str, Any], *, n_draws: int, seed: int,
) -> np.ndarray:
    """The ground-truth theory's prior-predictive P(B) per pair (exact
    expectation over parameter draws; no Bernoulli sampling noise)."""
    path = CANONICAL_YAML_DIR / f"{family}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no ground-truth YAML for family {family!r} at {path}")
    mean, _ = prior_predictive(Theory.from_yaml(path), pairs, context,
                               n_draws=n_draws, seed=seed)
    return mean


def fit_targets(
    families: Any, hc: pd.DataFrame, pairs: list[Pair],
    context: dict[str, Any], args: argparse.Namespace,
) -> dict[str, tuple[list[float], list[float]]]:
    """(n, k) per family. Human target: real trial counts, shared by all.
    Ground-truth target: `gt_pseudo_n` pseudo-trials per pair at the
    family's ground-truth P(B); canonical references are fit to humans."""
    human = (hc["n"].astype(float).tolist(), hc["k"].astype(float).tolist())
    out: dict[str, tuple[list[float], list[float]]] = {}
    for fam in sorted(set(families)):
        if args.fit_target == "human" or fam == "canonical":
            out[fam] = human
            continue
        p = ground_truth_p_b(fam, pairs, context, n_draws=args.gt_draws, seed=args.seed)
        n = np.full(len(pairs), args.gt_pseudo_n)
        out[fam] = (n.tolist(), (n * p).tolist())
    return out


def targets_table(targets: dict[str, tuple[list[float], list[float]]],
                  pairs: list[Pair]) -> pd.DataFrame:
    """Long table of the fit target P(B) = k / n per (family, stimulus pair)."""
    rows = []
    for fam, (n, k) in targets.items():
        for (a, b), ni, ki in zip(pairs, n, k):
            rows.append({"family": fam, "option_a": "".join(map(str, a)),
                         "option_b": "".join(map(str, b)), "p_b_target": ki / ni})
    return pd.DataFrame(rows)


def _target_name(fit_target: str) -> str:
    return "Ground truth" if fit_target == "ground_truth" else "Human"


def replot(args: argparse.Namespace) -> int:
    """Redraw the per-LLM figures from saved CSVs. Older output dirs have no
    targets.csv; it is then rebuilt from `--fit-target` (and saved)."""
    import matplotlib
    matplotlib.use("Agg")

    out = args.replot
    models_df, preds_df, targets_df = load_saved_outputs(out)
    if targets_df is None:
        hc = human_counts(args.human_data)
        pairs: list[Pair] = list(zip(hc["option_a"], hc["option_b"]))
        targets_df = targets_table(
            fit_targets(models_df["family"], hc, pairs, task_context(HUMAN_VALIDITIES), args),
            pairs)
        targets_df.to_csv(out / "targets.csv", index=False)
    written = plot_llm_figures(models_df, preds_df, targets_df, out,
                               target_name=_target_name(args.fit_target))
    for p in written:
        print(f"wrote {p}")
    return 0


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, nargs="+",
                    help="One or more autocog run directories (each with rounds/).")
    ap.add_argument("--replot", type=Path, default=None, metavar="OUT_DIR",
                    help="Skip fitting: redraw the per-LLM figures from a previous "
                         "run's saved CSVs in OUT_DIR.")
    ap.add_argument("--which", choices=("all", "surfaced"), default="all",
                    help="Every theory the run entertained, or only surfaced ones.")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="Use only each run's first N rounds (matched-cycle comparisons).")
    ap.add_argument("--include-canonical", action="store_true",
                    help=f"Also include the canonical heuristics {CANONICAL_NAMES}.")
    ap.add_argument("--human-data", type=Path, default=HUMAN_DATA_DEFAULT,
                    help="Hilbig-format choice data: its stimulus pairs are the fit "
                         "stimuli, and its choices the target under --fit-target human.")
    ap.add_argument("--fit-target", choices=("human", "ground_truth"), default="human",
                    help="Fit to pooled human choices, or to each run's ground-truth "
                         "model (recovery runs).")
    ap.add_argument("--gt-draws", type=int, default=500,
                    help="Parameter draws for the ground-truth P(B) target.")
    ap.add_argument("--gt-pseudo-n", type=float, default=100.0,
                    help="Pseudo-trials per pair for the ground-truth target (scales "
                         "NLL only; the best-fit parameters do not depend on it).")
    ap.add_argument("--all-pairs", action="store_true",
                    help="Compare across ground-truth families too.")
    ap.add_argument("--grid", choices=("hilbig", "all"), default="hilbig",
                    help="Stimuli to compare predictions on: the fit stimuli, or all "
                         "ordered pairs of distinct binary profiles.")
    ap.add_argument("--n-random", type=int, default=200,
                    help="Random-search draws per discrete combo before local search.")
    ap.add_argument("--n-starts", type=int, default=3, help="Nelder-Mead restarts.")
    ap.add_argument("--n-prior-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=60.0,
                    help="Seconds per model for fitting (and again for prior draws); "
                         "on expiry the best-so-far fit is kept and flagged fit_truncated.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    if args.replot is not None:
        return replot(args)
    if not args.run_dir:
        ap.error("--run-dir is required (or --replot OUT_DIR)")
    if args.out is None:
        if len(args.run_dir) != 1:
            ap.error("--out is required with more than one --run-dir")
        args.out = args.run_dir[0] / "analysis" / "compare_models"
    args.out.mkdir(parents=True, exist_ok=True)

    models = load_models(args.run_dir, which=args.which, max_rounds=args.max_rounds,
                         include_canonical=args.include_canonical)
    hc = human_counts(args.human_data)
    fit_pairs: list[Pair] = list(zip(hc["option_a"], hc["option_b"]))
    n_feat = len(fit_pairs[0][0])
    validities = HUMAN_VALIDITIES
    if n_feat != len(validities):
        raise SystemExit(f"human data has {n_feat} features; expected {len(validities)}")
    grid = fit_pairs if args.grid == "hilbig" else full_grid(n_feat)
    ctx = task_context(validities)
    target = (f"{int(hc['n'].sum())} human trials" if args.fit_target == "human"
              else "each run's ground-truth model")
    print(f"{len(models)} models; fitting to {target} on {len(fit_pairs)} stimulus "
          f"pairs; comparing on {len(grid)} pairs.")

    targets = fit_targets({m.family for m in models}, hc, fit_pairs, ctx, args)

    jobs = [{
        "model_id": m.model_id, "theory": m.theory.model_dump(), "context": ctx,
        "fit_pairs": fit_pairs, "n": targets[m.family][0], "k": targets[m.family][1],
        "real_counts": args.fit_target == "human",
        "grid_pairs": grid, "n_random": args.n_random, "n_starts": args.n_starts,
        "n_prior_draws": args.n_prior_draws, "seed": args.seed,
        "time_budget": args.time_budget,
    } for m in models]
    results: list[dict[str, Any]] = [None] * len(jobs)  # type: ignore[list-item]

    def _report(i: int, done: int) -> None:
        r = results[i]["row"]
        flag = " (truncated)" if r["fit_truncated"] else ""
        print(f"  [{done}/{len(jobs)}] {r['model']}  {r['eval_seconds']:.1f}s{flag}", flush=True)

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(_evaluate_one, j): i for i, j in enumerate(jobs)}
            for done, f in enumerate(as_completed(futs), 1):
                results[futs[f]] = f.result()
                _report(futs[f], done)
    else:
        for i, j in enumerate(jobs):
            results[i] = _evaluate_one(j)
            _report(i, i + 1)

    # models.csv
    meta = pd.DataFrame([{"model": m.model_id, "run": m.run, "llm": m.llm,
                          "family": m.family, "noise": m.noise, "replicate": m.replicate,
                          "label": m.label, "role": m.role, "generation": m.generation} for m in models])
    models_df = meta.merge(pd.DataFrame([r["row"] for r in results]), on="model")
    models_df.to_csv(args.out / "models.csv", index=False)
    group_summary(models_df).to_csv(args.out / "group_summary.csv", index=False)
    (args.out / "fit_params.json").write_text(json.dumps(
        {m.model_id: r["params"] for m, r in zip(models, results)}, indent=2, default=str))

    # predictions.csv
    human_lookup = {(a, b): p for a, b, p in zip(hc.option_a, hc.option_b, hc.p_b_human)}
    pred_rows = []
    preds: dict[str, dict[str, np.ndarray]] = {"fit": {}, "prior": {}}
    for m, r in zip(models, results):
        preds["fit"][m.model_id] = np.asarray(r["p_fit"], dtype=float)
        preds["prior"][m.model_id] = np.asarray(r["p_prior"], dtype=float)
        for (a, b), pf, pp in zip(grid, r["p_fit"], r["p_prior"]):
            pred_rows.append({"model": m.model_id, "option_a": "".join(map(str, a)),
                              "option_b": "".join(map(str, b)), "p_b_fit": pf,
                              "p_b_prior": pp, "p_b_human": human_lookup.get((a, b), np.nan)})
    preds_df = pd.DataFrame(pred_rows)
    preds_df.to_csv(args.out / "predictions.csv", index=False)
    targets_df = targets_table(targets, fit_pairs)
    targets_df.to_csv(args.out / "targets.csv", index=False)

    # pairwise.csv
    pw = pairwise_table(models, preds, all_pairs=args.all_pairs)
    pw.to_csv(args.out / "pairwise.csv", index=False)
    if len({m.llm for m in models}) > 1 and len(pw):
        group_pairwise(pw, models_df).to_csv(args.out / "group_pairwise.csv", index=False)

    ids = [m.model_id for m in models]
    for mode in ("fit", "prior"):
        square(pw, ids, f"{mode}_pearson_r", 1.0).to_csv(args.out / f"matrix_{mode}_pearson_r.csv")
        square(pw, ids, f"{mode}_sse", 0.0).to_csv(args.out / f"matrix_{mode}_sse.csv")
    square(pw, ids, "code_ncd", 0.0).to_csv(args.out / "matrix_code_ncd.csv")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        for mode, word in (("fit", "best-fit"), ("prior", "prior-predictive")):
            plot_heatmap(square(pw, ids, f"{mode}_pearson_r", 1.0),
                         title=f"Prediction correlation ({word})", cmap="RdBu_r",
                         vmin=-1, vmax=1, out_base=args.out / f"similarity_{mode}_pearson_r")
            plot_heatmap(square(pw, ids, f"{mode}_sse", 0.0),
                         title=f"Prediction SSE ({word}, {len(grid)} pairs)", cmap="viridis_r",
                         vmin=0, vmax=None, out_base=args.out / f"similarity_{mode}_sse")
        plot_heatmap(square(pw, ids, "code_ncd", 0.0), title="Code distance (NCD)",
                     cmap="viridis_r", vmin=0, vmax=1, out_base=args.out / "similarity_code_ncd")
        plot_complexity(models_df, args.out / "complexity")
        plot_llm_figures(models_df, preds_df, targets_df, args.out,
                         target_name=_target_name(args.fit_target))

    failed = models_df[models_df["fit_error"].notna()]
    for _, r in failed.iterrows():
        print(f"  fit failed for {r.model}: {r.fit_error}")
    n_trunc = int(models_df["fit_truncated"].sum())
    if n_trunc:
        print(f"  {n_trunc} fit(s) hit the {args.time_budget:.0f}s time budget (fit_truncated)")
    cols = ["model", "role", "n_free_params", "code_lines", "code_ast_nodes",
            "desc_words", "fit_mse", "bic", "prior_pred_sd", "uses_history"]
    with pd.option_context("display.width", 220, "display.max_rows", 60):
        if len(models_df) <= 60:
            print(models_df[cols].round(3).to_string(index=False))
        else:
            gs = group_summary(models_df)
            keep = ["family", "llm", "n_runs", "n_models"] + [
                f"{c}_mean" for c in ("n_free_params", "code_lines", "desc_words",
                                      "fit_mse", "prior_pred_sd")]
            print(gs[keep].round(3).to_string(index=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
