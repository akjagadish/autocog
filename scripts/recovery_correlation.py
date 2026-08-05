"""End-to-end ground-truth recovery analysis for the Hilbig (2014) task.

For each held-out ground-truth family (TTB / WADD / Tallying) and action-
noise level (epsilon in {0.0, 0.05, 0.3}), compare three model groups
against the ground truth sampled WITH the run's action noise, in stimulus
choice-proportion space, via Pearson correlation:

  seed     - round-0 starting (competitor) theories autocog was given.
  surfaced - theories autocog discovered (un-killed last round + final
             replacement).
  gt       - the ground-truth theory itself, sampled WITH the run's action
             noise. The recovery ceiling at that noise level.

Every quantity is a SAMPLED choice proportion of choosing option B per
unique stimulus pair (Bernoulli draw from the predicted P(B)); the gt bar
AND the reference both additionally apply epsilon-greedy action noise at the
run's ε (independent seeds), mirroring `src/experiment.py:173`. The reference
is thus the gt theory sampled the same way at the run's ε, NOT at epsilon=0.

Outputs (under --out / --csv / --mse-out, default the synthetic results dir):
  recovery_correlation.csv  - long format, one row per (run-dir, theory)
  recovery_correlation.png  - 1x3 grid: cols = held-out theory,
                              x = noise level, bars = {seed, surfaced, gt}
  recovery_mse.png          - same grid but y = MSE; includes a `random`
                              0.5 baseline as a no-information floor.

Both plots also show two extra surfaced bars, selected metric-specifically
(max r / min MSE) at plot time only — neither is written to the on-disk CSV:
  surfaced (best per run)     - each run-dir's best single surfaced theory,
                                then averaged across runs (mean-of-bests).
  surfaced (best across runs) - the single globally-best surfaced theory per
                                family, pooling every run-dir (one theory).

Reference caveat: ε-greedy action noise contracts each stimulus's choice
proportion toward 0.5 by an approximately affine map (p' ≈ (1-ε)·p + ε·0.5).
Because the reference is now sampled at the run's ε too, the two roles read
differently:
  * gt bar — an independent noisy draw of the SAME generator, so at every ε it
    matches the reference up to finite-sample variance: correlation ≈ 1, MSE at
    the sampling floor. It is the recovery ceiling at that noise level (flat
    across ε by construction — only the reference's own sampling noise erodes it).
  * seed / surfaced — replayed at ε=0 (clean predictions). Measured against the
    CONTRACTED noisy reference, their MSE picks up the contraction gap and their
    correlation is eroded by the reference's noise, both growing with ε. That is
    the cost of NOT matching the generator's noise, and it is what these metrics
    are meant to surface.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.theory import Theory  # noqa: E402
from scripts.eval_hilbig import (  # noqa: E402
    HUMAN_VALIDITIES,
    HUMAN_RATING_MAX,
    HUMAN_DATA_DEFAULT,
    CANONICAL_YAML_DIR,
    _sample_hilibig_params,
    _predict_p_b,
    load_human_choices,
    resolve_base_theories,
    resolve_surfaced_theories,
)
from scripts.figure_style import (  # noqa: E402
    FONTSIZE,
    ROLE_COLOR,
    role_label,
    save_figure,
    style_axes,
)

# Defaults describe the recovery analysis reported in the paper: the three
# canonical sampling heuristics under `results/recovery/`, at the noise-free
# condition plus the two action-noise levels reported in the manuscript
# (epsilon = 0.5 and 0.75). The epsilon = 1.0 runs also present under that root
# are not a reported noise level — they are the source of the synthetic
# `random` family baseline, which `build_random_family_rows` pulls in directly.
FAMILIES_DEFAULT: tuple[str, ...] = (
    "ttb_sampling", "wadd_sampling", "tallying_sampling",
)
NOISES_DEFAULT: tuple[float, ...] = (0.0, 0.5, 0.75)
N_DRAWS_DEFAULT: int = 100
RESULTS_ROOT_DEFAULT: Path = _REPO_ROOT / "results" / "recovery"
# Map family name -> ground-truth YAML stem (ew alias kept for parity).
GROUND_TRUTH_YAML: dict[str, str] = {
    "ttb": "ttb", "wadd": "wadd", "tallying": "tallying", "ew": "ew",
}


def unique_stimulus_pairs(
    human_data: Path = HUMAN_DATA_DEFAULT,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Unique (option_a, option_b) pairs from the canonical Hilbig Exp1 file,
    order-preserving dedupe. These are the fixed stimuli every model is
    replayed on (rating_max=1, validities [0.9,0.8,0.7,0.6])."""
    df = load_human_choices(human_data)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for a, b in zip(df["option_a"], df["option_b"]):
        key = (tuple(int(x) for x in a), tuple(int(x) for x in b))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def simulate_choice_proportions(
    theory: Theory,
    *,
    stimulus_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    validities: list[float],
    rating_max: int,
    n_draws: int,
    seed: int,
    action_noise: float = 0.0,
) -> np.ndarray:
    """Per-stimulus proportion of choosing option B over `n_draws` parameter
    draws. For each draw, sample one subject's parameters, then per stimulus
    draw a choice ~ Bernoulli(P(B)); with probability `action_noise` replace
    the choice with a uniform 50/50 coin (epsilon-greedy action noise,
    mirroring src/experiment.py:173). Determinism: stdlib `random` seeds the
    parameter sampler and a numpy Generator seeds the Bernoulli + noise
    draws, both keyed on `seed`.

    Divergence from `src/experiment.py`: unlike `experiment.simulate`, which
    applies the theory's deterministic argmax `policy` in the non-noise branch,
    here the base choice is a probability-matching Bernoulli draw from P(B);
    only the epsilon-greedy action-noise block is mirrored from
    `src/experiment.py:173`."""
    if not (0.0 <= action_noise <= 1.0):
        raise ValueError(f"action_noise must be in [0, 1]; got {action_noise!r}.")
    n_features = len(validities)
    random.seed(seed)                       # Theory.sample_parameters -> stdlib random
    rng = np.random.default_rng(seed)       # Bernoulli + action-noise draws
    counts = np.zeros(len(stimulus_pairs), dtype=float)
    for d in range(n_draws):
        params = _sample_hilibig_params(
            theory, validities=validities, rating_max=rating_max,
            n_features=n_features, seed_override=seed + d,
        )
        for j, (a, b) in enumerate(stimulus_pairs):
            p_b = _predict_p_b(theory, params, a, b)
            if action_noise > 0.0 and rng.random() < action_noise:
                choice = int(rng.integers(0, 2))
            else:
                choice = 1 if rng.random() < p_b else 0
            counts[j] += choice
    return counts / float(n_draws)


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two per-stimulus vectors. Returns NaN when
    either vector is constant (correlation undefined)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two per-stimulus vectors. Unlike Pearson r,
    MSE is NOT affine-invariant, so it is sensitive to the ε-greedy noise that
    contracts proportions toward 0.5 (the metric that actually 'sees' noise)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


def correlation_for_theory(
    theory: Theory,
    target: np.ndarray,
    *,
    stimulus_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    validities: list[float],
    rating_max: int,
    n_draws: int,
    seed: int,
    action_noise: float = 0.0,
) -> float:
    """Simulate `theory`'s choice proportions and Pearson-correlate them
    against the ground-truth `target` vector (sampled at the run's noise)."""
    props = simulate_choice_proportions(
        theory, stimulus_pairs=stimulus_pairs, validities=validities,
        rating_max=rating_max, n_draws=n_draws, seed=seed,
        action_noise=action_noise,
    )
    return pearson_r(props, target)


def metrics_for_theory(
    theory: Theory,
    target: np.ndarray,
    *,
    stimulus_pairs,
    validities,
    rating_max,
    n_draws,
    seed,
    action_noise: float = 0.0,
) -> dict[str, float]:
    """Simulate `theory`'s choice proportions ONCE and return both metrics
    vs the ground-truth `target`: {'pearson_r': ..., 'mse': ...}."""
    props = simulate_choice_proportions(
        theory, stimulus_pairs=stimulus_pairs, validities=validities,
        rating_max=rating_max, n_draws=n_draws, seed=seed,
        action_noise=action_noise,
    )
    return {"pearson_r": pearson_r(props, target), "mse": mse(props, target)}


# Matches `history` *used* (history.foo / history[...]) — NOT the `history`
# parameter name that every `def predict(..., history)` signature carries.
_HISTORY_USE = re.compile(r"history\s*[.\[]")


def theory_reads_history(theory: Theory) -> bool:
    """True iff the theory's `predict` source actually reads the trial
    history (e.g. `history.get(...)`, `history["response"]`). Such theories
    are sequential and must be simulated through the real trial loop;
    memoryless theories (no history use) keep the cheap independent path."""
    return bool(_HISTORY_USE.search(theory.predict_source))


def simulate_sequence(
    theory: Theory,
    *,
    stimulus_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    validities: list[float],
    n_runs: int,
    seed: int,
    action_noise: float = 0.0,
) -> tuple[np.ndarray, list[list[int]]]:
    """Sequence-aware simulation for history-dependent theories, reusing the
    exact path AutoCog generates data with: ``DecisionMakingBinaryExperiment``
    presents the canonical pairs (repeated/shuffled to fill MAX_TRIALS) and
    accumulates each realized choice into ``history['response']`` so sequential
    theories see the same contract they were written against. ε-greedy action
    noise mirrors the runs (`experiment.simulate`).

    Returns ``(per_pair_proportions, response_sequences)``:
      * ``per_pair_proportions`` — proportion of choosing option B per unique
        pair, aligned to ``stimulus_pairs`` order (feeds Pearson r / MSE).
      * ``response_sequences`` — one realized 0/1 response list per subject in
        trial order (feeds the lag-1 autocorrelation metric).

    Determinism: stdlib ``random`` (parameter sampling) and the noise RNG are
    seeded from ``seed``; the experiment's internal trial-order shuffle is
    unseeded by design, so only the order-invariant aggregates above are
    reproducible (see the recovery design discussion)."""
    if not (0.0 <= action_noise <= 1.0):
        raise ValueError(f"action_noise must be in [0, 1]; got {action_noise!r}.")
    from src.decision_making_binary_features.experiment import (
        DecisionMakingBinaryExperiment,
    )

    exp = DecisionMakingBinaryExperiment(
        validities=list(validities),
        trial_a_ratings=[list(a) for a, _ in stimulus_pairs],
        trial_b_ratings=[list(b) for _, b in stimulus_pairs],
    )
    random.seed(seed)  # parameter sampling -> stdlib random (as elsewhere)
    df = exp.simulate(
        theory, n_runs=n_runs, action_noise=action_noise,
        rng=np.random.default_rng(seed),
    )

    # Per-subject response sequence, in trial order (row order within subject).
    sequences = [
        [int(r) for r in g["response"].tolist()]
        for _, g in df.groupby("subject_id", sort=True)
    ]

    # Per-unique-pair proportion of choosing option B (response == 1).
    keys = [
        (tuple(int(v) for v in a), tuple(int(v) for v in b))
        for a, b in zip(df["option_a_ratings"], df["option_b_ratings"])
    ]
    sums: dict[tuple, float] = {}
    counts: dict[tuple, int] = {}
    for k, r in zip(keys, df["response"].to_numpy(dtype=float)):
        sums[k] = sums.get(k, 0.0) + r
        counts[k] = counts.get(k, 0) + 1
    props = np.array(
        [sums[p] / counts[p] if counts.get(p) else np.nan for p in stimulus_pairs],
        dtype=float,
    )
    return props, sequences


def lag1_autocorr(sequences: list[list[int]]) -> float:
    """Mean lag-1 autocorrelation of realized 0/1 response sequences (Pearson
    r between r[1:] and r[:-1], averaged over subjects). Alternation -> ~-1,
    perseveration -> ~+1, memoryless -> ~0. Constant sequences (no variance)
    are skipped; returns NaN if none are usable."""
    vals: list[float] = []
    for seq in sequences:
        r = np.asarray(seq, dtype=float)
        if r.size < 2:
            continue
        cur, prev = r[1:], r[:-1]
        if cur.std() == 0.0 or prev.std() == 0.0:
            continue
        vals.append(float(np.corrcoef(cur, prev)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def _row_metrics(
    theory: Theory,
    target: np.ndarray,
    *,
    stimulus_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    n_draws: int,
    seed: int,
    action_noise: float,
) -> dict[str, float]:
    """Per-theory recovery metrics with history-aware routing.

    History-reading theories run through ``simulate_sequence`` (real trial
    loop): per-pair proportions feed Pearson r / MSE and the realized response
    sequences feed lag-1 autocorrelation. Memoryless theories keep the existing
    independent path (so ttb/wadd/tallying numbers stay byte-identical) and get
    an analytic ``lag1_autocorr = 0.0`` (no serial dependence by construction)."""
    if theory_reads_history(theory):
        props, sequences = simulate_sequence(
            theory, stimulus_pairs=stimulus_pairs, validities=HUMAN_VALIDITIES,
            n_runs=n_draws, seed=seed, action_noise=action_noise,
        )
        return {
            "pearson_r": pearson_r(props, target),
            "mse": mse(props, target),
            "lag1_autocorr": lag1_autocorr(sequences),
        }
    m = metrics_for_theory(
        theory, target, stimulus_pairs=stimulus_pairs,
        validities=HUMAN_VALIDITIES, rating_max=HUMAN_RATING_MAX,
        n_draws=n_draws, seed=seed, action_noise=action_noise,
    )
    return {"pearson_r": m["pearson_r"], "mse": m["mse"], "lag1_autocorr": 0.0}


def discover_run_dirs(
    results_root: Path,
    *,
    families: list[str],
    noises: list[float] | None = None,
) -> list[tuple[str, float, Path]]:
    """Enumerate autocog run-dirs as (family, noise, path), sorted
    deterministically. Layout:
        <results_root>/<family>/noise=<eps>/hdm_ground_truth_<family>_*_run*
    `noises=None` accepts every `noise=*` dir found."""
    out: list[tuple[str, float, Path]] = []
    for family in families:
        family_root = results_root / family
        if not family_root.is_dir():
            print(f"[recovery] no dir for family {family!r}: {family_root}",
                  file=sys.stderr)
            continue
        found_noise_vals: set[float] = set()
        for noise_dir in sorted(family_root.glob("noise=*")):
            if not noise_dir.is_dir():
                continue
            noise_val = float(noise_dir.name.removeprefix("noise="))
            if noises is not None and noise_val not in noises:
                continue
            found_noise_vals.add(noise_val)
            # Match either prefix: hdm_ (original synthetic) or dmb_
            # (corrected-theories binary). Both name the same role.
            run_dirs = sorted(
                noise_dir.glob(f"*ground_truth_{family}_*_run*")
            )
            if not run_dirs:
                print(f"[recovery] no run-dirs under {noise_dir}",
                      file=sys.stderr)
            for run_dir in run_dirs:
                if (run_dir / "rounds").is_dir():
                    out.append((family, noise_val, run_dir))
        # Fix 2a: warn about explicitly-requested noise values with no dir.
        if noises is not None:
            for n in noises:
                if n not in found_noise_vals:
                    print(f"[recovery] warning: no noise={n} dir for family {family!r}",
                          file=sys.stderr)
    return out


def _theory_rows(
    theories: dict[str, Theory],
    *,
    role: str,
    family: str,
    noise: float,
    run_dir: Path,
    target: np.ndarray,
    stimulus_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    n_draws: int,
    base_seed_for_role: int,
    action_noise: float,
) -> list[dict[str, Any]]:
    """One row per theory: its Pearson r AND MSE to the noisy gt target."""
    rows: list[dict[str, Any]] = []
    for i, (label, theory) in enumerate(theories.items()):
        m = _row_metrics(
            theory, target, stimulus_pairs=stimulus_pairs,
            n_draws=n_draws, seed=base_seed_for_role + i,
            action_noise=action_noise,
        )
        rows.append({
            "family": family, "noise": noise, "run_dir": run_dir.name,
            "role": role, "theory_label": label,
            "n_stimuli": len(stimulus_pairs),
            "pearson_r": m["pearson_r"], "mse": m["mse"],
            "lag1_autocorr": m["lag1_autocorr"],
        })
    return rows


def build_results_long(
    *,
    results_root: Path,
    families: list[str],
    noises: list[float] | None,
    stimulus_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    n_draws: int,
    base_seed: int,
    reference_action_noise: float | None = None,
) -> pd.DataFrame:
    """Long-format results: one row per (run-dir, theory) across the seed,
    surfaced, and gt roles. The reference target is the gt theory sampled WITH
    each run's action noise (one per (family, noise), independent seed);
    seed/surfaced theories are replayed at eps=0 and compared against that noisy
    reference; the gt bar is the gt theory replayed at the run's noise level (an
    independent noisy draw of the same generator — the recovery ceiling).

    `reference_action_noise` overrides the noise the REFERENCE is sampled at:
    None (default) keeps the per-(family, noise) noisy reference described above;
    a fixed float (e.g. 0.0) pins every column's reference to that noise — set to
    0.0 to score all roles against the CLEAN gt@0 instead of the noisy gt@ε.
    Only the reference changes; every role's own predictions are untouched, so a
    None-call and a 0.0-call differ solely in the target each row is scored
    against (the role props are byte-identical across the two)."""
    run_dirs = discover_run_dirs(
        results_root, families=families, noises=noises,
    )
    # Per-(family, noise) NOISY reference: the gt theory sampled WITH that
    # noise level. Seed banding keeps each reference independent of the seed/
    # surfaced/gt-bar draws (which live in the 100k/300k/700k bands):
    #   900_000 + fam_idx*100_000 + noise_idx*1_000 — one slot per (fam, noise)
    # where noise_idx indexes the sorted noise levels present for that family.
    fam_index = {fam: i for i, fam in enumerate(families)}
    noises_by_family: dict[str, list[float]] = {}
    for fam, noise, _ in run_dirs:
        levels = noises_by_family.setdefault(fam, [])
        if noise not in levels:
            levels.append(noise)
    for levels in noises_by_family.values():
        levels.sort()
    targets: dict[tuple[str, float], np.ndarray] = {}
    for family, noise, _ in run_dirs:
        if (family, noise) in targets:
            continue
        gt_theory = Theory.from_yaml(
            CANONICAL_YAML_DIR / f"{GROUND_TRUTH_YAML.get(family, family)}.yaml"
        )
        noise_idx = noises_by_family[family].index(noise)
        ref_noise = noise if reference_action_noise is None else reference_action_noise
        targets[(family, noise)] = simulate_choice_proportions(
            gt_theory, stimulus_pairs=stimulus_pairs,
            validities=HUMAN_VALIDITIES, rating_max=HUMAN_RATING_MAX,
            n_draws=n_draws,
            seed=(base_seed + 900_000 + fam_index[family] * 100_000
                  + noise_idx * 1_000),
            action_noise=ref_noise,
        )

    rows: list[dict[str, Any]] = []
    for r_idx, (family, noise, run_dir) in enumerate(run_dirs):
        target = targets[(family, noise)]
        gt_theory = Theory.from_yaml(
            CANONICAL_YAML_DIR / f"{GROUND_TRUTH_YAML.get(family, family)}.yaml"
        )
        seed_theories = resolve_base_theories(run_dir)
        surfaced_theories = resolve_surfaced_theories(run_dir)
        # Seed-offset band layout (all offsets from base_seed):
        #   100_000 + r_idx*1_000  — seed theories
        #   300_000 + r_idx*1_000  — surfaced theories
        #   500_000 + r_idx*1_000  — gt_clean (clean gt vs noisy ref)
        #   700_000 + r_idx*1_000  — gt bar
        #   900_000 + fam_idx*100_000 + noise_idx*1_000 — noisy refs (above)
        # Bands stay collision-free up to ~200 run-dirs (actual ~27), so
        # independence between roles holds throughout any realistic run.
        rows += _theory_rows(
            seed_theories, role="seed", family=family, noise=noise,
            run_dir=run_dir, target=target, stimulus_pairs=stimulus_pairs,
            n_draws=n_draws, base_seed_for_role=base_seed + 100_000 + r_idx * 1_000,
            action_noise=0.0,
        )
        rows += _theory_rows(
            surfaced_theories, role="surfaced", family=family, noise=noise,
            run_dir=run_dir, target=target, stimulus_pairs=stimulus_pairs,
            n_draws=n_draws, base_seed_for_role=base_seed + 300_000 + r_idx * 1_000,
            action_noise=0.0,
        )
        # gt bar: the gt theory replayed WITH the run's action noise.
        gt_m = _row_metrics(
            gt_theory, target, stimulus_pairs=stimulus_pairs,
            n_draws=n_draws, seed=base_seed + 700_000 + r_idx * 1_000,
            action_noise=noise,
        )
        rows.append({
            "family": family, "noise": noise, "run_dir": run_dir.name,
            "role": "gt", "theory_label": "gt",
            "n_stimuli": len(stimulus_pairs),
            "pearson_r": gt_m["pearson_r"], "mse": gt_m["mse"],
            "lag1_autocorr": gt_m["lag1_autocorr"],
        })
        # gt_clean: the CLEAN gt (gt@eps=0 predictions) vs the noisy gt@eps
        # reference — i.e. MSE between the ground truth and ITS OWN noisy
        # version. Unlike the gt row (gt@eps vs gt@eps, flat at the sampling
        # floor), this RISES with eps as the noisy version contracts toward 0.5
        # while the clean prediction does not. Band 500_000 + r_idx*1_000
        # (between surfaced and gt-bar bands, collision-free at realistic
        # run-dir counts).
        gt_clean_m = _row_metrics(
            gt_theory, target, stimulus_pairs=stimulus_pairs,
            n_draws=n_draws, seed=base_seed + 500_000 + r_idx * 1_000,
            action_noise=0.0,
        )
        rows.append({
            "family": family, "noise": noise, "run_dir": run_dir.name,
            "role": "gt_clean", "theory_label": "gt_clean",
            "n_stimuli": len(stimulus_pairs),
            "pearson_r": gt_clean_m["pearson_r"], "mse": gt_clean_m["mse"],
            "lag1_autocorr": gt_clean_m["lag1_autocorr"],
        })
        # random baseline: chooses uniformly every trial => proportion 0.5 on
        # every stimulus. Constant vector => Pearson r undefined (NaN); used as
        # a no-information FLOOR in the MSE plot only.
        random_props = np.full(len(stimulus_pairs), 0.5, dtype=float)
        rows.append({
            "family": family, "noise": noise, "run_dir": run_dir.name,
            "role": "random", "theory_label": "random",
            "n_stimuli": len(stimulus_pairs),
            "pearson_r": pearson_r(random_props, target),  # NaN (constant)
            "mse": mse(random_props, target),
            "lag1_autocorr": 0.0,  # uniform coin -> no serial dependence
        })
    return pd.DataFrame(rows)


def build_random_family_rows(
    *,
    results_root: Path,
    canonical_families: list[str],
    stimulus_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    n_draws: int,
    base_seed: int,
    eps: float = 1.0,
) -> pd.DataFrame:
    """Long-format rows for a synthetic ``random`` ground-truth family.

    There is no autocog run whose GT is "random", so the data is drawn from the
    canonical families' ε=`eps` run-dirs: at ε=1 every choice is a uniform coin,
    so those runs *are* "GT = random choices". The seed/surfaced theories autocog
    was given / discovered there are replayed at ε=0 (clean) and scored against a
    FIXED 0.5 reference vector (the true random process) — NOT the per-run noisy
    gt reference the canonical families use. All such run-dirs across
    `canonical_families` are pooled into one family ``random``.

    Rows are labelled ``noise=0.0`` (the OUTPUT axis label) so the Random panel
    renders single-noise like the other non-canonical families; ``eps`` only
    selects the source run-dirs. Pearson r is NaN (constant reference); mse is
    finite. Seed bands ``base_seed + 2_000_000`` (seed) / ``+ 2_200_000``
    (surfaced) sit ABOVE build_results_long's highest band (the noisy-reference
    band tops out near 1.7M for 9 families), so these draws stay independent."""
    reference = np.full(len(stimulus_pairs), 0.5, dtype=float)
    run_dirs: list[tuple[str, float, Path]] = []
    for family in canonical_families:
        run_dirs += discover_run_dirs(
            results_root, families=[family], noises=[eps],
        )
    rows: list[dict[str, Any]] = []
    for r_idx, (_family, _noise, run_dir) in enumerate(run_dirs):
        rows += _theory_rows(
            resolve_base_theories(run_dir), role="seed", family="random",
            noise=0.0, run_dir=run_dir, target=reference,
            stimulus_pairs=stimulus_pairs, n_draws=n_draws,
            base_seed_for_role=base_seed + 2_000_000 + r_idx * 1_000,
            action_noise=0.0,
        )
        rows += _theory_rows(
            resolve_surfaced_theories(run_dir), role="surfaced", family="random",
            noise=0.0, run_dir=run_dir, target=reference,
            stimulus_pairs=stimulus_pairs, n_draws=n_draws,
            base_seed_for_role=base_seed + 2_200_000 + r_idx * 1_000,
            action_noise=0.0,
        )
    return pd.DataFrame(rows)


def summarise(long_df: pd.DataFrame, metric: str = "pearson_r") -> pd.DataFrame:
    """Two-stage aggregation matching the spec: first pool theories WITHIN a
    run-dir (mean metric per (family, noise, run_dir, role)), then take
    mean +/- SEM ACROSS run-dirs per (family, noise, role). NaN values are
    dropped before pooling. Works for any metric column (pearson_r or mse)."""
    nan_rows = long_df[long_df[metric].isna()]
    if not nan_rows.empty:
        dropped = nan_rows[["family", "noise", "role"]].drop_duplicates().to_dict("records")
        print(f"[recovery] dropping {len(nan_rows)} NaN {metric} row(s); affected (family,noise,role): {dropped}",
              file=sys.stderr)
    clean = long_df[long_df[metric].notna()].copy()
    per_run = (
        clean
        .groupby(["family", "noise", "run_dir", "role"], observed=True)[metric]
        .mean()
        .reset_index()
    )
    grouped = (
        per_run
        .groupby(["family", "noise", "role"], observed=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_runs"})
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n_runs"]).where(
        grouped["n_runs"] > 1, other=np.inf,
    )
    grouped.loc[grouped["n_runs"] <= 1, "sem"] = 0.0
    return grouped


# Two "best surfaced" reductions, distinguished by the grouping granularity:
#   * per run    — best surfaced theory WITHIN each run-dir; `summarise` then
#                  averages those per-run bests across runs (mean-of-bests).
#   * across runs — the SINGLE globally-best surfaced theory per (family,
#                  noise), pooling every run-dir (one theory, one bar).
_BEST_PER_RUN_KEYS: tuple[str, ...] = ("family", "noise", "run_dir")
_BEST_ACROSS_RUNS_KEYS: tuple[str, ...] = ("family", "noise")
_ROLE_BEST_PER_RUN = "surfaced (best per run)"
_ROLE_BEST_ACROSS_RUNS = "surfaced (best across runs)"


def _best_surfaced_by_metric(
    long_df: pd.DataFrame, *, metric: str, higher_is_better: bool,
    group_keys: tuple[str, ...], role_label: str,
) -> pd.DataFrame:
    """Within each `group_keys` group, keep the single surfaced theory with the
    best `metric` (max if higher_is_better else min), re-labelled `role_label`.
    NaN-`metric` rows are ignored; a group with no valid surfaced row
    contributes nothing. Returned rows carry the chosen theory's full row with
    `role` overwritten."""
    surf = long_df[(long_df["role"] == "surfaced") & long_df[metric].notna()]
    if surf.empty:
        return long_df.iloc[0:0].copy()
    keep_idx = []
    for _, g in surf.groupby(list(group_keys), observed=True):
        keep_idx.append(g[metric].idxmax() if higher_is_better else g[metric].idxmin())
    best = long_df.loc[keep_idx].copy()
    best["role"] = role_label
    return best


def _best_surfaced_by_autocorr(
    long_df: pd.DataFrame, *, group_keys: tuple[str, ...], role_label: str,
) -> pd.DataFrame:
    """Like `_best_surfaced_by_metric` but for lag-1 autocorr, where the
    recovery ideal is not a fixed extreme but the gt's OWN value (e.g. -1 for
    alternating, 0 for memoryless). Each surfaced row is scored by its distance
    to its run-dir's gt autocorr; the per-group minimum |distance| wins. Rows
    whose run-dir lacks a valid gt autocorr contribute nothing."""
    gt = long_df[(long_df["role"] == "gt") & long_df["lag1_autocorr"].notna()]
    gt_by_run = {
        (r.family, r.noise, r.run_dir): r.lag1_autocorr
        for r in gt.itertuples(index=False)
    }
    surf = long_df[(long_df["role"] == "surfaced")
                   & long_df["lag1_autocorr"].notna()].copy()
    if surf.empty:
        return long_df.iloc[0:0].copy()
    surf["_dist"] = [
        abs(r.lag1_autocorr - gt_by_run[(r.family, r.noise, r.run_dir)])
        if (r.family, r.noise, r.run_dir) in gt_by_run else np.nan
        for r in surf.itertuples(index=False)
    ]
    surf = surf[surf["_dist"].notna()]
    if surf.empty:
        return long_df.iloc[0:0].copy()
    keep_idx = [
        g["_dist"].idxmin()
        for _, g in surf.groupby(list(group_keys), observed=True)
    ]
    best = long_df.loc[keep_idx].copy()
    best["role"] = role_label
    return best


def surfaced_best_rows(
    long_df: pd.DataFrame, *, metric: str, higher_is_better: bool,
) -> pd.DataFrame:
    """Best surfaced theory PER run-dir (role 'surfaced (best per run)')."""
    return _best_surfaced_by_metric(
        long_df, metric=metric, higher_is_better=higher_is_better,
        group_keys=_BEST_PER_RUN_KEYS, role_label=_ROLE_BEST_PER_RUN,
    )


def surfaced_best_across_runs_rows(
    long_df: pd.DataFrame, *, metric: str, higher_is_better: bool,
) -> pd.DataFrame:
    """Single globally-best surfaced theory per (family, noise) across ALL
    run-dirs (role 'surfaced (best across runs)')."""
    return _best_surfaced_by_metric(
        long_df, metric=metric, higher_is_better=higher_is_better,
        group_keys=_BEST_ACROSS_RUNS_KEYS, role_label=_ROLE_BEST_ACROSS_RUNS,
    )


def surfaced_best_autocorr_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    """Best (closest-to-gt autocorr) surfaced theory PER run-dir."""
    return _best_surfaced_by_autocorr(
        long_df, group_keys=_BEST_PER_RUN_KEYS, role_label=_ROLE_BEST_PER_RUN,
    )


def surfaced_best_across_runs_autocorr_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    """Single globally-best (closest-to-gt autocorr) surfaced theory per
    (family, noise) across all run-dirs."""
    return _best_surfaced_by_autocorr(
        long_df, group_keys=_BEST_ACROSS_RUNS_KEYS,
        role_label=_ROLE_BEST_ACROSS_RUNS,
    )


GROUND_TRUTHS_PLOT: tuple[str, ...] = (
    "ttb", "wadd", "tallying",
    "ttb_sampling", "wadd_sampling", "tallying_sampling",
)
ROLES_PLOT: tuple[str, ...] = ("seed", "surfaced", "gt")


def family_plot_order(
    families_present: Any, priority: tuple[str, ...] = GROUND_TRUTHS_PLOT,
) -> list[str]:
    """Priority-first family ordering: families that appear in `priority` come
    first (in that order), then any others present in first-seen order. Default
    `priority` is GROUND_TRUTHS_PLOT (canonical-first); callers can pass a custom
    order (e.g. the per-model FAMILY_PLOT_ORDER). Accepts any iterable of family
    names — a DataFrame column or a plain list."""
    present = list(dict.fromkeys(families_present))
    return ([f for f in priority if f in present]
            + [f for f in present if f not in priority])


def _noise_axis_is_informative(summary: pd.DataFrame) -> bool:
    """True iff the figure spans more than one noise level. Single-noise
    figures (e.g. non-canonical families only run at ε=0) have a meaningless
    noise x-axis, so callers drop the ticks and label."""
    return summary["noise"].nunique() > 1


def _panel_title(
    family: str, show_title: bool, title_formatter=None,
) -> str | None:
    """Per-panel title or None when titles are suppressed. With no
    `title_formatter` the family is uppercased (the combined-grid behaviour);
    pass a formatter (e.g. per-model `heuristic_title`) to override."""
    if not show_title:
        return None
    fmt = title_formatter if title_formatter is not None else str.upper
    return fmt(family)


def _figure_width(n_panels: int, fig_width: float | None) -> float:
    """Total figure width in inches: a fixed `fig_width` when given (used to
    pool many panels into a compact figure ~2 canonical panels wide), else the
    default 6in per panel."""
    return fig_width if fig_width is not None else 6.0 * n_panels


# Per-role linestyle for the reference lines (baseline_roles): the random floor
# is dotted, the gt ceiling dashed, so the two are distinct beyond colour.
# gt = flat best-achievable floor (dashed); gt_clean = clean-vs-noisy gt
# baseline that rises with ε (dotted); random = no-information floor (dotted).
_BASELINE_LINESTYLE: dict[str, str] = {"random": ":", "gt": "--", "gt_clean": ":"}


def _legend_label(role: str, legend_labels: dict[str, str] | None) -> str:
    """Per-figure legend label: the `legend_labels` override when the role is
    present, else the global `role_label`. Lets one figure relabel a role
    (e.g. the clean-reference figure, where gt / gt_clean swap meaning) without
    mutating the shared ROLE_DISPLAY map."""
    if legend_labels and role in legend_labels:
        return legend_labels[role]
    return role_label(role)


def plot_grid(
    summary: pd.DataFrame, out_path: Path, *,
    roles: tuple[str, ...] = ROLES_PLOT,
    ylabel: str = "Pearson r to\nnoisy ground truth",
    title: str | None = None,
    show_title: bool = True,
    title_formatter=None,
    baseline_roles: tuple[str, ...] = (),
    fig_width: float | None = None,
    legend_labels: dict[str, str] | None = None,
    share_y: bool = True,
    family_order: tuple[str, ...] | None = None,
    bar_edgecolor: str = "black",
) -> None:
    """1x3 grid: cols = held-out theory, x = noise level, bars = role.
    Mirrors hilibig_battery_base_vs_surfaced.png but a single correlation
    row with {seed, surfaced, gt} bars. When the figure spans a single noise
    level (e.g. non-canonical families only run at ε=0), the noise x-axis
    carries no information and is dropped. `show_title=False` suppresses the
    per-panel family title; `title_formatter` overrides how it is rendered.

    Each role in `baseline_roles` (e.g. "random", "gt") is drawn NOT as a bar
    but as a horizontal segment spanning each noise level's bar group at that
    level's value — reference floor/ceiling lines (styled per
    `_BASELINE_LINESTYLE`). They must be disjoint from `roles` (else a role
    would be drawn as both a bar and a line).

    `share_y=True` (default) keeps one y-scale across panels; pass False to give
    each family panel its own y-limits (e.g. when seed magnitudes differ by
    family and cross-panel bar heights are not comparable)."""
    overlap = set(baseline_roles) & set(roles)
    if overlap:
        raise ValueError(
            f"baseline_roles {sorted(overlap)} must not also be bar roles "
            f"(roles={roles}); they are drawn as lines, not bars."
        )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show_noise = _noise_axis_is_informative(summary)

    # Known families first (in GROUND_TRUTHS_PLOT order), then any others
    # present (e.g. sampling variants) appended in first-seen order.
    families = family_plot_order(
        summary["family"], priority=family_order or GROUND_TRUTHS_PLOT,
    )
    if not families:
        # Empty summary (e.g. every family all-NaN for this metric after
        # summarise drops them) — nothing to plot; avoid a 1x0 subplots crash.
        print(f"[recovery] plot_grid: empty summary, skipping {out_path}",
              file=sys.stderr)
        return
    fig, axes = plt.subplots(
        1, len(families), figsize=(_figure_width(len(families), fig_width), 4),
        sharey=share_y, squeeze=False,
    )
    for col, fam in enumerate(families):
        ax = axes[0][col]
        sub = summary[summary.family == fam]
        noises = sorted(sub.noise.unique())
        x = np.arange(len(noises))
        width = 0.8 / len(roles)
        for i, role in enumerate(roles):
            rrows = sub[sub.role == role].set_index("noise")
            means = np.array(
                [rrows.loc[n, "mean"] if n in rrows.index else np.nan
                 for n in noises], dtype=float)
            sems = np.array(
                [rrows.loc[n, "sem"] if n in rrows.index else 0.0
                 for n in noises], dtype=float)
            ax.bar(
                x - 0.4 + width * (i + 0.5), means, width,
                yerr=sems, color=ROLE_COLOR[role],
                # thin edge keeps the cream "best across runs" bar visible in
                # the combined grid; pool / per-model figures pass "none".
                edgecolor=bar_edgecolor, linewidth=0.5,
                label=_legend_label(role, legend_labels) if col == 0 else None,
            )
        for b_role in baseline_roles:
            # Reference line per noise level, spanning that level's bar group
            # (x±0.4, the full 0.8-wide group). Labelled once for the shared
            # legend; styled per role (random dotted, gt dashed).
            brows = sub[sub.role == b_role].set_index("noise")
            linestyle = _BASELINE_LINESTYLE.get(b_role, ":")
            for k, n in enumerate(noises):
                if n not in brows.index:
                    continue
                yval = float(brows.loc[n, "mean"])
                ax.plot(
                    [x[k] - 0.4, x[k] + 0.4], [yval, yval],
                    linestyle=linestyle, color=ROLE_COLOR[b_role],
                    linewidth=2,
                    label=(_legend_label(b_role, legend_labels)
                           if (col == 0 and k == 0) else None),
                )
        if show_noise:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{n:g}" for n in noises])
            style_axes(ax, xlabel="action noise (ε)",
                       title=_panel_title(fam, show_title, title_formatter))
        else:
            # Single noise level: the x-axis is uninformative — drop ticks
            # and label, keep the bars centred under the panel title.
            ax.set_xticks([])
            style_axes(ax, title=_panel_title(fam, show_title, title_formatter))
    axes[0][0].set_ylabel(ylabel, fontsize=FONTSIZE)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)),
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               fontsize=FONTSIZE - 2)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    save_figure(fig, out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Ground-truth recovery on the Hilbig task: correlate sampled "
            "choice proportions of seed / surfaced / gt models against the "
            "ground truth sampled at the run's noise, across families and "
            "noise levels."
        )
    )
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT_DEFAULT)
    p.add_argument("--families", nargs="+", default=list(FAMILIES_DEFAULT))
    p.add_argument(
        "--noises", nargs="*", type=float, default=None,
        help="Noise levels to include (e.g. 0.0 0.05 0.3). Empty = all found.",
    )
    p.add_argument("--human-data", type=Path, default=HUMAN_DATA_DEFAULT)
    p.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--csv", type=Path,
        default=RESULTS_ROOT_DEFAULT / "recovery_correlation.csv",
    )
    p.add_argument(
        "--out", type=Path,
        default=RESULTS_ROOT_DEFAULT / "recovery_correlation.png",
    )
    p.add_argument(
        "--mse-out", type=Path,
        default=RESULTS_ROOT_DEFAULT / "recovery_mse.png",
    )
    p.add_argument(
        "--autocorr-out", type=Path,
        default=RESULTS_ROOT_DEFAULT / "recovery_autocorr.png",
    )
    p.add_argument(
        "--title", type=str,
        default=(
            "Ground-truth recovery (Hilbig) - Pearson r of sampled choice "
            "proportions to noisy ground truth"
        ),
    )
    args = p.parse_args(argv)

    # Fix 1: normalise empty list (--noises with no values) to None so
    # discover_run_dirs accepts all noise dirs (spec §7).
    noises = args.noises if args.noises else None

    pairs = unique_stimulus_pairs(args.human_data)
    print(f"[recovery] stimuli={len(pairs)} families={args.families} "
          f"noises={noises} n_draws={args.n_draws} seed={args.seed}")

    long_df = build_results_long(
        results_root=args.results_root, families=args.families,
        noises=noises, stimulus_pairs=pairs,
        n_draws=args.n_draws, base_seed=args.seed,
    )
    if long_df.empty:
        print("[recovery] no results — check --results-root / --families.",
              file=sys.stderr)
        return 1

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.csv, index=False)

    # Two extra bars per metric, added at plot time only (NOT written to the
    # on-disk CSV): 'surfaced (best per run)' = each run-dir's best surfaced
    # theory averaged across runs; 'surfaced (best across runs)' = the single
    # globally-best surfaced theory per family. Selection is metric-specific
    # (max r for correlation, min mse for MSE).
    corr_long = pd.concat(
        [long_df,
         surfaced_best_rows(long_df, metric="pearson_r", higher_is_better=True),
         surfaced_best_across_runs_rows(long_df, metric="pearson_r", higher_is_better=True)],
        ignore_index=True,
    )
    summary_corr = summarise(corr_long, "pearson_r")
    plot_grid(
        summary_corr, args.out,
        roles=("seed", "surfaced", "surfaced (best per run)",
               "surfaced (best across runs)", "gt"),
        ylabel="Pearson r to\nnoisy ground truth",
        title=args.title,
    )

    mse_long = pd.concat(
        [long_df,
         surfaced_best_rows(long_df, metric="mse", higher_is_better=False),
         surfaced_best_across_runs_rows(long_df, metric="mse", higher_is_better=False)],
        ignore_index=True,
    )
    summary_mse = summarise(mse_long, "mse")
    plot_grid(
        summary_mse, args.mse_out,
        roles=("random", "seed", "surfaced", "surfaced (best per run)",
               "surfaced (best across runs)", "gt"),
        ylabel="MSE to noisy\nground truth (lower = better)",
        title="Ground-truth recovery (Hilbig) - MSE of sampled choice "
              "proportions to noisy ground truth",
    )
    # lag-1 response autocorrelation: the sequential signature. Carries the
    # recovery signal for stimulus-independent serial models (e.g. alternating)
    # that the per-stimulus-pair corr/MSE structurally cannot see. NaN rows
    # (constant sequences) are dropped by summarise.
    autocorr_long = pd.concat(
        [long_df,
         surfaced_best_autocorr_rows(long_df),
         surfaced_best_across_runs_autocorr_rows(long_df)],
        ignore_index=True,
    )
    summary_autocorr = summarise(autocorr_long, "lag1_autocorr")
    plot_grid(
        summary_autocorr, args.autocorr_out,
        roles=("seed", "surfaced", "surfaced (best per run)",
               "surfaced (best across runs)", "gt"),
        ylabel="lag-1 response autocorrelation",
        title="Ground-truth recovery (Hilbig) - lag-1 response "
              "autocorrelation (sequential signature)",
    )

    print(summary_corr.to_string(index=False))
    print(summary_mse.to_string(index=False))
    print(summary_autocorr.to_string(index=False))
    print(f"[recovery] wrote {args.csv}, {args.out}, {args.mse_out}, "
          f"{args.autocorr_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
