"""Tests for ground-truth-only epsilon-greedy action noise in
`Experiment.simulate`.

Rationale: the heuristic theories already contain an in-predict `epsilon`
lapse that gets argmax-ed away by their deterministic policies, so the
ground-truth simulation is effectively noiseless today. We want a knob
that injects **policy-level** epsilon-greedy noise only when simulating
the ground-truth subject, leaving the theory YAMLs untouched (this is a
synthetic-test-case knob, not a modelling claim).

The contract under test:
  simulate(theory, n_runs=N, action_noise=eps, rng=rng)
    - action_noise=0.0 (default): identical output to simulate(theory, n_runs=N)
    - action_noise=1.0: chosen action is uniform over len(probs), independent
      of the theory's policy
    - 0 < action_noise < 1: with probability action_noise the action is
      replaced by a uniform draw; otherwise it's the theory.policy output
    - rng seeds the noise draw so runs are reproducible

The test theory used below returns a deterministic p=[1, 0] on every trial,
so we can separate "policy chose 0" (theory) from "policy chose 1" (flip).
Option B is the flipped option, so observed flip rate ≈ eps/2 on a binary
task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.heuristic_decision_making.experiment import (
    HeuristicDecisionMakingExperiment,
)
from src.theory import Theory


def _always_a_theory() -> Theory:
    """A theory that always returns p=[1.0, 0.0] — argmax policy picks A (0)."""
    predict_src = (
        "def predict(parameters, state, history):\n"
        "    import numpy as np\n"
        "    return np.array([1.0, 0.0])\n"
    )
    policy_src = (
        "def policy(probs):\n"
        "    import numpy as np\n"
        "    return int(np.argmax(probs))\n"
    )
    return Theory(
        description="Always-A (deterministic)",
        predict_source=predict_src,
        policy_source=policy_src,
        parameters={},
    )


def _hdm_experiment(n_trials: int = 20) -> HeuristicDecisionMakingExperiment:
    # Binary features, easy-to-dissociate trials. Actual design doesn't
    # matter for these tests — we only care about the action at each trial.
    trial_a = [[1, 0]] * n_trials
    trial_b = [[0, 1]] * n_trials
    return HeuristicDecisionMakingExperiment(
        validities=[0.9, 0.6],
        rating_max=1,
        trial_a_ratings=trial_a,
        trial_b_ratings=trial_b,
    )


def test_action_noise_default_matches_plain_simulate():
    """action_noise=0.0 must preserve the pre-existing (deterministic) output."""
    exp_plain = _hdm_experiment()
    exp_noisy = _hdm_experiment()
    theory = _always_a_theory()

    df_plain = exp_plain.simulate(theory, n_runs=3)
    # Pass action_noise=0.0 with a seeded rng; should still match byte-for-byte
    # because the noise branch never fires.
    df_noisy = exp_noisy.simulate(
        theory, n_runs=3, action_noise=0.0, rng=np.random.default_rng(42)
    )
    assert df_plain["response"].tolist() == df_noisy["response"].tolist()


def test_action_noise_zero_deterministic_all_a():
    """Without noise the always-A theory picks 0 on every trial."""
    exp = _hdm_experiment()
    theory = _always_a_theory()
    df = exp.simulate(theory, n_runs=2)
    # Every response must be 0 (option A).
    assert set(df["response"].unique()) == {0}


def test_action_noise_one_uniform_over_options():
    """action_noise=1.0 on an always-A theory yields ~50/50 choices."""
    exp = _hdm_experiment(n_trials=30)
    theory = _always_a_theory()
    df = exp.simulate(
        theory, n_runs=50, action_noise=1.0, rng=np.random.default_rng(0)
    )
    # With all-noise on a binary task, expect P(response=1) ~ 0.5.
    p1 = float((df["response"] == 1).mean())
    assert 0.4 <= p1 <= 0.6, (
        f"expected ~50/50 under action_noise=1.0, got P(response=1)={p1:.3f}"
    )


def test_action_noise_intermediate_flip_rate_matches_eps_over_two():
    """On a deterministic always-A theory with binary actions, the observed
    flip rate P(response=1) under epsilon-greedy is exactly eps/2 in
    expectation (with prob eps we pick uniform; half those land on B).
    """
    exp = _hdm_experiment(n_trials=20)
    theory = _always_a_theory()
    eps = 0.4
    df = exp.simulate(
        theory, n_runs=200, action_noise=eps, rng=np.random.default_rng(7)
    )
    p1 = float((df["response"] == 1).mean())
    # Expected flip rate eps/2 = 0.20. Allow ~3 SE slack on n=4000 trials
    # of a Bernoulli(0.2) (SE ~ 0.006, so ±0.02 is very comfortable).
    assert abs(p1 - eps / 2) < 0.02, (
        f"flip rate {p1:.3f} deviates from eps/2={eps / 2:.3f}"
    )


def test_action_noise_rng_is_reproducible():
    """Two runs with the same seed produce the same action sequence."""
    theory = _always_a_theory()

    exp_a = _hdm_experiment()
    df_a = exp_a.simulate(
        theory, n_runs=3, action_noise=0.3, rng=np.random.default_rng(123)
    )
    exp_b = _hdm_experiment()
    df_b = exp_b.simulate(
        theory, n_runs=3, action_noise=0.3, rng=np.random.default_rng(123)
    )
    # Different trial shuffles come from default_rng() inside the experiment
    # post_init (not the noise rng we pass in). So compare the noise-driven
    # *sequence of responses per subject* — with an always-A theory the
    # trial order doesn't affect which responses flip, only where they land
    # in the df. Simplest stable check: total flips must match.
    flips_a = int((df_a["response"] == 1).sum())
    flips_b = int((df_b["response"] == 1).sum())
    assert flips_a == flips_b, (
        f"same-seed runs produced different flip counts: {flips_a} vs {flips_b}"
    )


def test_action_noise_rejects_negative_and_over_one():
    """action_noise must be in [0, 1]; bad values raise rather than clip."""
    exp = _hdm_experiment()
    theory = _always_a_theory()
    with pytest.raises(ValueError):
        exp.simulate(theory, n_runs=1, action_noise=-0.1)
    with pytest.raises(ValueError):
        exp.simulate(theory, n_runs=1, action_noise=1.5)


# ---------------------------------------------------------------------------
# Integration tests: exercise the exact chain `main_heuristic_decision_making`
# uses end-to-end on a real TTB ground-truth YAML (no LLM calls needed).
# ---------------------------------------------------------------------------


def test_end_to_end_trace_on_real_ttb_yaml():
    """Mirror exactly what main() + run_round() do for the ground-truth
    simulate block, using the real TTB YAML loaded from disk:

        gt_noise_rng = (
            np.random.default_rng(gt_seed) if gt_epsilon > 0 else None
        )
        experiment.simulate(
            ground_truth.theory,
            n_runs=REAL_N_SUBJECTS,
            action_noise=gt_epsilon,
            rng=gt_noise_rng,
        )

    With validities=[0.9, 0.5] and stimuli (A=[1,0], B=[0,1]) on every
    trial, TTB's top-validity cue always favors A, so noiseless TTB picks
    A 100% of the time. Under ε=0.5 epsilon-greedy noise the theoretical
    flip rate is ε/2 = 0.25 (half of ε fires, half of those land on B).
    """
    from src.theory import Theory

    ttb_yaml_path = Path(__file__).resolve().parents[1] / (
        "theories/heuristic_decision_making/ttb.yaml"
    )
    ttb_theory = Theory.from_yaml(str(ttb_yaml_path))

    # Dominance-design stimuli: A strictly dominates B on every cue, so
    # whichever feature TTB picks first (its subjective validities are
    # sampled per-subject from [0, 1] — NOT the experiment's validities)
    # the first consulted cue already favors A. Noiseless TTB therefore
    # picks A on every trial regardless of the random cue-ordering sample,
    # which is what we need to pin down a ground-truth flip rate.
    validities = [0.9, 0.5]
    n_unique = 8
    exp_clean = HeuristicDecisionMakingExperiment(
        validities=validities,
        rating_max=1,
        trial_a_ratings=[[1, 1]] * n_unique,
        trial_b_ratings=[[0, 0]] * n_unique,
    )
    exp_noisy = HeuristicDecisionMakingExperiment(
        validities=validities,
        rating_max=1,
        trial_a_ratings=[[1, 1]] * n_unique,
        trial_b_ratings=[[0, 0]] * n_unique,
    )

    # --- trace main()'s gt_noise_rng construction exactly --------------------
    gt_epsilon_clean = 0.0
    gt_epsilon_noisy = 0.5
    gt_seed = 7
    rng_clean = (
        np.random.default_rng(gt_seed) if gt_epsilon_clean > 0.0 else None
    )
    rng_noisy = (
        np.random.default_rng(gt_seed) if gt_epsilon_noisy > 0.0 else None
    )
    assert rng_clean is None, "noiseless path must skip RNG construction"
    assert rng_noisy is not None, "noisy path must materialize RNG"

    # --- trace run_round()'s simulate call exactly ---------------------------
    from src.run_config import REAL_N_SUBJECTS

    df_clean = exp_clean.simulate(
        ttb_theory,
        n_runs=REAL_N_SUBJECTS,
        action_noise=gt_epsilon_clean,
        rng=rng_clean,
    )
    df_noisy = exp_noisy.simulate(
        ttb_theory,
        n_runs=REAL_N_SUBJECTS,
        action_noise=gt_epsilon_noisy,
        rng=rng_noisy,
    )

    # Noiseless TTB on these stimuli must pick A every trial.
    assert (df_clean["response"] == 0).all(), (
        f"noiseless TTB must pick A (0) on every trial; "
        f"got {(df_clean['response'] == 0).mean():.3f} A-rate"
    )
    # Noisy TTB: expected flip rate ε/2 = 0.25. REAL_N_SUBJECTS * n_trials
    # rows; with flip-Bernoulli(0.25) SE is tiny. Allow ±0.05 slack.
    p1_noisy = float((df_noisy["response"] == 1).mean())
    assert abs(p1_noisy - 0.25) < 0.05, (
        f"expected ε=0.5 TTB flip rate ≈ 0.25, got {p1_noisy:.3f} "
        f"(n_rows={len(df_noisy)})"
    )


def test_action_noise_kwarg_only_threaded_from_hdm_main():
    """Static check: the only caller that passes `action_noise=` or `rng=` to
    `simulate(...)` is `main_heuristic_decision_making.py`. The pi/arbiter/
    feedback/observation/main.py paths must NOT thread noise — otherwise
    the competitor theories' internal Welch-test simulations would be
    noised and their discriminability check would be biased.

    This is a regression guard: if someone later adds the kwarg to another
    call site by accident, this fails loudly.
    """
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    # Scan production code only: everything under src/ plus the two main
    # entry-point scripts at the repo root. The tests/ tree is intentionally
    # excluded — its whole purpose is to exercise these kwargs.
    search_paths = [
        str(repo_root / "src"),
        str(repo_root / "main_heuristic_decision_making.py"),
        str(repo_root / "main.py"),
    ]
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-E", r"action_noise=|\brng=", *search_paths],
        capture_output=True,
        text=True,
    )
    # Only the definition (src/experiment.py) and the intended caller
    # (main_heuristic_decision_making.py) may mention the kwargs.
    allowed_prefixes = (
        str(repo_root / "src" / "experiment.py"),
        str(repo_root / "main_heuristic_decision_making.py"),
    )
    offending = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # grep -rn output format: "<path>:<lineno>:<content>"
        path = line.split(":", 1)[0]
        if any(path.startswith(p) for p in allowed_prefixes):
            continue
        offending.append(line)
    assert not offending, (
        "action_noise / rng kwargs must only be threaded in "
        "experiment.py (definition) and main_heuristic_decision_making.py "
        "(caller). Found unexpected call sites:\n  "
        + "\n  ".join(offending)
    )


def test_main_script_rejects_out_of_range_gt_epsilon():
    """End-to-end CLI check: the main script must reject --gt_epsilon
    outside [0, 1] at argument-parse time (before any LLM/theory load)."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "main_heuristic_decision_making.py"

    # Out-of-range values must exit non-zero.
    for bad_eps in ("1.5", "-0.1"):
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--gt_epsilon",
                bad_eps,
                "--run_id",
                "pytest_bad_eps",
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=20,
        )
        assert result.returncode != 0, (
            f"--gt_epsilon={bad_eps} should exit non-zero; "
            f"got rc={result.returncode}, stderr={result.stderr!r}"
        )
        assert "--gt_epsilon must be in [0, 1]" in result.stderr, (
            f"expected range-error on stderr for --gt_epsilon={bad_eps}; "
            f"got stderr={result.stderr!r}"
        )
