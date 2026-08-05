"""Barebone component substitutions for ablation conditions (lesions /
no-intelligence baselines). Replacement controls live in src/controls.py.
Original pipeline modules are untouched; everything here is opt-in via
main_ablation_binary.py."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from src.controls import make_jsd_to_self_metric
from src.decision_making_binary_features.experiment import DecisionMakingBinaryExperiment
from src.jsd import choice_matrix
from src.logger import log
from src.metric import Metric
from src.observation import Observation, Observations
from src.pi import AutoPi
from src.run_config import REAL_N_SUBJECTS


def random_design(
    rng: np.random.Generator, *, n_features: int = 4, n_pairs: int = 16
) -> DecisionMakingBinaryExperiment:
    """A valid random binary design: random validities with a spread,
    random non-identical rating pairs."""
    validities = np.round(rng.uniform(0.55, 0.95, n_features), 2)
    validities[0] = 0.95  # guarantee a spread (spec: avoid uniform validities)
    validities[-1] = 0.55
    pairs_a, pairs_b = [], []
    while len(pairs_a) < n_pairs:
        a = rng.integers(0, 2, n_features).tolist()
        b = rng.integers(0, 2, n_features).tolist()
        if a != b:
            pairs_a.append(a)
            pairs_b.append(b)
    return DecisionMakingBinaryExperiment(
        validities=validities.tolist(),
        trial_a_ratings=pairs_a,
        trial_b_ratings=pairs_b,
    )


def _derive_rng(seed: int, label: str, round_idx: int) -> np.random.Generator:
    """Reproducible, call-order-independent rng for one (seed, slot, round).

    Both slots share `round_idx` within a round, so the slot `label`
    differentiates them; `round_idx = len(pool)` grows across rounds, so each
    round draws fresh designs. Order-independent because it depends only on the
    three inputs, not on any prior draws."""
    label_hash = zlib.crc32(label.encode())
    seed_seq = np.random.SeedSequence([int(seed), int(label_hash), int(round_idx)])
    return np.random.default_rng(seed_seq)


class BlindAutoPi(AutoPi):
    """blind_design ablation: experiments are random valid designs synthesized
    programmatically (`random_design`) — no LLM, no theory conditioning, no
    adaptivity. A design is accepted iff valid + simulable, which holds by
    construction. The measurement metric is the baseline's theory-aware LLM
    metric (first valid kept; NO Welch gate), so all downstream machinery
    (set_data, backfill, arbitration, leaderboard) is unchanged. The
    experiment-level loop redraws a fresh design only when every metric
    proposal errors on the current one."""

    def __init__(self, *, seed: int, n_features: int = 4, n_pairs: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.n_features = n_features
        self.n_pairs = n_pairs

    def propose_round(
        self,
        adversary: AutoPi,
        pool: Observations,
        *,
        workspace: Path | None = None,
        n_runs: int = 50,
        max_experiments: int = 3,
        max_metrics: int = 4,
        real_n_subjects: int = REAL_N_SUBJECTS,
        alpha: float = 0.01,
        **_ignored,  # accepted for orchestrator-call compatibility; unused here
    ) -> Observation:
        log(f"[blind_design] proposing round for {self.label} vs {adversary.label}")
        round_idx = len(pool)
        rng = _derive_rng(self.seed, self.label, round_idx)
        failed_metric_attempts: list[tuple[Metric, str]] = []
        last: Observation | None = None

        for k_exp in range(max_experiments):
            # Fresh random valid design each iteration (rng is stateful).
            # valid + simulable is automatic, so simulate is NOT guarded — an
            # error here is a real bug and should surface loudly.
            experiment = random_design(
                rng, n_features=self.n_features, n_pairs=self.n_pairs
            )
            data_self = experiment.simulate(self._theory, n_runs=n_runs)
            data_adv = experiment.simulate(adversary.theory, n_runs=n_runs)

            for k_metric in range(max_metrics):
                metric = self._llm_propose_metric(
                    adversary,
                    experiment,
                    ledger=failed_metric_attempts,
                    real_n_subjects=real_n_subjects,
                    alpha=alpha,
                    workspace=workspace,
                    log_label=f"metric_exp{k_exp:02d}_attempt_{k_metric:02d}",
                )
                # Coerce values to float INSIDE the try (like the baseline,
                # src/pi.py): a malformed LLM metric can return a non-scalar
                # (ndarray/Series/str). Coercing here turns that into a caught
                # rejection instead of an uncaught ValidationError at
                # record_prediction time, which would crash the whole run.
                try:
                    est_self = metric(data_self.copy())
                    est_adv = metric(data_adv.copy())
                    v_self = float(est_self.value) if est_self.value is not None else None
                    v_adv = float(est_adv.value) if est_adv.value is not None else None
                    var_self = est_self.variance
                    var_adv = est_adv.variance
                    outcome = None
                except Exception as e:
                    v_self = v_adv = None
                    var_self = var_adv = None
                    outcome = f"eval error: {type(e).__name__}: {e}"
                valid = v_self is not None and v_adv is not None

                last = Observation(
                    experiment=experiment,
                    metric=metric,
                    proposer_theory=self._theory,
                    proposer_label=self.label,
                )
                last.record_prediction(v_self, variance=var_self, label=self.label)
                last.record_prediction(v_adv, variance=var_adv, label=adversary.label)
                if valid:
                    print(
                        f"[blind_design exp{k_exp} metric{k_metric}] "
                        f"self={v_self:.4f} adv={v_adv:.4f} -> ACCEPT (no Welch gate)"
                    )
                    return last
                failed_metric_attempts.append((metric, outcome or "metric returned None"))

            print(
                f"[blind_design exp{k_exp}] all {max_metrics} metrics errored; "
                f"redrawing a fresh random design"
            )

        assert last is not None  # max_experiments >= 1 and max_metrics >= 1
        print(
            f"[blind_design] no metric evaluated validly after "
            f"{max_experiments} designs × {max_metrics} metrics; returning last"
        )
        return last


class BlindJsdAutoPi(AutoPi):
    """blind_design_jsd ablation: random valid designs (like `BlindAutoPi` —
    no LLM, no theory conditioning, no adaptivity) PAIRED WITH the fixed
    lag-1 JSD-to-self metric (like `JsdAutoPi`).

    There is NO acceptance gate: the first valid random design is kept and
    measured, mirroring `BlindAutoPi`'s no-selection philosophy. The metric is
    built IDENTICALLY to `JsdAutoPi` (`make_jsd_to_self_metric` on the proposing
    theory's choice matrix at `jsd_n_runs`), so downstream scoring is directly
    comparable. The ONLY differences from `JsdAutoPi` are the design source
    (random vs LLM) and the absence of the JSD threshold gate — so the
    blind/jsd pair isolates whether adaptive/LLM design buys anything on top of
    the JSD metric. A fresh design is redrawn only when the current one makes
    simulation/metric evaluation error (a degenerate draw), never for failing
    to separate the theories."""

    def __init__(self, *, seed: int, n_features: int = 4, n_pairs: int = 16,
                 jsd_n_runs: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.n_features = n_features
        self.n_pairs = n_pairs
        self.jsd_n_runs = jsd_n_runs

    def propose_round(
        self,
        adversary: AutoPi,
        pool: Observations,
        *,
        workspace: Path | None = None,
        n_runs: int = 50,
        max_experiments: int = 3,
        **_ignored,  # jsd_threshold / max_metrics / alpha are not used here
    ) -> Observation:
        log(f"[blind_design_jsd] proposing round for {self.label} vs {adversary.label}")
        round_idx = len(pool)
        rng = _derive_rng(self.seed, self.label, round_idx)

        for k_exp in range(max_experiments):
            # Fresh random valid design each iteration (rng is stateful). The
            # design is valid + simulable by construction; an error in the JSD
            # pipeline means a degenerate draw, so redraw rather than crash a
            # multi-round run (mirrors BlindAutoPi's redraw-on-error).
            experiment = random_design(
                rng, n_features=self.n_features, n_pairs=self.n_pairs
            )
            try:
                m_self = choice_matrix(self._theory, experiment, n_runs=self.jsd_n_runs)
                metric = make_jsd_to_self_metric(experiment, m_self)
                est_self = metric(experiment.simulate(self._theory, n_runs=n_runs))
                est_adv = metric(experiment.simulate(adversary.theory, n_runs=n_runs))
                obs = Observation(
                    experiment=experiment,
                    metric=metric,
                    proposer_theory=self._theory,
                    proposer_label=self.label,
                )
                obs.record_prediction(
                    est_self.value, variance=est_self.variance, label=self.label
                )
                obs.record_prediction(
                    est_adv.value, variance=est_adv.variance, label=adversary.label
                )
                # Inside the try (like JsdAutoPi): if a degenerate metric ever
                # yields a non-float value, the format raises here and is
                # caught below -> redraw, rather than crashing the round.
                print(
                    f"[blind_design_jsd exp{k_exp}] "
                    f"self={est_self.value:.4f} adv={est_adv.value:.4f} "
                    f"-> ACCEPT (no separation gate)"
                )
            except Exception as e:
                print(
                    f"[blind_design_jsd exp{k_exp}] eval failed "
                    f"({type(e).__name__}: {e}); redrawing a fresh random design"
                )
                continue
            return obs

        # Every random design errored in the JSD pipeline (pathological — valid
        # designs are simulable by construction). Fail loudly rather than
        # silently returning a malformed observation.
        raise RuntimeError(
            f"[blind_design_jsd] all {max_experiments} random designs errored "
            f"in JSD evaluation for {self.label}"
        )
