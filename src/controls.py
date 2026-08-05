"""Replacement controls — targeted component substitutions.

Original pipeline modules are untouched; opt-in via main_ablation_binary.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.jsd import _per_trial_bernoulli, _per_trial_lag1, choice_matrix, jsd
from src.logger import log
from src.metric import Metric
from src.observation import Observation, Observations
from src.pi import AutoPi
from src.prompts.experiment_proposal import _format_ledger

if TYPE_CHECKING:
    from src.experiment import Experiment
    from src.theory import Theory


NEUTRAL_SYSTEM_PROMPT = """\
You are a neutral experimental designer in computational cognitive science, \
working in the {domain} domain.

You are given two candidate theories of decision making, THEORY 1 and THEORY 2, \
each operationalized as code. Your only goal is to design an experiment whose \
data will best distinguish the two theories — you have no stake in either theory.

A useful proposal targets a *quantitative* dissociation between the two theories — \
how they respond differently to specific stimuli in addition to differences in overall \
performance.
"""


NEUTRAL_TEMPLATE = """\
## EXPERIMENTAL DOMAIN
{experiment_description}

{design_header}

Subjects see the following instructions:
{introduction_text}

## THEORY 1
{theory_1_description}

## THEORY 2
{theory_2_description}

## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
{ledger}

## RESPONSE FORMAT
Design one new experiment that maximally distinguishes THEORY 1 from THEORY 2.
Return a JSON object with the following fields:
{instruction_format}
"""


def render_neutral_experiment_proposal(
    *,
    experiment_class: type["Experiment"],
    theory_1: "Theory",
    theory_2: "Theory",
    ledger: list["Experiment"] | None = None,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for proposing an experiment with a
    neutral framing: "distinguish THEORY 1 from THEORY 2" instead of the
    original advocacy framing. Mirrors `src.prompts.experiment_proposal.render`
    in everything else (domain header, design header, instructions, ledger,
    response schema).
    """
    system_prompt = NEUTRAL_SYSTEM_PROMPT.format(domain=experiment_class.name)

    user_prompt = NEUTRAL_TEMPLATE.format(
        experiment_description=experiment_class.description,
        design_header=experiment_class.pretty_print_header(),
        introduction_text=experiment_class.introduction_text,
        theory_1_description=theory_1.pretty_print(),
        theory_2_description=theory_2.pretty_print(),
        ledger=_format_ledger(ledger or []),
        instruction_format=experiment_class.instruction_format(),
    )
    return system_prompt, user_prompt


class NeutralAutoPi(AutoPi):
    """neutral_proposer ablation: same metric + Welch acceptance machinery
    as control (inherited propose_round); only the experiment-proposal
    prompt is swapped for a neutral, non-advocative framing."""

    def _llm_propose_experiment(
        self,
        adversary: "AutoPi",
        *,
        ledger: list["Experiment"],
        workspace: Path | None = None,
        log_label: str | None = None,
    ) -> "Experiment":
        log(f"[neutral] proposing experiment ({self.label} slot, neutral framing)")
        system_prompt, user_prompt = render_neutral_experiment_proposal(
            experiment_class=self.experiment_class,
            theory_1=self._theory,
            theory_2=adversary.theory,
            ledger=ledger,
        )
        parsed = self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=self.experiment_class,
            workspace=workspace,
            log_label=log_label,
        )
        return self.experiment_class(**parsed.model_dump())


def make_projection_metric(experiment, m_self: np.ndarray, m_adv: np.ndarray) -> Metric:
    """A fixed (non-LLM) Metric: each subject's choices projected onto the
    discriminant profile `w = P_self(B) − P_adv(B)`, computed at proposal time
    from the two theories' simulated choice matrices on `experiment`.

    Weights are keyed by trial-pair CONTENT (option_a_ratings,
    option_b_ratings), NOT trial position: every (re)construction of an
    experiment reshuffles its trial order (`model_post_init`, unseeded), and
    this metric is persisted and re-applied to data simulated from
    reconstructed instances (crash-resume, backfill, rescoring). Position-
    keyed weights would silently misalign there; content-keyed weights are
    order-invariant. Scalar + per-subject variance, so every downstream
    consumer (set_data, backfill, distances, Welch) works unchanged."""
    p_self = _per_trial_bernoulli(m_self)[:, 1]
    p_adv = _per_trial_bernoulli(m_adv)[:, 1]
    w_by_pair: dict[str, list[float]] = {}
    for (a, b), ws, wa in zip(experiment._trials, p_self, p_adv):
        w_by_pair.setdefault(str((tuple(a), tuple(b))), []).append(float(ws - wa))
    weights = {k: float(np.mean(v)) for k, v in w_by_pair.items()}
    source = (
        f"W = {weights!r}\n"
        "def metric(data):\n"
        "    import numpy as np\n"
        "    vals = []\n"
        "    for _, subj in data.groupby('subject_id', sort=False):\n"
        "        keys = [\n"
        "            str((tuple(a), tuple(b)))\n"
        "            for a, b in zip(subj['option_a_ratings'], subj['option_b_ratings'])\n"
        "        ]\n"
        "        w = np.array([W.get(k, 0.0) for k in keys])\n"
        "        r = subj['response'].to_numpy(dtype=float)\n"
        "        vals.append(float(np.mean(r * w)))\n"
        "    return float(np.mean(vals))\n"
    )
    return Metric(
        metric_source=source,
        rationale=(
            "Auto-generated projection metric (jsd_metric control): mean "
            "per-subject projection of choices onto the per-trial choice-"
            "probability difference between the two proposing theories."
        ),
    )


def _conditional_choice_profile(content_keys: list[str], m: np.ndarray) -> dict[str, float]:
    """P(response==1 | content_t, response_{t-1}) aggregated over all trials
    t>=1 and all subjects of a simulated choice matrix `m`. Keys are
    "<content>|<prev>" — content-keyed (not position-keyed) because every
    experiment (re)construction reshuffles trial order."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n_trials = m.shape[1]
    for t in range(1, n_trials):
        ckey = content_keys[t]
        prev_col, curr_col = m[:, t - 1], m[:, t]
        for prev in (0, 1):
            mask = prev_col == prev
            k = f"{ckey}|{prev}"
            sums[k] = sums.get(k, 0.0) + float(curr_col[mask].sum())
            counts[k] = counts.get(k, 0) + int(mask.sum())
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def make_sequence_projection_metric(experiment, m_self: np.ndarray, m_adv: np.ndarray) -> Metric:
    """History-aware analog of make_projection_metric.

    Keys the discriminant on the STATE these theories act in: (current trial
    content, previous response). Weight for a state is
    `P_self(B | content, prev) − P_adv(B | content, prev)`, estimated from the
    two theories' simulated choice matrices on `experiment`. The metric
    projects each subject's choices onto these weights.

    This subsumes the static projection: for stimulus-driven theories the
    prev-response dependence is ~flat (so it reduces to the content metric),
    while for history-driven theories (perseveration/alternation) the content
    dependence is ~flat but the prev-response term carries all the signal.

    Unlike make_projection_metric, this is intentionally NOT trial-order
    invariant — it reads `response_{t-1}`, so it depends on each subject's
    realized trial sequence (preserved on disk in trial order). That is the
    price of seeing history, and it is required to score history-dependent
    ground truths at all."""
    content_keys = [str((tuple(a), tuple(b))) for a, b in experiment._trials]
    p_self = _conditional_choice_profile(content_keys, m_self)
    p_adv = _conditional_choice_profile(content_keys, m_adv)
    weights = {
        k: float(p_self.get(k, 0.5) - p_adv.get(k, 0.5))
        for k in set(p_self) | set(p_adv)
    }
    source = (
        f"W = {weights!r}\n"
        "def metric(data):\n"
        "    import numpy as np\n"
        "    vals = []\n"
        "    for _, subj in data.groupby('subject_id', sort=False):\n"
        "        a = list(subj['option_a_ratings'])\n"
        "        b = list(subj['option_b_ratings'])\n"
        "        r = subj['response'].to_numpy(dtype=int)\n"
        "        acc = []\n"
        "        for t in range(1, len(r)):\n"
        "            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))\n"
        "            acc.append(float(r[t]) * W.get(key, 0.0))\n"
        "        vals.append(float(np.mean(acc)) if acc else 0.0)\n"
        "    return float(np.mean(vals))\n"
    )
    return Metric(
        metric_source=source,
        rationale=(
            "Auto-generated history-aware projection metric (jsd_metric "
            "control): per-subject projection of choices onto the "
            "(content, previous-response)-conditioned choice-probability "
            "difference between the two proposing theories."
        ),
    )


def make_jsd_to_self_metric(experiment, m_self: np.ndarray) -> Metric:
    """An absolute (data-anchored-scale) Metric matching the baseline LLM
    metric's structure: defined at PROPOSAL time from the proposing theory's
    simulations alone, frozen thereafter, and computed from the input data
    alone.

    metric(data) = count-weighted mean, over the (content, prev-response)
    states visited by `data`, of the binary Jensen-Shannon divergence between
    the data's conditional choice probability P̂(B | state) and the proposing
    theory's baked one. Scale [0, ln 2]; 0 = behaves exactly like the
    proposing theory.

    Unlike the projection metric (weights = P_self − P_adv, which annihilate
    behavior the two pool theories SHARE), nothing is projected out: any
    deviation of the data from the proposing theory, in any direction, raises
    the value. On this metric the proposer predicts ≈ 0 (self-JSD floor) and
    the adversary predicts ≈ the acceptance JSD, so the acceptance gate
    doubles as the baseline's "do the theories separate on this metric?"
    check. States unseen in the baked simulations default to 0.5
    (uninformative), mirroring make_sequence_projection_metric."""
    content_keys = [str((tuple(a), tuple(b))) for a, b in experiment._trials]
    profile = _conditional_choice_profile(content_keys, m_self)
    source = (
        f"P_REF = {profile!r}\n"
        "def metric(data):\n"
        "    import numpy as np\n"
        "    def jsd2(p, q):\n"
        "        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)\n"
        "        v = 0.0\n"
        "        for x, y in ((1.0 - p, 1.0 - q), (p, q)):\n"
        "            m = 0.5 * (x + y)\n"
        "            if x > 0:\n"
        "                v += 0.5 * x * np.log(x / m)\n"
        "            if y > 0:\n"
        "                v += 0.5 * y * np.log(y / m)\n"
        "        return float(v)\n"
        "    sums, counts = {}, {}\n"
        "    for _, subj in data.groupby('subject_id', sort=False):\n"
        "        a = list(subj['option_a_ratings'])\n"
        "        b = list(subj['option_b_ratings'])\n"
        "        r = subj['response'].to_numpy(dtype=int)\n"
        "        for t in range(1, len(r)):\n"
        "            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))\n"
        "            sums[key] = sums.get(key, 0) + int(r[t])\n"
        "            counts[key] = counts.get(key, 0) + 1\n"
        "    num = den = 0.0\n"
        "    for k, n in counts.items():\n"
        "        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))\n"
        "        den += n\n"
        "    return float(num / den) if den else 0.0\n"
    )
    return Metric(
        metric_source=source,
        rationale=(
            "Auto-generated JSD-to-self metric (jsd_metric control): "
            "sequence-aware Jensen-Shannon divergence (nats, 0 to ln 2) "
            "between the dataset's conditional choice profile and the "
            "proposing theory's, over (trial content, previous response) "
            "states. 0 means the data behaves exactly like the proposing "
            "theory; ln 2 means maximally different."
        ),
    )


class JsdAutoPi(AutoPi):
    """jsd_metric control: pis propose experiments via the SAME LLM prompt
    as baseline, but acceptance replaces LLM-metric + Welch with
    sequence-aware JSD > threshold, and the observation's metric is the
    auto-generated JSD-to-self metric — absolute divergence from the
    proposing theory's conditional choice profile, structurally matching
    the baseline metric (frozen at proposal time, computed from the data
    alone) but with zero metric-proposal LLM calls."""

    # jsd_n_runs MUST match the n_runs used by calibrate_jsd_threshold.py:
    # plug-in JSD is upward-biased at finite samples, and the bias only
    # cancels in `estimate > threshold` when both sides are estimated at
    # the same sample size. Do not tune this down.
    def __init__(self, *, jsd_threshold: float, jsd_n_runs: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.jsd_threshold = jsd_threshold
        self.jsd_n_runs = jsd_n_runs

    def propose_round(
        self,
        adversary: AutoPi,
        pool: Observations,
        *,
        workspace: Path | None = None,
        n_runs: int = 50,
        max_experiments: int = 3,
        **_ignored,  # max_metrics / alpha / real_n_subjects are baseline-only knobs
    ) -> Observation:
        log(f"[jsd_metric] proposing round for {self.label} vs {adversary.label}")
        rejected: list["Experiment"] = []
        last: Observation | None = None
        for k_exp in range(max_experiments):
            experiment = self._llm_propose_experiment(
                adversary,
                ledger=list(pool.experiments) + rejected,
                workspace=workspace,
                log_label=f"experiment_attempt_{k_exp:02d}",
            )
            # Mirror baseline AutoPi.propose_round: a single bad experiment
            # (degenerate design that makes simulation/JSD/metric throw) must
            # be rejected, not crash a multi-round run. Catch, log, reject,
            # and try the next proposal.
            try:
                m_self = choice_matrix(self._theory, experiment, n_runs=self.jsd_n_runs)
                m_adv = choice_matrix(adversary.theory, experiment, n_runs=self.jsd_n_runs)
                seq = float(np.mean([
                    jsd(a, b)
                    for a, b in zip(_per_trial_lag1(m_self), _per_trial_lag1(m_adv))
                ]))
                accepted = seq > self.jsd_threshold
                print(
                    f"[jsd_metric exp{k_exp}] sequence_jsd={seq:.4f} "
                    f"threshold={self.jsd_threshold:.4f} "
                    f"-> {'ACCEPT' if accepted else 'reject'}"
                )
                # JSD-to-self: absolute divergence from the proposing
                # theory's conditional choice profile, matching the baseline
                # metric's structure (frozen at proposal time, computed from
                # the data alone). On it the proposer predicts ~0 and the
                # adversary predicts ~`seq`, so acceptance == "the theories
                # separate on this metric". Replaced the projection metric,
                # whose pool-anchored weights were blind to pool-shared
                # errors (anti_majority, run 1).
                metric = make_jsd_to_self_metric(experiment, m_self)
                obs = Observation(
                    experiment=experiment,
                    metric=metric,
                    proposer_theory=self._theory,
                    proposer_label=self.label,
                )
                est_self = metric(experiment.simulate(self._theory, n_runs=n_runs))
                est_adv = metric(experiment.simulate(adversary.theory, n_runs=n_runs))
                obs.record_prediction(
                    est_self.value, variance=est_self.variance, label=self.label
                )
                obs.record_prediction(
                    est_adv.value, variance=est_adv.variance, label=adversary.label
                )
            except Exception as e:
                print(
                    f"[jsd_metric exp{k_exp}] eval failed "
                    f"({type(e).__name__}: {e}); rejecting."
                )
                rejected.append(experiment)
                continue
            last = obs
            if accepted:
                return last
            rejected.append(experiment)
        if last is None:
            # Every proposal failed JSD evaluation (pathological). Fall back to
            # the baseline metric+Welch path so the round still yields a valid
            # Observation instead of crashing the run.
            print(
                "[jsd_metric] all proposals failed JSD eval; "
                "falling back to baseline propose_round."
            )
            return super().propose_round(
                adversary, pool, workspace=workspace,
                n_runs=n_runs, max_experiments=max_experiments,
            )
        print(
            f"[jsd_metric] no design beat threshold after {max_experiments} "
            f"attempts; returning last"
        )
        return last
