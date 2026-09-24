# generation_iter_02_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Decision Making (Binary Features) domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across the majority of experiments. 
You will see a list of theories that have been proposed in the past but you should only use them as inspiration and not to choose from them. Propose a new theory that is different. 
If they fail to do so, you will receive feedback on their performance on the same experiment(s) and you will have to propose another new theory and model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the candidate's `aggregate_loss` is compared against the running-best loss; strict improvement -> ACCEPTED (the candidate becomes the new running-best base); otherwise -> REJECTED (the candidate is discarded and the base is unchanged). You do NOT need to manually "revert" a regressed edit — the gate already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. Use this as ground truth on which past critic advice actually moved the loop forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight critic advice whose previous candidates were REJECTED, and reinforce / extend advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit should plausibly land at-or-below it, otherwise the gate will reject your attempt and the base stays put.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) to a *string* value that specifies its domain. Every value MUST be a string — never a bare list, number, tuple, or expression. Use exactly one of these notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{v1, v2, ...}"
   Example: "{1, 2}"

3. Vector of intervals whose length is set by the experiment — a bracketed tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), used when the parameter takes its value from an experiment-defined constant rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {2, 3}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in `parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent new identifier names. If a parameter's shape doesn't fit any of those variables, fall back to a literal interval (notation 1) or discrete set (notation 2). Use these names so the model adapts to any experiment in this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
- n_features: Number of expert ratings per option (LLM-proposed via `validities` length).
- validities: Per-expert validities (LLM-proposed; each in [0.5, 1.0]); fixed across all trials.

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared `parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is domain-specific — see the experiment description above and the `history` key list below, which mirrors the per-trial variables carried in `state`). Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a list-of-dicts. The per-trial keys are:
  Each value below is a Python list in trial order; entry `i` is the value for trial `i`. On the first trial all lists are empty.
  - `"option_a_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
  - `"option_b_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Propose a validity-weighted inverse-clean-count theory without an orientation mixture, without a top-cue bonus, and with exchangeable tied cues. All subjects treat an expert rating of 0 as clean/favorable and 1 as a defect, preserving the Experiment 1 result of zero positively valenced subjects. Every discriminating cue contributes continuous weighted evidence rather than entering an all-or-none tally: option A gains w_j when feature j has A=0/B=1 and loses w_j when A=1/B=0. Subject-specific weights are monotone, heavy-tailed transforms of communicated validity, e.g., w_j = exp(a_i * logit(v_j) + b_i), so high-validity cues dominate when validity gaps are clear but low-validity cues can still matter enough to produce large individual differences on difficult designs like Experiment 10. Choice is a softmax over net evidence with subject-specific gain and lapse/epsilon noise. When net evidence is near zero or no cue discriminates, subjects inspect high-validity experts in order; if several experts share the highest validity, the order is randomly permuted per trial/subject or their weight is split equally, avoiding the stable-sorting artifact that inflates top-cue effects in Experiment 2. This new theory should keep pi_5's successful universal inverse-valence and variance behavior while replacing inclusion gating and the deterministic tie-break with continuous weighting and tied-cue exchangeability.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.760)

**Description**
Subjects carry a latent valence orientation, with roughly 86% treating a binary expert rating of 1 as a defect and 0 as clean, while the remainder use the opposite positive valence. On each trial, each feature is probabilistically gated into an unweighted tally according to a validity logistic whose top-validity cue is included with high probability, while lower-validity cues are included often enough for their majorities to matter. Tally choices follow a softmax with subject-specific gain. Exact zero tallies are resolved by scanning only included cues in descending validity order and choosing the first valence-consistent discriminator. In addition, when the included tally is just one cue away from zero, the single highest-validity included discriminator receives a small attention bonus inside the tally, making the top cue somewhat more decisive without overriding larger lower-cue majorities. Subject-level parameters are drawn from narrowed heavy-tailed or logit-normal distributions to keep between-subject variance modest where the data are consistent while preserving heterogeneity where it is large.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import norm


def _trunc_qnorm(u, mu, sigma, lo, hi):
    u = float(np.clip(u, 0.0, 1.0))
    q_lo = norm.cdf((lo - mu) / sigma)
    q_hi = norm.cdf((hi - mu) / sigma)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * norm.ppf(q)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # Subject-level transforms: raw uniform draws are mapped through
    # truncated normal / log-normal / logit-normal distributions.
    u_alpha = (float(parameters['alpha_raw']) - 0.01) / 0.98
    u_alpha = float(np.clip(u_alpha, 0.0, 1.0))
    alpha = float(np.exp(_trunc_qnorm(
        u_alpha, np.log(6.5), 0.20, np.log(5.5), np.log(8.5))))

    # Live gamma for the validity gate.  It is not overwritten below; only
    # the single highest-validity cue's inclusion probability is pinned high.
    u_gamma = (float(parameters['gamma_raw']) - 0.01) / 0.98
    u_gamma = float(np.clip(u_gamma, 0.0, 1.0))
    gamma = float(_trunc_qnorm(u_gamma, -2.95, 0.35, -3.85, -2.45))

    u_bt = (float(parameters['beta_tally_raw']) - 0.01) / 0.98
    u_bt = float(np.clip(u_bt, 0.0, 1.0))
    beta_tally = float(np.exp(_trunc_qnorm(
        u_bt, np.log(1.85), 0.28, np.log(1.20), np.log(2.90))))

    u_tb = (float(parameters['beta_tiebreak_raw']) - 0.01) / 0.98
    u_tb = float(np.clip(u_tb, 0.0, 1.0))
    beta_tiebreak = float(np.exp(_trunc_qnorm(
        u_tb, np.log(2.00), 0.35, np.log(1.20), np.log(3.20))))

    u_eps = (float(parameters['epsilon_raw']) - 0.01) / 0.98
    u_eps = float(np.clip(u_eps, 0.0, 1.0))
    logit_lo = np.log(0.03 / 0.97)
    logit_hi = np.log(0.16 / 0.84)
    logit_mu = np.log(0.07 / 0.93)
    z_eps = _trunc_qnorm(u_eps, logit_mu, 0.45, logit_lo, logit_hi)
    epsilon = float(1.0 / (1.0 + np.exp(-z_eps)))

    # Small top-cue attention bonus, used only for near-zero tally margins.
    u_top = (float(parameters['top_bonus_raw']) - 0.01) / 0.98
    u_top = float(np.clip(u_top, 0.0, 1.0))
    extra = float(_trunc_qnorm(u_top, 0.40, 0.15, 0.00, 0.80))
    w_top = 1.0 + max(0.0, min(0.80, extra))

    # Latent valence orientation.  Approximately 86% use inverse valence.
    inv = float(parameters['orientation_u']) < 0.86

    # Inclusion probabilities from a validity logistic gate.
    # Only the highest-validity cue's inclusion probability is forced high.
    logit_j = alpha * val + gamma
    p_j = _sigmoid(logit_j)
    if n_features > 0:
        j_top = int(np.argmax(val))
        p_j[j_top] = float(np.clip(p_j[j_top], 0.95, 0.998))
    p_j = np.clip(p_j, 0.02, 0.99)

    included = np.flatnonzero(np.random.uniform(size=n_features) < p_j)
    if included.size == 0:
        return np.ones(2) / 2.0

    # Valence-consistent cue comparison on the included cues only.
    if inv:
        good_a = (a == 0) & (b == 1)
        good_b = (a == 1) & (b == 0)
    else:
        good_a = (a == 1) & (b == 0)
        good_b = (a == 0) & (b == 1)

    d_raw = float(np.sum(good_a[included]) - np.sum(good_b[included]))
    d = d_raw

    # Top-cue attention bonus.  It operates inside the tally path only, and
    # only when the current tally is one cue away from zero.  Larger
    # lower-cue majorities are not overridden, and exact-zero ties still
    # go to the validity-ordered tie-break scan.
    if d_raw != 0.0 and abs(d_raw) <= 1.0:
        disc = [j for j in included if a[j] != b[j]]
        if disc:
            j_att = max(disc, key=lambda j: float(val[j]))
            if inv:
                if a[j_att] == 0 and b[j_att] == 1:
                    d = d_raw + (w_top - 1.0)
                elif a[j_att] == 1 and b[j_att] == 0:
                    d = d_raw - (w_top - 1.0)
            else:
                if a[j_att] == 1 and b[j_att] == 0:
                    d = d_raw + (w_top - 1.0)
                elif a[j_att] == 0 and b[j_att] == 1:
                    d = d_raw - (w_top - 1.0)

    if d != 0.0:
        scores = np.array([d, -d], dtype=float)
        beta = beta_tally
    else:
        discriminating = [j for j in included if a[j] != b[j]]
        if len(discriminating) == 0:
            return np.ones(2) / 2.0

        discriminating = sorted(discriminating, key=lambda j: -val[j])
        winner = None
        for j in discriminating:
            if inv:
                if a[j] == 0 and b[j] == 1:
                    winner = 0
                    break
                elif a[j] == 1 and b[j] == 0:
                    winner = 1
                    break
            else:
                if a[j] == 1 and b[j] == 0:
                    winner = 0
                    break
                elif a[j] == 0 and b[j] == 1:
                    winner = 1
                    break

        if winner is None:
            return np.ones(2) / 2.0

        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = beta_tiebreak

    logits = beta * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- orientation_u: [0, 1]
- alpha_raw: [0.01, 0.99]
- gamma_raw: [0.01, 0.99]
- beta_tally_raw: [0.01, 0.99]
- beta_tiebreak_raw: [0.01, 0.99]
- epsilon_raw: [0.01, 0.99]
- top_bonus_raw: [0.01, 0.99]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.5117 (var=0.0161) vs this=-0.3700 (var=0.2536)
- Experiment 2: real=-0.6438 (var=0.0430) vs this=-0.6238 (var=0.2467)
- Experiment 3: real=-0.4300 (var=0.0221) vs this=-0.3550 (var=0.1762)
- Experiment 4: real=0.4925 (var=0.0658) vs this=0.4681 (var=0.1627)
- Experiment 5: real=0.2658 (var=0.1584) vs this=0.3300 (var=0.1102)
- Experiment 6: real=0.0867 (var=0.4244) vs this=0.0358 (var=0.0678)
- Experiment 7: real=0.0725 (var=0.0580) vs this=0.1800 (var=0.1187)
- Experiment 8: real=1.0100 (var=2.8518) vs this=1.1075 (var=2.7433)
- Experiment 9: real=0.3050 (var=0.0605) vs this=0.1742 (var=0.1020)
- Experiment 10: real=-0.1667 (var=0.0053) vs this=-0.1839 (var=0.0030)
- Experiment 11: real=0.0000 (var=0.0045) vs this=0.0400 (var=0.0007)
- Experiment 12: real=1.2973 (var=0.2536) vs this=1.9342 (var=0.0366)


---

### `pi_4_1` (overall score: 0.634)

**Description**
People invert the valence of binary expert ratings: a 1 is treated as a costly defect and a 0 as a clean, favorable signal. Across all features they compute an inverse-tally advantage for A, weakly weighting the inverse cues by the experts' stated validities. When the raw inverse tally is nonzero, choice is a noisy softmax over the two weakly validity-weighted inverse tallies. When the raw inverse tally is exactly tied, people do not guess uniformly; instead they break the tie lexicographically by consulting features in descending validity order and choosing the option that has a clean 0 on the first discriminating feature. Response noise is captured by a softmax plus an independent uniform lapse. This differs from Take The Best because it aggregates all cues rather than stopping at the first discriminator, and from plain inverse tallying because ties are resolved by validity-ordered clean-cue use instead of guessing.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import norm


def _trunc_qnorm(u, mu, sigma, lo, hi):
    """Map a (0,1) draw to a normal quantile truncated to [lo, hi]."""
    u = float(np.clip(u, 0.0, 1.0))
    q_lo = norm.cdf((lo - mu) / sigma)
    q_hi = norm.cdf((hi - mu) / sigma)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * norm.ppf(q)


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # --- Subject-level parameter transforms ---------------------------------
    # Raw declared ranges are mapped through truncated normal/lognormal
    # distributions so the intended medians and dispersions are realized
    # per subject rather than sampled uniformly.
    beta_tally_raw = float(parameters['beta_tally'])
    beta_tiebreak_raw = float(parameters['beta_tiebreak'])
    epsilon_raw = float(parameters['epsilon'])
    validity_weight_raw = float(parameters['validity_weight'])

    # Lower median inverse temperature to soften the inverse-tally gradient,
    # especially the s = +/-2 separation in Experiment 4, while retaining a
    # moderately wide support for between-subject variance.
    u_beta = (beta_tally_raw - 0.22) / (0.85 - 0.22)
    beta_tally = np.exp(_trunc_qnorm(
        u_beta, np.log(0.45), 0.40, np.log(0.22), np.log(0.85)))
    beta_tally = float(np.clip(beta_tally, 0.22, 0.85))

    # Keep tie-break temperature near 1.0 but widen the subject-level range.
    # The median is nudged to 1.02 to hold Experiment 3 near its observed
    # anti-TTB tie-break rate once the lapse distribution is widened.
    u_tie = (beta_tiebreak_raw - 0.50) / (1.70 - 0.50)
    beta_tiebreak = np.exp(_trunc_qnorm(
        u_tie, np.log(1.02), 0.40, np.log(0.50), np.log(1.70)))
    beta_tiebreak = float(np.clip(beta_tiebreak, 0.50, 1.70))

    # Shift the single skewed validity-weight distribution upward as advised:
    # median near 1.60 and a small upper tail toward 2.40. This preserves a
    # single unimodal distribution with no rejected mixture component.
    u_w = (validity_weight_raw - 0.60) / (2.40 - 0.60)
    z_w = _trunc_qnorm(
        u_w, np.log(1.10), 0.40, np.log(0.10), np.log(1.90))
    validity_weight = 0.50 + np.exp(z_w)
    validity_weight = float(np.clip(validity_weight, 0.60, 2.40))

    # Widen the lapse distribution only modestly, centered near 0.09, so
    # central tendencies in Experiments 3, 5, and 7 are not flattened while
    # enough heterogeneity is added.
    logit_lo = np.log(0.04 / 0.96)
    logit_hi = np.log(0.18 / 0.82)
    logit_mu = np.log(0.09 / 0.91)
    u_eps = (epsilon_raw - 0.04) / (0.18 - 0.04)
    z_eps = _trunc_qnorm(u_eps, logit_mu, 0.40, logit_lo, logit_hi)
    epsilon = 1.0 / (1.0 + np.exp(-z_eps))
    epsilon = float(np.clip(epsilon, 0.04, 0.18))

    # --- Inverse clean-cue comparison --------------------------------------
    clean_a = (b > a)
    clean_b = (a > b)
    raw_s = float(np.sum(clean_a) - np.sum(clean_b))

    if raw_s != 0.0:
        # Weak-to-moderate, all-cue validity weighting. Mean-centering keeps
        # the weights near one so low-validity majorities can still move the
        # tally when present.
        if validity_weight > 0.0 and n_features > 1:
            weights = 1.0 + validity_weight * (val - np.mean(val))
        else:
            weights = np.ones(n_features)

        s = float(np.sum(weights * clean_a) - np.sum(weights * clean_b))
        scores = np.array([s, -s])
        beta = beta_tally
    else:
        # Exact raw tally tie: consult features in descending validity order
        # and choose the alternative with a clean 0 on the first discriminator.
        cue_order = np.argsort(-val, kind='stable')
        winner = None
        for j in cue_order:
            if a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break

        if winner is None:
            return np.ones(2) / 2.0

        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = beta_tiebreak

    logits = beta * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0

`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta_tally: [0.22, 0.85]
- beta_tiebreak: [0.50, 1.70]
- epsilon: [0.04, 0.18]
- validity_weight: [0.60, 2.40]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.5117 (var=0.0161) vs this=-0.5442 (var=0.0102)
- Experiment 2: real=-0.6438 (var=0.0430) vs this=-0.6038 (var=0.0238)
- Experiment 3: real=-0.4300 (var=0.0221) vs this=-0.4029 (var=0.0226)
- Experiment 4: real=0.4925 (var=0.0658) vs this=0.5069 (var=0.0237)
- Experiment 5: real=0.2658 (var=0.1584) vs this=0.2446 (var=0.0156)
- Experiment 6: real=0.0867 (var=0.4244) vs this=0.1075 (var=0.0264)
- Experiment 7: real=0.0725 (var=0.0580) vs this=-0.0400 (var=0.0672)
- Experiment 8: real=1.0100 (var=2.8518) vs this=0.5012 (var=0.5143)
- Experiment 9: real=0.3050 (var=0.0605) vs this=0.4188 (var=0.0158)
- Experiment 10: real=-0.1667 (var=0.0053) vs this=-0.2374 (var=0.0002)
- Experiment 11: real=0.0000 (var=0.0045) vs this=0.0000 (var=0.0015)
- Experiment 12: real=1.2973 (var=0.2536) vs this=1.3853 (var=0.0963)


---

### `pi_3` (overall score: 0.592)

**Description**
People invert the meaning of binary expert ratings: a rating of 1 is treated as a defect or costly negative signal, and a rating of 0 is treated as a clean or favorable signal. Across all features, they tally how often each option is defect-free relative to the other option, and choose the option with the larger defect-free tally. Ties lead to guessing. Choice probability is a noisy softmax over the two defect tallies with inverse temperature beta, plus an independent lapse process that produces uniform guessing with probability epsilon. The heuristic uses all features equally, ignores validities and past trials, and differs from Take The Best by aggregating rather than stopping at the first discriminating cue, and from standard Tallying by reversing the valence of the binary ratings.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Inverse-tallying expects a (2, n_features) stimulus; got shape {stim.shape}.')

    a, b = stim[0], stim[1]

    # Defect-free wins: A has a 0 where B has a 1.
    a_wins_defects = float(np.sum(b > a))
    # Defect-free wins: B has a 0 where A has a 1.
    b_wins_defects = float(np.sum(a > b))

    scores = np.array([a_wins_defects, b_wins_defects])

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Numerically stable softmax over inverse-cue tallies.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [1.0, 5.0]
- epsilon: [0.30, 0.40]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.5117 (var=0.0161) vs this=-0.5379 (var=0.0084)
- Experiment 2: real=-0.6438 (var=0.0430) vs this=-0.6069 (var=0.0076)
- Experiment 3: real=-0.4300 (var=0.0221) vs this=0.0175 (var=0.0062)
- Experiment 4: real=0.4925 (var=0.0658) vs this=0.6381 (var=0.0147)
- Experiment 5: real=0.2658 (var=0.1584) vs this=-0.2708 (var=0.0095)
- Experiment 6: real=0.0867 (var=0.4244) vs this=-0.5333 (var=0.0223)
- Experiment 7: real=0.0725 (var=0.0580) vs this=-0.0025 (var=0.0792)
- Experiment 8: real=1.0100 (var=2.8518) vs this=1.7625 (var=0.1534)
- Experiment 9: real=0.3050 (var=0.0605) vs this=-0.0142 (var=0.0125)
- Experiment 10: real=-0.1667 (var=0.0053) vs this=-0.1529 (var=0.0007)
- Experiment 11: real=0.0000 (var=0.0045) vs this=0.0000 (var=0.0019)
- Experiment 12: real=1.2973 (var=0.2536) vs this=1.2888 (var=0.0583)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2897 -> ACCEPTED
- iter 2: loss=0.1615 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.1615 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    A = np.array(data['option_a_ratings'].tolist(), dtype=int)
    B = np.array(data['option_b_ratings'].tolist(), dtype=int)
    y = 1.0 - data['response'].to_numpy(dtype=float)

    top_a = A[:, 0] > B[:, 0]
    top_b = A[:, 0] < B[:, 0]

    lower_a_wins = (A[:, 1:] > B[:, 1:]).sum(axis=1)
    lower_b_wins = (B[:, 1:] > A[:, 1:]).sum(axis=1)
    lower_net = lower_a_wins - lower_b_wins

    high = lower_net >= 1
    low = lower_net <= -1

    def diff(top_mask):
        hi = y[top_mask & high]
        lo = y[top_mask & low]
        if len(hi) == 0 or len(lo) == 0:
            return 0.0
        return float(hi.mean() - lo.mean())

    return (diff(top_a) + diff(top_b)) / 2.0
```

**Observed (real) value:** -0.5117 (var=0.0161)
**Previous candidate values (this loop):**
  - iter 1: -0.1133 (var=0.0285) (Δ vs real +0.3983)
  - iter 2 (most recent): -0.5275 (var=0.0171) (Δ vs real -0.0158)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0092 (var=0.0043)
- pi_2: 0.6258 (var=0.0234)
- pi_3: -0.5379 (var=0.0084)
- pi_4: -0.5413 (var=0.0057)
- pi_5: -0.5308 (var=0.0142)
- pi_4_1: -0.5442 (var=0.0102)
- pi_6: -0.3700 (var=0.2536)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_rows = data['option_a_ratings'].tolist()
    b_rows = data['option_b_ratings'].tolist()
    tally = []
    for ra, rb in zip(a_rows, b_rows):
        a_wins = sum(1 for x, y in zip(ra, rb) if x > y)
        b_wins = sum(1 for x, y in zip(ra, rb) if y > x)
        tally.append(a_wins - b_wins)
    tally = np.asarray(tally, dtype=float)
    choose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    pos_rate = choose_a[tally > 0].mean()
    neg_rate = choose_a[tally < 0].mean()
    return float(pos_rate - neg_rate)
```

**Observed (real) value:** -0.6438 (var=0.0430)
**Previous candidate values (this loop):**
  - iter 1: -0.1831 (var=0.0471) (Δ vs real +0.4606)
  - iter 2 (most recent): -0.5206 (var=0.0297) (Δ vs real +0.1231)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7112 (var=0.0379)
- pi_1: -0.0013 (var=0.0059)
- pi_3: -0.6069 (var=0.0076)
- pi_4: -0.5913 (var=0.0097)
- pi_5: -0.5938 (var=0.0426)
- pi_4_1: -0.6038 (var=0.0238)
- pi_6: -0.6238 (var=0.2467)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.55])
    cue_order = np.argsort(-validities, kind='stable')

    a = np.array([list(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([list(r) for r in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int)

    n = a.shape[0]
    if n == 0:
        return 0.0

    predicted = np.full(n, -1, dtype=int)
    for cue in cue_order:
        undecided = predicted == -1
        disc = np.logical_and(undecided, a[:, cue] != b[:, cue])
        if not np.any(disc):
            continue
        predicted[disc] = np.where(a[disc, cue] > b[disc, cue], 0, 1)

    decided = predicted != -1
    if not np.any(decided):
        return 0.0

    match = (resp[decided] == predicted[decided]).astype(float)
    return float(np.mean(2.0 * match - 1.0))
```

**Observed (real) value:** -0.4300 (var=0.0221)
**Previous candidate values (this loop):**
  - iter 1: -0.4242 (var=0.0196) (Δ vs real +0.0058)
  - iter 2 (most recent): -0.3150 (var=0.0345) (Δ vs real +0.1150)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7113 (var=0.0398)
- pi_3: 0.0175 (var=0.0062)
- pi_2: 0.0117 (var=0.0115)
- pi_4: -0.4575 (var=0.0063)
- pi_5: -0.4596 (var=0.0245)
- pi_4_1: -0.4029 (var=0.0226)
- pi_6: -0.3550 (var=0.1762)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.array(data['option_a_ratings'].tolist(), dtype=int)
    b = np.array(data['option_b_ratings'].tolist(), dtype=int)
    # inverse-tally advantage for A: #(A=0,B=1) - #(A=1,B=0)
    s = (b > a).sum(axis=1) - (a > b).sum(axis=1)
    mask_pos = s == 2
    mask_neg = s == -2
    pos = data.loc[mask_pos, 'response']
    neg = data.loc[mask_neg, 'response']
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    p_pos = (pos == 0).mean()
    p_neg = (neg == 0).mean()
    return float(p_pos - p_neg)
```

**Observed (real) value:** 0.4925 (var=0.0658)
**Previous candidate values (this loop):**
  - iter 1: 0.0394 (var=0.0140) (Δ vs real -0.4531)
  - iter 2 (most recent): 0.4769 (var=0.0377) (Δ vs real -0.0156)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6381 (var=0.0147)
- pi_1: -0.0244 (var=0.0050)
- pi_2: -0.7525 (var=0.0447)
- pi_4: 0.4313 (var=0.0096)
- pi_5: 0.4237 (var=0.0670)
- pi_4_1: 0.5069 (var=0.0237)
- pi_6: 0.4681 (var=0.1627)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

PAIR_SIGNS = {
    ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)): 1.0,
    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): 1.0,
    ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((0, 0, 0, 0, 0, 1), (0, 0, 1, 0, 0, 0)): 1.0,
    ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,
    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1)): -1.0,
    ((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,
    ((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0)): 1.0,
    ((0, 0, 0, 0, 1, 1), (1, 0, 0, 0, 0, 0)): 1.0,
    ((0, 0, 0, 1, 1, 1), (1, 0, 1, 0, 0, 0)): 1.0,
}

def metric(data: pd.DataFrame) -> float:
    contributions = []
    for _, row in data.iterrows():
        key = (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))
        sign = PAIR_SIGNS.get(key)
        if sign is None:
            continue
        y = 1.0 if row['response'] == 0 else -1.0
        contributions.append(y * sign)
    return float(np.mean(contributions)) if contributions else 0.0
```

**Observed (real) value:** 0.2658 (var=0.1584)
**Previous candidate values (this loop):**
  - iter 1: 0.6833 (var=0.0167) (Δ vs real +0.4175)
  - iter 2 (most recent): 0.2621 (var=0.0297) (Δ vs real -0.0037)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2971 (var=0.0096)
- pi_3: -0.2708 (var=0.0095)
- pi_1: -0.6967 (var=0.0260)
- pi_2: 0.3704 (var=0.0144)
- pi_5: 0.3858 (var=0.0225)
- pi_4_1: 0.2446 (var=0.0156)
- pi_6: 0.3300 (var=0.1102)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    validities = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.6, 0.55], dtype=float)
    weights = 1.0 + 2.0 * (validities - validities.mean())

    pos = []
    neg = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        clean_a = (b > a).astype(float)
        clean_b = (a > b).astype(float)
        diff = clean_a - clean_b
        d = float(np.sum(diff))
        s = float(np.dot(weights, diff))
        y = 1.0 - float(row['response'])
        if d < 0.0 and s > 0.0:
            pos.append(y)
        elif d > 0.0 and s < 0.0:
            neg.append(y)

    if not pos or not neg:
        return 0.0

    return float(np.mean(pos) - np.mean(neg))
```

**Observed (real) value:** 0.0867 (var=0.4244)
**Previous candidate values (this loop):**
  - iter 1: 0.8733 (var=0.0064) (Δ vs real +0.7867)
  - iter 2 (most recent): 0.3808 (var=0.1137) (Δ vs real +0.2942)
**Other theories' values on this metric (for reference):**
- pi_3: -0.5333 (var=0.0223)
- pi_4: 0.2158 (var=0.0210)
- pi_1: -0.7375 (var=0.0404)
- pi_2: 0.6717 (var=0.0518)
- pi_5: 0.1383 (var=0.0736)
- pi_4_1: 0.1075 (var=0.0264)
- pi_6: 0.0358 (var=0.0678)

### Experiment 7
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    a0 = data['option_a_ratings'].apply(lambda x: x[0]).to_numpy(dtype=float)
    b0 = data['option_b_ratings'].apply(lambda x: x[0]).to_numpy(dtype=float)
    ld = data.apply(lambda row: sum([1 for x, y in zip(row['option_a_ratings'][1:], row['option_b_ratings'][1:]) if x != y]), axis=1).to_numpy(dtype=float)
    choose_a = 1.0 - data['response'].to_numpy(dtype=float)
    level = np.where(ld == 1, 1, np.where(ld == 3, 2, np.where(ld == 5, 3, 0)))
    topA = ((a0 == 0) & (b0 == 1))
    topB = ((a0 == 1) & (b0 == 0))
    m1 = choose_a[topA & (level == 1)].mean() if (topA & (level == 1)).any() else 0.0
    m2 = choose_a[topB & (level == 3)].mean() if (topB & (level == 3)).any() else 0.0
    m3 = choose_a[topA & (level == 3)].mean() if (topA & (level == 3)).any() else 0.0
    m4 = choose_a[topB & (level == 1)].mean() if (topB & (level == 1)).any() else 0.0
    return float(m1 + m2 - m3 - m4)
```

**Observed (real) value:** 0.0725 (var=0.0580)
**Previous candidate values (this loop):**
  - iter 1: 0.0863 (var=0.0247) (Δ vs real +0.0138)
  - iter 2 (most recent): -0.0800 (var=0.0872) (Δ vs real -0.1525)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0025 (var=0.0525)
- pi_5: 0.3325 (var=0.0612)
- pi_1: 0.0288 (var=0.0221)
- pi_2: 0.0137 (var=0.0602)
- pi_3: -0.0025 (var=0.0792)
- pi_4_1: -0.0400 (var=0.0672)
- pi_6: 0.1800 (var=0.1187)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    counts = {}
    totals = {}

    for row in data.itertuples(index=False):
        a = [int(x) for x in row.option_a_ratings]
        b = [int(x) for x in row.option_b_ratings]

        # h = 0 means the three high-validity cues favour B under inverse valence;
        # h = 1 is the mirror where they favour A.
        h = 0 if sum(a[:3]) > sum(b[:3]) else 1

        # Number of opposing low-validity cues.
        n = sum(1 for i in range(3, len(a)) if a[i] != b[i])

        key = (h, n)
        totals[key] = totals.get(key, 0) + 1
        if int(row.response) == 0:
            counts[key] = counts.get(key, 0) + 1

    props = {}
    for h in (0, 1):
        for n in (3, 4, 5):
            tot = totals.get((h, n), 0)
            props[(h, n)] = (counts.get((h, n), 0) / tot) if tot > 0 else 0.5

    d3 = props[(0, 3)] - props[(1, 3)]
    d4 = props[(0, 4)] - props[(1, 4)]
    d5 = props[(0, 5)] - props[(1, 5)]

    return float(-d3 + d4 + 2.0 * d5)
```

**Observed (real) value:** 1.0100 (var=2.8518)
**Previous candidate values (this loop):**
  - iter 1: -1.8138 (var=0.0404) (Δ vs real -2.8237)
  - iter 2 (most recent): -1.6350 (var=0.1386) (Δ vs real -2.6450)
**Other theories' values on this metric (for reference):**
- pi_5: 1.3512 (var=0.8533)
- pi_4: -0.1263 (var=0.1938)
- pi_1: 1.4112 (var=0.1813)
- pi_2: -2.0162 (var=0.4200)
- pi_3: 1.7625 (var=0.1534)
- pi_4_1: 0.5012 (var=0.5143)
- pi_6: 1.1075 (var=2.7433)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

_VALIDITIES = np.array([0.95, 0.90, 0.65, 0.64, 0.63, 0.62, 0.60, 0.60])
_FIRST_CUE_ORDER = np.argsort(-_VALIDITIES, kind='stable')


def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=np.int64)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=np.int64)
    r = np.asarray(data['response'].tolist(), dtype=np.int64)

    n = len(data)
    first_col = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)

    for j in _FIRST_CUE_ORDER:
        differs = a[:, j] != b[:, j]
        newly = differs & ~assigned
        first_col[newly] = j
        assigned |= differs

    valid = first_col >= 0
    if not valid.any():
        return 0.0

    idx_valid = np.nonzero(valid)[0]
    cols_valid = first_col[valid]
    a_first = a[idx_valid, cols_valid]
    b_first = b[idx_valid, cols_valid]
    favors_a = (a_first == 0) & (b_first == 1)

    high_mask = np.zeros(n, dtype=bool)
    low_mask = np.zeros(n, dtype=bool)
    high_mask[idx_valid[favors_a]] = True
    low_mask[idx_valid[~favors_a]] = True

    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0

    chose_a = r == 0
    return float(chose_a[high_mask].mean() - chose_a[low_mask].mean())

```

**Observed (real) value:** 0.3050 (var=0.0605)
**Previous candidate values (this loop):**
  - iter 1: 0.4604 (var=0.0229) (Δ vs real +0.1554)
  - iter 2 (most recent): 0.4629 (var=0.0302) (Δ vs real +0.1579)
**Other theories' values on this metric (for reference):**
- pi_4_1: 0.4188 (var=0.0158)
- pi_5: 0.2196 (var=0.0342)
- pi_1: -0.7067 (var=0.0351)
- pi_2: 0.0204 (var=0.0104)
- pi_3: -0.0142 (var=0.0125)
- pi_4: 0.4296 (var=0.0093)
- pi_6: 0.1742 (var=0.1020)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    scores = []
    for sid, sub in data.groupby('subject_id'):
        low = []
        for a, b, resp in zip(sub['option_a_ratings'], sub['option_b_ratings'], sub['response']):
            s = 0
            for x, y in zip(a, b):
                if x == 0 and y == 1:
                    s += 1
                elif x == 1 and y == 0:
                    s -= 1
            if s > 0:
                low.append(1.0 if resp == 0 else 0.0)
            elif s < 0:
                low.append(1.0 if resp == 1 else 0.0)
        if not low:
            scores.append(0.0)
        else:
            q = float(np.mean(low))
            scores.append(q * q - q)
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1667 (var=0.0053)
**Previous candidate values (this loop):**
  - iter 1: -0.0390 (var=0.0006) (Δ vs real +0.1277)
  - iter 2 (most recent): -0.1309 (var=0.0056) (Δ vs real +0.0357)
**Other theories' values on this metric (for reference):**
- pi_5: -0.2057 (var=0.0022)
- pi_4_1: -0.2374 (var=0.0002)
- pi_1: -0.1112 (var=0.0033)
- pi_2: -0.1264 (var=0.0029)
- pi_3: -0.1529 (var=0.0007)
- pi_4: -0.2445 (var=0.0000)
- pi_6: -0.1839 (var=0.0030)

### Experiment 11
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_list = data['option_a_ratings'].tolist()
    b_list = data['option_b_ratings'].tolist()
    k = np.array([sum(1 for x, y in zip(a, b) if x == 0 and y == 1) for a, b in zip(a_list, b_list)], dtype=float)
    resp = np.asarray(data['response'], dtype=float)
    subject_ids = data['subject_id'].to_numpy()
    informative = k != 4
    inv_pred = np.where(k < 4, 1.0, 0.0)
    inv_match = resp == inv_pred

    def subject_consistency(mask):
        valid = informative & mask
        if not valid.any():
            return 0.5
        return float(np.mean(inv_match[valid]))

    unique_ids = np.unique(subject_ids)
    if len(unique_ids) > 1:
        positive_count = []
        for sid in unique_ids:
            s = subject_consistency(subject_ids == sid)
            positive_count.append(1.0 if s < 0.5 else 0.0)
        return float(np.mean(positive_count))
    s = subject_consistency(np.ones(len(data), dtype=bool))
    return float(max(s, 1.0 - s))

```

**Observed (real) value:** 0.0000 (var=0.0045)
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.0068) (Δ vs real +0.0000)
  - iter 2 (most recent): 0.0000 (var=0.0053) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0400 (var=0.0007)
- pi_5: 0.0000 (var=0.0037)
- pi_1: 0.9200 (var=0.0019)
- pi_2: 1.0000 (var=0.0083)
- pi_3: 0.0000 (var=0.0019)
- pi_4: 0.0000 (var=0.0011)
- pi_4_1: 0.0000 (var=0.0015)

### Experiment 12
**Design**
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def score_one(g):
        a = np.array([list(x) for x in g['option_a_ratings']], dtype=int)
        b = np.array([list(x) for x in g['option_b_ratings']], dtype=int)
        resp = g['response'].to_numpy().astype(int)

        d = ((a == 0) & (b == 1)).sum(axis=1) - ((a == 1) & (b == 0)).sum(axis=1)
        choice_a = (resp == 0).astype(float)

        far = np.abs(d) == 3
        far_pos = far & (d == 3)
        far_neg = far & (d == -3)
        if np.any(far_pos) and np.any(far_neg):
            orient_diff = float(np.mean(choice_a[far_pos]) - np.mean(choice_a[far_neg]))
            orient = 1.0 if orient_diff >= 0.0 else -1.0
        else:
            corr = float(np.corrcoef(d.astype(float), choice_a)[0, 1])
            orient = 1.0 if np.isfinite(corr) and corr >= 0.0 else -1.0

        aligned_d = orient * d
        majority_opt = np.where(aligned_d > 0, 0, 1)

        raw_top_dir = np.where((a[:, 0] == 0) & (b[:, 0] == 1), 1.0, -1.0)
        top_opt = np.where(orient * raw_top_dir > 0.0, 0, 1)
        top_align = (top_opt == majority_opt)
        followed_top = (resp == top_opt).astype(float)

        near = np.abs(d) == 1
        far2 = np.abs(d) == 3

        def arcs(mask):
            vals = followed_top[mask]
            n = vals.shape[0]
            if n == 0:
                return 0.0
            p = (float(np.sum(vals)) + 0.5) / (float(n) + 1.0)
            p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            return float(np.arcsin(np.sqrt(p)))

        near_aligned = arcs(near & top_align)
        near_opposed = arcs(near & ~top_align)
        far_aligned = arcs(far2 & top_align)
        far_opposed = arcs(far2 & ~top_align)

        return (near_aligned - near_opposed) + (far_aligned - far_opposed)

    if data.empty:
        return 0.0

    scores = []
    for sid in data['subject_id'].unique():
        sub = data[data['subject_id'] == sid]
        scores.append(score_one(sub))
    return float(np.mean(scores))
```

**Observed (real) value:** 1.2973 (var=0.2536)
**Previous candidate values (this loop):**
  - iter 1: 1.5455 (var=0.1657) (Δ vs real +0.2482)
  - iter 2 (most recent): 1.2256 (var=0.2100) (Δ vs real -0.0717)
**Other theories' values on this metric (for reference):**
- pi_5: 1.5845 (var=0.1245)
- pi_6: 1.9342 (var=0.0366)
- pi_1: 0.1091 (var=0.0296)
- pi_2: 1.6796 (var=0.3099)
- pi_3: 1.2888 (var=0.0583)
- pi_4: 1.3818 (var=0.0296)
- pi_4_1: 1.3853 (var=0.0963)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
People uniformly treat a binary expert rating of 1 as a defect and 0 as a clean/favorable signal. Every discriminating expert cue contributes continuous subjective evidence, with cue weights that are monotone in communicated validity but much flatter than a steep log-linear ordering: w_j = exp(a_i * (logit(v_j) - mean(logit(v)))) normalized to mean one, with subject-specific a_i drawn from a low-median, heavy-tailed distribution. High-validity experts remain more influential when validity gaps are clear, but lower-validity majorities are not swamped by a single top cue. Choice is a softmax over net evidence with subject-specific gain and lapse, and near-zero evidence is resolved by a validity-ordered scan in which tied top-validity cues are randomly permuted per trial. There is no valence-orientation mixture and no separate top-cue bonus.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import norm


def _trunc_qnorm(u, mu, sigma, lo, hi):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = norm.cdf((lo - mu) / sigma)
    q_hi = norm.cdf((hi - mu) / sigma)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * norm.ppf(q)


def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a_row = stim[0]
    b_row = stim[1]
    n_features = stim.shape[1]
    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # ---- Subject-level parameter shaping ------------------------------
    # Validity-weight slope: now low-median and capped so the highest cue
    # receives only a modest weight advantage over the lowest cue.
    u_s = float(np.clip((float(parameters['slope_a']) - 0.05) / 1.05, 0.0, 1.0))
    slope_a = float(np.exp(_trunc_qnorm(
        u_s, np.log(0.30), 0.65, np.log(0.05), np.log(1.10))))

    # Choice gain: slightly softened relative to the previous candidate so
    # that moderate weighted-tally advantages produce graded rather than
    # near-deterministic choice.
    u_g = float(np.clip((float(parameters['gain']) - 0.10) / 1.40, 0.0, 1.0))
    gain = float(np.exp(_trunc_qnorm(
        u_g, np.log(0.40), 0.50, np.log(0.10), np.log(1.50))))

    lo_logit = np.log(0.03 / 0.97)
    hi_logit = np.log(0.16 / 0.84)
    mu_logit = np.log(0.08 / 0.92)
    u_e = float(np.clip((float(parameters['epsilon']) - 0.03) / 0.13, 0.0, 1.0))
    z_eps = _trunc_qnorm(u_e, mu_logit, 0.35, lo_logit, hi_logit)
    epsilon = float(_inv_logit(z_eps))

    u_tg = float(np.clip((float(parameters['tie_gain']) - 0.25) / 0.95, 0.0, 1.0))
    tie_gain = float(np.exp(_trunc_qnorm(
        u_tg, np.log(0.50), 0.45, np.log(0.25), np.log(1.20))))

    u_tt = float(np.clip((float(parameters['tie_tol']) - 0.05) / 0.40, 0.0, 1.0))
    tie_tol = float(_trunc_qnorm(u_tt, 0.19, 0.11, 0.05, 0.45))

    # ---- Continuous inverse-valence weights ---------------------------
    # 0 is clean/favorable, 1 is a defect.
    v = np.clip(val, 0.50001, 0.9999)
    logit_v = np.log(v / (1.0 - v))
    center = float(np.mean(logit_v))
    w = np.exp(slope_a * (logit_v - center))
    w = w / np.mean(w)  # mean weight is exactly one

    changed_mask = a_row != b_row
    if not np.any(changed_mask):
        return np.ones(2) / 2.0

    # A=0/B=1 gives +w_j; A=1/B=0 gives -w_j.
    diff = (b_row > a_row).astype(float) - (a_row > b_row).astype(float)
    net_evidence = float(np.dot(w, diff))

    if abs(net_evidence) <= tie_tol:
        # Inconclusive continuous evidence: lexicographic scan of the
        # highest-validity changed cues, with tied cues exchangeable.
        disc = np.flatnonzero(changed_mask)
        top_val = float(np.max(val[disc]))
        tied = disc[np.isclose(val[disc], top_val, rtol=1e-12, atol=1e-12)]
        if tied.size == 0:
            tied = disc
        idx = int(tied[np.random.randint(tied.size)])

        if a_row[idx] == 0.0 and b_row[idx] == 1.0:
            sign = 1.0   # clean cue favors A
        elif a_row[idx] == 1.0 and b_row[idx] == 0.0:
            sign = -1.0  # clean cue favors B
        else:
            return np.ones(2) / 2.0

        scores = np.array([sign, -sign], dtype=float)
        beta_eff = tie_gain
    else:
        scores = np.array([net_evidence, -net_evidence], dtype=float)
        beta_eff = gain

    logits = beta_eff * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- slope_a: [0.05, 1.10]
- gain: [0.10, 1.50]
- epsilon: [0.03, 0.16]
- tie_gain: [0.25, 1.20]
- tie_tol: [0.05, 0.45]
- validities: validities

`rationale`: The previous candidate's dominant failure was the steep validity-to-weight mapping. With slope_a drawn from a lognormal median near 1.8 and support up to 5.0, the highest-validity cue received roughly 20-25 times the weight of the lowest cue, causing a single top cue to override lower-cue coalitions in Experiments 1, 2, 4, and 8, and producing near-deterministic validity-following in Experiments 5, 6, and 10. This edit keeps the prescribed continuous weighted-evidence mechanism but retunes the slope distribution to a low-median heavy-tailed distribution: median slope_a is 0.30 with a hard cap of 1.10, so w_j = exp(a * (logit(v_j) - mean(logit(v)))) remains monotone in validity but gives the top cue only a modest advantage. Lower-validity majorities therefore move choice again, recovering inverse-tally dominance in Experiments 1, 2, and 4, bringing Experiment 6 conflict cells toward chance, and flipping Experiment 8 in the correct direction. The softmax gain is also slightly softened from a median of 0.55 to 0.40 so that clear but moderate weighted-tally differences produce graded rather than saturating choices. All other mechanisms are preserved: universal inverse valence, no orientation mixture, continuous validity-weighted evidence, lapse, and the randomized tied-cue fallback that keeps tied top-validity cues exchangeable.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter's prescribed family: universal inverse valence (0=clean, 1=defect), continuous validity-weighted evidence instead of inclusion gating, no orientation mixture, no top-cue bonus, softmax plus subject-level lapse, and an exchangeable random tie-break among tied top-validity cues in the inconclusiveness fallback. It reproduces part of pi_5's successes: Experiment 11 is exact (0.0000 vs 0.0000, var 0.0068), Experiment 3 is essentially on target (-0.4242 vs -0.4300), Experiment 7 is the best-fitting theory we have (0.0863 vs 0.0725), and Experiments 9 and 12 are within tolerance (0.4604 vs 0.3050; 1.5455 vs 1.2973). However, the candidate badly misses the bulk of the empirical patterns. Experiment 6 is off by +0.79 (0.8733 vs real 0.0867): the candidate follows validity-weighted evidence on the metric's conflict cells, while humans are near chance there. Experiment 8 is wrong-signed (-1.8138 vs +1.0100), the single biggest failure. Experiment 4 collapses to near chance (0.0394 vs 0.4925): humans track the ±2 inverse tally strongly, but the candidate's steep weights let high-validity cue alignments cancel the majority direction. Experiments 1 and 2 are far too weak (-0.1133 vs -0.5117; -0.1831 vs -0.6438): the single top cue outvotes the five-cue lower coalition, whereas humans let the lower coalition dominate. Experiments 5 and 10 show over-extreme validity-following (0.6833 vs 0.2658; -0.0390 vs -0.1667, i.e. q≈0.04 vs q≈0.21). The unifying diagnosis is that the subject-specific slope, drawn from a lognormal with median 1.8 (range 0.30-5.00), makes w_j = exp(a·(logit(v_j) − mean logit v)) far too steep: with the 6-7 cue validity arrays the top cue receives roughly 20-25× the weight of the bottom cue. Humans' behavior across Experiments 1, 2, 4, 5, 6, 8 and 10 is consistent with near-equal (tally-like) weights plus only modest validity sensitivity, not with such a steep validity ordering.
Rationale: Stay inside the prescribed family but flatten the validity-to-weight mapping, since the current log-linear slope distribution is the main miscalibration. Concretely: (1) replace the exponent with a much gentler monotone transform, e.g. w_j = exp(a·(logit(v_j) − mean logit v)) with a drawn from a tightened distribution (median ≈ 0.3-0.5, hard cap ≈ 1.0-1.2 instead of the current lognormal(1.8, 0.65) capped at 5.0), or use a near-linear transform like w_j = 1 + c·(v_j − mean v) with small subject-level c, keeping the mean normalization. Predicted effects of this single knob: Experiments 1 and 2 recover strong inverse-valence effects because the five-cue lower coalition (≈5·w̄) now overrides the single top cue exactly as humans do; Experiment 4 regains s=±2 tally sensitivity (net ≈ ±2w̄ gives p_A ≈ 0.75/0.25 before lapse, matching the observed ≈0.49 majority effect); Experiment 6's validity/tally conflict cells move to near-chance, matching the observed 0.0867; Experiment 5 drops toward the observed 0.2658; Experiment 10 moderates to q≈0.2; and the Experiment 8 sign flips positive, where pi_5's tally-like behavior also sits (+1.3512 vs real +1.0100). (2) Preserve everything that currently works: keep the universal inverse valence (Experiment 11 exact), keep the softmax gain and lapse mechanics but verify they can still separate s=−2/0/+2 tallies at the clicker level, and keep the randomized tied-cue fallback that made Experiments 3 and 7 land well. Do not reintroduce orientation mixtures, inclusion gating, top-cue bonuses, or deterministic stable-sort tie-breaks. This is a within-family parameter/transform tuning, not a mechanism redesign.

**Outcome of this advice:** iter 1 candidate loss=0.2897 -> iter 2 candidate loss=0.1615 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter's prescribed family: universal inverse valence (0=clean, 1=defect), continuous validity-weighted evidence, no orientation mixture, no top-cue bonus, softmax with subject-level gain/lapse, and exchangeable tied top-cue fallback. The flattening of the validity-to-weight slope from the previous iteration was clearly the right move: Experiments 1, 4, 5, 10, 11, and 12 are now essentially on target, and Experiments 2 and 6 improved substantially. The remaining failures are concentrated and diagnostic. Experiment 8 is still badly wrong-signed (-1.635 vs real +1.010, a -2.645 gap): the candidate behaves as if the three high-validity cues win at every opposing low-cue count, which produces the metric's near -2 signature. Real subjects let the low-validity majority take over as the number of opposing low cues grows. This is a weight-ratio calibration problem: the current slope_a distribution (median 0.30, hard cap 1.10, heavy right tail) still gives top/bottom weight ratios near or above 2 for many subjects, so 3 high-validity cues beat 5 low-validity cues. Experiment 6 remains too validity-following in conflict cells (0.381 vs real 0.087), indicating the same residual top-cue overweighting. Experiment 3 is now too weak (-0.315 vs -0.430): adjacent-cue net evidence is small and the top-cue fallback is not being used strongly or often enough to reproduce the observed validity-ordered choices. Experiments 2 and 9 also show mild residual top-validity overweighting in the same direction.
Rationale: Stay inside the prescribed continuous validity-weighted evidence family, but complete the slope calibration that was only partially done. Concretely: (1) Further lower and cap the validity-weight slope. Move slope_a from median 0.30 / cap 1.10 to median about 0.10-0.15 with a hard cap around 0.25-0.30, keeping a heavy-tailed shape but cutting the right tail. This should keep the top/bottom weight ratio in roughly the 1.2-1.5 range, so a 3-cue high-validity coalition still wins against 3 low cues in Experiment 8, but 4-5 low cues override it, flipping Experiment 8 positive. It should also reduce the residual validity-following in Experiments 2, 6, and 9. (2) Tighten tie_tol: median around 0.12-0.15 and hard cap around 0.20-0.25, instead of the current 0.19 median / 0.45 cap. This keeps Experiment 3's tiny adjacent-cue net evidence inside the fallback region while ensuring Experiment 8's low-majority net evidence is outside it and actually drives choice. (3) Strengthen the fallback itself by raising tie_gain from median 0.50 to about 0.80-1.00, so Experiment 3's validity-ordered fallback choices are more deterministic. (4) If flattening the slope weakens the strong tally effects in Experiments 1, 2, and 4 too much, compensate by moderately raising the softmax gain from median 0.40 toward 0.55-0.70, but monitor Experiment 6 so conflict cells stay near chance. Keep the universal inverse valence, lapse, exchangeable tied-cue fallback, and the absence of orientation mixtures and top-cue bonuses unchanged.

**Outcome of this advice:** iter 2 candidate loss=0.1615 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- theory: The theoretical claim this theory makes about cognition.
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this theory design; used to explain why this theory is better then the previous ones.


## Response

```json
{
  "theory": "People universally treat a binary expert rating of 0 as clean/favorable and 1 as a defect. Every discriminating cue contributes continuous validity-weighted evidence rather than entering an all-or-none tally: option A gains w_j when feature j has A=0/B=1 and loses w_j when A=1/B=0, with subject-specific weights w_j = exp(a_i * (logit(v_j) - mean(logit(v)))) normalized to mean one. The validity-weight slope a_i is small and heavily regularized, so high-validity experts dominate only modestly while low-validity majorities still move choice. Choice is a softmax over net evidence with subject-specific gain and lapse. When net evidence is inconclusive, subjects inspect the highest-validity discriminating cues in order, with tied top-validity cues randomly permuted per trial. There is no valence-orientation mixture and no separate top-cue bonus.",
  "predict": "import numpy as np\nfrom scipy.stats import norm\n\n\ndef _trunc_qnorm(u, mu, sigma, lo, hi):\n    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))\n    q_lo = norm.cdf((lo - mu) / sigma)\n    q_hi = norm.cdf((hi - mu) / sigma)\n    q = q_lo + u * (q_hi - q_lo)\n    return mu + sigma * norm.ppf(q)\n\n\ndef _inv_logit(z):\n    return 1.0 / (1.0 + np.exp(-z))\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus.')\n\n    a_row = stim[0]\n    b_row = stim[1]\n    n_features = stim.shape[1]\n    val = np.asarray(parameters['validities'], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError('validities length mismatch')\n\n    # ---- Subject-level parameter shaping ------------------------------\n    # Validity-weight slope: strongly flattened and capped so high-validity\n    # cues receive only a modest weight advantage.\n    u_s = float(np.clip((float(parameters['slope_a']) - 0.02) / 0.28, 0.0, 1.0))\n    slope_a = float(np.exp(_trunc_qnorm(\n        u_s, np.log(0.12), 0.70, np.log(0.02), np.log(0.30))))\n\n    # Choice gain: kept moderate. Flattening the validity weights is the\n    # main correction for the strong tally effects, so gain is not raised.\n    u_g = float(np.clip((float(parameters['gain']) - 0.10) / 1.40, 0.0, 1.0))\n    gain = float(np.exp(_trunc_qnorm(\n        u_g, np.log(0.40), 0.50, np.log(0.10), np.log(1.50))))\n\n    lo_logit = np.log(0.03 / 0.97)\n    hi_logit = np.log(0.16 / 0.84)\n    mu_logit = np.log(0.08 / 0.92)\n    u_e = float(np.clip((float(parameters['epsilon']) - 0.03) / 0.13, 0.0, 1.0))\n    z_eps = _trunc_qnorm(u_e, mu_logit, 0.35, lo_logit, hi_logit)\n    epsilon = float(_inv_logit(z_eps))\n\n    # Raised tie-break gain, but only moderately to avoid over-sharpening\n    # the first-cue-following metric in Experiment 9.\n    u_tg = float(np.clip((float(parameters['tie_gain']) - 0.30) / 1.00, 0.0, 1.0))\n    tie_gain = float(np.exp(_trunc_qnorm(\n        u_tg, np.log(0.70), 0.45, np.log(0.30), np.log(1.30))))\n\n    # Tightened inconclusive-evidence threshold.\n    u_tt = float(np.clip((float(parameters['tie_tol']) - 0.05) / 0.17, 0.0, 1.0))\n    tie_tol = float(_trunc_qnorm(u_tt, 0.13, 0.045, 0.05, 0.22))\n\n    # ---- Continuous inverse-valence weights ---------------------------\n    # 0 is clean/favorable, 1 is a defect.\n    v = np.clip(val, 0.50001, 0.9999)\n    logit_v = np.log(v / (1.0 - v))\n    center = float(np.mean(logit_v))\n    w = np.exp(slope_a * (logit_v - center))\n    w = w / np.mean(w)  # mean weight is exactly one\n\n    changed_mask = a_row != b_row\n    if not np.any(changed_mask):\n        return np.ones(2) / 2.0\n\n    # A=0/B=1 gives +w_j; A=1/B=0 gives -w_j.\n    diff = (b_row > a_row).astype(float) - (a_row > b_row).astype(float)\n    net_evidence = float(np.dot(w, diff))\n\n    if abs(net_evidence) <= tie_tol:\n        # Inconclusive continuous evidence: lexicographic scan of the\n        # highest-validity changed cues, with tied cues exchangeable.\n        disc = np.flatnonzero(changed_mask)\n        top_val = float(np.max(val[disc]))\n        tied = disc[np.isclose(val[disc], top_val, rtol=1e-12, atol=1e-12)]\n        if tied.size == 0:\n            tied = disc\n        idx = int(tied[np.random.randint(tied.size)])\n\n        if a_row[idx] == 0.0 and b_row[idx] == 1.0:\n            sign = 1.0   # clean cue favors A\n        elif a_row[idx] == 1.0 and b_row[idx] == 0.0:\n            sign = -1.0  # clean cue favors B\n        else:\n            return np.ones(2) / 2.0\n\n        scores = np.array([sign, -sign], dtype=float)\n        beta_eff = tie_gain\n    else:\n        scores = np.array([net_evidence, -net_evidence], dtype=float)\n        beta_eff = gain\n\n    logits = beta_eff * scores\n    logits = logits - np.max(logits)\n    p_core = np.exp(logits)\n    p_core = p_core / p_core.sum()\n\n    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0:\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "slope_a": "[0.02, 0.30]",
    "gain": "[0.10, 1.50]",
    "epsilon": "[0.03, 0.16]",
    "tie_gain": "[0.30, 1.30]",
    "tie_tol": "[0.05, 0.22]",
    "validities": "validities"
  },
  "rationale": "The minimal edit completes the validity-weight calibration identified in the prior feedback. The slope distribution is moved from median 0.30 / cap 1.10 to median 0.12 / cap 0.30, keeping a heavy tail but cutting the extreme right tail. This lowers the top/bottom cue-weight ratio enough that three high-validity cues no longer beat five low-validity cues in Experiment 8, letting the low-validity majority take over and flipping that metric positive. It also reduces the residual validity-following in Experiments 2, 6, and 9. The inconclusive-evidence threshold is tightened to median 0.13 / cap 0.22 so small adjacent-cue evidence in Experiment 3 still invokes the fallback while larger low-majority evidence in Experiment 8 drives softmax choice. The tie-break gain is raised from median 0.50 to median 0.70, strengthening the validity-ordered fallback for Experiment 3, but it is kept moderate rather than pushed to 0.90 because Experiment 9 is already too first-cue-following. The softmax gain is deliberately left unchanged: flattening the validity weights already sharpens the strong inverse-tally effects in Experiments 1, 2, and 4, and raising gain would risk overshooting Experiments 4 and 10."
}
```

## Usage

```json
{
  "input_tokens": 26001,
  "output_tokens": 24953,
  "reasoning_tokens": 22808,
  "provider": "Baidu"
}
```
