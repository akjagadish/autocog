# generation_iter_03_attempt_00

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
- THEORY 1 = `pi_4_1`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_4_1`).

Propose a valence-mixture validity-gated tally theory. Each subject has a latent valence orientation: with probability about 0.8 they treat 1 as a defect and 0 as clean (inverse valence), otherwise they use the opposite positive valence. Within a trial, each feature j enters the tally with probability logistic(alpha * validity_j + gamma); calibrate alpha and gamma so the highest-validity cues are included with probability >0.95, which should produce the near-zero top/lower-cue interaction in E9 and moderate tie-break contrast in E1, while lower-validity cues are included with moderate probability so their majorities can still move the tally in E2 and E10. The included tally drives a noisy softmax with subject-specific beta_tally. Exact zero tallies are broken by scanning included cues in descending validity order and choosing the valence-consistent cue (0 for inverse subjects, 1 for positive subjects) on the first discriminator; if none discriminates, guess. Subject-level orientation, alpha, gamma, beta_tally, beta_tiebreak, and epsilon should be drawn from heavy-tailed or logit-normal distributions to reproduce the large observed between-subject variances. This differs from pi_4_1 by replacing deterministic all-cue tie-breaking with probabilistic inclusion and adding valence mixture; it differs from pi_5 by adding valence mixture and making high-validity cue inclusion nearly deterministic rather than merely high probability.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4_1` (overall score: 0.577)

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


---

### `pi_3` (overall score: 0.527)

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


---

### `pi_4` (overall score: 0.525)

**Description**
People invert the valence of binary expert ratings: a 1 is treated as a costly defect and a 0 as a clean, favorable signal. Across all features they compute an inverse-tally advantage for A, weakly weighting the inverse cues by the experts' stated validities. When the raw inverse tally is nonzero, choice is a noisy softmax over the two weakly validity-weighted inverse tallies. When the raw inverse tally is exactly tied, people do not guess uniformly; instead they break the tie lexicographically by consulting features in descending validity order and choosing the option that has a clean 0 on the first discriminating feature. Response noise is captured by a softmax plus an independent uniform lapse. This differs from Take The Best because it aggregates all cues rather than stopping at the first discriminator, and from plain inverse tallying because ties are resolved by validity-ordered clean-cue use instead of guessing.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

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

    beta_tally = float(parameters['beta_tally'])
    beta_tiebreak = float(parameters['beta_tiebreak'])
    epsilon = float(parameters['epsilon'])
    validity_weight = float(parameters['validity_weight'])

    clean_a = (b > a)
    clean_b = (a > b)
    raw_s = float(np.sum(clean_a) - np.sum(clean_b))

    if raw_s != 0:
        if validity_weight > 0.0 and n_features > 1:
            weights = 1.0 + validity_weight * (val - np.mean(val))
        else:
            weights = np.ones(n_features)
        s = float(np.sum(weights * clean_a) - np.sum(weights * clean_b))
        scores = np.array([s, -s])
        beta = beta_tally
    else:
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

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * np.ones(n_opts) / n_opts


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
- beta_tally: [0.43, 0.47]
- beta_tiebreak: [1.00, 1.08]
- epsilon: [0.09, 0.11]
- validity_weight: [1.90, 2.10]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.5117 (var=0.0161) vs this=-0.5413 (var=0.0057)
- Experiment 2: real=-0.6438 (var=0.0430) vs this=-0.5913 (var=0.0097)
- Experiment 3: real=-0.4300 (var=0.0221) vs this=-0.4575 (var=0.0063)
- Experiment 4: real=0.4925 (var=0.0658) vs this=0.4313 (var=0.0096)
- Experiment 5: real=0.2658 (var=0.1584) vs this=0.2971 (var=0.0096)
- Experiment 6: real=0.0867 (var=0.4244) vs this=0.2158 (var=0.0210)
- Experiment 7: real=0.0725 (var=0.0580) vs this=-0.0025 (var=0.0525)
- Experiment 8: real=1.0100 (var=2.8518) vs this=-0.1263 (var=0.1938)
- Experiment 9: real=0.3050 (var=0.0605) vs this=0.4296 (var=0.0093)
- Experiment 10: real=-0.1667 (var=0.0053) vs this=-0.2445 (var=0.0000)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1282 -> ACCEPTED
- iter 2: loss=0.2332 -> REJECTED
- iter 3: loss=0.1966 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1282 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.5133 (var=0.1467) (Δ vs real -0.0017)
  - iter 2: -0.4154 (var=0.1645) (Δ vs real +0.0962)
  - iter 3 (most recent): -0.4204 (var=0.0818) (Δ vs real +0.0912)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0092 (var=0.0043)
- pi_2: 0.6258 (var=0.0234)
- pi_3: -0.5379 (var=0.0084)
- pi_4: -0.5413 (var=0.0057)
- pi_5: -0.5308 (var=0.0142)
- pi_4_1: -0.5442 (var=0.0102)

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
  - iter 1: -0.5563 (var=0.2649) (Δ vs real +0.0875)
  - iter 2: -0.4994 (var=0.1633) (Δ vs real +0.1444)
  - iter 3 (most recent): -0.3387 (var=0.1333) (Δ vs real +0.3050)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7112 (var=0.0379)
- pi_1: -0.0013 (var=0.0059)
- pi_3: -0.6069 (var=0.0076)
- pi_4: -0.5913 (var=0.0097)
- pi_5: -0.5938 (var=0.0426)
- pi_4_1: -0.6038 (var=0.0238)

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
  - iter 1: -0.3387 (var=0.1388) (Δ vs real +0.0912)
  - iter 2: -0.4892 (var=0.0276) (Δ vs real -0.0592)
  - iter 3 (most recent): -0.4375 (var=0.0985) (Δ vs real -0.0075)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7113 (var=0.0398)
- pi_3: 0.0175 (var=0.0062)
- pi_2: 0.0117 (var=0.0115)
- pi_4: -0.4575 (var=0.0063)
- pi_5: -0.4596 (var=0.0245)
- pi_4_1: -0.4029 (var=0.0226)

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
  - iter 1: 0.4456 (var=0.1925) (Δ vs real -0.0469)
  - iter 2: 0.2700 (var=0.0726) (Δ vs real -0.2225)
  - iter 3 (most recent): 0.2200 (var=0.0421) (Δ vs real -0.2725)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6381 (var=0.0147)
- pi_1: -0.0244 (var=0.0050)
- pi_2: -0.7525 (var=0.0447)
- pi_4: 0.4313 (var=0.0096)
- pi_5: 0.4237 (var=0.0670)
- pi_4_1: 0.5069 (var=0.0237)

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
  - iter 1: 0.1554 (var=0.1563) (Δ vs real -0.1104)
  - iter 2: 0.4204 (var=0.0740) (Δ vs real +0.1546)
  - iter 3 (most recent): 0.4792 (var=0.1807) (Δ vs real +0.2133)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2971 (var=0.0096)
- pi_3: -0.2708 (var=0.0095)
- pi_1: -0.6967 (var=0.0260)
- pi_2: 0.3704 (var=0.0144)
- pi_5: 0.3858 (var=0.0225)
- pi_4_1: 0.2446 (var=0.0156)

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
  - iter 1: 0.0908 (var=0.0755) (Δ vs real +0.0042)
  - iter 2: 0.2700 (var=0.0520) (Δ vs real +0.1833)
  - iter 3 (most recent): 0.5625 (var=0.1145) (Δ vs real +0.4758)
**Other theories' values on this metric (for reference):**
- pi_3: -0.5333 (var=0.0223)
- pi_4: 0.2158 (var=0.0210)
- pi_1: -0.7375 (var=0.0404)
- pi_2: 0.6717 (var=0.0518)
- pi_5: 0.1383 (var=0.0736)
- pi_4_1: 0.1075 (var=0.0264)

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
  - iter 1: 0.2713 (var=0.1159) (Δ vs real +0.1988)
  - iter 2: 0.4375 (var=0.1581) (Δ vs real +0.3650)
  - iter 3 (most recent): 0.0575 (var=0.0333) (Δ vs real -0.0150)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0025 (var=0.0525)
- pi_5: 0.3325 (var=0.0612)
- pi_1: 0.0288 (var=0.0221)
- pi_2: 0.0137 (var=0.0602)
- pi_3: -0.0025 (var=0.0792)
- pi_4_1: -0.0400 (var=0.0672)

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
  - iter 1: 1.0388 (var=1.1173) (Δ vs real +0.0288)
  - iter 2: 0.3088 (var=0.3698) (Δ vs real -0.7012)
  - iter 3 (most recent): -0.0537 (var=0.2483) (Δ vs real -1.0637)
**Other theories' values on this metric (for reference):**
- pi_5: 1.3512 (var=0.8533)
- pi_4: -0.1263 (var=0.1938)
- pi_1: 1.4112 (var=0.1813)
- pi_2: -2.0162 (var=0.4200)
- pi_3: 1.7625 (var=0.1534)
- pi_4_1: 0.5012 (var=0.5143)

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
  - iter 1: 0.1521 (var=0.0321) (Δ vs real -0.1529)
  - iter 2: 0.1371 (var=0.0221) (Δ vs real -0.1679)
  - iter 3 (most recent): 0.2250 (var=0.1197) (Δ vs real -0.0800)
**Other theories' values on this metric (for reference):**
- pi_4_1: 0.4188 (var=0.0158)
- pi_5: 0.2196 (var=0.0342)
- pi_1: -0.7067 (var=0.0351)
- pi_2: 0.0204 (var=0.0104)
- pi_3: -0.0142 (var=0.0125)
- pi_4: 0.4296 (var=0.0093)

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
  - iter 1: -0.2098 (var=0.0013) (Δ vs real -0.0432)
  - iter 2: -0.2424 (var=0.0001) (Δ vs real -0.0757)
  - iter 3 (most recent): -0.2400 (var=0.0002) (Δ vs real -0.0733)
**Other theories' values on this metric (for reference):**
- pi_5: -0.2057 (var=0.0022)
- pi_4_1: -0.2374 (var=0.0002)
- pi_1: -0.1112 (var=0.0033)
- pi_2: -0.1264 (var=0.0029)
- pi_3: -0.1529 (var=0.0007)
- pi_4: -0.2445 (var=0.0000)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Subjects carry a latent valence orientation: roughly 85% read a binary expert rating of 1 as a defect and 0 as clean (inverse valence), while the remaining subjects use the opposite positive valence. On each trial, each feature is probabilistically gated into an unweighted tally with inclusion probability logistic(alpha * validity_j + gamma). The highest-validity cue is included with probability above 0.95, while lower-validity cues are included with moderate probability so that their majorities can still move the tally. The gated tally drives a high-gain softmax with subject-specific beta_tally. When the gated tally is exactly zero, subjects scan only the included cues in descending validity order and choose the first valence-consistent cue; if no included cue discriminates, they guess. All core subject-level parameters are drawn from heavy-tailed or logit-normal transforms to preserve large between-subject variance.

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
    # truncated log-normal / normal / logit-normal distributions.
    u_alpha = (float(parameters['alpha_raw']) - 0.01) / 0.98
    u_alpha = float(np.clip(u_alpha, 0.0, 1.0))
    alpha = float(np.exp(_trunc_qnorm(
        u_alpha, np.log(6.5), 0.30, np.log(4.5), np.log(9.5))))

    u_gamma = (float(parameters['gamma_raw']) - 0.01) / 0.98
    u_gamma = float(np.clip(u_gamma, 0.0, 1.0))
    gamma = float(_trunc_qnorm(
        u_gamma, -4.8, 0.85, -6.5, -2.8))

    u_bt = (float(parameters['beta_tally_raw']) - 0.01) / 0.98
    u_bt = float(np.clip(u_bt, 0.0, 1.0))
    beta_tally = float(np.exp(_trunc_qnorm(
        u_bt, np.log(1.8), 0.55, np.log(0.5), np.log(5.0))))

    u_tb = (float(parameters['beta_tiebreak_raw']) - 0.01) / 0.98
    u_tb = float(np.clip(u_tb, 0.0, 1.0))
    beta_tiebreak = float(np.exp(_trunc_qnorm(
        u_tb, np.log(1.75), 0.45, np.log(0.6), np.log(4.5))))

    u_eps = (float(parameters['epsilon_raw']) - 0.01) / 0.98
    u_eps = float(np.clip(u_eps, 0.0, 1.0))
    logit_lo = np.log(0.02 / 0.98)
    logit_hi = np.log(0.22 / 0.78)
    logit_mu = np.log(0.09 / 0.91)
    z_eps = _trunc_qnorm(u_eps, logit_mu, 0.65, logit_lo, logit_hi)
    epsilon = float(1.0 / (1.0 + np.exp(-z_eps)))

    # Latent valence orientation.  The probability of inverse valence is
    # approximately 0.85, with the remainder using positive valence.
    inv = float(parameters['orientation_u']) < 0.85

    # Inclusion probabilities from a validity logistic gate.
    # Force the highest-validity cue to be included with p >= 0.95.
    vmax = float(np.max(val))
    p_top = _sigmoid(alpha * vmax + gamma)
    p_top = float(np.clip(p_top, 0.95, 0.998))
    gamma_eff = np.log(p_top / (1.0 - p_top)) - alpha * vmax

    logit_j = alpha * val + gamma_eff
    p_j = 1.0 / (1.0 + np.exp(-logit_j))
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

    d = float(np.sum(good_a[included]) - np.sum(good_b[included]))

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
- validities: validities

`rationale`: This theory implements the prescribed valence-mixture validity-gated tally mechanism while correcting the failures of pi_4_1 and pi_5. Unlike pi_4_1, cue use is probabilistic rather than all-or-none, and an inverse/positive valence mixture is added; this should reduce the overstrong top/lower-cue interaction in Experiment 9 and the under-predicted compensation effect in Experiment 8. Unlike pi_5, the highest-validity cue is included nearly deterministically (p >= 0.95), so Experiment 3 retains a strong anti-take-the-best tie-break signal, while lower-validity cues are still included often enough for their majorities to affect Experiments 2, 8, and 10. The valence mixture is compensated by high-gain tally and tie-break softmax processes, which keeps the inverse-tally contrasts in Experiments 1, 2, and 4 near the observed human values. Subject-level parameters are log-normal, truncated-normal, or logit-normal, preserving the large between-subject variances seen across experiments.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter's prescribed family (valence-mixture validity-gated tally with near-certain top-cue inclusion, softmax tally, included-cue tie-break scan by descending validity, heavy-tailed subject parameters) and it wins decisively on some experiments: E1 is nearly exact (-0.513 vs -0.512), E6 is essentially exact (0.091 vs 0.087), E8 is far better than any prior theory (1.039 vs real 1.010), and E10 is close (-0.210 vs -0.167). However, three experiments show clear directional failures that are too large to ignore. E7 over-shoots by a factor of roughly 3.7 (0.271 vs 0.073): lower-validity cue majorities are moving choices far more than humans allow, meaning the top cue loses too often when the lower tally outvotes it. E9 under-shoots by half (0.152 vs 0.305): the first-discriminating-cue contrast is diluted, so the validity-ordered tie-break scan is not decisive enough and/or the valence-mixture plus epsilon dilution is too strong. E5 also under-shoots (0.155 vs 0.266). E2 and E3 under-shoot the inverse-tally contrast by roughly 0.09 each (-0.556 vs -0.644 and -0.339 vs -0.430). The between-subject variances are also badly inflated in E1-E4 (candidate 0.15-0.26 vs real 0.02-0.07), indicating the heavy-tailed gate parameter draws are too wide even though E5 variance happens to match. The core tension to resolve is that E2/E10 want lower-cue majorities to matter while E7 wants the top cue to dominate a count of lower cues; a pure unweighted count tally with one monotone logistic gating curve struggles to produce both.
Rationale: Keep the prescribed family but make three targeted changes, in priority order. (1) Make the included tally validity-weighted rather than a pure count: each included cue j contributes weight exp(k * validity_j) with a subject-specific k drawn near 2.5-3.0. A single top cue can then beat two or three mid-validity cues in E7 (fixing the 0.271 over-shoot toward 0.1), while the larger lower-cue blocks in E2, E8, and E10 still move the tally (k must stay below roughly 3.5 or the two-vs-four cue splits in E2 flip sign; check this numerically). If weighting is judged off-limits, the fallback is to push alpha up to 9-12 and lower gamma so mid-validity cues fall to p around 0.35-0.45, but that will further dilute E9 and E2, so weighting is the preferred minimal change. (2) Strengthen the inverse contrasts globally: raise the inverse-valence share from 0.85 to roughly 0.90-0.93 and lower the epsilon mean toward 0.05-0.06. For E9 specifically also raise the mean beta_tiebreak (e.g., logit-normal centered at log(3) instead of log(1.75)) so the descending-validity scan decides more sharply; E9's contrast is only about half of real and needs both less dilution and a more decisive scan. This scale-up also lifts E2, E3, and E5, and is safe because E1, E4, E6, and E8 are currently at or slightly below real and can absorb a modest increase. (3) Shrink the between-subject heterogeneity of the gate parameters: narrow the alpha truncation to roughly 5.0-8.5 with sigma 0.15 and gamma to roughly -5.5 to -3.5 with sigma 0.6, so the E1-E4 variances drop from 0.15-0.26 toward the observed 0.02-0.07 while keeping enough heterogeneity for E5 and E8. The earlier heavy tails overshot the actual between-subject spread on the tally-contrast metrics.

**Outcome of this advice:** iter 1 candidate loss=0.1282 -> iter 2 candidate loss=0.2332 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: This candidate was rejected by the accept gate (loss 0.2332 against the running best 0.1282), so it will not become the base. It regressed on seven of ten experiments while helping only a few. E6 moved from a near-exact 0.0908 to 0.2700 against real 0.0867: raising alpha to 8 made inclusion strongly validity-selective, so the gated tally effectively became a validity-weighted tally and produced exactly the high-versus-low validity weighting interaction the metric is designed to detect. E1 fell from -0.5133 to -0.4154 (real -0.5117), E4 from 0.4456 to 0.2700 (real 0.4925), and E8 collapsed from 1.0388 to 0.3088 (real 1.0100): these are tally-path dilutions caused mainly by lowering beta_tally from about 1.8 to 0.85. E7 did not improve under the sharper gate; it worsened from 0.2713 to 0.4375 against real 0.0725, so sharper inclusion was the wrong lever for the top-cue-versus-lower-majority cells. E10 also worsened from -0.2098 to -0.2424 (real -0.1667), meaning lower-validity blocks carry even less weight than humans give them. E5 overshot from 0.1554 to 0.4204 (real 0.2658), showing that the combined inverse-valence share 0.90, beta_tiebreak 2.5, and epsilon 0.07 over-corrected. The real gains are narrow: E3 moved to -0.4892 (real -0.4300) with variance 0.0276 (real 0.0221), and E4 variance fell to 0.0726 (real 0.0658). These support keeping the narrowed gate parameter dispersions, but not the other changes. The implementation also drifts from the arbiter's prescription in one respect: gamma_raw is transformed into gamma but then overwritten by gamma_eff recomputed from the p_top clip, so gamma is effectively dead and the stated alpha-and-gamma calibration is not actually performed.
Rationale: Do not repeat the previous rejected direction (continuous validity weighting or further alpha sharpening). Build on the iter-1 base and make smaller edits. (1) Restore beta_tally centered near log(1.8) with a narrow dispersion (about 0.25-0.35), because the 0.85 median diluted E1, E2, E4, E8, and E10. (2) Restore alpha centered near 6.5; alpha 8 broke E6 and did not fix E7. Keep the narrowed alpha and gamma dispersions only, since those brought E3 and E4 variances close to real. (3) Restore gamma as a live parameter rather than deriving gamma_eff from a p_top clip, so the arbiter's alpha-and-gamma calibration is real; keep the top cue inclusion pinned above 0.95 through the chosen gamma center. (4) Use intermediate valence and decision settings to avoid re-oscillation: inverse valence share about 0.86-0.88 (between the accepted 0.85 and the rejected 0.90), epsilon near 0.08, and beta_tiebreak near 2.0-2.25 (between the accepted 1.75 and the rejected 2.5). This should lift E2, E3, and E9 modestly without re-triggering the E5 overshoot. (5) For E7, change mechanism within the family instead of sharpening the gate: add a modest subject-level attention bonus so the single highest-validity included discriminating cue contributes weight w_top (median around 1.5-2.5, bounded 1.0-3.0) to the otherwise unweighted tally, and/or extend the exact-zero tie-break to near-zero margins with |d| <= 1 so the descending-validity included-cue scan resolves small deficits in favor of the top cue. Both keep lower-validity majorities in E2 and E10 able to move the tally while preventing one to three lower cues in E7 from outvoting the top cue, pushing E7 from about 0.27 toward the observed 0.07. Re-check E6 after this change and require it to stay near 0.09-0.15: a top-only bonus is much less correlated with the metric's linear validity weights than a full continuous weighting, so it should be safe. (6) For E9, inspect the inclusion probabilities at its 0.62-0.65 first-discriminator positions; if raising those p values to about 0.75-0.85 via the live gamma is possible without lifting E7's lower-cue p, do so, otherwise rely on the strengthened tie-break and the slightly higher inverse share. Keep the exact-zero guess rule and the included-cue descending-validity scan so the proposal remains inside the prescribed valence-mixture validity-gated tally family.

**Outcome of this advice:** iter 2 candidate loss=0.2332 -> iter 3 candidate loss=0.1966 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate was rejected by the accept gate (aggregate loss 0.1966 vs running best 0.1282), and the per-experiment numbers show exactly why. It collapsed nearly every experiment that requires lower-validity cue majorities to move the tally: E2 fell to -0.3387 (real -0.6438, accepted base -0.5563), E4 to 0.2200 (real 0.4925, base 0.4456), E8 to -0.0537 (real 1.0100, base 1.0388), E10 to -0.2400 (real -0.1667, base -0.2098), and it distorted E6 to 0.5625 (real 0.0867, base 0.0908), E5 to 0.4792 (real 0.2658, base 0.1554), and E1 to -0.4204 (real -0.5117, base -0.5133). The single cause is the widened near-zero tie-break arm: because inclusion probabilities for mid-to-high validities are roughly 0.7 to 0.95 and the experiments have few features, the gated tally lands in -1, 0, +1 on a very large share of trials. On all those trials the descending-validity included-cue scan discards the lower-validity majority and hands the decision to the first high-validity discriminator, which is an override of the majority tally, not a tie-break. That is precisely why E2/E4/E8/E10 lost their lower-cue contrast and why E5/E6 inflated: near-zero tallies occur constantly, so the tally path is bypassed most of the time. The candidate's genuine wins are E7 (0.0575 vs real 0.0725, from base 0.2713), E3 (-0.4375 vs real -0.4300, from base -0.3387), and E9 (0.2250 vs real 0.3050, from base 0.1521), plus better E4 variance (0.0421 vs real 0.0658). Those wins are real but they were purchased with the same near-zero arm and the strengthened tie-break settings that destroyed the majority-driven experiments. The between-subject variances on E1/E3 are still inflated relative to the real data (E1 0.0818 vs 0.0161, E3 0.0985 vs 0.0221), so the narrowing of subject-level draws was only partially sufficient.
Rationale: The loop already rejected this candidate, so the next build will be on the accepted iter-1 base. Do not carry the abs(d) <= 1 arm forward: it is the single largest error and it is not a minor tuning issue. Build on iter-1 with these in-family edits, in priority order. (1) Restore the tie-break threshold to exact zero only (d == 0, not |d| <= 1), and restore the iter-1 tally path: alpha centered near 6.5 with a narrow dispersion, beta_tally centered near 1.8, live gamma that is not overwritten, and the top-cue inclusion pin kept in 0.95-0.998. This should bring E1, E2, E4, E6, E8, and E10 back to their near-real iter-1 values. (2) Keep only the components of the rejected candidate that demonstrably helped: the narrowed alpha/gamma/beta_tally subject dispersions (these reduced E3/E4 variances) and the live-gamma structure. Do not re-introduce global validity weighting or alpha sharpening, and do not re-introduce the wide tie-break arm; both prior directions were rejected by the gate. (3) Rebalance the tie-break decisiveness because iter-3 overshot E5 badly at beta_tiebreak 2.1. Grid beta_tiebreak (roughly 1.5 to 1.8), inverse-valence share (0.85 to 0.87), and epsilon (0.07 to 0.09), and choose the combination that keeps E5 inside 0.20 to 0.32 while keeping E9 as high as possible and holding E2 <= -0.60, E4 >= 0.42, E1 <= -0.48, and E6 inside 0.05 to 0.15. E5 and E9 are in tension through the tie-break, so this must be a constrained search rather than a single knob move. (4) For E7, do not use the near-zero arm again. E7 is a near-zero metric for symmetric mechanisms and most reference theories sit between -0.04 and +0.03, so first re-measure iter-1's E7 value with a larger simulation sample to separate the 0.2713 point estimate from sampling noise (its variance was 0.1159). If the overshoot is systematic, attack it by narrowing the gate-parameter dispersions further so the top cue is reliably included with probability at least 0.95 in every subject; if it is still too high, add the previously suggested but unimplemented small subject-level attention bonus for only the single highest-validity included discriminator, median weight about 1.3-1.6 and bounded 1.0-2.0, applied inside the tally only, with no override of tallies whose magnitude is 2 or more. Re-check E6 and E10 after any such bonus and require E6 to stay 0.05-0.15 and E10 not to fall below -0.23.

**Outcome of this advice:** iter 3 candidate loss=0.1966 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Subjects carry a latent valence orientation, with roughly 86% treating a binary expert rating of 1 as a defect and 0 as clean, while the remainder use the opposite positive valence. On each trial, each feature is probabilistically gated into an unweighted tally according to a validity logistic whose top-validity cue is included with high probability, while lower-validity cues are included often enough for their majorities to matter. Tally choices follow a softmax with subject-specific gain. Exact zero tallies are resolved by scanning only included cues in descending validity order and choosing the first valence-consistent discriminator. In addition, when the included tally is just one cue away from zero, the single highest-validity included discriminator receives a small attention bonus inside the tally, making the top cue somewhat more decisive without overriding larger lower-cue majorities. Subject-level parameters are drawn from narrowed heavy-tailed or logit-normal distributions to keep between-subject variance modest where the data are consistent while preserving heterogeneity where it is large.",
  "predict": "import numpy as np\nfrom scipy.stats import norm\n\n\ndef _trunc_qnorm(u, mu, sigma, lo, hi):\n    u = float(np.clip(u, 0.0, 1.0))\n    q_lo = norm.cdf((lo - mu) / sigma)\n    q_hi = norm.cdf((hi - mu) / sigma)\n    q = q_lo + u * (q_hi - q_lo)\n    return mu + sigma * norm.ppf(q)\n\n\ndef _sigmoid(x):\n    return 1.0 / (1.0 + np.exp(-x))\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus.')\n\n    a = stim[0]\n    b = stim[1]\n    val = np.asarray(parameters['validities'], dtype=float)\n    n_features = stim.shape[1]\n    if val.shape[0] != n_features:\n        raise ValueError('validities length mismatch')\n\n    # Subject-level transforms: raw uniform draws are mapped through\n    # truncated normal / log-normal / logit-normal distributions.\n    u_alpha = (float(parameters['alpha_raw']) - 0.01) / 0.98\n    u_alpha = float(np.clip(u_alpha, 0.0, 1.0))\n    alpha = float(np.exp(_trunc_qnorm(\n        u_alpha, np.log(6.5), 0.20, np.log(5.5), np.log(8.5))))\n\n    # Live gamma for the validity gate.  It is not overwritten below; only\n    # the single highest-validity cue's inclusion probability is pinned high.\n    u_gamma = (float(parameters['gamma_raw']) - 0.01) / 0.98\n    u_gamma = float(np.clip(u_gamma, 0.0, 1.0))\n    gamma = float(_trunc_qnorm(u_gamma, -3.15, 0.35, -3.85, -2.45))\n\n    u_bt = (float(parameters['beta_tally_raw']) - 0.01) / 0.98\n    u_bt = float(np.clip(u_bt, 0.0, 1.0))\n    beta_tally = float(np.exp(_trunc_qnorm(\n        u_bt, np.log(1.85), 0.28, np.log(1.20), np.log(2.90))))\n\n    u_tb = (float(parameters['beta_tiebreak_raw']) - 0.01) / 0.98\n    u_tb = float(np.clip(u_tb, 0.0, 1.0))\n    beta_tiebreak = float(np.exp(_trunc_qnorm(\n        u_tb, np.log(2.00), 0.35, np.log(1.20), np.log(3.20))))\n\n    u_eps = (float(parameters['epsilon_raw']) - 0.01) / 0.98\n    u_eps = float(np.clip(u_eps, 0.0, 1.0))\n    logit_lo = np.log(0.03 / 0.97)\n    logit_hi = np.log(0.16 / 0.84)\n    logit_mu = np.log(0.07 / 0.93)\n    z_eps = _trunc_qnorm(u_eps, logit_mu, 0.45, logit_lo, logit_hi)\n    epsilon = float(1.0 / (1.0 + np.exp(-z_eps)))\n\n    # Small top-cue attention bonus, used only for near-zero tally margins.\n    u_top = (float(parameters['top_bonus_raw']) - 0.01) / 0.98\n    u_top = float(np.clip(u_top, 0.0, 1.0))\n    extra = float(_trunc_qnorm(u_top, 0.40, 0.15, 0.00, 0.80))\n    w_top = 1.0 + max(0.0, min(0.80, extra))\n\n    # Latent valence orientation.  Approximately 86% use inverse valence.\n    inv = float(parameters['orientation_u']) < 0.86\n\n    # Inclusion probabilities from a validity logistic gate.\n    # Only the highest-validity cue's inclusion probability is forced high.\n    logit_j = alpha * val + gamma\n    p_j = _sigmoid(logit_j)\n    if n_features > 0:\n        j_top = int(np.argmax(val))\n        p_j[j_top] = float(np.clip(p_j[j_top], 0.95, 0.998))\n    p_j = np.clip(p_j, 0.02, 0.99)\n\n    included = np.flatnonzero(np.random.uniform(size=n_features) < p_j)\n    if included.size == 0:\n        return np.ones(2) / 2.0\n\n    # Valence-consistent cue comparison on the included cues only.\n    if inv:\n        good_a = (a == 0) & (b == 1)\n        good_b = (a == 1) & (b == 0)\n    else:\n        good_a = (a == 1) & (b == 0)\n        good_b = (a == 0) & (b == 1)\n\n    d_raw = float(np.sum(good_a[included]) - np.sum(good_b[included]))\n    d = d_raw\n\n    # Top-cue attention bonus.  It operates inside the tally path only, and\n    # only when the current tally is one cue away from zero.  Larger\n    # lower-cue majorities are not overridden, and exact-zero ties still\n    # go to the validity-ordered tie-break scan.\n    if d_raw != 0.0 and abs(d_raw) <= 1.0:\n        disc = [j for j in included if a[j] != b[j]]\n        if disc:\n            j_att = max(disc, key=lambda j: float(val[j]))\n            if inv:\n                if a[j_att] == 0 and b[j_att] == 1:\n                    d = d_raw + (w_top - 1.0)\n                elif a[j_att] == 1 and b[j_att] == 0:\n                    d = d_raw - (w_top - 1.0)\n            else:\n                if a[j_att] == 1 and b[j_att] == 0:\n                    d = d_raw + (w_top - 1.0)\n                elif a[j_att] == 0 and b[j_att] == 1:\n                    d = d_raw - (w_top - 1.0)\n\n    if d != 0.0:\n        scores = np.array([d, -d], dtype=float)\n        beta = beta_tally\n    else:\n        discriminating = [j for j in included if a[j] != b[j]]\n        if len(discriminating) == 0:\n            return np.ones(2) / 2.0\n\n        discriminating = sorted(discriminating, key=lambda j: -val[j])\n        winner = None\n        for j in discriminating:\n            if inv:\n                if a[j] == 0 and b[j] == 1:\n                    winner = 0\n                    break\n                elif a[j] == 1 and b[j] == 0:\n                    winner = 1\n                    break\n            else:\n                if a[j] == 1 and b[j] == 0:\n                    winner = 0\n                    break\n                elif a[j] == 0 and b[j] == 1:\n                    winner = 1\n                    break\n\n        if winner is None:\n            return np.ones(2) / 2.0\n\n        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])\n        beta = beta_tiebreak\n\n    logits = beta * scores\n    logits = logits - np.max(logits)\n    p_core = np.exp(logits)\n    p_core = p_core / p_core.sum()\n\n    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0\n",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0:\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "orientation_u": "[0, 1]",
    "alpha_raw": "[0.01, 0.99]",
    "gamma_raw": "[0.01, 0.99]",
    "beta_tally_raw": "[0.01, 0.99]",
    "beta_tiebreak_raw": "[0.01, 0.99]",
    "epsilon_raw": "[0.01, 0.99]",
    "top_bonus_raw": "[0.01, 0.99]",
    "validities": "validities"
  },
  "rationale": "This is a minimal edit of the accepted base that applies the latest feedback without carrying forward the rejected |d|<=1 tie-break override. The exact-zero tie-break is preserved, alpha/beta/gamma dispersions are narrowed to reduce inflated between-subject variances, gamma is now used directly with only the top cue's inclusion probability pinned high, and the tally gain is kept close to 1.8. The inverse-valence share is raised only slightly to 0.86, epsilon is centered at 0.07, and beta_tiebreak is moved to 2.0 to lift the under-predicted inverse-tally and tie-break contrasts in E2, E3, E4, E5, and E9 while avoiding the prior E5 overshoot caused by the wide near-zero arm. For E7, a small top-cue attention bonus is added only inside the tally path for |d|<=1 and d!=0, so the highest-validity discriminator is more decisive in near-zero tallies without overriding larger lower-cue majorities; this directly targets the E7 overshoot while protecting E6 and E10."
}
```

## Usage

```json
{
  "input_tokens": 23842,
  "output_tokens": 22526,
  "reasoning_tokens": 19830,
  "provider": "Baidu"
}
```
