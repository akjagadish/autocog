# generation_iter_00_attempt_00

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
**Other theories' values on this metric (for reference):**
- pi_5: -0.2057 (var=0.0022)
- pi_4_1: -0.2374 (var=0.0002)
- pi_1: -0.1112 (var=0.0033)
- pi_2: -0.1264 (var=0.0029)
- pi_3: -0.1529 (var=0.0007)
- pi_4: -0.2445 (var=0.0000)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

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
  "theory": "Subjects carry a latent valence orientation: roughly 85% read a binary expert rating of 1 as a defect and 0 as clean (inverse valence), while the remaining subjects use the opposite positive valence. On each trial, each feature is probabilistically gated into an unweighted tally with inclusion probability logistic(alpha * validity_j + gamma). The highest-validity cue is included with probability above 0.95, while lower-validity cues are included with moderate probability so that their majorities can still move the tally. The gated tally drives a high-gain softmax with subject-specific beta_tally. When the gated tally is exactly zero, subjects scan only the included cues in descending validity order and choose the first valence-consistent cue; if no included cue discriminates, they guess. All core subject-level parameters are drawn from heavy-tailed or logit-normal transforms to preserve large between-subject variance.",
  "predict": "import numpy as np\nfrom scipy.stats import norm\n\n\ndef _trunc_qnorm(u, mu, sigma, lo, hi):\n    u = float(np.clip(u, 0.0, 1.0))\n    q_lo = norm.cdf((lo - mu) / sigma)\n    q_hi = norm.cdf((hi - mu) / sigma)\n    q = q_lo + u * (q_hi - q_lo)\n    return mu + sigma * norm.ppf(q)\n\n\ndef _sigmoid(x):\n    return 1.0 / (1.0 + np.exp(-x))\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus.')\n\n    a = stim[0]\n    b = stim[1]\n    val = np.asarray(parameters['validities'], dtype=float)\n    n_features = stim.shape[1]\n    if val.shape[0] != n_features:\n        raise ValueError('validities length mismatch')\n\n    # Subject-level transforms: raw uniform draws are mapped through\n    # truncated log-normal / normal / logit-normal distributions.\n    u_alpha = (float(parameters['alpha_raw']) - 0.01) / 0.98\n    u_alpha = float(np.clip(u_alpha, 0.0, 1.0))\n    alpha = float(np.exp(_trunc_qnorm(\n        u_alpha, np.log(6.5), 0.30, np.log(4.5), np.log(9.5))))\n\n    u_gamma = (float(parameters['gamma_raw']) - 0.01) / 0.98\n    u_gamma = float(np.clip(u_gamma, 0.0, 1.0))\n    gamma = float(_trunc_qnorm(\n        u_gamma, -4.8, 0.85, -6.5, -2.8))\n\n    u_bt = (float(parameters['beta_tally_raw']) - 0.01) / 0.98\n    u_bt = float(np.clip(u_bt, 0.0, 1.0))\n    beta_tally = float(np.exp(_trunc_qnorm(\n        u_bt, np.log(1.8), 0.55, np.log(0.5), np.log(5.0))))\n\n    u_tb = (float(parameters['beta_tiebreak_raw']) - 0.01) / 0.98\n    u_tb = float(np.clip(u_tb, 0.0, 1.0))\n    beta_tiebreak = float(np.exp(_trunc_qnorm(\n        u_tb, np.log(1.75), 0.45, np.log(0.6), np.log(4.5))))\n\n    u_eps = (float(parameters['epsilon_raw']) - 0.01) / 0.98\n    u_eps = float(np.clip(u_eps, 0.0, 1.0))\n    logit_lo = np.log(0.02 / 0.98)\n    logit_hi = np.log(0.22 / 0.78)\n    logit_mu = np.log(0.09 / 0.91)\n    z_eps = _trunc_qnorm(u_eps, logit_mu, 0.65, logit_lo, logit_hi)\n    epsilon = float(1.0 / (1.0 + np.exp(-z_eps)))\n\n    # Latent valence orientation.  The probability of inverse valence is\n    # approximately 0.85, with the remainder using positive valence.\n    inv = float(parameters['orientation_u']) < 0.85\n\n    # Inclusion probabilities from a validity logistic gate.\n    # Force the highest-validity cue to be included with p >= 0.95.\n    vmax = float(np.max(val))\n    p_top = _sigmoid(alpha * vmax + gamma)\n    p_top = float(np.clip(p_top, 0.95, 0.998))\n    gamma_eff = np.log(p_top / (1.0 - p_top)) - alpha * vmax\n\n    logit_j = alpha * val + gamma_eff\n    p_j = 1.0 / (1.0 + np.exp(-logit_j))\n    p_j = np.clip(p_j, 0.02, 0.99)\n\n    included = np.flatnonzero(np.random.uniform(size=n_features) < p_j)\n    if included.size == 0:\n        return np.ones(2) / 2.0\n\n    # Valence-consistent cue comparison on the included cues only.\n    if inv:\n        good_a = (a == 0) & (b == 1)\n        good_b = (a == 1) & (b == 0)\n    else:\n        good_a = (a == 1) & (b == 0)\n        good_b = (a == 0) & (b == 1)\n\n    d = float(np.sum(good_a[included]) - np.sum(good_b[included]))\n\n    if d != 0.0:\n        scores = np.array([d, -d], dtype=float)\n        beta = beta_tally\n    else:\n        discriminating = [j for j in included if a[j] != b[j]]\n        if len(discriminating) == 0:\n            return np.ones(2) / 2.0\n\n        discriminating = sorted(discriminating, key=lambda j: -val[j])\n        winner = None\n        for j in discriminating:\n            if inv:\n                if a[j] == 0 and b[j] == 1:\n                    winner = 0\n                    break\n                elif a[j] == 1 and b[j] == 0:\n                    winner = 1\n                    break\n            else:\n                if a[j] == 1 and b[j] == 0:\n                    winner = 0\n                    break\n                elif a[j] == 0 and b[j] == 1:\n                    winner = 1\n                    break\n\n        if winner is None:\n            return np.ones(2) / 2.0\n\n        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])\n        beta = beta_tiebreak\n\n    logits = beta * scores\n    logits = logits - np.max(logits)\n    p_core = np.exp(logits)\n    p_core = p_core / p_core.sum()\n\n    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0:\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "orientation_u": "[0, 1]",
    "alpha_raw": "[0.01, 0.99]",
    "gamma_raw": "[0.01, 0.99]",
    "beta_tally_raw": "[0.01, 0.99]",
    "beta_tiebreak_raw": "[0.01, 0.99]",
    "epsilon_raw": "[0.01, 0.99]",
    "validities": "validities"
  },
  "rationale": "This theory implements the prescribed valence-mixture validity-gated tally mechanism while correcting the failures of pi_4_1 and pi_5. Unlike pi_4_1, cue use is probabilistic rather than all-or-none, and an inverse/positive valence mixture is added; this should reduce the overstrong top/lower-cue interaction in Experiment 9 and the under-predicted compensation effect in Experiment 8. Unlike pi_5, the highest-validity cue is included nearly deterministically (p >= 0.95), so Experiment 3 retains a strong anti-take-the-best tie-break signal, while lower-validity cues are still included often enough for their majorities to affect Experiments 2, 8, and 10. The valence mixture is compensated by high-gain tally and tie-break softmax processes, which keeps the inverse-tally contrasts in Experiments 1, 2, and 4 near the observed human values. Subject-level parameters are log-normal, truncated-normal, or logit-normal, preserving the large between-subject variances seen across experiments."
}
```

## Usage

```json
{
  "input_tokens": 16742,
  "output_tokens": 42504,
  "reasoning_tokens": 40266,
  "provider": "Baidu"
}
```
