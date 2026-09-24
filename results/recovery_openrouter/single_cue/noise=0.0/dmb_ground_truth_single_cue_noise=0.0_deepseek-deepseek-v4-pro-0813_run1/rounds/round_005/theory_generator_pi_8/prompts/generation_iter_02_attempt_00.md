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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Replace pi_7 with a distinctiveness-gated low-cue dominance theory. Keep the useful median/MAD-normalized deviance idea from pi_5, but do not use pi_7's hard share cap or its size-suppressed rank baseline. In the new theory, a below-median cue's weight should increase with both its deviance and its distinctiveness, defined as the normalized validity gap to the next-lowest cue. A single extreme low cue should therefore receive very large weight, producing the strong follow-the-0.50-cue behavior observed in Experiment 2, while a homogeneous below-median bloc should share weight more evenly. Add a small and strictly bounded high-validity rank baseline whose strength depends on high-bloc size and low-cue distinctiveness, allowing the high-cue sweep in Experiment 1 to produce a contrast near 0.48 and preserving the positive high-validity correlation in Experiment 12 without overriding an extreme outlier. Replace the current product-of-gates beta machinery with a simpler entropy/conflict gate: choice temperature should be low when weight entropy is low and one distinct low cue dominates the signed evidence, moderate when a high-cue sweep opposes a solitary outlier, and near-chance when evidence is low-spread, balanced, or ambiguous due to many similar cues, as in Experiments 8 and 9. This theory differs from pi_7 by removing the cap-and-redistribution core and by making the high-rank baseline conditional rather than size-suppressed; it differs from pi_5 by adding explicit cue distinctiveness and a simpler entropy-based stochasticity term, so it can sharpen Experiment 2 and damp the overpredicted Experiment 8 while retaining pi_5's good fits elsewhere.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.813)

**Description**
People assign choice weight by each expert's rank inside the advertised validity ensemble, strongly discounting higher ranks. The steepness of that discount is fixed, but the weight floor given to the high-validity tail is gated by how large the lower-validity bloc is. When low-validity experts form a large subgroup of the ensemble, a small floor is added so that the many high-validity cues are not completely ignored; this prevents the extreme rank-reversal saturation seen in balanced low/high ensembles. When low-validity experts are a small minority, the floor is zero, the high-validity tail is almost fully suppressed, and the response temperature is lowered so that the remaining single-cue contrasts are not over-saturated. Signed weighted tallies, with ties contributing zero, then pass through the regime-appropriate fixed-temperature softmax with a small lapse probability.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            'Rank-salience integration expects a (2, n_features) stimulus; got shape {}.'.format(stim.shape)
        )

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    n_feat = validities.shape[0]
    if n_feat != stim.shape[1]:
        raise ValueError(
            'validities length {} != n_features {}.'.format(n_feat, stim.shape[1])
        )

    alpha = float(parameters['alpha'])
    delta_high = float(parameters['delta_high'])
    delta_low = float(parameters['delta_low'])
    beta_high = float(parameters['beta_high'])
    beta_low = float(parameters['beta_low'])
    epsilon = float(parameters['epsilon'])
    spread_frac = float(parameters['spread_frac'])
    bottom_share_threshold = float(parameters['bottom_share_threshold'])

    vmin = float(np.min(validities))
    vmax = float(np.max(validities))
    if vmax - vmin <= 1e-12:
        bottom_share = 1.0
    else:
        lower_bloc = validities <= vmin + spread_frac * (vmax - vmin)
        bottom_share = float(np.mean(lower_bloc))

    if bottom_share >= bottom_share_threshold:
        delta = delta_high
        beta = beta_high
    else:
        delta = delta_low
        beta = beta_low

    ranks = rankdata(validities, method='average')
    if n_feat > 1:
        u = (ranks - 1.0) / (n_feat - 1.0)
    else:
        u = np.array([0.5])

    raw_weights = np.power(1.0 - u, alpha) + delta
    total = raw_weights.sum()
    if total <= 0.0:
        weights = np.ones(n_feat) / max(n_feat, 1)
    else:
        weights = raw_weights / total

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - np.max(logits))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts) / n_opts


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- alpha: [4.60, 4.90]
- delta_high: [0.11, 0.13]
- delta_low: {0.0}
- beta_high: [2.95, 3.05]
- beta_low: [1.65, 1.80]
- epsilon: {0.02}
- spread_frac: {0.25}
- bottom_share_threshold: {0.42}
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1359 (var=0.0002)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.8021 (var=0.0109)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=-0.3640 (var=0.0059)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.7150 (var=0.0209)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.7621 (var=0.0044)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.3529 (var=0.0067)
- Experiment 7: real=-0.0103 (var=0.0009) vs this=0.0282 (var=0.0018)
- Experiment 8: real=0.0472 (var=0.0067) vs this=0.0794 (var=0.0104)
- Experiment 9: real=0.0032 (var=0.0007) vs this=0.0500 (var=0.0010)
- Experiment 10: real=0.2001 (var=0.0006) vs this=0.1905 (var=0.0004)
- Experiment 11: real=0.4833 (var=0.0144) vs this=0.6117 (var=0.0132)
- Experiment 12: real=0.8375 (var=0.0133) vs this=0.5475 (var=0.0085)


---

### `pi_7` (overall score: 0.701)

**Description**
People evaluate each advertised expert validity relative to the ensemble median, so that cues below the median acquire context-sensitive salience through a median-normalized deviance score. A rank-based baseline keeps higher-validity cues from being completely ignored, but only when the genuine low-validity outlier bloc is small; as the number of below-median outlier cues grows, this high-validity baseline is smoothly suppressed. Cue weights are capped by a smooth, low-bloc-size-dependent maximal share, so that no single cue can monopolize the decision while still allowing redistribution toward high-validity cues when several low-validity cues compete. Choice gain is set by smooth functions of low-outlier bloc size, validity gap, and effective number of competing cues.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Bounded contextual salience expects a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    n_feat = validities.shape[0]
    if n_feat != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    c = float(parameters['c'])
    median = float(np.median(validities))
    abs_dev = np.abs(validities - median)
    mad = float(np.median(abs_dev))
    scale = mad + c
    if scale <= 1e-12:
        scale = 1e-12

    deviance = np.maximum(0.0, median - validities) / scale

    low_dev_threshold = float(parameters['low_dev_threshold'])
    low_bloc_frac = float(np.mean(deviance > low_dev_threshold))

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    dev_weight = w0 + kappa * np.power(deviance, gamma)

    ranks = rankdata(validities, method='average')
    if n_feat > 1:
        u = (ranks - 1.0) / (n_feat - 1.0)
    else:
        u = np.array([0.5])

    b_scale = float(parameters['b_scale'])
    b_base = float(parameters['b_base'])
    b_alpha = float(parameters['b_alpha'])
    gap_center = float(parameters['gap_center'])
    gap_width = float(parameters['gap_width'])

    gap_norm = (float(np.max(validities)) - float(np.min(validities))) / (float(np.median(validities)) + c)
    gap_activ = 1.0 / (1.0 + np.exp(-float(np.clip((gap_norm - gap_center) / gap_width, -500.0, 500.0))))

    rank_gate_center = float(parameters['rank_gate_center'])
    rank_gate_width = float(parameters['rank_gate_width'])
    rank_gate = 1.0 / (1.0 + np.exp(float(np.clip((rank_gate_center - low_bloc_frac) / rank_gate_width, -500.0, 500.0))))
    rank_base = b_scale * (b_base + gap_activ) * np.power(u, b_alpha) * rank_gate

    raw = dev_weight + rank_base
    total = float(np.sum(raw))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw / total

    cap_base = float(parameters['cap_base'])
    cap_peak = float(parameters['cap_peak'])
    cap_center = float(parameters['cap_center'])
    cap_width = float(parameters['cap_width'])
    cap_share = cap_base + (cap_peak - cap_base) * np.exp(-0.5 * ((low_bloc_frac - cap_center) / cap_width) ** 2)
    cap_share = float(np.clip(cap_share, 0.20, 0.50))

    for _ in range(100):
        max_w = float(np.max(weights)) if weights.size else 0.0
        if max_w <= cap_share + 1e-12:
            break
        weights = np.minimum(weights, cap_share)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
            break
        weights = weights / total_w

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    sorted_w = np.sort(weights)[::-1]
    top1 = float(sorted_w[0]) if sorted_w.size > 0 else 1.0
    top2 = float(sorted_w[1]) if sorted_w.size > 1 else top1
    top3 = float(sorted_w[2]) if sorted_w.size > 2 else (float(sorted_w[-1]) if sorted_w.size > 0 else 0.0)
    top3_sim = 1.0 - float(top1 - top3)
    effective_n = 1.0 / max(float(np.sum(weights * weights)), 1e-12)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -500.0, 500.0))))

    share_hi_center = float(parameters['share_hi_center'])
    share_hi_width = float(parameters['share_hi_width'])
    gap_hi_center = float(parameters['gap_hi_center'])
    gap_hi_width = float(parameters['gap_hi_width'])
    share_gate = sigmoid((top1 - share_hi_center) / share_hi_width)
    weight_gap_gate = sigmoid(((top1 - top2) - gap_hi_center) / gap_hi_width)
    high_gate = share_gate * weight_gap_gate

    eff_n_center = float(parameters['eff_n_center'])
    eff_n_width = float(parameters['eff_n_width'])
    top3_sim_center = float(parameters['top3_sim_center'])
    top3_sim_width = float(parameters['top3_sim_width'])
    low_bloc_center = float(parameters['low_bloc_center'])
    low_bloc_width = float(parameters['low_bloc_width'])

    eff_gate = np.exp(-0.5 * ((effective_n - eff_n_center) / eff_n_width) ** 2)
    balance_gate = np.exp(-0.5 * ((top3_sim - top3_sim_center) / top3_sim_width) ** 2)
    low_size_gate = np.exp(-0.5 * ((low_bloc_frac - low_bloc_center) / low_bloc_width) ** 2)
    low_gate = eff_gate * balance_gate * low_size_gate

    beta_base = float(parameters['beta_base'])
    beta_floor_hi = float(parameters['beta_floor_hi'])
    beta_boost_lo = float(parameters['beta_boost_lo'])
    beta_eff = beta_base + beta_boost_lo * low_gate
    beta_eff = beta_eff * (1.0 - high_gate) + beta_floor_hi * high_gate
    beta_eff = float(np.clip(beta_eff, 0.10, 12.0))

    epsilon = float(parameters['epsilon'])
    logits = np.array([evidence_A, 0.0], dtype=float)
    z = beta_eff * (logits - float(np.max(logits)))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts, dtype=float) / float(n_opts)


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- c: [0.008, 0.014]
- low_dev_threshold: {0.70}
- rank_gate_center: {0.19}
- rank_gate_width: {0.07}
- w0: [0.005, 0.015]
- kappa: [1.8, 2.2]
- gamma: [1.3, 1.5]
- b_scale: [0.30, 0.50]
- b_base: [0.15, 0.25]
- b_alpha: [1.3, 1.7]
- gap_center: {0.38}
- gap_width: {0.10}
- cap_base: {0.26}
- cap_peak: {0.44}
- cap_center: {0.22}
- cap_width: {0.10}
- share_hi_center: {0.85}
- share_hi_width: {0.02}
- gap_hi_center: {0.40}
- gap_hi_width: {0.06}
- eff_n_center: {3.0}
- eff_n_width: {0.14}
- top3_sim_center: {0.95}
- top3_sim_width: {0.07}
- low_bloc_center: {0.48}
- low_bloc_width: {0.10}
- beta_base: [2.4, 2.8]
- beta_floor_hi: [0.35, 0.50]
- beta_boost_lo: [0.35, 0.60]
- epsilon: [0.04, 0.06]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1246 (var=0.0002)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.7272 (var=0.0075)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=-0.2726 (var=0.0068)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.6675 (var=0.0201)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.8321 (var=0.0036)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.4271 (var=0.0084)
- Experiment 7: real=-0.0103 (var=0.0009) vs this=0.0224 (var=0.0016)
- Experiment 8: real=0.0472 (var=0.0067) vs this=0.0680 (var=0.0084)
- Experiment 9: real=0.0032 (var=0.0007) vs this=0.0023 (var=0.0005)
- Experiment 10: real=0.2001 (var=0.0006) vs this=0.2038 (var=0.0002)
- Experiment 11: real=0.4833 (var=0.0144) vs this=0.7900 (var=0.0109)
- Experiment 12: real=0.8375 (var=0.0133) vs this=0.3594 (var=0.0070)


---

### `pi_5` (overall score: 0.667)

**Description**
People normalize each advertised cue validity relative to the current task ensemble, giving salience only to cues whose validity falls below the ensemble median. Low-validity outliers dominate the signed feature tally, with ties contributing zero. The decisiveness of the softmax choice rule is itself context-gated: when a single cue becomes a near-monopoly in normalized decision weight, the response temperature is quenched toward chance; when a small set of low-validity cues is nearly co-equal, the response is sharpened. This keeps equal-weight tallying as a limiting case while allowing reversed-validity-like and regime-sensitive choice behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus array.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    median = np.median(validities)
    mad = np.median(np.abs(validities - median))
    c = float(parameters['c'])

    denom = mad + c
    if denom <= 0.0:
        denom = 1e-12
    deviance = np.maximum(0.0, median - validities) / denom

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    weights = w0 + kappa * np.power(deviance, gamma)

    total = weights.sum()
    if total <= 0.0:
        weights = np.ones_like(weights) / max(weights.size, 1)
    else:
        weights = weights / total

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    sorted_w = np.sort(weights)[::-1]
    nw = sorted_w.shape[0]
    top1 = sorted_w[0]
    top2 = sorted_w[1] if nw > 1 else top1
    top3 = sorted_w[2] if nw > 2 else (sorted_w[-1] if nw > 0 else 0.0)
    top3_similarity = 1.0 - float(top1 - top3)
    effective_n = 1.0 / max(float(np.sum(weights ** 2)), 1e-12)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -500.0, 500.0))))

    share_hi_center = float(parameters['share_hi_center'])
    share_hi_width = float(parameters['share_hi_width'])
    gap_hi_center = float(parameters['gap_hi_center'])
    gap_hi_width = float(parameters['gap_hi_width'])
    beta_floor_hi = float(parameters['beta_floor_hi'])

    share_gate = sigmoid((top1 - share_hi_center) / share_hi_width)
    gap_gate = sigmoid(((top1 - top2) - gap_hi_center) / gap_hi_width)
    high_gate = share_gate * gap_gate

    eff_n_center = float(parameters['eff_n_center'])
    eff_n_width = float(parameters['eff_n_width'])
    top3_sim_center = float(parameters['top3_sim_center'])
    top3_sim_width = float(parameters['top3_sim_width'])
    beta_boost_lo = float(parameters['beta_boost_lo'])

    eff_gate = np.exp(-0.5 * ((effective_n - eff_n_center) / eff_n_width) ** 2)
    balance_gate = np.exp(-0.5 * ((top3_similarity - top3_sim_center) / top3_sim_width) ** 2)
    low_gate = eff_gate * balance_gate

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    beta_eff = beta + beta_boost_lo * low_gate
    beta_eff = beta_eff * (1.0 - high_gate) + beta_floor_hi * high_gate
    beta_eff = float(np.clip(beta_eff, 0.05, 15.0))

    logits = np.array([evidence_A, 0.0])
    z = beta_eff * (logits - np.max(logits))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts) / n_opts


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- w0: [0.005, 0.04]
- kappa: [1.5, 4.0]
- gamma: [1.0, 1.5]
- c: [0.005, 0.02]
- beta: [1.7, 2.3]
- epsilon: [0.12, 0.18]
- share_hi_center: [0.84, 0.92]
- share_hi_width: [0.012, 0.035]
- gap_hi_center: [0.42, 0.58]
- gap_hi_width: [0.03, 0.08]
- beta_floor_hi: [0.25, 0.35]
- eff_n_center: [2.95, 3.10]
- eff_n_width: [0.10, 0.18]
- top3_sim_center: [0.92, 0.98]
- top3_sim_width: [0.04, 0.10]
- beta_boost_lo: [0.8, 1.4]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1025 (var=0.0003)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.7328 (var=0.0080)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=-0.3414 (var=0.0125)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.6958 (var=0.0145)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.8250 (var=0.0030)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.3704 (var=0.0099)
- Experiment 7: real=-0.0103 (var=0.0009) vs this=0.0068 (var=0.0020)
- Experiment 8: real=0.0472 (var=0.0067) vs this=-0.0010 (var=0.0097)
- Experiment 9: real=0.0032 (var=0.0007) vs this=0.0003 (var=0.0008)
- Experiment 10: real=0.2001 (var=0.0006) vs this=0.1290 (var=0.0004)
- Experiment 11: real=0.4833 (var=0.0144) vs this=0.4950 (var=0.0193)
- Experiment 12: real=0.8375 (var=0.0133) vs this=0.6706 (var=0.0116)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1155 -> ACCEPTED
- iter 2: loss=0.0744 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.0744 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    def subject_slope(sdf):
        a = np.array(sdf['option_a_ratings'].tolist(), dtype=float)
        b = np.array(sdf['option_b_ratings'].tolist(), dtype=float)
        resp = sdf['response'].to_numpy(dtype=float)
        n = a.shape[0]

        top = np.full(n, -1, dtype=int)
        top[a[:, 0] > b[:, 0]] = 0
        top[b[:, 0] > a[:, 0]] = 1
        valid = top >= 0

        opp_wins = np.zeros(n, dtype=int)
        for j in range(1, a.shape[1]):
            opp_wins[(top == 0) & (b[:, j] > a[:, j])] += 1
            opp_wins[(top == 1) & (a[:, j] > b[:, j])] += 1

        choose_top = np.where(resp == top, 1.0, 0.0)

        x = opp_wins[valid].astype(float)
        y = choose_top[valid]
        if x.size < 2 or np.ptp(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    slopes = data.groupby('subject_id', sort=False).apply(subject_slope)
    vals = np.atleast_1d(np.asarray(slopes, dtype=float))
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** -0.0959 (var=0.0010)
**Previous candidate values (this loop):**
  - iter 1: -0.1346 (var=0.0002) (Δ vs real -0.0387)
  - iter 2 (most recent): -0.1163 (var=0.0003) (Δ vs real -0.0204)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0006 (var=0.0002)
- pi_2: -0.1360 (var=0.0009)
- pi_3: -0.1493 (var=0.0006)
- pi_4: -0.1505 (var=0.0003)
- pi_5: -0.1025 (var=0.0003)
- pi_6: -0.1359 (var=0.0002)
- pi_7: -0.1246 (var=0.0002)

### Experiment 2
**Design**
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.6], dtype=float)
    cue_order = np.argsort(-validities, kind='stable')

    y = data['response'].astype(float).to_numpy()
    dirs = np.zeros(len(data), dtype=float)
    margins = np.zeros(len(data), dtype=float)

    for i, (_, row) in enumerate(data.iterrows()):
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)

        margins[i] = float(np.sum(a > b) - np.sum(b > a))

        direction = 0.0
        for j in cue_order:
            if a[j] > b[j]:
                direction = 1.0
                break
            if b[j] > a[j]:
                direction = -1.0
                break
        dirs[i] = direction

    d_pos = dirs == 1.0
    d_neg = dirs == -1.0
    m_pos = margins > 0.0
    m_neg = margins < 0.0

    directional_contrast = y[d_neg].mean() - y[d_pos].mean()
    margin_contrast = y[m_neg].mean() - y[m_pos].mean()

    return float(directional_contrast - margin_contrast)

```

**Observed (real) value:** -0.8592 (var=0.0589)
**Previous candidate values (this loop):**
  - iter 1: -0.7985 (var=0.0094) (Δ vs real +0.0607)
  - iter 2 (most recent): -0.6581 (var=0.0165) (Δ vs real +0.2011)
**Other theories' values on this metric (for reference):**
- pi_2: -0.7597 (var=0.0423)
- pi_1: 0.7279 (var=0.0420)
- pi_3: -0.6946 (var=0.0665)
- pi_4: -0.8321 (var=0.0089)
- pi_5: -0.7328 (var=0.0080)
- pi_6: -0.8021 (var=0.0109)
- pi_7: -0.7272 (var=0.0075)

### Experiment 3
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data):
    pairs_ordered = [
        (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
        (0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0),
        (1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0),
        (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0),
        (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
        (1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1),
        (1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1),
        (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)
    ]
    order = {p: i for i, p in enumerate(pairs_ordered)}

    def pair_key(row):
        return tuple(list(row['option_a_ratings']) + list(row['option_b_ratings']))

    keys = data.apply(pair_key, axis=1)
    pos = keys.map(order)
    if pos.isna().any():
        return 0.0

    rank = (pos % 8).to_numpy(dtype=float) + 1.0
    chose_a = (data['response'].to_numpy() == 0).astype(float)

    subj_corrs = []
    for sid in data['subject_id'].unique():
        mask = data['subject_id'].to_numpy() == sid
        if np.std(rank[mask]) == 0.0 or np.std(chose_a[mask]) == 0.0:
            subj_corrs.append(0.0)
            continue
        r = float(np.corrcoef(rank[mask], chose_a[mask])[0, 1])
        subj_corrs.append(r)

    return float(np.mean(subj_corrs))
```

**Observed (real) value:** -0.3485 (var=0.0198)
**Previous candidate values (this loop):**
  - iter 1: -0.5104 (var=0.0062) (Δ vs real -0.1619)
  - iter 2 (most recent): -0.4343 (var=0.0060) (Δ vs real -0.0858)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5096 (var=0.0191)
- pi_2: 0.0215 (var=0.0080)
- pi_1: 0.5469 (var=0.0262)
- pi_4: -0.2535 (var=0.0057)
- pi_5: -0.3414 (var=0.0125)
- pi_6: -0.3640 (var=0.0059)
- pi_7: -0.2726 (var=0.0068)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    high_a = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    low_a = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)

    a_key = data['option_a_ratings'].apply(tuple)
    mask_high_a = a_key == high_a
    mask_low_a = a_key == low_a

    if mask_high_a.sum() == 0 or mask_low_a.sum() == 0:
        return 0.0

    pA_high = 1.0 - float(data.loc[mask_high_a, 'response'].mean())
    pA_low = 1.0 - float(data.loc[mask_low_a, 'response'].mean())

    return float(pA_high - pA_low)

```

**Observed (real) value:** -0.6533 (var=0.0604)
**Previous candidate values (this loop):**
  - iter 1: -0.7417 (var=0.0169) (Δ vs real -0.0883)
  - iter 2 (most recent): -0.6425 (var=0.0170) (Δ vs real +0.0108)
**Other theories' values on this metric (for reference):**
- pi_2: -0.1783 (var=0.0317)
- pi_3: 0.8683 (var=0.0156)
- pi_1: 0.6692 (var=0.0641)
- pi_4: -0.7908 (var=0.0142)
- pi_5: -0.6958 (var=0.0145)
- pi_6: -0.7150 (var=0.0209)
- pi_7: -0.6675 (var=0.0201)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    total = 0.0
    n = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_low = int(a[0]) + int(a[1])
        b_low = int(b[0]) + int(b[1])
        if a_low == b_low:
            continue
        low_side = 'A' if a_low > b_low else 'B'
        opposing = b if low_side == 'A' else a
        k = sum(1 for j in range(2, 9) if int(opposing[j]) == 1)
        if 3 <= k <= 6:
            chose_low = ((low_side == 'A' and int(row['response']) == 0) or
                         (low_side == 'B' and int(row['response']) == 1))
            total += float(chose_low)
            n += 1
    if n == 0:
        return 0.5
    return total / float(n)
```

**Observed (real) value:** 0.8275 (var=0.0166)
**Previous candidate values (this loop):**
  - iter 1: 0.9500 (var=0.0014) (Δ vs real +0.1225)
  - iter 2 (most recent): 0.9058 (var=0.0019) (Δ vs real +0.0783)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8642 (var=0.0036)
- pi_2: 0.1492 (var=0.0089)
- pi_1: 0.1346 (var=0.0111)
- pi_3: 0.0708 (var=0.0019)
- pi_5: 0.8250 (var=0.0030)
- pi_6: 0.7621 (var=0.0044)
- pi_7: 0.8321 (var=0.0036)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    pos1 = {
        ((1, 1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((0, 1, 1, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 0, 1, 1)),
        ((0, 1, 1, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((0, 0, 1, 1, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0, 1, 1)),
    }
    neg1 = {
        ((0, 0, 1, 0, 0, 1, 1, 0), (1, 1, 0, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0, 0, 0)),
        ((0, 0, 0, 1, 0, 1, 1, 0), (0, 1, 1, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1, 1, 1), (0, 1, 1, 1, 0, 0, 0, 0)),
    }
    early2 = {
        ((1, 1, 1, 1, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((1, 1, 1, 1, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0, 1, 1)),
        ((0, 1, 1, 1, 1, 1, 0, 0), (1, 0, 0, 0, 0, 0, 1, 1)),
        ((0, 1, 1, 1, 0, 0, 1, 1), (1, 0, 0, 0, 1, 1, 0, 0)),
    }
    late2 = {
        ((0, 0, 1, 1, 0, 1, 1, 1), (1, 1, 0, 0, 1, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 0, 0, 0, 0, 0)),
        ((0, 0, 1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 0, 0, 0, 0)),
        ((1, 0, 0, 0, 1, 1, 1, 1), (0, 1, 1, 1, 0, 0, 0, 0)),
    }
    akey = data['option_a_ratings'].apply(lambda x: tuple([int(v) for v in x]))
    bkey = data['option_b_ratings'].apply(lambda x: tuple([int(v) for v in x]))
    pairs = list(zip(akey, bkey))
    p_a = 1.0 - data['response'].to_numpy(dtype=float)
    p_pos1 = p_a[[k in pos1 for k in pairs]].mean()
    p_neg1 = p_a[[k in neg1 for k in pairs]].mean()
    p_early2 = p_a[[k in early2 for k in pairs]].mean()
    p_late2 = p_a[[k in late2 for k in pairs]].mean()
    block1_contrast = p_pos1 - p_neg1
    block2_contrast = p_early2 - p_late2
    return float(0.5 * (block1_contrast + block2_contrast))
```

**Observed (real) value:** 0.1525 (var=0.0118)
**Previous candidate values (this loop):**
  - iter 1: 0.4483 (var=0.0068) (Δ vs real +0.2958)
  - iter 2 (most recent): 0.3875 (var=0.0082) (Δ vs real +0.2350)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0092 (var=0.0068)
- pi_4: 0.6663 (var=0.0054)
- pi_1: -0.4425 (var=0.0114)
- pi_3: -0.6067 (var=0.0096)
- pi_5: 0.3704 (var=0.0099)
- pi_6: 0.3529 (var=0.0067)
- pi_7: 0.4271 (var=0.0084)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    # High-validity experts are the first six rating positions.
    # Low-validity experts are positions 6,7,8; their split identifies the block.
    blocks = {}
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        high_diff = a[:6] - b[:6]
        h = float(np.sum(high_diff))
        key = tuple(int(x) for x in high_diff)
        low_sign = int((a[6] - b[6]) + (a[7] - b[7]) + (a[8] - b[8]))
        yA = 1.0 if int(row.response) == 0 else 0.0
        d = blocks.setdefault(low_sign, {})
        entry = d.setdefault(key, [0.0, 0, h])
        entry[0] += yA
        entry[1] += 1
    vals = []
    for d in blocks.values():
        if len(d) < 2:
            continue
        entries = list(d.values())
        hs = np.array([e[2] for e in entries], dtype=float)
        ps = np.array([e[0] / e[1] if e[1] > 0 else 0.5 for e in entries], dtype=float)
        z = hs - hs.mean()
        denom = float(np.sqrt(np.mean(z * z)))
        if denom <= 0.0:
            continue
        vals.append(float(np.mean((z / denom) * ps)))
    return float(np.mean(vals)) if vals else 0.0
```

**Observed (real) value:** -0.0103 (var=0.0009)
**Previous candidate values (this loop):**
  - iter 1: 0.0417 (var=0.0017) (Δ vs real +0.0520)
  - iter 2 (most recent): 0.0030 (var=0.0019) (Δ vs real +0.0132)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2803 (var=0.0012)
- pi_5: 0.0068 (var=0.0020)
- pi_1: 0.1888 (var=0.0043)
- pi_2: 0.2909 (var=0.0059)
- pi_3: 0.3502 (var=0.0022)
- pi_6: 0.0282 (var=0.0018)
- pi_7: 0.0224 (var=0.0016)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np

def metric(data):
    chose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    a_high = np.array([sum(r[2:]) for r in data['option_a_ratings']], dtype=float)
    b_high = np.array([sum(r[2:]) for r in data['option_b_ratings']], dtype=float)
    high_evidence = a_high - b_high
    if float(high_evidence.std()) < 1e-12 or float(chose_a.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(high_evidence, chose_a)[0, 1])
```

**Observed (real) value:** 0.0472 (var=0.0067)
**Previous candidate values (this loop):**
  - iter 1: -0.0118 (var=0.0112) (Δ vs real -0.0590)
  - iter 2 (most recent): 0.0012 (var=0.0091) (Δ vs real -0.0460)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0010 (var=0.0097)
- pi_4: 0.6671 (var=0.0037)
- pi_1: 0.3105 (var=0.0223)
- pi_2: 0.6061 (var=0.0242)
- pi_3: 0.7504 (var=0.0068)
- pi_6: 0.0794 (var=0.0104)
- pi_7: 0.0680 (var=0.0084)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df['y'] = (df['response'] == 0).astype(float)
    df['low_a'] = df['option_a_ratings'].apply(lambda r: int(np.sum(np.asarray(r)[:4])))
    df['high_a'] = df['option_a_ratings'].apply(lambda r: int(np.sum(np.asarray(r)[4:9])))
    df['low_tie'] = (df['low_a'] == 2).astype(float)
    df['a_maj'] = (df['low_a'] == 3).astype(float)

    slopes = []
    for subj_id, sub in df.groupby('subject_id', sort=False):
        X = np.column_stack((
            np.ones(len(sub)),
            sub['low_tie'].to_numpy(dtype=float),
            sub['a_maj'].to_numpy(dtype=float),
            sub['high_a'].to_numpy(dtype=float)
        ))
        y = sub['y'].to_numpy(dtype=float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        slopes.append(float(beta[3]))

    return float(np.mean(slopes))
```

**Observed (real) value:** 0.0032 (var=0.0007)
**Previous candidate values (this loop):**
  - iter 1: -0.0034 (var=0.0004) (Δ vs real -0.0066)
  - iter 2 (most recent): 0.0022 (var=0.0010) (Δ vs real -0.0010)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0500 (var=0.0010)
- pi_5: 0.0003 (var=0.0008)
- pi_1: 0.1056 (var=0.0010)
- pi_2: 0.1807 (var=0.0017)
- pi_3: 0.2205 (var=0.0005)
- pi_4: 0.1640 (var=0.0005)
- pi_7: 0.0023 (var=0.0005)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np

def metric(data):
    validities = np.array([0.5, 0.84, 0.843, 0.846, 0.849, 0.852, 0.855, 0.858, 0.861, 0.864, 0.867])
    order = np.argsort(validities)
    ranks = np.empty(len(validities), dtype=float)
    ranks[order] = np.arange(1, len(validities) + 1, dtype=float)
    u = (ranks - 1.0) / (len(validities) - 1.0)
    alpha = 4.75
    raw = np.power(1.0 - u, alpha)
    weights = raw / raw.sum()

    A = np.stack([np.asarray(v, dtype=float) for v in data['option_a_ratings']])
    B = np.stack([np.asarray(v, dtype=float) for v in data['option_b_ratings']])
    signed_evidence = (A - B) @ weights
    chose_a = 1.0 - np.asarray(data['response'], dtype=float)
    return float(np.mean(signed_evidence * chose_a))
```

**Observed (real) value:** 0.2001 (var=0.0006)
**Previous candidate values (this loop):**
  - iter 1: 0.2101 (var=0.0002) (Δ vs real +0.0099)
  - iter 2 (most recent): 0.2034 (var=0.0002) (Δ vs real +0.0033)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1290 (var=0.0004)
- pi_6: 0.1905 (var=0.0004)
- pi_1: 0.1377 (var=0.0003)
- pi_2: 0.2228 (var=0.0005)
- pi_3: 0.2121 (var=0.0003)
- pi_4: 0.2385 (var=0.0002)
- pi_7: 0.2038 (var=0.0002)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    response = np.asarray(data['response'], dtype=float)
    choice_a = (response == 0).astype(float)

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    diff = a - b

    low_favors_a = diff[:, 0] > 0.5
    high_a_count = np.sum(diff[:, 1:], axis=1)

    la_k0 = low_favors_a & (high_a_count == 0)
    la_k7 = low_favors_a & (high_a_count == 7)
    lb_k0 = (~low_favors_a) & (high_a_count == 0)
    lb_k7 = (~low_favors_a) & (high_a_count == 7)

    def prop_a(mask):
        n = int(np.sum(mask))
        if n == 0:
            return 0.0
        return float(np.sum(choice_a[mask]) / n)

    la_contrast = prop_a(la_k7) - prop_a(la_k0)
    lb_contrast = prop_a(lb_k7) - prop_a(lb_k0)
    return 0.5 * (la_contrast + lb_contrast)

```

**Observed (real) value:** 0.4833 (var=0.0144)
**Previous candidate values (this loop):**
  - iter 1: 0.5050 (var=0.0026) (Δ vs real +0.0217)
  - iter 2 (most recent): 0.5200 (var=0.0043) (Δ vs real +0.0367)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7900 (var=0.0109)
- pi_5: 0.4950 (var=0.0193)
- pi_1: 0.8517 (var=0.0151)
- pi_2: 0.8650 (var=0.0147)
- pi_3: 0.9367 (var=0.0079)
- pi_4: 0.9033 (var=0.0065)
- pi_6: 0.6117 (var=0.0132)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = data['option_a_ratings']
    b = data['option_b_ratings']

    # Feature index 0 is the 0.50-validity cue.
    # Feature indices 1..5 are the five 0.80-validity cues.
    f1 = a.apply(lambda x: float(x[0])) - b.apply(lambda x: float(x[0]))
    grp = a.apply(lambda x: float(sum(x[1:6]))) - b.apply(lambda x: float(sum(x[1:6])))

    # High-conflict trials: the lone 0.50 cue points opposite the 0.80 bloc,
    # and the 0.80 bloc has a net majority of at least 3 (i.e. +/-3 or +/-5).
    conflict = ((f1 * grp) < 0) & (grp.abs() >= 3)

    if int(conflict.sum()) == 0:
        return 0.5

    # Follow the lone 0.50 cue: choose A when it favors A, choose B when it favors B.
    follow_lone = ((f1 > 0) & (data['response'] == 0)) | ((f1 < 0) & (data['response'] == 1))

    return float(follow_lone[conflict].mean())

```

**Observed (real) value:** 0.8375 (var=0.0133)
**Previous candidate values (this loop):**
  - iter 1: 0.9487 (var=0.0016) (Δ vs real +0.1112)
  - iter 2 (most recent): 0.8750 (var=0.0037) (Δ vs real +0.0375)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6706 (var=0.0116)
- pi_7: 0.3594 (var=0.0070)
- pi_1: 0.3181 (var=0.0072)
- pi_2: 0.1288 (var=0.0070)
- pi_3: 0.0569 (var=0.0024)
- pi_4: 0.1156 (var=0.0048)
- pi_6: 0.5475 (var=0.0085)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
People anchor each advertised expert validity to the current ensemble rather than using the absolute validity scale. Experts below the ensemble median earn decision weight through a median/MAD-normalized deviance score, and that deviance-based weight is multiplicatively amplified by cue distinctiveness: the normalized validity distance to the nearest other below-median cue, or to the median when a below-median cue is solitary. A homogeneous low-validity bloc therefore shares weight evenly, while a single extreme low-validity cue becomes highly salient and tends to dominate the signed evidence. A small, strictly bounded rank-based baseline is added for above-median experts; its strength increases with the size of the high-validity bloc and is reduced when lower-validity cues are highly distinctive, so a large high-validity sweep can contribute without overriding an extreme low-cue outlier. Choice stochasticity is governed by a simple entropy/conflict gate: when normalized cue-weight entropy is low, evidence is spread strongly, and a distinctive low cue dominates the signed tally, the choice gain is high; when a high-validity sweep opposes a solitary low cue, gain is pushed toward a moderate regime; and when many similar cues produce high weight entropy, weak evidence spread, or balanced evidence, the gain collapses toward chance.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Distinctive low-cue dominance expects a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    n_feat = validities.shape[0]
    if n_feat != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    c = float(parameters['c'])
    median = float(np.median(validities))
    abs_dev = np.abs(validities - median)
    mad = float(np.median(abs_dev))
    scale = mad + c
    if scale <= 1e-12:
        scale = 1e-12

    dev = np.maximum(0.0, median - validities) / scale
    below = validities < median - 1e-12

    # Distinctiveness is the normalized gap to the nearest other below-median
    # cue, or to the median when there is only one below-median cue.
    distinct = np.zeros(n_feat, dtype=float)
    low_idx = np.where(below)[0]
    if low_idx.size == 1:
        distinct[low_idx[0]] = dev[low_idx[0]]
    elif low_idx.size > 1:
        for i in low_idx:
            others = [j for j in low_idx if j != i]
            if others:
                dgap = min(abs(float(validities[j]) - float(validities[i])) for j in others)
                distinct[i] = dgap / scale
            else:
                distinct[i] = dev[i]

    gamma = float(parameters['gamma'])
    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    lam = float(parameters['lam'])
    distinct_scale = float(parameters['distinct_scale'])
    eta = float(parameters['eta'])

    dev_bounded = np.minimum(dev, 100.0)
    dist_score = np.tanh(distinct / distinct_scale)

    low_raw = np.zeros(n_feat, dtype=float)
    low_raw[below] = (
        w0
        + kappa * np.power(dev_bounded[below], gamma)
        * (1.0 + lam * np.power(dist_score[below], eta))
    )

    # Small, conditional high-validity rank baseline.
    high_mask = ~below
    high_raw = np.zeros(n_feat, dtype=float)
    if high_mask.any():
        if below.any():
            dmax = float(np.max(distinct))
        else:
            dmax = 0.0

        d_gate = float(parameters['d_gate'])
        baseline_factor = 1.0 / (1.0 + (dmax / d_gate) ** 2)

        vh = validities[high_mask]
        nh = int(vh.size)
        high_frac = nh / float(n_feat)
        ranks = rankdata(vh, method='average')
        if nh > 1:
            u = (ranks - 1.0) / (nh - 1.0)
        else:
            u = np.array([1.0], dtype=float)

        b_scale = float(parameters['b_scale'])
        b_pow = float(parameters['b_pow'])
        b_base = float(parameters['b_base'])
        b_rank = float(parameters['b_rank'])
        b_alpha = float(parameters['b_alpha'])
        b_cap = float(parameters['b_cap'])

        raw_h = (
            b_scale
            * np.power(high_frac, b_pow)
            * baseline_factor
            * (b_base + b_rank * np.power(u, b_alpha))
        )
        raw_h = np.minimum(raw_h, b_cap)
        high_raw[high_mask] = raw_h

    raw = low_raw + high_raw
    total = float(np.sum(raw))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw / total

    diff = stim[0] - stim[1]
    evidence = float(np.dot(weights, diff))

    low_w = weights * below.astype(float)
    high_w = weights * high_mask.astype(float)
    low_evidence = float(np.dot(low_w, diff))
    high_evidence = float(np.dot(high_w, diff))

    if low_evidence * high_evidence < 0.0:
        conflict = min(abs(low_evidence), abs(high_evidence)) / (abs(evidence) + 1e-12)
        conflict = float(np.clip(conflict, 0.0, 1.0))
    else:
        conflict = 0.0

    en = -np.sum(weights * np.log(weights + 1e-12))
    entropy = float(en / np.log(max(n_feat, 2)))

    H_thresh = float(parameters['H_thresh'])
    s_H = float(parameters['s_H'])
    concentration = 1.0 / (1.0 + np.exp((entropy - H_thresh) / s_H))

    if below.any():
        dmax = float(np.max(distinct))
    else:
        dmax = 0.0
    D_half = float(parameters['D_half'])
    distinct_factor = 0.5 + 0.5 * np.tanh(dmax / D_half)

    low_share = abs(low_evidence) / (abs(low_evidence) + abs(high_evidence) + 1e-12)

    e_half = float(parameters['e_half'])
    spread = abs(evidence) / (abs(evidence) + e_half)

    beta_base = float(parameters['beta_base'])
    beta_range = float(parameters['beta_range'])
    conflict_damp = float(parameters['conflict_damp'])
    beta_min = float(parameters['beta_min'])
    beta_max = float(parameters['beta_max'])

    drive = concentration * low_share * distinct_factor * spread
    beta = beta_base + beta_range * drive * (1.0 - conflict_damp * conflict)
    beta = float(np.clip(beta, beta_min, beta_max))

    epsilon = float(parameters['epsilon'])
    logits = np.array([evidence, 0.0], dtype=float)
    z = beta * (logits - float(np.max(logits)))
    e = np.exp(z)
    core = e / np.sum(e)

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts, dtype=float) / float(n_opts)


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- c: [0.008, 0.015]
- gamma: [1.25, 1.50]
- w0: [0.06, 0.12]
- kappa: [2.3, 3.0]
- lam: {0.6}
- distinct_scale: [1.4, 2.4]
- eta: {1.0}
- b_scale: [0.30, 0.50]
- b_pow: {1.0}
- b_base: [0.02, 0.07]
- b_rank: [0.10, 0.20]
- b_alpha: [0.9, 1.2]
- b_cap: [0.15, 0.30]
- d_gate: [8.5, 14.0]
- H_thresh: [0.78, 0.88]
- s_H: [0.05, 0.10]
- D_half: [1.0, 2.0]
- e_half: [0.10, 0.30]
- beta_base: [0.35, 0.58]
- beta_range: [3.6, 5.4]
- conflict_damp: [0.35, 0.60]
- beta_min: [0.10, 0.25]
- beta_max: [4.6, 6.4]
- epsilon: [0.10, 0.16]
- validities: validities

`rationale`: This is a minimal recalibration of the previous distinctiveness-gated low-cue dominance model rather than a mechanism change. The predict/policy code is unchanged; only the sampled parameter ranges are edited along the lines of the critic feedback. Solitary low-cue overdominance is tempered by lowering lam to 0.6 and widening distinct_scale, while kappa and w0 are increased to preserve strong homogeneous below-median blocs such as Experiment 2. d_gate is substantially increased so the high-validity rank baseline is less suppressed at moderate distinctiveness, which should pull the lone-0.50 follow rate in Experiment 12 toward ~0.84 instead of ~0.95 while leaving cases with near-zero or very large distinctiveness largely intact. The beta ceiling and beta range are reduced from the previous extreme values, and epsilon is raised, to remove the over-sharpened ceiling effects that produced too-high contrasts in Experiments 5, 6, and 12. The high-baseline scale and cap are lowered slightly so low-cue evidence can remain dominant in homogeneous low-bloc cases. Together these edits keep the same distinctiveness-gated low-cue dominance architecture but soften the previous systematic over-sharpening and increase between-subject stochasticity.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter's prescribed distinctiveness-gated low-cue dominance family and captures the main qualitative signatures: strong follow-the-low-cue behavior in Exp2 and Exp12, near-chance behavior in Exp8/Exp9, good high-cue sweep contrast in Exp11, and a close fit on Exp10. However, the model systematically over-sharpens low-cue dominance and is too deterministic. Exp12 is 0.949 vs real 0.838, Exp5 is 0.950 vs 0.828, Exp6 is 0.448 vs 0.153, Exp3 is -0.510 vs -0.349, Exp4 is -0.742 vs -0.653, and Exp1 is -0.135 vs -0.096. These are all in the same direction but too extreme. The candidate also produces much lower between-subject variance than the real data in most experiments (e.g., Exp2 var 0.009 vs 0.059, Exp12 0.002 vs 0.013, Exp5 0.001 vs 0.017), indicating insufficient stochasticity or parameter heterogeneity. Exp7/Exp8/Exp9 are close enough to zero to be acceptable despite small sign differences. Exp2 is slightly under the real value (-0.798 vs -0.859), so a global weakening of low-cue weight would be the wrong fix.
Rationale: Keep the same mechanism family but recalibrate the sharpness and variance. Concretely: (1) lower beta_max/beta_range and/or raise epsilon to remove the ceiling effects in high-drive conditions that cause Exp5, Exp6, and Exp12 to overshoot; (2) increase d_gate so the high-validity baseline is less suppressed at moderate distinctiveness, which should pull Exp12's follow-the-0.50 rate from ~0.95 toward ~0.84 while leaving Exp10 (very large distinctiveness) and Exp2 (dmax=0) essentially unchanged; (3) if softening weakens Exp2, compensate by slightly increasing kappa/w0 for homogeneous below-median blocs, since those have distinctiveness near zero and will not re-trigger the solitary-outlier overdominance; (4) widen parameter ranges for beta and epsilon to restore realistic between-subject variance. Optionally reduce lam or increase distinct_scale to temper the multiplicative distinctiveness boost for solitary low cues. Do not switch away from the distinctiveness-gated low-cue dominance architecture.

**Outcome of this advice:** iter 1 candidate loss=0.1155 -> iter 2 candidate loss=0.0744 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter's prescribed distinctiveness-gated low-cue dominance family, and the recalibration produced a real aggregate improvement (loss 0.1155 -> 0.0744). It now fits well on Experiments 4, 7, 8, 9, 10, and 11, and is near-target on 1 and 12. The remaining problems are that the softening was applied too uniformly. Experiment 2 regressed substantially (-0.658 vs real -0.859, and worse than iter 1's -0.799), meaning homogeneous low-cue blocs lost too much drive. At the same time the model remains over-sharp on Experiments 5 (0.906 vs 0.828), 6 (0.388 vs 0.153), and 3 (-0.434 vs -0.349). So the entropy/conflict gate is not yet condition-specific enough: it under-drives a homogeneous multi-cue low bloc while still over-driving several high-distinctness/conflict conditions. Subject-level variance also remains much lower than human data in several experiments (e.g. Exp2 0.017 vs 0.059, Exp4 0.017 vs 0.060, Exp5 0.002 vs 0.017, Exp12 0.004 vs 0.013), suggesting insufficient between-subject stochasticity or parameter heterogeneity.
Rationale: Keep the prescribed mechanism family, but make the temperature/drive gate regime-dependent instead of globally softened. (1) The main Exp2 miss is likely caused by `distinct_factor = 0.5 + 0.5*tanh(dmax/D_half)` halving drive when the low-cue bloc is homogeneous (dmax=0). Distinctiveness already enters the cue weights, so using it again in the inverse-temperature drive double-counts it and inappropriately damps non-distinct low blocs. In the next iteration, floor `distinct_factor` at ~0.9 for multi-cue low blocs with dmax near 0, or remove it from the drive term and let entropy/low_share/conflict govern temperature. This should push Exp2 back toward the observed -0.86 while leaving Exp12 nearly unchanged because dmax is already large. (2) To avoid re-overshooting Exp5/6/12, make conflict damping condition-specific rather than globally weaker: keep or strengthen conflict damping for high-distinct/solitary low-cue conflicts, and keep beta_max/beta_range modest there. Do not broadly lower beta again, since that is what weakened Exp2. (3) Leave the high-validity rank baseline broadly alone because Exp4/10/11/12 are currently close, and only re-sweep it if these fits drift. (4) Increase true between-subject heterogeneity—wider epsilon/beta/gamma/kappa distributions or a subject-level gain term—because the candidate is still far too consistent across subjects.

**Outcome of this advice:** iter 2 candidate loss=0.0744 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "People anchor each advertised expert validity to the current ensemble, so cues below the ensemble median earn decision weight through a median/MAD-normalized deviance score that is multiplicatively amplified by cue distinctiveness. A homogeneous low-validity bloc shares weight evenly, while a single extreme low-validity cue becomes highly salient and dominates the signed evidence. A small, strictly bounded rank-based baseline keeps high-validity cues from being ignored, with strength that increases with high-bloc size and decreases when low cues are highly distinctive. Choice stochasticity is governed by a conditional entropy/conflict gate: homogeneous multi-cue low blocs receive a high temperature drive floor, whereas conflict damping is strengthened selectively when the high-validity bloc outnumbers multiple low cues or when a low cue is solitary and highly distinctive. This preserves strong homogeneous low-bloc choices and extreme-outlier following without allowing high-cue sweeps or high-distinctness conflicts to over-saturate.",
  "predict": "import numpy as np\nfrom scipy.stats import rankdata\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Distinctive low-cue dominance expects a (2, n_features) stimulus.')\n\n    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)\n    n_feat = validities.shape[0]\n    if n_feat != stim.shape[1]:\n        raise ValueError('validities length must match n_features.')\n\n    c = float(parameters['c'])\n    median = float(np.median(validities))\n    abs_dev = np.abs(validities - median)\n    mad = float(np.median(abs_dev))\n    scale = mad + c\n    if scale <= 1e-12:\n        scale = 1e-12\n\n    dev = np.maximum(0.0, median - validities) / scale\n    below = validities < median - 1e-12\n\n    distinct = np.zeros(n_feat, dtype=float)\n    low_idx = np.where(below)[0]\n    if low_idx.size == 1:\n        distinct[low_idx[0]] = dev[low_idx[0]]\n    elif low_idx.size > 1:\n        for i in low_idx:\n            others = [j for j in low_idx if j != i]\n            if others:\n                dgap = min(abs(float(validities[j]) - float(validities[i])) for j in others)\n                distinct[i] = dgap / scale\n            else:\n                distinct[i] = dev[i]\n\n    gamma = float(parameters['gamma'])\n    w0 = float(parameters['w0'])\n    kappa = float(parameters['kappa'])\n    lam = float(parameters['lam'])\n    distinct_scale = float(parameters['distinct_scale'])\n    eta = float(parameters['eta'])\n\n    dev_bounded = np.minimum(dev, 100.0)\n    dist_score = np.tanh(distinct / distinct_scale)\n\n    low_raw = np.zeros(n_feat, dtype=float)\n    low_raw[below] = (\n        w0\n        + kappa * np.power(dev_bounded[below], gamma)\n        * (1.0 + lam * np.power(dist_score[below], eta))\n    )\n\n    high_mask = ~below\n    high_raw = np.zeros(n_feat, dtype=float)\n    if high_mask.any():\n        if below.any():\n            dmax_base = float(np.max(distinct))\n        else:\n            dmax_base = 0.0\n\n        d_gate = float(parameters['d_gate'])\n        baseline_factor = 1.0 / (1.0 + (dmax_base / d_gate) ** 2)\n\n        vh = validities[high_mask]\n        nh = int(vh.size)\n        high_frac = nh / float(n_feat)\n        ranks = rankdata(vh, method='average')\n        if nh > 1:\n            u = (ranks - 1.0) / (nh - 1.0)\n        else:\n            u = np.array([1.0], dtype=float)\n\n        b_scale = float(parameters['b_scale'])\n        b_pow = float(parameters['b_pow'])\n        b_base = float(parameters['b_base'])\n        b_rank = float(parameters['b_rank'])\n        b_alpha = float(parameters['b_alpha'])\n        b_cap = float(parameters['b_cap'])\n\n        raw_h = (\n            b_scale\n            * np.power(high_frac, b_pow)\n            * baseline_factor\n            * (b_base + b_rank * np.power(u, b_alpha))\n        )\n        raw_h = np.minimum(raw_h, b_cap)\n        high_raw[high_mask] = raw_h\n\n    raw = low_raw + high_raw\n    total = float(np.sum(raw))\n    if total <= 0.0:\n        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))\n    else:\n        weights = raw / total\n\n    diff = stim[0] - stim[1]\n    evidence = float(np.dot(weights, diff))\n\n    low_w = weights * below.astype(float)\n    high_w = weights * high_mask.astype(float)\n    low_evidence = float(np.dot(low_w, diff))\n    high_evidence = float(np.dot(high_w, diff))\n\n    if low_evidence * high_evidence < 0.0:\n        conflict = min(abs(low_evidence), abs(high_evidence)) / (abs(evidence) + 1e-12)\n        conflict = float(np.clip(conflict, 0.0, 1.0))\n    else:\n        conflict = 0.0\n\n    en = -np.sum(weights * np.log(weights + 1e-12))\n    entropy = float(en / np.log(max(n_feat, 2)))\n\n    H_thresh = float(parameters['H_thresh'])\n    s_H = float(parameters['s_H'])\n    concentration = 1.0 / (1.0 + np.exp((entropy - H_thresh) / s_H))\n\n    if below.any():\n        dmax = float(np.max(distinct))\n    else:\n        dmax = 0.0\n\n    n_low = int(low_idx.size)\n    D_half = float(parameters['D_half'])\n    base_distinct_factor = 0.5 + 0.5 * np.tanh(dmax / D_half)\n\n    hom_count_thresh = float(parameters['hom_count_thresh'])\n    hom_dmax_thresh = float(parameters['hom_dmax_thresh'])\n    hom_floor = float(parameters['hom_floor'])\n    homogeneous_low_bloc = (n_low >= hom_count_thresh) and (dmax < hom_dmax_thresh)\n    if homogeneous_low_bloc:\n        distinct_factor = hom_floor + (1.0 - hom_floor) * np.tanh(dmax / D_half)\n    else:\n        distinct_factor = base_distinct_factor\n\n    low_share = abs(low_evidence) / (abs(low_evidence) + abs(high_evidence) + 1e-12)\n\n    e_half = float(parameters['e_half'])\n    spread = abs(evidence) / (abs(evidence) + e_half)\n\n    beta_base = float(parameters['beta_base'])\n    beta_range = float(parameters['beta_range'])\n    conflict_damp = float(parameters['conflict_damp'])\n    c_asym = float(parameters['c_asym'])\n    c_distinct = float(parameters['c_distinct'])\n    c_solitary = float(parameters['c_solitary'])\n    beta_min = float(parameters['beta_min'])\n    beta_max = float(parameters['beta_max'])\n\n    high_frac_for_gate = float(np.mean(high_mask))\n    low_frac = n_low / float(n_feat)\n    asym = float(np.clip(high_frac_for_gate - low_frac, 0.0, 1.0))\n    distinct_conf = float(np.tanh(dmax / D_half))\n    solitary = 1.0 if n_low == 1 else 0.0\n    asym_term = asym * (1.0 if n_low > 1 else 0.0)\n\n    conflict_damp_eff = (\n        conflict_damp\n        + c_asym * asym_term\n        + c_distinct * distinct_conf\n        + c_solitary * solitary\n    )\n    conflict_damp_eff = float(np.clip(conflict_damp_eff, 0.0, 0.95))\n\n    drive = concentration * low_share * distinct_factor * spread\n    beta = beta_base + beta_range * drive * (1.0 - conflict_damp_eff * conflict)\n    beta = float(np.clip(beta, beta_min, beta_max))\n\n    epsilon = float(parameters['epsilon'])\n    logits = np.array([evidence, 0.0], dtype=float)\n    z = beta * (logits - float(np.max(logits)))\n    e = np.exp(z)\n    core = e / np.sum(e)\n\n    n_opts = core.shape[0]\n    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts, dtype=float) / float(n_opts)\n",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=float)\n    probs = probs / probs.sum()\n    return int(np.random.choice(probs.shape[0], p=probs))\n",
  "parameters": {
    "c": "[0.007, 0.016]",
    "gamma": "[1.15, 1.55]",
    "w0": "[0.05, 0.13]",
    "kappa": "[2.0, 3.2]",
    "lam": "{0.6}",
    "distinct_scale": "[1.2, 2.6]",
    "eta": "{1.0}",
    "b_scale": "[0.25, 0.55]",
    "b_pow": "{1.0}",
    "b_base": "[0.01, 0.09]",
    "b_rank": "[0.08, 0.22]",
    "b_alpha": "[0.9, 1.3]",
    "b_cap": "[0.12, 0.33]",
    "d_gate": "[7.0, 16.0]",
    "H_thresh": "[0.75, 0.90]",
    "s_H": "[0.04, 0.12]",
    "D_half": "[0.8, 2.2]",
    "hom_count_thresh": "{3}",
    "hom_dmax_thresh": "[0.25, 0.60]",
    "hom_floor": "[0.82, 0.95]",
    "e_half": "[0.08, 0.35]",
    "beta_base": "[0.30, 0.60]",
    "beta_range": "[3.2, 5.6]",
    "conflict_damp": "[0.26, 0.50]",
    "c_asym": "[0.18, 0.42]",
    "c_distinct": "[0.06, 0.14]",
    "c_solitary": "[0.03, 0.10]",
    "beta_min": "[0.08, 0.28]",
    "beta_max": "[4.2, 5.8]",
    "epsilon": "[0.08, 0.22]",
    "validities": "validities"
  },
  "rationale": "This is a minimal edit of the accepted distinctiveness-gated low-cue dominance model. The key change addresses the prior uniform softening: distinctiveness is no longer allowed to halve the stochasticity drive for homogeneous multi-cue low blocs. When at least three below-median cues are nearly identical, the drive factor is floored near 0.9 instead of falling to 0.5, restoring the strong homogeneous low-bloc behavior needed in Experiment 2 while leaving the two-cue high-bloc cases in Experiment 5 on the original path. Conflict damping is made condition-specific rather than global: the base damping term is lowered, but additive boosts now apply when the high-validity bloc outnumbers multiple low cues, when low-cue distinctiveness is large, or when a low cue is solitary. This keeps high-cue sweeps and solitary-outlier conflicts from re-overshooting in Experiments 5, 6, 11, and 12 while allowing the Experiment 2 boost. The high-validity rank baseline is left broadly intact, and several parameter ranges are widened to restore realistic between-subject variance."
}
```

## Usage

```json
{
  "input_tokens": 31099,
  "output_tokens": 28613,
  "reasoning_tokens": 25557,
  "provider": "Baidu"
}
```
