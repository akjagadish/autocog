# generation_iter_08_attempt_00

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
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 1 (= `pi_8`).

Replace pi_8 with a new theory, e.g. 'Gated validity-gap compensation' (GVC). Keep the successful low-cue core from pi_9: below-median cues receive median/MAD deviance-anchored weight, with lone-diagnostic-cue amplification and count-dependent attenuation for multiple simultaneously diagnostic low cues. Change the high-cue side qualitatively: drop the rank-based high-cue baseline entirely. Instead give at/above-median cues only a small baseline plus a bonus proportional to a validity-gap score, e.g. (validity - median)/scale, and gate that bonus by a low-cue configuration factor. The gate should be near zero when a lone below-median cue is diagnostic (preserving Exp1 and Exp14 lone-low following), small when low cues have very large deviance (preserving Exp2, 7, 10, 11), and larger when below-median cues form a homogeneous bloc of three or more (allowing the high-cue influence seen in Exp4, 6, and 13). Conflict between low and high evidence should further damp the high-cue bonus. Finally, use a fixed-temperature softmax with no lapse floor, since Exp15 estimates lapse at 0.0. This differs from pi_8 by removing the rank-based high-cue baseline and the entropy/conflict temperature machinery, and differs from pi_9 by replacing its rank-ordered high-cue bonus with a validity-gap bonus gated by low-bloc homogeneity/deviance, which should also reduce the over-predicted high-cue contrast in Exp8 while retaining high-cue effects where validity gaps are large.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_9` (overall score: 0.839)

**Description**
Deviance-anchored, state-gated high-cue bonus with diagnostic low-cue attenuation (DASH). People anchor each advertised expert validity to the current ensemble median. Below-median cues earn weight from a median/MAD-normalized deviance score, but when two or more below-median cues are diagnostic on a trial their weights are attenuated by a count-dependent factor, preventing co-equal low-validity blocs from dominating choice. A solitary diagnostic below-median cue instead receives an amplified weight, capturing strong lone-cue following. Above-median cues receive only a tiny at-median baseline plus a rank-ordered bonus that is gated by current signed low-cue evidence and low-versus-high conflict. Choice is a fixed-temperature softmax with no lapse floor.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('DASH expects a (2, n_features) stimulus.')

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
    above = validities > median + 1e-12
    at_median = ~below & ~above

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lambda'])
    sigma = float(parameters['sigma'])
    w_hi0 = float(parameters['w_hi0'])

    raw = np.zeros(n_feat, dtype=float)
    dev_bounded = np.minimum(dev, 100.0)
    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))
    if n_active >= 2:
        phi = float(parameters['phi'])
        att = 1.0 / (1.0 + phi * (n_active - 1))
        raw[below] = raw[below] * att
    elif n_active == 1:
        i = int(np.flatnonzero(active_below)[0])
        amp = 1.0 + lam * np.tanh(float(dev[i]) / sigma)
        raw[i] = raw[i] * amp

    raw[at_median] = w_hi0

    if np.any(above):
        above_idx = np.flatnonzero(above)
        vh = validities[above_idx]
        ranks = rankdata(vh, method='average')
        if vh.size > 1:
            u = (ranks - 1.0) / (vh.size - 1.0)
        else:
            u = np.ones(1, dtype=float)
        rho = float(parameters['rho'])
        alpha = float(parameters['alpha'])
        bonus_pre = rho * np.power(np.clip(u, 0.0, 1.0), alpha)
        raw_pre_high = w_hi0 + bonus_pre
    else:
        raw_pre_high = np.empty(0, dtype=float)

    raw_pre = raw.copy()
    if np.any(above):
        raw_pre[above] = raw_pre_high

    low_mask = below
    high_mask = ~below
    E_low = float(np.dot(raw_pre[low_mask], diff[low_mask])) if np.any(low_mask) else 0.0
    E_high_pre = float(np.dot(raw_pre[high_mask], diff[high_mask])) if np.any(high_mask) else 0.0

    if E_low * E_high_pre < 0.0:
        conflict = 2.0 * min(abs(E_low), abs(E_high_pre)) / (abs(E_low) + abs(E_high_pre) + 1e-12)
    else:
        conflict = 0.0

    a = float(parameters['a'])
    K = float(parameters['K'])
    low_gate = abs(E_low) / (abs(E_low) + a)
    conflict_gate = 1.0 / (1.0 + (conflict / K) ** 2)
    gate = low_gate * conflict_gate

    if np.any(above):
        above_idx = np.flatnonzero(above)
        vh = validities[above_idx]
        ranks = rankdata(vh, method='average')
        if vh.size > 1:
            u = (ranks - 1.0) / (vh.size - 1.0)
        else:
            u = np.ones(1, dtype=float)
        rho = float(parameters['rho'])
        alpha = float(parameters['alpha'])
        bonus = rho * np.power(np.clip(u, 0.0, 1.0), alpha) * gate
        raw[above] = w_hi0 + bonus

    total = float(np.sum(raw))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw / total

    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters['beta'])
    logits = np.array([evidence_A, 0.0], dtype=float)
    z = beta * (logits - float(np.max(logits)))
    e = np.exp(z)
    probs = e / np.sum(e)
    return probs


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- c: [0.008, 0.013]
- w0: [0.006, 0.015]
- kappa: [1.8, 2.4]
- gamma: [1.2, 1.5]
- lambda: [2.2, 2.8]
- sigma: [0.20, 0.30]
- phi: [0.25, 0.50]
- w_hi0: [0.0008, 0.0015]
- rho: [0.018, 0.028]
- alpha: [2.5, 3.5]
- a: [0.22, 0.38]
- K: [0.35, 0.55]
- beta: [1.8, 2.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1230 (var=0.0002)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.7776 (var=0.0100)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=-0.4851 (var=0.0096)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.7383 (var=0.0103)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.8846 (var=0.0017)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.4829 (var=0.0072)
- Experiment 7: real=-0.0103 (var=0.0009) vs this=0.0194 (var=0.0024)
- Experiment 8: real=0.0472 (var=0.0067) vs this=0.0032 (var=0.0066)
- Experiment 9: real=0.0032 (var=0.0007) vs this=0.0048 (var=0.0012)
- Experiment 10: real=0.2001 (var=0.0006) vs this=0.1985 (var=0.0003)
- Experiment 11: real=0.4833 (var=0.0144) vs this=0.4800 (var=0.0077)
- Experiment 12: real=0.8375 (var=0.0133) vs this=0.7462 (var=0.0067)
- Experiment 13: real=0.0000 (var=0.0000) vs this=0.0530 (var=0.0135)
- Experiment 14: real=0.0278 (var=0.0063) vs this=0.0706 (var=0.0146)
- Experiment 15: real=0.7742 (var=0.0352) vs this=0.7493 (var=0.0087)
- Experiment 16: real=0.8392 (var=0.0101) vs this=0.8350 (var=0.0017)


---

### `pi_6` (overall score: 0.729)

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
- Experiment 13: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0233)
- Experiment 14: real=0.0278 (var=0.0063) vs this=0.3933 (var=0.0108)
- Experiment 15: real=0.7742 (var=0.0352) vs this=0.4036 (var=0.0073)
- Experiment 16: real=0.8392 (var=0.0101) vs this=0.7431 (var=0.0023)


---

### `pi_8` (overall score: 0.640)

**Description**
People anchor each advertised expert validity to the current ensemble, so cues below the ensemble median earn decision weight through a median/MAD-normalized deviance score that is multiplicatively amplified by cue distinctiveness. A homogeneous low-validity bloc shares weight evenly, while a single extreme low-validity cue becomes highly salient and dominates the signed evidence. A small, strictly bounded rank-based baseline keeps high-validity cues from being ignored, with strength that increases with high-bloc size and decreases when low cues are highly distinctive. Choice stochasticity is governed by a conditional entropy/conflict gate: homogeneous multi-cue low blocs receive a high temperature drive floor, whereas conflict damping is strengthened selectively when the high-validity bloc outnumbers multiple low cues or when a low cue is solitary and highly distinctive. This preserves strong homogeneous low-bloc choices and extreme-outlier following without allowing high-cue sweeps or high-distinctness conflicts to over-saturate.

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

    high_mask = ~below
    high_raw = np.zeros(n_feat, dtype=float)
    if high_mask.any():
        if below.any():
            dmax_base = float(np.max(distinct))
        else:
            dmax_base = 0.0

        d_gate = float(parameters['d_gate'])
        baseline_factor = 1.0 / (1.0 + (dmax_base / d_gate) ** 2)

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

    n_low = int(low_idx.size)
    D_half = float(parameters['D_half'])
    base_distinct_factor = 0.5 + 0.5 * np.tanh(dmax / D_half)

    hom_count_thresh = float(parameters['hom_count_thresh'])
    hom_dmax_thresh = float(parameters['hom_dmax_thresh'])
    hom_floor = float(parameters['hom_floor'])
    homogeneous_low_bloc = (n_low >= hom_count_thresh) and (dmax < hom_dmax_thresh)
    if homogeneous_low_bloc:
        distinct_factor = hom_floor + (1.0 - hom_floor) * np.tanh(dmax / D_half)
    else:
        distinct_factor = base_distinct_factor

    low_share = abs(low_evidence) / (abs(low_evidence) + abs(high_evidence) + 1e-12)

    e_half = float(parameters['e_half'])
    spread = abs(evidence) / (abs(evidence) + e_half)

    beta_base = float(parameters['beta_base'])
    beta_range = float(parameters['beta_range'])
    conflict_damp = float(parameters['conflict_damp'])
    c_asym = float(parameters['c_asym'])
    c_distinct = float(parameters['c_distinct'])
    c_solitary = float(parameters['c_solitary'])
    beta_min = float(parameters['beta_min'])
    beta_max = float(parameters['beta_max'])

    high_frac_for_gate = float(np.mean(high_mask))
    low_frac = n_low / float(n_feat)
    asym = float(np.clip(high_frac_for_gate - low_frac, 0.0, 1.0))
    distinct_conf = float(np.tanh(dmax / D_half))
    solitary = 1.0 if n_low == 1 else 0.0
    asym_term = asym * (1.0 if n_low > 1 else 0.0)

    conflict_damp_eff = (
        conflict_damp
        + c_asym * asym_term
        + c_distinct * distinct_conf
        + c_solitary * solitary
    )
    conflict_damp_eff = float(np.clip(conflict_damp_eff, 0.0, 0.95))

    drive = concentration * low_share * distinct_factor * spread
    beta = beta_base + beta_range * drive * (1.0 - conflict_damp_eff * conflict)
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
- c: [0.007, 0.016]
- gamma: [1.15, 1.55]
- w0: [0.05, 0.13]
- kappa: [2.0, 3.2]
- lam: {0.6}
- distinct_scale: [1.2, 2.6]
- eta: {1.0}
- b_scale: [0.25, 0.55]
- b_pow: {1.0}
- b_base: [0.01, 0.09]
- b_rank: [0.08, 0.22]
- b_alpha: [0.9, 1.3]
- b_cap: [0.12, 0.33]
- d_gate: [7.0, 16.0]
- H_thresh: [0.75, 0.90]
- s_H: [0.04, 0.12]
- D_half: [0.8, 2.2]
- hom_count_thresh: {3}
- hom_dmax_thresh: [0.25, 0.60]
- hom_floor: [0.82, 0.95]
- e_half: [0.08, 0.35]
- beta_base: [0.30, 0.60]
- beta_range: [3.2, 5.6]
- conflict_damp: [0.26, 0.50]
- c_asym: [0.18, 0.42]
- c_distinct: [0.06, 0.14]
- c_solitary: [0.03, 0.10]
- beta_min: [0.08, 0.28]
- beta_max: [4.2, 5.8]
- epsilon: [0.08, 0.22]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1140 (var=0.0002)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.7515 (var=0.0116)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=-0.4045 (var=0.0105)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.7300 (var=0.0175)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.8754 (var=0.0037)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.4717 (var=0.0121)
- Experiment 7: real=-0.0103 (var=0.0009) vs this=0.0207 (var=0.0025)
- Experiment 8: real=0.0472 (var=0.0067) vs this=-0.0014 (var=0.0073)
- Experiment 9: real=0.0032 (var=0.0007) vs this=-0.0043 (var=0.0007)
- Experiment 10: real=0.2001 (var=0.0006) vs this=0.2034 (var=0.0002)
- Experiment 11: real=0.4833 (var=0.0144) vs this=0.5033 (var=0.0080)
- Experiment 12: real=0.8375 (var=0.0133) vs this=0.8650 (var=0.0050)
- Experiment 13: real=0.0000 (var=0.0000) vs this=0.0342 (var=0.0105)
- Experiment 14: real=0.0278 (var=0.0063) vs this=0.0667 (var=0.0215)
- Experiment 15: real=0.7742 (var=0.0352) vs this=0.1440 (var=0.0168)
- Experiment 16: real=0.8392 (var=0.0101) vs this=0.5506 (var=0.0025)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1932 -> ACCEPTED
- iter 2: loss=0.1021 -> ACCEPTED
- iter 3: loss=0.0786 -> ACCEPTED
- iter 4: loss=0.0631 -> ACCEPTED
- iter 5: loss=0.0580 -> ACCEPTED
- iter 6: loss=0.0632 -> REJECTED
- iter 7: loss=0.0678 -> REJECTED
- iter 8: loss=0.0625 -> REJECTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.0580 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.1206 (var=0.0003) (Δ vs real -0.0247)
  - iter 2: -0.1176 (var=0.0003) (Δ vs real -0.0217)
  - iter 3: -0.1195 (var=0.0002) (Δ vs real -0.0236)
  - iter 4: -0.1002 (var=0.0002) (Δ vs real -0.0043)
  - iter 5: -0.0981 (var=0.0002) (Δ vs real -0.0022)
  - iter 6: -0.1001 (var=0.0004) (Δ vs real -0.0041)
  - iter 7: -0.0996 (var=0.0004) (Δ vs real -0.0037)
  - iter 8 (most recent): -0.1123 (var=0.0003) (Δ vs real -0.0164)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0006 (var=0.0002)
- pi_2: -0.1360 (var=0.0009)
- pi_3: -0.1493 (var=0.0006)
- pi_4: -0.1505 (var=0.0003)
- pi_5: -0.1025 (var=0.0003)
- pi_6: -0.1359 (var=0.0002)
- pi_7: -0.1246 (var=0.0002)
- pi_8: -0.1140 (var=0.0002)
- pi_9: -0.1230 (var=0.0002)

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
  - iter 1: -0.6811 (var=0.0146) (Δ vs real +0.1781)
  - iter 2: -0.6896 (var=0.0150) (Δ vs real +0.1696)
  - iter 3: -0.7440 (var=0.0117) (Δ vs real +0.1151)
  - iter 4: -0.6431 (var=0.0136) (Δ vs real +0.2161)
  - iter 5: -0.6440 (var=0.0112) (Δ vs real +0.2151)
  - iter 6: -0.6239 (var=0.0101) (Δ vs real +0.2353)
  - iter 7: -0.6271 (var=0.0105) (Δ vs real +0.2321)
  - iter 8 (most recent): -0.7064 (var=0.0133) (Δ vs real +0.1528)
**Other theories' values on this metric (for reference):**
- pi_2: -0.7597 (var=0.0423)
- pi_1: 0.7279 (var=0.0420)
- pi_3: -0.6946 (var=0.0665)
- pi_4: -0.8321 (var=0.0089)
- pi_5: -0.7328 (var=0.0080)
- pi_6: -0.8021 (var=0.0109)
- pi_7: -0.7272 (var=0.0075)
- pi_8: -0.7515 (var=0.0116)
- pi_9: -0.7776 (var=0.0100)

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
  - iter 1: -0.5111 (var=0.0040) (Δ vs real -0.1626)
  - iter 2: -0.4977 (var=0.0067) (Δ vs real -0.1492)
  - iter 3: -0.5365 (var=0.0079) (Δ vs real -0.1880)
  - iter 4: -0.4294 (var=0.0083) (Δ vs real -0.0809)
  - iter 5: -0.4469 (var=0.0064) (Δ vs real -0.0984)
  - iter 6: -0.4320 (var=0.0113) (Δ vs real -0.0835)
  - iter 7: -0.4318 (var=0.0069) (Δ vs real -0.0833)
  - iter 8 (most recent): -0.4943 (var=0.0057) (Δ vs real -0.1458)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5096 (var=0.0191)
- pi_2: 0.0215 (var=0.0080)
- pi_1: 0.5469 (var=0.0262)
- pi_4: -0.2535 (var=0.0057)
- pi_5: -0.3414 (var=0.0125)
- pi_6: -0.3640 (var=0.0059)
- pi_7: -0.2726 (var=0.0068)
- pi_8: -0.4045 (var=0.0105)
- pi_9: -0.4851 (var=0.0096)

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
  - iter 1: -0.7900 (var=0.0167) (Δ vs real -0.1367)
  - iter 2: -0.7875 (var=0.0135) (Δ vs real -0.1342)
  - iter 3: -0.7783 (var=0.0114) (Δ vs real -0.1250)
  - iter 4: -0.6408 (var=0.0224) (Δ vs real +0.0125)
  - iter 5: -0.6775 (var=0.0151) (Δ vs real -0.0242)
  - iter 6: -0.6583 (var=0.0148) (Δ vs real -0.0050)
  - iter 7: -0.6433 (var=0.0150) (Δ vs real +0.0100)
  - iter 8 (most recent): -0.6975 (var=0.0241) (Δ vs real -0.0442)
**Other theories' values on this metric (for reference):**
- pi_2: -0.1783 (var=0.0317)
- pi_3: 0.8683 (var=0.0156)
- pi_1: 0.6692 (var=0.0641)
- pi_4: -0.7908 (var=0.0142)
- pi_5: -0.6958 (var=0.0145)
- pi_6: -0.7150 (var=0.0209)
- pi_7: -0.6675 (var=0.0201)
- pi_8: -0.7300 (var=0.0175)
- pi_9: -0.7383 (var=0.0103)

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
  - iter 1: 0.8971 (var=0.0022) (Δ vs real +0.0696)
  - iter 2: 0.8971 (var=0.0020) (Δ vs real +0.0696)
  - iter 3: 0.8992 (var=0.0026) (Δ vs real +0.0717)
  - iter 4: 0.8287 (var=0.0026) (Δ vs real +0.0012)
  - iter 5: 0.8246 (var=0.0022) (Δ vs real -0.0029)
  - iter 6: 0.8213 (var=0.0038) (Δ vs real -0.0062)
  - iter 7: 0.8246 (var=0.0029) (Δ vs real -0.0029)
  - iter 8 (most recent): 0.8675 (var=0.0030) (Δ vs real +0.0400)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8642 (var=0.0036)
- pi_2: 0.1492 (var=0.0089)
- pi_1: 0.1346 (var=0.0111)
- pi_3: 0.0708 (var=0.0019)
- pi_5: 0.8250 (var=0.0030)
- pi_6: 0.7621 (var=0.0044)
- pi_7: 0.8321 (var=0.0036)
- pi_8: 0.8754 (var=0.0037)
- pi_9: 0.8846 (var=0.0017)

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
  - iter 1: 0.4571 (var=0.0081) (Δ vs real +0.3046)
  - iter 2: 0.4629 (var=0.0084) (Δ vs real +0.3104)
  - iter 3: 0.4821 (var=0.0091) (Δ vs real +0.3296)
  - iter 4: 0.3821 (var=0.0104) (Δ vs real +0.2296)
  - iter 5: 0.3733 (var=0.0107) (Δ vs real +0.2208)
  - iter 6: 0.3738 (var=0.0102) (Δ vs real +0.2213)
  - iter 7: 0.3833 (var=0.0089) (Δ vs real +0.2308)
  - iter 8 (most recent): 0.4213 (var=0.0091) (Δ vs real +0.2687)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0092 (var=0.0068)
- pi_4: 0.6663 (var=0.0054)
- pi_1: -0.4425 (var=0.0114)
- pi_3: -0.6067 (var=0.0096)
- pi_5: 0.3704 (var=0.0099)
- pi_6: 0.3529 (var=0.0067)
- pi_7: 0.4271 (var=0.0084)
- pi_8: 0.4717 (var=0.0121)
- pi_9: 0.4829 (var=0.0072)

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
  - iter 1: 0.0024 (var=0.0015) (Δ vs real +0.0126)
  - iter 2: 0.0216 (var=0.0015) (Δ vs real +0.0319)
  - iter 3: 0.0164 (var=0.0020) (Δ vs real +0.0266)
  - iter 4: 0.0171 (var=0.0026) (Δ vs real +0.0273)
  - iter 5: -0.0045 (var=0.0025) (Δ vs real +0.0058)
  - iter 6: 0.0164 (var=0.0030) (Δ vs real +0.0267)
  - iter 7: -0.0015 (var=0.0027) (Δ vs real +0.0088)
  - iter 8 (most recent): 0.0101 (var=0.0019) (Δ vs real +0.0204)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2803 (var=0.0012)
- pi_5: 0.0068 (var=0.0020)
- pi_1: 0.1888 (var=0.0043)
- pi_2: 0.2909 (var=0.0059)
- pi_3: 0.3502 (var=0.0022)
- pi_6: 0.0282 (var=0.0018)
- pi_7: 0.0224 (var=0.0016)
- pi_8: 0.0207 (var=0.0025)
- pi_9: 0.0194 (var=0.0024)

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
  - iter 1: 0.0056 (var=0.0112) (Δ vs real -0.0416)
  - iter 2: 0.0109 (var=0.0090) (Δ vs real -0.0362)
  - iter 3: 0.0042 (var=0.0105) (Δ vs real -0.0430)
  - iter 4: -0.0017 (var=0.0082) (Δ vs real -0.0489)
  - iter 5: 0.0021 (var=0.0080) (Δ vs real -0.0451)
  - iter 6: 0.0174 (var=0.0091) (Δ vs real -0.0298)
  - iter 7: 0.0069 (var=0.0067) (Δ vs real -0.0403)
  - iter 8 (most recent): 0.0082 (var=0.0096) (Δ vs real -0.0389)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0010 (var=0.0097)
- pi_4: 0.6671 (var=0.0037)
- pi_1: 0.3105 (var=0.0223)
- pi_2: 0.6061 (var=0.0242)
- pi_3: 0.7504 (var=0.0068)
- pi_6: 0.0794 (var=0.0104)
- pi_7: 0.0680 (var=0.0084)
- pi_8: -0.0014 (var=0.0073)
- pi_9: 0.0032 (var=0.0066)

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
  - iter 1: -0.0012 (var=0.0005) (Δ vs real -0.0044)
  - iter 2: 0.0036 (var=0.0008) (Δ vs real +0.0004)
  - iter 3: 0.0032 (var=0.0007) (Δ vs real +0.0001)
  - iter 4: 0.0104 (var=0.0009) (Δ vs real +0.0072)
  - iter 5: -0.0004 (var=0.0007) (Δ vs real -0.0036)
  - iter 6: 0.0012 (var=0.0007) (Δ vs real -0.0020)
  - iter 7: 0.0018 (var=0.0008) (Δ vs real -0.0014)
  - iter 8 (most recent): 0.0033 (var=0.0009) (Δ vs real +0.0001)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0500 (var=0.0010)
- pi_5: 0.0003 (var=0.0008)
- pi_1: 0.1056 (var=0.0010)
- pi_2: 0.1807 (var=0.0017)
- pi_3: 0.2205 (var=0.0005)
- pi_4: 0.1640 (var=0.0005)
- pi_7: 0.0023 (var=0.0005)
- pi_8: -0.0043 (var=0.0007)
- pi_9: 0.0048 (var=0.0012)

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
  - iter 1: 0.1986 (var=0.0002) (Δ vs real -0.0015)
  - iter 2: 0.1942 (var=0.0003) (Δ vs real -0.0059)
  - iter 3: 0.1969 (var=0.0003) (Δ vs real -0.0032)
  - iter 4: 0.1857 (var=0.0003) (Δ vs real -0.0145)
  - iter 5: 0.1865 (var=0.0003) (Δ vs real -0.0136)
  - iter 6: 0.1814 (var=0.0003) (Δ vs real -0.0187)
  - iter 7: 0.1817 (var=0.0005) (Δ vs real -0.0184)
  - iter 8 (most recent): 0.1914 (var=0.0003) (Δ vs real -0.0087)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1290 (var=0.0004)
- pi_6: 0.1905 (var=0.0004)
- pi_1: 0.1377 (var=0.0003)
- pi_2: 0.2228 (var=0.0005)
- pi_3: 0.2121 (var=0.0003)
- pi_4: 0.2385 (var=0.0002)
- pi_7: 0.2038 (var=0.0002)
- pi_8: 0.2034 (var=0.0002)
- pi_9: 0.1985 (var=0.0003)

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
  - iter 1: 0.4933 (var=0.0072) (Δ vs real +0.0100)
  - iter 2: 0.5050 (var=0.0071) (Δ vs real +0.0217)
  - iter 3: 0.4950 (var=0.0076) (Δ vs real +0.0117)
  - iter 4: 0.5017 (var=0.0115) (Δ vs real +0.0183)
  - iter 5: 0.5000 (var=0.0111) (Δ vs real +0.0167)
  - iter 6: 0.5067 (var=0.0108) (Δ vs real +0.0233)
  - iter 7: 0.5300 (var=0.0110) (Δ vs real +0.0467)
  - iter 8 (most recent): 0.5033 (var=0.0122) (Δ vs real +0.0200)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7900 (var=0.0109)
- pi_5: 0.4950 (var=0.0193)
- pi_1: 0.8517 (var=0.0151)
- pi_2: 0.8650 (var=0.0147)
- pi_3: 0.9367 (var=0.0079)
- pi_4: 0.9033 (var=0.0065)
- pi_6: 0.6117 (var=0.0132)
- pi_8: 0.5033 (var=0.0080)
- pi_9: 0.4800 (var=0.0077)

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
  - iter 1: 0.7631 (var=0.0057) (Δ vs real -0.0744)
  - iter 2: 0.7500 (var=0.0071) (Δ vs real -0.0875)
  - iter 3: 0.7412 (var=0.0066) (Δ vs real -0.0963)
  - iter 4: 0.8169 (var=0.0047) (Δ vs real -0.0206)
  - iter 5: 0.8287 (var=0.0056) (Δ vs real -0.0088)
  - iter 6: 0.8306 (var=0.0046) (Δ vs real -0.0069)
  - iter 7: 0.8287 (var=0.0045) (Δ vs real -0.0088)
  - iter 8 (most recent): 0.8612 (var=0.0034) (Δ vs real +0.0237)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6706 (var=0.0116)
- pi_7: 0.3594 (var=0.0070)
- pi_1: 0.3181 (var=0.0072)
- pi_2: 0.1288 (var=0.0070)
- pi_3: 0.0569 (var=0.0024)
- pi_4: 0.1156 (var=0.0048)
- pi_6: 0.5475 (var=0.0085)
- pi_8: 0.8650 (var=0.0050)
- pi_9: 0.7462 (var=0.0067)

### Experiment 13
**Design**
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a_low = data['option_a_ratings'].apply(lambda v: int(v[0]) + int(v[1]) + int(v[2]))
    b_low = data['option_b_ratings'].apply(lambda v: int(v[0]) + int(v[1]) + int(v[2]))
    d = (a_low - b_low).to_numpy(dtype=float)
    resp = data['response'].to_numpy(dtype=float)

    follow = np.where(d > 0, 1.0 - resp, resp)
    mask = np.abs(d) >= 1.0
    d = d[mask]
    resp = resp[mask]
    follow = follow[mask]

    unan = np.abs(d) == 3.0
    split = np.abs(d) == 1.0
    n_u = int(unan.sum())
    n_s = int(split.sum())
    if n_u < 5 or n_s < 5:
        return 0.0

    p_u = float(np.clip(follow[unan].mean(), 0.001, 0.999))
    p_s = float(np.clip(follow[split].mean(), 0.001, 0.999))

    def logit(p):
        p = float(p)
        return float(np.log(p / (1.0 - p)))

    emax = min(0.45, 2.0 * (1.0 - p_u) - 1e-3, 2.0 * (1.0 - p_s) - 1e-3)
    if emax <= 0.0:
        return 0.0

    best_e = 0.0
    best_loss = 1e12
    for j in range(401):
        e = emax * float(j) / 400.0
        denom = 1.0 - e
        if denom <= 0.0:
            continue
        q_u = (p_u - 0.5 * e) / denom
        q_s = (p_s - 0.5 * e) / denom
        if q_u <= 0.0 or q_u >= 1.0 or q_s <= 0.0 or q_s >= 1.0:
            continue
        loss = abs(logit(q_u) - 3.0 * logit(q_s))
        if loss < best_loss:
            best_loss = loss
            best_e = e
    return float(best_e)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.0537 (var=0.0137) (Δ vs real +0.0537)
  - iter 2: 0.0000 (var=0.0100) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0102) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0233) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0306) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0214) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.0208) (Δ vs real +0.0000)
  - iter 8 (most recent): 0.0000 (var=0.0161) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0342 (var=0.0105)
- pi_5: 0.1512 (var=0.0183)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0103)
- pi_3: 0.0000 (var=0.0039)
- pi_4: 0.0080 (var=0.0088)
- pi_6: 0.0000 (var=0.0233)
- pi_7: 0.0585 (var=0.0121)
- pi_9: 0.0530 (var=0.0135)

### Experiment 14
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np

    ratings = np.array(data['option_a_ratings'].tolist(), dtype=float)
    responses = np.asarray(data['response'].tolist(), dtype=int)
    subjects = np.asarray(data['subject_id'].tolist())
    choose_a = (responses == 0).astype(float)

    rel_idx = np.arange(1, 16)
    tallies = ratings[:, rel_idx].sum(axis=1)

    low = tallies <= 4
    high = tallies >= 8

    vals = []
    for sid in np.unique(subjects):
        idx = subjects == sid
        low_idx = idx & low
        high_idx = idx & high
        if np.count_nonzero(low_idx) == 0 or np.count_nonzero(high_idx) == 0:
            continue
        vals.append(float(np.mean(choose_a[high_idx]) - np.mean(choose_a[low_idx])))

    if not vals:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.0278 (var=0.0063)
**Previous candidate values (this loop):**
  - iter 1: 0.4189 (var=0.0089) (Δ vs real +0.3911)
  - iter 2: 0.1967 (var=0.0265) (Δ vs real +0.1689)
  - iter 3: 0.0117 (var=0.0112) (Δ vs real -0.0161)
  - iter 4: 0.0100 (var=0.0120) (Δ vs real -0.0178)
  - iter 5: 0.0122 (var=0.0133) (Δ vs real -0.0156)
  - iter 6: 0.0217 (var=0.0109) (Δ vs real -0.0061)
  - iter 7: 0.0006 (var=0.0141) (Δ vs real -0.0272)
  - iter 8 (most recent): 0.0244 (var=0.0155) (Δ vs real -0.0033)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2411 (var=0.0134)
- pi_8: 0.0667 (var=0.0215)
- pi_1: 0.3689 (var=0.0171)
- pi_2: 0.6983 (var=0.0351)
- pi_3: 0.8700 (var=0.0083)
- pi_4: 0.7717 (var=0.0078)
- pi_6: 0.3933 (var=0.0108)
- pi_7: 0.3833 (var=0.0125)
- pi_9: 0.0706 (var=0.0146)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np


def metric(data):
    a0 = data['option_a_ratings'].apply(lambda x: int(x[0]))
    b0 = data['option_b_ratings'].apply(lambda x: int(x[0]))
    low_a = (a0 == 1) & (b0 == 0)
    low_b = (a0 == 0) & (b0 == 1)
    resp = data['response'].astype(float)
    ra = resp[low_a].mean()
    rb = resp[low_b].mean()
    if np.isnan(ra) or np.isnan(rb):
        return 0.0
    return float(rb - ra)

```

**Observed (real) value:** 0.7742 (var=0.0352)
**Previous candidate values (this loop):**
  - iter 1: 0.4636 (var=0.0271) (Δ vs real -0.3107)
  - iter 2: 0.7973 (var=0.0065) (Δ vs real +0.0231)
  - iter 3: 0.7720 (var=0.0076) (Δ vs real -0.0022)
  - iter 4: 0.6818 (var=0.0061) (Δ vs real -0.0924)
  - iter 5: 0.6498 (var=0.0063) (Δ vs real -0.1244)
  - iter 6: 0.6622 (var=0.0075) (Δ vs real -0.1120)
  - iter 7: 0.6422 (var=0.0082) (Δ vs real -0.1320)
  - iter 8 (most recent): 0.7276 (var=0.0058) (Δ vs real -0.0467)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1440 (var=0.0168)
- pi_9: 0.7493 (var=0.0087)
- pi_1: -0.0071 (var=0.0075)
- pi_2: 0.0831 (var=0.0046)
- pi_3: 0.0964 (var=0.0021)
- pi_4: 0.0831 (var=0.0052)
- pi_5: 0.3538 (var=0.0183)
- pi_6: 0.4036 (var=0.0073)
- pi_7: 0.2671 (var=0.0058)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np


def metric(data: pd.DataFrame) -> float:
    low_a = np.asarray([x[0] for x in data['option_a_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=float)
    follow_low = ((low_a == 1.0) & (resp == 0.0)) | ((low_a == 0.0) & (resp == 1.0))
    return float(np.mean(follow_low))
```

**Observed (real) value:** 0.8392 (var=0.0101)
**Previous candidate values (this loop):**
  - iter 1: 0.5683 (var=0.0027) (Δ vs real -0.2708)
  - iter 2: 0.7535 (var=0.0037) (Δ vs real -0.0856)
  - iter 3: 0.8796 (var=0.0019) (Δ vs real +0.0404)
  - iter 4: 0.8163 (var=0.0016) (Δ vs real -0.0229)
  - iter 5: 0.8223 (var=0.0017) (Δ vs real -0.0169)
  - iter 6: 0.8165 (var=0.0013) (Δ vs real -0.0227)
  - iter 7: 0.8133 (var=0.0017) (Δ vs real -0.0258)
  - iter 8 (most recent): 0.8544 (var=0.0012) (Δ vs real +0.0152)
**Other theories' values on this metric (for reference):**
- pi_9: 0.8350 (var=0.0017)
- pi_8: 0.5506 (var=0.0025)
- pi_1: 0.5008 (var=0.0009)
- pi_2: 0.5677 (var=0.0017)
- pi_3: 0.5329 (var=0.0011)
- pi_4: 0.6112 (var=0.0009)
- pi_5: 0.6748 (var=0.0043)
- pi_6: 0.7431 (var=0.0023)
- pi_7: 0.6675 (var=0.0024)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Conflict-gated validity-gap compensation with strict homogeneous-bloc licensing and stronger opposition-sensitive lone amplification (GVC-strict). People anchor each advertised expert validity to the ensemble median. Below-median cues earn decision weight through a median/MAD-normalized deviance score. A single diagnostically deviant low cue is strongly amplified, with amplification increasing in the number of opposed at-or-above-median cues. When multiple below-median cues are diagnostic, a homogeneous low-deviant group is attenuated by cue count, but an extreme focal low cue is exempted from that group attenuation so it can still dominate the signed evidence. At-or-above-median cues receive no default weight; their validity-gap bonus is licensed only by a very homogeneous, sufficiently deviant bloc of three or more simultaneously diagnostic below-median cues, and is sharply damped by low-versus-high conflict. Choice is a fixed-temperature softmax with no lapse floor.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('GVC expects a (2, n_features) stimulus.')

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
    at_or_above = ~below
    gap = np.maximum(0.0, validities - median) / scale

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lambda'])
    sigma = float(parameters['sigma'])
    phi = float(parameters['phi'])
    nu = float(parameters['nu'])
    opp_cap = float(parameters['opp_cap'])
    focal_ratio = float(parameters['focal_ratio'])

    raw = np.zeros(n_feat, dtype=float)
    dev_bounded = np.minimum(dev, 100.0)
    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))

    def lone_amplification(idx):
        base_amp = 1.0 + lam * np.tanh(float(dev[idx]) / sigma)
        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)
        n_opp = int(np.count_nonzero(opp))
        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))

    if n_active == 1:
        focal = int(np.flatnonzero(active_below)[0])
        raw[focal] = raw[focal] * lone_amplification(focal)
    elif n_active >= 2:
        att = 1.0 / (1.0 + phi * (n_active - 1))
        active_devs = dev[active_below]
        i_max = int(np.argmax(active_devs))
        max_dev = float(active_devs[i_max])
        med_dev = float(np.median(active_devs))
        if med_dev > 1e-12 and max_dev > focal_ratio * med_dev:
            focal = int(np.flatnonzero(active_below)[i_max])
            others = below.copy()
            others[focal] = False
            raw[others] = raw[others] * att
            raw[focal] = raw[focal] * lone_amplification(focal)
        else:
            raw[below] = raw[below] * att

    gate_base = float(parameters['gate_base'])
    lone_floor = float(parameters['lone_floor'])
    D_half = float(parameters['D_half'])
    q_dev = float(parameters['q_dev'])
    b_boost = float(parameters['b_boost'])
    D_boost = float(parameters['D_boost'])
    H_spread = float(parameters['H_spread'])
    K = float(parameters['K'])

    gate = gate_base
    if n_active == 1:
        gate *= lone_floor

    if n_active > 0:
        active_devs = dev[active_below]
        dmax = float(np.max(active_devs))
    else:
        dmax = 0.0

    if dmax > 1e-12:
        gate *= 1.0 / (1.0 + (dmax / D_half) ** q_dev)

    if n_active >= 3:
        low_vals = validities[active_below]
        spread = float(np.std(low_vals))
        spread_norm = spread / scale
        hom = 1.0 / (1.0 + (spread_norm / H_spread) ** 12.0)
        mean_dev = float(np.mean(dev[active_below]))
        act = float(np.tanh(mean_dev / D_boost))
        gate += b_boost * hom * act

    w_hi0 = float(parameters['w_hi0'])
    rho = float(parameters['rho'])
    alpha = float(parameters['alpha'])

    gap_bounded = np.minimum(gap, 100.0)
    hi_pre = np.zeros(n_feat, dtype=float)
    if np.any(at_or_above):
        hi_pre[at_or_above] = w_hi0 + rho * np.power(gap_bounded[at_or_above], alpha)

    E_low = float(np.dot(raw, diff))
    E_high_pre = float(np.dot(hi_pre, diff))

    if E_low * E_high_pre < 0.0:
        conflict = 2.0 * min(abs(E_low), abs(E_high_pre)) / (abs(E_low) + abs(E_high_pre) + 1e-12)
        gate *= 1.0 / (1.0 + (conflict / K) ** 2.0)

    gate = float(np.maximum(gate, 0.0))

    raw_high = np.zeros(n_feat, dtype=float)
    if np.any(at_or_above):
        raw_high[at_or_above] = w_hi0 + rho * np.power(gap_bounded[at_or_above], alpha) * gate

    raw_all = raw + raw_high
    total = float(np.sum(raw_all))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw_all / total

    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters['beta'])
    logits = np.array([evidence_A, 0.0], dtype=float)
    z = beta * (logits - float(np.max(logits)))
    e = np.exp(z)
    probs = e / np.sum(e)
    return probs


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- c: [0.008, 0.012]
- w0: [0.007, 0.012]
- kappa: [1.9, 2.3]
- gamma: [1.25, 1.45]
- lambda: [4.5, 5.5]
- sigma: [0.10, 0.18]
- phi: [0.55, 0.75]
- nu: [0.15, 0.25]
- opp_cap: {5}
- focal_ratio: [3.0, 4.0]
- gate_base: {0.0}
- lone_floor: [0.00, 0.03]
- D_half: [0.35, 0.55]
- q_dev: {2}
- b_boost: [0.02, 0.08]
- D_boost: [0.6, 1.2]
- H_spread: [0.005, 0.02]
- K: [0.30, 0.50]
- w_hi0: {0.0}
- rho: [0.08, 0.14]
- alpha: [1.1, 1.8]
- beta: [1.4, 1.8]
- validities: validities

`rationale`: Minimal in-family repair of the running-best GVC-focal candidate. The critic's main diagnosis was that the validity-gap bonus was still too readily licensed and insufficiently suppressed under low-versus-high conflict. Here K is moved from [0.70, 1.00] to [0.30, 0.50], sharply damping the high-cue bonus in conflict, which should fix the Exp2 direction and recover Exp15 without a global beta increase. The homogeneous low-bloc license is made stricter by reducing b_boost to [0.02, 0.08], lowering D_boost to [0.6, 1.2], narrowing H_spread to [0.005, 0.02], and raising the homogeneity exponent from 8 to 12, so Exp6-like near-median blocs no longer reopen the gate while genuinely deviant homogeneous blocs in Exp4/13 can still do so. To preserve lone-low following under stronger conflict damping, lone amplification is strengthened: lambda [4.5, 5.5], nu [0.15, 0.25], and opp_cap fixed at 5. Beta, phi, and rho are deliberately left unchanged to avoid re-breaking Exp3/4/5. The fixed-temperature softmax, no lapse floor, and hard-zero high-cue default are retained.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The GVC candidate gets many of the core low-cue patterns right. Exp2, Exp7, Exp8, Exp9, Exp10, Exp11, and Exp12 are all in the right qualitative range, with Exp10 and Exp11 excellent, and Exp7 and Exp9 near zero as observed. However the candidate fails badly on three experiments. Exp14 is the worst: observed 0.0278, candidate 0.4189. In Exp14 the low-cue configuration appears never to be diagnostic, yet GVC initializes the high-cue gate at 1.0 and leaves it fully open, so the validity-gap bonus applies without any low-cue licensing. Humans show essentially no high-cue contrast there. Exp15 and Exp16 show the opposite failure: observed lone-low following of 0.7742 and 0.8392, candidate only 0.4636 and 0.5683. The lone diagnostic low cue is too easily diluted by the many at-or-above-median cues, likely because w_hi0 remains nonzero and accumulates across many high features, and the lone_floor gate is not close enough to zero. Exp6 is also clearly over-predicted: observed 0.1525, candidate 0.4571, indicating the homogeneous below-median bloc boost reopens high-cue compensation too strongly. There are also moderate overshoots in Exp3, Exp4, and Exp5, though those are less decisive than Exp14 through Exp16.
Rationale: Keep the GVC family and fixed-temperature softmax, but repair the gate and high-cue baseline. First, the gate should not default to 1.0. Add a small gate_base parameter, roughly 0.00 to 0.10, and make the homogeneous below-median bloc boost additive rather than multiplicative on a tiny base, so that no-active-low trials like Exp14 produce almost no high-cue influence while Exp4, Exp6, and Exp13 can still retrieve their required high-cue effects. Second, for the lone-low failures in Exp15 and Exp16, set w_hi0 to zero or near 1e-5 and shrink lone_floor to roughly 0.00 to 0.03, so many at-or-above-median cues cannot collectively dilute one diagnostic low cue. A modest increase in lambda may help but should be monitored against Exp1. Third, reduce the homogeneity boost b_boost from 1.5 to 2.8 down to roughly 0.4 to 1.0, or tighten H_spread and conflict damping K, to bring Exp6 down toward the observed 0.1525. These are in-family gate and parameter repairs, not a replacement of the prescribed GVC mechanism.

**Outcome of this advice:** iter 1 candidate loss=0.1932 -> iter 2 candidate loss=0.1021 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: GVC-d is a substantial improvement over the previous iteration. Exp13 matches exactly, Exp9 is nearly exact, Exp10 and Exp11 are strong, Exp15 has been repaired to within about 0.02 of observed, and Exp8 stays near zero. The prior gate repair clearly helped Exp14, Exp15, and Exp16. However the candidate is not yet experiment-invariant. The largest remaining failures are Exp6, where the homogeneous below-median bloc still opens the high-cue gate far too strongly, producing 0.4629 against an observed 0.1525, and Exp14, where no-active-low trials still receive a residual high-cue bonus through gate_base, producing 0.1967 against 0.0278. Lone-low following is also still too weak in Exp12 and Exp16: 0.7500 and 0.7535 against observed 0.8375 and 0.8392. This suggests the nonzero gate_base, multiplied by lone_floor, leaves enough high-cue weight to dilute a single diagnostic low cue. There are also moderate overshoots in Exp5, Exp3, and Exp4, with errors of roughly 0.07 to 0.15. Exp2 remains underweighted on the high-validity side, and the previous b_boost reduction did not move Exp6, so a sharper homogeneity trigger rather than another broad b_boost shrink is needed.
Rationale: Stay within the prescribed GVC family and keep the fixed-temperature softmax with no lapse. Make targeted in-family repairs. First, set gate_base to 0 or at most 0.005; since w_hi0 is already zero, this removes the unlicensed high-cue bonus on no-active-low trials like Exp14 and should also remove residual high-cue dilution on lone-low trials in Exp12 and Exp16. If Exp2 loses directional contrast because of this, compensate by increasing rho to about 0.04 to 0.08 or raising beta, not by reopening gate_base. Second, tighten the bloc license for Exp6: lower b_boost to roughly 0.10 to 0.35 and sharpen H_spread to about 0.05 to 0.15, or increase the homogeneity exponent, so only very homogeneous three-plus below-median blocs can open the gate; recheck Exp4 and Exp13 to ensure their high-cue contributions remain. Third, strengthen count-dependent attenuation for multiple active low cues by raising phi to about 0.40 to 0.60, which should pull Exp5 from 0.897 toward 0.827, while monitoring Exp10. Fourth, if Exp12 and Exp16 remain below about 0.80 after gate_base is zeroed, raise lambda to about 2.8 to 3.2 with sigma near 0.15 to 0.25 rather than restoring w_hi0 or any high-cue baseline.

**Outcome of this advice:** iter 2 candidate loss=0.1021 -> iter 3 candidate loss=0.0786 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current GVC-z candidate is a major improvement over the previous iterations and remains faithful to the prescribed gated validity-gap compensation family. Several experiments now match almost exactly: Exp9 is essentially exact, Exp13 remains exactly 0.0, Exp14 has been repaired from 0.4189 to 0.0117 against an observed 0.0278, Exp15 is near perfect at 0.7720 against 0.7742, and Exp10/Exp11 continue to fit well. The hard-zero gate default and the removal of the high-cue baseline clearly worked. However the candidate is still not experiment-invariant. The worst remaining failure is Exp6, where the homogeneous below-median bloc gate still opens too strongly: candidate 0.4821 against observed 0.1525, an error of +0.3296. The prior b_boost and H_spread changes did not fix it, which suggests the problem is that the homogeneity boost is still licensed independently of how diagnostic the low-cue bloc actually is. Exp3 is also substantially over-predicted (-0.5365 against -0.3485), as is Exp4 (-0.7783 against -0.6533). These suggest that multiple low cues still accumulate too much choice pressure, partly because the fixed-temperature softmax is saturating under high beta. Exp5 remains too high (0.8992 against 0.8275) for the same reason. On the high-cue side, Exp2 is still under-predicted (-0.7440 against -0.8592), so licensed validity-gap compensation may be too weak. Lone-low behavior is still asymmetric: Exp12 is too weak (0.7412 against 0.8375) while Exp16 is too strong (0.8796 against 0.8392), indicating the lone amplification term lacks the contextual opposition sensitivity needed to separate these cases.
Rationale: Regenerate with targeted in-family repairs. First, fix Exp6 by making the homogeneous low-bloc license more stringent and diagnosis-sensitive. The homogeneous boost should not be a bare additive gate += b_boost * hom for any spread below H_spread. Raise the homogeneity exponent to roughly 6-10 or narrow H_spread to about 0.01-0.04, and additionally scale b_boost by a low-cue activation factor such as tanh(mean(dev[active_below]) / D_boost) with D_boost around 1.0-2.0. This keeps a strong license for genuinely diagnostic homogeneous blocs in Exp4 and Exp13, while suppressing the near-median homogeneous blocs that likely drive the Exp6 over-prediction. Second, after tightening that gate, increase the licensed high-cue response to recover Exp2: raise rho to roughly 0.08-0.14 or loosen conflict damping by raising K to about 0.6-1.0, while keeping gate_base and w_hi0 at zero. Third, reduce overall saturation in multi-low-cue configurations: lower beta to about 1.4-1.8 and slightly strengthen phi to about 0.55-0.75, which should pull Exp3, Exp4, and Exp5 toward their observed values. To avoid losing lone-low following, pair this with stronger lone amplification, with lambda around 3.4-4.2 and sigma near 0.10-0.18. Fourth, make lone amplification opposition-sensitive: multiply the lone amplification term by 1 + nu * min(n_opposing_at_or_above, cap), with nu around 0.05-0.15 and cap around 3-5. This should raise Exp12, where a lone below-median cue faces several opposing above-median cues, without further inflating Exp16, where lone-cue following is already too strong. Keep the fixed-temperature softmax with no lapse and the hard-zero high-cue default; these are parameter-level and gating repairs within the prescribed GVC family.

**Outcome of this advice:** iter 3 candidate loss=0.0786 -> iter 4 candidate loss=0.0631 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current GVC-focal candidate is the best iteration so far and remains within the prescribed gated validity-gap family. Several experiments are now near exact or close, including Exp5 (0.8287 vs 0.8275), Exp4 (-0.6408 vs -0.6533), Exp13 (exact 0.0000), Exp1, Exp9, Exp11, Exp12, Exp14, and Exp16, all within roughly 0.02-0.03 of observed. However the candidate is not yet experiment-invariant. The largest remaining failure is Exp6, where the candidate gives 0.3821 against an observed 0.1525, an over-prediction of about 0.23. Exp2 also moved in the wrong direction: the candidate gives -0.6431 against -0.8592, an error of about 0.216, after the previous iteration had -0.7440. Exp15 regressed from 0.7720 to 0.6818 against observed 0.7742, an error of -0.092. Exp3 remains somewhat too negative at -0.4294 vs -0.3485. These failures share a common diagnosis: the high-cue validity-gap bonus is still too readily licensed and insufficiently suppressed when low and high evidence conflict. The conflict damping term uses K in 0.70-1.00, which is too permissive, and the homogeneous low-bloc boost, though tightened, still reopens the high-cue gate too much in Exp6 and in conflict-heavy configurations. The low-cue side is otherwise well calibrated after the beta and phi changes.
Rationale: Keep the GVC family, fixed-temperature softmax, no lapse floor, and hard-zero high-cue default. First, tighten the conflict gate substantially by lowering K from 0.70-1.00 to an intermediate range around 0.30-0.50. This should close the validity-gap bonus under opposed low-high evidence, pushing Exp2 toward -0.86, reducing the Exp6 overshoot, and recovering much of the lost Exp15 lone-low following without a global beta increase. This partly reverses the previous relaxation of K, but the current data show that move overshot; use an intermediate value rather than returning to the old strong setting. Second, make the homogeneous low-bloc license still more stringent: lower b_boost to about 0.02-0.08, steepen the activation by lowering D_boost to about 0.6-1.2, and/or narrow H_spread to about 0.005-0.02 with an exponent of 10-12. Third, do not raise beta globally, because the current values have fixed Exp3, Exp4, and Exp5; if Exp15 and Exp10 remain slightly under-following after conflict tightening, strengthen lone-cue amplification instead by raising lambda toward 4.5-5.5 and increasing nu toward 0.15-0.25 with opp_cap 5. Recheck Exp4 and Exp13 after tightening the conflict and homogeneity gate to ensure their high-cue contributions are not lost. These are all parameter-level and gating repairs within the prescribed mechanism, not a family change.

**Outcome of this advice:** iter 4 candidate loss=0.0631 -> iter 5 candidate loss=0.0580 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The GVC-strict candidate is a further improvement and was correctly accepted, but it is not yet experiment-invariant. Many experiments are now near exact or well within noise: Exp1 (-0.0981 vs -0.0959), Exp4 (-0.6775 vs -0.6533), Exp5 (0.8246 vs 0.8275), Exp7 (-0.0045 vs -0.0103), Exp9 (-0.0004 vs 0.0032), Exp10 (0.1865 vs 0.2001), Exp11 (0.5000 vs 0.4833), Exp12 (0.8287 vs 0.8375), Exp13 exact, Exp14 (0.0122 vs 0.0278), and Exp16 (0.8223 vs 0.8392). The remaining failures are concentrated and diagnostic. Exp2 is still substantially under-weighted on the high-validity side (-0.6440 vs -0.8592), and notably the previous K tightening did not move it at all, which suggests the conflict-damping term is not the binding constraint there. Exp6 remains over-predicted (0.3733 vs 0.1525), indicating the homogeneous low-bloc license still reopens high-cue influence too easily in near-median configurations. Exp15 regressed further (0.6498 vs 0.7742), which is likely residual gate leakage on lone-low trials because lone_floor is still allowed to be nonzero. Exp3 remains moderately too negative (-0.4469 vs -0.3485).
Rationale: Keep the GVC family, the fixed-temperature softmax, no lapse floor, hard-zero high-cue default, and the accepted K and homogeneity repairs, but make three targeted in-family repairs. First, for Exp2 the homogeneous low-bloc bonus is being nearly annihilated by the global one-over-one-plus-(dmax/D_half)^2 suppression that applies even when n_active >= 3 and the below-median cues form a perfectly homogeneous bloc. Exempt a sufficiently deviant homogeneous bloc from the global dmax suppression, or apply that suppression only to lone and non-bloc low configurations, and modestly raise b_boost into roughly 0.10 to 0.25 while keeping H_spread tight and D_boost low. This should restore the licensed validity-gap compensation needed in Exp2 without broadly reopening the gate. Second, for Exp6 make the bloc license require both homogeneity and sufficiently large deviance from median, for example by gating on the minimum or median active below-median deviance rather than only the mean, so near-median homogeneous blocs fail the threshold while strict low-validity blocs in Exp2 pass. Third, set lone_floor exactly 0.0 so a single diagnostic below-median cue receives no high-cue bonus at all; the current range allows nonzero draws that dilute lone-cue weight and likely produce the Exp15 shortfall. Recheck Exp12 and Exp16 after this change, since they should improve slightly but not overshoot. Avoid global beta or rho increases because the current values are protecting Exp3 through Exp5 and Exp8. These are parameter-level and gating repairs within the prescribed GVC mechanism, not a change of family.

**Outcome of this advice:** iter 5 candidate loss=0.0580 -> iter 6 candidate loss=0.0632 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The iter 6 candidate was rejected by the accept gate (loss 0.0632 vs running best 0.0580), so the loop will revert to the accepted iter 5 base. The candidate made three main changes relative to iter 5: it exempted licensed homogeneous low blocs from the global dmax suppression, added a median-deviance threshold for bloc licensing, and forced lone_floor to exactly 0.0. These changes produced visible improvements in Exp14 (0.0122 -> 0.0217), Exp15 (0.6498 -> 0.6622), Exp8 (0.0021 -> 0.0174), Exp9 (-0.0004 -> 0.0012), Exp3 (-0.4469 -> -0.4320), and Exp4 (-0.6775 -> -0.6583). However they also produced regressions in Exp2 (-0.6440 -> -0.6239), Exp7 (-0.0045 -> 0.0164), Exp10 (0.1865 -> 0.1814), Exp11 (0.5000 -> 0.5067), Exp16 (0.8223 -> 0.8165), and Exp1 (-0.0981 -> -0.1001). Most importantly, the two largest errors under the accepted base were not meaningfully repaired: Exp2 moved in the wrong direction, and Exp6 was essentially unchanged (0.3733 -> 0.3738). The aggregate loss therefore increased. The rejected changes did not address the binding bottleneck: in multiple-active-low-cue trials, the below-median raw weights still dominate the at-or-above-median validity-gap bonus, so the high-cue side remains too weak in Exp2 and the low-cue side remains too strong in Exp6. The lone_floor and bloc-license modifications were secondary to this low-cue weight imbalance.
Rationale: Regenerate with the accepted iter 5 base, but do not repeat the rejected iter 6 changes. The new direction should be a low-cue weight recalibration rather than further gate/license surgery. First, reduce kappa from the current 1.9-2.3 to roughly 1.0-1.4 and/or increase gamma from 1.25-1.45 to 1.55-1.80. This lowers the decision weight of below-median cues whose median/MAD deviance score is below 1, which is exactly the regime in Exp2 (low validity 0.6 vs median 0.65) and likely in Exp6. Because the weights are normalized, this change leaves lone-low and zero-high-gate trials essentially unchanged, so it should not damage Exp15, Exp16, Exp12, Exp14, or Exp5. Reducing the low-cue dominance should pull Exp2 toward the observed -0.8592, pull Exp6 down from 0.3733 toward 0.1525, and also bring Exp3 closer to -0.3485. Second, keep gate_base=0, w_hi0=0, the fixed-temperature softmax, and no lapse floor. If Exp2 remains too weak after the kappa/gamma change, compensate with a modest increase in rho (0.14-0.22) or b_boost, but only after verifying Exp6 does not reopen. Third, for the remaining Exp15 shortfall, do not continue to rely on lone_floor=0.0. Instead, lower focal_ratio from 3.0-4.0 to 2.0-2.5 and/or increase lambda to 5.5-6.5 with nu around 0.25-0.35, so the most deviant low cue is amplified even when other below-median cues oppose it. Monitor Exp16 to ensure lone-low following does not exceed 0.85. These are all parameter-level and gating repairs within the prescribed GVC family.

**Outcome of this advice:** iter 6 candidate loss=0.0632 -> iter 7 candidate loss=0.0678 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The current iter 7 candidate was rejected by the accept gate (loss 0.0678 vs accepted iter 5 loss 0.0580). Its main regressions are concentrated in exactly the experiments that were already the accepted base's weakest points: Exp2 moved from -0.6440 to -0.6271 against real -0.8592, Exp6 moved from 0.3733 to 0.3833 against real 0.1525, and Exp15 moved from 0.6498 to 0.6422 against real 0.7742. The candidate also slightly worsened Exp16, Exp10, and especially Exp11 (0.5300 vs real 0.4833). The improvements it did achieve, mostly in Exp3, Exp4, Exp8, and Exp9, are real but too small to compensate. This confirms the prior diagnosis that the aggressive low-cue weight recalibration (low kappa, high gamma) is not the binding knob: it did not fix the homogeneous low-bloc gate problem in Exp6, did not restore the strong margin-driven choice in Exp2, and further weakened lone-low following in Exp15. The candidate remains within the prescribed GVC family, so the problem is parameter/gating calibration rather than mechanism choice.
Rationale: Regenerate on top of the unchanged accepted iter 5 base, but do not repeat the rejected kappa/gamma recalibration in its aggressive form. A smaller intermediate step is preferable: kappa near 1.5-1.7 and gamma near 1.40-1.55, i.e. between the accepted base's high low-cue weights and the rejected candidate's suppressed low-cue weights. This should recover much of the lone-low strength lost in Exp15, Exp16, and Exp10 without reproducing the over-dominance that previously hurt Exp2 and Exp6. The binding remaining problem for Exp2 and Exp6 is the high-cue validity-gap gate: it is still opening too much for moderately deviant homogeneous below-median blocs. Tighten that gate specifically by (a) increasing D_half from 0.35-0.55 to roughly 0.60-0.90 and/or increasing q_dev from 2 to 3, so the dmax suppression more strongly closes the gate in the deviance regime around 0.8; (b) increasing D_boost from 0.6-1.2 to roughly 1.2-2.0 so the homogeneity boost activation is smaller for moderate mean deviance; and (c) keeping b_boost small, around 0.005-0.03, while retaining tight H_spread. These gate changes should pull Exp6 down toward 0.15 and make Exp2 more margin-dominated (more negative) without destroying the high-cue contribution needed in Exp4 and Exp13. For the residual Exp15 lone-low shortfall, rather than further low-cue amplification, nudge the fixed-temperature softmax beta upward modestly from 1.4-1.8 to about 1.7-2.0. This increases single-cue decisiveness where the low cue is diagnostic; it must be paired with the tighter gate above so Exp6 does not over-saturate. Keep gate_base=0, w_hi0=0, no lapse floor, and the prescribed GVC structure. These are parameter-level and gating repairs within the arbiter's mechanism family, not a switch of mechanism.

**Outcome of this advice:** iter 7 candidate loss=0.0678 -> iter 8 candidate loss=0.0625 -> the gate marked it REJECTED.

### Iteration 8 (most recent — address this)
Verdict: regenerate
Interpretation: The iter 8 candidate is a mixed recalibration of the accepted iter 5 base and the accept gate is right to reject it: its aggregate loss 0.0625 is worse than the accepted 0.0580, because it fixed a few weak experiments by regressing the ones the base already fitted. Compared with the accepted base, the candidate improves Exp15 (0.6498 to 0.7276, still short of the real 0.7742), Exp14 (0.0122 to 0.0244, real 0.0278), and Exp2 (-0.6440 to -0.7064, real -0.8592), and keeps Exp9 essentially exact. But it regresses Exp5 from 0.8246 to 0.8675 (real 0.8275), Exp1 from -0.0981 to -0.1123 (real -0.0959), Exp3 from -0.4469 to -0.4943 (real -0.3485), and Exp7 from -0.0045 to 0.0101 (real -0.0103). Most importantly, the extra gate tightening did not touch the Exp6 bottleneck and instead made it worse: 0.3733 at the base to 0.4213 now, against the real 0.1525. This confirms my iter 7 advice misfired: D_half, q_dev, D_boost and b_boost surgery does not control the Exp6 block contrast, and the kappa cut plus beta raise damaged the near-perfect Exp1 and Exp5. The persistent failures are therefore not gate thresholds but the shape of the below-median deviance weighting (too convex in dev, so cue position matters too much) and the shape of the above-median gap bonus (alpha above 1 concentrates too much weight on the single most extreme above-median cue). The candidate remains firmly within the prescribed GVC family, with gate_base 0, w_hi0 0, fixed-temperature softmax, and no lapse, so the remedy is parameter-shape and threshold recalibration, not a mechanism change.
Rationale: Regenerate on top of the unchanged accepted iter 5 base, but do not repeat any of the last three rejected directions (bloc exemption and b_boost raise, aggressive low-cue recalibration, and the iter 7 compound of intermediate kappa with D_half tightening and a beta raise). Pivot to the untested shape knobs. First, restore the accepted base's proven values: kappa 1.9-2.3, beta 1.40-1.80, D_half 0.35-0.55, q_dev 2, D_boost 0.6-1.2, b_boost 0.02-0.08; keep the iter 8 piece that demonstrably helped lone-low trials, namely lone_floor exactly 0.0. Second, attack Exp2 by lowering the gap exponent alpha from 1.1-1.8 to about 0.85-1.15, keeping rho unchanged. With alpha near 1 the validity-gap bonus spreads from the single top cue onto the 0.8 and 0.7 above-median cues, and in Exp2 the disagreement rows (direction from the top cue, margin from the block) will flip E_high_pre onto the margin side, removing the conflict flag and letting the high-cue side reinforce the margin instead of the top cue. This targets the remaining +0.15 error in Exp2 without reopening gate_base or b_boost. Third, attack Exp6 by flattening positional sensitivity instead of closing the gate: lower gamma from 1.25-1.45 to roughly 1.05-1.25 (the opposite of the rejected 1.55-1.80 raise), which compresses the below-median deviance weights toward each other and shrinks the block contrast driven by which positions carry the low cues, while leaving the total low-cue mass forExp5 and Exp10 essentially unchanged. Pair this with a focal_ratio raise from 3.0-4.0 to about 5.0-7.0 (the opposite of the rejected lowering to 2.0-2.5), so the extreme-cue exemption triggers less often and multi-active-low trials receive the uniform count attenuation that Exp6, Exp3 and Exp5 prefer. Fourth, handle the residual Exp15 shortfall only after these base-restoring changes: if lone-low following is still under 0.75, raise beta in a smaller band 1.80-2.00 and compensate the Exp5 overshoot risk with phi 0.65-0.85 rather than reopening any gate, since Exp15's metric is pinned by the softmax temperature once all other weights are zero. After lowering alpha, re-check Exp4 and Exp13 to confirm their licensed high-cue effects survive; if they weaken, use a modest rho nudge to 0.10-0.16, not b_boost. All of these are parameter-shape and threshold adjustments within the arbiter's prescribed GVC mechanism; keep the hard-zero high-cue default, no-lapse fixed-temperature softmax, and conflict gating intact.

**Outcome of this advice:** iter 8 candidate loss=0.0625 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Flattened-deviance validity-gap compensation (FD-GVC). People anchor each advertised expert validity to the ensemble median. Below-median cues earn decision weight from a median/MAD-normalized deviance score, but that deviance function is now deliberately compressive so position-to-position differences among below-median cues matter less and multi-cue homogeneous low blocs are attenuated by cue count rather than by sharply convex outlier salience. A solitary diagnostically deviant low cue is amplified, but the extreme-cue exemption among multiple active low cues is made stricter so near-median low blocs receive uniform count attenuation. At-or-above-median cues have no default weight; their validity-gap bonus is near-linear in the gap score and is licensed only by a very homogeneous, sufficiently deviant bloc of three or more simultaneously diagnostic below-median cues, with strong damping under low-versus-high conflict. Choice is a fixed-temperature softmax with no lapse floor.",
  "predict": "import numpy as np\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('GVC expects a (2, n_features) stimulus.')\n\n    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)\n    n_feat = validities.shape[0]\n    if n_feat != stim.shape[1]:\n        raise ValueError('validities length must match n_features.')\n\n    c = float(parameters['c'])\n    median = float(np.median(validities))\n    abs_dev = np.abs(validities - median)\n    mad = float(np.median(abs_dev))\n    scale = mad + c\n    if scale <= 1e-12:\n        scale = 1e-12\n\n    dev = np.maximum(0.0, median - validities) / scale\n    below = validities < median - 1e-12\n    at_or_above = ~below\n    gap = np.maximum(0.0, validities - median) / scale\n\n    w0 = float(parameters['w0'])\n    kappa = float(parameters['kappa'])\n    gamma = float(parameters['gamma'])\n    lam = float(parameters['lambda'])\n    sigma = float(parameters['sigma'])\n    phi = float(parameters['phi'])\n    nu = float(parameters['nu'])\n    opp_cap = float(parameters['opp_cap'])\n    focal_ratio = float(parameters['focal_ratio'])\n\n    raw = np.zeros(n_feat, dtype=float)\n    dev_bounded = np.minimum(dev, 100.0)\n    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)\n\n    diff = stim[0] - stim[1]\n    active_below = below & (np.abs(diff) > 1e-12)\n    n_active = int(np.count_nonzero(active_below))\n\n    def lone_amplification(idx):\n        base_amp = 1.0 + lam * np.tanh(float(dev[idx]) / sigma)\n        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)\n        n_opp = int(np.count_nonzero(opp))\n        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))\n\n    if n_active == 1:\n        focal = int(np.flatnonzero(active_below)[0])\n        raw[focal] = raw[focal] * lone_amplification(focal)\n    elif n_active >= 2:\n        att = 1.0 / (1.0 + phi * (n_active - 1))\n        active_devs = dev[active_below]\n        i_max = int(np.argmax(active_devs))\n        max_dev = float(active_devs[i_max])\n        med_dev = float(np.median(active_devs))\n        if med_dev > 1e-12 and max_dev > focal_ratio * med_dev:\n            focal = int(np.flatnonzero(active_below)[i_max])\n            others = below.copy()\n            others[focal] = False\n            raw[others] = raw[others] * att\n            raw[focal] = raw[focal] * lone_amplification(focal)\n        else:\n            raw[below] = raw[below] * att\n\n    gate_base = float(parameters['gate_base'])\n    lone_floor = float(parameters['lone_floor'])\n    D_half = float(parameters['D_half'])\n    q_dev = float(parameters['q_dev'])\n    b_boost = float(parameters['b_boost'])\n    D_boost = float(parameters['D_boost'])\n    H_spread = float(parameters['H_spread'])\n    K = float(parameters['K'])\n\n    gate = gate_base\n    if n_active == 1:\n        gate *= lone_floor\n\n    if n_active > 0:\n        active_devs = dev[active_below]\n        dmax = float(np.max(active_devs))\n    else:\n        dmax = 0.0\n\n    if dmax > 1e-12:\n        gate *= 1.0 / (1.0 + (dmax / D_half) ** q_dev)\n\n    if n_active >= 3:\n        low_vals = validities[active_below]\n        spread = float(np.std(low_vals))\n        spread_norm = spread / scale\n        hom = 1.0 / (1.0 + (spread_norm / H_spread) ** 12.0)\n        mean_dev = float(np.mean(dev[active_below]))\n        act = float(np.tanh(mean_dev / D_boost))\n        gate += b_boost * hom * act\n\n    w_hi0 = float(parameters['w_hi0'])\n    rho = float(parameters['rho'])\n    alpha = float(parameters['alpha'])\n\n    gap_bounded = np.minimum(gap, 100.0)\n    hi_pre = np.zeros(n_feat, dtype=float)\n    if np.any(at_or_above):\n        hi_pre[at_or_above] = w_hi0 + rho * np.power(gap_bounded[at_or_above], alpha)\n\n    E_low = float(np.dot(raw, diff))\n    E_high_pre = float(np.dot(hi_pre, diff))\n\n    if E_low * E_high_pre < 0.0:\n        conflict = 2.0 * min(abs(E_low), abs(E_high_pre)) / (abs(E_low) + abs(E_high_pre) + 1e-12)\n        gate *= 1.0 / (1.0 + (conflict / K) ** 2.0)\n\n    gate = float(np.maximum(gate, 0.0))\n\n    raw_high = np.zeros(n_feat, dtype=float)\n    if np.any(at_or_above):\n        raw_high[at_or_above] = w_hi0 + rho * np.power(gap_bounded[at_or_above], alpha) * gate\n\n    raw_all = raw + raw_high\n    total = float(np.sum(raw_all))\n    if total <= 0.0:\n        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))\n    else:\n        weights = raw_all / total\n\n    evidence_A = float(np.dot(weights, diff))\n\n    beta = float(parameters['beta'])\n    logits = np.array([evidence_A, 0.0], dtype=float)\n    z = beta * (logits - float(np.max(logits)))\n    e = np.exp(z)\n    probs = e / np.sum(e)\n    return probs\n",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=float)\n    probs = probs / probs.sum()\n    return int(np.random.choice(probs.shape[0], p=probs))\n",
  "parameters": {
    "c": "[0.008, 0.012]",
    "w0": "[0.007, 0.012]",
    "kappa": "[1.9, 2.3]",
    "gamma": "[1.05, 1.25]",
    "lambda": "[4.5, 5.5]",
    "sigma": "[0.10, 0.18]",
    "phi": "[0.65, 0.85]",
    "nu": "[0.15, 0.25]",
    "opp_cap": "{5}",
    "focal_ratio": "[5.0, 7.0]",
    "gate_base": "{0.0}",
    "lone_floor": "{0.0}",
    "D_half": "[0.35, 0.55]",
    "q_dev": "{2}",
    "b_boost": "[0.02, 0.08]",
    "D_boost": "[0.6, 1.2]",
    "H_spread": "[0.005, 0.02]",
    "K": "[0.30, 0.50]",
    "w_hi0": "{0.0}",
    "rho": "[0.08, 0.14]",
    "alpha": "[0.85, 1.15]",
    "beta": "[1.80, 2.00]",
    "validities": "validities"
  },
  "rationale": "Minimal in-family repair of the accepted GVC-strict candidate, following the latest advice and avoiding the three previously rejected repair directions. The low-cue deviance exponent is flattened from gamma 1.25-1.45 to 1.05-1.25, compressing positional sensitivity among below-median cues and directly targeting the Exp6 block-contrast over-prediction. The focal exemption is made stricter by raising focal_ratio from 3.0-4.0 to 5.0-7.0, so near-median multi-low configurations receive uniform count attenuation while a genuinely extreme lone low cue can still dominate. The high-cue validity-gap bonus is linearized by lowering alpha from 1.1-1.8 to 0.85-1.15, spreading licensed high-side weight away from the single top cue in Exp2 so high-cue evidence aligns with the block margin rather than the top-cue direction. lone_floor is fixed at exactly 0.0 to eliminate all residual high-cue leakage on lone-low trials. To recover the remaining Exp15 lone-low following, beta is raised only to the smaller advised band 1.80-2.00 and phi raised to 0.65-0.85 to protect Exp5 from over-saturation. gate_base and w_hi0 remain strictly zero, the fixed-temperature softmax remains lapse-free, and all conflict/homogeneity gate machinery from the accepted base is otherwise unchanged."
}
```

## Usage

```json
{
  "input_tokens": 47400,
  "output_tokens": 15548,
  "reasoning_tokens": 12974,
  "provider": "Baidu"
}
```
