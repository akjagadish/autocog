# generation_iter_06_attempt_00

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
- THEORY 1 = `pi_10`
- THEORY 2 = `pi_11`
- The recommendation below acts on THEORY 2 (= `pi_11`).

Replace pi_11 with a new theory that keeps the shared low-cue machinery (median anchoring of advertised validities, MAD-normalized deviance weight for below-median cues, lone low-cue amplification, and count/deviance-based attenuation of multiple low cues) but replaces the sign-flexible rank-ordered consensus term and the lapse floor with a low-bloc-size-gated high-cue term. Concretely: (i) when no below-median cues are diagnostic, fall back to equal weighting of all cues or median-cue majority so high cues do not dominate (captures Exp2); (ii) when a small number, roughly one or two, of below-median cues are diagnostic and their signed evidence is balanced or weak, give above-median cues a positive validity-gap weight so choices follow the high block (captures Exp1); (iii) when a large homogeneous bloc of below-median cues, roughly three or more per side, is diagnostic and balanced, make the high-cue gate negative or strongly downweighted so choices oppose the high block (captures Exp19); (iv) when low evidence is strongly one-sided, damp the high-cue term toward zero as conflict resolution; and (v) use a fixed-temperature softmax with no lapse floor (captures Exp15). The gate should be a continuous function of the number or mass of diagnostic below-median cues relative to the high bloc, not a strict three-cue license as in pi_10 and not an always-on rank bonus as in pi_11. This should preserve the good fits on Exp3-8, 14, 17, 18, and 20 while fixing the qualitative failures.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_10` (overall score: 0.829)

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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1024 (var=0.0003)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.6397 (var=0.0123)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=-0.4246 (var=0.0071)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.6833 (var=0.0225)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.8383 (var=0.0040)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.3608 (var=0.0101)
- Experiment 7: real=-0.0103 (var=0.0009) vs this=0.0036 (var=0.0016)
- Experiment 8: real=0.0472 (var=0.0067) vs this=-0.0054 (var=0.0055)
- Experiment 9: real=0.0032 (var=0.0007) vs this=-0.0062 (var=0.0005)
- Experiment 10: real=0.2001 (var=0.0006) vs this=0.1841 (var=0.0004)
- Experiment 11: real=0.4833 (var=0.0144) vs this=0.5083 (var=0.0098)
- Experiment 12: real=0.8375 (var=0.0133) vs this=0.8344 (var=0.0038)
- Experiment 13: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0173)
- Experiment 14: real=0.0278 (var=0.0063) vs this=-0.0217 (var=0.0149)
- Experiment 15: real=0.7742 (var=0.0352) vs this=0.6627 (var=0.0044)
- Experiment 16: real=0.8392 (var=0.0101) vs this=0.8198 (var=0.0021)
- Experiment 17: real=0.3783 (var=0.0022) vs this=0.6129 (var=0.0043)
- Experiment 18: real=0.0004 (var=0.0000) vs this=0.0003 (var=0.0000)
- Experiment 19: real=7.8384 (var=0.7057) vs this=-1.2702 (var=0.7861)
- Experiment 20: real=-0.7183 (var=0.0285) vs this=-0.4392 (var=0.0094)


---

### `pi_9` (overall score: 0.781)

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
- Experiment 17: real=0.3783 (var=0.0022) vs this=0.5133 (var=0.0025)
- Experiment 18: real=0.0004 (var=0.0000) vs this=-0.0098 (var=0.0000)
- Experiment 19: real=7.8384 (var=0.7057) vs this=2.1073 (var=0.8937)
- Experiment 20: real=-0.7183 (var=0.0285) vs this=-0.5467 (var=0.0130)


---

### `pi_11` (overall score: 0.689)

**Description**
Median anchoring of advertised validities gives below-median cues MAD-normalized deviance weight. Lone diagnostic low cues receive strong amplification, while multiple diagnostic low cues receive count-based attenuation that is deviance-selective: an extreme below-median singleton is partially exempt from attenuation without demoting inactive low cues. Above-median cues contribute through a sign-flexible rank-ordered consensus term gated by the signed evidence carried by actually diagnostic low cues. Balanced low evidence permits a negative consensus weight, opposing low evidence gates the high-cue term toward zero, and weak low evidence permits only a small capped positive rank bonus. Large comparable multi-low coalitions receive a linear soft attenuation of their total active low evidence to prevent aligned low blocs from over-saturating choice. Choice is a fixed-temperature softmax with a tiny lapse floor.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('DSA-SFCC expects a (2, n_features) stimulus.')

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
    at_or_above = ~below
    gap = np.maximum(0.0, validities - median) / scale

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    lone_gain = float(parameters['lone_gain'])
    lone_sigma = float(parameters['lone_sigma'])
    phi = float(parameters['phi'])
    nu = float(parameters['nu'])
    opp_cap = float(parameters['opp_cap'])

    raw = np.zeros(n_feat, dtype=float)
    dev_bounded = np.minimum(dev, 100.0)
    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)
    pre_att_raw = raw.copy()

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))
    use_low_cap = False

    def lone_amplification(idx):
        base_amp = 1.0 + lone_gain * np.tanh(float(dev[idx]) / lone_sigma)
        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)
        n_opp = int(np.count_nonzero(opp))
        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))

    if n_active == 1:
        focal = int(np.flatnonzero(active_below)[0])
        raw[focal] = raw[focal] * lone_amplification(focal)
    elif n_active >= 2:
        base_att = 1.0 / (1.0 + phi * (n_active - 1))
        d_act = dev[active_below]
        d_ref = float(np.median(d_act)) + 1e-12
        out_slope = float(parameters['out_slope'])
        out_thresh = float(parameters['out_thresh'])
        out_gain = float(parameters['out_gain'])

        ratio = np.clip(d_act / d_ref, 0.0, 20.0)
        outlier = 1.0 / (1.0 + np.exp(-out_slope * (ratio - out_thresh)))
        att_factors = base_att + (1.0 - base_att) * out_gain * outlier
        raw[active_below] = raw[active_below] * att_factors

        outlier_max = float(np.max(outlier))
        cap_out_gate = float(parameters['cap_out_gate'])
        n_cap_min = int(float(parameters['n_cap_min']))
        use_low_cap = (outlier_max < cap_out_gate) and (n_active >= n_cap_min)

    high_base = np.zeros(n_feat, dtype=float)
    above_idx = np.flatnonzero(above)
    if above_idx.size > 0:
        hi_gap = gap[above]
        order = np.argsort(hi_gap)
        m = above_idx.size
        ranks = np.empty(m, dtype=float)
        ranks[order] = np.arange(1, m + 1, dtype=float)
        if m > 1:
            u = (ranks - 1.0) / (m - 1.0)
        else:
            u = np.ones(m, dtype=float)
        rho_hi = float(parameters['rho_hi'])
        alpha_hi = float(parameters['alpha_hi'])
        bonus = rho_hi * np.power(np.clip(u, 0.0, 1.0), alpha_hi)
        high_base[above] = bonus
        total_hi = float(np.sum(high_base))
        h_cap = float(parameters['h_cap'])
        if total_hi > h_cap and total_hi > 1e-12:
            high_base[above] *= h_cap / total_hi

    total_low = float(np.sum(raw))
    E_low = float(np.dot(raw, diff))
    total_hi = float(np.sum(high_base))
    E_high = float(np.dot(high_base, diff))

    active_low_mass = float(np.sum(raw[active_below]))
    low_strength = abs(E_low) / (active_low_mass + 1e-12) if active_low_mass > 1e-12 else 0.0
    low_bal_eps = float(parameters['low_bal_eps'])
    min_low_mass = float(parameters['min_low_mass'])
    pre_att_active_mass = float(np.sum(pre_att_raw[active_below]))
    has_low_evidence = pre_att_active_mass > min_low_mass
    low_bal = 1.0 if (has_low_evidence and low_strength <= low_bal_eps) else 0.0
    weak_thresh = float(parameters['weak_thresh'])
    low_weak = 1.0 / (1.0 + (low_strength / weak_thresh) ** 2)

    active_high = above & (np.abs(diff) > 1e-12)
    n_ha = int(np.count_nonzero(active_high))
    consensus = 0.0
    if n_ha > 0:
        w_act = high_base[active_high]
        total_act = float(np.sum(w_act))
        if total_act > 1e-12:
            E_act = float(np.dot(w_act, diff[active_high]))
            align_share = 0.5 + 0.5 * abs(E_act) / total_act
            n_min_hi = float(parameters['n_min_hi'])
            size_factor = float(np.tanh(n_ha / max(n_min_hi, 1e-12)))
            consensus = align_share * size_factor

    cons_thresh = float(parameters['cons_thresh'])
    cons_s = float(parameters['cons_s'])
    zz = (cons_thresh - consensus) / cons_s
    zz = float(np.clip(zz, -60.0, 60.0))
    strong = 1.0 / (1.0 + float(np.exp(zz)))

    strong_neg = low_bal * strong
    h_pos = float(parameters['h_pos'])
    h_neg = float(parameters['h_neg'])
    g = h_pos * low_weak * (1.0 - strong_neg) - h_neg * strong_neg
    g = float(np.clip(g, -h_neg, h_pos))

    E_low_evidence = E_low
    if use_low_cap:
        low_multi_scale = float(parameters['low_multi_scale'])
        E_low_evidence = low_multi_scale * E_low

    denom = total_low + total_hi
    if denom <= 1e-12:
        evidence = 0.0
    else:
        evidence = (E_low_evidence + g * E_high) / denom

    beta = float(parameters['beta'])
    logits = np.array([evidence, 0.0], dtype=float)
    z = beta * (logits - float(np.max(logits)))
    z = np.clip(z, -60.0, 60.0)
    e = np.exp(z)
    core = e / np.sum(e)

    epsilon = float(parameters['epsilon'])
    probs = (1.0 - epsilon) * core + epsilon * np.ones(2, dtype=float) / 2.0
    probs = probs / np.sum(probs)
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
- lone_gain: [5.5, 6.5]
- lone_sigma: [0.10, 0.18]
- phi: [1.2, 1.5]
- nu: [0.15, 0.25]
- opp_cap: {5}
- out_gain: [0.50, 0.80]
- out_thresh: [2.2, 2.8]
- out_slope: [3.5, 5.5]
- cap_out_gate: {0.45}
- n_cap_min: {4}
- low_multi_scale: [0.60, 0.80]
- rho_hi: [0.025, 0.035]
- alpha_hi: [3.0, 4.5]
- h_cap: [0.06, 0.12]
- h_pos: [0.15, 0.25]
- h_neg: [1.30, 1.55]
- min_low_mass: {0.10}
- low_bal_eps: {0.02}
- weak_thresh: [0.12, 0.20]
- n_min_hi: {2}
- cons_thresh: [0.82, 0.88]
- cons_s: {0.04}
- beta: [1.85, 2.20]
- epsilon: {0.001}
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.0895 (var=0.0003)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.7337 (var=0.0110)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=-0.5035 (var=0.0056)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.5908 (var=0.0222)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.8846 (var=0.0020)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.3121 (var=0.0077)
- Experiment 7: real=-0.0103 (var=0.0009) vs this=0.0097 (var=0.0020)
- Experiment 8: real=0.0472 (var=0.0067) vs this=-0.0115 (var=0.0070)
- Experiment 9: real=0.0032 (var=0.0007) vs this=0.0018 (var=0.0009)
- Experiment 10: real=0.2001 (var=0.0006) vs this=0.1913 (var=0.0003)
- Experiment 11: real=0.4833 (var=0.0144) vs this=0.5017 (var=0.0068)
- Experiment 12: real=0.8375 (var=0.0133) vs this=0.8413 (var=0.0040)
- Experiment 13: real=0.0000 (var=0.0000) vs this=0.1185 (var=0.0145)
- Experiment 14: real=0.0278 (var=0.0063) vs this=0.0383 (var=0.0140)
- Experiment 15: real=0.7742 (var=0.0352) vs this=0.7404 (var=0.0043)
- Experiment 16: real=0.8392 (var=0.0101) vs this=0.8092 (var=0.0019)
- Experiment 17: real=0.3783 (var=0.0022) vs this=0.3569 (var=0.0017)
- Experiment 18: real=0.0004 (var=0.0000) vs this=0.0024 (var=0.0000)
- Experiment 19: real=7.8384 (var=0.7057) vs this=-23.9023 (var=1.1453)
- Experiment 20: real=-0.7183 (var=0.0285) vs this=0.1733 (var=0.0156)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1685 -> ACCEPTED
- iter 2: loss=0.1311 -> ACCEPTED
- iter 3: loss=0.1031 -> ACCEPTED
- iter 4: loss=0.1360 -> REJECTED
- iter 5: loss=0.1009 -> ACCEPTED
- iter 6: loss=0.1443 -> REJECTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.1009 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.1049 (var=0.0002) (Δ vs real -0.0090)
  - iter 2: -0.1046 (var=0.0003) (Δ vs real -0.0087)
  - iter 3: -0.1046 (var=0.0003) (Δ vs real -0.0087)
  - iter 4: -0.1049 (var=0.0003) (Δ vs real -0.0090)
  - iter 5: -0.1129 (var=0.0003) (Δ vs real -0.0170)
  - iter 6 (most recent): -0.1207 (var=0.0003) (Δ vs real -0.0247)
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
- pi_10: -0.1024 (var=0.0003)
- pi_11: -0.0895 (var=0.0003)

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
  - iter 1: -0.6574 (var=0.0128) (Δ vs real +0.2018)
  - iter 2: -0.6581 (var=0.0211) (Δ vs real +0.2011)
  - iter 3: -0.7003 (var=0.0147) (Δ vs real +0.1589)
  - iter 4: -0.6828 (var=0.0106) (Δ vs real +0.1764)
  - iter 5: -0.7574 (var=0.0100) (Δ vs real +0.1018)
  - iter 6 (most recent): -0.7589 (var=0.0107) (Δ vs real +0.1003)
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
- pi_10: -0.6397 (var=0.0123)
- pi_11: -0.7337 (var=0.0110)

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
  - iter 1: -0.4841 (var=0.0037) (Δ vs real -0.1356)
  - iter 2: -0.4676 (var=0.0070) (Δ vs real -0.1191)
  - iter 3: -0.4926 (var=0.0061) (Δ vs real -0.1441)
  - iter 4: -0.4857 (var=0.0063) (Δ vs real -0.1372)
  - iter 5: -0.5387 (var=0.0069) (Δ vs real -0.1902)
  - iter 6 (most recent): -0.5329 (var=0.0070) (Δ vs real -0.1844)
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
- pi_10: -0.4246 (var=0.0071)
- pi_11: -0.5035 (var=0.0056)

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
  - iter 1: -0.7092 (var=0.0130) (Δ vs real -0.0558)
  - iter 2: -0.6817 (var=0.0200) (Δ vs real -0.0283)
  - iter 3: -0.7008 (var=0.0140) (Δ vs real -0.0475)
  - iter 4: -0.6942 (var=0.0210) (Δ vs real -0.0408)
  - iter 5: -0.7633 (var=0.0119) (Δ vs real -0.1100)
  - iter 6 (most recent): -0.7875 (var=0.0091) (Δ vs real -0.1342)
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
- pi_10: -0.6833 (var=0.0225)
- pi_11: -0.5908 (var=0.0222)

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
  - iter 1: 0.8500 (var=0.0044) (Δ vs real +0.0225)
  - iter 2: 0.8517 (var=0.0022) (Δ vs real +0.0242)
  - iter 3: 0.8387 (var=0.0034) (Δ vs real +0.0112)
  - iter 4: 0.8588 (var=0.0031) (Δ vs real +0.0312)
  - iter 5: 0.8688 (var=0.0035) (Δ vs real +0.0413)
  - iter 6 (most recent): 0.8892 (var=0.0021) (Δ vs real +0.0617)
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
- pi_10: 0.8383 (var=0.0040)
- pi_11: 0.8846 (var=0.0020)

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
  - iter 1: 0.4188 (var=0.0074) (Δ vs real +0.2662)
  - iter 2: 0.4108 (var=0.0068) (Δ vs real +0.2583)
  - iter 3: 0.4096 (var=0.0079) (Δ vs real +0.2571)
  - iter 4: 0.4217 (var=0.0067) (Δ vs real +0.2692)
  - iter 5: 0.4733 (var=0.0067) (Δ vs real +0.3208)
  - iter 6 (most recent): 0.4750 (var=0.0098) (Δ vs real +0.3225)
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
- pi_10: 0.3608 (var=0.0101)
- pi_11: 0.3121 (var=0.0077)

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
  - iter 1: 0.0270 (var=0.0018) (Δ vs real +0.0372)
  - iter 2: 0.0050 (var=0.0023) (Δ vs real +0.0152)
  - iter 3: 0.0095 (var=0.0026) (Δ vs real +0.0197)
  - iter 4: 0.0196 (var=0.0022) (Δ vs real +0.0298)
  - iter 5: 0.0106 (var=0.0029) (Δ vs real +0.0208)
  - iter 6 (most recent): 0.0089 (var=0.0023) (Δ vs real +0.0192)
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
- pi_10: 0.0036 (var=0.0016)
- pi_11: 0.0097 (var=0.0020)

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
  - iter 1: 0.0082 (var=0.0071) (Δ vs real -0.0390)
  - iter 2: 0.0027 (var=0.0091) (Δ vs real -0.0444)
  - iter 3: 0.0077 (var=0.0084) (Δ vs real -0.0394)
  - iter 4: 0.0125 (var=0.0067) (Δ vs real -0.0347)
  - iter 5: 0.0094 (var=0.0089) (Δ vs real -0.0378)
  - iter 6 (most recent): -0.0101 (var=0.0090) (Δ vs real -0.0573)
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
- pi_10: -0.0054 (var=0.0055)
- pi_11: -0.0115 (var=0.0070)

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
  - iter 1: -0.0003 (var=0.0004) (Δ vs real -0.0035)
  - iter 2: 0.0028 (var=0.0008) (Δ vs real -0.0004)
  - iter 3: -0.0011 (var=0.0008) (Δ vs real -0.0043)
  - iter 4: -0.0049 (var=0.0008) (Δ vs real -0.0080)
  - iter 5: -0.0001 (var=0.0007) (Δ vs real -0.0033)
  - iter 6 (most recent): 0.0025 (var=0.0006) (Δ vs real -0.0007)
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
- pi_10: -0.0062 (var=0.0005)
- pi_11: 0.0018 (var=0.0009)

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
  - iter 1: 0.1912 (var=0.0004) (Δ vs real -0.0089)
  - iter 2: 0.1859 (var=0.0003) (Δ vs real -0.0142)
  - iter 3: 0.1892 (var=0.0003) (Δ vs real -0.0110)
  - iter 4: 0.1893 (var=0.0004) (Δ vs real -0.0108)
  - iter 5: 0.1930 (var=0.0002) (Δ vs real -0.0071)
  - iter 6 (most recent): 0.1987 (var=0.0002) (Δ vs real -0.0014)
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
- pi_10: 0.1841 (var=0.0004)
- pi_11: 0.1913 (var=0.0003)

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
  - iter 1: 0.4917 (var=0.0128) (Δ vs real +0.0083)
  - iter 2: 0.4867 (var=0.0104) (Δ vs real +0.0033)
  - iter 3: 0.5017 (var=0.0085) (Δ vs real +0.0183)
  - iter 4: 0.4917 (var=0.0098) (Δ vs real +0.0083)
  - iter 5: 0.4933 (var=0.0038) (Δ vs real +0.0100)
  - iter 6 (most recent): 0.5167 (var=0.0100) (Δ vs real +0.0333)
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
- pi_10: 0.5083 (var=0.0098)
- pi_11: 0.5017 (var=0.0068)

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
  - iter 1: 0.8381 (var=0.0048) (Δ vs real +0.0006)
  - iter 2: 0.8375 (var=0.0037) (Δ vs real +0.0000)
  - iter 3: 0.8337 (var=0.0052) (Δ vs real -0.0038)
  - iter 4: 0.8400 (var=0.0046) (Δ vs real +0.0025)
  - iter 5: 0.8744 (var=0.0044) (Δ vs real +0.0369)
  - iter 6 (most recent): 0.8819 (var=0.0033) (Δ vs real +0.0444)
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
- pi_10: 0.8344 (var=0.0038)
- pi_11: 0.8413 (var=0.0040)

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
  - iter 1: 0.0000 (var=0.0156) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0173) (Δ vs real +0.0000)
  - iter 3: 0.0385 (var=0.0208) (Δ vs real +0.0385)
  - iter 4: 0.0860 (var=0.0194) (Δ vs real +0.0860)
  - iter 5: 0.0000 (var=0.0136) (Δ vs real +0.0000)
  - iter 6 (most recent): 0.0532 (var=0.0187) (Δ vs real +0.0532)
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
- pi_10: 0.0000 (var=0.0173)
- pi_11: 0.1185 (var=0.0145)

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
  - iter 1: 0.3233 (var=0.0087) (Δ vs real +0.2956)
  - iter 2: 0.3406 (var=0.0149) (Δ vs real +0.3128)
  - iter 3: 0.0500 (var=0.0148) (Δ vs real +0.0222)
  - iter 4: 0.0633 (var=0.0143) (Δ vs real +0.0356)
  - iter 5: 0.0672 (var=0.0087) (Δ vs real +0.0394)
  - iter 6 (most recent): 0.0250 (var=0.0128) (Δ vs real -0.0028)
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
- pi_10: -0.0217 (var=0.0149)
- pi_11: 0.0383 (var=0.0140)

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
  - iter 1: 0.5729 (var=0.0122) (Δ vs real -0.2013)
  - iter 2: 0.6982 (var=0.0098) (Δ vs real -0.0760)
  - iter 3: 0.6920 (var=0.0072) (Δ vs real -0.0822)
  - iter 4: 0.6902 (var=0.0057) (Δ vs real -0.0840)
  - iter 5: 0.7658 (var=0.0060) (Δ vs real -0.0084)
  - iter 6 (most recent): 0.7676 (var=0.0045) (Δ vs real -0.0067)
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
- pi_10: 0.6627 (var=0.0044)
- pi_11: 0.7404 (var=0.0043)

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
  - iter 1: 0.6196 (var=0.0029) (Δ vs real -0.2196)
  - iter 2: 0.7010 (var=0.0027) (Δ vs real -0.1381)
  - iter 3: 0.7917 (var=0.0036) (Δ vs real -0.0475)
  - iter 4: 0.8198 (var=0.0026) (Δ vs real -0.0194)
  - iter 5: 0.8735 (var=0.0012) (Δ vs real +0.0344)
  - iter 6 (most recent): 0.7900 (var=0.0028) (Δ vs real -0.0492)
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
- pi_10: 0.8198 (var=0.0021)
- pi_11: 0.8092 (var=0.0019)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np

VALIDITIES = np.array([0.7985, 0.7985, 0.7985, 0.7985, 0.7985, 0.7985, 0.8, 0.8, 0.8, 0.8, 0.86, 0.88, 0.9, 0.95, 0.99])
MEDIAN = float(np.median(VALIDITIES))
HIGH_MASK = VALIDITIES > MEDIAN + 1e-12
HIGH_WEIGHTS = VALIDITIES - MEDIAN

def metric(data):
    a = data['option_a_ratings'].tolist()
    b = data['option_b_ratings'].tolist()
    r = data['response'].to_numpy()
    match = []
    for ai, bi, ri in zip(a, b, r):
        ai_arr = np.asarray(ai, dtype=float)
        bi_arr = np.asarray(bi, dtype=float)
        ev = float(np.dot(HIGH_WEIGHTS[HIGH_MASK], ai_arr[HIGH_MASK] - bi_arr[HIGH_MASK]))
        if ev > 1e-12:
            rec = 0
        elif ev < -1e-12:
            rec = 1
        else:
            continue
        match.append(1.0 if ri == rec else 0.0)
    if not match:
        return float('nan')
    return float(np.mean(match))
```

**Observed (real) value:** 0.3783 (var=0.0022)
**Previous candidate values (this loop):**
  - iter 1: 0.2606 (var=0.0030) (Δ vs real -0.1177)
  - iter 2: 0.2754 (var=0.0025) (Δ vs real -0.1029)
  - iter 3: 0.3008 (var=0.0021) (Δ vs real -0.0775)
  - iter 4: 0.2698 (var=0.0023) (Δ vs real -0.1085)
  - iter 5: 0.2971 (var=0.0032) (Δ vs real -0.0813)
  - iter 6 (most recent): 0.2338 (var=0.0020) (Δ vs real -0.1446)
**Other theories' values on this metric (for reference):**
- pi_10: 0.6129 (var=0.0043)
- pi_9: 0.5133 (var=0.0025)
- pi_1: 0.7302 (var=0.0056)
- pi_2: 0.7373 (var=0.0043)
- pi_3: 0.7998 (var=0.0012)
- pi_4: 0.7013 (var=0.0013)
- pi_5: 0.5038 (var=0.0035)
- pi_6: 0.5517 (var=0.0019)
- pi_7: 0.5256 (var=0.0024)
- pi_8: 0.4973 (var=0.0024)
- pi_11: 0.3569 (var=0.0017)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    a = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b = np.array(data['option_b_ratings'].tolist(), dtype=float)

    low_side = (b[:, 0] > a[:, 0]).astype(float)
    resp = data['response'].to_numpy(dtype=float)
    agree_low = (resp == low_side).astype(float)

    k = (a[:, 2:] != b[:, 2:]).sum(axis=1).astype(float)

    frame = data[['subject_id']].copy()
    frame['agree_low'] = agree_low
    frame['k'] = k

    slopes = []
    for sid, sub in frame.groupby('subject_id'):
        x = sub['k'].to_numpy(dtype=float)
        y = sub['agree_low'].to_numpy(dtype=float)
        xm = x.mean()
        ym = y.mean()
        denom = np.sum((x - xm) ** 2)
        if denom <= 1e-12:
            slopes.append(0.0)
        else:
            slopes.append(float(np.sum((x - xm) * (y - ym)) / denom))

    return float(np.mean(slopes)) if slopes else float('nan')
```

**Observed (real) value:** 0.0004 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: -0.0118 (var=0.0000) (Δ vs real -0.0122)
  - iter 2: -0.0074 (var=0.0001) (Δ vs real -0.0078)
  - iter 3: -0.0014 (var=0.0000) (Δ vs real -0.0018)
  - iter 4: 0.0020 (var=0.0000) (Δ vs real +0.0016)
  - iter 5: -0.0034 (var=0.0001) (Δ vs real -0.0038)
  - iter 6 (most recent): -0.0039 (var=0.0001) (Δ vs real -0.0043)
**Other theories' values on this metric (for reference):**
- pi_9: -0.0098 (var=0.0000)
- pi_10: 0.0003 (var=0.0000)
- pi_1: -0.0154 (var=0.0000)
- pi_2: -0.0150 (var=0.0000)
- pi_3: -0.0178 (var=0.0000)
- pi_4: -0.0182 (var=0.0000)
- pi_5: -0.0135 (var=0.0000)
- pi_6: -0.0182 (var=0.0000)
- pi_7: -0.0180 (var=0.0000)
- pi_8: -0.0037 (var=0.0000)
- pi_11: 0.0024 (var=0.0000)

### Experiment 19
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    kA = data['option_a_ratings'].apply(lambda x: sum(x[7:13]))
    low = data.loc[kA <= 2, 'response'].mean()
    high = data.loc[kA >= 3, 'response'].mean()
    contrast = low - high
    n_subj = data['subject_id'].nunique()
    se = np.sqrt(0.25 * (1.0 / 48.0 + 1.0 / 48.0))
    return float(contrast / (se / np.sqrt(n_subj)))
```

**Observed (real) value:** 7.8384 (var=0.7057)
**Previous candidate values (this loop):**
  - iter 1: 33.3420 (var=0.5688) (Δ vs real +25.5036)
  - iter 2: 27.1643 (var=2.3845) (Δ vs real +19.3260)
  - iter 3: -0.3464 (var=7.7943) (Δ vs real -8.1848)
  - iter 4: -27.6839 (var=1.1911) (Δ vs real -35.5223)
  - iter 5: -1.1547 (var=12.0267) (Δ vs real -8.9931)
  - iter 6 (most recent): -24.1332 (var=8.8101) (Δ vs real -31.9716)
**Other theories' values on this metric (for reference):**
- pi_10: -1.2702 (var=0.7861)
- pi_11: -23.9023 (var=1.1453)
- pi_1: 24.9704 (var=1.1071)
- pi_2: 45.0911 (var=2.0926)
- pi_3: 59.8135 (var=0.5103)
- pi_4: 41.7713 (var=0.5290)
- pi_5: 11.9800 (var=1.1921)
- pi_6: 10.3923 (var=0.9683)
- pi_7: 16.4545 (var=0.9383)
- pi_8: 3.8105 (var=1.0363)
- pi_9: 2.1073 (var=0.8937)

### Experiment 20
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    top_a = data['option_a_ratings'].apply(lambda x: int(x[-1]))
    top_b = data['option_b_ratings'].apply(lambda x: int(x[-1]))
    top_favors_a = (top_a > top_b).astype(int)
    chose_a = (data['response'] == 0).astype(int)
    congruent = (top_favors_a == chose_a).astype(int)
    return float(2.0 * congruent.mean() - 1.0)

```

**Observed (real) value:** -0.7183 (var=0.0285)
**Previous candidate values (this loop):**
  - iter 1: -0.4767 (var=0.0103) (Δ vs real +0.2417)
  - iter 2: -0.4571 (var=0.0063) (Δ vs real +0.2612)
  - iter 3: -0.1329 (var=0.0086) (Δ vs real +0.5854)
  - iter 4: -0.1192 (var=0.0120) (Δ vs real +0.5992)
  - iter 5: -0.0529 (var=0.0105) (Δ vs real +0.6654)
  - iter 6 (most recent): -0.0383 (var=0.0148) (Δ vs real +0.6800)
**Other theories' values on this metric (for reference):**
- pi_11: 0.1733 (var=0.0156)
- pi_10: -0.4392 (var=0.0094)
- pi_1: 0.6879 (var=0.0443)
- pi_2: -0.7512 (var=0.0213)
- pi_3: -0.5600 (var=0.1248)
- pi_4: -0.8175 (var=0.0028)
- pi_5: -0.4387 (var=0.0077)
- pi_6: -0.8054 (var=0.0024)
- pi_7: -0.3554 (var=0.0109)
- pi_8: -0.0658 (var=0.0075)
- pi_9: -0.5467 (var=0.0130)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
People anchor advertised expert validities to the ensemble median and assign decision weight to below-median cues by a median/MAD-normalized deviance score. A solitary diagnostic low cue is strongly amplified, multiple diagnostic low cues are attenuated by count and deviance-selective mechanisms, and all early below-median weighting is shared across experiments. Above-median cues enter only through a continuous validity-gap gate whose sign and strength depend on the size, homogeneity, and balance of the diagnostic low-cue bloc, and on the concentration of the high-cue validity-gap profile. The gate is not a strict license and not an always-on rank bonus: no diagnostic low cues trigger an equal-weight fallback; a small homogeneous balanced low bloc licenses weak positive high-cue following; a large homogeneous low bloc turns the high-cue term negative or nearly off; one-sided low evidence damps the gate; and a standout top cue can push the negative gate beyond its diffuse-bloc bound. Choice is a fixed-temperature softmax with no lapse floor.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('GVC-continuous expects a (2, n_features) stimulus.')

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
    gap = np.maximum(0.0, validities - median) / scale
    below = validities < median - 1e-12
    at_or_above = ~below

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))

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

    def lone_amplification(idx):
        base_amp = 1.0 + lam * np.tanh(float(dev[idx]) / sigma)
        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)
        n_opp = int(np.count_nonzero(opp))
        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))

    if n_active == 1:
        focal = int(np.flatnonzero(active_below)[0])
        raw[focal] = raw[focal] * lone_amplification(focal)
    elif n_active >= 2:
        base_att = 1.0 / (1.0 + phi * (n_active - 1))
        active_devs = dev[active_below]
        i_max = int(np.argmax(active_devs))
        max_dev = float(active_devs[i_max])
        med_dev = float(np.median(active_devs))
        if med_dev > 1e-12 and max_dev > focal_ratio * med_dev:
            focal = int(np.flatnonzero(active_below)[i_max])
            others = below.copy()
            others[focal] = False
            raw[others] = raw[others] * base_att
            raw[focal] = raw[focal] * lone_amplification(focal)
        else:
            raw[below] = raw[below] * base_att

    beta = float(parameters['beta'])
    fallback_beta_scale = float(parameters['fallback_beta_scale'])

    if n_active == 0:
        evidence = float(np.mean(diff))
        beta_eff = beta * fallback_beta_scale
    else:
        beta_eff = beta
        mass_active = float(np.sum(raw[active_below]))
        E_low = float(np.dot(raw, diff))
        imbalance = 0.0
        if mass_active > 1e-12:
            imbalance = min(1.0, abs(E_low) / mass_active)

        K_bal = float(parameters['K_bal'])
        low_balance = 1.0 / (1.0 + (imbalance / K_bal) ** 2.0)

        H_spread = float(parameters['H_spread'])
        if n_active >= 2:
            spread_norm = float(np.std(validities[active_below])) / scale
            homog = 1.0 / (1.0 + (spread_norm / H_spread) ** 4.0)
        else:
            homog = 1.0

        D_half = float(parameters['D_half'])
        dmax = float(np.max(dev[active_below]))
        dev_damp = 1.0 / (1.0 + (dmax / D_half) ** 2.0)

        h_pos = float(parameters['h_pos'])
        h_neg = float(parameters['h_neg'])
        n_small = float(parameters['n_small'])
        s_small = float(parameters['s_small'])
        n_large = float(parameters['n_large'])
        s_large = float(parameters['s_large'])

        z_small = (float(n_active) - n_small) / s_small
        small_win = float(np.exp(-z_small * z_small))

        z_large = (float(n_active) - n_large) / s_large
        z_large = float(np.clip(z_large, -50.0, 50.0))
        large_win = 1.0 / (1.0 + float(np.exp(-z_large)))

        conc_boost_gain = float(parameters['conc_boost_gain'])
        conc_power = float(parameters['conc_power'])
        if np.any(at_or_above):
            gap_above = gap[at_or_above]
            sum_gap = float(np.sum(gap_above))
            if sum_gap > 1e-12:
                max_gap_share = float(np.max(gap_above)) / sum_gap
            else:
                max_gap_share = 0.0
            conc_boost = 1.0 + conc_boost_gain * (max_gap_share ** conc_power)
        else:
            conc_boost = 1.0

        gate = h_pos * small_win * homog - h_neg * large_win * homog
        gate = gate * low_balance * dev_damp
        if n_active == 1:
            gate = gate * float(parameters['one_cue_gate_scale'])
        lower_bound = -h_neg * conc_boost
        gate = float(np.clip(gate, lower_bound, h_pos))

        rho = float(parameters['rho'])
        alpha = float(parameters['alpha'])
        branch_boost = float(parameters['branch_boost'])
        hi_boost = 1.0
        if n_active >= 2:
            hi_boost = 1.0 + branch_boost * small_win * homog * low_balance

        hi_pre = np.zeros(n_feat, dtype=float)
        gap_bounded = np.minimum(gap, 100.0)
        hi_pre[at_or_above] = rho * hi_boost * np.power(gap_bounded[at_or_above], alpha)

        raw_high = gate * hi_pre
        raw_all = raw + raw_high
        if n_active == 1:
            norm = float(np.sum(raw[below]))
        else:
            norm = float(np.sum(raw[below])) + float(np.sum(np.abs(raw_high[at_or_above])))
        if norm <= 1e-12:
            evidence = float(np.mean(diff))
        else:
            evidence = float(np.dot(raw_all, diff) / norm)

    logits = np.array([evidence, 0.0], dtype=float)
    z = beta_eff * (logits - float(np.max(logits)))
    z = np.clip(z, -60.0, 60.0)
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
- kappa: [2.30, 2.80]
- gamma: [1.25, 1.45]
- lambda: [7.5, 10.5]
- sigma: [0.14, 0.22]
- phi: [0.38, 0.55]
- nu: [0.15, 0.25]
- opp_cap: {5}
- focal_ratio: [3.0, 4.0]
- rho: [0.08, 0.14]
- alpha: [1.1, 1.8]
- h_pos: [0.02, 0.07]
- h_neg: [0.12, 0.20]
- n_small: [1.0, 1.8]
- s_small: [0.30, 0.55]
- n_large: [7.5, 9.2]
- s_large: [1.5, 2.5]
- K_bal: [0.40, 0.65]
- H_spread: [0.003, 0.012]
- D_half: [0.40, 0.65]
- beta: [1.8, 2.3]
- fallback_beta_scale: [0.08, 0.20]
- one_cue_gate_scale: [0.15, 0.35]
- conc_boost_gain: [1.2, 2.5]
- conc_power: {2}
- branch_boost: [0.3, 0.8]
- validities: validities

`rationale`: This is a minimal in-family edit of the accepted iter-3 continuous-gate base, implementing the iter-4 feedback while avoiding the rejected h_neg restoration. (1) The negative gate is now concentration-driven: the lower clamp is scaled by 1 + conc_boost_gain * max_gap_share^conc_power, so a standout top cue (Exp20) can push anti-top weight below the previous hard -h_neg bound, whereas Exp19's diffuse high-gap profile keeps near-iter-3 strength. (2) Exp19 is pulled structurally out of the negative window by moving n_large up to 7.5-9.2 and widening s_large, rather than raising h_neg, which crashed Exp19 in the rejected candidate. (3) The positive small-bloc gate is now multiplied by the same low-bloc homogeneity factor, specifically attenuating heterogeneous multi-cue low blocs (Exp6) without touching h_pos; Exp2/Exp17 are compensated by a branch-restricted hi_boost that multiplies rho only for balanced, homogeneous, small low-bloc cases. (4) In the n_active==1 branch the normalization drops the near-zero high-cue terms and uses sum(raw[below]), reducing dilution of the lone low cue (Exp15/Exp16), with one_cue_gate_scale lowered to further limit high-cue interference. (5) beta is raised slightly for noise-based metrics while fallback_beta_scale is tightened so the n_active==0 equal-weight fallback (Exp14) does not re-inflate. Architecture, no-lapse fixed-temperature softmax, and the shared low-cue machinery are preserved.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a faithful implementation of the arbiter's continuous low-bloc-gated high-cue mechanism and fixes the sign of Exp19, which pi_11 got backwards, while preserving several strong fits: Exp1 (-0.1049 vs -0.0959), Exp5 (0.850 vs 0.828), Exp9-13, and Exp18 are all near observed values, and Exp20 has the correct anti-top-cue sign even if too shallow (-0.477 vs -0.718). The main problem is magnitude calibration of the high-cue gate and the relative potency of the low-cue branch. Exp19 saturates: 33.34 vs observed 7.84, meaning the large homogeneous low-bloc anti-high gate is too strong. Exp14 (0.323 vs 0.028) and Exp6 (0.419 vs 0.153) overshoot in the opposite direction, indicating the small-bloc positive high-cue window is too strong or too wide. Exp15 (0.573 vs 0.774) and Exp16 (0.620 vs 0.839) under-follow the lone low cue, so the preserved low-cue machinery is being outcompeted by the above-median contribution. Exp2 (-0.657 vs -0.859) and Exp17 (0.261 vs 0.378) are also shallow. These are consistent with a gate whose positive and negative gains are too coarsely scaled, not with the wrong architecture.
Rationale: Keep the same low-cue machinery and continuous gate family. Rebalance parameters rather than redesign: (1) reduce h_neg from 0.70-1.10 to roughly 0.30-0.50, and/or shift n_large up toward 5.5-6.5 with a wider s_large, so the Exp19 contrast falls from 33 toward the observed 7.8 instead of hitting a near-deterministic anti-high choice. (2) tighten the positive small-bloc window by lowering s_small to about 0.5-0.8 and reducing h_pos to about 0.05-0.12; the current window is broad enough to produce too much high-block following in Exp14 and Exp6. (3) increase low-cue potency, e.g. raise kappa to 2.3-2.8 and lambda to 5.5-7.0, or re-normalize so the lone low cue is not diluted by the positive high-cue term, which should lift Exp15 and Exp16 toward their observed values. (4) Re-check Exp2 and Exp17 after those changes; if Exp2 remains too shallow, increase rho or alpha selectively inside the balanced small-bloc branch instead of raising h_pos globally. Do not reintroduce a lapse floor or the sign-flexible rank consensus term; the architecture is in-family and only the gate strength and low/high weight balance need correction.

**Outcome of this advice:** iter 1 candidate loss=0.1685 -> iter 2 candidate loss=0.1311 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: This candidate stays inside the prescribed continuous low-bloc-gated high-cue family and the parameter-level recalibration clearly helped overall: aggregate loss dropped from 0.1685 to 0.1311 and the gate accepted it. The wins are visible per experiment: Exp15 moved from 0.573 to 0.698 (real 0.774), Exp16 from 0.620 to 0.701 (real 0.839), Exp19 from 33.3 to 27.2 (real 7.8), Exp4 from 0.709 to 0.682 (real 0.653), Exp3 from 0.485 to 0.468 (real 0.349), and Exp7, Exp9, Exp11 tightened to near zero deltas. Several experiments are essentially perfect or excellent: Exp1 (-0.105 vs -0.096), Exp5 (0.852 vs 0.828), Exp9 (0.0028 vs 0.0032), Exp10 (0.186 vs 0.200), Exp11 (0.487 vs 0.483), Exp12 (0.8375 vs 0.8375), Exp13 (0.0 vs 0.0), Exp18 (-0.007 vs 0.0004). So the low-cue potency boost (kappa, lambda) worked as intended. However the candidate still misses the arbiter's target experiments by large margins. Exp19 remains about 3.5 times too strong at 27.2 vs 7.8 observed. Exp14 got WORSE after the h_pos/s_small reduction (0.323 to 0.341 vs 0.028 observed), which means that lever does not drive the Exp14 overshoot. Exp6 is still high (0.411 vs 0.153). Exp20 is still too shallow (-0.457 vs -0.718). Exp2 is unchanged and shallow (-0.658 vs -0.859). Exp16 (-0.138) and Exp15 (-0.076) still under-follow the lone low cue. This is a calibration and branching problem, not a wrong architecture, but it is not yet experiment-invariant enough to ship.
Rationale: Keep the same machine and gate family, but do not repeat the h_pos/s_small lever for Exp14 since lowering it failed there. Concrete next moves, in priority order: (1) Exp19 is still about 3.5x too large, so cut the negative gate much harder: h_neg to about 0.08-0.20, n_large up to 6.5-8.0, s_large wider (1.2-2.5), and relax the multi-cue attenuation phi to about 0.35-0.55 or add a floor to base_att so the low bloc keeps enough weight to partially counter the negative gate; the observed raw contrast is a partial reversal (about 0.15), not a full flip, so the low branch and the negative high term should be near-equal rather than the high term dominating. (2) Exp14 worsened despite weaker h_pos, so the +0.34 overshoot probably comes from the n_active == 0 equal-weight fallback integrating the 15 aligned cues with the full beta; soften that branch specifically, e.g., an effective temperature multiplier on beta of about 0.3-0.7 used only in the n_active == 0 fallback, or a count/redundancy cap on the fallback evidence so a large homogeneous bloc of tied-or-high cues is not integrated linearly. Re-check Exp2 after this, since the fallback also drives that metric and it currently sits 0.20 too high. (3) Exp20's negative gate is too weak while Exp19's is too strong, so gate the negative term on the concentration of the high bloc: multiply the negative gate by a factor that grows when the highest-validity cue stands alone or where the high-bloc gap is large, e.g., a term based on the maximum gap of the at-or-above cues; this raises the anti-top magnitude in Exp20 without touching Exp19's homogeneous bloc. (4) Exp15/16 still under-follow the lone low cue: raise lambda to 6.5-9.0 and sigma to 0.12-0.20, and in the n_active == 1 branch reduce the dilution of the lone cue by making the positive high-cue contribution smaller there (e.g., scale the one-cue branch gate by about 0.5), rather than boosting beta globally, which would re-inflate Exp14. (5) Exp2 and Exp17 remain shallow: raise rho or alpha selectively inside the balanced small-bloc branch (as previously suggested and not yet tried) instead of raising h_pos globally. Do not reintroduce the lapse floor or the sign-flexible rank consensus term, and do not switch mechanism families; these are all in-family calibrations of the gate shape, the fallback normalization, and the low/high weight balance.

**Outcome of this advice:** iter 2 candidate loss=0.1311 -> iter 3 candidate loss=0.1031 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: This candidate is faithful to the arbiter's prescribed continuous low-bloc-gated high-cue family and the aggregate loss improved from 0.1311 to 0.1031. The fallback-temperature fix worked as intended: Exp14 moved from 0.341 to 0.050 against observed 0.028. The lone-cue changes lifted Exp16 from 0.701 to 0.792 and Exp15 remains reasonably close at 0.692 versus 0.774. Exp1, Exp5, Exp9 through Exp12, and Exp18 are all near the observed values. However the current parameter set overshot the Exp19 correction: it now gives -0.346 whereas the real value is +7.838, a full sign reversal. The same weakened negative gate left Exp20 too shallow at -0.133 versus -0.718. Exp6 remains too high at 0.410 versus 0.153, Exp2 remains too shallow at -0.700 versus -0.859, Exp17 is still low at 0.301 versus 0.378, Exp3 is too negative at -0.493 versus -0.349, and Exp13 regressed from 0.000 to 0.039. These are calibration failures of the gate strength and low/high weight balance rather than evidence against the prescribed architecture.
Rationale: Regenerate with in-family calibration, not redesign. First, restore h_neg to an intermediate band around 0.30 to 0.50, and move n_large to about 5.8 to 6.8 with s_large around 1.0 to 1.8. This is the midpoint between the earlier saturating 0.70 to 1.10 regime and the current 0.12 to 0.20 regime; it should recover a positive but partial Exp19 contrast instead of the current negative one. Second, strengthen the Exp20 anti-top pathway through the concentration factor: raise conc_gain to about 1.5 to 2.5 or use max_gap_share raised to a power greater than 1, so a standout top cue receives a much stronger negative multiplier without re-amplifying Exp19's less concentrated high bloc. Third, trim the positive small-bloc window that is driving Exp6 over-following: lower h_pos to about 0.01 to 0.04 and/or set n_small to 0.8 to 1.5 with s_small to 0.20 to 0.40. If Exp2 and Exp17 become too shallow as a result, raise rho or alpha inside the balanced small-bloc branch rather than restoring h_pos globally. Fourth, to close the remaining Exp15 and Exp16 gaps, lower one_cue_gate_scale to about 0.15 to 0.35 and increase lambda toward 8 to 11 with sigma around 0.12 to 0.22. Keep fallback_beta_scale near its current range because it fixed Exp14. Recheck Exp3, Exp8, and Exp13 after these balance changes. Do not add a lapse floor or the sign-flexible rank-consensus term, and stay inside the continuous gated family.

**Outcome of this advice:** iter 3 candidate loss=0.1031 -> iter 4 candidate loss=0.1360 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate stays inside the prescribed continuous low-bloc-gated high-cue family, but its recalibration overshot: aggregate loss rose from 0.1031 (accepted iter-3 base) to 0.1360 and the gate marked it REJECTED. The single biggest failure is Exp19: the proposed restoration of h_neg to 0.30-0.50 with n_large 5.8-6.8 turned a small negative contrast (-0.35) into a strongly negative one (-27.68), the opposite of the intended 'partial positive' recovery (real +7.84). Tellingly, the Exp20 anti-top fix (conc_gain 1.5-2.5) did essentially nothing: Exp20 moved only from -0.133 to -0.119 vs real -0.718. This strongly suggests the negative gate is already hitting its hard clip at [-h_neg, h_pos], so multiplying by conc_factor cannot amplify it — the parameter was inert at its upper range. Secondary regressions vs the accepted base: Exp17 (0.301 -> 0.270, real 0.378), Exp13 (0.039 -> 0.086, real 0.000), Exp14 (0.050 -> 0.063, real 0.028), Exp6 (0.410 -> 0.422, real 0.153), Exp2 (-0.700 -> -0.683, real -0.859), Exp7 (0.0095 -> 0.0196, real -0.010), Exp5 (0.839 -> 0.859, real 0.828). Gains were small: Exp16 (0.792 -> 0.820, real 0.839) and Exp11/Exp12 marginal. The architecture looks right and most experiments (Exp1, 4, 9, 10, 18, 8) are stable; the problem is the shape/bounds of the gate as a function of low-bloc size/homogeneity, low-evidence balance, and high-gap concentration, plus the normalization that dilutes the lone low cue.
Rationale: The iter-4 direction was rejected, so do NOT repeat 'restore h_neg to 0.30-0.50 with n_large 5.8-6.8' — that lever crashed Exp19. Build on the accepted iter-3 base with different, smaller in-family moves: (1) keep iter-3 h_neg and instead make the MAGNITUDE of the negative gate concentration-driven: remove or scale the gate's lower clamp by the concentration factor (e.g., effective lower bound -h_neg * (1 + conc_gain_p * max_gap_share^p)) so Exp20's stand-out top cue can push the anti-top weight well below -h_neg, while Exp19's diffuse high-gap profile stays near iter-3 strength. The inertness of conc_gain in iter 4 implicates the clamp or the norm; fix that rather than raising h_neg. (2) Pull Exp19 out of the negative window structurally: raise n_large to roughly 7.0-8.5 and/or require BOTH homogeneity and a n_active range for the negative term, so Exp19's configuration gets a near-zero or slightly positive gate while Exp20 still qualifies through its distinct high-gap concentration. If Exp19's lows are balanced, a small fixed positive weight in the imbalance~0 / diffuse-high case (e.g., gate ~ +0.01-0.03) plus the low branch would push the contrast toward the observed +7.84 without re-saturating to iter-2's +27. (3) Exp6 over-following: do not touch h_pos again (already tiny and the lever was rejected); instead attenuate the positive gate for heterogeneous multi-cue low blocs (stronger H_spread or phi dependence), then compensate Exp2/Exp17 with modest rho/alpha increases restricted to the balanced, homogeneous small-bloc branch — these two experiments remain the shallowest after Exp20. (4) Lone low cue (Exp15): keep lambda 8-11 but reduce dilution in the n_active==1 branch by excluding the near-zero high-cue terms from the normalization (use sum(raw[below]) as norm there), which should lift Exp15 without touching Exp16. (5) After rebalancing, re-check Exp13 and Exp14 and raise beta slightly (2.0-2.5) or tighten fallback_beta_scale (0.08-0.20) if noise-based metrics regress. Keep the no-lapse fixed-temperature softmax, the normalization scheme, and the continuous gate — all changes are within the prescribed gate family.

**Outcome of this advice:** iter 4 candidate loss=0.1360 -> iter 5 candidate loss=0.1009 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate was accepted by the gate (aggregate loss 0.1009, best so far) and it is a faithful in-family implementation of the prescribed continuous low-bloc-gated high-cue architecture with the shared low-cue machinery, no-lapse fixed-temperature softmax, and continuous gate. It genuinely fixed several previous weak points: Exp13 is back to exactly 0.000 (real 0.000), Exp15 is now 0.766 against 0.774 (previously 0.69), Exp16 is 0.874 against 0.839, and Exp1, Exp5, Exp9, Exp10, Exp11, Exp18 all remain close. However, the two signature targets the arbiter assigned this family remain qualitatively missed, and one experiment regressed badly. Exp19 is still the wrong sign: candidate -1.15 against real +7.84, with huge subject variance (12.03), indicating the negative anti-high gate never opens in the intended regime and choices sit near a flip boundary. Exp20 remains ~10-14 times too shallow: -0.053 against -0.718, so the concentration-driven anti-top amplification is essentially inert. The reason is visible in the code: conc_boost only rescales the lower clamp bound, not the gate magnitude, and the gate magnitude itself is dominated by large_win, which is nearly zero because n_large sits at 7.5-9.2 while the diagnostic low-bloc counts in these experiments are much smaller, plus low_balance and dev_damp further shrink it. So the negative term never reaches -h_neg, let alone beyond it. Secondary problems: Exp6 worsened to 0.473 against 0.153 (the positive small-bloc pathway still over-follows high cues), Exp2 remains shallow (-0.757 vs -0.859), Exp17 is low (0.297 vs 0.378), Exp14 slightly overshoots (0.067 vs 0.028), and Exp3/Exp4 drifted a bit. These are calibration and gate-shape problems, not evidence against the architecture.
Rationale: Regenerate with in-family calibrations, keeping h_neg fixed since raising it to 0.30-0.50 was the rejected lever that crashed Exp19. Priority 1: make the negative gate actually fire. (a) Apply the concentration factor to the gate magnitude, not only the clip bound: compute gate = (h_pos * small_win - h_neg * large_win * (1 + conc_boost_gain * max_gap_share ^ conc_power)) * homog * low_balance * dev_damp and keep the lower clip at -h_neg * (1 + conc_boost_gain * max_gap_share ^ conc_power); consider conc_power = 1 and conc_boost_gain in 2.0-3.5 so an Exp20 standout top cue drives the gate well below its diffuse value. (b) The large-window trigger is centered wrong for Exp19: n_large 7.5-9.2 with s_large 1.5-2.5 means large_win is near zero at the low-bloc counts that actually occur. Move n_large to roughly 3.0-5.0 with s_large 0.8-1.5, or better, replace absolute n_active with a continuous function of the diagnostic below-median cue mass relative to the high bloc as the arbiter specified, so the negative gate opens for Exp19 without re-saturating to iter-2's +27. (c) Check that Exp19's low cues are actually classified below-median and that dev_damp/low_balance are not suppressing them; consider raising D_half or relaxing dev_damp for large homogeneous balanced blocs. Priority 2: fix Exp6 without touching h_pos again. Cap branch_boost to 0.05-0.25 and raise H_spread to 0.02-0.06 so heterogeneous multi-cue low blocs lose the positive high-cue boost while homogeneous small blocs (Exp2/Exp17) keep it; if Exp2 and Exp17 dip, compensate only inside the balanced homogeneous branch with modest rho/alpha increases rather than h_pos. Priority 3: tighten fallback_beta_scale to 0.05-0.12 to bring Exp14 from 0.067 toward 0.028. Keep the n_active==1 normalization that uses sum(raw[below]) (it fixed Exp15 and Exp16), keep the no-lapse fixed-temperature softmax, and do not reintroduce the sign-flexible rank consensus or a lapse floor. After these changes, re-check Exp3, Exp4, Exp12, and Exp16; Exp16 is now slightly over (0.874 vs 0.839), so a small lambda or beta reduction may be needed once the gate is recalibrated.

**Outcome of this advice:** iter 5 candidate loss=0.1009 -> iter 6 candidate loss=0.1443 -> the gate marked it REJECTED.

### Iteration 6 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is a faithful in-family implementation of the prescribed continuous low-bloc-gated high-cue architecture, but its recalibration is badly miscalibrated and the gate rejected it: aggregate loss rose from the accepted 0.1009 (iter 5) to 0.1443. The dominant failure is Exp19: the candidate now produces -24.13 against the observed +7.84, essentially reproducing the rejected pi_11 value of -23.90. This happened because moving n_large from 7.5-9.2 down to 3.5-5.0 opened the negative large_win branch on the low-bloc counts actually present in Exp19, but with the wrong overall sign/strength. The same negative gate over-fires in Exp17, which fell from 0.297 (iter 5) to 0.234 against the real 0.378. Exp20 remains almost flat at -0.038 against -0.718: the concentration factor is present in the gate arithmetic now, but it is multiplied afterwards by low_balance and dev_damp, which are near their minimum when low evidence is strongly one-sided, so the anti-top contribution never reaches usable magnitude. Exp6 is still a large miss at 0.475 versus 0.153, showing the H_spread and branch_boost changes did not address the over-following problem. Secondary regressions from the current parameter set include Exp16 (0.790 vs 0.839 real, caused by the lambda reduction), Exp13 (0.053 vs 0.000), and mild drift in Exp4, Exp5, Exp8, Exp11 and Exp12. The one clear success is Exp14: the tightened fallback_beta_scale moved it to 0.025 against 0.028. Exp15 and Exp10 also remain good, but the losses are far larger than the gains.
Rationale: Build on the accepted iter-5 base, not on this rejected candidate. Keep the fallback_beta_scale tightening that fixed Exp14, but discard the n_large=3.5-5.0, the broad negative large_win activation, the H_spread widening, the lambda reduction and the phi reduction; those changes caused the Exp19/Exp17/Exp16 regressions. Concretely: (1) restore n_large to roughly the iter-5 range and do not let low-bloc count alone activate the negative gate. Instead gate the negative branch on high-bloc concentration--for example, a separate concentration-activation term based on max_gap_share--so Exp19's diffuse high-gap profile gets a near-zero or small positive high-cue gate and the low-cue branch can drive the observed positive contrast, while Exp20's standout top cue still receives a strong negative contribution. (2) For Exp20, stop multiplying the concentration-negative term by the full low_balance and dev_damp suppression. Those factors are near zero exactly when low evidence is one-sided and the top cue stands out; use a floor or an interaction that lets high concentration override the conflict damping, e.g. multiply the negative concentration term by something like (0.3+0.7*low_balance) and (0.3+0.7*dev_damp), or split it into an only partially damped negative channel. This is the in-family fix for why conc_boost remains inert. (3) For Exp6, do not reuse H_spread/branch_boost as the main lever since it did not move the metric. Shrink the positive small-bloc window more directly: n_small toward 0.8-1.2, s_small toward 0.10-0.25, lower h_pos toward 0.005-0.02, and cap branch_boost near zero for n_active>=2. If Exp2 and Exp17 then become too shallow, compensate only inside the balanced homogeneous small-bloc branch via a modest rho/alpha increase, not via h_pos or branch_boost. (4) Restore lambda to the higher accepted range (about 8-11) to recover Exp16 and keep Exp15, while leaving one_cue_gate_scale low so lone-cue choices do not get diluted. (5) After these changes, explicitly re-check Exp13 should return to 0.000, and Exp17 should return toward the observed 0.378 rather than being pulled negative by the large-bloc gate. Do not change mechanism family and do not reintroduce a lapse floor or sign-flexible rank consensus.

**Outcome of this advice:** iter 6 candidate loss=0.1443 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "People anchor advertised expert validities to the ensemble median and assign decision weight to below-median cues by a median/MAD-normalized deviance score. A solitary diagnostic low cue is strongly amplified, multiple diagnostic low cues are attenuated by count and deviance-selective mechanisms, and all early below-median weighting is shared across experiments. Above-median cues enter only through a continuous validity-gap gate whose positive branch uses a narrow small-bloc window and whose negative branch is activated by high-bloc validity-gap concentration, not by low-bloc count alone. The negative branch is only partially damped by low-evidence imbalance and extreme deviance, so a standout top cue can receive strong anti-following weight while a diffuse high-gap profile leaves the low-cue branch free to drive choice. Choice is a fixed-temperature softmax with no lapse floor.",
  "predict": "import numpy as np\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('GVC-continuous expects a (2, n_features) stimulus.')\n\n    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)\n    n_feat = validities.shape[0]\n    if n_feat != stim.shape[1]:\n        raise ValueError('validities length must match n_features.')\n\n    c = float(parameters['c'])\n    median = float(np.median(validities))\n    abs_dev = np.abs(validities - median)\n    mad = float(np.median(abs_dev))\n    scale = mad + c\n    if scale <= 1e-12:\n        scale = 1e-12\n\n    dev = np.maximum(0.0, median - validities) / scale\n    gap = np.maximum(0.0, validities - median) / scale\n    below = validities < median - 1e-12\n    at_or_above = ~below\n\n    diff = stim[0] - stim[1]\n    active_below = below & (np.abs(diff) > 1e-12)\n    n_active = int(np.count_nonzero(active_below))\n\n    w0 = float(parameters['w0'])\n    kappa = float(parameters['kappa'])\n    gamma = float(parameters['gamma'])\n    lam = float(parameters['lambda'])\n    sigma = float(parameters['sigma'])\n    phi = float(parameters['phi'])\n    nu = float(parameters['nu'])\n    opp_cap = float(parameters['opp_cap'])\n    focal_ratio = float(parameters['focal_ratio'])\n\n    raw = np.zeros(n_feat, dtype=float)\n    dev_bounded = np.minimum(dev, 100.0)\n    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)\n\n    def lone_amplification(idx):\n        base_amp = 1.0 + lam * np.tanh(float(dev[idx]) / sigma)\n        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)\n        n_opp = int(np.count_nonzero(opp))\n        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))\n\n    if n_active == 1:\n        focal = int(np.flatnonzero(active_below)[0])\n        raw[focal] = raw[focal] * lone_amplification(focal)\n    elif n_active >= 2:\n        base_att = 1.0 / (1.0 + phi * (n_active - 1))\n        active_devs = dev[active_below]\n        i_max = int(np.argmax(active_devs))\n        max_dev = float(active_devs[i_max])\n        med_dev = float(np.median(active_devs))\n        if med_dev > 1e-12 and max_dev > focal_ratio * med_dev:\n            focal = int(np.flatnonzero(active_below)[i_max])\n            others = below.copy()\n            others[focal] = False\n            raw[others] = raw[others] * base_att\n            raw[focal] = raw[focal] * lone_amplification(focal)\n        else:\n            raw[below] = raw[below] * base_att\n\n    beta = float(parameters['beta'])\n    fallback_beta_scale = float(parameters['fallback_beta_scale'])\n\n    if n_active == 0:\n        evidence = float(np.mean(diff))\n        beta_eff = beta * fallback_beta_scale\n    else:\n        beta_eff = beta\n        mass_active = float(np.sum(raw[active_below]))\n        E_low = float(np.dot(raw, diff))\n        imbalance = 0.0\n        if mass_active > 1e-12:\n            imbalance = min(1.0, abs(E_low) / mass_active)\n\n        K_bal = float(parameters['K_bal'])\n        low_balance = 1.0 / (1.0 + (imbalance / K_bal) ** 2.0)\n\n        H_spread = float(parameters['H_spread'])\n        if n_active >= 2:\n            spread_norm = float(np.std(validities[active_below])) / scale\n            homog = 1.0 / (1.0 + (spread_norm / H_spread) ** 4.0)\n        else:\n            homog = 1.0\n\n        D_half = float(parameters['D_half'])\n        dmax = float(np.max(dev[active_below]))\n        dev_damp = 1.0 / (1.0 + (dmax / D_half) ** 2.0)\n\n        h_pos = float(parameters['h_pos'])\n        h_neg = float(parameters['h_neg'])\n        neg_bal_floor = float(parameters['neg_bal_floor'])\n        neg_dev_floor = float(parameters['neg_dev_floor'])\n        n_small = float(parameters['n_small'])\n        s_small = float(parameters['s_small'])\n        n_large = float(parameters['n_large'])\n        s_large = float(parameters['s_large'])\n\n        z_small = (float(n_active) - n_small) / s_small\n        small_win = float(np.exp(-z_small * z_small))\n\n        z_large = (float(n_active) - n_large) / s_large\n        z_large = float(np.clip(z_large, -50.0, 50.0))\n        large_win = 1.0 / (1.0 + float(np.exp(-z_large)))\n\n        conc_boost_gain = float(parameters['conc_boost_gain'])\n        conc_power = float(parameters['conc_power'])\n        max_gap_share = 0.0\n        if np.any(at_or_above):\n            gap_above = gap[at_or_above]\n            sum_gap = float(np.sum(gap_above))\n            if sum_gap > 1e-12:\n                max_gap_share = float(np.max(gap_above)) / sum_gap\n            else:\n                max_gap_share = 0.0\n            conc_boost = 1.0 + conc_boost_gain * (max_gap_share ** conc_power)\n        else:\n            conc_boost = 1.0\n\n        # Concentration-activated negative gate.  The low-bloc-count window\n        # only moderates the negative branch instead of gating it on its own.\n        conc_act = max_gap_share ** conc_power\n        conc_mult = 1.0 + conc_boost_gain * conc_act\n        neg_count_factor = 0.35 + 0.65 * large_win\n        neg_damp_bal = neg_bal_floor + (1.0 - neg_bal_floor) * low_balance\n        neg_damp_dev = neg_dev_floor + (1.0 - neg_dev_floor) * dev_damp\n\n        gate = (h_pos * small_win * homog * low_balance * dev_damp\n                - h_neg * homog * conc_act * conc_mult * neg_count_factor\n                * neg_damp_bal * neg_damp_dev)\n        if n_active == 1:\n            gate = gate * float(parameters['one_cue_gate_scale'])\n        lower_bound = -h_neg * conc_boost\n        gate = float(np.clip(gate, lower_bound, h_pos))\n\n        rho = float(parameters['rho'])\n        alpha = float(parameters['alpha'])\n        branch_boost = float(parameters['branch_boost'])\n        hi_boost = 1.0\n        if n_active >= 2:\n            hi_boost = 1.0 + branch_boost * small_win * homog * low_balance\n\n        hi_pre = np.zeros(n_feat, dtype=float)\n        gap_bounded = np.minimum(gap, 100.0)\n        hi_pre[at_or_above] = rho * hi_boost * np.power(gap_bounded[at_or_above], alpha)\n\n        raw_high = gate * hi_pre\n        raw_all = raw + raw_high\n        if n_active == 1:\n            norm = float(np.sum(raw[below]))\n        else:\n            norm = float(np.sum(raw[below])) + float(np.sum(np.abs(raw_high[at_or_above])))\n        if norm <= 1e-12:\n            evidence = float(np.mean(diff))\n        else:\n            evidence = float(np.dot(raw_all, diff) / norm)\n\n    logits = np.array([evidence, 0.0], dtype=float)\n    z = beta_eff * (logits - float(np.max(logits)))\n    z = np.clip(z, -60.0, 60.0)\n    e = np.exp(z)\n    probs = e / np.sum(e)\n    return probs\n",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=float)\n    probs = probs / probs.sum()\n    return int(np.random.choice(probs.shape[0], p=probs))\n",
  "parameters": {
    "c": "[0.008, 0.012]",
    "w0": "[0.007, 0.012]",
    "kappa": "[2.30, 2.80]",
    "gamma": "[1.25, 1.45]",
    "lambda": "[8.0, 11.0]",
    "sigma": "[0.14, 0.22]",
    "phi": "[0.38, 0.55]",
    "nu": "[0.15, 0.25]",
    "opp_cap": "{5}",
    "focal_ratio": "[3.0, 4.0]",
    "rho": "[0.08, 0.14]",
    "alpha": "[1.1, 1.8]",
    "h_pos": "[0.005, 0.02]",
    "h_neg": "[0.12, 0.20]",
    "neg_bal_floor": "[0.25, 0.45]",
    "neg_dev_floor": "[0.25, 0.45]",
    "n_small": "[0.8, 1.2]",
    "s_small": "[0.10, 0.25]",
    "n_large": "[7.5, 9.2]",
    "s_large": "[1.5, 2.5]",
    "K_bal": "[0.40, 0.65]",
    "H_spread": "[0.003, 0.012]",
    "D_half": "[0.40, 0.65]",
    "beta": "[1.8, 2.3]",
    "fallback_beta_scale": "[0.05, 0.12]",
    "one_cue_gate_scale": "[0.15, 0.35]",
    "conc_boost_gain": "[2.5, 4.5]",
    "conc_power": "{2}",
    "branch_boost": "[0.0, 0.08]",
    "validities": "validities"
  },
  "rationale": "This is a minimal in-family edit of the accepted iter-5 continuous-gate base applying the iter-6 feedback without repeating the rejected iter-6 parameter moves. (1) The negative gate no longer fires from low-bloc count alone: n_large/s_large are restored near the iter-5 ranges and now enter only through a count moderating factor, while the main activation is high-bloc validity-gap concentration via max_gap_share^conc_power. Exp19's diffuse high-gap profile therefore gets a near-zero high-cue gate and lets the low-cue branch produce the positive observed contrast, whereas Exp20's standout top cue activates a strong anti-top contribution. (2) The concentration factor is multiplied directly into the gate magnitude, not just the lower clip, and the negative channel is only partially damped by low_balance/dev_damp through neg_bal_floor and neg_dev_floor, fixing the iter-5 inertness. (3) The positive small-bloc window is tightened exactly as advised (n_small 0.8-1.2, s_small 0.10-0.25, h_pos 0.005-0.02, branch_boost near zero) to reduce Exp6 over-following. (4) lambda is restored to 8-11 for Exp16/Exp15, while the n_active==1 normalization and low one_cue_gate_scale are preserved. (5) fallback_beta_scale is tightened to 0.05-0.12 to keep the Exp14 gain. No lapse floor, no sign-flexible rank consensus, and no mechanism-family switch are introduced."
}
```

## Usage

```json
{
  "input_tokens": 55853,
  "output_tokens": 29317,
  "reasoning_tokens": 26031,
  "provider": "AtlasCloud"
}
```
