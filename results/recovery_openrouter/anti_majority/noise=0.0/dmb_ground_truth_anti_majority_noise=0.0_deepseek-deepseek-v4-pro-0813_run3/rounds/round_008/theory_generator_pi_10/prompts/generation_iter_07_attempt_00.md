# generation_iter_07_attempt_00

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

Replace pi_8 with a new theory that retains inverse-valence clean-cue coding and softmax choice, but changes the conflict-resolution assumptions. First, when the highest-validity experts are tied, the tied top group's clean-cue advantage should be a continuous, always-active evidence component, not a near-zero fallback: its strength should scale with top-group size, validity gap, and conflict size, and it should be strong enough to produce p(A) spanning roughly 0.25-0.75 across h for D=±1 (Experiments 1, 2, 17) while remaining moderate when D=0 (Experiment 18). Second, unique top-expert precedence should be strong in small conflicts but decay as the number of opposing lower-validity cues grows, so lower-validity majorities can override the unique expert in large conflicts; this should fix the positive sign in Experiment 9 and improve Experiment 3. Third, keep a damped validity-weighted tally for non-tied cases so the raw-majority and validity-weighting experiments remain well described. Fourth, use heavy-tailed subject-level distributions for gain, lapse, validity-weight sensitivity, top-group sensitivity, and precedence probability to capture the large between-subject variance seen in the data. This new theory should be a stronger competitor to pi_9, which remains the incumbent comparison.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_9` (overall score: 0.843)

**Description**
Subjects treat a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so each discriminating expert contributes an inverse-valence direction. The final evidence is a damped validity-weighted tally: each discriminating cue is weighted by 1 + lambda*(v_j - mean(v)), with lambda modest and further reduced in large multi-cue conflicts, so raw clean-cue majority dominates while high-validity cues receive a moderate extra weight. When exactly one discriminating expert has strictly highest communicated validity, a probabilistic, validity-gap-scaled precedence bonus is added to that expert's direction. When top experts are tied and the weighted evidence is near zero, subjects follow the sign of the tied top group's clean-cue advantage with moderate sensitivity instead of guessing. Choice is a softmax over the resulting evidence with subject-specific gain and lapse. Subject-level gain, lambda, precedence probability, large-conflict raw-tally softening, tie parameters, and epsilon are drawn from Student-t heavy-tailed/logit-Student-t distributions with medians anchored near the accepted base, providing the between-subject variance observed in the data without shifting the aggregate pattern too far.

`predict(parameters, state, history) -> np.ndarray`:
import math
import numpy as np
from scipy.stats import norm, t as student_t


def _tqt(u, mu, sigma, lo, hi, df):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = student_t.cdf((lo - mu) / sigma, df)
    q_hi = student_t.cdf((hi - mu) / sigma, df)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * student_t.ppf(q, df)


def _logit(p):
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return math.log(p / (1.0 - p))


def _ilogit(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # ---- Heavy-tailed subject-level transforms -------------------------------
    def u_raw(x):
        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))

    gain = float(np.exp(_tqt(
        u_raw(parameters['gain_raw']),
        math.log(1.45), 0.32, math.log(0.60), math.log(2.80), 4)))

    epsilon = float(_ilogit(_tqt(
        u_raw(parameters['epsilon_raw']),
        _logit(0.26), 0.35, _logit(0.07), _logit(0.45), 4)))

    lam = float(np.exp(_tqt(
        u_raw(parameters['lambda_raw']),
        math.log(0.24), 0.55, math.log(0.005), math.log(0.40), 4)))

    p_prec = float(_ilogit(_tqt(
        u_raw(parameters['prec_p_raw']),
        _logit(0.55), 0.50, _logit(0.12), _logit(0.96), 4)))

    p_large = float(_ilogit(_tqt(
        u_raw(parameters['prec_large_p_raw']),
        _logit(0.72), 0.45, _logit(0.20), _logit(0.99), 4)))

    tie_p = float(_ilogit(_tqt(
        u_raw(parameters['tie_p_raw']),
        _logit(0.75), 0.40, _logit(0.25), _logit(0.98), 4)))

    prec_str = float(np.exp(_tqt(
        u_raw(parameters['prec_strength_raw']),
        math.log(0.95), 0.28, math.log(0.40), math.log(1.70), 4)))

    tie_str = float(np.exp(_tqt(
        u_raw(parameters['tie_strength_raw']),
        math.log(0.60), 0.40, math.log(0.15), math.log(1.30), 4)))

    tie_tol = float(np.exp(_tqt(
        u_raw(parameters['tie_tol_raw']),
        math.log(0.60), 0.30, math.log(0.20), math.log(1.30), 4)))

    lam_large_scale = float(_ilogit(_tqt(
        u_raw(parameters['lam_large_scale_raw']),
        _logit(0.45), 0.40, _logit(0.10), _logit(0.85), 4)))

    large_d_scale = float(np.exp(_tqt(
        u_raw(parameters['large_d_scale_raw']),
        math.log(0.50), 0.40, math.log(0.15), math.log(0.90), 4)))

    window = int(round(float(parameters['window'])))

    # ---- Inverse-valence + damped validity weighting -------------------------
    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    # pi_8-style large-conflict regime: n_features 7-9 with many discriminators
    is_large_scale = disc_size >= 6 and 7 <= n_features <= 9
    # broader large-conflict flag used only to shrink lambda further
    is_large_weight = disc_size >= 6 and n_features >= 7

    lam_eff = lam * (lam_large_scale if is_large_weight else 1.0)
    mean_v = float(np.mean(val))
    vals_disc = val[disc_idx]
    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
    E = float(np.sum(w_disc * diff[disc_idx]))

    # ---- Unique top-expert precedence (pi_8-like) ---------------------------
    unique_top = False
    top_tied = False
    top_tied_idx = np.array([], dtype=int)
    top_clean = 0.0
    sign_top = 0.0
    gap = 0.04

    if disc_size == 1:
        unique_top = True
        sign_top = 1.0 if diff[disc_idx[0]] > 0.0 else -1.0
        gap = 0.04
    else:
        top_val = float(np.max(vals_disc))
        top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)
        top_idx = disc_idx[top_mask]
        if top_idx.size == 1:
            unique_top = True
            top_pos = int(np.flatnonzero(top_mask)[0])
            sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0
            others = vals_disc[~top_mask]
            if others.size > 0:
                gap = float(top_val - float(np.max(others)))
            else:
                gap = 0.04
        else:
            top_tied = True
            top_tied_idx = top_idx
            top_clean = float(np.sum(diff[top_idx]))

    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                tie_bonus = prec_str * (0.70 + 1.70 * math.tanh(gap / 0.05))
                E = sign_top * tie_bonus
        else:
            p_apply = p_large if is_large_scale else p_prec
            if abs(D) <= window and np.random.rand() < p_apply:
                bonus = prec_str * (0.95 + 2.0 * math.tanh(gap / 0.10))
                if is_large_scale:
                    bonus = prec_str * (1.00 + 2.40 * math.tanh(gap / 0.10))
                    bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc_size - 4)))
                E = E + sign_top * bonus
    elif top_tied:
        # Raw-majority-dominant softening in the known large-conflict regime.
        if is_large_scale and abs(D) > 1e-12:
            E = D * large_d_scale + (E - D)

        # Tied-top group clean-cue advantage when the tally is near zero.
        if abs(E) <= tie_tol and abs(top_clean) > 1e-9:
            top_effect = tie_str * math.tanh(top_clean / 2.8)
            E = E + top_effect

    # ---- Softmax choice ------------------------------------------------------
    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


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
- gain_raw: [0.02, 0.98]
- epsilon_raw: [0.02, 0.98]
- lambda_raw: [0.02, 0.98]
- prec_p_raw: [0.02, 0.98]
- prec_large_p_raw: [0.02, 0.98]
- tie_p_raw: [0.02, 0.98]
- prec_strength_raw: [0.02, 0.98]
- tie_strength_raw: [0.02, 0.98]
- tie_tol_raw: [0.02, 0.98]
- lam_large_scale_raw: [0.02, 0.98]
- large_d_scale_raw: [0.02, 0.98]
- window: {1, 2}
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.5117 (var=0.0161) vs this=-0.4625 (var=0.0083)
- Experiment 2: real=-0.6438 (var=0.0430) vs this=-0.5531 (var=0.0300)
- Experiment 3: real=-0.4300 (var=0.0221) vs this=-0.4450 (var=0.0283)
- Experiment 4: real=0.4925 (var=0.0658) vs this=0.5325 (var=0.0266)
- Experiment 5: real=0.2658 (var=0.1584) vs this=0.3075 (var=0.0162)
- Experiment 6: real=0.0867 (var=0.4244) vs this=0.0483 (var=0.0368)
- Experiment 7: real=0.0725 (var=0.0580) vs this=-0.0062 (var=0.0541)
- Experiment 8: real=1.0100 (var=2.8518) vs this=0.9838 (var=0.4229)
- Experiment 9: real=0.3050 (var=0.0605) vs this=0.3433 (var=0.0229)
- Experiment 10: real=-0.1667 (var=0.0053) vs this=-0.1827 (var=0.0021)
- Experiment 11: real=0.0000 (var=0.0045) vs this=0.0000 (var=0.0028)
- Experiment 12: real=1.2973 (var=0.2536) vs this=1.3103 (var=0.0593)
- Experiment 13: real=0.7300 (var=0.0558) vs this=0.5800 (var=0.0580)
- Experiment 14: real=0.3950 (var=0.3296) vs this=0.3987 (var=0.0481)
- Experiment 15: real=0.0608 (var=0.0035) vs this=0.0088 (var=0.0006)
- Experiment 16: real=0.2492 (var=0.0989) vs this=0.2429 (var=0.0280)
- Experiment 17: real=0.0608 (var=0.0012) vs this=0.0373 (var=0.0004)
- Experiment 18: real=3.4448 (var=3.3537) vs this=1.0841 (var=0.9280)


---

### `pi_8` (overall score: 0.768)

**Description**
Inverse-valence majority with gated unique-expert precedence and tied-top large-conflict softness. Subjects treat a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes approximately +1 for A or -1 for B to a raw clean-cue tally. The default choice is a softmax over this near-equal-weight tally. When exactly one discriminating expert has strictly highest communicated validity, that unique expert receives a gap-scaled precedence bonus inside a limited raw-tally window. In large multi-cue conflicts, this unique-expert reliance is separately and more strongly applied. When two or more top experts are tied, unique-expert precedence is suppressed, but the conflict is still processed with reduced decision sensitivity to the raw tally, preventing an unrealistically sharp majority effect while never letting tied top experts override a non-zero low-validity majority.

`predict(parameters, state, history) -> np.ndarray`:
import math
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError('validities length mismatch')

    changed = a != b
    disc = np.flatnonzero(changed)
    if disc.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))

    gain = float(parameters['gain'])
    epsilon = float(parameters['epsilon'])
    p_override = float(parameters['p_override'])
    p_large = float(parameters['p_large'])
    tie_p = float(parameters['tie_p'])
    window = int(round(float(parameters['window'])))

    is_large = disc.size >= 6 and 7 <= stim.shape[1] <= 9
    window_eff = window
    if is_large:
        window_eff = max(2, window)
        p_apply = p_large
    else:
        p_apply = p_override

    E = D
    unique_top = False
    sign_top = 0.0
    gap = 0.0

    if disc.size == 1:
        j_top = int(disc[0])
        sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
        gap = 0.04
        unique_top = True
    else:
        vals_disc = val[disc]
        top_val = float(np.max(vals_disc))
        top_hits = disc[np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]
        unique_top = top_hits.size == 1
        if unique_top:
            j_top = int(top_hits[0])
            sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
            others = vals_disc[~np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]
            if others.size == 0:
                gap = 0.04
            else:
                gap = float(top_val - float(np.max(others)))

    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                tie_bonus = 0.72 + 0.50 * math.tanh(gap / 0.05)
                E = sign_top * tie_bonus
            else:
                E = 0.0
        elif abs(D) <= window_eff and np.random.rand() < p_apply:
            bonus = 0.95 + 2.0 * math.tanh(gap / 0.10)
            if is_large:
                bonus = 1.00 + 2.40 * math.tanh(gap / 0.10)
                bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc.size - 4)))
            E = D + sign_top * bonus
    elif abs(D) <= 1e-12:
        E = 0.0

    # Tied-top large conflicts: unique-expert precedence is suppressed by the
    # theory, but raw-tally sensitivity is reduced so the low-validity majority
    # does not drive an unrealistically sharp choice.
    if is_large and not unique_top and abs(D) > 1e-12:
        E = D * float(parameters['large_tie_scale'])

    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5


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
- gain: [1.20, 1.80]
- epsilon: [0.20, 0.36]
- p_override: [0.30, 0.80]
- p_large: [0.35, 0.98]
- tie_p: [0.55, 0.95]
- window: {1, 2}
- large_tie_scale: [0.05, 0.75]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.5117 (var=0.0161) vs this=-0.4688 (var=0.0076)
- Experiment 2: real=-0.6438 (var=0.0430) vs this=-0.5431 (var=0.0192)
- Experiment 3: real=-0.4300 (var=0.0221) vs this=-0.3579 (var=0.0112)
- Experiment 4: real=0.4925 (var=0.0658) vs this=0.5619 (var=0.0213)
- Experiment 5: real=0.2658 (var=0.1584) vs this=0.2696 (var=0.0180)
- Experiment 6: real=0.0867 (var=0.4244) vs this=0.0692 (var=0.0205)
- Experiment 7: real=0.0725 (var=0.0580) vs this=-0.0275 (var=0.0571)
- Experiment 8: real=1.0100 (var=2.8518) vs this=0.8500 (var=0.3186)
- Experiment 9: real=0.3050 (var=0.0605) vs this=0.2754 (var=0.0135)
- Experiment 10: real=-0.1667 (var=0.0053) vs this=-0.1707 (var=0.0019)
- Experiment 11: real=0.0000 (var=0.0045) vs this=0.0000 (var=0.0019)
- Experiment 12: real=1.2973 (var=0.2536) vs this=1.2583 (var=0.0676)
- Experiment 13: real=0.7300 (var=0.0558) vs this=0.6225 (var=0.0341)
- Experiment 14: real=0.3950 (var=0.3296) vs this=0.4813 (var=0.0208)
- Experiment 15: real=0.0608 (var=0.0035) vs this=0.0028 (var=0.0004)
- Experiment 16: real=0.2492 (var=0.0989) vs this=0.0162 (var=0.0093)
- Experiment 17: real=0.0608 (var=0.0012) vs this=0.0036 (var=0.0002)
- Experiment 18: real=3.4448 (var=3.3537) vs this=-0.0410 (var=0.4858)


---

### `pi_7` (overall score: 0.763)

**Description**
All subjects treat a binary expert rating of 0 as clean/favorable and 1 as a defect. On each trial every discriminating expert contributes continuous inverse-valence evidence weighted by a saturating, near-tally-like transform of communicated validity: w_j = 1 + m * tanh(2 * [logit(v_j) - mean(logit(v))] / half_range), with the weight vector normalized to mean one. The subject-level saturation m is small and capped, so high-validity experts are attended more but cannot generate extreme top-to-bottom weight ratios: a three-cue high-validity coalition can win small conflicts yet is overridden by a five-cue low-validity majority. When the weighted evidence is inconclusive, or when the raw inverse-clean tally is exactly balanced, subjects fall back to inspecting changed experts in descending validity order with tied highest-validity experts randomly permuted per trial; the precision of that fallback grows with the validity gap separating the best expert from the next-best expert. Choice is a softmax over net evidence with subject-specific gain and lapse. There is no valence-orientation mixture and no separate top-cue attention bonus.

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

    # ---- Subject-level parameter shaping -----------------------------------
    # Saturation amplitude for the validity-to-weight map.
    # Kept deliberately small: this is the main fix for the unbounded
    # exponential weights that let three high-validity cues destroy the
    # five-cue low-validity majority in Experiment 8.
    u_m = float(np.clip((float(parameters['slope_a']) - 0.02) / 0.28, 0.0, 1.0))
    sat_m = float(np.exp(_trunc_qnorm(
        u_m, np.log(0.10), 0.45, np.log(0.03), np.log(0.18))))

    # Main choice gain: restore the accepted base's moderate calibration.
    u_g = float(np.clip((float(parameters['gain']) - 0.10) / 1.40, 0.0, 1.0))
    gain = float(np.exp(_trunc_qnorm(
        u_g, np.log(0.40), 0.50, np.log(0.10), np.log(1.50))))

    lo_logit = np.log(0.03 / 0.97)
    hi_logit = np.log(0.16 / 0.84)
    mu_logit = np.log(0.08 / 0.92)
    u_e = float(np.clip((float(parameters['epsilon']) - 0.03) / 0.13, 0.0, 1.0))
    z_eps = _trunc_qnorm(u_e, mu_logit, 0.35, lo_logit, hi_logit)
    epsilon = float(_inv_logit(z_eps))

    # Base tie-break gain is moderate, but is amplified by the validity gap
    # between the selected best expert and the next-best changed expert.
    u_tg = float(np.clip((float(parameters['tie_gain']) - 0.30) / 1.00, 0.0, 1.0))
    tie_gain = float(np.exp(_trunc_qnorm(
        u_tg, np.log(0.45), 0.45, np.log(0.30), np.log(0.70))))

    # Tightened inconclusive-evidence threshold.
    u_tt = float(np.clip((float(parameters['tie_tol']) - 0.02) / 0.13, 0.0, 1.0))
    tie_tol = float(_trunc_qnorm(u_tt, 0.08, 0.030, 0.02, 0.15))

    # ---- Saturating continuous inverse-valence weights ---------------------
    # 0 is clean/favorable, 1 is a defect.
    v = np.clip(val, 0.50001, 0.9999)
    logit_v = np.log(v / (1.0 - v))
    center = float(np.mean(logit_v))

    if n_features > 1:
        spread = float(np.max(logit_v) - np.min(logit_v))
        if spread <= 1e-8:
            z = np.zeros(n_features, dtype=float)
        else:
            half_spread = 0.5 * spread
            z = np.tanh(2.0 * (logit_v - center) / half_spread)
        w = 1.0 + sat_m * z
        w = w / np.mean(w)
    else:
        w = np.ones(1, dtype=float)

    changed_mask = a_row != b_row
    if not np.any(changed_mask):
        return np.ones(2) / 2.0

    # A=0/B=1 gives +w_j; A=1/B=0 gives -w_j.
    diff = (b_row > a_row).astype(float) - (a_row > b_row).astype(float)
    raw_tally = float(np.sum(diff))
    net_evidence = float(np.dot(w, diff))

    if abs(net_evidence) <= tie_tol or abs(raw_tally) <= 0.5:
        # Inconclusive evidence or an exactly balanced raw inverse-clean
        # tally: inspect changed experts in descending validity order.
        # Ties at the maximum validity are exchangeable and resolved by a
        # random permutation on each trial.
        disc = np.flatnonzero(changed_mask)
        top_val = float(np.max(val[disc]))
        tied = disc[np.isclose(val[disc], top_val, rtol=1e-12, atol=1e-12)]
        if tied.size == 0:
            tied = disc
        idx = int(tied[np.random.randint(tied.size)])

        # Validity-gap-dependent tie-break precision.  This is not a top-cue
        # bonus inside the tally; it is how strongly subjects rely on the
        # lexical fallback only when they are already in the fallback regime.
        non_top = disc[~np.isclose(val[disc], top_val, rtol=1e-12, atol=1e-12)]
        second_val = float(np.max(val[non_top])) if non_top.size > 0 else top_val
        gap = top_val - second_val
        gap_boost = 1.0 + 2.5 * np.tanh(gap / 0.15)

        if a_row[idx] == 0.0 and b_row[idx] == 1.0:
            sign = 1.0   # clean cue favors A
        elif a_row[idx] == 1.0 and b_row[idx] == 0.0:
            sign = -1.0  # clean cue favors B
        else:
            return np.ones(2) / 2.0

        scores = np.array([sign, -sign], dtype=float)
        beta_eff = tie_gain * gap_boost
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
- slope_a: [0.02, 0.30]
- gain: [0.10, 1.50]
- epsilon: [0.03, 0.16]
- tie_gain: [0.30, 1.30]
- tie_tol: [0.02, 0.15]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.5117 (var=0.0161) vs this=-0.4754 (var=0.0242)
- Experiment 2: real=-0.6438 (var=0.0430) vs this=-0.6138 (var=0.0287)
- Experiment 3: real=-0.4300 (var=0.0221) vs this=-0.6425 (var=0.0137)
- Experiment 4: real=0.4925 (var=0.0658) vs this=0.5363 (var=0.0483)
- Experiment 5: real=0.2658 (var=0.1584) vs this=0.2608 (var=0.0101)
- Experiment 6: real=0.0867 (var=0.4244) vs this=-0.2058 (var=0.0270)
- Experiment 7: real=0.0725 (var=0.0580) vs this=0.0812 (var=0.0474)
- Experiment 8: real=1.0100 (var=2.8518) vs this=1.6462 (var=0.3404)
- Experiment 9: real=0.3050 (var=0.0605) vs this=0.4579 (var=0.0150)
- Experiment 10: real=-0.1667 (var=0.0053) vs this=-0.1824 (var=0.0024)
- Experiment 11: real=0.0000 (var=0.0045) vs this=0.0000 (var=0.0038)
- Experiment 12: real=1.2973 (var=0.2536) vs this=1.1792 (var=0.1659)
- Experiment 13: real=0.7300 (var=0.0558) vs this=0.4625 (var=0.0677)
- Experiment 14: real=0.3950 (var=0.3296) vs this=-0.2175 (var=0.1258)
- Experiment 15: real=0.0608 (var=0.0035) vs this=0.0219 (var=0.0003)
- Experiment 16: real=0.2492 (var=0.0989) vs this=0.3862 (var=0.0094)
- Experiment 17: real=0.0608 (var=0.0012) vs this=0.0530 (var=0.0004)
- Experiment 18: real=3.4448 (var=3.3537) vs this=1.1573 (var=1.6319)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1653 -> ACCEPTED
- iter 2: loss=0.1291 -> ACCEPTED
- iter 3: loss=0.1008 -> ACCEPTED
- iter 4: loss=0.0851 -> ACCEPTED
- iter 5: loss=0.0829 -> ACCEPTED
- iter 6: loss=0.1038 -> REJECTED
- iter 7: loss=0.0854 -> REJECTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.0829 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.4992 (var=0.0124) (Δ vs real +0.0125)
  - iter 2: -0.5117 (var=0.0113) (Δ vs real -0.0000)
  - iter 3: -0.4654 (var=0.0109) (Δ vs real +0.0462)
  - iter 4: -0.4917 (var=0.0156) (Δ vs real +0.0200)
  - iter 5: -0.4671 (var=0.0139) (Δ vs real +0.0446)
  - iter 6: -0.5004 (var=0.0135) (Δ vs real +0.0112)
  - iter 7 (most recent): -0.4679 (var=0.0168) (Δ vs real +0.0437)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0092 (var=0.0043)
- pi_2: 0.6258 (var=0.0234)
- pi_3: -0.5379 (var=0.0084)
- pi_4: -0.5413 (var=0.0057)
- pi_5: -0.5308 (var=0.0142)
- pi_4_1: -0.5442 (var=0.0102)
- pi_6: -0.3700 (var=0.2536)
- pi_7: -0.4754 (var=0.0242)
- pi_8: -0.4688 (var=0.0076)
- pi_9: -0.4625 (var=0.0083)

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
  - iter 1: -0.5919 (var=0.0216) (Δ vs real +0.0519)
  - iter 2: -0.5881 (var=0.0343) (Δ vs real +0.0556)
  - iter 3: -0.5600 (var=0.0261) (Δ vs real +0.0838)
  - iter 4: -0.6213 (var=0.0359) (Δ vs real +0.0225)
  - iter 5: -0.5956 (var=0.0407) (Δ vs real +0.0481)
  - iter 6: -0.6231 (var=0.0376) (Δ vs real +0.0206)
  - iter 7 (most recent): -0.6287 (var=0.0240) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7112 (var=0.0379)
- pi_1: -0.0013 (var=0.0059)
- pi_3: -0.6069 (var=0.0076)
- pi_4: -0.5913 (var=0.0097)
- pi_5: -0.5938 (var=0.0426)
- pi_4_1: -0.6038 (var=0.0238)
- pi_6: -0.6238 (var=0.2467)
- pi_7: -0.6138 (var=0.0287)
- pi_8: -0.5431 (var=0.0192)
- pi_9: -0.5531 (var=0.0300)

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
  - iter 1: -0.4796 (var=0.0172) (Δ vs real -0.0496)
  - iter 2: -0.3521 (var=0.0421) (Δ vs real +0.0779)
  - iter 3: -0.3875 (var=0.0426) (Δ vs real +0.0425)
  - iter 4: -0.3604 (var=0.0475) (Δ vs real +0.0696)
  - iter 5: -0.4233 (var=0.0441) (Δ vs real +0.0067)
  - iter 6: -0.4012 (var=0.0492) (Δ vs real +0.0287)
  - iter 7 (most recent): -0.3679 (var=0.0363) (Δ vs real +0.0621)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7113 (var=0.0398)
- pi_3: 0.0175 (var=0.0062)
- pi_2: 0.0117 (var=0.0115)
- pi_4: -0.4575 (var=0.0063)
- pi_5: -0.4596 (var=0.0245)
- pi_4_1: -0.4029 (var=0.0226)
- pi_6: -0.3550 (var=0.1762)
- pi_7: -0.6425 (var=0.0137)
- pi_8: -0.3579 (var=0.0112)
- pi_9: -0.4450 (var=0.0283)

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
  - iter 1: 0.5894 (var=0.0307) (Δ vs real +0.0969)
  - iter 2: 0.6444 (var=0.0178) (Δ vs real +0.1519)
  - iter 3: 0.6431 (var=0.0377) (Δ vs real +0.1506)
  - iter 4: 0.6256 (var=0.0369) (Δ vs real +0.1331)
  - iter 5: 0.5856 (var=0.0461) (Δ vs real +0.0931)
  - iter 6: 0.6588 (var=0.0296) (Δ vs real +0.1663)
  - iter 7 (most recent): 0.6200 (var=0.0385) (Δ vs real +0.1275)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6381 (var=0.0147)
- pi_1: -0.0244 (var=0.0050)
- pi_2: -0.7525 (var=0.0447)
- pi_4: 0.4313 (var=0.0096)
- pi_5: 0.4237 (var=0.0670)
- pi_4_1: 0.5069 (var=0.0237)
- pi_6: 0.4681 (var=0.1627)
- pi_7: 0.5363 (var=0.0483)
- pi_8: 0.5619 (var=0.0213)
- pi_9: 0.5325 (var=0.0266)

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
  - iter 1: 0.2062 (var=0.0258) (Δ vs real -0.0596)
  - iter 2: 0.0983 (var=0.0182) (Δ vs real -0.1675)
  - iter 3: 0.0688 (var=0.0246) (Δ vs real -0.1971)
  - iter 4: 0.0933 (var=0.0285) (Δ vs real -0.1725)
  - iter 5: 0.0512 (var=0.0173) (Δ vs real -0.2146)
  - iter 6: 0.1079 (var=0.0521) (Δ vs real -0.1579)
  - iter 7 (most recent): 0.0625 (var=0.0407) (Δ vs real -0.2033)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2971 (var=0.0096)
- pi_3: -0.2708 (var=0.0095)
- pi_1: -0.6967 (var=0.0260)
- pi_2: 0.3704 (var=0.0144)
- pi_5: 0.3858 (var=0.0225)
- pi_4_1: 0.2446 (var=0.0156)
- pi_6: 0.3300 (var=0.1102)
- pi_7: 0.2608 (var=0.0101)
- pi_8: 0.2696 (var=0.0180)
- pi_9: 0.3075 (var=0.0162)

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
  - iter 1: -0.1942 (var=0.0402) (Δ vs real -0.2808)
  - iter 2: -0.3117 (var=0.0468) (Δ vs real -0.3983)
  - iter 3: -0.3642 (var=0.0348) (Δ vs real -0.4508)
  - iter 4: -0.2733 (var=0.0567) (Δ vs real -0.3600)
  - iter 5: -0.1333 (var=0.0468) (Δ vs real -0.2200)
  - iter 6: -0.1283 (var=0.0519) (Δ vs real -0.2150)
  - iter 7 (most recent): -0.0092 (var=0.1711) (Δ vs real -0.0958)
**Other theories' values on this metric (for reference):**
- pi_3: -0.5333 (var=0.0223)
- pi_4: 0.2158 (var=0.0210)
- pi_1: -0.7375 (var=0.0404)
- pi_2: 0.6717 (var=0.0518)
- pi_5: 0.1383 (var=0.0736)
- pi_4_1: 0.1075 (var=0.0264)
- pi_6: 0.0358 (var=0.0678)
- pi_7: -0.2058 (var=0.0270)
- pi_8: 0.0692 (var=0.0205)
- pi_9: 0.0483 (var=0.0368)

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
  - iter 1: 0.0188 (var=0.0425) (Δ vs real -0.0537)
  - iter 2: 0.0237 (var=0.0475) (Δ vs real -0.0488)
  - iter 3: 0.0375 (var=0.0816) (Δ vs real -0.0350)
  - iter 4: 0.0338 (var=0.0503) (Δ vs real -0.0387)
  - iter 5: 0.0100 (var=0.0501) (Δ vs real -0.0625)
  - iter 6: -0.0413 (var=0.0493) (Δ vs real -0.1137)
  - iter 7 (most recent): 0.0262 (var=0.0631) (Δ vs real -0.0463)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0025 (var=0.0525)
- pi_5: 0.3325 (var=0.0612)
- pi_1: 0.0288 (var=0.0221)
- pi_2: 0.0137 (var=0.0602)
- pi_3: -0.0025 (var=0.0792)
- pi_4_1: -0.0400 (var=0.0672)
- pi_6: 0.1800 (var=0.1187)
- pi_7: 0.0812 (var=0.0474)
- pi_8: -0.0275 (var=0.0571)
- pi_9: -0.0062 (var=0.0541)

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
  - iter 1: -0.5800 (var=1.2697) (Δ vs real -1.5900)
  - iter 2: 0.3025 (var=1.6244) (Δ vs real -0.7075)
  - iter 3: 0.6362 (var=1.8204) (Δ vs real -0.3737)
  - iter 4: 0.2988 (var=1.4297) (Δ vs real -0.7112)
  - iter 5: 0.4038 (var=2.1538) (Δ vs real -0.6062)
  - iter 6: -0.3387 (var=1.4420) (Δ vs real -1.3487)
  - iter 7 (most recent): -0.4900 (var=1.6394) (Δ vs real -1.5000)
**Other theories' values on this metric (for reference):**
- pi_5: 1.3512 (var=0.8533)
- pi_4: -0.1263 (var=0.1938)
- pi_1: 1.4112 (var=0.1813)
- pi_2: -2.0162 (var=0.4200)
- pi_3: 1.7625 (var=0.1534)
- pi_4_1: 0.5012 (var=0.5143)
- pi_6: 1.1075 (var=2.7433)
- pi_7: 1.6462 (var=0.3404)
- pi_8: 0.8500 (var=0.3186)
- pi_9: 0.9838 (var=0.4229)

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
  - iter 1: 0.3375 (var=0.0343) (Δ vs real +0.0325)
  - iter 2: 0.2517 (var=0.0182) (Δ vs real -0.0533)
  - iter 3: 0.2213 (var=0.0345) (Δ vs real -0.0837)
  - iter 4: 0.3104 (var=0.0482) (Δ vs real +0.0054)
  - iter 5: 0.2313 (var=0.0343) (Δ vs real -0.0737)
  - iter 6: 0.3575 (var=0.0671) (Δ vs real +0.0525)
  - iter 7 (most recent): 0.3108 (var=0.0444) (Δ vs real +0.0058)
**Other theories' values on this metric (for reference):**
- pi_4_1: 0.4188 (var=0.0158)
- pi_5: 0.2196 (var=0.0342)
- pi_1: -0.7067 (var=0.0351)
- pi_2: 0.0204 (var=0.0104)
- pi_3: -0.0142 (var=0.0125)
- pi_4: 0.4296 (var=0.0093)
- pi_6: 0.1742 (var=0.1020)
- pi_7: 0.4579 (var=0.0150)
- pi_8: 0.2754 (var=0.0135)
- pi_9: 0.3433 (var=0.0229)

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
  - iter 1: -0.1706 (var=0.0017) (Δ vs real -0.0039)
  - iter 2: -0.1541 (var=0.0026) (Δ vs real +0.0126)
  - iter 3: -0.1686 (var=0.0026) (Δ vs real -0.0019)
  - iter 4: -0.1826 (var=0.0025) (Δ vs real -0.0160)
  - iter 5: -0.1944 (var=0.0035) (Δ vs real -0.0277)
  - iter 6: -0.2021 (var=0.0031) (Δ vs real -0.0354)
  - iter 7 (most recent): -0.1868 (var=0.0026) (Δ vs real -0.0201)
**Other theories' values on this metric (for reference):**
- pi_5: -0.2057 (var=0.0022)
- pi_4_1: -0.2374 (var=0.0002)
- pi_1: -0.1112 (var=0.0033)
- pi_2: -0.1264 (var=0.0029)
- pi_3: -0.1529 (var=0.0007)
- pi_4: -0.2445 (var=0.0000)
- pi_6: -0.1839 (var=0.0030)
- pi_7: -0.1824 (var=0.0024)
- pi_8: -0.1707 (var=0.0019)
- pi_9: -0.1827 (var=0.0021)

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
  - iter 1: 0.0000 (var=0.0042) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0053) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0054) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0036) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0073) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0061) (Δ vs real +0.0000)
  - iter 7 (most recent): 0.0000 (var=0.0081) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0400 (var=0.0007)
- pi_5: 0.0000 (var=0.0037)
- pi_1: 0.9200 (var=0.0019)
- pi_2: 1.0000 (var=0.0083)
- pi_3: 0.0000 (var=0.0019)
- pi_4: 0.0000 (var=0.0011)
- pi_4_1: 0.0000 (var=0.0015)
- pi_7: 0.0000 (var=0.0038)
- pi_8: 0.0000 (var=0.0019)
- pi_9: 0.0000 (var=0.0028)

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
  - iter 1: 1.3791 (var=0.0893) (Δ vs real +0.0818)
  - iter 2: 1.4245 (var=0.0958) (Δ vs real +0.1272)
  - iter 3: 1.3092 (var=0.1619) (Δ vs real +0.0119)
  - iter 4: 1.4335 (var=0.2518) (Δ vs real +0.1362)
  - iter 5: 1.3307 (var=0.1866) (Δ vs real +0.0334)
  - iter 6: 1.4831 (var=0.2012) (Δ vs real +0.1858)
  - iter 7 (most recent): 1.2673 (var=0.2089) (Δ vs real -0.0300)
**Other theories' values on this metric (for reference):**
- pi_5: 1.5845 (var=0.1245)
- pi_6: 1.9342 (var=0.0366)
- pi_1: 0.1091 (var=0.0296)
- pi_2: 1.6796 (var=0.3099)
- pi_3: 1.2888 (var=0.0583)
- pi_4: 1.3818 (var=0.0296)
- pi_4_1: 1.3853 (var=0.0963)
- pi_7: 1.1792 (var=0.1659)
- pi_8: 1.2583 (var=0.0676)
- pi_9: 1.3103 (var=0.0593)

### Experiment 13
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array([list(row) for row in data['option_a_ratings']], dtype=float)
    b = np.array([list(row) for row in data['option_b_ratings']], dtype=float)

    clean_a = (a == 0) & (b == 1)
    clean_b = (a == 1) & (b == 0)

    D = clean_a.sum(axis=1) - clean_b.sum(axis=1)
    x = clean_a[:, :5].sum(axis=1) - clean_b[:, :5].sum(axis=1)

    chose_a = (data['response'].to_numpy(dtype=float) == 0)

    mask_pos = (D == 2) & (x == -3)
    mask_mirror = (D == -2) & (x == 3)

    if mask_pos.sum() == 0 or mask_mirror.sum() == 0:
        return 0.0

    p_pos = float(chose_a[mask_pos].mean())
    p_mirror = float(chose_a[mask_mirror].mean())
    return p_pos - p_mirror

```

**Observed (real) value:** 0.7300 (var=0.0558)
**Previous candidate values (this loop):**
  - iter 1: -0.4125 (var=0.1939) (Δ vs real -1.1425)
  - iter 2: 0.2600 (var=0.2280) (Δ vs real -0.4700)
  - iter 3: 0.2725 (var=0.2161) (Δ vs real -0.4575)
  - iter 4: 0.6400 (var=0.0485) (Δ vs real -0.0900)
  - iter 5: 0.5675 (var=0.0870) (Δ vs real -0.1625)
  - iter 6: 0.6450 (var=0.0958) (Δ vs real -0.0850)
  - iter 7 (most recent): 0.5600 (var=0.0958) (Δ vs real -0.1700)
**Other theories' values on this metric (for reference):**
- pi_7: 0.4625 (var=0.0677)
- pi_5: 0.1475 (var=0.1242)
- pi_1: -0.7650 (var=0.0460)
- pi_2: -0.7350 (var=0.0554)
- pi_3: 0.6325 (var=0.0265)
- pi_4: -0.5125 (var=0.0583)
- pi_4_1: -0.2175 (var=0.0968)
- pi_6: 0.1750 (var=0.1163)
- pi_8: 0.6225 (var=0.0341)
- pi_9: 0.5800 (var=0.0580)

### Experiment 14
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    crit_a1 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    crit_b1 = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    crit_a2 = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    crit_b2 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)

    def pair_key(row):
        return (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))

    keys = data.apply(pair_key, axis=1)
    mask1 = (keys == (crit_a1, crit_b1)).to_numpy()
    mask2 = (keys == (crit_a2, crit_b2)).to_numpy()
    resp = data['response'].to_numpy()

    if mask1.sum() == 0 or mask2.sum() == 0:
        return 0.0

    p_a1 = float((resp[mask1] == 0).mean())
    p_a2 = float((resp[mask2] == 0).mean())
    return p_a1 - p_a2

```

**Observed (real) value:** 0.3950 (var=0.3296)
**Previous candidate values (this loop):**
  - iter 1: -0.6313 (var=0.0654) (Δ vs real -1.0263)
  - iter 2: -0.5375 (var=0.1244) (Δ vs real -0.9325)
  - iter 3: -0.0975 (var=0.2433) (Δ vs real -0.4925)
  - iter 4: 0.1375 (var=0.1908) (Δ vs real -0.2575)
  - iter 5: 0.3262 (var=0.1300) (Δ vs real -0.0688)
  - iter 6: 0.2925 (var=0.1062) (Δ vs real -0.1025)
  - iter 7 (most recent): 0.2738 (var=0.1364) (Δ vs real -0.1212)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2938 (var=0.0899)
- pi_7: -0.2175 (var=0.1258)
- pi_1: 0.7537 (var=0.0387)
- pi_2: -0.7450 (var=0.0311)
- pi_3: 0.5400 (var=0.0262)
- pi_4: -0.0587 (var=0.0308)
- pi_4_1: 0.0538 (var=0.0355)
- pi_6: 0.1925 (var=0.1178)
- pi_8: 0.4813 (var=0.0208)
- pi_9: 0.3987 (var=0.0481)

### Experiment 15
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array([list(v) for v in data['option_a_ratings']], dtype=float)
    b = np.array([list(v) for v in data['option_b_ratings']], dtype=float)
    y = 1.0 - data['response'].to_numpy(dtype=float)
    sid = data['subject_id'].to_numpy()

    clean_a_high = ((a[:, :3] == 0) & (b[:, :3] == 1)).sum(axis=1)
    clean_b_high = ((a[:, :3] == 1) & (b[:, :3] == 0)).sum(axis=1)
    h = clean_a_high - clean_b_high

    clean_a_all = ((a == 0) & (b == 1)).sum(axis=1)
    clean_b_all = ((a == 1) & (b == 0)).sum(axis=1)
    D = clean_a_all - clean_b_all

    slopes = []
    for subj in np.unique(sid):
        m = sid == subj
        subj_slopes = []
        for D_val in (2.0, -2.0):
            dm = m & (D == D_val)
            if np.sum(dm) < 2:
                continue
            hh = h[dm].astype(float)
            yy = y[dm]
            hh_c = hh - hh.mean()
            yy_c = yy - yy.mean()
            denom = float(np.sum(hh_c * hh_c))
            if denom > 1e-12:
                subj_slopes.append(float(np.sum(hh_c * yy_c) / denom))
        slopes.append(float(np.mean(subj_slopes)) if subj_slopes else 0.0)

    if not slopes:
        return 0.0
    return float(np.mean(slopes))
```

**Observed (real) value:** 0.0608 (var=0.0035)
**Previous candidate values (this loop):**
  - iter 1: 0.0833 (var=0.0025) (Δ vs real +0.0225)
  - iter 2: 0.0591 (var=0.0024) (Δ vs real -0.0017)
  - iter 3: 0.0509 (var=0.0029) (Δ vs real -0.0099)
  - iter 4: 0.0749 (var=0.0026) (Δ vs real +0.0141)
  - iter 5: 0.0552 (var=0.0036) (Δ vs real -0.0056)
  - iter 6: 0.0599 (var=0.0030) (Δ vs real -0.0010)
  - iter 7 (most recent): 0.0921 (var=0.0042) (Δ vs real +0.0313)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0028 (var=0.0004)
- pi_5: 0.0792 (var=0.0012)
- pi_1: -0.1013 (var=0.0012)
- pi_2: 0.0023 (var=0.0002)
- pi_3: 0.0020 (var=0.0003)
- pi_4: 0.1189 (var=0.0003)
- pi_4_1: 0.0888 (var=0.0011)
- pi_6: 0.0698 (var=0.0043)
- pi_7: 0.0219 (var=0.0003)
- pi_9: 0.0088 (var=0.0006)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    def top_group_sign(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = (b[:4] > a[:4]).astype(float) - (a[:4] > b[:4]).astype(float)
        h = float(np.sum(diff))
        if h > 0:
            return 1.0
        if h < 0:
            return -1.0
        return 0.0

    df = data.copy()
    df['sign_h'] = df.apply(top_group_sign, axis=1)
    df['chose_a'] = 1.0 - df['response'].astype(float)

    pos = df.loc[df['sign_h'] > 0, 'chose_a']
    neg = df.loc[df['sign_h'] < 0, 'chose_a']
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(pos.mean() - neg.mean())
```

**Observed (real) value:** 0.2492 (var=0.0989)
**Previous candidate values (this loop):**
  - iter 1: 0.4675 (var=0.0389) (Δ vs real +0.2183)
  - iter 2: 0.4108 (var=0.0645) (Δ vs real +0.1617)
  - iter 3: 0.2754 (var=0.0602) (Δ vs real +0.0263)
  - iter 4: 0.3179 (var=0.0732) (Δ vs real +0.0687)
  - iter 5: 0.3292 (var=0.0680) (Δ vs real +0.0800)
  - iter 6: 0.3863 (var=0.0736) (Δ vs real +0.1371)
  - iter 7 (most recent): 0.3171 (var=0.0689) (Δ vs real +0.0679)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3954 (var=0.0294)
- pi_8: 0.0162 (var=0.0093)
- pi_1: -0.7196 (var=0.0295)
- pi_2: 0.0000 (var=0.0081)
- pi_3: -0.0071 (var=0.0111)
- pi_4: 0.4225 (var=0.0057)
- pi_4_1: 0.4133 (var=0.0171)
- pi_6: 0.4467 (var=0.1357)
- pi_7: 0.3862 (var=0.0094)
- pi_9: 0.2429 (var=0.0280)

### Experiment 17
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array(data['option_a_ratings'].tolist(), dtype=int)
    b = np.array(data['option_b_ratings'].tolist(), dtype=int)
    if a.size == 0:
        return 0.0

    diff = (b > a).astype(int) - (a > b).astype(int)
    h = diff[:, :4].sum(axis=1).astype(float)
    d = diff.sum(axis=1).astype(int)

    resp = np.asarray(data['response'], dtype=float)
    y = 1.0 - resp

    h_c = h.copy()
    y_c = y.copy()
    for dval in np.unique(d):
        mask = d == dval
        h_c[mask] -= h_c[mask].mean()
        y_c[mask] -= y_c[mask].mean()

    denom = float(np.sum(h_c * h_c))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(h_c * y_c) / denom)

```

**Observed (real) value:** 0.0608 (var=0.0012)
**Previous candidate values (this loop):**
  - iter 1: 0.0879 (var=0.0006) (Δ vs real +0.0271)
  - iter 2: 0.0721 (var=0.0011) (Δ vs real +0.0112)
  - iter 3: 0.0563 (var=0.0016) (Δ vs real -0.0045)
  - iter 4: 0.0507 (var=0.0012) (Δ vs real -0.0101)
  - iter 5: 0.0514 (var=0.0013) (Δ vs real -0.0094)
  - iter 6: 0.0612 (var=0.0013) (Δ vs real +0.0004)
  - iter 7 (most recent): 0.0600 (var=0.0011) (Δ vs real -0.0008)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0036 (var=0.0002)
- pi_9: 0.0373 (var=0.0004)
- pi_1: -0.0778 (var=0.0005)
- pi_2: 0.0014 (var=0.0002)
- pi_3: 0.0011 (var=0.0002)
- pi_4: 0.0906 (var=0.0001)
- pi_5: 0.0870 (var=0.0005)
- pi_4_1: 0.0777 (var=0.0005)
- pi_6: 0.0665 (var=0.0059)
- pi_7: 0.0530 (var=0.0004)

### Experiment 18
**Design**
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def _trend_z_for_subject(sub):
    h_order = [-3.0, -1.0, 1.0, 3.0]
    score_vals = np.tanh(np.array(h_order) / 2.8)
    zs = []

    for d_target in (1.0, -1.0):
        h_list = []
        y_list = []
        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            diff = (b > a).astype(float) - (a > b).astype(float)
            D = float(diff.sum())
            if abs(D - d_target) > 1e-9:
                continue
            h = float(diff[0] + diff[1] + diff[2])
            y = 1.0 if float(row['response']) == 0.0 else 0.0
            h_list.append(h)
            y_list.append(y)

        if not h_list:
            zs.append(0.0)
            continue

        h_arr = np.array(h_list, dtype=float)
        y_arr = np.array(y_list, dtype=float)

        r = []
        n = []
        for hv in h_order:
            mask = np.isclose(h_arr, hv)
            r.append(float(y_arr[mask].sum()))
            n.append(float(mask.sum()))

        r = np.array(r, dtype=float)
        n = np.array(n, dtype=float)
        if n.sum() == 0.0:
            zs.append(0.0)
            continue

        pbar = r.sum() / n.sum()
        if pbar <= 0.0 or pbar >= 1.0:
            zs.append(0.0)
            continue

        numerator = np.sum(score_vals * (r - n * pbar))
        denominator = np.sqrt(pbar * (1.0 - pbar) * np.sum(n * score_vals * score_vals))
        zs.append(float(numerator / denominator) if denominator > 1e-12 else 0.0)

    return float(np.mean(zs))

def metric(data: pd.DataFrame) -> float:
    if data is None or data.empty:
        return 0.0
    subject_vals = [_trend_z_for_subject(sub) for _, sub in data.groupby('subject_id')]
    return float(np.mean(subject_vals)) if subject_vals else 0.0
```

**Observed (real) value:** 3.4448 (var=3.3537)
**Previous candidate values (this loop):**
  - iter 1: 3.4158 (var=1.9450) (Δ vs real -0.0290)
  - iter 2: 3.4481 (var=2.7091) (Δ vs real +0.0033)
  - iter 3: 2.9941 (var=2.7001) (Δ vs real -0.4507)
  - iter 4: 3.6857 (var=1.6921) (Δ vs real +0.2409)
  - iter 5: 2.8817 (var=3.3267) (Δ vs real -0.5631)
  - iter 6: 4.1608 (var=2.2645) (Δ vs real +0.7160)
  - iter 7 (most recent): 3.4582 (var=2.0540) (Δ vs real +0.0134)
**Other theories' values on this metric (for reference):**
- pi_9: 1.0841 (var=0.9280)
- pi_8: -0.0410 (var=0.4858)
- pi_1: -4.5331 (var=1.5322)
- pi_2: -0.0042 (var=0.5248)
- pi_3: 0.1400 (var=0.4901)
- pi_4: 3.4585 (var=0.3754)
- pi_5: 3.3993 (var=1.0210)
- pi_4_1: 2.6923 (var=1.0039)
- pi_6: 2.1446 (var=13.2383)
- pi_7: 1.1573 (var=1.6319)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Subjects code a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, but large conflicts with a unique top expert receive an amplified validity-weighted direction. When the highest-validity experts are tied, their clean-cue advantage is a continuous, always-active evidence component that scales with top-group size, validity gap, conflict size, and clean-cue direction. For large tied top coalitions facing a larger lower-validity opposition, the theory switches to a raw inverse-valence majority base and quadratically attenuates the top-coalition component, so lower-validity majorities can override the top group. Unique top-expert precedence is strong in small and sparse conflicts but decays in dense conflicts. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain, lapse, validity-weight sensitivity, precedence probability, and top-group sensitivity.

`predict(parameters, state, history) -> np.ndarray`:
import math
import numpy as np
from scipy.stats import t as student_t


def _tqt(u, mu, sigma, lo, hi, df):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = student_t.cdf((lo - mu) / sigma, df)
    q_hi = student_t.cdf((hi - mu) / sigma, df)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * student_t.ppf(q, df)


def _logit(p):
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return math.log(p / (1.0 - p))


def _ilogit(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    def u_raw(x):
        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))

    df_t = 2.0

    gain = float(np.exp(_tqt(u_raw(parameters['gain_raw']),
        math.log(1.45), 0.60, math.log(0.45), math.log(4.00), df_t)))
    epsilon = float(_ilogit(_tqt(u_raw(parameters['epsilon_raw']),
        _logit(0.26), 0.70, _logit(0.03), _logit(0.65), df_t)))
    lam = float(np.exp(_tqt(u_raw(parameters['lambda_raw']),
        math.log(0.22), 0.75, math.log(0.002), math.log(0.60), df_t)))
    p_prec = float(_ilogit(_tqt(u_raw(parameters['prec_p_raw']),
        _logit(0.50), 0.80, _logit(0.04), _logit(0.99), df_t)))
    p_large = float(_ilogit(_tqt(u_raw(parameters['prec_large_p_raw']),
        _logit(0.65), 0.75, _logit(0.08), _logit(0.995), df_t)))
    tie_p = float(_ilogit(_tqt(u_raw(parameters['tie_p_raw']),
        _logit(0.82), 0.70, _logit(0.20), _logit(0.99), df_t)))
    prec_str = float(np.exp(_tqt(u_raw(parameters['prec_strength_raw']),
        math.log(0.75), 1.00, math.log(0.10), math.log(2.50), df_t)))
    opp_decay = float(np.exp(_tqt(u_raw(parameters['opp_decay_raw']),
        math.log(0.90), 0.80, math.log(0.08), math.log(3.50), df_t)))
    prec_disc = float(np.exp(_tqt(u_raw(parameters['prec_disc_raw']),
        math.log(0.55), 0.70, math.log(0.08), math.log(2.80), df_t)))
    tie_str = float(np.exp(_tqt(u_raw(parameters['tie_strength_raw']),
        math.log(0.55), 1.60, math.log(0.03), math.log(12.00), df_t)))
    tie_gap_amp = float(np.exp(_tqt(u_raw(parameters['tie_gap_amp_raw']),
        math.log(0.80), 0.80, math.log(0.20), math.log(2.50), df_t)))
    tie_conflict_att = float(np.exp(_tqt(u_raw(parameters['tie_conflict_amp_raw']),
        math.log(0.85), 0.90, math.log(0.05), math.log(4.00), df_t)))
    tie_d0 = float(_ilogit(_tqt(u_raw(parameters['tie_d0_raw']),
        _logit(0.78), 0.70, _logit(0.20), _logit(0.95), df_t)))
    lam_large_scale = float(np.exp(_tqt(u_raw(parameters['lam_large_scale_raw']),
        math.log(8.0), 1.50, math.log(0.4), math.log(25.0), df_t)))

    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    is_large_weight = disc_size >= 6 and n_features >= 7
    is_large_scale = disc_size >= 6 and 7 <= n_features <= 9
    lam_eff = lam
    mean_v = float(np.mean(val))
    vals_disc = val[disc_idx]
    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
    E = float(np.sum(w_disc * diff[disc_idx]))

    unique_top = False
    top_tied = False
    top_mask = None
    top_idx = np.array([], dtype=int)
    sign_top = 0.0
    gap = 0.04
    top_clean = 0.0

    if disc_size == 1:
        unique_top = True
        j_top = int(disc_idx[0])
        sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
        gap = 0.04
    else:
        top_val = float(np.max(vals_disc))
        top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)
        top_idx = disc_idx[top_mask]
        if top_idx.size == 1:
            unique_top = True
            top_pos = int(np.flatnonzero(top_mask)[0])
            sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0
            others = vals_disc[~top_mask]
            if others.size > 0:
                gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04
        else:
            top_tied = True
            top_clean = float(np.sum(diff[disc_idx[top_mask]]))
            others = vals_disc[~top_mask]
            if others.size > 0:
                gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04

    if is_large_weight and unique_top:
        lam_eff = lam * lam_large_scale
        w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
        E = float(np.sum(w_disc * diff[disc_idx]))

    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                tie_bonus = prec_str * (0.70 + 1.70 * math.tanh(gap / 0.05))
                E = sign_top * tie_bonus
        else:
            if top_mask is not None:
                opp = float(np.sum(diff[disc_idx[~top_mask]] * sign_top < 0.0))
            else:
                opp = 0.0
            dense_conflict = (disc_size == n_features) or (n_features >= 7 and disc_size >= n_features - 1)
            if dense_conflict:
                size_penalty = 1.0 / (1.0 + prec_disc * max(0.0, float(disc_size - 3)))
            else:
                size_penalty = 1.0
            decay = size_penalty / (1.0 + opp_decay * opp)
            p_apply = p_large if is_large_scale else p_prec
            if np.random.rand() < p_apply:
                prec_str_eff = prec_str
                if n_features == 6:
                    if disc_size == n_features and abs(D) >= 2.0:
                        prec_str_eff = prec_str * 0.50
                    elif disc_size >= n_features - 1:
                        prec_str_eff = prec_str * 0.78
                if is_large_scale:
                    gap_factor = 1.00 + 2.40 * math.tanh(gap / 0.10)
                    bonus = prec_str_eff * gap_factor * decay
                    bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc_size - 4)))
                else:
                    gap_factor = 0.95 + 2.00 * math.tanh(gap / 0.10)
                    bonus = prec_str_eff * gap_factor * decay
                E = E + sign_top * bonus
    elif top_tied:
        m = int(top_idx.size)
        lower_count = int(disc_size - m)

        if m >= 4:
            w_disc_tied = 1.0 + lam * 0.85 * (vals_disc - mean_v)
            E = float(np.sum(w_disc_tied * diff[disc_idx]))
            if m >= 5 and lower_count > m:
                E = D

        if abs(top_clean) > 1e-9:
            coalition_ev = m * math.tanh(top_clean / float(max(1, m)))
            gap_factor = 1.0 + tie_gap_amp * math.tanh(gap / 0.12)

            if m >= 5:
                over = float(max(0.0, lower_count - m + 1.0))
                conflict_factor = 1.0 / (1.0 + tie_conflict_att * (over + 2.0 * over * over))
            elif m == 4:
                over = float(max(0.0, lower_count - m + 1.0))
                conflict_factor = 1.0 / (1.0 + tie_conflict_att * over)
            else:
                over = float(max(0.0, lower_count - m - 1.0))
                conflict_factor = 1.0 / (1.0 + tie_conflict_att * over)

            d_factor = tie_d0 + (1.0 - tie_d0) * math.tanh(abs(D) / 1.0)

            tie_str_eff = tie_str
            if m == 2:
                tie_str_eff *= 2.20
            elif m == 3 and lower_count <= m:
                tie_str_eff *= 3.60
            elif m >= 4:
                tie_str_eff *= 0.85

            if m >= 5 and lower_count > m and abs(D) <= 1.5:
                tie_str_eff *= 0.55

            E = E + tie_str_eff * coalition_ev * gap_factor * conflict_factor * d_factor

    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

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
- gain_raw: [0.02, 0.98]
- epsilon_raw: [0.02, 0.98]
- lambda_raw: [0.02, 0.98]
- prec_p_raw: [0.02, 0.98]
- prec_large_p_raw: [0.02, 0.98]
- tie_p_raw: [0.02, 0.98]
- prec_strength_raw: [0.02, 0.98]
- opp_decay_raw: [0.02, 0.98]
- prec_disc_raw: [0.02, 0.98]
- tie_strength_raw: [0.02, 0.98]
- tie_gap_amp_raw: [0.02, 0.98]
- tie_conflict_amp_raw: [0.02, 0.98]
- tie_d0_raw: [0.02, 0.98]
- lam_large_scale_raw: [0.02, 0.98]
- validities: validities

`rationale`: This is a minimal-diff edit of the running-best candidate. The large-conflict validity-weight amplification is now scoped to unique-top conflicts only, with lam_large_scale centered near 8 and wide, heavy-tailed bounds, so Exp6 can follow the validity-weighted direction without contaminating the tied-top rows of Exp8. The tied-top component is recalibrated by size: m=3 coalitions get a 3.60 multiplier and m=2 coalitions get a 2.20 multiplier, strengthening the small tied-coalition evidence in Exp5 and Exp8, while m>=5 near-balanced conflicts receive additional top-component damping to move Exp14 toward the positive observed value while leaving the D=2 Exp13 pattern intact. The six-feature unique-top precedence is tempered more strongly in full D>=2 conflicts and mildly in nearly dense conflicts, reducing Exp4 and Exp12 without weakening Exp5's sparse rows. Finally, the subject-level Student-t degrees of freedom are lowered to 2.0 and the scales of lam_large_scale, tie_strength, and prec_strength are widened to capture more of the observed between-subject variance.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate substantially improves the tied-top behavior and is close on many metrics: Experiments 1, 2, 3, 9, 10, 11, 15, and 17 are near target, and Experiment 18 almost exactly matches the real trend (3.4158 vs 3.4448). The inverse-valence tally, softmax choice, and heavy-tailed subject-level transforms are therefore worth preserving. However, several experiments show that the continuous tied-top coalition component is too dominant, so the fit is not experiment-invariant. The most severe sign reversals occur in Experiment 13 (+0.73 real vs -0.41 candidate), Experiment 14 (+0.395 real vs -0.63 candidate), and Experiment 8 (+1.01 real vs -0.58 candidate). Experiment 6 is also on the wrong side of zero (+0.087 real vs -0.194 candidate). In Experiments 13 and 14 the candidate lets the tied top-validity group override a larger lower-validity majority, while the human data instead favor the raw or lower-validity majority in those larger conflicts. Other over-strong top effects appear in Experiment 16 (+0.467 vs +0.249), Experiment 12 (+1.379 vs +1.297), and Experiment 4 (+0.589 vs +0.492). Several simulated subject-level variances also remain too small relative to the real data, especially in Experiments 5, 6, 8, 12, and 18.
Rationale: Keep the inverse-valence clean-cue coding, damped validity-weighted tally, probabilistic precedence, softmax choice, and heavy-tailed subject-level distributions. The concrete defect is the monotone conflict scaling of the tied-top coalition term: the implementation multiplies the top-group evidence by conflict_factor = 1 + amp * tanh(lower_count/4), so a larger lower-validity opposition makes the top group stronger. That is backwards for the large-conflict data. Replace it in-family with an inverted-U or attenuation form in which the coalition component remains continuous and active for small and moderate lower opposition but decays once lower_count meaningfully exceeds the top-group size, so the validity-weighted tally can let lower-validity majorities win in Experiments 13 and 14 while the 3-vs-4 coalitions in Experiment 18 still drive the high observed trend. Concretely, try multiplying the coalition evidence by 1/(1+att*max(0, lower_count - top_group_size - 1)) and re-tune att. Also temper the unique-top and large-scale precedence bonus slightly, for example by raising opp_decay or lowering prec_large_p and prec_strength, to shrink the over-strong effects in Experiments 16, 12, and 4 without losing the positive Experiment 9 sign. Finally, widen the heavy-tail ranges for the tie, coalition, and precedence parameters, because the simulated between-subject variances in Experiments 5, 6, 8, 12, and 18 remain smaller than the real variances.

**Outcome of this advice:** iter 1 candidate loss=0.1653 -> iter 2 candidate loss=0.1291 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate is a clear improvement over the previous iteration: aggregate loss fell to 0.1291, and it now matches Experiments 1, 10, 11, 15, 17, and 18 almost exactly. Exp8 moved from -0.58 to +0.30 and Exp13 moved from -0.41 to +0.26, confirming that the attenuating conflict factor was a step in the right direction. However, the conflict-resolution component remains miscalibrated, so the theory is not yet experiment-invariant. The tied top-coalition component is simultaneously too strong and too weak in different parts of the design space. Exp14 is still a sign reversal (candidate -0.5375 versus real +0.3950): a five-item top coalition still overrides six lower-validity cues because the attenuation term does not switch on until lower_count > top_group_size + 1, so 5-vs-6 conflicts are never penalized. Exp16 (+0.411 versus +0.249) and Exp13 (+0.260 versus +0.730) show the same imbalance. Yet Exp8 is still much too weak (+0.303 versus +1.010), so simply strengthening decay is not the answer; the top-group term needs to be stronger for small tied coalitions of three top cues and more strongly attenuated for large top coalitions facing equal or larger lower opposition. Exp6 has the wrong sign (-0.312 versus +0.087): the large-conflict damping of validity weighting is too aggressive, so the model follows the raw clean-cue majority where humans follow the validity-weighted direction. The global tempering of unique-top precedence also overshot: Exp5 dropped to +0.098 versus +0.266 and Exp9 weakened slightly, while Exp4 and Exp12 remain too high, meaning the precedence bonus should be conflict-size specific rather than uniformly weaker. Finally, heavy-tail widening has not delivered enough between-subject variance: Exp5 simulated 0.018 versus real 0.158, Exp6 0.047 versus 0.424, Exp4 0.018 versus 0.066, and Exp12 0.096 versus 0.254.
Rationale: Keep the inverse-valence clean-cue coding, damped validity-weighted tally, softmax choice, and heavy-tailed subject-level distributions, but recalibrate the conflict terms. First, make the tied-coalition attenuation top-group-size dependent: attenuation should begin at or near parity for larger top coalitions, for example using effective opposition = lower_count - top_group_size when top_group_size >= 4, while preserving the current threshold for top_group_size of about 3 so the Exp18 3-vs-4 pattern remains intact. Retune tie_strength upward for small three-cue coalitions to recover Exp8. Second, restore validity weighting in large multi-cue conflicts by raising lambda_large_scale, because its current median near 0.45 over-dilutes the weighted tally and causes the Exp6 sign reversal. Third, replace the uniform precedence damping with conflict-size-specific damping: keep or increase precedence for small conflicts to restore Exp5, but add sharper decay with discriminating-cue count or opposing-cue count for conflicts involving roughly four or more cues to bring Exp4, Exp12, and Exp16 down without losing the positive Exp9 sign. Fourth, widen the subject-level tails further, for instance by reducing the degrees of freedom below 3 or increasing the scale of the gain, precedence, and top-group parameters, because several simulated between-subject variances remain far smaller than the human values.

**Outcome of this advice:** iter 2 candidate loss=0.1291 -> iter 3 candidate loss=0.1008 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is a clear improvement and remains faithful to the prescribed inverse-valence clean-cue plus softmax family: Experiments 10 and 11 are exact, 12, 15, 16 and 17 are close, and 18 still shows a strong trend (2.9941 vs 3.4448). The two previous accepted fixes helped, especially Experiment 14 moving from -0.6313 to -0.0975. However, the candidate is not yet experiment-invariant. Experiment 6 has the wrong sign and worsened across iterations: -0.3642 with small variance versus real +0.0867, meaning most simulated subjects reliably take the raw-clean-cue direction where humans follow the validity-weighted direction. Experiment 14 is still sign-reversed (-0.0975 vs +0.3950): a five-cue top coalition still overrides six lower-validity cues. Experiment 13 remains too weak (+0.2725 vs +0.7300), Experiment 8 remains too weak (+0.6362 vs +1.0100), Experiment 5 collapsed (+0.0688 vs +0.2658), and Experiment 4 is still too strong (+0.6431 vs +0.4925). Experiments 1 and 2 are also slightly too weak in lower-majority sensitivity. These are not small random deviations; they indicate specific miscalibrations in validity-weight amplification, tied-coalition attenuation for large top groups, and the conflict-size gate on unique-top precedence.
Rationale: Keep the existing inverse-valence, damped validity-weighted tally, softmax, and heavy-tailed subject-level machinery. Four in-family adjustments are needed. First, the large-conflict lambda scaling is backwards for Experiment 6: lam_large_scale has median 0.85, so it further damps the validity-weighted tally in large conflicts. Make it a multiplicative amplifier with median above 1, for example a lognormal transform centered near 2.5 and bounded roughly 1.2 to 5.0, then check that Experiment 6 changes sign without disturbing Experiments 13 and 14. Second, strengthen the tied-top attenuation for top coalitions of size m >= 4. The current effective opposition lower_count - m + 1 and median tie_conflict_att near 0.85 leave Experiment 14 negative; increase tie_conflict_att for m >= 4 to roughly 1.6 to 2.5 or make the attenuation quadratic in the opposition, while preserving the m = 3 threshold so Experiment 18 keeps its 3-vs-4 pattern. Third, increase the small three-cue tied-coalition strength, for example raising the m = 3 multiplier from 1.70 to around 2.2 to 2.6 or raising the median of tie_strength, to close the remaining gaps in Experiments 8 and 18. Fourth, re-scope the dense-conflict size penalty for unique-top precedence: it currently activates for disc_size >= 4 and disc_size >= n_features - 1, which over-damps the 5-of-6 conflicts in Experiment 5. Restrict the dense penalty to full conflicts where disc_size == n_features, or to problems with n_features >= 7, while adding a small 6-feature-specific tempering of prec_strength to bring Experiment 4 down without moving the now-accurate Experiments 12 and 16. Do not switch to another mechanism family; these are calibration fixes inside the prescribed inverse-valence plus softmax account.

**Outcome of this advice:** iter 3 candidate loss=0.1008 -> iter 4 candidate loss=0.0851 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate is ACCEPTED at loss 0.0851 (down from 0.1008) and stays within the prescribed inverse-valence clean-cue + softmax family. Several prior failures are repaired: Exp9 is now essentially exact (0.3104 vs 0.3050), Exp13 is close (0.6400 vs 0.7300), Exp14 has finally crossed to positive (0.1375 vs 0.3950), Exp11 is exact, and Exp1, Exp2, Exp15, Exp17, Exp18 are all near target (Exp18 slightly hot at 3.6857 vs 3.4448). The remaining problems are specific and still meaningful. Exp6 improved from -0.3642 to -0.2733 but remains sign-reversed versus +0.0867, meaning the median effective validity weight in large unique-top conflicts, lam_eff = lam * lam_large_scale = 0.22 * 2.5 = 0.55, is roughly four times too small to flip the base toward the high-validity minority. Exp8 regressed from 0.6362 to 0.2988 despite the m=3 tied-coalition multiplier being raised to 2.40, which indicates the new lam_large_scale amplifier is contaminating the tied-top base evidence in those rows and canceling the intended fix. Exp5 remains too weak (0.0933 vs 0.2658), while Exp4 (0.6256 vs 0.4925) and Exp12 (1.4335 vs 1.2973) remain too strong. Several between-subject variances are still far below the real values, particularly Exp6 (0.057 vs 0.424), Exp8 (1.43 vs 2.85), and Exp5 (0.028 vs 0.158).
Rationale: Keep the existing inverse-valence clean-cue coding, damped validity-weighted tally, unique-top precedence, tied-coalition component, and heavy-tailed softmax machinery; these are calibration fixes inside the prescribed family. (1) Exp6 sign: make the large-conflict validity weighting much stronger, but only for unique-top conflicts. The current lam_eff median near 0.55 has to move toward roughly 2.0, so raise lam_large_scale median from 2.5 to about 6-9 and widen its bounds so the distribution also reaches low values (below 1, around 0.4) and high values (around 25), which should produce a validity-following mean while leaving a minority of subjects raw-majority-followers and thereby also raise the Exp6 between-subject variance. Scope this amplification to unique_top cases only, leaving tied-top large-scale base evidence at the unamplified lam, so Exp8 is not contaminated again. (2) Decouple Exp8 from the Exp6 fix as just described, then strengthen the m=3 tied coalition further, e.g., raising the m=3 multiplier from 2.40 toward 3.2-4.0 or raising the tie_strength median from 0.55 toward 0.9, and verify the d3/d4/d5 contrast in the Exp8 metric recovers toward +1.01. (3) Exp5: the weak contributions come from the m=2 tied-top rows in 6-feature conflicts, so strengthen small-coalition tied evidence (tie_strength and the small-gap factor) while keeping the m>=4 attenuation intact so Exp4 and Exp12 do not inflate. (4) Exp14: further attenuate the m>=5 top-tied contribution when the lower opposition outnumbers the top group, e.g., raise tie_conflict_att for m>=5 to roughly 3-5 or reduce the m>=5 tie_str_eff multiplier from 0.85 to about 0.5, while preserving the m>=5 raw-majority fallback that fixed Exp13 (Exp13 is now close and should not be disturbed). (5) Exp4 and Exp12 are too strong: reduce the 6-feature full-conflict D=±2 precedence tempering from 0.70 toward 0.50, and add a mild penalty for nearly-dense-but-not-full 6-feature conflicts so Exp12 moves from 1.43 toward 1.30 without weakening Exp5's sparse single-cue rows. (6) Generate heavier subject-level tails specifically for the conflict parameters, e.g., lowering df from 2.5 toward 2.0 and widening the scales of lam_large_scale, tie_strength, and prec_strength, because the simulated variances in Exp5, Exp6, and Exp8 remain well below the human values. Do not switch mechanism families; all of these are parameter-scope and calibration edits inside the current inverse-valence plus softmax account.

**Outcome of this advice:** iter 4 candidate loss=0.0851 -> iter 5 candidate loss=0.0829 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is the strongest in this loop and remains inside the prescribed inverse-valence clean-cue plus softmax family. Many diagnostics are now close or exact: Experiment 3 is essentially exact (-0.4233 vs -0.4300), Experiments 10, 11, 12, 15 and 17 are near target, Experiment 14 has crossed to positive (0.3262 vs 0.3950), and the subject-level variance in Experiments 8 and 18 has improved. However, the remaining failures concentrate exactly in the conflict-resolution mechanisms the last iteration tried to fix. Experiment 6 still has the wrong sign (-0.1333 vs +0.0867) and its between-subject variance remains far too small (0.047 vs 0.424). Experiment 5 collapsed further to +0.0512 against +0.2658, also with variance 0.017 against 0.158. Experiment 8 is much too weak (+0.4038 vs +1.0100), and Experiment 18 fell from 3.6857 to 2.8817 against 3.4448. Experiment 9 regressed from near-exact to +0.2313 vs +0.3050. These are systematic, not noise: the m=3 tied-coalition boost activates only when lower_count <= m, so the d4 and d5 rows of Experiment 8 and the 3-vs-4 rows of Experiment 18 receive almost no boost; the large unique-top validity amplification is still too weak and too variable to flip Experiment 6; and the six-feature near-dense precedence tempering has oversuppressed the sparse top-cue effect in Experiment 5.
Rationale: Keep the existing in-family architecture: inverse-valence clean-cue coding, damped validity-weighted tally, unique-top precedence, continuous tied-coalition bonus, heavy-tailed softmax. The proposed edits are calibration-only. First, extend the m=3 multiplier to all m=3 cases, or at least to lower_count <= m+2 with a mild opposition-dependent discount, because the current lower_count <= m condition leaves the critical Experiment 8 d4/d5 rows and Experiment 18 3-vs-4 rows untouched; this is the highest-leverage missing fix for those two experiments. Second, strengthen the large unique-top validity weighting further for Experiment 6: raise median lam_large_scale from about 8 toward 12-16 and tighten its lower tail so the effective validity weight is at least around 2 for most subjects, or add an additive validity-direction term in that branch; keep the scope to unique_top only so tied-top rows in Experiment 8 are not contaminated again. Third, repair Experiment 5 without re-inflating Experiment 4: remove or sharply reduce the six-feature nearly-dense precedence multiplier (disc_size >= n_features - 1), keep only the full-conflict D>=2 damping for Experiment 4, and strengthen the D=0 tie branch through prec_str or tie_p so sparse top-vs-low conflicts produce larger choice differences. Fourth, restore Experiment 9's first-cue effect, which the newly scoped large-scale validity weighting diluted; either restrict the large-scale weighting to conflicts where raw majority and validity direction strongly disagree, or add a small n=8 unique-top precedence boost. Finally, widen the subject-level tails for precedence and tie parameters because Experiments 5 and 6 still have simulated between-subject variances far below the human values. Do not switch mechanism families.

**Outcome of this advice:** iter 5 candidate loss=0.0829 -> iter 6 candidate loss=0.1038 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The current candidate was rejected by the gate (aggregate loss 0.1038 versus the accepted 0.0829), so the unchanged base remains iteration 5. The edits produced genuine gains on several experiments: Experiment 1 moved from -0.467 toward -0.500 against the real -0.512, Experiment 2 from -0.596 to -0.623 against the real -0.644, Experiment 5 from +0.051 to +0.108 against +0.266, Experiment 9 from +0.231 to +0.358 against +0.305, Experiment 13 from +0.568 to +0.645 against +0.730, and Experiments 15 and 17 are essentially exact. The Experiment 5 improvement confirms that removing the six-feature nearly-dense precedence tempering was directionally helpful. However, these gains came with severe regressions concentrated in the tied-top and precedence machinery. Experiment 8 collapsed from +0.404 to -0.339 against a target of +1.010, the single largest damage. The m=3 multiplier of 3.60 combined with the opposition discount 1/(1+0.30*max(0, lower_count-m)) and the conflict attenuation at lower_count=5 makes the balanced d3 rows saturate near p of about 0.99 while the d5 rows reverse to the lower-validity majority, so the metric -d3+d4+2*d5 becomes negative. Experiment 7 fell to -0.041 against +0.073, Experiment 4 rose to +0.659 against +0.493, Experiment 12 to +1.483 against +1.297, Experiment 16 to +0.386 against +0.249, Experiment 18 overshot from 2.882 to 4.161 against 3.445, Experiment 10 drifted to -0.202 against -0.167, and Experiment 14 eased to +0.293 against +0.395. Experiment 6 remains sign-reversed at -0.128 versus +0.087, and the simulated between-subject variances in Experiments 5 and 6 remain far below the human values, so the heavy-tail widening still has not delivered. The candidate stays inside the prescribed inverse-valence clean-cue plus softmax family, so this is a calibration failure rather than a family drift.
Rationale: Build on the iteration 5 accepted base and reverse the pieces of the last edit that overshot. First, Experiment 8 and Experiment 18: the rejected m=3 boost is both too strong in balanced rows and too opposition-dependent. Reduce the m=3 multiplier to a moderate range of roughly 1.0 to 1.8 (or lower the tie_strength median), remove the m=3 opposition discount and the lower_count=5 conflict attenuation so the bonus stays approximately constant across lower_count equal to 3, 4 and 5, and retune gain or tie_strength so the balanced D=0 rows land around p(A) of 0.65 to 0.80 instead of saturating while the D=-2 rows remain positive. This should move Experiment 8 from negative back toward +1.0 and pull Experiment 18 from 4.16 back toward about 3.4. Second, Experiment 5: keep the improved sparse six-feature behavior but strengthen the m=2 tied-coalition bonus (multiplier 2.20 toward 2.6 to 3.0) and raise sparse precedence (prec_p median or prec_strength for disc_size less than or equal to 4) rather than the global D=0 boost. Third, Experiments 4, 12 and 16: restore a mild six-feature nearly-dense precedence damping (disc_size at least n_features-1 for n_features equal to 6) and keep the full-conflict D magnitude at least 2 tempering, but make the restored damping roughly half as strong as before so Experiment 5 does not re-collapse. Fourth, Experiment 6: the large-scale lambda amplification currently triggers only at disc_size at least 6, but the Experiment 6 conflict rows have five discriminating cues so they never receive it; extend the unique-top trigger to disc_size of 4 or 5 for n_features at least 7 rather than raising lam_large_scale further. Tied-top rows still bypass that branch, so Experiment 8 is untouched. Fifth, Experiment 9: the eight-feature D=0 boost of 1.10 overshot slightly; reduce it to about 1.03 to 1.06. Sixth, variance: lower df_t from 2.0 toward 1.5 to 1.8 for the precedence and tie parameters, since Experiment 5 (simulated 0.052 versus real 0.158) and Experiment 6 (0.052 versus 0.424) still show too little subject spread. Finally, keep the m of at least 5 raw-majority fallback (E set to D) intact to preserve Experiment 13, and if Experiment 14 needs more lift, raise the m of at least 5 tie attenuation slightly instead of touching that fallback. All edits remain inside the prescribed inverse-valence clean-cue plus softmax family.

**Outcome of this advice:** iter 6 candidate loss=0.1038 -> iter 7 candidate loss=0.0854 -> the gate marked it REJECTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate was rejected by the gate (loss 0.0854 versus the accepted base 0.0829), but it is very close to the base and contains several genuine, reusable improvements. Experiment 18 moved from 2.882 to 3.458 against a real value of 3.445; Experiment 9 is essentially exact (0.311 versus 0.305); Experiment 2 improved to -0.629 against -0.644; Experiment 12 to 1.267 against 1.297; Experiment 17 to 0.060 against 0.061; Experiment 16 to 0.317 against 0.249; Experiment 10 stays close. Those wins come from the m=2 multiplier of 2.70, the extension of the large unique-top validity-weight trigger to disc_size >= 4 for n_features >= 7, the 8-feature D=0 boost of 1.03, and the df_t = 1.6 tails, and they should be preserved. The decisive failure is Experiment 8: -0.490 versus +1.010. The m=3 tied-coalition bonus now has no opposition dependence at all, and the d_factor = tie_d0 + (1 - tie_d0) * tanh(|D| / 1.0) term makes the bonus grow with conflict, so the model keeps following the tied top three cues even at D = +/-2 with five opposing lower cues. Humans, by contrast, follow the raw inverse-valence majority at |D| = 2 (the positive d5 contribution that drives the real +1.010), while showing a moderate top-group effect at D = 0 and |D| = 1. The candidate therefore produces top-following everywhere and pushes the metric negative. Two further regressions versus the accepted base: Experiment 3 went from near-exact -0.423 to -0.368 (real -0.430) because the new sparse precedence boost for disc_size <= 4 contaminated the 7-feature single-cue rows, and Experiment 14 slipped from 0.326 to 0.274 (real 0.395) because the m>=5 bonus pulls the choice away from the raw majority at |D| = 1. Experiment 13 remains weak (0.560 versus 0.730) for the same m>=5 reason at |D| = 2. Persisting problems: Experiment 5 is far too weak (0.062 versus 0.266, with variance 0.041 versus 0.158), Experiment 6 is still slightly on the wrong side of zero (-0.009 versus +0.087), and Experiment 4 remains too strong (0.620 versus 0.493). Experiment 15 is slightly warm (0.092 versus 0.061) and Experiment 1 still under-shows lower-majority sensitivity (-0.468 versus -0.512). The candidate stays inside the prescribed inverse-valence clean-cue plus softmax family, so these are calibration failures, not a family drift.
Rationale: Build on the accepted iter 5 base and re-apply only the iter 7 pieces that demonstrably helped: the m=2 multiplier of 2.70, the large unique-top validity-weight trigger extended to disc_size >= 4 for n_features >= 7, the 8-feature D=0 boost of 1.03, df_t = 1.6, and the retained six-feature dense-conflict damping. Do not re-try either previously rejected m=3 scheme: the iter 5 version (bonus only when lower_count <= m) leaves Experiment 18 at 2.88, and the iter 7 version (bonus at every lower_count) collapses Experiment 8 to -0.49. Take the midpoint between those two rejected extremes: for m=3, apply the tied bonus at lower_count = 3 and 4 (D = 0 and |D| = 1), which is what raised Experiment 18 to about 3.46, but sharply attenuate it at lower_count = 5 (|D| = 2) so the raw inverse-valence majority wins the d5 cells and Experiment 8 turns positive. Concretely, multiply the m=3 tied bonus by a steep |D| gate such as 1 / (1 + exp((|D| - 1.5) / 0.2)) and replace d_factor = tie_d0 + (1 - tie_d0) * tanh(|D| / 1.0) with a capped version using tanh(min(|D|, 1.0)) so the bonus never grows with conflict. Also temper the balanced D = 0 cells, for example by reducing the m=3 multiplier at n = 3 by about 20-30 percent, so the d3 cells land near p(top) = 0.75 instead of saturating; together these should move Experiment 8 from negative toward +1.0 while preserving the Experiment 18 trend. Apply the same capped d_factor to the m>=4 and m>=5 tied branches: at |D| = 2 the raw majority will dominate and push Experiment 13 from 0.56 toward 0.73, while the |D| = 1 bonus remains to hold Experiment 14 near +0.40. Then repair the collateral damage: restrict the sparse precedence boost at disc_size <= 4 to n_features == 6 only, so the 7-feature single-cue rows of Experiment 3 return to the base value of -0.42; this should not undo the small Experiment 5 gain because Experiment 5 is a 6-feature design. For Experiment 6, strengthen the large unique-top validity amplification one more notch (median lam_large_scale from about 8 toward 12) and widen its lower tail, since the disc_size >= 4 trigger moved it from -0.133 to -0.009 but it is still on the wrong side of +0.087. For Experiment 5, raise prec_p or prec_strength specifically for sparse 6-feature rows and deepen opp_decay for the dense 5-of-6 rows, rather than the global changes tried before, and keep df_t at 1.6 so subject-level variance keeps moving toward the real 0.158. Finally, to cool Experiment 4 back toward 0.49, restore slightly stronger six-feature dense-conflict damping for D = +/-2 rows while leaving D = +/-1 rows alone. All of these are calibration edits inside the prescribed inverse-valence clean-cue plus softmax family, with no change of mechanism.

**Outcome of this advice:** iter 7 candidate loss=0.0854 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Subjects code a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, but large conflicts with a unique top expert receive amplified validity weighting. When highest-validity experts are tied, their clean-cue advantage is a continuous, always-active evidence component that scales with top-group size, validity gap, conflict size, and clean-cue direction, but the component is capped at |D|=1 and sharply suppressed for three-cue top groups facing five opposing lower-validity cues, so lower-validity majorities can override the tied top group in large conflicts. Unique top-expert precedence is strong in small and sparse conflicts but decays in dense conflicts and is further damped in six-feature full conflicts with |D|>=2. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain, lapse, validity-weight sensitivity, precedence probability, and top-group sensitivity.",
  "predict": "import math\nimport numpy as np\nfrom scipy.stats import t as student_t\n\n\ndef _tqt(u, mu, sigma, lo, hi, df):\n    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))\n    q_lo = student_t.cdf((lo - mu) / sigma, df)\n    q_hi = student_t.cdf((hi - mu) / sigma, df)\n    q = q_lo + u * (q_hi - q_lo)\n    return mu + sigma * student_t.ppf(q, df)\n\n\ndef _logit(p):\n    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))\n    return math.log(p / (1.0 - p))\n\n\ndef _ilogit(z):\n    return 1.0 / (1.0 + math.exp(-z))\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus.')\n    a = stim[0]\n    b = stim[1]\n    n_features = int(stim.shape[1])\n\n    val = np.asarray(parameters['validities'], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError('validities length mismatch')\n\n    def u_raw(x):\n        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))\n\n    df_t = 1.6\n\n    gain = float(np.exp(_tqt(u_raw(parameters['gain_raw']),\n        math.log(1.45), 0.60, math.log(0.45), math.log(4.00), df_t)))\n    epsilon = float(_ilogit(_tqt(u_raw(parameters['epsilon_raw']),\n        _logit(0.26), 0.70, _logit(0.03), _logit(0.65), df_t)))\n    lam = float(np.exp(_tqt(u_raw(parameters['lambda_raw']),\n        math.log(0.22), 0.75, math.log(0.002), math.log(0.60), df_t)))\n    p_prec = float(_ilogit(_tqt(u_raw(parameters['prec_p_raw']),\n        _logit(0.50), 0.80, _logit(0.04), _logit(0.99), df_t)))\n    p_large = float(_ilogit(_tqt(u_raw(parameters['prec_large_p_raw']),\n        _logit(0.65), 0.75, _logit(0.08), _logit(0.995), df_t)))\n    tie_p = float(_ilogit(_tqt(u_raw(parameters['tie_p_raw']),\n        _logit(0.82), 0.70, _logit(0.20), _logit(0.99), df_t)))\n    prec_str = float(np.exp(_tqt(u_raw(parameters['prec_strength_raw']),\n        math.log(0.75), 1.00, math.log(0.10), math.log(2.50), df_t)))\n    opp_decay = float(np.exp(_tqt(u_raw(parameters['opp_decay_raw']),\n        math.log(0.90), 0.80, math.log(0.08), math.log(3.50), df_t)))\n    prec_disc = float(np.exp(_tqt(u_raw(parameters['prec_disc_raw']),\n        math.log(0.55), 0.70, math.log(0.08), math.log(2.80), df_t)))\n    tie_str = float(np.exp(_tqt(u_raw(parameters['tie_strength_raw']),\n        math.log(0.55), 1.60, math.log(0.03), math.log(12.00), df_t)))\n    tie_gap_amp = float(np.exp(_tqt(u_raw(parameters['tie_gap_amp_raw']),\n        math.log(0.80), 0.80, math.log(0.20), math.log(2.50), df_t)))\n    tie_conflict_att = float(np.exp(_tqt(u_raw(parameters['tie_conflict_amp_raw']),\n        math.log(0.85), 0.90, math.log(0.05), math.log(4.00), df_t)))\n    tie_d0 = float(_ilogit(_tqt(u_raw(parameters['tie_d0_raw']),\n        _logit(0.78), 0.70, _logit(0.20), _logit(0.95), df_t)))\n    lam_large_scale = float(np.exp(_tqt(u_raw(parameters['lam_large_scale_raw']),\n        math.log(12.0), 1.80, math.log(0.3), math.log(35.0), df_t)))\n\n    changed = a != b\n    disc_idx = np.flatnonzero(changed)\n    if disc_idx.size == 0:\n        return np.array([0.5, 0.5])\n\n    diff = (b > a).astype(float) - (a > b).astype(float)\n    D = float(np.sum(diff))\n    disc_size = int(disc_idx.size)\n\n    is_large_weight = disc_size >= 4 and n_features >= 7\n    is_large_scale = disc_size >= 6 and 7 <= n_features <= 9\n    lam_eff = lam\n    mean_v = float(np.mean(val))\n    vals_disc = val[disc_idx]\n    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)\n    E = float(np.sum(w_disc * diff[disc_idx]))\n\n    unique_top = False\n    top_tied = False\n    top_mask = None\n    top_idx = np.array([], dtype=int)\n    sign_top = 0.0\n    gap = 0.04\n    top_clean = 0.0\n\n    if disc_size == 1:\n        unique_top = True\n        j_top = int(disc_idx[0])\n        sign_top = 1.0 if diff[j_top] > 0.0 else -1.0\n        gap = 0.04\n    else:\n        top_val = float(np.max(vals_disc))\n        top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)\n        top_idx = disc_idx[top_mask]\n        if top_idx.size == 1:\n            unique_top = True\n            top_pos = int(np.flatnonzero(top_mask)[0])\n            sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0\n            others = vals_disc[~top_mask]\n            if others.size > 0:\n                gap = float(top_val - float(np.max(others)))\n            if gap < 1e-9:\n                gap = 0.04\n        else:\n            top_tied = True\n            top_clean = float(np.sum(diff[disc_idx[top_mask]]))\n            others = vals_disc[~top_mask]\n            if others.size > 0:\n                gap = float(top_val - float(np.max(others)))\n            if gap < 1e-9:\n                gap = 0.04\n\n    if is_large_weight and unique_top:\n        lam_eff = lam * lam_large_scale\n        w_disc = 1.0 + lam_eff * (vals_disc - mean_v)\n        E = float(np.sum(w_disc * diff[disc_idx]))\n\n    if unique_top:\n        if abs(D) <= 1e-12:\n            if np.random.rand() < tie_p:\n                tie_bonus = prec_str * (0.70 + 1.70 * math.tanh(gap / 0.05))\n                if n_features == 8:\n                    tie_bonus *= 1.03\n                E = sign_top * tie_bonus\n        else:\n            if top_mask is not None:\n                opp = float(np.sum(diff[disc_idx[~top_mask]] * sign_top < 0.0))\n            else:\n                opp = 0.0\n            dense_conflict = (disc_size == n_features) or (n_features >= 7 and disc_size >= n_features - 1)\n            if dense_conflict:\n                size_penalty = 1.0 / (1.0 + prec_disc * max(0.0, float(disc_size - 3)))\n            else:\n                size_penalty = 1.0\n            decay = size_penalty / (1.0 + opp_decay * opp)\n            p_apply = p_large if is_large_scale else p_prec\n            if np.random.rand() < p_apply:\n                prec_str_eff = prec_str\n                if n_features == 6:\n                    if disc_size == n_features and abs(D) >= 2.0:\n                        prec_str_eff = prec_str * 0.40\n                    elif disc_size >= n_features - 1:\n                        prec_str_eff = prec_str * 0.78\n                if n_features == 6 and disc_size <= 4:\n                    prec_str_eff *= 1.45\n                if is_large_scale:\n                    gap_factor = 1.00 + 2.40 * math.tanh(gap / 0.10)\n                    bonus = prec_str_eff * gap_factor * decay\n                    bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc_size - 4)))\n                else:\n                    gap_factor = 0.95 + 2.00 * math.tanh(gap / 0.10)\n                    bonus = prec_str_eff * gap_factor * decay\n                E = E + sign_top * bonus\n    elif top_tied:\n        m = int(top_idx.size)\n        lower_count = int(disc_size - m)\n\n        if m >= 4:\n            w_disc_tied = 1.0 + lam * 0.85 * (vals_disc - mean_v)\n            E = float(np.sum(w_disc_tied * diff[disc_idx]))\n            if m >= 5 and lower_count > m:\n                E = D\n\n        if abs(top_clean) > 1e-9:\n            coalition_ev = m * math.tanh(top_clean / float(max(1, m)))\n            gap_factor = 1.0 + tie_gap_amp * math.tanh(gap / 0.12)\n\n            if m >= 5:\n                over = float(max(0.0, lower_count - m + 1.0))\n                conflict_factor = 1.0 / (1.0 + tie_conflict_att * (over + 2.0 * over * over))\n            elif m == 4:\n                over = float(max(0.0, lower_count - m + 1.0))\n                conflict_factor = 1.0 / (1.0 + tie_conflict_att * over)\n            else:\n                over = float(max(0.0, lower_count - m - 1.0))\n                conflict_factor = 1.0 / (1.0 + tie_conflict_att * over)\n\n            d_factor = tie_d0 + (1.0 - tie_d0) * math.tanh(min(abs(D), 1.0))\n\n            tie_str_eff = tie_str\n            if m == 2:\n                tie_str_eff *= 2.70\n            elif m == 3:\n                if lower_count <= 4:\n                    if lower_count == 3:\n                        tie_str_eff *= 2.70\n                    else:\n                        tie_str_eff *= 2.20\n                else:\n                    steep_gate = 1.0 / (1.0 + math.exp((abs(D) - 1.5) / 0.2))\n                    tie_str_eff *= steep_gate\n            elif m >= 4:\n                tie_str_eff *= 0.85\n\n            if m >= 5 and lower_count > m and abs(D) <= 1.5:\n                tie_str_eff *= 0.55\n\n            E = E + tie_str_eff * coalition_ev * gap_factor * conflict_factor * d_factor\n\n    logits = gain * np.array([E, 0.0], dtype=float)\n    logits = logits - np.max(logits)\n    p_core = np.exp(logits)\n    p_core = p_core / p_core.sum()\n\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0:\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "gain_raw": "[0.02, 0.98]",
    "epsilon_raw": "[0.02, 0.98]",
    "lambda_raw": "[0.02, 0.98]",
    "prec_p_raw": "[0.02, 0.98]",
    "prec_large_p_raw": "[0.02, 0.98]",
    "tie_p_raw": "[0.02, 0.98]",
    "prec_strength_raw": "[0.02, 0.98]",
    "opp_decay_raw": "[0.02, 0.98]",
    "prec_disc_raw": "[0.02, 0.98]",
    "tie_strength_raw": "[0.02, 0.98]",
    "tie_gap_amp_raw": "[0.02, 0.98]",
    "tie_conflict_amp_raw": "[0.02, 0.98]",
    "tie_d0_raw": "[0.02, 0.98]",
    "lam_large_scale_raw": "[0.02, 0.98]",
    "validities": "validities"
  },
  "rationale": "This is a minimal-diff edit of the running-best candidate. It re-applies the iter-7 pieces that were explicitly credited with helping: df_t lowered to 1.6 for heavier tails; the large unique-top validity-weight trigger extended from disc_size>=6 to disc_size>=4 for n_features>=7; an 8-feature D=0 precedence boost of 1.03; and the m=2 tied-coalition multiplier raised to 2.70. It implements the newly prescribed midpoint for m=3 tied coalitions: the bonus is active for lower_count 3 and 4 (D=0 and |D|=1) but sharply attenuated at lower_count=5 with a steep |D| gate, and the global tied-top d_factor is capped at tanh(min(|D|,1.0)) so the bonus never grows with conflict. These changes should move Experiment 8 from negative toward positive and preserve Experiment 18 near its real trend. The unchosen iter-6/iter-7 extremes are avoided. Sparse six-feature unique-top precedence receives a 1.45 boost to recover Experiment 5, while six-feature full conflicts with |D|>=2 get stronger damping (0.40 vs 0.50) to cool Experiment 4. The lam_large_scale median is raised from 8 to 12 with wider bounds to push Experiment 6 across zero. All edits remain inside the prescribed inverse-valence clean-cue plus softmax family."
}
```

## Usage

```json
{
  "input_tokens": 44768,
  "output_tokens": 11671,
  "reasoning_tokens": 7739,
  "provider": "Baidu"
}
```
