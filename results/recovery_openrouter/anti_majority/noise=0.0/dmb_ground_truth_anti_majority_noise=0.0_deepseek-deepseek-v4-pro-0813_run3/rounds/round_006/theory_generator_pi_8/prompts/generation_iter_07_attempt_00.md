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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Propose a new theory: inverse-valence majority with unique-expert precedence. The theory should retain the inverse-valence assumption, treating rating 0 as a clean/favorable cue and rating 1 as a defect, so every discriminating feature contributes approximately +1 for A or -1 for B to a raw clean-cue tally. The new ingredient is that if exactly one discriminating expert has a strictly highest communicated validity, that unique expert receives a privileged override weight, active within a limited raw-tally window such as abs(D) <= 3, with strength increasing in the validity gap to the next-best expert. If two or more experts are tied at the maximum validity, the unique-expert override is suppressed and choice is driven by a near-equal-weight tally over all discriminating cues. Exact raw-tally ties can be broken by choices among the changed experts, but tied top experts should not be allowed to override a non-zero low-validity majority. The model should include subject-level parameters for the probability of applying unique-expert precedence, the maximum override window, softmax choice gain, and a lapse rate, all on distributions matched to the observed between-subject variance. This theory differs from pi_7 by removing continuous logit-tanh weighted evidence for every feature and from pi_5 by removing probabilistic cue exclusion and by tying the privileged role to uniqueness of the top expert rather than generic validity order. It should be instantiated with parameter ranges that allow near-equal majority weighting and a small but occasionally decisive unique-top bonus.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

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


---

### `pi_6` (overall score: 0.730)

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
- Experiment 13: real=0.7300 (var=0.0558) vs this=0.1750 (var=0.1163)
- Experiment 14: real=0.3950 (var=0.3296) vs this=0.1925 (var=0.1178)


---

### `pi_3` (overall score: 0.611)

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
- Experiment 13: real=0.7300 (var=0.0558) vs this=0.6325 (var=0.0265)
- Experiment 14: real=0.3950 (var=0.3296) vs this=0.5400 (var=0.0262)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1038 -> ACCEPTED
- iter 2: loss=0.0661 -> ACCEPTED
- iter 3: loss=0.0660 -> ACCEPTED
- iter 4: loss=0.0975 -> REJECTED
- iter 5: loss=0.0962 -> REJECTED
- iter 6: loss=0.0592 -> ACCEPTED
- iter 7: loss=0.0586 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 7 at loss=0.0586 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.3421 (var=0.0104) (Δ vs real +0.1696)
  - iter 2: -0.4492 (var=0.0055) (Δ vs real +0.0625)
  - iter 3: -0.4729 (var=0.0085) (Δ vs real +0.0387)
  - iter 4: -0.4925 (var=0.0103) (Δ vs real +0.0192)
  - iter 5: -0.5329 (var=0.0068) (Δ vs real -0.0213)
  - iter 6: -0.4792 (var=0.0092) (Δ vs real +0.0325)
  - iter 7 (most recent): -0.4700 (var=0.0108) (Δ vs real +0.0417)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0092 (var=0.0043)
- pi_2: 0.6258 (var=0.0234)
- pi_3: -0.5379 (var=0.0084)
- pi_4: -0.5413 (var=0.0057)
- pi_5: -0.5308 (var=0.0142)
- pi_4_1: -0.5442 (var=0.0102)
- pi_6: -0.3700 (var=0.2536)
- pi_7: -0.4754 (var=0.0242)

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
  - iter 1: -0.3763 (var=0.0214) (Δ vs real +0.2675)
  - iter 2: -0.5050 (var=0.0279) (Δ vs real +0.1388)
  - iter 3: -0.5563 (var=0.0160) (Δ vs real +0.0875)
  - iter 4: -0.5350 (var=0.0382) (Δ vs real +0.1088)
  - iter 5: -0.5769 (var=0.0141) (Δ vs real +0.0669)
  - iter 6: -0.5306 (var=0.0211) (Δ vs real +0.1131)
  - iter 7 (most recent): -0.5444 (var=0.0217) (Δ vs real +0.0994)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7112 (var=0.0379)
- pi_1: -0.0013 (var=0.0059)
- pi_3: -0.6069 (var=0.0076)
- pi_4: -0.5913 (var=0.0097)
- pi_5: -0.5938 (var=0.0426)
- pi_4_1: -0.6038 (var=0.0238)
- pi_6: -0.6238 (var=0.2467)
- pi_7: -0.6138 (var=0.0287)

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
  - iter 1: -0.4338 (var=0.0091) (Δ vs real -0.0038)
  - iter 2: -0.4271 (var=0.0110) (Δ vs real +0.0029)
  - iter 3: -0.3887 (var=0.0095) (Δ vs real +0.0413)
  - iter 4: -0.0921 (var=0.0099) (Δ vs real +0.3379)
  - iter 5: -0.3692 (var=0.0122) (Δ vs real +0.0608)
  - iter 6: -0.3983 (var=0.0109) (Δ vs real +0.0317)
  - iter 7 (most recent): -0.3583 (var=0.0165) (Δ vs real +0.0717)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7113 (var=0.0398)
- pi_3: 0.0175 (var=0.0062)
- pi_2: 0.0117 (var=0.0115)
- pi_4: -0.4575 (var=0.0063)
- pi_5: -0.4596 (var=0.0245)
- pi_4_1: -0.4029 (var=0.0226)
- pi_6: -0.3550 (var=0.1762)
- pi_7: -0.6425 (var=0.0137)

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
  - iter 1: 0.3606 (var=0.0172) (Δ vs real -0.1319)
  - iter 2: 0.5025 (var=0.0195) (Δ vs real +0.0100)
  - iter 3: 0.5587 (var=0.0236) (Δ vs real +0.0663)
  - iter 4: 0.5112 (var=0.0298) (Δ vs real +0.0188)
  - iter 5: 0.5550 (var=0.0143) (Δ vs real +0.0625)
  - iter 6: 0.5556 (var=0.0247) (Δ vs real +0.0631)
  - iter 7 (most recent): 0.5281 (var=0.0303) (Δ vs real +0.0356)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6381 (var=0.0147)
- pi_1: -0.0244 (var=0.0050)
- pi_2: -0.7525 (var=0.0447)
- pi_4: 0.4313 (var=0.0096)
- pi_5: 0.4237 (var=0.0670)
- pi_4_1: 0.5069 (var=0.0237)
- pi_6: 0.4681 (var=0.1627)
- pi_7: 0.5363 (var=0.0483)

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
  - iter 1: 0.3146 (var=0.0140) (Δ vs real +0.0488)
  - iter 2: 0.3492 (var=0.0164) (Δ vs real +0.0833)
  - iter 3: 0.2296 (var=0.0127) (Δ vs real -0.0362)
  - iter 4: 0.1479 (var=0.0099) (Δ vs real -0.1179)
  - iter 5: 0.2062 (var=0.0141) (Δ vs real -0.0596)
  - iter 6: 0.2037 (var=0.0125) (Δ vs real -0.0621)
  - iter 7 (most recent): 0.2492 (var=0.0141) (Δ vs real -0.0167)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2971 (var=0.0096)
- pi_3: -0.2708 (var=0.0095)
- pi_1: -0.6967 (var=0.0260)
- pi_2: 0.3704 (var=0.0144)
- pi_5: 0.3858 (var=0.0225)
- pi_4_1: 0.2446 (var=0.0156)
- pi_6: 0.3300 (var=0.1102)
- pi_7: 0.2608 (var=0.0101)

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
  - iter 1: 0.0292 (var=0.0283) (Δ vs real -0.0575)
  - iter 2: 0.0267 (var=0.0329) (Δ vs real -0.0600)
  - iter 3: -0.0550 (var=0.0179) (Δ vs real -0.1417)
  - iter 4: 0.1758 (var=0.0326) (Δ vs real +0.0892)
  - iter 5: -0.0683 (var=0.0252) (Δ vs real -0.1550)
  - iter 6: -0.0700 (var=0.0254) (Δ vs real -0.1567)
  - iter 7 (most recent): 0.0100 (var=0.0445) (Δ vs real -0.0767)
**Other theories' values on this metric (for reference):**
- pi_3: -0.5333 (var=0.0223)
- pi_4: 0.2158 (var=0.0210)
- pi_1: -0.7375 (var=0.0404)
- pi_2: 0.6717 (var=0.0518)
- pi_5: 0.1383 (var=0.0736)
- pi_4_1: 0.1075 (var=0.0264)
- pi_6: 0.0358 (var=0.0678)
- pi_7: -0.2058 (var=0.0270)

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
  - iter 1: 0.0550 (var=0.0487) (Δ vs real -0.0175)
  - iter 2: 0.0212 (var=0.0393) (Δ vs real -0.0512)
  - iter 3: -0.0250 (var=0.0448) (Δ vs real -0.0975)
  - iter 4: 0.0012 (var=0.0473) (Δ vs real -0.0713)
  - iter 5: 0.3075 (var=0.0570) (Δ vs real +0.2350)
  - iter 6: 0.0262 (var=0.0558) (Δ vs real -0.0462)
  - iter 7 (most recent): 0.0313 (var=0.0532) (Δ vs real -0.0412)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0025 (var=0.0525)
- pi_5: 0.3325 (var=0.0612)
- pi_1: 0.0288 (var=0.0221)
- pi_2: 0.0137 (var=0.0602)
- pi_3: -0.0025 (var=0.0792)
- pi_4_1: -0.0400 (var=0.0672)
- pi_6: 0.1800 (var=0.1187)
- pi_7: 0.0812 (var=0.0474)

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
  - iter 1: 1.3262 (var=0.1793) (Δ vs real +0.3163)
  - iter 2: 1.8413 (var=0.1475) (Δ vs real +0.8313)
  - iter 3: 1.7437 (var=0.1235) (Δ vs real +0.7338)
  - iter 4: 1.7763 (var=0.1986) (Δ vs real +0.7663)
  - iter 5: 1.7112 (var=0.1547) (Δ vs real +0.7012)
  - iter 6: 1.6962 (var=0.1472) (Δ vs real +0.6863)
  - iter 7 (most recent): 1.7475 (var=0.1181) (Δ vs real +0.7375)
**Other theories' values on this metric (for reference):**
- pi_5: 1.3512 (var=0.8533)
- pi_4: -0.1263 (var=0.1938)
- pi_1: 1.4112 (var=0.1813)
- pi_2: -2.0162 (var=0.4200)
- pi_3: 1.7625 (var=0.1534)
- pi_4_1: 0.5012 (var=0.5143)
- pi_6: 1.1075 (var=2.7433)
- pi_7: 1.6462 (var=0.3404)

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
  - iter 1: 0.2900 (var=0.0086) (Δ vs real -0.0150)
  - iter 2: 0.3329 (var=0.0068) (Δ vs real +0.0279)
  - iter 3: 0.3079 (var=0.0137) (Δ vs real +0.0029)
  - iter 4: 0.0492 (var=0.0116) (Δ vs real -0.2558)
  - iter 5: 0.3050 (var=0.0126) (Δ vs real +0.0000)
  - iter 6: 0.2796 (var=0.0122) (Δ vs real -0.0254)
  - iter 7 (most recent): 0.2717 (var=0.0102) (Δ vs real -0.0333)
**Other theories' values on this metric (for reference):**
- pi_4_1: 0.4188 (var=0.0158)
- pi_5: 0.2196 (var=0.0342)
- pi_1: -0.7067 (var=0.0351)
- pi_2: 0.0204 (var=0.0104)
- pi_3: -0.0142 (var=0.0125)
- pi_4: 0.4296 (var=0.0093)
- pi_6: 0.1742 (var=0.1020)
- pi_7: 0.4579 (var=0.0150)

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
  - iter 1: -0.2128 (var=0.0012) (Δ vs real -0.0462)
  - iter 2: -0.1911 (var=0.0024) (Δ vs real -0.0244)
  - iter 3: -0.1705 (var=0.0023) (Δ vs real -0.0038)
  - iter 4: -0.1837 (var=0.0034) (Δ vs real -0.0170)
  - iter 5: -0.1738 (var=0.0016) (Δ vs real -0.0071)
  - iter 6: -0.1703 (var=0.0016) (Δ vs real -0.0037)
  - iter 7 (most recent): -0.1766 (var=0.0017) (Δ vs real -0.0100)
**Other theories' values on this metric (for reference):**
- pi_5: -0.2057 (var=0.0022)
- pi_4_1: -0.2374 (var=0.0002)
- pi_1: -0.1112 (var=0.0033)
- pi_2: -0.1264 (var=0.0029)
- pi_3: -0.1529 (var=0.0007)
- pi_4: -0.2445 (var=0.0000)
- pi_6: -0.1839 (var=0.0030)
- pi_7: -0.1824 (var=0.0024)

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
  - iter 1: 0.0000 (var=0.0021) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0022) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0020) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0025) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0027) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0028) (Δ vs real +0.0000)
  - iter 7 (most recent): 0.0000 (var=0.0025) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0400 (var=0.0007)
- pi_5: 0.0000 (var=0.0037)
- pi_1: 0.9200 (var=0.0019)
- pi_2: 1.0000 (var=0.0083)
- pi_3: 0.0000 (var=0.0019)
- pi_4: 0.0000 (var=0.0011)
- pi_4_1: 0.0000 (var=0.0015)
- pi_7: 0.0000 (var=0.0038)

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
  - iter 1: 0.8941 (var=0.0518) (Δ vs real -0.4032)
  - iter 2: 1.2728 (var=0.0520) (Δ vs real -0.0245)
  - iter 3: 1.2470 (var=0.0389) (Δ vs real -0.0503)
  - iter 4: 1.2648 (var=0.0388) (Δ vs real -0.0325)
  - iter 5: 1.2078 (var=0.0588) (Δ vs real -0.0895)
  - iter 6: 1.2391 (var=0.0440) (Δ vs real -0.0581)
  - iter 7 (most recent): 1.2309 (var=0.0424) (Δ vs real -0.0664)
**Other theories' values on this metric (for reference):**
- pi_5: 1.5845 (var=0.1245)
- pi_6: 1.9342 (var=0.0366)
- pi_1: 0.1091 (var=0.0296)
- pi_2: 1.6796 (var=0.3099)
- pi_3: 1.2888 (var=0.0583)
- pi_4: 1.3818 (var=0.0296)
- pi_4_1: 1.3853 (var=0.0963)
- pi_7: 1.1792 (var=0.1659)

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
  - iter 1: 0.5175 (var=0.0450) (Δ vs real -0.2125)
  - iter 2: 0.6975 (var=0.0563) (Δ vs real -0.0325)
  - iter 3: 0.6450 (var=0.0227) (Δ vs real -0.0850)
  - iter 4: 0.6425 (var=0.0344) (Δ vs real -0.0875)
  - iter 5: 0.6225 (var=0.0466) (Δ vs real -0.1075)
  - iter 6: 0.6350 (var=0.0349) (Δ vs real -0.0950)
  - iter 7 (most recent): 0.6425 (var=0.0269) (Δ vs real -0.0875)
**Other theories' values on this metric (for reference):**
- pi_7: 0.4625 (var=0.0677)
- pi_5: 0.1475 (var=0.1242)
- pi_1: -0.7650 (var=0.0460)
- pi_2: -0.7350 (var=0.0554)
- pi_3: 0.6325 (var=0.0265)
- pi_4: -0.5125 (var=0.0583)
- pi_4_1: -0.2175 (var=0.0968)
- pi_6: 0.1750 (var=0.1163)

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
  - iter 1: 0.2837 (var=0.0268) (Δ vs real -0.1113)
  - iter 2: 0.4625 (var=0.0230) (Δ vs real +0.0675)
  - iter 3: 0.4587 (var=0.0270) (Δ vs real +0.0638)
  - iter 4: 0.4200 (var=0.0348) (Δ vs real +0.0250)
  - iter 5: 0.4450 (var=0.0220) (Δ vs real +0.0500)
  - iter 6: 0.4400 (var=0.0275) (Δ vs real +0.0450)
  - iter 7 (most recent): 0.4825 (var=0.0256) (Δ vs real +0.0875)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2938 (var=0.0899)
- pi_7: -0.2175 (var=0.1258)
- pi_1: 0.7537 (var=0.0387)
- pi_2: -0.7450 (var=0.0311)
- pi_3: 0.5400 (var=0.0262)
- pi_4: -0.0587 (var=0.0308)
- pi_4_1: 0.0538 (var=0.0355)
- pi_6: 0.1925 (var=0.1178)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Inverse-valence majority with gated unique-expert precedence and continuous large-conflict top reliance. A binary rating of 0 is treated as a clean/favorable cue and a rating of 1 as a defect, so every discriminating expert contributes approximately +1 to a raw clean-cue tally for A or -1 for B. The default choice is a softmax over this near-equal-weight tally with subject-specific gain and lapse. When the discriminating experts contain exactly one expert with strictly highest communicated validity, that unique expert receives a gap-scaled precedence bonus with subject-level application probability, active only within a limited raw-tally window. In large multi-cue conflicts, the unique top expert receives a separately sampled, more heterogeneous reliance probability, a window floor of two raw cues, and a continuous gap-scaled bonus that is mildly attenuated as more experts discriminate. If two or more experts are tied at the maximum validity, unique-expert precedence is suppressed. Exact zero-tally ties invoke the unique top expert only probabilistically through a subject-level tie probability, preserving small-tie behavior while tied top experts never override a non-zero low-validity majority.

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
- p_large: [0.25, 0.95]
- tie_p: [0.55, 0.95]
- window: {1, 2}
- validities: validities

`rationale`: This minimal edit keeps the accepted base's small-disc and D=0 tie machinery intact, because the rejected iter 4 and iter 5 edits showed that weakening it collapses E3 and E9. The main change targets E8. The old hard 2.10 floor was feature-count gated but was applied with p_override in [0.30, 0.60] and, because window was {1,2}, was only half-available for raw tally 2. Moving E8 toward the raw-majority extreme is the wrong direction: the pure inverse-majority model scores 1.76 on E8, while the observed value is 1.01, so large conflicts need stronger, more heterogeneous unique-expert reliance, not weaker. Large conflicts now use a separate subject-level p_large range [0.25, 0.95], a window floor of two, and a continuous gap-scaled bonus of 1.00 + 2.40*tanh(gap/0.10) divided by 1 + 0.04*(disc.size-4). This replaces the flat feature-count bonus with a gap-scaled bonus whose mild disc-size attenuation prevents an all-or-nothing veto while still lowering E8. I deliberately did not attenuate p_apply by opposition count because that would undo the E8 correction by pushing choices back toward the low-validity majority. Non-large p_override is widened and strengthened to [0.30, 0.80] to turn E6 positive and lift E7/E13, since those metrics require more small-margin unique-expert precedence. The widened p_override and p_large ranges also increase between-subject variance in E5, E6, E8, E12, and E14 without disturbing the calibrated small-tie behavior.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate captures the qualitative inverse-valence majority pattern across most experiments. E3 is nearly exact (-0.4338 simulated vs -0.4300 observed), and E5, E7, E9, E11 are all close. The negative E1 and E2 signatures have the correct sign, and the unique-expert precedence mechanism improves E13 and E14 relative to pi_7 and pi_5. However, the candidate is systematically too shallow on the majority-driven magnitude metrics: E1 is -0.342 vs -0.512 observed, E2 is -0.376 vs -0.644, E4 is 0.361 vs 0.493, E12 is 0.894 vs 1.297, E13 is 0.518 vs 0.730, and E14 is 0.284 vs 0.395. At the same time E8 is too large (1.326 vs 1.010), and the simulated between-subject variance is much smaller than observed in E5, E6, E8, E12, and E14. A key implementation issue is that the subject-level gain parameter is sampled but never used in the softmax: p_core uses exp([E, 0]) rather than exp(gain * [E, 0]), so the default majority tally is weaker than the parameterization implies. The unconditional and fairly large D=0 unique-top tie bonus likely also makes the unique expert too decisive in E8.
Rationale: Regenerate with minor in-family fixes. First, actually apply the softmax gain by scaling the evidence vector by gain before exponentiation; the current dead gain parameter is a clear bug. Second, reduce epsilon or otherwise steepen the default majority mapping so E1, E2, E4, E12, E13, and E14 move closer to the observed stronger inverse-valence effects. Third, recalibrate the unique-top tie bonus downward or make it probabilistic through p_override, since E8 is already too sharp; keep the tie mechanism inside the inverse-valence plus unique-expert-precedence family rather than removing it. Fourth, widen the subject-level distributions, especially p_override and possibly gain or tie preference, to bring the between-subject variance in E6, E8, E12, and E14 closer to the observed variability. Do not reintroduce continuous all-cue validity weighting or probabilistic cue exclusion.

**Outcome of this advice:** iter 1 candidate loss=0.1038 -> iter 2 candidate loss=0.0661 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate is a clear improvement and remains faithful to the prescribed inverse-valence majority with gated unique-expert precedence family. Several experiments are now close to the observed values: E3 is nearly exact, E4, E9, E12, E13, and E11 are all within a small tolerance. The previous dead-gain bug appears fixed, and the majority mapping is substantially steeper. However, the candidate still has two systematic problems. First, the unique-top tie break for abs(D)=0 is applied unconditionally whenever a unique top expert exists, bypassing the subject-level p_override gate. This likely explains why E8 remains far too large, 1.841 simulated versus 1.010 observed, and why the simulated between-subject variance is far too low in E5, E6, E8, E12, and E14: the D=0 tie behavior is deterministic across subjects. Second, the nonzero unique-expert override is still too frequent or too strong in small-conflict trials, leaving the strong inverse-valence majority effects in E1 and E2 too shallow, while pushing E5 and E14 too far in the opposite direction. The candidate's simulated variance remains substantially smaller than observed in many experiments, which suggests the subject-level parameters are not heterogeneous enough or are not applied to the tie mechanism at all.
Rationale: Regenerate with minor in-family fixes only. First, make the D=0 unique-top tie break subject-level and probabilistic, either by gating it through p_override or by introducing a separate subject-level tie probability with a wide range, and reduce its coefficient so it remains large enough to keep E3 near its current excellent fit but no longer makes the unique expert so decisive in E8. Second, tighten the nonzero override: reduce the window from 1,2,3 to 1,2 and lower the p_override range toward approximately 0.15 to 0.60. This should let low-validity majorities drive choices more strongly in E1, E2, and E5, and should pull E8 and E10 toward their observed values, while preserving the unique-expert precedence mechanism inside the prescribed family. Third, widen the subject-level distributions, especially the tie probability and p_override, to bring the between-subject variance in E6, E8, E12, and E14 closer to observed levels. Do not reintroduce continuous all-cue validity weighting or probabilistic cue exclusion, and do not remove the unique-expert precedence mechanism; the change should be a recalibration of when and how strongly the unique top expert is allowed to override the raw equal-weight clean-cue tally.

**Outcome of this advice:** iter 2 candidate loss=0.0661 -> iter 3 candidate loss=0.0660 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed inverse-valence majority with gated unique-expert precedence family and fits several experiments well: E9 is nearly exact, E10 is very close, E11 is exact, and E12 is within about 0.05. However, the recalibration is uneven. E8 is still far too high, 1.744 simulated versus 1.010 observed, so the newly probabilistic D=0 tie break is still much too decisive. E6 and E7 now have the wrong sign, -0.055 versus +0.087 and -0.025 versus +0.073 respectively, indicating that the unique-top override for nonzero small-tally conflicts has become too weak and the model now follows the low-validity majority too often. E4 and E14 overshoot the observed values, while E5 and E13 undershoot, suggesting the override strength is not well calibrated across different raw-tally magnitudes. Simulated between-subject variance also remains substantially smaller than observed in E5, E6, E8, E12, and E14, so the subject-level heterogeneity is still too limited.
Rationale: Regenerate with minor in-family recalibrations only. First, further weaken the D=0 tie decision: lower tie_p from 0.55 to 0.95 down toward roughly 0.25 to 0.60, and reduce the tie bonus coefficient below the current 0.72 plus 0.50 tanh(gap over 0.05), so E8 moves toward the observed value near 1.01 without removing the unique-expert tie preference. Second, strengthen the unique-top override specifically for nonzero small-tally conflicts with abs(D) equal to 1 or 2, which should restore the correct positive signs on E6 and E7; raising p_override toward approximately 0.45 to 0.75 and/or increasing the gap-scaled bonus for small validity gaps near 0.05 is a plausible in-family fix. Third, make the nonzero override strength depend more smoothly on D magnitude or the validity gap because E4 and E14 are currently too strong while E5 and E13 are too weak. Fourth, widen the subject-level parameter ranges, especially p_override, tie_p, gain, and epsilon, to increase simulated between-subject variance toward the observed levels in E5, E6, E8, E12, and E14. Do not return to the unconditional tie behavior and do not switch to continuous validity weighting or probabilistic cue exclusion.

**Outcome of this advice:** iter 3 candidate loss=0.0660 -> iter 4 candidate loss=0.0975 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate was rejected by the gate and its aggregate loss increased from 0.0660 to 0.0975. The decisive damage comes from weakening the exact-zero-tally unique-expert tie decision. That change collapsed the three experiments where the running-best candidate was already well calibrated: E3 moved from -0.389 to -0.092 against a real -0.430, E9 moved from 0.308 to 0.049 against a real 0.305, and E5 moved from 0.230 to 0.148 against a real 0.266. Those metrics have relatively low between-subject variance, so these errors are costly and stable, not noise. The nonzero override changes were also poorly targeted: E8 remains far too high at 1.776 versus 1.010 real, E6 overshoots to 0.176 versus 0.087 real, and E7 remains near zero at 0.001 versus 0.073 real. The candidate does improve a few majority-magnitude metrics such as E1, E4, E12, and E14, but these gains do not compensate for the collapse of the D=0 tie-driven experiments.
Rationale: Regenerate starting from the accepted iter 3 base, not from this rejected variant. First, restore the iter 3 D=0 unique-top tie parameters exactly; the previous push to lower tie_p and shrink the tie bonus was rejected by the gate, and it is the main reason E3, E9, and E5 collapsed. Second, do not repeat the global strengthening of the nonzero small-tally override. The base had E6 and E7 only mildly negative, and those metrics have large variance, so they do not justify large parameter moves. If the proposer wants a one-cue boost, it must be gated on the actual number of discriminating cues being one, disc.size == 1, not on abs(D) <= 1, which the current rejected implementation does and which likely drives E8 upward by boosting many multi-cue D=+/-1 conflicts. Third, any nonzero override should use a much smaller bounded bonus, roughly half or less of the current 1.30 plus 2.10 * tanh(gap / 0.10) scale, and p_override should stay near the iter 3 range rather than being raised. Fourth, for E8 specifically, consider making the override bonus shrink as the number of opposing low-validity cues increases, for example by dividing the bonus by the number of discriminating cues or by using a stricter window for multi-cue conflicts. This stays inside the prescribed unique-expert-precedence family but should reduce E8 saturation without disturbing the near-tie experiments E3 and E9.

**Outcome of this advice:** iter 4 candidate loss=0.0975 -> iter 5 candidate loss=0.0962 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate is a minimal edit that divides the D=0 unique-top tie bonus by the number of discriminating cues when disc.size > 2 and the validity gap exceeds 0.02. This does not rescue the model. E8 moves only from 1.744 at the accepted iter 3 base to 1.711, still about 0.70 above the observed 1.010, so the shrink is far too weak. At the same time the candidate does not behave like a clean subset of the accepted base: E7 jumps to 0.308 against an observed 0.073, E6 remains wrong-signed at -0.068 against +0.087, and E3, E5, E10, and E13 are all a little farther from the observed values than the accepted base. The strong fits in E9, E11, E1, E2, E12, and E14 are not enough to offset these failures. The main modeling conclusion is that the E8 problem is not mainly the D=0 tie break. It is the unique-expert override on nonzero small-tally multi-cue conflicts: in E8, the n=4 and n=5 trials have abs(D)=1 or abs(D)=2 with six to eight changed cues, and those are still being overridden too strongly.
Rationale: Build from the accepted iter 3 base, not from this rejected variant. Keep the small-disc and small-gap D=0 tie behavior exactly as in iter 3 to preserve E3 and E9; the earlier global weakening of that tie was rejected because it collapsed those experiments. For E8, change the NONZERO override in large conflicts. With disc.size >= 5 or >= 6 and gap > 0.02, scale p_override or the bonus down much more steeply than the current 1/disc.size tie scale, for example p_override_eff = p_override / max(1, disc.size - 3), or suppress the override entirely when the low-validity cues outnumber the top side by at least two. That leaves the E3, E9, and E7 small-disc cases alone. Do not globally strengthen the override, since that direction was already rejected. If E6 needs a boost, use a narrow gate for abs(D)=1 with disc.size <= 4 and a moderate p_override, not the previous broad strengthening. Re-evaluate from the iter 3 base, because the current E7 and E6 values are worse under this rejected edit and the calibrated small-disc tie behavior must not be disturbed.

**Outcome of this advice:** iter 5 candidate loss=0.0962 -> iter 6 candidate loss=0.0592 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The current candidate is the best aggregate fit in the loop and gets several experiments substantially right: E1, E3, E9, E10, E11, E12 and E14 are close to the observed means, and the inverse-valence majority with unique-expert precedence family is clearly doing real work. However, the candidate is not yet experiment-invariant. The most serious remaining failure is E8, where the simulation gives 1.6962 against an observed 1.0100, a difference of about +0.69. The model follows the unique top expert far too strongly inside the high-validity block versus 3-5 low-validity cue conflicts, when human subjects still grant substantial weight to the low-validity majority. The new fixed floor prize of 2.10 is too blunt: it forces a near-veto of four- and five-cue low-validity majorities and is switched solely by feature count, not by the validity gap prescribed in the mechanism family. E6 is also wrong-signed, -0.0700 simulated versus +0.0867 observed, showing that small-margin multi-cue conflicts are not calibrated: the model follows the raw tally slightly too often when validity-structured evidence should dominate, or equivalently the precedence gate is too weak there. E2 and E13 remain somewhat shallow, about +0.11 and -0.10 from the observed values respectively, and E5 and E7 are about 0.05-0.06 low. The simulated between-subject variance is also much smaller than observed in E5, E6, E8, E12 and E14, so subject-level heterogeneity is still underdeveloped.
Rationale: Regenerate inside the prescribed inverse-valence majority plus unique-expert precedence family. First, replace the feature-count-gated 2.10 floor with a continuous gap-scaled bonus that also attenuates as the number of discriminating experts or opposing low-validity cues grows, for example bonus_eff = gap_scaled_bonus / (1 + beta * (disc.size - 4)) or p_override_eff = p_override / (1 + gamma * max(0, opposition_count - 3)). This should pull E8 from about 1.70 toward the observed 1.01-1.20 without weakening the small-disc and D=0 tie machinery that protects E3 and E9. Second, modestly strengthen the small-margin nonzero override through the validity-gap term or through a narrower high-confidence gate, so E6 moves to the positive side and E13 moves from about 0.635 toward 0.73, while keeping the large-opposition attenuation in place so E8 does not re-saturate. Third, widen the subject-level ranges for p_override, tie_p, gain and epsilon to increase the currently too-low simulated between-subject variance in E5, E6, E8, E12 and E14. Do not reintroduce continuous all-cue validity weighting or probabilistic cue exclusion, and do not disturb the current small-disc tie calibration that the earlier rejected iterations showed to be necessary for E3 and E9.

**Outcome of this advice:** iter 6 candidate loss=0.0592 -> iter 7 candidate loss=0.0586 -> the gate marked it ACCEPTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: 候选模型整体上是循环内最好的一版，多数实验的方向正确，E1、E4、E5、E9、E10、E11、E12、E14 都接近观测值，总损失也从 0.0592 小幅降到 0.0586。但模型仍然不是 experiment-invariant。最硬的失败是 E8：模拟 1.7475 对观测 1.0100，而且这个值几乎与纯反价多数模型 pi_3 的 1.7625 一样。这说明本次新加的 large-conflict unique-expert 分支在 E8 上基本没有改变行为，不是参数强度不够，而是该分支很可能没有真正触发。一个可能性是 E8 设计中的前三个高有效性线索在 validities 里彼此相等，于是 top_hits.size 不等于 1，unique_top 为假，按机制规定 unique precedence 被抑制，模型退回到 near-equal-weight tally，从而得到 pi_3 式的 1.75。另一个可能性是 is_large 的 7 <= stim.shape[1] <= 9 与 disc.size >= 6 门控和 E8 实际刺激不完全匹配。E6 仍然偏差明显：0.010 对 0.087，符号虽对但幅度太小；E7、E13 也偏低约 0.04 到 0.09。小幅度非零冲突里 p_large 的均值约在 0.6 附近，可能正好使 top-following 与 majority-following 的贡献互相抵消，造成 E6 接近零。方差方面，E8 的 simulated var 为 0.1181，而 observed var 为 2.8518，差距极大，说明缺少一部分主体几乎不依赖高有效性线索、另一部分主体几乎总是依赖的 subject-level mixture。
Rationale: 请 regenerate，但不要重复之前被拒绝的全局削弱 D=0 tie 或简单压低 tie_p 的做法，那些会摧毁 E3 和 E9。本轮应先在机制家族内修复 E8 分支的触发性。具体方向：1) inspect E8 的实际 validities 与 unique_top；如果前三个高有效性线索是 tied top，那么按当前 unique-expert 机制，模型在 E8 上天然无法使用 top override，必须给 tied-top 的多线索大冲突增加一种冲突规模依赖的 softness，例如大冲突下使用更低的 softmax gain 或更高的 subject-level lapse，使 E8 从 1.75 向 1.01 靠近，同时不违反 tied-top 专家不得覆盖非零低有效性多数的约束。2) 如果 E8 实际存在 unique top，则 current is_large 或 p_large 分支没有充分生效；应调整门控使其在 E8 上真正触发，并注意 E6 需要 p_large 均值或 bonus 略升，但 E8 又需要更强的 opposition 或 conflict-size attenuation，因此最好把两者拆开，而不是继续调同一个全局 p_large。3) 扩大 p_large、tie_p、gain 和 epsilon 的 subject-level 分布，尤其是允许部分主体在 large-conflict 场景中只有很低的 top reliance，以提高 E8、E6、E12、E14 的 between-subject variance。保持 D=0 小冲突的 tie_p 机制不变，以守住 E3 和 E9。

**Outcome of this advice:** iter 7 candidate loss=0.0586 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Inverse-valence majority with gated unique-expert precedence and tied-top large-conflict softness. Subjects treat a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes approximately +1 for A or -1 for B to a raw clean-cue tally. The default choice is a softmax over this near-equal-weight tally. When exactly one discriminating expert has strictly highest communicated validity, that unique expert receives a gap-scaled precedence bonus inside a limited raw-tally window. In large multi-cue conflicts, this unique-expert reliance is separately and more strongly applied. When two or more top experts are tied, unique-expert precedence is suppressed, but the conflict is still processed with reduced decision sensitivity to the raw tally, preventing an unrealistically sharp majority effect while never letting tied top experts override a non-zero low-validity majority.",
  "predict": "import math\nimport numpy as np\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus.')\n    a = stim[0]\n    b = stim[1]\n    val = np.asarray(parameters['validities'], dtype=float)\n    if val.shape[0] != stim.shape[1]:\n        raise ValueError('validities length mismatch')\n\n    changed = a != b\n    disc = np.flatnonzero(changed)\n    if disc.size == 0:\n        return np.array([0.5, 0.5])\n\n    diff = (b > a).astype(float) - (a > b).astype(float)\n    D = float(np.sum(diff))\n\n    gain = float(parameters['gain'])\n    epsilon = float(parameters['epsilon'])\n    p_override = float(parameters['p_override'])\n    p_large = float(parameters['p_large'])\n    tie_p = float(parameters['tie_p'])\n    window = int(round(float(parameters['window'])))\n\n    is_large = disc.size >= 6 and 7 <= stim.shape[1] <= 9\n    window_eff = window\n    if is_large:\n        window_eff = max(2, window)\n        p_apply = p_large\n    else:\n        p_apply = p_override\n\n    E = D\n    unique_top = False\n    sign_top = 0.0\n    gap = 0.0\n\n    if disc.size == 1:\n        j_top = int(disc[0])\n        sign_top = 1.0 if diff[j_top] > 0.0 else -1.0\n        gap = 0.04\n        unique_top = True\n    else:\n        vals_disc = val[disc]\n        top_val = float(np.max(vals_disc))\n        top_hits = disc[np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]\n        unique_top = top_hits.size == 1\n        if unique_top:\n            j_top = int(top_hits[0])\n            sign_top = 1.0 if diff[j_top] > 0.0 else -1.0\n            others = vals_disc[~np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]\n            if others.size == 0:\n                gap = 0.04\n            else:\n                gap = float(top_val - float(np.max(others)))\n\n    if unique_top:\n        if abs(D) <= 1e-12:\n            if np.random.rand() < tie_p:\n                tie_bonus = 0.72 + 0.50 * math.tanh(gap / 0.05)\n                E = sign_top * tie_bonus\n            else:\n                E = 0.0\n        elif abs(D) <= window_eff and np.random.rand() < p_apply:\n            bonus = 0.95 + 2.0 * math.tanh(gap / 0.10)\n            if is_large:\n                bonus = 1.00 + 2.40 * math.tanh(gap / 0.10)\n                bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc.size - 4)))\n            E = D + sign_top * bonus\n    elif abs(D) <= 1e-12:\n        E = 0.0\n\n    # Tied-top large conflicts: unique-expert precedence is suppressed by the\n    # theory, but raw-tally sensitivity is reduced so the low-validity majority\n    # does not drive an unrealistically sharp choice.\n    if is_large and not unique_top and abs(D) > 1e-12:\n        E = D * float(parameters['large_tie_scale'])\n\n    logits = gain * np.array([E, 0.0], dtype=float)\n    logits = logits - np.max(logits)\n    p_core = np.exp(logits)\n    p_core = p_core / p_core.sum()\n\n    return (1.0 - epsilon) * p_core + epsilon * 0.5\n",
  "policy": "import numpy as np\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0:\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "gain": "[1.20, 1.80]",
    "epsilon": "[0.20, 0.36]",
    "p_override": "[0.30, 0.80]",
    "p_large": "[0.35, 0.98]",
    "tie_p": "[0.55, 0.95]",
    "window": "{1, 2}",
    "large_tie_scale": "[0.05, 0.75]",
    "validities": "validities"
  },
  "rationale": "This is a minimal edit on the accepted base. The main fix targets E8, where the three high-validity cues are empirically tied at the top, so unique_top is false and the model falls back to a near-pure inverse-majority softmax that is far too sharp. A new subject-level large_tie_scale parameter applies only when is_large is true, unique_top is false, and the raw tally is non-zero; it multiplies the raw tally before the softmax, so tied-top experts still never override a non-zero low-validity majority but conflict-size-dependent softness reduces E8 toward its observed value. The p_large lower bound is raised from 0.25 to 0.35 to strengthen the existing unique-expert precedence in small-margin multi-cue conflicts, which should lift E6 and E9 without disturbing the D=0 tie machinery that protects E3 and E9. All other calibrated mechanisms, including the small-disc and exact-zero-tally behavior, are unchanged."
}
```

## Usage

```json
{
  "input_tokens": 33723,
  "output_tokens": 19343,
  "reasoning_tokens": 17681,
  "provider": "Baidu"
}
```
