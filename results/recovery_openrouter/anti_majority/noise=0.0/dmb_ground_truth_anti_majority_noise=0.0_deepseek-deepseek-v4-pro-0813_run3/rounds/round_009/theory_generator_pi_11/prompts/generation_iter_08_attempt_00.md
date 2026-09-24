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
- THEORY 1 = `pi_10`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 2 (= `pi_9`).

Propose a new theory in which subjects combine a damped validity-weighted inverse-valence tally with a continuously active, smoothly attenuated top-coalition bonus. Unlike pi_9, the top-coalition effect must not be gated by |D|<=window or abs(E)<=tie_tol; it should survive nonzero D at modest strength, so E1 top-following is partial (~0.30-0.40, not 0.145) and E20's h-trend is large (~3.4). Unlike pi_10, the tied-top component should be much weaker and saturating: no large m=2 multiplier of 2.7, no extreme m-based amplifications, and no sharp suppression for three-cue top groups facing five lower cues; it should increase gradually with top-group size and validity gap, decay smoothly with opposition and net tally, and use tanh-type saturation so E2 stays near 0.09 and E16 near 0.40. Unique-top precedence should be probabilistic and still available in large multi-cue conflicts but with conflict-dependent damping that keeps it partial rather than ceiling-level. Large-conflict validity weighting should follow pi_9's damped-lambda style rather than pi_10's boost, letting raw majority dominate when |D| is large while the small top-coalition bonus partially counters it. Finally, subject-level parameters should remain heavy-tailed but with enough variance in the top-coalition and precedence components to reproduce the observed between-subject variances (e.g., E1 ~0.05, E8 ~0.42, E16 ~0.33), avoiding pi_9's near-zero population variance.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_10` (overall score: 0.825)

**Description**
Subjects code a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, but large conflicts with a unique top expert receive amplified validity weighting. When highest-validity experts are tied, their clean-cue advantage is a continuous, always-active evidence component that scales with top-group size, validity gap, conflict size, and clean-cue direction, but the component is capped at |D|=1 and sharply suppressed for three-cue top groups facing five opposing lower-validity cues, so lower-validity majorities can override the tied top group in large conflicts. Unique top-expert precedence is strong in small and sparse conflicts but decays in dense conflicts and is further damped in six-feature full conflicts with |D|>=2. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain, lapse, validity-weight sensitivity, precedence probability, and top-group sensitivity.

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

    df_t = 1.6

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
        math.log(12.0), 1.80, math.log(0.3), math.log(35.0), df_t)))

    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    is_large_weight = disc_size >= 4 and n_features >= 7
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
                if n_features == 8:
                    tie_bonus *= 1.03
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
                        prec_str_eff = prec_str * 0.40
                    elif disc_size >= n_features - 1:
                        prec_str_eff = prec_str * 0.78
                if n_features == 6 and disc_size <= 4:
                    prec_str_eff *= 1.45
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

            d_factor = tie_d0 + (1.0 - tie_d0) * math.tanh(min(abs(D), 1.0))

            tie_str_eff = tie_str
            if m == 2:
                tie_str_eff *= 2.70
            elif m == 3:
                if lower_count <= 4:
                    if lower_count == 3:
                        tie_str_eff *= 2.70
                    else:
                        tie_str_eff *= 2.20
                else:
                    steep_gate = 1.0 / (1.0 + math.exp((abs(D) - 1.5) / 0.2))
                    tie_str_eff *= steep_gate
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.5117 (var=0.0161) vs this=-0.5287 (var=0.0134)
- Experiment 2: real=-0.6438 (var=0.0430) vs this=-0.5869 (var=0.0358)
- Experiment 3: real=-0.4300 (var=0.0221) vs this=-0.4000 (var=0.0432)
- Experiment 4: real=0.4925 (var=0.0658) vs this=0.6331 (var=0.0232)
- Experiment 5: real=0.2658 (var=0.1584) vs this=0.0975 (var=0.0284)
- Experiment 6: real=0.0867 (var=0.4244) vs this=0.1267 (var=0.1923)
- Experiment 7: real=0.0725 (var=0.0580) vs this=0.0213 (var=0.0926)
- Experiment 8: real=1.0100 (var=2.8518) vs this=1.3738 (var=0.2713)
- Experiment 9: real=0.3050 (var=0.0605) vs this=0.2175 (var=0.0401)
- Experiment 10: real=-0.1667 (var=0.0053) vs this=-0.1847 (var=0.0032)
- Experiment 11: real=0.0000 (var=0.0045) vs this=0.0000 (var=0.0065)
- Experiment 12: real=1.2973 (var=0.2536) vs this=1.4052 (var=0.1406)
- Experiment 13: real=0.7300 (var=0.0558) vs this=0.6250 (var=0.1075)
- Experiment 14: real=0.3950 (var=0.3296) vs this=0.1500 (var=0.1973)
- Experiment 15: real=0.0608 (var=0.0035) vs this=0.0120 (var=0.0004)
- Experiment 16: real=0.2492 (var=0.0989) vs this=0.3746 (var=0.0795)
- Experiment 17: real=0.0608 (var=0.0012) vs this=0.0545 (var=0.0013)
- Experiment 18: real=3.4448 (var=3.3537) vs this=3.9261 (var=1.6381)
- Experiment 19: real=0.3422 (var=0.0518) vs this=0.3828 (var=0.0886)
- Experiment 20: real=0.0900 (var=0.0603) vs this=0.3738 (var=0.1102)


---

### `pi_9` (overall score: 0.819)

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
- Experiment 19: real=0.3422 (var=0.0518) vs this=0.1447 (var=0.0036)
- Experiment 20: real=0.0900 (var=0.0603) vs this=0.0181 (var=0.0090)


---

### `pi_6` (overall score: 0.768)

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
- Experiment 15: real=0.0608 (var=0.0035) vs this=0.0698 (var=0.0043)
- Experiment 16: real=0.2492 (var=0.0989) vs this=0.4467 (var=0.1357)
- Experiment 17: real=0.0608 (var=0.0012) vs this=0.0665 (var=0.0059)
- Experiment 18: real=3.4448 (var=3.3537) vs this=2.1446 (var=13.2383)
- Experiment 19: real=0.3422 (var=0.0518) vs this=0.3608 (var=0.0509)
- Experiment 20: real=0.0900 (var=0.0603) vs this=0.0394 (var=0.0047)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1332 -> ACCEPTED
- iter 2: loss=0.0699 -> ACCEPTED
- iter 3: loss=0.0996 -> REJECTED
- iter 4: loss=0.1074 -> REJECTED
- iter 5: loss=0.0884 -> REJECTED
- iter 6: loss=0.1378 -> REJECTED
- iter 7: loss=0.1384 -> REJECTED
- iter 8: loss=0.0841 -> REJECTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.0699 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.5367 (var=0.0178) (Δ vs real -0.0250)
  - iter 2: -0.5429 (var=0.0111) (Δ vs real -0.0313)
  - iter 3: -0.4821 (var=0.0155) (Δ vs real +0.0296)
  - iter 4: -0.5354 (var=0.0112) (Δ vs real -0.0238)
  - iter 5: -0.4812 (var=0.0131) (Δ vs real +0.0304)
  - iter 6: -0.5042 (var=0.0107) (Δ vs real +0.0075)
  - iter 7: -0.4767 (var=0.0174) (Δ vs real +0.0350)
  - iter 8 (most recent): -0.5042 (var=0.0166) (Δ vs real +0.0075)
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
- pi_10: -0.5287 (var=0.0134)

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
  - iter 1: -0.6731 (var=0.0315) (Δ vs real -0.0294)
  - iter 2: -0.5919 (var=0.0291) (Δ vs real +0.0519)
  - iter 3: -0.5487 (var=0.0432) (Δ vs real +0.0950)
  - iter 4: -0.5737 (var=0.0433) (Δ vs real +0.0700)
  - iter 5: -0.5431 (var=0.0385) (Δ vs real +0.1006)
  - iter 6: -0.6206 (var=0.0263) (Δ vs real +0.0231)
  - iter 7: -0.5056 (var=0.0542) (Δ vs real +0.1381)
  - iter 8 (most recent): -0.5350 (var=0.0321) (Δ vs real +0.1088)
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
- pi_10: -0.5869 (var=0.0358)

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
  - iter 1: -0.3133 (var=0.0360) (Δ vs real +0.1167)
  - iter 2: -0.3254 (var=0.0497) (Δ vs real +0.1046)
  - iter 3: -0.3912 (var=0.0352) (Δ vs real +0.0388)
  - iter 4: -0.3796 (var=0.0583) (Δ vs real +0.0504)
  - iter 5: -0.4983 (var=0.0440) (Δ vs real -0.0683)
  - iter 6: -0.5537 (var=0.0282) (Δ vs real -0.1237)
  - iter 7: -0.3900 (var=0.0374) (Δ vs real +0.0400)
  - iter 8 (most recent): -0.4479 (var=0.0512) (Δ vs real -0.0179)
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
- pi_10: -0.4000 (var=0.0432)

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
  - iter 1: 0.6588 (var=0.0260) (Δ vs real +0.1663)
  - iter 2: 0.6269 (var=0.0349) (Δ vs real +0.1344)
  - iter 3: 0.4925 (var=0.0433) (Δ vs real +0.0000)
  - iter 4: 0.6100 (var=0.0361) (Δ vs real +0.1175)
  - iter 5: 0.5250 (var=0.0300) (Δ vs real +0.0325)
  - iter 6: 0.5931 (var=0.0327) (Δ vs real +0.1006)
  - iter 7: 0.5075 (var=0.0517) (Δ vs real +0.0150)
  - iter 8 (most recent): 0.5644 (var=0.0328) (Δ vs real +0.0719)
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
- pi_10: 0.6331 (var=0.0232)

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
  - iter 1: 0.0379 (var=0.0219) (Δ vs real -0.2279)
  - iter 2: 0.1321 (var=0.0256) (Δ vs real -0.1337)
  - iter 3: 0.2025 (var=0.0456) (Δ vs real -0.0633)
  - iter 4: 0.0667 (var=0.0286) (Δ vs real -0.1992)
  - iter 5: 0.2583 (var=0.0498) (Δ vs real -0.0075)
  - iter 6: 0.1883 (var=0.0294) (Δ vs real -0.0775)
  - iter 7: 0.1542 (var=0.0354) (Δ vs real -0.1117)
  - iter 8 (most recent): 0.1313 (var=0.0344) (Δ vs real -0.1346)
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
- pi_10: 0.0975 (var=0.0284)

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
  - iter 1: -0.3650 (var=0.0351) (Δ vs real -0.4517)
  - iter 2: -0.2483 (var=0.0584) (Δ vs real -0.3350)
  - iter 3: -0.0158 (var=0.0737) (Δ vs real -0.1025)
  - iter 4: -0.2542 (var=0.0406) (Δ vs real -0.3408)
  - iter 5: -0.0775 (var=0.0967) (Δ vs real -0.1642)
  - iter 6: -0.1367 (var=0.0546) (Δ vs real -0.2233)
  - iter 7: 0.0025 (var=0.0675) (Δ vs real -0.0842)
  - iter 8 (most recent): -0.3225 (var=0.0301) (Δ vs real -0.4092)
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
- pi_10: 0.1267 (var=0.1923)

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
  - iter 1: 0.0400 (var=0.0562) (Δ vs real -0.0325)
  - iter 2: 0.0863 (var=0.0284) (Δ vs real +0.0138)
  - iter 3: 0.0400 (var=0.0478) (Δ vs real -0.0325)
  - iter 4: 0.0888 (var=0.0549) (Δ vs real +0.0163)
  - iter 5: -0.0100 (var=0.0538) (Δ vs real -0.0825)
  - iter 6: -0.0200 (var=0.0530) (Δ vs real -0.0925)
  - iter 7: 0.0725 (var=0.0479) (Δ vs real +0.0000)
  - iter 8 (most recent): 0.0588 (var=0.0508) (Δ vs real -0.0137)
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
- pi_10: 0.0213 (var=0.0926)

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
  - iter 1: 1.6362 (var=1.0526) (Δ vs real +0.6263)
  - iter 2: 0.5838 (var=1.6699) (Δ vs real -0.4262)
  - iter 3: 0.3875 (var=1.4308) (Δ vs real -0.6225)
  - iter 4: -0.2562 (var=2.2310) (Δ vs real -1.2662)
  - iter 5: -0.0687 (var=1.9377) (Δ vs real -1.0787)
  - iter 6: -0.5050 (var=1.1583) (Δ vs real -1.5150)
  - iter 7: -0.0900 (var=1.4078) (Δ vs real -1.1000)
  - iter 8 (most recent): 0.7600 (var=1.3030) (Δ vs real -0.2500)
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
- pi_10: 1.3738 (var=0.2713)

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
  - iter 1: 0.2812 (var=0.0423) (Δ vs real -0.0238)
  - iter 2: 0.3133 (var=0.0408) (Δ vs real +0.0083)
  - iter 3: 0.3771 (var=0.0329) (Δ vs real +0.0721)
  - iter 4: 0.2742 (var=0.0343) (Δ vs real -0.0308)
  - iter 5: 0.4129 (var=0.0512) (Δ vs real +0.1079)
  - iter 6: 0.4538 (var=0.0375) (Δ vs real +0.1488)
  - iter 7: 0.2596 (var=0.0270) (Δ vs real -0.0454)
  - iter 8 (most recent): 0.3633 (var=0.0391) (Δ vs real +0.0583)
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
- pi_10: 0.2175 (var=0.0401)

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
  - iter 1: -0.1423 (var=0.0027) (Δ vs real +0.0244)
  - iter 2: -0.1657 (var=0.0026) (Δ vs real +0.0009)
  - iter 3: -0.1906 (var=0.0031) (Δ vs real -0.0240)
  - iter 4: -0.1738 (var=0.0033) (Δ vs real -0.0071)
  - iter 5: -0.1734 (var=0.0031) (Δ vs real -0.0067)
  - iter 6: -0.1951 (var=0.0021) (Δ vs real -0.0284)
  - iter 7: -0.2193 (var=0.0015) (Δ vs real -0.0526)
  - iter 8 (most recent): -0.1820 (var=0.0023) (Δ vs real -0.0153)
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
- pi_10: -0.1847 (var=0.0032)

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
  - iter 1: 0.0000 (var=0.0047) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0045) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0055) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0049) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0033) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0033) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.0047) (Δ vs real +0.0000)
  - iter 8 (most recent): 0.0000 (var=0.0036) (Δ vs real +0.0000)
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
- pi_10: 0.0000 (var=0.0065)

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
  - iter 1: 1.5635 (var=0.1851) (Δ vs real +0.2662)
  - iter 2: 1.4297 (var=0.1343) (Δ vs real +0.1324)
  - iter 3: 1.5954 (var=0.1373) (Δ vs real +0.2981)
  - iter 4: 1.4832 (var=0.1837) (Δ vs real +0.1859)
  - iter 5: 1.4731 (var=0.1526) (Δ vs real +0.1758)
  - iter 6: 1.4731 (var=0.1753) (Δ vs real +0.1758)
  - iter 7: 1.5685 (var=0.1767) (Δ vs real +0.2712)
  - iter 8 (most recent): 1.4185 (var=0.1696) (Δ vs real +0.1212)
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
- pi_10: 1.4052 (var=0.1406)

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
  - iter 1: 0.5175 (var=0.1063) (Δ vs real -0.2125)
  - iter 2: 0.5200 (var=0.0752) (Δ vs real -0.2100)
  - iter 3: 0.4050 (var=0.1903) (Δ vs real -0.3250)
  - iter 4: 0.4075 (var=0.1980) (Δ vs real -0.3225)
  - iter 5: 0.4875 (var=0.1127) (Δ vs real -0.2425)
  - iter 6: 0.2525 (var=0.2178) (Δ vs real -0.4775)
  - iter 7: -0.0475 (var=0.3087) (Δ vs real -0.7775)
  - iter 8 (most recent): 0.5950 (var=0.0860) (Δ vs real -0.1350)
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
- pi_10: 0.6250 (var=0.1075)

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
  - iter 1: 0.0787 (var=0.2142) (Δ vs real -0.3163)
  - iter 2: 0.2800 (var=0.0913) (Δ vs real -0.1150)
  - iter 3: 0.0800 (var=0.1789) (Δ vs real -0.3150)
  - iter 4: 0.0225 (var=0.1993) (Δ vs real -0.3725)
  - iter 5: 0.2688 (var=0.0535) (Δ vs real -0.1262)
  - iter 6: 0.0050 (var=0.1989) (Δ vs real -0.3900)
  - iter 7: -0.1862 (var=0.1596) (Δ vs real -0.5812)
  - iter 8 (most recent): 0.3425 (var=0.0788) (Δ vs real -0.0525)
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
- pi_10: 0.1500 (var=0.1973)

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
  - iter 1: 0.0203 (var=0.0010) (Δ vs real -0.0405)
  - iter 2: 0.0601 (var=0.0029) (Δ vs real -0.0007)
  - iter 3: 0.0844 (var=0.0030) (Δ vs real +0.0236)
  - iter 4: 0.0671 (var=0.0035) (Δ vs real +0.0063)
  - iter 5: 0.0723 (var=0.0030) (Δ vs real +0.0115)
  - iter 6: 0.1139 (var=0.0035) (Δ vs real +0.0531)
  - iter 7: 0.0769 (var=0.0024) (Δ vs real +0.0160)
  - iter 8 (most recent): 0.0379 (var=0.0021) (Δ vs real -0.0229)
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
- pi_10: 0.0120 (var=0.0004)

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
  - iter 1: 0.3133 (var=0.0627) (Δ vs real +0.0642)
  - iter 2: 0.2463 (var=0.0596) (Δ vs real -0.0029)
  - iter 3: 0.3108 (var=0.0693) (Δ vs real +0.0617)
  - iter 4: 0.2762 (var=0.0735) (Δ vs real +0.0271)
  - iter 5: 0.2733 (var=0.0655) (Δ vs real +0.0242)
  - iter 6: 0.2846 (var=0.0652) (Δ vs real +0.0354)
  - iter 7: 0.4442 (var=0.0672) (Δ vs real +0.1950)
  - iter 8 (most recent): 0.3204 (var=0.0646) (Δ vs real +0.0713)
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
- pi_10: 0.3746 (var=0.0795)

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
  - iter 1: 0.0410 (var=0.0011) (Δ vs real -0.0198)
  - iter 2: 0.0542 (var=0.0013) (Δ vs real -0.0066)
  - iter 3: 0.0707 (var=0.0011) (Δ vs real +0.0099)
  - iter 4: 0.0481 (var=0.0011) (Δ vs real -0.0127)
  - iter 5: 0.0496 (var=0.0010) (Δ vs real -0.0112)
  - iter 6: 0.0722 (var=0.0014) (Δ vs real +0.0114)
  - iter 7: 0.0809 (var=0.0013) (Δ vs real +0.0201)
  - iter 8 (most recent): 0.0374 (var=0.0010) (Δ vs real -0.0234)
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
- pi_10: 0.0545 (var=0.0013)

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
  - iter 1: 1.5524 (var=1.9218) (Δ vs real -1.8925)
  - iter 2: 3.5746 (var=2.4559) (Δ vs real +0.1298)
  - iter 3: 2.5901 (var=2.4600) (Δ vs real -0.8547)
  - iter 4: 3.2814 (var=2.7353) (Δ vs real -0.1634)
  - iter 5: 3.1113 (var=2.0691) (Δ vs real -0.3335)
  - iter 6: 4.2630 (var=1.7986) (Δ vs real +0.8181)
  - iter 7: 4.0387 (var=2.0093) (Δ vs real +0.5939)
  - iter 8 (most recent): 2.9018 (var=2.8008) (Δ vs real -0.5431)
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
- pi_10: 3.9261 (var=1.6381)

### Experiment 19
**Design**
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
def metric(data):
    a = [row for row in data['option_a_ratings']]
    b = [row for row in data['option_b_ratings']]
    resp = data['response'].to_numpy()
    d_vals = []
    top_choices = []
    for ar, br, r in zip(a, b, resp):
        d = sum([1 if (y > x) else -1 for x, y in zip(ar, br) if x != y])
        d_vals.append(d)
        top_choices.append(((ar[0] == 0) and (br[0] == 1) and (r == 0)) or
                           ((ar[0] == 1) and (br[0] == 0) and (r == 1)))
    d_vals = np.array(d_vals)
    top_choices = np.array(top_choices, dtype=float)
    mask = np.abs(d_vals) == 3
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(top_choices[mask]))
```

**Observed (real) value:** 0.3422 (var=0.0518)
**Previous candidate values (this loop):**
  - iter 1: 0.1236 (var=0.0033) (Δ vs real -0.2186)
  - iter 2: 0.1711 (var=0.0064) (Δ vs real -0.1711)
  - iter 3: 0.1808 (var=0.0083) (Δ vs real -0.1614)
  - iter 4: 0.1789 (var=0.0098) (Δ vs real -0.1633)
  - iter 5: 0.2017 (var=0.0121) (Δ vs real -0.1406)
  - iter 6: 0.1883 (var=0.0128) (Δ vs real -0.1539)
  - iter 7: 0.2350 (var=0.0209) (Δ vs real -0.1072)
  - iter 8 (most recent): 0.1525 (var=0.0093) (Δ vs real -0.1897)
**Other theories' values on this metric (for reference):**
- pi_10: 0.3828 (var=0.0886)
- pi_9: 0.1447 (var=0.0036)
- pi_1: 0.1692 (var=0.0121)
- pi_2: 0.8639 (var=0.0094)
- pi_3: 0.1728 (var=0.0021)
- pi_4: 0.3950 (var=0.0032)
- pi_5: 0.3572 (var=0.0283)
- pi_4_1: 0.2986 (var=0.0101)
- pi_6: 0.3608 (var=0.0509)
- pi_7: 0.1628 (var=0.0101)
- pi_8: 0.1489 (var=0.0022)

### Experiment 20
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    import pandas as pd

    A = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    B = np.asarray(data['option_b_ratings'].tolist(), dtype=float)

    diff = (B > A).astype(float) - (A > B).astype(float)
    D = diff.sum(axis=1)
    top = diff[:, 0] + diff[:, 1]
    chose_a = (data['response'].to_numpy() == 0).astype(float)

    aux = pd.DataFrame({
        'sid': data['subject_id'].to_numpy(),
        'D': D,
        'top': top,
        'chose_a': chose_a,
    })

    def subj_metric(g):
        pp = g.loc[(g['D'] > 0) & (g['top'] > 0), 'chose_a'].mean()
        pn = g.loc[(g['D'] > 0) & (g['top'] < 0), 'chose_a'].mean()
        np_ = g.loc[(g['D'] < 0) & (g['top'] > 0), 'chose_a'].mean()
        nn = g.loc[(g['D'] < 0) & (g['top'] < 0), 'chose_a'].mean()
        return 0.5 * ((pp - pn) + (np_ - nn))

    scores = aux.groupby('sid', sort=False).apply(subj_metric)
    return float(scores.mean())

```

**Observed (real) value:** 0.0900 (var=0.0603)
**Previous candidate values (this loop):**
  - iter 1: 0.0325 (var=0.0121) (Δ vs real -0.0575)
  - iter 2: 0.1375 (var=0.0615) (Δ vs real +0.0475)
  - iter 3: 0.2013 (var=0.0575) (Δ vs real +0.1113)
  - iter 4: 0.2256 (var=0.0793) (Δ vs real +0.1356)
  - iter 5: 0.1981 (var=0.0756) (Δ vs real +0.1081)
  - iter 6: 0.1181 (var=0.0291) (Δ vs real +0.0281)
  - iter 7: 0.1656 (var=0.0412) (Δ vs real +0.0756)
  - iter 8 (most recent): 0.1150 (var=0.0295) (Δ vs real +0.0250)
**Other theories' values on this metric (for reference):**
- pi_9: 0.0181 (var=0.0090)
- pi_10: 0.3738 (var=0.1102)
- pi_1: -0.6606 (var=0.0427)
- pi_2: 0.0037 (var=0.0075)
- pi_3: -0.0150 (var=0.0113)
- pi_4: 0.1319 (var=0.0140)
- pi_5: 0.0819 (var=0.0153)
- pi_4_1: 0.0825 (var=0.0070)
- pi_6: 0.0394 (var=0.0047)
- pi_7: 0.0269 (var=0.0088)
- pi_8: 0.0094 (var=0.0065)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Subjects code a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, with lambda modest but only mildly reduced in large conflicts so high-validity cues still move moderately against raw majorities. A continuously active, smoothly attenuated top-coalition bonus follows tied highest-validity expert groups: it grows approximately linearly with top-group size for small coalitions (m=2-3) but is sharply and smoothly penalized for larger coalitions (m=4+), while opposition from lower-validity cues is damped in a way that scales with top-group size. Unique highest-validity expert precedence is probabilistic and survives large conflicts, but is attenuated when the unique top cue merely agrees with the raw majority. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain and lapse, preserving between-subject heterogeneity.

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

    # ---------- Heavy-tailed subject-level parameters -------------------
    def u_raw(x):
        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))

    df_ht = 1.8

    gain = float(np.exp(_tqt(
        u_raw(parameters['gain_raw']),
        math.log(1.35), 0.55, math.log(0.45), math.log(3.60), df_ht)))

    epsilon = float(_ilogit(_tqt(
        u_raw(parameters['epsilon_raw']),
        _logit(0.20), 0.65, _logit(0.03), _logit(0.55), df_ht)))

    lam = float(np.exp(_tqt(
        u_raw(parameters['lambda_raw']),
        math.log(0.24), 0.75, math.log(0.005), math.log(0.50), df_ht)))

    lam_large_scale = float(_ilogit(_tqt(
        u_raw(parameters['lam_large_scale_raw']),
        _logit(0.85), 0.45, _logit(0.10), _logit(0.98), df_ht)))

    p_prec = float(_ilogit(_tqt(
        u_raw(parameters['prec_p_raw']),
        _logit(0.75), 0.90, _logit(0.06), _logit(0.98), df_ht)))

    tie_p = float(_ilogit(_tqt(
        u_raw(parameters['tie_p_raw']),
        _logit(0.60), 0.75, _logit(0.15), _logit(0.99), df_ht)))

    prec_str = float(np.exp(_tqt(
        u_raw(parameters['prec_strength_raw']),
        math.log(0.95), 0.90, math.log(0.15), math.log(2.80), df_ht)))

    prec_gap_amp = float(np.exp(_tqt(
        u_raw(parameters['prec_gap_amp_raw']),
        math.log(0.90), 0.50, math.log(0.10), math.log(2.20), df_ht)))

    prec_opp_decay = float(np.exp(_tqt(
        u_raw(parameters['prec_opp_decay_raw']),
        math.log(0.50), 0.50, math.log(0.03), math.log(1.20), df_ht)))

    prec_d_decay = float(np.exp(_tqt(
        u_raw(parameters['prec_d_decay_raw']),
        math.log(0.10), 0.60, math.log(0.02), math.log(0.80), df_ht)))

    prec_conf_att = float(np.exp(_tqt(
        u_raw(parameters['prec_conf_att_raw']),
        math.log(0.06), 0.55, math.log(0.01), math.log(0.60), df_ht)))

    align_scale = float(_ilogit(_tqt(
        u_raw(parameters['align_scale_raw']),
        _logit(0.40), 0.60, _logit(0.05), _logit(0.90), df_ht)))

    align_d_att = float(np.exp(_tqt(
        u_raw(parameters['align_d_att_raw']),
        math.log(0.60), 0.60, math.log(0.03), math.log(1.60), df_ht)))

    tie_str = float(np.exp(_tqt(
        u_raw(parameters['tie_strength_raw']),
        math.log(1.10), 1.00, math.log(0.10), math.log(7.00), df_ht)))

    tie_coal_scale = float(np.exp(_tqt(
        u_raw(parameters['tie_coal_scale_raw']),
        math.log(2.80), 0.30, math.log(1.20), math.log(5.00), df_ht)))

    tie_size_amp = float(np.exp(_tqt(
        u_raw(parameters['tie_size_amp_raw']),
        math.log(0.50), 0.45, math.log(0.05), math.log(1.50), df_ht)))

    tie_large_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_large_decay_raw']),
        math.log(3.00), 0.45, math.log(0.80), math.log(9.00), df_ht)))

    tie_gap_amp = float(np.exp(_tqt(
        u_raw(parameters['tie_gap_amp_raw']),
        math.log(0.70), 0.55, math.log(0.05), math.log(2.00), df_ht)))

    tie_opp_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_opp_decay_raw']),
        math.log(0.60), 0.60, math.log(0.03), math.log(1.50), df_ht)))

    tie_opp_size = float(np.exp(_tqt(
        u_raw(parameters['tie_opp_size_raw']),
        math.log(0.80), 0.50, math.log(0.10), math.log(1.80), df_ht)))

    tie_d_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_d_decay_raw']),
        math.log(0.12), 0.60, math.log(0.02), math.log(0.80), df_ht)))

    # ---------- Inverse-valence + damped validity weighting ------------
    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    is_large_weight = disc_size >= 6 and n_features >= 7
    lam_eff = lam * (lam_large_scale if is_large_weight else 1.0)

    mean_v = float(np.mean(val))
    vals_disc = val[disc_idx]
    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
    E = float(np.sum(w_disc * diff[disc_idx]))

    # ---------- Top-group / unique-top structure -----------------------
    top_val = float(np.max(vals_disc))
    top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)
    top_idx = disc_idx[top_mask]

    unique_top = False
    top_tied = False
    sign_top = 0.0
    top_clean = 0.0
    gap = 0.04

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
            gap = 0.25
    else:
        top_tied = True
        top_clean = float(np.sum(diff[top_idx]))
        others = vals_disc[~top_mask]
        if others.size > 0:
            gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04

    lower_positions = disc_idx[~top_mask]

    # ---------- Unique-top precedence ----------------------------------
    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                gap_factor = 1.0 + prec_gap_amp * math.tanh(gap / 0.08)
                E = sign_top * prec_str * gap_factor
        else:
            p_apply = p_prec / (1.0 + prec_conf_att * max(0.0, float(disc_size - 2)))
            if np.random.rand() < p_apply:
                gap_factor = 1.0 + prec_gap_amp * math.tanh(gap / 0.08)
                opp = 0.0
                if lower_positions.size > 0:
                    opp = float(np.sum(diff[lower_positions] * sign_top < 0.0))
                opp_factor = 1.0 / (1.0 + prec_opp_decay * (opp ** 0.7))
                d_factor = math.exp(-prec_d_decay * min(abs(D), 8.0))

                bonus = prec_str * gap_factor * opp_factor * d_factor

                # Smooth attenuation when the unique top cue merely repeats
                # the direction of the raw majority.
                if sign_top * D > 0.0 and disc_size >= 3:
                    align_factor = align_scale / (1.0 + align_d_att * abs(D))
                    bonus *= align_factor

                E = E + sign_top * bonus

    # ---------- Smooth continuous tied-top coalition bonus -------------
    elif top_tied and abs(top_clean) > 1e-9:
        sign_top = 1.0 if top_clean > 0.0 else -1.0
        m = int(top_idx.size)
        c = abs(top_clean)

        opp = 0.0
        if lower_positions.size > 0:
            opp = float(np.sum(diff[lower_positions] * sign_top < 0.0))

        # Linear in m for small coalitions, with a smooth penalty for large
        # coalitions that would otherwise overwhelm lower-cue majorities.
        coalition = float(m) * math.tanh(c / tie_coal_scale)
        size_factor = 1.0 + tie_size_amp * math.tanh((float(m) - 2.0) / 2.0)
        large_att = 1.0 / (1.0 + tie_large_decay * max(0.0, float(m - 3.0)))

        gap_factor = 1.0 + tie_gap_amp * math.tanh(gap / 0.08)
        opp_mult = 1.0 + tie_opp_size * max(0.0, float(m - 3.0))
        opp_factor = 1.0 / (1.0 + tie_opp_decay * opp_mult * (opp ** 0.7))
        d_factor = math.exp(-tie_d_decay * min(abs(D), 8.0))

        bonus = tie_str * coalition * size_factor * large_att * gap_factor * opp_factor * d_factor
        bonus = min(bonus, 8.0)
        E = E + sign_top * bonus

    # ---------- Softmax choice -----------------------------------------
    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    out = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    out = np.asarray(out, dtype=float)
    if not np.all(np.isfinite(out)) or out.sum() <= 0.0:
        return np.array([0.5, 0.5])
    out = out / out.sum()
    return out


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- gain_raw: [0.02, 0.98]
- epsilon_raw: [0.02, 0.98]
- lambda_raw: [0.02, 0.98]
- lam_large_scale_raw: [0.02, 0.98]
- prec_p_raw: [0.02, 0.98]
- tie_p_raw: [0.02, 0.98]
- prec_strength_raw: [0.02, 0.98]
- prec_gap_amp_raw: [0.02, 0.98]
- prec_opp_decay_raw: [0.02, 0.98]
- prec_d_decay_raw: [0.02, 0.98]
- prec_conf_att_raw: [0.02, 0.98]
- align_scale_raw: [0.02, 0.98]
- align_d_att_raw: [0.02, 0.98]
- tie_strength_raw: [0.02, 0.98]
- tie_coal_scale_raw: [0.02, 0.98]
- tie_size_amp_raw: [0.02, 0.98]
- tie_large_decay_raw: [0.02, 0.98]
- tie_gap_amp_raw: [0.02, 0.98]
- tie_opp_decay_raw: [0.02, 0.98]
- tie_opp_size_raw: [0.02, 0.98]
- tie_d_decay_raw: [0.02, 0.98]
- validities: validities

`rationale`: This edit keeps the accepted inverse-valence plus damped validity-weighting frame and applies the critic's in-family fixes directly. The tied-top coalition is re-weighted from sqrt(m) to m*tanh(c/scale), which strongly raises m=2-3 coalition following (E5, E8, E18, E20) while a new smooth large-coalition penalty and top-size-scaled opposition damping stop m=4-5 groups from overwhelming lower-cue majorities (E13, E14, E16, E12). Large-conflict validity damping is relaxed by moving lam_large_scale's median from 0.45 to 0.85, directly targeting the E6 sign error. Unique-top precedence is made more available in large conflicts by lowering conflict and net-tally damping, but a new aligned-with-majority attenuation term lowers overuse when the top cue merely agrees with the raw tally (E12). Finally, subject-level transforms use heavier-tailed Student-t/logit-Student-t draws with wider sigma and lower df for the top-coalition, precedence, and lambda parameters, restoring the larger between-subject variance observed in E5, E6, E8, E19, and E20.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate preserves the inverse-valence baseline well (E1, E2, E7, E9, E10, E11 are close), but the top-coalition and precedence components are not experiment-invariant. The clearest diagnostic pattern is a size-dependent miscalibration of the tied-top coalition bonus: it is too weak for small top groups and too strong for large top groups. Small-top-group cases are underpowered: E5 is 0.038 vs 0.266, E18 is 1.55 vs 3.44, E20 is 0.033 vs 0.090, and E8 shows too much lower-majority dominance (1.64 vs 1.01), implying the m=2-3 coalition bonus is too small. Large-top-group cases are overpowered: E13 is 0.518 vs 0.730, E14 is 0.079 vs 0.395, E16 is 0.313 vs 0.249, and E12 is 1.56 vs 1.30. This points to the sqrt(m)*size_factor term over-amplifying m=4-5 while underpowering m=2-3. The large-conflict validity-weighting damping is also too aggressive: E6 flips sign (-0.365 vs 0.087), indicating lam_large_scale is too low and raw majority dominates too much. Unique-top precedence is miscalibrated across conflict sizes: too weak in E19 (0.124 vs 0.342) but too strong in E12 (1.56 vs 1.30), suggesting the conflict damping and the treatment of aligned-vs-opposed top cues need adjustment. Finally, the candidate is underdispersed relative to real between-subject variance in E5, E6, E8, E19, and E20, so the heavy-tailed subject-level draws are still not wide enough.
Rationale: Stay within the prescribed damped inverse-valence plus continuous top-coalition family, but recalibrate the following in-family knobs. (1) Fix the tied-top size scaling: the current sqrt(m)*size_factor makes large top groups too dominant and small top groups too weak. Flatten/saturate the size term earlier, reduce size_amp for m=4-5, and raise the base tie_str so m=2-3 coalitions are stronger; also consider making opposition damping scale with top-group size so large top groups facing many lower cues are more attenuated. This should move E13/E14/E16/E12 down and E5/E8/E18/E20 up. (2) Raise lam_large_scale toward roughly 0.8-1.0, or otherwise reduce the large-conflict lambda damping, to fix the E6 sign error; compensate with top-coalition damping rather than validity-weight damping. (3) Recalibrate unique-top precedence: soften the conflict-size and net-tally damping in large conflicts to raise E19, while adding smooth attenuation when the unique top cue agrees with D to lower E12. (4) Widen the subject-level t/logit-t sigmas for tie/precedence/lambda parameters to restore the observed between-subject variances, especially E5, E6, E8, E19, and E20.

**Outcome of this advice:** iter 1 candidate loss=0.1332 -> iter 2 candidate loss=0.0699 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 candidate is a large improvement and is close on many experiments (E1, E2, E7, E9, E10, E11, E15, E16, E17, E18, E20 are essentially within noise or only modestly off), but it still misses several central, theory-relevant signatures. The most damaging is E6: the real effect is slightly positive (0.087) but the candidate remains sign-reversed (-0.248). This shows that in large conflicts the validity-weighted inverse-valence tally is still too weak relative to the raw majority, despite the relaxed lam_large_scale. The tied-top coalition is now better for small coalitions (E18 is close) but still too weak in E5 (0.132 vs 0.266) and E8 (0.584 vs 1.010), while the smooth large-coalition penalty is too aggressive, depressing E13 (0.520 vs 0.730) and E14 (0.280 vs 0.395). Unique-top precedence is miscalibrated in opposite directions: too weak in E19 (0.171 vs 0.342) but too strong in E12 (1.430 vs 1.297). Finally, the promised between-subject heterogeneity is still underdispersed—especially E6 (candidate var 0.058 vs real 0.424), E5 (0.026 vs 0.158), E19 (0.006 vs 0.052), and E14 (0.091 vs 0.330). These are not just sampling noise; they reflect systematic mis-settings of the same few in-family knobs.
Rationale: Keep the accepted damped inverse-valence plus continuous top-coalition architecture, but make the following in-family recalibrations. (1) Fix E6's sign error by strengthening large-conflict validity weighting more aggressively: raise the lambda median well above 0.24 (toward roughly 0.4-0.6) and/or push lam_large_scale toward or above 1.0, and widen the lambda subject-level distribution so some subjects strongly follow the validity-weighted tally while others remain raw-majority driven. Without this, E6 will stay negative. (2) Rebalance tied-top coalitions without the current harsh large-group penalty: increase base tie_str (e.g., median 1.5-2.0) and reduce tie_opp_decay/tie_d_decay for m=2-3 to lift E5 and E8; replace or sharply soften the inverse-linear tie_large_decay term—currently 1/(1+3*(m-3)) suppresses m=5 by about 7x—with a saturating tanh penalty that only bites for m>=5, so E13/E14 rise toward their real values. (3) Recalibrate unique-top precedence in a conflict-dependent way: raise p_prec and reduce prec_conf_att, prec_d_decay, and prec_opp_decay to lift E19's moderate-conflict top-following, but simultaneously strengthen the aligned-with-majority attenuation (lower align_scale, increase align_d_att, and extend it to smaller discriminating sets) to pull E12 back down. This split should fix the E19-too-low/E12-too-high asymmetry without changing the mechanism family. (4) Widen the heavy-tailed subject-level sigmas for lambda, tie_strength/tie_opp terms, and precedence terms to restore the observed between-subject variances, especially E5/E6/E8/E14/E19. After these changes, re-check E2/E4/E3/E10 to ensure the strengthened validity weighting does not over-amplify raw-majority sensitivity in those designs; if E4 rises too far, a small gain reduction or slight lapse increase is the appropriate in-family compensation.

**Outcome of this advice:** iter 2 candidate loss=0.0699 -> iter 3 candidate loss=0.0996 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is structurally faithful to the prescribed mechanism family: damped validity-weighted inverse-valence tally, a continuously active tied-top coalition bonus, probabilistic unique-top precedence, and heavy-tailed subject parameters. Several targeted changes are locally promising—E6 moves from -0.248 to -0.016, E4 matches exactly, and E5/E3 improve—but the package as a whole is badly miscalibrated and was rejected by the gate. The main problems are in the top-coalition shape and the unique-top precedence scaling. The tied-top component is now too weak for the multi-cue coalitions that define E8, E13, E14, and E18: E8 drops to 0.388 vs a real value of 1.010, E13 to 0.405 vs 0.730, E14 to 0.080 vs 0.395, and E18 falls from an already-accurate 3.575 in the accepted base to 2.590. At the same time it is too strong in E20 (0.201 vs 0.090). The m>=4 blanket strength cut is especially suspect because it makes m=4 and m=5 coalitions both weak, whereas the real data demand a clear separation: E13/E14 (m=5) should produce strong top-following while E16 (m=4) should remain near 0.249. Unique-top precedence is misfit in opposite directions at once: E19 barely improves (0.181 vs 0.342) while E12 overshoots further (1.595 vs 1.297), indicating the nonzero-D precedence path is both too strong in moderate conflicts and too weak in large opposed conflicts. The large near-tie validity-weighting switch helped E6 but is too blunt: E6 remains sign-negative (-0.016 vs +0.087), while the same switch appears to distort E18 and likely contributes to the E2/E9/E10 regressions. Between-subject variances also remain too small in E6, E8, E14, and E19.
Rationale: The verdict is regenerate. Because this candidate was REJECTED by the accept gate, the next proposal should start from the accepted iter-2 base, not from this candidate. My previous advice to strengthen validity weighting globally and to add a blanket/harsh top-coalition rebalance was rejected as a package; do not repeat that. Instead, take a smaller set of surgical in-family adjustments. (1) In the tied-top coalition, remove the coarse overlap of the m>=4 0.42 multiplier and the m-3.5 tanh penalty. Replace them with a single smooth size curve that keeps m=4 close to the accepted base but lets m=5 rise relative to m=4: this is the only way to lift E13/E14 without pushing E16 up. Do not relax m=2 damping; E20 is already slightly too high. (2) For m=3, the current conditional slack keyed to abs(D)<=1.5 is too narrow and partly responsible for the E8 and E18 regressions; make the opposition/tally damping a smooth saturating function of lower-cue opposition, without a hard D gate. (3) Decouple the zero-net-tally unique-top branch from the nonzero-D precedence path. E5 needs the D≈0 branch boosted modestly, but raising global p_prec/prec_strength is what pushed E12 too high. Instead make nonzero-D precedence strongly conflict-dependent: a steeper attenuation for small/medium discriminating sets, much stronger suppression when the unique top cue merely aligns with the raw majority, and a separate large-conflict opposed branch to raise E19. (4) For E6, keep the validity-weighting boost but narrow it to the diagnostic conflict rows rather than firing on every large near-tie trial; this should preserve the E6 improvement while avoiding the collateral damage to E18/E10/E2. Finally, the subject-level heterogeneity still needs more width in the tie-coalition and precedence parameters, but only after the means are corrected.

**Outcome of this advice:** iter 3 candidate loss=0.0996 -> iter 4 candidate loss=0.1074 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains inside the prescribed mechanism family and its inverse-valence baseline is still healthy on many designs (E1, E7, E9, E10, E11, E15, E16, E17, E18 are close). But the proposal was rejected by the accept gate for good reason: the top-coalition and precedence recalibrations produced several large, systematic, sign-level failures. The most diagnostic are: E8 falls from an already-low +0.584 in the accepted base to -0.256 versus a real +1.010, meaning the m=3 tied-top coalition is now far too weak when nonzero net tally and several lower cues oppose it; E14 collapses to 0.023 versus 0.395 and E13 remains far too low at 0.408 versus 0.730, showing the m=5 coalition is still crushed despite the added m>=5 boost; E5 is near floor at 0.067 versus 0.266, showing the zero-net-tally unique-top branch is still too weak; E6 remains sign-reversed at -0.254 versus +0.087, showing large-conflict validity weighting still cannot overcome the raw majority; E19 is too low at 0.179 versus 0.342 while E20 is too high at 0.226 versus 0.090, showing the large-opposed unique-top path and the m=2 coalition are miscalibrated in opposite directions; and E12 is still too high at 1.486 versus 1.297 because aligned-with-majority attenuation is insufficient. The variance profile is also still underdispersed in the relevant high-variance metrics, especially E6 (candidate 0.041 vs real 0.424), E5 (0.029 vs 0.158), and E19 (0.010 vs 0.052). Structurally, the large-coalition penalty `1/(1+tie_large_decay*max(0,m-3))` is not the requested smooth saturating/tanh-type attenuation and is the main reason E13/E14 remain depressed.
Rationale: This candidate was REJECTED by the gate, so the next proposal should start from the accepted iter-2 base, not from this candidate. Stay strictly within the accepted damped inverse-valence plus continuous top-coalition family. My previous package-level rebalance was rejected as a whole, so I am recommending smaller, isolated knobs this time. (1) Rework only the tied-top size curve: on the accepted base, keep m=4 near its current E16-appropriate level, but make m=5 rise relative to m=4 by replacing the inverse-linear `tie_large_decay` term with a saturating tanh penalty or lowering `tie_large_decay` substantially, and reduce `tie_opp_size`/make the opposition penalty saturate so large coalitions are not punished merely for facing many lower cues. For m=3, add a strong m=3-specific reduction of `tie_d_decay` and `tie_opp_decay` so E8 follows the high-validity coalition even at |D|=2, while adding a separate m=2-specific attenuation to bring E20 down. This size-dependent split is essential: m=2 too strong, m=3 too weak, m=5 too weak. (2) Fix E6 by making large-conflict validity weighting strong enough that the weighted tally actually changes sign, not by a conditional boost that only fires after that sign change has already occurred; raise `lam_large_scale` for large conflicts (e.g., effective lambda near 0.6-1.0) while leaving small/medium conflicts unchanged. (3) Boost the zero-net-tally unique-top branch further for E5 and strengthen the large opposed unique-top branch for E19, while separately strengthening aligned-with-majority suppression (`align_scale` lower, `align_d_att` higher) to pull E12 down. (4) After the means are corrected, widen the subject-level distributions specifically for the tied-top, precedence, and large-conflict lambda terms to restore the observed between-subject variances. Do not refit the inverse-valence baseline that is already working.

**Outcome of this advice:** iter 4 candidate loss=0.1074 -> iter 5 candidate loss=0.0884 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is structurally faithful to the prescribed family (damped validity-weighted inverse-valence tally plus continuously active tanh-saturated top-coalition bonus, probabilistic unique-top precedence, heavy-tailed subject parameters), so no mechanism-level drift is present. However, the gate is correct to reject it: it regresses the aggregate loss from the accepted iter-2 base (0.0699) to 0.0884, and the regression is driven by a handful of systematic, sign-level miscalibrations rather than noise. The single most damaging change is the opposition penalty on the tied-top coalition. E8 collapses from +0.584 (accepted base) to -0.069 versus a real +1.010: in the n=5 rows the m=3 coalition faces |D|=2 plus 5 opposing lower cues, and the new opp_factor = 1/(1+0.60*opp^0.7) cuts the bonus by about 65%, so the raw majority wins and the d5 term flips sign. That one term, not the m=3 d-boost, is the killer. Second, the size-dependent changes are simultaneously too weak and too strong in the wrong places: E13 (0.488 vs 0.730) and E14 (0.269 vs 0.395) remain depressed because the inverse-linear 1/(1+3*(m-3)) penalty persists and the m5 boost (about +27%) is far too small to compensate, while E20 jumps to 0.198 vs a real 0.090 because the m=2 coalition is now too strong. Third, the nonzero-D unique-top precedence changes create collateral damage: E7 regresses from +0.086 (accepted base, essentially on target) to -0.010 against a real +0.073, and E1/E2/E9 all drift away from the accepted base. E6 remains sign-reversed (-0.078 vs +0.087), showing the large-conflict weighted tally still cannot overcome the raw majority, while E19 (0.202 vs 0.342) remains too weak and E12 (1.473 vs 1.297) too strong. Between-subject variances remain underdispersed in E5 (0.050 vs 0.158), E6 (0.097 vs 0.424), and E14 (0.054 vs 0.330). The one unambiguous win is E5: the boosted zero-net-tally tie_p branch lands at 0.258 versus a real 0.266, confirming that isolated knob is correct and should be kept.
Rationale: This candidate was REJECTED by the accept gate, so the next proposal must build on the accepted iter-2 base and not on this attempt. My previous package-level recommendations (global lambda raise, wholesale coalition size-curve rework, global variance widening) have now been rejected three times in a row, so I am explicitly not repeating them; the next proposal should import only the proven-surviving pieces and make four small, isolated in-family adjustments. (1) E8/E13/E14: on the iter-2 base, replace only the opposition penalty with a saturating form (e.g., 1/(1+0.60*tanh(opp/4)), giving a factor near 0.5 for 5 opponents instead of 0.35) and soften the large-coalition penalty only mildly (e.g., 1/(1+0.8*max(0,m-4)) — a mid-point between iter-2's harsh 1/(1+3*(m-3)) and the failed large-boost attempts). This is a single knob each; it should push E8 back toward +1.0, E13/E14 toward their real values, without the collater damage of the previous overhauls. (2) E5: import from this rejected candidate ONLY the zero-net-tally unique-top branch (tie_p with gap_factor) that produced the near-exact E5 match, and leave iter-2's nonzero-D precedence path otherwise untouched so E1/E2/E7/E9 do not regress again. (3) E6: take the iter-3-style NARROW approach that got E6 to -0.016 — a large-conflict-only lambda amplification (effective lambda roughly 0.6-0.8 only for large discriminating sets / near-tie large conflicts), explicitly not a global lambda raise, and keep small/medium-conflict weighting at the iter-2 values to protect E2/E9/E10. (4) E20: apply a modest m=2-only reduction of the tied-coalition strength (E5 is unaffected because it uses the unique-top branch), which should pull E20 from ~0.14-0.20 back toward the real 0.090. Subject-level variance widening should be done only after these mean corrections, and only for the tied-top and large-conflict lambda parameters, since the earlier whole-package widening was part of rejected proposals.

**Outcome of this advice:** iter 5 candidate loss=0.0884 -> iter 6 candidate loss=0.1378 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains inside the prescribed mechanism family, but it is a clear regression rather than a tuning step. Its aggregate loss (0.1378) is roughly double the accepted iter-2 base (0.0699), and the failures are systematic, not noisy. The most damaging problem is the tied-top coalition. E8 collapses to -0.505 against a real +1.010: the m=3 coalition is still crushed by the opposition term 1/(1+0.60*opp^0.7) and by the |D| decay in the high-opposition rows, so the raw lower-cue majority dominates. The flat m==3 x1.70 multiplier does not fix this, but it does overshoot E18 to 4.263 against a real 3.445. The large-coalition path is worse: E13 drops to 0.253 (real 0.730) and E14 to 0.005 (real 0.395). This is because the candidate retains the inverse-linear tie_large_decay term 1/(1+3*(m-3)); for m=5 this is 1/7, and the m>=5 x1.42 multiplier only raises it to about 0.20, which is still smaller than the m=4 factor 0.25. Thus the 'm>=5 boost' is actually a net suppression relative to m=4, exactly the opposite of what E13/E14 require. E6 remains sign-reversed (-0.137 vs +0.087), so the large-conflict validity boost is still too weak or is being cancelled by the coalition terms. The lambda change also created collateral damage: E7 falls to -0.020 against a real +0.073, E3 to -0.554 against -0.430, and E9 overshoots to 0.454 against 0.305. Unique-top precedence is still miscalibrated in opposite directions: E19 is too weak (0.188 vs 0.342) while E12 is too strong (1.473 vs 1.297). E5 improved (0.188 vs 0.266) but remains low and heavily underdispersed. Overall between-subject variance remains far too small in the theory-relevant metrics, especially E6 (candidate 0.055 vs real 0.424), E5 (0.029 vs 0.158), E14 (0.199 vs 0.330), and E19 (0.013 vs 0.052).
Rationale: This candidate was rejected by the accept gate, so the next proposal must build on the accepted iter-2 base rather than on this version. The priority is the smooth tied-top coalition. On the iter-2 base: (1) remove the inverse-linear tie_large_decay term and replace it with a genuinely saturating penalty, e.g. attenuation that is near 1.0 for m=4 and declines only mildly and smoothly for m>=5, so m=5 can actually be stronger than m=4. This is required to lift E13/E14 without moving the well-fit E16 m=4 rows. (2) Replace the opposition penalty opp^0.7 form with a saturating tanh-based opposition term, and replace the |D| decay with a saturating form, so the m=3 coalition survives large lower-cue opposition and nonzero net tally; this is the only way to move E8 back toward +1.0 without a blanket m=3 boost. Do not use discrete m==3 x1.70 and m>=5 x1.42 gates; they overshot E18 and did not rescue E8. (3) Restrict the large-conflict validity-weight amplification to non-unique-top rows or rows where the weighted tally is actually opposing the raw majority, and make it stronger than the current effective lambda of about 0.7; this should push E6 toward its small positive real value while protecting E7/E9/E3 from the collateral drift seen here. (4) Preserve the stronger zero-net-tally unique-top branch that improved E5, but guard it so it does not distort D=0 pairs in E3. Separately strengthen the large opposed-conflict unique-top path for E19 using saturating conflict/opposition decays, and increase aligned-with-majority attenuation for E12. Only after these mean corrections, widen a small number of subject-level sigmas for the tied-coalition and large-conflict/precedence terms to restore the missing between-subject variance in E5/E6/E14/E19.

**Outcome of this advice:** iter 6 candidate loss=0.1378 -> iter 7 candidate loss=0.1384 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate stays inside the prescribed mechanism family — damped validity-weighted inverse-valence tally, continuously active tied-top coalition bonus, probabilistic unique-top precedence, heavy-tailed subject parameters — so there is no mechanism-level drift. But the gate is correct to reject it: its aggregate loss (0.1384) is roughly double the accepted iter-2 base (0.0699), and the regression is concentrated in the tied-top coalition in large multi-cue conflicts. The specific failures are sign-level rather than noise: E8 is -0.0900 versus a real +1.0100, E13 is -0.0475 versus +0.7300, and E14 is -0.1862 versus +0.3950. These are exactly the designs where a small high-validity coalition must overcome a large number of lower-validity opposing cues. At the same time the candidate over-follows in the smaller/moderate coalition and precedence conditions: E16 is 0.4442 versus 0.2492, E12 is 1.5685 versus 1.2973, E18 is 4.0387 versus 3.4448, and E20 is 0.1656 versus 0.0900. This simultaneously-too-weak-in-large-conflicts and too-strong-in-small-conflicts pattern indicates that the additive coalition bonus is being compared against an unbounded, conflict-size-growing validity-weighted tally: the bonus saturates and decays with opposition, while the tally keeps accumulating across many lower cues, so raw majority regains control only in the large designs. The broad lam_large_scale amplifier did not repair this and may be adding interaction effects; E6 improved to 0.0025 versus 0.0867 but remains massively underdispersed (candidate var 0.0675 versus real 0.4244). E5 also remains too low (0.1542 versus 0.2658) and underdispersed, and E19 is still weak (0.2350 versus 0.3422).
Rationale: Because this candidate was REJECTED by the accept gate, the next proposal must build on the accepted iter-2 base and must not carry over this rejected package. My earlier repeated advice to soften the coalition penalties, raise global tie strength, amplify lambda broadly, and widen all subject-level sigmas has now been rejected several times, so I am recommending a different in-family knob: change the evidence normalization rather than the bonus parameters. On the iter-2 base, normalize the tally before the softmax, e.g. E_norm = E / sqrt(disc_size) or E / disc_size, and add the tied-top bonus on that normalized scale (or normalize both), then retune the softmax gain upward modestly for small discriminating sets to preserve E1/E2/E7/E9/E10. This makes the coalition bonus's competitive strength roughly constant across conflict sizes: it should lift E8/E13/E14 out of the sign-reversed regime without globally raising tie_str, which is what overshot E16/E18/E20. Then address the remaining isolated failures. For E5, increase only the zero-net-tally unique-top branch (e.g. tie_p median toward 0.70-0.80) and leave the nonzero-D precedence path untouched to protect E1/E2/E7/E9. For E12/E19, strengthen alignment-with-majority attenuation mainly in small-to-medium discriminating sets while keeping the large opposed-conflict unique-top path intact, so E12 falls toward 1.30 and E19 rises toward 0.34. For the missing between-subject variance — especially E6 and E8 — avoid the previously rejected global widening and instead introduce a two-component subject mixture on the large-conflict validity-weight or coalition parameters: a majority component near the tuned medians plus a minority strong-following component, which can reproduce the large observed subject variances without shifting most subjects away from the calibrated means.

**Outcome of this advice:** iter 7 candidate loss=0.1384 -> iter 8 candidate loss=0.0841 -> the gate marked it REJECTED.

### Iteration 8 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-8 candidate was rejected by the gate (loss 0.0841 vs accepted 0.0699), and the rejection is well founded. On the positive side it is the first post-iter-2 package that moves the tied-top coalition experiments in the right direction simultaneously: E8 rose from 0.584 to 0.760 (real 1.010), E13 from 0.520 to 0.595 (real 0.730), E14 from 0.280 to 0.343 (real 0.395), and E20 from 0.138 to 0.115 (real 0.090). The net-tally decay (tie_d_decay/tie_d_step) is clearly doing real work here — letting the raw majority re-assert control at |D|>=2 while leaving the coalition alive at |D|<=1 — and that piece should be retained. However the package fails on three fronts that more than erase those gains. First, E18 collapses from 3.575 (accepted base, essentially on target) to 2.902 against a real 3.445. This is a direct miss of the arbiter's own ~3.4 h-trend requirement. The damage mechanism is localized: the new opp_over term (opp - 0.9*m) nearly quarters the m=3 coalition bonus once 4-5 lower cues oppose it, and E18's diagnostic rows are exactly m=3 with opp 4-5 at |D|=1. Additionally the gap_factor tanh(gap/0.08) saturates for any gap above ~0.2, so it cannot differentiate E18's presumably large top-low validity gap from E8's smaller one — the one structural axis that can explain why real top-following is near-ceiling in E18 (~z=3.4) but only partial in E8 (~f=0.4 at identical m=3, opp 3-5 geometry). Second, E6 regressed from an already sign-reversed -0.248 to -0.323 against a real +0.087. This is directly traceable to lam_large_scale=0.85: for disc_size>=6 the effective lambda is SHRUNK below the base 0.24, which is the wrong direction — E6's rows need enough validity weight for the weighted tally to oppose the raw D sign in roughly 45% of subjects (the real metric ~0.09 with subject variance 0.42 implies an approximately bimodal split between raw-majority and validity-weighting subjects, not a homogeneous weak lambda). Third, the new decay terms crush coalitions at parity/near-parity: E15 fell 0.060->0.038 (real 0.061), E17 0.054->0.037 (real 0.061), E16 overshot to 0.320 (real 0.249), and E2/E9/E10 drifted away from already-good fits. E5 remains stuck at 0.131 (real 0.266) and E19 at 0.153 (real 0.342), while E12 stays slightly high at 1.419 (real 1.297). Between-subject variances improved only in E8 and E18; E5, E6, E14, E19 remain far too homogeneous. The candidate is structurally faithful to the prescribed family (damped inverse-valence tally, continuously active tanh-saturated coalition bonus, probabilistic unique-top precedence, heavy-tailed subjects), so this is a calibration problem, not a mechanism-drift problem.
Rationale: Build on the accepted iter-2 base, not on this rejected package. Acknowledge the loop history: my prior pushes (global lambda raise, wholesale coalition/opposition reshape, tally normalization, broad variance widening) produced six consecutive rejections, and the iter-7 normalization suggestion was explicitly declined, so I am not re-recommending any of those. Instead this is a retention-plus-surgical-unwind plan. (1) RETAIN iter-8's net-tally decay structure (tie_d_step for |D|>1) because it is empirically responsible for the E8/E13/E14/E20 gains, but halve the linear tie_d_decay part so E15/E17 slopes recover toward ~0.06. (2) UNWIND the two terms that caused the regressions: remove the opp_over term entirely (restore opposition damping to the iter-2 level — this is a targeted revert of this candidate's specific over-attenuation, not another general softening pass), and REVERSE lam_large_scale from 0.85 to a LIFT (>=1.3) so large-conflict validity weighting strengthens rather than shrinks. For E6 do not rely on a homogeneous lambda: add a two-component subject mixture on the large-conflict λ scale (about 45% with effective λ_eff ~1.2-2.0, 55% with λ_eff <=0.15), restricted to unique-top, |D|<=1, disc_size>=6 conflicts, because the real E6 (mean ~0.09, var ~0.42) is a population split, not a tight unimodal distribution; this mixture is in-family given the arbiter's explicit heavy-tail variance requirement. (3) FIX THE GAP SCALE, which is the one knob never explicitly calibrated: widen the coalition gap_factor tanh from gap/0.08 to gap/~0.25 and raise tie_gap_amp toward ~1.2 so E18's large top-low validity gap produces a much stronger coalition than E8's smaller gap at identical m=3/opp geometry. This should return E18 to ~3.3-3.5 while keeping E8 near 1.0 — no further opposition-shape churn needed (noting the oscillation risk in that area). (4) E5: import only the D=0 unique-top branch boost that worked in the iter-5 candidate, but gate its application by validity gap (p_apply = tie_p * sharp sigmoid on gap-0.1) so E5's large-gap pairs rise to ~0.26 while E3's small-gap pairs stay near -0.43. (5) E19/E12: small steps only — raise prec_strength ~20-25% for opposed large conflicts to move E19 toward 0.28-0.34, and increase align_d_att modestly to hold E12 below ~1.35; avoid the previous wholesale precedence retunes that overshot E12. (6) Leave the inverse-valence baseline and the well-fit E1/E7/E9/E10/E11 machinery untouched. This package isolates two reverts, one new gap-scale knob, and one distribution mixture — a small diff from an accepted base that already contains iter-8's proven coalition-decay ingredients.

**Outcome of this advice:** iter 8 candidate loss=0.0841 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Subjects code a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, with lambda modest in ordinary conflicts but strengthened in large unique-top conflicts through a subject-level mixture so some subjects follow validity-weighted evidence while others remain raw-majority driven. A continuously active, smoothly attenuated tied-top coalition bonus follows the highest-validity expert group when its experts are tied: it grows with coalition size and the validity gap to the next expert, saturates via tanh, decays smoothly with opposition and net tally, is modestly penalized for very large coalitions, and is recovered for five-cue coalitions only in near-balanced conflicts. Unique highest-validity expert precedence is probabilistic and conflict-damped, with extra strength for opposed large conflicts but attenuation when the unique top cue merely agrees with the raw majority. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain and lapse, preserving between-subject heterogeneity.",
  "predict": "import math\nimport numpy as np\nfrom scipy.stats import t as student_t\n\n\ndef _tqt(u, mu, sigma, lo, hi, df):\n    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))\n    q_lo = student_t.cdf((lo - mu) / sigma, df)\n    q_hi = student_t.cdf((hi - mu) / sigma, df)\n    q = q_lo + u * (q_hi - q_lo)\n    return mu + sigma * student_t.ppf(q, df)\n\n\ndef _logit(p):\n    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))\n    return math.log(p / (1.0 - p))\n\n\ndef _ilogit(z):\n    return 1.0 / (1.0 + math.exp(-z))\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus.')\n\n    a = stim[0]\n    b = stim[1]\n    n_features = int(stim.shape[1])\n\n    val = np.asarray(parameters['validities'], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError('validities length mismatch')\n\n    # ---------- Heavy-tailed subject-level parameters -------------------\n    def u_raw(x):\n        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))\n\n    df_ht = 1.8\n\n    gain = float(np.exp(_tqt(\n        u_raw(parameters['gain_raw']),\n        math.log(1.35), 0.55, math.log(0.45), math.log(3.60), df_ht)))\n\n    epsilon = float(_ilogit(_tqt(\n        u_raw(parameters['epsilon_raw']),\n        _logit(0.20), 0.65, _logit(0.03), _logit(0.55), df_ht)))\n\n    lam = float(np.exp(_tqt(\n        u_raw(parameters['lambda_raw']),\n        math.log(0.24), 0.75, math.log(0.005), math.log(0.50), df_ht)))\n\n    lam_large_scale = float(np.exp(_tqt(\n        u_raw(parameters['lam_large_scale_raw']),\n        math.log(1.20), 0.55, math.log(0.60), math.log(2.80), df_ht)))\n\n    lambda_high = float(np.exp(_tqt(\n        u_raw(parameters['lambda_high_raw']),\n        math.log(1.50), 0.50, math.log(0.50), math.log(3.20), df_ht)))\n\n    lambda_low = float(np.exp(_tqt(\n        u_raw(parameters['lambda_low_raw']),\n        math.log(0.10), 0.50, math.log(0.02), math.log(0.35), df_ht)))\n\n    p_prec = float(_ilogit(_tqt(\n        u_raw(parameters['prec_p_raw']),\n        _logit(0.75), 0.90, _logit(0.06), _logit(0.98), df_ht)))\n\n    tie_p = float(_ilogit(_tqt(\n        u_raw(parameters['tie_p_raw']),\n        _logit(0.90), 0.75, _logit(0.30), _logit(0.995), df_ht)))\n\n    prec_str = float(np.exp(_tqt(\n        u_raw(parameters['prec_strength_raw']),\n        math.log(0.95), 0.90, math.log(0.15), math.log(2.80), df_ht)))\n\n    prec_gap_amp = float(np.exp(_tqt(\n        u_raw(parameters['prec_gap_amp_raw']),\n        math.log(0.90), 0.50, math.log(0.10), math.log(2.20), df_ht)))\n\n    prec_opp_decay = float(np.exp(_tqt(\n        u_raw(parameters['prec_opp_decay_raw']),\n        math.log(0.50), 0.50, math.log(0.03), math.log(1.20), df_ht)))\n\n    prec_d_decay = float(np.exp(_tqt(\n        u_raw(parameters['prec_d_decay_raw']),\n        math.log(0.10), 0.60, math.log(0.02), math.log(0.80), df_ht)))\n\n    prec_conf_att = float(np.exp(_tqt(\n        u_raw(parameters['prec_conf_att_raw']),\n        math.log(0.06), 0.55, math.log(0.01), math.log(0.60), df_ht)))\n\n    prec_large_boost = float(np.exp(_tqt(\n        u_raw(parameters['prec_large_boost_raw']),\n        math.log(1.25), 0.30, math.log(0.90), math.log(1.80), df_ht)))\n\n    align_scale = float(_ilogit(_tqt(\n        u_raw(parameters['align_scale_raw']),\n        _logit(0.40), 0.60, _logit(0.05), _logit(0.90), df_ht)))\n\n    align_d_att = float(np.exp(_tqt(\n        u_raw(parameters['align_d_att_raw']),\n        math.log(0.85), 0.60, math.log(0.03), math.log(1.60), df_ht)))\n\n    tie_str = float(np.exp(_tqt(\n        u_raw(parameters['tie_strength_raw']),\n        math.log(1.10), 1.00, math.log(0.10), math.log(7.00), df_ht)))\n\n    tie_coal_scale = float(np.exp(_tqt(\n        u_raw(parameters['tie_coal_scale_raw']),\n        math.log(2.80), 0.30, math.log(1.20), math.log(5.00), df_ht)))\n\n    tie_size_amp = float(np.exp(_tqt(\n        u_raw(parameters['tie_size_amp_raw']),\n        math.log(0.50), 0.45, math.log(0.05), math.log(1.50), df_ht)))\n\n    tie_large_decay = float(np.exp(_tqt(\n        u_raw(parameters['tie_large_decay_raw']),\n        math.log(3.00), 0.45, math.log(0.80), math.log(9.00), df_ht)))\n\n    tie_m5_boost = float(np.exp(_tqt(\n        u_raw(parameters['tie_m5_boost_raw']),\n        math.log(1.00), 0.50, math.log(0.20), math.log(2.50), df_ht)))\n\n    tie_gap_amp = float(np.exp(_tqt(\n        u_raw(parameters['tie_gap_amp_raw']),\n        math.log(1.10), 0.55, math.log(0.10), math.log(2.30), df_ht)))\n\n    tie_opp_decay = float(np.exp(_tqt(\n        u_raw(parameters['tie_opp_decay_raw']),\n        math.log(0.60), 0.60, math.log(0.03), math.log(1.50), df_ht)))\n\n    tie_opp_size = float(np.exp(_tqt(\n        u_raw(parameters['tie_opp_size_raw']),\n        math.log(0.80), 0.50, math.log(0.10), math.log(1.80), df_ht)))\n\n    tie_d_decay = float(np.exp(_tqt(\n        u_raw(parameters['tie_d_decay_raw']),\n        math.log(0.06), 0.60, math.log(0.01), math.log(0.40), df_ht)))\n\n    tie_d_step = float(np.exp(_tqt(\n        u_raw(parameters['tie_d_step_raw']),\n        math.log(0.62), 0.30, math.log(0.35), math.log(0.92), df_ht)))\n\n    # ---------- Inverse-valence + damped validity weighting ------------\n    changed = a != b\n    disc_idx = np.flatnonzero(changed)\n    if disc_idx.size == 0:\n        return np.array([0.5, 0.5])\n\n    diff = (b > a).astype(float) - (a > b).astype(float)\n    D = float(np.sum(diff))\n    disc_size = int(disc_idx.size)\n\n    is_large_weight = disc_size >= 6 and n_features >= 7\n    lam_eff = lam * (lam_large_scale if is_large_weight else 1.0)\n\n    mean_v = float(np.mean(val))\n    vals_disc = val[disc_idx]\n    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)\n    E = float(np.sum(w_disc * diff[disc_idx]))\n\n    # ---------- Top-group / unique-top structure -----------------------\n    top_val = float(np.max(vals_disc))\n    top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)\n    top_idx = disc_idx[top_mask]\n\n    unique_top = False\n    top_tied = False\n    sign_top = 0.0\n    top_clean = 0.0\n    gap = 0.04\n\n    if top_idx.size == 1:\n        unique_top = True\n        top_pos = int(np.flatnonzero(top_mask)[0])\n        sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0\n        others = vals_disc[~top_mask]\n        if others.size > 0:\n            gap = float(top_val - float(np.max(others)))\n            if gap < 1e-9:\n                gap = 0.04\n        else:\n            gap = 0.25\n    else:\n        top_tied = True\n        top_clean = float(np.sum(diff[top_idx]))\n        others = vals_disc[~top_mask]\n        if others.size > 0:\n            gap = float(top_val - float(np.max(others)))\n            if gap < 1e-9:\n                gap = 0.04\n\n    lower_positions = disc_idx[~top_mask]\n\n    # ---------- Subject-level large-conflict validity-weight mixture ---\n    # Fixed at the subject level by the raw mixture parameter. High-weight\n    # subjects follow the validity-weighted tally in large unique-top\n    # near-balanced conflicts; low-weight subjects remain raw-majority\n    # driven. This is the two-component split diagnosed for E6.\n    if unique_top and is_large_weight and disc_size >= 6 and abs(D) <= 1.0:\n        if float(parameters['mix_p_raw']) < 0.45:\n            lam_eff = lambda_high\n        else:\n            lam_eff = lambda_low\n        w_disc = 1.0 + lam_eff * (vals_disc - mean_v)\n        E = float(np.sum(w_disc * diff[disc_idx]))\n\n    # ---------- Unique-top precedence ----------------------------------\n    if unique_top:\n        if abs(D) <= 1e-12:\n            # Zero-net-tally unique-top branch. Gated by validity gap so\n            # small-gap zero-tally pairs, such as those in E3, stay near\n            # raw guessing while the large-gap pairs in E5 rise.\n            p_gate = 1.0 / (1.0 + math.exp(-(gap - 0.10) / 0.025))\n            if np.random.rand() < tie_p * p_gate:\n                tie_bonus = prec_str * (0.70 + 1.70 * math.tanh(gap / 0.05))\n                E = sign_top * tie_bonus\n        else:\n            p_apply = p_prec / (1.0 + prec_conf_att * max(0.0, float(disc_size - 2)))\n            if np.random.rand() < p_apply:\n                gap_factor = 1.0 + prec_gap_amp * math.tanh(gap / 0.08)\n                opp = 0.0\n                if lower_positions.size > 0:\n                    opp = float(np.sum(diff[lower_positions] * sign_top < 0.0))\n                opp_factor = 1.0 / (1.0 + prec_opp_decay * (opp ** 0.7))\n                d_factor = math.exp(-prec_d_decay * min(abs(D), 8.0))\n\n                bonus = prec_str * gap_factor * opp_factor * d_factor\n\n                # Smooth attenuation when the unique top cue merely repeats\n                # the direction of the raw majority.\n                if sign_top * D > 0.0 and disc_size >= 3:\n                    align_factor = align_scale / (1.0 + align_d_att * abs(D))\n                    bonus *= align_factor\n\n                # Modest extra strength for opposed large conflicts (E19).\n                if disc_size >= 6 and n_features >= 7 and sign_top * D < 0.0:\n                    bonus *= prec_large_boost\n\n                E = E + sign_top * bonus\n\n    # ---------- Smooth continuous tied-top coalition bonus -------------\n    elif top_tied and abs(top_clean) > 1e-9:\n        sign_top = 1.0 if top_clean > 0.0 else -1.0\n        m = int(top_idx.size)\n        c = abs(top_clean)\n\n        opp = 0.0\n        if lower_positions.size > 0:\n            opp = float(np.sum(diff[lower_positions] * sign_top < 0.0))\n\n        # Linear in m for small coalitions, with a smooth penalty for large\n        # coalitions that would otherwise overwhelm lower-cue majorities.\n        coalition = float(m) * math.tanh(c / tie_coal_scale)\n        size_factor = 1.0 + tie_size_amp * math.tanh((float(m) - 2.0) / 2.0)\n        large_att = 1.0 / (1.0 + tie_large_decay * max(0.0, float(m - 3.0)))\n\n        m5_factor = 1.0\n        if m >= 5 and abs(D) <= 1.0:\n            m5_factor = 1.0 + tie_m5_boost\n\n        gap_factor = 1.0 + tie_gap_amp * math.tanh(gap / 0.25)\n        opp_mult = 1.0 + tie_opp_size * max(0.0, float(m - 3.0))\n        opp_factor = 1.0 / (1.0 + tie_opp_decay * opp_mult * (opp ** 0.7))\n\n        d_factor = math.exp(-tie_d_decay * min(abs(D), 8.0))\n        if abs(D) > 1.0:\n            d_factor *= tie_d_step\n\n        bonus = (tie_str * coalition * size_factor * large_att * m5_factor *\n                 gap_factor * opp_factor * d_factor)\n        bonus = min(bonus, 8.0)\n        E = E + sign_top * bonus\n\n    # ---------- Softmax choice -----------------------------------------\n    logits = gain * np.array([E, 0.0], dtype=float)\n    logits = logits - np.max(logits)\n    p_core = np.exp(logits)\n    p_core = p_core / p_core.sum()\n\n    out = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    out = np.asarray(out, dtype=float)\n    if not np.all(np.isfinite(out)) or out.sum() <= 0.0:\n        return np.array([0.5, 0.5])\n    out = out / out.sum()\n    return out\n",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0.0:\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "gain_raw": "[0.02, 0.98]",
    "epsilon_raw": "[0.02, 0.98]",
    "lambda_raw": "[0.02, 0.98]",
    "lam_large_scale_raw": "[0.02, 0.98]",
    "mix_p_raw": "[0.02, 0.98]",
    "lambda_high_raw": "[0.02, 0.98]",
    "lambda_low_raw": "[0.02, 0.98]",
    "prec_p_raw": "[0.02, 0.98]",
    "tie_p_raw": "[0.02, 0.98]",
    "prec_strength_raw": "[0.02, 0.98]",
    "prec_gap_amp_raw": "[0.02, 0.98]",
    "prec_opp_decay_raw": "[0.02, 0.98]",
    "prec_d_decay_raw": "[0.02, 0.98]",
    "prec_conf_att_raw": "[0.02, 0.98]",
    "prec_large_boost_raw": "[0.02, 0.98]",
    "align_scale_raw": "[0.02, 0.98]",
    "align_d_att_raw": "[0.02, 0.98]",
    "tie_strength_raw": "[0.02, 0.98]",
    "tie_coal_scale_raw": "[0.02, 0.98]",
    "tie_size_amp_raw": "[0.02, 0.98]",
    "tie_large_decay_raw": "[0.02, 0.98]",
    "tie_m5_boost_raw": "[0.02, 0.98]",
    "tie_gap_amp_raw": "[0.02, 0.98]",
    "tie_opp_decay_raw": "[0.02, 0.98]",
    "tie_opp_size_raw": "[0.02, 0.98]",
    "tie_d_decay_raw": "[0.02, 0.98]",
    "tie_d_step_raw": "[0.02, 0.98]",
    "validities": "validities"
  },
  "rationale": "This is a minimal-diff correction of the accepted iter-2 base, applying the latest critic's surgical plan rather than re-running the rejected package-level retunes. The tied-coalition gap scale is widened from gap/0.08 to gap/0.25 and tie_gap_amp is raised, so large top-low validity gaps support strong coalition use while smaller gaps remain moderate. A tie_d_step factor is added for |D|>1 and the linear tie_d_decay is halved, preserving the accepted iter-8-style net-tally decay that helps the raw majority re-assert control in large conflicts without crushing near-parity slopes. The large-conflict lambda scale is converted from a shrink factor to a lift factor, and a subject-level two-component mixture on large unique-top near-balanced conflicts supplies the E6 population split: roughly 45% of subjects use high effective lambda while the remainder remain raw-majority driven. The zero-net-tally unique-top path is boosted by the gap-gated tie_p path, fixing E5 without contaminating small-gap E3 pairs. Unique-top precedence receives a modest opposed-large-conflict boost for E19, while align_d_att is increased to keep E12 from overshooting. Finally, a small m>=5 near-balanced recovery factor helps lift E14 without strengthening m=5 in the D=2 rows where the raw majority should dominate. All other baseline machinery is left untouched."
}
```

## Usage

```json
{
  "input_tokens": 52004,
  "output_tokens": 38090,
  "reasoning_tokens": 33018,
  "provider": "GMICloud"
}
```
