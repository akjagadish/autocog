# generation_iter_04_attempt_00

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
- THEORY 2 = `pi_7_1`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Propose a brand-new theory to replace pi_6, retaining the shared baseline mixture (weak forward lexicographic with small beta, moderate lowest-validity-first lexicographic, and feature-win tallying) but replacing the binary 24-trial qualifying-fraction trapezoid trigger with a per-trial graded conflict-density evidence mechanism. The new theory should claim: people accumulate local evidence for the backward route only from trials that are dense (at least five of six features discriminating, or six of six) AND in which the highest and lowest validities point in opposite directions; top-tied fallback evidence is allowed only on fully discriminating (6/6) trials. Critically, evidence must NOT require the tally to be near zero: dense top-low conflict with tallies of +/-4 to +/-6 (E15/E16) drives strong reversal, while aligned fully-diagnostic near-tied Q-trial contexts (E1) and 4/6-density top-low conflicts (E2) emit zero evidence, keeping those metrics near baseline. The evidence enters a leaky asymmetric accumulator with a minimum qualifying count, hysteresis, and smooth activation of a strengthened lowest-validity-first route; there is no strong attenuation floor at high density, so all-complementary designs reverse strongly (targets: E8 top-cue contrast near -0.67; E15/E16 low-cue adherence near 0.84), yet the strengthened route is capped/calibrated to avoid over-reversing sparse top-tied conflicts (E13/E14 near 0) and to leave E1 probes near 0.57 and the E17 high-vs-low window differential near 0.05-0.06. Baseline weights and betas should be fitted to simultaneously reproduce E2 backward adherence near 0.69, E3-E7 mixture patterns, E5 near -0.13, E6 near -0.27, E9 near 0.15, and E18 near 0.64. This differs from pi_6 by abandoning the trailing-window frequency trigger and by predicting maximal rather than attenuated reversal at high density, and differs from pi_7_1 by dropping the near-tie tally gate, raising the effective density floor so 4/6 trials no longer drive the accumulator, restricting top-tied fallback to 6/6 trials, and removing the full-fraction penalty that currently suppresses E8/E15 reversal.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.906)

**Description**
People choose between two products through a baseline mixture of a weak forward lexicographic scan, a moderate lowest-validity-first lexicographic scan, and feature-win tallying. Decision makers continuously accumulate local evidence about cue reliability from recent discriminating trials. Evidence is graded by discrimination density: near-tied trials with only a few discriminating features contribute little or nothing, while dense five-of-six or fully discriminating trials drive a leaky accumulator. The accumulator has asymmetric rise and decay, a minimum qualifying count, hysteresis, and smooth activation of a strengthened lowest-validity-first route. When fully discriminating trials dominate the local evidence stream, the reversed route is gently capped below its maximum so strongly complementary designs reverse strongly but not excessively. When dense conflict is sparse, evidence decays back to baseline mixture weights, preserving near-zero active-inactive contrasts in sparse designs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(x, y, order, beta):
        winner = None
        for j in order:
            if x[j] > y[j]:
                winner = 0
                break
            if y[j] > x[j]:
                winner = 1
                break
        if winner is None:
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_backward_strong = float(parameters['beta_backward_strong'])
    beta_tally = float(parameters['beta_tally'])

    p_forward = lex_probabilities(a, b, descending, beta_forward)
    p_backward = lex_probabilities(a, b, ascending, beta_backward)
    p_shift = lex_probabilities(a, b, ascending, beta_backward_strong)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / np.sum(weights)
    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    tie_margin = float(parameters['tie_margin'])
    hi_lo_bonus = float(parameters['hi_lo_bonus'])
    base_evidence = float(parameters['base_evidence'])
    tied_top_evidence = float(parameters['tied_top_evidence'])
    density_floor = float(parameters['density_floor'])
    hi_idx = int(descending[0])
    lo_idx = int(ascending[0])

    def trial_evidence(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_discrim = int(np.sum(d != 0.0))
        density = n_discrim / max(1, n_features)
        tally = float(np.sum(d))
        near = 1.0 if abs(tally) <= tie_margin else 0.0
        if near <= 0.0 or density <= density_floor:
            return 0.0

        density_gain = (density - density_floor) / (1.0 - density_floor)
        hi_dir = np.sign(d[hi_idx])
        lo_dir = np.sign(d[lo_idx])
        disagree = 1.0 if (hi_dir != 0.0 and lo_dir != 0.0 and hi_dir == -lo_dir) else 0.0

        if hi_dir == 0.0:
            base = tied_top_evidence
        else:
            base = base_evidence + hi_lo_bonus * disagree

        return float(min(1.0, base * density_gain))

    evidence_seq = []
    full_flags = []
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        x = np.asarray(past_a[i], dtype=float)
        y = np.asarray(past_b[i], dtype=float)
        evidence_seq.append(trial_evidence(x, y))
        full_flags.append(1.0 if np.all((x - y) != 0.0) else 0.0)
    evidence_seq.append(trial_evidence(a, b))
    full_flags.append(1.0 if np.all((a - b) != 0.0) else 0.0)

    evidence_rise = float(parameters['evidence_rise'])
    evidence_decay = float(parameters['evidence_decay'])
    zero_decay_boost = float(parameters['zero_decay_boost'])
    threshold = float(parameters['conflict_threshold'])
    hysteresis = float(parameters['hysteresis'])
    activation_rise = float(parameters['activation_rise'])
    activation_decay = float(parameters['activation_decay'])
    min_qual = int(parameters['min_qualifying_count'])
    s_max = float(parameters['s_max'])
    full_penalty_threshold = float(parameters['full_penalty_threshold'])
    full_penalty_depth = float(parameters['full_penalty_depth'])

    X = 0.0
    S = 0.0
    n_seen = 0
    for idx, u in enumerate(evidence_seq):
        if u >= X:
            X += evidence_rise * (u - X)
        else:
            decay_mult = zero_decay_boost if u <= 0.0 else 1.0
            X -= evidence_decay * decay_mult * (X - u)

        if u > 0.0:
            n_seen += 1

        n_evidence = 0
        n_full = 0
        for k in range(idx + 1):
            if evidence_seq[k] > 0.0:
                n_evidence += 1
                if full_flags[k] > 0.5:
                    n_full += 1
        full_frac = (n_full / max(1, n_evidence)) if n_evidence > 0 else 0.0

        if full_frac > full_penalty_threshold:
            full_penalty = full_penalty_depth * ((full_frac - full_penalty_threshold) / (1.0 - full_penalty_threshold + 1e-9))
        else:
            full_penalty = 0.0
        s_max_eff = s_max * (1.0 - full_penalty)

        if n_seen < min_qual:
            target = 0.0
        else:
            threshold_eff = threshold - hysteresis if S > 0.5 else threshold
            target = s_max_eff if X >= threshold_eff else 0.0

        if target > S:
            S += activation_rise * (target - S)
        else:
            S -= activation_decay * (S - target)

    s = float(np.clip(S, 0.0, s_max))
    p = (1.0 - s) * p_base + s * p_shift

    epsilon = float(parameters['epsilon'])
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)

    p = np.clip(p, 0.0, None)
    return p / np.sum(p)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0
    probs = probs / probs.sum()
    return int(np.random.choice(probs.size, p=probs))


`parameters`:
- w_forward: [0.22, 0.32]
- w_backward: [0.42, 0.54]
- beta_forward: [0.08, 0.20]
- beta_backward: [1.30, 1.70]
- beta_tally: [1.80, 2.60]
- beta_backward_strong: [2.40, 4.00]
- tie_margin: [2.0, 3.0]
- hi_lo_bonus: [0.30, 0.55]
- base_evidence: [0.62, 0.78]
- tied_top_evidence: [0.50, 0.72]
- density_floor: {0.80}
- evidence_rise: [0.22, 0.42]
- evidence_decay: [0.02, 0.12]
- zero_decay_boost: [1.20, 2.50]
- conflict_threshold: [0.03, 0.05]
- hysteresis: [0.02, 0.05]
- activation_rise: [0.08, 0.30]
- activation_decay: [0.01, 0.08]
- min_qualifying_count: {2, 3}
- s_max: [0.75, 0.85]
- full_penalty_threshold: [0.85, 0.95]
- full_penalty_depth: [0.10, 0.18]
- epsilon: [0.0, 0.02]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2487 (var=0.0065) vs this=0.2331 (var=0.0017)
- Experiment 2: real=0.4196 (var=0.0128) vs this=0.4373 (var=0.0096)
- Experiment 3: real=-0.1300 (var=0.0159) vs this=-0.1158 (var=0.0055)
- Experiment 4: real=-0.2742 (var=0.0069) vs this=-0.2704 (var=0.0106)
- Experiment 5: real=-0.6583 (var=0.0598) vs this=-0.6529 (var=0.0087)
- Experiment 6: real=-0.6711 (var=0.0489) vs this=-0.5633 (var=0.0084)
- Experiment 7: real=0.1553 (var=0.0159) vs this=0.1667 (var=0.0021)
- Experiment 8: real=0.0150 (var=0.0027) vs this=0.0208 (var=0.0062)
- Experiment 9: real=-0.0283 (var=0.0032) vs this=-0.0013 (var=0.0043)
- Experiment 10: real=-0.0008 (var=0.0091) vs this=-0.0071 (var=0.0128)
- Experiment 11: real=-0.0050 (var=0.0198) vs this=0.0408 (var=0.0280)
- Experiment 12: real=-0.0178 (var=0.0089) vs this=-0.0200 (var=0.0134)
- Experiment 13: real=0.8425 (var=0.0101) vs this=0.7425 (var=0.0023)
- Experiment 14: real=0.8413 (var=0.0122) vs this=0.8350 (var=0.0017)
- Experiment 15: real=0.0553 (var=0.0224) vs this=0.0339 (var=0.0118)
- Experiment 16: real=0.6412 (var=0.0315) vs this=0.7974 (var=0.0100)
- Experiment 17: real=0.5710 (var=0.0228) vs this=0.7837 (var=0.0089)
- Experiment 18: real=0.6942 (var=0.0025) vs this=0.6952 (var=0.0018)


---

### `pi_7_1` (overall score: 0.793)

**Description**
People choose between two products through a baseline mixture of a weak forward lexicographic scan, a moderate lowest-validity-first lexicographic scan, and feature-win tallying. Decision makers continuously accumulate local evidence about cue reliability from recent discriminating trials. Evidence is graded by discrimination density: near-tied trials with only a few discriminating features contribute little or nothing, while dense five-of-six or fully discriminating trials drive a leaky accumulator. The accumulator has asymmetric rise and decay, a minimum qualifying count, hysteresis, and smooth activation of a strengthened lowest-validity-first route. When fully discriminating trials dominate the local evidence stream, the reversed route is gently capped below its maximum so strongly complementary designs reverse strongly but not excessively. When dense conflict is sparse, evidence decays back to baseline mixture weights, preserving near-zero active-inactive contrasts in sparse designs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(x, y, order, beta):
        winner = None
        for j in order:
            if x[j] > y[j]:
                winner = 0
                break
            if y[j] > x[j]:
                winner = 1
                break
        if winner is None:
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_backward_strong = float(parameters['beta_backward_strong'])
    beta_tally = float(parameters['beta_tally'])

    p_forward = lex_probabilities(a, b, descending, beta_forward)
    p_backward = lex_probabilities(a, b, ascending, beta_backward)
    p_shift = lex_probabilities(a, b, ascending, beta_backward_strong)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / np.sum(weights)
    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    tie_margin = float(parameters['tie_margin'])
    hi_lo_bonus = float(parameters['hi_lo_bonus'])
    base_evidence = float(parameters['base_evidence'])
    tied_top_evidence = float(parameters['tied_top_evidence'])
    density_floor = float(parameters['density_floor'])
    near_full_boost = float(parameters['near_full_boost'])
    top_tied_tally_min = float(parameters['top_tied_tally_min'])
    top_tied_bonus = float(parameters['top_tied_bonus'])
    hi_idx = int(descending[0])
    lo_idx = int(ascending[0])

    def trial_evidence(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_discrim = int(np.sum(d != 0.0))
        density = n_discrim / max(1, n_features)
        tally = float(np.sum(d))
        near = 1.0 if abs(tally) <= tie_margin else 0.0
        full = 1.0 if n_discrim >= n_features else 0.0
        if near <= 0.0:
            return 0.0

        hi_dir = np.sign(d[hi_idx])
        lo_dir = np.sign(d[lo_idx])
        disagree = 1.0 if (hi_dir != 0.0 and lo_dir != 0.0 and hi_dir == -lo_dir) else 0.0
        tally_align = 1.0 if (hi_dir == 0.0 or float(tally) * hi_dir >= 0.0) else 0.0

        if full > 0.5:
            density_gain = 1.0
            if hi_dir == 0.0:
                base = tied_top_evidence
            else:
                base = base_evidence + hi_lo_bonus * disagree
            return float(min(1.0, base * density_gain))

        # Non-full branch: top-tied fallback is narrowly gated by tally
        # magnitude so marginal top-tied sparse conflicts (e.g. E11) stay blocked,
        # while E13-style dense top-tied 5/6 conflicts can still emit evidence.
        if hi_dir == 0.0:
            fallback_hi_dir = 0.0
            for j in descending:
                if d[j] != 0.0:
                    fallback_hi_dir = float(np.sign(d[j]))
                    break
            if fallback_hi_dir == 0.0:
                return 0.0
            if abs(tally) < top_tied_tally_min:
                return 0.0
            disagree_t = 1.0 if (lo_dir != 0.0 and fallback_hi_dir != 0.0 and fallback_hi_dir == -lo_dir) else 0.0
            tally_align_t = 1.0 if (fallback_hi_dir != 0.0 and float(tally) * fallback_hi_dir >= 0.0) else 0.0
            if density <= density_floor or disagree_t <= 0.0 or tally_align_t <= 0.0:
                return 0.0
            density_gain = float(np.sqrt(max(0.0, (density - density_floor) / (1.0 - density_floor + 1e-9))))
            density_gain *= near_full_boost
            base = base_evidence + hi_lo_bonus + top_tied_bonus
            return float(min(1.0, base * density_gain))

        if density <= density_floor or disagree <= 0.0 or tally_align <= 0.0:
            return 0.0
        density_gain = float(np.sqrt(max(0.0, (density - density_floor) / (1.0 - density_floor + 1e-9))))
        density_gain *= near_full_boost
        base = base_evidence + hi_lo_bonus * disagree
        return float(min(1.0, base * density_gain))

    evidence_seq = []
    full_flags = []
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        x = np.asarray(past_a[i], dtype=float)
        y = np.asarray(past_b[i], dtype=float)
        evidence_seq.append(trial_evidence(x, y))
        full_flags.append(1.0 if np.all((x - y) != 0.0) else 0.0)
    evidence_seq.append(trial_evidence(a, b))
    full_flags.append(1.0 if np.all((a - b) != 0.0) else 0.0)

    evidence_rise = float(parameters['evidence_rise'])
    evidence_decay = float(parameters['evidence_decay'])
    zero_decay_boost = float(parameters['zero_decay_boost'])
    threshold = float(parameters['conflict_threshold'])
    saturation = float(parameters['saturation_evidence'])
    hysteresis = float(parameters['hysteresis'])
    activation_rise = float(parameters['activation_rise'])
    activation_decay = float(parameters['activation_decay'])
    min_qual = int(parameters['min_qualifying_count'])
    qualifying_window = int(parameters['qualifying_window'])
    full_frac_window = int(parameters['full_frac_window'])
    s_max = float(parameters['s_max'])
    full_penalty_center = float(parameters['full_penalty_threshold'])
    full_penalty_depth = float(parameters['full_penalty_depth'])
    full_penalty_width = float(parameters['full_penalty_width'])
    full_penalty_floor = float(parameters['full_penalty_floor'])

    X = 0.0
    S = 0.0

    for idx, u in enumerate(evidence_seq):
        if u >= X:
            X += evidence_rise * (u - X)
        else:
            decay_mult = zero_decay_boost if u <= 0.0 else 1.0
            X -= evidence_decay * decay_mult * (X - u)
        X = float(np.clip(X, 0.0, 1.0))

        q_start = max(0, idx - qualifying_window + 1)
        recent_q = np.asarray(evidence_seq[q_start:idx + 1], dtype=float)
        n_recent_qual = int(np.sum(recent_q > 0.0))

        f_start = max(0, idx - full_frac_window + 1)
        recent_flags = np.asarray(full_flags[f_start:idx + 1], dtype=float)
        window_len = idx - f_start + 1
        n_full = int(np.sum(recent_flags > 0.5))
        full_frac = n_full / max(1, window_len)

        width = max(full_penalty_width, 1e-9)
        rise_start = full_penalty_center - width
        fall_end = full_penalty_center + width
        if full_frac <= rise_start:
            full_penalty = 0.0
        elif full_frac < full_penalty_center:
            full_penalty = full_penalty_depth * max(0.0, (full_frac - rise_start)) / width
        elif full_frac < fall_end:
            full_penalty = full_penalty_floor + max(0.0, (full_penalty_depth - full_penalty_floor)) * max(0.0, (fall_end - full_frac)) / width
        else:
            full_penalty = full_penalty_floor
        full_penalty = float(np.clip(full_penalty, 0.0, 1.0))
        s_max_eff = s_max * (1.0 - full_penalty)

        if n_recent_qual < min_qual:
            target = 0.0
        else:
            threshold_eff = threshold - hysteresis if S > 0.5 else threshold
            if X <= threshold_eff:
                target = 0.0
            elif X >= saturation:
                target = s_max_eff
            else:
                target = s_max_eff * ((X - threshold_eff) / (saturation - threshold_eff + 1e-9))

        target = float(np.clip(target, 0.0, s_max_eff))
        if target > S:
            S += activation_rise * (target - S)
        else:
            S -= activation_decay * (S - target)
        S = float(np.clip(S, 0.0, s_max_eff))

    s = float(np.clip(S, 0.0, s_max))
    p = (1.0 - s) * p_base + s * p_shift

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)

    p = np.clip(p, 0.0, None)
    return p / np.sum(p)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0
    probs = probs / probs.sum()
    return int(np.random.choice(probs.size, p=probs))


`parameters`:
- w_forward: [0.25, 0.28]
- w_backward: [0.44, 0.52]
- beta_forward: [0.12, 0.20]
- beta_backward: [1.40, 1.60]
- beta_tally: [1.90, 2.40]
- beta_backward_strong: [2.60, 2.90]
- tie_margin: [2.5, 3.0]
- hi_lo_bonus: [0.45, 0.55]
- base_evidence: [0.38, 0.50]
- tied_top_evidence: [0.34, 0.44]
- density_floor: [0.56, 0.62]
- near_full_boost: [1.05, 1.12]
- evidence_rise: [0.35, 0.45]
- evidence_decay: [0.05, 0.08]
- zero_decay_boost: [1.20, 2.20]
- conflict_threshold: [0.03, 0.05]
- saturation_evidence: [0.25, 0.35]
- hysteresis: [0.02, 0.05]
- activation_rise: [0.22, 0.30]
- activation_decay: [0.04, 0.07]
- min_qualifying_count: {4}
- qualifying_window: {16}
- full_frac_window: {16}
- s_max: [0.72, 0.78]
- full_penalty_threshold: [0.50, 0.60]
- full_penalty_depth: [0.15, 0.25]
- full_penalty_width: [0.15, 0.22]
- full_penalty_floor: [0.04, 0.10]
- top_tied_tally_min: {2, 3}
- top_tied_bonus: [0.10, 0.20]
- epsilon: {0}
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2487 (var=0.0065) vs this=0.2379 (var=0.0032)
- Experiment 2: real=0.4196 (var=0.0128) vs this=0.4431 (var=0.0090)
- Experiment 3: real=-0.1300 (var=0.0159) vs this=-0.1113 (var=0.0052)
- Experiment 4: real=-0.2742 (var=0.0069) vs this=-0.2825 (var=0.0100)
- Experiment 5: real=-0.6583 (var=0.0598) vs this=-0.6750 (var=0.0079)
- Experiment 6: real=-0.6711 (var=0.0489) vs this=-0.5422 (var=0.0126)
- Experiment 7: real=0.1553 (var=0.0159) vs this=0.1767 (var=0.0022)
- Experiment 8: real=0.0150 (var=0.0027) vs this=0.0188 (var=0.0073)
- Experiment 9: real=-0.0283 (var=0.0032) vs this=0.0021 (var=0.0056)
- Experiment 10: real=-0.0008 (var=0.0091) vs this=0.0033 (var=0.0095)
- Experiment 11: real=-0.0050 (var=0.0198) vs this=0.0367 (var=0.0284)
- Experiment 12: real=-0.0178 (var=0.0089) vs this=-0.0478 (var=0.0131)
- Experiment 13: real=0.8425 (var=0.0101) vs this=0.7629 (var=0.0021)
- Experiment 14: real=0.8413 (var=0.0122) vs this=0.8244 (var=0.0014)
- Experiment 15: real=0.0553 (var=0.0224) vs this=0.0529 (var=0.0354)
- Experiment 16: real=0.6412 (var=0.0315) vs this=0.7112 (var=0.0135)
- Experiment 17: real=0.5710 (var=0.0228) vs this=0.5890 (var=0.0153)
- Experiment 18: real=0.6942 (var=0.0025) vs this=0.7494 (var=0.0014)


---

### `pi_4` (overall score: 0.750)

**Description**
People choose between two options through a stable, subject-level mixture of three comparison routes: (1) forward lexicographic scanning, inspecting cues in descending validity order and stopping at the first discriminating cue; (2) backward lexicographic scanning, inspecting cues in ascending validity order and stopping at the first discriminating cue in that reversed order; and (3) feature-win tallying, where each option receives one point per feature on which it strictly beats the other and ties contribute nothing. The forward and backward lexicographic weights jointly act as a subject-level polarity parameter for first-validity-cue effects: forward weight supports a positive first-discriminator effect, while backward weight supports a negative one. Each route's preferred option is passed through its own softmax choice rule, so responding is noisy but not controlled by a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Preferred validity vector comes from the experiment.  If the
    # parameter resolver ever leaves it unresolved, fall back to a
    # descending validity vector so the model is still scorable.
    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.ones(2, dtype=float) / 2.0

        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- w_forward: [0.22, 0.32]
- w_backward: [0.42, 0.54]
- beta_forward: [0.08, 0.20]
- beta_backward: [1.30, 1.70]
- beta_tally: [1.80, 2.60]
- epsilon: {0}
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2487 (var=0.0065) vs this=0.2306 (var=0.0024)
- Experiment 2: real=0.4196 (var=0.0128) vs this=0.4335 (var=0.0093)
- Experiment 3: real=-0.1300 (var=0.0159) vs this=-0.1325 (var=0.0113)
- Experiment 4: real=-0.2742 (var=0.0069) vs this=-0.2879 (var=0.0081)
- Experiment 5: real=-0.6583 (var=0.0598) vs this=-0.2833 (var=0.0124)
- Experiment 6: real=-0.6711 (var=0.0489) vs this=-0.0722 (var=0.0187)
- Experiment 7: real=0.1553 (var=0.0159) vs this=0.4160 (var=0.0041)
- Experiment 8: real=0.0150 (var=0.0027) vs this=-0.0162 (var=0.0110)
- Experiment 9: real=-0.0283 (var=0.0032) vs this=-0.0012 (var=0.0120)
- Experiment 10: real=-0.0008 (var=0.0091) vs this=0.0100 (var=0.0111)
- Experiment 11: real=-0.0050 (var=0.0198) vs this=0.0250 (var=0.0283)
- Experiment 12: real=-0.0178 (var=0.0089) vs this=-0.0067 (var=0.0117)
- Experiment 13: real=0.8425 (var=0.0101) vs this=0.5792 (var=0.0037)
- Experiment 14: real=0.8413 (var=0.0122) vs this=0.6440 (var=0.0025)
- Experiment 15: real=0.0553 (var=0.0224) vs this=0.0258 (var=0.0268)
- Experiment 16: real=0.6412 (var=0.0315) vs this=0.3880 (var=0.0187)
- Experiment 17: real=0.5710 (var=0.0228) vs this=0.4962 (var=0.0247)
- Experiment 18: real=0.6942 (var=0.0025) vs this=0.7056 (var=0.0015)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0982 -> ACCEPTED
- iter 2: loss=0.0845 -> ACCEPTED
- iter 3: loss=0.0873 -> REJECTED
- iter 4: loss=0.0908 -> REJECTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.0845 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = None
        # Cues are in descending-validity order, so the first discriminating
        # feature is the Take The Best cue.
        for j in range(len(a)):
            if a[j] > b[j]:
                pred = 0
                break
            if b[j] > a[j]:
                pred = 1
                break
        if pred is not None:
            matches.append(1.0 if r == pred else 0.0)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2487 (var=0.0065)
**Previous candidate values (this loop):**
  - iter 1: 0.2042 (var=0.0019) (Δ vs real -0.0446)
  - iter 2: 0.1681 (var=0.0017) (Δ vs real -0.0806)
  - iter 3: 0.1542 (var=0.0016) (Δ vs real -0.0946)
  - iter 4 (most recent): 0.1992 (var=0.0019) (Δ vs real -0.0496)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8723 (var=0.0096)
- pi_2: 0.1175 (var=0.0068)
- pi_3: 0.2483 (var=0.0023)
- pi_4: 0.2306 (var=0.0024)
- pi_5: 0.2258 (var=0.0024)
- pi_5_1: 0.2360 (var=0.0031)
- pi_5_2: 0.2146 (var=0.0032)
- pi_6: 0.2292 (var=0.0019)
- pi_7: 0.2331 (var=0.0017)
- pi_7_1: 0.2379 (var=0.0032)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        response_a = 1.0 if int(row["response"]) == 0 else 0.0

        # Take The Best style second-cue prediction.
        # By design, feature 0 is tied on every trial, so feature 1 is
        # the first discriminating cue.
        if a[1] > b[1]:
            second_wins_a = True
        elif b[1] > a[1]:
            second_wins_a = False
        else:
            second_wins_a = True

        second_score = response_a if second_wins_a else (1.0 - response_a)

        # Tallying total score. Feature-specific ties contribute zero.
        wins_a = float(np.sum(a > b))
        wins_b = float(np.sum(b > a))
        tally_d = wins_a - wins_b

        if tally_d > 0.0:
            tally_score = response_a
        elif tally_d < 0.0:
            tally_score = 1.0 - response_a
        else:
            # Tallying has no preference on a tied tally.
            tally_score = 0.5

        scores.append(tally_score - second_score)

    return float(np.mean(scores)) if scores else 0.0
```

**Observed (real) value:** 0.4196 (var=0.0128)
**Previous candidate values (this loop):**
  - iter 1: 0.4546 (var=0.0070) (Δ vs real +0.0350)
  - iter 2: 0.4581 (var=0.0059) (Δ vs real +0.0385)
  - iter 3: 0.5165 (var=0.0083) (Δ vs real +0.0969)
  - iter 4 (most recent): 0.4823 (var=0.0059) (Δ vs real +0.0627)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6304 (var=0.0462)
- pi_1: -0.6075 (var=0.0314)
- pi_3: 0.4169 (var=0.0101)
- pi_4: 0.4335 (var=0.0093)
- pi_5: 0.4321 (var=0.0103)
- pi_5_1: 0.4519 (var=0.0082)
- pi_5_2: 0.4796 (var=0.0123)
- pi_6: 0.4558 (var=0.0077)
- pi_7: 0.4373 (var=0.0096)
- pi_7_1: 0.4431 (var=0.0090)

### Experiment 3
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    a1 = data["option_a_ratings"].apply(lambda x: x[1])
    b1 = data["option_b_ratings"].apply(lambda x: x[1])
    f1_a = a1 > b1
    f1_b = b1 > a1
    chose_a = data["response"].astype(int) == 0
    if f1_a.sum() == 0 or f1_b.sum() == 0:
        return 0.0
    return float(chose_a[f1_a].mean() - chose_a[f1_b].mean())
```

**Observed (real) value:** -0.1300 (var=0.0159)
**Previous candidate values (this loop):**
  - iter 1: -0.1275 (var=0.0093) (Δ vs real +0.0025)
  - iter 2: -0.1404 (var=0.0062) (Δ vs real -0.0104)
  - iter 3: -0.1525 (var=0.0086) (Δ vs real -0.0225)
  - iter 4 (most recent): -0.1333 (var=0.0063) (Δ vs real -0.0033)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2375 (var=0.0045)
- pi_2: -0.0158 (var=0.0048)
- pi_1: 0.6592 (var=0.0472)
- pi_4: -0.1325 (var=0.0113)
- pi_5: -0.1446 (var=0.0066)
- pi_5_1: -0.1329 (var=0.0102)
- pi_5_2: -0.1471 (var=0.0086)
- pi_6: -0.1254 (var=0.0091)
- pi_7: -0.1158 (var=0.0055)
- pi_7_1: -0.1113 (var=0.0052)

### Experiment 4
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.88, 0.80, 0.72, 0.65, 0.60, 0.55, 0.52])
    order = np.argsort(-validities)
    pos = []
    neg = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        first = 'A'
        found = False
        for j in order:
            if a[j] > b[j]:
                first = 'A'
                found = True
                break
            if b[j] > a[j]:
                first = 'B'
                found = True
                break
        if not found:
            continue
        chose_a = 1.0 if row.response == 0 else 0.0
        if first == 'A':
            pos.append(chose_a)
        else:
            neg.append(chose_a)
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(np.mean(pos) - np.mean(neg))

```

**Observed (real) value:** -0.2742 (var=0.0069)
**Previous candidate values (this loop):**
  - iter 1: -0.2458 (var=0.0099) (Δ vs real +0.0283)
  - iter 2: -0.2642 (var=0.0134) (Δ vs real +0.0100)
  - iter 3: -0.3446 (var=0.0101) (Δ vs real -0.0704)
  - iter 4 (most recent): -0.2588 (var=0.0085) (Δ vs real +0.0154)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0109)
- pi_3: 0.2375 (var=0.0114)
- pi_1: 0.6925 (var=0.0418)
- pi_4: -0.2879 (var=0.0081)
- pi_5: -0.2883 (var=0.0135)
- pi_5_1: -0.3025 (var=0.0121)
- pi_5_2: -0.3025 (var=0.0138)
- pi_6: -0.2946 (var=0.0138)
- pi_7: -0.2704 (var=0.0106)
- pi_7_1: -0.2825 (var=0.0100)

### Experiment 5
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    a0 = np.array([int(r[0]) for r in data['option_a_ratings']])
    b0 = np.array([int(r[0]) for r in data['option_b_ratings']])
    choose_a = (np.asarray(data['response'], dtype=int) == 0).astype(float)
    top_a = a0 > b0
    top_b = ~top_a
    if not np.any(top_a) or not np.any(top_b):
        return 0.0
    return float(choose_a[top_a].mean() - choose_a[top_b].mean())
```

**Observed (real) value:** -0.6583 (var=0.0598)
**Previous candidate values (this loop):**
  - iter 1: -0.5596 (var=0.0062) (Δ vs real +0.0988)
  - iter 2: -0.5833 (var=0.0115) (Δ vs real +0.0750)
  - iter 3: -0.6121 (var=0.0040) (Δ vs real +0.0463)
  - iter 4 (most recent): -0.5400 (var=0.0071) (Δ vs real +0.1183)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2613 (var=0.0080)
- pi_4: -0.2833 (var=0.0124)
- pi_1: 0.6971 (var=0.0436)
- pi_2: -0.0012 (var=0.0086)
- pi_5: -0.6717 (var=0.0368)
- pi_5_1: -0.5975 (var=0.0258)
- pi_5_2: -0.7096 (var=0.0160)
- pi_6: -0.6721 (var=0.0051)
- pi_7: -0.6529 (var=0.0087)
- pi_7_1: -0.6750 (var=0.0079)

### Experiment 6
**Design**
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    if len(data) == 0:
        return 0.0

    a = np.array([list(x) for x in data['option_a_ratings']], dtype=int)
    b = np.array([list(x) for x in data['option_b_ratings']], dtype=int)
    diff = a - b

    top_cue = diff[:, 0]
    tally_diff = (diff > 0).sum(axis=1) - (diff < 0).sum(axis=1)

    chose_a = (data['response'].to_numpy() == 0).astype(float)

    top_a_plus = (top_cue > 0) & (tally_diff >= 1)
    top_b_minus = (top_cue < 0) & (tally_diff <= -1)

    mean_top_a = np.mean(chose_a[top_a_plus]) if np.any(top_a_plus) else 0.0
    mean_top_b = np.mean(chose_a[top_b_minus]) if np.any(top_b_minus) else 0.0

    return float(mean_top_a - mean_top_b)
```

**Observed (real) value:** -0.6711 (var=0.0489)
**Previous candidate values (this loop):**
  - iter 1: -0.2106 (var=0.0211) (Δ vs real +0.4606)
  - iter 2: -0.3911 (var=0.0151) (Δ vs real +0.2800)
  - iter 3: -0.3972 (var=0.0128) (Δ vs real +0.2739)
  - iter 4 (most recent): -0.2372 (var=0.0142) (Δ vs real +0.4339)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0722 (var=0.0187)
- pi_3: 0.9950 (var=0.0001)
- pi_1: 0.6939 (var=0.0333)
- pi_2: 0.7450 (var=0.0289)
- pi_5: -0.6683 (var=0.0378)
- pi_5_1: -0.6233 (var=0.0227)
- pi_5_2: -0.7394 (var=0.0104)
- pi_6: -0.6406 (var=0.0120)
- pi_7: -0.5633 (var=0.0084)
- pi_7_1: -0.5422 (var=0.0126)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    choices = []
    for _, row in data.iterrows():
        a = [int(x) for x in row['option_a_ratings']]
        b = [int(x) for x in row['option_b_ratings']]
        if a[0] == b[0]:
            continue
        top = 0 if a[0] > b[0] else 1
        top_vec = a if top == 0 else b
        other_vec = b if top == 0 else a
        d_top = 0
        for i in range(len(top_vec)):
            if top_vec[i] > other_vec[i]:
                d_top += 1
            elif other_vec[i] > top_vec[i]:
                d_top -= 1
        if d_top >= 0:
            choices.append(1.0 if int(row['response']) == top else 0.0)
    if not choices:
        return 0.5
    return float(np.mean(choices))
```

**Observed (real) value:** 0.1553 (var=0.0159)
**Previous candidate values (this loop):**
  - iter 1: 0.3477 (var=0.0051) (Δ vs real +0.1923)
  - iter 2: 0.2763 (var=0.0027) (Δ vs real +0.1210)
  - iter 3: 0.2753 (var=0.0040) (Δ vs real +0.1200)
  - iter 4 (most recent): 0.3270 (var=0.0029) (Δ vs real +0.1717)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1807 (var=0.0115)
- pi_4: 0.4160 (var=0.0041)
- pi_1: 0.8197 (var=0.0124)
- pi_2: 0.7093 (var=0.0045)
- pi_3: 0.8537 (var=0.0015)
- pi_5_1: 0.1960 (var=0.0071)
- pi_5_2: 0.1453 (var=0.0037)
- pi_6: 0.1447 (var=0.0028)
- pi_7: 0.1667 (var=0.0021)
- pi_7_1: 0.1767 (var=0.0022)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    validities = np.array([0.9, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5], dtype=float)
    j_hi = int(np.argmax(validities))
    j_lo = int(np.argmin(validities))

    a_arr = np.array([np.asarray(x, dtype=float) for x in data['option_a_ratings']])
    b_arr = np.array([np.asarray(x, dtype=float) for x in data['option_b_ratings']])
    resp = np.asarray(data['response'], dtype=float)

    hi_diff = a_arr[:, j_hi] - b_arr[:, j_hi]
    lo_diff = a_arr[:, j_lo] - b_arr[:, j_lo]
    d = (a_arr > b_arr).sum(axis=1) - (b_arr > a_arr).sum(axis=1)

    hi_dir = np.where(hi_diff > 0, 1, np.where(hi_diff < 0, -1, 0))
    lo_dir = np.where(lo_diff > 0, 1, np.where(lo_diff < 0, -1, 0))
    live = (hi_dir != 0) & (lo_dir != 0) & (hi_dir == -lo_dir) & (d * hi_dir >= 0)

    y = np.where(lo_dir > 0, (resp == 0).astype(float),
                 np.where(lo_dir < 0, (resp == 1).astype(float), 0.5))

    if np.sum(live) == 0 or np.sum(~live) == 0:
        return 0.0
    return float(np.mean(y[live]) - np.mean(y[~live]))
```

**Observed (real) value:** 0.0150 (var=0.0027)
**Previous candidate values (this loop):**
  - iter 1: 0.0338 (var=0.0062) (Δ vs real +0.0188)
  - iter 2: -0.0429 (var=0.0110) (Δ vs real -0.0579)
  - iter 3: 0.0046 (var=0.0121) (Δ vs real -0.0104)
  - iter 4 (most recent): -0.0067 (var=0.0115) (Δ vs real -0.0217)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0162 (var=0.0110)
- pi_5: 0.2750 (var=0.0251)
- pi_1: -0.0038 (var=0.0040)
- pi_2: 0.0008 (var=0.0059)
- pi_3: 0.0004 (var=0.0001)
- pi_5_1: -0.0021 (var=0.0064)
- pi_5_2: -0.0021 (var=0.0144)
- pi_6: 0.0238 (var=0.0106)
- pi_7: 0.0208 (var=0.0062)
- pi_7_1: 0.0188 (var=0.0073)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.90, 0.78, 0.66, 0.58, 0.52, 0.52])
    top_idx = int(np.argmax(validities))
    low_idx = np.where(validities == np.min(validities))[0]

    subject_effects = []
    for sid, sub in data.groupby('subject_id'):
        groups = {1: {'active': [], 'inactive': []},
                  -1: {'active': [], 'inactive': []}}
        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            top_dir = int(np.sign(a[top_idx] - b[top_idx]))
            if top_dir == 0:
                continue
            lows = [int(np.sign(a[j] - b[j])) for j in low_idx]
            nonzero = [d for d in lows if d != 0]
            if len(nonzero) > 0 and all(d == nonzero[0] for d in nonzero):
                low_dir = nonzero[0]
            else:
                low_dir = 0
            tally_d = int(np.sum(a - b))
            gate = (low_dir != 0 and top_dir == -low_dir and tally_d * top_dir >= 0)
            y_a = 1.0 - float(row['response'])
            signed_y = float(top_dir) * y_a
            groups[top_dir]['active' if gate else 'inactive'].append(signed_y)

        dir_effects = []
        for d in (1, -1):
            active = groups[d]['active']
            inactive = groups[d]['inactive']
            if len(active) > 0 and len(inactive) > 0:
                dir_effects.append(float(np.mean(inactive) - np.mean(active)))
        if dir_effects:
            subject_effects.append(float(np.mean(dir_effects)))

    if not subject_effects:
        return 0.0
    return float(np.mean(subject_effects))
```

**Observed (real) value:** -0.0283 (var=0.0032)
**Previous candidate values (this loop):**
  - iter 1: 0.0025 (var=0.0080) (Δ vs real +0.0308)
  - iter 2: 0.0054 (var=0.0101) (Δ vs real +0.0338)
  - iter 3: -0.0175 (var=0.0109) (Δ vs real +0.0108)
  - iter 4 (most recent): 0.0058 (var=0.0126) (Δ vs real +0.0342)
**Other theories' values on this metric (for reference):**
- pi_5_1: 0.2417 (var=0.0120)
- pi_4: -0.0012 (var=0.0120)
- pi_1: -0.0150 (var=0.0031)
- pi_2: 0.0033 (var=0.0040)
- pi_3: -0.0004 (var=0.0002)
- pi_5: -0.0029 (var=0.0043)
- pi_5_2: 0.0038 (var=0.0084)
- pi_6: -0.0358 (var=0.0108)
- pi_7: -0.0013 (var=0.0043)
- pi_7_1: 0.0021 (var=0.0056)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 1]

**Metric**
```python
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a0 = data['option_a_ratings'].apply(lambda x: x[0]).astype(int)
    b0 = data['option_b_ratings'].apply(lambda x: x[0]).astype(int)
    a7 = data['option_a_ratings'].apply(lambda x: x[7]).astype(int)
    b7 = data['option_b_ratings'].apply(lambda x: x[7]).astype(int)

    top_is_a = a0 > b0
    gate_active = a7 == b7

    choose_top = ((top_is_a & (data['response'] == 0)) |
                  (~top_is_a & (data['response'] == 1))).astype(float)

    if gate_active.sum() == 0 or (~gate_active).sum() == 0:
        return 0.0

    return float(choose_top[~gate_active].mean() - choose_top[gate_active].mean())

```

**Observed (real) value:** -0.0008 (var=0.0091)
**Previous candidate values (this loop):**
  - iter 1: 0.0054 (var=0.0151) (Δ vs real +0.0063)
  - iter 2: 0.0104 (var=0.0129) (Δ vs real +0.0112)
  - iter 3: -0.0025 (var=0.0136) (Δ vs real -0.0017)
  - iter 4 (most recent): 0.0046 (var=0.0127) (Δ vs real +0.0054)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0100 (var=0.0111)
- pi_5_1: 0.2717 (var=0.0167)
- pi_1: -0.0088 (var=0.0044)
- pi_2: -0.0075 (var=0.0048)
- pi_3: 0.0021 (var=0.0002)
- pi_5: -0.0029 (var=0.0062)
- pi_5_2: -0.0125 (var=0.0074)
- pi_6: 0.0096 (var=0.0118)
- pi_7: -0.0071 (var=0.0128)
- pi_7_1: 0.0033 (var=0.0095)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.5], dtype=float)
    descending = np.argsort(-validities)

    A = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    B = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    R = np.asarray(data['response'].tolist(), dtype=float)
    chose_a = (R == 0).astype(float)

    n = A.shape[0]
    if n == 0:
        return 0.0

    top_dir = np.full(n, -1, dtype=int)
    for j in descending:
        mask = (top_dir == -1) & (A[:, j] != B[:, j])
        top_dir[mask] = (B[mask, j] > A[mask, j]).astype(int)

    gate_active = (A[:, 0] != B[:, 0])
    cond = np.full(n, -1, dtype=int)
    cond[(top_dir == 0) & gate_active] = 0
    cond[(top_dir == 0) & (~gate_active)] = 1
    cond[(top_dir == 1) & gate_active] = 2
    cond[(top_dir == 1) & (~gate_active)] = 3

    cell_means = []
    for k in range(4):
        rows = (cond == k)
        if rows.sum() == 0:
            return 0.0
        cell_means.append(float(chose_a[rows].mean()))

    return float(cell_means[1] - cell_means[0] + cell_means[2] - cell_means[3])
```

**Observed (real) value:** -0.0050 (var=0.0198)
**Previous candidate values (this loop):**
  - iter 1: -0.0175 (var=0.0374) (Δ vs real -0.0125)
  - iter 2: 0.0117 (var=0.0373) (Δ vs real +0.0167)
  - iter 3: 0.0192 (var=0.0518) (Δ vs real +0.0242)
  - iter 4 (most recent): 0.0508 (var=0.0415) (Δ vs real +0.0558)
**Other theories' values on this metric (for reference):**
- pi_5_2: 0.6500 (var=0.0457)
- pi_4: 0.0250 (var=0.0283)
- pi_1: -0.0150 (var=0.0273)
- pi_2: 0.0342 (var=0.0199)
- pi_3: -0.0058 (var=0.0006)
- pi_5: 0.5325 (var=0.0892)
- pi_5_1: 0.5525 (var=0.0595)
- pi_6: 0.0367 (var=0.0440)
- pi_7: 0.0408 (var=0.0280)
- pi_7_1: 0.0367 (var=0.0284)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np

def metric(data):
    if data is None or len(data) == 0:
        return 0.0
    validities = np.array([0.9, 0.8, 0.72, 0.61, 0.55, 0.5])
    order = np.argsort(-validities, kind='stable')
    hi_idx = int(order[0])
    second_idx = int(order[1])
    lo_idx = int(order[-1])
    active_low = []
    inactive_low = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        resp = row.response
        if a.shape != validities.shape or b.shape != validities.shape:
            continue
        if a[lo_idx] == b[lo_idx]:
            continue
        low_dir = 1 if a[lo_idx] > b[lo_idx] else -1
        low_chosen = 1.0 if ((low_dir == 1 and resp == 0) or (low_dir == -1 and resp == 1)) else 0.0
        tally_d = int(np.sum(a > b) - np.sum(b > a))
        if a[hi_idx] != b[hi_idx]:
            hi_dir = 1 if a[hi_idx] > b[hi_idx] else -1
            if hi_dir == -low_dir and tally_d * hi_dir >= 0:
                active_low.append(low_chosen)
        else:
            if a[second_idx] != b[second_idx]:
                hi_dir = 1 if a[second_idx] > b[second_idx] else -1
                if hi_dir == -low_dir and tally_d * hi_dir >= 0:
                    inactive_low.append(low_chosen)
    if not active_low or not inactive_low:
        return 0.0
    return float(np.mean(active_low) - np.mean(inactive_low))
```

**Observed (real) value:** -0.0178 (var=0.0089)
**Previous candidate values (this loop):**
  - iter 1: 0.0339 (var=0.0101) (Δ vs real +0.0517)
  - iter 2: -0.0100 (var=0.0118) (Δ vs real +0.0078)
  - iter 3: 0.0172 (var=0.0147) (Δ vs real +0.0350)
  - iter 4 (most recent): -0.0200 (var=0.0140) (Δ vs real -0.0022)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0067 (var=0.0117)
- pi_5_2: 0.2894 (var=0.0132)
- pi_1: -0.0050 (var=0.0090)
- pi_2: 0.0017 (var=0.0129)
- pi_3: -0.0067 (var=0.0049)
- pi_5: 0.2600 (var=0.0188)
- pi_5_1: 0.2211 (var=0.0170)
- pi_6: -0.0206 (var=0.0128)
- pi_7: -0.0200 (var=0.0134)
- pi_7_1: -0.0478 (var=0.0131)

### Experiment 13
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.9, 0.8, 0.72, 0.62, 0.61, 0.6], dtype=float)
    low_idx = int(np.argmin(validities))
    a = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b = np.array(data['option_b_ratings'].tolist(), dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0:
        return 0.5
    a_low = a[:, low_idx]
    b_low = b[:, low_idx]
    valid = np.not_equal(a_low, b_low)
    if not np.any(valid):
        return 0.5
    low_cue_choice = np.where(a_low > b_low, 0, 1)
    response = data['response'].to_numpy(dtype=float)
    match = np.equal(response, low_cue_choice).astype(float)
    return float(np.mean(match[valid]))
```

**Observed (real) value:** 0.8425 (var=0.0101)
**Previous candidate values (this loop):**
  - iter 1: 0.6850 (var=0.0036) (Δ vs real -0.1575)
  - iter 2: 0.6998 (var=0.0024) (Δ vs real -0.1427)
  - iter 3: 0.7258 (var=0.0025) (Δ vs real -0.1167)
  - iter 4 (most recent): 0.6806 (var=0.0033) (Δ vs real -0.1619)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8048 (var=0.0022)
- pi_4: 0.5792 (var=0.0037)
- pi_1: 0.1235 (var=0.0079)
- pi_2: 0.3046 (var=0.0038)
- pi_3: 0.1665 (var=0.0008)
- pi_5: 0.7015 (var=0.0047)
- pi_5_1: 0.6865 (var=0.0051)
- pi_5_2: 0.7231 (var=0.0028)
- pi_7: 0.7425 (var=0.0023)
- pi_7_1: 0.7629 (var=0.0021)

### Experiment 14
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data):
    import numpy as np

    # Feature index 5 is the unique lowest-validity expert and discriminates in every pair.
    a5 = data['option_a_ratings'].apply(lambda x: int(x[5])).to_numpy()
    resp = data['response'].to_numpy(dtype=float)

    # A choice is feature-5-consistent if it picks the option whose feature 5 equals 1.
    match = np.where(a5 == 1, 1.0 - resp, resp)
    return float(np.mean(match))

```

**Observed (real) value:** 0.8413 (var=0.0122)
**Previous candidate values (this loop):**
  - iter 1: 0.8381 (var=0.0011) (Δ vs real -0.0031)
  - iter 2: 0.8494 (var=0.0015) (Δ vs real +0.0081)
  - iter 3: 0.8469 (var=0.0014) (Δ vs real +0.0056)
  - iter 4 (most recent): 0.8477 (var=0.0021) (Δ vs real +0.0065)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6440 (var=0.0025)
- pi_6: 0.7856 (var=0.0023)
- pi_1: 0.1590 (var=0.0096)
- pi_2: 0.4971 (var=0.0017)
- pi_3: 0.3656 (var=0.0013)
- pi_5: 0.7935 (var=0.0075)
- pi_5_1: 0.7956 (var=0.0031)
- pi_5_2: 0.8431 (var=0.0024)
- pi_7: 0.8350 (var=0.0017)
- pi_7_1: 0.8244 (var=0.0014)

### Experiment 15
**Design**
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Metric**
```python
import numpy as np

def metric(data):
    validities = np.array([0.9, 0.8, 0.72, 0.61, 0.6, 0.6])
    ascending = np.argsort(validities, kind='stable')
    n_features = len(validities)
    window = 24
    low_cut = 2
    high_cut = 3
    min_trial = 30

    subject_scores = []
    for subject_id, subj in data.groupby('subject_id', sort=False):
        subj = subj.reset_index(drop=True)
        n = len(subj)
        if n <= min_trial:
            continue

        is_full = np.zeros(n, dtype=bool)
        back_cons = np.zeros(n, dtype=float)
        has_winner = np.zeros(n, dtype=bool)

        for i in range(n):
            a = np.asarray(subj.loc[i, 'option_a_ratings'], dtype=float)
            b = np.asarray(subj.loc[i, 'option_b_ratings'], dtype=float)
            diff = a - b
            n_disc = int(np.sum(diff != 0.0))
            tally = float(np.sum(diff))
            if n_disc >= n_features and abs(tally) <= 3.0:
                is_full[i] = True

            winner = -1
            for j in ascending:
                if a[j] > b[j]:
                    winner = 0
                    break
                if b[j] > a[j]:
                    winner = 1
                    break
            if winner < 0:
                continue
            has_winner[i] = True
            back_cons[i] = 1.0 if int(subj.loc[i, 'response']) == winner else 0.0

        prefix = np.concatenate((np.zeros(1), np.cumsum(is_full.astype(float))))
        low_mask = np.zeros(n, dtype=bool)
        high_mask = np.zeros(n, dtype=bool)
        eligible = np.zeros(n, dtype=bool)

        for i in range(n):
            if i < min_trial or i < window - 1:
                continue
            start = i - window + 1
            full_count = prefix[i + 1] - prefix[start]
            eligible[i] = True
            if full_count < low_cut:
                low_mask[i] = True
            elif full_count >= high_cut:
                high_mask[i] = True

        non_full = ~is_full
        low_sel = eligible & low_mask & non_full & has_winner
        high_sel = eligible & high_mask & non_full & has_winner

        if not low_sel.any() or not high_sel.any():
            continue

        low_mean = float(np.mean(back_cons[low_sel]))
        high_mean = float(np.mean(back_cons[high_sel]))
        subject_scores.append(high_mean - low_mean)

    if not subject_scores:
        return 0.0
    return float(np.mean(subject_scores))
```

**Observed (real) value:** 0.0553 (var=0.0224)
**Previous candidate values (this loop):**
  - iter 1: 0.0136 (var=0.0352) (Δ vs real -0.0417)
  - iter 2: 0.0292 (var=0.0302) (Δ vs real -0.0261)
  - iter 3: -0.0040 (var=0.0215) (Δ vs real -0.0593)
  - iter 4 (most recent): 0.0022 (var=0.0258) (Δ vs real -0.0531)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2099 (var=0.0205)
- pi_7: 0.0339 (var=0.0118)
- pi_1: 0.0181 (var=0.0166)
- pi_2: 0.1104 (var=0.0222)
- pi_3: 0.1243 (var=0.0156)
- pi_4: 0.0258 (var=0.0268)
- pi_5: 0.0293 (var=0.0291)
- pi_5_1: 0.0363 (var=0.0247)
- pi_5_2: 0.0421 (var=0.0316)
- pi_7_1: 0.0529 (var=0.0354)

### Experiment 16
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    low_order = [3, 4, 5]
    burn = 24

    sub_diffs = []
    for _, sub in data.groupby('subject_id', sort=False):
        sub = sub.reset_index(drop=True)
        infos = []
        full_flags = []

        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            d = a - b
            full_flags.append(1.0 if np.all(d != 0.0) else 0.0)

            if d[0] == 0:
                infos.append(None)
                continue

            top_dir = 1.0 if d[0] > 0 else -1.0
            low_dir = 0.0
            for j in low_order:
                if d[j] != 0.0:
                    low_dir = 1.0 if d[j] > 0 else -1.0
                    break

            if low_dir == 0.0:
                infos.append(None)
                continue

            top_choice = 0 if top_dir > 0 else 1
            matched = 1.0 if int(row['response']) == top_choice else 0.0
            aligned = top_dir * low_dir > 0
            infos.append((matched, aligned))

        aligned_vals = []
        conflict_vals = []

        for i in range(burn, len(sub)):
            window = full_flags[i - 23:i + 1]
            frac = float(np.mean(window))
            if frac < 0.5:
                continue

            info = infos[i]
            if info is None:
                continue

            matched, aligned = info
            if aligned:
                aligned_vals.append(matched)
            else:
                conflict_vals.append(matched)

        if len(aligned_vals) == 0 or len(conflict_vals) == 0:
            continue

        sub_diffs.append(float(np.mean(aligned_vals) - np.mean(conflict_vals)))

    if not sub_diffs:
        return 0.0
    return float(np.mean(sub_diffs))

```

**Observed (real) value:** 0.6412 (var=0.0315)
**Previous candidate values (this loop):**
  - iter 1: 0.5256 (var=0.0230) (Δ vs real -0.1157)
  - iter 2: 0.6015 (var=0.0140) (Δ vs real -0.0397)
  - iter 3: 0.6217 (var=0.0186) (Δ vs real -0.0195)
  - iter 4 (most recent): 0.5623 (var=0.0189) (Δ vs real -0.0789)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7974 (var=0.0100)
- pi_6: 0.6026 (var=0.0164)
- pi_1: -0.0026 (var=0.0106)
- pi_2: 0.2110 (var=0.0313)
- pi_3: 0.2314 (var=0.0135)
- pi_4: 0.3880 (var=0.0187)
- pi_5: 0.5215 (var=0.0194)
- pi_5_1: 0.4401 (var=0.0172)
- pi_5_2: 0.4013 (var=0.0236)
- pi_7_1: 0.7112 (var=0.0135)

### Experiment 17
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    if data.empty:
        return 0.0
    A = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    B = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    y = data['response'].to_numpy(dtype=float)
    if A.ndim != 2 or A.shape[1] < 6:
        return 0.0
    d = A - B
    n_disc = np.sum(d != 0.0, axis=1)
    tally = np.sum(d, axis=1)
    d0 = d[:, 0]
    d4 = d[:, 4]
    d5 = d[:, 5]
    topA = (d0 > 0.0) & (d5 < 0.0) & (d4 <= 0.0) & (n_disc == 5) & (tally < 0.0)
    topB = (d0 < 0.0) & (d5 > 0.0) & (d4 >= 0.0) & (n_disc == 5) & (tally > 0.0)
    if topA.sum() == 0 or topB.sum() == 0:
        return 0.0
    pB_topA = float(np.mean(y[topA]))
    pB_topB = float(np.mean(y[topB]))
    return float(pB_topA - pB_topB)
```

**Observed (real) value:** 0.5710 (var=0.0228)
**Previous candidate values (this loop):**
  - iter 1: 0.6044 (var=0.0152) (Δ vs real +0.0335)
  - iter 2: 0.7139 (var=0.0149) (Δ vs real +0.1430)
  - iter 3: 0.7256 (var=0.0067) (Δ vs real +0.1546)
  - iter 4 (most recent): 0.5920 (var=0.0203) (Δ vs real +0.0211)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7742 (var=0.0080)
- pi_7_1: 0.5890 (var=0.0153)
- pi_1: -0.6544 (var=0.0529)
- pi_2: 0.7242 (var=0.0366)
- pi_3: 0.4645 (var=0.0136)
- pi_4: 0.4962 (var=0.0247)
- pi_5: 0.5211 (var=0.0189)
- pi_5_1: 0.5061 (var=0.0183)
- pi_5_2: 0.4869 (var=0.0207)
- pi_7: 0.7837 (var=0.0089)

### Experiment 18
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.75, 0.73], dtype=float)
    ascending = np.argsort(validities, kind='stable')
    A = np.array(data['option_a_ratings'].tolist(), dtype=float)
    B = np.array(data['option_b_ratings'].tolist(), dtype=float)
    resp = data['response'].to_numpy()
    diff = A - B
    scores = []
    for i in range(diff.shape[0]):
        bwd = -1
        for j in ascending:
            if diff[i, j] > 0.0:
                bwd = 0
                break
            if diff[i, j] < 0.0:
                bwd = 1
                break
        if bwd != -1:
            scores.append(1.0 if int(resp[i]) == bwd else 0.0)
    if len(scores) == 0:
        return 0.5
    return float(np.mean(scores))
```

**Observed (real) value:** 0.6942 (var=0.0025)
**Previous candidate values (this loop):**
  - iter 1: 0.7035 (var=0.0020) (Δ vs real +0.0094)
  - iter 2: 0.7060 (var=0.0017) (Δ vs real +0.0119)
  - iter 3: 0.7094 (var=0.0019) (Δ vs real +0.0152)
  - iter 4 (most recent): 0.6956 (var=0.0020) (Δ vs real +0.0015)
**Other theories' values on this metric (for reference):**
- pi_7_1: 0.7494 (var=0.0014)
- pi_6: 0.6875 (var=0.0016)
- pi_1: 0.6135 (var=0.0015)
- pi_2: 0.6225 (var=0.0032)
- pi_3: 0.6806 (var=0.0012)
- pi_4: 0.7056 (var=0.0015)
- pi_5: 0.7137 (var=0.0020)
- pi_5_1: 0.7254 (var=0.0025)
- pi_5_2: 0.7427 (var=0.0023)
- pi_7: 0.6952 (var=0.0018)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
People choose between two products through a baseline mixture of a weak forward lexicographic cue scan, a moderate lowest-validity-first lexicographic scan, and feature-win tallying. Decision makers continuously accumulate local evidence about cue ecology from dense discriminating trials. Only trials with at least five of six discriminating features—or fully discriminating trials—contribute evidence, and that evidence requires the highest-validity and lowest-validity cues to point in opposite directions. Fully discriminating top-cue-tied trials are a weak fallback source. The evidence enters a leaky asymmetric accumulator with a low minimum qualifying count, hysteresis, and smooth activation of a strengthened lowest-validity-first route. This calibrated high-density evidence path predicts strong reversal in conflict-dense designs while weak fallback evidence and accelerated zero-evidence decay keep sparse or top-tied contexts near baseline.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(x, y, order, beta):
        winner = None
        for j in order:
            if x[j] > y[j]:
                winner = 0
                break
            if y[j] > x[j]:
                winner = 1
                break
        if winner is None:
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_backward_strong = float(parameters['beta_backward_strong'])
    beta_tally = float(parameters['beta_tally'])

    p_forward = lex_probabilities(a, b, descending, beta_forward)
    p_backward = lex_probabilities(a, b, ascending, beta_backward)
    p_shift = lex_probabilities(a, b, ascending, beta_backward_strong)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / np.sum(weights)
    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    base_evidence = float(parameters['base_evidence'])
    tied_top_evidence = float(parameters['tied_top_evidence'])
    partial_gain = float(parameters['partial_gain'])
    conflict_threshold = float(parameters['conflict_threshold'])
    saturation_evidence = float(parameters['saturation_evidence'])
    hysteresis = float(parameters['hysteresis'])
    evidence_rise = float(parameters['evidence_rise'])
    evidence_decay = float(parameters['evidence_decay'])
    zero_decay_boost = float(parameters['zero_decay_boost'])
    activation_rise = float(parameters['activation_rise'])
    activation_decay = float(parameters['activation_decay'])
    min_qual = int(parameters['min_qualifying_count'])
    s_max = float(parameters['s_max'])
    epsilon = float(parameters.get('epsilon', 0.0))

    min_v = float(np.min(validities))
    n_tied_low = int(np.sum(np.isclose(validities, min_v, atol=1e-6)))
    low_tie_factor = float(parameters['low_tie_penalty']) ** max(0, n_tied_low - 1)
    s_max_eff = float(np.clip(s_max * low_tie_factor, 0.0, 1.0))

    hi_idx = int(descending[0])
    lo_idx = int(ascending[0])

    min_disc = n_features - 1 if n_features >= 6 else n_features

    def trial_evidence(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_disc = int(np.sum(d != 0.0))
        full = 1.0 if n_disc >= n_features else 0.0

        if n_disc < min_disc:
            return 0.0

        hi_dir = float(np.sign(d[hi_idx]))
        lo_dir = float(np.sign(d[lo_idx]))

        if hi_dir != 0.0 and lo_dir != 0.0 and hi_dir != lo_dir:
            gain = 1.0 if full > 0.5 else partial_gain
            return float(min(1.0, base_evidence * gain))

        if hi_dir == 0.0 and full > 0.5:
            fallback_hi_dir = 0.0
            for j in descending:
                if d[j] != 0.0:
                    fallback_hi_dir = float(np.sign(d[j]))
                    break
            if fallback_hi_dir != 0.0 and lo_dir != 0.0 and fallback_hi_dir != lo_dir:
                return float(min(1.0, tied_top_evidence))

        return 0.0

    evidence_seq = []
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        x = np.asarray(past_a[i], dtype=float)
        y = np.asarray(past_b[i], dtype=float)
        evidence_seq.append(trial_evidence(x, y))
    evidence_seq.append(trial_evidence(a, b))

    X = 0.0
    S = 0.0

    for idx, u in enumerate(evidence_seq):
        if u >= X:
            X += evidence_rise * (u - X)
        else:
            decay_mult = zero_decay_boost if u <= 0.0 else 1.0
            X -= evidence_decay * decay_mult * (X - u)
        X = float(np.clip(X, 0.0, 1.0))

        n_seen = 0
        for k in range(idx + 1):
            if evidence_seq[k] > 0.0:
                n_seen += 1

        if n_seen < min_qual:
            target = 0.0
        else:
            threshold_eff = conflict_threshold - hysteresis if S > 0.5 else conflict_threshold
            if X <= threshold_eff:
                target = 0.0
            elif X >= saturation_evidence:
                target = s_max_eff
            else:
                target = s_max_eff * ((X - threshold_eff) / (saturation_evidence - threshold_eff + 1e-9))

        target = float(np.clip(target, 0.0, s_max_eff))
        if target > S:
            S += activation_rise * (target - S)
        else:
            S -= activation_decay * (S - target)
        S = float(np.clip(S, 0.0, s_max_eff))

    s = float(np.clip(S, 0.0, s_max))
    p = (1.0 - s) * p_base + s * p_shift

    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)

    p = np.clip(p, 0.0, None)
    return p / np.sum(p)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0
    probs = probs / probs.sum()
    return int(np.random.choice(probs.size, p=probs))


`parameters`:
- w_forward: [0.20, 0.26]
- w_backward: [0.44, 0.50]
- beta_forward: [0.08, 0.16]
- beta_backward: [1.35, 1.55]
- beta_tally: [1.90, 2.30]
- beta_backward_strong: [3.60, 4.20]
- base_evidence: [0.78, 0.88]
- tied_top_evidence: [0.10, 0.25]
- partial_gain: [0.55, 0.75]
- low_tie_penalty: [0.95, 1.00]
- evidence_rise: [0.32, 0.42]
- evidence_decay: [0.03, 0.07]
- zero_decay_boost: [4.0, 6.0]
- conflict_threshold: [0.20, 0.26]
- saturation_evidence: [0.45, 0.60]
- hysteresis: [0.02, 0.05]
- activation_rise: [0.18, 0.26]
- activation_decay: [0.02, 0.05]
- min_qualifying_count: {2, 3}
- s_max: [0.62, 0.70]
- epsilon: [0.0, 0.01]
- validities: validities

`rationale`: This is a minimal calibration edit of the running-best candidate. The critic's diagnosis was that conflict-dense reversal was undermodeled because 5/6 dense evidence was too weak, the accumulator threshold was too high, and the qualifying count was too conservative. I raised partial_gain to 0.55-0.75, lowered conflict_threshold to 0.20-0.26, reduced min_qualifying_count to 2-3, sped up activation, relaxed the tied-low penalty, and sharply weakened the top-tied fallback. I also strengthened zero-evidence decay so sparse designs such as E11 do not accumulate spurious evidence after the threshold was lowered. Critically, I kept s_max modest and beta_backward_strong only moderately higher than before: the previous E14 fit was already near its 0.841 target, so a large global ceiling increase would have overshot low-cue adherence there. The gains are instead produced by faster, more reliable recruitment of the strengthened backward route in dense top-low conflict streams, which repairs E5/E6/E13/E16 without forcing E14 or the sparse top-tied metrics away from their targets.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed mechanism family: it replaces the trailing-window trigger with per-trial graded conflict-density evidence, imposes a density floor, excludes tally-nearness, uses a leaky asymmetric accumulator, and smooths activation of a strengthened lowest-validity-first route. Several baseline and sparse-conflict metrics are well reproduced (E3, E14, E17, E18, E10, E11). However, the conflict-dense reversal metrics are systematically undermodeled. E6 is a severe miss: real -0.671 vs candidate -0.211. E5 is too weak (-0.658 vs -0.560), E13 is far too low in low-cue adherence (0.843 vs 0.685), and E16 is under-reversed (0.641 vs 0.526). At the same time E7 is over-following the top cue (0.155 vs 0.348), implying the strengthened backward route is not activating strongly enough in the contexts it should. E12 has the wrong sign (real -0.018 vs candidate +0.034), suggesting the top-tied fallback evidence is too strong or too easily accumulated. The low between-subject variance on several reversal metrics also suggests the accumulator switches too uniformly across simulated subjects.
Rationale: The mechanism can remain intact, but its evidence/activation calibration is wrong. The main failure is that 5/6-density top-low conflicts supply too little evidence: with partial_gain only 0.20-0.32, the per-trial evidence never rises past conflict_threshold for many dense 5/6 streams, so min_qualifying_count=4 is rarely reached and the strengthened route stays near baseline. This explains E6/E13 under-reversal and likely E7's residual top-cue following. Raise partial_gain toward 0.6-0.9 (or make 5/6 evidence closer to full strength), lower conflict_threshold to around 0.15-0.25, and/or reduce min_qualifying_count to 2-3, while keeping the 4/6 gate so E2 remains near baseline. Additionally raise s_max to roughly 0.75-0.85 and/or beta_backward_strong to 4.0-5.0 to repair E5/E6/E13/E16, but keep the cap from overshooting E14. The top-tied fallback should be weakened (tied_top_evidence toward 0.10-0.25) because it is the likely cause of the E12 sign error. Finally, the low_tie_penalty should be relaxed toward 1.0 unless it can be shown necessary, since tied-low designs E15/E16 currently appear under-activated. These are in-family calibration changes, not a switch from the prescribed accumulator evidence mechanism.

**Outcome of this advice:** iter 1 candidate loss=0.0982 -> iter 2 candidate loss=0.0845 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate is a faithful implementation of the prescribed per-trial graded conflict-density accumulator: dense top-low conflict drives a strengthened lowest-validity-first route, 4/6 trials emit no evidence, top-tied fallback is restricted to fully discriminating trials, and there is no tally-nearness gate. The last round of calibration clearly helped: aggregate loss improved from 0.0982 to 0.0845, and the dense-conflict deficits were partially repaired (E5 moved from +0.099 to +0.075, E6 from +0.461 to +0.280, E13 from -0.158 to -0.143, E16 from -0.116 to -0.040). Several baseline and sparse metrics are already close (E3 -0.010, E4 +0.010, E10 +0.011, E12 +0.008, E14 +0.008, E18 +0.012). However, the theory is not yet experiment-invariant. E6 remains a large under-reversal (-0.391 vs -0.671, residual +0.280), E13 still under-follows the low cue (0.700 vs 0.843, residual -0.143), and the strengthened route now over-reverses in other dense-window contexts: E17 is 0.714 vs 0.571 (+0.143), E7 is 0.276 vs 0.155 (+0.121), and E8 has flipped sign relative to the near-zero real value (-0.043 vs +0.015). This pattern suggests the prior global strengthening was partly right but now too uniform: a single 5/6 evidence gain and single saturation ceiling cannot simultaneously distinguish the fully-discriminating unbalanced conflicts that drive E6 from the windowed n_disc=5 conflicts that drive E17, while sparse top-tied trials in E13 remain too reliant on the weak baseline backward route.
Rationale: Keep the prescribed accumulator family but recalibrate evidence grading rather than shifting all evidence strength in one direction. Specifically: (1) differentiate full 6/6 evidence from 5/6 evidence more sharply: keep full-discrimination evidence near ceiling but keep the 5/6 partial contribution moderate (partial_gain around 0.50-0.60, not pushed higher), and/or slow saturation for partial evidence via a slightly higher saturation_evidence or lower activation_rise; this should pull E17 back toward 0.57 and E7 toward 0.16 while preserving the E6/E15/E16 gains. (2) Rebalance the baseline mixture instead of relying on the accumulator for E13: shift weight from tallying into the moderate backward route (w_backward toward 0.50-0.58, w_tally toward 0.18-0.28) to lift top-tied sparse low-cue adherence; compensate any E14/E18 overshoot by modestly trimming s_max or beta_backward_strong. (3) Repair the E8 sign flip with a modest threshold/hysteresis increase or reduced zero-decay boost rather than another global lowering of threshold, and monitor E6 while doing so. (4) The candidate's between-subject variance on key contrast metrics remains too small (e.g. E6 var 0.015 vs real 0.049); add subject-level variability in accumulator parameters such as threshold, min_qualifying_count, or activation rates.

**Outcome of this advice:** iter 2 candidate loss=0.0845 -> iter 3 candidate loss=0.0873 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is still inside the prescribed accumulation family, and a few of its targeted changes did work: restricting non-six-feature environments to fully discriminating evidence repaired E8 (0.0046 vs 0.0150), and dense-conflict metrics improved to varying degrees (E5 -0.612 vs -0.658, E6 -0.397 vs -0.671, E13 0.726 vs 0.843, E16 0.622 vs 0.641). But the aggregate gate rejection is informative: these gains were purchased with broad collateral damage in contexts that should stay near baseline. The clearest failures are E2 (+0.097), E4 (-0.070), E1 (-0.095), E12 (+0.035), E15 (-0.059), and especially E17 (+0.155). The pattern says the candidate over-shifted the baseline mixture toward the moderate backward route and made the strengthened route too persistent, so low-cue/anti-top responding leaks into non-diagnostic or partial-evidence probe trials. The previous approach—rebalance w_backward upward and differentiate 6/6 from 5/6 via partial_gain alone—was rejected by the gate and should not be repeated. Two implementation details are diagnostic. First, tied_top_evidence is bounded at 0.08-0.18 while conflict_threshold is 0.24-0.30 and the leaky evidence X cannot exceed the input value; therefore the full-discrimination top-tied fallback can never cross threshold and is effectively inert, which likely explains why E13 remained too low despite the baseline shift. Second, E17 stayed too high despite moderate partial_gain because the accumulator's activation S decays very slowly (activation_decay 0.02-0.05), so full 6/6 evidence accumulated earlier in the sequence keeps the strong route active during later 5/6 window trials; the same carryover mechanism plausibly drives the E1 over-reversal. Thus the problem is less the evidence input gain than the activation ceiling and decay dynamics.
Rationale: Build on the accepted iter2 base rather than this rejected variant. Keep the one clearly useful change from the current attempt—only fully discriminating trials are informative when n_features is not 6—so E8 stays near zero. Do not raise the global baseline w_backward to 0.52-0.58; that blunt shift is the most likely cause of the E1/E2/E4/E12/E17 regressions. Instead: (1) make the full 6/6 evidence route stronger but give 5/6 evidence a separate, lower activation ceiling, e.g. a partial_s_max of roughly 0.15-0.25 that caps S regardless of how high X climbs; this should pull E17 and E7 downward while preserving the full-conflict reversal needed for E5/E6/E16. (2) Increase the leak/decay on S during trials without strong full evidence, for example raise activation_decay and evidence_decay or strengthen the zero-evidence decay channel, so aligned E1 probes and 5/6 window probes relax toward the baseline mixture rather than inheriting earlier saturation; keep full 6/6 conflict trials able to reactivate the route. (3) Make the full-discrimination top-tied fallback actually able to cross threshold by either modestly raising tied_top_evidence above conflict_threshold (e.g. 0.28-0.35) or lowering conflict_threshold slightly; this can lift E13's low-cue adherence without another global baseline change, and should not damage E12 because most E12 inactive probes are non-full. (4) Do not raise conflict_threshold or hysteresis further—E15 moved from +0.029 to -0.004 under that change—and consider returning to the accepted iter2 threshold levels. These are all in-family accumulator/calibration adjustments, not a switch of mechanism family.

**Outcome of this advice:** iter 3 candidate loss=0.0873 -> iter 4 candidate loss=0.0908 -> the gate marked it REJECTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is still inside the prescribed accumulator family, and it does fix a few of the rejected predecessor's worst probes: E8 is now near zero (-0.0067 vs real 0.0150), E12 is almost exact (-0.0200 vs -0.0178), E17 is close (0.5920 vs 0.5710), and E18 is nearly perfect (0.6956 vs 0.6942). However, these gains came at the cost of the dense-conflict reversals that were the main point of the theory. E6 collapses from the already-weak -0.3911 on the accepted iter2 base to -0.2372 against a real -0.6711. E5 also weakens (-0.5400 vs -0.6583), E13 remains far too low in low-cue adherence (0.6806 vs 0.8425), E15 drops to essentially zero (0.0022 vs 0.0553), and E16 regresses to 0.5623 vs 0.6412. At the same time several probe metrics moved the wrong way: E2 (0.4823 vs 0.4196), E7 (0.3270 vs 0.1553), E9 (0.0058 vs -0.0283), and E11 (0.0508 vs -0.0050). The aggregate gate rejection (0.0908 vs accepted 0.0845) matches this diagnosis: the candidate over-corrected the sparse/windowed probes while deactivating exactly the dense 5/6 top-low conflict contexts that need strong reversed responding. The likely culprit is the combination of the new very low partial_s_max (0.15-0.25) plus the strong non-evidence decay-to-zero: E6/E7/E5 contain many 5/6 top-low conflicts, and those trials now get only a weak route activation while every intervening non-evidence trial drags activation back. The previous advice to cap 5/6 activation and accelerate non-evidence decay was implemented and rejected, so it should not be repeated.
Rationale: Start from the accepted iter2 base, not this rejected variant. Keep the accepted base's accumulator profile (no partial_s_max ceiling, no aggressive non-evidence zero-target reset). The new direction should target a baseline calibration knob instead of the accumulator: reduce beta_tally substantially from [1.90, 2.30] toward [1.10, 1.60], while keeping the baseline mixture weights and route betas fixed initially. The current tally route is too sharp, and this single over-sharpness explains several otherwise contradictory errors: E2 is over-tallying, E6/E5/E7 are too top-following on aligned top/tally trials, E13 is under-following the low cue where tally opposes it, and E17's over-reversal is inflated by tallying agreeing with the low cue. A softer tally route should move these metrics toward the real values without needing to weaken the strengthened backward route, which is the change that caused the current dense-conflict regressions. If E13/E6 remain under-reversed after lowering beta_tally, the second step can be a small increase in beta_backward (not w_backward) or a slight increase in s_max, but the weight-shift and partial-ceiling approaches already failed and should not be reintroduced. This remains a calibration change within the prescribed baseline-mixture plus accumulator family, not a mechanism switch.

**Outcome of this advice:** iter 4 candidate loss=0.0908 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "People choose between two products through a baseline mixture of a weak forward lexicographic cue scan, a moderate lowest-validity-first lexicographic scan, and a soft feature-win tallying route. Decision makers continuously accumulate local evidence about cue ecology from dense discriminating trials. Only trials with at least five of six discriminating features, or fully discriminating trials, contribute evidence, and that evidence requires the highest-validity and lowest-validity cues to point in opposite directions. Fully discriminating top-cue-tied trials are a weak fallback source. The evidence enters a leaky asymmetric accumulator with a low minimum qualifying count, hysteresis, and smooth activation of a strengthened lowest-validity-first route. The tally route is deliberately kept soft so it does not dominate aligned top/tally trials, allowing the backward route and accumulator-driven reversal to express themselves in dense conflict contexts.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim == 3 and stim.shape[0] == 1:\n        stim = stim[0]\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus.')\n\n    a = stim[0]\n    b = stim[1]\n    n_features = int(stim.shape[1])\n\n    validities = parameters.get('validities')\n    if validities is None:\n        validities = list(np.linspace(0.9, 0.5, n_features))\n    validities = np.asarray(validities, dtype=float)\n    if validities.ndim == 0 or validities.shape[0] != n_features:\n        validities = np.linspace(0.9, 0.5, n_features)\n\n    descending = np.argsort(-validities, kind='stable')\n    ascending = np.argsort(validities, kind='stable')\n\n    def lex_probabilities(x, y, order, beta):\n        winner = None\n        for j in order:\n            if x[j] > y[j]:\n                winner = 0\n                break\n            if y[j] > x[j]:\n                winner = 1\n                break\n        if winner is None:\n            return np.full(2, 0.5, dtype=float)\n        z = float(beta) if winner == 0 else -float(beta)\n        p_a = 1.0 / (1.0 + np.exp(-z))\n        return np.array([p_a, 1.0 - p_a], dtype=float)\n\n    beta_forward = float(parameters['beta_forward'])\n    beta_backward = float(parameters['beta_backward'])\n    beta_backward_strong = float(parameters['beta_backward_strong'])\n    beta_tally = float(parameters['beta_tally'])\n\n    p_forward = lex_probabilities(a, b, descending, beta_forward)\n    p_backward = lex_probabilities(a, b, ascending, beta_backward)\n    p_shift = lex_probabilities(a, b, ascending, beta_backward_strong)\n\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    tally_scores = np.array([a_wins, b_wins], dtype=float)\n    z = beta_tally * (tally_scores - np.max(tally_scores))\n    e = np.exp(z)\n    p_tally = e / np.sum(e)\n\n    w_forward = float(parameters['w_forward'])\n    w_backward = float(parameters['w_backward'])\n    w_tally = 1.0 - w_forward - w_backward\n    if w_tally < 0.0:\n        w_tally = 0.0\n    weights = np.array([w_forward, w_backward, w_tally], dtype=float)\n    weights = weights / np.sum(weights)\n    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally\n\n    base_evidence = float(parameters['base_evidence'])\n    tied_top_evidence = float(parameters['tied_top_evidence'])\n    partial_gain = float(parameters['partial_gain'])\n    conflict_threshold = float(parameters['conflict_threshold'])\n    saturation_evidence = float(parameters['saturation_evidence'])\n    hysteresis = float(parameters['hysteresis'])\n    evidence_rise = float(parameters['evidence_rise'])\n    evidence_decay = float(parameters['evidence_decay'])\n    zero_decay_boost = float(parameters['zero_decay_boost'])\n    activation_rise = float(parameters['activation_rise'])\n    activation_decay = float(parameters['activation_decay'])\n    min_qual = int(parameters['min_qualifying_count'])\n    s_max = float(parameters['s_max'])\n    epsilon = float(parameters.get('epsilon', 0.0))\n\n    min_v = float(np.min(validities))\n    n_tied_low = int(np.sum(np.isclose(validities, min_v, atol=1e-6)))\n    low_tie_factor = float(parameters['low_tie_penalty']) ** max(0, n_tied_low - 1)\n    s_max_eff = float(np.clip(s_max * low_tie_factor, 0.0, 1.0))\n\n    hi_idx = int(descending[0])\n    lo_idx = int(ascending[0])\n\n    min_disc = n_features - 1 if n_features >= 6 else n_features\n\n    def trial_evidence(x, y):\n        x = np.asarray(x, dtype=float)\n        y = np.asarray(y, dtype=float)\n        d = x - y\n        n_disc = int(np.sum(d != 0.0))\n        full = 1.0 if n_disc >= n_features else 0.0\n\n        if n_disc < min_disc:\n            return 0.0\n\n        hi_dir = float(np.sign(d[hi_idx]))\n        lo_dir = float(np.sign(d[lo_idx]))\n\n        if hi_dir != 0.0 and lo_dir != 0.0 and hi_dir != lo_dir:\n            gain = 1.0 if full > 0.5 else partial_gain\n            return float(min(1.0, base_evidence * gain))\n\n        if hi_dir == 0.0 and full > 0.5:\n            fallback_hi_dir = 0.0\n            for j in descending:\n                if d[j] != 0.0:\n                    fallback_hi_dir = float(np.sign(d[j]))\n                    break\n            if fallback_hi_dir != 0.0 and lo_dir != 0.0 and fallback_hi_dir != lo_dir:\n                return float(min(1.0, tied_top_evidence))\n\n        return 0.0\n\n    evidence_seq = []\n    past_a = history.get('option_a_ratings', [])\n    past_b = history.get('option_b_ratings', [])\n    n_past = min(len(past_a), len(past_b))\n    for i in range(n_past):\n        x = np.asarray(past_a[i], dtype=float)\n        y = np.asarray(past_b[i], dtype=float)\n        evidence_seq.append(trial_evidence(x, y))\n    evidence_seq.append(trial_evidence(a, b))\n\n    X = 0.0\n    S = 0.0\n\n    for idx, u in enumerate(evidence_seq):\n        if u >= X:\n            X += evidence_rise * (u - X)\n        else:\n            decay_mult = zero_decay_boost if u <= 0.0 else 1.0\n            X -= evidence_decay * decay_mult * (X - u)\n        X = float(np.clip(X, 0.0, 1.0))\n\n        n_seen = 0\n        for k in range(idx + 1):\n            if evidence_seq[k] > 0.0:\n                n_seen += 1\n\n        if n_seen < min_qual:\n            target = 0.0\n        else:\n            threshold_eff = conflict_threshold - hysteresis if S > 0.5 else conflict_threshold\n            if X <= threshold_eff:\n                target = 0.0\n            elif X >= saturation_evidence:\n                target = s_max_eff\n            else:\n                target = s_max_eff * ((X - threshold_eff) / (saturation_evidence - threshold_eff + 1e-9))\n\n        target = float(np.clip(target, 0.0, s_max_eff))\n        if target > S:\n            S += activation_rise * (target - S)\n        else:\n            S -= activation_decay * (S - target)\n        S = float(np.clip(S, 0.0, s_max_eff))\n\n    s = float(np.clip(S, 0.0, s_max))\n    p = (1.0 - s) * p_base + s * p_shift\n\n    if epsilon > 0.0:\n        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)\n\n    p = np.clip(p, 0.0, None)\n    return p / np.sum(p)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=float)\n    if probs.size == 0:\n        return 0\n    probs = probs / probs.sum()\n    return int(np.random.choice(probs.size, p=probs))\n",
  "parameters": {
    "w_forward": "[0.20, 0.26]",
    "w_backward": "[0.44, 0.50]",
    "beta_forward": "[0.08, 0.16]",
    "beta_backward": "[1.35, 1.55]",
    "beta_tally": "[1.10, 1.60]",
    "beta_backward_strong": "[3.60, 4.20]",
    "base_evidence": "[0.78, 0.88]",
    "tied_top_evidence": "[0.10, 0.25]",
    "partial_gain": "[0.55, 0.75]",
    "low_tie_penalty": "[0.95, 1.00]",
    "evidence_rise": "[0.32, 0.42]",
    "evidence_decay": "[0.03, 0.07]",
    "zero_decay_boost": "[4.0, 6.0]",
    "conflict_threshold": "[0.20, 0.26]",
    "saturation_evidence": "[0.45, 0.60]",
    "hysteresis": "[0.02, 0.05]",
    "activation_rise": "[0.18, 0.26]",
    "activation_decay": "[0.02, 0.05]",
    "min_qualifying_count": "{2, 3}",
    "s_max": "[0.62, 0.70]",
    "epsilon": "[0.0, 0.01]",
    "validities": "validities"
  },
  "rationale": "This is a minimal calibration edit of the running-best candidate. The only change is beta_tally, reduced from [1.90, 2.30] to [1.10, 1.60]. The previous tally route was too sharp: on aligned top/tally trials it over-rode the backward and forward routes, producing excessive top-cue following in E2/E5/E6/E7 and E17, while opposing the low-cue route in E13. A softer tally route rebalances these mixed-trial choices without weakening the dense-conflict accumulator path, which is the change that caused the rejected dense-conflict regressions in the prior iteration. All other accumulator, evidence, and baseline-mixture settings are preserved."
}
```

## Usage

```json
{
  "input_tokens": 54700,
  "output_tokens": 9628,
  "reasoning_tokens": 6799,
  "provider": "Baidu"
}
```
