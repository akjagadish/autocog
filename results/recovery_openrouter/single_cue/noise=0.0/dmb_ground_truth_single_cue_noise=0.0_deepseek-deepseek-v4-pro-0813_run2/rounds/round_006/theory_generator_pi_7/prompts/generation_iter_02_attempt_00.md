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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 with a new adaptive cue-reliability theory. The new theory should keep a baseline forward-lexicographic, backward-lexicographic, and tally route, but instead of using fixed mixture weights it should continuously update subjective cue weights from local trial statistics. The key state variables should be the density of recent discriminating cues, the signed tally margin, and whether high-validity and low-validity cues disagree. When many recent trials are fully discriminating and near-tied, the agent should infer that the nominal validity ordering is locally uninformative and temporarily shift weight toward the lowest-validity or most recently discriminating cue, producing the strong negative top-cue effects seen in Exp7, Exp8, and Exp9. When such conflict is sparse or absent, the weights should decay back to the default mixture, preserving the near-zero active-inactive contrasts in Exp11 through Exp14. Unlike pi_6's fixed 24-trial trapezoidal trigger, this theory should use a smooth accumulation and decay process, possibly with an asymmetric learning rate and hysteresis, so that it can better capture gradual density effects and the remaining small mismatches. The parameter ranges should include baseline route weights, baseline softmax precisions, the maximum reversed cue weight, a conflict accumulation rate, a decay rate, a conflict threshold, and a response noise term.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.960)

**Description**
People choose between two products through a stable, subject-level mixture of three routes: a weak forward lexicographic scan, a moderate backward/recency lexicographic scan, and feature-win tallying. Choice is modulated by a context-sensitive amplification of a higher-precision backward lexicographic route. The decision maker monitors the trailing 24-trial frequency of fully diagnostic, near-tied-tally comparisons, ignores isolated occurrences via a minimum qualifying count, and maps the qualifying fraction through a band-limited trapezoidal sensitivity curve. Moderate densities of such trials produce the strongest backward amplification, which generates the strong negative top-cue effects seen in dense complementary designs; very high densities attenuate to a lower floor, protecting all-complementary blocks from over-reversal. Sparse designs that never contain fully diagnostic near-tied trials keep the trigger at exactly zero, so their behavior stays governed by the pi_4-like baseline mixture, preserving the near-zero active-inactive contrasts in those experiments.

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
    n = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n:
        validities = np.linspace(0.9, 0.5, n)

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
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    tie_margin = float(parameters['tie_margin'])

    def complementarity_signal(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_discrim = int(np.sum(d != 0.0))
        tally = float(np.sum(d > 0.0) - np.sum(d < 0.0))
        return 1.0 if (n_discrim >= n and abs(tally) <= tie_margin) else 0.0

    signals = []
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        signals.append(complementarity_signal(past_a[i], past_b[i]))
    signals.append(complementarity_signal(a, b))

    context_window = int(parameters['context_window'])
    if len(signals) > context_window:
        signals = signals[-context_window:]

    n_recent = max(1, len(signals))
    n_qual = float(np.sum(signals))
    frac = n_qual / n_recent

    min_count = int(parameters['min_qualifying_count'])
    rise_start = float(parameters['rise_start_frac'])
    peak_start = float(parameters['peak_start_frac'])
    peak_end = float(parameters['peak_end_frac'])
    fall_end = float(parameters['fall_end_frac'])
    s_max = float(parameters['s_max'])
    s_floor = min(float(parameters['s_floor']), s_max)

    if n_qual < min_count:
        s = 0.0
    else:
        if frac <= rise_start:
            s = 0.0
        elif frac <= peak_start:
            s = s_max * (frac - rise_start) / (peak_start - rise_start + 1e-9)
        elif frac <= peak_end:
            s = s_max
        elif frac <= fall_end:
            s = s_max - (s_max - s_floor) * (frac - peak_end) / (fall_end - peak_end + 1e-9)
        else:
            s = s_floor
    s = float(np.clip(s, 0.0, 1.0))

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_backward_strong = float(parameters['beta_backward_strong'])

    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)
    p_backward_strong = lex_probabilities(ascending, beta_backward_strong)

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
    weights = weights / np.sum(weights)

    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally
    p = (1.0 - s) * p_base + s * p_backward_strong
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
- beta_backward_strong: [2.50, 4.00]
- context_window: {24}
- tie_margin: [2.0, 3.0]
- min_qualifying_count: {2, 3}
- rise_start_frac: [0.03, 0.06]
- peak_start_frac: [0.08, 0.14]
- peak_end_frac: [0.26, 0.34]
- fall_end_frac: [0.40, 0.48]
- s_max: [0.84, 0.92]
- s_floor: [0.40, 0.50]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2487 (var=0.0065) vs this=0.2292 (var=0.0019)
- Experiment 2: real=0.4196 (var=0.0128) vs this=0.4558 (var=0.0077)
- Experiment 3: real=-0.1300 (var=0.0159) vs this=-0.1254 (var=0.0091)
- Experiment 4: real=-0.2742 (var=0.0069) vs this=-0.2946 (var=0.0138)
- Experiment 5: real=-0.6583 (var=0.0598) vs this=-0.6721 (var=0.0051)
- Experiment 6: real=-0.6711 (var=0.0489) vs this=-0.6406 (var=0.0120)
- Experiment 7: real=0.1553 (var=0.0159) vs this=0.1447 (var=0.0028)
- Experiment 8: real=0.0150 (var=0.0027) vs this=0.0238 (var=0.0106)
- Experiment 9: real=-0.0283 (var=0.0032) vs this=-0.0358 (var=0.0108)
- Experiment 10: real=-0.0008 (var=0.0091) vs this=0.0096 (var=0.0118)
- Experiment 11: real=-0.0050 (var=0.0198) vs this=0.0367 (var=0.0440)
- Experiment 12: real=-0.0178 (var=0.0089) vs this=-0.0206 (var=0.0128)
- Experiment 13: real=0.8425 (var=0.0101) vs this=0.8048 (var=0.0022)
- Experiment 14: real=0.8413 (var=0.0122) vs this=0.7856 (var=0.0023)


---

### `pi_4` (overall score: 0.748)

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


---

### `pi_5_2` (overall score: 0.513)

**Description**
People compare options through a conflict-gated dual-lexicographic/tallying system. A baseline mixture combines a weak forward Take-The-Best route, a moderate backward lexicographic route, and feature-win tallying. When the highest-validity and lowest-validity cues discriminate in opposite directions and the tally does not oppose the highest cue, the decision maker enters a conflict regime: tallying is suspended and the choice is made with high probability by a strong backward lexicographic scan over ascending validities. Subjects differ substantially in how strongly and deterministically they reverse on such conflict trials.

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
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_conflict_back = float(parameters['beta_conflict_back'])

    p_forward = lex_probabilities(descending, beta_forward)
    p_backward_base = lex_probabilities(ascending, beta_backward)
    p_backward_conflict = lex_probabilities(ascending, beta_conflict_back)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (scores - np.max(scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally_base = 1.0 - w_forward - w_backward
    if w_tally_base < 0.0:
        w_tally_base = 0.0
        denom = w_forward + w_backward
        w_forward /= denom
        w_backward /= denom

    v_max = float(np.max(validities))
    v_min = float(np.min(validities))
    top_positions = np.where(validities == v_max)[0]
    bottom_positions = np.where(validities == v_min)[0]

    if top_positions.size == 1 and bottom_positions.size == 1:
        j_hi = int(top_positions[0])
        j_lo = int(bottom_positions[0])
        hi_dir = 0
        if a[j_hi] > b[j_hi]:
            hi_dir = 1
        elif b[j_hi] > a[j_hi]:
            hi_dir = -1
        lo_dir = 0
        if a[j_lo] > b[j_lo]:
            lo_dir = 1
        elif b[j_lo] > a[j_lo]:
            lo_dir = -1
        tally_d = int(a_wins - b_wins)
        extreme_cue_conflict = (hi_dir != 0 and lo_dir != 0 and hi_dir == -lo_dir)
        live_conflict = bool(extreme_cue_conflict and (tally_d * hi_dir >= 0))
    else:
        live_conflict = False

    if live_conflict:
        w_conflict_back = float(parameters['w_conflict_back'])
        w_back_eff = w_conflict_back
        w_forward_eff = 1.0 - w_back_eff
        if w_forward_eff < 0.0:
            w_forward_eff = 0.0
            w_back_eff = 1.0
        w_tally_eff = 0.0
        p_backward = p_backward_conflict
    else:
        w_forward_eff = w_forward
        w_back_eff = w_backward
        w_tally_eff = w_tally_base
        p_backward = p_backward_base

    weights = np.array([w_forward_eff, w_back_eff, w_tally_eff], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)

    p = np.clip(p, 0.0, None)
    return p / p.sum()


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- w_forward: [0.17, 0.30]
- w_backward: [0.36, 0.60]
- beta_forward: [0.0, 0.6]
- beta_backward: [1.0, 2.5]
- beta_tally: [1.5, 3.0]
- w_conflict_back: [0.75, 0.95]
- beta_conflict_back: [1.8, 4.0]
- epsilon: {0}
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2487 (var=0.0065) vs this=0.2146 (var=0.0032)
- Experiment 2: real=0.4196 (var=0.0128) vs this=0.4796 (var=0.0123)
- Experiment 3: real=-0.1300 (var=0.0159) vs this=-0.1471 (var=0.0086)
- Experiment 4: real=-0.2742 (var=0.0069) vs this=-0.3025 (var=0.0138)
- Experiment 5: real=-0.6583 (var=0.0598) vs this=-0.7096 (var=0.0160)
- Experiment 6: real=-0.6711 (var=0.0489) vs this=-0.7394 (var=0.0104)
- Experiment 7: real=0.1553 (var=0.0159) vs this=0.1453 (var=0.0037)
- Experiment 8: real=0.0150 (var=0.0027) vs this=-0.0021 (var=0.0144)
- Experiment 9: real=-0.0283 (var=0.0032) vs this=0.0038 (var=0.0084)
- Experiment 10: real=-0.0008 (var=0.0091) vs this=-0.0125 (var=0.0074)
- Experiment 11: real=-0.0050 (var=0.0198) vs this=0.6500 (var=0.0457)
- Experiment 12: real=-0.0178 (var=0.0089) vs this=0.2894 (var=0.0132)
- Experiment 13: real=0.8425 (var=0.0101) vs this=0.7231 (var=0.0028)
- Experiment 14: real=0.8413 (var=0.0122) vs this=0.8431 (var=0.0024)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0583 -> ACCEPTED
- iter 2: loss=0.0407 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.0407 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.2323 (var=0.0026) (Δ vs real -0.0165)
  - iter 2 (most recent): 0.2269 (var=0.0019) (Δ vs real -0.0219)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8723 (var=0.0096)
- pi_2: 0.1175 (var=0.0068)
- pi_3: 0.2483 (var=0.0023)
- pi_4: 0.2306 (var=0.0024)
- pi_5: 0.2258 (var=0.0024)
- pi_5_1: 0.2360 (var=0.0031)
- pi_5_2: 0.2146 (var=0.0032)
- pi_6: 0.2292 (var=0.0019)

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
  - iter 1: 0.4254 (var=0.0062) (Δ vs real +0.0058)
  - iter 2 (most recent): 0.4373 (var=0.0079) (Δ vs real +0.0177)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6304 (var=0.0462)
- pi_1: -0.6075 (var=0.0314)
- pi_3: 0.4169 (var=0.0101)
- pi_4: 0.4335 (var=0.0093)
- pi_5: 0.4321 (var=0.0103)
- pi_5_1: 0.4519 (var=0.0082)
- pi_5_2: 0.4796 (var=0.0123)
- pi_6: 0.4558 (var=0.0077)

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
  - iter 1: -0.0954 (var=0.0105) (Δ vs real +0.0346)
  - iter 2 (most recent): -0.1242 (var=0.0062) (Δ vs real +0.0058)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2375 (var=0.0045)
- pi_2: -0.0158 (var=0.0048)
- pi_1: 0.6592 (var=0.0472)
- pi_4: -0.1325 (var=0.0113)
- pi_5: -0.1446 (var=0.0066)
- pi_5_1: -0.1329 (var=0.0102)
- pi_5_2: -0.1471 (var=0.0086)
- pi_6: -0.1254 (var=0.0091)

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
  - iter 1: -0.2700 (var=0.0092) (Δ vs real +0.0042)
  - iter 2 (most recent): -0.2954 (var=0.0084) (Δ vs real -0.0212)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0109)
- pi_3: 0.2375 (var=0.0114)
- pi_1: 0.6925 (var=0.0418)
- pi_4: -0.2879 (var=0.0081)
- pi_5: -0.2883 (var=0.0135)
- pi_5_1: -0.3025 (var=0.0121)
- pi_5_2: -0.3025 (var=0.0138)
- pi_6: -0.2946 (var=0.0138)

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
  - iter 1: -0.6583 (var=0.0073) (Δ vs real +0.0000)
  - iter 2 (most recent): -0.7183 (var=0.0060) (Δ vs real -0.0600)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2613 (var=0.0080)
- pi_4: -0.2833 (var=0.0124)
- pi_1: 0.6971 (var=0.0436)
- pi_2: -0.0012 (var=0.0086)
- pi_5: -0.6717 (var=0.0368)
- pi_5_1: -0.5975 (var=0.0258)
- pi_5_2: -0.7096 (var=0.0160)
- pi_6: -0.6721 (var=0.0051)

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
  - iter 1: -0.4433 (var=0.0145) (Δ vs real +0.2278)
  - iter 2 (most recent): -0.5567 (var=0.0136) (Δ vs real +0.1144)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0722 (var=0.0187)
- pi_3: 0.9950 (var=0.0001)
- pi_1: 0.6939 (var=0.0333)
- pi_2: 0.7450 (var=0.0289)
- pi_5: -0.6683 (var=0.0378)
- pi_5_1: -0.6233 (var=0.0227)
- pi_5_2: -0.7394 (var=0.0104)
- pi_6: -0.6406 (var=0.0120)

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
  - iter 1: 0.2277 (var=0.0027) (Δ vs real +0.0723)
  - iter 2 (most recent): 0.1570 (var=0.0020) (Δ vs real +0.0017)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1807 (var=0.0115)
- pi_4: 0.4160 (var=0.0041)
- pi_1: 0.8197 (var=0.0124)
- pi_2: 0.7093 (var=0.0045)
- pi_3: 0.8537 (var=0.0015)
- pi_5_1: 0.1960 (var=0.0071)
- pi_5_2: 0.1453 (var=0.0037)
- pi_6: 0.1447 (var=0.0028)

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
  - iter 1: 0.0125 (var=0.0129) (Δ vs real -0.0025)
  - iter 2 (most recent): 0.0246 (var=0.0066) (Δ vs real +0.0096)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0162 (var=0.0110)
- pi_5: 0.2750 (var=0.0251)
- pi_1: -0.0038 (var=0.0040)
- pi_2: 0.0008 (var=0.0059)
- pi_3: 0.0004 (var=0.0001)
- pi_5_1: -0.0021 (var=0.0064)
- pi_5_2: -0.0021 (var=0.0144)
- pi_6: 0.0238 (var=0.0106)

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
  - iter 1: 0.0008 (var=0.0125) (Δ vs real +0.0292)
  - iter 2 (most recent): -0.0075 (var=0.0050) (Δ vs real +0.0208)
**Other theories' values on this metric (for reference):**
- pi_5_1: 0.2417 (var=0.0120)
- pi_4: -0.0012 (var=0.0120)
- pi_1: -0.0150 (var=0.0031)
- pi_2: 0.0033 (var=0.0040)
- pi_3: -0.0004 (var=0.0002)
- pi_5: -0.0029 (var=0.0043)
- pi_5_2: 0.0038 (var=0.0084)
- pi_6: -0.0358 (var=0.0108)

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
  - iter 1: 0.0246 (var=0.0121) (Δ vs real +0.0254)
  - iter 2 (most recent): 0.0037 (var=0.0110) (Δ vs real +0.0046)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0100 (var=0.0111)
- pi_5_1: 0.2717 (var=0.0167)
- pi_1: -0.0088 (var=0.0044)
- pi_2: -0.0075 (var=0.0048)
- pi_3: 0.0021 (var=0.0002)
- pi_5: -0.0029 (var=0.0062)
- pi_5_2: -0.0125 (var=0.0074)
- pi_6: 0.0096 (var=0.0118)

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
  - iter 1: 0.0092 (var=0.0432) (Δ vs real +0.0142)
  - iter 2 (most recent): 0.0567 (var=0.0253) (Δ vs real +0.0617)
**Other theories' values on this metric (for reference):**
- pi_5_2: 0.6500 (var=0.0457)
- pi_4: 0.0250 (var=0.0283)
- pi_1: -0.0150 (var=0.0273)
- pi_2: 0.0342 (var=0.0199)
- pi_3: -0.0058 (var=0.0006)
- pi_5: 0.5325 (var=0.0892)
- pi_5_1: 0.5525 (var=0.0595)
- pi_6: 0.0367 (var=0.0440)

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
  - iter 1: 0.0206 (var=0.0135) (Δ vs real +0.0383)
  - iter 2 (most recent): 0.0006 (var=0.0127) (Δ vs real +0.0183)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0067 (var=0.0117)
- pi_5_2: 0.2894 (var=0.0132)
- pi_1: -0.0050 (var=0.0090)
- pi_2: 0.0017 (var=0.0129)
- pi_3: -0.0067 (var=0.0049)
- pi_5: 0.2600 (var=0.0188)
- pi_5_1: 0.2211 (var=0.0170)
- pi_6: -0.0206 (var=0.0128)

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
  - iter 1: 0.7408 (var=0.0022) (Δ vs real -0.1017)
  - iter 2 (most recent): 0.7921 (var=0.0019) (Δ vs real -0.0504)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8048 (var=0.0022)
- pi_4: 0.5792 (var=0.0037)
- pi_1: 0.1235 (var=0.0079)
- pi_2: 0.3046 (var=0.0038)
- pi_3: 0.1665 (var=0.0008)
- pi_5: 0.7015 (var=0.0047)
- pi_5_1: 0.6865 (var=0.0051)
- pi_5_2: 0.7231 (var=0.0028)

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
  - iter 1: 0.8273 (var=0.0029) (Δ vs real -0.0140)
  - iter 2 (most recent): 0.8704 (var=0.0014) (Δ vs real +0.0292)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6440 (var=0.0025)
- pi_6: 0.7856 (var=0.0023)
- pi_1: 0.1590 (var=0.0096)
- pi_2: 0.4971 (var=0.0017)
- pi_3: 0.3656 (var=0.0013)
- pi_5: 0.7935 (var=0.0075)
- pi_5_1: 0.7956 (var=0.0031)
- pi_5_2: 0.8431 (var=0.0024)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
People choose between two products through a baseline mixture of a weak forward lexicographic scan, a moderate lowest-validity-first lexicographic scan, and feature-win tallying. On top of that baseline, decision makers continuously accumulate local evidence about cue reliability. The key evidence is near-tied trials with high discrimination density: a graded density term replaces the old all-features-discriminate gate, so five-of-six discriminating trials contribute while sparse three- or four-of-six trials do not. Near-tied trials where the highest-validity cue is tied or absent provide smaller base evidence, while true opposite-direction disagreement between the highest- and lowest-validity cues adds a bonus. This evidence enters a leaky accumulator with asymmetric rise and decay, a minimum qualifying count, and hysteresis. When it passes a lowered threshold, weight smoothly and persistently shifts toward the lowest-validity-first route, which deepens reversal in dense partially-discriminating blocks and increases low-cue consistency in top-cue-tied blocks, while sparse designs decay back to the baseline mixture.

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
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        evidence_seq.append(trial_evidence(past_a[i], past_b[i]))
    evidence_seq.append(trial_evidence(a, b))

    evidence_rise = float(parameters['evidence_rise'])
    evidence_decay = float(parameters['evidence_decay'])
    threshold = float(parameters['conflict_threshold'])
    hysteresis = float(parameters['hysteresis'])
    activation_rise = float(parameters['activation_rise'])
    activation_decay = float(parameters['activation_decay'])
    min_qual = int(parameters['min_qualifying_count'])
    s_max = float(parameters['s_max'])

    X = 0.0
    S = 0.0
    n_seen = 0
    for u in evidence_seq:
        if u >= X:
            X += evidence_rise * (u - X)
        else:
            X -= evidence_decay * (X - u)
        if u > 0.0:
            n_seen += 1
        if n_seen < min_qual:
            target = 0.0
        else:
            threshold_eff = threshold - hysteresis if S > 0.5 else threshold
            target = s_max if X >= threshold_eff else 0.0
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
- hi_lo_bonus: [0.25, 0.45]
- base_evidence: [0.55, 0.65]
- tied_top_evidence: [0.38, 0.52]
- density_floor: {0.80}
- evidence_rise: [0.18, 0.40]
- evidence_decay: [0.02, 0.12]
- conflict_threshold: [0.03, 0.05]
- hysteresis: [0.02, 0.05]
- activation_rise: [0.08, 0.30]
- activation_decay: [0.01, 0.08]
- min_qualifying_count: {2, 3}
- s_max: [0.75, 0.85]
- epsilon: [0.0, 0.02]
- validities: validities

`rationale`: This is a minimal in-family repair of the accumulator-mixture model. The old hard gate required all features to discriminate, which blocked Exp6/Exp7 and Exp13 from reaching the reversed regime. The new trial_evidence replaces that binary gate with a graded discrimination-density term: evidence is scaled by n_discrim/n_features and only trials with density above a fixed 0.80 floor contribute. This lets five-of-six discriminating trials in Exp6/Exp7 and Exp13 accumulate while keeping the three- or four-of-six sparse trials in Exp11/Exp12 sub-threshold. The disagreement requirement is also weakened: near-tied trials with a tied or absent top cue now receive smaller tied-top base evidence, so Exp13 accumulates despite frequent top-cue ties, while true high-low disagreement retains a bonus. Finally, the reversed route is deepened by raising s_max to [0.75, 0.85] and the evidence threshold is lowered to [0.03, 0.05], closing the Exp6/Exp7/Exp13 under-reversal gaps. The asymmetric rise/decay, hysteresis, minimum qualifying count, and response-noise machinery are kept unchanged, so sparse designs still decay back to the baseline mixture and Exp5 remains strongly reversed.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate faithfully implements the arbiter's adaptive cue-reliability family: a leaky evidence accumulator with asymmetric rise/decay, hysteresis, a qualifying threshold, and a smooth mixture between baseline forward/backward/tally routes and a strengthened lowest-validity-first route. Fit is excellent on many experiments: Exp1 0.232 vs real 0.249, Exp2 0.425 vs 0.420, Exp4 -0.270 vs -0.274, Exp5 exactly -0.658 vs -0.658, Exp8 0.013 vs 0.015, Exp14 0.827 vs 0.841; the sparse active-inactive contrasts stay near zero (Exps 9-12 within ~0.04 of real). However, there is a consistent under-reversal in the partially-discriminating dense blocks: Exp6 -0.443 vs -0.671 (candidate +0.23 too weak), Exp7 0.228 vs 0.155 (+0.07 too weak, i.e., not reversed enough), Exp13 0.741 vs 0.843 (-0.10 too weak). The common cause is the evidence gate: it requires ALL features to discriminate (full = n_discrim >= n_features) and hi-lo disagreement. Exp5 and Exp14 are fully complementary, so the gate passes and the fit is near perfect; Exp6 and Exp7 contain mostly partially-tied trials, so almost no evidence accumulates and the model never reaches the deepened reversal; Exp13 has many top-cue ties so the disagreement clause also blocks evidence, yet humans show the strongest low-cue consistency there. Notably, pi_6's values on those same metrics (-0.641, 0.145, 0.805) demonstrate that this family can reach the real values; the candidate's smooth accumulator just needs recalibration, not replacement. The spoils in sparse designs (Exp11/12 near zero) exist precisely because those designs also lack fully discriminating trials, so any fix must be graded rather than binary to avoid firing evidence in Exps 11/12.
Rationale: Keep the adaptive cue-reliability family and the baseline mixture, but recalibrate the evidence gating and activation depth. (1) Replace the hard 'all features discriminate' gate with a graded discrimination-density term (e.g., scale evidence by n_discrim/n_features, with a floor such that Exp11/12's 3-4-of-6 trials stay sub-threshold while Exp6/7's 5-of-6 trials contribute). (2) Weaken the disagreement requirement: give a smaller base evidence on near-tied trials where the top cue is tied or absent, keeping only a bonus for true hi-lo disagreement, so Exp13 accumulates despite frequent hi ties. (3) Slightly deepen saturation (raise s_max toward ~0.85 and/or lower the conflict threshold) to close the remaining Exp6/7/13 gaps, while verifying Exp5 retains its exact -0.658 and Exps 8-12 remain within noise of zero. Keep asymmetric rise/decay, hysteresis, min qualifying count, and epsilon noise as-is; do not switch mechanism families, since the accumulator + mixture loop already captures the sparse-to-dense continuum and only the gating and ceiling need this minor in-family repair.

**Outcome of this advice:** iter 1 candidate loss=0.0583 -> iter 2 candidate loss=0.0407 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is a faithful in-family adaptive cue-reliability implementation and aggregate loss improved from 0.0583 to 0.0407. It is now very close on several experiments: Exp2 0.437 vs 0.420, Exp3 -0.124 vs -0.130, Exp7 0.157 vs 0.155, Exp8 0.025 vs 0.015, Exp10 0.004 vs -0.001. The graded density gate also fixed much of the previous under-reversal in Exp6 (-0.443 to -0.557 vs real -0.671) and Exp13 (0.741 to 0.792 vs real 0.843). However, four residual failures remain. Exp5 regressed from an exact -0.658 to -0.718, i.e., the reversal route now saturates too strongly in fully discriminating designs. Exp6 is still under-reversed by +0.114 and Exp13 remains -0.050 below the real low-cue consistency. Critically, Exp11 is +0.057 while the real active-inactive contrast is near zero, and it moved from +0.009 in iter 1 to +0.057 now. This indicates the relaxed graded evidence gate is firing on the occasional dense 5-of-6 trials embedded in an otherwise sparse design, violating the arbiter's requirement that sparse designs decay back to the baseline mixture.
Rationale: Keep the same adaptive cue-reliability accumulator-mixture family, but recalibrate the evidence gating in-family. The previous graded density repair was accepted and should be retained, but it needs a temporal-density constraint: make activation depend on a recent run of evidence trials (e.g., at least k of the last m trials produce nonzero trial_evidence), rather than a slow all-history leaky count. This should preserve the near-zero Exp11/Exp12 contrasts while still firing in the sustained dense blocks of Exp6/Exp7. Second, lower s_max from [0.75, 0.85] to roughly [0.65, 0.75] because the reversed route is now too deep in fully discriminating designs: Exp5 should move back from -0.718 toward -0.658, and Exp14 from 0.870 toward 0.841; this also reduces the modest over-reversal in Exp1 and Exp4. Third, compensate for the lower ceiling on under-reversed Exp6 by increasing evidence_rise and possibly hi_lo_bonus/base_evidence, so partially discriminating dense blocks reach that lower ceiling sooner while fully discriminating Exp5 simply stays capped. Fourth, raise tied_top_evidence modestly to close Exp13's remaining -0.050 gap without changing the sparse designs. Keep the baseline forward/backward/tally routes, asymmetric rise/decay, hysteresis, and response noise as-is. Do not switch mechanism families.

**Outcome of this advice:** iter 2 candidate loss=0.0407 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "People choose between two products through a baseline mixture of a weak forward lexicographic scan, a moderate lowest-validity-first lexicographic scan, and feature-win tallying. On top of that baseline, decision makers continuously accumulate locally diagnostic evidence. Activation rises only when the recent trial stream contains a sustained run of near-tied, high-discrimination trials, preventing isolated dense trials in sparse blocks from triggering a reversal. The reversed-route weight rises smoothly and is further modulated by discrimination composition: when the recent diagnostic run is dominated by fully discriminating trials, the reversed-route weight settles at a lower ceiling, while partially discriminating dense blocks can reach a higher ceiling. This produces strong low-validity-cue shifts in dense partially-discriminating blocks without over-shifting all-complementary blocks, and leaves sparse blocks governed by the baseline mixture.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim == 3 and stim.shape[0] == 1:\n        stim = stim[0]\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus.')\n\n    a = stim[0]\n    b = stim[1]\n    n_features = int(stim.shape[1])\n\n    validities = parameters.get('validities')\n    if validities is None:\n        validities = list(np.linspace(0.9, 0.5, n_features))\n    validities = np.asarray(validities, dtype=float)\n    if validities.ndim == 0 or validities.shape[0] != n_features:\n        validities = np.linspace(0.9, 0.5, n_features)\n\n    descending = np.argsort(-validities, kind='stable')\n    ascending = np.argsort(validities, kind='stable')\n\n    def lex_probabilities(x, y, order, beta):\n        winner = None\n        for j in order:\n            if x[j] > y[j]:\n                winner = 0\n                break\n            if y[j] > x[j]:\n                winner = 1\n                break\n        if winner is None:\n            return np.full(2, 0.5, dtype=float)\n        z = float(beta) if winner == 0 else -float(beta)\n        p_a = 1.0 / (1.0 + np.exp(-z))\n        return np.array([p_a, 1.0 - p_a], dtype=float)\n\n    beta_forward = float(parameters['beta_forward'])\n    beta_backward = float(parameters['beta_backward'])\n    beta_backward_strong = float(parameters['beta_backward_strong'])\n    beta_tally = float(parameters['beta_tally'])\n\n    p_forward = lex_probabilities(a, b, descending, beta_forward)\n    p_backward = lex_probabilities(a, b, ascending, beta_backward)\n    p_shift = lex_probabilities(a, b, ascending, beta_backward_strong)\n\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    tally_scores = np.array([a_wins, b_wins], dtype=float)\n    z = beta_tally * (tally_scores - np.max(tally_scores))\n    e = np.exp(z)\n    p_tally = e / np.sum(e)\n\n    w_forward = float(parameters['w_forward'])\n    w_backward = float(parameters['w_backward'])\n    w_tally = 1.0 - w_forward - w_backward\n    if w_tally < 0.0:\n        w_tally = 0.0\n    weights = np.array([w_forward, w_backward, w_tally], dtype=float)\n    weights = weights / np.sum(weights)\n    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally\n\n    tie_margin = float(parameters['tie_margin'])\n    hi_lo_bonus = float(parameters['hi_lo_bonus'])\n    base_evidence = float(parameters['base_evidence'])\n    tied_top_evidence = float(parameters['tied_top_evidence'])\n    density_floor = float(parameters['density_floor'])\n    hi_idx = int(descending[0])\n    lo_idx = int(ascending[0])\n\n    def trial_evidence(x, y):\n        x = np.asarray(x, dtype=float)\n        y = np.asarray(y, dtype=float)\n        d = x - y\n        n_discrim = int(np.sum(d != 0.0))\n        density = n_discrim / max(1, n_features)\n        tally = float(np.sum(d))\n        near = 1.0 if abs(tally) <= tie_margin else 0.0\n        if near <= 0.0 or density <= density_floor:\n            return 0.0\n\n        density_gain = (density - density_floor) / (1.0 - density_floor)\n        hi_dir = np.sign(d[hi_idx])\n        lo_dir = np.sign(d[lo_idx])\n        disagree = 1.0 if (hi_dir != 0.0 and lo_dir != 0.0 and hi_dir == -lo_dir) else 0.0\n\n        if hi_dir == 0.0:\n            base = tied_top_evidence\n        else:\n            base = base_evidence + hi_lo_bonus * disagree\n\n        return float(min(1.0, base * density_gain))\n\n    def trial_full(x, y):\n        x = np.asarray(x, dtype=float)\n        y = np.asarray(y, dtype=float)\n        d = x - y\n        return 1.0 if int(np.sum(d != 0.0)) >= n_features else 0.0\n\n    evidence_seq = []\n    full_seq = []\n    past_a = history.get('option_a_ratings', [])\n    past_b = history.get('option_b_ratings', [])\n    n_past = min(len(past_a), len(past_b))\n    for i in range(n_past):\n        evidence_seq.append(trial_evidence(past_a[i], past_b[i]))\n        full_seq.append(trial_full(past_a[i], past_b[i]))\n    evidence_seq.append(trial_evidence(a, b))\n    full_seq.append(trial_full(a, b))\n\n    evidence_rise = float(parameters['evidence_rise'])\n    evidence_decay = float(parameters['evidence_decay'])\n    threshold = float(parameters['conflict_threshold'])\n    hysteresis = float(parameters['hysteresis'])\n    activation_rise = float(parameters['activation_rise'])\n    activation_decay = float(parameters['activation_decay'])\n    min_qual = int(parameters['min_qualifying_count'])\n    s_max = float(parameters['s_max'])\n    run_window = int(parameters['run_window'])\n    run_min_count = int(parameters['run_min_count'])\n    full_start = float(parameters['full_density_start'])\n    full_end = float(parameters['full_density_end'])\n    s_full_floor = float(parameters['s_full_floor'])\n\n    recent_evidence = evidence_seq[-run_window:]\n    recent_full = full_seq[-run_window:]\n    n_recent_evidence = int(np.sum(np.asarray(recent_evidence, dtype=float) > 0.0))\n    run_active = n_recent_evidence >= run_min_count\n\n    if len(recent_full) == 0:\n        rho_full = 0.0\n    else:\n        rho_full = float(np.mean(recent_full))\n\n    if rho_full <= full_start:\n        full_pen = 0.0\n    elif rho_full >= full_end:\n        full_pen = 1.0\n    else:\n        full_pen = (rho_full - full_start) / (full_end - full_start + 1e-9)\n    full_pen = float(np.clip(full_pen, 0.0, 1.0))\n\n    X = 0.0\n    S = 0.0\n    n_seen = 0\n    for u in evidence_seq:\n        if u >= X:\n            X += evidence_rise * (u - X)\n        else:\n            X -= evidence_decay * (X - u)\n        if u > 0.0:\n            n_seen += 1\n\n        if (not run_active) or n_seen < min_qual:\n            target = 0.0\n        else:\n            threshold_eff = threshold - hysteresis if S > 0.5 else threshold\n            ceiling = s_max - full_pen * (s_max - s_full_floor)\n            if ceiling < 0.0:\n                ceiling = 0.0\n            target = ceiling if X >= threshold_eff else 0.0\n\n        if target > S:\n            S += activation_rise * (target - S)\n        else:\n            S -= activation_decay * (S - target)\n\n    s = float(np.clip(S, 0.0, s_max))\n    p = (1.0 - s) * p_base + s * p_shift\n\n    epsilon = float(parameters['epsilon'])\n    if epsilon > 0.0:\n        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)\n\n    p = np.clip(p, 0.0, None)\n    return p / np.sum(p)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=float)\n    if probs.size == 0:\n        return 0\n    probs = probs / probs.sum()\n    return int(np.random.choice(probs.size, p=probs))\n",
  "parameters": {
    "w_forward": "[0.22, 0.32]",
    "w_backward": "[0.42, 0.54]",
    "beta_forward": "[0.08, 0.20]",
    "beta_backward": "[1.30, 1.70]",
    "beta_tally": "[1.80, 2.60]",
    "beta_backward_strong": "[4.00, 6.00]",
    "tie_margin": "[2.0, 3.0]",
    "hi_lo_bonus": "[0.30, 0.50]",
    "base_evidence": "[0.60, 0.75]",
    "tied_top_evidence": "[0.50, 0.65]",
    "density_floor": "{0.80}",
    "evidence_rise": "[0.35, 0.55]",
    "evidence_decay": "[0.02, 0.12]",
    "conflict_threshold": "[0.03, 0.05]",
    "hysteresis": "[0.02, 0.05]",
    "activation_rise": "[0.20, 0.45]",
    "activation_decay": "[0.01, 0.08]",
    "min_qualifying_count": "{2, 3}",
    "s_max": "[0.72, 0.82]",
    "run_window": "{12}",
    "run_min_count": "{4}",
    "full_density_start": "[0.45, 0.60]",
    "full_density_end": "[0.70, 0.85]",
    "s_full_floor": "[0.48, 0.60]",
    "epsilon": "[0.0, 0.02]",
    "validities": "validities"
  },
  "rationale": "This edit keeps the accepted adaptive accumulator-mixture family but repairs the two newest mechanistic failures. First, isolated dense trials in sparse designs can no longer trigger reversal: activation now requires a recent run of at least four evidence-producing trials in the trailing twelve-trial window. That drives Exp11 and Exp12 back toward their near-zero active-inactive contrasts while still allowing sustained dense blocks in Exp6 and Exp7 to fire. Second, instead of globally lowering s_max, the model now applies a smooth full-discrimination-density penalty: when the recent diagnostic run is dominated by fully discriminating trials, the reversed-route ceiling falls to a lower floor, so all-complementary designs such as Exp5 and Exp14 saturate less and stop over-reversing, while partially discriminating dense designs keep the higher ceiling and still reverse strongly. To compensate for the lower saturated weight in those dense partial blocks, beta_backward_strong is raised and evidence/activation accumulation rates are increased, so the lower-validity route is more decisive when it is active. Tied-top evidence is also modestly increased to close the Exp13 low-cue-consistency gap."
}
```

## Usage

```json
{
  "input_tokens": 39682,
  "output_tokens": 23384,
  "reasoning_tokens": 20349,
  "provider": "Baidu"
}
```
