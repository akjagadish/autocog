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
- THEORY 1 = `pi_17`
- THEORY 2 = `pi_19`
- The recommendation below acts on THEORY 2 (= `pi_19`).

Propose a new theory, 'Relative Evidence Accumulation with Conflict Discounting'. In this theory, the decision-maker evaluates options by accumulating evidence, but the weight of each feature is dynamically discounted by the presence of conflicting features in the competing option (cross-option inhibition) rather than being diluted by features within the same option (self-inhibition). This avoids the 'perverse' self-dilution seen in Exp 1 and 2 (ensuring that adding supporting cues does not hurt an option), while successfully capturing the conflict-driven non-linearities and context effects observed in Exp 13, 14, 31, 32, and 33.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_16` (overall score: 0.615)

**Description**
Context-Dependent Lexicographic Accumulator (CDLA) with Symmetric Dispersion Effect: Decision-makers process information lexicographically, but the weight assigned to the first discriminating cue is contextually modulated by the dispersion of remaining cues in a symmetric manner. Both contradiction and consonance from secondary cues adjust the primary cue's weight. Furthermore, both primary and secondary cues can be weighted negatively, capturing baseline avoidance and perverse context effects.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    first_disc_idx = -1
    for i in cue_order:
        if a[i] != b[i]:
            first_disc_idx = i
            break
            
    if first_disc_idx == -1:
        return np.array([0.5, 0.5])
        
    w_first = float(parameters["w_first"])
    w_remaining = float(parameters["w_remaining"])
    dispersion_effect = float(parameters["dispersion_effect"])
    
    remaining_a = 0
    remaining_b = 0
    passed_first = False
    for i in cue_order:
        if passed_first:
            remaining_a += a[i]
            remaining_b += b[i]
        if i == first_disc_idx:
            passed_first = True
            
    first_diff = a[first_disc_idx] - b[first_disc_idx]
    
    if first_diff > 0:
        contradiction = remaining_b - remaining_a
    else:
        contradiction = remaining_a - remaining_b
        
    # The weight of the first cue is modulated symmetrically by the contradiction/consonance
    adj_first = w_first * np.exp(dispersion_effect * contradiction)
    
    ev_a = 0.0
    ev_b = 0.0
    
    if first_diff > 0:
        ev_a += adj_first
    else:
        ev_b += adj_first
        
    ev_a += w_remaining * remaining_a
    ev_b += w_remaining * remaining_b
    
    beta = float(parameters["beta"])
    z = beta * np.array([ev_a, ev_b])
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- w_first: [-10.0, 10.0]
- w_remaining: [-5.0, 5.0]
- dispersion_effect: [-2.0, 2.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6508 (var=0.0505) vs this=0.4779 (var=0.1411)
- Experiment 2: real=0.3011 (var=0.0238) vs this=0.5619 (var=0.1384)
- Experiment 3: real=0.6100 (var=0.0044) vs this=0.5181 (var=0.0904)
- Experiment 4: real=0.6383 (var=0.0300) vs this=0.4894 (var=0.1279)
- Experiment 5: real=0.0825 (var=0.1837) vs this=0.0025 (var=0.0366)
- Experiment 6: real=-0.0167 (var=0.0028) vs this=0.0017 (var=0.0020)
- Experiment 7: real=-0.2050 (var=0.2002) vs this=0.0162 (var=0.0381)
- Experiment 8: real=0.9324 (var=0.1377) vs this=1.5639 (var=0.2574)
- Experiment 9: real=1.6547 (var=0.1361) vs this=0.9474 (var=0.1545)
- Experiment 10: real=0.0885 (var=0.0487) vs this=0.0843 (var=0.2106)
- Experiment 11: real=-0.4292 (var=0.0555) vs this=0.0592 (var=0.0859)
- Experiment 12: real=-0.6711 (var=0.0499) vs this=0.0033 (var=0.1677)
- Experiment 13: real=0.2644 (var=0.0112) vs this=0.4183 (var=0.1303)
- Experiment 14: real=0.1350 (var=0.0065) vs this=0.4608 (var=0.1408)
- Experiment 15: real=0.8031 (var=0.0244) vs this=0.5038 (var=0.1611)
- Experiment 16: real=0.2611 (var=0.0294) vs this=0.3584 (var=0.0092)
- Experiment 17: real=0.2825 (var=0.0159) vs this=0.4537 (var=0.1374)
- Experiment 18: real=0.3458 (var=0.0444) vs this=0.4288 (var=0.1416)
- Experiment 19: real=0.1758 (var=0.0110) vs this=0.4921 (var=0.1261)
- Experiment 20: real=0.2025 (var=0.0829) vs this=-0.0075 (var=0.0196)
- Experiment 21: real=0.4050 (var=0.0313) vs this=0.5069 (var=0.0043)
- Experiment 22: real=0.5684 (var=0.0814) vs this=0.5274 (var=0.1550)
- Experiment 23: real=0.3133 (var=0.0161) vs this=0.3871 (var=0.0078)
- Experiment 24: real=0.3702 (var=0.0075) vs this=0.3656 (var=0.0105)
- Experiment 25: real=-0.0010 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 26: real=0.0154 (var=0.0544) vs this=0.0015 (var=0.0408)
- Experiment 27: real=0.8320 (var=0.0112) vs this=0.4980 (var=0.1427)
- Experiment 28: real=0.1467 (var=0.0053) vs this=0.4644 (var=0.0984)
- Experiment 29: real=-0.6071 (var=0.0412) vs this=0.0298 (var=0.2381)
- Experiment 30: real=-0.7262 (var=0.0611) vs this=0.0246 (var=0.3400)
- Experiment 31: real=0.5022 (var=0.1976) vs this=0.1956 (var=0.2897)
- Experiment 32: real=-0.0200 (var=0.0414) vs this=-0.0042 (var=0.0113)
- Experiment 33: real=0.1067 (var=0.2133) vs this=0.1267 (var=0.1296)
- Experiment 34: real=0.1022 (var=0.0050) vs this=0.0883 (var=0.0025)
- Experiment 35: real=0.0640 (var=0.0143) vs this=0.0220 (var=0.2577)
- Experiment 36: real=-0.0320 (var=0.0398) vs this=0.0080 (var=0.6215)


---

### `pi_18` (overall score: 0.453)

**Description**
Feature-Averaging with Cue-wise Cancellation and Soft Normalization

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities to reflect subjective evidence weight, keeping the 0.5 baseline
    w = np.maximum(0.0, val - 0.5) ** gamma
    
    # Cue-wise cancellation: only consider unique features
    diff = a - b
    unique_a = (diff > 0)
    unique_b = (diff < 0)
    
    # Feature-averaging: soft normalization interpolating between sum (alpha=0) and mean (alpha=1)
    n_a = np.sum(unique_a)
    sum_w_a = np.sum(w[unique_a])
    ev_a = sum_w_a / (1.0 + alpha * (n_a - 1.0)) if n_a > 0 else 0.0
    
    n_b = np.sum(unique_b)
    sum_w_b = np.sum(w[unique_b])
    ev_b = sum_w_b / (1.0 + alpha * (n_b - 1.0)) if n_b > 0 else 0.0
    
    # Convert evidence to probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


`parameters`:
- gamma: [0.1, 10.0]
- alpha: [0.0, 2.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6508 (var=0.0505) vs this=0.5608 (var=0.0162)
- Experiment 2: real=0.3011 (var=0.0238) vs this=0.4344 (var=0.0119)
- Experiment 3: real=0.6100 (var=0.0044) vs this=0.5400 (var=0.0053)
- Experiment 4: real=0.6383 (var=0.0300) vs this=0.5577 (var=0.0117)
- Experiment 5: real=0.0825 (var=0.1837) vs this=0.0413 (var=0.0653)
- Experiment 6: real=-0.0167 (var=0.0028) vs this=0.0294 (var=0.0087)
- Experiment 7: real=-0.2050 (var=0.2002) vs this=0.0375 (var=0.0747)
- Experiment 8: real=0.9324 (var=0.1377) vs this=1.5602 (var=0.0568)
- Experiment 9: real=1.6547 (var=0.1361) vs this=0.9884 (var=0.0315)
- Experiment 10: real=0.0885 (var=0.0487) vs this=0.3135 (var=0.2388)
- Experiment 11: real=-0.4292 (var=0.0555) vs this=0.0127 (var=0.0236)
- Experiment 12: real=-0.6711 (var=0.0499) vs this=-0.0122 (var=0.0332)
- Experiment 13: real=0.2644 (var=0.0112) vs this=0.5439 (var=0.0093)
- Experiment 14: real=0.1350 (var=0.0065) vs this=0.5704 (var=0.0178)
- Experiment 15: real=0.8031 (var=0.0244) vs this=0.5500 (var=0.0244)
- Experiment 16: real=0.2611 (var=0.0294) vs this=0.0779 (var=0.0063)
- Experiment 17: real=0.2825 (var=0.0159) vs this=0.5429 (var=0.0066)
- Experiment 18: real=0.3458 (var=0.0444) vs this=0.5437 (var=0.0077)
- Experiment 19: real=0.1758 (var=0.0110) vs this=0.5726 (var=0.0087)
- Experiment 20: real=0.2025 (var=0.0829) vs this=0.0162 (var=0.0389)
- Experiment 21: real=0.4050 (var=0.0313) vs this=0.4587 (var=0.0145)
- Experiment 22: real=0.5684 (var=0.0814) vs this=0.4532 (var=0.0175)
- Experiment 23: real=0.3133 (var=0.0161) vs this=0.0546 (var=0.0012)
- Experiment 24: real=0.3702 (var=0.0075) vs this=0.1235 (var=0.0051)
- Experiment 25: real=-0.0010 (var=0.0000) vs this=0.0001 (var=0.0001)
- Experiment 26: real=0.0154 (var=0.0544) vs this=0.1477 (var=0.1318)
- Experiment 27: real=0.8320 (var=0.0112) vs this=0.6070 (var=0.0268)
- Experiment 28: real=0.1467 (var=0.0053) vs this=0.5896 (var=0.0189)
- Experiment 29: real=-0.6071 (var=0.0412) vs this=0.0394 (var=0.0236)
- Experiment 30: real=-0.7262 (var=0.0611) vs this=0.0600 (var=0.0484)
- Experiment 31: real=0.5022 (var=0.1976) vs this=0.0756 (var=0.1523)
- Experiment 32: real=-0.0200 (var=0.0414) vs this=0.0133 (var=0.0113)
- Experiment 33: real=0.1067 (var=0.2133) vs this=0.1444 (var=0.1538)
- Experiment 34: real=0.1022 (var=0.0050) vs this=0.1483 (var=0.0064)
- Experiment 35: real=0.0640 (var=0.0143) vs this=-0.0460 (var=0.0593)
- Experiment 36: real=-0.0320 (var=0.0398) vs this=-0.1040 (var=0.1324)


---

### `pi_8` (overall score: 0.441)

**Description**
Conflict-Driven Strategy Selection: Decision-makers adaptively select their decision strategy based on the dissimilarity of the options' total cue counts. When options are highly dissimilar in the number of positive cues (indicating high conflict or complexity), individuals abandon compensatory processing and fall back entirely on a simple non-compensatory heuristic (Take The Best). Conversely, when the total cue counts are similar, individuals attempt to integrate all available information using a compensatory strategy (Tallying). This is modeled as a probabilistic mixture of TTB and Tallying, where the probability of using TTB scales directly with the absolute difference in total cue counts.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    # TTB Probabilities
    z_ttb = beta_ttb * scores_ttb
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta_tally * scores_tally
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally_dist = e_tally / e_tally.sum()
    
    # Conflict-Driven Weight
    gamma = float(parameters["gamma"])
    
    # Dissimilarity in total cue counts
    diff_cues = abs(np.sum(a) - np.sum(b))
    
    # Probability of using TTB increases linearly with diff_cues
    w_ttb = min(1.0, gamma * diff_cues / max(1.0, float(n_features)))
    
    epsilon = float(parameters["epsilon"])
    
    p_core = w_ttb * p_ttb_dist + (1.0 - w_ttb) * p_tally_dist
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta_ttb: [0.1, 10.0]
- beta_tally: [0.1, 10.0]
- gamma: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6508 (var=0.0505) vs this=0.7383 (var=0.0261)
- Experiment 2: real=0.3011 (var=0.0238) vs this=0.2467 (var=0.0310)
- Experiment 3: real=0.6100 (var=0.0044) vs this=0.6758 (var=0.0197)
- Experiment 4: real=0.6383 (var=0.0300) vs this=0.7631 (var=0.0297)
- Experiment 5: real=0.0825 (var=0.1837) vs this=0.0313 (var=0.0402)
- Experiment 6: real=-0.0167 (var=0.0028) vs this=-0.0100 (var=0.0058)
- Experiment 7: real=-0.2050 (var=0.2002) vs this=-0.0062 (var=0.0454)
- Experiment 8: real=0.9324 (var=0.1377) vs this=1.3323 (var=0.0562)
- Experiment 9: real=1.6547 (var=0.1361) vs this=0.9579 (var=0.0176)
- Experiment 10: real=0.0885 (var=0.0487) vs this=-0.0265 (var=0.0856)
- Experiment 11: real=-0.4292 (var=0.0555) vs this=0.1088 (var=0.0248)
- Experiment 12: real=-0.6711 (var=0.0499) vs this=0.0000 (var=0.0405)
- Experiment 13: real=0.2644 (var=0.0112) vs this=0.4861 (var=0.0031)
- Experiment 14: real=0.1350 (var=0.0065) vs this=0.4925 (var=0.0046)
- Experiment 15: real=0.8031 (var=0.0244) vs this=0.7508 (var=0.0329)
- Experiment 16: real=0.2611 (var=0.0294) vs this=0.0658 (var=0.0028)
- Experiment 17: real=0.2825 (var=0.0159) vs this=0.4938 (var=0.0038)
- Experiment 18: real=0.3458 (var=0.0444) vs this=0.4883 (var=0.0056)
- Experiment 19: real=0.1758 (var=0.0110) vs this=0.4742 (var=0.0058)
- Experiment 20: real=0.2025 (var=0.0829) vs this=0.0612 (var=0.0277)
- Experiment 21: real=0.4050 (var=0.0313) vs this=0.5000 (var=0.0067)
- Experiment 22: real=0.5684 (var=0.0814) vs this=0.5084 (var=0.0076)
- Experiment 23: real=0.3133 (var=0.0161) vs this=0.0563 (var=0.0017)
- Experiment 24: real=0.3702 (var=0.0075) vs this=0.0958 (var=0.0022)
- Experiment 25: real=-0.0010 (var=0.0000) vs this=0.0019 (var=0.0001)
- Experiment 26: real=0.0154 (var=0.0544) vs this=0.0138 (var=0.0854)
- Experiment 27: real=0.8320 (var=0.0112) vs this=0.7690 (var=0.0442)
- Experiment 28: real=0.1467 (var=0.0053) vs this=0.8060 (var=0.0123)
- Experiment 29: real=-0.6071 (var=0.0412) vs this=0.1056 (var=0.0333)
- Experiment 30: real=-0.7262 (var=0.0611) vs this=0.0477 (var=0.0203)
- Experiment 31: real=0.5022 (var=0.1976) vs this=-0.1356 (var=0.0885)
- Experiment 32: real=-0.0200 (var=0.0414) vs this=0.0000 (var=0.0215)
- Experiment 33: real=0.1067 (var=0.2133) vs this=0.1133 (var=0.1893)
- Experiment 34: real=0.1022 (var=0.0050) vs this=0.1083 (var=0.0041)
- Experiment 35: real=0.0640 (var=0.0143) vs this=0.3260 (var=0.0407)
- Experiment 36: real=-0.0320 (var=0.0398) vs this=0.3600 (var=0.1004)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3594 -> ACCEPTED
- iter 2: loss=0.3972 -> REJECTED
- iter 3: loss=0.4083 -> REJECTED
- iter 4: loss=0.4157 -> REJECTED
- iter 5: loss=0.4090 -> REJECTED
- iter 6: loss=0.4070 -> REJECTED
- iter 7: loss=0.3886 -> REJECTED
- iter 8: loss=0.3825 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.3594 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_winner = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(a)) if b[i] > a[i])
        
        tally_winner = 0 if a_wins > b_wins else (1 if b_wins > a_wins else -1)
        
        # Only consider trials where TTB and Tallying make opposing deterministic predictions
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner:
            ttb_matches.append(1 if row['response'] == ttb_winner else 0)
            
    if not ttb_matches:
        return 0.5
    return float(np.mean(ttb_matches))
```

**Observed (real) value:** 0.6508 (var=0.0505)
**Previous candidate values (this loop):**
  - iter 1: 0.4971 (var=0.0081) (Δ vs real -0.1538)
  - iter 2: 0.4508 (var=0.0189) (Δ vs real -0.2000)
  - iter 3: 0.4296 (var=0.0180) (Δ vs real -0.2213)
  - iter 4: 0.4642 (var=0.0120) (Δ vs real -0.1867)
  - iter 5: 0.4363 (var=0.0207) (Δ vs real -0.2146)
  - iter 6: 0.5121 (var=0.0149) (Δ vs real -0.1388)
  - iter 7: 0.4608 (var=0.0188) (Δ vs real -0.1900)
  - iter 8 (most recent): 0.4696 (var=0.0098) (Δ vs real -0.1813)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8471 (var=0.0094)
- pi_2: 0.1842 (var=0.0150)
- pi_3: 0.6208 (var=0.0346)
- pi_4: 0.6571 (var=0.0250)
- pi_5: 0.6525 (var=0.0167)
- pi_6: 0.5054 (var=0.0097)
- pi_7: 0.7004 (var=0.0332)
- pi_8: 0.7383 (var=0.0261)
- pi_9: 0.4642 (var=0.0870)
- pi_10: 0.6300 (var=0.0278)
- pi_11: 0.3767 (var=0.0225)
- pi_12: 0.7571 (var=0.0819)
- pi_13: 0.4533 (var=0.0873)
- pi_14: 0.3396 (var=0.0945)
- pi_15: 0.5775 (var=0.0644)
- pi_16: 0.4779 (var=0.1411)
- pi_17: 0.7825 (var=0.0205)
- pi_18: 0.5608 (var=0.0162)
- pi_19: 0.4725 (var=0.0286)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_mask = a_wins != b_wins
    if not np.any(valid_mask):
        return 0.5
        
    tally_preds = (b_wins > a_wins).astype(int)
    responses = data['response'].values
    
    matches = (tally_preds[valid_mask] == responses[valid_mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3011 (var=0.0238)
**Previous candidate values (this loop):**
  - iter 1: 0.5178 (var=0.0063) (Δ vs real +0.2167)
  - iter 2: 0.5525 (var=0.0125) (Δ vs real +0.2514)
  - iter 3: 0.5797 (var=0.0134) (Δ vs real +0.2786)
  - iter 4: 0.5825 (var=0.0147) (Δ vs real +0.2814)
  - iter 5: 0.5508 (var=0.0122) (Δ vs real +0.2497)
  - iter 6: 0.4811 (var=0.0117) (Δ vs real +0.1800)
  - iter 7: 0.5342 (var=0.0073) (Δ vs real +0.2331)
  - iter 8 (most recent): 0.5133 (var=0.0036) (Δ vs real +0.2122)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8328 (var=0.0095)
- pi_1: 0.1311 (var=0.0070)
- pi_3: 0.3628 (var=0.0340)
- pi_4: 0.3628 (var=0.0203)
- pi_5: 0.3386 (var=0.0210)
- pi_6: 0.4697 (var=0.0089)
- pi_7: 0.2425 (var=0.0259)
- pi_8: 0.2467 (var=0.0310)
- pi_9: 0.5097 (var=0.0656)
- pi_10: 0.3900 (var=0.0317)
- pi_11: 0.6008 (var=0.0199)
- pi_12: 0.2317 (var=0.0698)
- pi_13: 0.6303 (var=0.0479)
- pi_14: 0.5461 (var=0.1141)
- pi_15: 0.4664 (var=0.0528)
- pi_16: 0.5619 (var=0.1384)
- pi_17: 0.3364 (var=0.0082)
- pi_18: 0.4344 (var=0.0119)
- pi_19: 0.4992 (var=0.0140)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        pred = None
        # The validities are [0.95, 0.93, 0.91, 0.89, 0.5], so the cue order is simply 0 to 4.
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
                
        if pred is not None:
            agreements.append(1 if resp == pred else 0)
            
    if not agreements:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.6100 (var=0.0044)
**Previous candidate values (this loop):**
  - iter 1: 0.4685 (var=0.0090) (Δ vs real -0.1415)
  - iter 2: 0.4417 (var=0.0078) (Δ vs real -0.1683)
  - iter 3: 0.4602 (var=0.0137) (Δ vs real -0.1498)
  - iter 4: 0.4244 (var=0.0087) (Δ vs real -0.1856)
  - iter 5: 0.4069 (var=0.0179) (Δ vs real -0.2031)
  - iter 6: 0.4288 (var=0.0118) (Δ vs real -0.1812)
  - iter 7: 0.4977 (var=0.0097) (Δ vs real -0.1123)
  - iter 8 (most recent): 0.4852 (var=0.0023) (Δ vs real -0.1248)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8738 (var=0.0091)
- pi_3: 0.3508 (var=0.0099)
- pi_2: 0.3267 (var=0.0043)
- pi_4: 0.6810 (var=0.0113)
- pi_5: 0.4873 (var=0.0031)
- pi_6: 0.5006 (var=0.0039)
- pi_7: 0.7798 (var=0.0289)
- pi_8: 0.6758 (var=0.0197)
- pi_9: 0.5813 (var=0.0331)
- pi_10: 0.5865 (var=0.0337)
- pi_11: 0.3890 (var=0.0102)
- pi_12: 0.8083 (var=0.0358)
- pi_13: 0.3767 (var=0.0065)
- pi_14: 0.4292 (var=0.0756)
- pi_15: 0.5729 (var=0.0506)
- pi_16: 0.5181 (var=0.0904)
- pi_17: 0.6192 (var=0.0061)
- pi_18: 0.5400 (var=0.0053)
- pi_19: 0.4054 (var=0.0084)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    # TTB consults cues in order of validity (which corresponds to the feature index 0 to 4).
    # We can find the TTB choice by weighting the differences such that earlier features strictly dominate.
    diff = A - B
    weights = 10 ** np.arange(A.shape[1])[::-1]
    ttb_score = diff.dot(weights)
    
    # If ttb_score > 0, A is favored on the first discriminating cue (predict 0).
    # If ttb_score < 0, B is favored (predict 1).
    ttb_pred = (ttb_score < 0).astype(int)
    
    return float(np.mean(data['response'].values == ttb_pred))
```

**Observed (real) value:** 0.6383 (var=0.0300)
**Previous candidate values (this loop):**
  - iter 1: 0.4685 (var=0.0053) (Δ vs real -0.1698)
  - iter 2: 0.4175 (var=0.0170) (Δ vs real -0.2208)
  - iter 3: 0.4475 (var=0.0085) (Δ vs real -0.1908)
  - iter 4: 0.4006 (var=0.0169) (Δ vs real -0.2377)
  - iter 5: 0.4290 (var=0.0137) (Δ vs real -0.2094)
  - iter 6: 0.5138 (var=0.0079) (Δ vs real -0.1246)
  - iter 7: 0.4710 (var=0.0085) (Δ vs real -0.1673)
  - iter 8 (most recent): 0.4902 (var=0.0062) (Δ vs real -0.1481)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6185 (var=0.0157)
- pi_1: 0.8521 (var=0.0087)
- pi_2: 0.1623 (var=0.0094)
- pi_4: 0.7048 (var=0.0188)
- pi_5: 0.6348 (var=0.0233)
- pi_6: 0.5340 (var=0.0043)
- pi_7: 0.6963 (var=0.0213)
- pi_8: 0.7631 (var=0.0297)
- pi_9: 0.4179 (var=0.0528)
- pi_10: 0.5962 (var=0.0228)
- pi_11: 0.4073 (var=0.0196)
- pi_12: 0.7473 (var=0.0796)
- pi_13: 0.3390 (var=0.0671)
- pi_14: 0.4158 (var=0.0970)
- pi_15: 0.5404 (var=0.0509)
- pi_16: 0.4894 (var=0.1279)
- pi_17: 0.6208 (var=0.0059)
- pi_18: 0.5577 (var=0.0117)
- pi_19: 0.4940 (var=0.0135)

### Experiment 5
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (data['A_tuple'] == (1,0,1,0,0)) & (data['B_tuple'] == (0,1,0,1,1))
    t3_mask = (data['A_tuple'] == (1,0,0,0,1)) & (data['B_tuple'] == (0,1,1,1,0))
    t4_mask = (data['A_tuple'] == (1,1,0,0,0)) & (data['B_tuple'] == (1,0,1,1,0))
    t5_mask = (data['A_tuple'] == (1,1,0,0,0)) & (data['B_tuple'] == (1,0,0,1,1))
    
    p_A_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_A_t3 = 1.0 - data[t3_mask]['response'].mean()
    p_A_t4 = 1.0 - data[t4_mask]['response'].mean()
    p_A_t5 = 1.0 - data[t5_mask]['response'].mean()
    
    val = (p_A_t1 - p_A_t3) + (p_A_t5 - p_A_t4)
    
    if pd.isna(val):
        return 0.0
    return float(val)
```

**Observed (real) value:** 0.0825 (var=0.1837)
**Previous candidate values (this loop):**
  - iter 1: 0.0250 (var=0.0545) (Δ vs real -0.0575)
  - iter 2: 0.0687 (var=0.0444) (Δ vs real -0.0138)
  - iter 3: 0.0875 (var=0.0720) (Δ vs real +0.0050)
  - iter 4: 0.0925 (var=0.0710) (Δ vs real +0.0100)
  - iter 5: 0.0150 (var=0.0823) (Δ vs real -0.0675)
  - iter 6: 0.1625 (var=0.0998) (Δ vs real +0.0800)
  - iter 7: 0.0138 (var=0.0630) (Δ vs real -0.0688)
  - iter 8 (most recent): 0.0087 (var=0.0891) (Δ vs real -0.0738)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0050 (var=0.0408)
- pi_3: 0.2650 (var=0.1685)
- pi_1: 0.0012 (var=0.0293)
- pi_2: -0.0100 (var=0.0237)
- pi_5: 0.1613 (var=0.0413)
- pi_6: 0.0163 (var=0.0737)
- pi_7: 0.0187 (var=0.0499)
- pi_8: 0.0313 (var=0.0402)
- pi_9: -0.2188 (var=0.2266)
- pi_10: 0.1050 (var=0.0954)
- pi_11: 0.0825 (var=0.0662)
- pi_12: -0.1137 (var=0.1323)
- pi_13: -0.3000 (var=0.3262)
- pi_14: -0.0113 (var=0.0484)
- pi_15: -0.0050 (var=0.0500)
- pi_16: 0.0025 (var=0.0366)
- pi_17: -0.3162 (var=0.0979)
- pi_18: 0.0413 (var=0.0653)
- pi_19: 0.1913 (var=0.0838)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'A_str': a_str,
        'response': data['response']
    })
    
    # Strategy Mixture strictly predicts identical probabilities for Trials 1 & 7, and Trials 2 & 8.
    # T1 & T7: TTB predicts Option A, Tallying predicts Option A.
    # T2 & T8: TTB predicts Option B, Tallying predicts Option B.
    # WADD with non-linear scaling strongly differentiates these pairs based on specific cue validities.
    pairs = [
        ('1000111', '1010101'), # T1 vs T7
        ('0101010', '0111000')  # T8 vs T2
    ]
    
    scores = []
    for subj, grp in df.groupby('subject_id'):
        subj_score = 0
        for s_a, s_b in pairs:
            ra = grp[grp['A_str'] == s_a]['response'].values
            rb = grp[grp['A_str'] == s_b]['response'].values
            if len(ra) >= 2 and len(rb) >= 2:
                # Split-half cross-product provides an unbiased estimator of the squared difference
                # in true choice probabilities. Under Strategy Mixture, expected value is exactly 0.
                # Under WADD, the expected value is strictly positive.
                ra_even, ra_odd = ra[::2].mean(), ra[1::2].mean()
                rb_even, rb_odd = rb[::2].mean(), rb[1::2].mean()
                subj_score += (ra_even - rb_even) * (ra_odd - rb_odd)
        scores.append(subj_score)
        
    return float(np.mean(scores))
```

**Observed (real) value:** -0.0167 (var=0.0028)
**Previous candidate values (this loop):**
  - iter 1: 0.0217 (var=0.0081) (Δ vs real +0.0383)
  - iter 2: 0.0556 (var=0.0290) (Δ vs real +0.0722)
  - iter 3: 0.0278 (var=0.0297) (Δ vs real +0.0444)
  - iter 4: 0.0622 (var=0.0277) (Δ vs real +0.0789)
  - iter 5: 0.0378 (var=0.0342) (Δ vs real +0.0544)
  - iter 6: 0.1283 (var=0.0422) (Δ vs real +0.1450)
  - iter 7: -0.0189 (var=0.0167) (Δ vs real -0.0022)
  - iter 8 (most recent): 0.0211 (var=0.0145) (Δ vs real +0.0378)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9578 (var=0.3256)
- pi_4: 0.0056 (var=0.0017)
- pi_1: -0.0100 (var=0.0035)
- pi_2: 0.0028 (var=0.0039)
- pi_5: 0.0578 (var=0.0224)
- pi_6: 0.0361 (var=0.0279)
- pi_7: 0.0539 (var=0.0772)
- pi_8: -0.0100 (var=0.0058)
- pi_9: 0.0172 (var=0.0087)
- pi_10: 0.1400 (var=0.1147)
- pi_11: 0.0306 (var=0.0254)
- pi_12: 0.0089 (var=0.0012)
- pi_13: 0.3944 (var=0.2993)
- pi_14: -0.0078 (var=0.0028)
- pi_15: -0.0094 (var=0.0009)
- pi_16: 0.0017 (var=0.0020)
- pi_17: 0.0783 (var=0.0222)
- pi_18: 0.0294 (var=0.0087)
- pi_19: 0.2039 (var=0.1917)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_trial_type(a):
        a_tuple = tuple(a)
        if a_tuple == (1, 1, 0, 0, 1): return 1
        if a_tuple == (1, 0, 0, 1, 1): return 3
        if a_tuple == (1, 0, 0, 0, 1): return 4
        if a_tuple == (1, 0, 1, 0, 0): return 6
        return 0
        
    trial_types = data['option_a_ratings'].apply(get_trial_type)
    
    p_A = {}
    for t in [1, 3, 4, 6]:
        mask = trial_types == t
        if mask.sum() > 0:
            p_A[t] = np.mean(data.loc[mask, 'response'] == 0)
        else:
            p_A[t] = 0.5
            
    return float((p_A[1] - p_A[3]) + (p_A[6] - p_A[4]))
```

**Observed (real) value:** -0.2050 (var=0.2002)
**Previous candidate values (this loop):**
  - iter 1: -0.0463 (var=0.0733) (Δ vs real +0.1587)
  - iter 2: 0.0188 (var=0.0486) (Δ vs real +0.2238)
  - iter 3: 0.0250 (var=0.0836) (Δ vs real +0.2300)
  - iter 4: 0.0513 (var=0.0850) (Δ vs real +0.2562)
  - iter 5: 0.0600 (var=0.0739) (Δ vs real +0.2650)
  - iter 6: 0.1713 (var=0.0790) (Δ vs real +0.3762)
  - iter 7: -0.0262 (var=0.0613) (Δ vs real +0.1787)
  - iter 8 (most recent): -0.0088 (var=0.0403) (Δ vs real +0.1962)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0162 (var=0.0422)
- pi_5: 0.1437 (var=0.0461)
- pi_1: -0.0125 (var=0.0342)
- pi_2: 0.0000 (var=0.0255)
- pi_3: 0.2900 (var=0.1053)
- pi_6: 0.0575 (var=0.0679)
- pi_7: 0.0563 (var=0.0375)
- pi_8: -0.0062 (var=0.0454)
- pi_9: -0.1388 (var=0.1600)
- pi_10: 0.1387 (var=0.0677)
- pi_11: 0.0763 (var=0.0632)
- pi_12: -0.0225 (var=0.0382)
- pi_13: -0.2688 (var=0.1822)
- pi_14: -0.0200 (var=0.0346)
- pi_15: -0.0075 (var=0.0342)
- pi_16: 0.0162 (var=0.0381)
- pi_17: -0.4150 (var=0.1487)
- pi_18: 0.0375 (var=0.0747)
- pi_19: 0.2613 (var=0.1406)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    # Identify trials by their sum of cues (Tallying score proxy)
    sumA = A.sum(axis=1)
    sumB = B.sum(axis=1)
    
    # 1. Trials where one option has strictly more cues (Trials 4, 5, 8)
    mask_more_B = (sumB > sumA)
    mask_more_A = (sumA > sumB)
    
    more_cues_chosen = 0
    more_cues_total = 0
    if np.any(mask_more_B):
        more_cues_chosen += np.sum(resp[mask_more_B] == 1)
        more_cues_total += np.sum(mask_more_B)
    if np.any(mask_more_A):
        more_cues_chosen += np.sum(resp[mask_more_A] == 0)
        more_cues_total += np.sum(mask_more_A)
        
    p_more_cues = float(more_cues_chosen) / more_cues_total if more_cues_total > 0 else 0.5
    
    # 2. Trials where options have an equal number of cues (Trials 1, 2, 3, 6, 7)
    mask_equal = (sumA == sumB)
    
    ttb_winner_chosen = 0
    ttb_total = 0
    if np.any(mask_equal):
        # Cue 0 is the highest validity cue. In equal cue trials, 
        # the option with Cue 0 is always the TTB winner.
        mask_ttb_A = mask_equal & (A[:, 0] == 1)
        ttb_winner_chosen += np.sum(resp[mask_ttb_A] == 0)
        ttb_total += np.sum(mask_ttb_A)
        
        mask_ttb_B = mask_equal & (B[:, 0] == 1)
        ttb_winner_chosen += np.sum(resp[mask_ttb_B] == 1)
        ttb_total += np.sum(mask_ttb_B)
        
    p_ttb_winner = float(ttb_winner_chosen) / ttb_total if ttb_total > 0 else 0.5
    
    # The metric is a linear combination designed to cancel out the p_ttb parameter in the Mixture model
    return float(p_more_cues + 2.0 * p_ttb_winner)

```

**Observed (real) value:** 0.9324 (var=0.1377)
**Previous candidate values (this loop):**
  - iter 1: 1.5470 (var=0.0385) (Δ vs real +0.6146)
  - iter 2: 1.6078 (var=0.0440) (Δ vs real +0.6753)
  - iter 3: 1.6537 (var=0.1119) (Δ vs real +0.7212)
  - iter 4: 1.6479 (var=0.0417) (Δ vs real +0.7154)
  - iter 5: 1.5739 (var=0.0515) (Δ vs real +0.6414)
  - iter 6: 1.6121 (var=0.0378) (Δ vs real +0.6797)
  - iter 7: 1.6010 (var=0.0653) (Δ vs real +0.6686)
  - iter 8 (most recent): 1.5754 (var=0.0335) (Δ vs real +0.6430)
**Other theories' values on this metric (for reference):**
- pi_5: 1.7010 (var=0.0202)
- pi_4: 1.8784 (var=0.0226)
- pi_1: 1.8606 (var=0.0150)
- pi_2: 1.9030 (var=0.0303)
- pi_3: 1.9046 (var=0.0565)
- pi_6: 1.5333 (var=0.0268)
- pi_7: 1.7891 (var=0.0225)
- pi_8: 1.3323 (var=0.0562)
- pi_9: 1.6521 (var=0.0553)
- pi_10: 1.6938 (var=0.0342)
- pi_11: 1.7516 (var=0.1198)
- pi_12: 1.3804 (var=0.0691)
- pi_13: 1.4948 (var=0.1568)
- pi_14: 1.5803 (var=0.1489)
- pi_15: 1.8798 (var=0.0296)
- pi_16: 1.5639 (var=0.2574)
- pi_17: 1.5772 (var=0.0251)
- pi_18: 1.5602 (var=0.0568)
- pi_19: 1.7018 (var=0.0824)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_ratings = data['option_a_ratings'].apply(tuple)
    
    # Trial 3: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t3_mask = a_ratings == (1, 1, 0, 0, 0)
    # Trial 4: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 1]
    t4_mask = a_ratings == (0, 1, 1, 0, 0)
    
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t3) or pd.isna(p_a_t4):
        return 0.0
        
    return float(p_a_t3 + p_a_t4)
```

**Observed (real) value:** 1.6547 (var=0.1361)
**Previous candidate values (this loop):**
  - iter 1: 0.9842 (var=0.0229) (Δ vs real -0.6705)
  - iter 2: 0.9716 (var=0.0646) (Δ vs real -0.6832)
  - iter 3: 0.9453 (var=0.0331) (Δ vs real -0.7095)
  - iter 4: 0.9442 (var=0.0574) (Δ vs real -0.7105)
  - iter 5: 0.9032 (var=0.0707) (Δ vs real -0.7516)
  - iter 6: 1.1158 (var=0.0220) (Δ vs real -0.5389)
  - iter 7: 0.9116 (var=0.0453) (Δ vs real -0.7432)
  - iter 8 (most recent): 0.9926 (var=0.0300) (Δ vs real -0.6621)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7937 (var=0.0431)
- pi_6: 1.0674 (var=0.0256)
- pi_1: 0.9800 (var=0.0117)
- pi_2: 0.2621 (var=0.0461)
- pi_3: 1.0484 (var=0.0172)
- pi_5: 1.1011 (var=0.0213)
- pi_7: 0.9853 (var=0.0247)
- pi_8: 0.9579 (var=0.0176)
- pi_9: 0.7442 (var=0.1582)
- pi_10: 1.0505 (var=0.0208)
- pi_11: 0.8495 (var=0.1024)
- pi_12: 0.8695 (var=0.0941)
- pi_13: 0.7000 (var=0.1668)
- pi_14: 0.6095 (var=0.0947)
- pi_15: 0.5842 (var=0.0802)
- pi_16: 0.9474 (var=0.1545)
- pi_17: 0.8947 (var=0.0400)
- pi_18: 0.9884 (var=0.0315)
- pi_19: 1.0316 (var=0.0790)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    A_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    B_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    chose_A = 1.0 - data['response']
    
    m12 = ((A_str == '100100') & (B_str == '011000')) | ((A_str == '100110') & (B_str == '011001'))
    m34 = ((A_str == '011000') & (B_str == '100000')) | ((A_str == '011010') & (B_str == '100001'))
    m56 = ((A_str == '100000') & (B_str == '011100')) | ((A_str == '100010') & (B_str == '011101'))
    m78 = ((A_str == '001100') & (B_str == '100000')) | ((A_str == '001110') & (B_str == '100001'))
    
    def get_lo(mask):
        n = mask.sum()
        if n == 0:
            return 0.0
        x = chose_A[mask].sum()
        # Laplace smoothing to avoid log(0)
        p = (x + 0.5) / (n + 1.0)
        return np.log(p / (1.0 - p))
        
    lo12 = get_lo(m12)
    lo34 = get_lo(m34)
    lo56 = get_lo(m56)
    lo78 = get_lo(m78)
    
    # Numerator: Contrast where Mixture is exactly 0, WADD-DR is strictly positive
    num = lo34 - lo78
    # Denominator: Contrast that is positive for both and scales identically with beta
    denom = lo12 - lo56
    
    # Bounded normalized ratio to cancel out the beta variance
    return float(num / (abs(num) + abs(denom) + 0.1))
```

**Observed (real) value:** 0.0885 (var=0.0487)
**Previous candidate values (this loop):**
  - iter 1: 0.0645 (var=0.2344) (Δ vs real -0.0240)
  - iter 2: 0.1131 (var=0.1702) (Δ vs real +0.0246)
  - iter 3: 0.0854 (var=0.1789) (Δ vs real -0.0031)
  - iter 4: -0.1053 (var=0.2346) (Δ vs real -0.1938)
  - iter 5: 0.2041 (var=0.1592) (Δ vs real +0.1156)
  - iter 6: 0.3140 (var=0.1523) (Δ vs real +0.2255)
  - iter 7: 0.3533 (var=0.2306) (Δ vs real +0.2648)
  - iter 8 (most recent): 0.0555 (var=0.1807) (Δ vs real -0.0330)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3311 (var=0.1888)
- pi_4: -0.0236 (var=0.2122)
- pi_1: 0.2907 (var=0.3186)
- pi_2: -0.0310 (var=0.0885)
- pi_3: 0.3089 (var=0.1732)
- pi_5: 0.5910 (var=0.1844)
- pi_7: -0.4383 (var=0.2361)
- pi_8: -0.0265 (var=0.0856)
- pi_9: 0.1615 (var=0.2406)
- pi_10: 0.1827 (var=0.2096)
- pi_11: 0.1176 (var=0.1914)
- pi_12: -0.1220 (var=0.2496)
- pi_13: 0.1309 (var=0.1218)
- pi_14: 0.0758 (var=0.1555)
- pi_15: 0.0052 (var=0.0841)
- pi_16: 0.0843 (var=0.2106)
- pi_17: 0.2047 (var=0.1812)
- pi_18: 0.3135 (var=0.2388)
- pi_19: 0.2625 (var=0.1702)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    # Conflict trials: TTB prefers A (cue 1), but Tally prefers B (2 vs 3)
    t1 = (1, 0, 0, 0, 1)
    t2 = (1, 0, 0, 1, 0)
    t3 = (1, 0, 1, 0, 0)
    conflict_trials = {t1, t2, t3}
    
    # Agreement trial: TTB prefers A (cue 1), and Tally prefers A (2 vs 1)
    t6 = (1, 1, 0, 0, 0)
    
    subj_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        df_conflict = subj_df[subj_df['A_tuple'].isin(conflict_trials)]
        df_agree = subj_df[subj_df['A_tuple'] == t6]
        
        if len(df_conflict) == 0 or len(df_agree) == 0:
            continue
            
        # response = 0 means option A was chosen
        p_a_conflict = 1.0 - df_conflict['response'].mean()
        p_a_agree = 1.0 - df_agree['response'].mean()
        
        subj_diffs.append(p_a_agree - p_a_conflict)
        
    if not subj_diffs:
        return 0.0
        
    return float(np.mean(subj_diffs))
```

**Observed (real) value:** -0.4292 (var=0.0555)
**Previous candidate values (this loop):**
  - iter 1: -0.0158 (var=0.0264) (Δ vs real +0.4135)
  - iter 2: 0.1546 (var=0.0535) (Δ vs real +0.5838)
  - iter 3: 0.0612 (var=0.0427) (Δ vs real +0.4904)
  - iter 4: 0.1677 (var=0.0484) (Δ vs real +0.5969)
  - iter 5: 0.1673 (var=0.0631) (Δ vs real +0.5965)
  - iter 6: 0.1408 (var=0.0321) (Δ vs real +0.5700)
  - iter 7: 0.0950 (var=0.0334) (Δ vs real +0.5242)
  - iter 8 (most recent): 0.0523 (var=0.0266) (Δ vs real +0.4815)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1927 (var=0.0269)
- pi_7: 0.0235 (var=0.0242)
- pi_1: 0.0077 (var=0.0203)
- pi_2: 0.6223 (var=0.0421)
- pi_3: 0.0885 (var=0.0402)
- pi_5: 0.1638 (var=0.0160)
- pi_6: 0.0323 (var=0.0289)
- pi_8: 0.1088 (var=0.0248)
- pi_9: 0.0923 (var=0.0418)
- pi_10: 0.1050 (var=0.0177)
- pi_11: 0.1827 (var=0.0528)
- pi_12: 0.2415 (var=0.0382)
- pi_13: 0.0292 (var=0.0884)
- pi_14: 0.4273 (var=0.0675)
- pi_15: 0.3115 (var=0.0416)
- pi_16: 0.0592 (var=0.0859)
- pi_17: -0.2081 (var=0.0388)
- pi_18: 0.0127 (var=0.0236)
- pi_19: 0.2077 (var=0.0389)

### Experiment 12
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    t9_mask = (data['A_str'] == '11100') & (data['B_str'] == '00011')
    t10_mask = (data['A_str'] == '11010') & (data['B_str'] == '00101')
    t7_mask = (data['A_str'] == '10000') & (data['B_str'] == '01111')
    t8_mask = (data['A_str'] == '00111') & (data['B_str'] == '10000')
    
    ttb_t9 = 1.0 - data.loc[t9_mask, 'response'].mean() if t9_mask.sum() > 0 else 0.5
    ttb_t10 = 1.0 - data.loc[t10_mask, 'response'].mean() if t10_mask.sum() > 0 else 0.5
    ttb_t7 = 1.0 - data.loc[t7_mask, 'response'].mean() if t7_mask.sum() > 0 else 0.5
    ttb_t8 = data.loc[t8_mask, 'response'].mean() if t8_mask.sum() > 0 else 0.5
    
    agree = (ttb_t9 + ttb_t10) / 2.0
    disagree = (ttb_t7 + ttb_t8) / 2.0
    
    return float(agree - disagree)
```

**Observed (real) value:** -0.6711 (var=0.0499)
**Previous candidate values (this loop):**
  - iter 1: 0.0356 (var=0.0359) (Δ vs real +0.7067)
  - iter 2: 0.2311 (var=0.0865) (Δ vs real +0.9022)
  - iter 3: 0.1767 (var=0.0691) (Δ vs real +0.8478)
  - iter 4: 0.1822 (var=0.0875) (Δ vs real +0.8533)
  - iter 5: 0.1122 (var=0.0883) (Δ vs real +0.7833)
  - iter 6: 0.1533 (var=0.0533) (Δ vs real +0.8244)
  - iter 7: 0.1000 (var=0.0506) (Δ vs real +0.7711)
  - iter 8 (most recent): 0.0556 (var=0.0400) (Δ vs real +0.7267)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0444 (var=0.0356)
- pi_4: 0.2378 (var=0.0388)
- pi_1: -0.0222 (var=0.0131)
- pi_2: 0.6789 (var=0.0431)
- pi_3: 0.1478 (var=0.0759)
- pi_5: 0.1711 (var=0.0236)
- pi_6: 0.0478 (var=0.0311)
- pi_8: 0.0000 (var=0.0405)
- pi_9: 0.0789 (var=0.0387)
- pi_10: 0.0589 (var=0.0557)
- pi_11: 0.3389 (var=0.0800)
- pi_12: 0.0544 (var=0.0251)
- pi_13: 0.1067 (var=0.0601)
- pi_14: 0.4144 (var=0.1037)
- pi_15: 0.3922 (var=0.0949)
- pi_16: 0.0033 (var=0.1677)
- pi_17: -0.2833 (var=0.0318)
- pi_18: -0.0122 (var=0.0332)
- pi_19: 0.2856 (var=0.0681)

### Experiment 13
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the total number of positive cues is tied
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = a_sum == b_sum
    filtered = data[mask]
    
    if len(filtered) == 0:
        return 0.5
    
    # In these tied trials, check if the subject chose the option with the highest-validity cue (cue 0)
    a_cue0 = filtered['option_a_ratings'].apply(lambda x: x[0])
    chose_cue0 = ((a_cue0 == 1) & (filtered['response'] == 0)) | ((a_cue0 == 0) & (filtered['response'] == 1))
    
    return float(chose_cue0.mean())
```

**Observed (real) value:** 0.2644 (var=0.0112)
**Previous candidate values (this loop):**
  - iter 1: 0.5247 (var=0.0060) (Δ vs real +0.2603)
  - iter 2: 0.5356 (var=0.0055) (Δ vs real +0.2711)
  - iter 3: 0.5356 (var=0.0094) (Δ vs real +0.2711)
  - iter 4: 0.5631 (var=0.0073) (Δ vs real +0.2986)
  - iter 5: 0.5294 (var=0.0096) (Δ vs real +0.2650)
  - iter 6: 0.6181 (var=0.0131) (Δ vs real +0.3536)
  - iter 7: 0.5350 (var=0.0126) (Δ vs real +0.2706)
  - iter 8 (most recent): 0.5283 (var=0.0081) (Δ vs real +0.2639)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4861 (var=0.0031)
- pi_7: 0.7722 (var=0.0129)
- pi_1: 0.8689 (var=0.0074)
- pi_2: 0.5056 (var=0.0030)
- pi_3: 0.8119 (var=0.0117)
- pi_4: 0.7708 (var=0.0129)
- pi_5: 0.7458 (var=0.0159)
- pi_6: 0.5544 (var=0.0097)
- pi_9: 0.5758 (var=0.0727)
- pi_10: 0.7389 (var=0.0217)
- pi_11: 0.6389 (var=0.0203)
- pi_12: 0.2139 (var=0.0556)
- pi_13: 0.5275 (var=0.1160)
- pi_14: 0.4253 (var=0.0988)
- pi_15: 0.6722 (var=0.0189)
- pi_16: 0.4183 (var=0.1303)
- pi_17: 0.5478 (var=0.0097)
- pi_18: 0.5439 (var=0.0093)
- pi_19: 0.7128 (var=0.0206)

### Experiment 14
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive cues for options A and B
    a_sums = data['option_a_ratings'].apply(lambda x: sum(x))
    b_sums = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Isolate trials where both options have the same number of positive cues (Trials 1 and 2)
    mask = a_sums == b_sums
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    
    # In these trials, option A always possesses the most valid cue (cue 0)
    # We calculate the proportion of times the subject chose option A (response == 0)
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.1350 (var=0.0065)
**Previous candidate values (this loop):**
  - iter 1: 0.4904 (var=0.0047) (Δ vs real +0.3554)
  - iter 2: 0.5646 (var=0.0076) (Δ vs real +0.4296)
  - iter 3: 0.5229 (var=0.0104) (Δ vs real +0.3879)
  - iter 4: 0.5546 (var=0.0103) (Δ vs real +0.4196)
  - iter 5: 0.5500 (var=0.0172) (Δ vs real +0.4150)
  - iter 6: 0.6088 (var=0.0195) (Δ vs real +0.4738)
  - iter 7: 0.5450 (var=0.0090) (Δ vs real +0.4100)
  - iter 8 (most recent): 0.5212 (var=0.0070) (Δ vs real +0.3862)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7721 (var=0.0156)
- pi_8: 0.4925 (var=0.0046)
- pi_1: 0.8529 (var=0.0106)
- pi_2: 0.4979 (var=0.0064)
- pi_3: 0.8146 (var=0.0159)
- pi_4: 0.7725 (var=0.0116)
- pi_5: 0.7383 (var=0.0113)
- pi_6: 0.5767 (var=0.0130)
- pi_9: 0.5746 (var=0.0900)
- pi_10: 0.7417 (var=0.0261)
- pi_11: 0.6408 (var=0.0132)
- pi_12: 0.1625 (var=0.0471)
- pi_13: 0.5663 (var=0.1188)
- pi_14: 0.5204 (var=0.1096)
- pi_15: 0.7350 (var=0.0215)
- pi_16: 0.4608 (var=0.1408)
- pi_17: 0.5283 (var=0.0146)
- pi_18: 0.5704 (var=0.0178)
- pi_19: 0.6446 (var=0.0251)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a2 = data['option_a_ratings'].apply(lambda x: x[2])
    b2 = data['option_b_ratings'].apply(lambda x: x[2])
    
    mask = a2 != b2
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    a2_sub = a2[mask]
    b2_sub = b2[mask]
    
    ttb_pred = (b2_sub > a2_sub).astype(int)
    return float((subset['response'] == ttb_pred).mean())
```

**Observed (real) value:** 0.8031 (var=0.0244)
**Previous candidate values (this loop):**
  - iter 1: 0.4838 (var=0.0137) (Δ vs real -0.3192)
  - iter 2: 0.4408 (var=0.0238) (Δ vs real -0.3623)
  - iter 3: 0.3869 (var=0.0258) (Δ vs real -0.4162)
  - iter 4: 0.3469 (var=0.0303) (Δ vs real -0.4562)
  - iter 5: 0.4408 (var=0.0306) (Δ vs real -0.3623)
  - iter 6: 0.4615 (var=0.0276) (Δ vs real -0.3415)
  - iter 7: 0.4346 (var=0.0145) (Δ vs real -0.3685)
  - iter 8 (most recent): 0.4862 (var=0.0130) (Δ vs real -0.3169)
**Other theories' values on this metric (for reference):**
- pi_8: 0.7508 (var=0.0329)
- pi_9: 0.1408 (var=0.0120)
- pi_1: 0.8531 (var=0.0125)
- pi_2: 0.1223 (var=0.0100)
- pi_3: 0.6200 (var=0.0461)
- pi_4: 0.6008 (var=0.0357)
- pi_5: 0.6192 (var=0.0356)
- pi_6: 0.5285 (var=0.0127)
- pi_7: 0.7077 (var=0.0356)
- pi_10: 0.5869 (var=0.0497)
- pi_11: 0.3600 (var=0.0224)
- pi_12: 0.7992 (var=0.0527)
- pi_13: 0.3331 (var=0.0926)
- pi_14: 0.3246 (var=0.0925)
- pi_15: 0.4715 (var=0.0666)
- pi_16: 0.5038 (var=0.1611)
- pi_17: 0.8215 (var=0.0185)
- pi_18: 0.5500 (var=0.0244)
- pi_19: 0.4662 (var=0.0295)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Identify trials where the total number of cues is perfectly tied (diff_cues == 0) 
    # and the options are spatially symmetric (outer vs inner cues).
    # Trial 3: A=[1, 0, 0, 0, 0], B=[0, 0, 0, 0, 1]
    # Trial 4: A=[1, 1, 0, 0, 0], B=[0, 0, 0, 1, 1]
    mask = data['option_a_ratings'].apply(tuple).isin([(1, 0, 0, 0, 0), (1, 1, 0, 0, 0)])
    df_trial = data[mask]
    if len(df_trial) == 0:
        return 0.0
    
    # For the Competing model, diff_cues == 0 means 100% reliance on Tallying. 
    # Since the sum of cues is equal, Tallying predicts exactly 50/50, so subject means will be ~0.5.
    # For the Advocated model, extreme primacy or recency will drive choices deterministically 
    # towards A or B, so subject means will be near 0.0 or 1.0.
    # Measuring the absolute deviation from 0.5 captures this structural divergence.
    subj_means = df_trial.groupby('subject_id')['response'].mean()
    return float(np.mean(np.abs(subj_means - 0.5)))
```

**Observed (real) value:** 0.2611 (var=0.0294)
**Previous candidate values (this loop):**
  - iter 1: 0.0584 (var=0.0021) (Δ vs real -0.2026)
  - iter 2: 0.0763 (var=0.0032) (Δ vs real -0.1847)
  - iter 3: 0.0679 (var=0.0036) (Δ vs real -0.1932)
  - iter 4: 0.0984 (var=0.0056) (Δ vs real -0.1626)
  - iter 5: 0.0832 (var=0.0060) (Δ vs real -0.1779)
  - iter 6: 0.0674 (var=0.0027) (Δ vs real -0.1937)
  - iter 7: 0.0584 (var=0.0017) (Δ vs real -0.2026)
  - iter 8 (most recent): 0.0742 (var=0.0035) (Δ vs real -0.1868)
**Other theories' values on this metric (for reference):**
- pi_9: 0.3216 (var=0.0178)
- pi_8: 0.0658 (var=0.0028)
- pi_1: 0.3547 (var=0.0127)
- pi_2: 0.0679 (var=0.0037)
- pi_3: 0.1137 (var=0.0087)
- pi_4: 0.2447 (var=0.0108)
- pi_5: 0.1584 (var=0.0115)
- pi_6: 0.0684 (var=0.0033)
- pi_7: 0.1747 (var=0.0131)
- pi_10: 0.1621 (var=0.0206)
- pi_11: 0.0863 (var=0.0054)
- pi_12: 0.3395 (var=0.0186)
- pi_13: 0.2089 (var=0.0253)
- pi_14: 0.2716 (var=0.0199)
- pi_15: 0.2158 (var=0.0213)
- pi_16: 0.3584 (var=0.0092)
- pi_17: 0.0974 (var=0.0035)
- pi_18: 0.0779 (var=0.0063)
- pi_19: 0.0900 (var=0.0047)

### Experiment 17
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    correct = 0
    total = 0
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if sum(a) == sum(b):
            for i in range(len(a)):
                if a[i] != b[i]:
                    expected = 0 if a[i] > b[i] else 1
                    if resp == expected:
                        correct += 1
                    break
            total += 1
            
    return float(correct / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.2825 (var=0.0159)
**Previous candidate values (this loop):**
  - iter 1: 0.5133 (var=0.0048) (Δ vs real +0.2308)
  - iter 2: 0.5413 (var=0.0078) (Δ vs real +0.2588)
  - iter 3: 0.5267 (var=0.0171) (Δ vs real +0.2442)
  - iter 4: 0.5321 (var=0.0054) (Δ vs real +0.2496)
  - iter 5: 0.5325 (var=0.0102) (Δ vs real +0.2500)
  - iter 6: 0.5725 (var=0.0042) (Δ vs real +0.2900)
  - iter 7: 0.5713 (var=0.0120) (Δ vs real +0.2888)
  - iter 8 (most recent): 0.5067 (var=0.0040) (Δ vs real +0.2242)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4938 (var=0.0038)
- pi_10: 0.7017 (var=0.0218)
- pi_1: 0.8517 (var=0.0136)
- pi_2: 0.5033 (var=0.0039)
- pi_3: 0.7708 (var=0.0174)
- pi_4: 0.7521 (var=0.0140)
- pi_5: 0.6846 (var=0.0134)
- pi_6: 0.5333 (var=0.0069)
- pi_7: 0.7550 (var=0.0149)
- pi_9: 0.6763 (var=0.0302)
- pi_11: 0.5300 (var=0.0085)
- pi_12: 0.4238 (var=0.0184)
- pi_13: 0.5713 (var=0.0494)
- pi_14: 0.5308 (var=0.0766)
- pi_15: 0.7100 (var=0.0246)
- pi_16: 0.4537 (var=0.1374)
- pi_17: 0.7708 (var=0.0132)
- pi_18: 0.5429 (var=0.0066)
- pi_19: 0.5775 (var=0.0111)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where both options have the same total number of positive cues (zero conflict)
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    zero_diff = (sum_a == sum_b)
    
    subset = data[zero_diff]
    if len(subset) == 0:
        return 0.5
        
    # Identify which option possesses the highest-validity cue (index 0)
    a_has_cue1 = subset['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_cue1 = subset['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    # Calculate how often the subject chose the option with the highest-validity cue
    chose_a = (subset['response'] == 0)
    chose_b = (subset['response'] == 1)
    
    chose_highest_validity = (chose_a & a_has_cue1) | (chose_b & b_has_cue1)
    
    return float(chose_highest_validity.mean())
```

**Observed (real) value:** 0.3458 (var=0.0444)
**Previous candidate values (this loop):**
  - iter 1: 0.4879 (var=0.0036) (Δ vs real +0.1421)
  - iter 2: 0.5312 (var=0.0059) (Δ vs real +0.1854)
  - iter 3: 0.5304 (var=0.0091) (Δ vs real +0.1846)
  - iter 4: 0.5150 (var=0.0055) (Δ vs real +0.1692)
  - iter 5: 0.5421 (var=0.0126) (Δ vs real +0.1963)
  - iter 6: 0.5517 (var=0.0082) (Δ vs real +0.2058)
  - iter 7: 0.5363 (var=0.0058) (Δ vs real +0.1904)
  - iter 8 (most recent): 0.5088 (var=0.0038) (Δ vs real +0.1629)
**Other theories' values on this metric (for reference):**
- pi_10: 0.7100 (var=0.0283)
- pi_8: 0.4883 (var=0.0056)
- pi_1: 0.8488 (var=0.0116)
- pi_2: 0.4983 (var=0.0061)
- pi_3: 0.7408 (var=0.0154)
- pi_4: 0.7654 (var=0.0104)
- pi_5: 0.7238 (var=0.0132)
- pi_6: 0.5546 (var=0.0090)
- pi_7: 0.7325 (var=0.0216)
- pi_9: 0.6946 (var=0.0320)
- pi_11: 0.5208 (var=0.0060)
- pi_12: 0.5196 (var=0.0047)
- pi_13: 0.6421 (var=0.0262)
- pi_14: 0.4813 (var=0.1119)
- pi_15: 0.7438 (var=0.0279)
- pi_16: 0.4288 (var=0.1416)
- pi_17: 0.7392 (var=0.0154)
- pi_18: 0.5437 (var=0.0077)
- pi_19: 0.5954 (var=0.0132)

### Experiment 19
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive cues for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter for trials where both options have the same number of positive cues
    # (i.e., diff_cues == 0)
    mask = sum_a == sum_b
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(subset['response'] == 0))

```

**Observed (real) value:** 0.1758 (var=0.0110)
**Previous candidate values (this loop):**
  - iter 1: 0.4963 (var=0.0079) (Δ vs real +0.3205)
  - iter 2: 0.5442 (var=0.0125) (Δ vs real +0.3684)
  - iter 3: 0.5495 (var=0.0185) (Δ vs real +0.3737)
  - iter 4: 0.5253 (var=0.0069) (Δ vs real +0.3495)
  - iter 5: 0.5116 (var=0.0098) (Δ vs real +0.3358)
  - iter 6: 0.5574 (var=0.0112) (Δ vs real +0.3816)
  - iter 7: 0.5316 (var=0.0129) (Δ vs real +0.3558)
  - iter 8 (most recent): 0.5258 (var=0.0075) (Δ vs real +0.3500)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4742 (var=0.0058)
- pi_11: 0.5853 (var=0.0135)
- pi_1: 0.8453 (var=0.0113)
- pi_2: 0.5105 (var=0.0066)
- pi_3: 0.7842 (var=0.0210)
- pi_4: 0.7505 (var=0.0134)
- pi_5: 0.6884 (var=0.0161)
- pi_6: 0.5205 (var=0.0062)
- pi_7: 0.7732 (var=0.0187)
- pi_9: 0.6700 (var=0.0544)
- pi_10: 0.6905 (var=0.0233)
- pi_12: 0.2053 (var=0.0502)
- pi_13: 0.4011 (var=0.1217)
- pi_14: 0.5968 (var=0.0905)
- pi_15: 0.7458 (var=0.0231)
- pi_16: 0.4921 (var=0.1261)
- pi_17: 0.6858 (var=0.0124)
- pi_18: 0.5726 (var=0.0087)
- pi_19: 0.6084 (var=0.0154)

### Experiment 20
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    
    # Trial 3: A=[1,0,0,0,0] (sum=1), B=[0,1,1,1,1] (sum=4)
    mask_t3 = (sum_a == 1) & (sum_b == 4)
    # Trial 5: A=[1,1,0,0,0] (sum=2), B=[0,0,1,1,1] (sum=3)
    mask_t5 = (sum_a == 2) & (sum_b == 3)
    
    if not np.any(mask_t3) or not np.any(mask_t5):
        return 0.0
        
    responses = data['response'].values
    
    # Probability of choosing Option B in Trial 3 and Trial 5
    p_b_t3 = np.mean(responses[mask_t3] == 1)
    p_b_t5 = np.mean(responses[mask_t5] == 1)
    
    # Return the difference in probability of choosing B between Trial 5 and Trial 3
    return float(p_b_t5 - p_b_t3)
```

**Observed (real) value:** 0.2025 (var=0.0829)
**Previous candidate values (this loop):**
  - iter 1: -0.0025 (var=0.0323) (Δ vs real -0.2050)
  - iter 2: -0.0512 (var=0.0415) (Δ vs real -0.2537)
  - iter 3: -0.1325 (var=0.0726) (Δ vs real -0.3350)
  - iter 4: -0.1175 (var=0.0609) (Δ vs real -0.3200)
  - iter 5: -0.0787 (var=0.0511) (Δ vs real -0.2812)
  - iter 6: -0.2487 (var=0.0743) (Δ vs real -0.4512)
  - iter 7: -0.0575 (var=0.0400) (Δ vs real -0.2600)
  - iter 8 (most recent): 0.0038 (var=0.0360) (Δ vs real -0.1987)
**Other theories' values on this metric (for reference):**
- pi_11: -0.1675 (var=0.0405)
- pi_8: 0.0612 (var=0.0277)
- pi_1: -0.0125 (var=0.0145)
- pi_2: -0.0387 (var=0.0200)
- pi_3: -0.2275 (var=0.0912)
- pi_4: -0.0463 (var=0.0270)
- pi_5: -0.1750 (var=0.0239)
- pi_6: -0.0600 (var=0.0261)
- pi_7: -0.0337 (var=0.0453)
- pi_9: -0.0887 (var=0.0481)
- pi_10: -0.1213 (var=0.0588)
- pi_12: -0.0038 (var=0.0152)
- pi_13: -0.1350 (var=0.0840)
- pi_14: -0.0663 (var=0.0301)
- pi_15: -0.0688 (var=0.0219)
- pi_16: -0.0075 (var=0.0196)
- pi_17: 0.2375 (var=0.0603)
- pi_18: 0.0162 (var=0.0389)
- pi_19: -0.1538 (var=0.0844)

### Experiment 21
**Design**
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of positive cues for A and B
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    # Filter for zero-conflict (tied sum) trials
    tied_mask = sum_a == sum_b
    
    if not tied_mask.any():
        return 0.5
        
    tied_data = data[tied_mask]
    
    # The Competing theory predicts a boost to Reverse TTB on tied trials.
    # In the experimental design, Option A always wins the lowest-validity 
    # cue on the tied trials (Trial 1 and Trial 2).
    # The Advocated theory predicts exactly 50/50 on these trials.
    # We return the proportion of times Option A is chosen (response == 0).
    return float(np.mean(tied_data['response'] == 0))
```

**Observed (real) value:** 0.4050 (var=0.0313)
**Previous candidate values (this loop):**
  - iter 1: 0.4875 (var=0.0095) (Δ vs real +0.0825)
  - iter 2: 0.4425 (var=0.0118) (Δ vs real +0.0375)
  - iter 3: 0.4606 (var=0.0147) (Δ vs real +0.0556)
  - iter 4: 0.4437 (var=0.0131) (Δ vs real +0.0387)
  - iter 5: 0.4744 (var=0.0119) (Δ vs real +0.0694)
  - iter 6: 0.4244 (var=0.0120) (Δ vs real +0.0194)
  - iter 7: 0.4838 (var=0.0082) (Δ vs real +0.0787)
  - iter 8 (most recent): 0.4763 (var=0.0099) (Δ vs real +0.0712)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5000 (var=0.0067)
- pi_12: 0.8000 (var=0.0345)
- pi_1: 0.4969 (var=0.0031)
- pi_2: 0.4894 (var=0.0056)
- pi_3: 0.4581 (var=0.0092)
- pi_4: 0.4844 (var=0.0058)
- pi_5: 0.4219 (var=0.0071)
- pi_6: 0.4906 (var=0.0089)
- pi_7: 0.4894 (var=0.0055)
- pi_9: 0.5731 (var=0.0369)
- pi_10: 0.4575 (var=0.0085)
- pi_11: 0.4062 (var=0.0154)
- pi_13: 0.6406 (var=0.0538)
- pi_14: 0.4944 (var=0.0050)
- pi_15: 0.4969 (var=0.0047)
- pi_16: 0.5069 (var=0.0043)
- pi_17: 0.5837 (var=0.0119)
- pi_18: 0.4587 (var=0.0145)
- pi_19: 0.3531 (var=0.0187)

### Experiment 22
**Design**
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sums_a = data['option_a_ratings'].apply(sum)
    sums_b = data['option_b_ratings'].apply(sum)
    tied = data[sums_a == sums_b]
    if len(tied) == 0:
        return 0.5
    return float((tied['response'] == 0).mean())
```

**Observed (real) value:** 0.5684 (var=0.0814)
**Previous candidate values (this loop):**
  - iter 1: 0.4532 (var=0.0127) (Δ vs real -0.1153)
  - iter 2: 0.3979 (var=0.0110) (Δ vs real -0.1705)
  - iter 3: 0.4621 (var=0.0172) (Δ vs real -0.1063)
  - iter 4: 0.4395 (var=0.0106) (Δ vs real -0.1289)
  - iter 5: 0.4311 (var=0.0258) (Δ vs real -0.1374)
  - iter 6: 0.3489 (var=0.0227) (Δ vs real -0.2195)
  - iter 7: 0.4589 (var=0.0164) (Δ vs real -0.1095)
  - iter 8 (most recent): 0.4837 (var=0.0086) (Δ vs real -0.0847)
**Other theories' values on this metric (for reference):**
- pi_12: 0.8284 (var=0.0469)
- pi_8: 0.5084 (var=0.0076)
- pi_1: 0.1584 (var=0.0089)
- pi_2: 0.5116 (var=0.0059)
- pi_3: 0.1737 (var=0.0151)
- pi_4: 0.2558 (var=0.0164)
- pi_5: 0.2500 (var=0.0142)
- pi_6: 0.4611 (var=0.0132)
- pi_7: 0.2084 (var=0.0201)
- pi_9: 0.4111 (var=0.1285)
- pi_10: 0.2816 (var=0.0362)
- pi_11: 0.3716 (var=0.0228)
- pi_13: 0.5484 (var=0.1431)
- pi_14: 0.3879 (var=0.1063)
- pi_15: 0.2926 (var=0.0282)
- pi_16: 0.5274 (var=0.1550)
- pi_17: 0.4779 (var=0.0167)
- pi_18: 0.4532 (var=0.0175)
- pi_19: 0.3232 (var=0.0227)

### Experiment 23
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify zero-conflict trials where the total number of positive cues is equal
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    zero_conflict_mask = (a_sums == b_sums)
    
    df_zero = data[zero_conflict_mask]
    if df_zero.empty:
        return 0.0
        
    # Calculate proportion of A choices (response == 0) per subject
    p_a = (df_zero['response'] == 0).groupby(df_zero['subject_id']).mean()
    
    # Mean absolute deviation from 0.5 across subjects
    return float(np.mean(np.abs(p_a - 0.5)))

```

**Observed (real) value:** 0.3133 (var=0.0161)
**Previous candidate values (this loop):**
  - iter 1: 0.0592 (var=0.0017) (Δ vs real -0.2542)
  - iter 2: 0.0579 (var=0.0025) (Δ vs real -0.2554)
  - iter 3: 0.0633 (var=0.0060) (Δ vs real -0.2500)
  - iter 4: 0.0642 (var=0.0017) (Δ vs real -0.2492)
  - iter 5: 0.0638 (var=0.0035) (Δ vs real -0.2496)
  - iter 6: 0.0792 (var=0.0035) (Δ vs real -0.2342)
  - iter 7: 0.0696 (var=0.0043) (Δ vs real -0.2438)
  - iter 8 (most recent): 0.0504 (var=0.0018) (Δ vs real -0.2629)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0563 (var=0.0017)
- pi_13: 0.2817 (var=0.0201)
- pi_1: 0.3250 (var=0.0135)
- pi_2: 0.0521 (var=0.0017)
- pi_3: 0.2079 (var=0.0144)
- pi_4: 0.2804 (var=0.0146)
- pi_5: 0.1404 (var=0.0110)
- pi_6: 0.0675 (var=0.0028)
- pi_7: 0.2533 (var=0.0121)
- pi_9: 0.2367 (var=0.0190)
- pi_10: 0.1783 (var=0.0203)
- pi_11: 0.0700 (var=0.0032)
- pi_12: 0.3850 (var=0.0106)
- pi_14: 0.2350 (var=0.0202)
- pi_15: 0.2029 (var=0.0223)
- pi_16: 0.3871 (var=0.0078)
- pi_17: 0.1263 (var=0.0068)
- pi_18: 0.0546 (var=0.0012)
- pi_19: 0.0754 (var=0.0031)

### Experiment 24
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate sum of positive cues for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter for zero-difference trials (where total cue counts are equal)
    zero_diff = data[sum_a == sum_b].copy()
    
    # Create a unique string identifier for the trial types
    zero_diff['trial_type'] = zero_diff['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + '_' + zero_diff['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of times each subject chose Option A (response == 0) for each trial type
    prop_a = zero_diff.groupby(['subject_id', 'trial_type'])['response'].apply(lambda x: (x == 0).mean()).reset_index()
    
    # Calculate the absolute deviation from 0.5 (random guessing)
    prop_a['abs_dev'] = (prop_a['response'] - 0.5).abs()
    
    # Average the absolute deviation across trial types for each subject, then return the overall mean
    return float(prop_a.groupby('subject_id')['abs_dev'].mean().mean())
```

**Observed (real) value:** 0.3702 (var=0.0075)
**Previous candidate values (this loop):**
  - iter 1: 0.0937 (var=0.0014) (Δ vs real -0.2765)
  - iter 2: 0.1140 (var=0.0033) (Δ vs real -0.2561)
  - iter 3: 0.1004 (var=0.0019) (Δ vs real -0.2698)
  - iter 4: 0.1039 (var=0.0031) (Δ vs real -0.2663)
  - iter 5: 0.1161 (var=0.0053) (Δ vs real -0.2540)
  - iter 6: 0.1449 (var=0.0086) (Δ vs real -0.2253)
  - iter 7: 0.1277 (var=0.0083) (Δ vs real -0.2425)
  - iter 8 (most recent): 0.1000 (var=0.0019) (Δ vs real -0.2702)
**Other theories' values on this metric (for reference):**
- pi_13: 0.3481 (var=0.0112)
- pi_8: 0.0958 (var=0.0022)
- pi_1: 0.3684 (var=0.0095)
- pi_2: 0.0979 (var=0.0013)
- pi_3: 0.3025 (var=0.0075)
- pi_4: 0.2744 (var=0.0080)
- pi_5: 0.2260 (var=0.0141)
- pi_6: 0.1372 (var=0.0048)
- pi_7: 0.2681 (var=0.0117)
- pi_9: 0.2996 (var=0.0161)
- pi_10: 0.2449 (var=0.0134)
- pi_11: 0.1646 (var=0.0105)
- pi_12: 0.3611 (var=0.0193)
- pi_14: 0.3074 (var=0.0144)
- pi_15: 0.2088 (var=0.0158)
- pi_16: 0.3656 (var=0.0105)
- pi_17: 0.1347 (var=0.0051)
- pi_18: 0.1235 (var=0.0051)
- pi_19: 0.1856 (var=0.0100)

### Experiment 25
**Design**
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        # In all 10 trials, one option has 4 or 5 cues, the other has 0 or 1.
        # We identify the dominant option (the one with more 1s).
        if sum(a) > sum(b):
            return str(a) + str(b), 1 if resp == 0 else 0
        else:
            return str(b) + str(a), 1 if resp == 1 else 0

    mapped = data.apply(get_trial, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_dom'])
    df['subject_id'] = data['subject_id'].values
    
    counts = df.groupby(['subject_id', 'trial'])['chose_dom'].agg(['sum', 'count'])
    
    def calc_M(sub_df):
        valid = sub_df[sub_df['count'] > 1]
        if len(valid) < 2:
            return np.nan
        
        X = valid['sum'].values.astype(float)
        R = valid['count'].values.astype(float)
        Y = X / R
        
        # S2_Y is the sample variance of the observed choice proportions across the 10 trials
        S2_Y = np.var(Y, ddof=1)
        
        # W_t is the exact unbiased estimator of the binomial variance for trial t: p_t(1-p_t)/R_t
        W = X * (R - X) / (R**2 * (R - 1.0))
        mean_W = np.mean(W)
        
        # M is the unbiased estimator of the variance of the true underlying choice probabilities
        return S2_Y - mean_W

    M_per_subj = counts.groupby('subject_id').apply(calc_M).dropna()
    if M_per_subj.empty:
        return 0.0
        
    return float(M_per_subj.mean())
```

**Observed (real) value:** -0.0010 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.0003 (var=0.0001) (Δ vs real +0.0014)
  - iter 2: 0.0015 (var=0.0001) (Δ vs real +0.0025)
  - iter 3: -0.0003 (var=0.0000) (Δ vs real +0.0007)
  - iter 4: -0.0027 (var=0.0001) (Δ vs real -0.0017)
  - iter 5: 0.0044 (var=0.0001) (Δ vs real +0.0054)
  - iter 6: 0.0009 (var=0.0001) (Δ vs real +0.0019)
  - iter 7: 0.0003 (var=0.0001) (Δ vs real +0.0013)
  - iter 8 (most recent): 0.0143 (var=0.0006) (Δ vs real +0.0153)
**Other theories' values on this metric (for reference):**
- pi_14: 0.0002 (var=0.0001)
- pi_13: 0.0005 (var=0.0000)
- pi_1: 0.0001 (var=0.0000)
- pi_2: -0.0010 (var=0.0001)
- pi_3: 0.0025 (var=0.0001)
- pi_4: -0.0001 (var=0.0001)
- pi_5: -0.0002 (var=0.0001)
- pi_6: -0.0005 (var=0.0002)
- pi_7: 0.0011 (var=0.0001)
- pi_8: 0.0019 (var=0.0001)
- pi_9: 0.0061 (var=0.0002)
- pi_10: -0.0006 (var=0.0001)
- pi_11: -0.0016 (var=0.0001)
- pi_12: -0.0004 (var=0.0000)
- pi_15: -0.0005 (var=0.0000)
- pi_16: 0.0000 (var=0.0000)
- pi_17: 0.0182 (var=0.0006)
- pi_18: 0.0001 (var=0.0001)
- pi_19: 0.0012 (var=0.0001)

### Experiment 26
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    is_A = (data['response'] == 0).astype(float)
    
    t1 = (data['A_tuple'] == (1, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 1, 1, 0))
    t4 = (data['A_tuple'] == (1, 0, 0, 0, 1)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t5 = (data['A_tuple'] == (1, 1, 1, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 0))
    t7 = (data['A_tuple'] == (1, 0, 1, 1, 0)) & (data['B_tuple'] == (0, 1, 0, 0, 0))
    
    p1 = is_A[t1].mean() if t1.sum() > 0 else 0.5
    p4 = is_A[t4].mean() if t4.sum() > 0 else 0.5
    p5 = is_A[t5].mean() if t5.sum() > 0 else 0.5
    p7 = is_A[t7].mean() if t7.sum() > 0 else 0.5
    
    return float((p1 - p4) + (p5 - p7))
```

**Observed (real) value:** 0.0154 (var=0.0544)
**Previous candidate values (this loop):**
  - iter 1: -0.0077 (var=0.0791) (Δ vs real -0.0231)
  - iter 2: 0.0954 (var=0.0595) (Δ vs real +0.0800)
  - iter 3: 0.0323 (var=0.0599) (Δ vs real +0.0169)
  - iter 4: 0.1969 (var=0.0718) (Δ vs real +0.1815)
  - iter 5: 0.0338 (var=0.0905) (Δ vs real +0.0185)
  - iter 6: 0.2000 (var=0.1046) (Δ vs real +0.1846)
  - iter 7: 0.1031 (var=0.0707) (Δ vs real +0.0877)
  - iter 8 (most recent): 0.0262 (var=0.0875) (Δ vs real +0.0108)
**Other theories' values on this metric (for reference):**
- pi_13: -0.6923 (var=0.7401)
- pi_14: -0.0462 (var=0.0386)
- pi_1: -0.0323 (var=0.0341)
- pi_2: 0.0154 (var=0.0627)
- pi_3: 0.1354 (var=0.0877)
- pi_4: -0.0092 (var=0.0380)
- pi_5: 0.2585 (var=0.0577)
- pi_6: 0.0815 (var=0.0832)
- pi_7: 0.0492 (var=0.0560)
- pi_8: 0.0138 (var=0.0854)
- pi_9: -0.3308 (var=0.3165)
- pi_10: 0.1492 (var=0.0567)
- pi_11: 0.1662 (var=0.0718)
- pi_12: -0.8031 (var=0.1420)
- pi_15: -0.0354 (var=0.0400)
- pi_16: 0.0015 (var=0.0408)
- pi_17: -0.0954 (var=0.1085)
- pi_18: 0.1477 (var=0.1318)
- pi_19: 0.2908 (var=0.0718)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    df['a_first'] = df['option_a_ratings'].apply(lambda x: x[0])
    df['b_sum'] = df['option_b_ratings'].apply(sum)
    df['a_sum'] = df['option_a_ratings'].apply(sum)
    
    # T5, T6: A wins TTB (a_first == 1) and B has massive tally advantage (b_sum in [5, 6])
    mask_A = (df['a_first'] == 1) & (df['b_sum'].isin([5, 6]))
    p_A = (df.loc[mask_A, 'response'] == 0).mean() if mask_A.any() else 0.5
    
    # T9: B wins TTB (a_first == 0) and A has massive tally advantage (a_sum == 6)
    mask_B = (df['a_first'] == 0) & (df['a_sum'] == 6)
    p_B = (df.loc[mask_B, 'response'] == 1).mean() if mask_B.any() else 0.5
    
    return float((p_A + p_B) / 2.0)
```

**Observed (real) value:** 0.8320 (var=0.0112)
**Previous candidate values (this loop):**
  - iter 1: 0.4510 (var=0.0163) (Δ vs real -0.3810)
  - iter 2: 0.3805 (var=0.0179) (Δ vs real -0.4515)
  - iter 3: 0.4125 (var=0.0238) (Δ vs real -0.4195)
  - iter 4: 0.3715 (var=0.0222) (Δ vs real -0.4605)
  - iter 5: 0.4145 (var=0.0364) (Δ vs real -0.4175)
  - iter 6: 0.4650 (var=0.0225) (Δ vs real -0.3670)
  - iter 7: 0.4070 (var=0.0167) (Δ vs real -0.4250)
  - iter 8 (most recent): 0.4385 (var=0.0191) (Δ vs real -0.3935)
**Other theories' values on this metric (for reference):**
- pi_14: 0.2405 (var=0.0648)
- pi_15: 0.5480 (var=0.0879)
- pi_1: 0.8665 (var=0.0132)
- pi_2: 0.1265 (var=0.0098)
- pi_3: 0.5865 (var=0.1055)
- pi_4: 0.5895 (var=0.0309)
- pi_5: 0.5865 (var=0.0381)
- pi_6: 0.4670 (var=0.0248)
- pi_7: 0.7810 (var=0.0244)
- pi_8: 0.7690 (var=0.0442)
- pi_9: 0.4465 (var=0.1257)
- pi_10: 0.6190 (var=0.0457)
- pi_11: 0.2805 (var=0.0282)
- pi_12: 0.8245 (var=0.0590)
- pi_13: 0.3855 (var=0.1025)
- pi_16: 0.4980 (var=0.1427)
- pi_17: 0.7755 (var=0.0316)
- pi_18: 0.6070 (var=0.0268)
- pi_19: 0.4195 (var=0.0403)

### Experiment 28
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_ttb_choice(row):
        # Cue 0 is the highest validity cue (0.95)
        a_wins_ttb = row['option_a_ratings'][0] > row['option_b_ratings'][0]
        ttb_winner = 0 if a_wins_ttb else 1
        return 1.0 if row['response'] == ttb_winner else 0.0
        
    return float(data.apply(is_ttb_choice, axis=1).mean())
```

**Observed (real) value:** 0.1467 (var=0.0053)
**Previous candidate values (this loop):**
  - iter 1: 0.7050 (var=0.0270) (Δ vs real +0.5583)
  - iter 2: 0.7410 (var=0.0201) (Δ vs real +0.5944)
  - iter 3: 0.8631 (var=0.0084) (Δ vs real +0.7165)
  - iter 4: 0.7090 (var=0.0202) (Δ vs real +0.5623)
  - iter 5: 0.8521 (var=0.0152) (Δ vs real +0.7054)
  - iter 6: 0.7323 (var=0.0214) (Δ vs real +0.5856)
  - iter 7: 0.8533 (var=0.0117) (Δ vs real +0.7067)
  - iter 8 (most recent): 0.7562 (var=0.0232) (Δ vs real +0.6096)
**Other theories' values on this metric (for reference):**
- pi_15: 0.9419 (var=0.0103)
- pi_14: 0.8358 (var=0.0066)
- pi_1: 0.8494 (var=0.0101)
- pi_2: 0.8677 (var=0.0069)
- pi_3: 0.8154 (var=0.0156)
- pi_4: 0.9071 (var=0.0113)
- pi_5: 0.8698 (var=0.0077)
- pi_6: 0.5813 (var=0.0238)
- pi_7: 0.7802 (var=0.0151)
- pi_8: 0.8060 (var=0.0123)
- pi_9: 0.7481 (var=0.0211)
- pi_10: 0.7519 (var=0.0261)
- pi_11: 0.7129 (var=0.0139)
- pi_12: 0.8721 (var=0.0114)
- pi_13: 0.8573 (var=0.0078)
- pi_16: 0.4644 (var=0.0984)
- pi_17: 0.6400 (var=0.0077)
- pi_18: 0.5896 (var=0.0189)
- pi_19: 0.7556 (var=0.0291)

### Experiment 29
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['chose_A'] = 1 - data['response']
    data['tally_A'] = data['option_a_ratings'].apply(sum)
    data['tally_B'] = data['option_b_ratings'].apply(sum)
    data['tally_diff'] = data['tally_A'] - data['tally_B']
    
    pos_mean = data[data['tally_diff'] > 0]['chose_A'].mean()
    neg_mean = data[data['tally_diff'] < 0]['chose_A'].mean()
    
    if pd.isna(pos_mean): pos_mean = 0.5
    if pd.isna(neg_mean): neg_mean = 0.5
    
    return float(pos_mean - neg_mean)
```

**Observed (real) value:** -0.6071 (var=0.0412)
**Previous candidate values (this loop):**
  - iter 1: 0.2552 (var=0.0552) (Δ vs real +0.8623)
  - iter 2: 0.3323 (var=0.0744) (Δ vs real +0.9394)
  - iter 3: 0.5025 (var=0.0484) (Δ vs real +1.1096)
  - iter 4: 0.4379 (var=0.1115) (Δ vs real +1.0450)
  - iter 5: 0.4167 (var=0.0312) (Δ vs real +1.0237)
  - iter 6: 0.1952 (var=0.0533) (Δ vs real +0.8023)
  - iter 7: 0.3979 (var=0.0342) (Δ vs real +1.0050)
  - iter 8 (most recent): 0.2285 (var=0.0456) (Δ vs real +0.8356)
**Other theories' values on this metric (for reference):**
- pi_16: 0.0298 (var=0.2381)
- pi_15: 0.4731 (var=0.0711)
- pi_1: 0.0175 (var=0.0050)
- pi_2: 0.7331 (var=0.0311)
- pi_3: 0.0921 (var=0.0541)
- pi_4: 0.2338 (var=0.0241)
- pi_5: 0.2946 (var=0.0357)
- pi_6: 0.0677 (var=0.0350)
- pi_7: -0.0010 (var=0.0080)
- pi_8: 0.1056 (var=0.0333)
- pi_9: 0.2456 (var=0.0555)
- pi_10: 0.1169 (var=0.0287)
- pi_11: 0.3692 (var=0.0559)
- pi_12: 0.1290 (var=0.0699)
- pi_13: 0.4502 (var=0.1616)
- pi_14: 0.5008 (var=0.0910)
- pi_17: -0.1767 (var=0.0234)
- pi_18: 0.0394 (var=0.0236)
- pi_19: 0.3385 (var=0.0934)

### Experiment 30
**Design**
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    # Trial 1: A has 4 cues, B has 1 cue
    t1_mask = (a_sum == 4) & (b_sum == 1)
    
    # Trial 7: A has 1 cue, B has 4 cues
    t7_mask = (a_sum == 1) & (b_sum == 4)
    
    # response == 0 means Option A was chosen
    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_a_t7 = 1.0 - data.loc[t7_mask, 'response'].mean()
    
    # Handle cases where a subject might have missed a trial type (fallback to 0)
    if np.isnan(p_a_t1):
        p_a_t1 = 0.5
    if np.isnan(p_a_t7):
        p_a_t7 = 0.5
        
    return float(p_a_t1 - p_a_t7)
```

**Observed (real) value:** -0.7262 (var=0.0611)
**Previous candidate values (this loop):**
  - iter 1: 0.1554 (var=0.0804) (Δ vs real +0.8815)
  - iter 2: 0.3015 (var=0.1389) (Δ vs real +1.0277)
  - iter 3: 0.2292 (var=0.0872) (Δ vs real +0.9554)
  - iter 4: 0.3369 (var=0.1077) (Δ vs real +1.0631)
  - iter 5: 0.1308 (var=0.0862) (Δ vs real +0.8569)
  - iter 6: 0.2354 (var=0.1199) (Δ vs real +0.9615)
  - iter 7: 0.1492 (var=0.0728) (Δ vs real +0.8754)
  - iter 8 (most recent): 0.1154 (var=0.0536) (Δ vs real +0.8415)
**Other theories' values on this metric (for reference):**
- pi_15: 0.4292 (var=0.0864)
- pi_16: 0.0246 (var=0.3400)
- pi_1: 0.0308 (var=0.0118)
- pi_2: 0.7138 (var=0.0455)
- pi_3: 0.1200 (var=0.0703)
- pi_4: 0.3246 (var=0.0498)
- pi_5: 0.2323 (var=0.0306)
- pi_6: 0.0662 (var=0.0627)
- pi_7: 0.1046 (var=0.0695)
- pi_8: 0.0477 (var=0.0203)
- pi_9: 0.1631 (var=0.0823)
- pi_10: 0.1508 (var=0.0914)
- pi_11: 0.5292 (var=0.0818)
- pi_12: 0.0415 (var=0.0258)
- pi_13: 0.1938 (var=0.1066)
- pi_14: 0.5462 (var=0.1264)
- pi_17: -0.2677 (var=0.0576)
- pi_18: 0.0600 (var=0.0484)
- pi_19: 0.3369 (var=0.0968)

### Experiment 31
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_a = (1, 0, 0, 0, 0, 0)
    t1_b = (0, 1, 0, 0, 0, 0)
    t3_a = (1, 0, 1, 0, 1, 0)
    t3_b = (0, 1, 0, 0, 0, 0)
    
    t4_a = (1, 0, 0, 0, 0, 0)
    t4_b = (0, 1, 0, 1, 0, 0)
    t7_a = (1, 0, 1, 0, 1, 0)
    t7_b = (0, 1, 0, 1, 0, 1)
    
    mask_t1 = (a_tuples == t1_a) & (b_tuples == t1_b)
    mask_t3 = (a_tuples == t3_a) & (b_tuples == t3_b)
    mask_t4 = (a_tuples == t4_a) & (b_tuples == t4_b)
    mask_t7 = (a_tuples == t7_a) & (b_tuples == t7_b)
    
    diff1 = 0.0
    if mask_t1.sum() > 0 and mask_t3.sum() > 0:
        diff1 = data.loc[mask_t3, 'response'].mean() - data.loc[mask_t1, 'response'].mean()
        
    diff2 = 0.0
    if mask_t4.sum() > 0 and mask_t7.sum() > 0:
        diff2 = data.loc[mask_t7, 'response'].mean() - data.loc[mask_t4, 'response'].mean()
        
    return float(diff1 + diff2)
```

**Observed (real) value:** 0.5022 (var=0.1976)
**Previous candidate values (this loop):**
  - iter 1: 0.0844 (var=0.1573) (Δ vs real -0.4178)
  - iter 2: -0.1289 (var=0.1552) (Δ vs real -0.6311)
  - iter 3: -0.1511 (var=0.1525) (Δ vs real -0.6533)
  - iter 4: -0.2578 (var=0.1494) (Δ vs real -0.7600)
  - iter 5: -0.0911 (var=0.1460) (Δ vs real -0.5933)
  - iter 6: -0.0889 (var=0.0731) (Δ vs real -0.5911)
  - iter 7: -0.0667 (var=0.1412) (Δ vs real -0.5689)
  - iter 8 (most recent): 0.0022 (var=0.1286) (Δ vs real -0.5000)
**Other theories' values on this metric (for reference):**
- pi_17: 0.5044 (var=0.2465)
- pi_15: -0.4333 (var=0.1325)
- pi_1: 0.0311 (var=0.0543)
- pi_2: -0.6311 (var=0.1123)
- pi_3: -0.0756 (var=0.0950)
- pi_4: -0.2356 (var=0.0946)
- pi_5: -0.0533 (var=0.0979)
- pi_6: -0.0089 (var=0.1100)
- pi_7: -0.0822 (var=0.0681)
- pi_8: -0.1356 (var=0.0885)
- pi_9: -0.1400 (var=0.1382)
- pi_10: -0.0733 (var=0.0847)
- pi_11: -0.1533 (var=0.1585)
- pi_12: -0.7511 (var=0.1983)
- pi_13: -0.3733 (var=0.2028)
- pi_14: -0.4133 (var=0.1264)
- pi_16: 0.1956 (var=0.2897)
- pi_18: 0.0756 (var=0.1523)
- pi_19: -0.1378 (var=0.1104)

### Experiment 32
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify which option possesses the highest validity cue (Cue 0)
    has_cue_0_a = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    has_cue_0_b = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    # Determine if the subject chose the option with Cue 0
    chose_cue_0 = ((data['response'] == 0) & has_cue_0_a) | ((data['response'] == 1) & has_cue_0_b)
    
    # Count total cues in each option to identify trial types
    n_cues_a = data['option_a_ratings'].apply(sum)
    n_cues_b = data['option_b_ratings'].apply(sum)
    
    # Mask for 1 vs 1 cue trials (Trial 1 and 6)
    mask_1 = (n_cues_a == 1) & (n_cues_b == 1)
    # Mask for 3 vs 3 cue trials (Trial 3 and 8)
    mask_3 = (n_cues_a == 3) & (n_cues_b == 3)
    
    # Calculate the proportion of times the subject chose Cue 0 in each condition
    p1 = chose_cue_0[mask_1].mean()
    p3 = chose_cue_0[mask_3].mean()
    
    # Return the difference in choice probability
    return float(p1 - p3)
```

**Observed (real) value:** -0.0200 (var=0.0414)
**Previous candidate values (this loop):**
  - iter 1: 0.0125 (var=0.0242) (Δ vs real +0.0325)
  - iter 2: 0.0075 (var=0.0387) (Δ vs real +0.0275)
  - iter 3: 0.0325 (var=0.0227) (Δ vs real +0.0525)
  - iter 4: 0.0200 (var=0.0140) (Δ vs real +0.0400)
  - iter 5: 0.0175 (var=0.0288) (Δ vs real +0.0375)
  - iter 6: 0.0392 (var=0.0200) (Δ vs real +0.0592)
  - iter 7: -0.0033 (var=0.0191) (Δ vs real +0.0167)
  - iter 8 (most recent): 0.0258 (var=0.0184) (Δ vs real +0.0458)
**Other theories' values on this metric (for reference):**
- pi_15: -0.0017 (var=0.0212)
- pi_17: 0.2200 (var=0.0285)
- pi_1: -0.0225 (var=0.0095)
- pi_2: 0.0183 (var=0.0208)
- pi_3: 0.0842 (var=0.0125)
- pi_4: -0.0075 (var=0.0178)
- pi_5: 0.0067 (var=0.0164)
- pi_6: -0.0175 (var=0.0178)
- pi_7: 0.0458 (var=0.0127)
- pi_8: 0.0000 (var=0.0215)
- pi_9: 0.0033 (var=0.0180)
- pi_10: 0.0175 (var=0.0138)
- pi_11: -0.0092 (var=0.0181)
- pi_12: -0.3775 (var=0.0298)
- pi_13: -0.1408 (var=0.0613)
- pi_14: 0.0083 (var=0.0083)
- pi_16: -0.0042 (var=0.0113)
- pi_18: 0.0133 (var=0.0113)
- pi_19: 0.0192 (var=0.0210)

### Experiment 33
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    df['sig_A'] = df['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    df['sig_B'] = df['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    df['sig'] = df['sig_A'] + "_" + df['sig_B']
    
    means = df.groupby('sig')['response'].mean()
    
    t1 = "10000_01100"
    t2 = "10010_01110"
    t3 = "10011_01111"
    
    t4 = "01100_10000"
    t5 = "01110_10010"
    t6 = "01111_10011"
    
    t7 = "11000_00110"
    t8 = "11001_00111"
    
    t9 = "10100_01010"
    t10 = "10101_01011"
    
    diff = 0.0
    if t1 in means and t2 in means: diff += abs(means[t2] - means[t1])
    if t1 in means and t3 in means: diff += abs(means[t3] - means[t1])
    if t4 in means and t5 in means: diff += abs(means[t5] - means[t4])
    if t4 in means and t6 in means: diff += abs(means[t6] - means[t4])
    if t7 in means and t8 in means: diff += abs(means[t8] - means[t7])
    if t9 in means and t10 in means: diff += abs(means[t10] - means[t9])
    
    return float(diff)
```

**Observed (real) value:** 0.1067 (var=0.2133)
**Previous candidate values (this loop):**
  - iter 1: 0.3156 (var=0.1501) (Δ vs real +0.2089)
  - iter 2: 0.1000 (var=0.1725) (Δ vs real -0.0067)
  - iter 3: 0.2311 (var=0.1476) (Δ vs real +0.1244)
  - iter 4: 0.1667 (var=0.1085) (Δ vs real +0.0600)
  - iter 5: 0.1711 (var=0.1341) (Δ vs real +0.0644)
  - iter 6: 0.0644 (var=0.1665) (Δ vs real -0.0422)
  - iter 7: 0.1711 (var=0.1419) (Δ vs real +0.0644)
  - iter 8 (most recent): 0.2044 (var=0.1116) (Δ vs real +0.0978)
**Other theories' values on this metric (for reference):**
- pi_17: 0.6000 (var=0.3276)
- pi_18: 0.1444 (var=0.1538)
- pi_1: 0.0822 (var=0.1247)
- pi_2: 0.1622 (var=0.1036)
- pi_3: 0.1067 (var=0.1028)
- pi_4: 0.1311 (var=0.1591)
- pi_5: 0.0889 (var=0.1205)
- pi_6: 0.2244 (var=0.0804)
- pi_7: 0.1600 (var=0.1419)
- pi_8: 0.1133 (var=0.1893)
- pi_9: 0.1222 (var=0.1497)
- pi_10: 0.1444 (var=0.1698)
- pi_11: 0.0956 (var=0.1451)
- pi_12: 0.1267 (var=0.1652)
- pi_13: 0.0756 (var=0.1101)
- pi_14: 0.0533 (var=0.1180)
- pi_15: 0.1067 (var=0.1577)
- pi_16: 0.1267 (var=0.1296)
- pi_19: 0.1067 (var=0.1936)

### Experiment 34
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Create string representation for A and B options to identify trials
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Map rows to trial types
    def get_trial(row):
        a, b = row['A_str'], row['B_str']
        if a == '10000' and b == '01100': return 1
        if a == '10010' and b == '01110': return 2
        if a == '10011' and b == '01111': return 3
        if a == '01000' and b == '00110': return 4
        if a == '11000' and b == '10110': return 5
        if a == '11001' and b == '10111': return 6
        if a == '00100' and b == '00011': return 7
        if a == '01100' and b == '01011': return 8
        return 0

    data['trial'] = data.apply(get_trial, axis=1)
    
    # Calculate P(choose A) for each trial. Response 0 means A was chosen.
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # Group by subject and trial
    subj_trial = data[data['trial'] > 0].groupby(['subject_id', 'trial'])['chose_A'].mean().unstack(fill_value=np.nan)
    
    # For each subject, compute the absolute difference in choice probabilities 
    # between the base trial and the trial with the most added shared features.
    diffs = []
    for subj in subj_trial.index:
        p = subj_trial.loc[subj]
        d1 = abs(p.get(1, np.nan) - p.get(3, np.nan))
        d2 = abs(p.get(4, np.nan) - p.get(6, np.nan))
        d3 = abs(p.get(7, np.nan) - p.get(8, np.nan))
        
        valid_d = [d for d in [d1, d2, d3] if pd.notna(d)]
        if valid_d:
            diffs.append(np.mean(valid_d))
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
```

**Observed (real) value:** 0.1022 (var=0.0050)
**Previous candidate values (this loop):**
  - iter 1: 0.1522 (var=0.0031) (Δ vs real +0.0500)
  - iter 2: 0.1761 (var=0.0052) (Δ vs real +0.0739)
  - iter 3: 0.1678 (var=0.0055) (Δ vs real +0.0656)
  - iter 4: 0.1333 (var=0.0048) (Δ vs real +0.0311)
  - iter 5: 0.1789 (var=0.0058) (Δ vs real +0.0767)
  - iter 6: 0.1567 (var=0.0037) (Δ vs real +0.0544)
  - iter 7: 0.1406 (var=0.0049) (Δ vs real +0.0383)
  - iter 8 (most recent): 0.1556 (var=0.0067) (Δ vs real +0.0533)
**Other theories' values on this metric (for reference):**
- pi_18: 0.1483 (var=0.0064)
- pi_17: 0.2783 (var=0.0119)
- pi_1: 0.1111 (var=0.0060)
- pi_2: 0.0928 (var=0.0030)
- pi_3: 0.1522 (var=0.0041)
- pi_4: 0.1378 (var=0.0053)
- pi_5: 0.1322 (var=0.0040)
- pi_6: 0.1667 (var=0.0047)
- pi_7: 0.1250 (var=0.0030)
- pi_8: 0.1083 (var=0.0041)
- pi_9: 0.1333 (var=0.0032)
- pi_10: 0.1472 (var=0.0043)
- pi_11: 0.1583 (var=0.0048)
- pi_12: 0.0850 (var=0.0034)
- pi_13: 0.1222 (var=0.0060)
- pi_14: 0.1167 (var=0.0045)
- pi_15: 0.1356 (var=0.0055)
- pi_16: 0.0883 (var=0.0025)
- pi_19: 0.1422 (var=0.0046)

### Experiment 35
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))
    
    t1 = (a_str == '10000') & (b_str == '01000')
    t4 = (a_str == '10111') & (b_str == '01000')
    
    if t1.sum() == 0 or t4.sum() == 0:
        return 0.0
        
    p_a_t1 = 1.0 - data.loc[t1, 'response'].mean()
    p_a_t4 = 1.0 - data.loc[t4, 'response'].mean()
    
    return float(p_a_t4 - p_a_t1)
```

**Observed (real) value:** 0.0640 (var=0.0143)
**Previous candidate values (this loop):**
  - iter 1: 0.0060 (var=0.0598) (Δ vs real -0.0580)
  - iter 2: 0.1320 (var=0.0798) (Δ vs real +0.0680)
  - iter 3: 0.1200 (var=0.0416) (Δ vs real +0.0560)
  - iter 4: 0.1400 (var=0.0472) (Δ vs real +0.0760)
  - iter 5: 0.0220 (var=0.0461) (Δ vs real -0.0420)
  - iter 6: 0.0460 (var=0.0289) (Δ vs real -0.0180)
  - iter 7: -0.0160 (var=0.0509) (Δ vs real -0.0800)
  - iter 8 (most recent): 0.0480 (var=0.0369) (Δ vs real -0.0160)
**Other theories' values on this metric (for reference):**
- pi_17: -0.3560 (var=0.0849)
- pi_19: 0.0720 (var=0.0508)
- pi_1: 0.0060 (var=0.0266)
- pi_2: 0.3300 (var=0.0249)
- pi_3: 0.0480 (var=0.0213)
- pi_4: 0.1780 (var=0.0253)
- pi_5: 0.0140 (var=0.0416)
- pi_6: 0.0260 (var=0.0599)
- pi_7: 0.0060 (var=0.0374)
- pi_8: 0.3260 (var=0.0407)
- pi_9: 0.1800 (var=0.0680)
- pi_10: 0.0260 (var=0.0331)
- pi_11: 0.1960 (var=0.0468)
- pi_12: 0.6800 (var=0.0984)
- pi_13: 0.2760 (var=0.1470)
- pi_14: 0.3680 (var=0.1286)
- pi_15: 0.2800 (var=0.0328)
- pi_16: 0.0220 (var=0.2577)
- pi_18: -0.0460 (var=0.0593)

### Experiment 36
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Response 0 means choosing Option A, so 1.0 - response is the indicator for choosing A
    p_a = 1.0 - data['response']
    
    m1 = p_a[(a_str == '10000') & (b_str == '01000')].mean()
    m2 = p_a[(a_str == '10100') & (b_str == '01000')].mean()
    m3 = p_a[(a_str == '10110') & (b_str == '01000')].mean()
    m4 = p_a[(a_str == '10111') & (b_str == '01000')].mean()
    
    # Fallback to 0.5 if any trial type is completely missing (should not happen in this design)
    m1 = 0.5 if np.isnan(m1) else m1
    m2 = 0.5 if np.isnan(m2) else m2
    m3 = 0.5 if np.isnan(m3) else m3
    m4 = 0.5 if np.isnan(m4) else m4
    
    # Compare the later trials (more unique features) to the earlier trials
    return float((m4 + m3) - (m2 + m1))
```

**Observed (real) value:** -0.0320 (var=0.0398)
**Previous candidate values (this loop):**
  - iter 1: 0.0720 (var=0.1012) (Δ vs real +0.1040)
  - iter 2: 0.0580 (var=0.1200) (Δ vs real +0.0900)
  - iter 3: 0.1520 (var=0.1457) (Δ vs real +0.1840)
  - iter 4: 0.1680 (var=0.0690) (Δ vs real +0.2000)
  - iter 5: 0.0880 (var=0.1115) (Δ vs real +0.1200)
  - iter 6: 0.0160 (var=0.1241) (Δ vs real +0.0480)
  - iter 7: 0.0380 (var=0.0920) (Δ vs real +0.0700)
  - iter 8 (most recent): 0.1440 (var=0.1093) (Δ vs real +0.1760)
**Other theories' values on this metric (for reference):**
- pi_19: 0.0760 (var=0.0914)
- pi_17: -0.3280 (var=0.1464)
- pi_1: 0.0800 (var=0.0608)
- pi_2: 0.3740 (var=0.0687)
- pi_3: 0.0660 (var=0.0522)
- pi_4: 0.1600 (var=0.0336)
- pi_5: 0.0760 (var=0.0730)
- pi_6: -0.0420 (var=0.1076)
- pi_7: 0.0840 (var=0.0769)
- pi_8: 0.3600 (var=0.1004)
- pi_9: 0.1980 (var=0.1002)
- pi_10: 0.0180 (var=0.0815)
- pi_11: 0.1720 (var=0.1156)
- pi_12: 0.6960 (var=0.1520)
- pi_13: 0.4280 (var=0.1272)
- pi_14: 0.4480 (var=0.1745)
- pi_15: 0.1760 (var=0.0466)
- pi_16: 0.0080 (var=0.6215)
- pi_18: -0.1040 (var=0.1324)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Relative Evidence Accumulation with Conflict Discounting: Decision-makers evaluate options by accumulating evidence from their features, but this evidence is dynamically discounted by the presence of conflicting features in the competing option (cross-option inhibition). This avoids perverse self-dilution (adding cues to an option never hurts it) while capturing conflict-driven non-linearities and context effects through mutual suppression.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    alpha = float(parameters["alpha"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of cue validities (centered at chance 0.5)
    w = np.maximum(0.0, val - 0.5) ** alpha
    
    # Cross-option conflict: number of features present in the competitor but absent in the focal option
    conflict_a = np.sum(b * (1.0 - a))
    conflict_b = np.sum(a * (1.0 - b))
    
    # Evidence accumulation with exponential conflict discounting
    ev_a = np.sum(a * w) * np.exp(-gamma * conflict_a)
    ev_b = np.sum(b * w) * np.exp(-gamma * conflict_b)
    
    # Softmax choice rule
    z = beta * np.array([ev_a, ev_b])
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


`parameters`:
- alpha: [0.0, 5.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's recommendation, this model implements 'Relative Evidence Accumulation with Conflict Discounting'. Instead of normalizing by the total number of features within an option (which causes perverse self-dilution in Exp 1 and 2), the evidence for an option is discounted by the number of conflicting features in the competing option (cross-option inhibition). This ensures that adding supporting cues strictly benefits an option, while the mutual suppression mechanism naturally captures the conflict-driven non-linearities and context effects required for Experiments 13, 14, 31, 32, and 33.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory 'Relative Evidence Accumulation with Conflict Discounting' was accepted by the gate (achieving a new best aggregate loss), but its specific predictions reveal a major calibration issue: the model predicts ~0.5 (random guessing) across many key experiments (e.g., Exps 1, 2, 3, 4, 13, 14, 17, 18). This is a classic symptom of evidence vanishing to zero. Because the parameter `gamma` can reach up to 10.0, the exponential discounting term `np.exp(-gamma * conflict)` frequently evaluates to near zero for both options. Consequently, `ev_a` and `ev_b` become ~0, making the softmax logits roughly equal and forcing the choice probabilities to 0.5 regardless of `beta`.
Rationale: The model's core mechanism is sound and successfully improved the aggregate loss, but the exponential conflict discounting is too aggressive and crushes evidence to zero, causing the model to default to 0.5 probability on many trials. To fix this while staying strictly within the prescribed family, adjust the conflict discounting mechanism: either drastically reduce the range of `gamma` (e.g., [0.0, 1.0] or [0.0, 2.0]), or replace the exponential decay with a softer divisive discounting (e.g., `1.0 / (1.0 + gamma * conflict_a)`). This will preserve the relative evidence accumulation while preventing the logits from vanishing.

**Outcome of this advice:** iter 1 candidate loss=0.3594 -> iter 2 candidate loss=0.3972 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The attempt to fix the random guessing (0.5) predictions by replacing exponential discounting with divisive discounting was rejected by the gate, indicating it worsened the overall fit. The root cause of the vanishing logits in the accepted base is likely not just the conflict discounting, but the cue weighting itself: `w = np.maximum(0.0, val - 0.5) ** alpha` produces microscopic values when `alpha` is large (e.g., 0.2^5 = 0.00032). When these tiny weights are further discounted, the total evidence becomes too small for `beta` to meaningfully amplify, forcing the softmax to output 0.5.
Rationale: Since the divisive discounting was rejected, revert to the accepted base's exponential discounting mechanism. To fix the vanishing logits problem, normalize the cue weights so they sum to 1 (`w = w / np.sum(w)`). This ensures the accumulated evidence remains on a stable scale. Additionally, reduce the upper bound of `alpha` to 3.0 and increase the upper bound of `beta` to 50.0 so the model has enough gain to make deterministic predictions when needed.

**Outcome of this advice:** iter 2 candidate loss=0.3972 -> iter 3 candidate loss=0.4083 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The attempt to normalize cue weights and alter the bounds of alpha and beta (Iter 3) was rejected by the gate, as was the attempt to use divisive discounting (Iter 2). This indicates that the original cue weighting and exponential discounting equations from the accepted base (Iter 1) are structurally superior, but they are suffering from a parameterization issue. Specifically, the base model still outputs ~0.5 (random guessing) on many critical experiments because the parameter `gamma` can reach up to 10.0. When `gamma` is large, `np.exp(-gamma * conflict)` evaluates to effectively zero, crushing the accumulated evidence `ev_a` and `ev_b` to zero. This causes the softmax logits to vanish, forcing choice probabilities to 0.5 regardless of the beta parameter.
Rationale: Since both structural deviations from the accepted base (divisive discounting and weight normalization) were rejected by the gate, we must revert to the exact mathematical formulation of the accepted base (Iter 1). To solve the vanishing evidence problem, simply restrict the upper bound of `gamma` to a much smaller value, such as 1.5 or 2.0 (i.e., `gamma: [0.0, 2.0]`). This will allow the exponential discounting to act as a soft penalty rather than an absolute zero-multiplier, preserving the cross-option inhibition mechanism while allowing `beta` to effectively scale the remaining evidence differences into deterministic predictions.

**Outcome of this advice:** iter 3 candidate loss=0.4083 -> iter 4 candidate loss=0.4157 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The attempt to restrict the `gamma` range to `[0.0, 2.0]` (Iteration 4) was rejected by the accept gate, just as the previous attempts at divisive discounting and weight normalization were. This indicates that the original mathematical formulation and parameter ranges of the Iteration 1 base are structurally optimal within this family, but the model still struggles with outputting ~0.5 (random guessing) on many critical experiments. The root cause is likely that the combination of `(val - 0.5) ** alpha` (which yields inherently small values) and `np.exp(-gamma * conflict)` (which further shrinks them) produces evidence differences that are simply too small for the current `beta` parameter to amplify. When evidence differences are microscopic, a `beta` capped at 20.0 is insufficient to push the softmax logits away from 0, resulting in 50/50 predictions.
Rationale: Since structural changes (divisive discounting, normalization) and restricting `gamma` bounds have all been rejected, you must revert exactly to the Iteration 1 base formulation and parameter ranges (including `gamma: [0.0, 10.0]`). To solve the vanishing evidence problem, you should solely increase the upper bound of the inverse temperature `beta` to a much larger value (e.g., `[0.1, 100.0]` or even `[0.1, 200.0]`). This will allow the model to adequately scale the microscopic evidence differences produced by the exponential discounting into deterministic choice probabilities, without altering the proven structural equations of the accepted base.

**Outcome of this advice:** iter 4 candidate loss=0.4157 -> iter 5 candidate loss=0.4090 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The attempt to expand beta (Iteration 5) was rejected by the gate, just like the previous attempts to restrict gamma or normalize weights. The consistent rejection of these scaling/temperature fixes reveals a deeper structural flaw: the model's predictions are defaulting to 0.5 because when it does make deterministic predictions, it often predicts the *wrong* option, incurring massive log-loss penalties. For example, in Exp 1 (TTB vs Tallying), Option A has 1 high-validity cue and Option B has 3 low-validity cues. The current unweighted conflict count (`conflict_a = np.sum(b * (1.0 - a))`) gives Option A a massive conflict penalty of 3, and Option B a penalty of 1. This causes the model to strongly favor the Tallying option (B), whereas humans favor the TTB option (A). Increasing beta or restricting gamma simply amplifies this incorrect prediction, which is why the gate keeps rejecting your changes.
Rationale: To fix the directional predictions without leaving the prescribed mechanism family, the cross-option conflict must be weighted by the cue validities. A low-validity cue should not exert the same inhibitory force as a high-validity cue. Revert to the Iteration 1 base parameter ranges, but modify the conflict equations to be weighted by `w`: `conflict_a = np.sum(b * (1.0 - a) * w)` and `conflict_b = np.sum(a * (1.0 - b) * w)`. This ensures that high-validity cues drive both evidence accumulation and cross-option inhibition, allowing the model to correctly predict TTB-like behavior in conflict experiments.

**Outcome of this advice:** iter 5 candidate loss=0.4090 -> iter 6 candidate loss=0.4070 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The attempt to weight the cross-option conflict by cue validities (Iteration 6) was rejected by the accept gate, just as previous attempts to adjust parameter bounds or swap the discounting function were. The accepted Iteration 1 base remains the best-performing model, but it still suffers from predicting ~0.5 (random guessing) on many trials. The persistent failure of scaling fixes (beta, gamma) points to a numerical precision issue in the initial cue weighting step.
Rationale: Since all structural deviations and bound adjustments have been rejected, we must revert exactly to the Iteration 1 base (including the unweighted conflict count `conflict_a = np.sum(b * (1.0 - a))`). The root cause of the vanishing evidence is the equation `w = np.maximum(0.0, val - 0.5) ** alpha`. Because `val - 0.5` is at most 0.5, raising it to a power up to 5.0 shrinks the maximum possible weight to 0.03125. When this already tiny evidence is exponentially discounted, it becomes too small for `beta` to amplify, causing the softmax to return 0.5. To fix this without changing the mechanism or bounds, simply scale the centered validities to a maximum of 1.0 before applying the exponent: `w = (np.maximum(0.0, val - 0.5) * 2.0) ** alpha`. This mathematically preserves the Iteration 1 structure while preventing the raw evidence from vanishing.

**Outcome of this advice:** iter 6 candidate loss=0.4070 -> iter 7 candidate loss=0.3886 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The attempt to scale the centered validities (Iteration 6) was rejected by the accept gate, just as previous attempts to adjust parameter bounds, normalize weights, and swap the discounting function were. The accepted Iteration 1 base remains the best-performing model, but it still predicts ~0.5 (random guessing) on many trials. The persistent failure of scaling and temperature fixes indicates that the model is actively 'hedging' to avoid massive log-loss penalties from predicting the wrong option. This implies that the core directional signal of the model is misaligned with human behavior on certain key trials.
Rationale: Since all structural deviations, bound adjustments, and weight scalings have been rejected, we must revert to the Iteration 1 base's exponential discounting mechanism. The root cause of the hedging behavior is likely the definition of 'conflict' as unique features (`np.sum(b * (1.0 - a))`). This specific definition creates pathological edge cases where an option is heavily penalized incorrectly, forcing the model to flatten its predictions to 0.5 to survive. To fix this while staying strictly in-family, redefine the cross-option conflict as simply the *total* number of features in the competing option, i.e., `conflict_a = np.sum(b)` and `conflict_b = np.sum(a)`. This provides a smoother, more stable cross-option inhibition that scales linearly with the competitor's raw size, rather than its unique differences, preventing the severe miscalibrations that cause the hedging.

**Outcome of this advice:** iter 7 candidate loss=0.3886 -> iter 8 candidate loss=0.3825 -> the gate marked it REJECTED.

### Iteration 8 (most recent — address this)
Verdict: regenerate
Interpretation: The attempt to redefine conflict as the total number of competitor features (Iteration 8) was rejected by the accept gate, just as previous structural and parameter adjustments were. The accepted Iteration 1 base remains the best-performing model, but it still struggles with 0.5 (random guessing) predictions on many trials. The persistent failure of option-level conflict discounting (whether unique or total features) indicates a deeper structural issue with how the inhibition is applied. By discounting the *total* evidence of an option (`np.exp(-gamma * conflict_a)`), the model uniformly crushes all evidence, including the highly valid unique cues that should drive the choice. This leads to vanishing logits and 50/50 predictions.
Rationale: Since option-level discounting has consistently failed or caused hedging, the cross-option inhibition must be applied at the *feature level*. Revert to the Iteration 1 parameter ranges, but modify the evidence accumulation so that each feature's weight is individually discounted by the presence of that specific feature in the competing option: `ev_a = np.sum(a * w * np.exp(-gamma * b))` and `ev_b = np.sum(b * w * np.exp(-gamma * a))`. This ensures that shared features are heavily discounted (effectively ignored, matching empirical behavior), while unique features retain their full weight to drive deterministic choices, eliminating the vanishing evidence problem without leaving the prescribed mechanism family.

**Outcome of this advice:** iter 8 candidate loss=0.3825 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
```json
{
  "theory": "Relative Evidence Accumulation with Feature-Level Conflict Discounting",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    alpha = float(parameters[\"alpha\"])\n    gamma = float(parameters[\"gamma\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Non-linear weighting of cue validities (centered at chance 0.5)\n    w = np.maximum(0.0, val - 0.5) ** alpha\n    \n    # Feature-level evidence accumulation with cross-option conflict discounting\n    ev_a = np.sum(a * w * np.exp(-gamma * b))\n    ev_b = np.sum(b * w * np.exp(-gamma * a))\n    \n    # Softmax choice rule\n    z = beta * np.array([ev_a, ev_b])\n    z = z - np.max(z)\n    p = np.exp(z)\n    p = p / np.sum(p)\n    \n    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return np.random.choice(len(probs), p=probs)\n",
  "parameters": {
    "alpha": "[0.0, 5.0]",
    "gamma": "[0.0, 10.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Applying the critic's Iteration 8 feedback: previous attempts to use option-level conflict discounting failed because they uniformly crushed all evidence, including the highly valid unique cues that should drive deterministic choices, resulting in 50/50 random guessing. By moving the exponential discounting to the feature level, each feature's weight is individually discounted only if that specific feature is also present in the competing option. This correctly marginalizes shared features while allowing unique features to retain their full weight and drive the choice, solving the vanishing evidence problem while strictly maintaining the prescribed mechanism family."
}
```
```

## Usage

```json
{
  "prompt_token_count": 65229,
  "candidates_token_count": 623,
  "total_token_count": 66786
}
```
