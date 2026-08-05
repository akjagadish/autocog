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
- THEORY 1 = `pi_14`
- THEORY 2 = `pi_15`
- The recommendation below acts on THEORY 1 (= `pi_14`).

Propose a brand-new theory that moves away from simple tallying or static mixtures. Consider a Sequential Evidence Accumulation model with dynamic thresholds, or a Lexicographic model with context-dependent cue weighting (where the weight of the first cue depends on the dispersion of the remaining cues). The new theory must account for the extreme dominance of the first cue in certain massive-disadvantage scenarios (Exp 1) while also explaining the severe drop in TTB adherence in other baseline contexts (Exp 2).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_8` (overall score: 0.447)

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


---

### `pi_12` (overall score: 0.440)

**Description**
Context-Dependent Strategy Selection with Reverse TTB (Bottom-Up Heuristic): Decision-makers adapt their decision strategy based on the overall equality of evidence. When the total number of positive cues is unequal, they rely on a mixture of Tallying and standard Take-The-Best (evaluating from most to least valid). However, when options present a tied sum of positive cues, it triggers a 'Bottom-Up' or 'Reverse TTB' heuristic. In this mode, subjects evaluate cues starting from the least valid to the most valid, resolving the tie by favoring the option that possesses discriminating lower-validity cues. This naturally explains the paradoxical avoidance of the highest-validity cue in tied-sum scenarios, as the highest-validity cue is evaluated last and is thus overridden by differences in the lower-validity cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Orderings for Top-Down (TTB) and Bottom-Up (Reverse TTB)
    cue_order_top_down = np.argsort(-val, kind="stable").tolist()
    cue_order_bottom_up = np.argsort(val, kind="stable").tolist()
    
    # 1. Standard TTB (Top-Down)
    winner_ttb = 0.5
    for j in cue_order_top_down:
        if a[j] > b[j]:
            winner_ttb = 1.0
            break
        elif b[j] > a[j]:
            winner_ttb = 0.0
            break
            
    # 2. Bottom-Up TTB (Reverse TTB)
    winner_bottom_up = 0.5
    for j in cue_order_bottom_up:
        if a[j] > b[j]:
            winner_bottom_up = 1.0
            break
        elif b[j] > a[j]:
            winner_bottom_up = 0.0
            break
            
    # 3. Tallying
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        winner_tally = 1.0
    elif b_wins > a_wins:
        winner_tally = 0.0
    else:
        winner_tally = 0.5
        
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Strategy mixture logits
    z_strats = np.array([
        float(parameters["w_ttb"]),
        float(parameters["w_tally"]),
        float(parameters["w_bottom_up"])
    ])
    
    # Context-dependent boosts
    if sum_a == sum_b:
        z_strats[2] += float(parameters["gamma_tie"])  # Boost Bottom-Up when sums are tied
    else:
        z_strats[0] += float(parameters["gamma_diff"]) # Boost Standard TTB when sums differ
        
    e_strats = np.exp(z_strats - np.max(z_strats))
    p_strats = e_strats / np.sum(e_strats)
    
    # Expected probability of choosing Option A
    p_a = (p_strats[0] * winner_ttb + 
           p_strats[1] * winner_tally + 
           p_strats[2] * winner_bottom_up)
    
    # Convert to choice probabilities with temperature
    beta = float(parameters["beta"])
    z = beta * np.array([p_a, 1.0 - p_a])
    e = np.exp(z - np.max(z))
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- w_ttb: [0.0, 5.0]
- w_tally: [0.0, 5.0]
- w_bottom_up: [0.0, 5.0]
- gamma_tie: [0.0, 10.0]
- gamma_diff: [0.0, 10.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.2]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6508 (var=0.0505) vs this=0.7571 (var=0.0819)
- Experiment 2: real=0.3011 (var=0.0238) vs this=0.2317 (var=0.0698)
- Experiment 3: real=0.6100 (var=0.0044) vs this=0.8083 (var=0.0358)
- Experiment 4: real=0.6383 (var=0.0300) vs this=0.7473 (var=0.0796)
- Experiment 5: real=0.0825 (var=0.1837) vs this=-0.1137 (var=0.1323)
- Experiment 6: real=-0.0167 (var=0.0028) vs this=0.0089 (var=0.0012)
- Experiment 7: real=-0.2050 (var=0.2002) vs this=-0.0225 (var=0.0382)
- Experiment 8: real=0.9324 (var=0.1377) vs this=1.3804 (var=0.0691)
- Experiment 9: real=1.6547 (var=0.1361) vs this=0.8695 (var=0.0941)
- Experiment 10: real=0.0885 (var=0.0487) vs this=-0.1220 (var=0.2496)
- Experiment 11: real=-0.4292 (var=0.0555) vs this=0.2415 (var=0.0382)
- Experiment 12: real=-0.6711 (var=0.0499) vs this=0.0544 (var=0.0251)
- Experiment 13: real=0.2644 (var=0.0112) vs this=0.2139 (var=0.0556)
- Experiment 14: real=0.1350 (var=0.0065) vs this=0.1625 (var=0.0471)
- Experiment 15: real=0.8031 (var=0.0244) vs this=0.7992 (var=0.0527)
- Experiment 16: real=0.2611 (var=0.0294) vs this=0.3395 (var=0.0186)
- Experiment 17: real=0.2825 (var=0.0159) vs this=0.4238 (var=0.0184)
- Experiment 18: real=0.3458 (var=0.0444) vs this=0.5196 (var=0.0047)
- Experiment 19: real=0.1758 (var=0.0110) vs this=0.2053 (var=0.0502)
- Experiment 20: real=0.2025 (var=0.0829) vs this=-0.0038 (var=0.0152)
- Experiment 21: real=0.4050 (var=0.0313) vs this=0.8000 (var=0.0345)
- Experiment 22: real=0.5684 (var=0.0814) vs this=0.8284 (var=0.0469)
- Experiment 23: real=0.3133 (var=0.0161) vs this=0.3850 (var=0.0106)
- Experiment 24: real=0.3702 (var=0.0075) vs this=0.3611 (var=0.0193)
- Experiment 25: real=-0.0010 (var=0.0000) vs this=-0.0004 (var=0.0000)
- Experiment 26: real=0.0154 (var=0.0544) vs this=-0.8031 (var=0.1420)
- Experiment 27: real=0.8320 (var=0.0112) vs this=0.8245 (var=0.0590)
- Experiment 28: real=0.1467 (var=0.0053) vs this=0.8721 (var=0.0114)


---

### `pi_6` (overall score: 0.410)

**Description**
Weighted Additive with Diminishing Returns (WADD-DR): Decision-makers integrate all available discriminating cues but apply a sub-additive (concave) transformation to the accumulated evidence. Cue validities are first scaled non-linearly to reflect subjective weighting. Then, the total accumulated evidence for each option undergoes a concave transformation before being converted to choice probabilities. This naturally accounts for the dilution effect when multiple weaker cues are added, pulling choice probabilities toward 0.5, while maintaining sensitivity to the overall balance of evidence.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities: subtract 0.5 so a random cue provides 0 evidence, then apply non-linear scaling
    w = np.maximum(0.0, val - 0.5) ** gamma
    
    # Accumulate evidence for each option based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Apply sub-additive (concave) transformation to accumulated evidence
    # alpha < 1 yields diminishing returns for additional evidence
    ev_a_trans = (ev_a + 1e-9) ** alpha
    ev_b_trans = (ev_b + 1e-9) ** alpha
    
    # Convert transformed evidence to choice probabilities via softmax
    scores = np.array([ev_a_trans, ev_b_trans])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- gamma: [0.1, 30.0]
- alpha: [0.01, 1.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6508 (var=0.0505) vs this=0.5054 (var=0.0097)
- Experiment 2: real=0.3011 (var=0.0238) vs this=0.4697 (var=0.0089)
- Experiment 3: real=0.6100 (var=0.0044) vs this=0.5006 (var=0.0039)
- Experiment 4: real=0.6383 (var=0.0300) vs this=0.5340 (var=0.0043)
- Experiment 5: real=0.0825 (var=0.1837) vs this=0.0163 (var=0.0737)
- Experiment 6: real=-0.0167 (var=0.0028) vs this=0.0361 (var=0.0279)
- Experiment 7: real=-0.2050 (var=0.2002) vs this=0.0575 (var=0.0679)
- Experiment 8: real=0.9324 (var=0.1377) vs this=1.5333 (var=0.0268)
- Experiment 9: real=1.6547 (var=0.1361) vs this=1.0674 (var=0.0256)
- Experiment 10: real=0.0885 (var=0.0487) vs this=0.3311 (var=0.1888)
- Experiment 11: real=-0.4292 (var=0.0555) vs this=0.0323 (var=0.0289)
- Experiment 12: real=-0.6711 (var=0.0499) vs this=0.0478 (var=0.0311)
- Experiment 13: real=0.2644 (var=0.0112) vs this=0.5544 (var=0.0097)
- Experiment 14: real=0.1350 (var=0.0065) vs this=0.5767 (var=0.0130)
- Experiment 15: real=0.8031 (var=0.0244) vs this=0.5285 (var=0.0127)
- Experiment 16: real=0.2611 (var=0.0294) vs this=0.0684 (var=0.0033)
- Experiment 17: real=0.2825 (var=0.0159) vs this=0.5333 (var=0.0069)
- Experiment 18: real=0.3458 (var=0.0444) vs this=0.5546 (var=0.0090)
- Experiment 19: real=0.1758 (var=0.0110) vs this=0.5205 (var=0.0062)
- Experiment 20: real=0.2025 (var=0.0829) vs this=-0.0600 (var=0.0261)
- Experiment 21: real=0.4050 (var=0.0313) vs this=0.4906 (var=0.0089)
- Experiment 22: real=0.5684 (var=0.0814) vs this=0.4611 (var=0.0132)
- Experiment 23: real=0.3133 (var=0.0161) vs this=0.0675 (var=0.0028)
- Experiment 24: real=0.3702 (var=0.0075) vs this=0.1372 (var=0.0048)
- Experiment 25: real=-0.0010 (var=0.0000) vs this=-0.0005 (var=0.0002)
- Experiment 26: real=0.0154 (var=0.0544) vs this=0.0815 (var=0.0832)
- Experiment 27: real=0.8320 (var=0.0112) vs this=0.4670 (var=0.0248)
- Experiment 28: real=0.1467 (var=0.0053) vs this=0.5813 (var=0.0238)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4818 -> ACCEPTED
- iter 2: loss=0.4089 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.4089 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.2142 (var=0.0791) (Δ vs real -0.4367)
  - iter 2 (most recent): 0.4958 (var=0.1341) (Δ vs real -0.1550)
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
  - iter 1: 0.8286 (var=0.0637) (Δ vs real +0.5275)
  - iter 2 (most recent): 0.5694 (var=0.1198) (Δ vs real +0.2683)
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
  - iter 1: 0.3175 (var=0.0451) (Δ vs real -0.2925)
  - iter 2 (most recent): 0.5746 (var=0.0874) (Δ vs real -0.0354)
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
  - iter 1: 0.1152 (var=0.0430) (Δ vs real -0.5231)
  - iter 2 (most recent): 0.5775 (var=0.1369) (Δ vs real -0.0608)
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
  - iter 1: 0.0012 (var=0.0235) (Δ vs real -0.0813)
  - iter 2 (most recent): 0.0050 (var=0.0168) (Δ vs real -0.0775)
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
  - iter 1: 0.0072 (var=0.0015) (Δ vs real +0.0239)
  - iter 2 (most recent): 0.0067 (var=0.0041) (Δ vs real +0.0233)
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
  - iter 1: 0.0125 (var=0.0127) (Δ vs real +0.2175)
  - iter 2 (most recent): 0.0150 (var=0.0220) (Δ vs real +0.2200)
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
  - iter 1: 1.5987 (var=0.4384) (Δ vs real +0.6662)
  - iter 2 (most recent): 1.8526 (var=0.2707) (Δ vs real +0.9201)
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
  - iter 1: 0.2453 (var=0.1101) (Δ vs real -1.4095)
  - iter 2 (most recent): 0.6611 (var=0.1408) (Δ vs real -0.9937)
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
  - iter 1: 0.0092 (var=0.1276) (Δ vs real -0.0793)
  - iter 2 (most recent): -0.1198 (var=0.2149) (Δ vs real -0.2083)
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
  - iter 1: 0.7015 (var=0.0667) (Δ vs real +1.1308)
  - iter 2 (most recent): 0.3531 (var=0.1409) (Δ vs real +0.7823)
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
  - iter 1: 0.8856 (var=0.0284) (Δ vs real +1.5567)
  - iter 2 (most recent): 0.3689 (var=0.1411) (Δ vs real +1.0400)
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
  - iter 1: 0.4839 (var=0.1743) (Δ vs real +0.2194)
  - iter 2 (most recent): 0.6333 (var=0.1239) (Δ vs real +0.3689)
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
  - iter 1: 0.4708 (var=0.1598) (Δ vs real +0.3358)
  - iter 2 (most recent): 0.6075 (var=0.1197) (Δ vs real +0.4725)
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
  - iter 1: 0.0908 (var=0.0139) (Δ vs real -0.7123)
  - iter 2 (most recent): 0.4631 (var=0.1601) (Δ vs real -0.3400)
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
  - iter 1: 0.3700 (var=0.0177) (Δ vs real +0.1089)
  - iter 2 (most recent): 0.3632 (var=0.0109) (Δ vs real +0.1021)
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
  - iter 1: 0.5367 (var=0.1772) (Δ vs real +0.2542)
  - iter 2 (most recent): 0.6354 (var=0.1362) (Δ vs real +0.3529)
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
  - iter 1: 0.3971 (var=0.1501) (Δ vs real +0.0513)
  - iter 2 (most recent): 0.5933 (var=0.1295) (Δ vs real +0.2475)
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
  - iter 1: 0.4721 (var=0.1555) (Δ vs real +0.2963)
  - iter 2 (most recent): 0.7074 (var=0.0903) (Δ vs real +0.5316)
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
  - iter 1: -0.0913 (var=0.0560) (Δ vs real -0.2938)
  - iter 2 (most recent): -0.0925 (var=0.0643) (Δ vs real -0.2950)
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
  - iter 1: 0.5006 (var=0.0036) (Δ vs real +0.0956)
  - iter 2 (most recent): 0.4894 (var=0.0026) (Δ vs real +0.0844)
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
  - iter 1: 0.4842 (var=0.1613) (Δ vs real -0.0842)
  - iter 2 (most recent): 0.3763 (var=0.1198) (Δ vs real -0.1921)
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
  - iter 1: 0.3646 (var=0.0222) (Δ vs real +0.0513)
  - iter 2 (most recent): 0.3504 (var=0.0150) (Δ vs real +0.0371)
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
  - iter 1: 0.3996 (var=0.0142) (Δ vs real +0.0295)
  - iter 2 (most recent): 0.3393 (var=0.0134) (Δ vs real -0.0309)
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
  - iter 1: 0.0001 (var=0.0000) (Δ vs real +0.0011)
  - iter 2 (most recent): -0.0008 (var=0.0000) (Δ vs real +0.0002)
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
  - iter 1: -0.0231 (var=0.0424) (Δ vs real -0.0385)
  - iter 2 (most recent): 0.0123 (var=0.0335) (Δ vs real -0.0031)
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
  - iter 1: 0.0610 (var=0.0036) (Δ vs real -0.7710)
  - iter 2 (most recent): 0.4865 (var=0.1591) (Δ vs real -0.3455)
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
  - iter 1: 0.9383 (var=0.0025) (Δ vs real +0.7917)
  - iter 2 (most recent): 0.8708 (var=0.0073) (Δ vs real +0.7242)
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

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Context-Dependent Lexicographic Accumulator (CDLA) with Bidirectional Dispersion Effect: Decision-makers process information in a lexicographic manner, but the weight assigned to the first discriminating cue is contextually modulated by the dispersion of the remaining cues. When the remaining cues strongly contradict the first cue, the subjective weight of the first cue can either decay or amplify depending on the individual's specific sensitivity (dispersion effect). This allows the model to empirically learn the correct direction of the context modulation, capturing both the extreme dominance of the first cue in certain scenarios (e.g., massive disadvantage) and the graded drop in Take-The-Best adherence when opposing evidence accumulates significantly.

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
        contradiction = max(0, remaining_b - remaining_a)
    else:
        contradiction = max(0, remaining_a - remaining_b)
        
    # The weight of the first cue is modulated by the contradiction
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
- w_first: [1.0, 10.0]
- w_remaining: [0.0, 5.0]
- dispersion_effect: [-2.0, 2.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Applying the minimal diff requested by the critic: renamed `dispersion_decay` to `dispersion_effect` and set its range to `[-2.0, 2.0]`. The update equation was modified to `adj_first = w_first * np.exp(dispersion_effect * contradiction)` to allow the model to learn whether massive contradiction strengthens or weakens the primary cue, rather than forcing a decay. I also widened the ranges for `epsilon` and `beta` to better accommodate the low TTB adherence observed in Experiment 28.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate implements the prescribed Context-Dependent Lexicographic Accumulator, explicitly modulating the weight of the first discriminating cue based on the contradiction from remaining cues. However, the model strictly forces the first cue's weight to *decay* when contradiction is high (via `adj_first = w_first * np.exp(-dispersion_decay * contradiction)` with `dispersion_decay` in [0.0, 2.0]). This leads to completely backward predictions in key experiments: in Exp 1 (massive disadvantage), observed TTB adherence is high (0.6508), but the candidate's simulated TTB adherence drops to 0.2142 because the first cue is penalized. Similarly, the candidate fails on Exp 27 and Exp 28, missing the observed baseline behaviors.
Rationale: To fix the backward predictions while remaining faithful to the arbiter's prescribed Context-Dependent Lexicographic Accumulator family, the model must be allowed to learn the correct direction of the context modulation. Rename `dispersion_decay` to `dispersion_effect` and expand its parameter range to `[-2.0, 2.0]`. Then use `adj_first = w_first * np.exp(dispersion_effect * contradiction)`. This minor adjustment allows the model to empirically determine whether massive contradiction strengthens or weakens the primary cue. Additionally, ensure `epsilon` and `beta` ranges are wide enough to capture the low TTB adherence in Exp 28.

**Outcome of this advice:** iter 1 candidate loss=0.4818 -> iter 2 candidate loss=0.4089 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate successfully improved the aggregate loss (from 0.4818 to 0.4089) by allowing the dispersion effect to be bidirectional, which helped fit Experiment 1 and Experiment 9 much better. However, the model still catastrophically fails on Experiments 11, 12, and 28. In Experiments 11 and 12, the observed differences between agreement and conflict trials are strongly negative (e.g., -0.4292 and -0.6711), meaning that subjects are paradoxically choosing the TTB-favored option *more* frequently when the remaining cues contradict it, or avoiding options with many positive remaining cues. The candidate currently predicts positive values for these metrics because the parameter `w_remaining` is strictly bounded to [0.0, 5.0], forcing the remaining cues to always add positive evidence.
Rationale: To capture the perverse/negative effect of remaining cues observed in Experiments 11, 12, and 28, the model must be allowed to assign a negative weight to the tally of remaining cues. Keep the current functional form exactly as is, but widen the parameter range of `w_remaining` to `[-5.0, 5.0]` (instead of `[0.0, 5.0]`). This minor adjustment remains completely within the prescribed Context-Dependent Lexicographic Accumulator family while giving the optimization routine the freedom to learn whether secondary cues are used compensatorily or aversively in this specific experimental domain.

**Outcome of this advice:** iter 2 candidate loss=0.4089 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Context-Dependent Lexicographic Accumulator (CDLA) with Bidirectional Dispersion Effect and Negative Secondary Weights: Decision-makers process information lexicographically, but the weight assigned to the first discriminating cue is contextually modulated by the dispersion of remaining cues. Additionally, the remaining cues can be weighted negatively, capturing a perverse effect where subjects might avoid options with many positive secondary cues, leading to higher adherence to the first cue when contradicted.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    \n    first_disc_idx = -1\n    for i in cue_order:\n        if a[i] != b[i]:\n            first_disc_idx = i\n            break\n            \n    if first_disc_idx == -1:\n        return np.array([0.5, 0.5])\n        \n    w_first = float(parameters[\"w_first\"])\n    w_remaining = float(parameters[\"w_remaining\"])\n    dispersion_effect = float(parameters[\"dispersion_effect\"])\n    \n    remaining_a = 0\n    remaining_b = 0\n    passed_first = False\n    for i in cue_order:\n        if passed_first:\n            remaining_a += a[i]\n            remaining_b += b[i]\n        if i == first_disc_idx:\n            passed_first = True\n            \n    first_diff = a[first_disc_idx] - b[first_disc_idx]\n    \n    if first_diff > 0:\n        contradiction = max(0, remaining_b - remaining_a)\n    else:\n        contradiction = max(0, remaining_a - remaining_b)\n        \n    # The weight of the first cue is modulated by the contradiction\n    adj_first = w_first * np.exp(dispersion_effect * contradiction)\n    \n    ev_a = 0.0\n    ev_b = 0.0\n    \n    if first_diff > 0:\n        ev_a += adj_first\n    else:\n        ev_b += adj_first\n        \n    ev_a += w_remaining * remaining_a\n    ev_b += w_remaining * remaining_b\n    \n    beta = float(parameters[\"beta\"])\n    z = beta * np.array([ev_a, ev_b])\n    z = z - np.max(z)\n    p = np.exp(z)\n    p = p / np.sum(p)\n    \n    epsilon = float(parameters[\"epsilon\"])\n    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "w_first": "[1.0, 10.0]",
    "w_remaining": "[-5.0, 5.0]",
    "dispersion_effect": "[-2.0, 2.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, I kept the functional form of the Context-Dependent Lexicographic Accumulator exactly intact but widened the parameter range of `w_remaining` to `[-5.0, 5.0]`. This minor adjustment allows the model to assign negative weights to the tally of remaining cues, giving the optimization routine the freedom to capture the perverse/negative effect of secondary cues observed in Experiments 11, 12, and 28, where subjects paradoxically choose the TTB-favored option more when contradicted."
}
```
```

## Usage

```json
{
  "prompt_token_count": 41331,
  "candidates_token_count": 1025,
  "total_token_count": 43325
}
```
