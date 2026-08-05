# generation_iter_05_attempt_00

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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a new theory that blends the strict rank-based nature of Take-The-Best with the compensatory sensitivity of WADD. For example, a 'Mixture of TTB and WADD' (rather than Tallying) or a 'Threshold-based WADD' where cues only contribute to a compensatory sum if their validities exceed a certain threshold. A TTB/WADD mixture would allow strict one-reason decision making to dominate (explaining Exp 1's near-zero difference), while the WADD component would evaluate the exact validities of the compensating cues when TTB is not used (explaining Exp 2's high variance).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.544)

**Description**
People use a Weighted Additive (WADD) strategy to make decisions, where each feature is weighted by its subjective importance (a non-linear transformation of its validity). The total score for an option is the sum of the weighted features it possesses. This compensatory mechanism allows multiple lower-validity cues to sometimes outweigh a single high-validity cue, naturally interpolating between Take The Best (when validity differences are heavily magnified) and Tallying (when validities are ignored). Response noise and lapses account for stochasticity in choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities into subjective weights using a power function.
    # gamma = 0 yields equal weights (Tallying); gamma -> inf yields lexicographic weights (Take The Best).
    weights = val ** gamma
    
    # Compute weighted additive scores for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

`parameters`:
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 30.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6533 (var=0.0443) vs this=0.6640 (var=0.0320)
- Experiment 2: real=0.3696 (var=0.0477) vs this=0.3227 (var=0.0319)
- Experiment 3: real=0.4021 (var=0.0113) vs this=0.3475 (var=0.0281)
- Experiment 4: real=0.2479 (var=0.0104) vs this=0.3542 (var=0.0643)
- Experiment 5: real=0.0133 (var=0.0126) vs this=0.1833 (var=0.0475)
- Experiment 6: real=0.1327 (var=0.0146) vs this=0.0503 (var=0.0036)


---

### `pi_4` (overall score: 0.515)

**Description**
People possess an adaptive toolbox of decision heuristics, primarily Take The Best (TTB) and Tallying, and select among them probabilistically. TTB is a non-compensatory, one-reason decision rule that relies solely on the most valid discriminating cue, while Tallying is an equal-weight compensatory rule that simply counts the number of cues favoring each option. By probabilistically mixing these distinct strategies on a trial-by-trial basis (or reflecting a population-level mixture of strategy users), decision-makers exhibit intermediate compensatory behavior. This avoids the need for a unified, complex non-linear cue weighting mechanism like Weighted Additive (WADD) models. Decision stochasticity is accounted for by the probabilistic strategy selection, strategy-specific softmax response noise, and an overarching lapse rate. The mixture is heavily biased toward TTB, reflecting the empirical dominance of one-reason decision making in these environments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take The Best (TTB) --- 
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # --- Tallying ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # --- Strategy Mixture ---
    p_ttb_weight = float(parameters["p_ttb"])
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_tally_core
    
    # --- Lapse Noise ---
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- p_ttb: [0.5, 1.0]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6533 (var=0.0443) vs this=0.7275 (var=0.0221)
- Experiment 2: real=0.3696 (var=0.0477) vs this=0.2754 (var=0.0178)
- Experiment 3: real=0.4021 (var=0.0113) vs this=0.2535 (var=0.0108)
- Experiment 4: real=0.2479 (var=0.0104) vs this=0.2737 (var=0.0273)
- Experiment 5: real=0.0133 (var=0.0126) vs this=0.0033 (var=0.0228)
- Experiment 6: real=0.1327 (var=0.0146) vs this=0.0105 (var=0.0012)


---

### `pi_1` (overall score: 0.394)

**Description**
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6533 (var=0.0443) vs this=0.8408 (var=0.0108)
- Experiment 2: real=0.3696 (var=0.0477) vs this=0.1496 (var=0.0077)
- Experiment 3: real=0.4021 (var=0.0113) vs this=0.1415 (var=0.0107)
- Experiment 4: real=0.2479 (var=0.0104) vs this=0.1227 (var=0.0108)
- Experiment 5: real=0.0133 (var=0.0126) vs this=-0.0167 (var=0.0247)
- Experiment 6: real=0.1327 (var=0.0146) vs this=0.0128 (var=0.0008)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2518 -> ACCEPTED
- iter 2: loss=0.3210 -> REJECTED
- iter 3: loss=0.2309 -> ACCEPTED
- iter 4: loss=0.1851 -> ACCEPTED
- iter 5: loss=0.2421 -> REJECTED
Running-best (last ACCEPTED) base: iter 4 at loss=0.1851 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_consistent = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if ttb_winner == resp:
            ttb_consistent += 1
            
    return float(ttb_consistent / len(data))
```

**Observed (real) value:** 0.6533 (var=0.0443)
**Previous candidate values (this loop):**
  - iter 1: 0.6965 (var=0.0401) (Δ vs real +0.0431)
  - iter 2: 0.7879 (var=0.0265) (Δ vs real +0.1346)
  - iter 3: 0.5913 (var=0.0533) (Δ vs real -0.0621)
  - iter 4: 0.6271 (var=0.0479) (Δ vs real -0.0262)
  - iter 5 (most recent): 0.7771 (var=0.0287) (Δ vs real +0.1238)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8408 (var=0.0108)
- pi_2: 0.1517 (var=0.0089)
- pi_3: 0.6640 (var=0.0320)
- pi_4: 0.7275 (var=0.0221)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tallying_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        if row['response'] == tally_choice:
            tallying_consistent += 1
        total += 1
        
    if total == 0:
        return 0.5
    return float(tallying_consistent / total)
```

**Observed (real) value:** 0.3696 (var=0.0477)
**Previous candidate values (this loop):**
  - iter 1: 0.2831 (var=0.0338) (Δ vs real -0.0865)
  - iter 2: 0.2375 (var=0.0195) (Δ vs real -0.1321)
  - iter 3: 0.4410 (var=0.0541) (Δ vs real +0.0715)
  - iter 4: 0.3817 (var=0.0475) (Δ vs real +0.0121)
  - iter 5 (most recent): 0.2712 (var=0.0442) (Δ vs real -0.0983)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8775 (var=0.0070)
- pi_1: 0.1496 (var=0.0077)
- pi_3: 0.3227 (var=0.0319)
- pi_4: 0.2754 (var=0.0178)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4021 (var=0.0113)
**Previous candidate values (this loop):**
  - iter 1: 0.3065 (var=0.0295) (Δ vs real -0.0956)
  - iter 2: 0.2335 (var=0.0196) (Δ vs real -0.1685)
  - iter 3: 0.4083 (var=0.0341) (Δ vs real +0.0062)
  - iter 4: 0.3625 (var=0.0421) (Δ vs real -0.0396)
  - iter 5 (most recent): 0.2529 (var=0.0363) (Δ vs real -0.1492)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1415 (var=0.0107)
- pi_3: 0.3475 (var=0.0281)
- pi_2: 0.7240 (var=0.0034)
- pi_4: 0.2535 (var=0.0108)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # In all 8 trials of this design, Option A is constructed to be the Take-The-Best (TTB) choice
    # (it always possesses the highest-validity discriminating cue). Option B is the compensatory
    # choice (it possesses multiple lower-validity cues).
    # This metric simply calculates the overall proportion of times Option B was chosen.
    # TTB will only choose B due to lapse noise (epsilon), keeping this metric low.
    # WADD will choose B systematically when gamma is low enough to allow compensation,
    # leading to a significantly higher mean.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.2479 (var=0.0104)
**Previous candidate values (this loop):**
  - iter 1: 0.3702 (var=0.0406) (Δ vs real +0.1223)
  - iter 2: 0.2629 (var=0.0198) (Δ vs real +0.0150)
  - iter 3: 0.4500 (var=0.0627) (Δ vs real +0.2021)
  - iter 4: 0.3935 (var=0.0630) (Δ vs real +0.1456)
  - iter 5 (most recent): 0.3204 (var=0.0664) (Δ vs real +0.0725)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3542 (var=0.0643)
- pi_1: 0.1227 (var=0.0108)
- pi_2: 0.8721 (var=0.0048)
- pi_4: 0.2737 (var=0.0273)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project lists to tuples for hashable comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 6: A has cue 1 (0.95), B has cue 2 (0.85)
    t6_A = (1, 0, 0, 0, 0)
    t6_B = (0, 1, 0, 0, 0)
    
    # Trial 8: A has cue 3 (0.75), B has cue 4 (0.65)
    t8_A = (0, 0, 1, 0, 0)
    t8_B = (0, 0, 0, 1, 0)
    
    mask_6 = (a_tuples == t6_A) & (b_tuples == t6_B)
    mask_8 = (a_tuples == t8_A) & (b_tuples == t8_B)
    
    if not mask_6.any() or not mask_8.any():
        return 0.0
        
    # Calculate probability of choosing A (response == 0)
    pA_t6 = (data.loc[mask_6, 'response'] == 0).mean()
    pA_t8 = (data.loc[mask_8, 'response'] == 0).mean()
    
    return float(pA_t6 - pA_t8)
```

**Observed (real) value:** 0.0133 (var=0.0126)
**Previous candidate values (this loop):**
  - iter 1: 0.0333 (var=0.0292) (Δ vs real +0.0200)
  - iter 2: 0.0767 (var=0.0258) (Δ vs real +0.0633)
  - iter 3: 0.0617 (var=0.0338) (Δ vs real +0.0483)
  - iter 4: -0.0033 (var=0.0294) (Δ vs real -0.0167)
  - iter 5 (most recent): 0.0350 (var=0.0253) (Δ vs real +0.0217)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0033 (var=0.0228)
- pi_3: 0.1833 (var=0.0475)
- pi_1: -0.0167 (var=0.0247)
- pi_2: -0.0250 (var=0.0456)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Create a hashable surrogate for the trial type using Option B's ratings
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the mean response (probability of choosing B) for each trial type
    trial_means = data.groupby(b_str)['response'].mean()
    
    # Calculate the standard deviation of these means
    return float(np.std(trial_means))
```

**Observed (real) value:** 0.1327 (var=0.0146)
**Previous candidate values (this loop):**
  - iter 1: 0.0220 (var=0.0018) (Δ vs real -0.1107)
  - iter 2: 0.0183 (var=0.0022) (Δ vs real -0.1143)
  - iter 3: 0.0490 (var=0.0045) (Δ vs real -0.0836)
  - iter 4: 0.0455 (var=0.0037) (Δ vs real -0.0872)
  - iter 5 (most recent): 0.0494 (var=0.0056) (Δ vs real -0.0832)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0503 (var=0.0036)
- pi_4: 0.0105 (var=0.0012)
- pi_1: 0.0128 (var=0.0008)
- pi_2: 0.0135 (var=0.0010)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Decision-makers use an adaptive toolbox of strategies, dynamically selecting between Take The Best (TTB) and Weighted Additive (WADD) evaluation on a trial-by-trial basis. The probability of using TTB over WADD depends on the conflict between the cues: when the top discriminating cue is much stronger than the opposing cues, TTB dominates; when multiple opposing cues rival the top cue's validity, the decision-maker is more likely to switch to the compensatory WADD strategy. By applying independent scaling weights to the top cue's validity and the sum of opposing validities, the model can penalize a large number of weak opposing cues, keeping TTB dominant in those cases while still allowing WADD to differentiate between strong compensatory profiles.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take The Best (TTB) ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    v_top = 0.0
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            v_top = val[j]
            break
        if b[j] > a[j]:
            winner_ttb = 1
            v_top = val[j]
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
        p_ttb_weight = 0.5
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
        # Dynamic mixture weight based on conflict
        v_opp_sum = 0.0
        for j in range(len(val)):
            if winner_ttb == 0 and b[j] > a[j]:
                v_opp_sum += val[j]
            elif winner_ttb == 1 and a[j] > b[j]:
                v_opp_sum += val[j]
                
        theta_top = float(parameters["theta_top"])
        theta_opp = float(parameters["theta_opp"])
        bias_ttb = float(parameters["bias_ttb"])
        z_mix = theta_top * v_top - theta_opp * v_opp_sum + bias_ttb
        p_ttb_weight = 1.0 / (1.0 + np.exp(-z_mix))
        
    # --- Weighted Additive (WADD) ---
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores_wadd = np.array([score_a, score_b])
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd_core = e_wadd / np.sum(e_wadd)
    
    # --- Strategy Mixture ---
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_wadd_core
    
    # --- Lapse Noise ---
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- theta_top: [0.0, 20.0]
- theta_opp: [0.0, 20.0]
- bias_ttb: [-10.0, 10.0]
- gamma: [0.0, 10.0]
- beta_ttb: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: To address the over-prediction of compensatory choices in Experiment 4 and increase variance in Experiment 6, we decouple the scaling of the top cue's validity and the sum of opposing cues' validities in the mixture weight calculation. By introducing independent parameters `theta_top` and `theta_opp`, the model can penalize the opposing evidence if it consists of many weak cues (as in Exp 4), keeping TTB dominant. At the same time, it maintains the sensitivity to strong compensatory profiles required to produce the variance across trial types in Exp 6.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate model successfully implemented the requested TTB/WADD mixture and achieved a new best aggregate loss, which is why it was accepted. It balances Exp 1 and Exp 2 reasonably well. However, it still struggles with Experiment 4 (overestimating the choice of the compensatory Option B) and Experiment 6 (drastically underestimating the variance across different compensatory trial types). In Exp 6, the observed standard deviation of means across trial types is 0.1327, but the candidate only produces 0.0220. This indicates that the WADD component is either being washed out by TTB or lacks the sensitivity to differentiate between the specific validities of the compensating cues.
Rationale: While the TTB/WADD mixture is the correct family and improved the overall loss, the current parameterization suppresses the variance needed for Exp 6 and over-predicts compensatory choices in Exp 4. To fix this within the prescribed family, adjust the mixture mechanism: instead of a single static `p_ttb` parameter, consider making the reliance on WADD vs TTB sensitive to the difference in validities (e.g., if the best cue is much better than the rest, TTB dominates; if the validities are close, WADD dominates). Alternatively, simply widen the range of `gamma` or adjust `beta_wadd` to allow the WADD component to produce sharper, more differentiated probabilities across the different compensatory cue combinations in Exp 6.

**Outcome of this advice:** iter 1 candidate loss=0.2518 -> iter 2 candidate loss=0.3210 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate was REJECTED by the accept gate because its overall loss worsened (0.3210 vs 0.2518). While the proposer followed the alternative advice to widen the parameter ranges for `gamma` and the inverse temperatures, this approach failed to increase the variance for Experiment 6 (it actually dropped to 0.0183 compared to the target 0.1327) and caused an overestimation of TTB-consistency in Experiment 1. The static mixture weight `p_ttb` combined with extreme non-linear scaling parameters is not capturing the nuanced trade-off between strict one-reason decision making and compensatory evaluation.
Rationale: Because this candidate was rejected, the loop will revert to the previous accepted base (iter 1). The attempt to fix Exp 6 by merely widening `gamma` and `beta` failed. For the next attempt on the iter 1 base, DO NOT just widen the static parameter ranges. Instead, implement the other approach suggested in the previous critique: make the mixture weight dynamic. Specifically, calculate `p_ttb` on a trial-by-trial basis as a function of the cues present (e.g., based on the difference between the top discriminating cue's validity and the sum/max of the opposing cues' validities). This dynamic gating between TTB and WADD will allow the model to selectively apply compensatory logic only when the compensatory cues are strong enough to rival the top cue, addressing the variance in Exp 6 without breaking Exp 4.

**Outcome of this advice:** iter 2 candidate loss=0.3210 -> iter 3 candidate loss=0.2309 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate successfully implemented the dynamic gating mechanism and was ACCEPTED by the gate, achieving a new best overall loss (0.2309). The dynamic mixture significantly improved the fit for Experiment 3 (mean response 0.4083 vs real 0.4021) and doubled the variance in Experiment 6 (0.0490 vs previous 0.0220). However, the model now severely over-predicts compensatory choices in Experiment 4 (0.4500 vs real 0.2479). This happens because the conflict metric `v_opp_sum` simply sums all opposing validities. In Experiment 4, Option B has many opposing cues, causing `v_opp_sum` to be very large, which drives `p_ttb_weight` to near zero and lets WADD dominate too often. The variance in Experiment 6 is also still short of the 0.1327 target.
Rationale: The dynamic mixture is the right approach, but the conflict metric needs refinement to prevent WADD from over-dominating when there are many weak opposing cues (as in Exp 4). Adjust the calculation of the mixture weight: instead of a simple sum `v_opp_sum`, consider using the maximum of the opposing validities (`v_opp_max`), or apply independent scaling parameters for `v_top` and the opposing evidence (e.g., `theta_top * v_top - theta_opp * v_opp_sum + bias_ttb`). This will allow the model to penalize the opposing evidence if it consists of many weak cues, keeping TTB dominant in Exp 4 while still allowing WADD to differentiate between strong compensatory profiles in Exp 6.

**Outcome of this advice:** iter 3 candidate loss=0.2309 -> iter 4 candidate loss=0.1851 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate was ACCEPTED and achieved a new best loss (0.1851). The introduction of independent parameters `theta_top` and `theta_opp` successfully reduced the over-prediction of compensatory choices in Experiment 4 (dropping from 0.4500 to 0.3935) while maintaining excellent fits on Experiments 1, 2, and 3. However, the model still struggles to fully suppress WADD in Exp 4 (observed 0.2479) and still lacks the variance required across trial types in Exp 6 (observed 0.1327 vs candidate 0.0455). Because `v_opp_sum` simply adds up the opposing validities linearly, a large number of weak cues (as in Exp 4) can still accumulate enough mass to rival the top cue and trigger a switch to WADD.
Rationale: To further suppress compensatory choices in Exp 4 and boost the variance in Exp 6 within the current TTB/WADD mixture family, change how opposing cues are aggregated in the mixture weight calculation. Instead of a linear sum of opposing validities (`v_opp_sum += val[j]`), apply a non-linear penalty to weak cues by raising the opposing validities to a power before summing (e.g., `v_opp_sum += val[j] ** kappa`, with a new parameter `kappa` bounded between 1.0 and 10.0), or simply use the maximum opposing validity (`v_opp_max`). This will ensure that multiple weak cues (Exp 4) fail to trigger a switch to WADD, while strong opposing cues (Exp 6) successfully compete with the top cue.

**Outcome of this advice:** iter 4 candidate loss=0.1851 -> iter 5 candidate loss=0.2421 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate was REJECTED by the accept gate because its overall loss worsened (0.2421 vs the running best 0.1851). While introducing the non-linear penalty 'kappa' to the opposing validities slightly improved the suppression of compensatory choices in Experiment 4 (0.3204 down from 0.3935), it severely disrupted the balance in Experiments 1, 2, and 3. Specifically, TTB consistency in Exp 1 shot up too high, and the mean response in Exp 3 dropped significantly (0.2529 vs observed 0.4021). The non-linear scaling distorted the mixture weights across the board, failing to provide the targeted fix for Exp 4 without breaking the others.
Rationale: Because this candidate was rejected, the loop will revert to the previous accepted base (iter 4). The attempt to fix Exp 4 using a non-linear penalty ('kappa') on the sum of opposing cues failed. For the next iteration on the iter 4 base, replace the sum of opposing validities with the MAXIMUM opposing validity (`v_opp_max`). By defining the conflict purely as the difference between the top discriminating cue and the single strongest opposing cue (e.g., `z_mix = theta_top * v_top - theta_opp * v_opp_max + bias_ttb`), you completely prevent any accumulation of weak cues (as in Exp 4) from triggering WADD, without needing to introduce extreme non-linear scaling parameters that break the well-calibrated fits in Exps 1-3.

**Outcome of this advice:** iter 5 candidate loss=0.2421 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Decision-makers use an adaptive toolbox of strategies, dynamically selecting between Take The Best (TTB) and Weighted Additive (WADD) evaluation on a trial-by-trial basis. The probability of using TTB over WADD depends on the conflict between the cues: when the top discriminating cue is much stronger than the strongest opposing cue, TTB dominates; when a single opposing cue rivals the top cue's validity, the decision-maker is more likely to switch to the compensatory WADD strategy. By defining conflict based on the maximum opposing validity rather than the sum, the model prevents a large number of weak opposing cues from improperly triggering compensatory evaluation, keeping TTB dominant in those cases while maintaining sensitivity to strong compensatory profiles.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n    \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # --- Take The Best (TTB) ---\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    winner_ttb = None\n    v_top = 0.0\n    for j in cue_order:\n        if a[j] > b[j]:\n            winner_ttb = 0\n            v_top = val[j]\n            break\n        if b[j] > a[j]:\n            winner_ttb = 1\n            v_top = val[j]\n            break\n            \n    if winner_ttb is None:\n        p_ttb_core = np.array([0.5, 0.5])\n        p_ttb_weight = 0.5\n    else:\n        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])\n        beta_ttb = float(parameters[\"beta_ttb\"])\n        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))\n        e_ttb = np.exp(z_ttb)\n        p_ttb_core = e_ttb / np.sum(e_ttb)\n        \n        # Dynamic mixture weight based on conflict with the MAX opposing cue\n        v_opp_max = 0.0\n        for j in range(len(val)):\n            if winner_ttb == 0 and b[j] > a[j]:\n                if val[j] > v_opp_max:\n                    v_opp_max = val[j]\n            elif winner_ttb == 1 and a[j] > b[j]:\n                if val[j] > v_opp_max:\n                    v_opp_max = val[j]\n                \n        theta_top = float(parameters[\"theta_top\"])\n        theta_opp = float(parameters[\"theta_opp\"])\n        bias_ttb = float(parameters[\"bias_ttb\"])\n        z_mix = theta_top * v_top - theta_opp * v_opp_max + bias_ttb\n        p_ttb_weight = 1.0 / (1.0 + np.exp(-z_mix))\n        \n    # --- Weighted Additive (WADD) ---\n    gamma = float(parameters[\"gamma\"])\n    weights = val ** gamma\n    score_a = np.sum(weights * a)\n    score_b = np.sum(weights * b)\n    scores_wadd = np.array([score_a, score_b])\n    \n    beta_wadd = float(parameters[\"beta_wadd\"])\n    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))\n    e_wadd = np.exp(z_wadd)\n    p_wadd_core = e_wadd / np.sum(e_wadd)\n    \n    # --- Strategy Mixture ---\n    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_wadd_core\n    \n    # --- Lapse Noise ---\n    epsilon = float(parameters[\"epsilon\"])\n    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])\n    \n    return p_final\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "theta_top": "[0.0, 20.0]",
    "theta_opp": "[0.0, 20.0]",
    "bias_ttb": "[-10.0, 10.0]",
    "gamma": "[0.0, 10.0]",
    "beta_ttb": "[0.1, 20.0]",
    "beta_wadd": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "To address the over-prediction of compensatory choices in Experiment 4 while avoiding the disruption caused by non-linear scaling in iter 5, we replace the sum of opposing validities with the MAXIMUM opposing validity (`v_opp_max`). This ensures that the conflict is defined purely as the difference between the top discriminating cue and the single strongest opposing cue. As a result, an accumulation of weak cues (as in Exp 4) cannot trigger a switch to WADD, keeping TTB dominant, while strong opposing cues (as in Exp 6) can successfully compete with the top cue."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14845,
  "candidates_token_count": 1468,
  "total_token_count": 17067
}
```
