# generation_iter_01_attempt_00

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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace Theory 2 with a 'Rank-Dependent Evidence Accumulation' (or Rank-Weighted Additive) theory. Instead of a probabilistic mixture of two separate heuristics, propose that subjects use a single compensatory mechanism where the subjective weight of each feature is determined by its rank order of validity, rather than its raw validity score (e.g., weights decay exponentially by rank). This naturally bridges the gap between TTB (steep exponential decay) and Tallying (flat weights), allowing for compensatory trade-offs (like WADD) while naturally discounting lower-ranked cues. This single-process approach might better capture the intermediate choice probabilities across the experimental suite without relying on an arbitrary mixture parameter.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.797)

**Description**
WADD / Take-The-Best Strategy Mixture Theory with Calibrated Determinism: Decision-makers probabilistically choose between a fully compensatory Weighted Additive (WADD) strategy and a non-compensatory Take-The-Best (TTB) strategy on any given trial. By mixing WADD and TTB, the model captures both dominance violations (via WADD) and choice probability plateaus (via TTB). Independent inverse temperatures allow each strategy to scale properly. To match the high level of equivocation in human data across multiple experiments, the determinism of these strategies is strongly bounded, ensuring predictions remain appropriately probabilistic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_wadd = float(parameters["beta_wadd"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    p_wadd_weight = float(parameters["p_wadd_weight"])
    
    # --- WADD Prediction ---
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores_wadd = np.array([score_a, score_b])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- Take-The-Best (TTB) Prediction ---
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
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Mixture & Noise ---
    p_mix = p_wadd_weight * p_wadd + (1.0 - p_wadd_weight) * p_ttb
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta_wadd: [0.0, 3.0]
- beta_ttb: [0.0, 3.0]
- p_wadd_weight: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3747 (var=0.0051) vs this=0.5238 (var=0.0281)
- Experiment 2: real=0.6400 (var=0.0095) vs this=0.5296 (var=0.0326)
- Experiment 3: real=0.5863 (var=0.0190) vs this=0.6275 (var=0.0122)
- Experiment 4: real=0.5317 (var=0.0228) vs this=0.6717 (var=0.0158)
- Experiment 5: real=0.4700 (var=0.0206) vs this=0.3492 (var=0.0201)
- Experiment 6: real=0.1378 (var=0.0902) vs this=0.1111 (var=0.1141)
- Experiment 7: real=0.4500 (var=0.0325) vs this=0.5708 (var=0.0219)
- Experiment 8: real=0.0767 (var=0.0235) vs this=0.0717 (var=0.0104)


---

### `pi_4` (overall score: 0.382)

**Description**
Strategy Mixture Theory with Decoupled Determinism: Decision-makers probabilistically choose between Tallying and Take-The-Best on any given trial. Because Tallying operates on raw integer count differences (which can be large) while TTB operates on binary 1/0 differences, the two heuristics have different intrinsic scales. To account for this, the model uses independent inverse temperatures (beta_tally and beta_ttb) for each strategy. This allows the decision process to independently calibrate the determinism of each heuristic, avoiding artificial scale distortions when balancing the two strategies via the mixture weight.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    p_tally_weight = float(parameters["p_tally_weight"])
    
    # --- Tallying Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # --- Take-The-Best (TTB) Prediction ---
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
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Mixture & Noise ---
    p_mix = p_tally_weight * p_tally + (1.0 - p_tally_weight) * p_ttb
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta_tally: [0.1, 20.0]
- beta_ttb: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_tally_weight: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3747 (var=0.0051) vs this=0.5258 (var=0.0613)
- Experiment 2: real=0.6400 (var=0.0095) vs this=0.5837 (var=0.0394)
- Experiment 3: real=0.5863 (var=0.0190) vs this=0.6769 (var=0.0151)
- Experiment 4: real=0.5317 (var=0.0228) vs this=0.6825 (var=0.0197)
- Experiment 5: real=0.4700 (var=0.0206) vs this=0.1492 (var=0.0135)
- Experiment 6: real=0.1378 (var=0.0902) vs this=0.0289 (var=0.0774)
- Experiment 7: real=0.4500 (var=0.0325) vs this=0.8408 (var=0.0103)
- Experiment 8: real=0.0767 (var=0.0235) vs this=-0.0204 (var=0.0072)


---

### `pi_2` (overall score: 0.322)

**Description**
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3747 (var=0.0051) vs this=0.1318 (var=0.0071)
- Experiment 2: real=0.6400 (var=0.0095) vs this=0.8767 (var=0.0063)
- Experiment 3: real=0.5863 (var=0.0190) vs this=0.4850 (var=0.0067)
- Experiment 4: real=0.5317 (var=0.0228) vs this=0.4892 (var=0.0105)
- Experiment 5: real=0.4700 (var=0.0206) vs this=0.1350 (var=0.0111)
- Experiment 6: real=0.1378 (var=0.0902) vs this=0.0244 (var=0.0643)
- Experiment 7: real=0.4500 (var=0.0325) vs this=0.8592 (var=0.0146)
- Experiment 8: real=0.0767 (var=0.0235) vs this=-0.0158 (var=0.0081)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4026 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.4026 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            if b[i] > a[i]:
                return 1
        return -1
    
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if ttb_predict(a, b) == r:
            matches += 1
            
    return float(matches) / len(data) if len(data) > 0 else 0.0
```

**Observed (real) value:** 0.3747 (var=0.0051)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5338 (var=0.0723) (Δ vs real +0.1591)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8407 (var=0.0059)
- pi_2: 0.1318 (var=0.0071)
- pi_3: 0.3573 (var=0.0363)
- pi_4: 0.5258 (var=0.0613)
- pi_5: 0.5238 (var=0.0281)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Calculate first feature for each option (most valid cue)
    first_a = data['option_a_ratings'].apply(lambda x: x[0])
    first_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where Tallying and TTB make strictly opposite predictions
    # Tallying decides by sum(A) vs sum(B)
    # TTB decides by A[0] vs B[0]
    opposing_mask = ((sum_a > sum_b) & (first_a < first_b)) | ((sum_a < sum_b) & (first_a > first_b))
    opposing_data = data[opposing_mask]
    
    if len(opposing_data) == 0:
        return 0.5
        
    # Determine what Tallying would choose: 0 (A) if sum_a > sum_b else 1 (B)
    tallying_choice = (opposing_data['option_a_ratings'].apply(sum) < opposing_data['option_b_ratings'].apply(sum)).astype(int)
    
    # Calculate the proportion of responses that match the Tallying prediction
    match_rate = (opposing_data['response'] == tallying_choice).mean()
    
    return float(match_rate)
```

**Observed (real) value:** 0.6400 (var=0.0095)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5312 (var=0.1169) (Δ vs real -0.1088)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8767 (var=0.0063)
- pi_1: 0.1408 (var=0.0115)
- pi_3: 0.7929 (var=0.0253)
- pi_4: 0.5837 (var=0.0394)
- pi_5: 0.5296 (var=0.0326)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    wadd_match = 0
    n_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            
            if wadd_a > wadd_b:
                expected = 0
            elif wadd_b > wadd_a:
                expected = 1
            else:
                continue
                
            if row['response'] == expected:
                wadd_match += 1
            n_ties += 1
            
    if n_ties == 0:
        return 0.5
    return wadd_match / n_ties
```

**Observed (real) value:** 0.5863 (var=0.0190)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8438 (var=0.0217) (Δ vs real +0.2575)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7831 (var=0.0163)
- pi_2: 0.4850 (var=0.0067)
- pi_1: 0.8438 (var=0.0137)
- pi_4: 0.6769 (var=0.0151)
- pi_5: 0.6275 (var=0.0122)

### Experiment 4
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target_pairs = [
        ( (1,0,0,0), (0,1,0,0) ),
        ( (1,0,1,0), (0,1,0,1) )
    ]
    
    chosen_has_f1 = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        is_target = False
        for p in target_pairs:
            if (a == p[0] and b == p[1]) or (a == p[1] and b == p[0]):
                is_target = True
                break
                
        if is_target:
            chose_a = (row['response'] == 0)
            chosen_opt = a if chose_a else b
            
            if chosen_opt[0] == 1:
                chosen_has_f1.append(1)
            else:
                chosen_has_f1.append(0)
                
    if len(chosen_has_f1) == 0:
        return 0.5
        
    return float(np.mean(chosen_has_f1))
```

**Observed (real) value:** 0.5317 (var=0.0228)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7875 (var=0.0290) (Δ vs real +0.2558)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4892 (var=0.0105)
- pi_3: 0.7533 (var=0.0292)
- pi_1: 0.8408 (var=0.0152)
- pi_4: 0.6825 (var=0.0197)
- pi_5: 0.6717 (var=0.0158)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = tuple(a)
        b = tuple(b)
        if a == (1, 0, 0, 1, 1) and b == (0, 1, 1, 0, 0):
            wadd_choices += int(r == 1)
            total += 1
        elif a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 1, 1):
            wadd_choices += int(r == 0)
            total += 1
    return float(wadd_choices / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4700 (var=0.0206)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1392 (var=0.0158) (Δ vs real -0.3308)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7258 (var=0.0583)
- pi_4: 0.1492 (var=0.0135)
- pi_1: 0.1558 (var=0.0126)
- pi_2: 0.1350 (var=0.0111)
- pi_5: 0.3492 (var=0.0201)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Response = 0 means option A was chosen
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    t1_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 1, 1))
    t2_mask = (data['A_tuple'] == (1, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 1, 1, 1))
    t9_mask = (data['A_tuple'] == (1, 1, 1, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 1))
    t10_mask = (data['A_tuple'] == (1, 1, 1, 1, 0)) & (data['B_tuple'] == (0, 0, 0, 0, 1))
    
    p_t1 = data.loc[t1_mask, 'chose_A'].mean()
    p_t2 = data.loc[t2_mask, 'chose_A'].mean()
    p_t9 = data.loc[t9_mask, 'chose_A'].mean()
    p_t10 = data.loc[t10_mask, 'chose_A'].mean()
    
    if pd.isna(p_t1) or pd.isna(p_t2) or pd.isna(p_t9) or pd.isna(p_t10):
        return 0.0
        
    return (p_t2 - p_t1) + (p_t10 - p_t9)

```

**Observed (real) value:** 0.1378 (var=0.0902)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2444 (var=0.1407) (Δ vs real +0.1067)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0289 (var=0.0774)
- pi_3: 0.4778 (var=0.0920)
- pi_1: 0.0156 (var=0.0514)
- pi_2: 0.0244 (var=0.0643)
- pi_5: 0.1111 (var=0.1141)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1_a = (1, 0, 0, 0, 1, 1, 1)
    t1_b = (0, 1, 1, 1, 0, 0, 0)
    
    t2_a = (0, 1, 1, 1, 0, 0, 0)
    t2_b = (1, 0, 0, 0, 1, 1, 1)
    
    matches = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a == t1_a and b == t1_b:
            matches.append(1 if row['response'] == 0 else 0)
        elif a == t2_a and b == t2_b:
            matches.append(1 if row['response'] == 1 else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4500 (var=0.0325)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7258 (var=0.0220) (Δ vs real +0.2758)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5708 (var=0.0219)
- pi_4: 0.8408 (var=0.0103)
- pi_1: 0.8233 (var=0.0134)
- pi_2: 0.8592 (var=0.0146)
- pi_3: 0.2250 (var=0.0656)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.9, 0.85, 0.8, 0.6, 0.55, 0.5])
    
    def get_wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * val) - np.sum(b * val)
        
    wadd_diffs = data.apply(get_wadd_diff, axis=1)
    
    set1_mask = wadd_diffs > 0.1
    set2_mask = wadd_diffs < 0.1
    
    chose_a = (data['response'] == 0).astype(float)
    
    p_a_set1 = chose_a[set1_mask].mean()
    p_a_set2 = chose_a[set2_mask].mean()
    
    if np.isnan(p_a_set1) or np.isnan(p_a_set2):
        return 0.0
        
    return float(p_a_set1 - p_a_set2)

```

**Observed (real) value:** 0.0767 (var=0.0235)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1642 (var=0.0622) (Δ vs real +0.0875)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0204 (var=0.0072)
- pi_5: 0.0717 (var=0.0104)
- pi_1: -0.0017 (var=0.0033)
- pi_2: -0.0158 (var=0.0081)
- pi_3: 0.4712 (var=0.0664)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Rank-Dependent Evidence Accumulation: Decision-makers use a single, unified compensatory process rather than probabilistically switching between distinct heuristics. The subjective weight of each feature is determined by its validity rank rather than its raw validity score, decaying exponentially. A decay rate parameter naturally interpolates between Take-The-Best (steep decay) and Tallying (flat weights), allowing for WADD-like compensatory trade-offs while naturally discounting lower-ranked cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Assign exponentially decaying weights based on rank
    weights = np.zeros_like(val)
    for rank, cue_idx in enumerate(cue_order):
        weights[cue_idx] = gamma ** rank
        
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add random lapse noise
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- gamma: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's feedback, this model replaces the brittle probabilistic mixture of Tallying and Take-The-Best with a single Rank-Dependent Evidence Accumulation mechanism. Instead of relying on raw validity scores, the model assigns weights to features based on their validity rank, decaying exponentially by a factor of 'gamma'. When gamma approaches 0, the model acts exactly like Take-The-Best. When gamma approaches 1, it acts like Tallying. Intermediate values produce WADD-like compensatory behavior that naturally discounts lower-ranked cues. This unified approach inherently resolves the arbitrary scale mismatches of mixture models while capturing the full spectrum of heuristic behaviors.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The Rank-Dependent Evidence Accumulation model was accepted and provides a solid improvement, successfully implementing the prescribed mechanism. However, it still struggles with specific empirical patterns: it strongly overpredicts the metrics in Experiments 3, 4, and 7 (predicting ~0.7-0.84 where human data is ~0.45-0.58), and drastically underpredicts Experiment 5 (0.13 vs human 0.47). The strict exponential decay (`gamma ** rank`) might impose an inflexible shape on the weights, forcing the model into an awkward compromise between TTB-like and Tallying-like behavior across different feature subset sizes.
Rationale: To better capture the intermediate choice probabilities and fix the mismatch in Exps 3, 5, and 7, adjust the rank-based weighting function. Instead of a strict exponential decay, try implementing a power-law decay for the weights: `1.0 / (rank + 1.0) ** gamma`. This provides a different discounting curve that often fits human cognitive biases better than exponential decay. Adjust the `gamma` parameter range accordingly (e.g., [0.0, 5.0]) to allow for both flat weights (gamma=0) and steep TTB-like drop-offs (high gamma). Keep the rest of the prescribed mechanism intact.

**Outcome of this advice:** iter 1 candidate loss=0.4026 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Rank-Dependent Evidence Accumulation with Power-Law Decay: Decision-makers use a single compensatory process where the subjective weight of each feature is determined by its validity rank rather than its raw validity score. To better capture human cognitive biases and intermediate choice probabilities, these weights decay according to a power law (1 / (rank + 1)^gamma) rather than an exponential function. This allows for a more flexible discounting curve that can naturally bridge Take-The-Best (steep decay) and Tallying (flat weights) while better fitting empirical choice patterns across different feature subset sizes.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    gamma = float(parameters[\"gamma\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Sort cues by validity descending\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    \n    # Assign power-law decaying weights based on rank\n    weights = np.zeros_like(val)\n    for rank, cue_idx in enumerate(cue_order):\n        weights[cue_idx] = 1.0 / ((rank + 1.0) ** gamma)\n        \n    score_a = np.sum(a * weights)\n    score_b = np.sum(b * weights)\n    \n    scores = np.array([score_a, score_b])\n    \n    # Softmax with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Add random lapse noise\n    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    \n    return p_final",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probs = np.asarray(probabilities, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "gamma": "[0.0, 5.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's feedback, this edit replaces the strict exponential decay function with a power-law decay function (`1.0 / (rank + 1.0) ** gamma`). The `gamma` parameter range is adjusted to `[0.0, 5.0]` to accommodate this change. Power-law decay provides a different, potentially more flexible discounting curve that often matches human cognitive biases better than exponential decay, which should help correct the over/under-predictions observed in Experiments 3, 4, 5, and 7."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14739,
  "candidates_token_count": 797,
  "total_token_count": 15724
}
```
