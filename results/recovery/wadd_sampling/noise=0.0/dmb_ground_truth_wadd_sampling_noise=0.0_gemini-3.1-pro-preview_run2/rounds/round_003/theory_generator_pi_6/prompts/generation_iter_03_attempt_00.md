# generation_iter_03_attempt_00

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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Propose a 'Rank-Based Weighting' theory. Instead of scaling raw validities or computing log-odds, humans might ignore the exact numerical validities and instead rely on their ordinal rank. The theory should posit that decision-makers sort the cues by validity and assign weights that decay based on rank (e.g., w_i = alpha ** rank_i). This ordinal approach captures bounded rationality—reducing cognitive load by discarding precise probabilities—while still strongly weighting the most valid cues, offering a robust alternative to the continuous transformations of WADD.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.735)

**Description**
People integrate information from multiple cues by computing a weighted sum of the feature values, where the weights are a non-linear transformation of the cues' validities. This Weighted Additive (WADD) strategy is compensatory but incorporates a scaling parameter (gamma) that can exponentiate the validities. This allows the decision-maker to dynamically re-balance attention—either steepening the weights to behave more like Take The Best, or flattening them to behave more like Tallying—to match the empirical balance of conflict resolution.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    # Weighted Additive (WADD) heuristic with exponentiated validities.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Exponentiate validities to allow non-linear scaling of weights
    weights = val ** gamma
    
    # Compute the weighted sum of features for each option
    # using the scaled cue validities as weights.
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 5.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3520 (var=0.0355) vs this=0.4170 (var=0.0576)
- Experiment 2: real=0.6600 (var=0.0377) vs this=0.6763 (var=0.0203)
- Experiment 3: real=0.3862 (var=0.0372) vs this=0.4656 (var=0.0213)
- Experiment 4: real=0.6225 (var=0.0889) vs this=0.8100 (var=0.0256)
- Experiment 5: real=0.6217 (var=0.0568) vs this=0.5975 (var=0.0257)
- Experiment 6: real=0.6025 (var=0.0898) vs this=0.4706 (var=0.0862)
- Experiment 7: real=0.2200 (var=0.1117) vs this=0.2816 (var=0.0715)
- Experiment 8: real=0.0211 (var=0.2810) vs this=0.0463 (var=0.2994)


---

### `pi_4` (overall score: 0.496)

**Description**
Heuristic Mixture (Strategy Selection) Theory: Decision-makers do not rely on a single strategy; instead, they probabilistically select between different heuristics on each trial. Specifically, they mix between a non-compensatory, lexicographic strategy (Take-The-Best) and a non-weighted, compensatory strategy (Tallying). A mixture parameter governs the probability of using TTB versus Tallying, allowing the model to capture intermediate choice behaviors by blending one-reason decision making with simple feature-counting.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take-The-Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        ttb_scores = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        ttb_scores = np.array([0.0, 1.0])
    else:
        ttb_scores = np.array([0.0, 0.0])
        
    beta = float(parameters["beta"])
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Strategy Mixture
    p_mix = float(parameters["p_mix"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_mix * p_ttb + (1.0 - p_mix) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.25]
- p_mix: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3520 (var=0.0355) vs this=0.4920 (var=0.0537)
- Experiment 2: real=0.6600 (var=0.0377) vs this=0.5563 (var=0.0670)
- Experiment 3: real=0.3862 (var=0.0372) vs this=0.5700 (var=0.0501)
- Experiment 4: real=0.6225 (var=0.0889) vs this=0.7338 (var=0.0340)
- Experiment 5: real=0.6217 (var=0.0568) vs this=0.2908 (var=0.0232)
- Experiment 6: real=0.6025 (var=0.0898) vs this=0.9294 (var=0.0055)
- Experiment 7: real=0.2200 (var=0.1117) vs this=0.0132 (var=0.0121)
- Experiment 8: real=0.0211 (var=0.2810) vs this=0.0305 (var=0.3155)


---

### `pi_2` (overall score: 0.437)

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
- Experiment 1: real=0.3520 (var=0.0355) vs this=0.1600 (var=0.0102)
- Experiment 2: real=0.6600 (var=0.0377) vs this=0.8627 (var=0.0063)
- Experiment 3: real=0.3862 (var=0.0372) vs this=0.2577 (var=0.0061)
- Experiment 4: real=0.6225 (var=0.0889) vs this=0.5112 (var=0.0187)
- Experiment 5: real=0.6217 (var=0.0568) vs this=0.4850 (var=0.0101)
- Experiment 6: real=0.6025 (var=0.0898) vs this=0.8750 (var=0.0084)
- Experiment 7: real=0.2200 (var=0.1117) vs this=0.0163 (var=0.0068)
- Experiment 8: real=0.0211 (var=0.2810) vs this=-0.7379 (var=0.0311)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3658 -> ACCEPTED
- iter 2: loss=0.3624 -> ACCEPTED
- iter 3: loss=0.3757 -> REJECTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.3624 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        a_wins = sum(1 for i in range(4) if a[i] > b[i])
        b_wins = sum(1 for i in range(4) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if resp == ttb_winner:
                ttb_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_consistent / total)

```

**Observed (real) value:** 0.3520 (var=0.0355)
**Previous candidate values (this loop):**
  - iter 1: 0.5407 (var=0.0854) (Δ vs real +0.1887)
  - iter 2: 0.5917 (var=0.0703) (Δ vs real +0.2397)
  - iter 3 (most recent): 0.5357 (var=0.0747) (Δ vs real +0.1837)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8683 (var=0.0089)
- pi_2: 0.1600 (var=0.0102)
- pi_3: 0.4170 (var=0.0576)
- pi_4: 0.4920 (var=0.0537)
- pi_5: 0.5613 (var=0.0028)

### Experiment 2
**Design**
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Fixed validities from the experimental design
    val = np.array([0.65, 0.95, 0.55, 0.75, 0.85])
    cue_order = np.argsort(-val, kind='stable').tolist()
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue  # Tallying predicts a tie, skip
            
        # TTB prediction
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        # Only consider trials where the two heuristics make STRICTLY OPPOSITE predictions
        if ttb_pred is not None and tally_pred != ttb_pred:
            matches.append(1.0 if row['response'] == tally_pred else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.6600 (var=0.0377)
**Previous candidate values (this loop):**
  - iter 1: 0.4070 (var=0.0647) (Δ vs real -0.2530)
  - iter 2: 0.4067 (var=0.0796) (Δ vs real -0.2533)
  - iter 3 (most recent): 0.4257 (var=0.0849) (Δ vs real -0.2343)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8627 (var=0.0063)
- pi_1: 0.1273 (var=0.0083)
- pi_3: 0.6763 (var=0.0203)
- pi_4: 0.5563 (var=0.0670)
- pi_5: 0.6290 (var=0.0061)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_has_f0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_f0 = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    chose_f0 = (a_has_f0 & chose_a) | (b_has_f0 & chose_b)
    return float(chose_f0.mean())
```

**Observed (real) value:** 0.3862 (var=0.0372)
**Previous candidate values (this loop):**
  - iter 1: 0.7019 (var=0.0512) (Δ vs real +0.3156)
  - iter 2: 0.6646 (var=0.0503) (Δ vs real +0.2783)
  - iter 3 (most recent): 0.6602 (var=0.0427) (Δ vs real +0.2740)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4656 (var=0.0213)
- pi_2: 0.2577 (var=0.0061)
- pi_1: 0.8477 (var=0.0145)
- pi_4: 0.5700 (var=0.0501)
- pi_5: 0.4963 (var=0.0037)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 3 where A=[1, 0, 1, 0, 0] and B=[0, 1, 0, 1, 0]
    # This is the only trial where the sum of features is 2 for both options.
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    mask = (a_sums == 2) & (b_sums == 2)
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.6225 (var=0.0889)
**Previous candidate values (this loop):**
  - iter 1: 0.8000 (var=0.0362) (Δ vs real +0.1775)
  - iter 2: 0.7837 (var=0.0205) (Δ vs real +0.1612)
  - iter 3 (most recent): 0.8488 (var=0.0120) (Δ vs real +0.2262)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5112 (var=0.0187)
- pi_3: 0.8100 (var=0.0256)
- pi_1: 0.8475 (var=0.0247)
- pi_4: 0.7338 (var=0.0340)
- pi_5: 0.7863 (var=0.0247)

### Experiment 5
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify the tie-breaker trials (Trials 1 and 6) where Option A is [0, 1, 1, 0, 0]
    is_target = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (0, 1, 1, 0, 0))
    if not is_target.any():
        return 0.5
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[is_target, 'response'] == 0))
```

**Observed (real) value:** 0.6217 (var=0.0568)
**Previous candidate values (this loop):**
  - iter 1: 0.3025 (var=0.0389) (Δ vs real -0.3192)
  - iter 2: 0.3142 (var=0.0470) (Δ vs real -0.3075)
  - iter 3 (most recent): 0.3575 (var=0.0442) (Δ vs real -0.2642)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5975 (var=0.0257)
- pi_4: 0.2908 (var=0.0232)
- pi_1: 0.1517 (var=0.0103)
- pi_2: 0.4850 (var=0.0101)
- pi_5: 0.7117 (var=0.0234)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10011') & (b_str == '01100')
    t2_mask = (a_str == '01100') & (b_str == '10011')
    
    t1_choices = data.loc[t1_mask, 'response'] == 0
    t2_choices = data.loc[t2_mask, 'response'] == 1
    
    total_target_choices = t1_choices.sum() + t2_choices.sum()
    total_trials = t1_mask.sum() + t2_mask.sum()
    
    if total_trials == 0:
        return 0.5
    return float(total_target_choices / total_trials)
```

**Observed (real) value:** 0.6025 (var=0.0898)
**Previous candidate values (this loop):**
  - iter 1: 0.8300 (var=0.0163) (Δ vs real +0.2275)
  - iter 2: 0.8187 (var=0.0153) (Δ vs real +0.2162)
  - iter 3 (most recent): 0.8137 (var=0.0146) (Δ vs real +0.2112)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9294 (var=0.0055)
- pi_3: 0.4706 (var=0.0862)
- pi_1: 0.8731 (var=0.0090)
- pi_2: 0.8750 (var=0.0084)
- pi_5: 0.2062 (var=0.0213)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1: Target has cues [1, 1, 0, 0, 0], Opposing has [0, 0, 1, 1, 1]
    is_t1_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Identify Trial 2: Target has cue [1, 0, 0, 0, 0], Opposing has [0, 1, 1, 0, 0]
    is_t2_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    is_t2_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    
    # Calculate P(Target | T1)
    t1_target_choices = (is_t1_fwd & (data['response'] == 0)).sum() + (is_t1_rev & (data['response'] == 1)).sum()
    t1_total = is_t1_fwd.sum() + is_t1_rev.sum()
    p_t1 = float(t1_target_choices / t1_total) if t1_total > 0 else 0.5
    
    # Calculate P(Target | T2)
    t2_target_choices = (is_t2_fwd & (data['response'] == 0)).sum() + (is_t2_rev & (data['response'] == 1)).sum()
    t2_total = is_t2_fwd.sum() + is_t2_rev.sum()
    p_t2 = float(t2_target_choices / t2_total) if t2_total > 0 else 0.5
    
    # Return the simple difference between the two choice probabilities
    return p_t1 - p_t2

```

**Observed (real) value:** 0.2200 (var=0.1117)
**Previous candidate values (this loop):**
  - iter 1: 0.1453 (var=0.0654) (Δ vs real -0.0747)
  - iter 2: 0.1111 (var=0.0334) (Δ vs real -0.1089)
  - iter 3 (most recent): 0.0779 (var=0.0324) (Δ vs real -0.1421)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2816 (var=0.0715)
- pi_5: 0.4916 (var=0.0813)
- pi_1: 0.0226 (var=0.0071)
- pi_2: 0.0163 (var=0.0068)
- pi_4: 0.0132 (var=0.0121)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    # Trial 2: Option A is [1, 0, 0, 0, 0], Option B is [0, 1, 0, 0, 1]
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    # Trial 4: Option A is [0, 1, 0, 1, 1], Option B is [1, 0, 1, 0, 0]
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 1))
    
    t2_resp = data.loc[t2_mask, 'response'].mean()
    t4_resp = data.loc[t4_mask, 'response'].mean()
    
    if pd.isna(t2_resp) or pd.isna(t4_resp):
        return 0.0
        
    # Response is 1 if Option B is chosen, 0 if Option A is chosen.
    return float(t4_resp - t2_resp)
```

**Observed (real) value:** 0.0211 (var=0.2810)
**Previous candidate values (this loop):**
  - iter 1: 0.3189 (var=0.3083) (Δ vs real +0.2979)
  - iter 2: 0.2874 (var=0.3204) (Δ vs real +0.2663)
  - iter 3 (most recent): 0.3474 (var=0.3539) (Δ vs real +0.3263)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5779 (var=0.0844)
- pi_3: 0.0463 (var=0.2994)
- pi_1: 0.6674 (var=0.0579)
- pi_2: -0.7379 (var=0.0311)
- pi_4: 0.0305 (var=0.3155)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Rank-Based Weighting with Directional Cues: Decision-makers evaluate cues based on their informational value (distance from 0.5 validity) rather than raw validity. They sort cues by this importance and assign weights that decay exponentially based on ordinal rank (w = direction * alpha ^ rank). This approach captures bounded rationality by discarding precise probability information in favor of a simpler ordinal hierarchy, while correctly accounting for anti-correlated cues (validity < 0.5) by flipping their sign.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute cue importance (distance from 0.5) and direction
    importance = np.abs(val - 0.5)
    direction = np.sign(val - 0.5)
    
    # Calculate ranks (0 is highest importance, 1 is second highest, etc.)
    order = np.argsort(-importance, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(val))
    
    alpha = float(parameters["alpha"])
    # Compute weights: direction * alpha^rank. 
    weights = direction * (alpha ** ranks)
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Addressed the critic's feedback by modifying the ranking logic to sort cues by their importance (absolute distance from 0.5) rather than raw validity. Cues with validities < 0.5 are highly predictive of the opposing option, so they now correctly receive a high rank but a negative weight. This minimal edit fixes the model's failure on experiments with anti-correlated cues while preserving the core Rank-Based Weighting mechanism.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The Rank-Based Weighting model fits poorly on several experiments (e.g., Exp 2, 5, 8). For instance, in Experiment 5, the model predicts a choice proportion of 0.3025 whereas human data shows 0.6217. A core limitation of the current implementation is that it sorts cues strictly by raw validities descending, completely failing to account for cues with validities < 0.5. In binary choice tasks, a validity of 0.2 is actually highly informative (it strongly predicts the *other* option), meaning its effective rank should be high and its weight should be negative (or the cue should be flipped).
Rationale: The candidate implements the prescribed Rank-Based Weighting family but misses a critical detail of cue processing: validities < 0.5. To fix this while staying strictly within the arbiter's recommended family, make the following minor adjustments: 1) Compute cue importance as `abs(val - 0.5)`. 2) Sort cues by this importance (descending) to assign the ordinal ranks. 3) Compute the final weights by multiplying the rank-based decay by the direction of the cue: `weight = np.sign(val - 0.5) * (alpha ** ranks)`. This ensures that highly predictive but anti-correlated cues receive appropriately high rank and correctly penalize the options that possess them.

**Outcome of this advice:** iter 1 candidate loss=0.3658 -> iter 2 candidate loss=0.3624 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The model's performance improved after correctly handling cue directionality (loss dropped to 0.3624 and was ACCEPTED). However, it still struggles with several experiments, most notably Experiment 2 (0.4067 vs real 0.6600) and Experiment 5 (0.3142 vs real 0.6217). A potential issue lies in how ranks are assigned: by using `np.argsort`, cues with identical importance (e.g., validities of 0.8 and 0.2 both have an absolute distance of 0.3 from 0.5) are arbitrarily assigned different sequential ranks. This breaks the symmetry of equally informative cues and introduces arbitrary weighting biases.
Rationale: The candidate successfully incorporated directional cue importance, which improved the fit and was accepted by the gate. To further refine this within the prescribed Rank-Based Weighting family, the ranking mechanism must handle tied importances correctly. Currently, `np.argsort` forces a strict ordering even for ties. Modify the rank computation to assign the *same* rank to cues with equal importance. You can achieve this by using `scipy.stats.rankdata` with `method='dense'` (or `method='min'`), ensuring that equally predictive cues receive the exact same weight. Keep the rest of the logic intact.

**Outcome of this advice:** iter 2 candidate loss=0.3624 -> iter 3 candidate loss=0.3757 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The attempt to use dense ranking for ties worsened the overall loss (0.3757 vs 0.3624) and was rejected by the gate. The model continues to underperform on Experiment 2 and Experiment 5. The rejection suggests that either strict sequential ranking (even with arbitrary tie-breaking) is empirically a better fit, or that the dense ranking fundamentally distorted the weight distribution. Since the loop will automatically revert to the accepted Iteration 2 base, we should explore a different dimension of the rank-based weighting family: the functional form of the decay.
Rationale: The previous recommendation to use dense ranking for ties was rejected by the gate, so the loop will revert to the Iteration 2 base (which uses `np.argsort`). Building on that accepted base, let's change the shape of the rank-based decay. The current exponential decay (`alpha ** rank`) might drop off too aggressively or not smoothly enough to capture the compensatory behavior seen in Exp 2 and 5. Instead, implement a power-law decay: compute weights as `weights = direction * (1.0 / (ranks + 1.0) ** alpha)`. To support this, change the `alpha` parameter range to `[0.0, 5.0]`, allowing the model to fit both very flat and very steep power-law decays. Keep the directional importance (`abs(val - 0.5)`) logic intact.

**Outcome of this advice:** iter 3 candidate loss=0.3757 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Rank-Based Weighting with Power-Law Decay: Decision-makers evaluate cues based on their informational value (distance from 0.5 validity) rather than raw validity. They sort cues by this importance and assign weights that decay according to a power law based on ordinal rank (w = direction * (1 / (rank + 1)^alpha)). This approach captures bounded rationality by discarding precise probability information in favor of a simpler ordinal hierarchy, while correctly accounting for anti-correlated cues and providing a flexible decay shape that can model both compensatory and non-compensatory choice patterns.",
  "predict": "import numpy as np\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Compute cue importance (distance from 0.5) and direction\n    importance = np.abs(val - 0.5)\n    direction = np.sign(val - 0.5)\n    \n    # Calculate ranks (0 is highest importance, 1 is second highest, etc.)\n    order = np.argsort(-importance, kind=\"stable\")\n    ranks = np.empty_like(order)\n    ranks[order] = np.arange(len(val))\n    \n    alpha = float(parameters[\"alpha\"])\n    # Compute weights using power-law decay\n    weights = direction * (1.0 / (ranks + 1.0) ** alpha)\n    \n    score_a = np.sum(a * weights)\n    score_b = np.sum(b * weights)\n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax choice rule with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "import numpy as np\n\ndef policy(probs):\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities = probabilities / np.sum(probabilities)\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "alpha": "[0.0, 5.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Changed the rank-based decay function from exponential to a power-law decay, as suggested by the critic. This allows for a wider range of decay shapes (from very flat to very steep) to better capture the compensatory behavior observed in the experiments. Expanded the alpha parameter range to [0.0, 5.0] to accommodate this new functional form, while keeping the directional importance logic from the previously accepted iteration intact."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14779,
  "candidates_token_count": 822,
  "total_token_count": 16213
}
```
