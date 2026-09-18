# generation_iter_00_attempt_00

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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a theory of adaptive or conditional decision-making that blends validity-ordered cascade with equal-weight tallying based on task structure. The key insight from the data: subjects default to Tallying (equal-weight feature counting) when the validity structure is ambiguous or when features are roughly balanced in discriminative power (Experiment 1). However, when the task explicitly creates a sharp validity cliff (Experiment 2: validities [0.95, 0.90, ...] with a large drop to 0.65), subjects partially engage in validity-aware cascade reasoning, though with substantial heterogeneity. The new theory should specify: (1) A primary mechanism: Tallying with equal-weight feature counting as the baseline heuristic. (2) A triggering condition for validity-ordered cascade: When the ratio or gap between top-validity cues and lower cues exceeds a threshold (e.g., >0.25 validity gap), subjects shift partially toward TTB-like cascade. (3) A mixing parameter α that interpolates between Tallying (α=0) and TTB (α=1), modulated by the validity structure. (4) Uncertainty or noise in the validity perception: subjects may not perfectly learn or apply stated validities, introducing heterogeneity. This theory would predict: Experiment 1 (validities [0.9, 0.7, 0.6, 0.55], gaps ≤0.3): α ≈ 0, behavior ≈ Tallying (observed 0.40 ✓). Experiment 2 (validities [0.95, 0.90, 0.65, ...], sharp 0.25 gap): α partially elevated, behavior intermediate between Tallying and TTB (observed 0.23, between -0.44 and 0.46 ✓, with higher variance due to heterogeneous α). The new theory should be called something like 'Adaptive Validity-Aware Tallying' or 'Conditional Cascade-Tallying Hybrid' and should include a principled mechanism for when and how much subjects weight validity ordering versus equal aggregation.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.000)

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
- Experiment 1: real=0.4012 (var=0.0172) vs this=0.8472 (var=0.0098)
- Experiment 2: real=0.2300 (var=0.2070) vs this=0.4587 (var=0.0220)


---

### `pi_2` (overall score: 0.000)

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
- Experiment 1: real=0.4012 (var=0.0172) vs this=0.4122 (var=0.0053)
- Experiment 2: real=0.2300 (var=0.2070) vs this=-0.4417 (var=0.0226)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Metric: Preference for high-validity-cue-favoring choice on conflict trials.
    # TTB should strongly prefer choices where the highest-validity cue (feature 0)
    # discriminates, even when lower-validity cues favor the opposite. Tallying
    # treats all cues equally, so it should show much weaker preference for the
    # high-validity-cue winner on these conflict trials.
    #
    # Strategy: identify trials where feature 0 discriminates (A vs B differ on
    # feature 0) AND the majority of features 1-3 favor the opposite option.
    # On such trials, TTB picks the feature-0 winner; Tallying often picks the
    # opposite (majority of 3 lower cues). We measure the fraction of conflict
    # trials where the subject chose the feature-0 winner.
    #
    # This metric is tight per subject (binary choice on a fixed trial set)
    # and has a large mean gap between theories.
    
    conflict_trials = []
    
    for idx, row in data.iterrows():
        a_ratings = np.asarray(row['option_a_ratings'], dtype=int)
        b_ratings = np.asarray(row['option_b_ratings'], dtype=int)
        response = int(row['response'])
        
        # Check if feature 0 discriminates
        if a_ratings[0] == b_ratings[0]:
            continue
        
        # Determine which option wins on feature 0
        feature_0_winner = 0 if a_ratings[0] > b_ratings[0] else 1
        
        # Count wins on features 1-3
        a_wins_lower = np.sum(a_ratings[1:] > b_ratings[1:])
        b_wins_lower = np.sum(b_ratings[1:] > a_ratings[1:])
        
        # Conflict trial: feature 0 favors one option, majority of features 1-3 favor the other
        # We need strict majority (at least 2 out of 3) of lower cues to favor the opposite
        if a_wins_lower > b_wins_lower and feature_0_winner == 1:
            # Feature 0 favors B, but A wins on majority of lower cues
            conflict_trials.append((1, response))  # TTB predicts B (1), Tallying often predicts A (0)
        elif b_wins_lower > a_wins_lower and feature_0_winner == 0:
            # Feature 0 favors A, but B wins on majority of lower cues
            conflict_trials.append((0, response))  # TTB predicts A (0), Tallying often predicts B (1)
    
    if len(conflict_trials) == 0:
        return 0.5  # No conflict trials; neutral
    
    # Compute fraction of conflict trials where subject chose the feature-0 winner
    # (i.e., the TTB prediction)
    ttb_correct = sum(1 for predicted, actual in conflict_trials if predicted == actual)
    return float(ttb_correct) / len(conflict_trials)
```

**Observed (real) value:** 0.4012 (var=0.0172)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8472 (var=0.0098)
- pi_2: 0.4122 (var=0.0053)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Measure the degree to which subjects employ a validity-ordered cascade
    (TTB) versus equal-weight tallying (Tallying).
    
    Key insight: On trials where the two highest-validity features (0, 1) tie
    between options, TTB must consult feature 2 (validity 0.65) to break the tie,
    while Tallying counts all features equally. This creates a sharp dissociation:
    - TTB predicts choice based on feature 2 after tie-breaking at features 0-1.
    - Tallying predicts choice based on the majority vote across all features.
    
    We compute a "cascade fidelity" score: the proportion of trials where the
    subject's choice aligns with TTB's prediction (given the validity cascade)
    minus the proportion where it aligns with Tallying's prediction.
    
    On the critical tie-breaking trials (1-4, 7-8), TTB and Tallying diverge.
    On control trials (9-12), both theories largely agree. On low-validity trials
    (5-6), both theories agree. The metric isolates the cascade effect.
    
    Validities: [0.95, 0.90, 0.65, 0.60, 0.55]
    """
    validities = np.array([0.95, 0.90, 0.65, 0.60, 0.55])
    
    # Define the 12 trial pairs (from the experimental design)
    trial_pairs = [
        ([1, 1, 1, 0, 0], [1, 1, 0, 1, 1]),  # trial 1
        ([1, 1, 0, 1, 1], [1, 1, 1, 0, 0]),  # trial 2
        ([1, 1, 1, 0, 0], [1, 1, 0, 1, 1]),  # trial 3
        ([1, 1, 0, 1, 1], [1, 1, 1, 0, 0]),  # trial 4
        ([0, 0, 1, 0, 0], [0, 0, 0, 1, 1]),  # trial 5
        ([0, 0, 0, 1, 1], [0, 0, 1, 0, 0]),  # trial 6
        ([0, 0, 1, 0, 0], [0, 0, 0, 1, 1]),  # trial 7
        ([0, 0, 0, 1, 1], [0, 0, 1, 0, 0]),  # trial 8
        ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),  # trial 9
        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),  # trial 10
        ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),  # trial 11
        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),  # trial 12
    ]
    
    # Helper: compute TTB prediction (0 = choose A, 1 = choose B, None = tie)
    def ttb_predict(a, b, vals):
        cue_order = np.argsort(-vals, kind='stable')
        for j in cue_order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None  # All tied
    
    # Helper: compute Tallying prediction (0 = choose A, 1 = choose B, None = tie)
    def tallying_predict(a, b):
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(b)) if b[i] > a[i])
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        else:
            return None  # Tied
    
    # Count agreement with each theory, excluding ties
    ttb_matches = 0
    tallying_matches = 0
    valid_trials = 0
    
    for idx, row in data.iterrows():
        a = np.array(row['option_a_ratings'], dtype=int)
        b = np.array(row['option_b_ratings'], dtype=int)
        response = int(row['response'])
        
        ttb_pred = ttb_predict(a, b, validities)
        tallying_pred = tallying_predict(a, b)
        
        # Only count trials where both theories make a definite prediction
        if ttb_pred is not None and tallying_pred is not None:
            valid_trials += 1
            if response == ttb_pred:
                ttb_matches += 1
            if response == tallying_pred:
                tallying_matches += 1
    
    if valid_trials == 0:
        return 0.0
    
    # Cascade fidelity: TTB agreement rate minus Tallying agreement rate
    # Positive values favor TTB; negative values favor Tallying.
    ttb_rate = ttb_matches / valid_trials
    tallying_rate = tallying_matches / valid_trials
    
    return ttb_rate - tallying_rate
```

**Observed (real) value:** 0.2300 (var=0.2070)
**Other theories' values on this metric (for reference):**
- pi_2: -0.4417 (var=0.0226)
- pi_1: 0.4587 (var=0.0220)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

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
  "theory": "Adaptive Validity-Aware Tallying with Conditional Cascade Engagement. Subjects default to equal-weight feature tallying (counting wins across all features) as their primary heuristic. However, when the task structure creates a sharp validity cliff—defined by the maximum gap between adjacent validity-ordered features exceeding a learned threshold—subjects partially shift toward validity-ordered cascade reasoning (Take The Best). The degree of shift is controlled by a mixing parameter α that interpolates between pure Tallying (α=0) and pure TTB (α=1). The mixing parameter is modulated by the maximum validity gap: when gap > threshold, α increases; when gap ≤ threshold, α ≈ 0. Subjects also exhibit uncertainty in their perception of stated validities, introducing individual heterogeneity in both the threshold and the strength of the cascade engagement. This theory predicts: Experiment 1 (validities [0.9, 0.7, 0.6, 0.55], max gap 0.2): max_gap < threshold, so α ≈ 0, behavior ≈ pure Tallying (observed 0.40 ✓). Experiment 2 (validities [0.95, 0.90, 0.65, ...], gap 0.25 between features 1 and 2): max_gap > threshold for many subjects, so α partially elevated, behavior intermediate between Tallying and TTB (observed 0.23, consistent with mixed strategy ✓, with higher variance due to heterogeneous threshold and α_max).",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    from scipy.special import softmax\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Adaptive Validity-Aware Tallying expects (2, n_features) stimulus; got {stim.shape}.\"\n        )\n    \n    n_features = stim.shape[1]\n    a, b = stim[0], stim[1]\n    \n    # Extract validities and parameters\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {validities.shape[0]} != n_features {n_features}.\"\n        )\n    \n    gap_threshold = float(parameters[\"gap_threshold\"])\n    alpha_max = float(parameters[\"alpha_max\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Compute the maximum gap between consecutive validity-ordered features\n    sorted_vals = np.sort(validities)[::-1]  # Descending order\n    max_gap = 0.0\n    for i in range(len(sorted_vals) - 1):\n        gap = sorted_vals[i] - sorted_vals[i + 1]\n        max_gap = max(max_gap, gap)\n    \n    # Modulate alpha based on max_gap relative to threshold\n    # Linear interpolation: alpha = 0 when max_gap <= threshold,\n    # alpha = alpha_max when max_gap >= threshold + 0.1 (or some margin)\n    if max_gap <= gap_threshold:\n        alpha = 0.0\n    else:\n        # Smooth ramp from threshold to threshold + margin\n        margin = 0.15\n        alpha = min(alpha_max, (max_gap - gap_threshold) / margin)\n    \n    # --- Tallying component ---\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    tallying_scores = np.array([a_wins, b_wins])\n    \n    # --- TTB component ---\n    cue_order = np.argsort(-validities, kind=\"stable\").tolist()\n    ttb_winner = None\n    for j in cue_order:\n        if a[j] > b[j]:\n            ttb_winner = 0\n            break\n        if b[j] > a[j]:\n            ttb_winner = 1\n            break\n    \n    if ttb_winner is None:\n        ttb_scores = np.ones(2) / 2.0  # Tie: uniform guess\n    else:\n        ttb_scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])\n    \n    # --- Blend: alpha interpolates between Tallying and TTB ---\n    # alpha = 0 -> pure Tallying; alpha = 1 -> pure TTB\n    blended_scores = (1.0 - alpha) * tallying_scores + alpha * ttb_scores\n    \n    # --- Softmax with noise ---\n    z = beta * (blended_scores - blended_scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n    \n    n_opts = p_core.shape[0]\n    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n    \n    return p_final",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "gap_threshold": "[0.15, 0.35]",
    "alpha_max": "[0.0, 1.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "This theory directly implements the arbiter's prescription for an adaptive validity-aware tallying system. Key innovations:\n\n1. **Primary Mechanism (Tallying Baseline)**: The model defaults to equal-weight feature tallying—counting wins across all features—which explains Experiment 1's observed behavior (0.40) where validities are relatively balanced (gaps ≤ 0.2). Pure Tallying predicts 0.4122 on Experiment 1, close to observed 0.4012.\n\n2. **Triggering Condition (Validity Cliff Detection)**: The model computes the maximum gap between consecutively validity-ordered features. When this gap exceeds a learned threshold (gap_threshold), subjects engage in validity-ordered cascade reasoning. Experiment 2 has validities [0.95, 0.90, 0.65, ...] with a 0.25 gap between features 1 and 2, which exceeds typical thresholds, triggering cascade engagement.\n\n3. **Mixing Parameter (α)**: A continuous interpolation between Tallying (α=0) and TTB (α=1) is modulated by the validity gap. The blended score is `(1-α) * tallying_score + α * ttb_score`. This allows intermediate behavior: on Experiment 2, if α ≈ 0.5 on average, the model produces predictions between pure Tallying (-0.44) and pure TTB (0.46), consistent with observed 0.23 and the higher variance (var=0.2070) reflecting heterogeneous α across subjects.\n\n4. **Heterogeneity via Parameter Learning**: Both gap_threshold and alpha_max are free parameters sampled per subject, capturing the observed high variance in Experiment 2 (var=0.2070 vs var=0.0053 for pure Tallying). Some subjects may have low thresholds (cascade easily), others high thresholds (stick to Tallying), and some may cap α_max low (weak cascade engagement).\n\n5. **Noise and Softmax**: Response noise enters through softmax over the blended scores (beta controls temperature) plus lapse rate epsilon, consistent with both prior theories.\n\n**Why This Overcomes Prior Failures**:\n- `pi_1` (TTB) scored 0.8472 on Exp 1 (far too high, predicting strong cascade) and 0.4587 on Exp 2 (too high, not capturing the heterogeneity and intermediate behavior). The new theory fixes this by making cascade engagement conditional on validity structure and subject-specific.\n- `pi_2` (pure Tallying) scored 0.4122 on Exp 1 (close, good) but -0.4417 on Exp 2 (far too negative, missing the cascade shift). The new theory captures the partial shift toward TTB when validity gaps are large, landing near 0.23.\n- The adaptive mechanism respects the observation that subjects are sensitive to task structure (validity gaps) but do not fully commit to either strategy, maintaining a flexible blend that varies across subjects and contexts."
}
```
```

## Usage

```json
{
  "input_tokens": 8221,
  "output_tokens": 2256
}
```
