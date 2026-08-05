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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Propose a 'Sequential Evidence Accumulation' or 'Probabilistic Cue Integration' theory. Instead of a deterministic non-linear weighting (like the current WADD) or a strict mixture of two rigid heuristics (like Strategy Mixture), this theory should assume that decision-makers accumulate evidence by sampling cues probabilistically based on their validities. If a threshold is reached, a decision is made; otherwise, it falls back to a noisy integration of all sampled cues. This would naturally produce the ~0.50 choice proportions on conflict trials (mimicking Strategy Mixture) while allowing sensitivity to cue clustering (fixing the failure on Exp 2).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.824)

**Description**
Non-linear Weighted Additive (WADD) Theory with Power Weights and Max-Normalized Validities: Decision-makers integrate all available cues simultaneously in a compensatory manner. The subjective weight assigned to each cue is determined by a power transformation of its objective validity, scaled by a parameter gamma. To ensure gamma operates consistently across experiments with varying validity scales, the objective validities are first normalized by their maximum value. This stable, bounded non-linear mapping ensures that cues with zero validity receive zero subjective weight, while allowing the model to naturally balance the top cue against the remaining cues on conflict trials. A temperature-parameterized softmax choice rule is used to smoothly control stochasticity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Max-normalize validities to make gamma scale-invariant across experiments
    v_norm = validities / np.max(validities)
    
    # Power transformation of normalized validities to subjective weights
    weights = v_norm ** gamma
    weights = weights / np.sum(weights)
    
    # Compute weighted sum of features for each option
    scores = stim @ weights
    
    # Standard softmax choice rule with temperature
    z = scores / tau
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- tau: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 20.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4867 (var=0.0082) vs this=0.4275 (var=0.0242)
- Experiment 2: real=0.5089 (var=0.0125) vs this=0.5344 (var=0.0115)
- Experiment 3: real=0.5121 (var=0.0035) vs this=0.5342 (var=0.0057)
- Experiment 4: real=0.5188 (var=0.0033) vs this=0.5587 (var=0.0073)
- Experiment 5: real=0.1010 (var=0.0012) vs this=0.1458 (var=0.0063)
- Experiment 6: real=-0.0084 (var=0.0161) vs this=0.0642 (var=0.0186)
- Experiment 7: real=0.0673 (var=0.0007) vs this=0.1089 (var=0.0153)
- Experiment 8: real=0.0075 (var=0.0276) vs this=0.0887 (var=0.0389)
- Experiment 9: real=0.0172 (var=0.0001) vs this=0.0183 (var=0.0001)
- Experiment 10: real=-0.0300 (var=0.0316) vs this=0.0687 (var=0.0289)


---

### `pi_4` (overall score: 0.709)

**Description**
Strategy Mixture Theory: Instead of relying on a single heuristic, decision-makers draw from a repertoire of strategies on a trial-by-trial basis. Specifically, individuals mix between a non-compensatory lexicographic rule (Take-The-Best) and a compensatory rule (Tallying). On any given trial, a subject employs TTB with probability `p_ttb` and Tallying with probability `1 - p_ttb`. This intra-individual strategy variation naturally accounts for the aggregate ~0.50 choice proportions observed in conflict trials where the two heuristics prescribe different options, while a relatively stable mixture proportion across the population explains the low between-subject variance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (TTB)
    order = np.argsort(validities)[::-1]
    score_ttb = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            score_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            score_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    score_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Softmax for TTB
    z_ttb = beta * score_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    prob_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * score_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    prob_tally = e_tally / np.sum(e_tally)
    
    # Mix the two strategies
    p_core = p_ttb * prob_ttb + (1.0 - p_ttb) * prob_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.4, 0.6]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4867 (var=0.0082) vs this=0.4908 (var=0.0117)
- Experiment 2: real=0.5089 (var=0.0125) vs this=0.5067 (var=0.0155)
- Experiment 3: real=0.5121 (var=0.0035) vs this=0.5231 (var=0.0036)
- Experiment 4: real=0.5188 (var=0.0033) vs this=0.5460 (var=0.0047)
- Experiment 5: real=0.1010 (var=0.0012) vs this=0.1108 (var=0.0011)
- Experiment 6: real=-0.0084 (var=0.0161) vs this=0.0032 (var=0.0287)
- Experiment 7: real=0.0673 (var=0.0007) vs this=0.0774 (var=0.0020)
- Experiment 8: real=0.0075 (var=0.0276) vs this=0.0113 (var=0.0353)
- Experiment 9: real=0.0172 (var=0.0001) vs this=0.0401 (var=0.0003)
- Experiment 10: real=-0.0300 (var=0.0316) vs this=-0.5325 (var=0.0410)


---

### `pi_2` (overall score: 0.267)

**Description**
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
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
- weights: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4867 (var=0.0082) vs this=0.5358 (var=0.1008)
- Experiment 2: real=0.5089 (var=0.0125) vs this=0.5700 (var=0.1165)
- Experiment 3: real=0.5121 (var=0.0035) vs this=0.3550 (var=0.0231)
- Experiment 4: real=0.5188 (var=0.0033) vs this=0.4254 (var=0.0137)
- Experiment 5: real=0.1010 (var=0.0012) vs this=0.2882 (var=0.0098)
- Experiment 6: real=-0.0084 (var=0.0161) vs this=0.2305 (var=0.1023)
- Experiment 7: real=0.0673 (var=0.0007) vs this=0.4926 (var=0.0429)
- Experiment 8: real=0.0075 (var=0.0276) vs this=0.2762 (var=0.1069)
- Experiment 9: real=0.0172 (var=0.0001) vs this=0.0839 (var=0.0028)
- Experiment 10: real=-0.0300 (var=0.0316) vs this=0.0438 (var=0.3439)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3349 -> ACCEPTED
- iter 2: loss=0.4131 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.3349 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f0 = np.array([x[0] for x in data['option_a_ratings']])
    a_f1 = np.array([x[1] for x in data['option_a_ratings']])
    b_f0 = np.array([x[0] for x in data['option_b_ratings']])
    b_f1 = np.array([x[1] for x in data['option_b_ratings']])
    
    a_sum2 = a_f0 + a_f1
    b_sum2 = b_f0 + b_f1
    
    # Identify critical trials (trials 1 and 2) where one option has the two highest 
    # validity features (sum=2) and the other has none of them (sum=0) but wins on the rest.
    mask = ((a_sum2 == 0) & (b_sum2 == 2)) | ((a_sum2 == 2) & (b_sum2 == 0))
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    a_sum2_rel = a_sum2[mask]
    
    # Tallying prefers the option with more features (which here means the one with 0 on the first two features)
    tally_chose_a = (a_sum2_rel == 0) & (responses == 0)
    tally_chose_b = (a_sum2_rel == 2) & (responses == 1)
    
    return float(np.mean(tally_chose_a | tally_chose_b))

```

**Observed (real) value:** 0.4867 (var=0.0082)
**Previous candidate values (this loop):**
  - iter 1: 0.2450 (var=0.0277) (Δ vs real -0.2417)
  - iter 2 (most recent): 0.6150 (var=0.0380) (Δ vs real +0.1283)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8483 (var=0.0104)
- pi_2: 0.5358 (var=0.1008)
- pi_3: 0.1608 (var=0.0119)
- pi_4: 0.4908 (var=0.0117)
- pi_5: 0.6233 (var=0.0676)
- pi_6: 0.4275 (var=0.0242)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if a[0] == 1 and a[1] == 1 and b[0] == 0 and b[1] == 0:
            if resp == 0:
                wadd_choices += 1
            total += 1
        elif a[0] == 0 and a[1] == 0 and b[0] == 1 and b[1] == 1:
            if resp == 1:
                wadd_choices += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(wadd_choices / total)

```

**Observed (real) value:** 0.5089 (var=0.0125)
**Previous candidate values (this loop):**
  - iter 1: 0.8133 (var=0.0180) (Δ vs real +0.3044)
  - iter 2 (most recent): 0.3600 (var=0.0573) (Δ vs real -0.1489)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5700 (var=0.1165)
- pi_1: 0.1611 (var=0.0137)
- pi_3: 0.8411 (var=0.0167)
- pi_4: 0.5067 (var=0.0155)
- pi_5: 0.3200 (var=0.0887)
- pi_6: 0.5344 (var=0.0115)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    return float(np.mean(data['response'] == 0))
```

**Observed (real) value:** 0.5121 (var=0.0035)
**Previous candidate values (this loop):**
  - iter 1: 0.5419 (var=0.0167) (Δ vs real +0.0298)
  - iter 2 (most recent): 0.2817 (var=0.0125) (Δ vs real -0.2304)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8408 (var=0.0088)
- pi_2: 0.3550 (var=0.0231)
- pi_1: 0.1950 (var=0.0070)
- pi_4: 0.5231 (var=0.0036)
- pi_5: 0.2629 (var=0.0095)
- pi_6: 0.5342 (var=0.0057)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        if pred != -1:
            matches.append(1 if r == pred else 0)
    return float(np.mean(matches)) if matches else 0.0
```

**Observed (real) value:** 0.5188 (var=0.0033)
**Previous candidate values (this loop):**
  - iter 1: 0.6112 (var=0.0132) (Δ vs real +0.0925)
  - iter 2 (most recent): 0.3429 (var=0.0088) (Δ vs real -0.1758)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4254 (var=0.0137)
- pi_3: 0.8435 (var=0.0120)
- pi_1: 0.2565 (var=0.0043)
- pi_4: 0.5460 (var=0.0047)
- pi_5: 0.3323 (var=0.0101)
- pi_6: 0.5587 (var=0.0073)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Create a hashable trial identifier
    data['trial_id'] = data.apply(
        lambda row: (tuple(row['option_a_ratings']), tuple(row['option_b_ratings'])), 
        axis=1
    )
    
    # Identify conflict trials where TTB and Tallying prescribe different options.
    # TTB relies on the first cue (index 0). Tallying relies on the sum of cues.
    def is_conflict(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_a = a[0] > b[0]
        ttb_b = b[0] > a[0]
        tally_a = sum(a) > sum(b)
        tally_b = sum(b) > sum(a)
        
        if ttb_a and tally_b:
            return True
        if ttb_b and tally_a:
            return True
        return False

    data['is_conflict'] = data.apply(is_conflict, axis=1)
    conflict_data = data[data['is_conflict']]
    
    if len(conflict_data) == 0:
        return 0.0
        
    # For each subject and each unique conflict trial, compute the choice proportion.
    # response == 0 means A, response == 1 means B. 
    # The mean of response is the proportion of B choices (p_B).
    # The absolute difference from 0.5 measures how deterministic the subject's choice is.
    grouped = conflict_data.groupby(['subject_id', 'trial_id'])['response'].mean().reset_index()
    grouped['extremity'] = (grouped['response'] - 0.5).abs()
    
    # Average the extremity of preferences per subject across all conflict trials,
    # then return the grand mean.
    subj_extremity = grouped.groupby('subject_id')['extremity'].mean()
    return float(subj_extremity.mean())

```

**Observed (real) value:** 0.1010 (var=0.0012)
**Previous candidate values (this loop):**
  - iter 1: 0.1918 (var=0.0028) (Δ vs real +0.0908)
  - iter 2 (most recent): 0.2433 (var=0.0106) (Δ vs real +0.1423)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1108 (var=0.0011)
- pi_2: 0.2882 (var=0.0098)
- pi_1: 0.3488 (var=0.0097)
- pi_3: 0.3578 (var=0.0067)
- pi_5: 0.2812 (var=0.0163)
- pi_6: 0.1458 (var=0.0063)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 2: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t2_mask = a_tuples == (1, 1, 0, 0, 0)
    # Trial 4: A=[0, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t4_mask = a_tuples == (0, 1, 0, 0, 0)
    
    if not t2_mask.any() or not t4_mask.any():
        return 0.0
        
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t4)
```

**Observed (real) value:** -0.0084 (var=0.0161)
**Previous candidate values (this loop):**
  - iter 1: 0.1326 (var=0.0169) (Δ vs real +0.1411)
  - iter 2 (most recent): 0.2011 (var=0.0414) (Δ vs real +0.2095)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2305 (var=0.1023)
- pi_4: 0.0032 (var=0.0287)
- pi_1: -0.0084 (var=0.0137)
- pi_3: 0.0189 (var=0.0159)
- pi_5: 0.1916 (var=0.0995)
- pi_6: 0.0642 (var=0.0186)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Identify conflict trials where TTB and Tallying point to different options
    conflict_a_strs = ['10000', '01110', '11000', '01011', '00111']
    df_conflict = data[data['a_str'].isin(conflict_a_strs)]
    
    if len(df_conflict) == 0:
        return 0.0
        
    # Calculate the mean response for each subject and each trial type
    grouped = df_conflict.groupby(['subject_id', 'a_str'])['response'].mean()
    
    # Calculate within-subject consistency: 4 * (p - 0.5)^2
    # This maps p=0.5 to 0.0 (coin flip) and p=0.0 or 1.0 to 1.0 (deterministic)
    consistency = 4.0 * ((grouped - 0.5) ** 2)
    
    return float(consistency.mean())
```

**Observed (real) value:** 0.0673 (var=0.0007)
**Previous candidate values (this loop):**
  - iter 1: 0.2441 (var=0.0334) (Δ vs real +0.1768)
  - iter 2 (most recent): 0.3797 (var=0.0633) (Δ vs real +0.3125)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0774 (var=0.0020)
- pi_5: 0.4415 (var=0.0609)
- pi_1: 0.6161 (var=0.0637)
- pi_2: 0.4926 (var=0.0429)
- pi_3: 0.5082 (var=0.0580)
- pi_6: 0.1089 (var=0.0153)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: TTB option is '10000', Tally option is '01110'
    t1_ttb_chosen = ((a_str == '10000') & (b_str == '01110') & (data['response'] == 0)) | \
                    ((a_str == '01110') & (b_str == '10000') & (data['response'] == 1))
    t1_trials = ((a_str == '10000') & (b_str == '01110')) | ((a_str == '01110') & (b_str == '10000'))
    
    # Trial 2: TTB option is '11000', Tally option is '00111'
    t2_ttb_chosen = ((a_str == '11000') & (b_str == '00111') & (data['response'] == 0)) | \
                    ((a_str == '00111') & (b_str == '11000') & (data['response'] == 1))
    t2_trials = ((a_str == '11000') & (b_str == '00111')) | ((a_str == '00111') & (b_str == '11000'))
    
    p1 = t1_ttb_chosen.sum() / t1_trials.sum() if t1_trials.sum() > 0 else 0.0
    p2 = t2_ttb_chosen.sum() / t2_trials.sum() if t2_trials.sum() > 0 else 0.0
    
    return float(p2 - p1)
```

**Observed (real) value:** 0.0075 (var=0.0276)
**Previous candidate values (this loop):**
  - iter 1: 0.2013 (var=0.0240) (Δ vs real +0.1938)
  - iter 2 (most recent): 0.0988 (var=0.0203) (Δ vs real +0.0913)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3337 (var=0.1490)
- pi_4: 0.0113 (var=0.0353)
- pi_1: 0.0312 (var=0.0155)
- pi_2: 0.2762 (var=0.1069)
- pi_3: 0.0225 (var=0.0153)
- pi_6: 0.0887 (var=0.0389)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    
    def get_ttb_option(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0
        
    df['ttb_pref'] = df.apply(get_ttb_option, axis=1)
    df['chose_ttb'] = (df['response'] == df['ttb_pref']).astype(float)
    df['trial_key'] = df.apply(lambda r: (tuple(r['option_a_ratings']), tuple(r['option_b_ratings'])), axis=1)
    
    props = df.groupby(['subject_id', 'trial_key'])['chose_ttb'].mean().reset_index()
    subj_vars = props.groupby('subject_id')['chose_ttb'].var(ddof=0)
    
    return float(subj_vars.mean())
```

**Observed (real) value:** 0.0172 (var=0.0001)
**Previous candidate values (this loop):**
  - iter 1: 0.0392 (var=0.0009) (Δ vs real +0.0220)
  - iter 2 (most recent): 0.0780 (var=0.0016) (Δ vs real +0.0609)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0401 (var=0.0003)
- pi_6: 0.0183 (var=0.0001)
- pi_1: 0.1107 (var=0.0022)
- pi_2: 0.0839 (var=0.0028)
- pi_3: 0.0090 (var=0.0001)
- pi_5: 0.0941 (var=0.0027)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: A=[1, 0, 0, 1, 1], B=[0, 1, 1, 0, 0]
    # WADD favors B (relies on cues 2 and 3 > cue 1 due to non-linear weighting)
    # Strategy Mixture favors A (TTB favors A, Tallying favors A)
    t1_mask = data['A_key'] == (1, 0, 0, 1, 1)
    
    # Trial 3: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 0]
    # WADD favors A (cues 2 and 3 > cue 1)
    # Strategy Mixture favors B (TTB favors B, Tallying ties)
    t3_mask = data['A_key'] == (0, 1, 1, 0, 0)
    
    t1_resp = data.loc[t1_mask, 'response'].mean()
    t3_resp = data.loc[t3_mask, 'response'].mean()
    
    if pd.isna(t1_resp): t1_resp = 0.5
    if pd.isna(t3_resp): t3_resp = 0.5
    
    return float(t1_resp - t3_resp)
```

**Observed (real) value:** -0.0300 (var=0.0316)
**Previous candidate values (this loop):**
  - iter 1: 0.2200 (var=0.0291) (Δ vs real +0.2500)
  - iter 2 (most recent): -0.1775 (var=0.0811) (Δ vs real -0.1475)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0687 (var=0.0289)
- pi_4: -0.5325 (var=0.0410)
- pi_1: -0.3800 (var=0.0347)
- pi_2: 0.0438 (var=0.3439)
- pi_3: -0.7050 (var=0.0478)
- pi_5: -0.2675 (var=0.0436)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Sequential Probabilistic Cue Integration: Decision-makers evaluate options by sequentially sampling cues with a probability proportional to their validity. Upon sampling a cue, evidence is accumulated for the option it favors. If the absolute difference in accumulated evidence reaches a decision threshold, the process terminates and the favored option is chosen. If a maximum number of samples is reached without crossing the threshold, the decision-maker falls back to a noisy integration of all accumulated evidence. This mechanism naturally interpolates between non-compensatory heuristics (low threshold, high validity bias) and compensatory strategies (high threshold, uniform sampling), effectively reproducing both strategy mixture patterns and sensitivity to cue clustering.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    theta = int(parameters["theta"])
    max_samples = int(parameters["max_samples"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    cue_diffs = a - b
    
    # Normalize validities to max 1 for stable gamma scaling across experiments
    v_norm = validities / np.max(validities)
    p = v_norm ** gamma
    p_sum = np.sum(p)
    if p_sum == 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p_sum
        
    # Monte Carlo approximation of the choice probabilities
    N = 1000
    samples = np.random.choice(len(validities), size=(N, max_samples), p=p)
    step_evidence = cue_diffs[samples]
    cum_evidence = np.cumsum(step_evidence, axis=1)
    
    hit_A = cum_evidence >= theta
    hit_B = cum_evidence <= -theta
    hit_any = hit_A | hit_B
    
    ever_hit = np.any(hit_any, axis=1)
    first_hit_idx = np.argmax(hit_any, axis=1)
    
    # Decisions for simulated runs that hit the threshold
    final_ev_hit = cum_evidence[np.arange(N), first_hit_idx]
    prob_A_hit = (final_ev_hit > 0).astype(float)
    prob_A_hit = np.where(final_ev_hit == 0, 0.5, prob_A_hit)
    
    # Decisions for simulated runs that exhausted max_samples without hitting threshold
    final_ev_no_hit = cum_evidence[:, -1]
    z = beta * final_ev_no_hit
    z = np.clip(z, -100, 100)  # Prevent overflow in exp
    prob_A_no_hit = 1.0 / (1.0 + np.exp(-z))
    
    # Aggregate overall probability of choosing A
    prob_A_overall = np.where(ever_hit, prob_A_hit, prob_A_no_hit)
    expected_p_A = np.mean(prob_A_overall)
    
    p_core = np.array([expected_p_A, 1.0 - expected_p_A])
    
    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- theta: {1, 2, 3, 4, 5}
- max_samples: {3, 5, 7, 10, 15}
- gamma: [0.0, 10.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's guidance, this theory implements a probabilistic cue sampling mechanism where decision-makers accumulate evidence by drawing cues weighted by their validities (controlled by 'gamma'). If the accumulated evidence differential reaches a threshold ('theta'), a deterministic choice is made. Otherwise, if the 'max_samples' limit is reached, it falls back to a noisy integration of the sampled cues (parameterized by inverse temperature 'beta'). This approach unifies the deterministic one-reason stopping rules of Take-The-Best (low threshold, high validity bias) with the compensatory integration of Tallying/WADD (high threshold, uniform or moderate sampling). The intra-individual stochasticity induced by the sampling process naturally reproduces the ~0.50 conflict trial choice proportions seen in Strategy Mixture, while the integration fallback resolves the deterministic failures on cue clustering (Experiment 2).

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory successfully implements the arbiter's prescribed Sequential Probabilistic Cue Integration mechanism and was accepted by the gate, establishing a new running-best loss. However, an analysis of the per-experiment fits shows substantial deviations on Exp 1, Exp 2, and Exp 10. Specifically, on Exp 2, the candidate predicts ~81% TTB-consistent choices compared to the observed ~51%. This indicates that the model is too heavily biased toward the most valid cue, likely because the sampling occurs with replacement, 'gamma' can be very high, and 'theta' can be low, causing the model to repeatedly sample the first cue and cross the threshold immediately. It fails to adequately capture the compensatory Tallying-like behavior that humans exhibit when cues are clustered.
Rationale: To fix the over-reliance on the first cue (which breaks Exp 2 and Exp 10), make minor adjustments to the prescribed mechanism: 1) Constrain the 'gamma' parameter range (e.g., [0.0, 3.0]) so that the sampling distribution does not become entirely degenerate on the first cue. 2) Increase the upper bound of the threshold 'theta' (e.g., up to 10) so the model is forced to accumulate more evidence before stopping. 3) For the fallback mechanism (when max_samples is reached), consider integrating the actual unweighted cue differences of the stimulus (true Tallying) rather than the sum of the sampled cues, as sampling with replacement heavily distorts the tally. Keep the core probabilistic sampling intact.

**Outcome of this advice:** iter 1 candidate loss=0.3349 -> iter 2 candidate loss=0.4131 -> the gate marked it REJECTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate was REJECTED by the gate, as the aggregate loss worsened from 0.3349 to 0.4131. The previous advice to constrain gamma, increase theta, and change the fallback mechanism to true Tallying overcorrected the model's behavior. For instance, in Exp 2, the TTB-consistent choices dropped from 0.8133 to 0.3600 (real is 0.5089), and in Exp 3 and 4, the metrics dropped significantly below 0.5. The model became too noisy and compensatory, losing the necessary TTB-like features that fit the other experiments.
Rationale: Since the previous advice was rejected by the gate, the proposer will build on the accepted iter 1 base. To reduce the over-reliance on the first cue (Exp 2) without overcorrecting, we should avoid changing the fallback to true Tallying and instead keep it as the sum of sampled cues. To smooth the sampling distribution more gracefully, replace the power-law validity weighting with a softmax function: `p = exp(gamma * v_norm) / sum(...)`, where `gamma` acts as an inverse temperature (range [0.0, 10.0]). Keep `theta` in a moderate range (e.g., {1, 2, 3, 4, 5}) so the model can still stop early on strong evidence but occasionally samples deeper.

**Outcome of this advice:** iter 2 candidate loss=0.4131 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Sequential Probabilistic Cue Integration: Decision-makers evaluate options by sequentially sampling cues with a probability proportional to their validity. Upon sampling a cue, evidence is accumulated for the option it favors. If the absolute difference in accumulated evidence reaches a decision threshold, the process terminates and the favored option is chosen. If a maximum number of samples is reached without crossing the threshold, the decision-maker falls back to a noisy integration of all accumulated evidence. This mechanism naturally interpolates between non-compensatory heuristics (low threshold, high validity bias) and compensatory strategies (high threshold, uniform sampling), effectively reproducing both strategy mixture patterns and sensitivity to cue clustering.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    theta = int(parameters[\"theta\"])\n    max_samples = int(parameters[\"max_samples\"])\n    gamma = float(parameters[\"gamma\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    a, b = stim[0], stim[1]\n    cue_diffs = a - b\n    \n    # Normalize validities to max 1 for stable gamma scaling across experiments\n    v_norm = validities / np.max(validities)\n    \n    # Softmax for cue sampling probabilities (gamma acts as inverse temperature)\n    z_p = gamma * v_norm\n    z_p = z_p - np.max(z_p)\n    e_p = np.exp(z_p)\n    p = e_p / np.sum(e_p)\n        \n    # Monte Carlo approximation of the choice probabilities\n    N = 1000\n    samples = np.random.choice(len(validities), size=(N, max_samples), p=p)\n    step_evidence = cue_diffs[samples]\n    cum_evidence = np.cumsum(step_evidence, axis=1)\n    \n    hit_A = cum_evidence >= theta\n    hit_B = cum_evidence <= -theta\n    hit_any = hit_A | hit_B\n    \n    ever_hit = np.any(hit_any, axis=1)\n    first_hit_idx = np.argmax(hit_any, axis=1)\n    \n    # Decisions for simulated runs that hit the threshold\n    final_ev_hit = cum_evidence[np.arange(N), first_hit_idx]\n    prob_A_hit = (final_ev_hit > 0).astype(float)\n    prob_A_hit = np.where(final_ev_hit == 0, 0.5, prob_A_hit)\n    \n    # Decisions for simulated runs that exhausted max_samples without hitting threshold\n    final_ev_no_hit = cum_evidence[:, -1]\n    z = beta * final_ev_no_hit\n    z = np.clip(z, -100, 100)  # Prevent overflow in exp\n    prob_A_no_hit = 1.0 / (1.0 + np.exp(-z))\n    \n    # Aggregate overall probability of choosing A\n    prob_A_overall = np.where(ever_hit, prob_A_hit, prob_A_no_hit)\n    expected_p_A = np.mean(prob_A_overall)\n    \n    p_core = np.array([expected_p_A, 1.0 - expected_p_A])\n    \n    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "theta": "{1, 2, 3, 4, 5}",
    "max_samples": "{3, 5, 7, 10, 15}",
    "gamma": "[0.0, 10.0]",
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "We revert to the successful Iteration 1 base and apply a minimal edit: replacing the power-law validity weighting for cue sampling with a softmax function parameterized by inverse temperature 'gamma'. This smoothly controls the probability of sampling the most valid cue, preventing it from becoming completely degenerate and over-predicting TTB-like behavior on conflict trials, without the overcorrection seen when replacing the fallback mechanism with true Tallying."
}
```
```

## Usage

```json
{
  "prompt_token_count": 17211,
  "candidates_token_count": 1163,
  "total_token_count": 20322
}
```
