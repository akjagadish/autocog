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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace the Take-The-Best theory with a Naïve Bayes (Log-Odds) Weighted Additive model. Instead of linearly centering validities around chance (val - 0.5) or using a strict non-compensatory heuristic, this theory posits that decision-makers weigh each feature by its log-odds (log(validity / (1 - validity))). This provides a theoretically grounded, non-linear evidence accumulation scheme that naturally scales cue strengths, allowing for a more nuanced compensatory integration that might better match the observed choice probabilities.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.467)

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
- Experiment 1: real=0.2566 (var=0.0090) vs this=0.2040 (var=0.0039)
- Experiment 2: real=0.0866 (var=0.0009) vs this=0.0003 (var=0.0001)
- Experiment 3: real=0.1453 (var=0.0027) vs this=0.0492 (var=0.0006)
- Experiment 4: real=0.0896 (var=0.0017) vs this=0.0003 (var=0.0001)
- Experiment 5: real=0.2225 (var=0.0052) vs this=0.2370 (var=0.0044)
- Experiment 6: real=0.0192 (var=0.0004) vs this=0.0004 (var=0.0001)
- Experiment 7: real=0.1608 (var=0.0034) vs this=0.0936 (var=0.0019)
- Experiment 8: real=0.0624 (var=0.0010) vs this=0.0003 (var=0.0001)
- Experiment 9: real=0.0774 (var=0.0018) vs this=0.0421 (var=0.0007)
- Experiment 10: real=0.3282 (var=0.0036) vs this=0.2474 (var=0.0051)


---

### `pi_3` (overall score: 0.456)

**Description**
Decision-makers use a Weighted Additive (WADD) strategy, evaluating each option by summing its features weighted by their respective validities, and choose the option with the highest weighted sum. Choices are subject to softmax noise and independent lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted additive sums
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
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
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2566 (var=0.0090) vs this=0.1758 (var=0.0032)
- Experiment 2: real=0.0866 (var=0.0009) vs this=0.0192 (var=0.0003)
- Experiment 3: real=0.1453 (var=0.0027) vs this=0.0003 (var=0.0001)
- Experiment 4: real=0.0896 (var=0.0017) vs this=0.0559 (var=0.0009)
- Experiment 5: real=0.2225 (var=0.0052) vs this=0.2242 (var=0.0052)
- Experiment 6: real=0.0192 (var=0.0004) vs this=0.0179 (var=0.0004)
- Experiment 7: real=0.1608 (var=0.0034) vs this=0.0612 (var=0.0021)
- Experiment 8: real=0.0624 (var=0.0010) vs this=0.0115 (var=0.0003)
- Experiment 9: real=0.0774 (var=0.0018) vs this=0.0254 (var=0.0011)
- Experiment 10: real=0.3282 (var=0.0036) vs this=0.2448 (var=0.0037)


---

### `pi_5` (overall score: 0.299)

**Description**
Decision-makers use a Compensatory Weighted Additive (WADD) strategy where they integrate all available cues, weighting each by its validity centered around chance (validity - 0.5). This linear integration allows multiple weaker cues to appropriately accumulate and potentially override a single strong cue, avoiding the extreme dominance that log-odds weighting can cause. Choices are made probabilistically via a softmax function over the accumulated evidence, with an independent lapse rate accounting for random guesses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Extract validities
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Linear weighting centered around chance (0.5) to prevent extreme weights
    weights = val - 0.5
    
    # Weighted additive sums
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2566 (var=0.0090) vs this=0.0905 (var=0.0010)
- Experiment 2: real=0.0866 (var=0.0009) vs this=0.0900 (var=0.0022)
- Experiment 3: real=0.1453 (var=0.0027) vs this=0.0222 (var=0.0003)
- Experiment 4: real=0.0896 (var=0.0017) vs this=0.1143 (var=0.0037)
- Experiment 5: real=0.2225 (var=0.0052) vs this=0.1368 (var=0.0019)
- Experiment 6: real=0.0192 (var=0.0004) vs this=0.0826 (var=0.0009)
- Experiment 7: real=0.1608 (var=0.0034) vs this=0.0006 (var=0.0001)
- Experiment 8: real=0.0624 (var=0.0010) vs this=0.0622 (var=0.0007)
- Experiment 9: real=0.0774 (var=0.0018) vs this=0.0003 (var=0.0001)
- Experiment 10: real=0.3282 (var=0.0036) vs this=0.1237 (var=0.0026)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
P_REF = {'((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.8565906455353691, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.8373248983280615, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.15898767034393252, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.14881865602945687, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.1520648398301814, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.16161158895427796, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8453125, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.8523255813953489, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.15064102564102563, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.1757028112449799, '((0, 1, 1, 1), (1, 0, 0, 0))|0': 0.8585072515033605, '((0, 1, 1, 1), (1, 0, 0, 0))|1': 0.8393309680689306}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2566 (var=0.0090)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0005 (var=0.0002)
- pi_2: 0.2040 (var=0.0039)
- pi_3: 0.1758 (var=0.0032)
- pi_4: 0.0001 (var=0.0001)
- pi_5: 0.0905 (var=0.0010)
- pi_6: 0.0010 (var=0.0002)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.1402002861230329, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.1497695852534562, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.5043969849246231, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.48375451263537905, '((0, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.5103884372177055, '((0, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.5065913370998116, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|0': 0.8472344161545216, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|1': 0.8648302370275465, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.16136919315403422, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.1365568544102019, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.847210994341148, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8530416951469583, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.12895927601809956, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.1579330422125182, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.494475138121547, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.4640449438202247, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|0': 0.8562322946175638, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|1': 0.854251012145749, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.5150684931506849, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.4637096774193548}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0866 (var=0.0009)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0003 (var=0.0001)
- pi_1: 0.1925 (var=0.0062)
- pi_3: 0.0192 (var=0.0003)
- pi_4: 0.1827 (var=0.0057)
- pi_5: 0.0900 (var=0.0022)
- pi_6: 0.1982 (var=0.0048)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.26369728025226646, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.29727187206020694, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.6797778806108283, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.6608756080611535, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.22816114878340646, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.2598353156450137, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.32966136270909835, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.35422106179286333, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.1391884951206985, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.1355111917725348, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.3110687022900763, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.3377551020408163, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.18157181571815717, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.2274401473296501, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.600963644327639, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.5854214123006833}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.1453 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0003 (var=0.0001)
- pi_2: 0.0492 (var=0.0006)
- pi_1: 0.0441 (var=0.0010)
- pi_4: 0.0502 (var=0.0009)
- pi_5: 0.0222 (var=0.0003)
- pi_6: 0.0473 (var=0.0009)

### Experiment 4
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.8586609989373007, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8712757830404889, '((0, 1, 0, 0, 0), (1, 0, 1, 0, 0))|0': 0.8609756097560975, '((0, 1, 0, 0, 0), (1, 0, 1, 0, 0))|1': 0.8570909090909091, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.15002239140170176, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.14257888585897935, '((1, 0, 0, 1, 1), (0, 1, 1, 1, 0))|0': 0.5017473789316026, '((1, 0, 0, 1, 1), (0, 1, 1, 1, 0))|1': 0.49553092599213444, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.49635036496350365, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.4961554026709834, '((0, 1, 1, 1, 0), (1, 0, 0, 1, 1))|0': 0.4830261881668283, '((0, 1, 1, 1, 0), (1, 0, 0, 1, 1))|1': 0.49488677867056247}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0896 (var=0.0017)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0003 (var=0.0001)
- pi_3: 0.0559 (var=0.0009)
- pi_1: 0.1186 (var=0.0038)
- pi_4: 0.1235 (var=0.0032)
- pi_5: 0.1143 (var=0.0037)
- pi_6: 0.1488 (var=0.0028)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.14216867469879518, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.15532646048109966, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.15469613259668508, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.15073272854152128, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.8447457627118644, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.8497959183673469, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.8611491108071135, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.8594507269789984, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.15014436958614052, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.14148103552077063, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.8241758241758241, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.8639021796916534, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.14147018030513175, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.14308426073131955, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.8389261744966443, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.8534768211920529, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|0': 0.14962251201098148, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|1': 0.1568785197103781, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.8593523441275979, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.8193343898573693}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2225 (var=0.0052)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0008 (var=0.0002)
- pi_2: 0.2370 (var=0.0044)
- pi_1: 0.0007 (var=0.0002)
- pi_3: 0.2242 (var=0.0052)
- pi_5: 0.1368 (var=0.0019)
- pi_6: 0.0009 (var=0.0001)

### Experiment 6
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0, 1, 1), (0, 1, 0, 1, 0, 0))|0': 0.12998137802607077, '((1, 1, 1, 0, 1, 1), (0, 1, 0, 1, 0, 0))|1': 0.13947990543735225, '((0, 1, 0, 0, 1, 0), (1, 0, 1, 0, 0, 0))|0': 0.4952642835319279, '((0, 1, 0, 0, 1, 0), (1, 0, 1, 0, 0, 0))|1': 0.5225933202357563, '((1, 0, 1, 0, 0, 1), (0, 0, 0, 1, 0, 0))|0': 0.14586799778147533, '((1, 0, 1, 0, 0, 1), (0, 0, 0, 1, 0, 0))|1': 0.13480146813480146, '((0, 1, 0, 0, 0, 0), (1, 0, 1, 0, 1, 1))|0': 0.8642156862745098, '((0, 1, 0, 0, 0, 0), (1, 0, 1, 0, 1, 1))|1': 0.8666666666666667, '((1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1))|0': 0.4955338153977031, '((1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1))|1': 0.4932526756630991, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0))|0': 0.8539269634817409, '((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0))|1': 0.8593359514459121}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0192 (var=0.0004)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0004 (var=0.0001)
- pi_4: 0.2287 (var=0.0049)
- pi_1: 0.2123 (var=0.0067)
- pi_3: 0.0179 (var=0.0004)
- pi_5: 0.0826 (var=0.0009)
- pi_6: 0.2276 (var=0.0052)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.17096165933375235, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.21396054628224584, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|0': 0.49543639475185397, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|1': 0.48299845440494593, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.49304377332880894, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.5013491635186185, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.22198603097479502, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.26808228268082285, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.2222222222222222, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.25272727272727274, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.30790872147546106, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.31667707682698315}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.1608 (var=0.0034)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0006 (var=0.0001)
- pi_2: 0.0936 (var=0.0019)
- pi_1: 0.0611 (var=0.0016)
- pi_3: 0.0612 (var=0.0021)
- pi_4: 0.0662 (var=0.0018)
- pi_6: 0.0714 (var=0.0016)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1))|0': 0.8675363773206222, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1))|1': 0.8643434971997511, '((0, 1, 1, 1, 1, 0), (1, 0, 0, 0, 0, 1))|0': 0.1512301013024602, '((0, 1, 1, 1, 1, 0), (1, 0, 0, 0, 0, 1))|1': 0.13210099188458072, '((1, 0, 0, 0, 0, 0), (0, 1, 0, 1, 1, 0))|0': 0.8493231312536786, '((1, 0, 0, 0, 0, 0), (0, 1, 0, 1, 1, 0))|1': 0.8616517622304051, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.8547832071576049, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.8695854680950164, '((1, 0, 1, 1, 0, 0), (0, 1, 0, 0, 1, 1))|0': 0.506993006993007, '((1, 0, 1, 1, 0, 0), (0, 1, 0, 0, 1, 1))|1': 0.48770053475935826, '((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0))|0': 0.14600409836065573, '((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0))|1': 0.14381067961165048, '((1, 1, 0, 0, 1, 0), (0, 0, 1, 1, 0, 1))|0': 0.5154083204930663, '((1, 1, 0, 0, 1, 0), (0, 0, 1, 1, 0, 1))|1': 0.5013032145960035, '((1, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 1))|0': 0.5055849500293945, '((1, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 1))|1': 0.49025803054239075}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0624 (var=0.0010)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0003 (var=0.0001)
- pi_5: 0.0622 (var=0.0007)
- pi_1: 0.2016 (var=0.0053)
- pi_3: 0.0115 (var=0.0003)
- pi_4: 0.2075 (var=0.0057)
- pi_6: 0.2130 (var=0.0042)

### Experiment 9
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (1, 0, 1, 0, 1))|0': 0.5024608501118568, '((0, 1, 0, 0, 0), (1, 0, 1, 0, 1))|1': 0.4922737306843267, '((0, 1, 0, 0, 0), (1, 0, 0, 1, 0))|0': 0.7202024850437183, '((0, 1, 0, 0, 0), (1, 0, 0, 1, 0))|1': 0.7183098591549296, '((1, 0, 0, 0, 1), (0, 0, 0, 1, 0))|0': 0.41073512252042005, '((1, 0, 0, 0, 1), (0, 0, 0, 1, 0))|1': 0.4037685060565276, '((0, 1, 0, 0, 0), (0, 0, 1, 0, 0))|0': 0.18383371824480368, '((0, 1, 0, 0, 0), (0, 0, 1, 0, 0))|1': 0.1855787476280835, '((0, 1, 0, 1, 0), (1, 1, 0, 0, 1))|0': 0.5897531787584144, '((0, 1, 0, 1, 0), (1, 1, 0, 0, 1))|1': 0.590310442144873, '((0, 1, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.5946054750402576, '((0, 1, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.5941278065630398}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0774 (var=0.0018)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0003 (var=0.0001)
- pi_6: 0.1052 (var=0.0022)
- pi_1: 0.1043 (var=0.0035)
- pi_2: 0.0421 (var=0.0007)
- pi_3: 0.0254 (var=0.0011)
- pi_4: 0.1099 (var=0.0027)

### Experiment 10
**Design**
  A=[0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 1, 0), (0, 0, 0, 0, 1))|0': 0.117357910906298, '((0, 1, 0, 1, 0), (0, 0, 0, 0, 1))|1': 0.12610441767068273, '((0, 0, 0, 1, 0), (1, 1, 0, 0, 1))|0': 0.11905667554203118, '((0, 0, 0, 1, 0), (1, 1, 0, 0, 1))|1': 0.14048825426070935, '((1, 0, 1, 0, 1), (0, 1, 0, 0, 0))|0': 0.8811976819059885, '((1, 0, 1, 0, 1), (0, 1, 0, 0, 0))|1': 0.8701298701298701, '((0, 1, 0, 1, 0), (1, 0, 1, 1, 1))|0': 0.12163934426229508, '((0, 1, 0, 1, 0), (1, 0, 1, 1, 1))|1': 0.13542857142857143, '((0, 0, 0, 1, 0), (1, 1, 0, 0, 0))|0': 0.12852552659764369, '((0, 0, 0, 1, 0), (1, 1, 0, 0, 0))|1': 0.1345672836418209, '((0, 1, 0, 0, 1), (0, 0, 0, 1, 0))|0': 0.8712940355772585, '((0, 1, 0, 0, 1), (0, 0, 0, 1, 0))|1': 0.8773926539058459}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.3282 (var=0.0036)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0004 (var=0.0001)
- pi_5: 0.1237 (var=0.0026)
- pi_1: 0.0028 (var=0.0004)
- pi_2: 0.2474 (var=0.0051)
- pi_3: 0.2448 (var=0.0037)
- pi_4: 0.0004 (var=0.0001)


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
  "theory": "Decision-makers use a Naïve Bayes (Log-Odds) Weighted Additive strategy where they integrate all available cues, weighting each by its log-odds (log(validity / (1 - validity))). This provides a theoretically grounded, non-linear evidence accumulation scheme that naturally scales cue strengths, allowing for a nuanced compensatory integration. Choices are made probabilistically via a softmax function over the accumulated evidence, with an independent lapse rate accounting for random guesses.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    # Extract validities\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Clip validities to avoid division by zero or log(0) in log-odds calculation\n    # Validities are provided in [0.5, 1.0]\n    val_clipped = np.clip(val, 1e-4, 1.0 - 1e-4)\n    \n    # Calculate log-odds weights\n    weights = np.log(val_clipped / (1.0 - val_clipped))\n    \n    # Weighted additive sums\n    score_a = np.sum(a * weights)\n    score_b = np.sum(b * weights)\n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax choice rule with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "The Naïve Bayes (Log-Odds) model provides a normative foundation for evidence accumulation. By transforming validities into log-odds, the model naturally captures the non-linear relationship between cue validity and evidential weight (e.g., a cue with 0.9 validity is much more than twice as strong as one with 0.6). This addresses the limitations of linear weighting (which under-weights highly valid cues) and non-compensatory heuristics (which ignore weaker cues), better matching the observed choice probabilities through a nuanced compensatory integration."
}
```
```

## Usage

```json
{
  "prompt_token_count": 21895,
  "candidates_token_count": 715,
  "total_token_count": 24001
}
```
