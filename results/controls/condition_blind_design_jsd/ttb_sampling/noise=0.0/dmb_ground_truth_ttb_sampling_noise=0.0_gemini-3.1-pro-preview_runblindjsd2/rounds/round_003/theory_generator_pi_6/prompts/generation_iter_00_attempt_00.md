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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Propose a 'Tallying' (or Equal Weight) heuristic theory. In this theory, decision-makers simply count the number of positive features for each option and choose the one with the higher tally, ignoring the specific validities of the cues. If there is a tie, they guess randomly. This represents another classic, non-compensatory bounded rationality heuristic that completely contrasts with both the single-cue focus of Take The Best and the complex integration of Weighted Additive models.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.942)

**Description**
People use the 'Take The Best' (TTB) heuristic to make binary choices. They search through features in descending order of their subjective or stated validities. The first feature that discriminates between the two options (i.e., one option has a positive feature and the other does not) strictly determines the choice, ignoring all other features. If no features discriminate, they guess uniformly. Response noise is modeled via a simple lapse rate where the individual occasionally guesses randomly instead of following the deterministic rule.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by descending validity
    order = np.argsort(validities)[::-1]
    
    epsilon = float(parameters["epsilon"])
    
    # Default to uniform guess if no features discriminate
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0748 (var=0.0005) vs this=0.0708 (var=0.0006)
- Experiment 2: real=0.0238 (var=0.0003) vs this=0.0238 (var=0.0003)
- Experiment 3: real=0.0037 (var=0.0003) vs this=0.0009 (var=0.0001)
- Experiment 4: real=0.0221 (var=0.0005) vs this=0.0330 (var=0.0004)
- Experiment 5: real=0.0007 (var=0.0001) vs this=0.0010 (var=0.0001)
- Experiment 6: real=0.0550 (var=0.0003) vs this=0.0495 (var=0.0004)
- Experiment 7: real=0.0039 (var=0.0001) vs this=0.0005 (var=0.0001)
- Experiment 8: real=0.0056 (var=0.0001) vs this=0.0038 (var=0.0001)


---

### `pi_5` (overall score: 0.791)

**Description**
Weighted Additive (WADD) Theory: Decision-makers integrate all available information by weighting each feature according to its validity. Instead of ignoring feature validities (like Equal-Weight) or ignoring all but the best feature (like Take The Best), this compensatory heuristic transforms the stated validities into log-odds weights to represent the amount of evidence each feature provides. The overall subjective value of an option is computed by summing the weights of its positive features. Choice probabilities are then generated by applying a softmax function to these values, meaning multiple weak cues can potentially outweigh a single strong cue. Response variability is captured via the softmax temperature and an independent random lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities using log-odds for principled evidence weighting.
    # Clip to avoid log(0) or division by zero for validities of 1.0.
    v_clipped = np.clip(validities, 0.501, 0.999)
    weights = np.log(v_clipped / (1.0 - v_clipped))
    
    # Calculate the weighted additive score for each option
    score_a = float(np.sum(a * weights))
    score_b = float(np.sum(b * weights))
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax over the total scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Incorporate uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0748 (var=0.0005) vs this=0.0625 (var=0.0007)
- Experiment 2: real=0.0238 (var=0.0003) vs this=0.0174 (var=0.0002)
- Experiment 3: real=0.0037 (var=0.0003) vs this=0.0020 (var=0.0002)
- Experiment 4: real=0.0221 (var=0.0005) vs this=0.0366 (var=0.0004)
- Experiment 5: real=0.0007 (var=0.0001) vs this=0.0028 (var=0.0002)
- Experiment 6: real=0.0550 (var=0.0003) vs this=0.0445 (var=0.0003)
- Experiment 7: real=0.0039 (var=0.0001) vs this=0.0022 (var=0.0001)
- Experiment 8: real=0.0056 (var=0.0001) vs this=0.0018 (var=0.0001)


---

### `pi_1` (overall score: 0.078)

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
- Experiment 1: real=0.0748 (var=0.0005) vs this=0.0010 (var=0.0001)
- Experiment 2: real=0.0238 (var=0.0003) vs this=0.0090 (var=0.0002)
- Experiment 3: real=0.0037 (var=0.0003) vs this=0.0456 (var=0.0004)
- Experiment 4: real=0.0221 (var=0.0005) vs this=0.0103 (var=0.0002)
- Experiment 5: real=0.0007 (var=0.0001) vs this=0.0057 (var=0.0002)
- Experiment 6: real=0.0550 (var=0.0003) vs this=0.0016 (var=0.0001)
- Experiment 7: real=0.0039 (var=0.0001) vs this=0.0161 (var=0.0002)
- Experiment 8: real=0.0056 (var=0.0001) vs this=0.0412 (var=0.0003)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 0, 1, 1))|0': 0.43703703703703706, '((0, 1, 1, 1), (1, 0, 1, 1))|1': 0.5026143790849673, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.8568872987477638, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.8424336973478939, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.15728155339805824, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.17142857142857143, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.1486784140969163, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.1547085201793722, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.125, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.1565217391304348, '((0, 1, 0, 0), (1, 1, 1, 0))|0': 0.8710166919575114, '((0, 1, 0, 0), (1, 1, 1, 0))|1': 0.8692033293697978, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8631578947368421, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8813008130081301, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8595360824742269, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8349609375, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.11976047904191617, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.1157243816254417, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.5056980056980057, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.5151515151515151, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.527147087857848, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4714104193138501, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8561643835616438, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8524229074889867, '((1, 0, 0, 1), (1, 1, 0, 1))|0': 0.8854961832061069, '((1, 0, 0, 1), (1, 1, 0, 1))|1': 0.8607021517553793, '((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.17066666666666666, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.1368421052631579, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.8222222222222222, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.8921810699588477}
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

**Observed (real) value:** 0.0748 (var=0.0005)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0010 (var=0.0001)
- pi_2: 0.0115 (var=0.0007)
- pi_3: 0.0708 (var=0.0006)
- pi_4: 0.0009 (var=0.0001)
- pi_5: 0.0625 (var=0.0007)

### Experiment 2
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8401037165082109, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.80248833592535, '((0, 0, 1, 1), (0, 1, 1, 0))|0': 0.6113207547169811, '((0, 0, 1, 1), (0, 1, 1, 0))|1': 0.6108108108108108, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.34727061556329847, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.3184238551650692, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8583850931677018, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8170854271356784, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.22069317023445464, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.23318385650224216, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.38506224066390043, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.3277310924369748, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.15520833333333334, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.17023809523809524, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.760061919504644, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.7381889763779528, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.1650114591291062, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.1955193482688391, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.14410134600158353, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.19366852886405958, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.7028301886792453, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.6401515151515151, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.20399666944213155, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.28046744574290483, '((1, 0, 1, 0), (0, 0, 1, 1))|0': 0.3624242424242424, '((1, 0, 1, 0), (0, 0, 1, 1))|1': 0.37333333333333335, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.148636763412489, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.16289592760180996}
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

**Observed (real) value:** 0.0238 (var=0.0003)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0008 (var=0.0002)
- pi_1: 0.0090 (var=0.0002)
- pi_3: 0.0238 (var=0.0003)
- pi_4: 0.0087 (var=0.0002)
- pi_5: 0.0174 (var=0.0002)

### Experiment 3
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12193588937774984, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.18181818181818182, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8410087719298246, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8862612612612613, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.8716814159292036, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.8671875, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8713745271122321, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8917576961271102, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.11723329425556858, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.14064914992272023, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.8592017738359202, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.8775055679287305, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.1415929203539823, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.12388724035608309, '((1, 0, 1, 1), (0, 1, 1, 1))|0': 0.11632100991884581, '((1, 0, 1, 1), (0, 1, 1, 1))|1': 0.12590448625180897, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.1254125412541254, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.10998877665544332, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.12875536480686695, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.1191904047976012, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.09761549925484352, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.13100436681222707, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8807649043869517, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8858397365532382, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.14576271186440679, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.1366120218579235, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8717948717948718, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.8830255057167986, '((0, 0, 1, 1), (1, 1, 0, 0))|0': 0.9044117647058824, '((0, 0, 1, 1), (1, 1, 0, 0))|1': 0.8616071428571429}
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

**Observed (real) value:** 0.0037 (var=0.0003)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0009 (var=0.0001)
- pi_2: 0.0369 (var=0.0021)
- pi_1: 0.0456 (var=0.0004)
- pi_4: 0.0448 (var=0.0002)
- pi_5: 0.0020 (var=0.0002)

### Experiment 4
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8526829268292683, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8296774193548387, '((1, 1, 0, 0), (1, 0, 0, 1))|0': 0.40131578947368424, '((1, 1, 0, 0), (1, 0, 0, 1))|1': 0.40673076923076923, '((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.42889137737961924, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.38588754134509373, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.6492082825822169, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.5985699693564862, '((1, 1, 0, 1), (1, 1, 0, 0))|0': 0.2508710801393728, '((1, 1, 0, 1), (1, 1, 0, 0))|1': 0.234341252699784, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.5834127740705434, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.5845539280958721, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.25029655990510086, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.2251828631138976, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.19047619047619047, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.20503597122302158, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.6282051282051282, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.617231638418079, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.8502512562814071, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8285714285714286, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.4478114478114478, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.45324532453245325, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.826625386996904, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.8303249097472925, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.5434782608695652, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.5343811394891945, '((1, 0, 1, 0), (0, 0, 1, 0))|0': 0.2383025027203482, '((1, 0, 1, 0), (0, 0, 1, 0))|1': 0.21112372304199772, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.7726063829787234, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.7958015267175572}
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

**Observed (real) value:** 0.0221 (var=0.0005)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0017 (var=0.0004)
- pi_3: 0.0330 (var=0.0004)
- pi_1: 0.0103 (var=0.0002)
- pi_4: 0.0056 (var=0.0002)
- pi_5: 0.0366 (var=0.0004)

### Experiment 5
**Design**
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8621700879765396, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8805903398926654, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.11504424778761062, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.10695187165775401, '((0, 0, 0, 1), (1, 0, 0, 1))|0': 0.8700440528634361, '((0, 0, 0, 1), (1, 0, 0, 1))|1': 0.8845291479820628, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.8587731811697575, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.8789808917197452, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8701594533029613, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8765613519470977, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.864406779661017, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.9080036798528058, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.12398190045248869, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.1223021582733813, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.8806306306306306, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.8892543859649122, '((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.8705357142857143, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.8809171597633136, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.11958146487294469, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.10698496905393456, '((0, 1, 1, 1), (0, 0, 1, 1))|0': 0.1348973607038123, '((0, 1, 1, 1), (0, 0, 1, 1))|1': 0.11001788908765653, '((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.8956228956228957, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.8767876787678768, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8747795414462081, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.8738738738738738, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.13157894736842105, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.11469534050179211, '((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.12855007473841554, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.12024756852343059}
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

**Observed (real) value:** 0.0007 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0010 (var=0.0001)
- pi_4: 0.0088 (var=0.0007)
- pi_1: 0.0057 (var=0.0002)
- pi_2: 0.0141 (var=0.0010)
- pi_5: 0.0028 (var=0.0002)

### Experiment 6
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.4839572192513369, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.5056100981767181, '((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.8486725663716814, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.8488479262672811, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8265486725663717, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8720647773279352, '((1, 0, 1, 1), (1, 1, 1, 1))|0': 0.8492537313432836, '((1, 0, 1, 1), (1, 1, 1, 1))|1': 0.8469026548672567, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.8308702791461412, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8547439126784215, '((0, 0, 1, 0), (0, 1, 1, 0))|0': 0.8071428571428572, '((0, 0, 1, 0), (0, 1, 1, 0))|1': 0.8763157894736842, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.5056179775280899, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.5016307893020222, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.4763779527559055, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.49489795918367346, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.19220055710306408, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.14989590562109645, '((1, 1, 0, 0), (0, 0, 1, 0))|0': 0.14414414414414414, '((1, 1, 0, 0), (0, 0, 1, 0))|1': 0.1826086956521739, '((1, 0, 0, 0), (0, 1, 0, 1))|0': 0.8669673055242391, '((1, 0, 0, 0), (0, 1, 0, 1))|1': 0.8466593647316539, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.5126903553299492, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.46732837055417703, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.844106463878327, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.8437190900098912, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8591954022988506, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8768115942028986, '((1, 0, 0, 0), (0, 0, 1, 0))|0': 0.4816112084063047, '((1, 0, 0, 0), (0, 0, 1, 0))|1': 0.515052888527258}
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

**Observed (real) value:** 0.0550 (var=0.0003)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0010 (var=0.0002)
- pi_3: 0.0495 (var=0.0004)
- pi_1: 0.0016 (var=0.0001)
- pi_2: 0.0107 (var=0.0004)
- pi_5: 0.0445 (var=0.0003)

### Experiment 7
**Design**
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.132890365448505, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.12709030100334448, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12929061784897025, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.11661341853035144, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8674377224199288, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.863905325443787, '((0, 1, 1, 0), (0, 1, 0, 1))|0': 0.13922651933701657, '((0, 1, 1, 0), (0, 1, 0, 1))|1': 0.13631284916201117, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8639269406392694, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.8780141843971632, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.11243484735666419, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.12253829321663019, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1341991341991342, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.12557077625570776, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8754171301446051, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8690344062153164, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.15171650055370986, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.12374581939799331, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8614457831325302, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8785211267605634, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8621987951807228, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8326271186440678, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.8367952522255193, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.8738898756660746, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.1319910514541387, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.1390728476821192, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.8669064748201439, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.8677325581395349, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.14099783080260303, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.1366742596810934, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.1492007104795737, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.13056379821958458}
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

**Observed (real) value:** 0.0039 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0005 (var=0.0001)
- pi_5: 0.0022 (var=0.0001)
- pi_1: 0.0161 (var=0.0002)
- pi_2: 0.0179 (var=0.0006)
- pi_4: 0.0164 (var=0.0002)

### Experiment 8
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.8622305529522024, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.8703956343792633, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.8287153652392947, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.872416250890948, '((0, 0, 1, 1), (0, 0, 1, 0))|0': 0.24222222222222223, '((0, 0, 1, 1), (0, 0, 1, 0))|1': 0.23407407407407407, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.8309859154929577, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.875968992248062, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.8403361344537815, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.8640483383685801, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8552631578947368, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8661825726141079, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.8056206088992974, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.8361252731245448, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.840958605664488, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.8419090231170768, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.2574626865671642, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.2519582245430809, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.15707620528771385, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.15519253208868145, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.8338164251207729, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.869281045751634, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8418604651162791, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8476190476190476, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.24970131421744324, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.2523364485981308, '((1, 0, 1, 1), (0, 1, 1, 0))|0': 0.14170692431561996, '((1, 0, 1, 1), (0, 1, 1, 0))|1': 0.1662425784563189, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.8603238866396761, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.8591117917304747, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.8641390205371248, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.882604970008569}
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

**Observed (real) value:** 0.0056 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0018 (var=0.0001)
- pi_3: 0.0038 (var=0.0001)
- pi_1: 0.0412 (var=0.0003)
- pi_2: 0.0233 (var=0.0009)
- pi_4: 0.0392 (var=0.0003)


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
  "theory": "Take The Best with Validity-Dependent Confidence",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Sort features by descending validity\n    order = np.argsort(validities)[::-1]\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Default to uniform guess if no features discriminate\n    p_core = np.array([0.5, 0.5])\n    \n    for idx in order:\n        if a[idx] != b[idx]:\n            # Convert validity to log-odds weight\n            v = np.clip(validities[idx], 0.501, 0.999)\n            w = np.log(v / (1.0 - v))\n            \n            # Assign weight to the option that has the feature\n            scores = np.zeros(2)\n            if a[idx] > b[idx]:\n                scores[0] = w\n            else:\n                scores[1] = w\n                \n            # Softmax to convert to probability\n            z = beta * scores\n            z -= np.max(z)\n            e = np.exp(z)\n            p_core = e / np.sum(e)\n            break\n            \n    # Blend with uniform lapse\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return np.random.choice(len(probs), p=probs)",
  "parameters": {
    "beta": "[0.0, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "I am ignoring the arbiter's suggestion to propose a Tallying (Equal Weight) heuristic. As demonstrated by the prior theory `pi_1`, Tallying completely ignores cue validities and results in a very poor fit to the data (overall score 0.078). The high score of `pi_3` (Take The Best, 0.942) strongly indicates that subjects use cue validities to prioritize features. Instead of Tallying, I propose a refinement of Take The Best called 'Take The Best with Validity-Dependent Confidence'. In this theory, decision-makers still search through features in descending order of validity and stop at the first discriminating cue. However, rather than making a strictly deterministic choice, their probability of choosing the favored option scales with the log-odds validity of that discriminating cue. This captures the intuition that people are more confident and less noisy when the deciding cue is highly valid, and more equivocal when it is weak, subsuming strict TTB as a special case when beta is large."
}
```
```

## Usage

```json
{
  "prompt_token_count": 25316,
  "candidates_token_count": 754,
  "total_token_count": 31856
}
```
