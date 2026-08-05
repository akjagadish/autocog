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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Propose a 'Tallying with Validity-Ranked Tie-Breaking' theory. Instead of complex weighted sums or strict one-reason decision making, assume subjects use a boundedly rational approach to minimize cognitive load: they simply count the number of positive features for each option (Tallying). They prefer the option with the higher tally. If the tallies are tied, they fall back to a non-compensatory tie-breaker, inspecting the most valid cue to decide. The model should use this score difference in a softmax function to allow for stochasticity and include a lapse rate for random guessing.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.198)

**Description**
Decision-makers use a non-compensatory heuristic, specifically Take-The-Best (TTB), but their choices are also influenced by sequential dependencies like choice inertia. They evaluate options by inspecting features sequentially in descending order of validity. The first feature that discriminates between the options determines the preferred option, which is assigned a base value. However, the final choice probability is modulated by an inertia bias added to the option chosen in the previous trial. This combines frugal, one-reason decision making with psychological momentum (inertia), passed through a softmax function with a lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    score_a, score_b = 0.0, 0.0
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner == 0:
        score_a = 1.0
    elif winner == 1:
        score_b = 1.0
        
    # Incorporate choice inertia from the previous trial
    inertia = float(parameters["inertia"])
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
        if prev_resp == 0:
            score_a += inertia
        elif prev_resp == 1:
            score_b += inertia
            
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
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
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- inertia: [-10.0, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2658 (var=0.0000) vs this=0.0649 (var=0.0044)
- Experiment 2: real=0.2554 (var=0.0000) vs this=0.0605 (var=0.0022)
- Experiment 3: real=0.2601 (var=0.0008) vs this=0.0558 (var=0.0039)
- Experiment 4: real=0.2320 (var=0.0050) vs this=0.0681 (var=0.0040)
- Experiment 5: real=0.2336 (var=0.0002) vs this=0.0073 (var=0.0020)
- Experiment 6: real=0.2239 (var=0.0074) vs this=0.0385 (var=0.0033)
- Experiment 7: real=0.2150 (var=0.0000) vs this=0.0017 (var=0.0018)
- Experiment 8: real=0.2221 (var=0.0005) vs this=0.0010 (var=0.0019)


---

### `pi_4` (overall score: 0.187)

**Description**
Decision-makers evaluate options using a Weighted Additive (WADD) strategy, but their choices are subject to sequential dependencies, specifically choice inertia. The value of each option is computed as a weighted sum of its features, where weights correspond to cue validities. Additionally, a baseline bias (inertia) is added to the score of the option that was chosen in the immediately preceding trial. This reflects a psychological tendency to repeat previous actions (or avoid them, if inertia is negative). Choice probabilities are generated via a softmax function with an inverse temperature, along with a lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    # Incorporate choice inertia from the previous trial
    inertia = float(parameters["inertia"])
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
        if prev_resp == 0:
            score_a += inertia
        elif prev_resp == 1:
            score_b += inertia
            
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- inertia: [-20.0, 20.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2658 (var=0.0000) vs this=0.0793 (var=0.0023)
- Experiment 2: real=0.2554 (var=0.0000) vs this=0.0562 (var=0.0022)
- Experiment 3: real=0.2601 (var=0.0008) vs this=0.0660 (var=0.0021)
- Experiment 4: real=0.2320 (var=0.0050) vs this=0.0609 (var=0.0026)
- Experiment 5: real=0.2336 (var=0.0002) vs this=0.0007 (var=0.0016)
- Experiment 6: real=0.2239 (var=0.0074) vs this=0.0494 (var=0.0023)
- Experiment 7: real=0.2150 (var=0.0000) vs this=0.0015 (var=0.0011)
- Experiment 8: real=0.2221 (var=0.0005) vs this=0.0012 (var=0.0015)


---

### `pi_2` (overall score: 0.143)

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
- Experiment 1: real=0.2658 (var=0.0000) vs this=0.0194 (var=0.0002)
- Experiment 2: real=0.2554 (var=0.0000) vs this=0.0009 (var=0.0001)
- Experiment 3: real=0.2601 (var=0.0008) vs this=0.0258 (var=0.0002)
- Experiment 4: real=0.2320 (var=0.0050) vs this=0.0144 (var=0.0002)
- Experiment 5: real=0.2336 (var=0.0002) vs this=0.0441 (var=0.0009)
- Experiment 6: real=0.2239 (var=0.0074) vs this=0.0172 (var=0.0002)
- Experiment 7: real=0.2150 (var=0.0000) vs this=0.0438 (var=0.0006)
- Experiment 8: real=0.2221 (var=0.0005) vs this=0.0494 (var=0.0009)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.126044039483675, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.15734989648033126, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8735119047619048, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8728070175438597, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.16129032258064516, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.1453940066592675, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.85997171145686, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.8664226898444648, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8376068376068376, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8716216216216216, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.8649262202043133, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.8324265505984766, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.16248153618906944, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.12911843276936777, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.1632047477744807, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.1483126110124334, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.13676731793960922, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.13501483679525222, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.12455516014234876, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.15680473372781065, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8619676945668135, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8543342269883825, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8610855565777369, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8401122019635343, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8388520971302428, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8893838158871566, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.14516129032258066, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.1252236135957066, '((1, 1, 0, 1), (0, 1, 1, 0))|0': 0.13047445255474452, '((1, 1, 0, 1), (0, 1, 1, 0))|1': 0.15346534653465346, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8712328767123287, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8567375886524823}
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

**Observed (real) value:** 0.2658 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0009 (var=0.0001)
- pi_2: 0.0194 (var=0.0002)
- pi_3: 0.0020 (var=0.0002)
- pi_4: 0.0793 (var=0.0023)
- pi_5: 0.0649 (var=0.0044)

### Experiment 2
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 1), (0, 1, 1, 1))|0': 0.862453531598513, '((0, 0, 0, 1), (0, 1, 1, 1))|1': 0.8751258811681772, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.47289156626506024, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.4876760563380282, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.13242574257425743, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.14818548387096775, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.8419282511210763, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8601321585903083, '((0, 1, 1, 0), (0, 0, 0, 0))|0': 0.14113785557986872, '((0, 1, 1, 0), (0, 0, 0, 0))|1': 0.14672686230248308, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.15040183696900114, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.1496232508073197, '((0, 0, 0, 0), (0, 1, 1, 0))|0': 0.8631346578366446, '((0, 0, 0, 0), (0, 1, 1, 0))|1': 0.8534675615212528, '((0, 1, 0, 1), (1, 1, 1, 1))|0': 0.8433179723502304, '((0, 1, 0, 1), (1, 1, 1, 1))|1': 0.8630393996247655, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.4905814905814906, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.5077720207253886, '((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.13493377483443708, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.13682432432432431, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.1644325290437891, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.17033773861967694, '((1, 0, 1, 1), (1, 0, 0, 0))|0': 0.14156626506024098, '((1, 0, 1, 1), (1, 0, 0, 0))|1': 0.14925373134328357, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.8377483443708609, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8534675615212528, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.8529048207663782, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8425832492431887, '((0, 0, 0, 0), (1, 0, 1, 1))|0': 0.8533627342888643, '((0, 0, 0, 0), (1, 0, 1, 1))|1': 0.8779395296752519, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.14985163204747776, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.15008880994671403}
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

**Observed (real) value:** 0.2554 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0009 (var=0.0001)
- pi_1: 0.0281 (var=0.0001)
- pi_3: 0.0040 (var=0.0001)
- pi_4: 0.0562 (var=0.0022)
- pi_5: 0.0605 (var=0.0022)

### Experiment 3
**Design**
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
P_REF = {'((0, 0, 0, 1), (1, 0, 1, 0))|0': 0.8241525423728814, '((0, 0, 0, 1), (1, 0, 1, 0))|1': 0.8621987951807228, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.1432408236347359, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.1683748169838946, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.1524024024024024, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.17094017094017094, '((0, 1, 0, 0), (0, 0, 1, 0))|0': 0.13938053097345132, '((0, 1, 0, 0), (0, 0, 1, 0))|1': 0.14397321428571427, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.1352154531946508, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.16519823788546256, '((0, 1, 1, 0), (0, 1, 1, 1))|0': 0.831140350877193, '((0, 1, 1, 0), (0, 1, 1, 1))|1': 0.847972972972973, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.8285302593659942, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.8318264014466547, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8353863381858903, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8412348401323043, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.18046709129511676, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.13393528969149737, '((0, 1, 1, 1), (1, 1, 1, 1))|0': 0.8490909090909091, '((0, 1, 1, 1), (1, 1, 1, 1))|1': 0.8514285714285714, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.14977973568281938, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.16704035874439463, '((0, 0, 0, 0), (0, 1, 1, 1))|0': 0.8558382257012394, '((0, 0, 0, 0), (0, 1, 1, 1))|1': 0.8239700374531835, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.14167294649585532, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.17758985200845667, '((0, 0, 1, 0), (0, 0, 0, 0))|0': 0.16057585825027684, '((0, 0, 1, 0), (0, 0, 0, 0))|1': 0.13154960981047936, '((0, 1, 1, 1), (0, 1, 0, 0))|0': 0.16817359855334538, '((0, 1, 1, 1), (0, 1, 0, 0))|1': 0.16714697406340057, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.8523076923076923, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.8458823529411764}
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

**Observed (real) value:** 0.2601 (var=0.0008)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0013 (var=0.0001)
- pi_3: 0.0160 (var=0.0003)
- pi_2: 0.0258 (var=0.0002)
- pi_4: 0.0660 (var=0.0021)
- pi_5: 0.0558 (var=0.0039)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.13957759412304868, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.180028129395218, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.8734723220704529, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8337408312958435, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.14118629908103592, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.15754560530679934, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8639705882352942, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8455056179775281, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.8453038674033149, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.8277310924369747, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.16892502258355918, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.19336219336219337, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.14307342922028765, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.18997912317327767, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.1967020023557126, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.23809523809523808, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.130348913759052, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.2491103202846975, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1337099811676083, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.14769647696476965, '((1, 1, 1, 1), (1, 0, 1, 0))|0': 0.13020833333333334, '((1, 1, 1, 1), (1, 0, 1, 0))|1': 0.15432098765432098, '((0, 1, 1, 0), (1, 1, 0, 0))|0': 0.6285452881976212, '((0, 1, 1, 0), (1, 1, 0, 0))|1': 0.6025459688826026, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8790072388831437, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.885954381752701, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.1439312567132116, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.11392405063291139, '((1, 0, 0, 0), (0, 1, 0, 0))|0': 0.1753902662993572, '((1, 0, 0, 0), (0, 1, 0, 0))|1': 0.19127988748241911, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.13533284564740308, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.18475750577367206}
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

**Observed (real) value:** 0.2320 (var=0.0050)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0010 (var=0.0003)
- pi_1: 0.0048 (var=0.0001)
- pi_2: 0.0144 (var=0.0002)
- pi_4: 0.0609 (var=0.0026)
- pi_5: 0.0681 (var=0.0040)

### Experiment 5
**Design**
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0), (0, 1, 0, 0))|0': 0.5198836081474297, '((1, 0, 1, 0), (0, 1, 0, 0))|1': 0.47074122236671, '((1, 0, 1, 1), (0, 1, 0, 0))|0': 0.467966573816156, '((1, 0, 1, 1), (0, 1, 0, 0))|1': 0.4332409972299169, '((0, 1, 1, 0), (0, 1, 0, 0))|0': 0.4904522613065327, '((0, 1, 1, 0), (0, 1, 0, 0))|1': 0.4360248447204969, '((1, 1, 0, 1), (0, 1, 0, 0))|0': 0.495850622406639, '((1, 1, 0, 1), (0, 1, 0, 0))|1': 0.45095693779904306, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.49638802889576883, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.42719614921780985, '((0, 1, 0, 1), (0, 0, 1, 1))|0': 0.5505154639175258, '((0, 1, 0, 1), (0, 0, 1, 1))|1': 0.4614457831325301, '((1, 1, 1, 1), (0, 1, 0, 0))|0': 0.47426067907995617, '((1, 1, 1, 1), (0, 1, 0, 0))|1': 0.42728297632469, '((0, 0, 1, 0), (1, 1, 0, 0))|0': 0.4968421052631579, '((0, 0, 1, 0), (1, 1, 0, 0))|1': 0.4623529411764706, '((0, 1, 0, 0), (1, 0, 1, 0))|0': 0.5115529285330468, '((0, 1, 0, 0), (1, 0, 1, 0))|1': 0.4473835537665325, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.5056746532156369, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.4512022630834512, '((1, 1, 1, 1), (1, 0, 1, 1))|0': 0.5224839400428265, '((1, 1, 1, 1), (1, 0, 1, 1))|1': 0.45958429561200925, '((0, 0, 1, 0), (0, 1, 0, 0))|0': 0.5144385026737968, '((0, 0, 1, 0), (0, 1, 0, 0))|1': 0.46936416184971097, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.5223068552774756, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.4699205448354143, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.49892008639308855, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.4473684210526316}
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

**Observed (real) value:** 0.2336 (var=0.0002)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0007 (var=0.0016)
- pi_3: 0.0526 (var=0.0009)
- pi_1: 0.0741 (var=0.0021)
- pi_2: 0.0441 (var=0.0009)
- pi_5: 0.0073 (var=0.0020)

### Experiment 6
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.17132867132867133, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.15309446254071662, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.6379928315412187, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.6723027375201288, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.7880794701986755, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.81438127090301, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.7609427609427609, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.7769485903814262, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.7936210131332082, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8484609313338595, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.797427652733119, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.8488918737407656, '((0, 1, 0, 1), (0, 0, 1, 1))|0': 0.6291666666666667, '((0, 1, 0, 1), (0, 0, 1, 1))|1': 0.6820512820512821, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8179775280898877, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8435424354243543, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.19402985074626866, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.16729559748427672, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.7983193277310925, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8370165745856354, '((1, 1, 1, 1), (0, 1, 1, 1))|0': 0.1960352422907489, '((1, 1, 1, 1), (0, 1, 1, 1))|1': 0.14635958395245172, '((0, 0, 0, 1), (0, 0, 1, 0))|0': 0.6612377850162866, '((0, 0, 0, 1), (0, 0, 1, 0))|1': 0.6757679180887372, '((0, 1, 1, 1), (1, 1, 1, 0))|0': 0.7945544554455446, '((0, 1, 1, 1), (1, 1, 1, 0))|1': 0.8359598853868195, '((1, 0, 0, 1), (1, 0, 1, 0))|0': 0.6296296296296297, '((1, 0, 0, 1), (1, 0, 1, 0))|1': 0.6730158730158731, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.834920634920635, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8589743589743589, '((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.8554913294797688, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.8501805054151624}
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

**Observed (real) value:** 0.2239 (var=0.0074)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0018 (var=0.0002)
- pi_4: 0.0494 (var=0.0023)
- pi_1: 0.0101 (var=0.0001)
- pi_2: 0.0172 (var=0.0002)
- pi_5: 0.0385 (var=0.0033)

### Experiment 7
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 1, 1, 0))|0': 0.5142296368989205, '((0, 1, 1, 1), (1, 1, 1, 0))|1': 0.5096030729833547, '((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.4219977553310887, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.41804180418041803, '((1, 1, 0, 0), (1, 0, 1, 0))|0': 0.5243781094527363, '((1, 1, 0, 0), (1, 0, 1, 0))|1': 0.5006289308176101, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.5032894736842105, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.5168918918918919, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.5138461538461538, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.5357575757575758, '((0, 0, 0, 0), (1, 1, 1, 1))|0': 0.502262443438914, '((0, 0, 0, 0), (1, 1, 1, 1))|1': 0.5436681222707423, '((1, 1, 0, 0), (1, 0, 0, 0))|0': 0.4811529933481153, '((1, 1, 0, 0), (1, 0, 0, 0))|1': 0.47216035634743875, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.5050847457627119, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.5049180327868853, '((0, 0, 0, 1), (1, 0, 0, 1))|0': 0.5043196544276458, '((0, 0, 0, 1), (1, 0, 0, 1))|1': 0.5137299771167048, '((0, 0, 1, 1), (1, 1, 0, 0))|0': 0.48259860788863107, '((0, 0, 1, 1), (1, 1, 0, 0))|1': 0.47761194029850745, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.43991416309012876, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.47119815668202764, '((1, 1, 0, 0), (0, 1, 0, 1))|0': 0.4946107784431138, '((1, 1, 0, 0), (0, 1, 0, 1))|1': 0.4736196319018405, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.5406852248394004, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.5311778290993071, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.4575892857142857, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.47013274336283184, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.486784140969163, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.4663677130044843}
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

**Observed (real) value:** 0.2150 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0015 (var=0.0011)
- pi_5: 0.0017 (var=0.0018)
- pi_1: 0.0629 (var=0.0016)
- pi_2: 0.0438 (var=0.0006)
- pi_3: 0.0571 (var=0.0013)

### Experiment 8
**Design**
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 0), (0, 1, 0, 1))|0': 0.4529991047448523, '((0, 1, 1, 0), (0, 1, 0, 1))|1': 0.4480234260614934, '((1, 1, 1, 1), (0, 0, 0, 1))|0': 0.4255555555555556, '((1, 1, 1, 1), (0, 0, 0, 1))|1': 0.35777777777777775, '((0, 0, 1, 0), (1, 1, 1, 1))|0': 0.5586538461538462, '((0, 0, 1, 0), (1, 1, 1, 1))|1': 0.4789473684210526, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.42639040348964014, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.37712344280860705, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.4421768707482993, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.4254215304798962, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.43743536711478803, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.42016806722689076, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.44195953141640043, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.3879210220673635, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.578544061302682, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.4894179894179894, '((1, 0, 0, 1), (0, 1, 0, 0))|0': 0.464746772591857, '((1, 0, 0, 1), (0, 1, 0, 0))|1': 0.42370744010088274, '((0, 0, 0, 0), (1, 1, 0, 0))|0': 0.5597548518896833, '((0, 0, 0, 0), (1, 1, 0, 0))|1': 0.4774665042630938, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.5431034482758621, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.5045871559633027, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.44678609062170704, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.44418331374853115, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.5573604060913706, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.47116564417177914, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.4754601226993865, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.4209245742092457, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.4399164054336468, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.40569395017793597, '((1, 1, 1, 0), (1, 0, 1, 1))|0': 0.4461942257217848, '((1, 1, 1, 0), (1, 0, 1, 1))|1': 0.3902439024390244}
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

**Observed (real) value:** 0.2221 (var=0.0005)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0010 (var=0.0019)
- pi_4: 0.0012 (var=0.0015)
- pi_1: 0.0457 (var=0.0014)
- pi_2: 0.0494 (var=0.0009)
- pi_3: 0.0575 (var=0.0011)


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
  "theory": "Decision-makers use a boundedly rational strategy called Tallying with Validity-Ranked Tie-Breaking to minimize cognitive load while still making informed choices. They first evaluate options by simply counting the number of positive features for each (Tallying) and prefer the option with the highest tally. If the tallies are tied, they fall back to a non-compensatory tie-breaker, sequentially inspecting cues in descending order of their validity until one discriminates between the options. The resulting scores (tallies plus a potential tie-breaking bonus) are passed through a softmax function to generate choice probabilities, allowing for stochasticity, and a lapse rate is included to account for random guessing.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Expected a (2, n_features) stimulus.\")\n\n    a, b = stim[0], stim[1]\n    \n    # Tally positive features\n    score_a = np.sum(a)\n    score_b = np.sum(b)\n    \n    # Validity-Ranked Tie-Breaking\n    if score_a == score_b:\n        val = np.asarray(parameters[\"validities\"], dtype=float)\n        cue_order = np.argsort(-val, kind=\"stable\").tolist()\n        tie_bonus = float(parameters[\"tie_bonus\"])\n        \n        for j in cue_order:\n            if a[j] > b[j]:\n                score_a += tie_bonus\n                break\n            elif b[j] > a[j]:\n                score_b += tie_bonus\n                break\n                \n    scores = np.array([score_a, score_b])\n\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Softmax with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()  # Ensure valid probabilities\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "tie_bonus": "[0.1, 10.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's guidance, this theory replaces strict one-reason decision making (Take-The-Best) and compensatory weighted sums (WADD) with a simpler boundedly rational approach: Tallying. Subjects count the number of positive features and choose the option with the higher tally. When a tie occurs, they rely on a non-compensatory tie-breaker, inspecting cues in descending order of validity. A separate 'tie_bonus' parameter controls how strongly the tie-breaker influences the softmax score, allowing the model to smoothly integrate the tie-breaker into the stochastic choice framework alongside a lapse rate."
}
```
```

## Usage

```json
{
  "prompt_token_count": 25991,
  "candidates_token_count": 844,
  "total_token_count": 28819
}
```
