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

Propose a new theory that explicitly models sequential dependencies. The new theory should combine a core decision strategy (like Weighted Additive value computation) with a 'choice inertia' or 'stickiness' mechanism. Specifically, the model's predict function should adjust the baseline choice probabilities by increasing the likelihood of repeating the previous trial's action (extracted from the `history` object), reflecting a cognitive tendency to stick with the same physical response or side when uncertain.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.303)

**Description**
People use a Thresholded Tallying heuristic to compare options. To simplify decision making, individuals ignore cues whose validity falls below a certain subjective threshold. For the remaining cues, they disregard the exact validities and simply tally the number of positive features for each option. Decisions are then made probabilistically based on the difference in tallies using a softmax function, with occasional random lapses. The choice rule operates in a softer, probabilistic regime to account for human behavioral variance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Select features with validity at or above the subjective threshold
    mask = val >= threshold
    
    # Tally positive features for each option among selected cues
    score_a = np.sum(a[mask])
    score_b = np.sum(b[mask])
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


`parameters`:
- beta: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- threshold: [0.5, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1278 (var=0.0018) vs this=0.0288 (var=0.0009)
- Experiment 2: real=0.0115 (var=0.0003) vs this=0.0274 (var=0.0006)
- Experiment 3: real=0.0631 (var=0.0005) vs this=0.0126 (var=0.0004)
- Experiment 4: real=0.0422 (var=0.0006) vs this=0.0287 (var=0.0006)
- Experiment 5: real=0.0322 (var=0.0002) vs this=0.0281 (var=0.0006)
- Experiment 6: real=0.1498 (var=0.0008) vs this=0.0434 (var=0.0006)
- Experiment 7: real=0.0797 (var=0.0006) vs this=0.0246 (var=0.0007)
- Experiment 8: real=0.0803 (var=0.0011) vs this=0.0014 (var=0.0003)


---

### `pi_2` (overall score: 0.171)

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
- Experiment 1: real=0.1278 (var=0.0018) vs this=0.0710 (var=0.0005)
- Experiment 2: real=0.0115 (var=0.0003) vs this=0.0008 (var=0.0001)
- Experiment 3: real=0.0631 (var=0.0005) vs this=0.0113 (var=0.0002)
- Experiment 4: real=0.0422 (var=0.0006) vs this=0.0010 (var=0.0001)
- Experiment 5: real=0.0322 (var=0.0002) vs this=0.0035 (var=0.0005)
- Experiment 6: real=0.1498 (var=0.0008) vs this=0.0522 (var=0.0004)
- Experiment 7: real=0.0797 (var=0.0006) vs this=0.0080 (var=0.0001)
- Experiment 8: real=0.0803 (var=0.0011) vs this=0.0287 (var=0.0005)


---

### `pi_3` (overall score: 0.085)

**Description**
People use a Weighted Additive (WADD) strategy to compare options. They compute a subjective value for each option by summing its feature values weighted by their explicit validities. Decisions are made probabilistically using a softmax function over these values, with occasional random lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1278 (var=0.0018) vs this=0.0545 (var=0.0004)
- Experiment 2: real=0.0115 (var=0.0003) vs this=0.0086 (var=0.0001)
- Experiment 3: real=0.0631 (var=0.0005) vs this=0.0013 (var=0.0001)
- Experiment 4: real=0.0422 (var=0.0006) vs this=0.0191 (var=0.0003)
- Experiment 5: real=0.0322 (var=0.0002) vs this=0.0011 (var=0.0001)
- Experiment 6: real=0.1498 (var=0.0008) vs this=0.0258 (var=0.0002)
- Experiment 7: real=0.0797 (var=0.0006) vs this=0.0011 (var=0.0003)
- Experiment 8: real=0.0803 (var=0.0011) vs this=0.0264 (var=0.0006)


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
P_REF = {'((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8544303797468354, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8515850144092219, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.8507135016465422, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.8616422947131609, '((1, 0, 0, 0), (1, 1, 1, 0))|0': 0.8434684684684685, '((1, 0, 0, 0), (1, 1, 1, 0))|1': 0.8585526315789473, '((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.18723404255319148, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.13909774436090225, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8309278350515464, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.84106463878327, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.8507042253521127, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8477064220183487, '((0, 1, 0, 0), (1, 1, 1, 0))|0': 0.8584070796460177, '((0, 1, 0, 0), (1, 1, 1, 0))|1': 0.8328358208955224, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.1354625550660793, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.13452914798206278, '((1, 0, 1, 1), (0, 0, 0, 0))|0': 0.1414048059149723, '((1, 0, 1, 1), (0, 0, 0, 0))|1': 0.15550239234449761, '((0, 1, 1, 1), (1, 0, 1, 1))|0': 0.8586309523809523, '((0, 1, 1, 1), (1, 0, 1, 1))|1': 0.8572695035460993, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.8174442190669371, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.8569242540168325, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.1520935960591133, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.1417004048582996, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.1292817679558011, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.12960893854748604, '((1, 0, 0, 1), (1, 1, 0, 1))|0': 0.8486547085201793, '((1, 0, 0, 1), (1, 1, 0, 1))|1': 0.8458149779735683, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.17146974063400577, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.13471971066907776}
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

**Observed (real) value:** 0.1278 (var=0.0018)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0007 (var=0.0002)
- pi_2: 0.0710 (var=0.0005)
- pi_3: 0.0545 (var=0.0004)
- pi_4: 0.0021 (var=0.0001)
- pi_5: 0.0288 (var=0.0009)

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
P_REF = {'((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.15658362989323843, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.15976331360946747, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.85766092475068, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8407460545193687, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.14026602176541716, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.14285714285714285, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.1414496833216045, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.16358839050131926, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8337531486146096, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.852882703777336, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.8533834586466166, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.8027522935779816, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.13660179640718562, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.1810344827586207, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.12374042724707779, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.13941018766756033, '((0, 0, 1, 1), (0, 1, 1, 0))|0': 0.5021645021645021, '((0, 0, 1, 1), (0, 1, 1, 0))|1': 0.4897260273972603, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8671428571428571, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.835, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.12324324324324325, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.136, '((1, 1, 1, 1), (1, 0, 0, 0))|0': 0.1130820399113082, '((1, 1, 1, 1), (1, 0, 0, 0))|1': 0.13870246085011187, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.1437837837837838, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.1382857142857143, '((1, 0, 1, 0), (0, 0, 1, 1))|0': 0.512396694214876, '((1, 0, 1, 0), (0, 0, 1, 1))|1': 0.5260196905766527}
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

**Observed (real) value:** 0.0115 (var=0.0003)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0008 (var=0.0001)
- pi_1: 0.0303 (var=0.0002)
- pi_3: 0.0086 (var=0.0001)
- pi_4: 0.0328 (var=0.0002)
- pi_5: 0.0274 (var=0.0006)

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
P_REF = {'((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.14212152420185376, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.1640530759951749, '((1, 1, 0, 1), (0, 1, 1, 1))|0': 0.3063973063973064, '((1, 1, 0, 1), (0, 1, 1, 1))|1': 0.2948294829482948, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12403100775193798, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.14327485380116958, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.7960526315789473, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.78125, '((0, 0, 0, 0), (0, 0, 1, 1))|0': 0.8595764272559853, '((0, 0, 0, 0), (0, 0, 1, 1))|1': 0.8515406162464986, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.8306451612903226, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.8304721030042919, '((0, 0, 0, 1), (0, 0, 0, 0))|0': 0.18655967903711135, '((0, 0, 0, 1), (0, 0, 0, 0))|1': 0.22042341220423411, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.171875, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.18017241379310345, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.18838992332968238, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.16347237880496054, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8280542986425339, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.832014072119613, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8572727272727273, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.8328571428571429, '((0, 0, 1, 1), (1, 1, 0, 0))|0': 0.7511664074650077, '((0, 0, 1, 1), (1, 1, 0, 0))|1': 0.7865168539325843, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.16783216783216784, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.19839679358717435, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.26578073089701, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.3070469798657718, '((1, 0, 1, 1), (0, 1, 1, 1))|0': 0.2703984819734345, '((1, 0, 1, 1), (0, 1, 1, 1))|1': 0.28820375335120646}
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

**Observed (real) value:** 0.0631 (var=0.0005)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0013 (var=0.0001)
- pi_2: 0.0113 (var=0.0002)
- pi_1: 0.0218 (var=0.0002)
- pi_4: 0.0275 (var=0.0002)
- pi_5: 0.0126 (var=0.0004)

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
P_REF = {'((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.5033185840707964, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.5044642857142857, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.5038335158817087, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4791431792559188, '((1, 1, 0, 0), (1, 0, 1, 1))|0': 0.834625322997416, '((1, 1, 0, 0), (1, 0, 1, 1))|1': 0.8469785575048733, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.8716773602199817, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.846262341325811, '((1, 1, 0, 1), (1, 1, 0, 0))|0': 0.16143497757847533, '((1, 1, 0, 1), (1, 1, 0, 0))|1': 0.1461100569259962, '((1, 1, 0, 0), (1, 0, 0, 1))|0': 0.5206991720331187, '((1, 1, 0, 0), (1, 0, 0, 1))|1': 0.4950911640953717, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.5170842824601367, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.5032537960954447, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.5125, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.5021739130434782, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.16395939086294417, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.1619631901840491, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.5163265306122449, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.5109756097560976, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8736383442265795, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8642803877703207, '((0, 0, 0, 1), (1, 1, 0, 0))|0': 0.8541666666666666, '((0, 0, 0, 1), (1, 1, 0, 0))|1': 0.8308080808080808, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.88470066518847, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.8641425389755011, '((1, 0, 1, 0), (0, 0, 1, 0))|0': 0.1368653421633554, '((1, 0, 1, 0), (0, 0, 1, 0))|1': 0.13870246085011187, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.1446700507614213, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.1541501976284585}
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

**Observed (real) value:** 0.0422 (var=0.0006)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0010 (var=0.0001)
- pi_3: 0.0191 (var=0.0003)
- pi_1: 0.0493 (var=0.0004)
- pi_4: 0.0568 (var=0.0005)
- pi_5: 0.0287 (var=0.0006)

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
P_REF = {'((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.8243512974051896, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.8306389530408006, '((0, 0, 0, 1), (1, 0, 0, 1))|0': 0.75, '((0, 0, 0, 1), (1, 0, 0, 1))|1': 0.8394308943089431, '((1, 1, 0, 0), (0, 1, 1, 0))|0': 0.20378457059679767, '((1, 1, 0, 0), (0, 1, 1, 0))|1': 0.20035938903863432, '((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.831081081081081, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8326086956521739, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.8106508875739645, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.8414539829853055, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.8162650602409639, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.8257042253521126, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.7980769230769231, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.864247311827957, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.15469613259668508, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.16089385474860335, '((0, 1, 1, 1), (0, 0, 1, 1))|0': 0.1354764638346728, '((0, 1, 1, 1), (0, 0, 1, 1))|1': 0.18945102260495156, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.16691068814055637, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.1378692927484333, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8472527472527472, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.848314606741573, '((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.863013698630137, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.8474025974025974, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.22109826589595374, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.18501805054151624, '((1, 0, 1, 1), (0, 1, 0, 1))|0': 0.1837037037037037, '((1, 0, 1, 1), (0, 1, 0, 1))|1': 0.16444444444444445, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.837573385518591, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.805439330543933}
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

**Observed (real) value:** 0.0322 (var=0.0002)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0011 (var=0.0001)
- pi_4: 0.0029 (var=0.0001)
- pi_1: 0.0015 (var=0.0001)
- pi_2: 0.0035 (var=0.0005)
- pi_5: 0.0281 (var=0.0006)

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
P_REF = {'((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.8792834890965732, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.8695436507936508, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8768115942028986, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.8729729729729729, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8824175824175824, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8910112359550562, '((1, 0, 0, 0), (0, 1, 0, 1))|0': 0.19534883720930232, '((1, 0, 0, 0), (0, 1, 0, 1))|1': 0.11798107255520504, '((1, 0, 1, 0), (1, 1, 1, 1))|0': 0.8701923076923077, '((1, 0, 1, 0), (1, 1, 1, 1))|1': 0.8706030150753769, '((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.1404833836858006, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.10896309314586995, '((1, 1, 0, 0), (0, 0, 1, 0))|0': 0.13004484304932734, '((1, 1, 0, 0), (0, 0, 1, 0))|1': 0.10635155096011817, '((1, 0, 1, 1), (1, 1, 1, 1))|0': 0.8691176470588236, '((1, 0, 1, 1), (1, 1, 1, 1))|1': 0.8901785714285714, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.8641425389755011, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.876940133037694, '((0, 1, 1, 1), (1, 1, 0, 1))|0': 0.8669623059866962, '((0, 1, 1, 1), (1, 1, 0, 1))|1': 0.8917716827279466, '((0, 0, 0, 1), (1, 1, 1, 1))|0': 0.884828349944629, '((0, 0, 0, 1), (1, 1, 1, 1))|1': 0.8717948717948718, '((1, 1, 0, 0), (0, 0, 0, 1))|0': 0.11346444780635401, '((1, 1, 0, 0), (0, 0, 0, 1))|1': 0.12554872695346794, '((0, 0, 0, 1), (1, 0, 0, 0))|0': 0.8833151581243184, '((0, 0, 0, 1), (1, 0, 0, 0))|1': 0.8935447338618346, '((0, 0, 1, 0), (0, 1, 1, 0))|0': 0.8888888888888888, '((0, 0, 1, 0), (0, 1, 1, 0))|1': 0.8853333333333333, '((1, 0, 0, 0), (0, 0, 1, 0))|0': 0.1291759465478842, '((1, 0, 0, 0), (0, 0, 1, 0))|1': 0.12065136935603257}
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

**Observed (real) value:** 0.1498 (var=0.0008)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0008 (var=0.0001)
- pi_3: 0.0258 (var=0.0002)
- pi_1: 0.0020 (var=0.0004)
- pi_2: 0.0522 (var=0.0004)
- pi_5: 0.0434 (var=0.0006)

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
P_REF = {'((1, 1, 1, 0), (0, 0, 1, 1))|0': 0.14130434782608695, '((1, 1, 1, 0), (0, 0, 1, 1))|1': 0.14798850574712644, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.1374223602484472, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.16796875, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.1437389770723104, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.14114114114114115, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.7966101694915254, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.8589440504334122, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.16150442477876106, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.16517857142857142, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.14174107142857142, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.1592920353982301, '((0, 1, 0, 0), (0, 1, 0, 1))|0': 0.8240223463687151, '((0, 1, 0, 0), (0, 1, 0, 1))|1': 0.8311808118081181, '((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.2047670639219935, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.20410490307867732, '((0, 0, 0, 0), (0, 1, 0, 0))|0': 0.8211382113821138, '((0, 0, 0, 0), (0, 1, 0, 0))|1': 0.8311688311688312, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8488888888888889, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8511111111111112, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.1305767138193689, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.13847900113507378, '((0, 1, 0, 0), (0, 1, 1, 0))|0': 0.858440575321726, '((0, 1, 0, 0), (0, 1, 1, 0))|1': 0.8288100208768268, '((0, 1, 1, 0), (0, 1, 0, 1))|0': 0.25467625899280577, '((0, 1, 1, 0), (0, 1, 0, 1))|1': 0.23710407239819004, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.7866909753874203, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.7823613086770982, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.1379638439581351, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.17623497997329773, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8618290258449304, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8117408906882592}
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

**Observed (real) value:** 0.0797 (var=0.0006)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0011 (var=0.0003)
- pi_5: 0.0246 (var=0.0007)
- pi_1: 0.0026 (var=0.0001)
- pi_2: 0.0080 (var=0.0001)
- pi_4: 0.0053 (var=0.0000)

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
P_REF = {'((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.7179144385026738, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.7899239543726235, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.3987012987012987, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.525242718446602, '((1, 0, 1, 1), (0, 1, 1, 0))|0': 0.4, '((1, 0, 1, 1), (0, 1, 1, 0))|1': 0.4097826086956522, '((0, 1, 1, 0), (1, 0, 1, 0))|0': 0.5949656750572082, '((0, 1, 1, 0), (1, 0, 1, 0))|1': 0.58207343412527, '((0, 0, 1, 1), (0, 0, 1, 0))|0': 0.4528301886792453, '((0, 0, 1, 1), (0, 0, 1, 0))|1': 0.500945179584121, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.4343163538873995, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.4487666034155598, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.7379310344827587, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.7369565217391304, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.6849865951742627, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.6641366223908919, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.49841772151898733, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.4803082191780822, '((0, 0, 0, 1), (0, 0, 1, 1))|0': 0.6206467661691543, '((0, 0, 0, 1), (0, 0, 1, 1))|1': 0.6556224899598394, '((1, 1, 0, 1), (1, 1, 1, 0))|0': 0.6007853403141361, '((1, 1, 0, 1), (1, 1, 1, 0))|1': 0.6013513513513513, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.5683192261185006, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.5724563206577595, '((0, 0, 1, 0), (0, 0, 0, 1))|0': 0.4384949348769899, '((0, 0, 1, 0), (0, 0, 0, 1))|1': 0.37962128043282234, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.6468571428571429, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.6443243243243243, '((0, 1, 1, 0), (1, 1, 1, 0))|0': 0.7374517374517374, '((0, 1, 1, 0), (1, 1, 1, 0))|1': 0.7478005865102639, '((0, 1, 0, 1), (0, 1, 0, 0))|0': 0.4662857142857143, '((0, 1, 0, 1), (0, 1, 0, 0))|1': 0.46594594594594596}
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

**Observed (real) value:** 0.0803 (var=0.0011)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0014 (var=0.0003)
- pi_3: 0.0264 (var=0.0006)
- pi_1: 0.0497 (var=0.0007)
- pi_2: 0.0287 (var=0.0005)
- pi_4: 0.0616 (var=0.0007)


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
  "theory": "Decisions are driven by a Weighted Additive (WADD) evaluation of the options, combined with a sequential 'choice inertia' mechanism. Individuals compute the subjective value of each option by weighting its features by their respective validities. However, when translating these values into choices, people exhibit a tendency to repeat their physical response from the immediately preceding trial. This inertia acts as a baseline shift in the evidence required to choose the previously selected option, capturing sequential dependencies in decision-making.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Weighted Additive value computation\n    score_a = np.sum(a * val)\n    score_b = np.sum(b * val)\n    \n    beta = float(parameters[\"beta\"])\n    rho = float(parameters[\"rho\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Choice inertia: boost the logit of the previously chosen option\n    bias_a = 0.0\n    bias_b = 0.0\n    if history and \"response\" in history and len(history[\"response\"]) > 0:\n        last_resp = history[\"response\"][-1]\n        if last_resp == 0:\n            bias_a = rho\n        elif last_resp == 1:\n            bias_b = rho\n            \n    logits = np.array([beta * score_a + bias_a, beta * score_b + bias_b])\n    \n    # Numerically stable softmax\n    logits -= np.max(logits)\n    exp_logits = np.exp(logits)\n    p_core = exp_logits / np.sum(exp_logits)\n    \n    # Trembling hand lapse\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    p = np.asarray(probabilities, dtype=float)\n    p /= p.sum()\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "beta": "[0.0, 20.0]",
    "rho": "[-2.0, 5.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "The arbiter pointed out the need to capture sequential dependencies in choices, specifically choice inertia. The previous models evaluated trials independently. By incorporating a 'rho' parameter that adds a logit bonus to the option chosen on the previous trial, this model directly captures the tendency to stick with the same response side. This is combined with a Weighted Additive core strategy to evaluate the options' merits, balancing stimulus-driven value with sequential response biases."
}
```
```

## Usage

```json
{
  "prompt_token_count": 25074,
  "candidates_token_count": 722,
  "total_token_count": 26887
}
```
