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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a 'Strategy Mixture' or 'Tallying' theory. A pure TTB model is too rigid. A new theory should posit that subjects either use a simple Tallying heuristic (counting the number of positive cues for each option, ignoring cue validities) or probabilistically alternate between TTB and WADD based on trial difficulty. Tallying is a strong baseline in binary choice tasks with multiple cues, and replacing TTB with Tallying (or a strategy mixture) will provide a structurally distinct hypothesis from WADD.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.408)

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
- Experiment 1: real=0.2689 (var=0.0008) vs this=0.0249 (var=0.0002)
- Experiment 2: real=0.2318 (var=0.0107) vs this=0.0012 (var=0.0001)
- Experiment 3: real=0.2723 (var=0.0005) vs this=0.0668 (var=0.0003)
- Experiment 4: real=0.2460 (var=0.0012) vs this=0.0401 (var=0.0002)


---

### `pi_3` (overall score: 0.115)

**Description**
Weighted Additive (WADD) strategy computes a compensatory score for each option by summing its feature values weighted by the log-odds of their respective cue validities. Choice is probabilistically determined by a softmax over these aggregated scores, allowing multiple weak cues to override a single strong cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds weights, clipping to avoid log(0) or division by zero
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Calculate weighted sum for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
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
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2689 (var=0.0008) vs this=0.0170 (var=0.0001)
- Experiment 2: real=0.2318 (var=0.0107) vs this=0.0070 (var=0.0001)
- Experiment 3: real=0.2723 (var=0.0005) vs this=0.0032 (var=0.0002)
- Experiment 4: real=0.2460 (var=0.0012) vs this=0.0007 (var=0.0001)


---

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
- Experiment 1: real=0.2689 (var=0.0008) vs this=0.0008 (var=0.0001)
- Experiment 2: real=0.2318 (var=0.0107) vs this=0.0116 (var=0.0001)
- Experiment 3: real=0.2723 (var=0.0005) vs this=0.0010 (var=0.0003)
- Experiment 4: real=0.2460 (var=0.0012) vs this=0.0037 (var=0.0002)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.14532374100719425, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.14027149321266968, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.14891304347826087, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.14204545454545456, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8696969696969697, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8291666666666667, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8757575757575757, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8614035087719298, '((1, 0, 0, 1), (1, 1, 1, 0))|0': 0.8546255506607929, '((1, 0, 0, 1), (1, 1, 1, 0))|1': 0.8430493273542601, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8666666666666667, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8621621621621621, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.17159763313609466, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.12529002320185614, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.1332547169811321, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.1441717791411043, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.7886710239651417, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8806860551826995, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.854389721627409, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.873218304576144, '((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8542825361512792, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.855715871254162, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.8677007299270073, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.8480113636363636, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.14301929625425652, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.1588683351468988, '((1, 1, 1, 0), (0, 1, 0, 0))|0': 0.1291053227633069, '((1, 1, 1, 0), (0, 1, 0, 0))|1': 0.15485278080697928, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.856043956043956, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8382022471910112, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.16374269005847952, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.12724014336917563}
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

**Observed (real) value:** 0.2689 (var=0.0008)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0008 (var=0.0001)
- pi_2: 0.0249 (var=0.0002)
- pi_3: 0.0170 (var=0.0001)

### Experiment 2
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.8162162162162162, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.8586345381526105, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.832, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8754838709677419, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.1724137931034483, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.14285714285714285, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.8335483870967741, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.855609756097561, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.8704883227176221, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.871331828442438, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8712871287128713, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8636779505946935, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.842031029619182, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.846929422548121, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.18151815181518152, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.1507537688442211, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8575539568345324, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8597285067873304, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.18433179723502305, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.1575984990619137, '((0, 0, 0, 0), (1, 0, 0, 0))|0': 0.8415094339622642, '((0, 0, 0, 0), (1, 0, 0, 0))|1': 0.8497512437810946, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.84, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.8457142857142858, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.48918640576725025, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.5271411338962606, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.5010482180293501, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4837490551776266, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.8520710059171598, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.8567615658362989, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.7663230240549829, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8436050364479788}
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

**Observed (real) value:** 0.2318 (var=0.0107)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0012 (var=0.0001)
- pi_1: 0.0116 (var=0.0001)
- pi_3: 0.0070 (var=0.0001)

### Experiment 3
**Design**
  A=[0, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.14613778705636743, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.1445874337623013, '((1, 1, 1, 1), (1, 1, 1, 0))|0': 0.14887794198139026, '((1, 1, 1, 1), (1, 1, 1, 0))|1': 0.1314156796390299, '((0, 1, 0, 1), (0, 1, 1, 1))|0': 0.8539944903581267, '((0, 1, 0, 1), (0, 1, 1, 1))|1': 0.8579465541490858, '((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.8591117917304747, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.8559622195985832, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.847084708470847, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.8653198653198653, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8435277382645804, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8505013673655424, '((1, 1, 0, 1), (1, 1, 0, 0))|0': 0.16152716593245228, '((1, 1, 0, 1), (1, 1, 0, 0))|1': 0.13047363717605004, '((1, 0, 1, 1), (1, 1, 0, 0))|0': 0.8445901639344262, '((1, 0, 1, 1), (1, 1, 0, 0))|1': 0.7781818181818182, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8793103448275862, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.865036231884058, '((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.8561253561253561, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.8715846994535519, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.15529753265602322, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.1422142214221422, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.15214180206794684, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.15672306322350846, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.8633257403189066, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.8322981366459627, '((0, 0, 1, 0), (1, 0, 0, 1))|0': 0.8514705882352941, '((0, 0, 1, 0), (1, 0, 0, 1))|1': 0.8321428571428572, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.16770186335403728, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.1404707668944571}
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

**Observed (real) value:** 0.2723 (var=0.0005)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0010 (var=0.0003)
- pi_3: 0.0032 (var=0.0002)
- pi_2: 0.0668 (var=0.0003)

### Experiment 4
**Design**
  A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8877937831690674, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.8253638253638254, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.11549295774647887, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.12018348623853212, '((1, 1, 0, 0), (1, 1, 0, 1))|0': 0.763235294117647, '((1, 1, 0, 0), (1, 1, 0, 1))|1': 0.7227272727272728, '((1, 0, 1, 0), (1, 0, 1, 1))|0': 0.7636761487964989, '((1, 0, 1, 0), (1, 0, 1, 1))|1': 0.7787810383747178, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.13499480789200416, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.14814814814814814, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.28645383951682485, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.31045241809672386, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8513853904282116, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8697813121272365, '((1, 1, 0, 1), (0, 0, 0, 1))|0': 0.10682226211849193, '((1, 1, 0, 1), (0, 0, 0, 1))|1': 0.13994169096209913, '((0, 1, 1, 0), (0, 1, 0, 0))|0': 0.1196319018404908, '((0, 1, 1, 0), (0, 1, 0, 0))|1': 0.15328467153284672, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.13957446808510637, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.1376, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.6856780735107731, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.675568743818002, '((0, 0, 0, 1), (0, 1, 0, 1))|0': 0.8723897911832946, '((0, 0, 0, 1), (0, 1, 0, 1))|1': 0.8284023668639053, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.12698412698412698, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.11929824561403508, '((1, 0, 1, 0), (0, 1, 1, 1))|0': 0.1417142857142857, '((1, 0, 1, 0), (0, 1, 1, 1))|1': 0.1408, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.8236559139784946, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.8091954022988506, '((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.14899328859060404, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.12985781990521328}
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

**Observed (real) value:** 0.2460 (var=0.0012)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0007 (var=0.0001)
- pi_1: 0.0037 (var=0.0002)
- pi_2: 0.0401 (var=0.0002)


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
  "theory": "Decision makers exhibit strategy variability, probabilistic alternating between a simple heuristic and a compensatory strategy. Specifically, subjects use a Strategy Mixture of Tallying (counting the number of positive cues for each option, ignoring cue validities) and Weighted Additive (WADD, computing a compensatory score based on log-odds of cue validities). This mixture allows the model to capture both fast, unweighted evidence accumulation and more deliberative, validity-weighted integration on different trials.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Strategy Mixture expects a (2, n_features) stimulus.\")\n    \n    a, b = stim[0], stim[1]\n    \n    # Tallying component\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    scores_tally = np.array([a_wins, b_wins])\n    \n    # WADD component\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    val_clipped = np.clip(val, 0.5001, 0.9999)\n    weights = np.log(val_clipped / (1.0 - val_clipped))\n    scores_wadd = np.dot(stim, weights)\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    w_mix = float(parameters[\"w_mix\"])\n    \n    # Tallying probabilities\n    z_tally = beta * (scores_tally - np.max(scores_tally))\n    e_tally = np.exp(z_tally)\n    p_tally = e_tally / np.sum(e_tally)\n    \n    # WADD probabilities\n    z_wadd = beta * (scores_wadd - np.max(scores_wadd))\n    e_wadd = np.exp(z_wadd)\n    p_wadd = e_wadd / np.sum(e_wadd)\n    \n    # Mixture\n    p_core = w_mix * p_tally + (1.0 - w_mix) * p_wadd\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "w_mix": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "The arbiter noted that pure TTB is too rigid and suggested exploring Tallying or a Strategy Mixture. Given that Tallying alone is a strong baseline (as seen in its leaderboard performance), a mixture model that probabilistically alternates between Tallying (simple cue counting) and WADD (compensatory validity-weighted integration) provides a flexible and structurally distinct hypothesis. This model allows for individual differences in strategy reliance via the w_mix parameter, capturing behavior that is sometimes heuristic and sometimes deliberative."
}
```
```

## Usage

```json
{
  "prompt_token_count": 15255,
  "candidates_token_count": 873,
  "total_token_count": 17554
}
```
