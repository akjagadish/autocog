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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a brand-new theory based on an Additive Difference Model with Probabilistic Feature Comparison. Instead of calculating absolute values for each option (as in WADD) or strictly applying a lexicographic stopping rule (as in TTB), assume decision-makers compare options attribute-by-attribute. The differences on each attribute are weighted non-linearly by their stated validities and accumulated into a relative evidence signal. Introduce a stochastic element where attention to features is proportional to their validities, and use a softmax function over the accumulated differences with a lapse rate to account for baseline noise.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.482)

**Description**
Take-The-Best with Validity-Scaled Confidence and Choice Inertia: Decision makers use a fast-and-frugal lexicographic heuristic, comparing options sequentially on features ordered by their stated validities. They stop at the first feature that discriminates between the options. However, the confidence (and therefore determinism) of their choice scales with the validity of the cue that resolved the decision. The final choice is also subject to an autoregressive motor/spatial bias (inertia) from the immediately preceding trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    scores = np.zeros(2)
    
    # Take-The-Best heuristic: find the first discriminating cue
    for j in cue_order:
        if a[j] > b[j]:
            scores[0] += val[j]
            break
        if b[j] > a[j]:
            scores[1] += val[j]
            break
            
    # Add choice inertia from the previous trial
    if history and "response" in history and len(history["response"]) > 0:
        last_choice = int(history["response"][-1])
        inertia = float(parameters["inertia"])
        scores[last_choice] += inertia
        
    # Softmax and lapse
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- inertia: [-5.0, 5.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2689 (var=0.0008) vs this=0.0711 (var=0.0029)
- Experiment 2: real=0.2318 (var=0.0107) vs this=0.0533 (var=0.0042)
- Experiment 3: real=0.2723 (var=0.0005) vs this=0.0578 (var=0.0041)
- Experiment 4: real=0.2460 (var=0.0012) vs this=0.0549 (var=0.0036)
- Experiment 5: real=0.2611 (var=0.0002) vs this=0.0580 (var=0.0029)
- Experiment 6: real=0.2658 (var=0.0017) vs this=0.0568 (var=0.0056)
- Experiment 7: real=0.2391 (var=0.0008) vs this=0.0431 (var=0.0030)
- Experiment 8: real=0.2027 (var=0.0000) vs this=0.0089 (var=0.0014)
- Experiment 9: real=0.2411 (var=0.0027) vs this=0.0263 (var=0.0031)
- Experiment 10: real=0.2314 (var=0.0001) vs this=0.0007 (var=0.0015)


---

### `pi_2` (overall score: 0.273)

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
- Experiment 5: real=0.2611 (var=0.0002) vs this=0.0037 (var=0.0002)
- Experiment 6: real=0.2658 (var=0.0017) vs this=0.0441 (var=0.0003)
- Experiment 7: real=0.2391 (var=0.0008) vs this=0.0093 (var=0.0001)
- Experiment 8: real=0.2027 (var=0.0000) vs this=0.0579 (var=0.0013)
- Experiment 9: real=0.2411 (var=0.0027) vs this=0.0118 (var=0.0002)
- Experiment 10: real=0.2314 (var=0.0001) vs this=0.0555 (var=0.0007)


---

### `pi_5` (overall score: 0.233)

**Description**
Choice-Inertia Weighted Additive Model: Decision makers evaluate options using a compensatory Weighted Additive (WADD) strategy, but their final valuation is biased by their choice on the immediately preceding trial. This history-dependent inertia acts as an autoregressive bias on the chosen response side (e.g., a motor or spatial bias), allowing the model to capture sequential dependencies such as the tendency to repeat or alternate responses independently of the option features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    # WADD base valuation
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    scores = np.dot(stim, weights)
    
    # Add choice inertia from the previous trial
    if history and "response" in history and len(history["response"]) > 0:
        last_choice = int(history["response"][-1])
        inertia = float(parameters["inertia"])
        scores[last_choice] += inertia
        
    # Softmax and lapse
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- inertia: [-5.0, 5.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2689 (var=0.0008) vs this=0.0350 (var=0.0029)
- Experiment 2: real=0.2318 (var=0.0107) vs this=0.0156 (var=0.0021)
- Experiment 3: real=0.2723 (var=0.0005) vs this=0.0450 (var=0.0019)
- Experiment 4: real=0.2460 (var=0.0012) vs this=0.0366 (var=0.0022)
- Experiment 5: real=0.2611 (var=0.0002) vs this=0.0187 (var=0.0025)
- Experiment 6: real=0.2658 (var=0.0017) vs this=0.0223 (var=0.0020)
- Experiment 7: real=0.2391 (var=0.0008) vs this=0.0199 (var=0.0031)
- Experiment 8: real=0.2027 (var=0.0000) vs this=0.0009 (var=0.0013)
- Experiment 9: real=0.2411 (var=0.0027) vs this=0.0233 (var=0.0017)
- Experiment 10: real=0.2314 (var=0.0001) vs this=0.0113 (var=0.0013)


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
- pi_4: 0.0198 (var=0.0002)
- pi_5: 0.0350 (var=0.0029)
- pi_6: 0.0711 (var=0.0029)

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
- pi_4: 0.0024 (var=0.0001)
- pi_5: 0.0156 (var=0.0021)
- pi_6: 0.0533 (var=0.0042)

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
- pi_4: 0.0210 (var=0.0006)
- pi_5: 0.0450 (var=0.0019)
- pi_6: 0.0578 (var=0.0041)

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
- pi_4: 0.0093 (var=0.0004)
- pi_5: 0.0366 (var=0.0022)
- pi_6: 0.0549 (var=0.0036)

### Experiment 5
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0), (1, 1, 1, 1))|0': 0.848851269649335, '((1, 1, 0, 0), (1, 1, 1, 1))|1': 0.8756423432682425, '((1, 1, 0, 1), (1, 0, 0, 1))|0': 0.1488933601609658, '((1, 1, 0, 1), (1, 0, 0, 1))|1': 0.15272448196469685, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.2138728323699422, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.20306859205776173, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.14391143911439114, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.15363128491620112, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.14601769911504425, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.19402985074626866, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8621908127208481, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8532934131736527, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.6896551724137931, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.6927860696517413, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.6811797752808989, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.6865808823529411, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.13060179257362356, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.13542688910696762, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.8680811808118081, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.8533519553072626, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.12791702679343128, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.16174183514774496, '((0, 1, 0, 0), (1, 1, 0, 0))|0': 0.8446215139442231, '((0, 1, 0, 0), (1, 1, 0, 0))|1': 0.8615090735434575, '((0, 0, 1, 0), (1, 1, 1, 0))|0': 0.8611599297012302, '((0, 0, 1, 0), (1, 1, 1, 0))|1': 0.8610271903323263, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8228346456692913, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8421733505821475, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.15481171548117154, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.132375189107413, '((1, 0, 0, 0), (1, 0, 1, 0))|0': 0.8642086330935251, '((1, 0, 0, 0), (1, 0, 1, 0))|1': 0.8473837209302325}
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

**Observed (real) value:** 0.2611 (var=0.0002)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0013 (var=0.0001)
- pi_3: 0.0030 (var=0.0001)
- pi_1: 0.0179 (var=0.0002)
- pi_2: 0.0037 (var=0.0002)
- pi_5: 0.0187 (var=0.0025)
- pi_6: 0.0580 (var=0.0029)

### Experiment 6
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8738738738738738, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.882494004796163, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.8158347676419966, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.8072100313479624, '((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.15524475524475526, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.1631336405529954, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.8671988388969522, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.8364928909952607, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.10869565217391304, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.13793103448275862, '((1, 1, 0, 0), (0, 0, 0, 0))|0': 0.12197686645636173, '((1, 1, 0, 0), (0, 0, 0, 0))|1': 0.127208480565371, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.11345454545454546, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.13176470588235295, '((1, 1, 1, 0), (1, 0, 1, 1))|0': 0.14825174825174825, '((1, 1, 1, 0), (1, 0, 1, 1))|1': 0.12442396313364056, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.12267657992565056, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.12747252747252746, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.11588921282798834, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.14719626168224298, '((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.7353448275862069, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.7328125, '((0, 0, 1, 1), (0, 1, 0, 1))|0': 0.8229461756373938, '((0, 0, 1, 1), (0, 1, 0, 1))|1': 0.8071297989031079, '((1, 0, 1, 0), (1, 1, 0, 1))|0': 0.8168044077134986, '((1, 0, 1, 0), (1, 1, 0, 1))|1': 0.8417132216014898, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.13426423200859292, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.13003452243958574, '((1, 0, 0, 1), (1, 1, 0, 0))|0': 0.8900343642611683, '((1, 0, 0, 1), (1, 1, 0, 0))|1': 0.8805031446540881, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12517193947730398, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.10624417520969245}
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

**Observed (real) value:** 0.2658 (var=0.0017)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0007 (var=0.0001)
- pi_4: 0.0123 (var=0.0003)
- pi_1: 0.0020 (var=0.0002)
- pi_2: 0.0441 (var=0.0003)
- pi_5: 0.0223 (var=0.0020)
- pi_6: 0.0568 (var=0.0056)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.5161637931034483, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.5240825688073395, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.36378205128205127, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.31037414965986393, '((0, 1, 0, 0), (1, 0, 0, 0))|0': 0.6975023126734505, '((0, 1, 0, 0), (1, 0, 0, 0))|1': 0.7009735744089013, '((0, 1, 0, 1), (0, 1, 1, 0))|0': 0.6290516206482593, '((0, 1, 0, 1), (0, 1, 1, 0))|1': 0.6380558428128231, '((1, 0, 0, 1), (0, 0, 1, 0))|0': 0.14984709480122324, '((1, 0, 0, 1), (0, 0, 1, 0))|1': 0.14184397163120568, '((0, 0, 1, 0), (0, 1, 0, 1))|0': 0.838405036726128, '((0, 0, 1, 0), (0, 1, 0, 1))|1': 0.8642266824085005, '((0, 1, 0, 1), (1, 0, 0, 1))|0': 0.6720085470085471, '((0, 1, 0, 1), (1, 0, 0, 1))|1': 0.6712962962962963, '((1, 1, 1, 1), (0, 1, 1, 0))|0': 0.13793103448275862, '((1, 1, 1, 1), (0, 1, 1, 0))|1': 0.1504297994269341, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.1532258064516129, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.16559485530546625, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8964757709251101, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8688340807174888, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.6379310344827587, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.687793427230047, '((1, 1, 0, 1), (1, 0, 0, 1))|0': 0.13604378420641125, '((1, 1, 0, 1), (1, 0, 0, 1))|1': 0.16314779270633398, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.12582014666152064, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.14469772051536176, '((0, 0, 1, 0), (1, 1, 1, 0))|0': 0.8667239896818573, '((0, 0, 1, 0), (1, 1, 1, 0))|1': 0.8445839874411303}
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

**Observed (real) value:** 0.2391 (var=0.0008)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0010 (var=0.0001)
- pi_5: 0.0199 (var=0.0031)
- pi_1: 0.0123 (var=0.0001)
- pi_2: 0.0093 (var=0.0001)
- pi_3: 0.0124 (var=0.0001)
- pi_6: 0.0431 (var=0.0030)

### Experiment 8
**Design**
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]

**Metric**
```python
P_REF = {'((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.3399103139013453, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.37664233576642336, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.45098039215686275, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.5226757369614512, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.5025536261491318, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.6321559074299634, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.4888211382113821, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.571078431372549, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.484548825710754, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.627648839556004, '((1, 0, 0, 0), (0, 1, 1, 0))|0': 0.3731527093596059, '((1, 0, 0, 0), (0, 1, 1, 0))|1': 0.4524291497975709, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.4384525205158265, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.5068637803590285, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.5042462845010616, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.5641025641025641, '((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.4496487119437939, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.48731501057082455, '((0, 1, 0, 1), (1, 1, 0, 1))|0': 0.662020905923345, '((0, 1, 0, 1), (1, 1, 0, 1))|1': 0.7273695420660277, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.4547770700636943, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.47783251231527096, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.63125, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.6452380952380953, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.5013368983957219, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.6387832699619772, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.30851063829787234, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.3018867924528302, '((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.4416167664670659, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.4807692307692308, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.5912653975363942, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.659316427783903}
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

**Observed (real) value:** 0.2027 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0009 (var=0.0013)
- pi_4: 0.0294 (var=0.0006)
- pi_1: 0.0525 (var=0.0011)
- pi_2: 0.0579 (var=0.0013)
- pi_3: 0.0495 (var=0.0010)
- pi_6: 0.0089 (var=0.0014)

### Experiment 9
**Design**
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
P_REF = {'((0, 0, 0, 0), (1, 1, 1, 1))|0': 0.8704453441295547, '((0, 0, 0, 0), (1, 1, 1, 1))|1': 0.8706326723323891, '((1, 0, 1, 0), (0, 0, 1, 0))|0': 0.15, '((1, 0, 1, 0), (0, 0, 1, 0))|1': 0.13137254901960785, '((0, 1, 0, 0), (0, 0, 1, 0))|0': 0.39369158878504673, '((0, 1, 0, 0), (0, 0, 1, 0))|1': 0.4110169491525424, '((1, 1, 0, 0), (0, 1, 1, 1))|0': 0.4532304725168756, '((1, 1, 0, 0), (0, 1, 1, 1))|1': 0.528178243774574, '((1, 0, 1, 0), (0, 1, 1, 1))|0': 0.43133802816901406, '((1, 0, 1, 0), (0, 1, 1, 1))|1': 0.5016233766233766, '((1, 0, 1, 1), (1, 1, 0, 1))|0': 0.5971638655462185, '((1, 0, 1, 1), (1, 1, 0, 1))|1': 0.5996462264150944, '((1, 1, 0, 1), (0, 0, 0, 1))|0': 0.13682432432432431, '((1, 1, 0, 1), (0, 0, 0, 1))|1': 0.12665562913907286, '((0, 0, 0, 1), (1, 0, 1, 0))|0': 0.8488745980707395, '((0, 0, 0, 1), (1, 0, 1, 0))|1': 0.8624787775891342, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.16997792494481237, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.10888252148997135, '((0, 0, 1, 0), (0, 1, 0, 1))|0': 0.8286252354048964, '((0, 0, 1, 0), (0, 1, 0, 1))|1': 0.8238482384823849, '((0, 0, 0, 0), (0, 1, 0, 1))|0': 0.8407445708376422, '((0, 0, 0, 0), (0, 1, 0, 1))|1': 0.8523409363745498, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.8274760383386581, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.8574310692669805, '((1, 0, 0, 1), (1, 0, 1, 0))|0': 0.6139705882352942, '((1, 0, 0, 1), (1, 0, 1, 0))|1': 0.6199186991869918, '((0, 0, 1, 0), (1, 0, 0, 1))|0': 0.8719298245614036, '((0, 0, 1, 0), (1, 0, 0, 1))|1': 0.8585365853658536, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.7010135135135135, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.6788079470198676}
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

**Observed (real) value:** 0.2411 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0013 (var=0.0002)
- pi_6: 0.0263 (var=0.0031)
- pi_1: 0.0222 (var=0.0002)
- pi_2: 0.0118 (var=0.0002)
- pi_3: 0.0132 (var=0.0002)
- pi_5: 0.0233 (var=0.0017)

### Experiment 10
**Design**
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.5417085427135678, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.5167701863354037, '((1, 0, 0, 1), (1, 0, 1, 0))|0': 0.5420765027322404, '((1, 0, 0, 1), (1, 0, 1, 0))|1': 0.48926553672316386, '((1, 1, 1, 1), (0, 1, 0, 1))|0': 0.4376899696048632, '((1, 1, 1, 1), (0, 1, 0, 1))|1': 0.4034440344403444, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.5974534769833496, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.47753530166880614, '((1, 1, 0, 0), (1, 0, 0, 0))|0': 0.4877750611246944, '((1, 1, 0, 0), (1, 0, 0, 0))|1': 0.44281524926686217, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.38907103825136613, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.40790960451977404, '((1, 0, 0, 1), (1, 1, 1, 1))|0': 0.581140350877193, '((1, 0, 0, 1), (1, 1, 1, 1))|1': 0.49324324324324326, '((0, 0, 1, 0), (1, 1, 1, 0))|0': 0.5845824411134903, '((0, 0, 1, 0), (1, 1, 1, 0))|1': 0.5484988452655889, '((1, 0, 0, 0), (0, 0, 1, 1))|0': 0.4152637485970819, '((1, 0, 0, 0), (0, 0, 1, 1))|1': 0.3872387238723872, '((1, 0, 1, 1), (0, 0, 0, 1))|0': 0.4424686192468619, '((1, 0, 1, 1), (0, 0, 0, 1))|1': 0.42298578199052134, '((1, 1, 1, 1), (0, 1, 0, 0))|0': 0.4067622950819672, '((1, 1, 1, 1), (0, 1, 0, 0))|1': 0.39805825242718446, '((0, 1, 1, 1), (1, 1, 1, 0))|0': 0.5773874862788145, '((0, 1, 1, 1), (1, 1, 1, 0))|1': 0.5331833520809899, '((0, 1, 1, 1), (0, 0, 0, 1))|0': 0.4966216216216216, '((0, 1, 1, 1), (0, 0, 0, 1))|1': 0.4166666666666667, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.4881720430107527, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.4264367816091954, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.5672645739910314, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.5, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.4263565891472868, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.4427083333333333}
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

**Observed (real) value:** 0.2314 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0007 (var=0.0015)
- pi_4: 0.0598 (var=0.0007)
- pi_1: 0.0588 (var=0.0009)
- pi_2: 0.0555 (var=0.0007)
- pi_3: 0.0539 (var=0.0010)
- pi_5: 0.0113 (var=0.0013)


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
  "theory": "Probabilistic Attention Additive Difference Model: Decision-makers compare options attribute-by-attribute rather than computing holistic absolute values. The relative evidence for each option is accumulated from the feature differences. However, the accumulation is doubly weighted: first, by the non-linear log-odds of the feature's validity, reflecting its objective diagnosticity; and second, by a stochastic attention mechanism where the probability of attending to a feature scales with its validity. This results in an expected evidence score that heavily discounts low-validity features. The final choice is generated via a softmax over the accumulated relative evidence, incorporating choice inertia from the previous trial and a baseline lapse rate for attentional lapses.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be of shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    val_clipped = np.clip(val, 0.5001, 0.9999)\n    \n    # Non-linear weighting based on stated validities (log-odds)\n    log_odds = np.log(val_clipped / (1.0 - val_clipped))\n    \n    # Stochastic attention: probability of attending scales with validity\n    gamma = float(parameters[\"gamma\"])\n    attention = val_clipped ** gamma\n    \n    # Expected effective weights for the accumulated difference\n    effective_weights = attention * log_odds\n    \n    scores = np.zeros(2)\n    # Accumulate relative evidence attribute-by-attribute\n    scores[0] = np.sum((a > b) * effective_weights)\n    scores[1] = np.sum((b > a) * effective_weights)\n    \n    # Add choice inertia from the previous trial\n    if history and \"response\" in history and len(history[\"response\"]) > 0:\n        last_choice = int(history[\"response\"][-1])\n        inertia = float(parameters[\"inertia\"])\n        scores[last_choice] += inertia\n        \n    # Softmax and lapse\n    beta = float(parameters[\"beta\"])\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    epsilon = float(parameters[\"epsilon\"])\n    n_opts = len(p_core)\n    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n    \n    return p_final\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "inertia": "[-5.0, 5.0]",
    "gamma": "[0.1, 5.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, this theory departs from strict lexicographic stopping (like TTB) and holistic valuation (like WADD). Instead, it uses an Additive Difference approach where features are compared one-by-one. It incorporates a non-linear weighting of the differences using log-odds, and introduces a probabilistic attention mechanism where the likelihood of considering a feature is proportional to its validity (parameterized by gamma). The expected evidence is then passed through a softmax function with choice inertia and a lapse rate, bridging the gap between compensatory and non-compensatory heuristics by heavily discounting less diagnostic cues while still allowing them to contribute probabilistically."
}
```
```

## Usage

```json
{
  "prompt_token_count": 31293,
  "candidates_token_count": 967,
  "total_token_count": 34463
}
```
