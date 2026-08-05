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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a brand-new theory based on a non-compensatory heuristic, such as Take-The-Best (TTB) or a Lexicographic strategy. Instead of computing a weighted sum of all features, the decision-maker compares the options sequentially based on cue validities, stopping at the first cue that discriminates between the two options. This would produce a completely different choice probability profile (often more deterministic on certain trials) and could better explain the high divergence from the WADD-based predictions.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.309)

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
- Experiment 1: real=0.2326 (var=0.0093) vs this=0.0006 (var=0.0004)
- Experiment 2: real=0.2312 (var=0.0009) vs this=0.1891 (var=0.0058)
- Experiment 3: real=0.2566 (var=0.0000) vs this=0.0260 (var=0.0003)
- Experiment 4: real=0.2497 (var=0.0000) vs this=0.1648 (var=0.0027)
- Experiment 5: real=0.2658 (var=0.0000) vs this=0.1139 (var=0.0014)
- Experiment 6: real=0.2074 (var=0.0151) vs this=0.0078 (var=0.0003)
- Experiment 7: real=0.2587 (var=0.0000) vs this=0.0605 (var=0.0009)
- Experiment 8: real=0.2302 (var=0.0019) vs this=0.0802 (var=0.0024)
- Experiment 9: real=0.2398 (var=0.0004) vs this=0.0107 (var=0.0002)
- Experiment 10: real=0.2003 (var=0.0000) vs this=0.0591 (var=0.0014)


---

### `pi_4` (overall score: 0.210)

**Description**
Decision-makers use the Take-The-Best (TTB) heuristic, searching through features in descending order of their validities. The search stops at the first feature that discriminates between the options, and the option favored by that feature is chosen. If no features discriminate, the choice is a random guess. To account for behavioral noise without parameter redundancy, deviations from the deterministic TTB rule are modeled using a single lapse rate parameter (epsilon), replacing the redundant softmax temperature found in prior TTB instantiations.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    cue_order = np.argsort(-val, kind='stable')
    a, b = stim[0], stim[1]
    
    p_core = np.array([0.5, 0.5])
    for idx in cue_order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters['epsilon'])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

`parameters`:
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2326 (var=0.0093) vs this=0.0071 (var=0.0003)
- Experiment 2: real=0.2312 (var=0.0009) vs this=0.1462 (var=0.0111)
- Experiment 3: real=0.2566 (var=0.0000) vs this=0.0276 (var=0.0003)
- Experiment 4: real=0.2497 (var=0.0000) vs this=0.1108 (var=0.0041)
- Experiment 5: real=0.2658 (var=0.0000) vs this=0.0913 (var=0.0016)
- Experiment 6: real=0.2074 (var=0.0151) vs this=0.0007 (var=0.0002)
- Experiment 7: real=0.2587 (var=0.0000) vs this=0.0531 (var=0.0006)
- Experiment 8: real=0.2302 (var=0.0019) vs this=0.0533 (var=0.0032)
- Experiment 9: real=0.2398 (var=0.0004) vs this=0.0073 (var=0.0002)
- Experiment 10: real=0.2003 (var=0.0000) vs this=0.0292 (var=0.0021)


---

### `pi_6` (overall score: 0.142)

**Description**
Weighted Additive with Choice Inertia: Decision-makers evaluate options by taking a weighted sum of their features, where the weights correspond to cue validities. However, choices are not made independently across trials; they are subject to choice inertia. A stickiness parameter biases the current choice toward the option selected in the immediately preceding trial. The final decision is made probabilistically via a softmax function over the adjusted option values, with an additional lapse rate to account for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters['validities'], dtype=float)
    
    # Weighted addition of features
    v_a = np.sum(val * a)
    v_b = np.sum(val * b)
    
    # Apply choice inertia (stickiness) if there is a previous trial
    stickiness = float(parameters['stickiness'])
    if len(history['response']) > 0:
        prev_response = history['response'][-1]
        if prev_response == 0:
            v_a += stickiness
        elif prev_response == 1:
            v_b += stickiness
            
    scores = np.array([v_a, v_b])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- stickiness: [-5.0, 5.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2326 (var=0.0093) vs this=0.0992 (var=0.0038)
- Experiment 2: real=0.2312 (var=0.0009) vs this=0.0384 (var=0.0043)
- Experiment 3: real=0.2566 (var=0.0000) vs this=0.0544 (var=0.0038)
- Experiment 4: real=0.2497 (var=0.0000) vs this=0.0426 (var=0.0027)
- Experiment 5: real=0.2658 (var=0.0000) vs this=0.0391 (var=0.0042)
- Experiment 6: real=0.2074 (var=0.0151) vs this=0.0479 (var=0.0036)
- Experiment 7: real=0.2587 (var=0.0000) vs this=0.0575 (var=0.0030)
- Experiment 8: real=0.2302 (var=0.0019) vs this=0.0153 (var=0.0024)
- Experiment 9: real=0.2398 (var=0.0004) vs this=0.0293 (var=0.0028)
- Experiment 10: real=0.2003 (var=0.0000) vs this=0.0006 (var=0.0022)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.12373831775700934, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.16864864864864865, '((0, 0, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.8707455429497569, '((0, 0, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.842756183745583, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13971830985915493, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.13315068493150686, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.14070162202942285, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.15911485774499473, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.1434994379917572, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.1439312567132116, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.862147406733394, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.8509272467902995, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.14093959731543623, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.15824175824175823, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.12565812565812565, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.17738359201773837}
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

**Observed (real) value:** 0.2326 (var=0.0093)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0006 (var=0.0004)
- pi_2: 0.2322 (var=0.0071)
- pi_3: 0.2387 (var=0.0045)
- pi_4: 0.0071 (var=0.0003)
- pi_5: 0.1687 (var=0.0091)
- pi_6: 0.0992 (var=0.0038)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|0': 0.15132924335378323, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|1': 0.15049226441631505, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8864696734059098, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8513513513513513, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.1476923076923077, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.1409090909090909, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.5088640840446487, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.5267958950969214, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.14424173318129987, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.16408668730650156, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.8602006688963211, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.8486187845303867, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8741418764302059, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8512396694214877, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.5034550839091807, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.4830569574621485, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|0': 0.12566641279512566, '((0, 1, 0, 1, 1), (1, 0, 0, 0, 0))|1': 0.15547378104875806, '((1, 1, 0, 0, 1), (1, 0, 1, 1, 0))|0': 0.504950495049505, '((1, 1, 0, 0, 1), (1, 0, 1, 1, 0))|1': 0.49954001839926404, '((1, 0, 1, 1, 0), (1, 1, 0, 0, 1))|0': 0.4824976348155156, '((1, 0, 1, 1, 0), (1, 1, 0, 0, 1))|1': 0.47282204020848845, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.13087248322147652, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.13186813186813187}
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

**Observed (real) value:** 0.2312 (var=0.0009)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0009 (var=0.0001)
- pi_1: 0.1891 (var=0.0058)
- pi_3: 0.0006 (var=0.0001)
- pi_4: 0.1462 (var=0.0111)
- pi_5: 0.0111 (var=0.0004)
- pi_6: 0.0384 (var=0.0043)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.8303078137332282, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8450802512212142, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.8291666666666667, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.819047619047619, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13747054202670855, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.13043478260869565, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 1))|0': 0.8546296296296296, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 1))|1': 0.8617283950617284, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.7611386138613861, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.7583025830258303, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.2328042328042328, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.22727272727272727, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.16793168880455409, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.17618469015795868, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15034965034965034, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.13228346456692913, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|0': 0.13085234093637454, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|1': 0.1769825918762089, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8602693602693603, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8611111111111112}
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

**Observed (real) value:** 0.2566 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0008 (var=0.0001)
- pi_2: 0.0579 (var=0.0005)
- pi_1: 0.0260 (var=0.0003)
- pi_4: 0.0276 (var=0.0003)
- pi_5: 0.0506 (var=0.0005)
- pi_6: 0.0544 (var=0.0038)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 1, 1))|0': 0.8378978534418949, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 1, 1))|1': 0.8474877723432637, '((0, 0, 0, 1, 0, 0), (1, 1, 1, 0, 0, 0))|0': 0.8636604774535809, '((0, 0, 0, 1, 0, 0), (1, 1, 1, 0, 0, 0))|1': 0.8693877551020408, '((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0))|0': 0.15860517435320584, '((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0))|1': 0.15806805708013172, '((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.8434393638170974, '((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.8457178841309824, '((0, 0, 1, 1, 0, 0), (1, 1, 0, 0, 0, 0))|0': 0.5163297045101088, '((0, 0, 1, 1, 0, 0), (1, 1, 0, 0, 0, 0))|1': 0.5230400957510473, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))|0': 0.5212169735788631, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))|1': 0.49829351535836175, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0))|0': 0.1387252276379218, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0))|1': 0.13791113675706868, '((0, 0, 1, 0, 1, 1), (1, 1, 0, 0, 0, 0))|0': 0.14910262310170272, '((0, 0, 1, 0, 1, 1), (1, 1, 0, 0, 0, 0))|1': 0.16398037841625787}
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

**Observed (real) value:** 0.2497 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0025 (var=0.0001)
- pi_3: 0.0874 (var=0.0014)
- pi_1: 0.1648 (var=0.0027)
- pi_4: 0.1108 (var=0.0041)
- pi_5: 0.0155 (var=0.0005)
- pi_6: 0.0426 (var=0.0027)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8727025187202179, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8732737611697806, '((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))|0': 0.13749114103472715, '((1, 0, 1, 0, 1), (0, 1, 0, 1, 0))|1': 0.15267947421638017, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 1))|0': 0.8467741935483871, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 1))|1': 0.8539603960396039, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|0': 0.12545018007202882, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 0))|1': 0.14216634429400388, '((0, 0, 0, 0, 1), (1, 1, 1, 1, 0))|0': 0.8807600950118765, '((0, 0, 0, 0, 1), (1, 1, 1, 1, 0))|1': 0.8521008403361344, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15547263681592038, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.1371308016877637, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.8691049085659288, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.8585189644792294, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.11428571428571428, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.13272727272727272, '((1, 1, 1, 1, 0), (0, 0, 0, 0, 1))|0': 0.1178082191780822, '((1, 1, 1, 1, 0), (0, 0, 0, 0, 1))|1': 0.15080645161290324, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.8625877632898696, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.8526130358191427}
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
- pi_3: 0.0008 (var=0.0001)
- pi_4: 0.0913 (var=0.0016)
- pi_1: 0.1139 (var=0.0014)
- pi_2: 0.0010 (var=0.0001)
- pi_5: 0.0124 (var=0.0011)
- pi_6: 0.0391 (var=0.0042)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.22847522847522847, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.30434782608695654, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0))|0': 0.2122549019607843, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0))|1': 0.33484848484848484, '((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0))|0': 0.22483221476510068, '((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0))|1': 0.30781758957654726, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1))|0': 0.20794958797867183, '((1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1))|1': 0.3108320251177394, '((1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0))|0': 0.19950980392156864, '((1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0))|1': 0.3303030303030303, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.2198751800288046, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.3338735818476499, '((1, 0, 0, 1, 0, 0), (0, 1, 1, 0, 1, 1))|0': 0.21141439205955334, '((1, 0, 0, 1, 0, 0), (0, 1, 1, 0, 1, 1))|1': 0.33138686131386863, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.20784883720930233, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.3333333333333333, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|0': 0.2015376166941241, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|1': 0.3229706390328152, '((0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.19639376218323587, '((0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.32098765432098764}
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

**Observed (real) value:** 0.2074 (var=0.0151)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0007 (var=0.0002)
- pi_3: 0.1229 (var=0.0018)
- pi_1: 0.0078 (var=0.0003)
- pi_2: 0.1282 (var=0.0016)
- pi_5: 0.0778 (var=0.0026)
- pi_6: 0.0479 (var=0.0036)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
P_REF = {'((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.8195797027165556, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.7909562638991846, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.20248166948674562, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.18555008210180624, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.8475336322869955, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8538499264345267, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13093196112064037, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.1388438681793625, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.8201219512195121, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.8131127450980392, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 1))|0': 0.8556298773690078, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 1))|1': 0.8815060908084164, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.26512543039842595, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.24377791959157627, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.1711273317112733, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.13772708069286016}
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

**Observed (real) value:** 0.2587 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0008 (var=0.0001)
- pi_5: 0.0557 (var=0.0008)
- pi_1: 0.0605 (var=0.0009)
- pi_2: 0.0733 (var=0.0005)
- pi_4: 0.0531 (var=0.0006)
- pi_6: 0.0575 (var=0.0030)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
P_REF = {'((0, 1, 0, 0, 0), (1, 0, 1, 1, 0))|0': 0.7085889570552147, '((0, 1, 0, 0, 0), (1, 0, 1, 1, 0))|1': 0.753949678174371, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.6917030567685589, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.7258064516129032, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.6758156395649922, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.732199481118478, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.28841169937060346, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.27575858619539845, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.49389747762408465, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.5027760641579272}
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

**Observed (real) value:** 0.2302 (var=0.0019)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0004 (var=0.0005)
- pi_3: 0.0430 (var=0.0009)
- pi_1: 0.0802 (var=0.0024)
- pi_2: 0.0107 (var=0.0004)
- pi_4: 0.0533 (var=0.0032)
- pi_6: 0.0153 (var=0.0024)

### Experiment 9
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0), (1, 1, 0, 0))|0': 0.49550502379693284, '((1, 1, 0, 0), (1, 1, 0, 0))|1': 0.49151550614394385, '((1, 0, 0, 0), (0, 1, 0, 0))|0': 0.2635102821616451, '((1, 0, 0, 0), (0, 1, 0, 0))|1': 0.2783300198807157, '((0, 0, 1, 1), (0, 0, 1, 1))|0': 0.4987684729064039, '((0, 0, 1, 1), (0, 0, 1, 1))|1': 0.48785425101214575, '((1, 1, 1, 0), (0, 0, 0, 1))|0': 0.13649564375605033, '((1, 1, 1, 0), (0, 0, 0, 1))|1': 0.14667535853976532, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8082103912764592, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.7961783439490446, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.1878144214645053, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.1932633903920486, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.865625, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8456521739130435, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.502724795640327, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.5107296137339056}
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

**Observed (real) value:** 0.2398 (var=0.0004)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0004 (var=0.0001)
- pi_6: 0.0293 (var=0.0028)
- pi_1: 0.0107 (var=0.0002)
- pi_2: 0.0197 (var=0.0003)
- pi_4: 0.0073 (var=0.0002)
- pi_5: 0.0209 (var=0.0003)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.4691035683202785, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.5507593924860112, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 0))|0': 0.49821109123434704, '((0, 0, 1, 1, 0), (1, 1, 0, 0, 0))|1': 0.5655226209048362, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.4782958199356913, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.5268166089965398, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.49784668389319553, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.5520581113801453, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.47906602254428343, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.5267702936096719, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.47961838681699914, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.5212510024057738, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.5942519019442096, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.6458504519309778, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.4825306893295562, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.5513928914505284, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.47809878844361603, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.5357950263752826, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.47058823529411764, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.5224123879380603, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.36980491942324, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.4430794430794431, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.4470588235294118, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.4991735537190083}
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

**Observed (real) value:** 0.2003 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0006 (var=0.0022)
- pi_3: 0.0409 (var=0.0012)
- pi_1: 0.0591 (var=0.0014)
- pi_2: 0.0087 (var=0.0003)
- pi_4: 0.0292 (var=0.0021)
- pi_5: 0.0022 (var=0.0002)


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
  "theory": "Take-The-Best with Choice Inertia: Decision-makers evaluate options using a non-compensatory lexicographic heuristic (Take-The-Best), searching through features in descending order of their validities and stopping at the first feature that discriminates between the options. However, choices are not made in a vacuum; they are subject to choice inertia. A stickiness parameter biases the current choice toward the option selected in the immediately preceding trial. The deterministic TTB preference and the inertial bias are integrated probabilistically via a softmax function, with an additional lapse rate to account for random behavioral errors.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters['validities'], dtype=float)\n    # Descending validity; argsort is stable so validity ties break toward earlier index\n    cue_order = np.argsort(-val, kind='stable')\n    \n    # Take-The-Best base scores\n    base_scores = np.array([0.5, 0.5])\n    for idx in cue_order:\n        if a[idx] > b[idx]:\n            base_scores = np.array([1.0, 0.0])\n            break\n        elif b[idx] > a[idx]:\n            base_scores = np.array([0.0, 1.0])\n            break\n            \n    # Apply choice inertia (stickiness) if there is a previous trial\n    stickiness = float(parameters['stickiness'])\n    if len(history['response']) > 0:\n        prev_response = int(history['response'][-1])\n        base_scores[prev_response] += stickiness\n        \n    beta = float(parameters['beta'])\n    epsilon = float(parameters['epsilon'])\n    \n    # Numerically stable softmax\n    z = beta * (base_scores - np.max(base_scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64)\n    p /= p.sum()\n    return int(np.random.choice(len(p), p=p))",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "stickiness": "[-5.0, 5.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "The arbiter recommended shifting to a non-compensatory heuristic like Take-The-Best (TTB) to better capture the deterministic nature of choices on certain trials, which WADD failed to do. Previously, TTB models (pi_1) performed well but lacked a mechanism to account for the sequential dependencies evaluated by the metric (which conditions on the previous response). By combining the lexicographic TTB stopping rule with the choice inertia (stickiness) mechanism from pi_6, this model retains the frugality and non-compensatory choice profile of TTB while effectively explaining trial-to-trial autocorrelation."
}
```
```

## Usage

```json
{
  "prompt_token_count": 24965,
  "candidates_token_count": 810,
  "total_token_count": 27756
}
```
