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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Propose a 'Weighted Additive with Choice Inertia' theory. This theory should maintain the compensatory feature integration of the WADD model (weighting each feature by its validity) but explicitly incorporate a sequential dependence mechanism. Specifically, it should add a 'choice inertia' parameter that provides a utility bonus to the option chosen on the immediately preceding trial, thereby naturally explaining the conditional response distributions that the current memoryless models fail to capture.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.164)

**Description**
Take-The-Best (Lexicographic) Heuristic: Decision-makers evaluate features sequentially in order of decreasing cue validity. The first feature that discriminates between the two options strictly determines the preference, and all remaining lower-validity features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory strategy is formalized by assigning a binary score to the winning option based on the first discriminating cue, which is then mapped to choice probabilities via a softmax function (to capture decision noise) and a uniform lapse rate (to capture random errors).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Take-The-Best expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    # stable sort ensures consistent tie-breaking for equal validities
    cue_order = np.argsort(-validities, kind="stable")
    
    score_a = 0.0
    score_b = 0.0
    
    # Evaluate cues sequentially
    for idx in cue_order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break

    scores = np.array([score_a, score_b])
    
    # If no cues discriminate, guess randomly
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
    
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
- Experiment 1: real=0.0648 (var=0.0004) vs this=0.0014 (var=0.0002)
- Experiment 2: real=0.0885 (var=0.0006) vs this=0.0140 (var=0.0001)
- Experiment 3: real=0.0796 (var=0.0006) vs this=0.0010 (var=0.0002)
- Experiment 4: real=0.0508 (var=0.0002) vs this=0.0402 (var=0.0004)
- Experiment 5: real=0.0463 (var=0.0003) vs this=0.0301 (var=0.0001)
- Experiment 6: real=0.1035 (var=0.0008) vs this=0.0251 (var=0.0002)
- Experiment 7: real=0.0640 (var=0.0007) vs this=0.0011 (var=0.0002)
- Experiment 8: real=0.0352 (var=0.0003) vs this=0.0053 (var=0.0004)


---

### `pi_1` (overall score: 0.139)

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
- Experiment 1: real=0.0648 (var=0.0004) vs this=0.0009 (var=0.0001)
- Experiment 2: real=0.0885 (var=0.0006) vs this=0.0096 (var=0.0002)
- Experiment 3: real=0.0796 (var=0.0006) vs this=0.0015 (var=0.0000)
- Experiment 4: real=0.0508 (var=0.0002) vs this=0.0429 (var=0.0003)
- Experiment 5: real=0.0463 (var=0.0003) vs this=0.0257 (var=0.0002)
- Experiment 6: real=0.1035 (var=0.0008) vs this=0.0219 (var=0.0002)
- Experiment 7: real=0.0640 (var=0.0007) vs this=0.0021 (var=0.0001)
- Experiment 8: real=0.0352 (var=0.0003) vs this=0.0046 (var=0.0003)


---

### `pi_4` (overall score: 0.107)

**Description**
Equal Weights (Tallying) Heuristic: Decision-makers evaluate options by simply counting the number of positive features (1s) each option possesses, completely ignoring cue validities or any differential weighting. The option with the higher total count of positive features is preferred. If both options have the same number of positive features, the decision-maker guesses randomly. This represents a simple, unweighted compensatory strategy where any positive feature can offset the lack of another. Choice probabilities are generated by applying a softmax function to these unweighted sums, capturing variability in how deterministically the count difference drives choice, while an independent lapse rate accounts for random errors or attentional lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Equal Weights expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Unweighted sum of positive features (Tallying)
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0648 (var=0.0004) vs this=0.0216 (var=0.0002)
- Experiment 2: real=0.0885 (var=0.0006) vs this=0.0012 (var=0.0002)
- Experiment 3: real=0.0796 (var=0.0006) vs this=0.0593 (var=0.0004)
- Experiment 4: real=0.0508 (var=0.0002) vs this=0.0058 (var=0.0001)
- Experiment 5: real=0.0463 (var=0.0003) vs this=0.0016 (var=0.0001)
- Experiment 6: real=0.1035 (var=0.0008) vs this=0.0124 (var=0.0002)
- Experiment 7: real=0.0640 (var=0.0007) vs this=0.0599 (var=0.0002)
- Experiment 8: real=0.0352 (var=0.0003) vs this=0.0030 (var=0.0001)


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
P_REF = {'((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8566591422121896, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8610503282275711, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8595555555555555, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8562962962962963, '((1, 0, 0, 1), (1, 1, 1, 0))|0': 0.8435277382645804, '((1, 0, 0, 1), (1, 1, 1, 0))|1': 0.8395624430264357, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.1537313432835821, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.1407079646017699, '((1, 0, 0, 0), (1, 1, 0, 0))|0': 0.8465703971119134, '((1, 0, 0, 0), (1, 1, 0, 0))|1': 0.8424855491329479, '((0, 1, 0, 0), (1, 0, 0, 1))|0': 0.8467261904761905, '((0, 1, 0, 0), (1, 0, 0, 1))|1': 0.8794326241134752, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.8613861386138614, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.8447024673439768, '((1, 0, 1, 0), (0, 0, 0, 1))|0': 0.14742857142857144, '((1, 0, 1, 0), (0, 0, 0, 1))|1': 0.1783783783783784, '((0, 0, 0, 1), (1, 0, 1, 1))|0': 0.8619528619528619, '((0, 0, 0, 1), (1, 0, 1, 1))|1': 0.8514851485148515, '((1, 1, 1, 0), (0, 1, 0, 0))|0': 0.1471861471861472, '((1, 1, 1, 0), (0, 1, 0, 0))|1': 0.16517189835575485, '((1, 1, 1, 0), (1, 0, 1, 0))|0': 0.1467065868263473, '((1, 1, 1, 0), (1, 0, 1, 0))|1': 0.14752650176678445, '((1, 0, 1, 1), (0, 0, 1, 1))|0': 0.14094775212636695, '((1, 0, 1, 1), (0, 0, 1, 1))|1': 0.19202363367799113, '((0, 1, 0, 0), (1, 1, 0, 1))|0': 0.8496993987975952, '((0, 1, 0, 0), (1, 1, 0, 1))|1': 0.8708685626441199, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8471810089020771, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8534635879218473, '((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.13849509269356597, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.16081540203850508, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.1365149833518313, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.13793103448275862}
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

**Observed (real) value:** 0.0648 (var=0.0004)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0009 (var=0.0001)
- pi_2: 0.0223 (var=0.0002)
- pi_3: 0.0206 (var=0.0002)
- pi_4: 0.0216 (var=0.0002)
- pi_5: 0.0014 (var=0.0002)

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
P_REF = {'((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.8082788671023965, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.8680089485458613, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.16129032258064516, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.13183520599250936, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.4983089064261556, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.5169769989047097, '((0, 1, 0, 1), (1, 0, 1, 1))|0': 0.8590604026845637, '((0, 1, 0, 1), (1, 0, 1, 1))|1': 0.8344370860927153, '((0, 0, 0, 0), (0, 0, 0, 1))|0': 0.7848101265822784, '((0, 0, 0, 0), (0, 0, 0, 1))|1': 0.8643634037108126, '((0, 0, 1, 0), (1, 0, 1, 0))|0': 0.844950213371266, '((0, 0, 1, 0), (1, 0, 1, 0))|1': 0.8568824065633546, '((0, 0, 1, 1), (1, 1, 0, 1))|0': 0.835820895522388, '((0, 0, 1, 1), (1, 1, 0, 1))|1': 0.8622715404699739, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.138996138996139, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.12512218963831867, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.8567454798331016, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.877890841813136, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.5, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.5089974293059126, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.8239700374531835, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8773646444879322, '((0, 1, 1, 0), (0, 0, 0, 1))|0': 0.14710252600297177, '((0, 1, 1, 0), (0, 0, 0, 1))|1': 0.1251109139307897, '((0, 0, 0, 1), (1, 1, 1, 0))|0': 0.8580786026200873, '((0, 0, 0, 1), (1, 1, 1, 0))|1': 0.8687782805429864, '((0, 1, 1, 0), (1, 1, 0, 1))|0': 0.861995753715499, '((0, 1, 1, 0), (1, 1, 0, 1))|1': 0.8585402558314522, '((0, 0, 0, 0), (1, 0, 0, 0))|0': 0.8573248407643312, '((0, 0, 0, 0), (1, 0, 0, 0))|1': 0.8551724137931035, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8632326820603907, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8771220695230396}
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

**Observed (real) value:** 0.0885 (var=0.0006)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0011 (var=0.0001)
- pi_1: 0.0096 (var=0.0002)
- pi_3: 0.0051 (var=0.0002)
- pi_4: 0.0012 (var=0.0002)
- pi_5: 0.0140 (var=0.0001)

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
P_REF = {'((0, 0, 0, 0), (1, 1, 0, 1))|0': 0.8103448275862069, '((0, 0, 0, 0), (1, 1, 0, 1))|1': 0.875, '((1, 0, 0, 0), (0, 0, 0, 0))|0': 0.15528634361233482, '((1, 0, 0, 0), (0, 0, 0, 0))|1': 0.14573991031390135, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.8577857785778578, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.8359941944847605, '((1, 0, 1, 1), (1, 1, 0, 0))|0': 0.8294243070362474, '((1, 0, 1, 1), (1, 1, 0, 0))|1': 0.8444778362133734, '((0, 1, 1, 1), (0, 0, 1, 0))|0': 0.15181058495821728, '((0, 1, 1, 1), (0, 0, 1, 0))|1': 0.13493530499075784, '((1, 1, 1, 0), (0, 0, 1, 0))|0': 0.1558538404175988, '((1, 1, 1, 0), (0, 0, 1, 0))|1': 0.1655773420479303, '((1, 1, 1, 0), (0, 1, 1, 1))|0': 0.1476147614761476, '((1, 1, 1, 0), (0, 1, 1, 1))|1': 0.14804063860667635, '((0, 1, 0, 1), (0, 1, 1, 1))|0': 0.8198847262247838, '((0, 1, 0, 1), (0, 1, 1, 1))|1': 0.8670886075949367, '((0, 0, 1, 0), (1, 0, 0, 1))|0': 0.8489525909592062, '((0, 0, 1, 0), (1, 0, 0, 1))|1': 0.8689809630459127, '((1, 1, 1, 1), (1, 1, 1, 0))|0': 0.14231974921630094, '((1, 1, 1, 1), (1, 1, 1, 0))|1': 0.15261845386533665, '((0, 1, 1, 1), (1, 0, 1, 0))|0': 0.8493449781659389, '((0, 1, 1, 1), (1, 0, 1, 0))|1': 0.8506787330316742, '((1, 0, 1, 1), (1, 1, 1, 0))|0': 0.8577680525164114, '((1, 0, 1, 1), (1, 1, 1, 0))|1': 0.8295711060948081, '((0, 0, 0, 0), (0, 0, 1, 0))|0': 0.8246013667425968, '((0, 0, 0, 0), (0, 0, 1, 0))|1': 0.8708765315739868, '((1, 0, 0, 1), (1, 0, 1, 1))|0': 0.8229166666666666, '((1, 0, 0, 1), (1, 0, 1, 1))|1': 0.851063829787234, '((1, 1, 0, 1), (1, 1, 0, 0))|0': 0.12705530642750373, '((1, 1, 0, 1), (1, 1, 0, 0))|1': 0.13793103448275862}
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

**Observed (real) value:** 0.0796 (var=0.0006)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0015 (var=0.0000)
- pi_3: 0.0493 (var=0.0003)
- pi_2: 0.0591 (var=0.0004)
- pi_4: 0.0593 (var=0.0004)
- pi_5: 0.0010 (var=0.0002)

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
P_REF = {'((1, 0, 1, 1), (1, 0, 0, 1))|0': 0.1539039039039039, '((1, 0, 1, 1), (1, 0, 0, 1))|1': 0.1517094017094017, '((1, 0, 1, 0), (0, 1, 1, 1))|0': 0.8102815177478581, '((1, 0, 1, 0), (0, 1, 1, 1))|1': 0.8331637843336724, '((0, 0, 0, 1), (0, 1, 0, 1))|0': 0.8107569721115537, '((0, 0, 0, 1), (0, 1, 0, 1))|1': 0.847457627118644, '((1, 1, 1, 0), (0, 1, 0, 1))|0': 0.14677103718199608, '((1, 1, 1, 0), (0, 1, 0, 1))|1': 0.15938303341902313, '((0, 0, 1, 1), (0, 0, 0, 1))|0': 0.1515837104072398, '((0, 0, 1, 1), (0, 0, 0, 1))|1': 0.1462882096069869, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.478494623655914, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.4578313253012048, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.15146579804560262, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.1888111888111888, '((0, 1, 1, 0), (0, 1, 0, 0))|0': 0.1648706896551724, '((0, 1, 1, 0), (0, 1, 0, 0))|1': 0.16628440366972477, '((1, 1, 1, 1), (1, 1, 0, 1))|0': 0.14054600606673406, '((1, 1, 1, 1), (1, 1, 0, 1))|1': 0.1467324290998767, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8410404624277457, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.8582677165354331, '((1, 0, 1, 0), (1, 0, 1, 1))|0': 0.8095890410958904, '((1, 0, 1, 0), (1, 0, 1, 1))|1': 0.8411214953271028, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.15300546448087432, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.11797752808988764, '((1, 1, 0, 1), (0, 0, 0, 1))|0': 0.14821591948764867, '((1, 1, 0, 1), (0, 0, 0, 1))|1': 0.14742014742014742, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8242640499553969, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.801178203240059, '((1, 0, 1, 0), (1, 1, 0, 0))|0': 0.5322878228782287, '((1, 0, 1, 0), (1, 1, 0, 0))|1': 0.5111731843575419, '((1, 1, 0, 0), (1, 1, 0, 1))|0': 0.8487903225806451, '((1, 1, 0, 0), (1, 1, 0, 1))|1': 0.8514851485148515}
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

**Observed (real) value:** 0.0508 (var=0.0002)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0008 (var=0.0001)
- pi_1: 0.0429 (var=0.0003)
- pi_2: 0.0043 (var=0.0001)
- pi_4: 0.0058 (var=0.0001)
- pi_5: 0.0402 (var=0.0004)

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
P_REF = {'((0, 0, 1, 0), (1, 1, 1, 0))|0': 0.8725314183123878, '((0, 0, 1, 0), (1, 1, 1, 0))|1': 0.8367346938775511, '((0, 1, 0, 1), (1, 1, 0, 0))|0': 0.5176151761517616, '((0, 1, 0, 1), (1, 1, 0, 0))|1': 0.4996505939902166, '((1, 0, 1, 1), (1, 0, 1, 0))|0': 0.13953488372093023, '((1, 0, 1, 1), (1, 0, 1, 0))|1': 0.18456375838926176, '((0, 1, 1, 0), (1, 0, 0, 0))|0': 0.14543404735062007, '((0, 1, 1, 0), (1, 0, 0, 0))|1': 0.15553121577217963, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.16082659478885894, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.17321688500727803, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8301886792452831, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8435374149659864, '((1, 0, 1, 0), (1, 0, 0, 0))|0': 0.15270935960591134, '((1, 0, 1, 0), (1, 0, 0, 0))|1': 0.14394904458598726, '((1, 1, 1, 1), (0, 0, 1, 0))|0': 0.15707620528771385, '((1, 1, 1, 1), (0, 0, 1, 0))|1': 0.12718786464410736, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.4788732394366197, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.5240963855421686, '((1, 0, 0, 0), (1, 0, 1, 0))|0': 0.860236220472441, '((1, 0, 0, 0), (1, 0, 1, 0))|1': 0.8507653061224489, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8608597285067874, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8438864628820961, '((0, 1, 0, 0), (1, 1, 0, 0))|0': 0.8354285714285714, '((0, 1, 0, 0), (1, 1, 0, 0))|1': 0.8475675675675676, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.1678463094034378, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.16522811344019728, '((0, 0, 1, 1), (1, 1, 1, 0))|0': 0.856425702811245, '((0, 0, 1, 1), (1, 1, 1, 0))|1': 0.8383084577114428, '((1, 1, 0, 0), (1, 1, 1, 1))|0': 0.8713450292397661, '((1, 1, 0, 0), (1, 1, 1, 1))|1': 0.8817204301075269, '((1, 1, 0, 1), (1, 0, 0, 1))|0': 0.13852813852813853, '((1, 1, 0, 1), (1, 0, 0, 1))|1': 0.13911472448057813}
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

**Observed (real) value:** 0.0463 (var=0.0003)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0016 (var=0.0001)
- pi_3: 0.0063 (var=0.0001)
- pi_1: 0.0257 (var=0.0002)
- pi_2: 0.0011 (var=0.0002)
- pi_5: 0.0301 (var=0.0001)

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
P_REF = {'((1, 0, 1, 0), (1, 1, 0, 1))|0': 0.8346379647749511, '((1, 0, 1, 0), (1, 1, 0, 1))|1': 0.8277634961439588, '((1, 1, 1, 0), (1, 0, 0, 1))|0': 0.12027833001988071, '((1, 1, 1, 0), (1, 0, 0, 1))|1': 0.14231738035264482, '((0, 0, 0, 1), (0, 1, 0, 0))|0': 0.8267477203647416, '((0, 0, 0, 1), (0, 1, 0, 0))|1': 0.7954545454545454, '((1, 1, 0, 1), (0, 0, 1, 1))|0': 0.1488862837045721, '((1, 1, 0, 1), (0, 0, 1, 1))|1': 0.12777191129883844, '((1, 0, 0, 1), (1, 1, 0, 0))|0': 0.8195020746887967, '((1, 0, 0, 1), (1, 1, 0, 0))|1': 0.8163841807909604, '((1, 1, 1, 0), (1, 0, 1, 1))|0': 0.1988388969521045, '((1, 1, 1, 0), (1, 0, 1, 1))|1': 0.1485148514851485, '((1, 0, 1, 0), (0, 0, 0, 0))|0': 0.11472448057813911, '((1, 0, 1, 0), (0, 0, 0, 0))|1': 0.13852813852813853, '((0, 0, 1, 0), (0, 0, 1, 1))|0': 0.8232104121475055, '((0, 0, 1, 0), (0, 0, 1, 1))|1': 0.8428246013667426, '((1, 1, 0, 0), (0, 0, 0, 0))|0': 0.11304347826086956, '((1, 1, 0, 0), (0, 0, 0, 0))|1': 0.15327102803738318, '((1, 0, 0, 1), (0, 0, 1, 1))|0': 0.4293286219081272, '((1, 0, 0, 1), (0, 0, 1, 1))|1': 0.42065868263473055, '((1, 1, 1, 1), (0, 0, 0, 0))|0': 0.12097407698350353, '((1, 1, 1, 1), (0, 0, 0, 0))|1': 0.12144212523719165, '((0, 0, 1, 1), (0, 0, 0, 0))|0': 0.1331521739130435, '((0, 0, 1, 1), (0, 0, 0, 0))|1': 0.12593984962406016, '((0, 0, 1, 1), (0, 1, 0, 1))|0': 0.572405929304447, '((0, 0, 1, 1), (0, 1, 0, 1))|1': 0.6099518459069021, '((0, 1, 0, 1), (0, 0, 0, 1))|0': 0.1233140655105973, '((0, 1, 0, 1), (0, 0, 0, 1))|1': 0.13910761154855644, '((0, 0, 1, 1), (1, 0, 0, 0))|0': 0.14508138711960367, '((0, 0, 1, 1), (1, 0, 0, 0))|1': 0.1834625322997416, '((0, 0, 1, 0), (1, 1, 0, 1))|0': 0.8810810810810811, '((0, 0, 1, 0), (1, 1, 0, 1))|1': 0.8742857142857143}
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

**Observed (real) value:** 0.1035 (var=0.0008)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0008 (var=0.0001)
- pi_4: 0.0124 (var=0.0002)
- pi_1: 0.0219 (var=0.0002)
- pi_2: 0.0120 (var=0.0002)
- pi_5: 0.0251 (var=0.0002)

### Experiment 7
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
P_REF = {'((0, 1, 0, 1), (0, 0, 1, 0))|0': 0.16690442225392296, '((0, 1, 0, 1), (0, 0, 1, 0))|1': 0.1492265696087352, '((0, 0, 1, 0), (1, 0, 0, 0))|0': 0.8384728340675477, '((0, 0, 1, 0), (1, 0, 0, 0))|1': 0.8489722966934763, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.10912343470483005, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.1495601173020528, '((1, 0, 0, 0), (1, 0, 1, 1))|0': 0.8663101604278075, '((1, 0, 0, 0), (1, 0, 1, 1))|1': 0.8672566371681416, '((1, 0, 0, 1), (1, 0, 0, 0))|0': 0.18376550169109357, '((1, 0, 0, 1), (1, 0, 0, 0))|1': 0.14895947426067907, '((1, 0, 0, 0), (1, 0, 0, 1))|0': 0.8370786516853933, '((1, 0, 0, 0), (1, 0, 0, 1))|1': 0.8252747252747252, '((0, 1, 0, 0), (0, 0, 0, 0))|0': 0.14014598540145987, '((0, 1, 0, 0), (0, 0, 0, 0))|1': 0.14977578475336323, '((1, 0, 0, 1), (0, 1, 1, 1))|0': 0.15948275862068967, '((1, 0, 0, 1), (0, 1, 1, 1))|1': 0.13677536231884058, '((1, 1, 0, 0), (1, 1, 1, 0))|0': 0.8485401459854015, '((1, 1, 0, 0), (1, 1, 1, 0))|1': 0.84375, '((0, 1, 1, 1), (1, 1, 0, 0))|0': 0.8449612403100775, '((0, 1, 1, 1), (1, 1, 0, 0))|1': 0.8315789473684211, '((1, 0, 0, 0), (0, 1, 1, 0))|0': 0.1412894375857339, '((1, 0, 0, 0), (0, 1, 1, 0))|1': 0.15779645191409897, '((0, 0, 1, 1), (0, 1, 1, 1))|0': 0.8469601677148847, '((0, 0, 1, 1), (0, 1, 1, 1))|1': 0.8669690098261527, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8352941176470589, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8660714285714286, '((1, 0, 0, 0), (1, 1, 0, 1))|0': 0.8468271334792122, '((1, 0, 0, 0), (1, 1, 0, 1))|1': 0.8419864559819413, '((1, 0, 1, 1), (0, 0, 1, 0))|0': 0.17982456140350878, '((1, 0, 1, 1), (0, 0, 1, 0))|1': 0.1388888888888889, '((0, 1, 0, 1), (1, 1, 0, 1))|0': 0.8676337262012692, '((0, 1, 0, 1), (1, 1, 0, 1))|1': 0.8436154949784792}
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

**Observed (real) value:** 0.0640 (var=0.0007)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0011 (var=0.0002)
- pi_3: 0.0471 (var=0.0003)
- pi_1: 0.0021 (var=0.0001)
- pi_2: 0.0552 (var=0.0003)
- pi_4: 0.0599 (var=0.0002)

### Experiment 8
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0), (0, 1, 1, 0))|0': 0.3313343328335832, '((1, 0, 1, 0), (0, 1, 1, 0))|1': 0.2965578111209179, '((1, 1, 0, 1), (1, 0, 1, 1))|0': 0.4525462962962963, '((1, 1, 0, 1), (1, 0, 1, 1))|1': 0.3856837606837607, '((1, 0, 1, 1), (1, 1, 1, 1))|0': 0.8444444444444444, '((1, 0, 1, 1), (1, 1, 1, 1))|1': 0.8493827160493828, '((0, 1, 0, 1), (1, 1, 1, 0))|0': 0.8880208333333334, '((0, 1, 0, 1), (1, 1, 1, 0))|1': 0.8488372093023255, '((1, 0, 0, 0), (1, 1, 1, 1))|0': 0.8564417177914111, '((1, 0, 0, 0), (1, 1, 1, 1))|1': 0.8700507614213198, '((1, 0, 1, 0), (0, 0, 1, 0))|0': 0.15994236311239193, '((1, 0, 1, 0), (0, 0, 1, 0))|1': 0.1518987341772152, '((1, 1, 0, 1), (1, 1, 1, 1))|0': 0.8163841807909604, '((1, 1, 0, 1), (1, 1, 1, 1))|1': 0.8498168498168498, '((0, 0, 1, 1), (1, 0, 1, 1))|0': 0.8389154704944178, '((0, 0, 1, 1), (1, 0, 1, 1))|1': 0.8499573742540495, '((0, 0, 1, 1), (1, 1, 1, 1))|0': 0.8468606431852986, '((0, 0, 1, 1), (1, 1, 1, 1))|1': 0.8500435919790759, '((0, 1, 0, 0), (1, 1, 1, 1))|0': 0.8858773181169758, '((0, 1, 0, 0), (1, 1, 1, 1))|1': 0.8771610555050046, '((0, 0, 0, 0), (1, 1, 1, 1))|0': 0.8677517802644964, '((0, 0, 0, 0), (1, 1, 1, 1))|1': 0.8494492044063647, '((0, 0, 1, 0), (0, 1, 1, 1))|0': 0.8380952380952381, '((0, 0, 1, 0), (0, 1, 1, 1))|1': 0.8542635658914729, '((1, 1, 1, 1), (1, 1, 0, 0))|0': 0.12440944881889764, '((1, 1, 1, 1), (1, 1, 0, 0))|1': 0.1339622641509434, '((0, 1, 1, 1), (0, 1, 0, 0))|0': 0.11645569620253164, '((0, 1, 1, 1), (0, 1, 0, 0))|1': 0.14796747967479676, '((0, 1, 1, 1), (0, 0, 0, 0))|0': 0.13561190738699008, '((0, 1, 1, 1), (0, 0, 0, 0))|1': 0.12541993281075028, '((1, 0, 0, 1), (0, 0, 0, 1))|0': 0.16666666666666666, '((1, 0, 0, 1), (0, 0, 0, 1))|1': 0.1431818181818182}
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

**Observed (real) value:** 0.0352 (var=0.0003)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0014 (var=0.0001)
- pi_5: 0.0053 (var=0.0004)
- pi_1: 0.0046 (var=0.0003)
- pi_2: 0.0029 (var=0.0004)
- pi_4: 0.0030 (var=0.0001)


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
  "theory": "Weighted Additive with Choice Inertia (WADD-CI): Decision-makers integrate all available feature information in a compensatory manner by weighting each feature by its cue validity. However, choices are not made in isolation; they are subject to sequential dependence. The model incorporates 'choice inertia' by adding a utility bonus to the option that was chosen on the immediately preceding trial, capturing the empirically observed conditional response distributions where people tend to repeat their past choices. The final preference is mapped to probabilities via a softmax function and an independent lapse rate.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"WADD-CI expects a (2, n_features) stimulus.\")\n    \n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Compensatory integration: Weighted Additive (WADD)\n    score_a = np.sum(a * validities)\n    score_b = np.sum(b * validities)\n    \n    # Sequential dependence: Choice Inertia\n    phi = float(parameters[\"phi\"])\n    if len(history[\"response\"]) > 0:\n        prev_resp = history[\"response\"][-1]\n        if prev_resp == 0:\n            score_a += phi\n        elif prev_resp == 1:\n            score_b += phi\n            \n    scores = np.array([score_a, score_b])\n    \n    # Softmax choice rule with lapse rate\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "phi": "[-5.0, 5.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's guidance, this theory implements a Weighted Additive (WADD) mechanism that integrates all features weighted by their validities, preserving the compensatory nature of the evaluation. To address the mechanistic failure of previous memoryless models that could not explain trial-to-trial dependencies, we introduce a 'choice inertia' parameter (phi). This parameter adds a utility bonus to the option chosen on the previous trial, allowing the model to naturally explain the distinct conditional response distributions seen in the experimental data."
}
```
```

## Usage

```json
{
  "prompt_token_count": 26188,
  "candidates_token_count": 763,
  "total_token_count": 28066
}
```
