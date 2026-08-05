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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Propose a pure 'Take-The-Best' (TTB) heuristic theory. Unlike Theory 1, which assumes a threshold and a fallback to tallying, or Theory 2, which assumes compensatory weighted addition, this new theory should posit that individuals strictly evaluate features in descending order of their validities and make a decision based *solely* on the first discriminating cue they find. To account for noise, include a lapse rate (epsilon) and perhaps a probability of misordering the cues. This provides a stark, non-compensatory alternative to WADD without the complex, unsupported tallying fallback.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.132)

**Description**
Soft Strategy Selection: Decision makers probabilistically mix between a non-compensatory one-reason heuristic (Take The Best) and a compensatory approach (Weighted Additive rule) across trials. The probability of deploying WADD versus TTB is modeled as a smooth logistic function of the cue dispersion (the absolute difference in WADD scores). This soft mixture allows for both the compensatory integration of WADD at higher dispersions and the non-compensatory dominance of TTB at lower dispersions, without forcing a stark discontinuity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)

    beta_wadd = float(parameters["beta_wadd"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    dispersion_threshold = float(parameters["dispersion_threshold"])
    steepness = float(parameters["steepness"])

    # --- WADD Component ---
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)

    # --- TTB Component ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)

    # --- Soft Mixture based on Cue Dispersion ---
    dispersion = abs(score_a_wadd - score_b_wadd)
    x = -steepness * (dispersion - dispersion_threshold)
    
    # Numerically stable logistic function
    if x > 100:
        trial_wadd_prob = 0.0
    elif x < -100:
        trial_wadd_prob = 1.0
    else:
        trial_wadd_prob = 1.0 / (1.0 + np.exp(x))
    
    p_core = trial_wadd_prob * p_wadd + (1.0 - trial_wadd_prob) * p_ttb
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta_wadd: [0.1, 50.0]
- beta_ttb: [0.1, 50.0]
- epsilon: [0.0, 0.2]
- dispersion_threshold: [0.0, 20.0]
- steepness: [0.1, 50.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2769 (var=0.0015) vs this=0.0090 (var=0.0007)
- Experiment 2: real=0.2751 (var=0.0028) vs this=0.3414 (var=0.0050)
- Experiment 3: real=0.2223 (var=0.0310) vs this=0.0071 (var=0.0015)
- Experiment 4: real=0.2346 (var=0.0145) vs this=0.2663 (var=0.0028)
- Experiment 5: real=0.1605 (var=0.0121) vs this=0.0081 (var=0.0011)
- Experiment 6: real=0.2360 (var=0.0005) vs this=0.1757 (var=0.0010)
- Experiment 7: real=0.2709 (var=0.0337) vs this=0.0012 (var=0.0008)
- Experiment 8: real=0.2454 (var=0.0011) vs this=0.2618 (var=0.0026)
- Experiment 9: real=0.2661 (var=0.0000) vs this=0.3712 (var=0.0018)
- Experiment 10: real=0.2405 (var=0.0000) vs this=0.1281 (var=0.0016)


---

### `pi_1` (overall score: 0.109)

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
- Experiment 1: real=0.2769 (var=0.0015) vs this=0.0002 (var=0.0003)
- Experiment 2: real=0.2751 (var=0.0028) vs this=0.2646 (var=0.0073)
- Experiment 3: real=0.2223 (var=0.0310) vs this=0.0010 (var=0.0001)
- Experiment 4: real=0.2346 (var=0.0145) vs this=0.2039 (var=0.0046)
- Experiment 5: real=0.1605 (var=0.0121) vs this=0.0039 (var=0.0001)
- Experiment 6: real=0.2360 (var=0.0005) vs this=0.1202 (var=0.0028)
- Experiment 7: real=0.2709 (var=0.0337) vs this=0.0038 (var=0.0004)
- Experiment 8: real=0.2454 (var=0.0011) vs this=0.1661 (var=0.0051)
- Experiment 9: real=0.2661 (var=0.0000) vs this=0.2695 (var=0.0070)
- Experiment 10: real=0.2405 (var=0.0000) vs this=0.0940 (var=0.0019)


---

### `pi_4` (overall score: 0.098)

**Description**
Decision makers adaptively switch between a non-compensatory one-reason heuristic (Take The Best) and a compensatory approach (Weighted Additive rule) across trials. Instead of a soft probabilistic mixture, individuals use a deterministic threshold on cue dispersion (the absolute difference in weighted cue values). If the dispersion exceeds a threshold, they fully commit to the compensatory rule; otherwise, they rely entirely on the one-reason heuristic. This hard-switching mechanism, combined with independent temperature scales for each strategy, captures the high choice determinism and divergence observed in human behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)

    beta_wadd = float(parameters["beta_wadd"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    dispersion_threshold = float(parameters["dispersion_threshold"])

    # --- WADD Component ---
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)

    # --- TTB Component ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)

    # --- Adaptive Mixture based on Cue Dispersion ---
    dispersion = abs(score_a_wadd - score_b_wadd)
    trial_wadd_prob = 1.0 if dispersion >= dispersion_threshold else 0.0
    
    p_core = trial_wadd_prob * p_wadd + (1.0 - trial_wadd_prob) * p_ttb
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta_wadd: [0.1, 20.0]
- beta_ttb: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- dispersion_threshold: [0.0, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2769 (var=0.0015) vs this=0.0075 (var=0.0072)
- Experiment 2: real=0.2751 (var=0.0028) vs this=0.2287 (var=0.0090)
- Experiment 3: real=0.2223 (var=0.0310) vs this=0.0013 (var=0.0003)
- Experiment 4: real=0.2346 (var=0.0145) vs this=0.1891 (var=0.0057)
- Experiment 5: real=0.1605 (var=0.0121) vs this=0.0017 (var=0.0008)
- Experiment 6: real=0.2360 (var=0.0005) vs this=0.0974 (var=0.0029)
- Experiment 7: real=0.2709 (var=0.0337) vs this=0.0089 (var=0.0014)
- Experiment 8: real=0.2454 (var=0.0011) vs this=0.1594 (var=0.0076)
- Experiment 9: real=0.2661 (var=0.0000) vs this=0.2421 (var=0.0087)
- Experiment 10: real=0.2405 (var=0.0000) vs this=0.0872 (var=0.0025)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.14435946462715105, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.1596409055425449, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.1597576530612245, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.15912636505460218, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.8542864283929018, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.8399058269570335, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.1660264618011097, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.15847482871611557, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.8423000284656988, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.8277689454160042}
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

**Observed (real) value:** 0.2769 (var=0.0015)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0002 (var=0.0003)
- pi_2: 0.2684 (var=0.0082)
- pi_3: 0.2532 (var=0.0067)
- pi_4: 0.0075 (var=0.0072)
- pi_5: 0.0090 (var=0.0007)
- pi_6: 0.2827 (var=0.0071)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.8737623762376238, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.8550632911392405, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|0': 0.8492590700051099, '((1, 1, 1, 0, 0), (1, 1, 0, 1, 1))|1': 0.8212956068503351, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.14496833216045038, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.1294171638366223, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8678861788617886, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8342105263157895, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.13399280575539568, '((0, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.14752906976744187, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.12575841147269717, '((1, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.13430330162283155, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.1375, '((1, 1, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.18194444444444444, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|0': 0.14272970561998216, '((0, 0, 0, 1, 1), (0, 0, 1, 0, 0))|1': 0.16715758468335787}
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

**Observed (real) value:** 0.2751 (var=0.0028)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0003 (var=0.0001)
- pi_1: 0.2646 (var=0.0073)
- pi_3: 0.0011 (var=0.0002)
- pi_4: 0.2287 (var=0.0090)
- pi_5: 0.3414 (var=0.0050)
- pi_6: 0.0005 (var=0.0002)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((0, 1, 0, 1, 1), (0, 0, 1, 1, 1))|0': 0.14172709294660515, '((0, 1, 0, 1, 1), (0, 0, 1, 1, 1))|1': 0.22614840989399293, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.15440210249671485, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.22482014388489208, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.1329155672823219, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.18661971830985916, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.15480895915678525, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.2322695035460993, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.13498349834983497, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.24912280701754386, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.13749183540169824, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.19144981412639406, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.15349294850770745, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.2395644283121597, '((0, 1, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.14965741074648395, '((0, 1, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.21062618595825428}
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

**Observed (real) value:** 0.2223 (var=0.0310)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0010 (var=0.0001)
- pi_3: 0.1537 (var=0.0036)
- pi_2: 0.1760 (var=0.0027)
- pi_4: 0.0013 (var=0.0003)
- pi_5: 0.0071 (var=0.0015)
- pi_6: 0.1728 (var=0.0023)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.7996438112199465, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8194988253719655, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.838126540673788, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8732905386547586, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.1927236971484759, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.14327253502511234, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8148951554591468, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8589405911618379, '((0, 1, 1, 0, 0), (0, 0, 1, 1, 1))|0': 0.7894736842105263, '((0, 1, 1, 0, 0), (0, 0, 1, 1, 1))|1': 0.8147261299012272, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.799184505606524, '((1, 1, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8554595443833464}
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

**Observed (real) value:** 0.2346 (var=0.0145)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0006 (var=0.0001)
- pi_1: 0.2039 (var=0.0046)
- pi_2: 0.0008 (var=0.0001)
- pi_4: 0.1891 (var=0.0057)
- pi_5: 0.2663 (var=0.0028)
- pi_6: 0.0011 (var=0.0001)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.18787033809689788, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.43912448700410395, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.15119549929676512, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.373015873015873, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.1584089323098395, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.3119891008174387, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|0': 0.1652276295133438, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 1))|1': 0.44813829787234044, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 0))|0': 0.1494956772334294, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 0))|1': 0.30097087378640774, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.17841726618705037, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.4268292682926829, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.2128186968838527, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.4677835051546392, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.1541740674955595, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.32993630573248406}
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

**Observed (real) value:** 0.1605 (var=0.0121)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0017 (var=0.0008)
- pi_3: 0.1393 (var=0.0029)
- pi_1: 0.0039 (var=0.0001)
- pi_2: 0.1361 (var=0.0037)
- pi_5: 0.0081 (var=0.0011)
- pi_6: 0.1482 (var=0.0026)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.5051792828685259, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.4955017301038062, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.8472032742155525, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.8176661264181524, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.7561881188118812, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.7838266384778013, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|0': 0.7674157303370787, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|1': 0.830939226519337, '((1, 0, 0, 1, 1), (0, 1, 0, 1, 1))|0': 0.32517482517482516, '((1, 0, 0, 1, 1), (0, 1, 0, 1, 1))|1': 0.3354330708661417, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.6804308797127468, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.6658259773013872, '((1, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.17035110533159947, '((1, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.1901893287435456, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 0))|0': 0.8219354838709677, '((1, 0, 0, 0, 0), (0, 1, 0, 1, 0))|1': 0.8347826086956521, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.23991195891415995, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.2670157068062827, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|0': 0.2180746561886051, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))|1': 0.18162083936324167}
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

**Observed (real) value:** 0.2360 (var=0.0005)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0012 (var=0.0002)
- pi_4: 0.0974 (var=0.0029)
- pi_1: 0.1202 (var=0.0028)
- pi_2: 0.0170 (var=0.0002)
- pi_5: 0.1757 (var=0.0010)
- pi_6: 0.0183 (var=0.0002)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.05563093622795115, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.18518518518518517, '((1, 0, 1, 0, 0), (0, 1, 0, 0, 0))|0': 0.057676685621445976, '((1, 0, 1, 0, 0), (0, 1, 0, 0, 0))|1': 0.08403361344537816, '((1, 0, 0, 0, 0), (0, 0, 1, 0, 1))|0': 0.06257668711656442, '((1, 0, 0, 0, 0), (0, 0, 1, 0, 1))|1': 0.18823529411764706, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.11834561834561834, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.40310077519379844, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.07235772357723577, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.21666666666666667, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|0': 0.06280587275693311, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 1))|1': 0.1814516129032258, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.08299430431244914, '((1, 0, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.3884297520661157, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.10661322645290582, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.33658536585365856, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.06282722513089005, '((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.17511520737327188, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.05218446601941747, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.11403508771929824}
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

**Observed (real) value:** 0.2709 (var=0.0337)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0012 (var=0.0008)
- pi_3: 0.1623 (var=0.0031)
- pi_1: 0.0038 (var=0.0004)
- pi_2: 0.1787 (var=0.0032)
- pi_4: 0.0089 (var=0.0014)
- pi_6: 0.1872 (var=0.0027)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.8124672603457308, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.8235904531304047, '((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))|0': 0.4947154471544715, '((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))|1': 0.49188034188034185, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.33564814814814814, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.3330078125, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8603793466807166, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.864920744314266, '((0, 1, 1, 0, 1), (1, 0, 0, 1, 0))|0': 0.18195488721804512, '((0, 1, 1, 0, 1), (1, 0, 0, 1, 0))|1': 0.18110516934046345, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.6511009870918755, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.6682743837084673}
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

**Observed (real) value:** 0.2454 (var=0.0011)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0007 (var=0.0001)
- pi_5: 0.2618 (var=0.0026)
- pi_1: 0.1661 (var=0.0051)
- pi_2: 0.0042 (var=0.0001)
- pi_4: 0.1594 (var=0.0076)
- pi_6: 0.0054 (var=0.0002)

### Experiment 9
**Design**
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.8702415063446582, '((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.8539325842696629, '((0, 1, 0, 0, 1, 1), (1, 0, 0, 1, 0, 0))|0': 0.14868255959849436, '((0, 1, 0, 0, 1, 1), (1, 0, 0, 1, 0, 0))|1': 0.13359920239282153, '((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0))|0': 0.15307820299500832, '((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0))|1': 0.1441291040623261, '((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.8624434389140272, '((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.8510791366906475, '((0, 0, 0, 1, 1, 1), (1, 0, 1, 0, 0, 0))|0': 0.14029051987767585, '((0, 0, 0, 1, 1, 1), (1, 0, 1, 0, 0, 0))|1': 0.1676829268292683, '((1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1))|0': 0.8629354654483152, '((1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1))|1': 0.8631375080697224, '((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.8320935175345378, '((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.8529522376833396, '((0, 0, 0, 1, 1, 1), (0, 1, 1, 0, 0, 0))|0': 0.1794871794871795, '((0, 0, 0, 1, 1, 1), (0, 1, 1, 0, 0, 0))|1': 0.13326337880377753}
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

**Observed (real) value:** 0.2661 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0003 (var=0.0002)
- pi_3: 0.1337 (var=0.0037)
- pi_1: 0.2695 (var=0.0070)
- pi_2: 0.0010 (var=0.0001)
- pi_4: 0.2421 (var=0.0087)
- pi_5: 0.3712 (var=0.0018)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
P_REF = {'((0, 1, 1, 0, 1), (1, 0, 0, 1, 0))|0': 0.17696237807710172, '((0, 1, 1, 0, 1), (1, 0, 0, 1, 0))|1': 0.17767362590541116, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.7570229843122948, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.7508499271491015, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.6563795485951175, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.6572841384556866, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.24498506188647035, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.23972323972323972, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.8305821025195482, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.8242594075260208, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.333466773418735, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.3310165073848827}
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

**Observed (real) value:** 0.2405 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0005 (var=0.0002)
- pi_6: 0.0596 (var=0.0009)
- pi_1: 0.0940 (var=0.0019)
- pi_2: 0.0657 (var=0.0010)
- pi_4: 0.0872 (var=0.0025)
- pi_5: 0.1281 (var=0.0016)


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
  "theory": "Probabilistic Take-The-Best (TTB) with Cue Misordering: Decision makers primarily rely on a non-compensatory one-reason heuristic, evaluating features strictly in descending order of their validities and choosing based solely on the first discriminating cue. However, due to cognitive noise or memory retrieval failures, there is a probability that the cue hierarchy is misordered (effectively randomizing the cue search order). When the order is randomized, the first discriminating cue encountered is uniformly distributed among all available discriminating cues. This provides a stark, non-compensatory alternative to WADD, where apparent 'tallying' behavior naturally emerges from random cue misordering rather than a distinct compensatory calculation. Response noise is handled via a softmax temperature on the final cue and an independent uniform lapse rate.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n\n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    p_random = float(parameters[\"p_random_order\"])\n\n    # --- Strict TTB (Validities Order) ---\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    winner_strict = None\n    for j in cue_order:\n        if a[j] > b[j]:\n            winner_strict = 0\n            break\n        elif b[j] > a[j]:\n            winner_strict = 1\n            break\n\n    p_strict = np.array([0.5, 0.5])\n    if winner_strict is not None:\n        scores = np.array([1.0, 0.0]) if winner_strict == 0 else np.array([0.0, 1.0])\n        z = beta * (scores - np.max(scores))\n        e = np.exp(z)\n        p_strict = e / np.sum(e)\n\n    # --- Random Order TTB (Misordered Cues) ---\n    # If the cue order is fully randomized, the first discriminating cue found\n    # is uniformly selected from the set of all discriminating cues.\n    discriminating_cues = []\n    for j in range(len(val)):\n        if a[j] > b[j]:\n            discriminating_cues.append(0)\n        elif b[j] > a[j]:\n            discriminating_cues.append(1)\n\n    p_rand = np.array([0.5, 0.5])\n    if len(discriminating_cues) > 0:\n        count_0 = sum(1 for w in discriminating_cues if w == 0)\n        count_1 = sum(1 for w in discriminating_cues if w == 1)\n        prob_0_wins = count_0 / len(discriminating_cues)\n        prob_1_wins = count_1 / len(discriminating_cues)\n        \n        # Softmax probabilities if a cue favoring option 0 is found first\n        scores_0 = np.array([1.0, 0.0])\n        z_0 = beta * (scores_0 - np.max(scores_0))\n        e_0 = np.exp(z_0)\n        p_win_0 = e_0 / np.sum(e_0)\n        \n        # Softmax probabilities if a cue favoring option 1 is found first\n        scores_1 = np.array([0.0, 1.0])\n        z_1 = beta * (scores_1 - np.max(scores_1))\n        e_1 = np.exp(z_1)\n        p_win_1 = e_1 / np.sum(e_1)\n        \n        # Expected probability under random cue misordering\n        p_rand = prob_0_wins * p_win_0 + prob_1_wins * p_win_1\n\n    # --- Mixture and Lapse ---\n    p_core = (1.0 - p_random) * p_strict + p_random * p_rand\n    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n\n    return p_final\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "p_random_order": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, this theory implements a pure Take-The-Best (TTB) non-compensatory heuristic, strictly evaluating features by validity and deciding based solely on the first discriminating cue. It avoids any complex compensatory WADD fallback. To account for behavioral noise beyond simple lapses, it introduces a probability of misordering the cues (`p_random_order`). When cues are misordered (modeled as a random permutation), the first discriminating cue encountered is mathematically equivalent to a uniform draw from all discriminating cues. This elegantly produces an expected choice probability similar to tallying, but it arises naturally from a non-compensatory mechanism (random cue search) rather than a distinct compensatory calculation. A lapse rate and softmax temperature capture baseline response noise."
}
```
```

## Usage

```json
{
  "prompt_token_count": 23658,
  "candidates_token_count": 1402,
  "total_token_count": 28437
}
```
