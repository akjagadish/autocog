# generation_iter_03_attempt_00

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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Propose a new theory that replaces the fixed validity-gated tallying of pi_4: people adaptively decide whether unweighted tallying is diagnostic in the current task, and use validity to resolve ties only when tally ties are frequent. In the new theory, on each trial compute the unweighted tally difference d = #A wins - #B wins and a validity-weighted evidence V = sum_j w_j * (A_j - B_j). From the recent or global stimulus history, compute a context statistic c such as the proportion of trials with |d| <= tau or the average absolute tally. The choice probability should be a mixture p = lambda(c) * p_valid + (1 - lambda(c)) * p_tally, with lambda increasing when tie prevalence is high. For low tie prevalence, lambda should be small, recovering near-zero tie-validity alignment as in Experiments 1 and 2. For high tie prevalence, lambda should be large, producing strong validity-aligned choices as in Experiments 5 and 6, but with enough trial or subject stochasticity to avoid pi_4's overly small variance. This new theory differs from pi_3 because it has a task-composition-dependent validity route rather than continuous zero-centered validity grading on every trial, and differs from pi_4 because the validity gate is not fixed but is switched by context. The next model should use the history argument that the current theories ignore, and parameter ranges should allow both weak and strong validity reliance, with moderate between-subject variability.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.546)

**Description**
People use a validity-graded compensatory integration rule. On every choice they inspect all features, compute a signed option advantage by summing each discriminating feature's contribution, and then choose via a softmax over that advantage plus an independent lapse. Unlike Take The Best there is no early stopping. Unlike pure Tallying, each feature's contribution is weighted by a validity-sensitive multiplier. The multiplier is anchored around equal weighting, w_j = 1 + kappa * (v_j - mean(v)), normalized so the average weight is exactly 1. Because kappa has a symmetric prior centered at zero, the population-level expected weight vector is the equal-weight vector, but individual subjects may slightly overweight or underweight higher-validity cues. Tied features contribute nothing.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    # Validity-graded compensatory integration.
    # state is the current pair of option feature vectors, shape (2, n_features).
    # Row 0 = option A, row 1 = option B. History is irrelevant because there is
    # no trial-by-trial feedback and the rule is applied independently each trial.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-graded integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Validity-sensitive weights around a strict equal-weight anchor.
    # The centered validities sum to zero, so the raw weights sum to n_features.
    # Normalizing by their mean keeps the scale of the advantage score directly
    # comparable to an unweighted signed tally when kappa is small.
    kappa = float(parameters["kappa"])
    centered_validities = validities - np.mean(validities)
    weights = 1.0 + kappa * centered_validities
    weights = weights / np.mean(weights)

    # Discriminating features contribute according to their weights; ties
    # have difference zero and thus contribute nothing.
    advantage = float(np.dot(weights, stim[0] - stim[1]))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Stable softmax over the signed advantage versus a zero-advantage boundary.
    # This reduces to logistic choice in the weighted advantage.
    scores = np.array([advantage, 0.0])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
import numpy as np

def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- kappa: [-0.10, 0.10]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.6137 (var=0.0420) vs this=-0.5590 (var=0.0211)
- Experiment 2: real=0.0659 (var=0.0002) vs this=0.0682 (var=0.0002)
- Experiment 3: real=2.6289 (var=0.2854) vs this=2.0844 (var=1.0595)
- Experiment 4: real=4.2015 (var=2.0036) vs this=1.1715 (var=2.0496)
- Experiment 5: real=-0.0467 (var=0.0156) vs this=-0.0333 (var=0.1778)
- Experiment 6: real=0.0204 (var=0.0176) vs this=0.0637 (var=0.0343)


---

### `pi_2` (overall score: 0.144)

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
- Experiment 1: real=-0.6137 (var=0.0420) vs this=-0.6093 (var=0.0194)
- Experiment 2: real=0.0659 (var=0.0002) vs this=0.0679 (var=0.0003)
- Experiment 3: real=2.6289 (var=0.2854) vs this=0.6756 (var=0.2991)
- Experiment 4: real=4.2015 (var=2.0036) vs this=0.1148 (var=0.3890)
- Experiment 5: real=-0.0467 (var=0.0156) vs this=0.0300 (var=0.0391)
- Experiment 6: real=0.0204 (var=0.0176) vs this=-0.0359 (var=0.0223)


---

### `pi_4` (overall score: 0.141)

**Description**
People use a validity-gated tallying rule. On every trial they first compute the unweighted tally difference d = #A wins - #B wins. If the tallies clearly differ, choice is driven by a softmax over the two unweighted tallies, exactly as in Tallying, so majority-tally behavior is preserved. Only when the tallies are tied, or within a small threshold tau of tied, is communicated validity used: the decision maker computes validity-weighted evidence V = sum_j w_j * (A_j - B_j), with weights w_j = (v_j - 0.5)^gamma increasing steeply and convexly in validity. Choice on these tie trials is then a separate softmax over V and -V, with its own inverse temperature beta_v and lapse epsilon_v, and with probability 1 - fallback_p the subject simply guesses instead of using this validity fallback. Thus validity does not continuously reweight every trial, as in a compensatory rule, nor are ties resolved by blind guessing, as in Tallying.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    # Validity-gated tallying.
    # state is the current pair of option feature vectors, shape (2, n_features).
    # Row 0 = option A, row 1 = option B. History is irrelevant because choices
    # are made independently on each trial with no feedback.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-gated tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    a, b = stim[0], stim[1]
    diff = a - b

    # Unweighted tally of strict feature-wise wins; tied features contribute nothing.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_d = a_wins - b_wins

    tau = float(parameters["tau"])

    def stable_softmax(scores, inv_temp):
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        p = e / np.sum(e)
        return p

    if np.abs(tally_d) > tau:
        # Unequal tallies: exactly the Tallying softmax, preserving majority-tally
        # choice behavior on clearly discriminating trials.
        scores = np.array([a_wins, b_wins])
        p_core = stable_softmax(scores, float(parameters["beta_tally"]))
        eps_tally = float(parameters["epsilon_tally"])
        probs = (1.0 - eps_tally) * p_core + eps_tally * np.array([0.5, 0.5])
    else:
        # Tally tie or near-tie: validity-gated integration instead of guessing.
        gamma = float(parameters["gamma"])
        weights = np.maximum(validities - 0.5, 0.0) ** gamma
        v = float(np.dot(weights, diff))

        scores_v = np.array([v, -v])
        p_core_v = stable_softmax(scores_v, float(parameters["beta_v"]))
        eps_v = float(parameters["epsilon_v"])
        p_valid = (1.0 - eps_v) * p_core_v + eps_v * np.array([0.5, 0.5])

        fallback_p = float(parameters["fallback_p"])
        probs = fallback_p * p_valid + (1.0 - fallback_p) * np.array([0.5, 0.5])

    probs = np.clip(probs, 0.0, 1.0)
    total = probs.sum()
    if total <= 0.0:
        probs = np.array([0.5, 0.5])
    else:
        probs = probs / total
    return probs


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0.0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta_tally: [0.1, 20.0]
- epsilon_tally: [0.0, 0.5]
- tau: [0.0, 0.5]
- gamma: [6.0, 10.0]
- beta_v: [100.0, 300.0]
- epsilon_v: [0.0, 0.2]
- fallback_p: [0.7, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.6137 (var=0.0420) vs this=-0.5890 (var=0.0178)
- Experiment 2: real=0.0659 (var=0.0002) vs this=0.0713 (var=0.0002)
- Experiment 3: real=2.6289 (var=0.2854) vs this=2.9744 (var=0.0172)
- Experiment 4: real=4.2015 (var=2.0036) vs this=3.6712 (var=1.2295)
- Experiment 5: real=-0.0467 (var=0.0156) vs this=0.7550 (var=0.0326)
- Experiment 6: real=0.0204 (var=0.0176) vs this=0.5795 (var=0.0652)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3038 -> ACCEPTED
- iter 2: loss=0.1113 -> ACCEPTED
- iter 3: loss=0.1467 -> REJECTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.1113 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.84, 0.76, 0.68, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind='stable')
    A = np.array([np.asarray(row, dtype=float) for row in data['option_a_ratings']])
    B = np.array([np.asarray(row, dtype=float) for row in data['option_b_ratings']])
    resp_is_a = (data['response'].to_numpy(dtype=float) == 0.0).astype(float)

    winner = np.full(A.shape[0], -1, dtype=int)
    unresolved = np.ones(A.shape[0], dtype=bool)
    for j in cue_order:
        diff = A[:, j] - B[:, j]
        first = unresolved & (diff != 0)
        winner[first] = np.where(diff[first] > 0, 0, 1)
        unresolved[first] = False

    tally = (A > B).sum(axis=1) - (B > A).sum(axis=1)
    side = np.where(winner == 0, 1.0, np.where(winner == 1, -1.0, 0.0))
    signed_weight = side * (np.abs(tally) + 1.0)

    numerator = float(np.sum(signed_weight * resp_is_a))
    denom = float(np.sum(signed_weight[signed_weight > 0]))
    if denom == 0.0:
        return 0.0
    return numerator / denom

```

**Observed (real) value:** -0.6137 (var=0.0420)
**Previous candidate values (this loop):**
  - iter 1: -0.5620 (var=0.0269) (Δ vs real +0.0517)
  - iter 2: -0.4937 (var=0.0264) (Δ vs real +0.1200)
  - iter 3 (most recent): -0.6102 (var=0.0109) (Δ vs real +0.0035)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7347 (var=0.0373)
- pi_2: -0.6093 (var=0.0194)
- pi_3: -0.5590 (var=0.0211)
- pi_4: -0.5890 (var=0.0178)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    resp = data['response'].astype(int).tolist()

    # Highest validity is at display index 4: [0.57, 0.7, 0.5, 0.77, 0.92, 0.84, 0.63, 0.52]
    top_idx = 4

    xs = []
    ys = []

    for a, b, r in zip(a_col, b_col, resp):
        if a[top_idx] == b[top_idx]:
            continue

        top_is_a = a[top_idx] > b[top_idx]

        a_wins = 0
        b_wins = 0
        for fa, fb in zip(a, b):
            if fa > fb:
                a_wins += 1
            elif fb > fa:
                b_wins += 1

        d = a_wins - b_wins
        if d == 0:
            continue

        # Signed tally advantage for the option favored by the top cue.
        signed_tally_for_top = d if top_is_a else -d

        # 1 if the subject chose the option favored by the top cue, else 0.
        top_chosen = 1.0 if ((top_is_a and r == 0) or ((not top_is_a) and r == 1)) else 0.0

        xs.append(float(signed_tally_for_top))
        ys.append(top_chosen)

    if len(xs) < 2:
        return 0.0

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    xm = x.mean()
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0:
        return 0.0

    slope = float(np.sum((x - xm) * (y - y.mean())) / sxx)
    return slope
```

**Observed (real) value:** 0.0659 (var=0.0002)
**Previous candidate values (this loop):**
  - iter 1: 0.0664 (var=0.0002) (Δ vs real +0.0005)
  - iter 2: 0.0664 (var=0.0003) (Δ vs real +0.0005)
  - iter 3 (most recent): 0.0874 (var=0.0000) (Δ vs real +0.0215)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0679 (var=0.0003)
- pi_1: -0.0002 (var=0.0000)
- pi_3: 0.0682 (var=0.0002)
- pi_4: 0.0713 (var=0.0002)

### Experiment 3
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    centered = validities - validities.mean()

    cap = 3.0
    subj_vals = []

    for _, sub in data.groupby('subject_id'):
        a = np.asarray(sub['option_a_ratings'].tolist(), dtype=float)
        b = np.asarray(sub['option_b_ratings'].tolist(), dtype=float)
        diff = a - b
        tally = diff.sum(axis=1)
        s = diff @ centered

        mask = np.abs(tally) < 1e-9
        if int(mask.sum()) < 4:
            continue

        s0 = s[mask]
        resp = sub['response'].to_numpy(dtype=float)
        y = (resp[mask] == 0.0).astype(float)

        score = float(np.sum(s0 * (y - 0.5)))
        null_sd = float(np.sqrt(0.25 * np.sum(s0 * s0)))
        if null_sd <= 1e-12:
            continue

        z = score / null_sd
        subj_vals.append(float(min(abs(z), cap)))

    if not subj_vals:
        return 0.0
    return float(np.mean(subj_vals))

```

**Observed (real) value:** 2.6289 (var=0.2854)
**Previous candidate values (this loop):**
  - iter 1: 1.0978 (var=0.6928) (Δ vs real -1.5311)
  - iter 2: 2.9956 (var=0.0010) (Δ vs real +0.3667)
  - iter 3 (most recent): 2.8633 (var=0.1715) (Δ vs real +0.2344)
**Other theories' values on this metric (for reference):**
- pi_3: 2.0844 (var=1.0595)
- pi_2: 0.6756 (var=0.2991)
- pi_1: 2.6856 (var=0.3983)
- pi_4: 2.9744 (var=0.0172)

### Experiment 4
**Design**
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0])
    centered = validities - validities.mean()

    def one_subject_stat(sub):
        A = np.array(sub['option_a_ratings'].tolist(), dtype=float)
        B = np.array(sub['option_b_ratings'].tolist(), dtype=float)
        y = (sub['response'].to_numpy() == 0).astype(float)
        diff = A - B
        tie_trial_mask = np.isclose(np.sum(diff, axis=1), 0.0)
        s = diff.dot(centered)
        score = np.sum(s[tie_trial_mask] * y[tie_trial_mask])
        null_var = 0.25 * np.sum(s[tie_trial_mask] ** 2)
        if null_var <= 0.0:
            return 0.0
        z = score / np.sqrt(null_var)
        return float(np.abs(z) - 0.7978845608028654)

    if data['subject_id'].nunique() == 1:
        return one_subject_stat(data)

    values = [one_subject_stat(sub) for _, sub in data.groupby('subject_id', sort=False)]
    if not values:
        return 0.0
    return float(np.mean(values))

```

**Observed (real) value:** 4.2015 (var=2.0036)
**Previous candidate values (this loop):**
  - iter 1: 0.1928 (var=0.4795) (Δ vs real -4.0086)
  - iter 2: 4.7317 (var=1.3946) (Δ vs real +0.5302)
  - iter 3 (most recent): 2.3309 (var=1.6451) (Δ vs real -1.8706)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1148 (var=0.3890)
- pi_3: 1.1715 (var=2.0496)
- pi_1: 3.6874 (var=2.2057)
- pi_4: 3.6712 (var=1.2295)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    diff = a - b
    total_tally = diff.sum(axis=1)
    high_tally = diff[:, 8:].sum(axis=1)
    chose_a = (data['response'].to_numpy() == 0).astype(float)
    subject = data['subject_id'].to_numpy()

    per_subject = []
    for sid in np.unique(subject):
        idx = subject == sid
        tie = total_tally[idx] == 0.0
        high_pos = high_tally[idx] > 0
        high_neg = high_tally[idx] < 0
        pos_trials = tie & high_pos
        neg_trials = tie & high_neg
        if pos_trials.sum() > 0 and neg_trials.sum() > 0:
            diff_rate = float(chose_a[idx][pos_trials].mean() - chose_a[idx][neg_trials].mean())
            per_subject.append(diff_rate)

    if not per_subject:
        return float('nan')

    return float(np.mean(per_subject))
```

**Observed (real) value:** -0.0467 (var=0.0156)
**Previous candidate values (this loop):**
  - iter 1: 0.0233 (var=0.0378) (Δ vs real +0.0700)
  - iter 2: -0.0333 (var=0.0389) (Δ vs real +0.0133)
  - iter 3 (most recent): 0.0117 (var=0.0461) (Δ vs real +0.0583)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0333 (var=0.1778)
- pi_4: 0.7550 (var=0.0326)
- pi_1: 0.6483 (var=0.0759)
- pi_2: 0.0300 (var=0.0391)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    validities = np.array([0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0], dtype=float)
    weights = (validities - 0.5) ** 6.0

    a = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b = np.array(data['option_b_ratings'].tolist(), dtype=float)
    resp = data['response'].to_numpy(dtype=float)

    diff = a - b
    tally = (a > b).sum(axis=1) - (b > a).sum(axis=1)
    tie_mask = tally == 0

    if not np.any(tie_mask):
        return 0.0

    v = diff[tie_mask] @ weights
    y = resp[tie_mask]
    signed_accuracy = np.sign(v) * (1.0 - 2.0 * y)
    abs_v = np.abs(v)
    total_weight = float(abs_v.sum())

    if total_weight <= 0.0:
        return 0.0

    return float(np.sum(abs_v * signed_accuracy) / total_weight)

```

**Observed (real) value:** 0.0204 (var=0.0176)
**Previous candidate values (this loop):**
  - iter 1: 0.0470 (var=0.0337) (Δ vs real +0.0267)
  - iter 2: 0.1942 (var=0.0332) (Δ vs real +0.1739)
  - iter 3 (most recent): 0.0204 (var=0.0393) (Δ vs real -0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5795 (var=0.0652)
- pi_3: 0.0637 (var=0.0343)
- pi_1: 0.0104 (var=0.0124)
- pi_2: -0.0359 (var=0.0223)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
People adaptively decide whether unweighted tallying is diagnostic in the current task. They track the prevalence of exact tally ties in the stimulus history, and use that context to gate a validity-based fallback. When ties are rare, choice is essentially pure unweighted tallying. When exact ties are pervasive, validity information is recruited to break ties, yielding a context-weighted mixture of a tally policy and a validity policy. The present revision recalibrates the context gate so that moderately high tie prevalence, as in Experiments 3 and 4, recruits validity strongly and consistently, while near-chance tie prevalence in Experiments 1, 2, 5, and 6 leaves tallying dominant.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Context-gated tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    a = stim[0]
    b = stim[1]
    diff = a - b

    # Unweighted tally of strict feature-wise wins.
    wins_a = float(np.sum(a > b))
    wins_b = float(np.sum(b > a))
    tally_d = wins_a - wins_b

    tau_tie = float(parameters["tau_tie"])

    # Context statistic: prevalence of exact (or near) tally ties in recent history.
    h_a = history.get("option_a_ratings", [])
    h_b = history.get("option_b_ratings", [])
    n_past = min(len(h_a), len(h_b))

    window = int(round(float(parameters["history_window"])))
    if window < 1:
        window = 1
    start = max(0, n_past - window)

    tie_count = 0.0
    for i in range(start, n_past):
        pa = np.asarray(h_a[i], dtype=float)
        pb = np.asarray(h_b[i], dtype=float)
        past_wins = float(np.sum(pa > pb)) - float(np.sum(pb > pa))
        if abs(past_wins) <= tau_tie:
            tie_count += 1.0

    current_tie = 1.0 if abs(tally_d) <= tau_tie else 0.0
    tie_count += current_tie
    total_considered = min(n_past, window) + 1.0
    c = tie_count / total_considered if total_considered > 0.0 else 0.0

    # Context gate: a narrow, sharp ramp around the threshold.
    c0 = float(parameters["c0"])
    gate_width = float(parameters["gate_width"])
    ctx_power = float(parameters["ctx_power"])
    lambda_min = float(parameters["lambda_min"])
    lambda_max = float(parameters["lambda_max"])

    g = float(np.clip((c - c0) / max(gate_width, 1e-12), 0.0, 1.0))
    lamb = lambda_min + (lambda_max - lambda_min) * (g ** ctx_power)

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        p = e / np.sum(e)
        return p

    # Tally policy, preserving majority-tally behavior when tallies discriminate.
    beta_t = float(parameters["beta_tally"])
    eps_t = float(parameters["epsilon_tally"])
    p_tally = stable_softmax(np.array([wins_a, wins_b]), beta_t)
    p_tally = (1.0 - eps_t) * p_tally + eps_t * np.array([0.5, 0.5])

    # Validity policy, used mainly when tallies are tied.
    gamma = float(parameters["gamma"])
    weights = np.maximum(validities - 0.5, 0.0) ** gamma
    v = float(np.dot(weights, diff))

    beta_v = float(parameters["beta_v"])
    eps_v = float(parameters["epsilon_v"])
    p_valid = stable_softmax(np.array([v, -v]), beta_v)
    p_valid = (1.0 - eps_v) * p_valid + eps_v * np.array([0.5, 0.5])

    # Context-weighted mixture.
    probs = lamb * p_valid + (1.0 - lamb) * p_tally
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0.0:
        probs = np.array([0.5, 0.5])
    else:
        probs = probs / total

    return probs


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))


`parameters`:
- beta_tally: [0.1, 20.0]
- epsilon_tally: [0.0, 0.5]
- tau_tie: [0.0, 0.5]
- lambda_min: [0.0, 0.02]
- lambda_max: [0.95, 1.0]
- c0: [0.55, 0.57]
- gate_width: [0.03, 0.05]
- ctx_power: [0.5, 0.8]
- history_window: [100.0, 200.0]
- beta_v: [10.0, 40.0]
- epsilon_v: [0.0, 0.1]
- gamma: [3.0, 7.0]
- validities: validities

`rationale`: The previous candidate kept the correct adaptive context-mixture architecture but under-recruited validity in the high tie-prevalence tasks because its gate threshold was too high and the normalized gain divided by 1 - c0 was extremely attenuating, especially with ctx_power in [1,4]. This edit follows the critic's diagnosis without switching mechanisms: c0 is lowered and narrowed into a band above the near-0.5 tie prevalence in Experiments 1, 2, 5, and 6 but below the roughly 0.58-0.83 tie prevalence in Experiments 3 and 4. The normalized denominator is replaced by a small gate_width so the gate rises sharply above threshold instead of being crushed, while ctx_power is moved below 1 so the response becomes stronger rather than weaker. lambda_max is raised close to 1, and lambda_min is lowered near 0 so low tie-prevalence conditions stay close to pure tallying. The history window is enlarged to use essentially the full run, which reduces noisy context estimates and should lower the inflated between-subject variance seen previously in Experiment 3.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed adaptive context-gated mixture architecture, and it performs well in the low tie-prevalence conditions. Experiment 1 is close (-0.562 vs -0.614 real), Experiment 2 is nearly exact (0.0664 vs 0.0659 real), and Experiments 5 and 6 correctly stay near zero validity alignment (0.023 vs -0.047 and 0.047 vs 0.020). The failure is concentrated in the high tie-prevalence tasks. Experiment 3 real tie-validity alignment is 2.629 but the candidate gives 1.098, and Experiment 4 real is 4.202 while the candidate gives only 0.193. In both cases the model systematically under-recruits the validity route on tied tallies. The likely cause is gate calibration: c0 is set between 0.60 and 0.80, while the high-tie designs have tie prevalence around 0.63 to 0.67, so many simulated subjects sit near or below the gate threshold. Even subjects above threshold get only a tiny normalized gain g = (c - c0) / (1 - c0), and the current ctx_power range 1.0 to 4.0 further shrinks that already small value. With lambda close to zero, choices on exact ties remain near the 0.5 tally policy, producing weak tie-validity alignment. The between-subject variance in Experiment 3 is also too large (0.693 vs 0.285 real), consistent with some subjects turning the gate on and others leaving it off.
Rationale: Keep the adaptive context mixture and the history-based tie prevalence statistic, but recalibrate the gate so moderate-to-high tie prevalence actually recruits validity strongly. Concretely, lower c0 to sit above the low-tie tasks and below the high-tie tasks, for example c0 in [0.50, 0.58] rather than [0.60, 0.80], and make the gate respond more sharply or less attenuatingly above threshold. Options include replacing ctx_power [1.0, 4.0] with an exponent below 1.0, such as [0.3, 0.8], or introducing a steep logistic gate centered around roughly 0.58 to 0.62 with a moderate gain. This should let tie prevalence near 0.65 produce lambda close to lambda_max while keeping tie prevalence near 0.5 or below guarded. It may also help to sample c0 over a narrower range or increase lambda_max toward 1.0 so that once the validity route is recruited it is recruited consistently, reducing the inflated subject variance in Experiment 3. Do not switch mechanism families; the history-gated mixture is appropriate and only its context threshold and gain need adjustment.

**Outcome of this advice:** iter 1 candidate loss=0.3038 -> iter 2 candidate loss=0.1113 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate is a clear aggregate improvement, dropping loss from 0.3038 to 0.1113 and matching Experiment 2 almost exactly. Experiment 5 is also close. The adaptive mixture is now recruiting validity strongly in the high-tie Experiments 3 and 4, but it has overshot: Experiment 3 gives 2.9956 versus 2.6289 real, with subject variance 0.0010 versus 0.2854, and Experiment 4 gives 4.7317 versus 4.2015. More importantly, Experiment 6 is badly over-aligned: 0.1942 versus 0.0204 real. The key diagnosis is that Experiments 3 and 6 have the same exact-tie prevalence, about 0.75, yet real humans show strong validity alignment in Experiment 3 and essentially chance alignment in Experiment 6. Because the current gate uses only tie prevalence, it cannot separate these two tasks and switches validity on for both. The current validity route is also too deterministic once gated on because lambda_max is near 1 and beta_v is high. Experiment 1 is also somewhat too high at -0.4937 versus -0.6137, suggesting that when the tally route dominates, the model is not tally-deterministic enough or is leaking a little validity influence.
Rationale: Keep the prescribed adaptive context-gated mixture, but revise the context statistic so it reflects whether validity is actually diagnostic in the task, not just whether tallies tie. A practical variant is to compute the mean signed validity-weighted evidence V across past tie trials, or to combine tie prevalence with a function of mean absolute V, or to require both the tie-prevalence threshold and a diagnostic-consistency threshold. In Experiment 6 the high-validity features alternate sides across tie trials, so mean V is near zero and the gate should remain low; in Experiments 3 and 4 the validity advantage is consistent across tie trials, so the gate should rise. This stays inside the prescribed adaptive validity-gate family and only changes the context summary. Also moderate the now excessive validity strength and determinism: move lambda_max down from [0.95, 1.0] to roughly [0.80, 0.90], lower beta_v to something like [5, 15] or [5, 20], and allow a small epsilon_v in the range 0.03 to 0.10. This should bring Experiments 3 and 4 near their real means and restore the moderate between-subject variance that Experiment 3 requires. To repair Experiment 1, make the low-tie tally route more decisive by raising the lower bound of beta_tally and restricting epsilon_tally, for example beta_tally in [5, 20] and epsilon_tally in [0.0, 0.1]. Do not switch mechanism families; this is a calibration and context-statistic revision of the accepted candidate.

**Outcome of this advice:** iter 2 candidate loss=0.1113 -> iter 3 candidate loss=0.1467 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate is a mixed result and has been rejected by the accept gate because aggregate loss regressed from 0.1113 to 0.1467. It nails Experiment 6 (0.0204 vs 0.0204 real), is very close on Experiment 1 (-0.6102 vs -0.6137), and is acceptable on Experiments 2 and 3. However, it badly under-recruits validity in Experiment 4: 2.3309 versus 4.2015 real, a gap of -1.8706. This single failure is the main driver of the aggregate loss increase. The diagnosis is that the joint multiplicative consistency gate, combined with the lowered lambda_max, lower beta_v, and nonzero epsilon_v, attenuates the validity route too broadly. In Experiment 4 the tie-history consistency is clearly nonzero and diagnostic, yet the product of two sigmoids plus reduced validity strength leaves simulated choices only moderately validity-aligned, while humans show very strong tie-validity alignment. Experiment 5 is also slightly worse in sign and magnitude than the accepted iter 2 candidate. The theory is still in the prescribed adaptive context-gated mixture family, but its current calibration overcorrects Experiment 6 at the cost of Experiment 4.
Rationale: Verdict is regenerate; the next proposal will be built on the unchanged accepted iter 2 base, not on this rejected attempt. The previous push to add a continuous joint consistency product and globally reduce lambda_max/beta_v was rejected by the gate, so do not repeat that full attenuation. Instead, add a narrow consistency veto on top of iter 2: compute the same linear-weight tie-history consistency statistic, but make its effect near-binary rather than multiplicative. If consistency is above a small floor, roughly 0.02 to 0.05, the validity route should stay nearly as strong as iter 2; if consistency is near zero, lambda should fall to lambda_min. A sharp logistic or step function, not a product of two sigmoids with ctx_power greater than 1, should preserve Experiment 4 while still suppressing Experiment 6. Also restore intermediate validity strength rather than the overly weak current settings: lambda_max near 0.90 to 0.95, beta_v near 10 to 30, and epsilon_v near 0.00 to 0.05. Subject-level variability should come from sampling tie_thresh or the consistency threshold so that some subjects straddle the task boundary, not from globally making every subject uncertain. Keep the decisive tally route to preserve the Experiment 1 and Experiment 2 gains, but focus the next revision on restoring strong Experiment 4 tie-validity alignment without reopening Experiment 6.

**Outcome of this advice:** iter 3 candidate loss=0.1467 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
{
  "theory": "People adaptively decide whether unweighted tallying is diagnostic in the current task. They monitor two context signals from stimulus history: the prevalence of tally ties and the directional consistency of validity-weighted evidence on those tie trials. When ties are rare, choice remains essentially unweighted tallying. When ties are prevalent and validity evidence points in a consistent direction across tie trials, a validity-based tie-break policy is recruited. The consistency check is near-binary: once enough tie trials have been observed and the mean signed linear-validity evidence is sufficiently far from zero relative to its mean absolute size, the context gate admits validity; otherwise choice stays with tallying.",
  "predict": "import numpy as np\n\n\ndef predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Context-gated tallying expects a (2, n_features) stimulus; got shape %s.' % (stim.shape,))\n\n    validities = np.asarray(parameters['validities'], dtype=float)\n    n_features = stim.shape[1]\n    if validities.shape[0] != n_features:\n        raise ValueError('validities length %d != n_features %d.' % (validities.shape[0], n_features))\n\n    a = stim[0]\n    b = stim[1]\n    diff = a - b\n\n    wins_a = float(np.sum(a > b))\n    wins_b = float(np.sum(b > a))\n    tally_d = wins_a - wins_b\n\n    tau_tie = float(parameters['tau_tie'])\n\n    h_a = history.get('option_a_ratings', [])\n    h_b = history.get('option_b_ratings', [])\n    n_past = min(len(h_a), len(h_b))\n\n    window = int(round(float(parameters['history_window'])))\n    if window < 1:\n        window = 1\n    start = max(0, n_past - window)\n\n    lin_weights = validities - 0.5\n    tie_count = 0.0\n    cons_num = 0.0\n    cons_den = 0.0\n\n    for i in range(start, n_past):\n        pa = np.asarray(h_a[i], dtype=float)\n        pb = np.asarray(h_b[i], dtype=float)\n        past_wins = float(np.sum(pa > pb)) - float(np.sum(pb > pa))\n        if abs(past_wins) <= tau_tie:\n            tie_count += 1.0\n            v_lin = float(np.dot(lin_weights, pa - pb))\n            cons_num += v_lin\n            cons_den += abs(v_lin)\n\n    current_tie = 1.0 if abs(tally_d) <= tau_tie else 0.0\n    tie_count += current_tie\n    if current_tie:\n        v_lin_current = float(np.dot(lin_weights, diff))\n        cons_num += v_lin_current\n        cons_den += abs(v_lin_current)\n\n    total_considered = min(n_past, window) + 1.0\n    c = tie_count / total_considered if total_considered > 0.0 else 0.0\n\n    consistency = abs(cons_num) / max(cons_den, 1e-12)\n\n    c0 = float(parameters['c0'])\n    gate_width = float(parameters['gate_width'])\n    ctx_power = float(parameters['ctx_power'])\n    lambda_min = float(parameters['lambda_min'])\n    lambda_max = float(parameters['lambda_max'])\n    consistency_floor = float(parameters['consistency_floor'])\n    min_cons_ties = int(parameters['min_cons_ties'])\n\n    g = float(np.clip((c - c0) / max(gate_width, 1e-12), 0.0, 1.0))\n\n    if tie_count >= min_cons_ties and consistency >= consistency_floor:\n        veto = 1.0\n    else:\n        veto = 0.0\n\n    lamb = lambda_min + (lambda_max - lambda_min) * (g ** ctx_power) * veto\n\n    def stable_softmax(scores, inv_temp):\n        scores = np.asarray(scores, dtype=float)\n        z = inv_temp * (scores - np.max(scores))\n        z = np.clip(z, -30.0, 30.0)\n        e = np.exp(z)\n        p = e / np.sum(e)\n        return p\n\n    beta_t = float(parameters['beta_tally'])\n    eps_t = float(parameters['epsilon_tally'])\n    p_tally = stable_softmax(np.array([wins_a, wins_b]), beta_t)\n    p_tally = (1.0 - eps_t) * p_tally + eps_t * np.array([0.5, 0.5])\n\n    gamma = float(parameters['gamma'])\n    weights = np.maximum(validities - 0.5, 0.0) ** gamma\n    v = float(np.dot(weights, diff))\n\n    beta_v = float(parameters['beta_v'])\n    eps_v = float(parameters['epsilon_v'])\n    p_valid = stable_softmax(np.array([v, -v]), beta_v)\n    p_valid = (1.0 - eps_v) * p_valid + eps_v * np.array([0.5, 0.5])\n\n    probs = lamb * p_valid + (1.0 - lamb) * p_tally\n    probs = np.clip(probs, 0.0, None)\n    total = probs.sum()\n    if total <= 0.0:\n        probs = np.array([0.5, 0.5])\n    else:\n        probs = probs / total\n\n    return probs\n",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    p = np.asarray(probs, dtype=np.float64)\n    p = np.clip(p, 0.0, None)\n    total = p.sum()\n    if total <= 0.0:\n        p = np.ones_like(p) / len(p)\n    else:\n        p = p / total\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "beta_tally": "[0.1, 20.0]",
    "epsilon_tally": "[0.0, 0.5]",
    "tau_tie": "[0.0, 0.5]",
    "lambda_min": "[0.0, 0.005]",
    "lambda_max": "[0.90, 0.95]",
    "c0": "[0.55, 0.57]",
    "gate_width": "[0.03, 0.05]",
    "ctx_power": "[0.5, 0.8]",
    "history_window": "[100.0, 200.0]",
    "beta_v": "[10.0, 30.0]",
    "epsilon_v": "[0.0, 0.05]",
    "gamma": "[3.0, 7.0]",
    "consistency_floor": "[0.02, 0.05]",
    "min_cons_ties": "{3, 4}",
    "validities": "validities"
  },
  "rationale": "This is a minimal-diff edit of the accepted context-gated tallying model. The previous version used only tie prevalence as the context signal, which let noisy tie-prevalence estimates in Experiments 1 and 6 occasionally switch on validity even though the validity evidence on tie trials was not directionally diagnostic. The edit adds a near-binary consistency veto: it computes the mean signed linear validity-weighted evidence on past tie trials, normalized by the mean absolute evidence. In Experiments 1 and 6 that statistic is essentially zero because the validity advantages are balanced or cancel across the stimulus set, so the veto pins lambda to lambda_min and prevents the false validity recruitment. In Experiments 3 and 4 the statistic is clearly nonzero, so the veto stays open and the tie-prevalence gate recruits validity strongly. A small minimum tie count prevents a single early tie trial from being treated as diagnostic. Validity strength is restored to an intermediate regime, lambda_max in [0.90, 0.95], beta_v in [10, 30], and epsilon_v in [0, 0.05], keeping Experiment 4 strongly validity-aligned while avoiding pi_4-style excessive determinism. The tally parameters are left unchanged so the low tie-prevalence conditions retain the accepted model's near-tallying behavior."
}
```

## Usage

```json
{
  "input_tokens": 19035,
  "output_tokens": 46211,
  "reasoning_tokens": 44208,
  "provider": "Baidu"
}
```
