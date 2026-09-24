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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Replace pi_3 with a stochastic bounded-sampling evidence-accumulation theory. The new theory should not use a hard stop at the first displayed discriminator. Instead, discriminating cues should be sampled stochastically, with sampling probability determined by both serial position and stated validity. Serial position should produce only a moderate primacy gradient on short lists and a weak or partially flattened gradient on long lists, rather than the extreme exponential gradient that pi_4 currently uses. Stated validity should remain influential on all list lengths, but with a compressive rather than all-or-none effect, so a 0.90 validity cue can compete with opposing tally evidence and produce near-zero direction effects as in Experiment 2. A bounded capacity function such as f(n) = 1 / (1 + kappa * (n - 5)) or f(n) = (5 / n)^gamma with small gamma around 0.3 to 1.0 should be used, so longer lists flatten attention only partially, never to a pure equal-weight tally. Decision noise should be nontrivial: inverse temperature around 1 to 3 and lapse epsilon around 0.1 to 0.2, matching the modest first-cue consistency of 0.62 in Experiment 5 and the moderate validity effects in Experiments 3 and 4. The new theory also needs to allow subject-level heterogeneity in cue weights, because Experiment 2 shows high between-subject variance (var=0.197). This theory differs from pi_3 by abandoning deterministic first-discriminator commitment, and differs from pi_4 by making cue weighting stochastic and bounded, with validity effects that do not disappear on long lists.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.486)

**Description**
People scan the displayed features from left to right and accumulate a signed preference signal, without ever committing to a hard lexicographic stop. Each discriminating feature contributes an increment whose weight is a continuous function of serial position, stated validity, and list length. For short lists the position weight is a steep exponential primacy gradient, so the first discriminator dominates probabilistically rather than absolutely. As list length exceeds working-memory capacity the gradient collapses toward a flat weighting function, making the rule behave like a tallying heuristic on long lists. The accumulated score is passed through a softmax with inverse temperature beta, and an independent lapse probability permits occasional intention-independent errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # Accept either a dict with option ratings or a (2, n_features) array.
    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Capacity-sensitive evidence accumulation expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Stated validities weakly modulate feature weight.
    if 'validities' in parameters and parameters['validities'] is not None:
        validities = np.asarray(parameters['validities'], dtype=float)
        if validities.shape[0] != n_features:
            validities = np.ones(n_features, dtype=float)
    else:
        validities = np.ones(n_features, dtype=float)

    lambda_5 = float(parameters['lambda_5'])
    capacity_power = float(parameters['capacity_power'])
    validity_power = float(parameters['validity_power'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Capacity-sensitive flattening. For n_features = 5 this factor is 1,
    # preserving a strong exponential primacy gradient. For longer lists it
    # shrinks rapidly, approximating equal tallying weights.
    capacity_factor = (5.0 / n_features) ** capacity_power

    lam = lambda_5 * capacity_factor
    nu = validity_power * capacity_factor

    positions = np.arange(n_features, dtype=float)
    weights = np.exp(-lam * positions) * (validities ** nu)

    diff = a - b
    score_a = float(np.sum(weights[diff > 0]))
    score_b = float(np.sum(weights[diff < 0]))
    relative_score = score_a - score_b

    # Stable softmax over the accumulated relative evidence.
    logits = np.array([beta * relative_score, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - epsilon) * core_probs + epsilon * np.full(2, 0.5, dtype=float)
    return probs


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- lambda_5: [0.692, 0.694]
- capacity_power: [14.0, 18.0]
- validity_power: [0.049, 0.051]
- beta: [4.51, 4.53]
- epsilon: [0.095, 0.105]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4400 (var=0.1503) vs this=-0.4933 (var=0.0313)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=-0.3458 (var=0.0038)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.6208 (var=0.0020)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=0.0280 (var=0.0391)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=-0.2019 (var=0.0042)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=0.6033 (var=0.0124)


---

### `pi_2` (overall score: 0.392)

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
- Experiment 1: real=-0.4400 (var=0.1503) vs this=0.0383 (var=0.0359)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=-0.0029 (var=0.0086)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.2706 (var=0.0057)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=-0.0147 (var=0.0280)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=0.0058 (var=0.0086)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=0.4608 (var=0.0209)


---

### `pi_1` (overall score: 0.194)

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
- Experiment 1: real=-0.4400 (var=0.1503) vs this=0.6650 (var=0.0546)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=0.6875 (var=0.0362)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.6206 (var=0.0018)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=0.2253 (var=0.0167)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=0.0092 (var=0.0061)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=-0.6992 (var=0.0453)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=int)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=int)
    r = data['response'].to_numpy(dtype=float)

    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    tie = a_wins == b_wins

    # Among tally ties, the high-validity cue at index 2 discriminates.
    cue_favors_b = tie & (b[:, 2] > a[:, 2])
    cue_favors_a = tie & (a[:, 2] > b[:, 2])

    if not np.any(cue_favors_b) or not np.any(cue_favors_a):
        return 0.0

    p_b_given_b_cue = float(r[cue_favors_b].mean())
    p_b_given_a_cue = float(r[cue_favors_a].mean())

    return p_b_given_b_cue - p_b_given_a_cue

```

**Observed (real) value:** -0.4400 (var=0.1503)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6650 (var=0.0546)
- pi_2: 0.0383 (var=0.0359)
- pi_3: -0.6500 (var=0.0231)
- pi_4: -0.4933 (var=0.0313)

### Experiment 2
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
import numpy as np

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    validities = np.array([0.78, 0.55, 0.91, 0.50, 0.65], dtype=float)
    cue_order = np.argsort(-validities, kind='stable')

    a = np.vstack(data['option_a_ratings'].to_numpy()).astype(float)
    b = np.vstack(data['option_b_ratings'].to_numpy()).astype(float)
    diff = a - b

    ordered_diff = diff[:, cue_order]
    nonzeros = ordered_diff != 0
    has_discriminator = nonzeros.any(axis=1)
    first_pos = nonzeros.argmax(axis=1)
    first_values = ordered_diff[np.arange(ordered_diff.shape[0]), first_pos]

    choose_a = (data['response'].to_numpy() == 0).astype(float)
    first_cue_favors_a = has_discriminator & (first_values > 0)
    first_cue_favors_b = has_discriminator & (first_values < 0)

    if not (first_cue_favors_a.any() and first_cue_favors_b.any()):
        return 0.0

    p_a_given_cue_a = choose_a[first_cue_favors_a].mean()
    p_a_given_cue_b = choose_a[first_cue_favors_b].mean()
    return float(p_a_given_cue_a - p_a_given_cue_b)
```

**Observed (real) value:** -0.3750 (var=0.0321)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0029 (var=0.0086)
- pi_1: 0.6875 (var=0.0362)
- pi_3: -0.3296 (var=0.0048)
- pi_4: -0.3458 (var=0.0038)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    matches = []
    for a_ratings, b_ratings, response in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        winner = None
        for a_val, b_val in zip(a_ratings, b_ratings):
            if a_val > b_val:
                winner = 0
                break
            if a_val < b_val:
                winner = 1
                break
        if winner is not None:
            matches.append(1.0 if response == winner else 0.0)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.6246 (var=0.0093)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8567 (var=0.0014)
- pi_2: 0.2706 (var=0.0057)
- pi_1: 0.6206 (var=0.0018)
- pi_4: 0.6208 (var=0.0020)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_vals = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b_vals = np.asarray(data['option_b_ratings'].tolist(), dtype=float)

    a_wins = np.sum(a_vals > b_vals, axis=1)
    b_wins = np.sum(b_vals > a_vals, axis=1)
    tally_tie = a_wins == b_wins

    differ = a_vals != b_vals
    has_disc = differ.any(axis=1)
    first_disc = np.argmax(differ.astype(int), axis=1)

    row_idx = np.arange(a_vals.shape[0])
    first_a_wins = a_vals[row_idx, first_disc] > b_vals[row_idx, first_disc]

    response = data['response'].to_numpy(dtype=float)

    selected = tally_tie & has_disc
    if not np.any(selected):
        return 0.0

    disc_positions = np.unique(first_disc[selected])
    per_pos_diff = []
    for pos in disc_positions:
        in_pos = selected & (first_disc == pos)
        b_first = in_pos & ~first_a_wins
        a_first = in_pos & first_a_wins
        if np.sum(b_first) == 0 or np.sum(a_first) == 0:
            continue
        per_pos_diff.append(
            float(np.mean(response[b_first]) - np.mean(response[a_first]))
        )

    if not per_pos_diff:
        return 0.0
    return float(np.mean(per_pos_diff))
```

**Observed (real) value:** -0.1173 (var=0.0257)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0147 (var=0.0280)
- pi_3: 0.7160 (var=0.0149)
- pi_1: 0.2253 (var=0.0167)
- pi_4: 0.0280 (var=0.0391)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    low = []
    high = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        diff = a - b
        nz = np.flatnonzero(diff)
        if nz.size == 0:
            continue
        pos = int(nz[0])
        favored = 0 if diff[pos] > 0 else 1
        consistent = 1.0 if int(row.response) == favored else 0.0
        if pos == 0:
            low.append(consistent)
        elif pos in (1, 2, 3):
            high.append(consistent)
    if not low or not high:
        return float('nan')
    return float(np.mean(high) - np.mean(low))
```

**Observed (real) value:** -0.0222 (var=0.0067)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1878 (var=0.0101)
- pi_4: -0.2019 (var=0.0042)
- pi_1: 0.0092 (var=0.0061)
- pi_2: 0.0058 (var=0.0086)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hi_a = []
    hi_b = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'])
        b = np.asarray(row['option_b_ratings'])
        diff = a != b
        if not np.any(diff):
            continue
        first_diff = int(np.argmax(diff))
        if first_diff != 1:
            continue
        if a[1] > b[1]:
            hi_a.append(float(row['response']))
        else:
            hi_b.append(float(row['response']))
    if not hi_a or not hi_b:
        return 0.0
    return float(np.mean(hi_a) - np.mean(hi_b))
```

**Observed (real) value:** -0.0633 (var=0.1971)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6033 (var=0.0124)
- pi_3: -0.8958 (var=0.0049)
- pi_1: -0.6992 (var=0.0453)
- pi_2: 0.4608 (var=0.0209)


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
{
  "theory": "People do not commit to the first discriminating cue. Instead, they stochastically sample a small bounded set of discriminating cues, with sampling probability governed jointly by a moderate serial-position primacy gradient and a compressive function of stated cue validity. Capacity limits partially flatten attention on longer lists through a bounded capacity factor f(n)=(5/n)^gamma, so attention becomes more equal but never a pure equal-weight tally. Sampled cues accumulate signed evidence, with a special but non-deterministic boost for the first displayed discriminator: if that cue is sampled it receives extra weight, but if it is not sampled the decision still proceeds using other sampled cues. The accumulated evidence is passed through a softmax with modest noise and an independent lapse process. Subject-level variation in evidence scaling and response bias creates between-subject heterogeneity.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # Accept either the dict format or a (2, n_features) array.\n    if isinstance(state, dict):\n        a = np.asarray(state['option_a_ratings'], dtype=float)\n        b = np.asarray(state['option_b_ratings'], dtype=float)\n        stim = np.vstack([a, b])\n    else:\n        stim = np.asarray(state, dtype=float)\n\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f'Bounded-sampling accumulator expects a (2, n_features) stimulus; got shape {stim.shape}.'\n        )\n\n    a = stim[0]\n    b = stim[1]\n    n_features = stim.shape[1]\n\n    validities = np.asarray(parameters.get('validities'), dtype=float)\n    if validities is None or validities.shape[0] != n_features:\n        validities = np.full(n_features, 0.75, dtype=float)\n\n    capacity_gamma = float(parameters['capacity_gamma'])\n    position_decay = float(parameters['position_decay'])\n    att_intercept = float(parameters['att_validity_intercept'])\n    att_slope = float(parameters['att_validity_slope'])\n    evidence_intercept = float(parameters['evidence_intercept'])\n    evidence_slope = float(parameters['evidence_slope'])\n    first_cue_bonus = float(parameters['first_cue_bonus'])\n    subject_scale = float(parameters['subject_scale'])\n    subject_bias = float(parameters['subject_bias'])\n    beta = float(parameters['beta'])\n    epsilon = float(parameters['epsilon'])\n    sample_size = int(parameters['sample_size'])\n\n    # Bounded capacity: partial, compressive flattening for long lists.\n    capacity = (5.0 / n_features) ** capacity_gamma\n\n    positions = np.arange(n_features, dtype=float)\n    att_validity = np.clip(\n        att_intercept + att_slope * (validities - 0.5), 0.05, 1.0\n    )\n\n    # Sampling probabilities combine a moderate position gradient with\n    # stated validity; the position gradient itself weakens on long lists.\n    sampling_weights = (\n        capacity\n        * np.exp(-position_decay * positions * capacity)\n        * att_validity\n    )\n\n    diff = a - b\n    discrim = diff != 0\n    signs = np.where(discrim, np.sign(diff), 0.0)\n    discrim_idx = np.flatnonzero(discrim)\n\n    if discrim_idx.size == 0:\n        return np.full(2, 0.5, dtype=float)\n\n    # Stochastic bounded sampling: choose up to sample_size discriminating\n    # cues without replacement. This is a soft capacity limit, not a\n    # deterministic stop at the first discriminator.\n    if discrim_idx.size <= sample_size:\n        sampled = discrim_idx.copy()\n    else:\n        w = sampling_weights[discrim_idx].astype(float)\n        if w.sum() <= 0.0:\n            w = np.ones(discrim_idx.size, dtype=float)\n        w = w / w.sum()\n        sampled = np.random.choice(\n            discrim_idx, size=sample_size, replace=False, p=w\n        )\n\n    first_disc = int(np.argmax(discrim))\n\n    evidence = subject_scale * np.clip(\n        evidence_intercept + evidence_slope * (validities - 0.5), 0.1, 1.2\n    )\n\n    score = 0.0\n    for j in sampled:\n        j = int(j)\n        multiplier = first_cue_bonus if j == first_disc else 1.0\n        score += signs[j] * evidence[j] * multiplier\n\n    logits = np.array([beta * score + subject_bias, 0.0], dtype=float)\n    logits = logits - np.max(logits)\n    exponentials = np.exp(logits)\n    core_probs = exponentials / np.sum(exponentials)\n\n    probs = (1.0 - epsilon) * core_probs + epsilon * np.full(2, 0.5, dtype=float)\n    return probs\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=float)\n    probs = probs / np.sum(probs)\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "sample_size": "{2}",
    "capacity_gamma": "[0.40, 0.60]",
    "position_decay": "[0.55, 0.75]",
    "att_validity_intercept": "[0.45, 0.55]",
    "att_validity_slope": "[0.40, 0.60]",
    "evidence_intercept": "[0.52, 0.60]",
    "evidence_slope": "[0.40, 0.55]",
    "first_cue_bonus": "[1.60, 2.10]",
    "subject_scale": "[0.75, 1.25]",
    "subject_bias": "[-0.30, 0.30]",
    "beta": "[1.80, 3.00]",
    "epsilon": "[0.10, 0.20]",
    "validities": "validities"
  },
  "rationale": "This theory replaces deterministic first-discriminator commitment with a real stochastic bounded sampler. A fixed small sample size avoids the hard lexicographic stop of pi_3, while the first displayed discriminator is merely given a moderate weight bonus when sampled, not an absolute veto. Serial position enters through a moderate exponential gradient that is additionally scaled by the bounded capacity factor (5/n)^gamma, so long lists flatten partially rather than collapsing to equal tallying or to a pure first-cue rule. Stated validity enters both sampling probability and evidence magnitude through compressive linear transformations, so a 0.90 cue is influential but can still compete with opposing tally evidence. The beta and epsilon ranges keep decision noise nontrivial, producing the modest first-cue consistency and near-zero single-position effects seen in Experiments 3 and 5. Subject-level variability in subject_scale and subject_bias, together with the sampled parameter ranges, supplies the cue-weight and response heterogeneity needed for the high between-subject variance in Experiment 6."
}
```

## Usage

```json
{
  "input_tokens": 11313,
  "output_tokens": 54385,
  "reasoning_tokens": 52677,
  "provider": "Baidu"
}
```
