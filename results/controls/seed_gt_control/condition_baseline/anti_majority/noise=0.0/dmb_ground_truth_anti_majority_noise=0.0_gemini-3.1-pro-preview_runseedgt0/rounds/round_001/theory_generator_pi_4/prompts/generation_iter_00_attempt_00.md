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
- The recommendation below acts on THEORY 2 (= `pi_3`).

Propose a brand-new theory based on a standard Take-The-Best (TTB) or a Validity-Weighted Additive (WADD) model. The data shows that subjects often prefer options with fewer overall positive features if those features have higher validities (which drives the low Tallying match rates in Exp 3 and 4). A standard TTB or WADD model with a softmax response and a lapse rate is a theoretically grounded, biologically plausible mechanism that will likely serve as a much stronger and more realistic competitor to Theory 1 than the current Anti-Tallying model.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 1.000)

**Description**
People are modeled as running three classic heuristics in parallel — Take The Best (one-reason, top-validity cue), Tallying (count of feature-wise wins), and Weighted Additive (validity-weighted sum) — reading off the single option each heuristic deterministically prefers, and then leaning toward whichever option the majority of the three did NOT prefer. With three binary voters at least two always agree, so a majority option always exists; the model targets its complement. Response noise enters only at the final, ensemble level: a softmax over the binary "flipped-majority" score with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options. The individual heuristics contribute only their hard preference (their argmax) to the vote — they carry no internal noise — so beta and epsilon describe noise in the contrarian response itself, not in the constituent heuristics.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Anti-majority-vote ensemble. Each of Tallying / TTB / WADD
    # contributes a single hard preference (the argmax of its
    # deterministic score — no per-model softmax or lapse). We take the
    # majority vote across the three binary votes and flip it: when at
    # least two models agree on an option, we target the other one.
    # Choice noise (softmax over beta, lapse over epsilon) is applied
    # ONCE here, to the flipped-majority decision.

    def _pref_tallying(stimulus):
        # Tallying (Dawes 1979; Gigerenzer & Goldstein 1999): count
        # strict feature-wise wins; prefer the option with more wins.
        # Ties (equal win counts) fall to option A via argmax.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        a, b = stim[0], stim[1]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        return int(np.argmax(np.array([a_wins, b_wins])))

    def _pref_ttb(parameters, stimulus):
        # Take The Best (Gigerenzer & Goldstein 1996): consult cues in
        # descending validity; the first discriminating cue decides.
        # If no cue discriminates, the score is uniform and argmax
        # falls to option A.
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
            scores = np.ones(2) / 2.0
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        return int(np.argmax(scores))

    def _pref_wadd(parameters, stimulus):
        # Weighted Additive (Payne, Bettman & Johnson 1993): prefer the
        # option with the higher validity-weighted sum. Ties fall to
        # option A via argmax.
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim != 2 or stim.shape[0] != 2:
            raise ValueError(
                f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
            )
        n_features = stim.shape[1]
        validities = np.asarray(parameters["validities"], dtype=float)
        w = np.asarray(parameters["weights"], dtype=float)
        if w.shape[0] != n_features:
            raise ValueError(
                f"weights length {w.shape[0]} != n_features {n_features}."
            )
        scores = stim @ (validities * w)
        return int(np.argmax(scores))

    # Each model's hard preference (no noise inside the models).
    c_tallying = _pref_tallying(stimulus)
    c_ttb = _pref_ttb(parameters, stimulus)
    c_wadd = _pref_wadd(parameters, stimulus)

    # Majority vote across the three binary votes, then flip it.
    votes = [c_tallying, c_ttb, c_wadd]
    majority = 0 if votes.count(0) >= 2 else 1
    flipped = 1 - majority

    # Binary score for the flipped-majority choice (winner = 1, loser = 0),
    # then ONE softmax + lapse over the two options.
    scores = np.array([1.0, 0.0]) if flipped == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # flipped score this collapses to sigmoid(beta) for the targeted
    # option (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic flip).
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
- weights: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2675 (var=0.0463) vs this=0.2288 (var=0.0317)
- Experiment 2: real=0.2350 (var=0.0495) vs this=0.2308 (var=0.0618)
- Experiment 3: real=0.5967 (var=0.0436) vs this=0.6208 (var=0.0656)
- Experiment 4: real=0.3583 (var=0.1157) vs this=0.4473 (var=0.1230)


---

### `pi_3` (overall score: 0.074)

**Description**
Anti-Tallying: Subjects systematically misinterpret the expert ratings, treating a rating of 0 as superior to 1. They tally the number of features where an option 'wins' under this reversed interpretation (i.e., has a 0 while the other has a 1) and prefer the option with more reversed wins. Choice noise is modeled via a softmax over these anti-tallies and a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Anti-Tallying expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    # Anti-Tallying: subjects treat 0 as better than 1.
    # Thus, option A wins a feature if a < b (i.e., a=0, b=1).
    a_anti_wins = float(np.sum(a < b))
    b_anti_wins = float(np.sum(b < a))
    
    scores = np.array([a_anti_wins, b_anti_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the anti-tallies with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2675 (var=0.0463) vs this=0.1494 (var=0.0090)
- Experiment 2: real=0.2350 (var=0.0495) vs this=0.1108 (var=0.0075)
- Experiment 3: real=0.5967 (var=0.0436) vs this=0.1338 (var=0.0093)
- Experiment 4: real=0.3583 (var=0.1157) vs this=0.8573 (var=0.0090)


---

### `pi_2` (overall score: 0.000)

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
- Experiment 1: real=0.2675 (var=0.0463) vs this=0.8573 (var=0.0075)
- Experiment 2: real=0.2350 (var=0.0495) vs this=0.8492 (var=0.0115)
- Experiment 3: real=0.5967 (var=0.0436) vs this=0.8596 (var=0.0086)
- Experiment 4: real=0.3583 (var=0.1157) vs this=0.1350 (var=0.0076)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins and row['response'] == 0:
            matches += 1
        elif b_wins > a_wins and row['response'] == 1:
            matches += 1
        elif a_wins == b_wins:
            matches += 0.5
    return float(matches / len(data))
```

**Observed (real) value:** 0.2675 (var=0.0463)
**Other theories' values on this metric (for reference):**
- pi_1: 0.2288 (var=0.0317)
- pi_2: 0.8573 (var=0.0075)
- pi_3: 0.1494 (var=0.0090)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_sums = data['option_a_ratings'].apply(np.sum)
    b_sums = data['option_b_ratings'].apply(np.sum)
    
    mask = a_sums != b_sums
    if mask.sum() == 0:
        return 0.5
        
    responses = data.loc[mask, 'response']
    a_sums_filtered = a_sums[mask]
    b_sums_filtered = b_sums[mask]
    
    tallying_choices = (b_sums_filtered > a_sums_filtered).astype(int)
    
    return float((responses == tallying_choices).mean())
```

**Observed (real) value:** 0.2350 (var=0.0495)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8492 (var=0.0115)
- pi_1: 0.2308 (var=0.0618)
- pi_3: 0.1108 (var=0.0075)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ones = data['option_a_ratings'].apply(sum)
    b_ones = data['option_b_ratings'].apply(sum)
    
    chosen_more_ones = np.where(
        data['response'] == 0,
        a_ones > b_ones,
        b_ones > a_ones
    )
    
    return float(np.mean(chosen_more_ones))
```

**Observed (real) value:** 0.5967 (var=0.0436)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6208 (var=0.0656)
- pi_3: 0.1338 (var=0.0093)
- pi_2: 0.8596 (var=0.0086)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Anti-Tallying prefers the option with more 0s.
    # In this design, option A has either three 1s and two 0s (A[0]=0) 
    # or three 0s and two 1s (A[0]=1).
    # If A[0] == 0, A has two 0s and B has three 0s, so Anti-Tallying prefers B (1).
    # If A[0] == 1, A has three 0s and B has two 0s, so Anti-Tallying prefers A (0).
    # Thus, Anti-Tallying always predicts (1 - A[0]).
    # Conversely, Anti-Majority always predicts A[0].
    # We measure the proportion of choices matching Anti-Tallying.
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    anti_tallying_pred = 1 - a_first
    return float((data['response'] == anti_tallying_pred).mean())
```

**Observed (real) value:** 0.3583 (var=0.1157)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8573 (var=0.0090)
- pi_1: 0.4473 (var=0.1230)
- pi_2: 0.1350 (var=0.0076)


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
  "theory": "People make decisions by weighing the features of each option according to their validities. They compute a Validity-Weighted Additive (WADD) score for each option, summing the validities of the features that are present (rating of 1). This allows a single highly valid feature to outweigh multiple features of lower validity, leading to choices that often contradict simple Tallying. The final choice is made probabilistically using a softmax over these WADD scores, with a base lapse rate to account for random errors or lapses in attention.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"WADD expects a (2, n_features) stimulus.\")\n\n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Calculate Validity-Weighted Additive (WADD) scores\n    score_a = np.sum(a * validities)\n    score_b = np.sum(b * validities)\n    \n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax over scores with numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's diagnosis, the Anti-Tallying mechanism is discarded in favor of a theoretically grounded Validity-Weighted Additive (WADD) model. The data indicates that subjects often prefer options with fewer overall positive features if those features have higher validities, which explains the low match rates for Tallying in Experiments 1 and 2. By directly multiplying the feature vectors by their provided validities, this WADD model naturally captures the trade-off between feature quantity and feature quality without resorting to an implausible interpretation of the ratings. The model incorporates a softmax response rule (controlled by beta) and a uniform lapse rate (epsilon) to provide a realistic, biologically plausible probabilistic choice mechanism."
}
```
```

## Usage

```json
{
  "prompt_token_count": 8012,
  "candidates_token_count": 699,
  "total_token_count": 10228
}
```
