# generation_iter_01_attempt_01

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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Discard unweighted Tallying and propose a VALIDITY-WEIGHTED EVIDENCE INTEGRATION theory that nests both extremes and can absorb the residuals pi_1 leaves behind. Sketch: on each trial the decision maker forms a weighted sum of the signed feature-wise comparisons, d = sum_j w_j * sign(a_j - b_j), where the weight of cue j is a monotone but strongly convex function of its stated validity, e.g. w_j proportional to (v_j - 0.5)^gamma or to logit(v_j)^gamma, normalized to sum to 1. The exponent gamma is a free subject-level parameter: gamma = 0 recovers Tallying (equal weights), moderate gamma recovers a WADD/log-odds integrator, and large gamma makes the top discriminating cue dominate so behaviour becomes near-lexicographic and mimics Take-The-Best. Choice is a softmax over d with inverse temperature beta plus a lapse epsilon. Parameter ranges should be centred so that most sampled subjects are steeply weighted (gamma large enough that the top cue usually wins conflicts) but a nontrivial minority integrate more evenly \u2014 this reproduces the observed intermediate values (~0.74 on the Experiment 1 top-cue metric, ~0.19 on the Experiment 2 tally metric) AND the inflated between-subject variance seen in Experiment 1, both of which pure TTB under-predicts. Key qualitative differences from pi_1 to be exploited by future experiments: (i) confidence/consistency should be graded in the weighted margin rather than constant, so trials decided only by a 0.55 cue should be noisier than unanimous trials; (ii) on tally-tie trials the model still follows the higher-validity cue (unlike Tallying), so it survives Experiment 2 family 3; (iii) it predicts occasional reversals when many mid-validity cues oppose the top cue with a large summed weight, which is precisely where the observed data deviate from strict TTB.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 1.000)

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
- Experiment 1: real=0.7367 (var=0.0369) vs this=0.8425 (var=0.0110)
- Experiment 2: real=0.1889 (var=0.0124) vs this=0.1217 (var=0.0145)


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
- Experiment 1: real=0.7367 (var=0.0369) vs this=0.1758 (var=0.0098)
- Experiment 2: real=0.1889 (var=0.0124) vs this=0.8583 (var=0.0090)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0467 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0467 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    a_all = data['option_a_ratings'].tolist()
    b_all = data['option_b_ratings'].tolist()
    resp = data['response'].tolist()

    hits = 0
    n = 0
    for a, b, r in zip(a_all, b_all, resp):
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # TTB winner: features are ordered by descending validity in this design,
        # so scan left to right for the first discriminating cue.
        ttb = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb is None:
            continue
        # Tally winner: count strict feature-wise wins.
        aw = float(np.sum(a > b))
        bw = float(np.sum(b > a))
        if aw == bw:
            continue
        tal = 0 if aw > bw else 1
        # Keep only trials where the two heuristics point to OPPOSITE options.
        if tal == ttb:
            continue
        n += 1
        try:
            rr = int(r)
        except Exception:
            continue
        if rr == ttb:
            hits += 1

    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.7367 (var=0.0369)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7688 (var=0.0135) (Δ vs real +0.0321)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8425 (var=0.0110)
- pi_2: 0.1758 (var=0.0098)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.60, 0.93, 0.55, 0.80, 0.68, 0.57], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind="stable")
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        n = a.shape[0]
        val = validities if validities.shape[0] == n else np.linspace(0.9, 0.55, n)

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_win = 0
        elif margin < 0:
            tally_win = 1
        else:
            continue  # tally undecided: excluded

        tw = ttb_winner(a, b, val)
        if tw < 0 or tw == tally_win:
            continue  # only trials where the two rules point opposite ways
        if abs(margin) < 3:
            continue  # keep only large-margin conflicts (strong tally signal)

        hits.append(1.0 if int(row["response"]) == tally_win else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1889 (var=0.0124)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2222 (var=0.0156) (Δ vs real +0.0333)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8583 (var=0.0090)
- pi_1: 0.1217 (var=0.0145)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Convex Validity-Weighted Evidence Integration (CVW).** On every trial the decision maker compares the two products cue by cue and forms a single scalar evidence signal that is a *weighted vote* over the signed feature-wise comparisons: d = sum_j w_j * sign(a_j - b_j). Cues that tie contribute nothing. The weight of a cue is a monotone but *strongly convex* function of its stated validity: w_j is proportional to ((v_j - 0.5) / (v_max - 0.5))^gamma, normalised to sum to one. The subjective exponent gamma expresses how sharply a person amplifies small differences in stated diagnosticity: gamma = 0 recovers unweighted Tallying (every expert counts the same), gamma near 1 recovers a linear/WADD-like integrator, and large gamma makes the single most valid discriminating cue swamp the summed weight of all remaining cues, so behaviour becomes effectively lexicographic and mimics Take-The-Best. Crucially, the model is *compensatory in principle but non-compensatory in practice*: the top cue usually wins, yet when many mid-validity cues line up against it their summed weight can approach or exceed the top cue's weight, producing genuine, systematic reversals of TTB rather than mere lapses. Because choice is a softmax on the *weighted margin* d (inverse temperature beta) plus an independent lapse epsilon, confidence is graded: unanimous or top-two-vs-bottom-three trials are near deterministic, whereas trials decided by a lone modest-validity cue against several opposing cues are close to a coin flip. Subject-level variation in gamma (how steeply people discount less valid experts) is the main source of between-subject heterogeneity, and it bites hardest exactly on the conflict trials where the top cue's weight and the opposing coalition's weight are nearly matched. Two structural predictions distinguish CVW from one-reason heuristics: on tally-tie trials the model still follows the higher-validity cue, and the same subject can look TTB-like in an environment with one dominant expert (e.g. validity .93 against a field of .55-.80) while looking noticeably more compensatory in an environment where validities are closely spaced.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---- unpack the trial stimulus -------------------------------------
    if isinstance(state, dict):
        a = np.asarray(state.get("option_a_ratings"), dtype=float).ravel()
        b = np.asarray(state.get("option_b_ratings"), dtype=float).ravel()
    else:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        a = np.asarray(arr[0], dtype=float).ravel()
        b = np.asarray(arr[1], dtype=float).ravel()

    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.ones(2) / 2.0
    a = a[:n]
    b = b[:n]

    # ---- validities -----------------------------------------------------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n or not np.all(np.isfinite(val)):
        # fallback: descending validities consistent with the usual design
        val = np.linspace(0.9, 0.55, n)

    gamma = float(parameters.get("gamma", 2.0))
    beta = float(parameters.get("beta", 10.0))
    epsilon = float(parameters.get("epsilon", 0.05))
    epsilon = min(max(epsilon, 0.0), 1.0)

    # ---- convex validity weighting --------------------------------------
    u = np.clip(val - 0.5, 1e-9, None)          # informativeness above chance
    umax = float(np.max(u))
    if not np.isfinite(umax) or umax <= 0.0:
        w = np.ones(n) / float(n)
    else:
        r = u / umax                             # scale-free, top cue = 1
        with np.errstate(over="ignore", under="ignore"):
            w = np.power(r, gamma)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        s = float(np.sum(w))
        w = (w / s) if s > 0 else (np.ones(n) / float(n))

    # ---- weighted vote over signed feature comparisons -------------------
    sgn = np.sign(a - b)                         # +1 favours A, -1 favours B
    score_a = float(np.sum(w * (sgn > 0)))
    score_b = float(np.sum(w * (sgn < 0)))

    # ---- softmax on the weighted margin (graded confidence) --------------
    z = beta * np.array([score_a, score_b], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / e.sum()

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- gamma: [1.6, 2.6]
- beta: [6.0, 15.0]
- epsilon: [0.0, 0.2]
- validities: validities

`rationale`: pi_1 (pure TTB) over-predicts one-reason behaviour (0.84 vs 0.74 in Exp 1; 0.12 vs 0.19 in Exp 2) and pi_2 (Tallying) is catastrophically wrong in the opposite direction. The two experiments demand a rule that is ~74% TTB-consistent when the top cue faces a coalition of moderately valid cues (Exp 1) but ~81% TTB-consistent when one expert (v=.93) towers over the rest (Exp 2). A single lapse parameter cannot produce that dissociation (it moves both metrics toward .5 in the same direction), and pure rank-based frugality cannot either, because the two designs are structurally near-identical in rank terms. What differs between the experiments is the *validity spread*: only a model whose weights are a convex function of the actual stated validities makes the top-cue margin larger in Exp 2 than in Exp 1, and therefore predicts more TTB-consistent choice in Exp 2 exactly as observed.

The implementation follows the arbiter's prescription: w_j proportional to ((v_j-0.5)/(v_max-0.5))^gamma, d = sum_j w_j sign(a_j-b_j), softmax(beta*d) plus lapse. Hand-computation on both designs with gamma around 2, beta around 11 and epsilon around 0.1 yields Exp-1 conflict trials of the form (1 cue vs 3-4 cues) with small positive margins (p_TTB ~ .65-.70), the top-two-vs-bottom-three trial near ceiling (p ~ .95), and Exp-2 conflicts with distinctly larger margins (p_TTB ~ .76-.80), giving pooled predictions of roughly 0.76 on the Exp-1 metric and 0.22 on the Exp-2 tally metric — much closer to the observed 0.7367 / 0.1889 than either extreme heuristic. Ranges are centred so that most sampled subjects are steeply weighted (near-lexicographic) while a minority integrate more evenly; because gamma variation bites hardest where the top cue's weight and the opposing coalition's weight are nearly matched (Exp 1's 1-vs-4 and 1-vs-3 trials), it inflates between-subject variance in Exp 1 more than in Exp 2, matching the observed variance asymmetry (0.0369 vs 0.0124) that pi_1 under-produces. The model is also experiment-invariant by construction: it reads validities and n_features from the experiment, degrades gracefully to equal weighting if validities are missing, follows the higher-validity cue on tally-tie trials, and guesses only when no cue discriminates.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate lands close to the human data on both point estimates and is clearly the best of the three theories on the joint criterion: Exp 1 TTB-consistency 0.7688 vs 0.7367 observed (pi_1 = 0.8425, pi_2 = 0.1758) and Exp 2 tally-following 0.2222 vs 0.1889 observed (pi_1 = 0.1217, pi_2 = 0.8583). Crucially it reproduces the *dissociation* that motivated the arbiter's recommendation: more compensatory behaviour in Exp 1 (closely spaced validities .9-.55) than in Exp 2 (one dominant .93 expert), which no single-lapse variant of a pure heuristic can do. The implementation is faithful to the prescribed family 
i.e. w_j \propto ((v_j-0.5)/(v_max-0.5))^gamma, d = sum w_j sign(a_j-b_j), softmax(beta*d) + lapse, weights read from the experiment's own validities, graceful fallback, tally-ties resolved by the higher-validity cue.

Two residuals remain. (1) Both metrics are biased in the SAME absolute direction (+0.032 on Exp 1, +0.033 on Exp 2), but that means they are biased in OPPOSITE directions in 'TTB-ness' space: the model is slightly too TTB-like in Exp 1 and slightly too tally-like in Exp 2. Global knobs (beta, epsilon, gamma centre) cannot fix both 
any move that helps one hurts the other 
so this residual is not addressable by a simple re-centring and should be treated as near the floor of this family unless the gamma *distribution* is reshaped. (2) The clearer, addressable miss is dispersion: between-subject variance on Exp 1 is 0.0135 vs 0.0369 observed (under-dispersed by ~2.7x), while Exp 2 variance is slightly over-produced (0.0156 vs 0.0124). The theory's own claim is that gamma heterogeneity should bite hardest exactly where the top cue and the opposing coalition are nearly matched (Exp 1's 1-vs-3 / 1-vs-4 conflicts) and hardly at all where one cue dominates (Exp 2). The current parameter ranges do not deliver that asymmetry, because a large share of the simulated heterogeneity comes from beta/epsilon (which spread both experiments about equally) rather than from gamma (which should spread Exp 1 selectively).
Rationale: The candidate was ACCEPTED and the mechanism family is correct 
keep the CVW predict/policy exactly as written and change only the parameter distribution. Targeted, in-family adjustment: (a) widen the gamma range substantially and asymmetrically, e.g. gamma in [0.6, 3.8], so a genuine minority of near-linear integrators coexists with a majority of steep, near-lexicographic subjects; this is the knob that inflates Exp-1 between-subject variance (currently 0.0135 vs 0.0369) without inflating Exp-2 variance, because with the .93 cue towering over the field the weight profile is near-degenerate for any gamma > ~1. (b) To avoid over-shooting Exp-2 variance (already 0.0156 vs 0.0124), *narrow* the nuisance-noise ranges that spread both experiments equally: tighten beta to roughly [9, 14] and epsilon to roughly [0.02, 0.10], so heterogeneity is carried by gamma rather than by lapse/temperature. (c) Choose the widened gamma range so its behavioural centre of mass stays where it is now (the current point estimates are only ~0.03 off and are off in mutually opposing directions, so do NOT re-centre gamma, beta or epsilon to chase either metric 
any global shift will trade one error for the other). Do not switch mechanism families; the target of this iteration is variance calibration in Exp 1 with point estimates held roughly fixed.

**Outcome of this advice:** iter 1 candidate loss=0.0467 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
(no text)
```

## Usage

```json
{
  "input_tokens": 14985,
  "output_tokens": 32768
}
```
