# generation_iter_04_attempt_00

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
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace pi_1 with a reliability-weighted evidence-integration theory, retaining pi_2 as the equal-weight benchmark. The new theory should claim that people integrate all discriminating expert ratings, weighting each cue by its communicated validity, rather than stopping after one cue or counting every cue equally. A principled decision variable is E = sum_j w(v_j)*(A_j-B_j), with a monotone reliability transformation such as w(v)=log(v/(1-v)); choice probability can be generated by a lapse-contaminated sigmoid of beta*E. This mechanism naturally predicts graded confidence from the total strength of evidence. In Experiment 1, lower-validity cues collectively outweigh the top discriminating cue on the conflict trials, but reliability weighting makes that opposition less extreme than equal-weight tallying, potentially matching the observed 0.4033. In Experiment 2, increasingly numerous concordant cues generally increase total weighted evidence, producing the strong margin-dependent profile that TTB cannot express. Allow subject-level variation in sensitivity and lapse, but keep the validity-weighting rule task-invariant and avoid explaining the result as a mixture of deterministic TTB and tallying types. Future discriminating designs should independently manipulate the unweighted tally, the highest-validity cue, and the reliability-weighted sum; this would separate the proposed theory from both existing heuristics.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 1.000)

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
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.1350 (var=0.0104)
- Experiment 2: real=89.1422 (var=7.5896) vs this=1.9009 (var=0.0916)


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
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.8450 (var=0.0143)
- Experiment 2: real=89.1422 (var=7.5896) vs this=0.5503 (var=0.0488)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4593 -> ACCEPTED
- iter 2: loss=0.5416 -> REJECTED
- iter 3: loss=0.2611 -> ACCEPTED
- iter 4: loss=0.2481 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 4 at loss=0.2481 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Choice rate for the Take-The-Best winner on TTB-vs-tally conflict trials."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # Validities are strictly descending in feature-index order, so the
        # first nonzero difference identifies the Take-The-Best winner.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        ttb_winner = 0 if diff[discriminating[0]] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Retain only directional conflicts between the two heuristics.
        if ttb_winner != tally_winner:
            aligned.append(float(int(row["response"]) == ttb_winner))

    # The selected design always contains conflict trials; the fallback keeps
    # the function defined for any unexpected slice.
    return float(np.mean(aligned)) if aligned else 0.5
```

**Observed (real) value:** 0.4033 (var=0.0042)
**Previous candidate values (this loop):**
  - iter 1: 0.2963 (var=0.0051) (Δ vs real -0.1071)
  - iter 2: 0.4842 (var=0.0182) (Δ vs real +0.0808)
  - iter 3: 0.3829 (var=0.0078) (Δ vs real -0.0204)
  - iter 4 (most recent): 0.3721 (var=0.0053) (Δ vs real -0.0312)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8450 (var=0.0143)
- pi_2: 0.1350 (var=0.0104)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Total log marginal-likelihood ratio favoring Tallying over TTB.
    # The parameter grids approximate uniform priors over the stated ranges.
    beta_grid = np.linspace(0.1, 20.0, 120)
    epsilon_grid = np.linspace(0.0, 0.5, 61)
    beta = beta_grid[:, None]
    epsilon = epsilon_grid[None, :]

    def logmeanexp(x):
        x = np.asarray(x, dtype=float)
        m = float(np.max(x))
        return float(m + np.log(np.mean(np.exp(x - m))))

    def subject_log_bf(df):
        margins = []
        correct = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_wins = int(np.sum(a > b))
            b_wins = int(np.sum(b > a))
            margin = abs(a_wins - b_wins)

            if margin == 0:
                continue

            winner = 0 if a_wins > b_wins else 1
            margins.append(float(margin))
            correct.append(float(int(row["response"]) == winner))

        if len(margins) == 0:
            return np.nan

        margins = np.asarray(margins, dtype=float)
        correct = np.asarray(correct, dtype=float)
        tiny = 1e-12

        # TTB assigns the same winner probability to every non-tied pair.
        core_ttb = 1.0 / (1.0 + np.exp(-beta))
        p_ttb = (1.0 - epsilon) * core_ttb + epsilon * 0.5
        p_ttb = np.clip(p_ttb, tiny, 1.0 - tiny)
        n_correct = float(np.sum(correct))
        n_error = float(correct.size - n_correct)
        ll_ttb = n_correct * np.log(p_ttb) + n_error * np.log1p(-p_ttb)

        # Tallying's winner probability depends on the observed tally margin.
        ll_tally = np.zeros_like(p_ttb, dtype=float)
        for d, y in zip(margins, correct):
            x = beta * d
            core = np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
            p = (1.0 - epsilon) * core + epsilon * 0.5
            p = np.clip(p, tiny, 1.0 - tiny)
            ll_tally += y * np.log(p) + (1.0 - y) * np.log1p(-p)

        log_ml_tally = logmeanexp(ll_tally)
        log_ml_ttb = logmeanexp(ll_ttb)
        return float(log_ml_tally - log_ml_ttb)

    if "subject_id" in data.columns:
        values = [subject_log_bf(g) for _, g in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_log_bf(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    # Independent participants contribute additive evidence. On a one-subject
    # slice this is simply that participant's log Bayes factor.
    return float(np.sum(values))

```

**Observed (real) value:** 89.1422 (var=7.5896)
**Previous candidate values (this loop):**
  - iter 1: 21.1189 (var=0.3461) (Δ vs real -68.0233)
  - iter 2: 3.2654 (var=0.0834) (Δ vs real -85.8768)
  - iter 3: 45.4224 (var=1.8477) (Δ vs real -43.7199)
  - iter 4 (most recent): 49.0766 (var=1.5310) (Δ vs real -40.0657)
**Other theories' values on this metric (for reference):**
- pi_2: 1.9009 (var=0.0916)
- pi_1: 0.5503 (var=0.0488)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Communicated-Reliability Accumulation theory proposes that people translate each expert's stated validity into diagnostic log-odds and integrate the evidence supplied by every discriminating rating. For expert j, the subjective reliability weight is w_j = logit(v_j)^gamma, where gamma is constrained close to one, and the net evidence favoring A is E = sum_j w_j(A_j-B_j). Thus, a highly reliable expert contributes somewhat more than a weak expert, but sufficiently numerous weaker experts can collectively overturn a stronger cue. Choice sensitivity varies across people, while occasional lapses produce validity-independent guessing. Because accumulation continues across all cues, the theory predicts graded choice confidence as the amount and reliability of concordant evidence increase; no learning from choice history is assumed because the task provides no outcome feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Communicated-Reliability Accumulation.
    # state has shape (2, n_features), with rows for options A and B.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Reliability accumulation expects shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Clipping only protects the log-odds transform at its mathematical
    # endpoints. A validity of .5 consequently receives zero weight.
    v = np.clip(validities, 0.5, 1.0 - 1e-12)
    log_odds = np.log(v) - np.log1p(-v)
    gamma = float(parameters["gamma"])
    reliability_weights = np.power(log_odds, gamma)

    a = stimulus[0]
    b = stimulus[1]
    evidence_for_a = float(np.dot(reliability_weights, a - b))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Symmetric logits make their difference beta * evidence_for_a, so the
    # core probability of A is sigmoid(beta * evidence_for_a).
    logits = np.array(
        [0.5 * beta * evidence_for_a, -0.5 * beta * evidence_for_a],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probabilities = (1.0 - epsilon) * p_core + epsilon * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities


`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.ones(len(probabilities), dtype=np.float64) / len(probabilities)
    else:
        probabilities /= total
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.2, 0.9]
- epsilon: [0.0, 0.03]
- gamma: [1.07, 1.08]
- validities: validities

`rationale`: This is a parameter-only edit to the accepted model. Reducing beta's upper bound from 1.2 to 0.9 should prevent the weighted-evidence sigmoid from saturating prematurely, preserving stronger probability differences between the evidence margins in Experiment 2. Tightening epsilon from [0, 0.1] to [0, 0.03] avoids capping consistency on the largest-margin trials. Finally, concentrating gamma in [1.07, 1.08] slightly increases the typical leverage of the most reliable cue, while removing unnecessary curvature variation; together with the lower sensitivity, this should move Experiment 1 modestly toward its observed TTB-winner rate and reduce its excessive between-subject variance. The all-cue, task-invariant reliability integration mechanism is otherwise unchanged.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate moves decisively in the right direction relative to both benchmarks, but it does not yet capture the human pattern closely enough. In Experiment 1, its TTB-winner rate is 0.2963 versus 0.4033 observed. Thus, reliability weighting still gives the collection of weaker, tally-consistent cues too much influence relative to the highest-validity cue. In Experiment 2, it produces qualitatively appropriate margin dependence and greatly improves on pi_1 and pi_2, but the log Bayes factor is only 21.1189 versus 89.1422 observed. The model therefore substantially underproduces the strong within-subject increase in choice consistency with evidence margin. Its between-subject variance is also much too small (0.3461 versus 7.5896), indicating overly restricted individual differences in sensitivity or lapse.
Rationale: Retain the prescribed all-cue reliability-weighted accumulation mechanism, but recalibrate it. Introduce a modest task-invariant curvature parameter in the reliability transform, for example w(v) = logit(v)^gamma with gamma constrained near but above 1. This should raise the Experiment 1 TTB-winner rate by giving the most reliable cue somewhat more leverage without reverting to one-reason stopping. Separately, the current beta range of [0.2, 2.0] appears too narrow to generate Experiment 2's strong margin-dependent profile or its subject heterogeneity. Broaden sensitivity substantially, ideally after normalizing weights to keep beta comparable across feature sets, and permit a wider subject-level lapse range. The tuning target is joint: approximately 0.403 on Experiment 1 while markedly increasing Experiment 2's margin-based Bayes factor and between-subject variance. Do not replace accumulation with a TTB/tally mixture; the accepted improvement shows that the recommended mechanism family is promising, but its validity-weight curvature and individual-difference ranges are miscalibrated.

**Outcome of this advice:** iter 1 candidate loss=0.4593 -> iter 2 candidate loss=0.5416 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed all-cue reliability-weighted accumulation family, but this instantiation is worse than the accepted base and fails the joint experimental pattern. Experiment 1 moves past the target: the TTB-winner rate rises from 0.2963 to 0.4842, exceeding the observed 0.4033. Its variance also rises to 0.0182 versus 0.0042 observed, suggesting excessive subject-level dispersion. More importantly, Experiment 2 deteriorates sharply: the log Bayes factor falls from 21.1189 in the accepted base to 3.2654, far below 89.1422. The superlinear validity transform makes later, lower-validity concordant cues contribute too little, while the broadened beta range allows early sigmoid saturation. Together these changes flatten the margin-dependent probability profile that Experiment 2 requires. The wider lapse range can further cap high-margin accuracy. Thus, the previous recommendation to combine gamma substantially above one with broadly expanded beta and lapse ranges was rejected by the gate and should not be repeated.
Rationale: Build on the unchanged accepted base using a smaller, more targeted adjustment. Keep reliability accumulation, but constrain curvature much closer to linear log-odds weighting, such as gamma in approximately [1.03, 1.12], rather than [1.2, 1.6]. This should seek an Experiment 1 value between the base's 0.2963 and the rejected candidate's 0.4842. Do not retain the very broad beta range: use a moderate normalized-evidence temperature range near or narrower than the accepted base, chosen to keep low-margin choices uncertain while avoiding saturation before the largest Experiment 2 margins. Likewise, narrow epsilon rather than widening it, because large lapses suppress the high-margin consistency needed for the large Bayes factor. If extra subject variation is needed, place it chiefly in moderate beta variation without admitting extreme saturating values. The immediate joint targets are an Experiment 1 TTB rate near 0.403 with lower variance and an Experiment 2 Bayes factor above the accepted base's 21.1; a narrow grid search over slight curvature and moderate temperature is preferable to another simultaneous expansion of all parameter ranges.

**Outcome of this advice:** iter 2 candidate loss=0.5416 -> iter 3 candidate loss=0.2611 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is a substantial accepted improvement and remains faithful to the prescribed reliability-weighted, all-cue accumulation family. Experiment 1 is now close: the predicted TTB-winner rate is 0.3829 versus 0.4033 observed, although its between-subject variance remains somewhat high (0.0078 versus 0.0042). Experiment 2 also improves markedly over the prior accepted base, with a log Bayes factor of 45.4224 rather than 21.1189. Nevertheless, it still reaches only about half the observed evidence for margin-dependent responding (89.1422), and its variance is much too low (1.8477 versus 7.5896). Thus, the mechanism captures the direction of both effects but still underproduces the steep, reliable change in choice consistency across evidence margins.
Rationale: Continue tuning from this accepted candidate rather than changing mechanism families. The previous successful advice—slight curvature, restricted sensitivity, and low lapse—should be extended cautiously. In particular, test a modestly lower or narrower beta range by reducing the current upper bound of 1.2 while retaining enough spread for subject differences. The improvement obtained by removing high beta values suggests that residual early sigmoid saturation is still flattening Experiment 2's margin profile. Also tighten epsilon further toward zero so high-margin choices are not capped by lapses. Avoid the previously rejected strategy of broadly expanding beta or lapse. For Experiment 1, only a very small increase in average curvature is warranted to move 0.3829 toward 0.4033; narrow or fix gamma near the upper-middle of the current interval rather than broadening it, which may also reduce the excessive Experiment 1 variance. A focused grid search over beta upper bound, low epsilon, and a nearly fixed gamma should target a materially larger Experiment 2 Bayes factor without sacrificing the now-close Experiment 1 fit.

**Outcome of this advice:** iter 3 candidate loss=0.2611 -> iter 4 candidate loss=0.2481 -> the gate marked it ACCEPTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed reliability-weighted, all-cue accumulation family and is the new running best. Experiment 1 is reasonably close: the TTB-winner rate is 0.3721 versus 0.4033, and its variance of 0.0053 is now close to the observed 0.0042. Experiment 2 remains a material failure, however. Although the log Bayes factor improved from 45.4224 to 49.0766, it is still far below the observed 89.1422, and between-subject variance remains only 1.5310 versus 7.5896. The model therefore predicts margin-sensitive responding in the correct direction but still makes responses across different evidence margins too similar. The latest accepted reduction of high-beta and high-lapse cases helped, so that direction was useful, but it is not sufficient.
Rationale: Continue within reliability-weighted accumulation, using a focused calibration rather than broadening beta or lapse, which was previously rejected. The remaining Experiment 2 deficit likely reflects both residual saturation at the high-beta end and gamma greater than one discounting the incremental contribution of later, lower-validity concordant cues. Try moving gamma modestly closer to linear log-odds weighting, approximately 1.00–1.05, so successive concordant cues produce a stronger graded evidence profile. Counterbalance the resulting tendency to favor the tally winner more strongly in Experiment 1 by shifting beta slightly downward while removing the least sensitive tail—for example, search narrow intermediate ranges around roughly 0.25–0.70 rather than merely lowering the upper bound while retaining a broad range. Keep epsilon near zero, around 0–0.02. This combination should preserve Experiment 1 near 0.403 while increasing separation among Experiment 2 margins. First optimize the mean Experiment 2 Bayes factor; then permit moderate beta dispersion around the best intermediate region to raise its underestimated subject variance without reintroducing the extreme high-beta saturation that previously failed.

**Outcome of this advice:** iter 4 candidate loss=0.2481 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Communicated-Reliability Accumulation theory proposes that people translate each expert's stated validity into diagnostic log-odds and integrate the evidence supplied by every discriminating rating. For expert j, the subjective reliability weight is w_j = logit(v_j)^gamma, where gamma is constrained close to one, and the net evidence favoring A is E = sum_j w_j(A_j-B_j). Thus, a highly reliable expert contributes somewhat more than a weak expert, but sufficiently numerous weaker experts can collectively overturn a stronger cue. Choice sensitivity varies across people, while occasional lapses produce validity-independent guessing. Because accumulation continues across all cues, the theory predicts graded choice confidence as the amount and reliability of concordant evidence increase; no learning from choice history is assumed because the task provides no outcome feedback.",
  "predict": "def predict(parameters, state, history):\n    # Communicated-Reliability Accumulation.\n    # state has shape (2, n_features), with rows for options A and B.\n    stimulus = np.asarray(state, dtype=np.float64)\n    if stimulus.ndim != 2 or stimulus.shape[0] != 2:\n        raise ValueError(\n            f\"Reliability accumulation expects shape (2, n_features); got {stimulus.shape}.\"\n        )\n\n    n_features = stimulus.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} does not match n_features {n_features}.\"\n        )\n\n    # Clipping only protects the log-odds transform at its mathematical\n    # endpoints. A validity of .5 consequently receives zero weight.\n    v = np.clip(validities, 0.5, 1.0 - 1e-12)\n    log_odds = np.log(v) - np.log1p(-v)\n    gamma = float(parameters[\"gamma\"])\n    reliability_weights = np.power(log_odds, gamma)\n\n    a = stimulus[0]\n    b = stimulus[1]\n    evidence_for_a = float(np.dot(reliability_weights, a - b))\n\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Symmetric logits make their difference beta * evidence_for_a, so the\n    # core probability of A is sigmoid(beta * evidence_for_a).\n    logits = np.array(\n        [0.5 * beta * evidence_for_a, -0.5 * beta * evidence_for_a],\n        dtype=np.float64,\n    )\n    logits -= np.max(logits)\n    p_core = np.exp(logits)\n    p_core /= p_core.sum()\n\n    probabilities = (1.0 - epsilon) * p_core + epsilon * 0.5\n    probabilities = np.clip(probabilities, 0.0, 1.0)\n    probabilities /= probabilities.sum()\n    return probabilities\n",
  "policy": "def policy(probs):\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities = np.clip(probabilities, 0.0, None)\n    total = probabilities.sum()\n    if not np.isfinite(total) or total <= 0.0:\n        probabilities = np.ones(len(probabilities), dtype=np.float64) / len(probabilities)\n    else:\n        probabilities /= total\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "beta": "[0.25, 0.70]",
    "epsilon": "[0.0, 0.02]",
    "gamma": "[1.03, 1.04]",
    "validities": "validities"
  },
  "rationale": "This is a parameter-only edit to the accepted model. Moving gamma from [1.07, 1.08] to [1.03, 1.04] makes communicated-validity weighting closer to linear log-odds integration, preserving more of the incremental evidence supplied by numerous lower-validity concordant cues in Experiment 2. Restricting beta to the intermediate range [0.25, 0.70] removes both the least-sensitive subjects and the high-sensitivity cases that saturate before evidence margins can be distinguished. Its slightly lower mean sensitivity also moves conflict choices toward uncertainty, counteracting the lower gamma's tendency to favor the tally-consistent evidence and helping retain the Experiment 1 TTB-winner rate near its target. Finally, tightening epsilon to [0, 0.02] prevents lapses from capping large-margin consistency. The all-cue reliability-accumulation mechanism and all implementation details are otherwise unchanged."
}
```

## Usage

```json
{
  "input_tokens": 10403,
  "output_tokens": 1401
}
```
