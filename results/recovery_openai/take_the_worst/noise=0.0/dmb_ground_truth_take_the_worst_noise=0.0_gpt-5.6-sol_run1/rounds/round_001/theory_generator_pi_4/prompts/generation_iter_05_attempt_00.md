# generation_iter_05_attempt_00

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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Replace pi_2 with a continuous load-adaptive validity-concentration theory. The new theory should posit that people always integrate the signs of discriminating cues, but the distribution of attention across those cues changes with comparison load. With few discriminating cues, attention is approximately uniform, producing tally-like choices and explaining Experiment 4. As load increases, limited representational precision causes attention to concentrate progressively on cues with higher communicated validity; the decision remains an integration process rather than switching categorically between two strategies. At loads 6-8, this concentration should produce strong leading-cue adherence on exact tally ties, explaining Experiment 2, and similar top-cue adherence whether tally and top cue agree or conflict, explaining Experiment 3. A gradual, heterogeneous concentration function should produce the positive but moderate crossover in Experiment 1 rather than pi_3's overly large predicted crossover. Any implementation should use only pair-level observables such as the number of discriminating cues and the validity ordering, never experiment identity. A useful operationalization is a signed weighted sum whose validity-weight exponent or attention concentration increases smoothly with discriminating load, with stable subject-level capacity or concentration parameters and ordinary response noise applied afterward. This creates a genuinely stronger competitor to pi_3: it shares the empirically necessary load sensitivity but rejects pi_3's discrete prospective recruitment of equal-weight versus first-cue strategies.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.980)

**Description**
Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Comparison expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    discriminating = np.flatnonzero(differences != 0)
    load = float(discriminating.size)
    if load == 0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["capacity"])
    gate_slope = float(parameters["gate_slope"])

    # Probability that exhaustive equal-weight integration is recruited.
    # This stable logistic gate approaches one below capacity and zero above it.
    gate_logit = gate_slope * (capacity - load)
    if gate_logit >= 0.0:
        exp_neg = np.exp(-gate_logit)
        p_integrate = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(gate_logit)
        p_integrate = exp_pos / (1.0 + exp_pos)

    # Manageable-load strategy: all discriminating cues receive equal weight.
    tally_margin = float(np.sum(np.sign(differences)))
    if tally_margin > 0.0:
        tally_scores = np.array([1.0, 0.0])
    elif tally_margin < 0.0:
        tally_scores = np.array([0.0, 1.0])
    else:
        tally_scores = np.array([0.0, 0.0])

    # Overload strategy: validity-ranked compression to the first diagnostic cue.
    cue_order = np.argsort(-validities, kind="stable")
    lex_scores = np.array([0.0, 0.0])
    for cue in cue_order:
        if differences[cue] > 0.0:
            lex_scores = np.array([1.0, 0.0])
            break
        if differences[cue] < 0.0:
            lex_scores = np.array([0.0, 1.0])
            break

    beta = float(parameters["beta"])

    def noisy_strategy_choice(scores):
        logits = beta * np.asarray(scores, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    # Response noise follows, rather than determines, strategy recruitment.
    p_tally = noisy_strategy_choice(tally_scores)
    p_lex = noisy_strategy_choice(lex_scores)
    p_core = p_integrate * p_tally + (1.0 - p_integrate) * p_lex

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- capacity: [4.5, 5.5]
- gate_slope: [2.0, 5.0]
- beta: [1.55, 2.05]
- epsilon: [0.0, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0300 (var=0.0056) vs this=-0.0028 (var=0.0068)
- Experiment 2: real=0.6683 (var=0.0367) vs this=0.6600 (var=0.0188)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.6512 (var=0.0114)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.8291 (var=0.0019)


---

### `pi_1` (overall score: 0.173)

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
- Experiment 1: real=-0.0300 (var=0.0056) vs this=-0.0167 (var=0.0052)
- Experiment 2: real=0.6683 (var=0.0367) vs this=-0.6542 (var=0.0564)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.0041 (var=0.0094)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.8391 (var=0.0151)


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
- Experiment 1: real=-0.0300 (var=0.0056) vs this=0.8033 (var=0.0243)
- Experiment 2: real=0.6683 (var=0.0367) vs this=0.7008 (var=0.0514)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.0066 (var=0.0037)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.5031 (var=0.0037)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1825 -> ACCEPTED
- iter 2: loss=0.2617 -> REJECTED
- iter 3: loss=0.2060 -> REJECTED
- iter 4: loss=0.2001 -> REJECTED
- iter 5: loss=0.1776 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.1776 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast top-cue adherence between tally-congruent and
    # tally-conflicting trials, matching on absolute tally margin.
    cells = {(d, c): [] for d in (2, 4, 6) for c in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the highest-validity cue in this experiment.
        if a[0] > b[0]:
            top_choice = 0
            top_relative_margin = int(np.sum(a > b) - np.sum(b > a))
        elif b[0] > a[0]:
            top_choice = 1
            top_relative_margin = int(np.sum(b > a) - np.sum(a > b))
        else:
            continue

        d = abs(top_relative_margin)
        if d not in (2, 4, 6) or top_relative_margin == 0:
            continue

        congruence = 1 if top_relative_margin > 0 else -1
        top_adherence = 1.0 if response == top_choice else -1.0
        cells[(d, congruence)].append(top_adherence)

    matched_contrasts = []
    for d in (2, 4, 6):
        congruent = cells[(d, 1)]
        conflicting = cells[(d, -1)]
        if len(congruent) == 0 or len(conflicting) == 0:
            continue
        # Division by two keeps the metric in [-1, 1].
        matched_contrasts.append(
            0.5 * (float(np.mean(congruent)) - float(np.mean(conflicting)))
        )

    if len(matched_contrasts) == 0:
        return float("nan")
    return float(np.mean(matched_contrasts))
```

**Observed (real) value:** -0.0300 (var=0.0056)
**Previous candidate values (this loop):**
  - iter 1: 0.0294 (var=0.0015) (Δ vs real +0.0594)
  - iter 2: 0.1478 (var=0.0099) (Δ vs real +0.1778)
  - iter 3: 0.0622 (var=0.0029) (Δ vs real +0.0922)
  - iter 4: 0.1050 (var=0.0042) (Δ vs real +0.1350)
  - iter 5 (most recent): 0.0356 (var=0.0022) (Δ vs real +0.0656)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0167 (var=0.0052)
- pi_2: 0.8033 (var=0.0243)
- pi_3: -0.0028 (var=0.0068)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # +1 denotes an A prediction and -1 a B prediction.
        tally_margin = np.sum(np.sign(d))
        if tally_margin == 0:
            continue
        tally_direction = 1.0 if tally_margin > 0 else -1.0

        # Validities decrease with feature index in this experiment.
        discriminating = np.flatnonzero(d != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_direction = 1.0 if d[first] > 0 else -1.0

        # Agreement trials cannot distinguish the theories.
        if tally_direction == ttb_direction:
            continue

        observed_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(observed_direction * tally_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.6683 (var=0.0367)
**Previous candidate values (this loop):**
  - iter 1: 0.5167 (var=0.0113) (Δ vs real -0.1517)
  - iter 2: 0.8175 (var=0.0068) (Δ vs real +0.1492)
  - iter 3: 0.5742 (var=0.0141) (Δ vs real -0.0942)
  - iter 4: 0.5667 (var=0.0211) (Δ vs real -0.1017)
  - iter 5 (most recent): 0.5758 (var=0.0157) (Δ vs real -0.0925)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7008 (var=0.0514)
- pi_1: -0.6542 (var=0.0564)
- pi_3: 0.6600 (var=0.0188)

### Experiment 3
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return a parity-matched low-to-high load crossover index."""
    if data is None or len(data) == 0:
        return float("nan")

    loads = []
    majority_consistent = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        loads.append(int(np.count_nonzero(diff)))

        tally_margin = float(np.sum(np.sign(diff)))
        if tally_margin > 0:
            majority_response = 0
        elif tally_margin < 0:
            majority_response = 1
        else:
            majority_response = -1

        majority_consistent.append(
            float(int(response) == majority_response)
            if majority_response >= 0 else np.nan
        )

    work = pd.DataFrame({
        "load": np.asarray(loads, dtype=int),
        "majority_consistent": np.asarray(majority_consistent, dtype=float),
    })

    p = {}
    for load in (3, 4, 6, 7, 8):
        values = work.loc[
            work["load"] == load, "majority_consistent"
        ].dropna()
        if len(values) == 0:
            return float("nan")
        p[load] = float(values.mean())

    odd_crossover = p[3] - p[7]
    even_crossover = p[4] - 0.5 * (p[6] + p[8])
    return float(0.5 * (odd_crossover + even_crossover))

```

**Observed (real) value:** 0.3612 (var=0.0106)
**Previous candidate values (this loop):**
  - iter 1: 0.6406 (var=0.0084) (Δ vs real +0.2794)
  - iter 2: 0.4144 (var=0.0171) (Δ vs real +0.0531)
  - iter 3: 0.6166 (var=0.0125) (Δ vs real +0.2553)
  - iter 4: 0.5587 (var=0.0089) (Δ vs real +0.1975)
  - iter 5 (most recent): 0.6872 (var=0.0075) (Δ vs real +0.3259)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6512 (var=0.0114)
- pi_2: 0.0066 (var=0.0037)
- pi_1: 0.0041 (var=0.0094)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate of validity-leading-cue-consistent choices on overload trials."""
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        discriminating = np.flatnonzero(differences != 0)

        # Focus on the 6- and 8-cue overload conditions.
        if discriminating.size < 6:
            continue

        # Validities are strictly descending by feature index, so the first
        # discriminating index is the highest-validity diagnostic cue.
        leading_cue = int(discriminating[0])
        lex_response = 0 if differences[leading_cue] > 0 else 1
        scores.append(float(int(row["response"]) == lex_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8756 (var=0.0078)
**Previous candidate values (this loop):**
  - iter 1: 0.8328 (var=0.0029) (Δ vs real -0.0428)
  - iter 2: 0.6334 (var=0.0047) (Δ vs real -0.2422)
  - iter 3: 0.7828 (var=0.0027) (Δ vs real -0.0928)
  - iter 4: 0.7794 (var=0.0035) (Δ vs real -0.0962)
  - iter 5 (most recent): 0.8434 (var=0.0023) (Δ vs real -0.0322)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5031 (var=0.0037)
- pi_3: 0.8291 (var=0.0019)
- pi_1: 0.8391 (var=0.0151)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Precision-Annealed Rank Integration theory: people always combine the signed evidence supplied by every discriminating cue. Comparison load does not recruit a different heuristic; instead, it continuously changes the precision with which cue priorities can be represented. At low load, the attention distribution is diffuse and discriminating cues receive nearly equal weight, approximating tallying. As the number of discriminating cues exceeds a stable representational capacity, attention is progressively concentrated toward cues higher in the communicated-validity ordering. Attention is implemented as a Gibbs distribution over validity ranks, with its concentration increasing smoothly with load. The load curve is mildly powered while anchored at high load, reducing premature concentration without weakening overload attention. The highest-validity cue retains its original Gibbs attention mass, while attention conditional on the remaining cues is flattened so that lower-validity coalitions remain collectively influential. The normalized weighted evidence is then translated into a choice through ordinary logistic response noise. Stable individual differences generate heterogeneous but gradual crossovers without categorical switching between tallying and Take The Best.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Precision-Annealed Rank Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    signs = np.sign(differences)
    discriminating = np.flatnonzero(signs != 0.0)
    load = float(discriminating.size)

    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    # A cue's rank is the number of cues having strictly greater communicated
    # validity. Thus equally valid cues receive exactly equal attention.
    ranks = np.asarray(
        [np.sum(validities > validities[j]) for j in range(n_features)],
        dtype=float,
    )

    capacity = float(parameters["capacity"])
    transition_width = float(parameters["transition_width"])
    concentration_gain = float(parameters["concentration_gain"])
    schedule_power = float(parameters["schedule_power"])

    # Smoothly increasing attention concentration. Powering the curve relative
    # to a load-7 anchor reduces premature low-load concentration while leaving
    # the overload concentration of the accepted model approximately intact.
    def stable_sigmoid(value):
        if value >= 0.0:
            return 1.0 / (1.0 + np.exp(-value))
        ex = np.exp(value)
        return ex / (1.0 + ex)

    x = (load - capacity) / transition_width
    concentration_fraction = stable_sigmoid(x)
    anchor_fraction = stable_sigmoid((7.0 - capacity) / transition_width)
    concentration = concentration_gain * anchor_fraction * (
        concentration_fraction / anchor_fraction
    ) ** schedule_power

    active_ranks = ranks[discriminating]
    log_attention = -concentration * active_ranks
    log_attention -= np.max(log_attention)
    attention = np.exp(log_attention)
    attention /= attention.sum()

    # Preserve the Gibbs mass assigned to the highest-validity active cue(s),
    # but flatten only the conditional distribution over the remaining cues.
    top_mask = active_ranks == np.min(active_ranks)
    tail_mask = ~top_mask
    if np.any(tail_mask):
        top_mass = float(np.sum(attention[top_mask]))
        tail_flatten = float(parameters["tail_flatten"])
        tail_logits = tail_flatten * log_attention[tail_mask]
        tail_logits -= np.max(tail_logits)
        tail_distribution = np.exp(tail_logits)
        tail_distribution /= tail_distribution.sum()

        adjusted_attention = np.zeros_like(attention)
        adjusted_attention[top_mask] = attention[top_mask]
        adjusted_attention[tail_mask] = (1.0 - top_mass) * tail_distribution
        attention = adjusted_attention

    # Positive evidence favors A and negative evidence favors B. Normalizing
    # attention keeps response precision comparable across different loads.
    evidence = float(np.dot(attention, signs[discriminating]))

    beta = float(parameters["beta"])
    logits = np.array([0.5 * beta * evidence, -0.5 * beta * evidence], dtype=float)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5], dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- capacity: [5.3, 5.8]
- transition_width: [0.6, 0.95]
- concentration_gain: [1.15, 1.55]
- schedule_power: [1.15, 1.35]
- tail_flatten: [0.45, 0.70]
- beta: [4.2, 5.4]
- epsilon: [0.0, 0.06]

`rationale`: This is a localized edit to the accepted iteration-1 model. First, a mildly powered concentration curve is normalized at load 7. It therefore lowers the consequential residual concentration around load 3, moving Experiment 2 toward greater tally consistency, while approximately preserving the accepted model's concentration at loads 6–8. Second, the original Gibbs mass of the highest-validity discriminating cue is retained exactly, but the conditional distribution of the remaining mass is flattened. This strengthens coalitions among lower-ranked cues on the high-load conflicts driving Experiment 3 without globally taking attention away from the leading cue, avoiding the Experiment 4 damage caused by the rejected uniform-attention mixture. The modification remains a single continuous integration process: every discriminating sign enters one weighted sum, and neither experiment identity nor a discrete strategy gate is used. All original capacity, gain, response-noise, and lapse ranges are retained; only narrow subject-level ranges for the two targeted shape parameters are added.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed continuous load-adaptive integration family and is a useful accepted starting point, but its fit is not yet experiment-invariant. Experiment 1 is close in magnitude, although the contrast has the wrong sign (candidate 0.0294 versus observed -0.0300). Experiment 4 is also reasonably close but modestly underpredicts leading-cue adherence (0.8328 versus 0.8756). The major failure is Experiment 3: the predicted low-to-high-load crossover is 0.6406, almost twice the observed 0.3612 and essentially as excessive as pi_3. Thus, despite avoiding an explicit strategy switch, the present sigmoid-plus-rank-Gibbs implementation produces a functionally abrupt crossover. Experiment 2 also shows too little tally-consistent responding on low-load cue-conflict trials (0.5167 versus 0.6683), indicating that even the small residual rank concentration at low load is consequential. Between-subject variance is additionally too low in Experiments 1, 2, and 4, suggesting overly narrow stable-parameter ranges.
Rationale: Retain the continuous signed-integration mechanism, but recalibrate its concentration schedule and heterogeneity. The current rank Gibbs weights amplify small positive concentration values across eight ordinal rank steps, so attention is already insufficiently uniform at low load and then becomes too concentrated around loads 6-8. Use a more delayed and broader annealing curve—such as a higher capacity, larger transition width, or a low-load-baseline-subtracted sigmoid—so concentration is effectively zero around load 3 but rises more gradually through loads 6-8. Softening the rank distance or using communicated-validity magnitudes rather than unit-spaced ranks would likewise prevent an overly sharp evidence reversal while remaining within the prescribed family. Compensate for any loss in Experiment 4 with somewhat greater response precision rather than restoring excessive attention concentration: a higher beta can preserve decisive choices when integrated evidence clearly favors the leading cue without forcing the large Experiment 3 crossover. Finally, widen the subject-level capacity, width, gain, and/or beta ranges to better match the observed variances. The next fit should primarily reduce Experiment 3's crossover, raise Experiment 2's tally consistency, and preserve Experiment 4 near 0.88; Experiment 1 should remain near zero.

**Outcome of this advice:** iter 1 candidate loss=0.1825 -> iter 2 candidate loss=0.2617 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed continuous load-adaptive integration family, but it was rejected because the recalibration traded one mismatch for several larger ones. Baseline subtraction successfully reduced Experiment 3's crossover from 0.6406 to 0.4144, close to the observed 0.3612. However, exact uniformity at low load combined with increased beta overshot Experiment 2's tally consistency: 0.8175 versus 0.6683. More seriously, the weakened high-load concentration collapsed Experiment 4 leading-cue adherence from 0.8328 to 0.6334, far below 0.8756, and moved Experiment 1's near-zero contrast sharply upward to 0.1478 versus -0.0300. The increased heterogeneity also overshot Experiment 1 and Experiment 3 variances while Experiment 2 and Experiment 4 variances remained too low. Thus the candidate no longer provides an experiment-invariant account, despite improving Experiment 3.
Rationale: Because this iteration was rejected, build from the accepted iteration-1 model rather than repeating the rejected push toward a broad, baseline-subtracted schedule and higher beta. Preserve iteration 1's high-load behavior, which already fit Experiments 1 and 4 reasonably well. Try a different in-family adjustment: change the geometry of attention rather than globally weakening concentration. Unit-spaced exponential rank weights can suppress all lower-ranked cues too uniformly; use normalized communicated-validity gaps, compressed rank distances, or a softened Gibbs normalization so the leading cue can retain a distinct advantage while the remaining cues continue to contribute collectively. This may preserve strong leading-cue adherence in Experiment 4 and the near-zero Experiment 1 congruence contrast without forcing Experiment 3's excessive crossover. At load 3, retain a small positive concentration rather than imposing exact uniformity, and calibrate it between the accepted and rejected versions so Experiment 2 falls between 0.5167 and 0.8175, near 0.6683. Avoid another broad increase in beta or parameter ranges: the rejected candidate shows that response precision cannot compensate for insufficient high-load concentration, and indiscriminate widening worsened some variance fits. The target is a localized modification of attention distance/normalization that reduces Experiment 3 while leaving the accepted base's Experiments 1 and 4 largely intact.

**Outcome of this advice:** iter 2 candidate loss=0.2617 -> iter 3 candidate loss=0.2060 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed continuous load-adaptive integration family, but it was rejected by the accept gate and is worse overall than the accepted iteration-1 base (loss 0.2060 versus 0.1825). Sublinear rank compression moved Experiment 2 in the desired direction, from 0.5167 to 0.5742 against an observed 0.6683, but did not solve the principal Experiment 3 failure: the crossover remains 0.6166 versus 0.3612. It also degraded the previously reasonable fits in Experiment 1 (0.0622 versus -0.0300) and Experiment 4 (0.7828 versus 0.8756). Thus compression of all larger rank distances increases lower-cue influence too indiscriminately: it weakens leading-cue choices in Experiment 4 without supplying enough tally-consistent evidence on the high-load conflict trials that drive Experiment 3. Between-subject variance also remains too low in Experiments 1, 2, and 4.
Rationale: Do not repeat the rejected global schedule recalibration from iteration 2 or the rejected sublinear rank compression from the current iteration. Starting from the accepted iteration-1 model, try a different, localized normalization within the same mechanism family: retain its original load-dependent Gibbs profile but add a modest residual equal-attention floor, for example attention = (1-rho)*Gibbs + rho*uniform, with a stable subject-level rho and a narrowly calibrated range. This separates a sharp leading-cue component from residual collective evidence instead of flattening every rank gap. On Experiment 3 conflict trials, the uniform component contributes in the tally direction and should reduce the excessive crossover; in many Experiment 4 overload trials with exact or near tally balance, it contributes little directional evidence, allowing the concentrated component to preserve leading-cue adherence. It should also move Experiment 2 upward toward 0.6683 and pull Experiment 1's congruence contrast toward zero. Tune rho conservatively and, only if necessary, make a small compensating adjustment to concentration gain rather than another broad beta increase. Targeted heterogeneity in rho may improve the understated subject variance without indiscriminately widening all parameter ranges. All modulation should remain based on load and the active validity ordering, never experiment identity.

**Outcome of this advice:** iter 3 candidate loss=0.2060 -> iter 4 candidate loss=0.2001 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed continuous load-adaptive integration family, but the current residual-uniform modification was rejected and does not beat the accepted iteration-1 base (loss 0.2001 versus 0.1825). It produces the intended directional changes in Experiments 2 and 3, but they are too small: Experiment 2 remains insufficiently tally-consistent (0.5667 versus 0.6683), while Experiment 3 still predicts an excessive crossover (0.5587 versus 0.3612). At the same time, uniformly reallocating attention away from the validity-weighted profile damages the previously stronger fits: Experiment 4 leading-cue adherence falls from the accepted base's 0.8328 to 0.7794, versus 0.8756 observed, and Experiment 1's contrast rises from 0.0294 to 0.1050, moving farther from the observed -0.0300. Thus a uniform mixture over all cues creates the wrong tradeoff: it weakens the leading cue globally rather than selectively strengthening collective lower-cue evidence. Variance is now reasonably close in Experiments 1 and 3 but remains understated in Experiments 2 and 4.
Rationale: The previous recommendation to add a residual uniform-attention component was rejected, so do not repeat it or merely retune rho. On top of the unchanged accepted iteration-1 base, try a top-mass-preserving modification of the Gibbs normalization: retain the original total attention assigned to the highest-validity discriminating cue, but flatten or redistribute only the conditional attention distribution among the remaining cues. This remains a single continuous signed integration mechanism, yet can let coalitions of lower-ranked cues contribute more strongly on Experiment 3 conflict trials without sacrificing the leading-cue mass needed by Experiment 4 or disturbing Experiment 1 as much as the global uniform mixture did. Separately, Experiment 2 requires a modest reduction of concentration specifically at load 3. Use a smooth schedule-shape parameter that lowers the low-load concentration while anchoring concentration around loads 6–8 to the accepted base—for example, a powered and high-load-renormalized sigmoid—rather than the previously rejected baseline subtraction or broad transition widening that collapsed Experiment 4. Calibrate this low-load change between the accepted base and the rejected exact-uniform version. Avoid broad beta increases and indiscriminate range widening; if additional heterogeneity is needed, place a narrow subject-level range on the tail-flattening or schedule-shape parameter. The next candidate should preserve Experiment 4 near or above 0.83 and Experiment 1 near zero while moving Experiment 2 toward 0.67 and Experiment 3 toward 0.36.

**Outcome of this advice:** iter 4 candidate loss=0.2001 -> iter 5 candidate loss=0.1776 -> the gate marked it ACCEPTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed continuous load-adaptive integration family and has been accepted by the gate, modestly improving aggregate loss from 0.1825 to 0.1776. The powered, high-load-anchored schedule helped Experiment 2, moving tally consistency from 0.5167 to 0.5758, while preserving good fits in Experiment 4 (0.8434 versus 0.8756) and a near-zero Experiment 1 contrast (0.0356 versus -0.0300). However, the principal theoretical target remains badly missed: Experiment 3's crossover increased from the accepted base's 0.6406 to 0.6872, nearly twice the observed 0.3612 and even farther into the pi_3-like regime the arbiter intended this theory to avoid. The current improvement in aggregate loss therefore masks a central qualitative failure. The direction of the Experiment 3 change is diagnostically important: flattening the conditional tail while preserving top-cue mass apparently redistributes attention toward lower-ranked cues that support the leading cue on these particular high-load configurations, rather than strengthening the opposing coalition. Subject-level variance also remains understated in Experiments 1, 2, and 4.
Rationale: Build on this accepted candidate, retaining the successful powered, high-load-anchored schedule and the top-mass-preserving architecture. Do not discard those changes: they improved Experiment 2 without causing the Experiment 1 and 4 damage seen in rejected iterations. Instead, reverse the unsuccessful curvature within the non-top tail. The current tail_flatten range of 0.45-0.70 makes the tail more uniform and worsens Experiment 3. Test a neutral-to-sharpened conditional tail, with the multiplier centered at or above 1, so the preserved tail mass is concentrated more on the next-highest-validity cues rather than spread toward the lowest ranks. In Experiment 3, cues immediately below the leader form the relevant opposing coalition, whereas spreading mass farther down the ranking can reinforce the leader; selective tail sharpening should therefore reduce the crossover while leaving the leading cue's total mass—and hence Experiment 4—largely intact. This is a different in-family adjustment from the previously rejected global rank compression and uniform mixture. Keep the low-load schedule adjustment because it helped Experiment 2, but consider a small additional increase in schedule power only if tail sharpening pushes Experiment 2 away from 0.6683; do not return to exact low-load uniformity or broad beta increases. Include the neutral value tail_flatten=1 in the search and allow moderate heterogeneity in this curvature parameter to address the understated variances. The decisive next target is a substantial reduction of Experiment 3 toward 0.36 while holding Experiment 4 above roughly 0.83, Experiment 1 near zero, and continuing Experiment 2's movement toward 0.67.

**Outcome of this advice:** iter 5 candidate loss=0.1776 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Precision-Annealed Rank Integration theory: people always combine the signed evidence supplied by every discriminating cue. Comparison load does not recruit a different heuristic; instead, it continuously changes the precision with which cue priorities can be represented. At low load, the attention distribution is diffuse and discriminating cues receive nearly equal weight, approximating tallying. As the number of discriminating cues exceeds a stable representational capacity, attention is progressively concentrated toward cues higher in the communicated-validity ordering. Attention is implemented as a Gibbs distribution over validity ranks, with its concentration increasing smoothly with load. The load curve is mildly powered while anchored at high load, reducing premature concentration without weakening overload attention. The highest-validity cue retains its original Gibbs attention mass, while attention conditional on the remaining cues is neutral-to-sharpened, concentrating residual attention on the next-highest-validity cues rather than diffusely spreading it over the lowest-ranked cues. The normalized weighted evidence is then translated into a choice through ordinary logistic response noise. Stable individual differences generate heterogeneous but gradual crossovers without categorical switching between tallying and Take The Best.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Precision-Annealed Rank Integration expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {validities.shape[0]} != n_features {n_features}.\"\n        )\n\n    differences = a - b\n    signs = np.sign(differences)\n    discriminating = np.flatnonzero(signs != 0.0)\n    load = float(discriminating.size)\n\n    if discriminating.size == 0:\n        return np.array([0.5, 0.5], dtype=float)\n\n    # A cue's rank is the number of cues having strictly greater communicated\n    # validity. Thus equally valid cues receive exactly equal attention.\n    ranks = np.asarray(\n        [np.sum(validities > validities[j]) for j in range(n_features)],\n        dtype=float,\n    )\n\n    capacity = float(parameters[\"capacity\"])\n    transition_width = float(parameters[\"transition_width\"])\n    concentration_gain = float(parameters[\"concentration_gain\"])\n    schedule_power = float(parameters[\"schedule_power\"])\n\n    # Smoothly increasing attention concentration. Powering the curve relative\n    # to a load-7 anchor reduces premature low-load concentration while leaving\n    # the overload concentration of the accepted model approximately intact.\n    def stable_sigmoid(value):\n        if value >= 0.0:\n            return 1.0 / (1.0 + np.exp(-value))\n        ex = np.exp(value)\n        return ex / (1.0 + ex)\n\n    x = (load - capacity) / transition_width\n    concentration_fraction = stable_sigmoid(x)\n    anchor_fraction = stable_sigmoid((7.0 - capacity) / transition_width)\n    concentration = concentration_gain * anchor_fraction * (\n        concentration_fraction / anchor_fraction\n    ) ** schedule_power\n\n    active_ranks = ranks[discriminating]\n    log_attention = -concentration * active_ranks\n    log_attention -= np.max(log_attention)\n    attention = np.exp(log_attention)\n    attention /= attention.sum()\n\n    # Preserve the Gibbs mass assigned to the highest-validity active cue(s),\n    # but sharpen only the conditional distribution over the remaining cues.\n    top_mask = active_ranks == np.min(active_ranks)\n    tail_mask = ~top_mask\n    if np.any(tail_mask):\n        top_mass = float(np.sum(attention[top_mask]))\n        tail_flatten = float(parameters[\"tail_flatten\"])\n        tail_logits = tail_flatten * log_attention[tail_mask]\n        tail_logits -= np.max(tail_logits)\n        tail_distribution = np.exp(tail_logits)\n        tail_distribution /= tail_distribution.sum()\n\n        adjusted_attention = np.zeros_like(attention)\n        adjusted_attention[top_mask] = attention[top_mask]\n        adjusted_attention[tail_mask] = (1.0 - top_mass) * tail_distribution\n        attention = adjusted_attention\n\n    # Positive evidence favors A and negative evidence favors B. Normalizing\n    # attention keeps response precision comparable across different loads.\n    evidence = float(np.dot(attention, signs[discriminating]))\n\n    beta = float(parameters[\"beta\"])\n    logits = np.array([0.5 * beta * evidence, -0.5 * beta * evidence], dtype=float)\n    logits -= np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    epsilon = float(parameters[\"epsilon\"])\n    probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5], dtype=float)\n    probs = np.clip(probs, 0.0, 1.0)\n    probs /= probs.sum()\n    return probs\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / len(probs)\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "capacity": "[5.3, 5.8]",
    "transition_width": "[0.6, 0.95]",
    "concentration_gain": "[1.15, 1.55]",
    "schedule_power": "[1.15, 1.35]",
    "tail_flatten": "[1.0, 1.45]",
    "beta": "[4.2, 5.4]",
    "epsilon": "[0.0, 0.06]"
  },
  "rationale": "This is a minimal edit to the accepted iteration-5 candidate. The successful powered, load-7-anchored concentration schedule and top-mass-preserving normalization are unchanged. Only the conditional-tail curvature range is reversed from flattening (0.45-0.70) to neutral-to-moderate sharpening (1.0-1.45). The previous flattening spread residual attention toward the lowest-ranked cues and increased Experiment 3's already excessive crossover. Sharpening instead directs the unchanged non-top mass toward the immediately subordinate, relatively valid cues, which form the coalition opposing the leading cue in the high-load Experiment 3 configurations. Because the total mass assigned to the leading cue is preserved exactly, this targeted redistribution should retain Experiment 4's strong leading-cue adherence and avoid the Experiment 1 damage caused by global uniform mixtures. Including 1.0 supplies the unmodified conditional Gibbs profile, while the range up to 1.45 introduces moderate stable heterogeneity. No experiment identity, strategy gate, broad beta change, or additional mechanism is introduced."
}
```

## Usage

```json
{
  "input_tokens": 15600,
  "output_tokens": 2264
}
```
