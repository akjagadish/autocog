# generation_iter_09_attempt_00

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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 with a new task-adaptive distinctiveness-and-redundancy theory, retaining pi_1 as the one-reason benchmark. The new theory should posit that subjects construct effective cue weights jointly from communicated validity, ordinal/presentation prominence, and the cue's distribution across the task. Isolated dissenting cues can receive a distinctiveness bonus, whereas large concordant coalitions are treated as redundant or suspiciously correlated and can be strongly discounted rather than monotonically accumulated. A conflict-dependent strategy gate should allow lexicographic anchoring when a cue is prominent or when top validities are tied, but distinctiveness-based selection when sparse opposing evidence stands out; importantly, this gate must be driven by observable profile and stimulus statistics rather than experiment-specific labels. This can accommodate strong first-cue adherence in Experiment 1, the negative crossover in Experiment 2 if lone dissent is more salient than a redundant multi-cue coalition, and below-chance highest-validity adherence in Experiment 4 when another cue or cue configuration is more distinctive. Include stable subject-level variation in attention to communicated validity, order prominence, distinctiveness, and redundancy to address the elevated observed variances. The new theory must make signed, trial-level predictions for Experiments 1, 2, and 4 simultaneously and preserve the near-null interaction in Experiment 5 and null prevalence result in Experiment 6; it should not merely add unrestricted experiment-specific mixtures or lapse rates.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.498)

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
- Experiment 1: real=1.0000 (var=0.0000) vs this=0.9900 (var=0.0049)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.8492 (var=0.0121)
- Experiment 3: real=0.0721 (var=0.0052) vs this=-0.0008 (var=0.0015)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0400 (var=0.0384)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.8244 (var=0.0081)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=0.0103 (var=0.0049)


---

### `pi_3` (overall score: 0.178)

**Description**
Conflict-Gated Redundancy-Discounted Attention (CGRDA) proposes that people integrate all discriminating cues, but cue influence depends jointly on instructed validity, serial position, directional redundancy, and the current conflict configuration. Instructed validity and early presentation increase a cue's baseline attentional weight. Successive endorsements of an option have diminishing marginal impact because repeated evidence is treated as partially redundant. When the numbers of cues favoring the two options are closely balanced, attention is reoriented toward the final discriminating cue, producing a conflict-gated recency contribution. Thus, an early compact set of reliable cues can defeat a larger but redundant set, while the last cue can reverse the decision in balanced conflicts. Primacy, redundancy discounting, recency strength, and conflict sensitivity are stable subject-level traits.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CGRDA expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive differences favor B and negative differences favor A.
    diff = stim[1] - stim[0]
    signs = np.sign(diff)
    discriminating = np.flatnonzero(signs != 0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    primacy = float(parameters["primacy"])
    redundancy_discount = float(parameters["redundancy_discount"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    recency_strength = float(parameters["recency_strength"])
    balance_sensitivity = float(parameters["balance_sensitivity"])
    beta = float(parameters["beta"])

    # Convert communicated validities to reliability evidence. Normalizing
    # preserves validity rank without making the score depend arbitrarily
    # on the absolute log-odds scale selected by an experiment.
    clipped = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(clipped / (1.0 - clipped))
    reliability /= max(float(np.max(reliability)), 1e-12)
    reliability = np.maximum(reliability, 0.05)

    count_a = 0
    count_b = 0
    integrated = 0.0
    total_weight = 0.0

    for j in discriminating:
        direction = float(signs[j])
        if direction > 0:
            count_b += 1
            repetition = count_b
        else:
            count_a += 1
            repetition = count_a

        validity_weight = reliability[j] ** validity_sensitivity
        position_weight = np.exp(-primacy * float(j))
        novelty_weight = float(repetition) ** (-redundancy_discount)
        weight = validity_weight * position_weight * novelty_weight

        integrated += direction * weight
        total_weight += weight

    # Put integrated evidence on a common bounded scale across feature
    # counts and experiments.
    core_evidence = integrated / max(total_weight, 1e-12)

    # Recency is strongest at exact tally balance and falls rapidly as one
    # side acquires more endorsements. The final discriminating cue, not
    # necessarily the least-valid cue, determines the direction of this
    # conflict-resolution contribution.
    tally_imbalance = abs(count_b - count_a)
    conflict_gate = np.exp(-balance_sensitivity * float(tally_imbalance))
    last_direction = float(signs[int(discriminating[-1])])
    recency_evidence = recency_strength * conflict_gate * last_direction

    choice_evidence = core_evidence + recency_evidence
    utilities = np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])

    x = beta * utilities
    x = x - np.max(x)
    probs = np.exp(x)
    probs /= probs.sum()
    return probs.astype(np.float64)


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
- primacy: [0.8, 1.4]
- redundancy_discount: [0.25, 1.25]
- validity_sensitivity: [0.3, 1.5]
- recency_strength: [1.15, 1.65]
- balance_sensitivity: [2.5, 5.0]
- beta: [0.8, 1.8]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.3298 (var=0.0054)
- Experiment 3: real=0.0721 (var=0.0052) vs this=-0.3629 (var=0.0033)
- Experiment 4: real=0.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.6381 (var=0.0031)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=-0.3150 (var=0.0123)


---

### `pi_2` (overall score: 0.066)

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
- Experiment 1: real=1.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.4848 (var=0.0022)
- Experiment 3: real=0.0721 (var=0.0052) vs this=-0.1762 (var=0.0033)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0600 (var=0.0564)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.1456 (var=0.0074)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=0.3781 (var=0.0114)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1680 -> ACCEPTED
- iter 2: loss=0.1000 -> ACCEPTED
- iter 3: loss=0.1293 -> REJECTED
- iter 4: loss=0.0727 -> ACCEPTED
- iter 5: loss=0.0672 -> ACCEPTED
- iter 6: loss=0.0867 -> REJECTED
- iter 7: loss=0.0628 -> ACCEPTED
- iter 8: loss=0.0675 -> REJECTED
- iter 9: loss=0.0503 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 9 at loss=0.0503 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Prevalence of subjects choosing the first-discriminating-cue winner above chance."""
    if data is None or len(data) == 0:
        return float("nan")

    def subject_score(subj):
        agreements = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            diff = a - b
            discriminating = np.flatnonzero(diff != 0)
            if discriminating.size == 0:
                continue
            first = int(discriminating[0])
            ttb_response = 0 if diff[first] > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

        if len(agreements) == 0:
            return float("nan")
        rate = float(np.mean(agreements))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        score = subject_score(subj)
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if len(scores) else float("nan")
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 9 (most recent): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices consistent with the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    aligned = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        favored = None

        # Features are already ordered from highest to lowest validity.
        for j in range(min(len(a), len(b))):
            if a[j] > b[j]:
                favored = 0
                break
            if b[j] > a[j]:
                favored = 1
                break

        # Omit completely nondiscriminating pairs, although none occur in
        # the specified design.
        if favored is not None:
            aligned.append(float(int(response) == favored))

    if not aligned:
        return float("nan")
    return float(np.mean(aligned))
```

**Observed (real) value:** 0.3242 (var=0.0174)
**Previous candidate values (this loop):**
  - iter 1: 0.6498 (var=0.0034) (Δ vs real +0.3256)
  - iter 2: 0.5542 (var=0.0041) (Δ vs real +0.2300)
  - iter 3: 0.5708 (var=0.0051) (Δ vs real +0.2467)
  - iter 4: 0.4777 (var=0.0049) (Δ vs real +0.1535)
  - iter 5: 0.4531 (var=0.0035) (Δ vs real +0.1290)
  - iter 6: 0.4890 (var=0.0052) (Δ vs real +0.1648)
  - iter 7: 0.4648 (var=0.0060) (Δ vs real +0.1406)
  - iter 8: 0.4371 (var=0.0053) (Δ vs real +0.1129)
  - iter 9 (most recent): 0.3877 (var=0.0071) (Δ vs real +0.0635)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)
- pi_4: 0.9925 (var=0.0001)

### Experiment 3
**Design**
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Balanced-minus-imbalanced change in choosing Expert 5's winner."""
    if data is None or len(data) == 0:
        return float("nan")

    interaction_scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 5 or b.size != a.size:
            continue

        diff = b - a
        signs = np.sign(diff)
        n_b = int(np.sum(signs > 0))
        n_a = int(np.sum(signs < 0))
        imbalance = abs(n_b - n_a)

        # The design's critical conditions are exact 2-vs-2 balance and
        # 3-vs-1 imbalance. Ignore any unexpected rows.
        if imbalance == 0:
            condition_sign = 1.0
        elif imbalance == 2:
            condition_sign = -1.0
        else:
            continue

        # Expert 5 is feature index 4 and always identifies the TTB winner.
        if diff[4] > 0:
            expert5_winner = 1
        elif diff[4] < 0:
            expert5_winner = 0
        else:
            continue

        chose_expert5_winner = float(int(row["response"]) == expert5_winner)
        # Averaging this score gives P(E5 winner | balanced) minus
        # P(E5 winner | imbalanced), since the schedule is balanced.
        interaction_scores.append(condition_sign * chose_expert5_winner)

    if len(interaction_scores) == 0:
        return float("nan")
    return float(np.mean(interaction_scores))
```

**Observed (real) value:** 0.0721 (var=0.0052)
**Previous candidate values (this loop):**
  - iter 1: 0.1690 (var=0.0046) (Δ vs real +0.0969)
  - iter 2: 0.1175 (var=0.0043) (Δ vs real +0.0454)
  - iter 3: 0.2496 (var=0.0078) (Δ vs real +0.1775)
  - iter 4: 0.1367 (var=0.0051) (Δ vs real +0.0646)
  - iter 5: 0.1427 (var=0.0042) (Δ vs real +0.0706)
  - iter 6: 0.1246 (var=0.0037) (Δ vs real +0.0525)
  - iter 7: 0.1119 (var=0.0049) (Δ vs real +0.0398)
  - iter 8: 0.1377 (var=0.0037) (Δ vs real +0.0656)
  - iter 9 (most recent): 0.1221 (var=0.0037) (Δ vs real +0.0500)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)
- pi_4: 0.0050 (var=0.0001)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        if len(df) == 0:
            return np.nan

        a = np.stack(df["option_a_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
        b = np.stack(df["option_b_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
        response = df["response"].to_numpy(dtype=int)

        # Expert 5 is uniquely most valid, so its favored option is the
        # Take-The-Best winner in this design.
        ttb_winner = (b[:, 4] > a[:, 4]).astype(int)
        chose_ttb_winner = (response == ttb_winner).astype(float)

        # Whether Expert 1 favors the same option as Expert 5.
        expert1_winner = (b[:, 0] > a[:, 0]).astype(int)
        expert1_supports = expert1_winner == ttb_winner

        if not np.any(expert1_supports) or not np.any(~expert1_supports):
            return np.nan

        crossover = (chose_ttb_winner[expert1_supports].mean() -
                     chose_ttb_winner[~expert1_supports].mean())

        # Classify a subject as showing a substantively sized primacy
        # crossover. The 0.12 margin suppresses chance sampling contrasts
        # around TTB's population prediction of exactly zero.
        return float(crossover > 0.12)

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        scores = [subject_score(g) for _, g in data.groupby("subject_id", sort=False)]
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        return float(scores.mean()) if scores.size else np.nan

    return subject_score(data)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 9 (most recent): 0.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0400 (var=0.0384)
- pi_2: 0.0600 (var=0.0564)
- pi_4: 0.0000 (var=0.0000)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices that follow the stable-order winner among the tied top cues."""
    if len(data) == 0:
        return float("nan")

    follows_top = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Feature 0 is consulted first under the communicated validity order.
        if a[0] > b[0]:
            top_choice = 0
        elif b[0] > a[0]:
            top_choice = 1
        else:
            continue

        follows_top.append(float(int(row["response"]) == top_choice))

    if len(follows_top) == 0:
        return float("nan")
    return float(np.mean(follows_top))
```

**Observed (real) value:** 0.7900 (var=0.0184)
**Previous candidate values (this loop):**
  - iter 1: 0.8702 (var=0.0036) (Δ vs real +0.0802)
  - iter 2: 0.8600 (var=0.0095) (Δ vs real +0.0700)
  - iter 3: 0.8677 (var=0.0145) (Δ vs real +0.0777)
  - iter 4: 0.8431 (var=0.0086) (Δ vs real +0.0531)
  - iter 5: 0.8131 (var=0.0098) (Δ vs real +0.0231)
  - iter 6: 0.8217 (var=0.0135) (Δ vs real +0.0317)
  - iter 7: 0.8204 (var=0.0090) (Δ vs real +0.0304)
  - iter 8: 0.8494 (var=0.0099) (Δ vs real +0.0594)
  - iter 9 (most recent): 0.8271 (var=0.0094) (Δ vs real +0.0371)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8244 (var=0.0081)
- pi_4: 0.0179 (var=0.0002)
- pi_2: 0.1456 (var=0.0074)
- pi_3: 0.6381 (var=0.0031)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Difference in adherence to the highest-validity cue between
    # one-opponent and multi-opponent coalitions.
    lone_top_choices = []
    multi_top_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Feature 0 is the uniquely highest-validity expert in this design.
        top_direction = np.sign(b[0] - a[0])  # +1 means B is top-cue winner
        if top_direction == 0:
            continue

        lower_directions = np.sign(b[1:] - a[1:])
        n_opponents = int(np.sum(lower_directions * top_direction < 0))
        chose_top_winner = float(
            int(row["response"]) == (1 if top_direction > 0 else 0)
        )

        if n_opponents == 1:
            lone_top_choices.append(chose_top_winner)
        elif n_opponents >= 2:
            multi_top_choices.append(chose_top_winner)

    # The fixed schedule supplies both classes for every subject. These
    # fallbacks keep the function scalar-valued on unexpected partial data.
    if len(lone_top_choices) == 0 or len(multi_top_choices) == 0:
        return 0.0

    return float(np.mean(lone_top_choices) - np.mean(multi_top_choices))
```

**Observed (real) value:** -0.5450 (var=0.0462)
**Previous candidate values (this loop):**
  - iter 1: -0.2756 (var=0.0125) (Δ vs real +0.2694)
  - iter 2: -0.4587 (var=0.0289) (Δ vs real +0.0863)
  - iter 3: -0.5737 (var=0.0539) (Δ vs real -0.0287)
  - iter 4: -0.5591 (var=0.0280) (Δ vs real -0.0141)
  - iter 5: -0.4987 (var=0.0334) (Δ vs real +0.0463)
  - iter 6: -0.4053 (var=0.0381) (Δ vs real +0.1397)
  - iter 7: -0.4953 (var=0.0383) (Δ vs real +0.0497)
  - iter 8: -0.5078 (var=0.0436) (Δ vs real +0.0372)
  - iter 9 (most recent): -0.4812 (var=0.0483) (Δ vs real +0.0638)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5559 (var=0.0104)
- pi_1: 0.0103 (var=0.0049)
- pi_2: 0.3781 (var=0.0114)
- pi_3: -0.3150 (var=0.0123)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Adaptive Distinctive Coalition Gating (ADCG) proposes that binary-feature decisions arise from task-adaptive cue weights combining communicated validity, presentation prominence, distributional rarity, and learned cue dependence. Directionally concordant cues are compressed as potentially redundant, while sparse dissent can dominate when its cues have reliably and repeatedly opposed the anchor's cues. Learned anti-concurrence is reliability-sensitive: disagreement near chance is strongly suppressed, whereas well-observed disagreement above a subject-specific threshold receives a bounded nonlinear gain. A profile-driven gate interpolates between coalition comparison and a lexicographic anchor, preserving anchoring against large redundant coalitions while permitting diagnostic opposition to suppress it. For compact opposition, gate suppression is controlled by a selective summary of the strongest reliably learned opposing relationship rather than an indiscriminate average across all cue pairs; larger coalitions retain mean aggregation and strong size decay. In addition, a unique, early-presented validity anchor loses accessibility within the coalition core when it faces compact, reliably anti-concurrent opposition. This attenuation targets residual anchoring without increasing dissent support and vanishes for tied-validity anchors, late anchors, and large opposing coalitions. When top validities are tied, presentation order supplies a bounded anchor but the distinctive-coalition core is attenuated by a stable subject-specific amount. Stable variation in validity attention, order prominence, anti-concurrence threshold and gain, diagnostic selectivity, anchor attenuation, tied-top attenuation, redundancy, gating, and temperature produces structured individual differences without experiment-specific strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"ADCG expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive directions favor B; negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_attention = float(parameters["validity_attention"])
    order_prominence = float(parameters["order_prominence"])
    distribution_attention = float(parameters["distribution_attention"])
    distinctiveness = float(parameters["distinctiveness"])
    dissent_bonus = float(parameters["dissent_bonus"])
    anti_concurrence_attention = float(parameters["anti_concurrence_attention"])
    anti_concurrence_threshold = float(parameters["anti_concurrence_threshold"])
    diagnostic_selectivity = float(parameters["diagnostic_selectivity"])
    anchor_attenuation = float(parameters["anchor_attenuation"])
    redundancy = float(parameters["redundancy"])
    lex_strength = float(parameters["lex_strength"])
    gate_sharpness = float(parameters["gate_sharpness"])
    gate_rarity = float(parameters["gate_rarity"])
    tie_gate_ceiling = float(parameters["tie_gate_ceiling"])
    tie_core_attenuation = float(parameters["tie_core_attenuation"])
    beta = float(parameters["beta"])

    # Communicated validity is represented on a reliability-evidence scale.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    reliability = (reliability - np.mean(reliability)) / (
        np.std(reliability) + 0.35
    )

    # Estimate how often each cue discriminates and how often cue pairs
    # concur. Symmetric pseudocounts make the model well-defined early in
    # the block while allowing randomized task experience to update weights.
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    discr_count = np.ones(n_features, dtype=float) * 2.0
    pair_both = np.ones((n_features, n_features), dtype=float) * 2.0
    pair_same = np.ones((n_features, n_features), dtype=float)

    for i in range(n_past):
        pa = np.asarray(past_a[i], dtype=float)
        pb = np.asarray(past_b[i], dtype=float)
        if pa.shape[0] != n_features or pb.shape[0] != n_features:
            continue
        s = np.sign(pb - pa)
        on = np.flatnonzero(s != 0)
        discr_count[on] += 1.0
        for q, j in enumerate(on):
            for k in on[q + 1:]:
                pair_both[j, k] += 1.0
                pair_both[k, j] += 1.0
                if s[j] == s[k]:
                    pair_same[j, k] += 1.0
                    pair_same[k, j] += 1.0

    discr_rate = discr_count / float(n_past + 4.0)
    reference_rate = float(np.mean(discr_rate))
    rarity = np.clip(reference_rate / discr_rate, 0.35, 3.5)

    # Learned anti-concurrence is high only for cue pairs that repeatedly
    # discriminate in opposite directions. Pair-specific observation counts
    # determine reliability, and a steep subject-specific threshold removes
    # weak or chance-level disagreement while retaining systematic dissent.
    anti_concurrence = np.clip(1.0 - pair_same / pair_both, 0.0, 1.0)
    pair_observations = np.maximum(pair_both - 2.0, 0.0)
    pair_confidence = pair_observations / (pair_observations + 2.0)
    pair_confidence *= np.clip((pair_observations - 1.0) / 3.0, 0.0, 1.0)
    threshold_baseline = 1.0 / (
        1.0 + np.exp(-18.0 * (0.5 - anti_concurrence_threshold))
    )
    thresholded_anti = 1.0 / (
        1.0 + np.exp(-18.0 * (anti_concurrence - anti_concurrence_threshold))
    )
    thresholded_anti = np.clip(
        (thresholded_anti - threshold_baseline)
        / max(1.0 - threshold_baseline, 1e-12),
        0.0,
        1.0,
    )
    learned_anti = pair_confidence * thresholded_anti

    # Baseline cue accessibility combines instructed validity, serial
    # prominence, and learned distributional rarity.
    if n_features > 1:
        position = 1.0 - np.arange(n_features, dtype=float) / (n_features - 1.0)
    else:
        position = np.ones(1, dtype=float)
    cue_weight = np.exp(
        validity_attention * reliability
        + order_prominence * position
        + distribution_attention * np.log(rarity)
    )

    # Cues that historically concur with their current coalition partners
    # are treated as less independent. A cue with no partner is not given
    # this historical redundancy penalty.
    for j in active:
        partners = [
            int(k) for k in active
            if k != j and directions[k] == directions[j]
        ]
        if partners:
            concurrence = [pair_same[j, k] / pair_both[j, k] for k in partners]
            corr_load = float(np.mean(concurrence))
            cue_weight[j] /= 1.0 + redundancy * corr_load

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    # Lexicographic anchor: highest communicated validity among currently
    # discriminating cues, with stable presentation order breaking ties.
    active_validity = validities[active]
    best_validity = float(np.max(active_validity))
    best_candidates = active[np.isclose(active_validity, best_validity, atol=1e-10)]
    lex_cue = int(np.min(best_candidates))
    lex_direction = float(directions[lex_cue])

    # A global top-validity tie attenuates profile-based singleton
    # amplification, preventing the first cue from receiving both a strong
    # order anchor and a full distinctiveness bonus.
    sorted_v = np.sort(validities)[::-1]
    if sorted_v.size >= 2:
        top_gap = float(sorted_v[0] - sorted_v[1])
        global_top_tie = float(np.exp(-80.0 * top_gap))
    else:
        global_top_tie = 1.0
    tie_core_multiplier = 1.0 - tie_core_attenuation * global_top_tie

    # Current-profile distinctiveness: members of the smaller directional
    # coalition stand out, while each coalition's accumulated contribution
    # is divided by a redundancy term.
    if n_a > 0 and n_b > 0:
        if n_a < n_b:
            cue_weight[a_cues] *= (float(n_b) / float(n_a)) ** (
                distinctiveness * tie_core_multiplier
            )
        elif n_b < n_a:
            cue_weight[b_cues] *= (float(n_a) / float(n_b)) ** (
                distinctiveness * tie_core_multiplier
            )

        # Sparse evidence opposing the validity-defined anchor receives its
        # ordinary bounded bonus. Additional influence is restricted to
        # dissenting cues with reliably learned above-threshold opposition.
        dissent_cues = a_cues if lex_direction > 0 else b_cues
        anchor_cues = b_cues if lex_direction > 0 else a_cues
        n_dissent = int(dissent_cues.size)
        if n_dissent > 0:
            for j in dissent_cues:
                learned_diagnosticity = 0.0
                if anchor_cues.size > 0:
                    learned_diagnosticity = float(
                        np.mean(learned_anti[j, anchor_cues])
                    )
                bonus = 1.0 + (dissent_bonus / float(n_dissent)) * (
                    1.0
                    + anti_concurrence_attention * learned_diagnosticity
                )
                cue_weight[j] *= bonus

            # A unique and presentation-prominent validity anchor becomes
            # less accessible in the coalition core when compact opposition
            # has a reliably learned anti-concurrent relationship with it.
            # Active-cue validity uniqueness leaves tied-top profiles intact;
            # squared prominence and size decay protect late anchors and
            # anchors facing large, plausibly redundant coalitions.
            if active_validity.size >= 2:
                sorted_active_v = np.sort(active_validity)[::-1]
                active_gap = float(sorted_active_v[0] - sorted_active_v[1])
                anchor_uniqueness = 1.0 - np.exp(-80.0 * active_gap)
            else:
                anchor_uniqueness = 1.0
            anchor_diagnosticity = float(
                np.max(learned_anti[dissent_cues, lex_cue])
            )
            compact_decay = np.exp(-1.5 * float(n_dissent - 1))
            anchor_signal = (
                anchor_uniqueness
                * float(position[lex_cue] ** 2)
                * anchor_diagnosticity
                * compact_decay
            )
            cue_weight[lex_cue] *= np.exp(
                -anchor_attenuation * anchor_signal
            )

    support_a = 0.0
    support_b = 0.0
    if n_a > 0:
        support_a = float(np.sum(cue_weight[a_cues])) / (float(n_a) ** redundancy)
    if n_b > 0:
        support_b = float(np.sum(cue_weight[b_cues])) / (float(n_b) ** redundancy)

    scale = support_a + support_b
    core_evidence = (support_b - support_a) / max(scale, 1e-12)

    # A near tie among the two globally strongest instructed validities
    # raises the stable-order gate, but anchoring has a subject-specific
    # ceiling rather than becoming obligatorily lexicographic.
    tied_top_gate = tie_gate_ceiling * global_top_tie

    # Profile-driven anchoring occurs when the earliest discriminating cue
    # is on an isolated minority side. It is attenuated when a sparse cue in
    # the opposing coalition is historically rare or reliably
    # anti-concurrent, while a large opposition remains interpretable as
    # redundant evidence.
    first_cue = int(active[0])
    first_side_count = n_b if directions[first_cue] > 0 else n_a
    other_side_count = n_a if directions[first_cue] > 0 else n_b
    if first_side_count > 0 and other_side_count > 0:
        isolation = 1.0 - float(first_side_count) / float(other_side_count)
        minority_first = float(first_side_count < other_side_count)
        opposing_cues = a_cues if directions[first_cue] > 0 else b_cues
        first_side_cues = b_cues if directions[first_cue] > 0 else a_cues
        opposing_rarity = float(np.max(np.maximum(np.log(rarity[opposing_cues]), 0.0)))
        opposing_distinctiveness = opposing_rarity / (
            np.log(3.5) * np.sqrt(float(other_side_count))
        )

        # For compact opposition, a reliability-weighted soft maximum lets
        # one strongly diagnostic relation disengage the anchor without
        # being diluted by weak pairings. Larger coalitions retain mean
        # aggregation, and all configurations retain exponential size decay.
        opposition_relations = learned_anti[
            np.ix_(opposing_cues, first_side_cues)
        ].reshape(-1)
        if other_side_count <= 3 and opposition_relations.size > 1:
            select_logits = diagnostic_selectivity * (
                opposition_relations - np.max(opposition_relations)
            )
            select_weights = np.exp(select_logits)
            select_weights /= np.sum(select_weights)
            learned_opposition = float(
                np.sum(select_weights * opposition_relations)
            )
        else:
            learned_opposition = float(np.mean(opposition_relations))
        learned_opposition *= np.exp(-1.5 * float(other_side_count - 1))
        opposing_distinctiveness += (
            anti_concurrence_attention * learned_opposition
        )
    else:
        isolation = 0.0
        minority_first = 0.0
        opposing_distinctiveness = 0.0

    profile_gate = minority_first / (
        1.0 + np.exp(-gate_sharpness * (isolation - 0.30))
    )
    profile_gate *= np.exp(-gate_rarity * opposing_distinctiveness)
    gate = 1.0 - (1.0 - tied_top_gate) * (1.0 - profile_gate)
    gate = float(np.clip(gate, 0.0, 1.0))

    choice_evidence = (
        (1.0 - gate) * core_evidence
        + gate * lex_strength * lex_direction
    )

    logits = beta * np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs.astype(np.float64)

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- validity_attention: [0.2, 0.9]
- order_prominence: [0.15, 0.7]
- distribution_attention: [0.8, 1.8]
- distinctiveness: [0.35, 0.9]
- dissent_bonus: [1.0, 2.2]
- anti_concurrence_attention: [1.0, 4.8]
- anti_concurrence_threshold: [0.62, 0.76]
- diagnostic_selectivity: [2.0, 8.0]
- anchor_attenuation: [0.8, 2.4]
- redundancy: [0.75, 1.25]
- lex_strength: [0.7, 1.4]
- gate_sharpness: [4.0, 11.0]
- gate_rarity: [1.0, 4.0]
- tie_gate_ceiling: [0.55, 0.85]
- tie_core_attenuation: [0.7, 0.95]
- beta: [0.6, 3.4]

`rationale`: This is a minimal edit to the accepted iteration-7 ADCG model. The successful gate-only soft maximum and the original mean-based direct dissent bonus are retained exactly; the rejected iteration-8 extension is not repeated. The sole mechanism added is localized attenuation of the lexicographic anchor's own coalition-core weight. Attenuation requires four observable conditions jointly: the anchor must be uniquely best among currently discriminating cues, presentation-prominent, opposed by a cue with high-confidence thresholded anti-concurrence, and faced by compact rather than large opposition. Its exponential coalition-size decay preserves the large-coalition anchoring responsible for Experiment 1, while squared positional prominence minimizes effects on the late Expert-5 anchor implicated in Experiment 3. Active-validity uniqueness makes the attenuation vanish when leading validities are tied, protecting Experiment 5. The new bounded anchor_attenuation trait supplies targeted subject variation where diagnostic compact opposition matters, especially Experiments 2 and 6, without broad changes to temperature, redundancy, dissent support, or tied-top gating.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: ADCG is faithful to the prescribed distinctiveness-and-redundancy family and captures several important qualitative results: Experiment 1 is exact, Experiment 4 is exact on its coarse prevalence measure, and Experiment 6 has the correct negative sign. However, the aggregate fit is not yet sufficient. The largest failure is Experiment 2, where highest-validity-cue adherence remains 0.6498 rather than the observed 0.3242; the model therefore still anchors too strongly and does not allow sparse dissent to reverse choices often enough. Experiment 6 similarly has only about half the observed contrast magnitude (-0.2756 versus -0.5450). Experiment 3 overshoots the balanced-minus-imbalanced effect (0.1690 versus 0.0721), suggesting that generic coalition-size compression is too strong when applied indiscriminately. Experiment 5 also overpredicts stable-order adherence (0.8702 versus 0.7900). Across Experiments 2, 5, and 6, simulated between-subject variance is much too low, despite the theory's stated commitment to stable heterogeneity.
Rationale: Keep ADCG's mechanism family, but recalibrate its gate and heterogeneity. The current profile gate automatically increases lexicographic anchoring whenever the earliest cue is an isolated minority. That rule helps Experiments 1 and 6 but also protects the highest-validity cue in Experiment 2, directly opposing the required crossover. Make this gate sensitive to the observable distinctiveness of the opposing coalition: minority-first anchoring should remain strong when the opposition is a large, redundant coalition, but should be suppressed when the opposition contains a historically rare or otherwise distinctive sparse cue. This should lower Experiment 2 adherence while preserving Experiment 1 and strengthening the intended contrast between sparse and redundant opposition in Experiment 6. Avoid a blanket increase in distinctiveness or redundancy, because Experiment 3 already shows excessive imbalance sensitivity; instead, reduce generic coalition-size compression there and transfer more of the effect to the rarity-conditioned gate. Soften the tied-top gate or lower its effective lexicographic ceiling to bring Experiment 5 from 0.87 toward 0.79. Finally, widen subject-level ranges or add bounded subject-level variation in the gate ceiling/threshold and choice temperature: current variances are roughly three- to fivefold too small in Experiments 2, 5, and 6. These are in-family adjustments to gating, normalization, and parameter ranges rather than a change of theory.

**Outcome of this advice:** iter 1 candidate loss=0.1680 -> iter 2 candidate loss=0.1000 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision is a substantial improvement and remains faithful to the prescribed task-adaptive distinctiveness-and-redundancy family. It exactly captures Experiments 1 and 4, moves Experiment 3 much closer to the observed effect (0.1175 versus 0.0721), and reproduces most of the large negative contrast in Experiment 6 (-0.4587 versus -0.5450). The previous advice to transfer influence from generic coalition compression to rarity-conditioned sparse dissent was therefore productive. Nevertheless, the central crossover remains too weak in Experiment 2: highest-validity adherence is 0.5542 versus 0.3242, leaving the model above chance when humans are substantially below chance. Experiment 5 also remains too lexicographic (0.8600 versus 0.7900). Stable heterogeneity improved, especially in Experiments 5 and 6, but variance is still understated in Experiment 2 (0.0041 versus 0.0174), Experiment 5 (0.0095 versus 0.0184), and Experiment 6 (0.0289 versus 0.0462).
Rationale: Continue the successful in-family direction, but make the targeted sparse-dissent mechanism more selective and stronger rather than increasing generic distinctiveness or redundancy. The current multiplier, 1 + dissent_bonus/n_dissent, still awards appreciable bonuses to larger dissenting coalitions and evidently does not reverse enough Experiment 2 choices. Use a more sharply singleton-focused profile function—such as a steep decay after one dissenting cue—or allow a wider upper range for the singleton bonus. Couple this to stronger suppression of the lexicographic gate whenever the anchor faces a sparse, profile-distinctive opponent; do not require historical rarity alone, because early-block pseudocounts and limited exposure may make that signal too weak. Observable current-profile sparsity can enter the same gate while remaining experiment-invariant. This adjustment is also directionally appropriate for the residual Experiment 6 error. At the same time, slightly reduce generic coalition-size compression again to bring Experiment 3 from 0.1175 toward 0.0721 and avoid using that generic term to solve Experiment 2. Lower the tied-top gate ceiling modestly to reduce Experiment 5 adherence toward 0.79. Finally, widen stable subject-level variation specifically in dissent sensitivity, gate threshold/ceiling, and temperature, while preserving their correlation if needed, because independent broad noise may disturb the excellent Experiment 1 fit without producing the required structured variance. Preserve the current strong minority-first anchoring for large redundant opposition, which is successfully maintaining Experiment 1.

**Outcome of this advice:** iter 2 candidate loss=0.1000 -> iter 3 candidate loss=0.1293 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed task-adaptive distinctiveness-and-redundancy family, and it preserves the exact fits in Experiments 1 and 4. It also brings Experiment 6 very close to its observed negative contrast (-0.5737 versus -0.5450), with approximately appropriate variance. However, the iteration was rejected because its sharper singleton mechanism worsened aggregate fit. Experiment 2 remains the central failure: highest-validity adherence is still above chance at 0.5708, whereas humans are substantially below chance at 0.3242, and its simulated variance remains far too low (0.0051 versus 0.0174). More decisively, Experiment 3 deteriorated from the accepted base's 0.1175 to 0.2496 against an observed 0.0721. Thus, sharply boosting every current-profile singleton is not selective enough: singleton configurations in Experiment 3 are being treated like the diagnostic dissent in Experiments 2 and 6. Experiment 5 also remains too order-lexicographic (0.8677 versus 0.7900), despite the reduced tied-top ceiling.
Rationale: Because this iteration was rejected, build from the unchanged iteration-2 base rather than continuing the rejected exponential-singleton modification. The previous recommendation to strengthen and sharpen current-profile singleton dissent did not help; do not repeat it or broaden the singleton bonus further. Instead, use the distributional component already required by ADCG to distinguish diagnostic dissent from an incidental singleton. A promising in-family adjustment is a bounded cue-level distinctiveness statistic based on how unusually often each cue disagrees with, or avoids concurrence with, the other active cues across the task history. Let sparse opposition receive extra weight or suppress the lexicographic gate only when the dissenting cue also has high learned anti-concurrence/distinctiveness. This should target the systematic crossover structure of Experiment 2 without producing the indiscriminate singleton amplification that inflated Experiment 3. Use smoothed block-history estimates and cap the effect early in the block so pseudocount noise does not dominate. Keep the accepted base's generic coalition compression and singleton function near their iteration-2 settings; the current stronger form overshot Experiment 6 and badly harmed Experiment 3. For Experiment 5, inspect the complete choice-evidence decomposition rather than lowering the tied-top ceiling again: the first cue is also a singleton against a coalition, so the distinctive-coalition core may continue favoring it even after gate reduction. Attenuate the core singleton bonus when the top communicated validities are tied, while retaining a bounded order gate, so tied-top adherence can move from 0.86 toward 0.79 without abolishing the required anchoring. Finally, attach stable subject variation to the new anti-concurrence sensitivity and correlate it with validity attention or gate suppression. Merely widening independent ranges failed to generate Experiment 2's observed variance and risks disrupting the exact Experiment 1 fit.

**Outcome of this advice:** iter 3 candidate loss=0.1293 -> iter 4 candidate loss=0.0727 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The accepted revision is the best candidate so far and supports the prescribed task-adaptive distinctiveness-and-redundancy family. The learned anti-concurrence adjustment was productive: Experiment 2 moved from 0.5542 to 0.4777, Experiment 5 improved from 0.8600 to 0.8431, Experiment 6 is now nearly exact (-0.5591 versus -0.5450), and the exact results in Experiments 1 and 4 were preserved. Nevertheless, the central qualitative result remains uncaptured. In Experiment 2, humans follow the highest-validity discriminating cue substantially below chance (0.3242), whereas the candidate remains approximately at chance and 0.1535 too high. Experiment 3 also remains too sensitive to coalition balance (0.1367 versus 0.0721), and Experiment 5 remains overly order-anchored (0.8431 versus 0.7900). Between-subject variance is still markedly understated in Experiment 2 (0.0049 versus 0.0174), Experiment 5 (0.0086 versus 0.0184), and Experiment 6 (0.0280 versus 0.0462). Thus, the model has improved substantially but does not yet reproduce the experiment-invariant signed pattern, especially the negative crossover in Experiment 2.
Rationale: Continue from this accepted anti-concurrence-based ADCG candidate, since the prior recommendation clearly helped, but make the learned diagnosticity effect more selective rather than simply increasing all singleton or anti-concurrence bonuses. The present linear threshold at anti_rate > 0.5 likely gives weak influence to many moderately opposing pairs, contributing to Experiment 3, while failing to produce sufficiently strong reversals for the most systematic dissent structure in Experiment 2. Replace it with a reliability-sensitive, steeper thresholded mapping: strongly shrink anti-concurrence near chance, but permit a larger bounded effect for cue pairs whose smoothed anti-concurrence is clearly and repeatedly above a higher threshold. Confidence could rise somewhat faster with informative pair observations rather than total trial count, while requiring a minimum effective pair count. Apply the resulting statistic consistently to both dissent weighting and suppression of the lexicographic gate, retaining the existing coalition-size decay that protects Experiment 1. This is preferable to repeating the rejected generic singleton sharpening. For Experiment 5, modestly strengthen the accepted tied-validity attenuation of the distinctive core, rather than again relying only on a lower tie-gate ceiling; the residual excess indicates that both the core and gate still favor the first cue. Preserve a bounded order gate so adherence remains well above chance. Finally, introduce structured subject heterogeneity through the threshold or gain of anti-concurrence sensitivity and tied-top attenuation, preferably correlated with validity attention, rather than broadly widening independent noise parameters. This should raise variance where strategy sensitivity matters without sacrificing the exact Experiment 1 result or the now excellent Experiment 6 fit.

**Outcome of this advice:** iter 4 candidate loss=0.0727 -> iter 5 candidate loss=0.0672 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current ADCG revision is faithful to the prescribed distinctiveness-and-redundancy family and is the best accepted candidate so far. It preserves exact fits in Experiments 1 and 4, brings Experiment 5 close to the observed stable-order adherence (0.8131 versus 0.7900), and retains the correct strong negative contrast in Experiment 6 (-0.4987 versus -0.5450). The reliability-sensitive anti-concurrence revision therefore helped overall, consistent with its accepted gate outcome. However, the principal signed result remains too weak: Experiment 2 highest-validity adherence is only slightly below chance at 0.4531, whereas humans show a much stronger reversal at 0.3242. Experiment 3 still overstates the balance effect (0.1427 versus 0.0721). Moreover, compared with iteration 4, Experiment 6 moved away from its nearly exact value, suggesting that stricter filtering removed some useful compact-opposition influence. Subject variance also remains substantially too low in Experiments 2 (0.0035 versus 0.0174) and 5 (0.0098 versus 0.0184).
Rationale: Retain the accepted thresholded anti-concurrence representation, but do not tighten its threshold further or return to generic singleton amplification; the former has begun weakening Experiment 6, while the latter was previously rejected and badly inflated Experiment 3. Instead, rebalance where high-confidence anti-concurrence acts. Its effect should be stronger in suppressing the lexicographic gate for compact, reliably opposing configurations, but weaker or more tightly capped in the direct multiplicative dissent-weight bonus. Currently one shared anti_concurrence_attention parameter drives both pathways. Splitting this into bounded gate-suppression and cue-weight gains, or applying separate normalizations, would remain fully within ADCG while allowing Experiment 2 and the negative Experiment 6 contrast to strengthen without magnifying Experiment 3's coalition-balance effect. Preserve the existing exponential protection against large opposing coalitions so Experiment 1 remains exact. Experiment 5 now needs only conservative tuning; keep the current mean tied-top attenuation rather than pushing it much further. To address variance, widen the subject-level range of the compact-opposition gate gain around its current mean and modestly widen tied-top attenuation around its current mean, rather than broadening generic distinctiveness or lapse-like temperature. This should generate structured heterogeneity specifically in Experiments 2, 5, and 6 while protecting their pooled means.

**Outcome of this advice:** iter 5 candidate loss=0.0672 -> iter 6 candidate loss=0.0867 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed task-adaptive distinctiveness-and-redundancy family and preserves the exact metric fits in Experiments 1 and 4. Experiment 5 is also reasonably close (0.8217 versus 0.7900), and Experiment 3 improves modestly relative to the accepted iteration-5 base (0.1246 versus 0.1427, observed 0.0721). Nevertheless, the current edit was rejected and worsened the two principal compact-opposition results: Experiment 2 rose from 0.4531 to 0.4890 rather than approaching the observed below-chance value of 0.3242, and Experiment 6 weakened from -0.4987 to -0.4053 rather than approaching -0.5450. Thus, reducing direct diagnostic weighting while broadly widening gate suppression removed useful anti-concurrence influence instead of reallocating it successfully. Experiment 2 also continues to show much too little between-subject variance (0.0052 versus 0.0174).
Rationale: Build from the unchanged accepted iteration-5 candidate. My previous recommendation to split the shared anti-concurrence parameter into a weak direct-weight gain and a much wider gate gain was rejected, so do not repeat that broad reallocation. Restore the accepted coupling or its iteration-5 direct diagnostic strength. Then try a different, structurally selective in-family adjustment: change how learned diagnostic opposition is aggregated for gate suppression. The current mean across all opposing-by-anchor cue pairs can dilute one highly reliable diagnostic relationship with several weak pairs. Use a bounded reliability-weighted maximum, top-k average, or softmax aggregation for compact opposition, while retaining mean aggregation or strong size decay for larger coalitions. This can make a genuinely diagnostic sparse opponent disengage anchoring in Experiments 2 and 6 without globally increasing direct singleton weights or generic redundancy, both of which risk inflating Experiment 3. Keep the accepted iteration-5 tied-top settings because Experiment 5 is already close, and avoid further broad gain-range widening. For the deficient Experiment-2 variance, place modest stable heterogeneity in the softmax/top-k selectivity or diagnosticity threshold—not the overall gate gain—so subjects differ in whether a strongest learned opposition relation controls the gate while the pooled mean remains calibrated. Preserve the large-coalition protection responsible for the exact Experiment-1 fit.

**Outcome of this advice:** iter 6 candidate loss=0.0867 -> iter 7 candidate loss=0.0628 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: The current candidate is the best accepted ADCG instantiation and remains faithful to the prescribed task-adaptive distinctiveness-and-redundancy family. The selective aggregation change was productive overall: aggregate loss fell from 0.0672 to 0.0628, Experiment 3 improved to 0.1119 versus 0.0721, Experiment 6 remains reasonably close at -0.4953 versus -0.5450, and the exact results in Experiments 1 and 4 were preserved. Experiment 5 is also close, though still somewhat too order-anchored at 0.8204 versus 0.7900. However, the principal quantitative failure remains Experiment 2. Highest-validity adherence is 0.4648 rather than 0.3242, so ADCG produces only a weak below-chance reversal where humans show a strong one. Its between-subject variance is also only 0.0060 versus 0.0174. Thus the model identifies the correct qualitative mechanism but still lets communicated validity and anchoring survive too often in compact, diagnostically opposing profiles. Experiment 3's residual overprediction indicates that this should not be repaired through generic distinctiveness or coalition compression.
Rationale: Build on this accepted candidate and retain the selective soft-maximum gate aggregation, since the previous advice produced a new best loss. Do not return to the rejected broad split between direct and gate anti-concurrence gains, and do not amplify all singleton dissent. Instead, make a modest extension of the successful diagnostic-selectivity mechanism. First, allow strongly selective aggregation to affect the direct dissent diagnosticity for compact opposition as well as gate suppression, while keeping a tight cap and the existing coalition-size decay. At present the gate uses a soft maximum, but the direct bonus still averages learned anti-concurrence across anchor cues; that mean can leave the coalition core favoring the anchor even after its gate is suppressed. A bounded softmax or top-k summary in this direct term should produce more actual reversals in Experiment 2 without recreating the indiscriminate singleton effect that damaged Experiment 3. Second, inspect the observation-confidence bottleneck: with only 12 trials, the current multiplicative confidence schedule can delay diagnostic learning until much of the block has passed. Let confidence saturate slightly faster only for compact relations that are consistently above the existing high anti-concurrence threshold; do not lower the threshold generally. A modest increase in the upper range of diagnostic selectivity, rather than broad gate-gain widening, may also lower the Experiment 2 mean and increase its deficient variance. Keep generic redundancy, tied-top attenuation, and large-coalition protection essentially fixed, because Experiments 1, 5, and 6 are already close and Experiment 3 still needs less—not more—generic balance sensitivity.

**Outcome of this advice:** iter 7 candidate loss=0.0628 -> iter 8 candidate loss=0.0675 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: The current ADCG remains faithful to the prescribed distinctiveness-and-redundancy family and preserves exact fits in Experiments 1 and 4. It also improves the targeted means relative to iteration 7 in Experiment 2 (0.4371 versus 0.4648) and Experiment 6 (-0.5078 versus -0.4953). Nevertheless, the accept gate rejected this extension because those gains were outweighed by collateral damage: Experiment 3 increased from 0.1119 to 0.1377 against an observed 0.0721, and Experiment 5 increased from 0.8204 to 0.8494 against 0.7900. The bounded direct soft-maximum therefore spreads diagnostic amplification into profiles where it increases unwanted coalition-balance and order effects. Moreover, Experiment 2 remains materially too high at 0.4371 versus 0.3242, with much too little between-subject variance (0.0053 versus 0.0174). Thus, even the rejected candidate still does not reproduce the strong human reversal.
Rationale: Build again from the unchanged iteration-7 accepted base. My previous recommendation to extend selective anti-concurrence aggregation from the gate into the direct dissent bonus was rejected, so do not repeat that extension or further accelerate its confidence schedule. Retain the successful gate-only soft maximum and restore the iteration-7 direct mean diagnosticity. Try a different targeted in-family adjustment: when a unique, presentation-prominent validity anchor faces compact opposition with high-confidence learned anti-concurrence, attenuate that anchor cue's validity/order accessibility in the coalition core rather than adding more weight to the dissenting cues. Scale this attenuation continuously by observable validity-gap uniqueness, anchor position, learned diagnosticity, and the existing opposition-size decay. This should facilitate actual reversals in Experiments 2 and 6 while leaving tied-top profiles in Experiment 5 largely untouched and avoiding the generic coalition amplification that worsened Experiment 3. Large opposition must remain protected by strong size decay to preserve Experiment 1. Use only modest subject variation in the attenuation threshold or strength to address Experiment 2's deficient variance; do not broadly widen temperature, redundancy, or direct dissent ranges. The objective is a localized reduction of residual anchor accessibility, not another increase in singleton or coalition support.

**Outcome of this advice:** iter 8 candidate loss=0.0675 -> iter 9 candidate loss=0.0503 -> the gate marked it ACCEPTED.

### Iteration 9 (most recent — address this)
Verdict: regenerate
Interpretation: The accepted anchor-accessibility attenuation is a clear improvement and remains faithful to the prescribed task-adaptive distinctiveness-and-redundancy family. It reduced aggregate loss from 0.0628 to 0.0503 and moved the central Experiment 2 result from 0.4648 to 0.3877 while preserving exact fits in Experiments 1 and 4. The candidate now captures the correct signs across the critical experiments, including below-chance highest-validity adherence in Experiment 2 and the strong negative coalition contrast in Experiment 6. However, several residual errors remain aligned rather than random. Experiment 2 still follows the highest-validity cue too often (0.3877 versus 0.3242), and its between-subject variance remains less than half the observed value (0.0071 versus 0.0174). Experiment 6 is also insufficiently negative (-0.4812 versus -0.5450), although its variance is now excellent. Experiment 3 continues to overstate the balance effect (0.1221 versus 0.0721), while Experiment 5 remains somewhat too order-anchored (0.8271 versus 0.7900) and under-variable. Thus, the model has captured the qualitative experiment-invariant pattern, but residual anchor accessibility remains slightly too strong in compact diagnostic opposition, while generic coalition-balance sensitivity remains too large.
Rationale: Continue from this accepted candidate and retain the localized anchor attenuation; the previous recommendation was accepted and produced the largest recent improvement. Make only a modest, targeted extension rather than adding direct dissent weight or returning to generic singleton amplification, both of which were previously rejected. Slightly widen the upper range of anchor_attenuation, or make its nonlinear effect steeper only when opposition is singleton/compact and learned anti-concurrence is already high-confidence. This should lower Experiment 2 adherence and strengthen Experiment 6's negative contrast without changing large-coalition anchoring in Experiment 1. Preserve the validity-gap, positional-prominence, tied-top exclusion, and coalition-size protections so the adjustment does not spread into Experiment 5 or the late Expert-5 structure of Experiment 3. Use the added attenuation range as stable subject-level heterogeneity to increase Experiment 2 variance, rather than widening temperature or generic distinctiveness. Separately, apply a very small reduction in generic coalition-balance sensitivity—preferably through a softer coalition-size normalization near balanced 2-vs-2 and 3-vs-1 profiles—to move Experiment 3 toward 0.072 without weakening the diagnostic anti-concurrence pathway. Keep the tied-top gate and core attenuation essentially fixed: Experiment 5's residual error is modest, and prior broader changes to tied-top or direct diagnostic amplification worsened its fit.

**Outcome of this advice:** iter 9 candidate loss=0.0503 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Adaptive Distinctive Coalition Gating (ADCG) proposes that binary-feature decisions arise from task-adaptive cue weights combining communicated validity, presentation prominence, distributional rarity, and learned cue dependence. Directionally concordant cues are compressed as potentially redundant, while sparse dissent can dominate when its cues have reliably and repeatedly opposed the anchor's cues. Learned anti-concurrence is reliability-sensitive: disagreement near chance is strongly suppressed, whereas well-observed disagreement above a subject-specific threshold receives a bounded nonlinear gain. A profile-driven gate interpolates between coalition comparison and a lexicographic anchor, preserving anchoring against large redundant coalitions while permitting diagnostic opposition to suppress it. For compact opposition, gate suppression is controlled by a selective summary of the strongest reliably learned opposing relationship rather than an indiscriminate average across all cue pairs; larger coalitions retain mean aggregation and strong size decay. In addition, a unique, early-presented validity anchor loses accessibility within the coalition core when it faces compact, reliably anti-concurrent opposition. This attenuation targets residual anchoring without increasing dissent support and vanishes for tied-validity anchors, late anchors, and large opposing coalitions. When top validities are tied, presentation order supplies a bounded anchor but the distinctive-coalition core is attenuated by a stable subject-specific amount. Stable variation in validity attention, order prominence, anti-concurrence threshold and gain, diagnostic selectivity, anchor attenuation, tied-top attenuation, redundancy, gating, and temperature produces structured individual differences without experiment-specific strategies.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"ADCG expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {validities.shape[0]} != n_features {n_features}.\"\n        )\n\n    # Positive directions favor B; negative directions favor A.\n    directions = np.sign(stim[1] - stim[0])\n    active = np.flatnonzero(directions != 0)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    validity_attention = float(parameters[\"validity_attention\"])\n    order_prominence = float(parameters[\"order_prominence\"])\n    distribution_attention = float(parameters[\"distribution_attention\"])\n    distinctiveness = float(parameters[\"distinctiveness\"])\n    dissent_bonus = float(parameters[\"dissent_bonus\"])\n    anti_concurrence_attention = float(parameters[\"anti_concurrence_attention\"])\n    anti_concurrence_threshold = float(parameters[\"anti_concurrence_threshold\"])\n    diagnostic_selectivity = float(parameters[\"diagnostic_selectivity\"])\n    anchor_attenuation = float(parameters[\"anchor_attenuation\"])\n    redundancy = float(parameters[\"redundancy\"])\n    lex_strength = float(parameters[\"lex_strength\"])\n    gate_sharpness = float(parameters[\"gate_sharpness\"])\n    gate_rarity = float(parameters[\"gate_rarity\"])\n    tie_gate_ceiling = float(parameters[\"tie_gate_ceiling\"])\n    tie_core_attenuation = float(parameters[\"tie_core_attenuation\"])\n    beta = float(parameters[\"beta\"])\n\n    # Communicated validity is represented on a reliability-evidence scale.\n    v = np.clip(validities, 0.500001, 0.999999)\n    reliability = np.log(v / (1.0 - v))\n    reliability = (reliability - np.mean(reliability)) / (\n        np.std(reliability) + 0.35\n    )\n\n    # Estimate how often each cue discriminates and how often cue pairs\n    # concur. Symmetric pseudocounts make the model well-defined early in\n    # the block while allowing randomized task experience to update weights.\n    past_a = history.get(\"option_a_ratings\", [])\n    past_b = history.get(\"option_b_ratings\", [])\n    n_past = min(len(past_a), len(past_b))\n\n    discr_count = np.ones(n_features, dtype=float) * 2.0\n    pair_both = np.ones((n_features, n_features), dtype=float) * 2.0\n    pair_same = np.ones((n_features, n_features), dtype=float)\n\n    for i in range(n_past):\n        pa = np.asarray(past_a[i], dtype=float)\n        pb = np.asarray(past_b[i], dtype=float)\n        if pa.shape[0] != n_features or pb.shape[0] != n_features:\n            continue\n        s = np.sign(pb - pa)\n        on = np.flatnonzero(s != 0)\n        discr_count[on] += 1.0\n        for q, j in enumerate(on):\n            for k in on[q + 1:]:\n                pair_both[j, k] += 1.0\n                pair_both[k, j] += 1.0\n                if s[j] == s[k]:\n                    pair_same[j, k] += 1.0\n                    pair_same[k, j] += 1.0\n\n    discr_rate = discr_count / float(n_past + 4.0)\n    reference_rate = float(np.mean(discr_rate))\n    rarity = np.clip(reference_rate / discr_rate, 0.35, 3.5)\n\n    # Learned anti-concurrence is high only for cue pairs that repeatedly\n    # discriminate in opposite directions. Pair-specific observation counts\n    # determine reliability, and a steep subject-specific threshold removes\n    # weak or chance-level disagreement while retaining systematic dissent.\n    anti_concurrence = np.clip(1.0 - pair_same / pair_both, 0.0, 1.0)\n    pair_observations = np.maximum(pair_both - 2.0, 0.0)\n    pair_confidence = pair_observations / (pair_observations + 2.0)\n    pair_confidence *= np.clip((pair_observations - 1.0) / 3.0, 0.0, 1.0)\n    threshold_baseline = 1.0 / (\n        1.0 + np.exp(-18.0 * (0.5 - anti_concurrence_threshold))\n    )\n    thresholded_anti = 1.0 / (\n        1.0 + np.exp(-18.0 * (anti_concurrence - anti_concurrence_threshold))\n    )\n    thresholded_anti = np.clip(\n        (thresholded_anti - threshold_baseline)\n        / max(1.0 - threshold_baseline, 1e-12),\n        0.0,\n        1.0,\n    )\n    learned_anti = pair_confidence * thresholded_anti\n\n    # Baseline cue accessibility combines instructed validity, serial\n    # prominence, and learned distributional rarity.\n    if n_features > 1:\n        position = 1.0 - np.arange(n_features, dtype=float) / (n_features - 1.0)\n    else:\n        position = np.ones(1, dtype=float)\n    cue_weight = np.exp(\n        validity_attention * reliability\n        + order_prominence * position\n        + distribution_attention * np.log(rarity)\n    )\n\n    # Cues that historically concur with their current coalition partners\n    # are treated as less independent. A cue with no partner is not given\n    # this historical redundancy penalty.\n    for j in active:\n        partners = [\n            int(k) for k in active\n            if k != j and directions[k] == directions[j]\n        ]\n        if partners:\n            concurrence = [pair_same[j, k] / pair_both[j, k] for k in partners]\n            corr_load = float(np.mean(concurrence))\n            cue_weight[j] /= 1.0 + redundancy * corr_load\n\n    a_cues = np.flatnonzero(directions < 0)\n    b_cues = np.flatnonzero(directions > 0)\n    n_a = int(a_cues.size)\n    n_b = int(b_cues.size)\n\n    # Lexicographic anchor: highest communicated validity among currently\n    # discriminating cues, with stable presentation order breaking ties.\n    active_validity = validities[active]\n    best_validity = float(np.max(active_validity))\n    best_candidates = active[np.isclose(active_validity, best_validity, atol=1e-10)]\n    lex_cue = int(np.min(best_candidates))\n    lex_direction = float(directions[lex_cue])\n\n    # A global top-validity tie attenuates profile-based singleton\n    # amplification, preventing the first cue from receiving both a strong\n    # order anchor and a full distinctiveness bonus.\n    sorted_v = np.sort(validities)[::-1]\n    if sorted_v.size >= 2:\n        top_gap = float(sorted_v[0] - sorted_v[1])\n        global_top_tie = float(np.exp(-80.0 * top_gap))\n    else:\n        global_top_tie = 1.0\n    tie_core_multiplier = 1.0 - tie_core_attenuation * global_top_tie\n\n    # Current-profile distinctiveness: members of the smaller directional\n    # coalition stand out, while each coalition's accumulated contribution\n    # is divided by a redundancy term.\n    if n_a > 0 and n_b > 0:\n        if n_a < n_b:\n            cue_weight[a_cues] *= (float(n_b) / float(n_a)) ** (\n                distinctiveness * tie_core_multiplier\n            )\n        elif n_b < n_a:\n            cue_weight[b_cues] *= (float(n_a) / float(n_b)) ** (\n                distinctiveness * tie_core_multiplier\n            )\n\n        # Sparse evidence opposing the validity-defined anchor receives its\n        # ordinary bounded bonus. Additional influence is restricted to\n        # dissenting cues with reliably learned above-threshold opposition.\n        dissent_cues = a_cues if lex_direction > 0 else b_cues\n        anchor_cues = b_cues if lex_direction > 0 else a_cues\n        n_dissent = int(dissent_cues.size)\n        if n_dissent > 0:\n            for j in dissent_cues:\n                learned_diagnosticity = 0.0\n                if anchor_cues.size > 0:\n                    learned_diagnosticity = float(\n                        np.mean(learned_anti[j, anchor_cues])\n                    )\n                bonus = 1.0 + (dissent_bonus / float(n_dissent)) * (\n                    1.0\n                    + anti_concurrence_attention * learned_diagnosticity\n                )\n                cue_weight[j] *= bonus\n\n            # A unique and presentation-prominent validity anchor becomes\n            # less accessible in the coalition core when compact opposition\n            # has a reliably learned anti-concurrent relationship with it.\n            # Active-cue validity uniqueness leaves tied-top profiles intact;\n            # squared prominence and size decay protect late anchors and\n            # anchors facing large, plausibly redundant coalitions.\n            if active_validity.size >= 2:\n                sorted_active_v = np.sort(active_validity)[::-1]\n                active_gap = float(sorted_active_v[0] - sorted_active_v[1])\n                anchor_uniqueness = 1.0 - np.exp(-80.0 * active_gap)\n            else:\n                anchor_uniqueness = 1.0\n            anchor_diagnosticity = float(\n                np.max(learned_anti[dissent_cues, lex_cue])\n            )\n            compact_decay = np.exp(-1.5 * float(n_dissent - 1))\n            anchor_signal = (\n                anchor_uniqueness\n                * float(position[lex_cue] ** 2)\n                * anchor_diagnosticity\n                * compact_decay\n            )\n            cue_weight[lex_cue] *= np.exp(\n                -anchor_attenuation * anchor_signal\n            )\n\n    support_a = 0.0\n    support_b = 0.0\n    if n_a > 0:\n        support_a = float(np.sum(cue_weight[a_cues])) / (float(n_a) ** redundancy)\n    if n_b > 0:\n        support_b = float(np.sum(cue_weight[b_cues])) / (float(n_b) ** redundancy)\n\n    scale = support_a + support_b\n    core_evidence = (support_b - support_a) / max(scale, 1e-12)\n\n    # A near tie among the two globally strongest instructed validities\n    # raises the stable-order gate, but anchoring has a subject-specific\n    # ceiling rather than becoming obligatorily lexicographic.\n    tied_top_gate = tie_gate_ceiling * global_top_tie\n\n    # Profile-driven anchoring occurs when the earliest discriminating cue\n    # is on an isolated minority side. It is attenuated when a sparse cue in\n    # the opposing coalition is historically rare or reliably\n    # anti-concurrent, while a large opposition remains interpretable as\n    # redundant evidence.\n    first_cue = int(active[0])\n    first_side_count = n_b if directions[first_cue] > 0 else n_a\n    other_side_count = n_a if directions[first_cue] > 0 else n_b\n    if first_side_count > 0 and other_side_count > 0:\n        isolation = 1.0 - float(first_side_count) / float(other_side_count)\n        minority_first = float(first_side_count < other_side_count)\n        opposing_cues = a_cues if directions[first_cue] > 0 else b_cues\n        first_side_cues = b_cues if directions[first_cue] > 0 else a_cues\n        opposing_rarity = float(np.max(np.maximum(np.log(rarity[opposing_cues]), 0.0)))\n        opposing_distinctiveness = opposing_rarity / (\n            np.log(3.5) * np.sqrt(float(other_side_count))\n        )\n\n        # For compact opposition, a reliability-weighted soft maximum lets\n        # one strongly diagnostic relation disengage the anchor without\n        # being diluted by weak pairings. Larger coalitions retain mean\n        # aggregation, and all configurations retain exponential size decay.\n        opposition_relations = learned_anti[\n            np.ix_(opposing_cues, first_side_cues)\n        ].reshape(-1)\n        if other_side_count <= 3 and opposition_relations.size > 1:\n            select_logits = diagnostic_selectivity * (\n                opposition_relations - np.max(opposition_relations)\n            )\n            select_weights = np.exp(select_logits)\n            select_weights /= np.sum(select_weights)\n            learned_opposition = float(\n                np.sum(select_weights * opposition_relations)\n            )\n        else:\n            learned_opposition = float(np.mean(opposition_relations))\n        learned_opposition *= np.exp(-1.5 * float(other_side_count - 1))\n        opposing_distinctiveness += (\n            anti_concurrence_attention * learned_opposition\n        )\n    else:\n        isolation = 0.0\n        minority_first = 0.0\n        opposing_distinctiveness = 0.0\n\n    profile_gate = minority_first / (\n        1.0 + np.exp(-gate_sharpness * (isolation - 0.30))\n    )\n    profile_gate *= np.exp(-gate_rarity * opposing_distinctiveness)\n    gate = 1.0 - (1.0 - tied_top_gate) * (1.0 - profile_gate)\n    gate = float(np.clip(gate, 0.0, 1.0))\n\n    choice_evidence = (\n        (1.0 - gate) * core_evidence\n        + gate * lex_strength * lex_direction\n    )\n\n    logits = beta * np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])\n    logits = logits - np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n    return probs.astype(np.float64)",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "validities": "validities",
    "validity_attention": "[0.2, 0.9]",
    "order_prominence": "[0.15, 0.7]",
    "distribution_attention": "[0.8, 1.8]",
    "distinctiveness": "[0.35, 0.9]",
    "dissent_bonus": "[1.0, 2.2]",
    "anti_concurrence_attention": "[1.0, 4.8]",
    "anti_concurrence_threshold": "[0.62, 0.76]",
    "diagnostic_selectivity": "[2.0, 8.0]",
    "anchor_attenuation": "[0.8, 3.2]",
    "redundancy": "[0.68, 1.15]",
    "lex_strength": "[0.7, 1.4]",
    "gate_sharpness": "[4.0, 11.0]",
    "gate_rarity": "[1.0, 4.0]",
    "tie_gate_ceiling": "[0.55, 0.85]",
    "tie_core_attenuation": "[0.7, 0.95]",
    "beta": "[0.6, 3.4]"
  },
  "rationale": "This is a two-range calibration of the accepted iteration-9 ADCG source; no equations or pathways are added. First, the upper bound of anchor_attenuation increases from 2.4 to 3.2 while its lower bound remains fixed. Because attenuation is already conditional on a unique validity anchor, early presentation, high-confidence anti-concurrence, and compact opposition, the added strength and subject heterogeneity are localized to the residual failures in Experiments 2 and 6. Tied-top profiles, late anchors, and large opposing coalitions remain protected by the existing multiplicative exclusions and size decay. Second, redundancy shifts modestly from [0.75, 1.25] to [0.68, 1.15]. This slightly softens generic coalition compression, reducing the excessive balanced-versus-imbalanced effect in Experiment 3. It should also modestly reduce excess top-cue adherence in Experiments 2 and 5 and strengthen the negative multi-opponent contrast in Experiment 6, while the unchanged large-coalition profile gate preserves Experiment 1. The successful gate-only soft maximum, mean direct diagnostic bonus, tied-top mechanisms, and all confidence computations are otherwise retained exactly."
}
```

## Usage

```json
{
  "input_tokens": 23596,
  "output_tokens": 5001
}
```
