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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Replace pi_2 (Tallying) with a *weighted-additive continuum* theory that nests both current heuristics as endpoints. Sketch: subjects integrate all cues, but with rank/validity-discounted weights w_i determined by the stated validities and a free non-compensatoriness exponent gamma, e.g. w_i proportional to (v_i - 0.5)^gamma or w_i = gamma^{rank_i} with gamma controlling how steeply weight decays down the validity ordering. Evidence for option A is sum_i w_i * (a_i - b_i), fed through a softmax with inverse temperature beta plus lapse epsilon. As gamma -> 0 the model becomes Tallying (equal weights); as gamma -> large it becomes Take-The-Best (top discriminating cue dominates, lower cues cannot overturn it); intermediate gamma yields log-odds/naive-Bayes-like Franklin's rule. Crucially, unlike TTB this model predicts (a) *graded* choice consistency with the weighted margin "" fewer top-cue-consistent choices when 4-5 lower cues oppose the top cue than when only 1-2 do, and (b) higher accuracy on unanimous 6-0 trials than on trials decided only by the 0.55 cue "" both of which the current designs' margin-scaling and graded-conflict families can measure. Parameter ranges should be centred on the strongly non-compensatory region (so the theory reproduces ~0.75-0.85 TTB agreement on conflict trials in both experiments) but with enough spread in gamma across simulated subjects to generate the larger between-subject variance seen in Exp1 (~0.03-0.04). This makes it a genuine competitor to pi_1: it matches TTB where TTB succeeded, and can additionally explain the systematic sub-ceiling conflict agreement and any margin-graded confidence that TTB forbids. Future experiments should target the discriminating prediction directly: within-conflict-family gradients (margin 1 vs 3 vs 5 against the top cue) and the 0.55-cue-only vs unanimous contrast.

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
**Other theories' values on this metric (for reference):**
- pi_2: 0.8583 (var=0.0090)
- pi_1: 0.1217 (var=0.0145)


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
  "theory": "**Graded Non-Compensatory Cue Integration (log-odds weights with a non-compensatoriness exponent).**\n\nPeople do not choose between one-reason stopping (Take-The-Best) and equal-weight counting (Tallying); those are the two endpoints of a single continuum of *weighted-additive* evidence integration. On every trial the decider forms a subjective diagnosticity weight for each expert from that expert's stated validity, using the log-odds of validity, d_i = log(v_i/(1-v_i)), which is the normatively correct evidence contribution of a binary cue. Crucially, the weights are then passed through a *non-compensatoriness exponent* gamma: w_i \u221d d_i^gamma, normalized to sum to one. Gamma expresses how sharply attention/credit is concentrated on the most diagnostic experts:\n\n- gamma \u2192 0: all weights equal \u2192 exact Tallying (majority of experts wins).\n- gamma = 1: weights equal to log-odds \u2192 the naive-Bayes / Franklin rule (optimal linear integration).\n- gamma \u2192 \u221e: weight collapses onto the single most valid discriminating expert \u2192 exact Take-The-Best (lower cues can never overturn it).\n\nEvidence for option A is E = \u03a3_i w_i (a_i \u2212 b_i), converted to a choice probability by a softmax with inverse temperature beta plus an independent lapse epsilon. Individuals differ in gamma (a stable trait sampled per subject) and in beta.\n\nBecause gamma in humans lies in the strongly non-compensatory but still *finite* region (roughly 1.2\u20131.8), the theory makes three predictions no pure heuristic makes: (1) choices agree with the top-cue direction most of the time but *sub-ceiling*, and the agreement is **graded by the weighted margin** \u2014 fewer top-cue-consistent choices when 4\u20135 lower cues oppose the top cue than when only 1\u20132 do, and near-perfect agreement when the two top cues jointly oppose the rest; (2) agreement with the top cue is higher in environments containing a *dominant* cue (e.g. validity .93 whose log-odds exceeds the exponent-shrunk sum of all remaining log-odds) than in environments with a smoothly graded validity profile, even for structurally identical conflict patterns; (3) unanimous or near-unanimous option pairs are decided essentially deterministically, while pairs decided only by a .55-validity cue approach chance. Prediction (2) is what pure rank-based rules (TTB, Tallying, geometric rank-discount schemes) structurally forbid: they are invariant to validity magnitudes and therefore must give the *same* conflict-agreement level in both environments.",
  "predict": "def predict(parameters, state, history):\n    # Graded Non-Compensatory Cue Integration.\n    # Weights come from the *magnitudes* of the stated validities:\n    #     d_i = log(v_i / (1 - v_i))          (log-odds diagnosticity)\n    #     w_i  proportional to d_i ** gamma    (gamma = non-compensatoriness)\n    # gamma -> 0  : equal weights            == Tallying\n    # gamma  = 1  : log-odds weights         == naive Bayes / Franklin's rule\n    # gamma -> inf: top cue dominates        == Take-The-Best\n    # Evidence  E = sum_i w_i * (a_i - b_i)  -> softmax(beta * E) -> lapse.\n    # History is ignored (no feedback is given in this domain).\n    import numpy as np\n\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim == 1:\n        # Flat vector: assume [A features..., B features...]\n        if stim.shape[0] % 2 != 0:\n            return np.ones(2) / 2.0\n        half = stim.shape[0] // 2\n        stim = np.vstack([stim[:half], stim[half:]])\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        return np.ones(2) / 2.0\n\n    n_features = int(stim.shape[1])\n    if n_features == 0:\n        return np.ones(2) / 2.0\n\n    # --- validities -------------------------------------------------\n    val = parameters.get(\"validities\", None)\n    try:\n        v = np.asarray(val, dtype=float).ravel()\n    except Exception:\n        v = np.array([])\n    if v.size != n_features or not np.all(np.isfinite(v)):\n        v = np.linspace(0.90, 0.55, n_features)\n    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)\n\n    # --- diagnosticity weights --------------------------------------\n    gamma = float(parameters[\"gamma\"])\n    d = np.log(v / (1.0 - v))              # >= 0, larger = more diagnostic\n    d = np.maximum(d, 1e-12)\n    with np.errstate(over=\"ignore\", under=\"ignore\", invalid=\"ignore\"):\n        w = np.power(d, gamma)\n    w = np.where(np.isfinite(w), w, 0.0)\n    tot = float(np.sum(w))\n    if tot <= 0.0 or not np.isfinite(tot):\n        w = np.ones(n_features) / float(n_features)\n    else:\n        w = w / tot\n\n    # --- weighted-additive evidence ---------------------------------\n    a = stim[0]\n    b = stim[1]\n    E = float(np.dot(w, a - b))\n\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Numerically stable softmax over [beta*E, 0]  (== sigmoid(beta*E) for A)\n    z = np.array([beta * E, 0.0], dtype=float)\n    z = z - np.max(z)\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n\n    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)\n    p = np.clip(p, 1e-12, 1.0)\n    p = p / p.sum()\n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64).ravel()\n    p = np.clip(p, 0.0, None)\n    s = p.sum()\n    if not np.isfinite(s) or s <= 0:\n        p = np.ones(p.shape[0]) / p.shape[0]\n    else:\n        p = p / s\n    return int(np.random.choice(len(p), p=p))",
  "parameters": {
    "gamma": "[1.2, 1.8]",
    "beta": "[7.0, 14.0]",
    "epsilon": "[0.0, 0.05]",
    "validities": "validities"
  },
  "rationale": "**Why the previous endpoints failed.** pi_1 (TTB) and pi_2 (Tallying) are both *rank-only* rules: they use the validity ORDER and nothing else. That is exactly why they mis-predict. TTB is at ceiling on Exp1 conflicts (0.84 vs human 0.74) and at floor on Exp2 (0.12 vs 0.19); Tallying is the mirror image. Neither can produce the empirically crucial fact that human top-cue agreement is *lower* in Exp1 (0.737) than in Exp2 (1 - 0.189 = 0.811) even though Exp1 and Exp2 contain structurally identical conflict patterns (top cue vs. 4 opposing lower cues). Any rank-based scheme \u2014 including the geometric rank-discount weights w_r = gamma^r the arbiter also mentioned \u2014 is provably unable to do this: for those weights the normalized top-cue share is *smaller* in the 6-cue environment, forcing Exp2 agreement BELOW Exp1, the wrong direction. I therefore implemented the arbiter's weighted-additive continuum in its *validity-magnitude* form (its first suggestion), because only magnitude-sensitive weights can express that Exp2 contains a dominant .93 expert while Exp1's validity profile is smoothly graded.\n\n**Why log-odds^gamma.** d_i = logit(v_i) is the normative evidence per binary cue, so gamma = 1 recovers Franklin/naive Bayes; gamma = 0 recovers Tallying exactly; gamma \u2192 \u221e recovers TTB exactly. The exponent is a single, interpretable non-compensatoriness dial that nests both prior theories as limits, satisfying the arbiter's brief while remaining experiment-invariant (it reads whatever validities the experiment supplies).\n\n**Quantitative check (hand-computed before submission).** With normalized weights u, the three Exp2 conflict families reduce to E = 2u\u2080+u\u2085-1, 2u\u2080-1, u\u2080+2u\u2081-1, and the four Exp1 conflict families to 2u\u2080-1, 2u\u2080+u\u2084-1, 2(u\u2080+u\u2081)-1, and (u\u2080+2u\u2081-1). At gamma = 1.5, beta = 10 the model yields Exp1 top-cue agreement \u2248 0.78 and Exp2 tally-agreement \u2248 0.188 (targets 0.737 / 0.189) \u2014 both far closer than either endpoint theory (TTB: 0.84 / 0.12; Tally: 0.18 / 0.86). Averaging over the declared gamma \u2208 [1.2, 1.8], beta \u2208 [7, 14], epsilon \u2208 [0, 0.05] gives \u2248 (0.76, 0.22): errors of ~0.02\u20130.03 per experiment versus 0.11 / 0.07 for pi_1.\n\n**Mechanistic novelty and new predictions.** Unlike TTB, agreement here is *graded by weighted margin*: Exp1's 'top-2 vs bottom-3' conflict is near-deterministic (E \u2248 0.62) while 'top-1 vs bottom-4' is barely above chance (E \u2248 0.11), and the trial where the top cue is tied and rank-2 fights ranks 3-5 is intermediate \u2014 producing sub-ceiling pooled agreement without invoking large lapse rates. Between-subject dispersion in gamma (a stable individual trait) generates genuine heterogeneity in conflict agreement, matching the large per-subject variance seen in Exp1, whereas noise-only accounts predict near-binomial variance. It further predicts high accuracy on unanimous 6-0 items and near-chance responding on items decided only by the .55 cue \u2014 the discriminating contrasts the arbiter asked future designs to target."
}
```

## Usage

```json
{
  "input_tokens": 9556,
  "output_tokens": 34585
}
```
