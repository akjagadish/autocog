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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Replace pi_2 with a context-sensitive ensemble salience weighting theory. The theory claims people first normalize each advertised validity relative to the ensemble of validities shown in the current task; a cue receives decision weight according to how much its validity stands out from the distribution, not according to raw validity alone. A simple implementation: compute a deviance score for cue j, such as d_j = abs(v_j - median(v)) / (MAD(v) + c), where MAD is median absolute deviation and c is a small floor, then set w_j = w0 + kappa * d_j ^ gamma; aggregate signed feature differences sum_j w_j * (A_j - B_j) through a softmax with inverse temperature beta and lapse epsilon, with feature ties contributing zero. This predicts Exp1 because 0.50 and 0.51 are dramatic low-validity outliers in a mostly 0.98 to 0.995 set, so they are strongly upweighted and produce low-side choice near 0.83. It predicts Exp2 because the validities form two balanced clusters, 0.50 to 0.53 and 0.97 to 0.995, no cue is an outlier, w_j collapses toward an equal-tally floor, and only a modest contrast around 0.15 remains. The same mechanism should preserve pi_4-like fits on the skewed Exp5 and Exp6 distributions and remain flexible on Exp3 and Exp4, while being a distinct competitor: it differs from pi_4 by using ensemble-relative salience instead of raw inverse validity, and from pi_2 by treating equal tally as a limiting case rather than a fixed rule. Parameter ranges should allow kappa near zero for the near-tally regime and moderately large kappa for the outlier regime, with w0 controlling the equal-weight floor.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.710)

**Description**
People treat advertised expert validity as a cue to ignore rather than trust: cues with low validity, especially those near chance, receive the largest decision weight, while very high-validity cues are discounted because they are seen as redundant or overstated. Choices are formed by a continuous reversed-validity weighted additive comparison of the two options, with signed feature differences weighted by an inverse diagnosticity function of the advertised validities. Ties on a feature contribute zero evidence. The accumulated evidence then passes through a softmax choice rule with inverse temperature beta, plus a small uniform lapse probability epsilon that captures attentional or motor noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Reversed-diagnosticity weighted additive integration.
    # state is expected to be array-like of shape (2, n_features),
    # with row 0 = option A and row 1 = option B.
    # History is ignored because validities are advertised in the
    # instructions and no trial-by-trial feedback is provided.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reversed-validity integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1:
        validities = validities.reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {stim.shape[1]}."
        )

    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])

    # Reversed diagnosticity weights. Low-validity cues receive large
    # weights; high-validity cues are discounted. The max guard keeps
    # weights non-negative even for perfect validities.
    weights = np.maximum(theta - validities, 0.0) ** gamma

    # Signed feature-wise evidence: positive favors A, negative favors B.
    # Feature ties contribute zero.
    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the two options [A, B].
    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - logits.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- theta: {1.0}
- gamma: [0.3, 0.4]
- beta: [4.8, 5.5]
- epsilon: [0.15, 0.25]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1505 (var=0.0003)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.8321 (var=0.0089)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=-0.2535 (var=0.0057)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.7908 (var=0.0142)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.8642 (var=0.0036)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.6663 (var=0.0054)


---

### `pi_2` (overall score: 0.563)

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
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1360 (var=0.0009)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.7597 (var=0.0423)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=0.0215 (var=0.0080)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=-0.1783 (var=0.0317)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.1492 (var=0.0089)
- Experiment 6: real=0.1525 (var=0.0118) vs this=0.0092 (var=0.0068)


---

### `pi_3` (overall score: 0.118)

**Description**
People do not stop after the first discriminating cue, nor do they weight every cue equally. Instead, they form a continuous decision value by comparing the two options on every feature, weighting each signed feature comparison by a monotonically increasing function of the advertised cue validity. Cues with validity near chance are either ignored or heavily discounted, while highly valid cues receive much larger weight. Ties on a feature contribute zero evidence. The resulting validity-weighted evidence sum is passed through a softmax with inverse temperature beta, plus a small uniform lapse probability epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Validity-weighted additive integration.
    # state is array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored because validities are
    # advertised in the instructions and no trial-by-trial feedback is given.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-weighted integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1:
        validities = validities.reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {stim.shape[1]}."
        )

    offset = float(parameters["offset"])
    gamma = float(parameters["gamma"])

    # Monotone diagnosticity weight: w_j = max(validity_j - offset, 0)^gamma.
    # For offset = 0.5 and gamma = 1 this is a validity-gain weighting,
    # giving chance cues zero weight.
    weights = np.maximum(validities - offset, 0.0) ** gamma

    # Signed feature-wise evidence: positive favors A, negative favors B.
    # Feature ties contribute zero.
    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over [A, B].
    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - logits.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))

`parameters`:
- beta: [1.0, 50.0]
- epsilon: [0.0, 0.25]
- offset: [0.0, 0.5]
- gamma: [0.5, 2.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0959 (var=0.0010) vs this=-0.1493 (var=0.0006)
- Experiment 2: real=-0.8592 (var=0.0589) vs this=-0.6946 (var=0.0665)
- Experiment 3: real=-0.3485 (var=0.0198) vs this=0.5096 (var=0.0191)
- Experiment 4: real=-0.6533 (var=0.0604) vs this=0.8683 (var=0.0156)
- Experiment 5: real=0.8275 (var=0.0166) vs this=0.0708 (var=0.0019)
- Experiment 6: real=0.1525 (var=0.0118) vs this=-0.6067 (var=0.0096)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    def subject_slope(sdf):
        a = np.array(sdf['option_a_ratings'].tolist(), dtype=float)
        b = np.array(sdf['option_b_ratings'].tolist(), dtype=float)
        resp = sdf['response'].to_numpy(dtype=float)
        n = a.shape[0]

        top = np.full(n, -1, dtype=int)
        top[a[:, 0] > b[:, 0]] = 0
        top[b[:, 0] > a[:, 0]] = 1
        valid = top >= 0

        opp_wins = np.zeros(n, dtype=int)
        for j in range(1, a.shape[1]):
            opp_wins[(top == 0) & (b[:, j] > a[:, j])] += 1
            opp_wins[(top == 1) & (a[:, j] > b[:, j])] += 1

        choose_top = np.where(resp == top, 1.0, 0.0)

        x = opp_wins[valid].astype(float)
        y = choose_top[valid]
        if x.size < 2 or np.ptp(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    slopes = data.groupby('subject_id', sort=False).apply(subject_slope)
    vals = np.atleast_1d(np.asarray(slopes, dtype=float))
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** -0.0959 (var=0.0010)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0006 (var=0.0002)
- pi_2: -0.1360 (var=0.0009)
- pi_3: -0.1493 (var=0.0006)
- pi_4: -0.1505 (var=0.0003)

### Experiment 2
**Design**
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.6], dtype=float)
    cue_order = np.argsort(-validities, kind='stable')

    y = data['response'].astype(float).to_numpy()
    dirs = np.zeros(len(data), dtype=float)
    margins = np.zeros(len(data), dtype=float)

    for i, (_, row) in enumerate(data.iterrows()):
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)

        margins[i] = float(np.sum(a > b) - np.sum(b > a))

        direction = 0.0
        for j in cue_order:
            if a[j] > b[j]:
                direction = 1.0
                break
            if b[j] > a[j]:
                direction = -1.0
                break
        dirs[i] = direction

    d_pos = dirs == 1.0
    d_neg = dirs == -1.0
    m_pos = margins > 0.0
    m_neg = margins < 0.0

    directional_contrast = y[d_neg].mean() - y[d_pos].mean()
    margin_contrast = y[m_neg].mean() - y[m_pos].mean()

    return float(directional_contrast - margin_contrast)

```

**Observed (real) value:** -0.8592 (var=0.0589)
**Other theories' values on this metric (for reference):**
- pi_2: -0.7597 (var=0.0423)
- pi_1: 0.7279 (var=0.0420)
- pi_3: -0.6946 (var=0.0665)
- pi_4: -0.8321 (var=0.0089)

### Experiment 3
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data):
    pairs_ordered = [
        (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
        (0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0),
        (1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0),
        (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0),
        (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
        (1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1),
        (1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1),
        (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)
    ]
    order = {p: i for i, p in enumerate(pairs_ordered)}

    def pair_key(row):
        return tuple(list(row['option_a_ratings']) + list(row['option_b_ratings']))

    keys = data.apply(pair_key, axis=1)
    pos = keys.map(order)
    if pos.isna().any():
        return 0.0

    rank = (pos % 8).to_numpy(dtype=float) + 1.0
    chose_a = (data['response'].to_numpy() == 0).astype(float)

    subj_corrs = []
    for sid in data['subject_id'].unique():
        mask = data['subject_id'].to_numpy() == sid
        if np.std(rank[mask]) == 0.0 or np.std(chose_a[mask]) == 0.0:
            subj_corrs.append(0.0)
            continue
        r = float(np.corrcoef(rank[mask], chose_a[mask])[0, 1])
        subj_corrs.append(r)

    return float(np.mean(subj_corrs))
```

**Observed (real) value:** -0.3485 (var=0.0198)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5096 (var=0.0191)
- pi_2: 0.0215 (var=0.0080)
- pi_1: 0.5469 (var=0.0262)
- pi_4: -0.2535 (var=0.0057)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    high_a = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    low_a = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)

    a_key = data['option_a_ratings'].apply(tuple)
    mask_high_a = a_key == high_a
    mask_low_a = a_key == low_a

    if mask_high_a.sum() == 0 or mask_low_a.sum() == 0:
        return 0.0

    pA_high = 1.0 - float(data.loc[mask_high_a, 'response'].mean())
    pA_low = 1.0 - float(data.loc[mask_low_a, 'response'].mean())

    return float(pA_high - pA_low)

```

**Observed (real) value:** -0.6533 (var=0.0604)
**Other theories' values on this metric (for reference):**
- pi_2: -0.1783 (var=0.0317)
- pi_3: 0.8683 (var=0.0156)
- pi_1: 0.6692 (var=0.0641)
- pi_4: -0.7908 (var=0.0142)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    total = 0.0
    n = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_low = int(a[0]) + int(a[1])
        b_low = int(b[0]) + int(b[1])
        if a_low == b_low:
            continue
        low_side = 'A' if a_low > b_low else 'B'
        opposing = b if low_side == 'A' else a
        k = sum(1 for j in range(2, 9) if int(opposing[j]) == 1)
        if 3 <= k <= 6:
            chose_low = ((low_side == 'A' and int(row['response']) == 0) or
                         (low_side == 'B' and int(row['response']) == 1))
            total += float(chose_low)
            n += 1
    if n == 0:
        return 0.5
    return total / float(n)
```

**Observed (real) value:** 0.8275 (var=0.0166)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8642 (var=0.0036)
- pi_2: 0.1492 (var=0.0089)
- pi_1: 0.1346 (var=0.0111)
- pi_3: 0.0708 (var=0.0019)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    pos1 = {
        ((1, 1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((0, 1, 1, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 0, 1, 1)),
        ((0, 1, 1, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((0, 0, 1, 1, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0, 1, 1)),
    }
    neg1 = {
        ((0, 0, 1, 0, 0, 1, 1, 0), (1, 1, 0, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0, 0, 0)),
        ((0, 0, 0, 1, 0, 1, 1, 0), (0, 1, 1, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1, 1, 1), (0, 1, 1, 1, 0, 0, 0, 0)),
    }
    early2 = {
        ((1, 1, 1, 1, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((1, 1, 1, 1, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0, 1, 1)),
        ((0, 1, 1, 1, 1, 1, 0, 0), (1, 0, 0, 0, 0, 0, 1, 1)),
        ((0, 1, 1, 1, 0, 0, 1, 1), (1, 0, 0, 0, 1, 1, 0, 0)),
    }
    late2 = {
        ((0, 0, 1, 1, 0, 1, 1, 1), (1, 1, 0, 0, 1, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 0, 0, 0, 0, 0)),
        ((0, 0, 1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 0, 0, 0, 0)),
        ((1, 0, 0, 0, 1, 1, 1, 1), (0, 1, 1, 1, 0, 0, 0, 0)),
    }
    akey = data['option_a_ratings'].apply(lambda x: tuple([int(v) for v in x]))
    bkey = data['option_b_ratings'].apply(lambda x: tuple([int(v) for v in x]))
    pairs = list(zip(akey, bkey))
    p_a = 1.0 - data['response'].to_numpy(dtype=float)
    p_pos1 = p_a[[k in pos1 for k in pairs]].mean()
    p_neg1 = p_a[[k in neg1 for k in pairs]].mean()
    p_early2 = p_a[[k in early2 for k in pairs]].mean()
    p_late2 = p_a[[k in late2 for k in pairs]].mean()
    block1_contrast = p_pos1 - p_neg1
    block2_contrast = p_early2 - p_late2
    return float(0.5 * (block1_contrast + block2_contrast))
```

**Observed (real) value:** 0.1525 (var=0.0118)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0092 (var=0.0068)
- pi_4: 0.6663 (var=0.0054)
- pi_1: -0.4425 (var=0.0114)
- pi_3: -0.6067 (var=0.0096)


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
  "theory": "People judge each advertised cue validity relative to the current task ensemble rather than using its raw value. A cue receives decision weight based on how strongly its validity deviates from the median validity, measured by a median-absolute-deviation score. Choices are then formed by a weighted signed tally of feature differences: each cue contributes a baseline floor weight plus a compressive salience bonus, ties contribute zero, and the accumulated evidence passes through a softmax choice rule with lapse. Equal-weight tallying emerges as the limiting case when no cue is an ensemble outlier.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError('Expected a (2, n_features) stimulus array.')\n\n    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)\n    if validities.shape[0] != stim.shape[1]:\n        raise ValueError('validities length must match n_features.')\n\n    median = np.median(validities)\n    mad = np.median(np.abs(validities - median))\n    c = float(parameters['c'])\n\n    deviance = np.abs(validities - median) / (mad + c)\n\n    w0 = float(parameters['w0'])\n    kappa = float(parameters['kappa'])\n    gamma = float(parameters['gamma'])\n\n    weights = w0 + kappa * np.power(deviance, gamma)\n\n    diff = stim[0] - stim[1]\n    evidence_A = float(np.dot(weights, diff))\n\n    beta = float(parameters['beta'])\n    epsilon = float(parameters['epsilon'])\n\n    logits = np.array([evidence_A, 0.0])\n    z = beta * (logits - np.max(logits))\n    e = np.exp(z)\n    core = e / e.sum()\n\n    n_opts = core.shape[0]\n    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts) / n_opts\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=float)\n    probs = probs / probs.sum()\n    return int(np.random.choice(probs.shape[0], p=probs))\n",
  "parameters": {
    "w0": "[0.10, 0.20]",
    "kappa": "[0.20, 0.60]",
    "gamma": "[0.35, 0.50]",
    "c": "[0.025, 0.045]",
    "beta": "[1.90, 2.40]",
    "epsilon": "[0.12, 0.18]",
    "validities": "validities"
  },
  "rationale": "This theory replaces fixed equal-weight tallying with ensemble-relative cue salience. In Experiment 1 and Experiment 5 the very low validity cues are dramatic outliers in a mostly high-validity ensemble, so they receive large salience bonuses and support the observed low-side choices without the extreme over-saturation of raw inverse-validity weighting. In Experiment 2 the validities form balanced clusters, so no cue deviates strongly from the median; weights collapse toward the w0 floor and behavior approximates tallying, which fits the strongly negative contrast. The moderate beta and epsilon values are chosen to preserve high low-side choice in Experiment 5 while reducing the over-strong contrasts that raw reversed-validity weighting produced in Experiment 6, and the w0/kappa ranges allow both near-tally and outlier-sensitive regimes across subjects."
}
```

## Usage

```json
{
  "input_tokens": 13979,
  "output_tokens": 38198,
  "reasoning_tokens": 37363,
  "provider": "Baidu"
}
```
