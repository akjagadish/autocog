# generation_iter_06_attempt_00

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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Replace pi_3 with a non-compensatory Inverted Take-the-Best / Anti-TTB heuristic. The theory assumes subjects order cues by stated validity, find the first discriminating cue, and then choose the option NOT favored by that cue with a subject-specific inversion probability q_i, drawn from a broad population distribution so that many subjects respond only weakly or sometimes follow the cue's recommendation. On trials with no discriminating cue, subjects use a subject-level bias, and a lapse probability adds further noise. Unlike pi_3, there is no linear accumulation of cue advantages and no continuous logit over a graded validity-weighted score; unlike pi_5, there is no saturating transform over the full validity-weighted difference and no deterministic subject-level sign. The new theory predicts near-flat within-sign slopes (Expt1), small weak-vs-strong gains (Expt2), moderate wrong-way TTB/first-cue effects (Expts 3–4), weak positive directional contrasts (Expts 5–6), near-zero aligned-vs-conflict effects (Expt7), and near-zero low-vs-high validity-sum effects (Expt8), while also generating the large between-subject variance seen in the real data. This makes it a stronger task-invariant competitor to pi_5 than pi_3 was.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.852)

**Description**
Compressive Negative Validity-Weighted integration with subject-level saturation (CNVW-tau). Each subject forms D = sum_j v_j^gamma_i * (A_j - B_j), where v_j are the instructed expert validities and gamma_i is a subject-level validity exponent. The weighted difference is passed through a strongly saturating bounded transform psi(D) = D / (tau_i + abs(D)), with a small subject-level half-saturation constant tau_i. Most subjects apply a negative sign, S_i = -lambda_i * psi(D) + bias_i, while a minority apply a positive sign. Choice probability is p(A) = epsilon_i/2 + (1 - epsilon_i) * sigmoid(beta_i * S_i). There is no feature-tally term and no conflict-gating interaction.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    """Compressive negative validity-weighted integration with subject-level tau.

    D = sum_j v_j^gamma * (A_j - B_j)
    psi(D) = D / (tau + abs(D))
    S = sign * lambda * psi(D) + bias
    p(A) = lapse/2 + (1-lapse) * sigmoid(beta * S)

    Most subjects use sign = -1; a minority draw sign = +1.
    history is intentionally ignored because there is no feedback.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "CNVW-tau expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    lam = float(parameters["lambda"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    bias = float(parameters["bias"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])
    tau = float(parameters["tau"])

    # Latent sign mixture: most subjects use negative-valence weighting;
    # a minority trust positive-valence evidence and get a positive sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Validity-weighted feature difference.
    d = float(np.dot(v ** gamma, a - b))

    # Subject-level strongly saturating bounded transform. The small
    # half-saturation constant tau forces most non-negligible validity-
    # weighted differences toward the sign bound, so sign and not
    # continuous magnitude drives the main choice contrast.
    psi = d / (tau + abs(d))

    score = sign * lam * psi + bias
    z = beta * score

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)

`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))

`parameters`:
- gamma: [1.5, 2.5]
- tau: [0.10, 0.25]
- lambda: [0.8, 2.0]
- beta: [0.8, 2.0]
- bias: [-0.25, 0.25]
- epsilon: [0.03, 0.10]
- p_positive: [0.03, 0.10]
- sign_draw: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4323 (var=0.0402) vs this=-0.2852 (var=0.0363)
- Experiment 2: real=-0.2650 (var=0.0713) vs this=-0.2133 (var=0.0171)
- Experiment 3: real=0.3622 (var=0.0979) vs this=0.5467 (var=0.1673)
- Experiment 4: real=0.5300 (var=0.0883) vs this=0.5175 (var=0.0843)
- Experiment 5: real=-0.0300 (var=0.0191) vs this=0.0050 (var=0.0269)
- Experiment 6: real=0.0117 (var=0.0033) vs this=0.0663 (var=0.0100)
- Experiment 7: real=0.0064 (var=0.0161) vs this=0.0393 (var=0.0171)
- Experiment 8: real=0.1200 (var=0.0096) vs this=0.1742 (var=0.0110)


---

### `pi_4` (overall score: 0.330)

**Description**
Validity-weighted evidence accumulation with conflict-gated confidence (VWEA-CG). People integrate binary feature differences by weighting each difference by its communicated validity raised to a subject-specific power. A majority of subjects use a negative-valence or contra-indicator rule, so a feature advantage counts against an option, while a minority use positive-valence trust. Critically, the effective decisiveness of that validity-weighted evidence is attenuated when the evidence strongly conflicts with the simple feature tally: when the validity-weighted score is large but the number of feature-wise wins is nearly tied, the decision is treated as less diagnostic. Choices are generated by a softmax over this conflict-gated signed evidence score, plus a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # Conflict-gated heterogeneous signed-validity weighting.
    # state is expected to be array-like of shape (2, n_features),
    # with row 0 = option A and row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Conflict-gated signed-validity model expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    bias_a = float(parameters["bias_a"])
    kappa = float(parameters["kappa"])
    conflict_power = float(parameters["conflict_power"])

    # Latent strategy draw: most subjects use negative-valence weighting,
    # while a minority use positive-valence trust.
    p_positive = float(parameters["p_positive"])
    valence_draw = float(parameters["valence_draw"])
    sign = 1.0 if valence_draw < p_positive else -1.0

    # Signed validity-weighted evidence and simple feature tally.
    weighted_diff = float(np.dot(v ** gamma, a - b))
    tally = float(np.sum(a > b) - np.sum(b > a))

    # Conflict between validity-weighted evidence and simple tally.
    # When validities push strongly toward one option but the tally is
    # nearly tied, confidence falls and the effective softmax gain shrinks.
    conflict = abs(weighted_diff) / (1.0 + abs(tally))
    attenuation = 1.0 / (1.0 + kappa * (conflict ** conflict_power))
    beta_eff = beta * attenuation

    score_a = sign * weighted_diff + bias_a
    z = beta_eff * score_a

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))
    return np.array([p_a, 1.0 - p_a], dtype=float)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


`parameters`:
- p_positive: [0.03, 0.07]
- valence_draw: [0.0, 1.0]
- gamma: [2.0, 2.4]
- beta: [4.2, 4.8]
- epsilon: [0.04, 0.08]
- bias_a: [0.0, 0.03]
- kappa: [2.3, 2.7]
- conflict_power: [3.8, 4.2]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4323 (var=0.0402) vs this=-0.3839 (var=0.0765)
- Experiment 2: real=-0.2650 (var=0.0713) vs this=-0.3037 (var=0.0165)
- Experiment 3: real=0.3622 (var=0.0979) vs this=0.5850 (var=0.1060)
- Experiment 4: real=0.5300 (var=0.0883) vs this=0.5467 (var=0.0492)
- Experiment 5: real=-0.0300 (var=0.0191) vs this=0.6350 (var=0.0201)
- Experiment 6: real=0.0117 (var=0.0033) vs this=-0.0158 (var=0.0128)
- Experiment 7: real=0.0064 (var=0.0161) vs this=0.0529 (var=0.0066)
- Experiment 8: real=0.1200 (var=0.0096) vs this=0.3106 (var=0.0090)


---

### `pi_2` (overall score: 0.088)

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
- Experiment 1: real=-0.4323 (var=0.0402) vs this=0.4517 (var=0.0090)
- Experiment 2: real=-0.2650 (var=0.0713) vs this=-0.0083 (var=0.0072)
- Experiment 3: real=0.3622 (var=0.0979) vs this=0.0017 (var=0.0075)
- Experiment 4: real=0.5300 (var=0.0883) vs this=0.0025 (var=0.0212)
- Experiment 5: real=-0.0300 (var=0.0191) vs this=0.6083 (var=0.0361)
- Experiment 6: real=0.0117 (var=0.0033) vs this=0.0254 (var=0.0198)
- Experiment 7: real=0.0064 (var=0.0161) vs this=0.0118 (var=0.0112)
- Experiment 8: real=0.1200 (var=0.0096) vs this=0.3414 (var=0.0202)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1660 -> ACCEPTED
- iter 2: loss=0.1395 -> ACCEPTED
- iter 3: loss=0.1302 -> ACCEPTED
- iter 4: loss=0.1023 -> ACCEPTED
- iter 5: loss=1.0000 -> REJECTED
- iter 6: loss=0.1057 -> REJECTED
Running-best (last ACCEPTED) base: iter 4 at loss=0.1023 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data):
    import numpy as np
    min_class = 12
    groups = {0: [], 1: []}
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)
        if a[0] > b[0]:
            grp = 0
        elif b[0] > a[0]:
            grp = 1
        else:
            continue
        tally = int(np.sum(a > b) - np.sum(b > a))
        chose_a = 1.0 if int(row['response']) == 0 else 0.0
        groups[grp].append((tally, chose_a))
    parts = []
    for grp in (0, 1):
        vals = groups[grp]
        if not vals:
            parts.append(0.0)
            continue
        d_a = [d for d, y in vals if y == 1.0]
        d_b = [d for d, y in vals if y == 0.0]
        if len(d_a) < min_class or len(d_b) < min_class:
            parts.append(0.0)
        else:
            parts.append((np.mean(d_a) - np.mean(d_b)) / 6.0)
    return float(np.mean(parts))
```

**Observed (real) value:** -0.4323 (var=0.0402)
**Previous candidate values (this loop):**
  - iter 1: -0.0116 (var=0.0052) (Δ vs real +0.4207)
  - iter 2: -0.0267 (var=0.0031) (Δ vs real +0.4056)
  - iter 3: -0.0989 (var=0.0068) (Δ vs real +0.3335)
  - iter 4: -0.0125 (var=0.0063) (Δ vs real +0.4198)
  - iter 5: n/a (var=n/a)
  - iter 6 (most recent): -0.0229 (var=0.0065) (Δ vs real +0.4094)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0157 (var=0.0016)
- pi_2: 0.4517 (var=0.0090)
- pi_3: -0.5184 (var=0.0087)
- pi_4: -0.3839 (var=0.0765)
- pi_5: -0.2852 (var=0.0363)

### Experiment 2
**Design**
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    validities = np.array([0.9, 0.8, 0.65, 0.55, 0.5])
    cue_order = np.argsort(-validities, kind='stable')
    a_when_ttb_a = []
    a_when_ttb_b = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner == -1:
            continue
        chose_a = 1 if int(row['response']) == 0 else 0
        if ttb_winner == 0:
            a_when_ttb_a.append(chose_a)
        else:
            a_when_ttb_b.append(chose_a)
    if not a_when_ttb_a or not a_when_ttb_b:
        return 0.0
    return float(np.mean(a_when_ttb_a) - np.mean(a_when_ttb_b))
```

**Observed (real) value:** -0.2650 (var=0.0713)
**Previous candidate values (this loop):**
  - iter 1: -0.0683 (var=0.1557) (Δ vs real +0.1967)
  - iter 2: -0.2225 (var=0.0176) (Δ vs real +0.0425)
  - iter 3: -0.2575 (var=0.0132) (Δ vs real +0.0075)
  - iter 4: -0.2800 (var=0.0090) (Δ vs real -0.0150)
  - iter 5: n/a (var=n/a)
  - iter 6 (most recent): -0.2671 (var=0.0139) (Δ vs real -0.0021)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0083 (var=0.0072)
- pi_1: 0.6883 (var=0.0485)
- pi_3: -0.2804 (var=0.0079)
- pi_4: -0.3037 (var=0.0165)
- pi_5: -0.2133 (var=0.0171)

### Experiment 3
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    import pandas as pd

    df = data.copy()
    df["pair_key"] = df.apply(
        lambda r: (tuple(r["option_a_ratings"]), tuple(r["option_b_ratings"])),
        axis=1,
    )

    nwa_negative_pairs = [
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0)),
        ((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
    ]
    nwa_positive_pairs = [
        ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 0), (1, 1, 1, 0, 0, 0)),
    ]

    def p_a_for_key(d, key):
        mask = d["pair_key"] == key
        return 1.0 - float(d.loc[mask, "response"].mean())

    def subject_score(d):
        neg = [p_a_for_key(d, k) for k in nwa_negative_pairs]
        pos = [p_a_for_key(d, k) for k in nwa_positive_pairs]
        return sum([pos[i] - neg[i] for i in range(3)]) / 3.0

    scores = df.groupby("subject_id", sort=False).apply(subject_score)
    return float(scores.mean())

```

**Observed (real) value:** 0.3622 (var=0.0979)
**Previous candidate values (this loop):**
  - iter 1: 0.1872 (var=0.1794) (Δ vs real -0.1750)
  - iter 2: 0.4067 (var=0.0296) (Δ vs real +0.0444)
  - iter 3: 0.1367 (var=0.0162) (Δ vs real -0.2256)
  - iter 4: 0.5083 (var=0.0192) (Δ vs real +0.1461)
  - iter 5: n/a (var=n/a)
  - iter 6 (most recent): 0.3517 (var=0.0171) (Δ vs real -0.0106)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8928 (var=0.0032)
- pi_2: 0.0017 (var=0.0075)
- pi_1: -0.7161 (var=0.0463)
- pi_4: 0.5850 (var=0.1060)
- pi_5: 0.5467 (var=0.1673)

### Experiment 4
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    diffs = [tuple(int(x) - int(y) for x, y in zip(a, b)) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    choose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    high_keys = {(-1, -1, 1, 1), (-1, 1, -1, 1)}
    low_keys = {(1, 1, -1, -1), (1, -1, 1, -1)}
    high_mask = np.array([d in high_keys for d in diffs], dtype=bool)
    low_mask = np.array([d in low_keys for d in diffs], dtype=bool)
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0
    return float(choose_a[high_mask].mean() - choose_a[low_mask].mean())
```

**Observed (real) value:** 0.5300 (var=0.0883)
**Previous candidate values (this loop):**
  - iter 1: 0.2792 (var=0.1667) (Δ vs real -0.2508)
  - iter 2: 0.0350 (var=0.0225) (Δ vs real -0.4950)
  - iter 3: 0.5358 (var=0.0139) (Δ vs real +0.0058)
  - iter 4: 0.5375 (var=0.0239) (Δ vs real +0.0075)
  - iter 5: n/a (var=n/a)
  - iter 6 (most recent): 0.5242 (var=0.0234) (Δ vs real -0.0058)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0025 (var=0.0212)
- pi_3: 0.6125 (var=0.0127)
- pi_1: -0.6867 (var=0.0422)
- pi_4: 0.5467 (var=0.0492)
- pi_5: 0.5175 (var=0.0843)

### Experiment 5
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    pair_specs = {
        'cp': ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
        'ap': ((1, 0, 1, 1, 1, 1), (0, 1, 0, 0, 0, 0)),
        'cn': ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),
        'an': ((0, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1)),
    }
    lookup = {v: k for k, v in pair_specs.items()}

    df = data.copy()
    df['_pair'] = [
        lookup.get((tuple(a), tuple(b)), 'other')
        for a, b in zip(df['option_a_ratings'], df['option_b_ratings'])
    ]
    df['_chose_a'] = (df['response'] == 0).astype(float)

    subject_effects = []
    for _, sub in df.groupby('subject_id'):
        means = sub.groupby('_pair')['_chose_a'].mean()
        if not {'cp', 'ap', 'cn', 'an'}.issubset(means.index):
            continue
        cp = means['cp']
        ap = means['ap']
        cn = means['cn']
        an = means['an']
        effect = (
            (abs(ap - 0.5) - abs(cp - 0.5))
            + (abs(an - 0.5) - abs(cn - 0.5))
        )
        subject_effects.append(effect)

    if not subject_effects:
        return float('nan')
    effects = np.asarray(subject_effects, dtype=float)
    if np.any(np.isnan(effects)):
        return float('nan')
    return float(effects.mean())
```

**Observed (real) value:** -0.0300 (var=0.0191)
**Previous candidate values (this loop):**
  - iter 1: 0.0108 (var=0.0209) (Δ vs real +0.0408)
  - iter 2: -0.0075 (var=0.0252) (Δ vs real +0.0225)
  - iter 3: 0.1292 (var=0.0389) (Δ vs real +0.1592)
  - iter 4: -0.0008 (var=0.0323) (Δ vs real +0.0292)
  - iter 5: n/a (var=n/a)
  - iter 6 (most recent): 0.1167 (var=0.0264) (Δ vs real +0.1467)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0017 (var=0.0039)
- pi_4: 0.6350 (var=0.0201)
- pi_1: -0.0033 (var=0.0180)
- pi_2: 0.6083 (var=0.0361)
- pi_5: 0.0050 (var=0.0269)

### Experiment 6
**Design**
  A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]

**Metric**
```python
import numpy as np

def metric(data):
    v = np.array([0.95, 0.9, 0.85, 0.8, 0.6, 0.55, 0.5], dtype=float)

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    z = (a - b).dot(v ** 2)

    resp = np.asarray(data['response'], dtype=float)

    low_mask = z < 0.5
    high_mask = z > 0.9

    lows = resp[low_mask]
    highs = resp[high_mask]

    if lows.size == 0 or highs.size == 0:
        return float('nan')

    p_a_low = float(np.mean(lows == 0))
    p_a_high = float(np.mean(highs == 0))

    return float(p_a_low - p_a_high)

```

**Observed (real) value:** 0.0117 (var=0.0033)
**Previous candidate values (this loop):**
  - iter 1: 0.0212 (var=0.0109) (Δ vs real +0.0096)
  - iter 2: -0.0175 (var=0.0152) (Δ vs real -0.0292)
  - iter 3: 0.0425 (var=0.0098) (Δ vs real +0.0308)
  - iter 4: -0.0046 (var=0.0087) (Δ vs real -0.0163)
  - iter 5: n/a (var=n/a)
  - iter 6 (most recent): -0.0025 (var=0.0094) (Δ vs real -0.0142)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0158 (var=0.0128)
- pi_3: 0.2812 (var=0.0041)
- pi_1: -0.0062 (var=0.0073)
- pi_2: 0.0254 (var=0.0198)
- pi_5: 0.0663 (var=0.0100)

### Experiment 7
**Design**
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.84, 0.74, 0.62, 0.52], dtype=float)
    weights = validities ** 2.25

    a_rows = [np.asarray(r, dtype=float) for r in data['option_a_ratings']]
    b_rows = [np.asarray(r, dtype=float) for r in data['option_b_ratings']]
    if len(a_rows) == 0:
        return 0.0

    a = np.vstack(a_rows)
    b = np.vstack(b_rows)
    d = (a - b) @ weights

    resp = data['response'].to_numpy(dtype=float)
    sign_consistent = np.zeros(len(data), dtype=float)
    sign_consistent[(d > 0.0) & (resp == 1.0)] = 1.0
    sign_consistent[(d < 0.0) & (resp == 0.0)] = 1.0

    x = np.abs(d)
    xc = x - x.mean()
    denom = np.sum(xc * xc)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(xc * sign_consistent) / denom)

```

**Observed (real) value:** 0.0064 (var=0.0161)
**Previous candidate values (this loop):**
  - iter 1: 0.0529 (var=0.0240) (Δ vs real +0.0465)
  - iter 2: 0.0036 (var=0.0173) (Δ vs real -0.0028)
  - iter 3: -0.0103 (var=0.0278) (Δ vs real -0.0167)
  - iter 4: -0.0028 (var=0.0219) (Δ vs real -0.0092)
  - iter 5: n/a (var=n/a)
  - iter 6 (most recent): 0.0162 (var=0.0368) (Δ vs real +0.0098)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1683 (var=0.0061)
- pi_5: 0.0393 (var=0.0171)
- pi_1: -0.0904 (var=0.0098)
- pi_2: 0.0118 (var=0.0112)
- pi_4: 0.0529 (var=0.0066)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    validities = np.array([0.95, 0.9, 0.8, 0.6, 0.5], dtype=float)
    weights = validities ** 2.25

    subject_contrasts = []
    for _, subj in data.groupby('subject_id'):
        cons = []
        high_mask = []
        low_mask = []

        for _, row in subj.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            d = float(np.dot(weights, a - b))
            ad = abs(d)

            # Negative-validity prediction: choose B when A has larger
            # validity-weighted advantage, otherwise choose A.
            pred_neg = 1 if d > 0 else 0
            cons.append(1.0 if int(row['response']) == pred_neg else 0.0)
            high_mask.append(ad >= 0.90)
            low_mask.append(ad <= 0.25)

        cons = np.array(cons)
        high_mask = np.array(high_mask, dtype=bool)
        low_mask = np.array(low_mask, dtype=bool)

        if high_mask.sum() == 0 or low_mask.sum() == 0:
            subject_contrasts.append(0.0)
            continue

        raw_contrast = float(cons[high_mask].mean() - cons[low_mask].mean())
        # Absolute value aligns negative-sign and positive-sign subjects.
        subject_contrasts.append(abs(raw_contrast))

    if not subject_contrasts:
        return 0.0
    return float(np.mean(subject_contrasts))
```

**Observed (real) value:** 0.1200 (var=0.0096)
**Previous candidate values (this loop):**
  - iter 1: 0.0856 (var=0.0033) (Δ vs real -0.0344)
  - iter 2: 0.0903 (var=0.0036) (Δ vs real -0.0297)
  - iter 3: 0.0803 (var=0.0036) (Δ vs real -0.0397)
  - iter 4: 0.0797 (var=0.0037) (Δ vs real -0.0403)
  - iter 5: n/a (var=n/a)
  - iter 6 (most recent): 0.0903 (var=0.0057) (Δ vs real -0.0297)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1742 (var=0.0110)
- pi_3: 0.3681 (var=0.0125)
- pi_1: 0.0822 (var=0.0037)
- pi_2: 0.3414 (var=0.0202)
- pi_4: 0.3106 (var=0.0090)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Validity-gated slipped-start Inverted Take-the-Best (GSS-ITTB). People order cues by stated validity and normally use the first discriminating cue, inverting its recommendation with subject-specific probability q_i. Attentional slips to later ordered cues are not driven by feature count alone; they occur mainly when the first discriminator's validity is below a threshold, and are strongly suppressed for first discriminators with validity 0.90 or higher. When a slip occurs, the later discriminating cue controls the response with a boosted inversion probability, reflecting stronger contra-indicator responding in the low-validity regime. Trials with no discriminating cue are resolved by subject-level bias, and a small lapse adds residual noise. There is no graded validity-weighted accumulation and no continuous logit over a compensatory score.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    """Validity-gated slipped-start Inverted Take-the-Best (GSS-ITTB).

    Cues are ordered by stated validity. The normative first
    discriminating cue sets the subject's base inversion probability
    q_norm. Attentional slips to later ordered cues are gated by the
    validity of that first discriminator: slips are allowed and
    strengthened when validity is below gate_low, ramp down between
    gate_low and 0.90, and are switched off at 0.90 or above. Slipped
    responses use a boosted inversion strength q_slip. If no cue
    discriminates, a subject-level bias is used. No graded cue
    accumulation enters the model.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "GSS-ITTB expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    n = int(a.size)

    q_lo = float(parameters["q_lo"])
    q_mid = float(parameters["q_mid"])
    q_90 = float(parameters["q_90"])
    q_92 = float(parameters["q_92"])
    q_hi = float(parameters["q_hi"])
    q_floor = float(parameters["q_floor"])
    bias = float(parameters["bias"])
    epsilon = float(parameters["epsilon"])
    late2_base = float(parameters["late2_base"])
    late2_slope = float(parameters["late2_slope"])
    late3_base = float(parameters["late3_base"])
    late3_slope = float(parameters["late3_slope"])
    gate_low = float(parameters["gate_low"])
    q_low_boost = float(parameters["q_low_boost"])
    max_slip = float(parameters["max_slip"])

    def q_of(vv):
        vv = float(vv)
        if vv > 0.97:
            return float(np.clip(q_floor, 0.0, 1.0))
        return float(np.interp(
            vv,
            np.array([0.50, 0.80, 0.90, 0.92, 0.95], dtype=float),
            np.array([q_lo, q_mid, q_90, q_92, q_hi], dtype=float),
        ))

    cue_order = np.argsort(-v, kind="stable")

    def first_disc(start_idx):
        for r in range(start_idx, n):
            j = int(cue_order[r])
            if a[j] != b[j]:
                return j
        return -1

    j0 = first_disc(0)

    if j0 < 0:
        # No discriminating cue anywhere: subject-level bias.
        p_core = bias
    else:
        # Base inversion strength is calibrated by the normative first
        # discriminating cue.
        q_norm = q_of(v[j0])

        # Feature-count pressure toward late search.
        extra = max(0.0, float(n - 5))
        late2 = float(np.clip(late2_base + late2_slope * extra, 0.0, 0.90))
        late3 = float(np.clip(late3_base + late3_slope * extra, 0.0, 0.60))

        # Validity gate on slip: full slip below gate_low, zero slip at
        # validity 0.90 or higher, linear ramp between.
        vj0 = float(v[j0])
        if vj0 < gate_low:
            gate = 1.0
        elif vj0 >= 0.90:
            gate = 0.0
        else:
            gate = (0.90 - vj0) / (0.90 - gate_low)
        gate = float(np.clip(gate, 0.0, 1.0))

        late2 *= gate
        late3 *= gate

        if late2 + late3 > max_slip:
            scale = max_slip / (late2 + late3)
            late2 *= scale
            late3 *= scale

        p0 = 1.0 - late2 - late3

        # Slipped-to cues are inverted more strongly in the gated regime.
        q_slip = float(np.clip(q_norm + q_low_boost * gate, 0.0, 1.0))

        def direction_prob(start_idx, q_val):
            j = first_disc(start_idx)
            if j < 0:
                return bias
            if a[j] > b[j]:
                # Effective cue favors A, so inverted tendency chooses B.
                return 1.0 - q_val
            # Effective cue favors B, so inverted tendency chooses A.
            return q_val

        p_core = (
            p0 * direction_prob(0, q_norm)
            + late2 * direction_prob(2, q_slip)
            + late3 * direction_prob(3, q_slip)
        )

    p_a_core = float(np.clip(p_core, 0.0, 1.0))

    # Uniform lapse over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


`parameters`:
- q_lo: [0.20, 0.40]
- q_mid: [0.55, 0.68]
- q_90: [0.66, 0.82]
- q_92: [0.48, 0.62]
- q_hi: [0.70, 0.86]
- q_floor: [0.45, 0.65]
- bias: [0.30, 0.70]
- epsilon: [0.02, 0.08]
- late2_base: [0.02, 0.08]
- late2_slope: [0.80, 1.10]
- late3_base: [0.02, 0.06]
- late3_slope: [0.10, 0.25]
- gate_low: [0.84, 0.88]
- q_low_boost: [0.10, 0.25]
- max_slip: [0.93, 0.98]
- validities: validities

`rationale`: This is a minimal edit of the prior SS-ITTB candidate. The main change is that late-cue slip is now validity-gated rather than feature-count-only: slip is allowed for first discriminators below gate_low, ramps down toward 0.90, and is zero at 0.90 or higher. This preserves the within-sign sensitivity needed for Experiment 1 while removing the slip that diluted the strong first-cue wrong-way effects in Experiments 3 and 5. The gated slip branch is strengthened by raising late2/late3 probabilities, increasing max_slip, and adding q_low_boost, which boosts inversion only when the first discriminator is in the gated low-validity regime. q_mid is raised to 0.55-0.68 to increase low-validity inversion, while q_90, q_92, and q_hi keep their previous centers to protect Experiments 2, 4, and 7. q_90, q_hi, and bias ranges are broadened to restore more realistic between-subject variance without changing their central tendencies. The model remains strictly lexicographic and non-compensatory.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate implements a genuinely non-compensatory inverted TTB rule and does well on the low-contrast experiments: Expt5 is 0.0108 vs real -0.0300, Expt6 is 0.0212 vs real 0.0117, Expt7 is 0.0529 vs real 0.0064, and Expt8 is 0.0856 vs real 0.1200. These are close enough and match the intended near-zero directional and validity-sum effects. However, the candidate systematically underpredicts the core wrong-way effects. Expt1 candidate is -0.0116 while real is -0.4323, essentially flat versus a large negative within-sign slope. Expt2 candidate is -0.0683 versus real -0.2650. Expt3 candidate is 0.1872 versus real 0.3622. Expt4 candidate is 0.2792 versus real 0.5300. All four are in the correct direction but much too weak, suggesting the subject inversion probability is not strong enough or applies too narrowly. The candidate also makes inversion conditional on the validity of the first discriminating cue via a logistic link, which is a drift from the prescribed subject-specific q_i drawn from a broad distribution. This likely explains the under-prediction: when the first discriminating cue has moderate or low stated validity, many simulated subjects follow the cue rather than invert it, while the real data show substantial inversion across these trials. The between-subject variance in Expt2 is also too high for the candidate, 0.1557 versus real 0.0713, indicating the population heterogeneity is wider than the data support.
Rationale: Regenerate while keeping the same non-compensatory anti-TTB mechanism family. The main needed change is to strengthen the first-cue inversion: shift the effective q distribution upward so the median subject is a clear inverter rather than near-chance, and reduce the dependence of q on cue validity. Better alignment with the prescribed family would be to sample a subject-specific q_i from a broad but appropriately centered distribution, rather than deriving q from theta_q plus gamma_q times validity. If the logistic calibration is retained, theta_q should be moved toward zero or positive and gamma_q may be lowered or kept small so weak-to-moderate first cues still produce substantial inversion. Also tighten the population variance somewhat to reduce the Expt2 variance from 0.156 toward the observed 0.071, and consider lowering epsilon only if needed. These changes should increase the Expt1-4 contrasts toward the real values while leaving Expt5-8 near their current good values, because those metrics are structurally near zero under a lexicographic inversion rule. Do not reintroduce graded cue accumulation or a partial-linear weighting account.

**Outcome of this advice:** iter 1 candidate loss=0.1660 -> iter 2 candidate loss=0.1395 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: Iter 2 is the best candidate so far and improves several experiments: Expt2 is -0.2225 versus real -0.2650, Expt3 is 0.4067 versus real 0.3622, Expt5 is -0.0075 versus real -0.0300, Expt6 is -0.0175 versus real 0.0117, Expt7 is 0.0036 versus real 0.0064, and Expt8 is 0.0903 versus real 0.1200. Those are all within a reasonable band. The serious remaining failures are Expt1 and Expt4. Expt1 remains almost flat at -0.0267 versus real -0.4323, so the empirical within-sign slope is not captured. Expt4 regressed sharply from iter1's 0.2792 down to 0.0350, versus real 0.5300, so the candidate now produces almost no wrong-way high-versus-low contrast where the real data show a large one. The likely culprit is the new high-validity cap: the model floors inversion near 0.5 for validity values at or above 0.90, which erases the very contrast Expt4 requires. The candidate also now under-produces between-subject variance: Expt1 0.0031 versus real 0.0402, Expt2 0.0176 versus 0.0713, Expt3 0.0296 versus 0.0979, and Expt4 0.0225 versus 0.0883. The previous request to reduce variance was overcorrected.
Rationale: Regenerate while keeping the non-compensatory inverted TTB core. The main in-family fix is to recalibrate the piecewise inversion curve. The pre-peak rise is too shallow: validity values around 0.75 to 0.85 currently produce q near 0.50 to 0.60, whereas the real Expt4 pattern implies inversion around 0.70 to 0.80 for the first discriminating cue in that design. Increase q_base or rise so q reaches roughly 0.70 to 0.80 before validity 0.90. The cap should remain steep only for extreme validities if it is needed to keep Expt7 near zero, but it must not flatten the validity range driving Expt4. Moderately widen the subject-level q distribution so the per-subject metric variances move toward the real values, but do not return to the overly broad iter1 distribution that gave Expt2 variance near 0.156; target Expt2 variance around 0.07 and Expt3/Expt4 around 0.09. Expt1 is likely hard for a strict first-discriminator rule because the metric conditions on the first cue; any additional within-sign sensitivity should be implemented through small in-family cue-order or cue-skip noise, not through graded accumulation or a compensatory weighted score.

**Outcome of this advice:** iter 2 candidate loss=0.1395 -> iter 3 candidate loss=0.1302 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is the best aggregate fit so far and confirms the value of moving the inversion plateau to high validity while keeping a dip near 0.92. Experiment 4 is now essentially exact, 0.5358 vs real 0.5300, and Experiment 2 is nearly exact, -0.2575 vs real -0.2650. Experiment 7 remains near zero, -0.0103 vs 0.0064, so the validity-dip logic is doing its job. Experiment 6 is also close, 0.0425 vs 0.0117. However, the candidate still has substantial failures. Experiment 1 remains far too flat: -0.0989 vs real -0.4323, capturing only about one quarter of the real within-sign wrong-way slope. Experiment 3 regressed sharply from the previous iteration's 0.4067 down to 0.1367 against a real 0.3622; this is almost certainly caused by the new feature-count-gated attentional slip, because Experiment 3 has six features and the slip dilutes the first-cue inversion. Experiment 5 also worsened from -0.0075 to 0.1292 against a real -0.0300, again consistent with the slip creating a directional asymmetry. The variance profile is also still too compressed for the core wrong-way experiments: Experiment 3 variance is 0.0162 vs real 0.0979, Experiment 4 is 0.0139 vs real 0.0883, and Experiment 2 is 0.0132 vs real 0.0713.
Rationale: Regenerate while keeping the non-compensatory SS-ITTB core and the current q-curve around validities 0.90, 0.92, and 0.95, which fixed Experiments 2, 4, and 7. The main defect is the slip trigger. Do not trigger late-cue search by feature count alone, because both Experiment 1 and Experiment 3 have six features yet need opposite amounts of slip. Instead, make the slip probability conditional on the validity of the first discriminating cue: allow strong late2 and late3 only when v(j0) is below roughly 0.85 to 0.88, and strongly suppress them when v(j0) is 0.90 or higher. This should restore Experiment 3's strong first-cue wrong-way effect while preserving the within-sign sensitivity that improved Experiment 1. To push Experiment 1 closer to the real -0.4323, strengthen the gated-slip branch and moderately raise inversion for the lower-validity first-discriminator region, for example q_mid around 0.55 to 0.68, without raising q_90 or q_hi, so Experiment 4 is not overshot. Keep the q(0.92) dip unchanged to protect Experiment 7. Also add somewhat more between-subject spread in the inversion-strength parameters specifically; the point estimates for the wrong-way effects are now decent, but the population variance in Experiments 1 through 4 is still several-fold lower than the real data, so broader q_90, q_hi, and bias ranges with targets around Experiment 2 variance 0.07 and Experiment 3 or 4 variance around 0.09 are appropriate. The proposed changes stay within the lexicographic inverted TTB family.

**Outcome of this advice:** iter 3 candidate loss=0.1302 -> iter 4 candidate loss=0.1023 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is the best aggregate fit so far and the validity-gated slipped-start inverted TTB core is working across several experiments: Expt2 is nearly exact at -0.2800 versus real -0.2650, Expt4 is essentially exact at 0.5375 versus real 0.5300, Expt5 is -0.0008 versus real -0.0300, Expt6 is -0.0046 versus real 0.0117, and Expt7 is -0.0028 versus real 0.0064. Expt8 remains somewhat low at 0.0797 versus real 0.1200. The two serious remaining failures are Expt1 and Expt3. Expt1 is still almost flat at -0.0125 versus real -0.4323, so the candidate's within-sign wrong-way slope is almost entirely absent. Expt3 is now overshot at 0.5083 versus real 0.3622, which suggests that when slip is switched off the first-cue inversion becomes too strong. The hard zero slip gate at first-cue validity 0.90 or higher is the likely cause of the Expt1 failure: once that gate closes, the model has no mechanism left to differentiate trials within a fixed first-cue sign. The population variances also remain too compressed: Expt1 0.0063 versus real 0.0402, Expt2 0.0090 versus 0.0713, Expt3 0.0192 versus 0.0979, and Expt4 0.0239 versus 0.0883.
Rationale: Regenerate while keeping the validity-gated inverted TTB family and the current central q values around 0.90, 0.92, and 0.95, which preserved Expt2, Expt4, and Expt7. The main needed fix is to soften the hard slip cutoff. Do not set slip probability to exactly zero at validity 0.90 or above; instead allow a small residual slip floor around 0.10 to 0.20 above 0.90, or move the upper ramp endpoint to about 0.95. This should reintroduce the late-cue dependency needed for Expt1 and may also pull Expt3 back from its current overshoot because the mixed direction of late2 and late3 slips will dilute the pure first-cue inversion. After that change, retune late2, late3, and q_low_boost. If Expt3 is still too high, nudge the central q_90 down only slightly and use the reintroduced slip, rather than first-cue inversion, to hold Expt4 near its real value. Also broaden subject-level heterogeneity directly: widen q_90, q_hi, q_mid, bias, and the slip parameters symmetrically around their current central values so per-subject variances in Expts 1-4 move toward the real range 0.04 to 0.10 instead of 0.006 to 0.024. The wider distribution should also lift Expt8 toward its real 0.12. Keep the q dip near validity 0.92 to protect Expt7, and do not add graded validity-weighted accumulation or a continuous compensatory score.

**Outcome of this advice:** iter 4 candidate loss=0.1023 -> iter 5 candidate loss=1.0000 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate never actually ran: every experiment metric in iteration 5 is n/a with an aggregate loss of 1.0000, which is the unscorable marker. This is not a fit-quality failure but a crash. Reading the predict implementation against the declared parameter block shows the immediate cause. The extraction block reads q_hi = float(parameters["q_hi"]), but the parameter dictionary contains q_95, not q_hi, so predict raises KeyError on its very first lines for every single trial. Independently, the nested q_of function builds its interpolation ordinates with q_95 in the y-array, but q_95 is never read into a local variable, so even if the KeyError were fixed the interpolation would raise a NameError whenever the first discriminating cue has validity at or below 0.97, which covers all designs here since no stated validity exceeds 0.95. With predict failing on every trial the simulated dataset is empty, so all eight metrics and their variances are n/a and no fit-quality conclusion about the SoftGate Late2-ITTB concept can be drawn from this round. Beyond the crash, the candidate also drifts well past the step size previously recommended: gate_floor is 0.38 to 0.55 where the prior advice asked for a small residual floor of roughly 0.10 to 0.20, late2_base plus late2_slope reaches 0.65 to 0.95 of trial mass in n=6 designs before gating, and max_slip allows the first cue to retain as little as a few percent of the decision mass. That configuration, if executed, would very likely overshoot Expt3, create directional asymmetry in Expt5, and risk destroying the near-zero slope in Expt7. The good elements retained from the accepted iter-4 base, namely the q(0.92) dip, the central q_90 and q_hi values, and the negligible late3 branch, are consistent with what previously fixed Expts 2, 4, and 7, and should be preserved in a corrected retry. The gate rejected this candidate, so the unchanged accepted iter-4 base at loss 0.1023 remains what the next proposal builds on.
Rationale: Regenerate on top of the accepted iter-4 base. First, fix the execution bug so the candidate is actually scorable: rename the parameter consistently, either read q_95 from the parameter dictionary and use it as the fifth interpolation ordinate in q_of, or rename the listed parameter to q_hi everywhere it is used, and remove the unused duplicate. No fit tuning matters until the simulation runs. Second, treat the substantive edit as a smaller step than what was attempted. The prior push toward a soft slip gate was never evaluated because of the crash, but do not repeat it at the attempted magnitude. Implement a modest residual gate instead: gate_low around 0.86 to 0.90, gate_hi around 0.93 to 0.96, and gate_floor around 0.05 to 0.15, so high-validity first cues retain a small late-cue branch rather than the 0.38 to 0.55 mass proposed here. Keep the late2 mass in the gated regime modest, roughly 0.15 to 0.35, with late3 near zero and a max_slip no higher than about 0.5, so the first discriminating cue remains the primary driver and Expts 2, 4, and 7 are not diluted. Preserve the iter-4 central q values around validity 0.90, the q(0.92) dip that protects Expt7, and the 0.95 point. After the fix, check that Expt1 moves toward the real -0.432 while Expt3 stays near 0.362 rather than the iter-4 overshoot of 0.508; if Expt3 is still too high, nudge q_90 down slightly rather than increasing slip mass. Finally, broaden subject-level heterogeneity only moderately, targeting per-subject variances around 0.04 to 0.10 in Expts 1 through 4, and avoid any introduction of graded weighted accumulation, which would leave the prescribed non-compensatory inverted TTB family.

**Outcome of this advice:** iter 5 candidate loss=1.0000 -> iter 6 candidate loss=0.1057 -> the gate marked it REJECTED.

### Iteration 6 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-6 candidate is scorable and fixes the target that motivated the soft gate: Expt3 is now nearly exact at 0.3517 versus real 0.3622, after the accepted iter-4 base overshot at 0.5083. Expt2 and Expt4 also remain close to the real values. However, the candidate was rejected because the soft-gate edit trades away Expt5 and still fails Expt1. Expt5 moves from the iter-4 near-zero value of -0.0008 to 0.1167 against real -0.0300, meaning the new slipped branch makes the aligned/anti-aligned pairs too asymmetric. Expt1 remains essentially flat at -0.0229 versus real -0.4323, so the core within-sign wrong-way slope is still absent. The between-subject variances are also far too compressed: candidate variances in Expts 1 through 4 are 0.0065, 0.0139, 0.0171, and 0.0234 against real 0.0402, 0.0713, 0.0979, and 0.0883. The likely structural reason is that the model only routes attention to cue-order ranks 2 and 3, never to the immediately subsequent rank-1 cue or to the remaining tail cues. That sparse rank set is exactly what creates the Expt5 alignment asymmetry and leaves too little within-sign variation for Expt1.
Rationale: Regenerate on top of the accepted iter-4 base rather than iter-6. My earlier push toward a soft validity gate with a floor of 0.05 to 0.15 was executed and rejected, so do not simply enlarge that gate. The next in-family revision should repair the slip distribution itself. In the current predict function, late2 and late3 call direction_prob with cue-order indices 2 and 3, so the second discriminating cue is skipped entirely. Add a rank-1 branch and distribute the late-cue mass across the next few ordered cues instead of only ranks 2 and 3. In Expt5 the rank-1 cue acts oppositely for the cp/ap and cn/an pairs, so including it should pull the positive 0.1167 effect back toward the real -0.0300, which in turn allows more total slip mass to be used to generate the Expt1 within-sign slope. A reasonable starting allocation is rank-1 mass 0.10 to 0.20, rank-2 mass 0.08 to 0.15, rank-3 mass 0.04 to 0.10, and a small tail mass of 0.02 to 0.06 over later cues, with total slip capped near 0.40 to 0.55 and slipped inversion q_slip around 0.78 to 0.90. Keep the first cue as the primary branch with at least 0.45 to 0.60 of decision mass. If Expt1 is still too flat, increase the tail-rank mass rather than raising the validity gate floor. Also broaden subject-level heterogeneity in the inversion curve, for example by sampling a per-subject multiplier on q values from roughly 0.70 to 1.20, so the Expt2-4 population variances move from the current 0.01 to 0.02 range toward the real 0.07 to 0.10 range. Preserve the q dip near validity 0.92 to protect Expt7, and keep the rule strictly lexicographic and non-compensatory.

**Outcome of this advice:** iter 6 candidate loss=0.1057 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Rank-distributed slipped Inverted Take-the-Best (RDS-ITTB). People order cues by stated validity and normally invert the first discriminating cue with a subject-specific probability. In six-cue choice sets, a modest subset of trials slips to immediately subsequent ordered cues, including the rank-1 cue and the tail cues, using the same inversion strength. The rank-1 slip mass is balanced against the rank-3 plus tail mass so aligned and anti-aligned pairs remain symmetric. With no discriminating cue, a subject-level bias is used, and a small lapse adds residual noise. The rule remains strictly lexicographic and non-compensatory: there is no graded validity-weighted accumulation and no continuous logit over a compensatory score.",
  "predict": "import numpy as np\n\n\ndef predict(parameters, state, history):\n    \"\"\"Rank-distributed slipped Inverted Take-the-Best (RDS-ITTB).\n\n    Cues are ordered by stated validity. The normative first\n    discriminating cue controls the first branch with inversion\n    probability q_norm. In six-cue designs, a modest share of trials\n    slips to the immediately following ordered cues, including the\n    rank-1 cue and the tail cues, all using the same inversion\n    strength q_slip. The rank-1 mass is balanced against the\n    rank-3+tail mass so aligned/anti-aligned pairs stay symmetric.\n    Trials with no discriminating cue use subject-level bias.\n    \"\"\"\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            \"RDS-ITTB expects a (2, n_features) stimulus; \"\n            f\"got shape {stim.shape}.\"\n        )\n\n    a = stim[0]\n    b = stim[1]\n\n    v = np.asarray(parameters[\"validities\"], dtype=float)\n    if v.size != a.size:\n        raise ValueError(\n            f\"validities length {v.size} does not match n_features {a.size}.\"\n        )\n\n    n = int(a.size)\n\n    q_lo = float(parameters[\"q_lo\"])\n    q_mid = float(parameters[\"q_mid\"])\n    q_90 = float(parameters[\"q_90\"])\n    q_92 = float(parameters[\"q_92\"])\n    q_hi = float(parameters[\"q_hi\"])\n    q_floor = float(parameters[\"q_floor\"])\n    bias = float(parameters[\"bias\"])\n    epsilon = float(parameters[\"epsilon\"])\n    slip_base = float(parameters[\"slip_base\"])\n    slip_gain = float(parameters[\"slip_gain\"])\n    w2 = float(parameters[\"w2\"])\n    w3 = float(parameters[\"w3\"])\n    w_tail = float(parameters[\"w_tail\"])\n    gate_low = float(parameters[\"gate_low\"])\n    q_low_boost = float(parameters[\"q_low_boost\"])\n    q_scale = float(parameters[\"q_scale\"])\n\n    def q_of(vv):\n        vv = float(vv)\n        if vv > 0.97:\n            base = float(np.clip(q_floor, 0.0, 1.0))\n        else:\n            base = float(np.interp(\n                vv,\n                np.array([0.50, 0.80, 0.90, 0.92, 0.95], dtype=float),\n                np.array([q_lo, q_mid, q_90, q_92, q_hi], dtype=float),\n            ))\n        return float(np.clip(q_scale * base, 0.0, 1.0))\n\n    cue_order = np.argsort(-v, kind=\"stable\")\n\n    def first_disc(start_idx):\n        for r in range(start_idx, n):\n            j = int(cue_order[r])\n            if a[j] != b[j]:\n                return j\n        return -1\n\n    j0 = first_disc(0)\n\n    if j0 < 0:\n        p_core = bias\n    else:\n        q_norm = q_of(v[j0])\n\n        vj0 = float(v[j0])\n        if vj0 < gate_low:\n            gate = 1.0\n        elif vj0 >= 0.90:\n            gate = 0.0\n        else:\n            gate = (0.90 - vj0) / (0.90 - gate_low)\n        gate = float(np.clip(gate, 0.0, 1.0))\n\n        # Use the distributed slip only in six-cue designs. This keeps\n        # four- and five-cue designs on the first discriminating cue and\n        # avoids a spurious validity-sum contrast in seven-cue designs.\n        if n == 6:\n            slip_total = float(np.clip(slip_base + slip_gain * gate, 0.0, 0.90))\n        else:\n            slip_total = 0.0\n\n        # Rank-1 mass balances rank-3 + tail mass, preserving\n        # aligned/anti-aligned symmetry in six-cue designs.\n        w1 = w3 + w_tail\n        branch_w = np.array([w1, w2, w3, w_tail], dtype=float)\n        branch_w /= branch_w.sum()\n\n        p0 = 1.0 - slip_total\n        q_slip = float(np.clip(q_norm + q_low_boost * gate, 0.0, 1.0))\n\n        def direction_prob(start_idx, q_val):\n            j = first_disc(start_idx)\n            if j < 0:\n                return bias\n            if a[j] > b[j]:\n                return 1.0 - q_val\n            return q_val\n\n        p_core = p0 * direction_prob(0, q_norm)\n        p_core += slip_total * branch_w[0] * direction_prob(1, q_slip)\n        p_core += slip_total * branch_w[1] * direction_prob(2, q_slip)\n        p_core += slip_total * branch_w[2] * direction_prob(3, q_slip)\n\n        tail_ranks = list(range(4, n))\n        if tail_ranks:\n            tail_contrib = sum(direction_prob(r, q_slip) for r in tail_ranks) / len(tail_ranks)\n        else:\n            tail_contrib = bias\n        p_core += slip_total * branch_w[3] * tail_contrib\n\n    p_a_core = float(np.clip(p_core, 0.0, 1.0))\n    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0\n    p_a = float(np.clip(p_a, 0.0, 1.0))\n\n    return np.array([p_a, 1.0 - p_a], dtype=float)\n",
  "policy": "import numpy as np\n\n\ndef policy(probs):\n    p = np.asarray(probs, dtype=float)\n    if p.sum() <= 0.0:\n        p = np.ones_like(p) / len(p)\n    else:\n        p = p / p.sum()\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "q_lo": "[0.20, 0.40]",
    "q_mid": "[0.55, 0.68]",
    "q_90": "[0.66, 0.82]",
    "q_92": "[0.48, 0.62]",
    "q_hi": "[0.70, 0.86]",
    "q_floor": "[0.45, 0.65]",
    "bias": "[0.30, 0.70]",
    "epsilon": "[0.02, 0.08]",
    "slip_base": "[0.45, 0.55]",
    "slip_gain": "[0.00, 0.05]",
    "w2": "[0.18, 0.22]",
    "w3": "[0.13, 0.17]",
    "w_tail": "[0.13, 0.17]",
    "gate_low": "[0.84, 0.88]",
    "q_low_boost": "[0.10, 0.25]",
    "q_scale": "[0.85, 1.15]",
    "validities": "validities"
  },
  "rationale": "This is a minimal edit of the accepted GSS-ITTB base that directly addresses the latest critique. The slipped branch is no longer restricted to cue-order ranks 2 and 3: a rank-1 branch is included, and the remaining slipped mass is distributed over the tail cues. The rank-1 mass is set equal to the rank-3 plus tail mass inside predict, which cancels the aligned/anti-aligned asymmetry that previously moved Experiment 5 away from its near-zero real value. Slip is active only for six-cue designs, which is where the within-sign slope in Experiment 1 and the over-strong first-cue effect in Experiment 3 need adjustment; seven-cue designs are kept on the first cue so Experiment 6 does not acquire a spurious validity-sum contrast, and four- and five-cue designs retain the accepted values for Experiments 2, 4, 7, and 8. A moderate q_scale multiplier broadens subject-level heterogeneity without changing the central inversion curve. The model remains lexicographic, non-compensatory, and inverted TTB throughout."
}
```

## Usage

```json
{
  "input_tokens": 21634,
  "output_tokens": 57871,
  "reasoning_tokens": 55584,
  "provider": "Baidu"
}
```
