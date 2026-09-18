# generation_iter_02_attempt_00

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

Replace pi_3 with a compensatory theory that keeps its good fits on Exps 2-4 but drops the (falsified) positional re-sorting and adds relative-margin determinacy. Sketch:

**'Readability-limited validity weighting with relative-margin determinacy'.** (1) Integration is compensatory: p(A) = logistic(beta * D) with lapse epsilon, where the evidence uses subjective weights w_j proportional to logit(v_j)^gamma with gamma ~ 1.2-2.0 (steep enough that in a validity-sorted environment the top cue usually beats a coalition of 3-4 weak cues, reproducing Exp 3 ~0.75 and Exp 1's high conflict consistency).
(2) **Binding readability, not position.** Weights are flattened toward EQUAL weighting by an amount kappa that depends on how easily the stated validities can be bound to displayed columns: w_eff = (1-kappa)*w + kappa*(1/n). kappa should be a monotone function of the *incongruence/unreadability* of the display, e.g. kappa = k0 + k1*(1 - |Kendall tau(validity rank, column order)|) with an extra increment when the single most valid expert is not in an extreme (first/last) column. Crucially kappa is SMALL for any monotone display, ascending or descending (Exp 1 tau=-1 and Exp 3 tau=+1 both readable -> validity weights preserved, which is exactly where pi_3 died), and LARGE for scrambled displays (Exp 2 -> kappa near 1 gives tallying: A/B ties at 0.5, F at ~0.9, metric ~0.65 vs observed 0.659; Exp 4 -> kappa ~0.4 gives ~0.48 vs observed 0.510). This makes the theory environmentally falsifiable in a direction-free way: it predicts that REVERSING a sorted display costs nothing, while permuting it costs a lot - the opposite of pi_3.
(3) **Relative-margin determinacy.** The logistic argument is the weighted difference normalised by the total endorsement mass in the pair: D = sum_j w_eff_j (x_Aj - x_Bj) / (c + sum_j w_eff_j (x_Aj + x_Bj)), with c a small constant. This yields near-indifference when both options carry almost no endorsements (Exp 1 families G/H: [00001] vs [00000] and [10000] vs [00000] -> ~chance) while conflict families with two endorsed options stay decisive, which is the only way to get Exp 1's metric NEGATIVE (agreement < conflict, observed -0.208) - something both current theories get wrong in sign or magnitude.
(4) Population structure: heterogeneous beta and mild gamma variation (no discrete strategy switching), so between-subject variance is largest in intermediate-margin/scrambled environments (matching the observed var ~0.037 in Exps 1-2 vs ~0.01-0.02 in Exps 3-4).

Parameter ranges should be set so the model reproduces, without per-experiment refitting: Exp 1 metric in [-0.30, -0.05], Exp 2 in [0.55, 0.75], Exp 3 in [0.70, 0.82], Exp 4 in [0.45, 0.60]. Future discriminating experiments: same validities presented sorted-ascending vs sorted-descending vs scrambled (readability theory predicts ascending = descending >> scrambled; pi_3 predicted ascending << descending), and single-endorsement-vs-zero-endorsement trials versus many-endorsement trials with the same absolute weighted margin (relative-margin determinacy predicts a large accuracy drop in the former; TTB predicts none).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.258)

**Description**
**WADD-γ with Positional Validity Misattribution (WADD-γ + position-default blending), with heterogeneous decision determinacy.**

People integrate cues compensatorily: each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, L_j = log(v_j/(1−v_j)), w_j ∝ L_j^γ; the decision rests on the normalised evidence margin d = Σ_j w_j (x_Aj − x_Bj), with p(A) logistic in β·d and an attentional lapse ε. γ ≈ 1.15–1.85 places the decision maker in the near-lexicographic but still compensatory regime: the top expert usually survives a coalition of dissenters, but a near-peer second cue plus a couple of moderate cues can cancel or overturn it.

The substantive claim about *where the weights get attached* is retained and is the theory's main falsifiable commitment. Validities are stated verbally in the instructions, but the decision maker must bind each stated validity to a *column* of the displayed rating vector. This binding is costly and error-prone, and people fall back on a powerful display heuristic: **importance decreases from left to right** (the first-listed expert is the best one, the last-listed the worst). Subjective weights are therefore a blend, with mixing weight λ, of the correctly-bound validity weights and the weights implied by the positional default (the same weight multiset re-sorted into descending order across columns):

  w_eff ∝ (1−λ)·L_j^γ + λ·sort_desc(L^γ)_j.

This is an *environmental* prediction, not a per-experiment fudge: conflict-trial accuracy is a property of the alignment between display order and validity order, while the integration rule itself stays a single stationary compensatory WADD-γ. Concretely, (i) scramble the column order of an environment whose cues are currently presented in descending validity and conflict accuracy must fall toward (and, on some families, below) chance; (ii) re-sort the columns of a scrambled environment into descending validity and the same subjects must become sharply more accurate and more top-cue-consistent; (iii) an explicit column-labelling manipulation that makes the validity→column binding visually trivial should abolish the drop. Because the blend is the exact identity whenever columns are already validity-ordered, λ is behaviourally inert in validity-sorted environments and can only be identified by comparing environments — precisely the invariance signature the theory stakes itself on.

The third claim is about **population structure**: the same stationary rule is held by everyone, but decision determinacy is heterogeneous across people. Some subjects convert modest weighted margins into near-deterministic choices, others remain near chance on all but the most lopsided evidence. That heterogeneity is carried by the logistic gain β and by mild variation in the validity-sensitivity exponent γ, not by discrete strategy switching; the theory therefore predicts that between-subject spread in conflict accuracy should be largest in environments with intermediate margins and should shrink towards zero in environments where every conflict family is near-cancelling.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # ------------------------------------------------------------------
    # WADD-gamma with positional validity misattribution.
    #   w_j      proportional to  logit(v_j) ** gamma
    #   w_eff    = (1-lam) * w + lam * sort_desc(w)     (position default)
    #   d        = sum_j w_eff_j * (x_Aj - x_Bj)        in [-1, 1]
    #   p(A)     = sigmoid(beta * d), mixed with lapse epsilon.
    # gamma -> 0 recovers Tallying, gamma -> large recovers Take-The-Best.
    # lam is the degree of reliance on the display heuristic "importance
    # decreases left-to-right"; it is a no-op when the experiment already
    # presents cues in descending validity order.
    # History is not used: the rule is stationary (no feedback is given).
    # ------------------------------------------------------------------
    import numpy as np

    # ---- unpack the stimulus into two rating vectors -------------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two option vectors
            half = stim.shape[0] // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            a, b = stim[0].astype(float), stim[1].astype(float)

    n = int(min(a.shape[0], b.shape[0]))
    a = a[:n]
    b = b[:n]
    if n == 0:
        return np.ones(2) / 2.0

    # ---- subjective importance weights from cue diagnosticity ----------
    val = parameters.get("validities", None)
    if val is None:
        v = np.full(n, 0.75, dtype=float)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]

    # validities live in (0.5, 1); clip for numerical safety
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)
    logit = np.log(v / (1.0 - v))          # diagnosticity on evidence scale
    logit = np.maximum(logit, 1e-9)        # strictly positive

    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    epsilon = float(parameters.get("epsilon", 0.0))
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    lam = float(parameters.get("lam", 0.0))
    lam = float(np.clip(lam, 0.0, 1.0))

    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(logit, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=np.finfo(float).max / (10.0 * n), neginf=0.0)

    # ---- positional misattribution: blend with the left-to-right default
    # The default binds the largest importance to the first displayed cue,
    # the next largest to the second, etc.  When the experiment already
    # orders columns by descending validity this blend is the identity.
    if n > 1 and lam > 0.0:
        w_pos = np.sort(w)[::-1]
        w = (1.0 - lam) * w + lam * w_pos

    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)      # degenerate -> equal weights
    else:
        w = w / s

    # ---- weighted additive evidence margin -----------------------------
    d = float(np.dot(w, a - b))            # in [-1, 1]

    # numerically stable two-alternative softmax == logistic on beta*d
    z = np.array([beta * d, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - epsilon) * p_core + epsilon * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))

`parameters`:
- gamma: [1.15, 1.85]
- beta: [1.0, 13.0]
- epsilon: [0.0, 0.14]
- lam: [0.64, 1.00]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7533 (var=0.0228) vs this=0.7375 (var=0.0102)
- Experiment 2: real=0.5100 (var=0.0136) vs this=0.5371 (var=0.0083)
- Experiment 3: real=-0.2081 (var=0.0368) vs this=0.6219 (var=0.0370)
- Experiment 4: real=0.6587 (var=0.0379) vs this=0.8380 (var=0.0096)


---

### `pi_2` (overall score: 0.021)

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
- Experiment 1: real=0.7533 (var=0.0228) vs this=0.1425 (var=0.0084)
- Experiment 2: real=0.5100 (var=0.0136) vs this=0.3058 (var=0.0050)
- Experiment 3: real=-0.2081 (var=0.0368) vs this=0.4544 (var=0.0220)
- Experiment 4: real=0.6587 (var=0.0379) vs this=0.6260 (var=0.0084)


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
- Experiment 1: real=0.7533 (var=0.0228) vs this=0.8375 (var=0.0108)
- Experiment 2: real=0.5100 (var=0.0136) vs this=0.8204 (var=0.0123)
- Experiment 3: real=-0.2081 (var=0.0368) vs this=-0.0092 (var=0.0059)
- Experiment 4: real=0.6587 (var=0.0379) vs this=0.1800 (var=0.0087)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0503 -> ACCEPTED
- iter 2: loss=0.0475 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.0475 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # Take-The-Best winner: first discriminating cue in validity order.
        # Feature columns are already ordered by descending validity in this design.
        ttb_winner = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        # Tallying winner: majority of feature-wise wins (ties excluded).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        if tally_winner == ttb_winner:
            continue  # not a conflict trial
        resp = row['response']
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        hits += 1.0 if int(resp) == ttb_winner else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.7533 (var=0.0228)
**Previous candidate values (this loop):**
  - iter 1: 0.8346 (var=0.0107) (Δ vs real +0.0813)
  - iter 2 (most recent): 0.8021 (var=0.0134) (Δ vs real +0.0488)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8375 (var=0.0108)
- pi_2: 0.1425 (var=0.0084)
- pi_3: 0.7375 (var=0.0102)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Fixed design validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65
    val = np.array([0.93, 0.58, 0.86, 0.51, 0.74, 0.65], dtype=float)

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape[0] != b.shape[0]:
            continue
        if a.shape[0] == val.shape[0]:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(a.shape[0])

        tally = int(np.sum(a > b)) - int(np.sum(b > a))

        winner = None
        for j in order:
            j = int(j)
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no cue discriminates: uninformative for both models

        ttb_sign = 1 if winner == 0 else -1
        # Keep only trials where the tally does NOT favour the TTB winner:
        # either the tally is exactly tied (Family D) or it points the
        # other way (Family E, directional conflict).
        if tally * ttb_sign > 0:
            continue

        n += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == winner:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5100 (var=0.0136)
**Previous candidate values (this loop):**
  - iter 1: 0.5500 (var=0.0052) (Δ vs real +0.0400)
  - iter 2 (most recent): 0.5242 (var=0.0060) (Δ vs real +0.0142)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3058 (var=0.0050)
- pi_1: 0.8204 (var=0.0123)
- pi_3: 0.5371 (var=0.0083)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = [0.55, 0.62, 0.70, 0.85, 0.93]

    def key(v):
        return ''.join([str(int(round(float(x)))) for x in v])

    # conflict families: TTB (top-validity cue) and a left-to-right positional
    # weighting scheme point to OPPOSITE options
    conflict_keys = {
        '10000|00001', '00001|10000',      # family A
        '01000|00010', '00010|01000',      # family B
        '11000|00011', '00011|11000',      # family E
        '00100|11000', '11000|00100',      # family F
    }
    # agreement families: both accounts favour the same option
    agree_keys = {
        '11111|00000', '00000|11111',      # family D (dominance)
        '00001|00000', '00000|00001',      # family G
        '10000|00000', '00000|10000',      # family H
    }

    def ttb_winner(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    n_conf = 0
    hit_conf = 0
    n_agree = 0
    hit_agree = 0

    for _, row in data.iterrows():
        try:
            a = [int(round(float(x))) for x in row['option_a_ratings']]
            b = [int(round(float(x))) for x in row['option_b_ratings']]
            resp = int(row['response'])
        except Exception:
            continue
        n = len(a)
        if len(b) != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fallback: assume columns are displayed in ascending validity
            v = np.linspace(0.55, 0.93, n)
        order = list(np.argsort(-v, kind='stable'))
        w = ttb_winner(a, b, order)
        if w < 0:
            continue
        k = key(a) + '|' + key(b)
        if k in conflict_keys:
            n_conf += 1
            hit_conf += 1 if resp == w else 0
        elif k in agree_keys:
            n_agree += 1
            hit_agree += 1 if resp == w else 0

    if n_conf == 0 or n_agree == 0:
        return 0.0

    return float(hit_agree) / n_agree - float(hit_conf) / n_conf

```

**Observed (real) value:** -0.2081 (var=0.0368)
**Previous candidate values (this loop):**
  - iter 1: -0.2094 (var=0.0076) (Δ vs real -0.0014)
  - iter 2 (most recent): -0.1936 (var=0.0171) (Δ vs real +0.0144)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0092 (var=0.0059)
- pi_3: 0.6219 (var=0.0370)
- pi_2: 0.4544 (var=0.0220)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (display order): col0..col5
    v_default = np.array([0.60, 0.93, 0.55, 0.86, 0.68, 0.78], dtype=float)

    def build(n):
        v = v_default.copy()
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
        v = np.clip(v, 0.5 + 1e-6, 1 - 1e-6)
        L = np.log(v / (1.0 - v))
        w = np.power(L, 1.5)
        wpos = np.sort(w)[::-1]
        weff = 0.2 * w + 0.8 * wpos          # lam = 0.8, mid of advocated box
        s = weff.sum()
        weff = weff / s if s > 0 else np.ones(n) / n
        cascade = list(np.argsort(-v, kind="stable"))
        return weff, cascade

    cache = {}
    num = 0.0
    den = 0.0

    for a_raw, b_raw, resp in zip(data["option_a_ratings"],
                                  data["option_b_ratings"],
                                  data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        if n not in cache:
            cache[n] = build(n)
        weff, cascade = cache[n]

        d = float(np.dot(weff, a - b))

        winner = None
        for j in cascade:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        if abs(d) < 0.20:                    # ambiguous / near-cancelling for WADD-pos
            continue
        pos_choice = 0 if d > 0 else 1
        if pos_choice == winner:             # models agree -> uninformative
            continue

        den += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        num += 1.0 if r == pos_choice else 0.0

    if den == 0:
        return 0.5
    return float(num / den)
```

**Observed (real) value:** 0.6587 (var=0.0379)
**Previous candidate values (this loop):**
  - iter 1: 0.6553 (var=0.0137) (Δ vs real -0.0033)
  - iter 2 (most recent): 0.6093 (var=0.0109) (Δ vs real -0.0493)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8380 (var=0.0096)
- pi_1: 0.1800 (var=0.0087)
- pi_2: 0.6260 (var=0.0084)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Readability-limited validity weighting with dissent aversion and endorsement discount (RVW-DE), with broad determinacy heterogeneity.**

A single stationary, fully compensatory integration rule underlies choice in all cue-based binary-feature environments; the apparent switching between 'take-the-best-like' and 'tallying-like' behaviour across environments is produced by modulators that are properties of the *display and the evidence pattern*, not of the strategy, plus wide between-person variation in decision determinacy.

(1) **Graded validity weighting.** Each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, w_j ∝ logit(v_j)^gamma, with gamma spread over ~1.1–2.0 across people. At the population mean this is steep enough that a clearly best expert usually survives a coalition of three or four weak experts (near-lexicographic behaviour), yet weak enough that a near-peer plus a couple of moderate cues can overturn the top cue — and people at the low-gamma end of the population are genuinely closer to tallying without ever switching rule.

(2) **Binding readability, not position.** Stated validities must be bound to displayed columns. When the display order is *monotone* in validity (ascending OR descending) this binding is trivial and the validity weights are used essentially intact. When the order is scrambled, binding fails partially and the weights are flattened toward EQUAL weighting: w_eff = (1-kappa)·w + kappa/n, with kappa = k0 + k1·(1-|Kendall tau(column index, validity)|) + k2·[best expert is not in an extreme column]. This is direction-free: reversing a sorted display costs nothing, permuting it costs a lot (the opposite of positional-misattribution accounts, which ascending-validity environments falsify). Superimposed is a *mild* left-to-right attention gradient phi that scales, but never re-sorts, the effective weights — a reading-order effect, not a validity misattribution.

(3) **Dissent aversion (consensus premium).** An option about which the experts are unanimous is perceived as a known quantity; conflicting testimony makes an option's quality ambiguous. Subjects add a direction-blind premium rho·c(x) with c(x) = (2·mean(x)-1)^4, so only near-unanimity matters. This predicts that a product unanimously rated 0 can be *preferred* to a product carrying a single weak endorsement — accuracy on 'one-endorsement vs zero-endorsement' pairs can fall below chance while genuine two-endorsement conflict pairs stay decisive and dominance pairs are untouched.

(4) **Endorsement discount.** Each positive endorsement carries a small credibility cost theta (faint-praise skepticism), inert whenever the two options carry equal endorsement counts.

(5) **Population structure.** Everyone holds the same rule; heterogeneity is *broad* and continuous — in decision determinacy beta (some people convert modest margins into near-deterministic choices, others stay near chance on all but lopsided evidence), in gamma, in lapse, and in the modulator magnitudes rho, theta and phi. There is no discrete strategy switching, yet the resulting between-subject spread in any conflict-consistency statistic is large (sd ~0.10-0.15) in every environment whose scored families sit at intermediate margins.

Evidence for A: D = Σ_j w_eff_j (x_Aj - x_Bj) - theta·(Σx_A - Σx_B) + rho·(c(x_A) - c(x_B)); p(A) = (1-eps)·logistic(beta·D) + eps/2.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Readability-limited validity weighting with dissent aversion and
    # endorsement discount (RVW-DE).
    #   w_j        proportional to logit(v_j) ** gamma
    #   w_eff      = (1-kappa)*w + kappa/n          (readability flattening)
    #   w_eff     *= (1 + phi * reading-order gradient)   (mild primacy)
    #   D          = w_eff.(a-b) - theta*(sum a - sum b) + rho*(coh(a)-coh(b))
    #   p(A)       = (1-eps)*logistic(beta*D) + eps/2
    # kappa is driven by how incongruent the column order is with the
    # stated validity order (direction-free: monotone displays are cheap,
    # scrambled displays are expensive).  coh(x) = (2*mean(x)-1)**4 is a
    # steeply convex consensus premium: only near-unanimity matters.
    # History is unused: no feedback is given, the rule is stationary.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = arr.shape[0] // 2
            a = arr[:half].astype(float)
            b = arr[half:half + half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float).ravel()[:n]
    b = np.asarray(b, dtype=float).ravel()[:n]

    # ---------------- validities --------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.9, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters --------------------------------------
    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    eps = float(np.clip(float(parameters.get("eps", 0.08)), 0.0, 1.0))
    k0 = float(parameters.get("k0", 0.07))
    k1 = float(parameters.get("k1", 0.80))
    k2 = float(parameters.get("k2", 0.35))
    phi = float(parameters.get("phi", 0.16))
    theta = float(parameters.get("theta", 0.035))
    rho = float(parameters.get("rho", 0.32))

    # ---------------- validity weights --------------------------------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(L, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(w))
    if (not np.isfinite(s)) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)
    else:
        w = w / s

    # ---------------- readability flattening --------------------------
    # Kendall tau between column index and stated validity (ties skipped).
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)

    jmax = int(np.argmax(v))
    off_extreme = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    kappa = k0 + k1 * incong + k2 * off_extreme
    kappa = float(np.clip(kappa, 0.0, 1.0))

    w_eff = (1.0 - kappa) * w + kappa * (1.0 / float(n))

    # ---------------- mild reading-order (primacy) gradient -----------
    if n > 1 and phi != 0.0:
        idx = np.arange(n, dtype=float)
        grad = 1.0 + phi * ((float(n - 1) - 2.0 * idx) / float(n - 1))
        grad = np.maximum(grad, 1e-6)
        w_eff = w_eff * grad
    ssum = float(np.sum(w_eff))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        w_eff = np.ones(n, dtype=float) / float(n)
    else:
        w_eff = w_eff / ssum

    # ---------------- evidence ----------------------------------------
    core = float(np.dot(w_eff, a - b))

    # endorsement discount: fixed credibility cost per positive rating
    cost = -theta * (float(np.sum(a)) - float(np.sum(b)))

    # dissent aversion: steeply convex consensus premium
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    coh_a = (2.0 * ma - 1.0) ** 4
    coh_b = (2.0 * mb - 1.0) ** 4
    cons = rho * (coh_a - coh_b)

    D = core + cost + cons

    # ---------------- logistic choice with lapse -----------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))

`parameters`:
- gamma: [1.10, 1.95]
- beta: [1.2, 13.0]
- eps: [0.0, 0.16]
- k0: [0.02, 0.14]
- k1: [0.65, 0.95]
- k2: [0.20, 0.50]
- phi: [0.05, 0.28]
- theta: [0.0, 0.07]
- rho: [0.18, 0.46]
- validities: validities

`rationale`: Minimal-diff edit: `predict` and `policy` are re-emitted VERBATIM; only the parameter boxes (and the matching in-code fallback defaults, which are never used when parameters are supplied) change. This directly implements the critic's two targeted knobs on the accepted base.

(1) **Shave top-cue determinacy.** gamma [1.35, 2.10] -> [1.10, 1.95] and beta [3.0, 16.0] -> [1.2, 13.0] (mean beta 9.5 -> 7.1). Hand-computing Exp 1's four conflict families with the weight vector the model builds for a descending 5-cue display (w_eff ≈ [.539,.291,.104,.047,.019] after the phi gradient) gives evidence margins D ≈ {0.167, 0.122, 0.69, 0.146}; averaging logistic(beta*D) over beta ~ U[1.2,13] (logistic is concave in beta for D>0, so widening downward lowers the mean) yields family mean ≈ 0.78, and with lapse eps ~ U[0,0.16] (mean .08) the pooled conflict consistency lands ≈ 0.76 vs 0.835 previously and 0.753 observed — removing the single largest point-estimate error. The same shift pulls Exp 2 from 0.550 toward ~0.52 (observed 0.510) and Exp 4 slightly toward 0.5, which the widened phi box (see below) offsets.

(2) **Protect Exp 3's negative sign while adding noise.** Extra lapse/noise compresses agreement-minus-conflict toward 0, so, exactly as the critic prescribed, I compensate by nudging rho UP rather than raising beta: rho [0.18, 0.36] -> [0.18, 0.46] (mean .27 -> .32). With rho=0.32 the ascending-display families give D_G ≈ +0.14, D_H ≈ -0.28 (single-endorsement pairs sit at/below chance) while conflict families A/B/E/F keep |D| = 0.10-0.68, so the raw difference is ≈ -0.225 and, after the mean lapse factor 0.92, ≈ -0.21 (observed -0.208). Dominance and all mixed-profile pairs are untouched because c(x) is quartic.

(3) **Fix the 2-5x under-dispersion.** Heterogeneity is carried entirely by widening continuous boxes — no discrete strategy mixture. beta now spans near-chance subjects (beta~1.5 -> Exp 1 metric ~0.57) to near-deterministic ones (beta~13 -> ~0.87); theta [0.015,.05] -> [0.0, 0.07] matters a lot in Exp 1 (it enters conflict margins as +3theta, i.e. 0 to 0.21 of extra evidence for the frugal winner) and rho/phi/kappa breadth adds spread in the scrambled environments. phi [0.08,0.22] -> [0.05,0.28] and the wider k0/k1/k2 boxes generate the between-subject spread in the two scrambled-display metrics (Exps 2 and 4) that the previous narrow boxes suppressed, while leaving the pooled means near their (already excellent) values because phi's mean is essentially unchanged.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a strong first pass and clearly the best-fitting theory so far: it is the only one that recovers the SIGN and near-exact magnitude of the Exp 3 agreement-minus-conflict metric (-0.209 vs -0.208 observed) while simultaneously landing Exp 4's scrambled positional metric almost dead on (0.655 vs 0.659). The dissent-aversion / consensus-premium term is doing exactly the work the arbiter's relative-margin term was supposed to do (near-chance or sub-chance on the single-endorsement-vs-nothing families G/H while dominance stays at ~1 and two-endorsement conflict families stay decisive), and the direction-free readability flattening keeps the good Exp 2 and Exp 4 behaviour without reintroducing pi_3's falsified positional re-sorting. So the substitution in ingredient (3) is justified and faithful to the spirit of the prescribed family (compensatory weighted-additive with readability-driven flattening); it is not a family switch.

Two residual mismatches remain, both systematic and both in the same direction. (i) The model is too deterministic in the validity-weighted direction in the monotone-descending 5-cue environment: Exp 1 conflict consistency is 0.835 vs 0.753 observed (+0.081, the single largest point-estimate error), and Exp 2 is likewise slightly high (0.550 vs 0.510). Both errors are consistent with gamma and/or beta sitting a notch too high, i.e. the top cue beats weak coalitions a bit too reliably. (ii) Between-subject variance is uniformly under-produced by roughly a factor of 2-5: Exp 1 0.011 vs 0.023, Exp 2 0.005 vs 0.014, Exp 3 0.008 vs 0.037, Exp 4 0.014 vs 0.038. The population structure claimed in point (5) of the theory is therefore not actually being realised at the sampled parameter breadth \u2014 the parameter boxes are too narrow, so simulated subjects behave nearly identically. Since var is scored alongside the point estimate, this is the largest remaining source of loss and it is present in all four experiments, not just one.
Rationale: The candidate was ACCEPTED and should be the base; I am asking only for small in-family recalibration, not a redesign. Keep the exact predict() structure (gamma-steepened logit weights, direction-free kappa flattening from 1-|tau| plus the off-extreme increment, mild non-reordering phi gradient, quartic consensus premium rho, endorsement discount theta, logistic + lapse). Two targeted knobs: (1) Shave the top-cue determinacy slightly \u2014 lower the gamma box from [1.35, 2.10] to roughly [1.15, 1.85] and shift the beta box down modestly (e.g. [2.5, 13]) so that Exp 1 conflict consistency falls from 0.835 toward ~0.76 and Exp 2 from 0.55 toward ~0.52. Verify while doing so that Exp 3 stays in [-0.25, -0.15] and Exp 4 in [0.60, 0.70]; if the extra noise starts pulling Exp 3 toward 0, compensate by nudging rho up within its existing box rather than by raising beta again. (2) Fix the under-dispersion: the between-subject variance is 2-5x too small in every experiment, so widen the per-subject sampling breadth of the heterogeneity-carrying parameters (beta especially, plus gamma, rho and eps) so that individual subjects genuinely differ in how near-lexicographic they are \u2014 a wide beta box (some subjects near beta~2, some near beta~14) will raise var in Exp 1/3/4 toward the observed 0.02-0.04 without moving the pooled means much, because the metric is close to linear in p over the relevant range. Do NOT introduce discrete strategy mixtures to buy variance; that would drift out of the prescribed single-rule compensatory family. This is the first iteration, so no prior advice conflicts; if the widened-beta / lower-gamma step is rejected by the gate, the next attempt should keep the widened heterogeneity but restore gamma nearer its current midpoint.

**Outcome of this advice:** iter 1 candidate loss=0.0503 -> iter 2 candidate loss=0.0475 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: Iteration 2 was ACCEPTED (0.0503 -> 0.0475) and the advice from iteration 1 (shave top-cue determinacy, widen the heterogeneity boxes) clearly moved things in the right direction on three of the four point estimates: Exp 1 went 0.835 -> 0.802 (observed 0.753), Exp 2 0.550 -> 0.524 (observed 0.510), and Exp 3 stayed on-sign at -0.194 (observed -0.208). Between-subject variance also rose everywhere (Exp 3 0.0076 -> 0.0171, Exp 1 0.0107 -> 0.0134), confirming that the population structure claimed in point (5) is finally doing some work. The candidate remains by far the best experiment-invariant account on the table: it is the only theory that gets Exp 3's NEGATIVE agreement-minus-conflict sign (pi_1 ~0, pi_2 +0.45, pi_3 +0.62) while keeping Exp 2 near 0.5 and Exp 4 in the 0.6s, i.e. the direction-free readability flattening plus the quartic consensus premium are carrying the cross-environment pattern, not per-experiment tuning.

Two residuals remain, and they now pull in OPPOSITE directions, which is diagnostic. (i) Exp 1 is still +0.049 too high: in the monotone-descending 5-cue display the model lets the top cue beat 3-4 weak cues more reliably than people do. (ii) Exp 4 has gone from 0.655 (near-perfect) to 0.609, i.e. -0.049 too LOW, and Exp 2 also drifted downward. In other words, the global determinacy cut (lower beta/gamma applied uniformly) bought the Exp 1 gain by paying almost exactly the same amount in the scrambled 6-cue environment. Further uniform noise is therefore exhausted as a knob \u2014 the next gain has to come from a modulator that separates monotone displays (Exps 1, 3) from scrambled ones (Exps 2, 4). (iii) Between-subject variance is still 2-3.5x too small in every experiment (Exp 1 0.013 vs 0.023, Exp 2 0.006 vs 0.014, Exp 3 0.017 vs 0.037, Exp 4 0.011 vs 0.038); given that var is scored alongside the point estimate this is now the single largest remaining loss component, and it is worst precisely in the two scrambled environments where the readability parameters (kappa components, phi) ought to be generating spread but evidently are not.
Rationale: Stay strictly inside the prescribed family (compensatory logistic integration over gamma-steepened logit weights, direction-free readability flattening kappa from 1-|tau| plus the off-extreme increment, non-reordering reading-order gradient phi, quartic consensus premium rho, endorsement discount theta, lapse). Do NOT re-introduce positional re-sorting and do NOT add discrete strategy mixtures. Three targeted, minor edits:

(1) STOP lowering beta globally. My iteration-1 push on beta/gamma was accepted, but its marginal product is now zero-sum: it fixed part of Exp 1 and broke Exp 4 by the same amount. Keep the beta box roughly where it is ([1.2, 13.0]) \u2014 do not shift its midpoint down again.

(2) Make the reading-order gradient READABILITY-DEPENDENT instead of constant. Replace phi with phi_eff = phi * (p0 + p1*kappa) (e.g. p0 ~ 0.3, p1 ~ 1.5, or simply phi_eff = phi*kappa/kappa_max), so the left-to-right gradient is weak when the display is monotone (Exps 1 and 3, kappa small) and strong when binding fails (Exps 2 and 4, kappa large). Mechanistically this is the same story the theory already tells \u2014 when validities cannot be bound to columns, people fall back on reading order \u2014 and it is not a re-sorting mechanism. Predicted effect: Exp 4 recovers from 0.609 back toward 0.65-0.66 and Exp 2 nudges toward 0.51, while Exp 1 and Exp 3 are essentially untouched (in Exp 1 the current constant phi is actively inflating the top-cue weight in the descending display, so removing it there also helps the +0.049 Exp 1 error).

(3) Attack the residual Exp 1 excess with a mechanism that is inert in Exp 3 rather than with more global noise. Two candidates, in order of preference: (a) shave only the LOW end of gamma (box [1.00, 1.90] instead of [1.10, 1.95]) so that some subjects genuinely tally in the 5-cue coalition families while the Exp 3 single-cue-vs-single-cue conflicts (whose sign is gamma-independent) are barely affected; (b) allow theta to span slightly negative values (e.g. [-0.04, 0.06]) \u2014 note that in Exp 1's conflict families theta currently ADDS +3*theta of evidence to the frugal (TTB) winner, so a small negative tail acts as a per-subject 'endorsement-count pull' that lowers Exp 1 without touching equal-count trials elsewhere. If Exp 3 drifts toward 0 as a result, compensate with rho (its box already reaches 0.46) rather than beta.

(4) Variance is now the biggest single loss term. Confirm the readability parameters (k0, k1, k2, phi) are actually resampled PER SUBJECT, and widen the ones that bite hardest in the scrambled displays (k1, k2, phi, and rho) so that Exp 2 and Exp 4 subject-level metrics spread to sd ~0.15-0.19 (var 0.03-0.04). Target envelope for the next candidate: Exp 1 in [0.75, 0.79], Exp 2 in [0.50, 0.53], Exp 3 in [-0.24, -0.18], Exp 4 in [0.63, 0.68], with variances at least ~1.5x their current values in all four.

**Outcome of this advice:** iter 2 candidate loss=0.0475 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Readability-limited validity weighting with readability-gated reading-order fallback, dissent aversion and endorsement discount (RVW-DE+), with broad determinacy heterogeneity.**\n\nA single stationary, fully compensatory integration rule underlies choice in all cue-based binary-feature environments; the apparent switching between 'take-the-best-like' and 'tallying-like' behaviour across environments is produced by modulators that are properties of the *display and the evidence pattern*, not of the strategy, plus wide between-person variation in decision determinacy.\n\n(1) **Graded validity weighting.** Each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, w_j \u221d logit(v_j)^gamma, with gamma spread over ~1.05\u20132.0 across people. At the population mean this is steep enough that a clearly best expert usually survives a coalition of three or four weak experts (near-lexicographic behaviour), yet weak enough that a near-peer plus a couple of moderate cues can overturn the top cue \u2014 and people at the low-gamma end of the population are genuinely closer to tallying without ever switching rule.\n\n(2) **Binding readability, not position.** Stated validities must be bound to displayed columns. When the display order is *monotone* in validity (ascending OR descending) this binding is trivial and the validity weights are used essentially intact. When the order is scrambled, binding fails partially and the weights are flattened toward EQUAL weighting: w_eff = (1-kappa)\u00b7w + kappa/n, with kappa = k0 + k1\u00b7(1-|Kendall tau(column index, validity)|) + k2\u00b7[best expert is not in an extreme column]. This is direction-free: reversing a sorted display costs nothing, permuting it costs a lot (the opposite of positional-misattribution accounts, which ascending-validity environments falsify).\n\n(3) **Readability-gated reading-order fallback.** The left-to-right attention gradient is not a fixed perceptual bias but a *fallback that is recruited only when validity-to-column binding fails*: phi_eff = phi\u00b7(r0 + r1\u00b7kappa). In a readable (monotone) display people have a usable importance ordering and barely rely on reading order; in a scrambled display, where no cue can be confidently bound to its stated validity, attention defaults to reading order and the gradient becomes substantial. Crucially the gradient only *scales* weights \u2014 it never re-sorts them \u2014 so it is not positional misattribution. This makes a sharp prediction: column-order effects on choice should be near-zero in validity-sorted displays (either direction) and large in scrambled displays, and an explicit column-labelling manipulation should abolish them entirely.\n\n(4) **Dissent aversion (consensus premium).** An option about which the experts are unanimous is perceived as a known quantity; conflicting testimony makes an option's quality ambiguous. Subjects add a direction-blind premium rho\u00b7c(x) with c(x) = (2\u00b7mean(x)-1)^4, so only near-unanimity matters. A product unanimously rated 0 can therefore be *preferred* to a product carrying a single weak endorsement \u2014 accuracy on 'one-endorsement vs zero-endorsement' pairs falls below chance while genuine two-endorsement conflict pairs stay decisive and dominance pairs are untouched.\n\n(5) **Endorsement discount / pull.** Each positive endorsement carries a small credibility adjustment theta (faint-praise skepticism for most people, a mild endorsement-count pull for some), inert whenever the two options carry equal endorsement counts.\n\n(6) **Population structure.** Everyone holds the same rule; heterogeneity is *broad* and continuous \u2014 in decision determinacy beta, in gamma, in lapse, and in the modulator magnitudes rho, theta, phi and the readability coefficients. There is no discrete strategy switching, yet between-subject spread in conflict-consistency statistics is large in every environment whose scored families sit at intermediate margins, and largest in scrambled displays where the readability parameters themselves vary between people.\n\nEvidence for A: D = \u03a3_j w_eff_j (x_Aj - x_Bj) - theta\u00b7(\u03a3x_A - \u03a3x_B) + rho\u00b7(c(x_A) - c(x_B)); p(A) = (1-eps)\u00b7logistic(beta\u00b7D) + eps/2.",
  "predict": "def predict(parameters, state, history):\n    # Readability-limited validity weighting with dissent aversion and\n    # endorsement discount (RVW-DE+).\n    #   w_j        proportional to logit(v_j) ** gamma\n    #   w_eff      = (1-kappa)*w + kappa/n          (readability flattening)\n    #   w_eff     *= (1 + phi_eff * reading-order gradient)\n    #                with phi_eff = phi*(r0 + r1*kappa): the reading-order\n    #                fallback is recruited only when binding fails.\n    #   D          = w_eff.(a-b) - theta*(sum a - sum b) + rho*(coh(a)-coh(b))\n    #   p(A)       = (1-eps)*logistic(beta*D) + eps/2\n    # kappa is driven by how incongruent the column order is with the\n    # stated validity order (direction-free: monotone displays are cheap,\n    # scrambled displays are expensive).  coh(x) = (2*mean(x)-1)**4 is a\n    # steeply convex consensus premium: only near-unanimity matters.\n    # History is unused: no feedback is given, the rule is stationary.\n    import numpy as np\n\n    # ---------------- unpack the two rating vectors --------------------\n    a = None\n    b = None\n    if isinstance(state, dict):\n        if \"option_a_ratings\" in state and \"option_b_ratings\" in state:\n            a = np.asarray(state[\"option_a_ratings\"], dtype=float).ravel()\n            b = np.asarray(state[\"option_b_ratings\"], dtype=float).ravel()\n    if a is None:\n        arr = np.asarray(state, dtype=float)\n        if arr.ndim == 1:\n            half = arr.shape[0] // 2\n            a = arr[:half].astype(float)\n            b = arr[half:half + half].astype(float)\n        else:\n            arr = arr.reshape(arr.shape[0], -1)\n            a = arr[0].astype(float)\n            b = arr[1].astype(float)\n\n    n = int(min(np.size(a), np.size(b)))\n    if n == 0:\n        return np.array([0.5, 0.5])\n    a = np.asarray(a, dtype=float).ravel()[:n]\n    b = np.asarray(b, dtype=float).ravel()[:n]\n\n    # ---------------- validities --------------------------------------\n    val = parameters.get(\"validities\", None)\n    if val is None:\n        v = np.linspace(0.9, 0.55, n)\n    else:\n        v = np.asarray(val, dtype=float).ravel().astype(float)\n        if v.shape[0] < n:\n            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])\n        v = v[:n]\n    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)\n\n    # ---------------- parameters --------------------------------------\n    gamma = float(parameters.get(\"gamma\", 1.5))\n    beta = float(parameters.get(\"beta\", 7.0))\n    eps = float(np.clip(float(parameters.get(\"eps\", 0.09)), 0.0, 1.0))\n    k0 = float(parameters.get(\"k0\", 0.08))\n    k1 = float(parameters.get(\"k1\", 0.78))\n    k2 = float(parameters.get(\"k2\", 0.35))\n    phi = float(parameters.get(\"phi\", 0.20))\n    theta = float(parameters.get(\"theta\", 0.02))\n    rho = float(parameters.get(\"rho\", 0.34))\n\n    # readability-gating of the reading-order fallback (fixed structure)\n    r0 = 0.35\n    r1 = 1.15\n\n    # ---------------- validity weights --------------------------------\n    L = np.log(v / (1.0 - v))\n    L = np.maximum(L, 1e-9)\n    with np.errstate(over=\"ignore\", invalid=\"ignore\"):\n        w = np.power(L, gamma)\n    w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)\n    s = float(np.sum(w))\n    if (not np.isfinite(s)) or s <= 0.0:\n        w = np.ones(n, dtype=float) / float(n)\n    else:\n        w = w / s\n\n    # ---------------- readability flattening --------------------------\n    # Kendall tau between column index and stated validity (ties skipped).\n    conc = 0\n    disc = 0\n    for i in range(n - 1):\n        for j in range(i + 1, n):\n            d = v[j] - v[i]\n            if d > 0:\n                conc += 1\n            elif d < 0:\n                disc += 1\n    tot = conc + disc\n    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)\n    incong = 1.0 - abs(tau)\n\n    jmax = int(np.argmax(v))\n    off_extreme = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0\n\n    kappa = k0 + k1 * incong + k2 * off_extreme\n    kappa = float(np.clip(kappa, 0.0, 1.0))\n\n    w_eff = (1.0 - kappa) * w + kappa * (1.0 / float(n))\n\n    # ------- reading-order (primacy) gradient, gated by readability ----\n    if n > 1 and phi != 0.0:\n        phi_eff = phi * (r0 + r1 * kappa)\n        phi_eff = float(np.clip(phi_eff, 0.0, 0.45))\n        idx = np.arange(n, dtype=float)\n        grad = 1.0 + phi_eff * ((float(n - 1) - 2.0 * idx) / float(n - 1))\n        grad = np.maximum(grad, 1e-6)\n        w_eff = w_eff * grad\n    ssum = float(np.sum(w_eff))\n    if (not np.isfinite(ssum)) or ssum <= 0.0:\n        w_eff = np.ones(n, dtype=float) / float(n)\n    else:\n        w_eff = w_eff / ssum\n\n    # ---------------- evidence ----------------------------------------\n    core = float(np.dot(w_eff, a - b))\n\n    # endorsement discount: credibility adjustment per positive rating\n    cost = -theta * (float(np.sum(a)) - float(np.sum(b)))\n\n    # dissent aversion: steeply convex consensus premium\n    ma = float(np.mean(a))\n    mb = float(np.mean(b))\n    coh_a = (2.0 * ma - 1.0) ** 4\n    coh_b = (2.0 * mb - 1.0) ** 4\n    cons = rho * (coh_a - coh_b)\n\n    D = core + cost + cons\n\n    # ---------------- logistic choice with lapse -----------------------\n    z = np.array([beta * D, 0.0], dtype=float)\n    z = z - np.max(z)\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n\n    p = (1.0 - eps) * p_core + eps * 0.5\n    p = np.clip(p, 1e-12, 1.0)\n    p = p / p.sum()\n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64).ravel()\n    p = np.clip(p, 0.0, None)\n    tot = p.sum()\n    if (not np.isfinite(tot)) or tot <= 0.0:\n        p = np.ones_like(p) / float(len(p))\n    else:\n        p = p / tot\n    return int(np.random.choice(len(p), p=p))",
  "parameters": {
    "gamma": "[1.05, 1.95]",
    "beta": "[1.2, 13.0]",
    "eps": "[0.0, 0.20]",
    "k0": "[0.02, 0.16]",
    "k1": "[0.55, 1.00]",
    "k2": "[0.10, 0.60]",
    "phi": "[0.02, 0.38]",
    "theta": "[-0.03, 0.07]",
    "rho": "[0.14, 0.54]",
    "validities": "validities"
  },
  "rationale": "Minimal-diff edit on the accepted base: `policy` is verbatim, `predict` changes only the three lines of the reading-order gradient block (plus fallback defaults), and the parameter boxes are widened. Beta's box is left EXACTLY where it was, as the critic demanded (further global determinacy cuts are zero-sum).\n\n(1) **Readability-gated reading-order fallback (the one mechanistic edit).** phi is replaced by phi_eff = phi*(0.35 + 1.15*kappa), capped at 0.45. This implements the critic's item (2) with the theory's own story: reading order is a *fallback* recruited only when validity-to-column binding fails, never a re-sorting. Concretely, computing kappa for each environment with mid-box coefficients: Exp 1 (descending 5-cue, tau=-1, top cue extreme) kappa\u22480.08 \u2192 phi_eff\u22480.44*phi (gradient cut ~56%); Exp 3 (ascending 5-cue) likewise kappa\u22480.08 \u2192 phi_eff small; Exp 2 (tau=-0.33, top cue in col0) kappa\u22480.61 \u2192 phi_eff\u22481.05*phi (essentially UNCHANGED, so the good 0.524 is preserved); Exp 4 (tau=+0.07, best cue in col1 \u2192 off-extreme increment fires) kappa\u21921.0 \u2192 phi_eff\u22481.50*phi. This is exactly the separation the critic asked for: it pushes Exp 4 back up from 0.609 toward ~0.65 (the scrambled metric scores agreement with a strongly left-weighted profile, and phi is the only source of that once kappa saturates), leaves Exp 2 flat, and *lowers* Exp 1 because the constant phi was previously inflating the leftmost (= most valid) column in the descending display.\n\n(2) **Exp 1 excess attacked with mechanisms inert in Exp 3.** (a) gamma's low end shaved to 1.05 so some subjects genuinely tally in the 5-cue coalition families [10000]v[01111] and [11000]v[10111], whose margins are already tiny (D\u22480.05-0.07 at mid-gamma); Exp 3's conflict families are single-cue-vs-single-cue and their sign is gamma-insensitive. (b) theta now spans negative values, [-0.03, 0.07]: in Exp 1's conflict families theta enters as +2\u03b8 or +3\u03b8 of extra evidence for the frugal winner, so the negative tail creates per-subject 'endorsement-count pull' subjects who go against TTB there, while all equal-endorsement-count trials (most of Exps 2-4 and Exp 3's conflict families A/B/E) are untouched.\n\n(3) **Exp 3 sign protection.** Reducing phi in the ascending display raises the true top cue's weight, which pushes conflict accuracy up (metric more negative) but also raises family-G accuracy; to keep the net comfortably in [-0.24,-0.18] rho's box is nudged up and widened to [0.14, 0.54] (mean 0.34 vs 0.32), since rho is the sole driver of the sub-chance single-endorsement families G/H that make the metric negative at all.\n\n(4) **Variance, the largest remaining loss term.** All heterogeneity parameters are resampled per subject by construction (they are declared parameter boxes). The boxes that bite in the scrambled displays are widened \u2014 k1 [0.55,1.00], k2 [0.10,0.60], phi [0.02,0.38], rho [0.14,0.54], eps [0,0.20] \u2014 and because phi_eff is now multiplied by kappa, the phi spread and the kappa spread compound multiplicatively in Exps 2 and 4, which is precisely where subject-level spread was most deficient (0.006 and 0.011 vs 0.014 and 0.038). In Exps 1 and 3 the widened theta, gamma and eps ranges supply the extra spread. No discrete strategy mixture is introduced; the population remains a single rule with continuous parameter variation."
}
```

## Usage

```json
{
  "input_tokens": 26770,
  "output_tokens": 12989
}
```
