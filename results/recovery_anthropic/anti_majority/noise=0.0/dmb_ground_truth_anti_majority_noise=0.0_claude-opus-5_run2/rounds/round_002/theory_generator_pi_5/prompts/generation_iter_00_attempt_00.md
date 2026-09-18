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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Replace PWEA (pi_3) with a NEW theory: 'Noisy Numerosity Comparison of Warning Flags (Weber Counting, WC)'.

Core claims:
1. Negative polarity and validity-blindness are retained (as in FC-LEX, and as forced by Exps 2-6): the subject forms the raw number of 1s for each option; the option with the smaller perceived count is chosen. Which experts carry the flags does not enter the comparison.
2. The counts are represented with scalar-variability / Weber noise, so discriminability depends on the RATIO (or scaled difference) of the two counts, not on the raw difference and not on a flat plateau. Concretely: p(choose fewer-flag option) = (1-eps)*Phi( (c_more - c_less) / (w * sqrt(c_more^2 + c_less^2 + c0)) ) + eps/2, with a subject-level Weber fraction w and a small additive constant c0 so that 0-vs-1 remains easy. This yields: 0-vs-1, 0-vs-5, 1-vs-4, 2-vs-4 easy (~0.8-0.9, matching Exps 3-6), but 2-vs-3 hard (~0.5-0.6, matching Exp 1's observed 0.517), which no parameterisation of FC-LEX can produce and which PWEA only reproduces by an overshoot to 0.24.
3. Exact count ties are undecidable by the counting mechanism and are resolved by a subject-level, cue-INDEPENDENT idiosyncratic bias (side/order/first-cue habit) that is the same across all equal-count pairs. This predicts exactly the ~0 Exp-2 metric (observed 0.067) that killed PWEA, while remaining agnostic about the absolute tie-choice level.
4. Optional heterogeneity term to capture Exp 1's bimodal between-subject variance (var=0.133): a subject-level probability that near-chance experts (validity below a subject cutoff, e.g. < ~0.6) are discarded before counting, so a minority of subjects reverse on 2-high-flag vs 3-low-flag pairs. Keep this fraction modest so it does not induce a large Exp-2 P1/P2 asymmetry, and make the filter discrete (all-or-none per expert), NOT graded weighting \u2014 gradedness is what Exp 2 rules out.

Parameter ranges should be chosen so that the model reproduces, in simulation: Exp1 metric ~0.5 with high between-subject spread; Exp2 metric ~0.0-0.1; Exp3 ~0.75-0.80; Exp4 ~ -0.63; Exp5 ~ -0.58; Exp6 ~ +0.66.

This new theory is a sharp competitor to FC-LEX: they agree on polarity and validity-blindness but disagree on magnitude invariance. Future experiments should therefore contrast count-ratio ladders holding the count difference fixed (0-vs-1, 1-vs-2, 2-vs-3, 3-vs-4) against difference ladders (0-vs-1, 0-vs-2, 0-vs-4), where FC-LEX predicts one flat plateau and WC predicts a monotone ratio-driven gradient.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.813)

**Description**
**Flag Counting with Lexicographic Tiebreak (FC-LEX).**

Subjects in this task do not read a binary expert rating of '1' as an endorsement; they read it as a *warning flag*. And, crucially, they do not weight those flags by the validities they were told about in the instructions. On every trial the subject performs a single, extremely cheap operation: count how many 1s each option carries, and reject the option carrying more flags.

Three claims:

1. **Negative polarity.** A '1' is a strike against an option (a flag/complaint/warning), so an option's *badness* is monotone in its number of 1s. The option with the strictly smaller flag count is chosen.

2. **Validity-blind, magnitude-blind counting.** All experts are counted equally: the validities are acknowledged in the instructions but are not converted into cue weights. Furthermore, only the *sign* of the count difference matters. Choice probability is therefore a single flat accuracy h = (1-eps)*sigmoid(beta) + eps/2 on every trial where the counts differ, regardless of whether the difference is 1 flag or 5 flags, and regardless of which experts contributed the flags. There is no psychometric function over a validity-weighted margin: a 1-vs-0 comparison and a 1-vs-5 comparison produce the same accuracy.

3. **Lexicographic tiebreak on equal counts.** When the two options carry exactly the same number of flags the counting rule is silent. The subject then falls back to a one-reason rule: locate the highest-validity expert who discriminates and let that single expert decide. Because the subject is unsure whether that expert's '1' should attract or repel (the counting rule just told them 1s are bad, but the tiebreak is invoked precisely because badness is balanced), the direction of this tiebreak is a free subject-level bias tie_bias, which sits near chance and is mildly repelled-by-1 on average. Equal-count pairs therefore yield near-chance performance, not the strongly graded reversals a weighted-mass model predicts.

The theory is deliberately non-compensatory in *magnitude* while being fully compensatory in *counting*: it agrees with a weighted-evidence account on every design tested so far (because in those designs the option with the single top-validity 1 also happens to carry the fewest flags), but it diverges sharply on equal-count pairs such as [1,0,0,0,0] vs [0,0,0,0,1], on pairs where more 1s coincides with smaller weighted mass, and on margin ladders, where FC-LEX predicts a flat accuracy plateau conditional on the sign of the count difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])
    tie_bias = float(parameters["tie_bias"])
    eps = min(max(eps, 0.0), 1.0)
    tie_bias = min(max(tie_bias, 0.0), 1.0)

    # ---------- 2. validity-blind flag count (negative polarity) ----------
    count_a = float(np.sum(a))
    count_b = float(np.sum(b))
    d = count_a - count_b   # positive => A carries MORE warning flags => reject A

    if abs(d) > 1e-9:
        # Sign-only rule: magnitude of the count difference is ignored.
        winner = 1 if d > 0 else 0   # fewer flags wins
        s = 1.0 / (1.0 + np.exp(-beta))   # sigmoid(beta * sign) for the winner
        p_core = np.zeros(2, dtype=float)
        p_core[winner] = s
        p_core[1 - winner] = 1.0 - s
    else:
        # ---------- 3. lexicographic tiebreak on equal flag counts ----------
        val = parameters.get("validities", None)
        try:
            val = np.asarray(val, dtype=float).ravel()
        except Exception:
            val = np.array([])
        if val.size != n_features:
            if n_features == 1:
                val = np.array([0.9])
            else:
                val = np.linspace(0.9, 0.55, n_features)

        order = np.argsort(-val, kind="stable")
        top1_owner = None   # index of the option carrying the '1' on the top discriminating cue
        for j in order:
            if a[j] > b[j]:
                top1_owner = 0
                break
            if b[j] > a[j]:
                top1_owner = 1
                break

        if top1_owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.zeros(2, dtype=float)
            p_core[top1_owner] = tie_bias
            p_core[1 - top1_owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------- 4. attention lapse ----------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
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
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [1.3, 2.7]
- epsilon: [0.05, 0.25]
- tie_bias: [0.10, 0.70]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7744 (var=0.0262) vs this=0.8083 (var=0.0076)
- Experiment 2: real=-0.6339 (var=0.0508) vs this=-0.6325 (var=0.0187)
- Experiment 3: real=-0.5767 (var=0.0708) vs this=-0.5608 (var=0.0168)
- Experiment 4: real=0.6600 (var=0.0633) vs this=0.6133 (var=0.0274)
- Experiment 5: real=0.5167 (var=0.1333) vs this=0.8367 (var=0.0105)
- Experiment 6: real=0.0667 (var=0.0622) vs this=-0.0117 (var=0.0319)


---

### `pi_1` (overall score: 0.555)

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
- Experiment 1: real=0.7744 (var=0.0262) vs this=0.8172 (var=0.0109)
- Experiment 2: real=-0.6339 (var=0.0508) vs this=-0.0008 (var=0.0076)
- Experiment 3: real=-0.5767 (var=0.0708) vs this=0.0012 (var=0.0066)
- Experiment 4: real=0.6600 (var=0.0633) vs this=0.0183 (var=0.0212)
- Experiment 5: real=0.5167 (var=0.1333) vs this=0.8500 (var=0.0208)
- Experiment 6: real=0.0667 (var=0.0622) vs this=-0.0100 (var=0.0127)


---

### `pi_3` (overall score: 0.552)

**Description**
**Polarised Weighted Evidence Accumulation (PWEA).**

People do NOT stop at the first discriminating cue. On every trial they integrate *all* expert ratings for each option into a single scalar impression, and then choose between the two impressions with a graded (margin-sensitive) rule.

Three claims:

1. **Compensatory integration with validity-graded weights.** Each cue j contributes w_j = ((v_j - 0.5)/0.5)^gamma to the option's score, where the weights are rescaled to have mean 1 so that the total evidence scale grows with the number of cues. The steepness exponent gamma is a free, subject-level parameter: gamma -> 0 gives equal weighting (pure tallying), gamma ~ 1 gives classic validity-proportional weighting, and large gamma approaches a lexicographic, Take-The-Best-like regime in which the top cue can outvote all the others. The family therefore *nests* both incumbent heuristics as special cases rather than opposing them.

2. **Polarity of endorsement is a free interpretive parameter s.** A binary rating of 1 is not intrinsically 'good'. It is a symbol whose desirability sign the subject must infer from the instructions/response mapping. In this task population the sign is systematically *negative*: an expert '1' is read as a flag/warning (or the A/B response mapping is inverted), so the option carrying the greater weighted mass of 1s is *rejected*. score_i = s * sum_j w_j x_ij with s < 0.

3. **Graded choice on the evidence margin, with genuinely heterogeneous evidence sensitivity.** Choice is a softmax over the two scores with inverse temperature beta, so the probability of choosing an option increases smoothly with the *magnitude* of the weighted difference (a strict prediction that one-reason stopping rules cannot make), plus an independent attention-lapse epsilon that replaces the choice with a coin flip. Exact ties in weighted score produce exactly 50/50. Critically, the effective evidence sensitivity k = beta*|s| is only *moderate* in this population and varies widely across subjects: many subjects are near-threshold on small-margin pairs, so their choices on narrow-margin conflict items are only weakly above chance, whereas the same subjects are near-deterministic on wide-margin (dominance-like) items. This within-family heterogeneity in k and gamma — not lapse alone — is what produces the between-subject dispersion of the data and keeps the pooled conflict-hit rate well below ceiling.

Because s < 0 and gamma stays in the compensatory range, the model reproduces both signature results with one mechanism: in conflict pairs the option carrying the single highest-validity 1 also carries the *smallest* weighted mass of 1s, so it is chosen (looking exactly like TTB), whereas in agreement/dominance pairs the option that both TTB and tallying favour carries the *largest* mass of 1s and is therefore rejected (looking anti-TTB). Because Exp-1 conflict margins are small and Exp-2 conflict/dominance margins are large, moderate k depresses the Exp-1 hit rate while leaving the Exp-2 contrast saturated — exactly the asymmetry the human data show.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two options
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    # ---------- 2. validity-derived weights ----------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n_features:
        # graceful fallback: linearly descending validities
        if n_features == 1:
            val = np.array([0.9])
        else:
            val = np.linspace(0.9, 0.55, n_features)

    gamma = float(parameters["gamma"])
    r = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)          # normalised validity strength
    w = np.power(r, max(gamma, 0.0))
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0.0:
        w = np.ones(n_features)
    else:
        w = w / m                                       # mean weight == 1 (scale grows with n cues)

    # ---------- 3. signed compensatory integration ----------
    s = float(parameters["polarity"])                   # negative: a '1' counts AGAINST the option
    score_a = s * float(np.dot(w, a))
    score_b = s * float(np.dot(w, b))
    scores = np.array([score_a, score_b], dtype=float)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # ---------- 4. margin-graded softmax + lapse ----------
    z = beta * scores
    z = z - np.max(z)                                    # numerically stable
    e = np.exp(z)
    p_core = e / e.sum()
    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

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
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- polarity: [-1.0, -0.35]
- gamma: [0.1, 1.15]
- beta: [2.0, 9.0]
- epsilon: [0.10, 0.55]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7744 (var=0.0262) vs this=0.7611 (var=0.0116)
- Experiment 2: real=-0.6339 (var=0.0508) vs this=-0.6858 (var=0.0254)
- Experiment 3: real=-0.5767 (var=0.0708) vs this=-0.6121 (var=0.0311)
- Experiment 4: real=0.6600 (var=0.0633) vs this=0.6550 (var=0.0311)
- Experiment 5: real=0.5167 (var=0.1333) vs this=0.2383 (var=0.0433)
- Experiment 6: real=0.0667 (var=0.0622) vs this=0.5383 (var=0.0564)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def ttb_winner(a, b):
        # features are ordered by descending validity (index 0 = 0.9, ... index 4 = 0.55)
        for j in range(len(a)):
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    def tally_winner(a, b):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            return 0
        if b_wins > a_wins:
            return 1
        return -1

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row["option_a_ratings"]]
            b = [int(v) for v in row["option_b_ratings"]]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        w_ttb = ttb_winner(a, b)
        w_tal = tally_winner(a, b)
        # keep only trials where the two heuristics make opposite, well-defined predictions
        if w_ttb < 0 or w_tal < 0 or w_ttb == w_tal:
            continue
        resp = row["response"]
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        n += 1
        if int(resp) == w_ttb:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.7744 (var=0.0262)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8172 (var=0.0109)
- pi_2: 0.1522 (var=0.0104)
- pi_3: 0.7611 (var=0.0116)
- pi_4: 0.8083 (var=0.0076)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agree_hits = []
    conflict_hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        diff = a - b
        nz = np.nonzero(diff)[0]
        if nz.size == 0:
            continue
        # features are listed in descending validity order in this design
        ttb_choice = 0 if diff[nz[0]] > 0 else 1
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin == 0:
            continue
        tally_choice = 0 if margin > 0 else 1
        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        chose_ttb = 1.0 if resp == ttb_choice else 0.0
        if tally_choice == ttb_choice and abs(margin) >= 3:
            agree_hits.append(chose_ttb)
        elif tally_choice != ttb_choice and abs(margin) >= 2:
            conflict_hits.append(chose_ttb)
    if len(agree_hits) == 0 or len(conflict_hits) == 0:
        return 0.0
    return float(np.mean(agree_hits) - np.mean(conflict_hits))
```

**Observed (real) value:** -0.6339 (var=0.0508)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7178 (var=0.0313)
- pi_1: -0.0008 (var=0.0076)
- pi_3: -0.6858 (var=0.0254)
- pi_4: -0.6325 (var=0.0187)

### Experiment 3
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def key(x):
        return ''.join([str(int(v)) for v in list(x)])

    # Class I / II 'polarity-reversal' pairs: the option favoured by the
    # top-validity expert ALSO carries the larger weighted mass of 1s.
    rev = {
        ('11111', '00000'), ('00000', '11111'),
        ('11100', '00011'), ('00011', '11100'),
        ('10000', '00000'), ('00000', '10000'),
        ('10000', '00010'), ('00010', '10000'),
    }
    # Class IV 'agreement' pairs: top-cue option carries the SMALLER mass of 1s.
    agr = {
        ('10000', '01111'), ('01111', '10000'),
        ('10000', '01100'), ('01100', '10000'),
    }

    hits_rev = []
    hits_agr = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            ka = key(a)
            kb = key(b)
        except Exception:
            continue
        pair = (ka, kb)
        in_rev = pair in rev
        in_agr = pair in agr
        if not (in_rev or in_agr):
            continue
        aa = int(list(a)[0])
        bb = int(list(b)[0])
        if aa == bb:
            continue
        winner = 0 if aa == 1 else 1
        hit = 1.0 if int(r) == winner else 0.0
        if in_rev:
            hits_rev.append(hit)
        else:
            hits_agr.append(hit)

    if len(hits_rev) == 0 or len(hits_agr) == 0:
        return 0.0
    return float(np.mean(hits_rev) - np.mean(hits_agr))
```

**Observed (real) value:** -0.5767 (var=0.0708)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0012 (var=0.0066)
- pi_3: -0.6121 (var=0.0311)
- pi_2: 0.5846 (var=0.0241)
- pi_4: -0.5608 (var=0.0168)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def as_str(x):
        return ''.join([str(int(v)) for v in list(x)])

    dom_hits = []
    conf_hits = []

    for _, row in data.iterrows():
        a = [int(v) for v in list(row['option_a_ratings'])]
        b = [int(v) for v in list(row['option_b_ratings'])]
        if len(a) != len(b) or len(a) == 0:
            continue
        # Cue order = feature index order (validities are strictly descending by design)
        winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hit = 1.0 if resp == winner else 0.0

        sa = as_str(a)
        sb = as_str(b)
        key = tuple(sorted([sa, sb]))

        # DOMINANCE base pair: [1,0,0,0,0] vs [0,0,0,0,0]  (trials 1 and 9)
        if key == ('00000', '10000'):
            dom_hits.append(hit)
        # CONFLICT base pair: [1,0,0,0,0] vs [0,1,1,1,1]  (trials 4 and 12)
        elif key == ('01111', '10000'):
            conf_hits.append(hit)

    if len(dom_hits) == 0 or len(conf_hits) == 0:
        return 0.0

    return float(np.mean(conf_hits) - np.mean(dom_hits))
```

**Observed (real) value:** 0.6600 (var=0.0633)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6550 (var=0.0311)
- pi_1: 0.0183 (var=0.0212)
- pi_2: -0.6800 (var=0.0529)
- pi_4: 0.6133 (var=0.0274)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # design-time validities (fixed by the experiment)
    val_default = np.array([0.97, 0.66, 0.60, 0.56, 0.52], dtype=float)

    def weights(n):
        if n == len(val_default):
            v = val_default.copy()
        elif n == 1:
            v = np.array([0.9])
        else:
            v = np.linspace(0.9, 0.55, n)
        r = np.clip((v - 0.5) / 0.5, 1e-6, 1.0)
        return r / float(np.mean(r))

    hits_strong = []
    hits_any = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(row["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(row["option_b_ratings"], dtype=float).ravel()
            resp = int(row["response"])
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        w = weights(a.size)
        ca = float(a.sum())
        cb = float(b.sum())
        if abs(ca - cb) < 1e-9:
            continue  # equal-count pairs are not informative here
        ma = float(np.dot(w, a))
        mb = float(np.dot(w, b))
        fewer = 0 if ca < cb else 1
        # does the option with FEWER flags carry the LARGER validity-weighted mass?
        conflict = (ma > mb) if fewer == 0 else (mb > ma)
        if not conflict:
            continue
        hit = 1.0 if resp == fewer else 0.0
        hits_any.append(hit)
        if abs(ma - mb) > 1.5:
            hits_strong.append(hit)

    if len(hits_strong) > 0:
        return float(np.mean(hits_strong))
    if len(hits_any) > 0:
        return float(np.mean(hits_any))
    return float("nan")

```

**Observed (real) value:** 0.5167 (var=0.1333)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8367 (var=0.0105)
- pi_3: 0.2383 (var=0.0433)
- pi_1: 0.8500 (var=0.0208)
- pi_2: 0.1383 (var=0.0121)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    tgt1 = tuple(sorted([(1, 0, 0, 0, 1), (0, 1, 1, 0, 0)]))
    tgt2 = tuple(sorted([(1, 0, 0, 0, 0), (0, 0, 0, 0, 1)]))

    x1 = []
    x2 = []

    for _, row in data.iterrows():
        try:
            a = tuple(int(v) for v in row["option_a_ratings"])
            b = tuple(int(v) for v in row["option_b_ratings"])
            r = int(row["response"])
        except Exception:
            continue
        if len(a) != 5 or len(b) != 5:
            continue
        key = tuple(sorted([a, b]))
        if key != tgt1 and key != tgt2:
            continue
        chosen = a if r == 0 else b
        other = b if r == 0 else a
        # feature 0 (validity 0.90) must discriminate on these pairs
        if chosen[0] == other[0]:
            continue
        hit = 1.0 if chosen[0] == 1 else 0.0
        if key == tgt1:
            x1.append(hit)
        else:
            x2.append(hit)

    if len(x1) == 0 or len(x2) == 0:
        return 0.0
    return float(np.mean(x1) - np.mean(x2))
```

**Observed (real) value:** 0.0667 (var=0.0622)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5383 (var=0.0564)
- pi_4: -0.0117 (var=0.0319)
- pi_1: -0.0100 (var=0.0127)
- pi_2: -0.0150 (var=0.0488)


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
  "theory": "**Weber Counting with Severity Tiebreak and Expert Triage (WC-ST).**\n\nA binary expert rating of '1' is read as a *warning flag*, not an endorsement, so an option's badness grows with the number of 1s it carries and the subject picks the option that seems to carry FEWER flags. Three mechanistic claims, plus one heterogeneity claim:\n\n1. **Negative polarity, validity-blind numerosity.** The decision variable is the raw *count* of flags per option, not a validity-weighted mass. Which experts contributed the flags is irrelevant to the count; the stated validities are acknowledged but never converted into cue weights. This is what makes the model look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs, with a single mechanism.\n\n2. **Scalar variability (Weber noise) on the counts.** Perceived numerosity is noisy with noise that scales with magnitude, so discriminability depends on the RATIO-scaled difference, not the raw difference:  s = (c_hi - c_lo) / sqrt(c_hi^2 + c_lo^2 + 1). The small additive constant 1 keeps 0-vs-1 easy. Crucially, the psychometric function over s is *threshold-like* (Weibull with slope exponent kappa > 1), i.e. accelerating near zero: comparisons whose count ratio falls below the subject's Weber threshold theta stay essentially at chance, while comparisons above it jump quickly to a lapse-limited ceiling. This is what produces near-chance 2-vs-3 (~0.53) at the same time as high 1-vs-2 (~0.73), 2-vs-4 (~0.78) and near-ceiling 0-vs-1 / 1-vs-4 / 0-vs-5 (~0.85) — a pattern that neither a flat sign-only rule (FC-LEX) nor a linear/Gaussian margin rule can generate.\n\n3. **Severity tiebreak on exact count ties.** When the two options carry exactly the same number of flags the counting mechanism is silent. The subject then asks which single flag is *worst*: the most reliable expert that discriminates decides, and the option carrying THAT expert's flag is (usually) rejected. The strength/direction of this fallback is a subject-level bias tie_bias = P(choose the option flagged by the top discriminating expert), sitting below 0.5 (mildly repelled). Because this rule keys on the top discriminating expert in exactly the same way for every equal-count pair, it predicts *no* differential across equal-count pairs that share the same top discriminating expert.\n\n4. **Discrete expert triage (minority).** A minority of subjects do not count everyone: they apply an all-or-none trust filter, ignoring the one or two least-valid experts entirely before counting (never a graded weighting — gradedness is ruled out by the equal-count data). Triage subjects therefore *reverse* on pairs whose flag surplus lives entirely on the discarded experts (e.g. 2 high-validity flags vs 3 low-validity flags), which injects genuine bimodal between-subject dispersion exactly where the human data show it, and creates the small positive asymmetry among equal-count pairs whose tie is broken by triage.\n\nWC-ST is a sharp competitor to sign-only flag counting: they agree on polarity and validity-blindness but disagree about magnitude invariance. WC-ST predicts a monotone *ratio-driven* accuracy gradient (0v1 > 1v2 > 2v3 > 3v4 at constant difference; 0v1 < 0v2 < 0v4 at constant smaller count), whereas the sign-only rule predicts one flat plateau.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ---------------- 1. parse stimulus ----------------\n    a = b = None\n    if isinstance(state, dict):\n        if \"option_a_ratings\" in state and \"option_b_ratings\" in state:\n            a = np.asarray(state[\"option_a_ratings\"], dtype=float).ravel()\n            b = np.asarray(state[\"option_b_ratings\"], dtype=float).ravel()\n    if a is None:\n        stim = np.asarray(state, dtype=float)\n        if stim.ndim == 1:\n            if stim.size % 2 != 0:\n                return np.ones(2) / 2.0\n            half = stim.size // 2\n            a, b = stim[:half], stim[half:]\n        else:\n            stim = stim.reshape(stim.shape[0], -1)\n            if stim.shape[0] < 2:\n                return np.ones(2) / 2.0\n            a, b = stim[0], stim[1]\n    a = np.asarray(a, dtype=float).ravel()\n    b = np.asarray(b, dtype=float).ravel()\n    if a.size == 0 or a.size != b.size:\n        return np.ones(2) / 2.0\n    n = a.size\n\n    # ---------------- 2. validities ----------------\n    val = parameters.get(\"validities\", None)\n    try:\n        val = np.asarray(val, dtype=float).ravel()\n    except Exception:\n        val = np.array([])\n    if val.size != n:\n        val = np.array([0.9]) if n == 1 else np.linspace(0.9, 0.55, n)\n\n    # ---------------- 3. subject-level parameters ----------------\n    theta = float(parameters[\"theta\"])\n    kappa = float(parameters[\"kappa\"])\n    eps = float(np.clip(float(parameters[\"epsilon\"]), 0.0, 1.0))\n    tie_bias = float(np.clip(float(parameters[\"tie_bias\"]), 0.0, 1.0))\n    ip = float(np.clip(float(parameters[\"ignore_prone\"]), 0.0, 1.0))\n    theta = max(theta, 1e-3)\n    kappa = max(kappa, 0.5)\n\n    # ---------------- 4. discrete expert triage (minority of subjects) ----\n    # most subjects count every expert; a minority discard the 1 (or 2)\n    # least-valid experts outright before counting.\n    if ip < 0.84:\n        n_drop = 0\n    elif ip < 0.96:\n        n_drop = 1\n    else:\n        n_drop = 2\n    mask = np.ones(n, dtype=bool)\n    if n_drop > 0 and n > 2:\n        order_asc = np.argsort(val, kind=\"stable\")   # ascending validity\n        dropped = 0\n        for idx in order_asc:\n            if dropped >= n_drop or (n - dropped) <= 2:\n                break\n            if val[idx] < 0.65:                       # only near-chance experts\n                mask[idx] = False\n                dropped += 1\n    af = a[mask]\n    bf = b[mask]\n    if af.size == 0:\n        af, bf, mask = a, b, np.ones(n, dtype=bool)\n\n    # ---------------- 5. Weber-noisy numerosity comparison --------------\n    ca = float(np.sum(af))\n    cb = float(np.sum(bf))\n\n    if abs(ca - cb) > 1e-9:\n        c_hi = max(ca, cb)\n        c_lo = min(ca, cb)\n        denom = np.sqrt(c_hi * c_hi + c_lo * c_lo + 1.0)\n        s = (c_hi - c_lo) / denom               # ratio-scaled evidence\n        x = (s / theta) ** kappa\n        x = float(np.clip(x, 0.0, 50.0))\n        core = 1.0 - 0.5 * np.exp(-x)           # Weibull psychometric fn\n        core = float(np.clip(core, 0.5, 1.0))\n        winner = 0 if ca < cb else 1            # fewer flags is chosen\n        p_core = np.empty(2, dtype=float)\n        p_core[winner] = core\n        p_core[1 - winner] = 1.0 - core\n    else:\n        # ------------- 6. severity tiebreak on exact count ties ----------\n        idxs = np.where(mask)[0]\n        owner = None\n        order = idxs[np.argsort(-val[idxs], kind=\"stable\")]\n        for j in order:\n            if a[j] > b[j]:\n                owner = 0\n                break\n            if b[j] > a[j]:\n                owner = 1\n                break\n        if owner is None:                        # fall back to full cue set\n            order_all = np.argsort(-val, kind=\"stable\")\n            for j in order_all:\n                if a[j] > b[j]:\n                    owner = 0\n                    break\n                if b[j] > a[j]:\n                    owner = 1\n                    break\n        if owner is None:\n            p_core = np.ones(2) / 2.0\n        else:\n            p_core = np.empty(2, dtype=float)\n            p_core[owner] = tie_bias             # option carrying the worst flag\n            p_core[1 - owner] = 1.0 - tie_bias\n\n    if not np.all(np.isfinite(p_core)):\n        p_core = np.ones(2) / 2.0\n\n    # ---------------- 7. attention lapse ----------------\n    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)\n    p = np.clip(p, 1e-12, 1.0)\n    p = p / p.sum()\n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = probs.sum()\n    if not np.isfinite(total) or total <= 0:\n        probs = np.ones(len(probs)) / len(probs)\n    else:\n        probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "theta": "[0.34, 0.46]",
    "kappa": "[4.5, 7.5]",
    "epsilon": "[0.12, 0.48]",
    "tie_bias": "[0.05, 0.45]",
    "ignore_prone": "[0, 1]",
    "validities": "validities"
  },
  "rationale": "The running-best theory (FC-LEX) nailed five metrics but failed badly on the 2-vs-3 count comparison (predicted 0.837, observed 0.517) because its accuracy is magnitude-blind: every unequal-count pair gets the same flat h. The graded-weight alternative (PWEA) fixed that cell only by over-rotating into validity-weighted evidence, which then produced a large spurious asymmetry between equal-count pairs (Exp 6: +0.54 vs observed 0.067). WC-ST resolves the dilemma with the arbiter's prescribed Weber-counting family plus two specific refinements that the arithmetic of the data demands.\n\n(i) *Threshold-shaped (Weibull), not Gaussian, psychometric function.* I checked analytically that a plain Phi() over the ratio-scaled count difference cannot simultaneously give acc(2,3)~0.52 and acc(1,2)~0.73: those two comparisons differ by only a factor 1.5 in s, so any near-linear link forces them together, collapsing Exp 1 to chance. An accelerating Weibull with kappa~6 expands that factor into a threshold non-linearity: s=0.267 (2v3) sits below threshold and stays near chance, while s>=0.41 (1v2, 2v4) is well above it and s>=0.6 (0v1, 1v4, 1v5, 0v5) is lapse-limited at ceiling. This single change is what makes the whole set of six metrics simultaneously reachable.\n\n(ii) *Severity (top-expert) tiebreak rather than a pure side bias.* Both Exp-6 target pairs are exact count ties whose top discriminating expert is feature 0, so the same tiebreak applies to both and the Exp-6 contrast comes out at ~0 — the cue-independence the arbiter demanded is achieved behaviourally without giving up the low tie-hit rate (~0.25) that Exp 3 needs for its rev cell (a coin-flip tiebreak would cost Exp 3 about 0.04).\n\n(iii) *Discrete expert triage in a small minority (~16% of subjects).* This is the arbiter's point-4 heterogeneity, implemented as an all-or-none discard of the one or two least-valid experts (never graded weighting, which Exp 2/Exp 6 rule out). Triage subjects reverse on Exp 5's [1,1,0,0,0] vs [0,0,1,1,1] pair (the flag surplus lives entirely on the discarded 0.56/0.52 experts), pushing Exp 5 down to ~0.50 and injecting the bimodal between-subject dispersion the data show (var=0.133), while also generating the small positive Exp-6 asymmetry (~+0.09 vs observed +0.067) and sharpening Exp 3 toward -0.55.\n\nHand-computation over the six designs at the centre of the declared ranges gives: Exp1 ~0.76, Exp2 ~-0.62, Exp3 ~-0.55, Exp4 ~0.69, Exp5 ~0.50, Exp6 ~0.09, against observed 0.774 / -0.634 / -0.577 / 0.660 / 0.517 / 0.067 — every cell within ~0.05, versus FC-LEX's 0.32 miss on Exp 5. The theory is also experiment-invariant by construction: nothing keys on a specific design, only on flag counts, count ratios, and validity rank order, so it makes crisp falsifiable predictions for the ratio-vs-difference ladders the arbiter proposed."
}
```

## Usage

```json
{
  "input_tokens": 19746,
  "output_tokens": 49911
}
```
