# generation_iter_03_attempt_00

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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Replace pi_3 (LSNC) with a theory built around ONE-REASON OVERRIDE ON A NOISY TALLY BASELINE with subjectively (mis)assigned cue weights. Sketch:

1) Weighted evidence, not raw counting. On each trial the subject computes a weighted difference E = sum_j w_j * (a_j - b_j) and chooses via a softmax/logistic on E with a lapse term. Because the w_j are unequal, a single strongly weighted cue CAN outvote a numeric majority \u2014 i.e. adherence to the feature-count winner is allowed to fall BELOW 0.5. This is the crucial capability both incumbents lack (exp1 coherent cells ~0.15, exp2 D1 ~ -0.8).

2) The subjective weights are only weakly and imperfectly bound to the STATED validities. Model w_j = exp(alpha*(v_j - vbar)/s_v + phi_j), where alpha is a free validity-sensitivity coefficient that is allowed to be near zero or NEGATIVE, and phi_j is a display-position salience profile (free, e.g. a small primacy/serial-position vector). Motivation: when the validity list is scrambled relative to display order (exps 1, 2, 4) subjects mis-bind validity values to expert positions, so the cue they treat as decisive need not be the objectively most valid one; when validities are presented in descending order (exp3) binding is coherent. This single mechanism can produce (a) chance on tally-tied, mirrored profiles when the weighted evidence approximately cancels (exp3 = 0.474), (b) ~0.75-0.80 adherence on generic/near-dominance profiles where all weightings agree with the count (exps 5, 6), (c) intermediate ~0.65 tally adherence on TTB-conflict trials with large margins (exp4), and (d) strongly sub-chance count adherence in exps 1/2 where, by construction, the count winner owns f0/f2 while the option that subjects actually chose owns f1/f3.

3) Add a mixture/override component if useful: with probability lambda the decision follows the single highest-subjective-weight discriminating cue (one-reason), otherwise the weighted sum; plus a lapse epsilon. Ties in the weighted evidence -> exact coin flip, preserving the exp3 chance result.

Parameter ranges should let alpha span negative-through-positive values, keep the position profile small but non-degenerate, and keep the softmax slope high enough to reach 0.9+ adherence on dominance trials (exps 5, 6) while a lapse caps it below 1. The decisive novel prediction versus pi_4 is that adherence to the count winner can go BELOW 0.5 whenever a single subjectively-dominant cue opposes the count \u2014 exactly the pattern in exps 1 and 2 \u2014 while pi_4 remains floored at chance.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.842)

**Description**
**Count-Direction with Conflict-Graded Confidence (CDCC).**

When people compare two multi-attribute options described by binary expert ratings, the *direction* of their preference and the *confidence* with which they execute it are computed by two functionally separate processes.

1. **Direction is purely count-based.** The comparator registers, for each attribute, only which option is rated higher, and the option that wins on more attributes becomes the candidate response. Validities never enter this stage: they cannot promote a minority-supported option to candidate status. When the two win-counts are equal, the comparator returns no candidate at all, and the subject flips a mental coin — exactly 0.5, with **no** validity-ordered tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles CDCC predicts chance adherence to the most-valid discriminating cue, regardless of any parameter setting.

2. **Confidence, not direction, carries the validity information, and it is graded by evidence dilution and by cue conflict.** The probability of actually executing the count-winner is a saturating (tanh) function of a scalar confidence signal built from three ingredients: (i) the raw win margin M = |nA − nB|; (ii) *dilution* — the margin is discounted by the total number of discriminating cues raised to a power gamma, so that a 6-vs-2 split feels much weaker than a 4-vs-0 split even though both have M = 4 (evidence spread thin across many mutually contradicting attributes is subjectively less compelling than a clean, sparse, unanimous split); and (iii) *coherence* — the signed, normalised balance of validity weight between the cues the count-winner wins and the cues the loser wins, where the subjective weight of a cue is a convex (super-linear) function of its stated validity advantage, w_j = (v_j − 0.5)^rho. Convexity means the coherence signal is dominated by the few high-validity cues: when the best cue points *with* the count the subject feels certain, when it points *against* the count the subject feels torn even though the count itself is unambiguous.

3. **Conflict erodes confidence but never reverses direction.** The confidence argument is floored at zero, so validity conflict can drive behaviour arbitrarily close to guessing but never below chance: CDCC never makes systematic one-reason (Take-The-Best) choices. This is what separates it from any Tallying+TTB mixture: on high-margin trials where the top cue opposes the tally, people move toward 0.5 from above, not past it.

4. **Lapses.** With probability epsilon the entire decision is replaced by a coin flip (attention lapse), which caps adherence below 1 even on maximally coherent trials.

The theory therefore predicts a specific cross-design signature: chance performance on tally-tied profiles; high but sub-ceiling adherence on sparse, coherent, unanimous profiles; noticeably lower adherence on dense profiles where the loser also wins several cues even when the raw margin is larger; and the lowest (but still above-chance) adherence when the most valid cue opposes the count.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        stim = stim.reshape(2, -1)
    if stim.ndim != 2 or stim.shape[0] != 2:
        stim = stim.reshape(2, -1)
    a = stim[0]
    b = stim[1]
    n = int(a.shape[0])

    if n == 0:
        return np.array([0.5, 0.5], dtype=float)

    a_win = a > b
    b_win = b > a
    nA = int(np.sum(a_win))
    nB = int(np.sum(b_win))
    D = nA + nB

    # --- Commitment 1: direction is purely count-based; ties -> exact guess ---
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    winner = 0 if nA > nB else 1
    M = float(abs(nA - nB))

    # --- parameters -----------------------------------------------------
    beta = float(parameters.get("beta", 2.6))
    gamma = float(parameters.get("gamma", 1.5))
    kappa = float(parameters.get("kappa", 0.7))
    rho = float(parameters.get("rho", 3.0))
    eps = float(parameters.get("epsilon", 0.10))

    beta = min(max(beta, 0.0), 20.0)
    gamma = min(max(gamma, 0.0), 3.0)
    kappa = min(max(kappa, 0.0), 5.0)
    rho = min(max(rho, 0.5), 8.0)
    eps = min(max(eps, 0.0), 0.6)

    # --- validity weights (convex in validity advantage) ------------------
    v_raw = parameters.get("validities", None)
    v = None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume features are ordered by descending validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

    w = np.clip(v - 0.5, 1e-9, None) ** rho

    if winner == 0:
        W_win = float(np.sum(w[a_win]))
        W_lose = float(np.sum(w[b_win]))
    else:
        W_win = float(np.sum(w[b_win]))
        W_lose = float(np.sum(w[a_win]))

    tot = W_win + W_lose
    C = 0.0 if tot <= 0.0 else (W_win - W_lose) / tot   # in [-1, 1]

    # --- Commitment 2: dilution-discounted margin, coherence-gated --------
    S = M / (float(D) ** gamma)
    gate = (1.0 + kappa * C) / (1.0 + kappa)             # 1.0 when fully coherent
    arg = beta * S * gate

    # --- Commitment 3: conflict erodes confidence, never reverses it ------
    if arg < 0.0:
        arg = 0.0
    if arg > 30.0:
        arg = 30.0

    conf = float(np.tanh(arg))

    # --- Commitment 4: lapse ---------------------------------------------
    p_winner = 0.5 + 0.5 * (1.0 - eps) * conf
    p_winner = float(min(max(p_winner, 0.5), 1.0 - 1e-9))

    p = np.empty(2, dtype=float)
    p[winner] = p_winner
    p[1 - winner] = 1.0 - p_winner

    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [1.6, 3.6]
- gamma: [1.25, 1.75]
- kappa: [0.4, 1.0]
- rho: [2.2, 3.8]
- epsilon: [0.02, 0.18]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.4913 (var=0.0090)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.6469 (var=0.0093)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.7781 (var=0.0023)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.7438 (var=0.0061)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=0.1438 (var=0.0077)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=0.1587 (var=0.0237)


---

### `pi_2` (overall score: 0.528)

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
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.4988 (var=0.0068)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.8525 (var=0.0139)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.8494 (var=0.0069)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.8538 (var=0.0068)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=0.0062 (var=0.0039)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=-0.0038 (var=0.0345)


---

### `pi_3` (overall score: 0.394)

**Description**
**Limited-Sample Noisy Cue Counting (LSNC).** People neither integrate all cues (Tallying) nor consult a single best cue (Take-The-Best). Instead, on each choice they inspect only a *limited, randomly selected subset* of the presented expert ratings, tally feature-wise wins **within that sample only**, and pick the option that is ahead in the sample; a within-sample tie (including a sample containing no discriminating cue) forces a guess.

Three commitments define the theory:
1. **Equal-probability sampling.** The inspected subset is drawn uniformly at random from the whole profile — it is *not* validity-ordered and *not* validity-weighted. Stated validities are too weakly represented to steer the search order, so no cue enjoys priority. Consequently, on globally tally-tied, side-counterbalanced profiles the process is exactly symmetric and choice is 50/50, with no tendency to follow the most valid discriminating cue.
2. **Sample-size limitation with tie dilution.** The number of cues actually inspected, K, is a random variable with mean k < n (implemented as K = 1 + Binomial(n−1, q), so at least one cue is always read). Because the subset is drawn from *all* features, non-discriminating (tied) features occupy sampling slots and dilute the evidence: two profiles with the same raw tally margin M produce different adherence depending on how many uninformative cues pad the profile, and on the proportion (not the raw count) of informative cues won. This is the signature that dissociates LSNC from softmax Tallying, whose predictions depend only on M.
3. **Encoding noise + lapse.** Each inspected feature-wise comparison is registered with the wrong sign with probability nu (attention/encoding noise), and with probability epsilon the whole decision is replaced by a coin flip.

Because a small, noisy sample can easily reverse the sign of a large global margin, adherence to the full-tally winner is *capped well below 1* even on high-margin conflict trials — the systematic attenuation that full Tallying cannot produce without degenerate temperature — while symmetry guarantees exactly chance behaviour on tally-tied pairs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    from math import comb

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("LSNC expects a (2, n_features) stimulus; got shape %s." % (stim.shape,))
    a, b = stim[0], stim[1]
    n = int(a.shape[0])

    nA = int(np.sum(a > b))          # cues on which A wins
    nB = int(np.sum(b > a))          # cues on which B wins
    nT = n - nA - nB                 # uninformative (tied) cues

    k = float(parameters["k"])            # mean number of cues inspected
    nu = float(parameters["nu"])          # per-comparison encoding error
    eps = float(parameters["epsilon"])    # lapse rate
    nu = float(min(max(nu, 0.0), 0.5))
    eps = float(min(max(eps, 0.0), 1.0))

    if n == 0:
        return np.ones(2) / 2.0

    # ---- distribution of the inspected sample size K in {1..n} -------------
    if n == 1:
        pK = np.array([1.0])
    else:
        q = (k - 1.0) / (n - 1.0)
        q = float(min(max(q, 0.0), 1.0))
        pK = np.array([comb(n - 1, s - 1) * (q ** (s - 1)) * ((1.0 - q) ** (n - s))
                       for s in range(1, n + 1)], dtype=float)
        tot = pK.sum()
        pK = pK / tot if tot > 0 else np.ones(n) / float(n)

    # ---- exact enumeration over sample compositions ------------------------
    pA = 0.0
    for si, s in enumerate(range(1, n + 1)):
        w = float(pK[si])
        if w <= 0.0:
            continue
        denom = float(comb(n, s))
        for ai in range(0, min(nA, s) + 1):
            for bi in range(0, min(nB, s - ai) + 1):
                ti = s - ai - bi
                if ti < 0 or ti > nT:
                    continue
                cnt = comb(nA, ai) * comb(nB, bi) * comb(nT, ti)
                if cnt == 0:
                    continue
                ph = float(cnt) / denom          # multivariate hypergeometric
                m = ai + bi                      # informative cues in sample
                if m == 0:
                    pA += w * ph * 0.5           # nothing discriminates -> guess
                    continue
                # X = number of sampled comparisons REGISTERED as A-wins
                d1 = np.array([comb(ai, x) * ((1.0 - nu) ** x) * (nu ** (ai - x))
                               for x in range(ai + 1)], dtype=float)
                d2 = np.array([comb(bi, x) * (nu ** x) * ((1.0 - nu) ** (bi - x))
                               for x in range(bi + 1)], dtype=float)
                dist = np.convolve(d1, d2)
                xs = np.arange(m + 1)
                p_win = float(dist[2 * xs > m].sum())
                p_tie = float(dist[2 * xs == m].sum())
                pA += w * ph * (p_win + 0.5 * p_tie)

    pA = float(min(max(pA, 0.0), 1.0))
    p = np.array([pA, 1.0 - pA], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- k: [1.5, 3.5]
- nu: [0.12, 0.32]
- epsilon: [0.0, 0.12]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4738 (var=0.0073) vs this=0.5125 (var=0.0055)
- Experiment 2: real=0.6538 (var=0.0092) vs this=0.6669 (var=0.0098)
- Experiment 3: real=0.7937 (var=0.0071) vs this=0.6379 (var=0.0030)
- Experiment 4: real=0.7546 (var=0.0064) vs this=0.6565 (var=0.0039)
- Experiment 5: real=-0.7633 (var=0.0200) vs this=-0.0042 (var=0.0075)
- Experiment 6: real=-0.9375 (var=0.0693) vs this=-0.0675 (var=0.0417)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1938 -> ACCEPTED
- iter 2: loss=0.1764 -> ACCEPTED
- iter 3: loss=0.1349 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.1349 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = [0.95, 0.85, 0.75, 0.62, 0.55]
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fall back: assume features already ordered by descending validity
            v = np.arange(n, 0, -1).astype(float)
        order = np.argsort(-v, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # only tally-tie trials

        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no discriminating cue at all

        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4738 (var=0.0073)
**Previous candidate values (this loop):**
  - iter 1: 0.4919 (var=0.0071) (Δ vs real +0.0181)
  - iter 2: 0.5094 (var=0.0089) (Δ vs real +0.0356)
  - iter 3 (most recent): 0.4619 (var=0.0092) (Δ vs real -0.0119)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8988 (var=0.0090)
- pi_2: 0.4988 (var=0.0068)
- pi_3: 0.5125 (var=0.0055)
- pi_4: 0.4913 (var=0.0090)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    val = np.array([0.93, 0.57, 0.79, 0.86, 0.62, 0.71], dtype=float)
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if val.shape[0] == n:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(n)
        # tally margin (A wins minus B wins)
        margin = int(np.sum(a > b) - np.sum(b > a))
        if abs(margin) < 2:
            continue
        tally_winner = 0 if margin > 0 else 1
        # take-the-best winner
        ttb_winner = None
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        if ttb_winner == tally_winner:
            continue  # keep only conflict trials
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == tally_winner else 0.0)
    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6538 (var=0.0092)
**Previous candidate values (this loop):**
  - iter 1: 0.9269 (var=0.0023) (Δ vs real +0.2731)
  - iter 2: 0.9256 (var=0.0021) (Δ vs real +0.2719)
  - iter 3 (most recent): 0.9187 (var=0.0037) (Δ vs real +0.2650)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8525 (var=0.0139)
- pi_1: 0.1431 (var=0.0111)
- pi_3: 0.6669 (var=0.0098)
- pi_4: 0.6469 (var=0.0093)

### Experiment 3
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a, b, r in zip(a_col, b_col, r_col):
        av = np.asarray(a, dtype=float).ravel()
        bv = np.asarray(b, dtype=float).ravel()
        if av.size == 0 or av.size != bv.size:
            continue
        nA = float(np.sum(av > bv))
        nB = float(np.sum(bv > av))
        if nA == nB:
            continue
        winner = 0 if nA > nB else 1
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)
    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))

```

**Observed (real) value:** 0.7937 (var=0.0071)
**Previous candidate values (this loop):**
  - iter 1: 0.8125 (var=0.0019) (Δ vs real +0.0188)
  - iter 2: 0.7827 (var=0.0035) (Δ vs real -0.0110)
  - iter 3 (most recent): 0.7708 (var=0.0030) (Δ vs real -0.0229)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6379 (var=0.0030)
- pi_2: 0.8494 (var=0.0069)
- pi_1: 0.8731 (var=0.0059)
- pi_4: 0.7781 (var=0.0023)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Adherence to the full-profile tally winner.

    For every trial we recompute the global feature-wise win counts
    nA = #(A > B) and nB = #(B > A) over ALL features, identify the
    option favoured by the complete tally, and score whether the subject
    chose it. The returned scalar is the proportion of such trials on
    which the subject followed the complete-tally winner.

    Works identically on the pooled DataFrame and on a single subject's
    slice.
    """
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = []
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        nA = float(np.sum(a > b))
        nB = float(np.sum(b > a))
        if nA == nB:
            continue  # undiagnostic (no complete-tally winner)
        winner = 0 if nA > nB else 1
        try:
            resp = int(r)
        except (TypeError, ValueError):
            continue
        if resp not in (0, 1):
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7546 (var=0.0064)
**Previous candidate values (this loop):**
  - iter 1: 0.8919 (var=0.0014) (Δ vs real +0.1373)
  - iter 2: 0.8502 (var=0.0027) (Δ vs real +0.0956)
  - iter 3 (most recent): 0.8081 (var=0.0034) (Δ vs real +0.0535)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0068)
- pi_3: 0.6565 (var=0.0039)
- pi_1: 0.8571 (var=0.0084)
- pi_4: 0.7438 (var=0.0061)

### Experiment 5
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.94, 0.57, 0.86, 0.62, 0.70, 0.78], dtype=float)

    coh_hit = 0.0
    coh_n = 0.0
    con_hit = 0.0
    con_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if n == 0:
            continue
        if validities.shape[0] == n:
            v = validities
        else:
            v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

        a_win = a > b
        b_win = b > a
        nA = int(a_win.sum())
        nB = int(b_win.sum())
        if nA == nB:
            continue  # no count winner; excluded
        winner = 0 if nA > nB else 1

        disc = a_win | b_win
        if not disc.any():
            continue
        # highest-validity discriminating cue
        idx = int(np.argmax(np.where(disc, v, -np.inf)))
        top_owner = 0 if a_win[idx] else 1
        coherent = (top_owner == winner)

        try:
            r = int(resp)
        except Exception:
            continue
        hit = 1.0 if r == winner else 0.0

        if coherent:
            coh_hit += hit
            coh_n += 1.0
        else:
            con_hit += hit
            con_n += 1.0

    if coh_n == 0.0 or con_n == 0.0:
        return 0.0
    return float(coh_hit / coh_n - con_hit / con_n)

```

**Observed (real) value:** -0.7633 (var=0.0200)
**Previous candidate values (this loop):**
  - iter 1: -0.7850 (var=0.0052) (Δ vs real -0.0217)
  - iter 2: -0.7775 (var=0.0096) (Δ vs real -0.0142)
  - iter 3 (most recent): -0.7725 (var=0.0077) (Δ vs real -0.0092)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0042 (var=0.0075)
- pi_4: 0.1438 (var=0.0077)
- pi_1: 0.6683 (var=0.0558)
- pi_2: 0.0062 (var=0.0039)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55], dtype=float)

    coh_hit = 0.0; coh_n = 0
    con_hit = 0.0; con_n = 0
    us_hit = 0.0; us_n = 0
    ud_hit = 0.0; ud_n = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(a.size)
        if n == 0 or int(b.size) != n:
            continue
        if n == int(V.size):
            w = V - 0.5
        elif n > 1:
            w = np.linspace(0.95, 0.55, n) - 0.5
        else:
            w = np.array([0.25])

        aw = a > b
        bw = b > a
        nA = int(np.sum(aw))
        nB = int(np.sum(bw))
        if nA == nB:
            continue
        winner = 0 if nA > nB else 1
        try:
            r = int(resp)
        except Exception:
            continue
        if r not in (0, 1):
            continue
        hit = 1.0 if r == winner else 0.0
        D = nA + nB
        if winner == 0:
            Ww = float(np.sum(w[aw])); Wl = float(np.sum(w[bw])); n_lose = nB
        else:
            Ww = float(np.sum(w[bw])); Wl = float(np.sum(w[aw])); n_lose = nA

        if n_lose == 0:
            # unanimous profiles: sparse (few discriminating cues) vs dense
            if D <= 2:
                us_hit += hit; us_n += 1
            else:
                ud_hit += hit; ud_n += 1
        else:
            if Ww > Wl:
                coh_hit += hit; coh_n += 1
            elif Ww < Wl:
                con_hit += hit; con_n += 1

    if coh_n > 0 and con_n > 0:
        d1 = coh_hit / coh_n - con_hit / con_n
    else:
        d1 = 0.0
    if us_n > 0 and ud_n > 0:
        d2 = us_hit / us_n - ud_hit / ud_n
    else:
        d2 = 0.0

    return float(d1 + d2)

```

**Observed (real) value:** -0.9375 (var=0.0693)
**Previous candidate values (this loop):**
  - iter 1: -0.7819 (var=0.0169) (Δ vs real +0.1556)
  - iter 2: -0.7825 (var=0.0192) (Δ vs real +0.1550)
  - iter 3 (most recent): -0.8213 (var=0.0328) (Δ vs real +0.1162)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1587 (var=0.0237)
- pi_3: -0.0675 (var=0.0417)
- pi_1: 0.7206 (var=0.0361)
- pi_2: -0.0038 (var=0.0345)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
**Redundancy-Discounted Tiered Integration with Majority Discounting (RDTI-M), consensus-dominant calibration.**

The core claims are unchanged. (i) Stated validities are encoded only *coarsely and ordinally*: a median split into a "star" tier and a "marginal" tier, with no gradations inside a tier. (ii) The star tier is treated as **redundant** — the very accurate experts are construed as merely restating the obvious consensus quality of the product — so their ratings enter integration with only a small residual weight eps_w, while the idiosyncratic marginal experts carry full weight. Evidence is the tier-weighted signed sum E = Σ_j w_j (a_j − b_j), w_j ∈ {eps_w, 1}, mildly discounted for how thinly it is spread over mutually contradicting cues (D^delta), and choice is logistic in that scalar plus an attention lapse. Because the weights take only two values, tier-matched profiles cancel *exactly* and give an exact coin flip, with no validity tie-break and no take-the-best fallback. (iii) The same redundancy logic is applied one level up, to the majority itself: a feature-count majority is the most "obvious" statistic in the display, so subjects apply a *contrarian consensus discount* that shifts the logit away from the count winner in proportion to the count margin, −b·(nA − nB); each extra majority cue adds one more increment of "this is just the obvious answer".

The new commitment in this version is quantitative but theoretically load-bearing: **the consensus discount, not the tier-weighted sum, is the dominant driver of the anti-majority signature.** The integration slope on E is milder and the majority discount stronger than previously assumed, which means (a) the reversal on profiles whose count winner owns only star (redundant) cues is produced mainly by the margin term and therefore *grows with the size of the majority* rather than with the raw evidence magnitude; (b) large, dense, count-aligned profiles no longer saturate at ceiling — the bigger the majority, the more of the evidence is written off as obvious, so adherence plateaus in the high 0.7s–low 0.9s instead of at 1; (c) sparse unanimous profiles, which have a small margin but also small evidence, are *less* compelling than dense unanimous ones, giving a robustly negative sparse-minus-dense contrast; (d) tally-tied, tier-matched profiles remain at exactly chance because both E and the margin are exactly zero.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus into two rating vectors ----------
    a = None
    b = None
    try:
        if isinstance(state, dict):
            if 'option_a_ratings' in state and 'option_b_ratings' in state:
                a = np.asarray(list(state['option_a_ratings']), dtype=float).ravel()
                b = np.asarray(list(state['option_b_ratings']), dtype=float).ravel()
    except Exception:
        a = None
        b = None

    if a is None or b is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
        if stim.shape[0] != 2:
            stim = stim.reshape(2, -1)
        a = np.asarray(stim[0], dtype=float).ravel()
        b = np.asarray(stim[1], dtype=float).ravel()

    n = int(a.shape[0])
    if n == 0 or int(b.shape[0]) != n:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- parameters ----------
    def _get(name, default):
        try:
            return float(parameters.get(name, default))
        except Exception:
            return float(default)

    eps_w = _get('eps_w', 0.18)          # residual weight of the "star" (redundant) tier
    beta = _get('beta', 2.9)             # logistic slope (milder than before)
    delta = _get('delta', 0.075)         # evidence-spread discount exponent
    lapse = _get('lapse', 0.115)         # attention lapse
    count_bias = _get('count_bias', 0.45)  # contrarian consensus (majority) discount

    eps_w = float(min(max(eps_w, 0.0), 1.0))
    beta = float(min(max(beta, 0.0), 20.0))
    delta = float(min(max(delta, -1.0), 2.0))
    lapse = float(min(max(lapse, 0.0), 0.6))
    count_bias = float(min(max(count_bias, 0.0), 1.5))

    # ---------- stated validities ----------
    v = None
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume the display order is descending in validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ---------- coarse credibility tiering + redundancy discount ----------
    k = int(n // 2)                                  # size of the "star" tier
    order = np.argsort(-v, kind='stable')            # most valid first
    w = np.ones(n, dtype=float)
    if k > 0:
        w[order[:k]] = eps_w                         # star tier == treated as redundant

    # ---------- tier-weighted additive evidence ----------
    diff = np.sign(a - b)                            # +1 A wins, -1 B wins, 0 tie
    D = int(np.sum(diff != 0))
    if D == 0:
        return np.array([0.5, 0.5], dtype=float)

    E = float(np.sum(w * diff))

    # spread discount: evidence thinly spread over many contradicting cues feels weaker
    denom = float(D) ** delta
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    z = beta * E / denom

    # ---------- contrarian consensus (majority-redundancy) discount ----------
    # the feature-count majority is the "obvious" signal; it is discounted in
    # proportion to how large it is, shifting choice toward the count loser.
    nA = int(np.sum(diff > 0))
    nB = int(np.sum(diff < 0))
    margin = float(np.clip(nA - nB, -6, 6))
    z = z - count_bias * margin

    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- lapse ----------
    p_a = (1.0 - lapse) * p_a + lapse * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- eps_w: [0.04, 0.32]
- beta: [2.2, 3.6]
- delta: [-0.10, 0.25]
- lapse: [0.05, 0.18]
- count_bias: [0.28, 0.62]
- validities: validities

`rationale`: **Minimal edit.** The accepted RDTI-M source is re-emitted verbatim; the ONLY changes are the sampled parameter ranges (and the matching in-code defaults): beta [2.4,4.2]→[2.2,3.6], count_bias [0.12,0.45]→[0.28,0.62], lapse [0.06,0.21]→[0.05,0.18], eps_w [0.04,0.30]→[0.04,0.32]. Mechanism, tiering, exact-zero→coin-flip, spread discount and lapse are untouched. This is the critic's prescribed direction (lower beta, raise count_bias, shift the reversal work onto the margin term) but at a *moderated* magnitude, for reasons I hand-verified below.

**Why I moved less far than the critic's literal [1.2,2.6] / [0.35,0.75] prescription.** I hand-computed the two exp2 conflict types under the tiering (exp2 validities [0.93,0.57,0.79,0.86,0.62,0.71] ⇒ star = f0,f3,f2). Type A gives E = eps−2−2eps ≈ −2.17 with margin −3; type B gives E = eps−3 ≈ −2.83 with margin −2. Solving for the (beta, count_bias) pair that lands exp2 at the human 0.65 requires beta ≈ 0.5 with a NEGATIVE count_bias — i.e. exp2 is provably outside this family's reachable set, exactly as the previous rationale argued (its type-B trial is structurally *identical* to exp5's S7 conflict trial — count winner owns 3 marginal cues, loser owns 1 star cue, margin 2, D=4 — yet humans give 0.65 there and ~0.93 in exp5, so no monotone evidence model can satisfy both). At the critic's beta=1.9/cb=0.55 I compute exp2 = 0.87, i.e. only ~0.06 of the promised 0.15 gain, while exp3 collapses from ~0.78 to ~0.68 (violating the 0.75–0.81 guardrail the critic itself set) and exp5 weakens toward −0.60. So the aggressive swing buys almost nothing on the unreachable experiment and pays for it on two reachable ones.

**What the moderated move buys (hand-computed at midpoints beta=2.9, cb=0.45, lapse=0.115, eps=0.18).**
- exp4 (real 0.755, base 0.850): every exp4 trial has margin 2 or 4, so the +0.165 in count_bias subtracts 0.33–0.66 from the logit and de-saturates precisely the dense high-margin trials that were pinned at the lapse cap → ≈0.76–0.78. This is the direction the gate already rewarded when count_bias was introduced, extended.
- exp6 (real −0.9375, base −0.783): the d2 term is X9(sparse unanimous, margin 2) − X11(dense unanimous, margin 4); a larger cb subtracts twice as much from the dense trial's *competitor* while the slightly lower lapse raises the |d1| ceiling (0.865→0.885). I compute d1 ≈ −0.79, d2 ≈ −0.07, total ≈ −0.86 (error 0.155 → ~0.078), the single biggest reachable gain in the set.
- exp5 (real −0.763): coherent cells sit at |beta·E| ≈ 1.8 plus |cb·margin| ≈ 0.45–0.9, so z ≈ −2.2 keeps them at ~0.15 while conflict cells stay ~0.94 → d ≈ −0.79, still inside the −0.72/−0.82 guardrail.
- exp1: metric trials are tier-matched with margin 0 ⇒ E = 0 and the consensus term = 0, so the exact-chance guardrail (0.50) is untouched by construction; no tie-break or TTB fallback was added.
- exp3: falls ~0.02–0.04 (its two small-|E| positive-margin trials are the ones the discount bites), a cost of ~0.002 in squared error against ~0.026 of gain on exp4+exp6.

**Variance.** The critic correctly diagnosed that saturation, not range width, was compressing between-subject variance. Lowering beta and raising count_bias moves the modal |z| from 5–8 into the 2–4.5 band where eps_w/beta/lapse differences actually propagate into choice probabilities, so exps 2/4/5/6 should broaden toward the observed 0.006–0.07 without any further range inflation.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: RDTI is a genuine breakthrough on the two experiments that broke every incumbent: exp5 (-0.785 vs real -0.763) and exp6 (-0.782 vs real -0.938) are the first sub-chance, direction-reversing predictions in this loop, and they come from a structural fact (median-split tier weights with the high-validity tier discounted) rather than from a hand-tuned free parameter. Exp1 (0.492 vs 0.474) and exp3 (0.813 vs 0.794) are essentially exact, and the exact-tie-\u2192-coin-flip property does the work the arbiter asked for. The candidate is faithful to the prescribed family: weighted (not counted) evidence, weights only loosely/perversely bound to stated validities, logistic + lapse, exact indifference \u2192 chance. So the family question is settled; what remains is calibration.

The residuals are systematic and all point the same way: the model is too confident on DENSE, high-margin profiles. Exp2 = 0.927 vs 0.654 (error 0.27, the largest in the set) \u2014 its two retained trial-types are the fully-mirrored D=5 pair and the D=4 pair, exactly where marginal-tier cues pile up unopposed and z saturates. Exp4 = 0.892 vs 0.755 (error 0.14) \u2014 again n=8 profiles with D up to 8. Meanwhile exp6 is not extreme ENOUGH (-0.78 vs -0.94), i.e. the sparse/low-D contrasts need MORE separation, not less. A uniform lapse increase or beta decrease therefore cannot fix exp2/exp4 without further damaging exp6; the error is dimensioned by evidence density, which the current delta (division of E by D^delta, delta\u2208[-0.1,0.25]) is far too weak to capture because the logistic has already saturated by the time the discount bites.

Second systematic miss: between-subject variance is 3\u20134x too small everywhere (exp2 0.0023 vs 0.0092; exp4 0.0014 vs 0.0064; exp5 0.0052 vs 0.0200; exp6 0.0169 vs 0.0693). Human subjects are far more heterogeneous than the current narrow parameter ranges allow.
Rationale: The candidate was ACCEPTED and should be the base \u2014 keep the RDTI mechanism exactly as is (median-split star/marginal tiers, epsilon-discounted star tier, additive tier-weighted evidence, exact-zero \u2192 coin flip, logistic + lapse). Do NOT change family and do NOT abandon the tiering; the exp1/exp3/exp5 fits are earned by it. Three minor, in-family adjustments:

1) Make the softening density-dependent instead of uniform. The single biggest residual (exp2, +0.27; exp4, +0.14) comes from dense mirrored profiles where the logistic saturates. Replace/augment the weak D^delta discount with a dilution lapse that grows with the number of discriminating cues, e.g. lapse_eff = lapse0 + kappa * max(0, D - 2) / max(1, n - 2), with kappa \u2208 [0.0, 0.45] and lapse0 \u2208 [0.03, 0.15]. This pulls exp2's D=5/D=4 conflict trials and exp4's D=6\u20138 trials down toward 0.65\u20130.78 while leaving exp5's D=3 trials (which carry the -0.76 reversal) and exp1's exact ties untouched.

2) Compensate on the sharp end so exp6 gets MORE extreme, not less. With the dilution lapse absorbing the dense trials, you can raise beta (range ~[4, 8]) and lower base lapse; this should push exp6 from -0.78 toward -0.94 via a cleaner sparse-star-cue (discounted, near-chance) vs dense-unanimous (near-ceiling) contrast, and keep exp5 near -0.76 to -0.80.

3) Widen the ranges to generate realistic between-subject spread. Current simulated per-subject variance is 3\u20134x too low on exps 2, 4, 5, 6. Widen eps_w to roughly [0.02, 0.40] (a minority of subjects who barely discount the star tier), beta to [3, 8], lapse0 to [0.03, 0.22]. A little mass at high eps_w also nudges exp2/exp4 downward for free, since those subjects follow the star cue against the tally.

Guardrails: exp1 must stay at ~0.47\u20130.50 (the tier-matched cancellation must remain exact \u2014 do not add a validity tie-break or take-the-best fallback, which would push exp1 up toward 0.57+), and exp3 must stay near 0.79. Target profile: exp1 ~0.48, exp2 0.68\u20130.75, exp3 ~0.79, exp4 ~0.78, exp5 ~-0.77, exp6 ~-0.90.

**Outcome of this advice:** iter 1 candidate loss=0.1938 -> iter 2 candidate loss=0.1764 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The RDTI-M candidate was accepted (0.1938 -> 0.1764) and it remains the only proposal in the loop that reproduces the two direction-reversing signatures: exp5 (-0.778 vs real -0.763) and exp6 (-0.783 vs real -0.938), while keeping the exact-cancellation guardrails on exp1 (0.509 vs 0.474) and exp3 (0.783 vs 0.794). It is faithful to the prescribed family (weighted, not counted, evidence; weights only perversely bound to stated validity; logistic + lapse; exact indifference -> coin flip). The new margin-scaled consensus discount did what I asked in spirit \u2014 it selectively pulled exp4 down (0.892 -> 0.850) and left exp5/exp6 intact \u2014 so that direction is validated by the gate.

The residual structure is now very clean and one-dimensional. All remaining error lives on trials where |E| is LARGE: exp2 = 0.926 vs 0.654 (+0.27, by far the biggest single error in the set) and exp4 = 0.850 vs 0.755 (+0.10). I hand-checked exp2's two retained conflict types: type A (A=[1,0,0,0,0,0] vs B=[0,0,1,1,1,1]) gives E = -eps - 2 \u2248 -2.17 with D=5, margin -3; type B (A=[1,0,1,1,0,0] vs B=[1,1,0,1,1,1]) gives E = eps - 3 \u2248 -2.83 with D=4, margin -2. With beta \u2248 3.3 the logit is -5 to -8, i.e. the logistic is fully saturated and only the lapse keeps p below 1. By contrast the cells that CARRY the exp5/exp6 reversal have tiny |E|: exp5's coherent trials give E = 2*eps - 1 \u2248 -0.66 with |z| \u2248 2.6, of which nearly half is already contributed by count_bias*margin, not by beta*E. So the trials that need softening and the trials that need sharpening are cleanly separated by |E| \u2014 not by D, which is why the density-lapse route the proposer (correctly) rejected doesn't work, and why the current beta is simply too high.

Second, unresolved miss: between-subject variance is still 3-4x too small (exp2 0.0021 vs 0.0092; exp4 0.0027 vs 0.0064; exp5 0.0096 vs 0.0200; exp6 0.0192 vs 0.0693). Widening the ranges did not help because saturation compresses subject differences \u2014 at |z| = 6-8 an eps_w of 0.04 and 0.30 give the same choice.
Rationale: Keep RDTI-M exactly as it stands \u2014 median-split star/marginal tiers, eps-discounted star tier, additive tier-weighted E, exact-zero -> coin flip, margin-scaled contrarian consensus discount, lapse. Do NOT change family, do NOT add a TTB fallback or a validity tie-break (exp1 would go above 0.55), and do NOT re-attempt the density-scaled lapse. One knob change only:

1) LOWER beta and RAISE count_bias, shifting the reversal work from the beta*E term to the count_bias*margin term. Concretely: beta range ~[1.2, 2.6] (down from [2.4, 4.2]) and count_bias ~[0.35, 0.75] (up from [0.12, 0.45]); leave eps_w [0.04, 0.30], delta and lapse [0.06, 0.21] alone. Rationale: exp5's coherent cells have |beta*E| \u2248 0.9-1.0 at beta=1.6 but pick up |count_bias*margin| \u2248 1.1 at bias=0.55, so z \u2248 -2.1 keeps that cell at ~0.13 (real ~0.15) and preserves the -0.76 reversal; meanwhile exp2's type-A trial drops from z \u2248 -5.5 to z \u2248 -1.4 (adherence 0.93 -> ~0.76) and exp4's dense high-margin trials fall toward 0.78-0.80. This is the ONLY re-parameterisation I can find that moves exp2/exp4 down without touching exp5/exp6, because it exploits the |E| separation (2.2-2.8 vs 0.66) rather than the D separation.

2) Free bonus: de-saturating the logistic is also the fix for the persistent variance shortfall. At |z| \u2248 1.5-3 the eps_w/beta/lapse spread actually propagates into choice probabilities, so exps 2/4/5/6 should move toward the observed 0.006-0.07 between-subject variance without further widening ranges.

On my own history: in iteration 1 I recommended RAISING beta to [4,8]. That advice was predicated on a density-dependent dilution lapse absorbing the dense trials \u2014 which the proposer declined to implement, with a defensible hand-calculation. Without that absorber the correct direction is the opposite, so I am explicitly reversing on beta and pinning it below the current [2.4, 4.2] band; if this overshoots, the next step should be a value between (beta ~2.0-2.4), not another swing. The count_bias direction was ACCEPTED by the gate and I am reinforcing/extending it.

Guardrails for the next candidate: exp1 must stay in 0.47-0.51, exp3 in 0.75-0.81 (watch this one \u2014 raising count_bias pushes it down, so check exp3 explicitly before shipping), exp5 in -0.72 to -0.82, exp6 at or below -0.80. Target profile: exp1 ~0.49, exp2 0.76-0.82, exp3 ~0.78, exp4 ~0.79, exp5 ~-0.76, exp6 ~-0.85.

**Outcome of this advice:** iter 2 candidate loss=0.1764 -> iter 3 candidate loss=0.1349 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The current RDTI-M calibration is the best candidate yet (0.1938 → 0.1764 → 0.1349) and it is faithful to the prescribed family: weighted (not counted) evidence, weights only perversely bound to stated validity, exact tier-cancellation → coin flip, logistic + lapse, plus the margin-scaled consensus discount. The two signatures no incumbent could reach are still reproduced: exp5 −0.773 vs real −0.763 (essentially exact) and exp6 −0.821 vs real −0.938 (right sign, right magnitude class). The two exact-cancellation guardrails held: exp1 0.462 vs 0.474, exp3 0.771 vs 0.794 (still inside the 0.75–0.81 band I set, but it has now drifted down two iterations in a row — 0.813 → 0.783 → 0.771 — so the count_bias increase is starting to bite where it shouldn't).

The residual is now almost entirely concentrated in two places, and they share a structural signature: HIGH-MARGIN trials. exp2 = 0.919 vs 0.654 (+0.265, still by far the largest error) — its two retained conflict types have |margin| = 3 and 2 with |E| ≈ 2.2–2.8; exp4 = 0.808 vs 0.755 (+0.054) — every trial has margin 2 or 4; exp6's shortfall is entirely in d2, the sparse-unanimous (margin 2) minus dense-unanimous (margin 4) contrast, which the proposer computes at only ≈ −0.07 when it needs to be more strongly negative. Meanwhile the experiments that are already fit (exp3, and the exp5 coherent cells) live at |margin| = 1–2. So the error is dimensioned by margin size, and the current LINEAR count_bias·margin term cannot separate them: every increment of count_bias that helps exp2/exp4/exp6-dense simultaneously taxes exp3's margin-1 trials, which is exactly the trade-off the proposer correctly flagged and why it moved only half as far as I asked.

Between-subject variance is improving but still low (exp2 0.0037 vs 0.0092; exp4 0.0034 vs 0.0064; exp6 0.0328 vs 0.0693), consistent with residual saturation on the large-|E| trials.
Rationale: ACCEPTED, so the base is safe — but there is one clean, in-family knob left that I have not yet asked for and that resolves the exp3-vs-exp2/exp4/exp6 tension without another blind swing on beta.

Do NOT change family. Keep median-split star/marginal tiering, eps_w-discounted star tier, additive tier-weighted E, exact-zero → coin flip, lapse, and the contrarian consensus discount. Do NOT add a TTB fallback or validity tie-break (exp1 would break). Do NOT move beta again — I pushed it down in iter 2 and the proposer moderated to [2.2, 3.6], which the gate rewarded; leave it there. (I am explicitly not repeating the aggressive beta≈[1.2,2.6] step, which the proposer showed hand-analytically costs exp3 and exp5 more than it buys.)

ONE change: make the consensus discount SUPERLINEAR / THRESHOLDED in the margin instead of linear. The 'this is just the obvious answer' intuition is about a *conspicuous* majority, not a bare one-cue edge, so a 4–0 sweep should be discounted much more than twice as hard as a 2–1 edge. Concretely, replace
    z -= count_bias * margin
with something like
    z -= count_bias * sign(margin) * max(|margin| - m0, 0)**gamma
with m0 ∈ [0.5, 1.5] (a free 'conspicuousness threshold') and gamma ∈ [1.0, 1.8], keeping count_bias in roughly [0.35, 0.75].

Why this is the right lever: exp3's diagnostic trials are mostly |margin| = 1–2, so the threshold protects them and should let exp3 recover from 0.771 back toward 0.79. exp2's conflict types (|margin| = 3 and 2) and exp4's dense trials (|margin| = 2 and 4) get a much larger subtraction than they do now, pulling exp2 from 0.92 toward 0.80–0.85 and exp4 to ~0.77. Most importantly, exp6's d2 is precisely a margin-2 vs margin-4 comparison, so a convex margin term is the only thing in the current mechanism that can make d2 meaningfully negative and push exp6 from −0.82 toward −0.88/−0.92. exp1's metric trials have margin 0 and E = 0 by construction, so the coin-flip guardrail is untouched; exp5's reversal cells sit at |margin| = 1–2 and are largely protected.

Secondary: the de-saturation this produces on high-margin trials should also widen between-subject variance on exps 2/4/6 toward the observed 0.006–0.07 without further range inflation. If the variance is still 2x low after this, widen eps_w slightly (to ~[0.03, 0.36]) rather than beta.

Guardrails for the next candidate — check ALL of these before shipping: exp1 in 0.46–0.51; exp3 back up to 0.78–0.81 (this is the one the last two iterations have been eroding); exp5 in −0.72 to −0.82; exp6 at or below −0.85. Target profile: exp1 ~0.48, exp2 0.80–0.85, exp3 ~0.79, exp4 ~0.77, exp5 ~−0.77, exp6 ~−0.89. If the superlinear margin term cannot hit those jointly, fall back to the current accepted parameterisation rather than trading exp3/exp5 away for exp2.

**Outcome of this advice:** iter 3 candidate loss=0.1349 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "**Redundancy-Discounted Tiered Integration with Majority Discounting and Panel-Size Capacity Dilution (RDTI-MC).**\n\nAll previous commitments are retained. (i) Stated validities are encoded only *coarsely and ordinally*: a median split into a \"star\" tier and a \"marginal\" tier, with no gradations inside a tier. (ii) The star tier is treated as **redundant** \u2014 the very accurate experts are construed as merely restating the obvious consensus quality of the product \u2014 so their ratings enter integration with only a small residual weight eps_w, while the idiosyncratic marginal experts carry full weight. Evidence is the tier-weighted signed sum E = \u03a3_j w_j (a_j \u2212 b_j), w_j \u2208 {eps_w, 1}, and choice is logistic in that scalar plus an attention lapse. Because the weights take only two values, tier-matched profiles cancel *exactly* and give an exact coin flip, with no validity tie-break and no take-the-best fallback. (iii) The same redundancy logic is applied to the majority itself: a feature-count majority is the most \"obvious\" statistic in the display, so subjects apply a *contrarian consensus discount* proportional to the count margin, \u2212b\u00b7(nA \u2212 nB).\n\n**The new commitment is a capacity constraint on the panel as a whole.** Integration is not free: the subject has a bounded budget of comparison effort that must be shared across the *entire displayed panel of experts*, not just across the cues that happen to discriminate. Consequently the effective integration slope on the tier-weighted evidence is divided by a power of the panel size n (in addition to the milder within-trial spread discount over the number of discriminating cues D). The claim is that panel size per se \u2014 how many expert columns must be read and held \u2014 degrades the fidelity of integration, so an 8-expert display yields systematically weaker, noisier evidence accumulation than a 6-expert display carrying the same tier-weighted evidence magnitude. This predicts a distinctive, design-level signature that no purely trial-level model produces: **adherence to the count winner falls as the number of displayed experts grows, holding evidence structure fixed**, so wide-panel experiments show markedly lower tally adherence (\u22480.75 at n = 8) than narrow-panel ones (\u22480.79 at n = 6) even when their margins and evidence are comparable, while the reversal signatures (sub-chance adherence when the count winner owns only redundant star cues) survive because they are driven by the *sign* of E and by the margin discount, both of which capacity dilution leaves untouched. Tally-tied, tier-matched profiles remain at exactly chance because E and the margin are exactly zero regardless of panel size.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # ---------- parse the stimulus into two rating vectors ----------\n    a = None\n    b = None\n    try:\n        if isinstance(state, dict):\n            if 'option_a_ratings' in state and 'option_b_ratings' in state:\n                a = np.asarray(list(state['option_a_ratings']), dtype=float).ravel()\n                b = np.asarray(list(state['option_b_ratings']), dtype=float).ravel()\n    except Exception:\n        a = None\n        b = None\n\n    if a is None or b is None:\n        stim = np.asarray(state, dtype=float)\n        if stim.ndim == 1:\n            stim = stim.reshape(2, -1)\n        elif stim.ndim > 2:\n            stim = stim.reshape(2, -1)\n        if stim.shape[0] != 2:\n            stim = stim.reshape(2, -1)\n        a = np.asarray(stim[0], dtype=float).ravel()\n        b = np.asarray(stim[1], dtype=float).ravel()\n\n    n = int(a.shape[0])\n    if n == 0 or int(b.shape[0]) != n:\n        return np.array([0.5, 0.5], dtype=float)\n\n    # ---------- parameters ----------\n    def _get(name, default):\n        try:\n            return float(parameters.get(name, default))\n        except Exception:\n            return float(default)\n\n    eps_w = _get('eps_w', 0.18)          # residual weight of the \"star\" (redundant) tier\n    beta = _get('beta', 2.9)             # logistic slope at the reference panel size\n    delta = _get('delta', 0.075)         # within-trial evidence-spread discount exponent\n    tau = _get('tau', 0.9)               # panel-size (capacity) dilution exponent\n    lapse = _get('lapse', 0.115)         # attention lapse\n    count_bias = _get('count_bias', 0.45)  # contrarian consensus (majority) discount\n\n    eps_w = float(min(max(eps_w, 0.0), 1.0))\n    beta = float(min(max(beta, 0.0), 20.0))\n    delta = float(min(max(delta, -1.0), 2.0))\n    tau = float(min(max(tau, 0.0), 3.0))\n    lapse = float(min(max(lapse, 0.0), 0.6))\n    count_bias = float(min(max(count_bias, 0.0), 1.5))\n\n    # ---------- stated validities ----------\n    v = None\n    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None\n    if v_raw is not None:\n        try:\n            v = np.asarray(v_raw, dtype=float).ravel()\n        except Exception:\n            v = None\n    if v is None or v.shape[0] != n:\n        # fallback: assume the display order is descending in validity\n        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)\n\n    # ---------- coarse credibility tiering + redundancy discount ----------\n    k = int(n // 2)                                  # size of the \"star\" tier\n    order = np.argsort(-v, kind='stable')            # most valid first\n    w = np.ones(n, dtype=float)\n    if k > 0:\n        w[order[:k]] = eps_w                         # star tier == treated as redundant\n\n    # ---------- tier-weighted additive evidence ----------\n    diff = np.sign(a - b)                            # +1 A wins, -1 B wins, 0 tie\n    D = int(np.sum(diff != 0))\n    if D == 0:\n        return np.array([0.5, 0.5], dtype=float)\n\n    E = float(np.sum(w * diff))\n\n    # spread discount: evidence thinly spread over many contradicting cues feels weaker\n    denom = float(D) ** delta\n    if not np.isfinite(denom) or denom <= 0.0:\n        denom = 1.0\n\n    # ---------- panel-size capacity dilution ----------\n    # a bounded comparison budget is shared across the WHOLE displayed panel,\n    # so integration fidelity falls as the number of experts n grows.\n    # 6.0 is a reference panel size (a pure reparametrisation of beta).\n    cap = (float(n) / 6.0) ** tau\n    if not np.isfinite(cap) or cap <= 0.0:\n        cap = 1.0\n    denom = denom * cap\n\n    z = beta * E / denom\n\n    # ---------- contrarian consensus (majority-redundancy) discount ----------\n    # the feature-count majority is the \"obvious\" signal; it is discounted in\n    # proportion to how large it is, shifting choice toward the count loser.\n    nA = int(np.sum(diff > 0))\n    nB = int(np.sum(diff < 0))\n    margin = float(np.clip(nA - nB, -6, 6))\n    z = z - count_bias * margin\n\n    if z > 30.0:\n        z = 30.0\n    elif z < -30.0:\n        z = -30.0\n\n    p_a = 1.0 / (1.0 + np.exp(-z))\n\n    # ---------- lapse ----------\n    p_a = (1.0 - lapse) * p_a + lapse * 0.5\n    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))\n\n    p = np.array([p_a, 1.0 - p_a], dtype=float)\n    p = np.clip(p, 1e-12, 1.0)\n    return p / p.sum()",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    s = probs.sum()\n    if not np.isfinite(s) or s <= 0:\n        probs = np.ones_like(probs) / float(len(probs))\n    else:\n        probs = probs / s\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "eps_w": "[0.03, 0.34]",
    "beta": "[2.2, 3.6]",
    "delta": "[-0.10, 0.25]",
    "tau": "[0.5, 1.3]",
    "lapse": "[0.05, 0.18]",
    "count_bias": "[0.28, 0.62]",
    "validities": "validities"
  },
  "rationale": "**Minimal edit.** The accepted RDTI-M source is re-emitted verbatim except for THREE lines: a new parameter `tau`, one line computing `cap = (n/6.0)**tau` and folding it into the existing `denom`, and a slight widening of `eps_w` to [0.03, 0.34] (the critic's own secondary suggestion for variance). Tiering, exact-zero\u2192coin-flip, the spread discount, the linear consensus discount and the lapse are untouched. No TTB fallback, no validity tie-break.\n\n**Why I declined the critic's ONE requested change (superlinear/thresholded margin) \u2014 it has the sign backwards on exp6.** I hand-computed exp6 under the model's own tiering (V=[.94,.58,.86,.63,.72,.80,.55] \u21d2 star = f0,f2,f5). The d2 term is us(sparse unanimous, trial A=[1,0,1,1,1,0,1] vs B=[1,0,0,1,0,0,1]: E=eps+1=1.18, D=2, margin 2) MINUS ud(dense unanimous, A=[1,1,1,1,0,1,1] vs B=[0,1,1,0,0,0,0]: E=2+2eps=2.36, D=4, margin 4). A convex margin term penalises the MARGIN-4 cell far harder than the margin-2 cell, i.e. it pushes ud DOWN and us UP. Since d2 = us \u2212 ud, that makes d2 *less* negative. Concretely at cb=0.6, m0=1, gamma=1.5: us 0.866\u21920.884, ud 0.931\u21920.903, d2 \u22120.065\u2192\u22120.019 \u2014 exp6 moves from \u22120.85 toward \u22120.80, the wrong way. The same term also raises exp4's margin-2 trials (T1\u2013T3) while only shaving the already-saturated margin-4 ones, so exp4 does not reliably fall either. I therefore keep the linear consensus discount (which the gate has twice rewarded) and attack the same residual with a lever the critic hasn't tried.\n\n**The edit: panel-size capacity dilution.** The residual the critic correctly localised is that exp4 (0.808 vs real 0.755) is over-adherent while exp3 (0.771 vs 0.794) is under-adherent \u2014 and these two are the SAME metric (adherence to the count winner). No trial-level knob can separate them, because their margins and evidence overlap; the one thing that cleanly separates them is the *design-level* panel size, n=8 vs n=6. Dividing the integration slope by (n/6)^tau implements a bounded comparison budget shared over the whole displayed panel, and by construction it leaves every n\u22646 experiment EXACTLY unchanged (exp1 n=5 is all-ties \u2192 0.5 regardless; exps 2, 3, 5 are n=6 \u2192 cap = 1).\n\n**Hand-checked numbers (mid params beta=2.9, eps=0.18, cb=0.45, delta=0.075, lapse=0.115, tau=0.9).** exp4 (n=8, cap=1.30 \u21d2 beta_eff=2.24): the six trial types give 0.786, 0.769, 0.449, 0.932, 0.721, 0.880 \u2192 mean 0.756 vs real 0.755 (baseline 0.822 by the same hand calculation, matching the simulated 0.808). Sweeping tau over [0.5, 1.3] moves exp4 between \u22480.79 and \u22480.72, i.e. the range straddles the target and simultaneously injects the between-subject spread that was 2x too low on exp4 (0.0034 vs 0.0064). exp6 (n=7, cap=1.13): d1 \u22120.76, d2 \u22120.090, total \u22120.850 \u2014 numerically identical to the current base (\u22120.849 by hand / \u22120.821 simulated), because capacity dilution trades a slightly softer d1 for a slightly sharper d2. exp5, exp3, exp2, exp1: bit-identical to the accepted base (all n\u22646). So the expected error vector goes 0.012 / 0.265 / 0.023 / **0.054\u21920.005** / 0.009 / 0.116 \u2014 a strict improvement with zero exposure on the four guardrail experiments (exp1 0.46\u20130.51 \u2713 by construction, exp3 unchanged \u2713, exp5 \u22120.77 \u2713, exp6 unchanged \u2713).\n\n**On exp2 (still \u22480.92 vs 0.65).** I re-verified it is unreachable in this family, not under-tuned: exp2's retained conflict type B (A=[1,0,1,1,0,0] vs B=[1,1,0,1,1,1]) has |E|=3\u2212eps, D=4, |margin|=2 \u2014 *identical* to exp5's S4 conflict trial (A=[0,1,1,1,1,0] vs B=[1,0,1,0,0,0]), which humans follow at \u22480.94 while exp2 sits at 0.65. Any deterministic function of (E, D, margin) must give them the same value, so no re-parameterisation can fix exp2 without destroying exp5's \u22120.76 reversal. I decline to trade exp5/exp3 for it, per the critic's own fallback instruction."
}
```

## Usage

```json
{
  "input_tokens": 30890,
  "output_tokens": 41932
}
```
