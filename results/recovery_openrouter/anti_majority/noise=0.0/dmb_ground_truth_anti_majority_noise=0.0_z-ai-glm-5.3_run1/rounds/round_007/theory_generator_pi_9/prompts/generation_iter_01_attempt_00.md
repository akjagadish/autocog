# generation_iter_01_attempt_00

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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_8_1`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Replace pi_7 with a 'Heavy-Tailed Graded Dual-Route Distrust' (HT-GDRD) theory that preserves the empirically validated core shared with the winner — endorsements as validity-scaled liabilities d_j = c0 + c1*v_j^kappa accumulated into a suspicion difference passed through sigmoid(beta*D), mixed with a graded fewer-endorsement route sigmoid(gamma*(cnt_B-cnt_A)) and a uniform lapse — but replaces BOTH pi_7's independent-uniform population and pi_8_1's tightly clipped bimodal bands with an explicitly over-dispersed, three-component population: (i) a skeptic majority (~60-70%) with a WIDE lapse range including genuinely low lapse, epsilon in [0.05, 0.25] (this alone repairs Exp 15's observed 0.28 ceiling prevalence and Exp 1's low-lapse subjects, and adds real dispersion to Exps 13/14 saturation cells); (ii) an ENLARGED counter minority (~20-25%, up from 13%) with route weight w in [0.30, 0.60], low lapse epsilon in [0.05, 0.15], and steep count slope gamma >= 2.5 — sized to reproduce Exp 2's observed SD 0.143, Exp 1's extreme population-max membership score ~0.98, and Exp 16's excess dispersion ~0.115 rather than GDRD-B's 0.059; and (iii) a small residual 'trust-leaning' mode (~8-12%) with a positive micro-trust term alpha*logit(v_j) dominant, capturing subjects who follow high-validity endorsers — which the data hint at in Exp 15 and in the observed variance of Exps 13/14 that neither distrust-only population reproduces. The mixture should be implemented with genuinely heavy-tailed (e.g., Beta-like, not narrow clipped) within-mode spreads rather than the tight 2-3-point bands GDRD-B uses, since every variance statistic in the data exceeds even GDRD-B's prediction by 1.5-6x. Parameter ranges must be re-tuned so the pooled means on Exps 3-12 (which both current theories already match well: e.g., -0.26, -0.55, -0.19, 0.78, 0.37, 0.51) are preserved while the second-moment statistics (Exps 1, 2, 13-16) are raised to the observed levels; the key degrees of freedom for this are the counter-mode mass and its w-floor, the skeptic lapse floor (0.05), and within-mode spread widths. This yields a theory that is a direct, falsifiable competitor to pi_8_1 on exactly the axes (tail mass, lapse floor, dispersion magnitude) where pi_8_1 currently underpredicts the data, rather than another variant of the falsified independent-box design.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_8_1` (overall score: 0.872)

**Description**
Graded Dual-Route Distrust with a Bimodal Strategy Population (GDRD-B). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic distrust): every endorsement is a liability whose magnitude grows monotonically in the endorser's RAW claimed validity, d_j = c0 + c1*v_j^kappa; accumulated suspicion P(X) = sum_j x_j*d_j is compared across options through sigmoid(beta*(P(B)-P(A))), plus a small residual micro-trust edge alpha*sum_j clip(logit(v_j),0,3)*(a_j-b_j) (capped so perfect experts exert bounded, not unbounded, trust pull). ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is BIMODAL in the route weight w: a large majority (~88%) of 'skeptics' (w in [0.85, 0.95], moderate lapse) coexists with a minority (~12%) of 'counters' (w in [0.50, 0.62], low lapse, decisive count slope gamma >= 2). The mixture is implemented as a quantile transform of a single uniform per-subject population variable u_pop, which holds the population-MEAN route weight at its base value while exploding the between-subject behavioral variance — the empirically observed 3-12x super-binomial variance on the saturation assays (Exps 11, 12) that no uniform range-widening could reproduce.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Graded Dual-Route Distrust with a Bimodal Strategy Population
    # (GDRD-B) -- neutrality-coupled skeptic-mode variant, final tuning
    # pass per the validated iter-4 critique: skeptic epsilon lowered
    # to [0.15, 0.23] (decompresses the Exp-3 contrast, raises the
    # Exp-13 ceiling prevalence toward 0.28) with the neutrality
    # pairing re-solved over the new range, and the counter w range
    # restored to the iter-2-validated [0.42, 0.56] (pulls Exp 6 back
    # toward 0.21). Everything else is unchanged.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # ROUTE 1 (analytic distrust, weight w): every endorsement is a
    # LIABILITY whose magnitude grows with the endorser's RAW claimed
    # validity:
    #   d_j = c0 + c1 * v_j^kappa          (kappa >= 1, no dampening
    #                                        near v = 0.5)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = (P(B) - P(A)) + alpha * sum_j clip(logit(v_j), 0, 3)*(a_j - b_j)
    #     (micro-trust term; the logit is CAPPED at 3 so perfect
    #      experts exert bounded, not unbounded, trust pull)
    #   P_skeptic(A) = sigmoid(beta * D)
    # ROUTE 2 (graded counting, weight 1-w): a CONTINUOUS preference
    #   for the option with fewer total endorsements:
    #   P_count(A) = sigmoid(gamma * (cnt_B - cnt_A))
    #
    # BIMODAL POPULATION: a single uniform population quantile u_pop
    # in [0, 1] maps onto a two-component mixture:
    #   u_pop < 0.13  -> 'counter' minority:  w in [0.42, 0.56],
    #                    epsilon in [0.12, 0.22], gamma in [2.0, 3.5]
    #   otherwise     -> 'skeptic' majority:   epsilon in [0.15, 0.23],
    #                    w COUPLED to epsilon along the neutrality curve
    #                        w = 1 - (0.25 - eps)/(1 - eps)
    #                    (clipped to [0.85, 0.95]) plus a tiny independent
    #                    per-subject spread w_spread in [-0.02, 0.02],
    #                    gamma in [1.5, 3.5].
    #
    # NEUTRALITY COUPLING: a skeptic's Exp-14 excess-dispersion
    # contribution is v = (x-0.21)^2 - (r-0.96)^2 with
    # x = (1-eps)(1-w) + eps/2 and r = 1 - eps/2. Setting x - 0.21 =
    # (r - 0.96) gives exactly (1-eps)(1-w) = 0.25 - eps, i.e.
    # w = 1 - (0.25 - eps)/(1 - eps). Tying w to eps along this curve
    # makes every skeptic's v approximately ZERO at ANY lapse level.
    # Re-solving over the new eps range [0.15, 0.23]: eps = 0.15 ->
    # w ~ 0.88, eps = 0.23 -> w ~ 0.97 (clipped at 0.95 -- the clip
    # only makes skeptic v slightly positive, the safe direction for
    # Exp 14). A key structural property: along the curve the
    # saturated-conflict compliance (1-eps)*w + eps/2 = 0.75 + eps/2
    # is approximately eps-invariant (~0.83-0.87 over the whole
    # range), so the excellent Exp 11/12 means are protected while
    # the epsilon reduction decompresses the lapse-sensitive assays
    # (Exp 3) and raises the Exp-13 ceiling prevalence.
    #
    # History is ignored: validities are communicated in the
    # instructions, so the distrust weights are fixed for the whole
    # block. beta is sampled per subject from [1.3, 2.4] (the range
    # validated by the iter-3 beta revert, which repaired Exps 3/10).
    if isinstance(state, dict):
        a_vec = np.asarray(state["option_a_ratings"], dtype=float).ravel()
        b_vec = np.asarray(state["option_b_ratings"], dtype=float).ravel()
        stim = np.vstack([a_vec, b_vec])
    else:
        stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GDRD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Power route: v = 1 is harmless (no logs). Trust route: guard
    # the logit against v = 1 (perfect experts) and v < 0.5.
    v_pow = np.clip(v, 0.5, 1.0)
    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])
    u_pop = float(parameters["u_pop"])
    w_spread = float(parameters["w_spread"])

    # --- Bimodal population mixture over (w, epsilon, gamma) ---
    # Quantile transform of one uniform variable. No within-mode
    # jitters (they failed twice: contaminating Exp 6 and Exp 13
    # while moving no variance metric). Only a tiny +/-0.02 w spread
    # around the skeptic neutrality curve remains.
    f_counter = 0.13
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority: near-balanced routes, low lapse, decisive
        # graded count slope (gamma >= 2, per the theory's prose).
        # w restored to the iter-2-validated [0.42, 0.56]: the higher-w
        # counters of the iter-4 range were less count-decisive on
        # Exp 6's gap/coalition cells, elevating that assay.
        t = u / f_counter
        w_mix = 0.42 + 0.14 * t          # [0.42, 0.56]
        epsilon = 0.12 + 0.10 * t        # [0.12, 0.22]
        gamma = 2.0 + 1.5 * t            # [2.0, 3.5]
    else:
        # Skeptic majority: analytic-distrust dominant. The route
        # weight is COUPLED to the lapse along the Exp-14 neutrality
        # curve w = 1 - (0.25 - eps)/(1 - eps). The epsilon range is
        # LOWERED from [0.18, 0.27] to [0.15, 0.23] (mean ~0.19):
        # this decompresses the lapse-sensitive Exp-3 contrast back
        # toward -0.50/-0.52 and raises the Exp-13 ceiling prevalence
        # toward the observed 0.28, while the coupling keeps every
        # skeptic's Exp-14 v near zero and holds the saturated
        # compliance ~0.75 + eps/2 (eps-invariant along the curve),
        # protecting the Exp 11/12 means. Clipped to the theory's
        # skeptic band [0.85, 0.95].
        t = (u - f_counter) / (1.0 - f_counter)
        epsilon = 0.15 + 0.08 * t        # [0.15, 0.23]
        w_neutral = 1.0 - (0.25 - epsilon) / (1.0 - epsilon)
        w_mix = float(min(max(w_neutral + w_spread, 0.85), 0.95))
        gamma = 1.5 + 2.0 * t            # [1.5, 3.5]
    epsilon = float(min(max(epsilon, 0.02), 0.35))

    # --- Route 1: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual micro-trust: a small validity-weighted endorsement
    # edge. logit(0.5) = 0 (coin-flip experts contribute nothing) and
    # the logit is CAPPED at 3 so perfect experts (v = 1.0) exert a
    # bounded pull.
    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)
    trust = float(np.dot(lg, a - b))

    # D > 0 favors A (B carries more accumulated suspicion, net of
    # the small trust edge).
    D = (pen_b - pen_a) + alpha * trust
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid

    # --- Route 2: graded endorsement-count preference ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    p_count_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))

    # --- Dual-route mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_a = float(min(max(p_a, 0.0), 1.0))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- c0: [0.06, 0.13]
- c1: [1.00, 1.35]
- kappa: [1.6, 2.1]
- beta: [1.3, 2.4]
- alpha: [0.0, 0.05]
- u_pop: [0, 1]
- w_spread: [-0.02, 0.02]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7117 (var=0.0205) vs this=0.7278 (var=0.0040)
- Experiment 2: real=-0.2562 (var=0.0128) vs this=-0.2659 (var=0.0036)
- Experiment 3: real=-0.5467 (var=0.0421) vs this=-0.5054 (var=0.0118)
- Experiment 4: real=-0.1938 (var=0.0027) vs this=-0.1963 (var=0.0008)
- Experiment 5: real=0.1143 (var=0.0077) vs this=0.1390 (var=0.0031)
- Experiment 6: real=0.1925 (var=0.0159) vs this=0.2350 (var=0.0077)
- Experiment 7: real=0.3815 (var=0.0238) vs this=0.3931 (var=0.0044)
- Experiment 8: real=0.7808 (var=0.0201) vs this=0.7629 (var=0.0041)
- Experiment 9: real=0.3729 (var=0.0159) vs this=0.4000 (var=0.0040)
- Experiment 10: real=0.5096 (var=0.0121) vs this=0.5415 (var=0.0033)
- Experiment 11: real=0.7875 (var=0.0537) vs this=0.7937 (var=0.0271)
- Experiment 12: real=0.8320 (var=0.0438) vs this=0.8320 (var=0.0072)
- Experiment 13: real=0.2800 (var=0.0133) vs this=0.3200 (var=0.0029)
- Experiment 14: real=0.1152 (var=0.0110) vs this=0.0594 (var=0.0004)
- Experiment 15: real=0.9846 (var=0.0914) vs this=0.7372 (var=0.0287)
- Experiment 16: real=0.1433 (var=0.0000) vs this=0.1323 (var=0.0000)


---

### `pi_8` (overall score: 0.713)

**Description**
Graded Dual-Route Distrust with a Bimodal Strategy Population (GDRD-B). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic distrust): every endorsement is a liability whose magnitude grows monotonically in the endorser's RAW claimed validity, d_j = c0 + c1*v_j^kappa; accumulated suspicion P(X) = sum_j x_j*d_j is compared across options through sigmoid(beta*(P(B)-P(A))), plus a small residual micro-trust edge alpha*sum_j clip(logit(v_j),0,3)*(a_j-b_j) (capped so perfect experts exert bounded, not unbounded, trust pull). ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is BIMODAL in the route weight w: a large majority (~88%) of 'skeptics' (w in [0.85, 0.95], moderate lapse) coexists with a minority (~12%) of 'counters' (w in [0.50, 0.62], low lapse, decisive count slope gamma >= 2). The mixture is implemented as a quantile transform of a single uniform per-subject population variable u_pop, which holds the population-MEAN route weight at its base value while exploding the between-subject behavioral variance — the empirically observed 3-12x super-binomial variance on the saturation assays (Exps 11, 12) that no uniform range-widening could reproduce.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Graded Dual-Route Distrust, bimodal-population variant (GDRD-B).
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # ROUTE 1 (analytic distrust, weight w): each endorsement is a
    # LIABILITY whose magnitude grows with the endorser's RAW claimed
    # validity:
    #   d_j = c0 + c1 * v_j^kappa          (kappa >= 1, no dampening
    #                                        near v = 0.5)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = (P(B) - P(A)) + alpha * sum_j clip(logit(v_j), 0, 3)*(a_j - b_j)
    #     (micro-trust term; the logit is CAPPED at 3 so perfect
    #      experts exert bounded, not unbounded, trust pull)
    #   P_skeptic(A) = sigmoid(beta * D)
    # ROUTE 2 (graded counting, weight 1-w): a CONTINUOUS preference
    # for the option with fewer total endorsements:
    #   P_count(A) = sigmoid(gamma * (cnt_B - cnt_A))
    #
    # BIMODAL POPULATION: instead of sampling w, gamma, epsilon from
    # independent uniform ranges (which shifts the population MEANS
    # whenever the ranges widen), a single uniform population quantile
    # u_pop in [0, 1] is deterministically mapped onto a two-component
    # mixture:
    #   u_pop < 0.12  -> 'counter' minority:  w in [0.50, 0.62],
    #                    epsilon in [0.05, 0.10], gamma in [2.0, 3.0]
    #   otherwise     -> 'skeptic' majority:   w in [0.85, 0.95],
    #                    epsilon in [0.08, 0.18], gamma in [1.5, 3.0]
    # This holds E[w] at the calibrated base value (~0.86) while
    # producing genuine between-subject bimodality — the mechanism the
    # real super-binomial variances on Exps 11/12 demand.
    #
    # History is ignored: validities are communicated in the
    # instructions, so the distrust weights are fixed for the whole
    # block. beta is sampled per subject from a wide range.
    if isinstance(state, dict):
        a_vec = np.asarray(state["option_a_ratings"], dtype=float).ravel()
        b_vec = np.asarray(state["option_b_ratings"], dtype=float).ravel()
        stim = np.vstack([a_vec, b_vec])
    else:
        stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GDRD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Power route: v = 1 is harmless (no logs). Trust route: guard
    # the logit against v = 1 (perfect experts) and v < 0.5.
    v_pow = np.clip(v, 0.5, 1.0)
    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])
    u_pop = float(parameters["u_pop"])

    # --- Bimodal population mixture over (w, epsilon, gamma) ---
    # Quantile transform of one uniform variable: the population mean
    # route weight stays at the calibrated base (~0.86) while the
    # between-subject distribution becomes bimodal.
    f_counter = 0.12
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority: near-balanced routes, low lapse, decisive
        # graded count slope.
        t = u / f_counter
        w_mix = 0.50 + 0.12 * t          # [0.50, 0.62]
        epsilon = 0.05 + 0.05 * t        # [0.05, 0.10]
        gamma = 2.0 + 1.0 * t            # [2.0, 3.0]
    else:
        # Skeptic majority: analytic-distrust dominant, moderate lapse.
        t = (u - f_counter) / (1.0 - f_counter)
        w_mix = 0.85 + 0.10 * t          # [0.85, 0.95]
        epsilon = 0.08 + 0.10 * t        # [0.08, 0.18]
        gamma = 1.5 + 1.5 * t            # [1.5, 3.0]
    w_mix = float(min(max(w_mix, 0.5), 0.95))

    # --- Route 1: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual micro-trust: a small validity-weighted endorsement
    # edge. logit(0.5) = 0 (coin-flip experts contribute nothing) and
    # the logit is CAPPED at 3 so perfect experts (v = 1.0) exert a
    # bounded pull (validated twice on the Exp-12 saturation assay).
    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)
    trust = float(np.dot(lg, a - b))

    # D > 0 favors A (B carries more accumulated suspicion, net of
    # the small trust edge).
    D = (pen_b - pen_a) + alpha * trust
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid

    # --- Route 2: graded endorsement-count preference ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    p_count_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))

    # --- Dual-route mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_a = float(min(max(p_a, 0.0), 1.0))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- c0: [0.06, 0.13]
- c1: [1.00, 1.35]
- kappa: [1.6, 2.1]
- beta: [1.3, 2.5]
- alpha: [0.0, 0.05]
- u_pop: [0, 1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7117 (var=0.0205) vs this=0.7667 (var=0.0024)
- Experiment 2: real=-0.2562 (var=0.0128) vs this=-0.2728 (var=0.0044)
- Experiment 3: real=-0.5467 (var=0.0421) vs this=-0.5946 (var=0.0135)
- Experiment 4: real=-0.1938 (var=0.0027) vs this=-0.2078 (var=0.0006)
- Experiment 5: real=0.1143 (var=0.0077) vs this=0.0829 (var=0.0023)
- Experiment 6: real=0.1925 (var=0.0159) vs this=0.2050 (var=0.0059)
- Experiment 7: real=0.3815 (var=0.0238) vs this=0.3794 (var=0.0044)
- Experiment 8: real=0.7808 (var=0.0201) vs this=0.7825 (var=0.0049)
- Experiment 9: real=0.3729 (var=0.0159) vs this=0.3844 (var=0.0045)
- Experiment 10: real=0.5096 (var=0.0121) vs this=0.5610 (var=0.0028)
- Experiment 11: real=0.7875 (var=0.0537) vs this=0.7937 (var=0.0289)
- Experiment 12: real=0.8320 (var=0.0438) vs this=0.8385 (var=0.0089)
- Experiment 13: real=0.2800 (var=0.0133) vs this=0.5200 (var=0.0013)
- Experiment 14: real=0.1152 (var=0.0110) vs this=0.1117 (var=0.0005)
- Experiment 15: real=0.9846 (var=0.0914) vs this=0.4974 (var=0.0098)
- Experiment 16: real=0.1433 (var=0.0000) vs this=0.1187 (var=0.0000)


---

### `pi_7` (overall score: 0.681)

**Description**
Softened Validity-Distrust with Count Mixture (SVD). Every expert endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity: d_j = c0 + c1*v_j^kappa, with no dampening near v=0.5 and no trust refund. Decision making is a dual-process mixture at the trial level: with probability w the subject is a 'skeptic' who integrates the accumulated-suspicion difference D = P(B) - P(A) through sigmoid(beta*D); with probability (1-w) the subject is a 'counter' who simply chooses the option with FEWER total endorsements (indifferent at matched counts). Two heterogeneity mechanisms soften the pooled extremes and capture between-subject variability: (i) the sigmoid slope beta is sampled from a WIDE per-subject range, so the population is a mixture of shallow and steep sigmoids — sensitive near D=0 (matched-count validity-composition trials) yet compressed at large |D| (unequal-count trials); (ii) the uniform lapse epsilon is also sampled from a wide per-subject range, which pulls every saturated large-D assay toward 0.5 while adding genuine between-subject dispersion in overall choice rates.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Softened Validity-Distrust with Count Mixture (SVD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Branch 1 (skeptic, prob w): each endorsement is a LIABILITY whose
    # magnitude grows with the endorser's RAW claimed validity:
    #   d_j = c0 + c1 * v_j^kappa   (kappa >= 1.5; NO dampening near
    #                                v = 0.5, NO trust refund)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = P(B) - P(A)  (positive favors A)
    #   P_skeptic(A) = sigmoid(beta * D)
    # Branch 2 (counter, prob 1-w): pure tallying on endorsement COUNTS;
    #   choose the option with FEWER total endorsements; exactly
    #   indifferent (0.5) at matched counts.
    # Mixture: P_core(A) = w * P_skeptic(A) + (1-w) * P_count(A),
    # mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # distrust weights are fixed for the whole block. beta AND epsilon
    # are sampled per subject from WIDE ranges: between-subject
    # heterogeneity in the sigmoid slope and in the lapse rate are
    # explicit parts of the theory.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SVD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v = np.clip(v, 0.5, 1.0)  # defensive; no logs taken, v=1 harmless.

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    w_mix = float(parameters["w"])

    # --- Skeptic branch: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion).
    D = pen_b - pen_a
    x = beta * D
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * x))  # stable sigmoid

    # --- Counter branch: pure endorsement-count tallying ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    if cnt_a < cnt_b:
        p_count_a = 1.0      # A has fewer endorsements -> prefer A
    elif cnt_a > cnt_b:
        p_count_a = 0.0
    else:
        p_count_a = 0.5      # matched counts -> indifferent

    # --- Dual-process mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- c0: [0.05, 0.12]
- c1: [0.95, 1.3]
- kappa: [1.5, 1.8]
- beta: [1.2, 2.6]
- w: [0.80, 0.92]
- epsilon: [0.12, 0.24]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7117 (var=0.0205) vs this=0.7453 (var=0.0030)
- Experiment 2: real=-0.2562 (var=0.0128) vs this=-0.2559 (var=0.0043)
- Experiment 3: real=-0.5467 (var=0.0421) vs this=-0.5546 (var=0.0132)
- Experiment 4: real=-0.1938 (var=0.0027) vs this=-0.1995 (var=0.0005)
- Experiment 5: real=0.1143 (var=0.0077) vs this=0.1000 (var=0.0018)
- Experiment 6: real=0.1925 (var=0.0159) vs this=0.2312 (var=0.0144)
- Experiment 7: real=0.3815 (var=0.0238) vs this=0.3960 (var=0.0030)
- Experiment 8: real=0.7808 (var=0.0201) vs this=0.7746 (var=0.0055)
- Experiment 9: real=0.3729 (var=0.0159) vs this=0.3889 (var=0.0038)
- Experiment 10: real=0.5096 (var=0.0121) vs this=0.5540 (var=0.0032)
- Experiment 11: real=0.7875 (var=0.0537) vs this=0.7812 (var=0.0147)
- Experiment 12: real=0.8320 (var=0.0438) vs this=0.8175 (var=0.0049)
- Experiment 13: real=0.2800 (var=0.0133) vs this=0.4000 (var=0.0038)
- Experiment 14: real=0.1152 (var=0.0110) vs this=0.0387 (var=0.0001)
- Experiment 15: real=0.9846 (var=0.0914) vs this=0.3118 (var=0.0035)
- Experiment 16: real=0.1433 (var=0.0000) vs this=0.0816 (var=0.0000)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0607 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0607 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(choose the Take-The-Best winner) on trials where TTB and Tallying
    make OPPOSITE deterministic predictions (i.e., trials where the two
    options win on different numbers of features, so the tallies disagree).

    High value -> TTB behavior; low value -> Tallying behavior.
    Works on the pooled DataFrame and on a single subject's slice.
    """
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_matrix(col):
        rows = [list(r) for r in col]
        return np.array(rows, dtype=float)

    a = _to_matrix(data["option_a_ratings"])
    b = _to_matrix(data["option_b_ratings"])
    resp = np.asarray(data["response"].to_numpy(), dtype=float)

    n_trials = a.shape[0]

    # --- TTB winner per trial (cue cascade, first discriminating cue) ---
    winners = np.full(n_trials, -1, dtype=int)
    for j in cue_order:
        undecided = winners < 0
        a_better = undecided & (a[:, j] > b[:, j])
        b_better = undecided & (b[:, j] > a[:, j])
        winners[a_better] = 0
        winners[b_better] = 1

    # --- Tallying score per trial (number of strict feature-wise wins) ---
    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Conflict trials: tallies differ (so Tallying has a strict winner that,
    # in this design, is always the option OPPOSITE to the TTB winner),
    # and TTB also has a strict winner.
    conflict = (a_wins != b_wins) & (winners >= 0)

    if not np.any(conflict):
        # No diagnostic trials -> uninformative, return chance.
        return 0.5

    agree = float(np.mean(resp[conflict] == winners[conflict]))
    return agree
```

**Observed (real) value:** 0.7117 (var=0.0205)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7436 (var=0.0035) (Δ vs real +0.0319)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8444 (var=0.0125)
- pi_2: 0.1203 (var=0.0072)
- pi_3: 0.6869 (var=0.0033)
- pi_4: 0.7792 (var=0.0018)
- pi_5: 0.7272 (var=0.0025)
- pi_6: 0.6844 (var=0.0026)
- pi_7: 0.7453 (var=0.0030)
- pi_8: 0.7667 (var=0.0024)
- pi_8_1: 0.7278 (var=0.0040)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signature: allegiance to the tally winner on CONFLICT trials
    # (trials where the single top-validity discriminating cue favors
    # the tally LOSER). Tallying predicts P(choose tally winner) > 0.5
    # there (it follows the cue majority against the best cue);
    # Take-The-Best predicts P < 0.5 (it follows the top cue against
    # the majority). We return P(W | conflict) - 0.5, so the sign
    # itself identifies the theory for any noise level beta > 0.
    validities = np.array([0.95, 0.90, 0.85, 0.65, 0.60, 0.50])
    cue_order = np.argsort(-validities, kind="stable")

    a_list = data["option_a_ratings"].tolist()
    b_list = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    conflict_w = []
    for a_raw, b_raw, r in zip(a_list, b_list, resp):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)

        a_w = float(np.sum(a > b))
        b_w = float(np.sum(b > a))
        if a_w == b_w:
            # Tally is tied -> heuristic guesses; uninformative cell.
            continue
        winner = 0 if a_w > b_w else 1

        # First discriminating cue in descending-validity order.
        top_favors = None
        for j in cue_order:
            if a[j] > b[j]:
                top_favors = 0
                break
            if b[j] > a[j]:
                top_favors = 1
                break
        if top_favors is None:
            continue  # no discriminating cue at all

        if top_favors == winner:
            continue  # congruent trial: both heuristics point the same way

        # Conflict trial: tally winner opposes the top cue's favorite.
        conflict_w.append(1.0 if int(r) == winner else 0.0)

    if len(conflict_w) == 0:
        return 0.0
    return float(np.mean(conflict_w)) - 0.5
```

**Observed (real) value:** -0.2562 (var=0.0128)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.2506 (var=0.0040) (Δ vs real +0.0056)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3394 (var=0.0118)
- pi_1: -0.3231 (var=0.0164)
- pi_3: -0.2566 (var=0.0023)
- pi_4: -0.2684 (var=0.0040)
- pi_5: -0.2456 (var=0.0038)
- pi_6: -0.1978 (var=0.0023)
- pi_7: -0.2559 (var=0.0043)
- pi_8: -0.2728 (var=0.0044)
- pi_8_1: -0.2659 (var=0.0036)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Validity-sensitivity contrast for the cue-2 cascade design.

    For every trial we compute (a) the Take-The-Best winner (first
    discriminating cue in validity order) and (b) the VWEI evidence
    E = sum_j [log(v_j/(1-v_j))]^3 * sign(a_j - b_j), expressed
    relative to the TTB winner (Erel = E * (+1 if TTB winner is A
    else -1)).

    Two trial classes:
      HIGH  : Erel >  3   -> pairs 1-4, where the 90% expert decides
             and VWEI strongly agrees (P(pick TTB winner) ~ 0.85).
      LOW   : Erel < -0.5 -> pairs 7-12 and 15-16, where cue 2 (or 3)
             decides but the weighted evidence of the opposing
             coalition dominates (P(pick TTB winner) ~ 0.35-0.47).

    Metric = P(choose TTB winner | HIGH) - P(choose TTB winner | LOW).

    TTB predicts the SAME choice probability on every non-tied pair
    (its score is binary), so this difference is 0 in expectation.
    VWEI predicts a large positive difference (~ +0.42). The contrast
    cancels each subject's overall response rate (beta / epsilon),
    leaving only the stimulus-driven validity sensitivity.
    """
    v = np.array([0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64])
    lo = np.log(v / (1.0 - v))
    w = lo ** 3.0
    order = np.argsort(-v, kind="stable")

    highs = []
    lows = []

    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.shape[0] != 9 or b.shape[0] != 9:
            continue

        # VWEI-style evidence for A over B (gamma = 3).
        d = np.sign(a - b)
        E = float(np.dot(w, d))

        # TTB winner: first discriminating cue in validity order.
        win = None
        for j in order:
            if a[j] > b[j]:
                win = 0
                break
            if b[j] > a[j]:
                win = 1
                break
        if win is None:
            continue  # no discriminating cue (does not occur here)

        # Evidence relative to the TTB winner.
        s = 1.0 if win == 0 else -1.0
        Erel = E * s

        r_val = int(r)
        chose_winner = 1.0 if r_val == win else 0.0

        if Erel > 3.0:
            highs.append(chose_winner)
        elif Erel < -0.5:
            lows.append(chose_winner)

    if not highs or not lows:
        return 0.0

    return float(np.mean(highs) - np.mean(lows))
```

**Observed (real) value:** -0.5467 (var=0.0421)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.5042 (var=0.0127) (Δ vs real +0.0425)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0108 (var=0.0045)
- pi_3: 0.4433 (var=0.0115)
- pi_2: 0.5433 (var=0.0293)
- pi_4: -0.4146 (var=0.0116)
- pi_5: -0.5708 (var=0.0133)
- pi_6: -0.4858 (var=0.0140)
- pi_7: -0.5546 (var=0.0132)
- pi_8: -0.5946 (var=0.0135)
- pi_8_1: -0.5054 (var=0.0118)

### Experiment 4
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Signed-evidence slope.

    For every trial we compute the VWEI evidence
        E = sum_j w_j * sign(a_j - b_j),  w_j = logit(v_j)^3
    and the TTB winner (first discriminating cue in validity order;
    validities are strictly descending so cue order = feature order).

    Define the signed regressor  x = +E if TTB's winner is A, x = -E if
    TTB's winner is B, and the outcome y = 1 iff the subject chose TTB's
    winner.  Return the OLS slope of y on x.

    TTB's probability of choosing its own winner is a single constant
    (softmax(beta) mixed with a lapse) on EVERY trial, so its expected
    slope is exactly 0.  VWEI's probability of choosing TTB's winner is
    sigmoid(beta * x) on every trial, giving a clearly positive slope
    (~0.05 at the design center, sign-stable over the whole parameter
    box).  The two theories are maximally separated on this statistic.
    """
    if data is None or len(data) == 0:
        return 0.0

    validities = np.array([0.74, 0.73, 0.70, 0.69, 0.68,
                           0.67, 0.66, 0.65, 0.64], dtype=float)
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-6)
    w = np.log(v / (1.0 - v)) ** 3.0

    def _to_vec(cell):
        if isinstance(cell, str):
            return [float(ch) for ch in cell if ch in '01']
        return [float(val) for val in cell]

    a = np.array([_to_vec(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([_to_vec(r) for r in data['option_b_ratings']], dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != len(w):
        return 0.0

    resp = data['response'].to_numpy(dtype=float)

    d = np.sign(a - b)                     # +1 favors A, -1 favors B
    E = d @ w                              # VWEI evidence for A

    disc = d != 0
    keep = disc.any(axis=1)                # drop degenerate all-tie rows
    d, E, resp, disc = d[keep], E[keep], resp[keep], disc[keep]
    if d.shape[0] == 0:
        return 0.0

    # First discriminating cue (validities strictly descending -> order 0..8)
    first_idx = disc.argmax(axis=1)
    rows = np.arange(d.shape[0])
    d_first = d[rows, first_idx]
    winner_a = d_first > 0                  # TTB's winner is option A

    # y = 1 iff subject chose TTB's winner; x = evidence signed toward winner
    y = np.where(winner_a, 1.0 - resp, resp)
    x = np.where(winner_a, E, -E)

    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-12:
        return 0.0
    yc = y - y.mean()
    slope = float(np.dot(xc, yc)) / denom
    return slope
```

**Observed (real) value:** -0.1938 (var=0.0027)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.1973 (var=0.0006) (Δ vs real -0.0034)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0475 (var=0.0011)
- pi_1: 0.0002 (var=0.0008)
- pi_2: 0.1895 (var=0.0024)
- pi_4: -0.2020 (var=0.0005)
- pi_5: -0.2142 (var=0.0004)
- pi_6: -0.1987 (var=0.0006)
- pi_7: -0.1995 (var=0.0005)
- pi_8: -0.2078 (var=0.0006)
- pi_8_1: -0.1963 (var=0.0008)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Top-cue allegiance under endorsement-majority conflict.

    Selected trials: the 95%-validity expert (feature index 0) endorses
    exactly one option, AND that option carries strictly MORE total
    endorsements (sum of ratings) than its rival.  On these trials the
    metric is the empirical rate of choosing the option endorsed by the
    top expert.

    VWEI (any gamma, beta, epsilon) must follow the top cue at ~0.98
    here: the amplified log-odds weight of v=.95 dominates every
    possible opposing coalition of the remaining cues, and 50%-expert
    endorsements contribute exactly zero weight.
    SDI must AVOID the top-cue option here (~0.15-0.25): every
    endorsement is a liability, and the top-cue option is also the
    endorsement-majority option.
    """
    df = data.reset_index(drop=True)

    a = np.array([list(x) for x in df['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in df['option_b_ratings']], dtype=float)
    resp = df['response'].to_numpy(dtype=float)

    if a.ndim != 2 or a.shape[0] == 0:
        return 0.5

    chose_a = resp < 0.5

    # Which option does the top (95%) expert endorse?
    top_a = a[:, 0] > 0.5
    top_b = b[:, 0] > 0.5
    # Keep only trials where exactly one option gets the top expert's
    # endorsement (guards against degenerate rows).
    xor_top = np.logical_xor(top_a, top_b)

    # Total endorsement counts per option.
    cnt_a = a.sum(axis=1)
    cnt_b = b.sum(axis=1)

    # Is the top-cue-endorsed option also the endorsement-majority
    # option (strictly more endorsements)?
    top_option_more = np.where(top_a, cnt_a > cnt_b, cnt_b > cnt_a)

    sel = xor_top & top_option_more
    if not np.any(sel):
        return 0.5

    # Did the subject choose the option endorsed by the top expert?
    followed = np.where(top_a[sel], chose_a[sel], ~chose_a[sel])
    return float(np.mean(followed))
```

**Observed (real) value:** 0.1143 (var=0.0077)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1514 (var=0.0075) (Δ vs real +0.0371)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1348 (var=0.0038)
- pi_3: 0.9867 (var=0.0003)
- pi_1: 0.8433 (var=0.0110)
- pi_2: 0.9057 (var=0.0054)
- pi_5: 0.0852 (var=0.0016)
- pi_6: 0.1043 (var=0.0024)
- pi_7: 0.1000 (var=0.0018)
- pi_8: 0.0829 (var=0.0023)
- pi_8_1: 0.1390 (var=0.0031)

### Experiment 6
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Double-dissociation composite built on the two strongest contrasts
    # engineered into this design:
    #
    # COMPONENT 1 ("big-gap single endorsers", pairs 6/18, 8 trials/subject):
    #   both options carry exactly ONE endorsement; the endorsers' validities
    #   differ by >= 0.25 (0.90 vs 0.60).  VWEI overwhelmingly follows the
    #   higher-validity endorser (P ~ 0.89 across its whole parameter box);
    #   SDI is structurally near-blind at matched endorsement counts
    #   (P ~ 0.50-0.55 across its whole box).
    #
    # COMPONENT 2 ("ridge-crest coalition", pairs 10/22, 8 trials/subject):
    #   one option carries a single 0.80-validity endorsement, the other a
    #   coalition of four 0.75-validity endorsements.  Score +1 for choosing
    #   the COALITION.  VWEI favors the coalition (P ~ 0.62); SDI strongly
    #   favors the single endorser (coalition P ~ 0.10).
    #
    # Metric = mean of the two rates.  Expected: VWEI ~ 0.75, SDI ~ 0.31.
    # Both components use only 8 trials each but their gaps point the SAME
    # way after sign alignment, so the composite has a large mean gap and
    # modest binomial-only between-subject variance.

    V = np.array([0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60])

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    scores_gap = []    # 1 = chose the higher-validity single endorser
    scores_coal = []    # 1 = chose the 4x0.75 coalition over the single 0.80 endorser

    for a_raw, b_raw, r in zip(a_cells, b_cells, resp):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != V.shape or b.shape != V.shape:
            continue
        ca = int(round(a.sum()))
        cb = int(round(b.sum()))
        chose_a = (int(r) == 0)

        if ca == 1 and cb == 1:
            # Single endorsement on each side: who endorsed, and how good
            # is that expert?
            ia = int(np.argmax(a))
            ib = int(np.argmax(b))
            va = V[ia]
            vb = V[ib]
            if abs(va - vb) >= 0.25:
                hi_is_a = (va > vb)
                scores_gap.append(1.0 if (chose_a == hi_is_a) else 0.0)

        elif (ca == 1 and cb == 4) or (ca == 4 and cb == 1):
            # Ridge-crest conflict: verify it is exactly the designed cell
            # (single 0.80 expert vs four 0.75 experts) and score the
            # coalition choice.
            if ca == 1:
                single_vec, coal_vec = a, b
                single_is_a = True
            else:
                single_vec, coal_vec = b, a
                single_is_a = False
            i_single = int(np.argmax(single_vec))
            coal_idx = np.where(coal_vec > 0.5)[0]
            if (
                abs(V[i_single] - 0.80) < 1e-9
                and len(coal_idx) == 4
                and bool(np.all(np.abs(V[coal_idx] - 0.75) < 1e-9))
            ):
                # +1 when the subject chose the coalition side
                scores_coal.append(1.0 if (chose_a != single_is_a) else 0.0)

    r_gap = float(np.mean(scores_gap)) if len(scores_gap) > 0 else 0.5
    r_coal = float(np.mean(scores_coal)) if len(scores_coal) > 0 else 0.5
    return 0.5 * (r_gap + r_coal)
```

**Observed (real) value:** 0.1925 (var=0.0159)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2425 (var=0.0115) (Δ vs real +0.0500)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7625 (var=0.0125)
- pi_4: 0.3037 (var=0.0089)
- pi_1: 0.4863 (var=0.0079)
- pi_2: 0.6925 (var=0.0122)
- pi_5: 0.1862 (var=0.0087)
- pi_6: 0.2387 (var=0.0095)
- pi_7: 0.2312 (var=0.0144)
- pi_8: 0.2050 (var=0.0059)
- pi_8_1: 0.2350 (var=0.0077)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Validity-following rate in theory-conflict cells"
    #
    # The 15 experts split into a HIGH-validity group (features 0-6:
    # validities 0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65) and a
    # LOW-validity group (features 7-14: validity 0.60). Every trial in
    # this design pits a "pure high" option against a "pure low"
    # option. Two families of cells separate SDI from VPS:
    #
    # Family 1 (matched endorsement counts, validity gap): SDI's
    #   penalty is a near-flat function of validity (delta ~ 0), so it
    #   is pinned at ~0.51 for the high-validity option (its weak
    #   alpha tie-break even pushes slightly ABOVE 0.5). VPS's
    #   penalty rises steeply in validity, so it systematically picks
    #   the LOW-validity option (~0.28).
    #
    # Family 2 (count conflict where the FEWER-endorsed option is the
    #   high-validity one): SDI follows the endorsement count and
    #   picks the high-validity option (~0.72). VPS's validity-scaled
    #   distrust makes the low-validity majority cheaper, so it picks
    #   the high-validity option only ~0.30 of the time.
    #
    # The metric is the average of P(choose the high-validity-group
    #   option) over these two families. SDI predicts ~0.62; VPS
    #   predicts ~0.29.

    V = np.array([0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65,
                  0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])

    f1_hits = 0.0
    f1_n = 0
    f2_hits = 0.0
    f2_n = 0

    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        m = int(min(a.size, b.size, V.size))
        if m == 0:
            continue
        a = a[:m]
        b = b[:m]

        top = V[:m] >= 0.65  # high-validity expert group
        a_top = float(np.sum(a[top]))
        a_bot = float(np.sum(a[~top]))
        b_top = float(np.sum(b[top]))
        b_bot = float(np.sum(b[~top]))

        # Require each option to be endorsed purely by one group
        # (all trials in this design satisfy this; guards oddities).
        if (a_top > 0 and a_bot > 0) or (b_top > 0 and b_bot > 0):
            continue
        ca = a_top + a_bot
        cb = b_top + b_bot
        if ca == 0 or cb == 0:
            continue

        a_is_top = a_top > 0
        b_is_top = b_top > 0
        if a_is_top == b_is_top:
            continue

        chose_a = (int(round(float(r))) == 0)
        chose_top = chose_a if a_is_top else (not chose_a)

        if ca == cb:
            # Family 1: matched counts, validity gap.
            f1_n += 1
            f2_unused = None
            f1_hits += 1.0 if chose_top else 0.0
        else:
            # Family 2: count conflict; keep only cells where the
            # FEWER-endorsed option is the high-validity one
            # (the cells where SDI and VPS point in opposite
            # directions). Cells where the fewer-endorsed option is
            # the low-validity one are agreed upon by both theories
            # and are excluded.
            fewer_is_top = (a_is_top and ca < cb) or (b_is_top and cb < ca)
            if fewer_is_top:
                f2_n += 1
                f2_hits += 1.0 if chose_top else 0.0

    p1 = f1_hits / f1_n if f1_n > 0 else np.nan
    p2 = f2_hits / f2_n if f2_n > 0 else np.nan
    vals = [p for p in (p1, p2) if not np.isnan(p)]
    if not vals:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 0.3815 (var=0.0238)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4184 (var=0.0162) (Δ vs real +0.0369)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6227 (var=0.0042)
- pi_5: 0.2844 (var=0.0060)
- pi_1: 0.8411 (var=0.0121)
- pi_2: 0.3319 (var=0.0035)
- pi_3: 0.9332 (var=0.0008)
- pi_6: 0.3286 (var=0.0033)
- pi_7: 0.3960 (var=0.0030)
- pi_8: 0.3794 (var=0.0044)
- pi_8_1: 0.3931 (var=0.0044)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Matched-count validity-reversal index.
    #
    # On trials where A and B carry the SAME number of endorsements (k vs k),
    # the two theories make opposite-sign predictions:
    #   VPS: the coalition of LOWER-validity endorsers is preferred
    #        (a high-validity endorsement is a bigger suspected defect),
    #        with preference strengthening in k.
    #   SDI: at matched count its validity sensitivity has the wrong sign
    #        and is bounded, so it stays at or below chance for the
    #        low-validity coalition on every such trial.
    #
    # The metric is the empirical proportion of choices of the
    # LOW-validity coalition across all matched-count trials
    # (design pairs 1-6 and their A/B mirrors 13-18; 48 of the 96
    # trials per subject). VPS predicts ~0.85; SDI predicts ~0.45.
    # It is well-defined on a single subject's slice (48 Bernoulli
    # trials) and on the pooled data.

    v = np.array([
        0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.85, 0.85, 0.75,
        0.65, 0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65, 0.65,
    ], dtype=float)

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape[0] != v.shape[0] or b.shape[0] != v.shape[0]:
            continue

        ca = float(a.sum())
        cb = float(b.sum())
        # Keep only matched-count trials with at least one endorsement
        # on each side (the validity-reversal assay cells).
        if ca == 0.0 or cb == 0.0 or ca != cb:
            continue

        # Mean validity of the endorsing coalition on each side.
        va = float(np.mean(v[a > 0]))
        vb = float(np.mean(v[b > 0]))
        if va == vb:
            continue

        low_is_a = va < vb
        chose_a = (int(row["response"]) == 0)
        hits.append(1.0 if (chose_a == low_is_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7808 (var=0.0201)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7388 (var=0.0116) (Δ vs real -0.0421)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8263 (var=0.0045)
- pi_4: 0.4600 (var=0.0059)
- pi_1: 0.1496 (var=0.0101)
- pi_2: 0.5108 (var=0.0054)
- pi_3: 0.0317 (var=0.0006)
- pi_6: 0.8054 (var=0.0034)
- pi_7: 0.7746 (var=0.0055)
- pi_8: 0.7825 (var=0.0049)
- pi_8_1: 0.7629 (var=0.0041)

### Experiment 9
**Design**
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Weighted proportion of choosing the side whose endorsements come from
    # MORE VALID experts, on matched-endorsement-count trials.
    #
    # Design validities (n_features=20, fixed): indices 0-5 are 0.50,
    # indices 6-19 are 0.60. On every trial where both options carry the
    # SAME number of endorsements, the flat count terms (c0) of both
    # DCD and VPS cancel exactly, so the choice is driven purely by the
    # validity composition of the endorsers:
    #   - VPS: penalty strictly increasing in v  -> always avoids the
    #     0.60 experts -> weighted P(hi-validity side) well below 0.5
    #     (mid ~0.38, box <= ~0.45).
    #   - DCD: dampened distrust (2v-1)^kappa collapses near chance and
    #     is opposed by the residual trust term alpha*logit(v) ->
    #     essentially at/above chance (mid ~0.51, box >= ~0.49).
    # Trials are weighted by m = |#0.60-endorsements difference|, which
    # emphasizes the cells (m=2,3,4) where the two theories diverge most.
    V = np.array([0.5] * 6 + [0.6] * 14, dtype=float)
    hi = (V > 0.55).astype(float)  # indicator of the 0.60-validity experts

    if data is None or len(data) == 0:
        return 0.5

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    num = 0.0
    den = 0.0
    for i in range(len(data)):
        a = np.asarray(a_col[i], dtype=float)
        b = np.asarray(b_col[i], dtype=float)
        if a.shape[0] != V.shape[0] or b.shape[0] != V.shape[0]:
            continue
        tA = float(np.sum(a))
        tB = float(np.sum(b))
        # Only matched-count trials: the flat c0 terms cancel and the
        # choice isolates the validity-composition sensitivity.
        if tA != tB:
            continue
        hA = float(np.dot(hi, a))  # number of 0.60-expert endorsements on A
        hB = float(np.dot(hi, b))
        if hA == hB:
            continue  # pure tie in validity composition
        m = abs(hA - hB)
        r = int(resp[i])
        chose_hi = (r == 0) if (hA > hB) else (r == 1)
        den += m
        if chose_hi:
            num += m

    if den <= 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.3729 (var=0.0159)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4142 (var=0.0056) (Δ vs real +0.0413)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5269 (var=0.0032)
- pi_5: 0.3820 (var=0.0036)
- pi_1: 0.7308 (var=0.0072)
- pi_2: 0.5032 (var=0.0040)
- pi_3: 0.5099 (var=0.0041)
- pi_4: 0.4895 (var=0.0044)
- pi_7: 0.3889 (var=0.0038)
- pi_8: 0.3844 (var=0.0045)
- pi_8_1: 0.4000 (var=0.0040)

### Experiment 10
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # COIN-FLIP ENDORSEMENT TOLERANCE.
    #
    # For every trial, count how many endorsements each option receives
    # from the five chance-validity (v = 0.50) experts (feature indices
    # 15-19 in this experiment's fixed validity vector). Every one of the
    # 24 unique pairs pits an option carrying coin-flip endorsements
    # against an option carrying none, so the trial is scored:
    #   1.0  if the subject chose the option with MORE 0.50-expert
    #        endorsements (the 'coin-flip tolerant' choice),
    #   0.0  otherwise.
    # The metric is the mean of this score over all trials.
    #
    # This is a direct assay of the one structural difference between the
    # two confidence-distrust theories: the geometry of the per-endorsement
    # distrust charge. VPS charges c0 + c1*v^kappa, so a 0.50 endorsement
    # still costs a substantial c1*0.25-0.33 distrust units and the
    # internal exchange rate between mid-validity (0.75) endorsements and
    # coin-flips is ~1.9 coin-flips per 0.75. DCD charges
    # c0 + c1*(2v-1)^kappa, which collapses to the bare floor c0 at
    # v = 0.50 (and refunds part of the 0.75 charge via the residual trust
    # term), yielding an exchange rate of only ~1.3. Hence VPS subjects
    # are systematically MORE willing to accept coin-flip endorsements,
    # across every cell family in the design:
    #   composition cells (mixed {v_hi, 0.50} vs homogeneous mid): VPS
    #     picks the mixed side (0.53-0.59), DCD avoids it (0.33-0.45);
    #   exchange cells (0.75-coalition vs k coin-flips): VPS keeps
    #     choosing the coin-flip side through k=3-4, DCD abandons it;
    #   calibration cells ({0.90} vs {0.50}, {0.50,0.50} vs {0.90,0.90}):
    #     VPS 0.73/0.86 vs DCD 0.65/0.76.
    # Mid-parameter predictions: VPS ~0.59, DCD ~0.45 (disjoint across
    # their full parameter boxes), with per-subject SD ~0.05-0.06 from
    # 96 Bernoulli trials.

    validities = np.array([
        0.92, 0.90, 0.90, 0.90, 0.85, 0.77, 0.77,
        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.72, 0.72,
        0.50, 0.50, 0.50, 0.50, 0.50,
    ])
    coin_mask = np.isclose(validities, 0.5)

    responses = data["response"].to_numpy()
    ratings_a = data["option_a_ratings"].tolist()
    ratings_b = data["option_b_ratings"].tolist()

    scores = []
    for i in range(len(ratings_a)):
        a = np.asarray(ratings_a[i], dtype=float).ravel()
        b = np.asarray(ratings_b[i], dtype=float).ravel()
        n = min(a.shape[0], b.shape[0], validities.shape[0])
        mask = coin_mask[:n]
        ca = float(np.sum(a[:n][mask]))
        cb = float(np.sum(b[:n][mask]))
        if ca == cb:
            # No coin-flip asymmetry on this trial; skip defensively.
            continue
        chose_a = (int(responses[i]) == 0)
        c_chosen = ca if chose_a else cb
        c_other = cb if chose_a else ca
        scores.append(1.0 if c_chosen > c_other else 0.0)

    if len(scores) == 0:
        return 0.5
    return float(np.mean(scores))
```

**Observed (real) value:** 0.5096 (var=0.0121)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5408 (var=0.0030) (Δ vs real +0.0312)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6071 (var=0.0030)
- pi_6: 0.4429 (var=0.0037)
- pi_1: 0.4433 (var=0.0012)
- pi_2: 0.6129 (var=0.0030)
- pi_3: 0.5190 (var=0.0010)
- pi_4: 0.3810 (var=0.0022)
- pi_7: 0.5540 (var=0.0032)
- pi_8: 0.5610 (var=0.0028)
- pi_8_1: 0.5415 (var=0.0033)

### Experiment 11
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Saturated-conflict compliance rate.
    #
    # Logic: using a FIXED reference parameterization of the shared skeptic
    # core (c0=0.10, c1=1.0, kappa=2.0 -> d_j = 0.10 + v_j^2), compute the
    # skeptic decision variable D = pen(B) - pen(A) for every trial, plus the
    # raw endorsement counts cnt_A, cnt_B.
    #
    # Keep only trials that are (i) SATURATED for the skeptic core
    # (|D| > 2.0, so any plausible beta in either theory's box drives the
    # sigmoid to ~1) and (ii) in CONFLICT: the skeptic-preferred side is the
    # side with MORE endorsements (the pure counter branch votes the other
    # way). In this fixed design these are exactly the count-parity-jump
    # cells pairs 3, 4 and their mirrors 15, 16 (16 trials per subject).
    #
    # On these cells the two theories make disjoint predictions:
    #   VPS:  P(pick skeptic side) = (1-eps)*~1 + eps/2  ~ 0.95 (box 0.92-0.98)
    #   SVD:  P = (1-eps)*(w*1 + (1-w)*0) + eps/2 = (1-eps)*w + eps/2
    #         ~ 0.80 (box 0.73-0.87, since the counter mixture drags the
    #         saturated response toward the fewer-endorsement side).
    # The metric is the observed fraction of choices that follow the
    # skeptic-preferred side on these cells.

    validities = [0.95] * 7 + [0.75] * 4 + [0.50] * 9
    d_ref = [0.10 + 1.0 * (v ** 2) for v in validities]

    hits = []
    for _, row in data.iterrows():
        a_raw = row["option_a_ratings"]
        b_raw = row["option_b_ratings"]
        a = [int(float(x)) for x in a_raw]
        b = [int(float(x)) for x in b_raw]

        pen_a = sum(di for di, xi in zip(d_ref, a) if xi == 1)
        pen_b = sum(di for di, xi in zip(d_ref, b) if xi == 1)
        D = pen_b - pen_a  # >0 favors A

        cnt_a = sum(1 for x in a if x == 1)
        cnt_b = sum(1 for x in b if x == 1)

        # (i) saturation for the skeptic core
        if abs(D) <= 2.0:
            continue
        # counter branch must be decisive (not indifferent)
        if cnt_a == cnt_b:
            continue

        skeptic_prefers_a = D > 0
        counter_prefers_a = cnt_a < cnt_b  # fewer endorsements preferred

        # (ii) conflict: counter branch votes AGAINST the saturated skeptic core
        if skeptic_prefers_a == counter_prefers_a:
            continue

        resp = int(row["response"])  # 0 = chose A, 1 = chose B
        followed_skeptic = (resp == 0) if skeptic_prefers_a else (resp == 1)
        hits.append(1.0 if followed_skeptic else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7875 (var=0.0537)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7338 (var=0.0254) (Δ vs real -0.0537)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7812 (var=0.0147)
- pi_5: 0.9525 (var=0.0034)
- pi_1: 0.1400 (var=0.0209)
- pi_2: 0.8375 (var=0.0163)
- pi_3: 0.0075 (var=0.0004)
- pi_4: 0.1500 (var=0.0081)
- pi_6: 0.9487 (var=0.0026)
- pi_8: 0.7937 (var=0.0289)
- pi_8_1: 0.7937 (var=0.0271)

### Experiment 12
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # COIN-SIDE CHOICE RATE ON SATURATED PERFECT-VS-COIN CELLS.
    #
    # Features 0..5 are the validity-1.0 ("perfect") experts; features
    # 6..14 are the coin-flip (v=0.5) experts.
    #
    # Selected cells: one option carries >= 4 perfect endorsements with
    # <= 7 total endorsements, the other carries 0 perfect and >= 5 coin
    # endorsements. This selects exactly the 10 saturated pairs
    # (1,2,3,4,6 and their A/B mirrors 13,14,15,16,18), i.e. 40 of the
    # 96 trials per subject. The staircase top rung (pairs 5/17) and the
    # exchange-locus pairs are deliberately excluded: there the two
    # theories overlap or the per-subject spread is maximal.
    #
    # The metric is the rate of CHOOSING THE COIN-ONLY SIDE (i.e. avoiding
    # the perfect-endorser side), pooled over floor trials and their
    # mirrors, which cancels any side bias.

    n_perf = 6
    df = data.reset_index(drop=True)

    hits = []
    for a_raw, b_raw, resp in zip(
        df["option_a_ratings"], df["option_b_ratings"], df["response"]
    ):
        a = [int(x) for x in a_raw]
        b = [int(x) for x in b_raw]

        perfA = sum([1 for i in range(len(a)) if a[i] == 1 and i < n_perf])
        coinA = sum([1 for i in range(len(a)) if a[i] == 1 and i >= n_perf])
        perfB = sum([1 for i in range(len(b)) if b[i] == 1 and i < n_perf])
        coinB = sum([1 for i in range(len(b)) if b[i] == 1 and i >= n_perf])

        a_heavy = (perfA >= 4) and ((perfA + coinA) <= 7)
        b_heavy = (perfB >= 4) and ((perfB + coinB) <= 7)
        a_coin = (perfA == 0) and (coinA >= 5)
        b_coin = (perfB == 0) and (coinB >= 5)

        if a_heavy and b_coin:
            # A is the perfect-heavy side, so the coin-only side is B.
            hits.append(1.0 if int(resp) == 1 else 0.0)
        elif b_heavy and a_coin:
            # B is the perfect-heavy side, so the coin-only side is A.
            hits.append(1.0 if int(resp) == 0 else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.8320 (var=0.0438)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7870 (var=0.0296) (Δ vs real -0.0450)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9385 (var=0.0024)
- pi_7: 0.8175 (var=0.0049)
- pi_1: 0.1350 (var=0.0072)
- pi_2: 0.7165 (var=0.0069)
- pi_3: 0.0050 (var=0.0001)
- pi_4: 0.0885 (var=0.0040)
- pi_6: 0.0525 (var=0.0010)
- pi_8: 0.8385 (var=0.0089)
- pi_8_1: 0.8320 (var=0.0072)

### Experiment 13
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    from ast import literal_eval

    # Fixed design: experts 0-5 have v=1.0, experts 6-7 have v=0.65,
    # experts 8-15 have v=0.5.
    PERF = (0, 1, 2, 3, 4, 5)
    MID = (6, 7)
    COIN = (8, 9, 10, 11, 12, 13, 14, 15)

    def _vec(cell):
        if isinstance(cell, str):
            cell = literal_eval(cell)
        return np.asarray(list(cell), dtype=float).ravel()

    def _saturation_side(a, b):
        """Saturation (pure lapse-assay) trials: one option carries only
        1-2 coin endorsements, the other carries only 4-5 perfect
        endorsements. Both routes of both theories saturate toward the
        coin-side option, so P(choose coin side) = 1 - epsilon/2.
        Returns 0 if A is the coin side, 1 if B is, else -1."""
        pa = int(round(sum(a[i] for i in PERF)))
        ma = int(round(sum(a[i] for i in MID)))
        ca = int(round(sum(a[i] for i in COIN)))
        pb = int(round(sum(b[i] for i in PERF)))
        mb = int(round(sum(b[i] for i in MID)))
        cb = int(round(sum(b[i] for i in COIN)))
        if ma == 0 and mb == 0:
            a_low = (pa == 0 and ca in (1, 2) and pb in (4, 5) and cb == 0)
            b_low = (pb == 0 and cb in (1, 2) and pa in (4, 5) and ca == 0)
            if a_low != b_low:
                return 0 if a_low else 1
        return -1

    def _rate(df):
        hits = []
        for a_cell, b_cell, resp in zip(
            df["option_a_ratings"].tolist(),
            df["option_b_ratings"].tolist(),
            df["response"].tolist(),
        ):
            side = _saturation_side(_vec(a_cell), _vec(b_cell))
            if side < 0:
                continue
            hits.append(1.0 if int(float(resp)) == side else 0.0)
        if not hits:
            return float("nan")
        return float(np.mean(hits))

    n_subj = int(pd.Series(data["subject_id"]).nunique())

    # Single subject's slice: that subject's saturation accuracy.
    if n_subj <= 1:
        return _rate(data)

    # Pooled sample: population-level prevalence of ceiling-level
    # saturation (the design's designated lapse falsification assay).
    rates = []
    for _, sdf in data.groupby("subject_id", sort=False):
        r = _rate(sdf)
        if np.isfinite(r):
            rates.append(r)
    if not rates:
        return float("nan")
    rates = np.asarray(rates, dtype=float)
    return float(np.mean(rates >= 0.95))
```

**Observed (real) value:** 0.2800 (var=0.0133)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2200 (var=0.0100) (Δ vs real -0.0600)
**Other theories' values on this metric (for reference):**
- pi_7: 0.4000 (var=0.0038)
- pi_8: 0.5200 (var=0.0013)
- pi_1: 0.0000 (var=0.0135)
- pi_2: 0.0000 (var=0.0088)
- pi_3: 0.0000 (var=0.0004)
- pi_4: 0.0000 (var=0.0507)
- pi_5: 0.6600 (var=0.0033)
- pi_6: 0.0000 (var=0.0022)
- pi_8_1: 0.3200 (var=0.0029)

### Experiment 14
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # ------------------------------------------------------------------
    # EXCESS-DISPERSION METRIC (conflict staircase + congruent asymptote)
    #
    # Per subject:
    #   x = P(choose the fewer-endorsement side | CONFLICT trials)
    #       Conflict trials: the option with FEWER endorsements carries the
    #       HIGHER total validity mass (the perfect-vs-coins staircase,
    #       8 pairs x 4 reps = 32 trials/subject).
    #   r = P(choose the fewer-endorsement side | SATURATED CONGRUENT
    #       trials) (fewer-endorsed side carries >= 3 LESS validity mass;
    #       8 pairs x 4 reps = 32 trials/subject).
    #   v = (x - 0.21)^2 - (r - 0.96)^2
    #
    # 0.21 = a priori conflict-rate reference (midpoint of the SVD
    # box-implied range [0.13, 0.29]); 0.96 = the asymptote ceiling
    # (1 - eps/2 at eps = 0.08), which sits above SVD's hard cap of
    # 0.94 (eps >= 0.12).
    #
    # v is each subject's squared conflict-rate deviation from the
    # reference that is NOT paid for by a lapse-driven asymptote
    # shortfall.  Under SVD, all between-subject conflict dispersion
    # comes from (w, eps) heterogeneity, and eps >= 0.12 forces
    # r <= 0.94, so the two squared terms cancel subject-by-subject
    # (E[v] ~ 0.001).  Under GDRD-B, the bimodal route-weight
    # population (12% counter mode at conflict 0.35-0.53, skeptic
    # majority at 0.12-0.23) generates conflict dispersion while the
    # asymptote stays at the ceiling (eps <= 0.18), so v > 0
    # (E[v] ~ 0.011).
    #
    # Population value: root-mean-square of v across subjects (the
    # excess dispersion in rate units).  Single subject: its own v
    # (its squared contribution to the population statistic).
    # ------------------------------------------------------------------
    VALIDITIES = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.86, 0.74,
                           0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    CONFLICT_REF = 0.21
    ASYMPTOTE_REF = 0.96
    MASS_GAP_MIN = 3.0

    df = data.reset_index(drop=True)
    A = np.vstack([np.asarray(x, dtype=float).ravel()
                   for x in df["option_a_ratings"]])
    B = np.vstack([np.asarray(x, dtype=float).ravel()
                   for x in df["option_b_ratings"]])
    resp = df["response"].to_numpy(dtype=float)

    n_feat = A.shape[1]
    valid = VALIDITIES[:n_feat]

    cnt_a = A.sum(axis=1)
    cnt_b = B.sum(axis=1)
    mass_a = A @ valid
    mass_b = B @ valid

    unequal = cnt_a != cnt_b
    fewer_a = cnt_a < cnt_b
    mass_fewer = np.where(fewer_a, mass_a, mass_b)
    mass_more = np.where(fewer_a, mass_b, mass_a)

    conflict = unequal & (mass_fewer > mass_more)
    asymptote = (unequal & (mass_fewer < mass_more)
                 & ((mass_more - mass_fewer) >= MASS_GAP_MIN))

    chose_fewer = np.where(fewer_a, resp == 0.0, resp == 1.0).astype(float)

    tmp = pd.DataFrame({"sid": df["subject_id"].to_numpy(),
                        "conflict": conflict,
                        "asymptote": asymptote,
                        "cf": chose_fewer})

    sub_ids = pd.unique(tmp["sid"])
    x = tmp[tmp["conflict"]].groupby("sid")["cf"].mean().reindex(sub_ids)
    r = tmp[tmp["asymptote"]].groupby("sid")["cf"].mean().reindex(sub_ids)

    v = (x - CONFLICT_REF) ** 2 - (r - ASYMPTOTE_REF) ** 2
    v = v.dropna().to_numpy()
    if v.size == 0:
        return 0.0
    if v.size == 1:
        # single subject: its squared excess deviation (contribution
        # to the population dispersion statistic)
        return float(v[0])
    # population: root-mean-squared excess dispersion, in rate units
    return float(np.sqrt(max(0.0, float(np.mean(v)))))
```

**Observed (real) value:** 0.1152 (var=0.0110)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1051 (var=0.0014) (Δ vs real -0.0101)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1117 (var=0.0005)
- pi_7: 0.0387 (var=0.0001)
- pi_1: 0.0000 (var=0.0145)
- pi_2: 0.0000 (var=0.0239)
- pi_3: 0.0000 (var=0.0015)
- pi_4: 0.3730 (var=0.0446)
- pi_5: 0.1587 (var=0.0002)
- pi_6: 0.0000 (var=0.0107)
- pi_8_1: 0.0594 (var=0.0004)

### Experiment 15
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # EXTREME COUNTER-MODE ASSAY.
    # Per subject we form S = r_parity + r_conflict, where
    #   r_parity   = rate of choosing the PERFECT-ENDORSEMENT-HEAVY option
    #                on the count-matched parity trials (8v8 / 6v6),
    #   r_conflict = rate of choosing the 8-perfect (fewer-endorsement)
    #                option on the 8v9 conflict trials.
    # Algebra (lapse eps, skeptic weight w, count-route mass X=(1-eps)(1-w)):
    #   SVD:      S = 1.5*X + eps, X in [0.06,0.18] indep of eps in [0.12,0.24]
    #             -> true per-subject S confined to [0.23, 0.47] (w>=0.80 caps
    #                the count route at 0.20 of the core mixture).
    #   GDRD-B skeptics (87%): X = 0.25 - eps (neutrality coupling)
    #             -> S = 0.375 - 0.5*eps in [0.26, 0.30]: pinned LOW.
    #   GDRD-B counters (13%): w in [0.42,0.56] -> X in [0.34,0.51]
    #             -> S in [0.74, 0.89]: DISJOINT from SVD's entire box.
    # The two theories' pooled means on these cells are matched by
    # construction, so the dissociation lives in the tail: only GDRD-B
    # produces subjects whose (parity, conflict) pair lands above the
    # 0.47 ceiling of SVD's box. We map S through a logistic centered
    # in the empty moat (0.72, scale 0.12) and take the POPULATION
    # MAXIMUM of this bounded membership score. Under SVD the maximum
    # is a noise draw from the low zone (m ~ 0.2-0.45); under GDRD-B
    # the ~13% counter mode puts at least one subject deep in the high
    # zone (m ~ 0.6-1.0). This directly operationalises the design's
    # own falsifier ("any subject with conflict rate >= 0.40 falsifies
    # SVD outright") while pooling the parity and conflict readouts to
    # cut per-subject binomial noise.
    df = data.reset_index(drop=True)

    def _vec(x):
        if isinstance(x, np.ndarray):
            return x.astype(float)
        return np.asarray(list(x), dtype=float)

    a_list = [_vec(x) for x in df["option_a_ratings"]]
    b_list = [_vec(x) for x in df["option_b_ratings"]]

    n_perf = 8  # experts 0-7 are the perfect (v=1.0) experts in this design
    pA = np.array([v[:n_perf].sum() for v in a_list], dtype=float)
    pB = np.array([v[:n_perf].sum() for v in b_list], dtype=float)
    nA = np.array([v.sum() for v in a_list], dtype=float)
    nB = np.array([v.sum() for v in b_list], dtype=float)

    resp = df["response"].to_numpy(dtype=float)  # 0 = chose A, 1 = chose B
    chose_A = resp < 0.5
    ph_is_A = pA > pB  # option with more perfect-expert endorsements
    chose_ph = np.where(ph_is_A, chose_A, ~chose_A)

    parity_mask = (nA == nB)           # trials 1-4: 8v8 and 6v6
    conflict_mask = ((nA + nB) == 17)  # trials 5-6: 8v9

    sid = df["subject_id"].to_numpy()

    def _m(S):
        z = (S - 0.72) / 0.12
        return 1.0 / (1.0 + np.exp(-z))

    scores = []
    for s in pd.unique(sid):
        m_sub = (sid == s)
        pm = parity_mask & m_sub
        cm = conflict_mask & m_sub
        r_p = float(chose_ph[pm].mean()) if pm.sum() > 0 else 0.15
        r_c = float(chose_ph[cm].mean()) if cm.sum() > 0 else 0.205
        scores.append(float(_m(r_p + r_c)))

    # pooled: the most counter-like subject in the sample;
    # single-subject slice: that subject's own membership score.
    return float(np.max(scores))
```

**Observed (real) value:** 0.9846 (var=0.0914)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.9997 (var=0.1169) (Δ vs real +0.0151)
**Other theories' values on this metric (for reference):**
- pi_7: 0.3118 (var=0.0035)
- pi_8_1: 0.7372 (var=0.0287)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.9305 (var=0.0600)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 0.0324 (var=0.0000)
- pi_6: 1.0000 (var=0.0000)
- pi_8: 0.4974 (var=0.0098)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Population dispersion of the per-subject PERFECT-SIDE INDEX.
    # Diagnostic trials are identified structurally from the rating
    # vectors using the design validity layout: features 0-5 are the
    # six perfect experts (v = 1.0), features 6-11 are mid-validity
    # experts, features 12-19 are the eight coin experts (v = 0.5).
    # A diagnostic trial has one option carrying exactly the six
    # perfect endorsements and nothing else, while the other option
    # carries only coin endorsements (6, 7 or 8 of them) and no
    # mid-validity or perfect endorsements. On these trials the
    # analytic distrust route is saturated toward the coin side under
    # BOTH theories, so the rate of choosing the 6-perfect option
    # isolates the count-route transmission plus lapse.
    # Per-subject predictions:
    #   GDRD-B skeptics (about 87 percent): index near 0.16
    #   GDRD-B counters (about 13 percent): index near 0.42
    #   SVD subjects (all): index confined to about 0.18 - 0.24
    # The pooled statistic is the between-subject standard deviation
    # of this index: about 0.12 under GDRD-B (bimodal population,
    # super-binomial dispersion) versus about 0.085 under SVD
    # (binomial noise plus a tiny true spread). On a single
    # subject's slice the dispersion collapses to a small strictly
    # positive scale floor (1e-3 times the subject's own index), so
    # the statistic is well defined at both the pooled and the
    # per-subject level and is nearly constant across subjects.
    n_feat = 20
    indices = []
    for sid in pd.unique(data['subject_id']):
        sdf = data[data['subject_id'] == sid]
        a_cells = sdf['option_a_ratings'].values
        b_cells = sdf['option_b_ratings'].values
        r_cells = sdf['response'].values
        hits = 0
        n = 0
        for i in range(len(sdf)):
            xa = a_cells[i]
            xb = b_cells[i]
            if isinstance(xa, str):
                xa = [float(t) for t in xa.strip('[]() ').split(',') if t.strip() != '']
            if isinstance(xb, str):
                xb = [float(t) for t in xb.strip('[]() ').split(',') if t.strip() != '']
            try:
                av = np.asarray(list(xa), dtype=float).ravel()
                bv = np.asarray(list(xb), dtype=float).ravel()
            except Exception:
                continue
            if av.shape[0] != n_feat or bv.shape[0] != n_feat:
                continue
            try:
                r = int(round(float(r_cells[i])))
            except Exception:
                continue
            ap = float(av[0:6].sum())
            am = float(av[6:12].sum())
            ac = float(av[12:20].sum())
            bp = float(bv[0:6].sum())
            bm = float(bv[6:12].sum())
            bc = float(bv[12:20].sum())
            a_perf = (ap == 6.0) and (am == 0.0) and (ac == 0.0)
            b_perf = (bp == 6.0) and (bm == 0.0) and (bc == 0.0)
            if a_perf == b_perf:
                continue
            if a_perf:
                om = bm
                op = bp
                oc = bc
            else:
                om = am
                op = ap
                oc = ac
            if om != 0.0 or op != 0.0:
                continue
            if oc != 6.0 and oc != 7.0 and oc != 8.0:
                continue
            chose_perf = (r == 0 and a_perf) or (r == 1 and b_perf)
            if chose_perf:
                hits += 1
            n += 1
        if n > 0:
            indices.append(hits / float(n))
    if len(indices) == 0:
        return 0.0
    vals = np.asarray(indices, dtype=float)
    m = float(np.mean(vals))
    v = float(np.var(vals, ddof=0))
    fl = 1e-3 * m
    return float(np.sqrt(v + fl * fl))
```

**Observed (real) value:** 0.1433 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1649 (var=0.0000) (Δ vs real +0.0215)
**Other theories' values on this metric (for reference):**
- pi_8_1: 0.1323 (var=0.0000)
- pi_7: 0.0816 (var=0.0000)
- pi_1: 0.1275 (var=0.0000)
- pi_2: 0.0956 (var=0.0000)
- pi_3: 0.0181 (var=0.0000)
- pi_4: 0.0529 (var=0.0000)
- pi_5: 0.0484 (var=0.0000)
- pi_6: 0.0523 (var=0.0000)
- pi_8: 0.1187 (var=0.0000)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Heavy-Tailed Graded Dual-Route Distrust (HT-GDRD). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic): every endorsement is a charge on its option — for the distrustful majority a LIABILITY whose magnitude grows with the endorser's raw claimed validity, d_j = c0 + c1*v_j^kappa; for a small 'trust-leaning' minority an ASSET proportional to the (capped) log-odds of validity, alpha*logit(v_j), that dominates the liability for high-validity endorsers. ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is an explicitly over-dispersed THREE-COMPONENT mixture driven by one uniform quantile u_pop: (i) a skeptic majority (~78%) with a wide, heavy-tailed lapse distribution (eps in [0.13, 0.27], Beta-like shaping) and a loose (not tightly coupled) route weight w in [0.82, 0.96]; (ii) an ENLARGED counter minority (~16%, up from 13%) with near-balanced routes (w in [0.46, 0.70]), low-moderate lapse, and a steep count slope gamma >= 2.5 — this mode carries the between-subject dispersion that the saturation/conflict assays (Exps 14, 16) demand; and (iii) a small trust-leaning mode (~6%) with a dominant positive micro-trust term (alpha in [0.50, 0.68]) and an elevated lapse (eps in [0.26, 0.34]) — these subjects follow high-validity (especially near-perfect) endorsers but are otherwise noisy, which reproduces the extreme counter-mode membership score of Exp 15 without poisoning the lapse-sensitive prevalence assay of Exp 13. Because every within-mode spread is wide and the mode means are re-balanced, the pooled means on the first-moment assays (Exps 3-12) are preserved while the second-moment statistics (Exps 13-16) are raised to the observed levels.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Heavy-Tailed Graded Dual-Route Distrust (HT-GDRD).
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # ROUTE 1 (weight w): per-endorsement charge
    #   d_j = c0 + c1 * v_j^kappa                     (distrust charge)
    #   D = pen(B) - pen(A) + alpha * sum_j clip(logit(v_j), 0, 3) * (a_j - b_j)
    #   P_route1(A) = sigmoid(beta * D)
    # For skeptics and counters alpha is a tiny residual micro-trust edge
    # (alpha in [0, 0.05]). For the trust-leaning mode alpha is DOMINANT
    # (0.50-0.68): high-validity endorsers become net assets.
    #
    # ROUTE 2 (weight 1-w): graded fewer-endorsement preference
    #   P_route2(A) = sigmoid(gamma * (cnt_B - cnt_A))
    #
    # THREE-COMPONENT HEAVY-TAILED POPULATION, driven by one uniform
    # quantile u_pop in [0, 1]:
    #   u < 0.16        -> COUNTER (16%):  w in [0.46, 0.70],
    #                      eps in [0.13, 0.23], gamma in [2.5, 4.0]
    #   0.16 <= u < 0.94 -> SKEPTIC (78%):  eps = 0.13 + 0.14*t^0.8
    #                      (heavy-tailed over [0.13, 0.27]),
    #                      w in [0.82, 0.96] (loose, plus w_spread),
    #                      gamma in [1.5, 3.5]
    #   u >= 0.94       -> TRUST (6%):     alpha in [0.50, 0.68],
    #                      w in [0.44, 0.56] (count-leaning),
    #                      eps in [0.26, 0.34], gamma in [1.5, 3.0]
    #
    # Design notes vs the arbiter's numerical sketch: the skeptic lapse
    # floor is 0.13 (not 0.05) because Exp 13's ceiling-prevalence assay
    # (observed 0.28) punishes any further low-lapse mass in the majority
    # mode; the genuinely low-lapse tail mass is placed in the counter
    # mode instead. The trust mode is given an ELEVATED lapse and a
    # count-leaning route weight so that its conflict/asymptote behavior
    # on the Exp-14 excess-dispersion assay stays non-negative and its
    # saturated perfect-vs-coin behavior on Exp 12 stays near the lapse
    # floor, preserving the pooled means the distrust core already fits.
    # History is ignored: validities are communicated in the instructions,
    # so all charges are fixed for the whole block.
    if isinstance(state, dict):
        a_vec = np.asarray(state["option_a_ratings"], dtype=float).ravel()
        b_vec = np.asarray(state["option_b_ratings"], dtype=float).ravel()
        stim = np.vstack([a_vec, b_vec])
    else:
        stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HT-GDRD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v_pow = np.clip(v, 0.5, 1.0)          # power route: v=1 harmless
    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)     # guard the logit

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha_base = float(parameters["alpha"])
    u_pop = float(parameters["u_pop"])
    w_spread = float(parameters["w_spread"])

    # --- Three-component heavy-tailed population mixture ---
    f_counter = 0.16
    f_trust_lo = 0.94
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority: near-balanced routes, low-moderate lapse,
        # decisive graded count slope (gamma >= 2.5).
        t = u / f_counter
        w_mix = 0.46 + 0.24 * t          # [0.46, 0.70]
        epsilon = 0.13 + 0.10 * t        # [0.13, 0.23]
        gamma = 2.5 + 1.5 * t            # [2.5, 4.0]
        alpha_mode = alpha_base
    elif u < f_trust_lo:
        # Skeptic majority: analytic distrust dominant, with a genuinely
        # wide (heavy-tailed, Beta-like) lapse spread and a LOOSE route
        # weight (no tight neutrality coupling -- the loose version is
        # what reproduces the observed Exp-14 excess dispersion).
        t = (u - f_counter) / (f_trust_lo - f_counter)
        epsilon = 0.13 + 0.14 * (t ** 0.8)   # [0.13, 0.27], mean ~0.21
        w_mix = 0.86 + 0.08 * t + w_spread     # ~[0.82, 0.96]
        w_mix = float(min(max(w_mix, 0.82), 0.96))
        gamma = 1.5 + 2.0 * t            # [1.5, 3.5]
        alpha_mode = alpha_base
    else:
        # Trust-leaning residual mode: the positive micro-trust term is
        # DOMINANT, so near-perfect / high-validity endorsers act as net
        # assets; the route weight is count-leaning and the lapse is
        # elevated (a validity-following but inattentive subject).
        t = (u - f_trust_lo) / (1.0 - f_trust_lo)
        alpha_mode = 0.50 + 0.18 * t     # [0.50, 0.68]
        w_mix = 0.44 + 0.12 * t          # [0.44, 0.56]
        epsilon = 0.26 + 0.08 * t        # [0.26, 0.34]
        gamma = 1.5 + 1.5 * t            # [1.5, 3.0]
    epsilon = float(min(max(epsilon, 0.02), 0.45))

    # --- Route 1: validity-proportional charge plus micro-trust edge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Micro-trust: logit(0.5) = 0 (coin-flip experts contribute nothing);
    # capped at 3 so perfect experts exert a bounded pull.
    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)
    trust_edge = float(np.dot(lg, a - b))

    # D > 0 favors A.
    D = (pen_b - pen_a) + alpha_mode * trust_edge
    p_route1_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid

    # --- Route 2: graded endorsement-count preference ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    p_route2_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))

    # --- Dual-route mixture ---
    p_a = w_mix * p_route1_a + (1.0 - w_mix) * p_route2_a
    p_a = float(min(max(p_a, 0.0), 1.0))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- c0: [0.06, 0.13]
- c1: [1.00, 1.35]
- kappa: [1.6, 2.1]
- beta: [1.3, 2.4]
- alpha: [0.0, 0.05]
- u_pop: [0, 1]
- w_spread: [-0.02, 0.02]
- validities: validities

`rationale`: (none)

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is a faithful implementation of the arbiter's prescribed HT-GDRD family (validity-scaled liability charge + graded count route + three-component over-dispersed population), and it was ACCEPTED by the gate. Qualitatively it captures every experiment's sign and gross magnitude — a large improvement over pi_7 on the second-moment assays (Exp 14: 0.1051 vs observed 0.1152, where pi_7 gave 0.0387; Exp 15: 0.9997 vs 0.9846; Exp 16: 0.1649 vs 0.1433, where pi_7 gave 0.0816). However, the residual errors are not noise — they form one coherent systematic pattern: the model is uniformly too 'soft', i.e. pulled toward chance. Every distrust/saturation assay deviates in the direction predicted by an excess of lapse: Exp 5 (0.1514 vs 0.1143), Exp 6 (0.2425 vs 0.1925), Exp 7 (0.4184 vs 0.3815), Exp 9 (0.4142 vs 0.3729), Exp 10 (0.5408 vs 0.5096) all overshoot toward 0.5, while the skeptic-compliance and saturation assays undershoot away from their ceilings: Exp 8 (0.7388 vs 0.7808), Exp 11 (0.7338 vs 0.7875), Exp 12 (0.7870 vs 0.8320), Exp 13 (0.22 vs 0.28), Exp 3 (|−0.5042| vs |−0.5467|). A single population-level knob — the lapse distribution — explains ~12 of the 16 deviations in sign. Critically, the proposer's stated justification for raising the skeptic lapse floor to 0.13 ('Exp 13's ceiling-prevalence assay punishes any further low-lapse mass') is empirically backwards: the candidate UNDERSHOOTS Exp 13 (0.22 vs observed 0.28), meaning the assay rewards more low-lapse mass, exactly as the arbiter's original 0.05 floor anticipated. The between-subject variance statistics (Exp 1: 0.0035 vs 0.0205; Exp 3: 0.0127 vs 0.0421; Exp 11: 0.0254 vs 0.0537; Exp 12: 0.0296 vs 0.0438) are also uniformly under-predicted, which is the same symptom: excessive lapse washes out subject-level differences. Because the errors are systematic and one-dimensional, a targeted in-family adjustment should beat the current loss rather than requiring re-design.
Rationale: Keep the HT-GDRD mechanism exactly as prescribed (dual route, three-component mixture, logit micro-trust mode) — it is working. The single mis-calibrated knob is the population lapse distribution, which is set too high everywhere (skeptic mean ~0.21, counter 0.13–0.23, trust 0.26–0.34). Recommended adjustments on the accepted base: (1) lower the skeptic lapse range to roughly [0.07, 0.22] with a heavy lower tail (the 0.13 floor starves the Exp 13 ceiling-prevalence assay: observed 0.28 > simulated 0.22, so more genuinely low-lapse mass is needed — do NOT repeat the claim that Exp 13 punishes low lapse; the data show the opposite); (2) lower the counter-mode lapse to ~[0.07, 0.17]; (3) lower the trust-mode lapse to ~[0.20, 0.29]. This one shift should simultaneously pull Exps 5–7, 9, 10 down toward observed, push Exps 8, 11, 12, 13 up toward observed, deepen Exp 3, and raise the under-predicted between-subject variances. Two guardrails while doing so: (a) Exp 1 already slightly overshoots (0.7436 vs 0.7117) and will rise with lower lapse — compensate with a small reduction in beta or c1 (mid-range rather than the top of their boxes); (b) Exp 16 already slightly overshoots dispersion (0.1649 vs 0.1433) — since lower lapse increases index dispersion, hold the counter-mode mass at ~16% (do NOT enlarge it toward the arbiter's 20–25% sketch; the current 16% is what fixed Exp 14 at 0.1051 vs 0.1152) and, if needed, slightly narrow the counter w-range ceiling from 0.70 to ~0.64. Do not switch mechanism families or restructure the mixture; this is a lapse-calibration fix within the prescribed HT-GDRD design.

**Outcome of this advice:** iter 1 candidate loss=0.0607 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
```json
{
  "theory": "Heavy-Tailed Graded Dual-Route Distrust (HT-GDRD). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic): every endorsement is a charge on its option — for the distrustful majority a liability whose magnitude grows with the endorser's raw claimed validity, d_j = c0 + c1*v_j^kappa; for a small 'trust-leaning' minority an asset proportional to the (capped) log-odds of validity, alpha*logit(v_j), that dominates the liability for high-validity endorsers. ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is an over-dispersed three-component mixture driven by one uniform quantile u_pop: (i) a skeptic majority (~78%) with a heavy-tailed lapse distribution concentrated at GENUINELY LOW lapse (eps in [0.07, 0.22], Beta-like shaping with mass near the floor) and a loose route weight w in [0.82, 0.96]; (ii) a counter minority (~16%) with near-balanced routes (w in [0.46, 0.64]), low lapse (eps in [0.07, 0.17]) and a steep count slope gamma >= 2.5; and (iii) a small trust-leaning mode (~6%) with dominant positive micro-trust (alpha in [0.50, 0.68]) and moderately elevated lapse (eps in [0.20, 0.29]). The single systematic mis-calibration diagnosed in iteration 1 — an everywhere-excess lapse pulling every assay toward chance and washing out between-subject variance — is repaired by lowering all three mode lapse bands, with a compensating reduction in the population sigmoid slope beta so the TTB-agreement assay (Exp 1) does not overshoot further.",
  "predict": "def predict(parameters, state, history):\n    # Heavy-Tailed Graded Dual-Route Distrust (HT-GDRD), lapse-\n    # recalibrated pass per the iter-1 critique. The mechanism is\n    # UNCHANGED (dual route, three-component mixture, logit micro-\n    # trust mode); the single mis-calibrated knob was the population\n    # lapse distribution, which was too high in every mode. Edits:\n    #   (1) skeptic lapse  [0.13, 0.27] -> [0.07, 0.22], heavy lower\n    #       tail (t**0.8 shaping) so mass concentrates near the floor\n    #       (feeds the Exp-13 ceiling-prevalence assay, observed 0.28);\n    #   (2) counter lapse   [0.13, 0.23] -> [0.07, 0.17];\n    #   (3) trust lapse     [0.26, 0.34] -> [0.20, 0.29];\n    #   (4) counter w ceiling 0.70 -> 0.64 (Exp-16 dispersion guard);\n    #   (5) beta range      [1.3, 2.4] -> [1.2, 2.0] (compensates the\n    #       Exp-1 overshoot that lower lapse would otherwise worsen).\n    # Counter-mode mass stays at 16% (do NOT enlarge; it is what holds\n    # Exp 14 near observed).\n    if isinstance(state, dict):\n        a_vec = np.asarray(state[\"option_a_ratings\"], dtype=float).ravel()\n        b_vec = np.asarray(state[\"option_b_ratings\"], dtype=float).ravel()\n        stim = np.vstack([a_vec, b_vec])\n    else:\n        stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"HT-GDRD expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    a, b = stim[0], stim[1]\n    n_features = stim.shape[1]\n\n    v = np.asarray(parameters[\"validities\"], dtype=float)\n    if v.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {v.shape[0]} != n_features {n_features}.\"\n        )\n    v_pow = np.clip(v, 0.5, 1.0)          # power route: v=1 harmless\n    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)     # guard the logit\n\n    kappa = float(parameters[\"kappa\"])\n    beta = float(parameters[\"beta\"])\n    c0 = float(parameters[\"c0\"])\n    c1 = float(parameters[\"c1\"])\n    alpha_base = float(parameters[\"alpha\"])\n    u_pop = float(parameters[\"u_pop\"])\n    w_spread = float(parameters[\"w_spread\"])\n\n    # --- Three-component heavy-tailed population mixture ---\n    f_counter = 0.16\n    f_trust_lo = 0.94\n    u = min(max(u_pop, 0.0), 1.0)\n    if u < f_counter:\n        # Counter minority: near-balanced routes, LOW lapse,\n        # decisive graded count slope (gamma >= 2.5).\n        t = u / f_counter\n        w_mix = 0.46 + 0.18 * t          # [0.46, 0.64]\n        epsilon = 0.07 + 0.10 * t        # [0.07, 0.17]\n        gamma = 2.5 + 1.5 * t            # [2.5, 4.0]\n        alpha_mode = alpha_base\n    elif u < f_trust_lo:\n        # Skeptic majority: analytic distrust dominant, with a\n        # heavy-tailed lapse spread concentrated at GENUINELY LOW\n        # lapse (floor 0.07) and a LOOSE route weight.\n        t = (u - f_counter) / (f_trust_lo - f_counter)\n        epsilon = 0.07 + 0.15 * (t ** 0.8)   # [0.07, 0.22], mean ~0.15\n        w_mix = 0.86 + 0.08 * t + w_spread     # ~[0.82, 0.96]\n        w_mix = float(min(max(w_mix, 0.82), 0.96))\n        gamma = 1.5 + 2.0 * t            # [1.5, 3.5]\n        alpha_mode = alpha_base\n    else:\n        # Trust-leaning residual mode: dominant positive micro-trust,\n        # count-leaning route weight, moderately elevated lapse.\n        t = (u - f_trust_lo) / (1.0 - f_trust_lo)\n        alpha_mode = 0.50 + 0.18 * t     # [0.50, 0.68]\n        w_mix = 0.44 + 0.12 * t          # [0.44, 0.56]\n        epsilon = 0.20 + 0.09 * t        # [0.20, 0.29]\n        gamma = 1.5 + 1.5 * t            # [1.5, 3.0]\n    epsilon = float(min(max(epsilon, 0.02), 0.45))\n\n    # --- Route 1: validity-proportional charge plus micro-trust edge ---\n    d = c0 + c1 * np.power(v_pow, kappa)\n    pen_a = float(np.dot(d, a))\n    pen_b = float(np.dot(d, b))\n\n    # Micro-trust: logit(0.5) = 0 (coin-flip experts contribute nothing);\n    # capped at 3 so perfect experts exert a bounded pull.\n    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)\n    trust_edge = float(np.dot(lg, a - b))\n\n    # D > 0 favors A.\n    D = (pen_b - pen_a) + alpha_mode * trust_edge\n    p_route1_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid\n\n    # --- Route 2: graded endorsement-count preference ---\n    cnt_a = float(np.sum(a))\n    cnt_b = float(np.sum(b))\n    p_route2_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))\n\n    # --- Dual-route mixture ---\n    p_a = w_mix * p_route1_a + (1.0 - w_mix) * p_route2_a\n    p_a = float(min(max(p_a, 0.0), 1.0))\n    p_core = np.array([p_a, 1.0 - p_a], dtype=float)\n\n    # Independent lapse: with probability epsilon pick uniformly.\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "c0": "[0.06, 0.13]",
    "c1": "[1.00, 1.35]",
    "kappa": "[1.6, 2.1]",
    "beta": "[1.2, 2.0]",
    "alpha": "[0.0, 0.05]",
    "u_pop": "[0, 1]",
    "w_spread": "[-0.02, 0.02]",
    "validities": "validities"
  },
  "rationale": "Minimal-diff lapse recalibration of the ACCEPTED HT-GDRD base, per the iter-1 critique. The critique identified one coherent systematic error: the model was uniformly too soft (pulled toward chance) on ~12 of 16 assays, with between-subject variances also under-predicted — the signature of an everywhere-excess lapse. The fix lowers all three mode lapse bands: skeptic [0.13,0.27]->[0.07,0.22] with heavy lower tail (t**0.8 shaping, mean ~0.15 vs prior ~0.21), counter [0.13,0.23]->[0.07,0.17], trust [0.26,0.34]->[0.20,0.29]. This simultaneously: pulls Exps 5-7, 9, 10 down toward observed (they overshot toward 0.5); pushes Exps 8, 11, 12, 13 up toward their ceilings (they undershot); deepens Exp 3's contrast toward -0.5467; and raises the under-predicted between-subject variances (Exps 1, 3, 11, 12) because lapse no longer washes out subject-level differences. Two guardrails from the critique are honored: (a) Exp 1 already overshot (0.7436 vs 0.7117) and would rise further with lower lapse, so beta's range is trimmed from [1.3,2.4] to [1.2,2.0] to compensate; (b) Exp 16 already slightly overshot dispersion (0.1649 vs 0.1433), so the counter-mode mass stays at 16% (not enlarged toward the arbiter's 20-25% sketch) and the counter w ceiling is narrowed from 0.70 to 0.64 to hold the count-route tail in check. No mechanism, route structure, or mixture architecture is changed — this is a pure parameter-band edit inside the prescribed HT-GDRD family."
}
```
```

## Usage

```json
{
  "input_tokens": 65802,
  "output_tokens": 3255,
  "reasoning_tokens": 528,
  "provider": "Sail Research"
}
```
