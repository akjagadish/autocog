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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Propose 'RECIPROCAL-TRUST DISTRUST TALLYING (RTDT)' as the replacement for pi_7. Architecture: (1) Keep the distrust-inversion kernel verbatim (inv_j = ((1-v_j) - c*min(1-v))^gamma / Z) but steepen it: gamma in [3.0, 4.2], tau in [0.92, 0.96]. This directly repairs the under-predicted anti-validity slope (Exp5 observed -0.576 vs -0.418) and the under-shot position-crossover composite (Exp15), and the stronger kernel dominance automatically weakens count/tally allegiance toward Exp4's observed 0.158. (2) Replace the trust ramp with a RECIPROCAL one: t_j = t0 * (t1/t0)^(j/(n-1))^kappa with t0 in [0.00, 0.12] — an even lower early-trust floor than PBDT's [0.25, 0.35], fixing Exp16's severe over-prediction of primacy-rescue (observed 0.132 vs PBDT 0.264) — and t1 in [1.0, 1.6] with kappa in [0.6, 1.2], so LATE high-validity experts are not merely restored but AMPLIFIED above the flat share. This late-trusted amplification is the key novel lever: it fixes Exp10 (final 0.90-validity expert followed at 0.396 while both current theories predict ~0.28) without raising the early floor. (3) Keep the floored convex primacy gate but at the moderate range (eta [4.5, 6.5], delta [0.15, 0.30]) — the steep-gate/low-floor corner that PBDT verified. (4) Attenuate the choice rule: beta in [2.8, 4.0], epsilon in [0.12, 0.18], since both current theories systematically over-commit on allegiance metrics (Exps 4, 6, 9). (5) Evidence integration stays additive over discriminating cues, d = sum_j w_j*sign(A_j - B_j), logistic in beta*d with uniform lapse. The theory differs from PBDT in exactly three quantifiable respects (near-zero t0, super-unity t1, steeper gamma) each targeted at a documented residual, and differs from PGDT in retaining the position-dependent trusted channel that the log-Bayes-factor evidence decisively supports. It should be falsifiable on: Exp16 (predict rescue rate near 0.10-0.20), Exp10 (FEDI near 0.38-0.45), Exp4 (allegiance near 0.15-0.22), while matching PBDT on Exps 1, 2, 15, 17 where PBDT already succeeds.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_9` (overall score: 0.837)

**Description**
POSITION-BRIDGED DISTRUST TALLYING (PBDT), un-bundled-gate edition. Subjects scan expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, with the gate restored to the steep end of its box (eta high, delta low) so late low-validity helper cues are crushed relative to early ones — the structural lever separating primacy-rescue trials from validity-position-confounded layouts. Stated validities are distrust-inverted (treated as error rates) via inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z, kept at the steep bridged range [2.6, 2.9] with a high distrust share tau, so the anti-validity gradient in confounded layouts is carried by the distrust kernel alone rather than by the gate. Trust in high-validity experts ramps with position from a small-but-nonzero floor t_j = t0 + (t1-t0)*(j/(n-1))^kappa, with a slightly concave-leaning ramp (kappa below 1) so late trusted experts recover nearly the full flat share without raising the early floor. Weights combine multiplicatively, w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, with noise set between the two previous iterations so slope/contrast metrics that were over-attenuated recover while anchor allegiances keep their margin.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # PBDT: Position-Bridged Distrust Tallying, UN-BUNDLED-GATE edition
    # (iter-3). Minimal-diff edit of the ACCEPTED iter-2 base per the
    # critic's iter-2 directive, which diagnosed that the iter-2 move (b)
    # had bundled an efficacious half (distrust strengthening: the Exp 3
    # win) with a counterproductive half (gate flattening: the Exp 14
    # and Exp 13 regressions). This iteration UN-BUNDLES them:
    #   (A) REVERT THE GATE FLATTENING, KEEP THE DISTRUST STRENGTHENING:
    #       eta [5.5,6.0] -> [6.5, 8.0] (steep end), delta [0.25,0.30]
    #       -> [0.15, 0.25] (low floor), while gamma stays [2.6, 2.9]
    #       and tau stays [0.91, 0.93] exactly as in iter-2. The steep
    #       gate crushes the late 50%-validity helpers on Exp 14's
    #       trusted-early side (0.350 -> toward 0.132) and restores the
    #       M/C position families on Exp 13 (1.35 -> toward 1.84);
    #       the anti-validity gradient in confounded layouts (Exp 3)
    #       is carried by the distrust kernel, which iter 2 verified.
    #   (B) PARTIALLY ROLL BACK THE NOISE: beta [3.2,4.0] -> [3.6, 4.4]
    #       and epsilon [0.13,0.16] -> [0.11, 0.14] -- between the
    #       iter-1 and iter-2 settings. Lifts the over-attenuated slope
    #       metrics (Exps 1, 11, 12, 13, 15) at a small cost on Exps
    #       4/7, which currently have margin. Exp 2 is held as a
    #       monitored constraint (structural, noise-insensitive).
    #   (C) WIDEN THE RAMP CONVEXITY DOWNWARD: kappa [1.0,1.5] ->
    #       [0.8, 1.5], a concave-leaning ramp so the late trusted
    #       expert recovers more of the flat share (Exp 8 FEDI
    #       0.294 -> toward 0.396) without touching t0, tau, or the
    #       gate. t0/t1 unchanged from iter-2.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PBDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share of the bracket (HIGH, kept)
    eta = float(parameters["eta"])        # convexity of the primacy gate (STEEP, reverted)
    delta = float(parameters["delta"])    # floor of the primacy gate (LOW, reverted)
    gamma = float(parameters["gamma"])    # steepness of distrust inversion (kept)
    c = float(parameters["c"])           # partial centering of error rates
    t0 = float(parameters["t0"])          # trust floor at position 1 (LOW, kept)
    t1 = float(parameters["t1"])          # trust residue at the last position (HIGH, kept)
    kappa = float(parameters["kappa"])   # convexity of the trust ramp (WIDENED DOWN)
    beta = float(parameters["beta"])     # choice consistency (PARTIALLY RESTORED)
    eps = float(parameters["epsilon"])   # uniform lapse (PARTIALLY RESTORED)

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma in [2.6, 2.9] and tau in [0.91, 0.93] are preserved verbatim
    # from iter-2 -- the verified carriers of the anti-validity gradient
    # in validity-position-confounded layouts (Exp 3).
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and position alone carries structure.
        w_inv = np.full(n, 1.0 / n)

    if n > 1:
        pos = np.arange(n, dtype=float)
        # (2) TRUST RAMP with widened-down convexity: low floor t0,
        # high t1, and kappa now allowed below 1.0 so the ramp is
        # concave-leaning -- the late trusted expert recovers more of
        # the flat share earlier in the tail (the Exp 8 FEDI lever)
        # without raising the early floor that Exp 14 needs kept low.
        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
        # (3) PRIMACY GATE REVERTED TO THE STEEP END: eta in [6.5, 8.0]
        # with delta in [0.15, 0.25]. The steep, low-floored gate
        # concentrates attention on the first one or two experts and
        # crushes the late tail -- the structural fix for Exp 14's
        # late-helper over-weighting and Exp 13's M/C position
        # families. The anti-validity regression in confounded layouts
        # is carried by the (preserved) steep distrust kernel instead.
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket
    # cue-by-cue. tau in [0.91, 0.93] keeps the distrust channel
    # dominant; the trusted channel is carried by the ramp (small
    # early, near-full and now earlier-recovering late).
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (5) Deterministic attention: every discriminating cue is attended;
    # no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule -- beta in [3.6, 4.4],
    # between the iter-1 and iter-2 calibrations, so the slope metrics
    # over-attenuated by iter-2's noise (Exps 1, 11, 12, 13, 15)
    # recover while Exps 4/7 keep their remaining margin.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Uniform lapse, epsilon in [0.11, 0.14]: partially rolled back
    # from iter-2's [0.13, 0.16] per the critic's split-the-difference
    # directive. Exp 2 is NOT chased with further noise -- it is held
    # as a monitored structural constraint of the distrust family.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- tau: [0.91, 0.93]
- eta: [6.5, 8.0]
- delta: [0.15, 0.25]
- gamma: [2.6, 2.9]
- c: [0.05, 0.25]
- t0: [0.25, 0.35]
- t1: [0.95, 1.00]
- kappa: [0.8, 1.5]
- beta: [3.6, 4.4]
- epsilon: [0.11, 0.14]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2080 (var=0.0060) vs this=0.1423 (var=0.0016)
- Experiment 2: real=0.1583 (var=0.0053) vs this=0.3146 (var=0.0051)
- Experiment 3: real=-0.5756 (var=0.0186) vs this=-0.4178 (var=0.0203)
- Experiment 4: real=0.8433 (var=0.0152) vs this=0.8792 (var=0.0043)
- Experiment 5: real=0.2313 (var=0.0046) vs this=0.1894 (var=0.0018)
- Experiment 6: real=0.3739 (var=0.0026) vs this=0.3208 (var=0.0017)
- Experiment 7: real=0.7878 (var=0.0086) vs this=0.8572 (var=0.0032)
- Experiment 8: real=0.3957 (var=0.0045) vs this=0.2789 (var=0.0034)
- Experiment 9: real=0.3414 (var=0.0042) vs this=0.2954 (var=0.0028)
- Experiment 10: real=-0.1729 (var=0.0118) vs this=-0.2701 (var=0.0071)
- Experiment 11: real=0.3148 (var=0.0095) vs this=0.2873 (var=0.0067)
- Experiment 12: real=0.5396 (var=0.0423) vs this=0.5986 (var=0.0372)
- Experiment 13: real=1.8356 (var=0.2112) vs this=1.6694 (var=0.1653)
- Experiment 14: real=0.1322 (var=0.0089) vs this=0.2639 (var=0.0052)
- Experiment 15: real=1.0358 (var=0.1926) vs this=1.0273 (var=0.1192)
- Experiment 16: real=0.3217 (var=0.0143) vs this=0.3709 (var=0.0137)
- Experiment 17: real=-9.1163 (var=0.3521) vs this=-8.3125 (var=0.4421)
- Experiment 18: real=-0.9319 (var=0.0016) vs this=-1.5890 (var=0.0021)


---

### `pi_7` (overall score: 0.801)

**Description**
PRIMACY-GATED DISTRUST TALLYING (PGDT), floored-gate edition with a rebalanced distrust share. Subjects scan the expert ratings in presentation order and attention is deployed through a steep convex primacy gate with a substantial nonzero floor: g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention but late experts retain a meaningful attention residue. Independently, the communicated validities are distrust-inverted exactly as in the shared kernel: subjects act as if each stated validity were the expert's error rate, inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z. The two kernels combine MULTIPLICATIVELY: w_j proportional to g_j * [(1-tau)/n + tau*inv_j], normalized. The refinement this iteration is a rebalanced distrust share: tau is lowered to [0.82, 0.92] so the flat component (1-tau)/n of the multiplicative bracket grows from ~1.5% to ~4-5% per cue. This directly feeds trusted high-validity LATE experts (whose bracket value is structurally pinned at the flat share regardless of gate shape), softens the distrust-driven allegiance contrasts that overshoot their targets, and pulls the effective weights off pure anti-validity where that overshoots. A modestly raised gate floor delta in [0.20, 0.45] compensates on decreasing-validity layouts by further concentrating the product weights onto the back-loaded low-validity experts, protecting the anti-validity regression. Evidence is integrated additively over all discriminating cues, d = sum_j w_j * sign(A_j - B_j), and choice is logistic in beta*d with a small uniform lapse epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # PGDT: Primacy-Gated Distrust Tallying, FLOORED-GATE edition,
    # REBALANCED-DISTRUST-SHARE variant.
    # iter-4 minimal-diff edit per critic's iter-3 directive:
    #   (1) PRIMARY KNOB: LOWER TAU. Range moved from [0.90, 0.97] to
    #       [0.82, 0.92] (still inside the arbiter's prescribed
    #       [0.80, 0.97]). This grows the flat share (1-tau)/n of the
    #       multiplicative bracket from ~1.5% to ~4-5% per cue, which is
    #       the ONLY lever that can move Exp8's final 0.90-validity
    #       expert (its weight is structurally pinned at the flat share
    #       regardless of gate shape), and simultaneously softens the
    #       over-committed distrust allegiances on Exp2/Exp4/Exp7.
    #   (2) COMPENSATING KNOB: RAISE DELTA CEILING. Range moved from
    #       [0.15, 0.40] to [0.20, 0.45] so the iter-2->3 trajectory's
    #       verified leverage on Exp3's back-loading (stronger
    #       anti-validity as the early-gate flattens) compensates the
    #       Exp3 weakening that the tau drop would otherwise cause.
    #   (3) Everything else frozen: eta, gamma, c, beta, epsilon.
    # Mechanism structure unchanged from the accepted iter-3 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PGDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the distrust kernel
    eta = float(parameters["eta"])       # convexity of the primacy gate
    delta = float(parameters["delta"])  # floor of the primacy gate
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])     # choice consistency
    eps = float(parameters["epsilon"])  # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and the gate alone carries position.
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GATE WITH RAISED FLOOR over presentation
    # positions. Cue j (0-indexed) gets gate mass
    #     g_j = delta + (1-delta) * ((n - j)/(n - 1))^eta
    # with delta now in [0.20, 0.45]: the FIRST position still carries
    # the dominant share, but every late position retains a substantial
    # attention residue, so the multiplicative product never approaches
    # a pure primacy gradient and late low-validity (maximum-distrust)
    # cues keep enough weight to carry the anti-validity signature.
    # The raised ceiling further back-loads the product on decreasing-
    # validity layouts, compensating the tau drop for Exp3/Exp5.
    if n > 1:
        pos = np.arange(n, dtype=float)
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        g = np.ones(1)

    # (3) MULTIPLICATIVE combination: the gate scales the distrust
    # bracket cue-by-cue. Where the bracket is flat (tied validities)
    # this reduces to the (floored) primacy gradient; where the bracket
    # nearly vanishes (trusted high-validity early experts) the gate
    # multiplies almost nothing and the distrust kernel decides where
    # the remaining weight mass sits. With tau lowered, the flat share
    # (1-tau)/n is large enough that trusted high-validity LATE experts
    # retain meaningful weight through the gate's floor residue -- the
    # structural fix for the under-weighted late-cue cluster (Exp6/8)
    # that no amount of gate flattening could rescue at tau >= 0.90.
    bracket = (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small uniform lapse; kept below IVC's range because the gate
    # itself supplies response dilution on late-cue trials.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- tau: [0.82, 0.92]
- eta: [4.0, 6.0]
- delta: [0.20, 0.45]
- gamma: [2.8, 3.5]
- c: [0.10, 0.25]
- beta: [2.5, 6.0]
- epsilon: [0.06, 0.15]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2080 (var=0.0060) vs this=0.1637 (var=0.0053)
- Experiment 2: real=0.1583 (var=0.0053) vs this=0.2996 (var=0.0066)
- Experiment 3: real=-0.5756 (var=0.0186) vs this=-0.4381 (var=0.0264)
- Experiment 4: real=0.8433 (var=0.0152) vs this=0.8925 (var=0.0035)
- Experiment 5: real=0.2313 (var=0.0046) vs this=0.2334 (var=0.0055)
- Experiment 6: real=0.3739 (var=0.0026) vs this=0.3028 (var=0.0052)
- Experiment 7: real=0.7878 (var=0.0086) vs this=0.8439 (var=0.0041)
- Experiment 8: real=0.3957 (var=0.0045) vs this=0.2832 (var=0.0037)
- Experiment 9: real=0.3414 (var=0.0042) vs this=0.2989 (var=0.0035)
- Experiment 10: real=-0.1729 (var=0.0118) vs this=-0.2325 (var=0.0136)
- Experiment 11: real=0.3148 (var=0.0095) vs this=0.2316 (var=0.0094)
- Experiment 12: real=0.5396 (var=0.0423) vs this=0.4678 (var=0.0344)
- Experiment 13: real=1.8356 (var=0.2112) vs this=1.1900 (var=0.2144)
- Experiment 14: real=0.1322 (var=0.0089) vs this=0.4311 (var=0.0074)
- Experiment 15: real=1.0358 (var=0.1926) vs this=1.2013 (var=0.2191)
- Experiment 16: real=0.3217 (var=0.0143) vs this=0.2921 (var=0.0253)
- Experiment 17: real=-9.1163 (var=0.3521) vs this=12.6306 (var=0.4648)
- Experiment 18: real=-0.9319 (var=0.0016) vs this=1.8638 (var=0.0018)


---

### `pi_8` (overall score: 0.773)

**Description**
POSITION-RECOVERED DISTRUST TALLYING (PRDT), flattened-inversion edition. Subjects scan the expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention while late experts retain a floor residue. The communicated validities are distrust-inverted — subjects act as if each stated validity were the expert's error rate — but the inversion kernel is now comparatively FLAT: inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z with gamma in [1.6, 2.2], so low-validity experts dominate only moderately rather than monopolizing the weight mass. Trust itself recovers with presentation position: t_j = t0 + (t1-t0)*(j/(n-1))^kappa with t0 pinned at exactly zero — total skepticism toward early-communicated high-validity claims — ramping to a strong late-position residue with low convexity so credibility returns across the mid-late positions. Weights combine multiplicatively: w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon. The theoretical claim: the anti-validity signature in human data is a MODERATE distrust gradient, not a steep one — the steep primacy gate supplies the position structure while a flattened distrust kernel lets early high-validity cues retain enough weight to produce the mild TTB-following, weak tally-allegiance, and moderate validity-adherence observed across the pooled experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # PRDT: Position-Recovered Distrust Tallying.
    # iter-6 FLATTENED-INVERSION pass per the critic's iter-5 directive.
    # Architecture UNCHANGED from the accepted iter-4 base (distrust
    # inversion + position-growing trust residue + floored primacy gate,
    # multiplicative combination, additive integration, logistic choice).
    # The iter-5 experiment falsified the tau lever empirically: lowering
    # tau moved the ENTIRE over-distrust cluster (Exps 1,2,3,5,7,10) AWAY
    # from target, because in decreasing-validity designs the trust ramp
    # favors the same late low-validity cues the distrust kernel favors.
    # The one knob that reduces error-rate-cue dominance WITHOUT shifting
    # weight into the position-ramping trust channel is GAMMA:
    #   (1) TAU REVERTED to the validated iter-4 range [0.90, 0.93].
    #   (2) MAIN LEVER -- GAMMA LOWERED to [1.6, 2.2]: flattening inv_j
    #       toward uniform lets the (now steep) primacy gate express
    #       itself as moderate high-validity following on Exps 1/2/3/5/7,
    #       deflates the late 0.50-validity helpers' distrust dominance
    #       on Exp 14, and RAISES the final 0.90-expert's distrust weight
    #       on Exp 8 -- the one channel that can break the structural
    #       t1*(1-tau)/n cap.
    #   (3) KEEP the iter-5 gate calibration the residuals validated
    #       despite the rejection: eta [5.0, 6.0], delta [0.25, 0.30]
    #       (produced the near-exact Exp 13 = 1.796 vs 1.836, best-ever
    #       Exp 14, improved Exps 8 and 11).
    #   (4) t0 PINNED AT EXACTLY 0.0 (position-1 trusted bracket must
    #       vanish); t1 raised to [0.85, 1.00] as the directed final
    #       nudge for Exp 8 (safe for Exp 14: the rescue option's late
    #       helpers are 0.50-validity, distrust-dominated, and the
    #       trusted expert there sits at position 1 where t0 ~ 0).
    #   (5) epsilon raised to [0.09, 0.13] per the Exp 4 tripwire
    #       (over-determinism 0.911 vs 0.843 persists); beta held at
    #       [4.0, 5.5] -- NOT cut, NOT raised.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PRDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share of the bracket
    eta = float(parameters["eta"])         # convexity of the primacy gate
    delta = float(parameters["delta"])    # floor of the primacy gate
    gamma = float(parameters["gamma"])    # steepness of distrust inversion (LOW)
    c = float(parameters["c"])             # partial centering of error rates
    t0 = float(parameters["t0"])           # trust residue at position 1 (= 0)
    t1 = float(parameters["t1"])           # trust residue at the last position
    kappa = float(parameters["kappa"])    # convexity of the trust ramp (LOW)
    beta = float(parameters["beta"])      # choice consistency
    eps = float(parameters["epsilon"])   # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma now in [1.6, 2.2]: the kernel is comparatively FLAT, so
    # low-validity cues dominate only moderately -- the moderate
    # anti-validity signature the pooled data demand.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and position alone carries structure.
        w_inv = np.full(n, 1.0 / n)

    if n > 1:
        pos = np.arange(n, dtype=float)
        # (2) POSITION-GROWING TRUST RESIDUE: total skepticism toward
        # early-communicated trusted claims (t0 = 0 exactly), progressive
        # credibility concession as trusted experts appear later, with
        # LOW convexity kappa so trust returns across mid-late positions.
        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
        # (3) STEEP CONVEX PRIMACY GATE WITH FLOOR over presentation order
        # (the iter-5 calibration validated by the residuals: eta in
        # [5.0, 6.0], delta in [0.25, 0.30]).
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket cue-by-cue.
    # The bracket's trusted channel is modulated by the position-growing
    # trust residue, so an early trusted expert's bracket vanishes (t0 = 0)
    # while a late trusted expert regains up to the full flat share (t1).
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (5) Deterministic attention: every discriminating cue is attended;
    # no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # Uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- tau: [0.90, 0.93]
- eta: [5.0, 6.0]
- delta: [0.25, 0.30]
- gamma: [1.6, 2.2]
- c: [0.0, 0.3]
- t0: [0.0, 0.0]
- t1: [0.85, 1.00]
- kappa: [1.5, 3.0]
- beta: [4.0, 5.5]
- epsilon: [0.09, 0.13]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2080 (var=0.0060) vs this=0.1093 (var=0.0009)
- Experiment 2: real=0.1583 (var=0.0053) vs this=0.4033 (var=0.0047)
- Experiment 3: real=-0.5756 (var=0.0186) vs this=-0.4835 (var=0.0191)
- Experiment 4: real=0.8433 (var=0.0152) vs this=0.9133 (var=0.0028)
- Experiment 5: real=0.2313 (var=0.0046) vs this=0.1841 (var=0.0032)
- Experiment 6: real=0.3739 (var=0.0026) vs this=0.3092 (var=0.0025)
- Experiment 7: real=0.7878 (var=0.0086) vs this=0.8472 (var=0.0044)
- Experiment 8: real=0.3957 (var=0.0045) vs this=0.2654 (var=0.0024)
- Experiment 9: real=0.3414 (var=0.0042) vs this=0.2936 (var=0.0038)
- Experiment 10: real=-0.1729 (var=0.0118) vs this=-0.2524 (var=0.0104)
- Experiment 11: real=0.3148 (var=0.0095) vs this=0.2535 (var=0.0084)
- Experiment 12: real=0.5396 (var=0.0423) vs this=0.5770 (var=0.0378)
- Experiment 13: real=1.8356 (var=0.2112) vs this=1.7233 (var=0.1382)
- Experiment 14: real=0.1322 (var=0.0089) vs this=0.3094 (var=0.0064)
- Experiment 15: real=1.0358 (var=0.1926) vs this=0.7713 (var=0.1699)
- Experiment 16: real=0.3217 (var=0.0143) vs this=0.4993 (var=0.0158)
- Experiment 17: real=-9.1163 (var=0.3521) vs this=-23.1646 (var=0.3828)
- Experiment 18: real=-0.9319 (var=0.0016) vs this=-4.8657 (var=0.0019)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0830 -> ACCEPTED
- iter 2: loss=0.0843 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0830 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Design validities (fixed by experiment): strictly decreasing,
    # so the TTB cue order is simply feature 0,1,2,3,4.
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_vec(x):
        if isinstance(x, str):
            return np.array(
                [int(v) for v in x.strip("[]()").replace(",", " ").split()]
            )
        return np.asarray(x, dtype=float)

    hits = []
    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = _to_vec(a_cell)
        b = _to_vec(b_cell)
        if a.size == 0 or a.shape != b.shape:
            continue

        # --- TTB winner: first discriminating cue in validity order ---
        ttb_winner = None
        for j in cue_order:
            if j >= a.size:
                continue
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue  # no discriminating cue: TTB guesses, uninformative

        # --- Tallying winner: strict feature-wise win counts ---
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally ties -> Tallying guesses at 0.5, dilutes contrast
        tally_winner = 0 if a_wins > b_wins else 1

        if tally_winner == ttb_winner:
            continue  # heuristics agree: uninformative

        # Pure conflict trial: TTB's option vs the tally majority's option.
        hits.append(1.0 if int(resp) == ttb_winner else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2080 (var=0.0060)
**Previous candidate values (this loop):**
  - iter 1: 0.1467 (var=0.0022) (Δ vs real -0.0613)
  - iter 2 (most recent): 0.1683 (var=0.0019) (Δ vs real -0.0397)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8443 (var=0.0128)
- pi_2: 0.1357 (var=0.0112)
- pi_3: 0.2377 (var=0.0050)
- pi_4: 0.2197 (var=0.0050)
- pi_5: 0.1807 (var=0.0022)
- pi_5_1: 0.1870 (var=0.0059)
- pi_6: 0.1750 (var=0.0040)
- pi_7: 0.1637 (var=0.0053)
- pi_8: 0.1093 (var=0.0009)
- pi_9: 0.1423 (var=0.0016)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-vs-TTB choice-allegiance contrast.

    For every trial we determine (i) the TALLYING winner (option with more
    strict feature-wise wins) and (ii) the TAKE-THE-BEST winner (winner on
    the highest-validity discriminating cue, validities fixed by design as
    [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]).  The metric is

        mean( response == tally winner ) - mean( response == TTB winner )

    On the 4 trial types where both heuristics pick the same option
    (trials 1, 3, 5, 7 and mirrors) the two indicators are identical and
    cancel.  On the 12 conflict trial types they are exact complements, so
    the metric reduces to 0.75 * (2*P(tally winner | conflict) - 1): strongly
    positive under Tallying, strongly negative under TTB.
    """
    validities = np.array([0.95, 0.9, 0.85, 0.65, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    tally_match = []
    ttb_match = []

    for ca, cb, cr in zip(a_col, b_col, r_col):
        a = np.asarray(ca, dtype=float).ravel()
        b = np.asarray(cb, dtype=float).ravel()
        r = int(cr)

        # Tallying winner: more strict feature-wise wins (ties on a feature
        # contribute to neither side).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1  # undiscriminating tally -> excluded

        # TTB winner: first discriminating cue in descending-validity order.
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break

        if tally_winner >= 0:
            tally_match.append(1.0 if r == tally_winner else 0.0)
        if ttb_winner >= 0:
            ttb_match.append(1.0 if r == ttb_winner else 0.0)

    if len(tally_match) == 0 or len(ttb_match) == 0:
        return 0.0

    return float(np.mean(tally_match) - np.mean(ttb_match))
```

**Observed (real) value:** 0.1583 (var=0.0053)
**Previous candidate values (this loop):**
  - iter 1: 0.3042 (var=0.0064) (Δ vs real +0.1458)
  - iter 2 (most recent): 0.2892 (var=0.0059) (Δ vs real +0.1308)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5479 (var=0.0187)
- pi_1: -0.4837 (var=0.0228)
- pi_3: 0.1658 (var=0.0093)
- pi_4: 0.2079 (var=0.0051)
- pi_5: 0.2308 (var=0.0086)
- pi_5_1: 0.2875 (var=0.0062)
- pi_6: 0.2408 (var=0.0082)
- pi_7: 0.2996 (var=0.0066)
- pi_8: 0.4033 (var=0.0047)
- pi_9: 0.3146 (var=0.0051)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_vec(x):
    """Robustly convert a cell (list/tuple/ndarray/str) to a float 1-D array."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.asarray(list(x), dtype=float)
    s = str(x).strip().replace("[", " ").replace("]", " ").replace(",", " ")
    return np.asarray(s.split(), dtype=float)


def metric(data: pd.DataFrame) -> float:
    df = data.reset_index(drop=True)
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    A = np.vstack([_to_vec(x) for x in df["option_a_ratings"]])
    B = np.vstack([_to_vec(x) for x in df["option_b_ratings"]])
    n_feat = A.shape[1]

    # Design validities [0.95, 0.85, 0.75, 0.65, 0.60]; fall back gracefully.
    v = np.array([0.95, 0.85, 0.75, 0.65, 0.60], dtype=float)
    if v.shape[0] != n_feat:
        v = np.linspace(0.95, 0.60, n_feat)

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = (np.asarray(df["response"], dtype=float) == 0.0).astype(float)

    a_win = A > B
    b_win = B > A

    # Signed tally margin (Tallying's ONLY explanatory variable).
    margin = a_win.sum(axis=1).astype(float) - b_win.sum(axis=1).astype(float)

    # Validity-weighted evidence difference (NVW's driving quantity).
    d = (a_win * v).sum(axis=1) - (b_win * v).sum(axis=1)

    # OLS: y ~ 1 + margin + d. Return the partial coefficient on d.
    X = np.column_stack([np.ones(n_rows), margin, d])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])
```

**Observed (real) value:** -0.5756 (var=0.0186)
**Previous candidate values (this loop):**
  - iter 1: -0.6225 (var=0.0132) (Δ vs real -0.0469)
  - iter 2 (most recent): -0.5569 (var=0.0150) (Δ vs real +0.0187)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2763 (var=0.0165)
- pi_2: -0.0069 (var=0.0141)
- pi_1: 0.7613 (var=0.0601)
- pi_4: -0.4120 (var=0.0294)
- pi_5: -0.5397 (var=0.0203)
- pi_5_1: -0.5621 (var=0.0241)
- pi_6: -0.4702 (var=0.0232)
- pi_7: -0.4381 (var=0.0264)
- pi_8: -0.4835 (var=0.0191)
- pi_9: -0.4178 (var=0.0203)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anchor-allegiance: proportion of trials with a large tally margin
    (|a_wins - b_wins| >= 6) on which the subject chose the option that
    wins on more features (the tally winner).

    Under Tallying, these are the unambiguous anchors (8-0 and 7-1 splits:
    design trials 7, 8, 21, 22, 23, 24 and their 4 repetitions = 24 trials
    per subject), where the softmax over tallies predicts near-deterministic
    choice for any non-trivial beta. Under NVW, the capacity-interference
    factor delta^4 crushes the effective inverse temperature to
    beta_eff <= 11 * 0.30^4 ~ 0.09, so even the maximum validity-weighted
    evidence difference (~6) yields P ~ sigmoid(0.5) <= 0.64; attention
    lapses and the uniform-mixing epsilon only pull this further toward
    0.5. The two theories therefore make sharply separated predictions on
    this proportion, and the A/B-mirrored anchors cancel side bias.
    """
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    n_anchor = 0
    n_hit = 0

    for ra, rb, resp in zip(a_col, b_col, r_col):
        va = np.asarray(ra, dtype=float)
        vb = np.asarray(rb, dtype=float)
        if va.shape != vb.shape:
            continue

        a_wins = float(np.sum(va > vb))
        b_wins = float(np.sum(vb > va))
        margin = a_wins - b_wins

        if abs(margin) >= 6:
            n_anchor += 1
            winner_is_a = margin > 0
            chose_a = (int(resp) == 0)
            if chose_a == winner_is_a:
                n_hit += 1

    if n_anchor == 0:
        return 0.5
    return float(n_hit) / float(n_anchor)

```

**Observed (real) value:** 0.8433 (var=0.0152)
**Previous candidate values (this loop):**
  - iter 1: 0.8358 (var=0.0058) (Δ vs real -0.0075)
  - iter 2 (most recent): 0.8508 (var=0.0047) (Δ vs real +0.0075)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8808 (var=0.0089)
- pi_3: 0.5300 (var=0.0122)
- pi_1: 0.6133 (var=0.0060)
- pi_4: 0.8133 (var=0.0065)
- pi_5: 0.8083 (var=0.0061)
- pi_5_1: 0.8217 (var=0.0065)
- pi_6: 0.9058 (var=0.0060)
- pi_7: 0.8925 (var=0.0035)
- pi_8: 0.9133 (var=0.0028)
- pi_9: 0.8792 (var=0.0043)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Validity-adherence rate on VALIDITY-vs-RECENCY conflict trials.
    #
    # For every trial we compute two decision variables from the presented
    # cue patterns:
    #   d_v = sum_j v_j * sign(A_j - B_j)   (validity-weighted evidence for A)
    #   d_r = sum_j (j/(n-1))^3 * sign(A_j - B_j)  (convex cue-order recency
    #                                              evidence for A)
    # A trial is a CONFLICT trial when the two point in opposite directions
    # and both are non-trivial in magnitude. On those trials the metric is
    # the proportion of choices that follow the VALIDITY-implied option.
    #
    # NVW (validity-weighted integration) predicts adherence clearly above
    # chance (~0.6); SWC (recency-dominated subjective weights) predicts
    # adherence well below chance (~0.3), because on these trials its
    # recency gradient favors the opposite option.

    validities = np.array([0.95, 0.90, 0.85, 0.62, 0.50, 0.50], dtype=float)
    n = int(validities.shape[0])
    # Convex recency profile over presentation positions (eta ~ 3).
    recency = (np.arange(n, dtype=float) / float(n - 1)) ** 3.0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    hits = []
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != n or b.shape[0] != n:
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue

        d_v = float(np.dot(validities, s))
        d_r = float(np.dot(recency, s))

        # Need a real conflict with non-trivial evidence on both axes.
        if abs(d_v) < 0.30 or abs(d_r) < 0.50:
            continue
        if np.sign(d_v) == np.sign(d_r):
            continue

        validity_winner_is_a = (d_v > 0.0)
        chose_a = (int(resp) == 0)
        hits.append(1.0 if (chose_a == validity_winner_is_a) else 0.0)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2313 (var=0.0046)
**Previous candidate values (this loop):**
  - iter 1: 0.1959 (var=0.0028) (Δ vs real -0.0353)
  - iter 2 (most recent): 0.2091 (var=0.0034) (Δ vs real -0.0222)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5988 (var=0.0039)
- pi_4: 0.2816 (var=0.0052)
- pi_1: 0.7519 (var=0.0061)
- pi_2: 0.5803 (var=0.0046)
- pi_5: 0.2544 (var=0.0025)
- pi_5_1: 0.2369 (var=0.0070)
- pi_6: 0.2559 (var=0.0082)
- pi_7: 0.2334 (var=0.0055)
- pi_8: 0.1841 (var=0.0032)
- pi_9: 0.1894 (var=0.0018)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-consistency rate: fraction of choices aligned with a convex
    # cue-order recency gradient, restricted to trials where that gradient
    # is strong and its direction is unambiguous across SWC's eta range.
    #
    # SWC (advocated): late cues dominate via convex recency -> rate well
    #   above 0.5 (predicted ~0.65-0.85 across its parameter box).
    # NVW (competing): validity-weighted integration with capacity
    #   interference (beta_eff = beta*delta^3 <= ~0.30 at n=7) -> choices
    #   near chance on nearly every trial, and anti-recency or neutral on
    #   the conflict trials -> rate ~0.50.
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.5

    a_arrs = [np.asarray(list(x), dtype=float) for x in df['option_a_ratings']]
    b_arrs = [np.asarray(list(x), dtype=float) for x in df['option_b_ratings']]
    resp = pd.to_numeric(df['response'], errors='coerce').to_numpy()

    if len(a_arrs) == 0 or a_arrs[0].ndim != 1:
        return 0.5
    n = int(a_arrs[0].shape[0])
    if n < 2:
        return 0.5

    pos = np.arange(n, dtype=float) / (n - 1.0)
    w_lo = pos ** 2.5   # SWC's eta lower corner
    w_mid = pos ** 3.5  # SWC's eta midpoint
    w_hi = pos ** 4.5   # SWC's eta upper corner

    consistent = []
    for a, b, r in zip(a_arrs, b_arrs, resp):
        if a.shape[0] != n or b.shape[0] != n:
            continue
        if not np.isfinite(r):
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue
        e_lo = float(np.dot(w_lo, s))
        e_mid = float(np.dot(w_mid, s))
        e_hi = float(np.dot(w_hi, s))
        # Require the recency gradient's direction to be invariant across
        # SWC's whole eta range (drops ambiguous trials, e.g. 1 late cue vs
        # 4 mid cues, where the sign flips with eta).
        if e_lo == 0.0 or e_hi == 0.0 or np.sign(e_lo) != np.sign(e_hi):
            continue
        # Require non-trivial gradient magnitude (drops near-degenerate
        # trials where even SWC predicts ~0.52).
        if abs(e_mid) < 0.25:
            continue
        fav_a = e_mid > 0.0
        chose_a = (int(r) == 0)  # response: 0 = chose A, 1 = chose B
        consistent.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(consistent) == 0:
        return 0.5
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.3739 (var=0.0026)
**Previous candidate values (this loop):**
  - iter 1: 0.3450 (var=0.0017) (Δ vs real -0.0289)
  - iter 2 (most recent): 0.3547 (var=0.0032) (Δ vs real -0.0192)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7261 (var=0.0028)
- pi_3: 0.4847 (var=0.0034)
- pi_1: 0.3794 (var=0.0022)
- pi_2: 0.3853 (var=0.0032)
- pi_5: 0.4408 (var=0.0038)
- pi_5_1: 0.4392 (var=0.0036)
- pi_6: 0.3361 (var=0.0034)
- pi_7: 0.3028 (var=0.0052)
- pi_8: 0.3092 (var=0.0025)
- pi_9: 0.3208 (var=0.0017)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anti-validity vs. recency disagreement score.

    On trials where IVC's distrust-inverted weights and SWC's recency-dominated
    weights point to OPPOSITE options, report the proportion of choices that
    follow the IVC-preferred option. IVC-simulated data -> well above 0.5;
    SWC-simulated data -> well below 0.5.
    """
    data = data.reset_index(drop=True)
    if len(data) == 0:
        return 0.5

    # Build stimulus matrices without hashing any list-valued cells.
    a = np.vstack([np.asarray(r, dtype=float) for r in data["option_a_ratings"]])
    b = np.vstack([np.asarray(r, dtype=float) for r in data["option_b_ratings"]])
    n = a.shape[1]

    # Design validities (fallback mirrors both theories' graceful degradation).
    v = np.array([0.5, 0.9, 0.6, 0.95, 0.5, 0.6], dtype=float)
    if v.shape[0] != n:
        v = np.linspace(0.95, 0.55, n)

    # --- IVC weights at mid-range parameters (tau=0.925, kappa=2.7) ---
    tau, kappa = 0.925, 2.7
    inv = np.power(1.0 - v, kappa)
    Z = float(inv.sum())
    w_inv = inv / Z if Z > 1e-12 else np.full(n, 1.0 / n)
    w_ivc = (1.0 - tau) / n + tau * w_inv

    # --- SWC weights at mid-range parameters (lam=0.9, rho=0.9, eta=3.5) ---
    lam, rho, eta = 0.9, 0.9, 3.5
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)
    g = (1.0 - rho) * vnorm + rho * pos
    w_swc = (1.0 - lam) / n + lam * g

    # Signed discriminating-cue vector per trial.
    s = np.sign(a - b)
    d_ivc = s @ w_ivc
    d_swc = s @ w_swc

    # Keep only trials where the two theories STRICTLY disagree in sign.
    # (d_ivc == 0, e.g. the validity-matched EQ trials, is excluded.)
    mask = (d_ivc * d_swc) < 0.0
    if not np.any(mask):
        return 0.5

    resp = np.asarray(data["response"].to_numpy())
    r = resp[mask]
    ivc_prefers_a = d_ivc[mask] > 0.0
    chose_ivc_preferred = np.where(ivc_prefers_a, r == 0, r == 1)
    return float(chose_ivc_preferred.mean())
```

**Observed (real) value:** 0.7878 (var=0.0086)
**Previous candidate values (this loop):**
  - iter 1: 0.8178 (var=0.0032) (Δ vs real +0.0300)
  - iter 2 (most recent): 0.8078 (var=0.0036) (Δ vs real +0.0200)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6389 (var=0.0060)
- pi_4: 0.3622 (var=0.0067)
- pi_1: 0.1533 (var=0.0097)
- pi_2: 0.5167 (var=0.0064)
- pi_3: 0.4511 (var=0.0081)
- pi_5_1: 0.6478 (var=0.0070)
- pi_6: 0.7750 (var=0.0124)
- pi_7: 0.8439 (var=0.0041)
- pi_8: 0.8472 (var=0.0044)
- pi_9: 0.8572 (var=0.0032)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Final-Expert Dominance Index (FEDI).

    Among trials where the LAST-presented expert (feature position n-1) is the
    ONLY cue favoring one option (i.e., that option's rating vector is all
    zeros except a 1 in the final position), compute the proportion of
    choices that follow the final expert's recommendation.

    A convex-recency weighting (SWC) predicts adherence well above 0.5
    (the last cue carries the largest subjective weight at every corner of
    its parameter box). A distrust-inverted validity weighting (IVC)
    predicts adherence below 0.5, because the final expert here has a high
    stated validity (0.90) and is therefore heavily discounted.
    """
    df = data.reset_index(drop=True)

    A = np.array(
        [list(map(int, x)) for x in df['option_a_ratings']], dtype=int
    )
    B = np.array(
        [list(map(int, x)) for x in df['option_b_ratings']], dtype=int
    )
    r = df['response'].to_numpy(dtype=int)

    if A.ndim != 2 or A.shape[0] == 0:
        return float('nan')

    n_feat = A.shape[1]
    last = n_feat - 1

    a_last = A[:, last]
    b_last = B[:, last]

    # Last expert discriminates between the options.
    disc = a_last != b_last

    # The option endorsed by the final expert wins ONLY that final cue
    # (its rating vector sums to 1). This selects the single-cue and
    # multi-cue 'ladder' duels against the last expert (trials 1-14 and
    # their mirrors) and excludes trials where the last-cue option also
    # wins other cues.
    a_sel = disc & (a_last == 1) & (A.sum(axis=1) == 1)
    b_sel = disc & (b_last == 1) & (B.sum(axis=1) == 1)
    sel = a_sel | b_sel

    if not np.any(sel):
        return float('nan')

    # response: 0 = chose A, 1 = chose B.
    # Did the subject follow the final expert's endorsed option?
    followed = np.where(a_sel[sel], r[sel] == 0, r[sel] == 1)

    return float(np.mean(followed))
```

**Observed (real) value:** 0.3957 (var=0.0045)
**Previous candidate values (this loop):**
  - iter 1: 0.3082 (var=0.0042) (Δ vs real -0.0875)
  - iter 2 (most recent): 0.3104 (var=0.0044) (Δ vs real -0.0854)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6796 (var=0.0044)
- pi_5: 0.3561 (var=0.0033)
- pi_1: 0.5529 (var=0.0024)
- pi_2: 0.4039 (var=0.0053)
- pi_3: 0.4300 (var=0.0036)
- pi_5_1: 0.3393 (var=0.0058)
- pi_6: 0.2818 (var=0.0035)
- pi_7: 0.2832 (var=0.0037)
- pi_8: 0.2654 (var=0.0024)
- pi_9: 0.2789 (var=0.0034)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-following rate on count-neutral trials.
    #
    # With all six validities equal (0.95), IVC's weight vector is EXACTLY
    # flat for every admissible parameter combination (the distrust kernel
    # is constant across features, so w_j = 1/n identically). Hence on any
    # trial where the two options win an equal number of discriminating
    # cues (count difference = 0), IVC's evidence d = 0 and P(choose A) =
    # 0.50 exactly -- the lapse preserves 0.5. This is parameter-free.
    #
    # SWC, in contrast, has strictly increasing weights in cue position
    # (lambda >= 0.8, rho >= 0.8, convex recency), and on count-neutral
    # trials its evidence reduces EXACTLY to d = lambda*rho*sum_j pos_j*s_j,
    # so it must prefer the option supported by the LATER cues, with
    # subject-level choice rates roughly in [0.60, 0.79].
    #
    # The metric: among count-neutral trials, the fraction of choices that
    # go to the option favored by the later (more recent) discriminating
    # cues. IVC -> 0.50 for every subject; SWC -> well above 0.50.

    def _mat(col):
        return np.array(
            [np.asarray(v, dtype=float).ravel() for v in list(col)],
            dtype=float,
        )

    A = _mat(data['option_a_ratings'])
    B = _mat(data['option_b_ratings'])
    resp = np.asarray(pd.Series(data['response']).to_numpy(), dtype=float)

    if A.size == 0 or A.ndim != 2 or A.shape[0] == 0 or B.shape[0] != A.shape[0]:
        return 0.5
    n = int(A.shape[1])
    if n < 2:
        return 0.5

    # Convex recency profile over cue positions (any strictly increasing
    # profile gives the same favored direction on count-neutral trials;
    # 3.5 matches SWC's mid-gradient). Direction is what matters here.
    pos = (np.arange(n, dtype=float) / (n - 1.0)) ** 3.5

    S = np.sign(A - B)            # +1: cue favors A, -1: favors B, 0: tie
    count = S.sum(axis=1)         # IVC's evidence direction (flat weights)
    rec = S @ pos                 # recency evidence for A

    neutral = (np.abs(count) < 1e-9) & (np.abs(rec) > 1e-9)
    if not np.any(neutral):
        return 0.5

    chose_a = resp[neutral] == 0
    fav_a = rec[neutral] > 0      # option supported by the later cues
    followed = np.where(fav_a, chose_a, ~chose_a)
    return float(np.mean(followed))
```

**Observed (real) value:** 0.3414 (var=0.0042)
**Previous candidate values (this loop):**
  - iter 1: 0.3275 (var=0.0035) (Δ vs real -0.0139)
  - iter 2 (most recent): 0.3425 (var=0.0038) (Δ vs real +0.0011)
**Other theories' values on this metric (for reference):**
- pi_5_1: 0.5043 (var=0.0036)
- pi_4: 0.6693 (var=0.0058)
- pi_1: 0.1557 (var=0.0121)
- pi_2: 0.4986 (var=0.0044)
- pi_3: 0.4986 (var=0.0040)
- pi_5: 0.5043 (var=0.0045)
- pi_6: 0.4118 (var=0.0064)
- pi_7: 0.2989 (var=0.0035)
- pi_8: 0.2936 (var=0.0038)
- pi_9: 0.2954 (var=0.0028)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recency-gradient choice score on theory-divergent trials.
    #
    # Design validities: experts at positions 1 and 4 (0-indexed 0 and 3)
    # have validity 0.95; the other four experts have validity 0.75.
    #
    # Per trial:
    #   s_j = sign(A_j - B_j)   which option expert j favors
    #   x   = sum_j p_j * s_j   recency evidence for A, with convex
    #                           position weights p_j = (j/5)^3.5
    #   y   = +1 if A chosen, -1 if B chosen
    #
    # Scored trial families (48 of the 96 trials per subject):
    # (1) VALIDITY-MATCHED trials (types 3/4, 9/10, 13/14, 17/18, 25/26
    #     plus mirrors): the two options' discriminating experts carry
    #     identical validity multisets. Any validity-only weighting (IVC)
    #     yields exactly zero evidence, so E[x*y] = 0 for EVERY parameter
    #     combination (the uniform lapse preserves 0.5). SWC's convex
    #     recency gradient makes E[x*y] strongly positive.
    # (2) VALIDITY-CONFOUNDED trials (types 5/6, 11/12, 23/24 plus
    #     mirrors): the option whose experts sit LATER also carries
    #     strictly more 0.95 experts. IVC's distrust inversion favors
    #     the EARLY option, so E[x*y] < 0; SWC's recency gradient still
    #     favors the LATE option, so E[x*y] > 0.
    #
    # Metric = mean of x*y over these trials.
    # SWC: strongly positive (~ +0.28 across its box, range ~ +0.17..+0.41).
    # IVC: zero on the matched family for all parameters, slightly
    # negative overall (~ -0.03).

    n = 6
    V = np.array([0.95, 0.75, 0.75, 0.95, 0.75, 0.75])
    P = (np.arange(n, dtype=float) / float(n - 1)) ** 3.5
    hi = V >= 0.9

    def to_arr(cell):
        if isinstance(cell, np.ndarray):
            return np.asarray(cell, dtype=float).ravel()
        if isinstance(cell, (list, tuple)):
            return np.asarray(list(cell), dtype=float).ravel()
        txt = str(cell).strip().strip('[]()')
        parts = txt.replace(',', ' ').split()
        return np.asarray([float(t) for t in parts], dtype=float).ravel()

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    r_cells = data['response'].tolist()

    scores = []
    for i in range(len(a_cells)):
        try:
            a = to_arr(a_cells[i])
            b = to_arr(b_cells[i])
        except Exception:
            continue
        if a.shape[0] != n or b.shape[0] != n:
            continue
        try:
            resp = int(r_cells[i])
        except Exception:
            continue
        if resp != 0 and resp != 1:
            continue

        s = np.sign(a - b)
        ia = np.where(s > 0)[0]
        ib = np.where(s < 0)[0]
        if ia.size == 0 or ib.size == 0:
            continue

        va = sorted(V[ia].tolist())
        vb = sorted(V[ib].tolist())
        matched = (va == vb)

        x = float(np.dot(P, s))
        if not matched:
            if x == 0:
                continue
            late_a = x > 0
            late_hi = int(hi[ia].sum()) if late_a else int(hi[ib].sum())
            early_hi = int(hi[ib].sum()) if late_a else int(hi[ia].sum())
            if late_hi <= early_hi:
                continue

        y = 1.0 if resp == 0 else -1.0
        scores.append(x * y)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1729 (var=0.0118)
**Previous candidate values (this loop):**
  - iter 1: -0.2276 (var=0.0108) (Δ vs real -0.0547)
  - iter 2 (most recent): -0.1895 (var=0.0095) (Δ vs real -0.0166)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2647 (var=0.0159)
- pi_5_1: -0.0459 (var=0.0117)
- pi_1: -0.3350 (var=0.0112)
- pi_2: 0.0242 (var=0.0128)
- pi_3: 0.0443 (var=0.0150)
- pi_5: -0.0563 (var=0.0178)
- pi_6: -0.2233 (var=0.0216)
- pi_7: -0.2325 (var=0.0136)
- pi_8: -0.2524 (var=0.0104)
- pi_9: -0.2701 (var=0.0071)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Within-allegiance-cell OLS slope of choice on reference primacy evidence.
    #
    # Identification: group every trial by (c1, c2) where
    #   c1 = number of 0.60-expert wins for A minus for B (positions 2,3,5),
    #   c2 = total signed cue count.
    # Any position-INVARIANT validity-weighted model (IVC at ANY parameter
    # combo, plain tallying, any lapse/transform) makes P(choose A) a function
    # of (c1, c2) ONLY. Demeaning the primacy regressor within these cells
    # therefore makes the population OLS slope EXACTLY 0 under IVC,
    # parameter-free, for every subject. PW-IVC's primacy gradient breaks the
    # cell-conditional independence, producing a strictly positive slope.

    ratings_a = data['option_a_ratings']
    ratings_b = data['option_b_ratings']
    A = np.array([np.asarray(r, dtype=float) for r in ratings_a], dtype=float)
    B = np.array([np.asarray(r, dtype=float) for r in ratings_b], dtype=float)
    if A.ndim != 2 or A.shape[0] == 0:
        return 0.0
    n = A.shape[1]

    # Reference primacy kernel, eta = 5 (middle of PW-IVC's eta range [3.5, 6.5]).
    if n > 1:
        pos = np.arange(n, dtype=float)
        wp = ((n - pos) / (n - 1.0)) ** 5.0
        wp = wp / wp.sum()
    else:
        wp = np.ones(1)

    S = np.sign(A - B)
    y = 1.0 - np.asarray(data['response'], dtype=float)  # 1 = chose A

    x1 = S.dot(wp)  # primacy evidence favoring A

    # Allegiance cell for this design: validities [0.95, 0.6, 0.6, 0.95, 0.6, 0.95]
    # -> 0.60 experts at 0-indexed positions 1, 2, 4.
    if n == 6:
        c1 = S[:, 1] + S[:, 2] + S[:, 4]
    else:
        c1 = np.zeros(A.shape[0])
    c2 = S.sum(axis=1)

    g = c1.astype(int).astype(str) + "_" + c2.astype(int).astype(str)
    df = pd.DataFrame({'x1': x1, 'y': y, 'g': g})

    # Demean primacy evidence within allegiance cell: removes ALL validity-based
    # mean structure (distrust kernel at any gamma/c/tau, flat counting, lapse).
    df['x1c'] = df['x1'] - df.groupby('g')['x1'].transform('mean')
    ssx = float((df['x1c'] ** 2).sum())
    if ssx <= 1e-12:
        return 0.0
    ybar = float(df['y'].mean())
    slope = float((df['x1c'] * (df['y'] - ybar)).sum() / ssx)
    return slope
```

**Observed (real) value:** 0.3148 (var=0.0095)
**Previous candidate values (this loop):**
  - iter 1: 0.2360 (var=0.0066) (Δ vs real -0.0788)
  - iter 2 (most recent): 0.2037 (var=0.0086) (Δ vs real -0.1111)
**Other theories' values on this metric (for reference):**
- pi_5_1: -0.0034 (var=0.0097)
- pi_6: 0.1752 (var=0.0110)
- pi_1: 0.4129 (var=0.0077)
- pi_2: -0.0132 (var=0.0118)
- pi_3: 0.0021 (var=0.0076)
- pi_4: -0.3642 (var=0.0070)
- pi_5: -0.0209 (var=0.0151)
- pi_7: 0.2316 (var=0.0094)
- pi_8: 0.2535 (var=0.0084)
- pi_9: 0.2873 (var=0.0067)

### Experiment 12
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Primacy-slope metric on validity-matched cells (with X2-family
    orientation absorbed by a dummy).

    Logic: on every trial where the two options' winning-cue validity
    multisets are IDENTICAL (the 20 matched cells: D0,Z1,D1,W1,W2,D2,
    D2b,D3,D4,D5 + mirrors), the distrust kernel and the flat counting
    kernel cancel EXACTLY, so IVC predicts P(A)=0.500 parameter-free
    (the lapse preserves 0.5). PW-IVC predicts a systematic deviation
    graded by the primacy contrast C = w_prim . sign(A-B), with a
    steep convex reference gradient (eta=5).

    We also include the six X2-family cells (2 wins per side, unequal
    validity multisets). Under IVC all X2 cells share ONE probability
    that depends only on orientation (A-won-the-0.9 vs mirror), so a
    single +-1 orientation dummy Z absorbs IVC's entire mean structure
    there; the within-family position spread then loads on C.

    Metric = OLS coefficient of C in y ~ 1 + C + Z, with y = +1 (chose
    A) / -1 (chose B). Under IVC this coefficient is EXACTLY 0 in
    expectation for every admissible parameter. Under PW-IVC it is
    strictly positive (roughly 2*(1-eps)*beta*tau*rho scale), typically
    0.15-0.8.
    """
    import numpy as np
    import pandas as pd

    n = 6
    v = np.array([0.9, 0.6, 0.6, 0.6, 0.6, 0.9], dtype=float)

    # Reference primacy kernel (eta = 5): convex, front-loaded.
    pos = np.arange(n, dtype=float)
    prim = ((n - pos) / (n - 1.0)) ** 5.0
    w_prim = prim / prim.sum()

    # Reference inverted-validity kernel (gamma = 3, c = 0.1).
    # Only the SIGN of its dot product with s is used (orientation).
    err = 1.0 - v
    cent = np.maximum(err - 0.1 * float(err.min()), 0.0)
    u = cent ** 3.0
    u = u / u.sum()

    a = np.array([[float(t) for t in r] for r in data['option_a_ratings']], dtype=float)
    b = np.array([[float(t) for t in r] for r in data['option_b_ratings']], dtype=float)
    y = 1.0 - 2.0 * np.asarray(data['response'], dtype=float).ravel()  # +1 = A

    m = a.shape[0]
    if m == 0:
        return 0.0

    s = np.sign(a - b)
    aWin = a > b
    bWin = b > a

    C = np.zeros(m)
    Z = np.zeros(m)
    keep = np.zeros(m, dtype=bool)

    for i in range(m):
        ka = tuple(sorted(v[aWin[i]].tolist()))
        kb = tuple(sorted(v[bWin[i]].tolist()))
        nA = int(aWin[i].sum())
        nB = int(bWin[i].sum())
        if ka == kb and nA > 0:
            # Matched-validity cell: distrust+flat kernels cancel exactly.
            keep[i] = True
            C[i] = float(np.dot(w_prim, s[i]))
        elif nA == 2 and nB == 2 and ka != kb:
            # X2-family cell: orientation dummy absorbs IVC's constant
            # allegiance effect; position spread loads on C.
            keep[i] = True
            C[i] = float(np.dot(w_prim, s[i]))
            Z[i] = float(np.sign(np.dot(u, s[i])))

    if int(keep.sum()) < 12:
        return 0.0

    Ck = C[keep]
    Zk = Z[keep]
    yk = y[keep]

    X = np.column_stack([np.ones_like(Ck), Ck, Zk])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, yk, rcond=None)
        return float(coef[1])
    except Exception:
        return 0.0
```

**Observed (real) value:** 0.5396 (var=0.0423)
**Previous candidate values (this loop):**
  - iter 1: 0.3941 (var=0.0372) (Δ vs real -0.1455)
  - iter 2 (most recent): 0.3892 (var=0.0379) (Δ vs real -0.1504)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4027 (var=0.0731)
- pi_5_1: 0.0336 (var=0.0381)
- pi_1: 0.8316 (var=0.0733)
- pi_2: -0.0087 (var=0.0294)
- pi_3: -0.0421 (var=0.0393)
- pi_4: -0.6945 (var=0.0377)
- pi_5: 0.0254 (var=0.0512)
- pi_7: 0.4678 (var=0.0344)
- pi_8: 0.5770 (var=0.0378)
- pi_9: 0.5986 (var=0.0372)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Trial specification keyed by (A ratings, B ratings) -> (family, aligned response).
    # Families (validities = [0.95, 0.5, 0.95, 0.5, 0.95, 0.5]; experts 1,3,5 are 95%, experts 2,4,6 are 50%):
    #   "H" : high-validity (95%) duels (trials 1-4); aligned = product winning the LATER expert (anti-primacy)
    #   "L" : low-validity (50%) duels (trials 7-10); aligned = product winning the EARLIER expert (primacy)
    #   "M" : distrust-neutralized position cells (trials 13-18); aligned = product winning expert 2
    #   "C" : count/validity-balanced coalition cells (trials 21-24); aligned = product winning experts {2,3}
    spec = {
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0)): ("H", 1),
        ((0, 0, 0, 0, 1, 0), (1, 0, 0, 0, 0, 0)): ("H", 0),
        ((1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)): ("H", 1),
        ((0, 0, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0)): ("H", 0),
        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): ("L", 0),
        ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): ("L", 1),
        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0)): ("L", 0),
        ((0, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)): ("L", 1),
        ((0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 1)): ("M", 0),
        ((1, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): ("M", 1),
        ((0, 1, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0)): ("M", 0),
        ((1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)): ("M", 1),
        ((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 1)): ("M", 0),
        ((1, 0, 0, 0, 0, 1), (0, 1, 1, 0, 0, 0)): ("M", 1),
        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0)): ("C", 0),
        ((0, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 0)): ("C", 1),
        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 0, 1, 1)): ("C", 0),
        ((0, 0, 0, 0, 1, 1), (0, 1, 1, 0, 0, 0)): ("C", 1),
    }

    hits = {"H": 0, "L": 0, "M": 0, "C": 0}
    tots = {"H": 0, "L": 0, "M": 0, "C": 0}

    a_cells = data["option_a_ratings"].tolist()
    b_cells = data["option_b_ratings"].tolist()
    r_vals = data["response"].tolist()

    for a_cell, b_cell, r in zip(a_cells, b_cells, r_vals):
        try:
            ka = tuple([int(x) for x in a_cell])
            kb = tuple([int(x) for x in b_cell])
            ri = int(r)
        except Exception:
            continue
        entry = spec.get((ka, kb))
        if entry is None:
            continue
        fam, al = entry
        tots[fam] += 1
        if ri == al:
            hits[fam] += 1

    def prop(fam):
        if tots[fam] == 0:
            return 0.5
        return hits[fam] / float(tots[fam])

    p_l = prop("L")  # P(earlier-expert product | low-validity duel)
    p_h = prop("H")  # P(later-expert product  | high-validity duel)
    p_m = prop("M")  # P(expert-2 product | distrust-neutralized cells)
    p_c = prop("C")  # P({2,3}-coalition product | balanced coalition cells)

    # (1) Validity-scaled position crossover: C = P(early|low duel) - P(early|high duel)
    #     = p_l + p_h - 1. Strictly positive under PGDT, strictly negative under PW-IVC
    #     at every corner of both parameter boxes (position effects are validity-
    #     AMPLIFIED under PGDT's multiplicative kernel, validity-INVARIANT under
    #     PW-IVC's additive kernel, where the position-1 expert always dominates).
    # (2) M-family: distrust terms cancel exactly in BOTH theories, leaving opposite-
    #     signed position architecture (PGDT: d>0 at every corner; PW-IVC: d<0 at every
    #     corner). Double-weighted because this is the only sign-DISJOINT family.
    # (3) Coalition cells: counting and distrust both cancel; only position varies.
    #     PGDT separates the coalition strongly (d in [0.08, 0.39]); PW-IVC caps at
    #     d <= 0.11. Double-weighted to use the 12 trials efficiently.
    crossover = p_l + p_h - 1.0
    m_term = 2.0 * (p_m - 0.5)
    c_term = 2.0 * (p_c - 0.5)

    return float(crossover + m_term + c_term)
```

**Observed (real) value:** 1.8356 (var=0.2112)
**Previous candidate values (this loop):**
  - iter 1: 1.4278 (var=0.0964) (Δ vs real -0.4078)
  - iter 2 (most recent): 1.3539 (var=0.1237) (Δ vs real -0.4817)
**Other theories' values on this metric (for reference):**
- pi_7: 1.1900 (var=0.2144)
- pi_6: -0.0794 (var=0.2347)
- pi_1: 0.0389 (var=0.0856)
- pi_2: -0.4917 (var=0.1577)
- pi_3: -0.2350 (var=0.2168)
- pi_4: -0.8106 (var=0.2217)
- pi_5: 0.0650 (var=0.2219)
- pi_5_1: 0.0894 (var=0.1646)
- pi_8: 1.7233 (var=0.1382)
- pi_9: 1.6694 (var=0.1653)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Critical "primacy-rescue" trial family: exactly one option carries the
    # position-1 trusted (0.95) expert, the opposing option carries only
    # early low-validity (0.50) experts. Six f1-side patterns (option that
    # holds the position-1 trusted expert) x two counterpart patterns.
    P = {
        (1, 0, 0, 0, 0, 1),  # {95%@1, 50%@6}          (L1 rung)
        (1, 0, 0, 0, 1, 1),  # {95%@1, 50%@5, 50%@6}   (L2 rung)
        (1, 0, 1, 0, 1, 1),  # {95%@1, 50%@3,5,6}     (L3 rung)
        (1, 0, 1, 0, 0, 0),  # {95%@1, 50%@3}          (L1e early helper)
        (1, 0, 1, 1, 0, 0),  # {95%@1, 50%@3, 95%@4}   (TB+ cell)
        (1, 0, 0, 0, 1, 0),  # {95%@1, 50%@5}          (FL cell)
    }
    Q = {
        (0, 1, 0, 0, 0, 0),  # {50%@2}
        (0, 1, 0, 1, 0, 0),  # {50%@2, 95%@4}
    }

    def to_key(x):
        return tuple(int(v) for v in x)

    hits = []
    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]
    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        a = to_key(a_raw)
        b = to_key(b_raw)
        r = int(r_raw)
        if a in P and b in Q:
            # trusted-early product is option A
            hits.append(1.0 if r == 0 else 0.0)
        elif b in P and a in Q:
            # trusted-early product is option B
            hits.append(1.0 if r == 1 else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1322 (var=0.0089)
**Previous candidate values (this loop):**
  - iter 1: 0.3306 (var=0.0122) (Δ vs real +0.1983)
  - iter 2 (most recent): 0.3461 (var=0.0061) (Δ vs real +0.2139)
**Other theories' values on this metric (for reference):**
- pi_6: 0.6911 (var=0.0080)
- pi_7: 0.4311 (var=0.0074)
- pi_1: 0.8344 (var=0.0128)
- pi_2: 0.7878 (var=0.0063)
- pi_3: 0.6217 (var=0.0047)
- pi_4: 0.6733 (var=0.0080)
- pi_5: 0.5822 (var=0.0076)
- pi_5_1: 0.5911 (var=0.0082)
- pi_8: 0.3094 (var=0.0064)
- pi_9: 0.2639 (var=0.0052)

### Experiment 15
**Design**
  A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Gate-vs-ramp discrimination score (theory-informed linear contrast).

    Each of the 32 design cells gets a fixed coefficient c equal (in sign and
    rough magnitude) to the predicted difference P(A | PGDT) - P(A | PRDT)
    for that cell.  The score is the mean of c * (+1 if chose A, -1 if chose B)
    over the session, x100.  Under PGDT (position-independent trusted bracket
    ordered by the primacy gate) the score is strongly positive; under PRDT
    (t0 = 0 trust ramp that kills early trusted experts) it is much smaller.
    Mirrored cells carry opposite coefficients, so side/position bias cancels.
    """
    if data is None or len(data) == 0:
        return 0.0
    for col in ("option_a_ratings", "option_b_ratings", "response"):
        if col not in data.columns:
            return 0.0

    # (A_ratings, B_ratings) -> contrast coefficient
    _TRIALS = [
        # --- trusted-position axis (sign-robust at every parameter corner) ---
        ((1, 1, 1, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1, 1),  0.050),  # D17
        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.050),
        ((1, 1, 1, 1, 1, 0, 1), (0, 1, 1, 1, 1, 1, 1),  0.045),  # D16
        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 1), -0.045),
        ((1, 1, 1, 1, 1, 1, 0), (1, 0, 1, 1, 1, 1, 1),  0.022),  # D27-type
        ((1, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.022),
        ((1, 1, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1, 1),  0.020),  # D26-type
        ((1, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 1), -0.020),
        ((1, 0, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1),  0.012),  # pos1 vs pos0
        ((0, 1, 1, 1, 1, 1, 1), (1, 0, 1, 1, 1, 1, 1), -0.012),
        ((1, 1, 1, 1, 1, 0, 0), (0, 0, 1, 1, 1, 1, 1),  0.060),  # BLK
        ((0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.060),
        ((1, 1, 1, 1, 1, 0, 0), (0, 1, 1, 1, 1, 1, 1),  0.040),  # BLKa
        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.040),
        ((1, 1, 1, 1, 1, 1, 0), (0, 0, 1, 1, 1, 1, 1),  0.070),  # T21
        ((0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.070),
        # --- kernel-steepness / anti-validity axis ---
        ((1, 1, 1, 0, 1, 1, 1), (1, 1, 0, 1, 0, 1, 1), -0.080),  # ML1
        ((1, 1, 0, 1, 0, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.080),
        ((1, 1, 1, 0, 1, 1, 1), (1, 1, 0, 1, 1, 1, 1), -0.050),  # ML2
        ((1, 1, 0, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.050),
        ((0, 0, 0, 1, 0, 0, 0), (1, 1, 1, 0, 1, 1, 1),  0.050),  # ANCH
        ((1, 1, 1, 0, 1, 1, 1), (0, 0, 0, 1, 0, 0, 0), -0.050),
        ((1, 1, 1, 0, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1), -0.015),  # TL
        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.015),
        ((1, 1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1, 1),  0.012),  # MM
        ((1, 1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1, 1), -0.012),
        ((1, 1, 1, 1, 0, 1, 0), (0, 1, 0, 1, 1, 1, 1),  0.045),  # mixed
        ((0, 1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1, 0), -0.045),
        ((1, 1, 1, 1, 1, 0, 0), (0, 1, 1, 0, 1, 1, 1),  0.005),  # MX
        ((0, 1, 1, 0, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.005),
        # TOT anchors: no theory contrast (lapse-only difference) -> weight 0
        ((1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0),  0.000),
        ((0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1),  0.000),
    ]
    W = {(a, b): c for a, b, c in _TRIALS}

    def _key(cell):
        # robustly convert a list/tuple/ndarray/string of binary ratings
        # into a hashable tuple of ints
        if isinstance(cell, str):
            s = cell.replace("[", " ").replace("]", " ").replace(",", " ")
            try:
                return tuple(int(round(float(t))) for t in s.split())
            except ValueError:
                return None
        try:
            arr = np.asarray(cell, dtype=float).ravel()
            return tuple(int(round(float(v))) for v in arr)
        except Exception:
            return None

    total = 0.0
    n = 0
    for a, b, r in zip(data["option_a_ratings"].tolist(),
                       data["option_b_ratings"].tolist(),
                       data["response"].tolist()):
        ka = _key(a)
        kb = _key(b)
        if ka is None or kb is None:
            continue
        c = W.get((ka, kb), 0.0)
        try:
            rv = int(round(float(r)))
        except (TypeError, ValueError):
            continue
        y = 1.0 if rv == 0 else -1.0  # +1 = chose A, -1 = chose B
        total += c * y
        n += 1

    if n == 0:
        return 0.0
    return 100.0 * total / n
```

**Observed (real) value:** 1.0358 (var=0.1926)
**Previous candidate values (this loop):**
  - iter 1: 0.9356 (var=0.1521) (Δ vs real -0.1003)
  - iter 2 (most recent): 0.7490 (var=0.1363) (Δ vs real -0.2868)
**Other theories' values on this metric (for reference):**
- pi_7: 1.2013 (var=0.2191)
- pi_8: 0.7713 (var=0.1699)
- pi_1: 0.7452 (var=0.1677)
- pi_2: -0.4130 (var=0.1912)
- pi_3: -0.0560 (var=0.2308)
- pi_4: -1.3777 (var=0.2014)
- pi_5: 0.7034 (var=0.1475)
- pi_5_1: 0.7465 (var=0.2677)
- pi_6: 1.6376 (var=0.1716)
- pi_9: 1.0273 (var=0.1192)

### Experiment 16
**Design**
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Two-stage, layout-adaptive theory discriminant.
    # Stage 1 infers which validity layout generated the data (the design
    # header lists [0.99, 0.99, 0.99, 0.75, 0.75, 0.75, 0.75, 0.75] while
    # the rationale/trial list is built around a 0.55 expert at position
    # 5). The inference uses only the clean two-cue duels involving
    # position 5: under the 0.55 layout BOTH theories prefer the
    # position-5-endorsed product with P ~ 0.7-0.9; under the all-0.75
    # layout that probability is ~ 0.5. This is a theory-agnostic,
    # mirror-balanced statistic, so it is safe to branch on.
    # Stage 2 computes a gap-weighted choice discriminant: for every unique
    # (A, B) rating pattern it evaluates the box-averaged predicted P(A)
    # of PRDT (distrust inversion with gamma in [1.6, 2.2], floored primacy
    # gate, position-growing trust ramp with t0 pinned at 0) and of PGDT
    # (gamma in [2.8, 3.5], position-independent flat share (1-tau)/8),
    # using each theory's own mechanics verbatim, and weights the observed
    # choose-A indicator by the signed predicted gap. Mirrored cells
    # automatically reinforce. The score is linear in the trial outcomes,
    # so the pooled value equals the mean of the per-subject values and
    # the metric works on single-subject slices.

    def _vec(x):
        try:
            arr = np.asarray(list(x), dtype=float)
            if arr.size == 0:
                raise ValueError("empty cell")
            return arr
        except Exception:
            s = str(x).strip()
            for ch in "[](),":
                s = s.replace(ch, " ")
            return np.asarray([float(t) for t in s.split() if t != ""], dtype=float)

    if data is None or len(data) == 0:
        return 0.0

    df = data.reset_index(drop=True)
    a_list = [_vec(x) for x in df["option_a_ratings"].tolist()]
    b_list = [_vec(x) for x in df["option_b_ratings"].tolist()]
    resp = df["response"].to_numpy(dtype=float)
    n = int(a_list[0].size)
    chose_a = 1.0 - resp

    if n != 8:
        return float(np.mean(chose_a))

    # ---------- Stage 1: layout inference ----------
    y5 = []
    for av, bv, ca in zip(a_list, b_list, chose_a):
        s = np.sign(av - bv)
        nz = np.where(np.abs(s) > 0.5)[0]
        if len(nz) == 2 and 4 in nz:
            endorsed_a = bool(s[4] > 0)
            picked_a = bool(ca > 0.5)
            y5.append(1.0 if (endorsed_a == picked_a) else 0.0)
    p5 = float(np.mean(y5)) if len(y5) > 0 else 0.5

    if p5 > 0.60:
        v = np.array([0.99, 0.99, 0.99, 0.75, 0.55, 0.75, 0.75, 0.75], dtype=float)
    else:
        v = np.array([0.99, 0.99, 0.99, 0.75, 0.75, 0.75, 0.75, 0.75], dtype=float)

    # ---------- Stage 2: reference theory predictions ----------
    def _prdt_w(tau, eta, delta, gamma, c, t1, kappa):
        err = 1.0 - v
        centered = np.maximum(err - c * float(err.min()), 0.0)
        inv = np.power(centered, gamma)
        Z = float(inv.sum())
        if Z > 1e-12:
            w_inv = inv / Z
        else:
            w_inv = np.full(8, 0.125)
        pos = np.arange(8, dtype=float)
        t = t1 * np.power(pos / 7.0, kappa)  # t0 pinned at exactly 0
        g = delta + (1.0 - delta) * np.power((8.0 - pos) / 7.0, eta)
        bracket = t * (1.0 - tau) / 8.0 + tau * w_inv
        raw = g * bracket
        Zt = float(raw.sum())
        if Zt > 1e-12:
            return raw / Zt
        return np.full(8, 0.125)

    def _pgdt_w(tau, eta, delta, gamma, c):
        err = 1.0 - v
        centered = np.maximum(err - c * float(err.min()), 0.0)
        inv = np.power(centered, gamma)
        Z = float(inv.sum())
        if Z > 1e-12:
            w_inv = inv / Z
        else:
            w_inv = np.full(8, 0.125)
        pos = np.arange(8, dtype=float)
        g = delta + (1.0 - delta) * np.power((8.0 - pos) / 7.0, eta)
        bracket = (1.0 - tau) / 8.0 + tau * w_inv
        raw = g * bracket
        Zt = float(raw.sum())
        if Zt > 1e-12:
            return raw / Zt
        return np.full(8, 0.125)

    def _pa(w, s, beta, eps):
        d = float(np.dot(w, s))
        p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
        return (1.0 - eps) * p + 0.5 * eps

    prdt_base = dict(tau=0.915, eta=5.5, delta=0.275, gamma=1.9, c=0.15,
                     t1=0.925, kappa=2.25, beta=4.75, eps=0.11)
    prdt_draws = [dict(prdt_base)]
    for gm in (1.6, 2.2):
        for ta in (0.90, 0.93):
            for be in (4.0, 5.5):
                for t1v in (0.85, 1.0):
                    dr = dict(prdt_base)
                    dr["gamma"] = gm
                    dr["tau"] = ta
                    dr["beta"] = be
                    dr["t1"] = t1v
                    prdt_draws.append(dr)
    prdt_edges = {"eta": (5.0, 6.0), "delta": (0.25, 0.30),
                  "kappa": (1.5, 3.0), "c": (0.0, 0.3), "eps": (0.09, 0.13)}
    for key in prdt_edges:
        for val in prdt_edges[key]:
            dr = dict(prdt_base)
            dr[key] = val
            prdt_draws.append(dr)

    pgdt_base = dict(tau=0.87, eta=5.0, delta=0.325, gamma=3.15, c=0.175,
                     beta=4.25, eps=0.105)
    pgdt_draws = [dict(pgdt_base)]
    for gm in (2.8, 3.5):
        for ta in (0.82, 0.92):
            for dl in (0.20, 0.45):
                for be in (2.5, 6.0):
                    dr = dict(pgdt_base)
                    dr["gamma"] = gm
                    dr["tau"] = ta
                    dr["delta"] = dl
                    dr["beta"] = be
                    pgdt_draws.append(dr)
    pgdt_edges = {"eta": (4.0, 6.0), "c": (0.10, 0.25), "eps": (0.06, 0.15)}
    for key in pgdt_edges:
        for val in pgdt_edges[key]:
            dr = dict(pgdt_base)
            dr[key] = val
            pgdt_draws.append(dr)

    # ---------- Stage 3: gap-weighted choice discriminant ----------
    info = {}
    for av, bv, ca in zip(a_list, b_list, chose_a):
        key = (tuple(int(round(x)) for x in av), tuple(int(round(x)) for x in bv))
        if key not in info:
            s = np.sign(av - bv)
            pp = []
            for dr in prdt_draws:
                w = _prdt_w(dr["tau"], dr["eta"], dr["delta"], dr["gamma"],
                            dr["c"], dr["t1"], dr["kappa"])
                pp.append(_pa(w, s, dr["beta"], dr["eps"]))
            pg = []
            for dr in pgdt_draws:
                w = _pgdt_w(dr["tau"], dr["eta"], dr["delta"], dr["gamma"], dr["c"])
                pg.append(_pa(w, s, dr["beta"], dr["eps"]))
            info[key] = [0, 0.0, float(np.mean(pp)) - float(np.mean(pg))]
        info[key][0] += 1
        info[key][1] += float(ca)

    num = 0.0
    den = 0.0
    for key in info:
        cnt, na, gap = info[key]
        if abs(gap) < 1e-9:
            continue
        y = na / float(cnt)
        num += gap * (2.0 * y - 1.0) * cnt
        den += abs(gap) * cnt
    if den <= 1e-12:
        return 0.0
    return float(num / den)
```

**Observed (real) value:** 0.3217 (var=0.0143)
**Previous candidate values (this loop):**
  - iter 1: 0.3575 (var=0.0141) (Δ vs real +0.0358)
  - iter 2 (most recent): 0.3156 (var=0.0125) (Δ vs real -0.0061)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4993 (var=0.0158)
- pi_7: 0.2921 (var=0.0253)
- pi_1: -0.3540 (var=0.0203)
- pi_2: 0.0555 (var=0.0223)
- pi_3: 0.0175 (var=0.0135)
- pi_4: 0.2437 (var=0.0175)
- pi_5: 0.2891 (var=0.0130)
- pi_5_1: 0.3482 (var=0.0169)
- pi_6: 0.1779 (var=0.0330)
- pi_9: 0.3709 (var=0.0137)

### Experiment 17
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

_CACHE = {}

_TRIALS = [
    ((1, 1, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1)),
    ((0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0)),
    ((1, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1)),
    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1)),
    ((1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1)),
    ((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 1, 1)),
    ((1, 1, 1, 1, 0, 0), (0, 0, 1, 1, 1, 1)),
    ((0, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 0)),
    ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
    ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),
    ((1, 0, 1, 1, 1, 1), (0, 1, 1, 1, 0, 1)),
    ((0, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1)),
    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 0)),
    ((1, 1, 1, 1, 0, 0), (1, 0, 1, 1, 1, 1)),
    ((1, 1, 0, 0, 1, 1), (0, 0, 1, 1, 0, 0)),
    ((0, 0, 1, 1, 0, 0), (1, 1, 0, 0, 1, 1)),
    ((1, 1, 0, 1, 1, 1), (0, 1, 1, 1, 1, 1)),
    ((0, 1, 1, 1, 1, 1), (1, 1, 0, 1, 1, 1)),
    ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)),
    ((0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1)),
    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0)),
    ((1, 1, 1, 1, 1, 0), (1, 0, 1, 1, 1, 1)),
    ((0, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 0)),
    ((1, 0, 1, 1, 1, 0), (0, 1, 1, 1, 0, 1)),
]

_V = np.array([0.95, 0.85, 0.60, 0.60, 0.85, 0.95])
_A = np.array([t[0] for t in _TRIALS], dtype=float)
_B = np.array([t[1] for t in _TRIALS], dtype=float)
_SIGN = np.sign(_A - _B)

_PG_LO = np.array([0.82, 4.0, 0.20, 2.8, 0.10, 2.5, 0.06])
_PG_HI = np.array([0.92, 6.0, 0.45, 3.5, 0.25, 6.0, 0.15])
_PB_LO = np.array([0.91, 6.5, 0.15, 2.6, 0.05, 0.25, 0.95, 0.80, 3.6, 0.11])
_PB_HI = np.array([0.93, 8.0, 0.25, 2.9, 0.25, 0.35, 1.00, 1.50, 4.4, 0.14])


def _pgdt_pvec(par):
    tau, eta, delta, gamma, c, beta, eps = par
    tau = float(tau); eta = float(eta); delta = float(delta)
    gamma = float(gamma); c = float(c); beta = float(beta); eps = float(eps)
    n = 6
    err = 1.0 - _V
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Zv = float(inv.sum())
    if Zv > 1e-12:
        w_inv = inv / Zv
    else:
        w_inv = np.full(n, 1.0 / n)
    pos = np.arange(n, dtype=float)
    g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    bracket = (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    w = raw / float(raw.sum())
    d = _SIGN.dot(w)
    p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    return (1.0 - eps) * p + eps * 0.5


def _pbdt_pvec(par):
    tau, eta, delta, gamma, c, t0, t1, kappa, beta, eps = par
    tau = float(tau); eta = float(eta); delta = float(delta)
    gamma = float(gamma); c = float(c); t0 = float(t0); t1 = float(t1)
    kappa = float(kappa); beta = float(beta); eps = float(eps)
    n = 6
    err = 1.0 - _V
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Zv = float(inv.sum())
    if Zv > 1e-12:
        w_inv = inv / Zv
    else:
        w_inv = np.full(n, 1.0 / n)
    pos = np.arange(n, dtype=float)
    t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
    g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    w = raw / float(raw.sum())
    d = _SIGN.dot(w)
    p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    return (1.0 - eps) * p + eps * 0.5


def _make_grid(lo, hi, n_random=192, seed=1234567):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    k = len(lo)
    m = 2 ** k
    idx = np.arange(m)
    bits = ((idx[:, None] >> np.arange(k)[None, :]) & 1).astype(float)
    corners = lo[None, :] * (1.0 - bits) + hi[None, :] * bits
    center = ((lo + hi) / 2.0)[None, :]
    rng = np.random.default_rng(seed)
    rnd = rng.uniform(lo, hi, size=(n_random, k))
    return np.vstack([corners, center, rnd])


def _weights():
    if 'wmap' in _CACHE:
        return _CACHE['wmap']
    pg_grid = _make_grid(_PG_LO, _PG_HI)
    pb_grid = _make_grid(_PB_LO, _PB_HI)
    nt = len(_TRIALS)
    p_pg = np.zeros(nt)
    for par in pg_grid:
        p_pg += _pgdt_pvec(par)
    p_pg /= float(len(pg_grid))
    p_pb = np.zeros(nt)
    for par in pb_grid:
        p_pb += _pbdt_pvec(par)
    p_pb /= float(len(pb_grid))
    p_pg = np.clip(p_pg, 0.02, 0.98)
    p_pb = np.clip(p_pb, 0.02, 0.98)
    lA = np.log(p_pg / p_pb)
    lB = np.log((1.0 - p_pg) / (1.0 - p_pb))
    wmap = {}
    for i in range(nt):
        wmap[_TRIALS[i]] = (float(lA[i]), float(lB[i]))
    _CACHE['wmap'] = wmap
    return wmap


def _to_tuple(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(int(round(float(v))) for v in x)
    s = str(x).replace("'", "").replace('"', '')
    s = s.strip().strip('[](){} ').replace(',', ' ')
    return tuple(int(round(float(p))) for p in s.split())


def metric(data: pd.DataFrame) -> float:
    wmap = _weights()
    if len(data) == 0:
        return 0.0
    ra = data['option_a_ratings'].apply(_to_tuple)
    rb = data['option_b_ratings'].apply(_to_tuple)
    resp = pd.to_numeric(data['response']).to_numpy()
    total = 0.0
    for (a, b), r in zip(zip(ra, rb), resp):
        pr = wmap.get((a, b))
        if pr is None:
            continue
        total += pr[0] if int(r) == 0 else pr[1]
    return float(total)
```

**Observed (real) value:** -9.1163 (var=0.3521)
**Previous candidate values (this loop):**
  - iter 1: -8.6363 (var=0.3496) (Δ vs real +0.4800)
  - iter 2 (most recent): -8.5160 (var=0.5011) (Δ vs real +0.6003)
**Other theories' values on this metric (for reference):**
- pi_7: 12.6306 (var=0.4648)
- pi_9: -8.3125 (var=0.4421)
- pi_1: 141.4785 (var=0.4190)
- pi_2: 89.6874 (var=0.3151)
- pi_3: 70.0949 (var=0.4149)
- pi_4: 73.0610 (var=0.3610)
- pi_5: 31.8064 (var=0.3769)
- pi_5_1: 27.1285 (var=0.3245)
- pi_6: 41.4469 (var=0.3949)
- pi_8: -23.1646 (var=0.3828)

### Experiment 18
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Early-trusted-channel contrast score (PBDT vs PGDT diagnostic).

    The design's decisive lever is the TRUSTED (flat-share) channel: PGDT's
    bracket (1-tau)/n + tau*inv_j is position-independent, so an early
    high-validity expert carries the full amplified position-1 gate weight;
    PBDT multiplies the flat share by a trust ramp rising from a low floor
    t0, crushing the early trusted weight by ~1/t0 while leaving late trusted
    experts nearly intact.  Behaviorally this shows up as a specific pattern
    of mirror-corrected choice probabilities across the R1/R1p/R2/R2p/S1/S2/
    S6/T16/T26/T12 cells (and weakly on the anchors).

    This metric computes, for every one of the 32 fixed trial pairs, the
    predicted P(choose A) under PGDT and under PBDT (both engines
    implemented below, each averaged over five parameter settings spanning
    its box), takes the per-trial contrast c_k = p_PGDT - p_PBDT, and scores
    the observed choices against that contrast:

        S = sum_rows c_k * y,   y = +1 if A chosen, -1 if B chosen
        z = S / sqrt(sum_rows c_k^2)

    z is the standardized behavioral expression of the early-trusted
    channel (positive = choices follow PGDT's strong early-trusted
    weighting; near zero / negative = choices follow PBDT's crushed
    early-trusted channel).  Because a single subject contributes only 6
    trials per cell and the per-trial contrast is small, the single-subject
    z is dominated by binomial noise; we therefore report an empirical-Bayes
    shrunken version

        metric = (n / (n + K)) * z,   K = 2000 pseudo-observations,

    which pulls sparse single-subject estimates strongly toward 0 (chance)
    while leaving the full pooled dataset (n ~ 2400) essentially
    unregularized.  The statistic is well defined on any slice of the data.
    """
    V = (0.95, 0.95, 0.75, 0.75, 0.60, 0.95)
    TRIALS = [
        ((0, 0, 1, 0, 0, 0), (1, 0, 0, 1, 0, 0)),
        ((1, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0)),
        ((0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 1)),
        ((0, 0, 0, 1, 0, 1), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (1, 0, 0, 1, 0, 1)),
        ((1, 0, 0, 1, 0, 1), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1)),
        ((0, 1, 0, 1, 0, 1), (0, 0, 1, 0, 0, 0)),
        ((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (1, 0, 1, 0, 0, 0)),
        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (0, 1, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 1), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 1)),
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)),
        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)),
        ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)),
        ((0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 0, 0, 1, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0)),
        ((1, 1, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0)),
        ((0, 0, 0, 0, 1, 0), (1, 1, 0, 0, 0, 1)),
        ((1, 1, 0, 0, 0, 1), (0, 0, 1, 1, 1, 0)),
        ((0, 0, 1, 1, 1, 0), (1, 1, 0, 0, 0, 1)),
        ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)),
        ((0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1)),
    ]

    v = np.asarray(V, dtype=float)
    err = 1.0 - v
    pos = np.arange(6, dtype=float)

    def pgdt_w(tau, eta, delta, gamma, c):
        # PGDT weights: floored convex primacy gate x position-independent
        # bracket (flat share + distrust kernel).
        cen = np.maximum(err - c * float(err.min()), 0.0)
        inv = cen ** gamma
        w_inv = inv / inv.sum()
        g = delta + (1.0 - delta) * ((6.0 - pos) / 5.0) ** eta
        br = (1.0 - tau) / 6.0 + tau * w_inv
        raw = g * br
        return raw / raw.sum()

    def pbdt_w(tau, eta, delta, gamma, c, t0, t1, kappa):
        # PBDT weights: steep low-floored gate x (trust-ramped flat share +
        # distrust kernel).
        cen = np.maximum(err - c * float(err.min()), 0.0)
        inv = cen ** gamma
        w_inv = inv / inv.sum()
        t = t0 + (t1 - t0) * (pos / 5.0) ** kappa
        g = delta + (1.0 - delta) * ((6.0 - pos) / 5.0) ** eta
        br = t * (1.0 - tau) / 6.0 + tau * w_inv
        raw = g * br
        return raw / raw.sum()

    def p_a(w, A, B, beta, eps):
        d = float(np.dot(w, np.sign(np.asarray(A, dtype=float) - np.asarray(B, dtype=float))))
        p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
        return (1.0 - eps) * p + 0.5 * eps

    # Parameter settings spanning each theory's box (midpoint, key-dim
    # variants, weak corner, strong corner).
    PG = [
        (0.87, 5.00, 0.325, 3.15, 0.175, 4.25, 0.105),
        (0.82, 5.00, 0.325, 3.15, 0.175, 4.25, 0.105),
        (0.92, 5.00, 0.325, 3.15, 0.175, 4.25, 0.105),
        (0.92, 4.00, 0.200, 2.80, 0.100, 2.50, 0.150),
        (0.82, 6.00, 0.450, 3.50, 0.250, 6.00, 0.060),
    ]
    PB = [
        (0.92, 7.25, 0.20, 2.75, 0.15, 0.30, 0.975, 1.15, 4.00, 0.125),
        (0.91, 7.25, 0.20, 2.75, 0.15, 0.25, 0.975, 1.15, 4.00, 0.125),
        (0.93, 7.25, 0.20, 2.75, 0.15, 0.35, 0.975, 1.15, 4.00, 0.125),
        (0.93, 6.50, 0.25, 2.60, 0.05, 0.35, 0.950, 1.50, 3.60, 0.140),
        (0.91, 8.00, 0.15, 2.90, 0.25, 0.25, 1.000, 0.80, 4.40, 0.110),
    ]

    pg_ws = [pgdt_w(s[0], s[1], s[2], s[3], s[4]) for s in PG]
    pb_ws = [pbdt_w(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]) for s in PB]

    contrast = {}
    for A, B in TRIALS:
        pG = float(np.mean([p_a(w, A, B, s[5], s[6]) for (w, s) in zip(pg_ws, PG)]))
        pB = float(np.mean([p_a(w, A, B, s[8], s[9]) for (w, s) in zip(pb_ws, PB)]))
        contrast[(A, B)] = pG - pB

    def as_key(x):
        return tuple(int(round(float(t))) for t in x)

    n = int(len(data))
    if n == 0:
        return 0.0

    S = 0.0
    Q = 0.0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'],
                        data['response']):
        key = (as_key(a), as_key(b))
        c = contrast.get(key)
        if c is None:
            continue
        if abs(c) < 1e-12:
            continue
        y = 1.0 if int(round(float(r))) == 0 else -1.0  # +1 = chose A
        S += c * y
        Q += c * c

    if Q <= 1e-24:
        return 0.0

    z = S / np.sqrt(Q)
    # Empirical-Bayes shrinkage toward 0 (chance-level expression of the
    # contrast) with a fixed pseudo-observation count: single-subject
    # slices (n=96) are shrunk hard; the full pooled dataset (n~2400) is
    # dominated by the data.
    lam = float(n) / (float(n) + 2000.0)
    return float(lam * z)
```

**Observed (real) value:** -0.9319 (var=0.0016)
**Previous candidate values (this loop):**
  - iter 1: -0.9673 (var=0.0016) (Δ vs real -0.0353)
  - iter 2 (most recent): -3.7961 (var=0.0023) (Δ vs real -2.8642)
**Other theories' values on this metric (for reference):**
- pi_9: -1.5890 (var=0.0021)
- pi_7: 1.8638 (var=0.0018)
- pi_1: 18.9139 (var=0.0033)
- pi_2: 18.1537 (var=0.0025)
- pi_3: 7.5747 (var=0.0025)
- pi_4: 9.8800 (var=0.0020)
- pi_5: 0.9857 (var=0.0019)
- pi_5_1: 0.3211 (var=0.0019)
- pi_6: 5.4088 (var=0.0025)
- pi_8: -4.8657 (var=0.0019)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
RECIPROCAL-TRUST DISTRUST TALLYING (RTDT). Subjects scan the expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention while late experts retain only a floor residue. Stated validities are distrust-inverted (treated as error rates) via a STEEP kernel inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z with gamma in [3.0, 4.2] and a high distrust share tau in [0.92, 0.96], so the anti-validity gradient in validity-position-confounded layouts is carried almost entirely by the distrust kernel. The novel component is the RECIPROCAL trust ramp: credibility of high-validity experts is not linearly restored with position but GEOMETRICALLY amplified, t_j = t0 * (t1/t0)^((j/(n-1))^kappa), with a near-zero early-trust floor t0 in [0.00, 0.12], a super-unity late trust residue t1 in [1.0, 1.6], and a sub-unity exponent kappa in [0.6, 1.2] so trust recovers early in the tail. Early trusted claims are therefore almost completely discounted (reciprocal distrust of front-loaded authority), while LATE high-validity experts are amplified ABOVE the flat share — the key lever for final-expert following without raising the early floor. Weights combine multiplicatively, w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, both attenuated (beta in [2.8, 4.0], epsilon in [0.12, 0.18]) to avoid over-commitment on allegiance metrics.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # RTDT: Reciprocal-Trust Distrust Tallying.
    # Replaces PGDT (pi_7) per the arbiter's directive. Three quantified
    # departures from the accepted PBDT base, each targeted at a
    # documented residual:
    #   (A) STEEPER DISTRUST KERNEL: gamma [2.6,2.9] -> [3.0, 4.2],
    #       tau [0.91,0.93] -> [0.92, 0.96]. Repairs the under-predicted
    #       anti-validity slope (Exp5/Exp3 cluster) and lets kernel
    #       dominance weaken count/tally allegiance (Exp4, Exp2).
    #   (B) RECIPROCAL TRUST RAMP: replaces the linear ramp with a
    #       geometric one, t_j = t0 * (t1/t0)^((j/(n-1))^kappa), with a
    #       NEAR-ZERO early floor t0 in [0.00, 0.12] and a SUPER-UNITY
    #       late residue t1 in [1.0, 1.6], kappa in [0.6, 1.2]. Fixes
    #       Exp16's over-predicted primacy-rescue (early trusted experts
    #       now carry almost no trusted-channel weight) and Exp8/Exp10's
    #       under-followed late 0.90-validity expert (late trusted
    #       experts are AMPLIFIED above the flat share).
    #   (C) ATTENUATED CHOICE RULE: beta [2.8, 4.0], epsilon [0.12, 0.18],
    #       since the distrust family systematically over-commits on
    #       allegiance metrics (Exps 4, 6, 9).
    # Gate retained at the moderate steep/low-floor corner the PBDT
    # trajectory verified: eta [4.5, 6.5], delta [0.15, 0.30].
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RTDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share (HIGH, steepened)
    eta = float(parameters["eta"])        # primacy-gate convexity (moderate-steep)
    delta = float(parameters["delta"])    # primacy-gate floor (low)
    gamma = float(parameters["gamma"])    # distrust-inversion steepness (STEEP)
    c = float(parameters["c"])            # partial centering of error rates
    t0 = float(parameters["t0"])          # early-trust floor (NEAR ZERO)
    t1 = float(parameters["t1"])          # late-trust residue (SUPER-UNITY)
    kappa = float(parameters["kappa"])    # reciprocal-ramp exponent (SUB-UNITY)
    beta = float(parameters["beta"])      # choice consistency (ATTENUATED)
    eps = float(parameters["epsilon"])    # uniform lapse (ATTENUATED)

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the (now steeper) gamma power, then
    # normalize. gamma in [3.0, 4.2] with tau in [0.92, 0.96] makes the
    # distrust kernel the dominant carrier of the anti-validity
    # gradient in validity-position-confounded layouts.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and position alone carries structure.
        w_inv = np.full(n, 1.0 / n)

    if n > 1:
        pos = np.arange(n, dtype=float)
        # (2) RECIPROCAL TRUST RAMP, computed in log space for numerical
        # stability: log t_j = log t0 + (j/(n-1))^kappa * (log t1 - log t0).
        # With t0 ~ 0 the early positions are pinned at near-zero trust
        # (reciprocal distrust of front-loaded authority), and because
        # kappa < 1 the normalized-position exponent rises quickly, trust
        # recovers early in the tail and the LAST positions exceed the
        # flat share (t1 > 1) -- late trusted experts are amplified.
        t0_eff = max(t0, 1e-4)  # guard the geometric form against t0 = 0
        t1_eff = max(t1, 1e-4)
        frac = np.power(pos / (n - 1.0), kappa)  # in [0, 1], concave-leaning
        t = np.exp(np.log(t0_eff) + frac * (np.log(t1_eff) - np.log(t0_eff)))
        # (3) STEEP FLOORED CONVEX PRIMACY GATE at the moderate corner:
        # eta in [4.5, 6.5], delta in [0.15, 0.30].
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket
    # cue-by-cue. The trusted channel is modulated by the reciprocal
    # ramp -- near-vanishing early, super-unity late -- while the steep
    # distrust kernel (tau >= 0.92) dominates the bracket elsewhere.
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (5) Deterministic attention: every discriminating cue is attended;
    # no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # (6) Numerically stable logistic choice rule -- beta in [2.8, 4.0]
    # (attenuated: the distrust family over-commits on allegiance
    # metrics), with a uniform lapse epsilon in [0.12, 0.18].
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- tau: [0.92, 0.96]
- eta: [4.5, 6.5]
- delta: [0.15, 0.30]
- gamma: [3.0, 4.2]
- c: [0.05, 0.25]
- t0: [0.0, 0.12]
- t1: [1.0, 1.6]
- kappa: [0.6, 1.2]
- beta: [2.8, 4.0]
- epsilon: [0.12, 0.18]
- validities: validities

`rationale`: RTDT implements the arbiter's prescription exactly, repairing the three documented mechanistic failures of the pi_7/pi_9 distrust family with three quantified, independently falsifiable levers. (1) STEEPER DISTRUST KERNEL (gamma [3.0, 4.2], tau [0.92, 0.96]): both current theories under-predict the anti-validity slope (Exp5 observed -0.576 vs PBDT -0.418, PGDT -0.438) and under-shoot the position-crossover composite (Exp15). A steeper inversion concentrates even more bracket mass on low-validity error-rate cues, deepening the anti-validity regression toward the observed value, while the stronger kernel dominance automatically weakens count/tally allegiance toward Exp4's observed 0.843 and Exp2's 0.158. (2) RECIPROCAL TRUST RAMP: PBDT's linear ramp with t0 in [0.25, 0.35] leaves the early trusted expert with 25-35% of the flat share, which is why PBDT over-predicts primacy-rescue (Exp16: 0.371 vs observed 0.322; PGDT 0.292 is closer but its flat trusted channel is position-independent, which Exp18 decisively rejects: PGDT +1.86 vs observed -0.93). The reciprocal form t_j = t0*(t1/t0)^((j/(n-1))^kappa) with t0 in [0.00, 0.12] crushes early trusted weight toward zero (fixing Exp16's rescue rate and further crushing Exp14's trusted-early side) while the super-unity t1 in [1.0, 1.6] with sub-unity kappa AMPLIFIES late trusted experts above the flat share -- the only structural channel that can lift Exp8's final 0.90-validity following (observed 0.396 vs PBDT 0.279) and Exp10's FEDI-adjacent signature without raising the early floor. (3) ATTENUATED CHOICE RULE (beta [2.8, 4.0], epsilon [0.12, 0.18]): the distrust family systematically over-commits on allegiance metrics (Exp4: 0.879 vs 0.843; Exp7: 0.857 vs 0.788; Exp9: 0.295 vs 0.341 needs less determinism in the recency direction); softer noise pulls all of these toward target at a small cost on slope metrics that the steeper kernel and amplified late trust compensate. The gate stays at the moderate steep/low-floor corner (eta [4.5, 6.5], delta [0.15, 0.30]) that the PBDT ACCEPTED trajectory verified, so Exps 1, 13, 15, 17 -- where PBDT already succeeds -- are structurally preserved. The reciprocal ramp is computed in log space with a 1e-4 guard on t0 so the [0.00, 0.12] range including exact zero remains numerically valid. Falsifiable predictions: Exp16 rescue rate near 0.10-0.20 (currently 0.371/0.292), Exp8 FEDI near 0.38-0.45 (currently ~0.28), Exp4 allegiance near 0.82-0.86, with Exp18's negative early-trusted contrast retained (the near-zero t0 pushes the PBDT-style crushed early-trusted channel even further toward the observed -0.93).

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate faithfully implements the arbiter's RTDT prescription (steep distrust kernel, reciprocal trust ramp with near-zero t0 and super-unity t1, attenuated choice rule, moderate primacy gate), and the gate ACCEPTED it at loss 0.0830. Per-experiment diagnosis: (i) Clear wins vs the pi_7 base — Exp3 anti-validity slope (-0.6225 vs pi_7's -0.4381, obs -0.5756), Exp6 (0.345 vs 0.303, obs 0.374), Exp7 (0.818 vs 0.844, obs 0.788), Exp9 (0.328 vs 0.299, obs 0.341), Exp13 (1.428 vs 1.190, obs 1.836), Exp14 (0.331 vs 0.431, obs 0.132), Exp15/16/17/18 all close to observed. (ii) Remaining systematic misses, judged against the between-subject variance: **Exp14 primacy-rescue is still ~2.1 SD too high** (0.3306 vs 0.1322; the arbiter's own falsifiable target was 0.10–0.20) — the early trusted channel is evidently not crushed enough at the fitted parameters, likely because t0 sat near its 0.12 ceiling and/or the gate corner is too soft. **Exp2 tally-allegiance is ~2 SD too high** (0.3042 vs 0.1583, essentially unchanged from pi_7's 0.2996 — the 'steeper gamma weakens allegiance' prediction did not materialize). **Exp8 FEDI undershoots** (0.3082 vs 0.3957; arbiter target 0.38–0.45) — the model anti-follows the final 0.90-validity expert too strongly. **Exp3 now slightly OVERSHOOTS** the anti-validity slope (-0.6225 vs -0.5756), and Exp5 adherence undershoots (0.196 vs 0.231) — both signs that gamma/tau over-steepened past their target. Exp12 (0.394 vs 0.540) and Exp11 (0.236 vs 0.315) undershoot the primacy slope, consistent with the reciprocal ramp flattening the g_j*t_j product on validity-matched cells, but both diffs are within 1 SD of the between-subject variance, so I weight them lightly. Critically, the overshoot on Exp3/Exp5/Exp8 points to a coherent single knob: gamma is too high, and lowering it simultaneously (a) pulls Exp3 back to -0.58, (b) lifts Exp5 toward 0.231, (c) reduces the over-anti-following of the high-validity final expert in Exp8, and (d) restores some high-validity (pro-TTB) weight that would soften Exp2's tally contrast — without touching the trusted-channel architecture that won Exp17/18.
Rationale: The candidate is accepted and becomes the base, but two >2SD residuals (Exp14, Exp2) plus a consistent over-steepening signature (Exp3 overshoot, Exp5/Exp8 undershoot) leave identifiable in-family gains. Keep the RTDT architecture verbatim and retune three knobs: (1) **Crush the early trusted floor harder**: narrow t0 from [0.00, 0.12] to [0.00, 0.05] (and check where the fit landed — the 0.33 rescue rate implies the optimizer used a soft corner). This is the direct lever on Exp14's 0.33-vs-0.13 miss and further secures Exp18's negative early-trusted contrast. Do NOT raise t0 to help Exp2 — that would undo Exp14; the Exp2/Exp14 conflict must be resolved on the kernel/noise side, not the trust side. (2) **Back off the kernel steepness**: gamma from [3.0, 4.2] down to roughly [2.7, 3.3] (tau can stay in [0.92, 0.96] or slip to [0.91, 0.95]). The steepening overshot its own target: Exp3 went from -0.44 to -0.62 past the observed -0.58, Exp5 adherence fell to 0.196 below the observed 0.231, and Exp8's final 0.90-expert is anti-followed too hard (0.308 vs 0.396). A slightly flatter kernel adds high-validity weight, which also pulls Exp2's tally contrast down from 0.304 toward 0.158 by restoring a little TTB-side following. (3) **Soften determinism a touch more for Exp2 only if (2) is insufficient**: raise the epsilon ceiling to ~0.20 or lower the beta floor to ~2.5, then verify Exp4 anchors stay >= 0.82 (currently 0.836 vs 0.843, so there is headroom). Optionally widen t1 upward ([1.0, 2.0]) or lower kappa toward [0.5, 0.9] to strengthen the late-trusted amplification for Exp8 without touching the early floor. Preserve the moderate gate corner (eta [4.5, 6.5], delta [0.15, 0.30]) — Exp15's composite and the Exp11/12 primacy slopes (already within noise) depend on it, and steepening the gate further would fight the t-ramp and flatten g*t more. Falsifiable checks for the next iteration: Exp14 rescue in [0.10, 0.22], Exp2 contrast in [0.15, 0.24], Exp8 FEDI in [0.35, 0.45], Exp3 slope in [-0.60, -0.54], with Exp4, Exp17, Exp18 held at their current near-perfect values.

**Outcome of this advice:** iter 1 candidate loss=0.0830 -> iter 2 candidate loss=0.0843 -> the gate marked it REJECTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-2 retune was REJECTED by the gate (0.0843 vs 0.0830), and the per-experiment trajectory shows exactly why: my iter-1 advice had one winning component and one losing component, and the losing component dominated. WINNER — the kernel backoff (gamma [2.7, 3.3]): Exp3 anti-validity slope moved from -0.6225 to -0.5569 (obs -0.5756, now within noise), Exp5 adherence 0.196→0.209 (obs 0.231), Exp9 0.328→0.343 (obs 0.341, essentially perfect), Exp10 -0.228→-0.190 (obs -0.173), and Exp2 improved marginally (0.304→0.289). LOSER — the trust-ramp changes (t0 crushed to [0.00, 0.05], t1 widened to [1.0, 2.0], kappa lowered to [0.5, 0.9]): they did NOT move their intended targets and broke several metrics that were previously fine. (a) Exp14 rescue went UP, not down (0.331→0.346 vs obs 0.132) — t0 is evidently not the rescue lever at all; the flatter kernel actually adds weight to the 95%@1 expert in the rescue option, offsetting any t0 effect. (b) Exp8 FEDI was untouched (0.308→0.310 vs obs 0.396) — the super-unity late-trust amplification bought nothing. (c) Exp18 catastrophically blew up: -0.967 (near-perfect, obs -0.932) → -3.80, a massive over-expression of the crushed-early-trusted pattern; with t0 pinned near 0 and t1 up to 2.0 the ramp ratio became extreme. (d) Every position/primacy composite also regressed: Exp13 1.428→1.354 (obs 1.836), Exp15 0.936→0.749 (obs 1.036), Exp11 0.236→0.204 (obs 0.315), Exp12 0.394→0.389 (obs 0.540), Exp17 slightly worse. The diagnosis is coherent: the extreme t-ramp flattened the effective g_j*t_j product, weakening the primacy architecture that Exp11-15 and Exp18 depend on, while the kernel backoff (the good change) delivered exactly its predicted gains.
Rationale: Return to the ACCEPTED iter-1 base and re-apply only the winning change, plus one NEW in-family lever for the primacy family of residuals. (1) KEEP the kernel backoff verbatim: gamma [2.7, 3.3], tau [0.91, 0.95]. This was the accepted part of iter-2 — Exp3/5/9/10 all moved toward observed, and it should be preserved. (2) REVERT the trust-ramp extremes: my iter-1 directive to crush t0 was refuted by the gate — rescue did not fall and Exp18 exploded. Restore t0 to a moderate band between the two flanking iterations, e.g. [0.03, 0.12], pull t1 back to [1.0, 1.6], and kappa to [0.6, 1.0]. The t1=2.0 / kappa=0.5 corner broke Exp18 without helping Exp8; Exp8's residual (0.31 vs 0.40, ~1.3 SD) should be tolerated rather than chased with ramp extremes that cost 60-SD elsewhere. (3) NEW LEVER for Exp14 (and the coherent undershooting cluster Exp11/12/13/15): steepen the primacy gate — eta from [4.5, 6.5] to roughly [5.5, 7.5], delta from [0.15, 0.30] to [0.12, 0.25]. Rationale: the Exp14 rescue trials are driven by the early 50%-validity expert (position 2) vs late 50% experts in the opposing option (the 95% experts carry negligible distrust weight), so a steeper gate directly pushes rescue toward the lapse floor (~0.13 observed). Simultaneously, ALL of Exp11 (0.20 vs 0.31), Exp12 (0.39 vs 0.54), Exp13 (1.35 vs 1.84), and Exp15 (0.75 vs 1.04) currently UNDERSHOOT their primacy composites — the common in-family knob is the gate, and my iter-1 instruction to freeze it is contradicted by the iter-2 evidence that the effective primacy signal is too weak once the ramp is reverted. Steepening the gate and reverting the ramp act in the same direction on g*t. (4) Keep the softened noise box (beta [2.5, 4.0], epsilon [0.12, 0.20]) — Exp2 improved to ~1.8 SD and Exp4 anchors retain headroom (0.851 vs 0.843); do not soften further. Falsifiable checks for the next iteration: Exp18 back in [-1.3, -0.6] (the single hardest constraint — if it leaves this band, the ramp is still too extreme), Exp14 rescue in [0.10, 0.22], Exp13 >= 1.5, Exp15 in [0.9, 1.2], Exp3 stays in [-0.60, -0.54], Exp4 >= 0.82, Exp17 in [-10, -8]. Do NOT re-crush t0 and do NOT re-widen t1 — both directions were tried and rejected by the gate.

**Outcome of this advice:** iter 2 candidate loss=0.0843 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "RECIPROCAL-TRUST DISTRUST TALLYING (RTDT), gate-steepened retune. Subjects scan expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, with the gate now at the steeper/low-floored corner (eta in [5.5, 7.5], delta in [0.12, 0.25]) so the first one or two experts carry the dominant share of attention while the late tail retains only a small floor residue. Stated validities are distrust-inverted (treated as error rates) via a moderately steep kernel inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z with gamma backed off to [2.7, 3.3] and tau in [0.91, 0.95] — the calibration the iter-2 experiment verified moves the anti-validity slope, validity adherence, and recency composites toward their observed values without overshooting. The reciprocal trust ramp is retained at its moderate band: t_j = t0*(t1/t0)^((j/(n-1))^kappa) with t0 in [0.03, 0.12] (early trusted claims heavily discounted but not annihilated), super-unity t1 in [1.0, 1.6] amplifying late high-validity experts above the flat share, and kappa in [0.6, 1.0] so trust recovers early in the tail. Weights combine multiplicatively, w_j proportional to g_j*[t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, with the softened noise box (beta [2.5, 4.0], epsilon [0.12, 0.20]) that iter-2 verified improves the tally-allegiance contrast while the anchors retain headroom.",
  "predict": "def predict(parameters, state, history):\n    # RTDT: Reciprocal-Trust Distrust Tallying, GATE-STEEPENED retune.\n    # Minimal-diff edit of the ACCEPTED iter-1 base per the critic's\n    # iter-2 directive. The iter-2 experiment decomposed cleanly:\n    #   WINNER (kept verbatim): the kernel backoff -- gamma [3.0,4.2] ->\n    #     [2.7, 3.3], tau [0.92,0.96] -> [0.91, 0.95]. This moved\n    #     Exp3 (-0.62 -> -0.557, obs -0.576), Exp5 (0.196 -> 0.209),\n    #     Exp9 (0.328 -> 0.343, obs 0.341), Exp10 (-0.228 -> -0.190)\n    #     and Exp2 (0.304 -> 0.289) all toward observed.\n    #   LOSER (reverted): the trust-ramp extremes (t0 crushed to\n    #     [0,0.05], t1 widened to [1.0,2.0], kappa lowered to [0.5,0.9])\n    #     broke Exp18 (-0.97 -> -3.80) and flattened the g*t primacy\n    #     product on Exps 11-15 without moving their intended targets\n    #     (Exp14 rescue went UP; Exp8 FEDI untouched). Reverted to the\n    #     moderate band: t0 [0.03, 0.12], t1 [1.0, 1.6], kappa [0.6, 1.0].\n    #   NEW LEVER: the primacy gate steepened -- eta [4.5,6.5] ->\n    #     [5.5, 7.5], delta [0.15,0.30] -> [0.12, 0.25]. The Exp14\n    #     rescue trials pivot on the early 50%-validity expert vs late\n    #     50% helpers, so a steeper gate pushes rescue toward the lapse\n    #     floor (~0.13 observed); simultaneously Exps 11/12/13/15 all\n    #     UNDERSHOOT their primacy composites, and steepening the gate\n    #     and reverting the ramp act in the same direction on g*t.\n    #   Noise box kept softened: beta [2.5, 4.0], epsilon [0.12, 0.20]\n    #     (iter-2 verified Exp2 improved to ~1.8 SD with anchor headroom).\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"RTDT expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    a, b = stim[0], stim[1]\n    n = int(stim.shape[1])\n\n    v = np.asarray(parameters[\"validities\"], dtype=float).ravel()\n    if v.shape[0] != n:\n        # Graceful fallback: strictly decreasing validities.\n        v = np.linspace(0.95, 0.55, n)\n    v = np.clip(v, 0.0, 1.0)\n\n    tau = float(parameters[\"tau\"])        # distrust share (BACKED OFF)\n    eta = float(parameters[\"eta\"])        # primacy-gate convexity (STEEPENED)\n    delta = float(parameters[\"delta\"])    # primacy-gate floor (LOW)\n    gamma = float(parameters[\"gamma\"])    # distrust-inversion steepness (BACKED OFF)\n    c = float(parameters[\"c\"])            # partial centering of error rates\n    t0 = float(parameters[\"t0\"])          # early-trust floor (MODERATE, reverted)\n    t1 = float(parameters[\"t1\"])          # late-trust residue (SUPER-UNITY, reverted)\n    kappa = float(parameters[\"kappa\"])    # reciprocal-ramp exponent (reverted)\n    beta = float(parameters[\"beta\"])      # choice consistency (SOFTENED)\n    eps = float(parameters[\"epsilon\"])    # uniform lapse (SOFTENED)\n\n    # (1) DISTRUST INVERSION with partial centering: treat the stated\n    # validity as an error rate; subtract a fraction c of the smallest\n    # error rate before raising to the gamma power, then normalize.\n    # gamma in [2.7, 3.3] with tau in [0.91, 0.95] is the verified\n    # calibration: steep enough to carry the anti-validity gradient in\n    # validity-position-confounded layouts, flat enough not to\n    # over-anti-follow high-validity cues (Exp3/5/8/10 all verified).\n    err = 1.0 - v\n    centered = np.maximum(err - c * float(err.min()), 0.0)\n    inv = np.power(centered, gamma)\n    Z_v = float(inv.sum())\n    if Z_v > 1e-12:\n        w_inv = inv / Z_v\n    else:\n        # Degenerate design (all validities tied at the extreme): the\n        # distrust kernel is flat and position alone carries structure.\n        w_inv = np.full(n, 1.0 / n)\n\n    if n > 1:\n        pos = np.arange(n, dtype=float)\n        # (2) RECIPROCAL TRUST RAMP at the moderate band, computed in\n        # log space for numerical stability:\n        #   log t_j = log t0 + (j/(n-1))^kappa * (log t1 - log t0).\n        # t0 in [0.03, 0.12] discounts early trusted claims without\n        # annihilating them (the iter-2 t0~0 corner exploded Exp18);\n        # t1 in [1.0, 1.6] with kappa in [0.6, 1.0] amplifies late\n        # trusted experts above the flat share moderately.\n        t0_eff = max(t0, 1e-4)  # guard the geometric form against t0 = 0\n        t1_eff = max(t1, 1e-4)\n        frac = np.power(pos / (n - 1.0), kappa)  # in [0, 1]\n        t = np.exp(np.log(t0_eff) + frac * (np.log(t1_eff) - np.log(t0_eff)))\n        # (3) STEEP FLOORED CONVEX PRIMACY GATE, steepened per the\n        # critic's iter-2 directive: eta in [5.5, 7.5], delta in\n        # [0.12, 0.25]. The steeper gate pushes Exp14's rescue rate\n        # toward the lapse floor and strengthens the primacy composites\n        # on Exps 11/12/13/15, which all currently undershoot.\n        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)\n    else:\n        t = np.array([1.0])\n        g = np.array([1.0])\n\n    # (4) MULTIPLICATIVE combination: the gate scales the bracket\n    # cue-by-cue. The trusted channel is modulated by the reciprocal\n    # ramp -- heavily discounted early, moderately amplified late --\n    # while the distrust kernel (tau >= 0.91) dominates the bracket\n    # elsewhere.\n    bracket = t * (1.0 - tau) / n + tau * w_inv\n    raw = g * bracket\n    Z_t = float(raw.sum())\n    if Z_t > 1e-12:\n        w = raw / Z_t\n    else:\n        w = np.full(n, 1.0 / n)\n\n    # (5) Deterministic attention: every discriminating cue is attended;\n    # no capacity interference, beta is design-independent.\n    s = np.sign(a - b)\n\n    # Additive integration of all discriminating cues.\n    d = float(np.dot(w, s))\n\n    # (6) Numerically stable logistic choice rule -- beta in [2.5, 4.0]\n    # with a uniform lapse epsilon in [0.12, 0.20]: the softened noise\n    # box the iter-2 experiment verified (Exp2 improved toward 0.158\n    # while Exp4 anchors retain headroom above 0.82).\n    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))\n    probs = np.array([p_a, 1.0 - p_a], dtype=float)\n    probs = (1.0 - eps) * probs + eps * 0.5\n    return probs",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = probs / probs.sum()  # guard against float drift\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "tau": "[0.91, 0.95]",
    "eta": "[5.5, 7.5]",
    "delta": "[0.12, 0.25]",
    "gamma": "[2.7, 3.3]",
    "c": "[0.05, 0.25]",
    "t0": "[0.03, 0.12]",
    "t1": "[1.0, 1.6]",
    "kappa": "[0.6, 1.0]",
    "beta": "[2.5, 4.0]",
    "epsilon": "[0.12, 0.20]",
    "validities": "validities"
  },
  "rationale": "This is a minimal-diff retune of the ACCEPTED iter-1 RTDT base that applies exactly the three components of the critic's iter-2 directive, each of which is now evidence-graded by the gate's accept/reject trajectory. (1) KEEP THE KERNEL BACKOFF VERBATIM (gamma [2.7, 3.3], tau [0.91, 0.95]): the iter-2 experiment, though rejected overall, showed this single change moved every intended target toward observed — Exp3 slope -0.6225 -> -0.5569 (obs -0.5756, now within noise), Exp5 adherence 0.196 -> 0.209 (obs 0.231), Exp9 0.328 -> 0.343 (obs 0.341, essentially perfect), Exp10 -0.228 -> -0.190 (obs -0.173), Exp2 0.304 -> 0.289 (obs 0.158). This is the winning half of iter-2 and is preserved without modification. (2) REVERT THE TRUST-RAMP EXTREMES: the iter-1 directive to crush t0 to [0, 0.05] and the iter-2 widening of t1 to 2.0 with kappa 0.5 were both refuted — Exp14 rescue went UP not down (t0 is not the rescue lever), Exp8 FEDI was untouched (0.308 -> 0.310), and Exp18 exploded from a near-perfect -0.967 to -3.80 because the extreme ramp ratio over-expressed the crushed-early-trusted pattern far past the observed -0.93. The ramp returns to a moderate band between the two flanking iterations: t0 [0.03, 0.12], t1 [1.0, 1.6], kappa [0.6, 1.0]. Exp8's residual (~1.3 SD) is tolerated rather than chased with ramp extremes that cost 60-SD on Exp18. (3) NEW LEVER — STEEPEN THE PRIMACY GATE (eta [4.5, 6.5] -> [5.5, 7.5], delta [0.15, 0.30] -> [0.12, 0.25]): this addresses the two coherent residual clusters the iter-1 base left open. First, Exp14's primacy-rescue (0.331 vs obs 0.132, >2 SD): the rescue trials pivot on the early 50%-validity expert in the rescue option vs late 50%-validity helpers in the opposing option (the 95% experts carry almost no distrust weight), so a steeper gate directly pushes the rescue rate toward the lapse floor ~0.13 — the structural fix t0 could not deliver. Second, the primacy-composite undershoot cluster (Exp11 0.236 vs 0.315, Exp12 0.394 vs 0.540, Exp13 1.428 vs 1.836, Exp15 0.936 vs 1.036): all four currently undershoot, and steepening the gate while reverting the ramp act in the same direction on the g_j*t_j product, restoring the effective primacy signal the extreme ramp had flattened. The softened noise box (beta [2.5, 4.0], epsilon [0.12, 0.20]) is kept as verified in iter-2: Exp2 improved to ~1.8 SD while Exp4 anchors retain headroom (0.851 vs 0.843). Falsifiable checks for this iteration: Exp18 back in [-1.3, -0.6] (the hardest constraint — if it leaves this band the ramp is still too extreme), Exp14 rescue in [0.10, 0.22], Exp13 >= 1.5, Exp15 in [0.9, 1.2], Exp3 stays in [-0.60, -0.54], Exp4 >= 0.82, Exp17 in [-10, -8]. The architecture, mechanism family, and all function structure are unchanged from the accepted base — only five parameter ranges (gamma, tau, eta, delta, t0/t1/kappa, beta/epsilon) are retuned."
}
```
```

## Usage

```json
{
  "input_tokens": 65183,
  "output_tokens": 3811,
  "reasoning_tokens": 145,
  "provider": "Sail Research"
}
```
