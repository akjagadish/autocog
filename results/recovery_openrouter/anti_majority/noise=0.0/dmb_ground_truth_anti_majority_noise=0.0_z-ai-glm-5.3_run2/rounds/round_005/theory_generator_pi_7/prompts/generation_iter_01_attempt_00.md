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
- THEORY 1 = `pi_3_1`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_3_1`).

Replace THEORY 1 (SCWI) with a new theory, tentatively 'Compressive Anti-Validity Integration with a Moderate Deterministic Tail' (CAVI), designed as a stronger competitor to CNNAI rather than a return of SCWI. Mechanism sketch: (1) Keep the sign inversion — a rating of 1 is comprehended as a defect, per-cue defect magnitude d_j = v_j + tau*v_j^2 with tau in [0.35, 0.65] (a wider, higher range than SCWI's, needed for Experiment 1's +2.88 near-knife excess). (2) NEW: within-option diminishing sensitivity — the option's subjective defect load is D = (sum_j d_j * x_j)^kappa with kappa in [0.55, 0.85] (or equivalently a log-compressive map), so an option's value is -D. Small stacks (near-knife cells like AL, S-cells, M-cells) keep near-linear sharp margins, while large stacked endorsements grow sub-proportionally aversive — this flattens the high-ratio psychometrics that both current theories overshoot (Exp 10's 5x0.90-vs-11x0.50 kill cells down toward 0.60, Exp 12's rise down toward 0.30, Exp 4's disagreement-vs-agreement gap up toward 0.27 because compressed margins sit closer to the noise scale). (3) Population mixture: ~14% true guessers (beta ~ 0.02-0.10), ~70% engaged with lognormal beta (median ~2.0, log-sd 0.35, clipped [0.8, 6.0]), and ~16% deterministic tail (beta uniform [4.0, 6.5], lapse 0.02-0.04) — this reproduces Experiment 2's g ≈ 0.09 (a small but nonzero near-ceiling bloc, refuting both CNNAI's 0.00 and SCWI's 0.36) while keeping population means CNNAI-like on Exps 5-9. (4) Context elasticity retained but recalibrated: per-bucket eta redraw keyed on the rounded mean endorsed validity (as in CNNAI), with the stable-to-redraw variance ratio set for latent cross-context consistency r ≈ 0.20-0.25, matching Experiment 11's observed 0.196 between SCWI's 0.37 and CNNAI's 0.11. (5) Softmax(beta * (D_B - D_A)) plus symmetric lapse (0.03-0.09 engaged) and per-trial attention diffusion (sd ~0.05). This theory differs from SCWI by the compressive nonlinearity, the smaller deterministic bloc, and the context-keyed elasticity; it differs from CNNAI by the stronger supralinear per-cue contrast, the explicit deterministic tail, and the compression that CNNAI lacks. Parameter ranges should be tuned so that: Exp 1 AL excess lands in [+2.4, +3.2]; Exp 2 g in [0.06, 0.15]; Exp 10 kill-composite in [0.55, 0.65]; Exp 12 rise in [0.22, 0.36]; Exp 11 cross-ladder correlation in [0.15, 0.30].

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.858)

**Description**
Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = -(v_j + tau * v_j^2)
    # per unit rating (anti-validity weighting, supralinear in the
    # instructed validity). Option values are the cue-weighted sums;
    # choice is a numerically stable softmax over beta * values, mixed
    # with a symmetric lapse. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])

    # Signed (anti-validity) cue weights: a rating of 1 is evidence
    # AGAINST an option, and the inversion strength grows supralinearly
    # with the instructed validity of the expert.
    w = -(val + tau * np.square(val))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [1.1, 1.7]
- tau: [0.25, 0.45]
- epsilon: [0.05, 0.15]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1747 (var=0.0108) vs this=-0.1877 (var=0.0014)
- Experiment 2: real=0.2737 (var=0.0930) vs this=0.1862 (var=0.0095)
- Experiment 3: real=0.1227 (var=0.0008) vs this=0.1378 (var=0.0002)
- Experiment 4: real=0.1625 (var=0.0119) vs this=0.1608 (var=0.0029)
- Experiment 5: real=0.6837 (var=0.0643) vs this=0.5531 (var=0.0069)
- Experiment 6: real=0.6300 (var=0.0703) vs this=0.5687 (var=0.0053)
- Experiment 7: real=0.6462 (var=0.0395) vs this=0.6412 (var=0.0050)
- Experiment 8: real=0.6044 (var=0.0653) vs this=0.6056 (var=0.0099)
- Experiment 9: real=0.1963 (var=0.0601) vs this=0.2390 (var=0.0208)
- Experiment 10: real=0.2963 (var=0.0806) vs this=0.3713 (var=0.0215)
- Experiment 11: real=2.8800 (var=0.0015) vs this=1.0800 (var=0.0009)
- Experiment 12: real=0.0913 (var=0.0729) vs this=0.0000 (var=0.0000)


---

### `pi_6` (overall score: 0.842)

**Description**
Context-Normalized Noisy Anti-Validity Integration (CNNAI), knife-avoidance recalibration. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity CONTEXT BUCKETS (the mean endorsed validity rounded to 0.1), so all trials sharing a bucket inherit one shared per-subject draw. The elasticity decomposes into a small stable per-subject component (eta_base ~ N(0.02, 0.09)) and a larger per-bucket redraw (~N(0.10, 0.19)), total hard-clipped to [-0.25, 0.70]; the variance ratio fixes latent cross-context consistency at r ~ 0.18 by construction. The engaged eta distribution (mean ~0.12, sd ~0.21) is deliberately centered well BELOW the kill-design knife region (eta ~ 0.31-0.34 where high-validity stacks flip to low-validity stacks), so the engaged bulk behaves SCWI-like in effective curvature while margins on the below-knife side stay large. A modest per-trial multiplicative attention diffusion (sd 0.05) supplies within-context stochasticity, choice is softmax(beta * values) with nearly homogeneous engaged beta (lognormal median ~2.14, log-sd 0.20) plus a narrow symmetric lapse (0.03-0.09), and a 13% true-guesser subpopulation is retained, uncorrelated with eta.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Context-Normalized Noisy Anti-Validity Integration (CNNAI),
    # iter 6 -- knife-avoidance recalibration of the accepted iter-5 base
    # per the iter-5 critic diagnosis. Mechanism family UNCHANGED:
    # additive anti-validity integration, context-referenced elasticity
    # w_j = -v_j * (v_j / v_ref)^eta with v_ref = mean validity of the
    # endorsed cues, per-trial attention diffusion, softmax + symmetric
    # lapse, guesser mixture. Changes vs the accepted iter-5 base:
    #   (A) ETA CENTER LOWERED 0.23 -> 0.12 (eta_base ~ N(0.02, 0.09) plus
    #       per-context redraw ~ N(0.10, 0.19), total clipped [-0.25, 0.70]).
    #       The kill composites (Exps 5/6/8) are near-knife statistics:
    #       their flip knives sit at eta ~ 0.30-0.34, and on the
    #       below-knife side the score margins GROW as eta decreases
    #       (e.g. Exp 5's 2x1.00-vs-5x0.50 cell: margin 0.14 at eta 0.23
    #       vs 0.33 at eta 0.10). Centering at 0.12 both moves engaged
    #       mass below the knives (P(eta < 0.33) ~ 0.83) and enlarges
    #       the margins, lifting the undershooting composites.
    #   (B) ATTENTION DIFFUSION CUT 0.10 -> 0.05: at 0.10 the per-trial
    #       margin noise (sd ~ 0.25 on a 12-cue cell) was comparable to
    #       the near-knife signals (~0.15-0.35), damping below-knife
    #       subjects to p ~ 0.55-0.6 -- the main softening channel on
    #       Exps 5/6/8. Per-trial noise averages out within a subject
    #       (iter-2 critic), so cutting it lifts means without touching
    #       between-subject variance.
    #   (C) CONTEXT KEY COARSENED to the mean endorsed validity rounded
    #       to 0.1 (a bucketed v_ref / ladder identifier). The critic's
    #       'rounded MAX endorsed validity' variant was NOT used: both
    #       Exp-9 ladders ({1.00,0.50} and {1.00,0.60} cells) have max
    #       validity 1.0, so a max-based key would map them to the SAME
    #       draw and re-couple them -- recreating the iter-1 r = 0.44
    #       failure the critic itself warns against. Bucketed mean keeps
    #       them distinct (0.6 vs 0.7) while merging within-experiment
    #       fragmentation (Exp 2: ~2-3 keys instead of ~6; Exp 5: 2 keys),
    #       so per-subject composites stop averaging over many
    #       independent eta draws.
    #   (D) EXP-9 COUPLING CHANNELS CLOSED: engaged epsilon narrowed
    #       0.02-0.17 -> 0.03-0.09 (stable lapse was the dominant
    #       residual cross-ladder trait), eta_base sd cut 0.13 -> 0.09
    #       (latent r = 0.0081/(0.0081+0.0361) ~ 0.18), beta median
    #       trimmed 2.34 -> 2.14 with log-sd 0.25 -> 0.20 (guards Exps
    #       1/3/10 against the noise cut, and removes the last sizeable
    #       stable-trait correlation channel).
    #   (E) v_ref STAYS MEAN-REFERENCED (not max): the weight ratio
    #       w(v_h)/w(v_l) = (v_h/v_l)^(1+eta) is invariant to v_ref, and
    #       max-referencing multiplies every margin by v_max^(-eta) <= 1,
    #       which would DEFLATE exactly the near-knife margins the
    #       composites need. The critic's goal (steeper high-validity
    #       region) is achieved via (A)+(B) instead.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CNNAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. These subjects sit at ~0.5 on
        # every diagnostic cell and dilute population means/slopes.
        t = u / 0.13
        beta = 0.02 + 0.08 * t            # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps       # 0.12 .. 0.25
        eta_base = 0.10                  # irrelevant at beta ~ 0
    else:
        # Engaged subpopulation (~87%). Small stable elasticity
        # component (sd 0.09) paired with a larger per-context redraw
        # (sd 0.19): the variance ratio fixes latent cross-context
        # consistency at r ~ 0.18 by construction. The stable lapse
        # range is NARROW (0.03-0.09) so it can no longer act as a
        # hidden coupling channel across contexts, and beta is nearly
        # homogeneous (lognormal sd 0.20).
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.76 + 0.20 * float(ndtri(bs)))  # median ~2.14
        beta = min(max(beta, 0.9), 4.5)
        epsilon = 0.03 + 0.06 * ps       # 0.03 .. 0.09

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    # v_ref = mean validity of the cues actually displayed (endorsed)
    # this trial; the context key is v_ref rounded to one decimal, so
    # all cells of one validity ladder share a single per-subject eta
    # draw while different ladders (e.g. Exp 9's 0.50-ladder at bucket
    # 0.6 vs its 0.60-ladder at bucket 0.7) draw independently.
    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw, centered low so the engaged bulk sits below
    # the kill knives (~0.31-0.34) with large below-knife margins:
    # engaged eta ~ N(0.12, 0.21), P(eta < 0.33) ~ 0.83.
    eta_ctx = eta_base + 0.10 + 0.19 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion on cue weights,
    # sd cut 0.10 -> 0.05: at 0.10 the margin noise on near-knife cells
    # was comparable to the signal and was the main softening channel
    # on the Exp 5/6/8 composites.
    att = np.exp(0.05 * np.random.randn(n_features))
    w = w * att

    scores = np.array([float(np.dot(w, ra)), float(np.dot(w, rb))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- subject_seed: [0, 1]
- regime_seed: [0, 1]
- eta_base_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1747 (var=0.0108) vs this=-0.1968 (var=0.0072)
- Experiment 2: real=0.2737 (var=0.0930) vs this=0.1650 (var=0.0201)
- Experiment 3: real=0.1227 (var=0.0008) vs this=0.1497 (var=0.0011)
- Experiment 4: real=0.1625 (var=0.0119) vs this=0.1342 (var=0.0135)
- Experiment 5: real=0.6837 (var=0.0643) vs this=0.6150 (var=0.0208)
- Experiment 6: real=0.6300 (var=0.0703) vs this=0.6208 (var=0.0179)
- Experiment 7: real=0.6462 (var=0.0395) vs this=0.6581 (var=0.0250)
- Experiment 8: real=0.6044 (var=0.0653) vs this=0.7011 (var=0.0213)
- Experiment 9: real=0.1963 (var=0.0601) vs this=0.1113 (var=0.0229)
- Experiment 10: real=0.2963 (var=0.0806) vs this=0.3438 (var=0.0509)
- Experiment 11: real=2.8800 (var=0.0015) vs this=0.3200 (var=0.0016)
- Experiment 12: real=0.0913 (var=0.0729) vs this=0.0000 (var=0.0000)


---

### `pi_5` (overall score: 0.638)

**Description**
Heterogeneous-Elasticity Anti-Validity Integration (HEAI), bimodal-population refinement. Subjects integrate ALL expert ratings additively into a per-option subjective value, but a rating of 1 is comprehended as a DEFECT (evidence against the option) — a sign inversion of the rating direction. The anti-weight of expert j is a per-subject power law of the instructed validity, w_j = -v_j^gamma, where gamma ('validity elasticity') is drawn from a BIMODAL continuous population: a shallow majority (75% of engaged subjects, gamma ~ N(1.04, 0.22) truncated [0.60, 1.42]) that sits cleanly below the kill-design count-ratio thresholds (~1.31-1.34) and therefore follows the high-validity stack, and a steep minority (25% of engaged subjects, gamma ~ N(2.10, 0.26) truncated [1.65, 2.75]) that sits cleanly above them, whose near-lexicographic top-cue dominance drives the Exp-2 dissociation and keeps the Exp-8/Exp-4 composites from overshooting. Inverse temperature is mode-linked (shallow: lognormal median ~2.6; steep: lognormal median ~3.3; both heavy-tailed), a 16% guesser subpopulation (beta ~ 0) and a symmetric lapse epsilon are retained unchanged. The population mean elasticity (~1.30) stays below the arbiter's 1.4 ceiling. The key change from the falsified wide-unimodal shape is that almost no mass is wasted in the ambiguous 1.2-1.6 band where subjects sit at chance on the diagnostic cells: the shallow mode's thin upper shoulder (1.32-1.42) is retained deliberately, because subjects just above the kill thresholds miss Exp 8 while still hitting the higher-threshold Exp 5/7 ladder cells — the lever that lets Exp 8's mean and variance stay calibrated without capping Exp 5/6/7.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Heterogeneous-Elasticity Anti-Validity Integration (HEAI), iter 3.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Mechanism (UNCHANGED from the accepted iter-2 base): every cue
    # contributes signed evidence w_j = -v_j**gamma per unit rating
    # (a rating of 1 is a defect; the anti-weight is a per-subject
    # power law of the instructed validity). Choice is a numerically
    # stable softmax over beta * option values, mixed with a symmetric
    # lapse epsilon. History is ignored (no trial-by-trial feedback).
    #
    # Population edits on top of the accepted iter-2 base (mechanism
    # untouched), implementing the iter-2 critic's diagnosis that a
    # wide UNIMODAL gamma dumps its spread into the ambiguous
    # threshold band (gamma ~ 1.2-1.6) where subjects sit at ~chance
    # on every diagnostic cell, capping the means on BOTH sides:
    #   * GAMMA MADE BIMODAL: the engaged population is now a mixture
    #     of a SHALLOW majority (75%: N(1.04, 0.22) trunc [0.60, 1.42])
    #     and a STEEP minority (25%: N(2.10, 0.26) trunc [1.65, 2.75]),
    #     with minimal mass in between. The shallow mode sits cleanly
    #     below the kill-design count-ratio thresholds (~1.31-1.34 on
    #     Exp 8; ~1.32+ on the Exp 5 ladder), so it drives high-stack
    #     adherence on Exp 5/6/7/8; its thin upper shoulder (1.32-1.42,
    #     ~10% of the mode) deliberately remains to keep Exp 8's mean
    #     and between-subject variance calibrated. The steep mode sits
    #     cleanly above ~1.8 (critic's prescription; extends to 2.75 so
    #     the top cue can dominate three lower cues), which is what
    #     lifts the Exp 2 dissociation mean/variance and holds Exp 4
    #     down. Population mean elasticity ~ 1.30 < 1.4 (arbiter box).
    #   * BETA MODE-LINKED AND HOTTER: shallow subjects draw lognormal
    #     beta, median exp(0.95) ~ 2.6 (up from ~2.3), sigma 0.70,
    #     clip [0.6, 7.0]; steep subjects draw a hotter lognormal,
    #     median exp(1.20) ~ 3.3, sigma 0.60, clip [1.2, 8.0] (the
    #     steep 'confident lexicographic' subjects are also the
    #     sharpest). Hotter shallow beta sharpens engaged adherence on
    #     Exp 5/6/7; hotter steep beta maximizes the Exp 2 contribution
    #     per steep subject while pushing steep Exp 8/Exp 4 hits down.
    #   * Guesser fraction (16%), epsilon ranges, seeds, and everything
    #     else are untouched.
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HEAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    # Keep CDF seeds strictly inside (0, 1) so ndtri stays finite.
    gs = min(max(_seed("gamma_seed"), 1e-6), 1.0 - 1e-6)
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    es = _seed("epsilon_seed")

    if u < 0.16:
        # Guesser subpopulation (~16%): beta ~ 0 makes the softmax core
        # essentially uniform on every trial; lapse is elevated.
        t = u / 0.16
        beta = 0.02 + 0.08 * t          # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * es      # 0.12 .. 0.25
        gamma = 1.12                    # irrelevant at beta ~ 0
    else:
        if gs < 0.75:
            # SHALLOW majority (~75% of engaged): gamma ~ N(1.04, 0.22)
            # truncated to [0.60, 1.42]. Cleanly below the kill-design
            # thresholds (~1.31-1.34), with a thin upper shoulder that
            # keeps Exp 8's mean/variance calibrated.
            t = gs / 0.75
            tt = min(max(t, 1e-4), 1.0 - 1e-4)
            gamma = 1.04 + 0.22 * float(ndtri(tt))
            gamma = min(max(gamma, 0.60), 1.42)

            # Hot shallow beta: lognormal median ~2.6, heavy tail.
            z_b = float(ndtri(bs))
            beta = math.exp(0.95 + 0.70 * z_b)
            beta = min(max(beta, 0.6), 7.0)
        else:
            # STEEP minority (~25% of engaged): gamma ~ N(2.10, 0.26)
            # truncated to [1.65, 2.75]. Cleanly above ~1.8 so the top
            # cue can dominate three lower cues (the Exp 2 lever).
            t = (gs - 0.75) / 0.25
            tt = min(max(t, 1e-4), 1.0 - 1e-4)
            gamma = 2.10 + 0.26 * float(ndtri(tt))
            gamma = min(max(gamma, 1.65), 2.75)

            # Steep subjects are also sharper: lognormal median ~3.3.
            z_b = float(ndtri(bs))
            beta = math.exp(1.20 + 0.60 * z_b)
            beta = min(max(beta, 1.2), 8.0)

        # Symmetric lapse for engaged subjects (unchanged).
        epsilon = 0.03 + 0.14 * es      # 0.03 .. 0.17

    # Anti-validity power-law cue weights: a rating of 1 counts AGAINST
    # the option, with magnitude v_j**gamma.
    w = -np.power(val, gamma)

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- regime_seed: [0, 1]
- gamma_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1747 (var=0.0108) vs this=-0.1898 (var=0.0057)
- Experiment 2: real=0.2737 (var=0.0930) vs this=0.1550 (var=0.0256)
- Experiment 3: real=0.1227 (var=0.0008) vs this=0.1165 (var=0.0040)
- Experiment 4: real=0.1625 (var=0.0119) vs this=0.1979 (var=0.0276)
- Experiment 5: real=0.6837 (var=0.0643) vs this=0.6019 (var=0.0710)
- Experiment 6: real=0.6300 (var=0.0703) vs this=0.7004 (var=0.0351)
- Experiment 7: real=0.6462 (var=0.0395) vs this=0.6056 (var=0.0640)
- Experiment 8: real=0.6044 (var=0.0653) vs this=0.6122 (var=0.0654)
- Experiment 9: real=0.1963 (var=0.0601) vs this=0.9529 (var=0.0191)
- Experiment 10: real=0.2963 (var=0.0806) vs this=0.1713 (var=0.0509)
- Experiment 11: real=2.8800 (var=0.0015) vs this=0.7200 (var=0.0008)
- Experiment 12: real=0.0913 (var=0.0729) vs this=0.0823 (var=0.0589)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1319 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1319 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Slope of P(choose the top-cue-favored option) regressed on the
    # tally margin of that favored option, across all trials where the
    # highest-validity discriminating cue exists (ties like trials 17/18
    # are skipped). Under TTB this probability is flat in the tally margin
    # (slope ~ 0); under Tallying it rises monotonically with the tally
    # margin (positive slope).
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.65])
    cue_order = np.argsort(-validities, kind="stable")

    xs = []
    ys = []
    ra_col = data["option_a_ratings"]
    rb_col = data["option_b_ratings"]
    resp_col = data["response"]

    for ra, rb, resp in zip(ra_col, rb_col, resp_col):
        a = np.asarray(ra, dtype=float)
        b = np.asarray(rb, dtype=float)

        # Find the first discriminating cue in descending-validity order.
        favored = None
        for j in cue_order:
            if a[j] > b[j]:
                favored = 0
                break
            if b[j] > a[j]:
                favored = 1
                break
        if favored is None:
            # Full tie: no top cue, no tally signal. Skip.
            continue

        # Tally margin from A's perspective, then re-sign so it is from
        # the top-cue-favored option's perspective.
        d = float(np.sum(a > b) - np.sum(b > a))
        if favored == 1:
            d = -d

        xs.append(d)
        ys.append(1.0 if int(resp) == favored else 0.0)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return 0.0

    xm = xs.mean()
    ym = ys.mean()
    sxx = float(np.sum((xs - xm) ** 2))
    if sxx <= 0.0:
        return 0.0

    slope = float(np.sum((xs - xm) * (ys - ym)) / sxx)
    return slope
```

**Observed (real) value:** -0.1747 (var=0.0108)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.1797 (var=0.0087) (Δ vs real -0.0050)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0023 (var=0.0008)
- pi_2: 0.1952 (var=0.0019)
- pi_3: -0.1877 (var=0.0014)
- pi_4: -0.1883 (var=0.0047)
- pi_3_1: -0.2228 (var=0.0055)
- pi_5: -0.1898 (var=0.0057)
- pi_6: -0.1968 (var=0.0072)

### Experiment 2
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Cue cascade order for validities [0.9, 0.8, 0.7, 0.6, 0.5]:
    # strictly descending, so TTB consults features 0,1,2,3,4 in order.
    cue_order = [0, 1, 2, 3, 4]

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            return np.array([float(v) for v in s.replace(',', ' ').split()])
        return np.asarray(x, dtype=float).ravel()

    follow_disagree = 0
    n_disagree = 0
    follow_agree = 0
    n_agree = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != 5 or b.size != 5:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))

        # Tallying must be decisive and the tally margin must be exactly 1:
        # on every such trial the softmax probability of following the tally
        # is sigmoid(beta * 1), identical across trials, so Tallying predicts
        # a perfectly FLAT follow-rate regardless of subset.
        if a_wins == b_wins:
            continue
        if abs(a_wins - b_wins) != 1:
            continue

        tally_picks_a = a_wins > b_wins

        ttb_picks_a = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_picks_a = True
                break
            if b[j] > a[j]:
                ttb_picks_a = False
                break
        if ttb_picks_a is None:
            continue

        chose_a = (float(resp) == 0.0)

        if ttb_picks_a == tally_picks_a:
            n_agree += 1
            if chose_a == tally_picks_a:
                follow_agree += 1
        else:
            n_disagree += 1
            if chose_a == tally_picks_a:
                follow_disagree += 1

    if n_disagree == 0 or n_agree == 0:
        return 0.0

    p_dis = follow_disagree / float(n_disagree)
    p_agr = follow_agree / float(n_agree)
    return p_dis - p_agr

```

**Observed (real) value:** 0.2737 (var=0.0930)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2019 (var=0.0142) (Δ vs real -0.0719)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0006 (var=0.0060)
- pi_1: -0.6981 (var=0.0507)
- pi_3: 0.1862 (var=0.0095)
- pi_4: 0.2669 (var=0.0115)
- pi_3_1: 0.1713 (var=0.0139)
- pi_5: 0.1550 (var=0.0256)
- pi_6: 0.1650 (var=0.0201)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Slope of P(choose A) regressed on the anti-validity weighted cue
    margin D = sum_j u_j * (b_j - a_j), with u_j = v_j + tau * v_j^2
    (tau fixed at 0.35, the midpoint of SCWI's tau range).

    D > 0 means option A carries fewer/smaller anti-validity 'defects'
    and is therefore preferred by SCWI. SCWI predicts a POSITIVE slope
    (~0.2-0.3 with beta in [1.1,1.7], epsilon in [0.05,0.15]).
    Tallying cannot produce a positive slope: on count-tied trials
    (pairs 1-8, 24) it outputs exactly 0.5 for any beta/epsilon, and on
    all remaining trials its choice probability tracks the raw count
    margin, which is strongly ANTI-correlated with D across this design
    (the tally winner is almost always the weighted-margin loser). So
    Tallying's slope is <= 0 (approximately 0 for small beta, clearly
    negative for large beta). The sign separation is parameter-free.
    """
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    tau = 0.35
    u = validities + tau * np.square(validities)

    def _to_arr(cell):
        arr = np.asarray(cell, dtype=float).ravel()
        if arr.size != u.size:
            raise ValueError(
                f"Expected {u.size} ratings, got {arr.size}"
            )
        return arr

    a = np.vstack([_to_arr(c) for c in data["option_a_ratings"]])
    b = np.vstack([_to_arr(c) for c in data["option_b_ratings"]])

    # Signed SCWI evidence in favor of A (positive -> A preferred).
    D = (b - a) @ u

    # response: 0 = chose A, 1 = chose B  ->  indicator of choosing A.
    choice_a = 1.0 - np.asarray(data["response"], dtype=float).ravel()

    Dc = D - D.mean()
    Yc = choice_a - choice_a.mean()
    denom = float(np.sum(Dc * Dc))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(Dc * Yc) / denom)
```

**Observed (real) value:** 0.1227 (var=0.0008)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1404 (var=0.0029) (Δ vs real +0.0177)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1378 (var=0.0002)
- pi_2: -0.1263 (var=0.0006)
- pi_1: -0.0888 (var=0.0010)
- pi_4: 0.1448 (var=0.0005)
- pi_3_1: 0.1444 (var=0.0038)
- pi_5: 0.1165 (var=0.0040)
- pi_6: 0.1497 (var=0.0011)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Conflict-set tally-following rate.
    #
    # Selection rule: keep only trials where (i) the tally is non-zero (one
    # option strictly wins more feature-wise comparisons) and (ii) a
    # validity-weighted signed integrator with anti-validity weights
    # w_j = v_j + tau * v_j^2 (tau = 0.35, the midpoint of SCWI's range)
    # STRONGLY prefers the tally LOSER (|evidence| > 0.8).
    # In this design that selects exactly Probe B (pairs 7-12, |D| in
    # [1.0, 1.3]) plus the anchors (pairs 13-14, |D| ~ 5.2), i.e. 8 of the
    # 16 unique pairs = 48 of 96 trials per subject.
    #
    # Metric value = proportion of choices that follow the TALLY WINNER on
    # those conflict trials.
    #   Tallying: P >= sigmoid(beta*|m|) with |m| in {1,3,6}  -> > 0.5 for
    #             ANY beta > 0, epsilon < 1; ~0.95+ for typical beta.
    #   SCWI:     P = (1-eps)*(1-sigmoid(beta*|D|)) + eps/2 ~ 0.16-0.20 on
    #             Probe B and ~ eps/2 ~ 0.075 on anchors -> far below 0.5.
    # The two theories are separated by ~0.4-0.8 in mean with tight
    # per-subject binomial variance (48 trials/subject).

    import numpy as np

    VALIDITIES = np.array([1.0, 0.95, 0.65, 0.5, 0.5, 0.5])
    TAU = 0.35      # midpoint of SCWI's tau box [0.25, 0.45]
    THRESH = 0.8    # cleanly above Probe A's |D| <= 0.08, below Probe B's |D| >= 1.0

    w = VALIDITIES + TAU * np.square(VALIDITIES)  # positive anti-validity magnitudes

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    n_sel = 0
    n_match = 0
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.size == 0 or a.size != b.size or a.size != w.size:
            continue

        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            # Full-tie trials (pairs 15/16): both theories say 0.5, uninformative.
            continue

        # Signed evidence for option A under a weighted anti-validity
        # integrator: value_A - value_B = sum_j w_j * (b_j - a_j).
        ev = float(np.dot(w, b - a))

        tally_a = a_wins > b_wins
        if tally_a:
            # A is the tally winner; keep only if the integrator strongly
            # prefers B (the tally loser).
            if ev > -THRESH:
                continue
        else:
            # B is the tally winner; keep only if the integrator strongly
            # prefers A (the tally loser).
            if ev < THRESH:
                continue

        r = int(round(float(resp)))  # 0 = chose A, 1 = chose B
        chose_tally_winner = (r == 0) if tally_a else (r == 1)
        n_sel += 1
        if chose_tally_winner:
            n_match += 1

    if n_sel == 0:
        return 0.5  # degenerate fallback (no conflict trials found)
    return float(n_match) / float(n_sel)
```

**Observed (real) value:** 0.1625 (var=0.0119)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1862 (var=0.0219) (Δ vs real +0.0237)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0090)
- pi_3: 0.1608 (var=0.0029)
- pi_1: 0.6796 (var=0.0042)
- pi_4: 0.1696 (var=0.0044)
- pi_3_1: 0.1388 (var=0.0265)
- pi_5: 0.1979 (var=0.0276)
- pi_6: 0.1342 (var=0.0135)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Instructed validities of this experiment (fixed at design time).
    VAL = np.array([1.0, 1.0, 0.85, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    n_f = VAL.size

    # Extreme points of each theory's ADMISSIBLE parameter box.
    # SCWI: w(v) = -(v + tau*v^2), tau in [0.25, 0.45].
    W_SC = [-(VAL + t * VAL ** 2) for t in (0.25, 0.45)]
    # PAHI: w(v) = -v^gamma, gamma in [1.4, 2.4].
    W_PA = [-(VAL ** g) for g in (1.4, 2.4)]

    # Minimum |evidence margin| required to call a preference "guaranteed"
    # across the whole parameter box of a theory.
    M_SC = 0.08
    M_PA = 0.05

    def parse_vec(cell):
        if isinstance(cell, str):
            s = cell.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks], dtype=float)
        return np.asarray(cell, dtype=float).ravel()

    def parse_resp(r):
        if isinstance(r, str):
            rs = r.strip().upper()
            if rs == 'A':
                return 0
            if rs == 'B':
                return 1
            return int(float(rs))
        return int(round(float(r)))

    hits = []
    for a_cell, b_cell, r_cell in zip(data['option_a_ratings'],
                                       data['option_b_ratings'],
                                       data['response']):
        try:
            a = parse_vec(a_cell)
            b = parse_vec(b_cell)
            if a.size != n_f or b.size != n_f:
                continue
            d_sc = [float(np.dot(w, a) - np.dot(w, b)) for w in W_SC]
            d_pa = [float(np.dot(w, a) - np.dot(w, b)) for w in W_PA]
            resp = parse_resp(r_cell)
        except Exception:
            continue

        sc_lo, sc_hi = min(d_sc), max(d_sc)
        if sc_lo > M_SC:
            sc_pref = 1      # SCWI prefers A for EVERY admissible tau
        elif sc_hi < -M_SC:
            sc_pref = -1     # SCWI prefers B for EVERY admissible tau
        else:
            continue         # preference not guaranteed -> skip

        pa_lo, pa_hi = min(d_pa), max(d_pa)
        if pa_lo > M_PA:
            pa_pref = 1      # PAHI prefers A for EVERY admissible gamma
        elif pa_hi < -M_PA:
            pa_pref = -1     # PAHI prefers B for EVERY admissible gamma
        else:
            continue

        if sc_pref == pa_pref:
            continue         # theories agree -> no diagnostic content

        # Trial is a guaranteed sign-opposition trial: SCWI's preferred
        # option is exactly the one PAHI rejects (and vice versa).
        scwi_option_is_a = (sc_pref == 1)
        chose_scwi = (resp == 0) if scwi_option_is_a else (resp == 1)
        hits.append(1.0 if chose_scwi else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6837 (var=0.0643)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4556 (var=0.0237) (Δ vs real -0.2281)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5531 (var=0.0069)
- pi_4: 0.2281 (var=0.0125)
- pi_1: 0.8350 (var=0.0129)
- pi_2: 0.1300 (var=0.0103)
- pi_3_1: 0.6506 (var=0.0203)
- pi_5: 0.6019 (var=0.0710)
- pi_6: 0.6150 (var=0.0208)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Top-cue adherence" on steepness-diagnostic ladder cells.
    # The design's validity vector: 4 cues at 1.00, 5 at 0.65, 5 at 0.60,
    # 14 at 0.50 (hardcoded from the experimental design).
    VALID = np.array([1.0] * 4 + [0.65] * 5 + [0.60] * 5 + [0.50] * 14)
    NF = VALID.size

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            return np.array([float(t) for t in s.replace(',', ' ').split()])
        return np.asarray(x, dtype=float).ravel()

    # Excluded cells (signature = (top-cue count, sorted low-cue validities)):
    #  (2, 4x0.50): both theories prefer the low option -> no separation;
    #  (2, 5x0.60) and (2, 5x0.65): PAHI splits across its gamma range on
    #  these cells -> they add between-subject noise without adding gap.
    EXCLUDE = {
        (2, (0.5, 0.5, 0.5, 0.5)),
        (2, (0.6, 0.6, 0.6, 0.6, 0.6)),
        (2, (0.65, 0.65, 0.65, 0.65, 0.65)),
    }

    hits = []
    for a_raw, b_raw, resp in zip(
        data['option_a_ratings'], data['option_b_ratings'], data['response']
    ):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != NF or b.size != NF:
            continue
        ta = int(np.sum(a[:4] > 0.5))
        tb = int(np.sum(b[:4] > 0.5))
        # Exactly one option must carry 1s on the perfect-validity cues.
        if ta > 0 and tb > 0:
            continue
        if ta == 0 and tb == 0:
            continue
        if ta > 0:
            top_vec, low_vec, top_idx, tc = a, b, 0, ta
        else:
            top_vec, low_vec, top_idx, tc = b, a, 1, tb
        if tc not in (2, 4):
            continue
        # Top option must have 1s ONLY on perfect cues; low option none.
        if np.sum(top_vec[4:] > 0.5) > 0:
            continue
        if np.sum(low_vec[:4] > 0.5) > 0:
            continue
        low_mask = low_vec > 0.5
        if int(np.sum(low_mask)) < 4:
            continue
        sig = (tc, tuple(sorted(float(x) for x in np.round(VALID[low_mask], 2))))
        if sig in EXCLUDE:
            continue
        hits.append(1.0 if int(resp) == top_idx else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6300 (var=0.0703)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4742 (var=0.0147) (Δ vs real -0.1558)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2267 (var=0.0083)
- pi_3: 0.5687 (var=0.0053)
- pi_1: 0.8783 (var=0.0092)
- pi_2: 0.1263 (var=0.0072)
- pi_3_1: 0.6646 (var=0.0194)
- pi_5: 0.7004 (var=0.0351)
- pi_6: 0.6208 (var=0.0179)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design-time constants (fixed by this experiment: 29 experts,
    # 6 @ .85, 5 @ .80, 6 @ .75, 12 @ .50).
    V = np.array([0.85] * 6 + [0.80] * 5 + [0.75] * 6 + [0.50] * 12, dtype=float)

    # Reference (midpoint) parameter values. The design guarantees that on
    # every critical cell the SIGN of each theory's evidence is constant
    # over its ENTIRE parameter box, so midpoint values classify trials
    # identically to any admissible parameter.
    TAU = 0.28      # SCWI box [0.15, 0.42]
    GAMMA = 1.9     # PAHI box [1.4, 2.4]
    w_scwi = -(V + TAU * np.square(V))   # SCWI anti-validity weights
    w_pahi = -(np.power(V, GAMMA))       # PAHI anti-validity weights

    def _to_arr(x):
        if isinstance(x, np.ndarray):
            return x.astype(float)
        if isinstance(x, (list, tuple)):
            return np.asarray(x, dtype=float)
        s = str(x).strip().strip('[]()')
        toks = [t for t in s.replace(',', ' ').split() if t != '']
        return np.array([float(t) for t in toks])

    hits = []
    for a_raw, b_raw, r_raw in zip(data['option_a_ratings'],
                                    data['option_b_ratings'],
                                    data['response']):
        try:
            a = _to_arr(a_raw)
            b = _to_arr(b_raw)
            r = int(r_raw)
        except Exception:
            continue
        if r not in (0, 1):
            continue
        n = int(min(a.size, b.size, V.size))
        if n == 0:
            continue
        diff = a[:n] - b[:n]
        d_scwi = float(np.dot(w_scwi[:n], diff))   # value_A - value_B under SCWI
        d_pahi = float(np.dot(w_pahi[:n], diff))   # value_A - value_B under PAHI

        # Keep only trials where the two theories make OPPOSITE predictions
        # with a non-trivial margin on both sides (this automatically selects
        # the knife-edge count-ratio cells and discards controls, ties,
        # anchors, and calibration trials, where the theories agree).
        if abs(d_scwi) <= 0.05 or abs(d_pahi) <= 0.05:
            continue
        if (d_scwi > 0) == (d_pahi > 0):
            continue

        scwi_option = 0 if d_scwi > 0 else 1   # 0 = A, 1 = B
        hits.append(1.0 if r == scwi_option else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6462 (var=0.0395)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5288 (var=0.0201) (Δ vs real -0.1175)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.6975 (var=0.0214)
- pi_4: 0.2169 (var=0.0147)
- pi_1: 0.8762 (var=0.0079)
- pi_2: 0.1344 (var=0.0070)
- pi_3: 0.6412 (var=0.0050)
- pi_5: 0.6056 (var=0.0640)
- pi_6: 0.6581 (var=0.0250)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Kill-composite: fraction of choices falling on the HIGH-validity stack
    # across the parameter-free sign-kill trials of this design.
    #
    # Kill trials pit a endorsements on higher-validity cues against b
    # endorsements on lower-validity cues (all other cues 0 for both
    # options, disjoint supports, each stack within one validity block)
    # at count ratios that lie strictly between the two theories'
    # weight-ratio windows:
    #   5 x 0.90 vs 7 x 0.70   (r = 1.40)
    #   9 x 0.70 vs 14 x 0.50  (r = 1.556)
    #   5 x 0.90 vs 11 x 0.50  (r = 2.20)
    # SCWI's ratio window w(v_h)/w(v_l) is BELOW each r for every admissible
    # tau, so SCWI always prefers the high-validity stack. PAHI's power-law
    # window is ABOVE each r for every admissible gamma, so PAHI always
    # prefers the low-validity stack. The metric is the fraction of kill
    # trials on which the subject chose the high-validity stack; mirrored
    # versions of each kill pair absorb side bias. It is well defined on
    # the pooled DataFrame and on a single subject slice (18 kill trials
    # per subject: 6 unique kill pairs x 3 repetitions).
    import numpy as np

    validities = np.array([0.9] * 5 + [0.7] * 9 + [0.5] * 14)
    n_feat = validities.size

    kill_keys = set()
    kill_keys.add(((5, 0.9), (7, 0.7)))
    kill_keys.add(((9, 0.7), (14, 0.5)))
    kill_keys.add(((5, 0.9), (11, 0.5)))

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = 0
    total = 0

    for i in range(len(a_col)):
        a_cell = a_col[i]
        b_cell = b_col[i]
        resp = r_col[i]
        try:
            a = np.asarray(a_cell, dtype=float).ravel()
            b = np.asarray(b_cell, dtype=float).ravel()
        except Exception:
            continue
        if a.size != n_feat or b.size != n_feat:
            continue

        ones_a = np.flatnonzero(a == 1)
        ones_b = np.flatnonzero(b == 1)
        if ones_a.size == 0 or ones_b.size == 0:
            continue
        # Kill trials have disjoint supports (no tied cues).
        if np.intersect1d(ones_a, ones_b).size > 0:
            continue

        va = validities[ones_a]
        vb = validities[ones_b]
        # Each stack must sit entirely within one validity block.
        if np.unique(va).size != 1 or np.unique(vb).size != 1:
            continue

        sig_a = (int(ones_a.size), float(va[0]))
        sig_b = (int(ones_b.size), float(vb[0]))
        key = tuple(sorted([sig_a, sig_b]))
        if key not in kill_keys:
            continue
        if sig_a[1] == sig_b[1]:
            continue

        # High-validity stack is the one whose cues have larger validity.
        h_is_a = sig_a[1] > sig_b[1]

        try:
            r = int(round(float(resp)))
        except Exception:
            continue
        # response: 0 = chose A, 1 = chose B.
        chose_h = (r == 0) if h_is_a else (r == 1)
        if chose_h:
            hits += 1
        total += 1

    if total == 0:
        return 0.5
    return float(hits) / float(total)
```

**Observed (real) value:** 0.6044 (var=0.0653)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5222 (var=0.0200) (Δ vs real -0.0822)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2078 (var=0.0291)
- pi_3_1: 0.7078 (var=0.0349)
- pi_1: 0.8611 (var=0.0123)
- pi_2: 0.1200 (var=0.0071)
- pi_3: 0.6056 (var=0.0099)
- pi_5: 0.6122 (var=0.0654)
- pi_6: 0.7011 (var=0.0213)

### Experiment 9
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def _vec(x):
        if isinstance(x, np.ndarray):
            a = x.astype(float).ravel()
            if a.size > 0:
                return a
        try:
            a = np.asarray(x, dtype=float).ravel()
            if a.size > 0:
                return a
        except (TypeError, ValueError):
            pass
        s = str(x).strip().strip('[]()')
        toks = [t for t in s.replace(',', ' ').split() if t != '']
        return np.asarray([float(t) for t in toks], dtype=float)

    df = data.reset_index(drop=True)
    if df.shape[0] == 0 or 'response' not in df.columns:
        return 0.0

    A_list = [_vec(v) for v in df['option_a_ratings'].tolist()]
    B_list = [_vec(v) for v in df['option_b_ratings'].tolist()]
    R = np.asarray([float(r) for r in df['response'].tolist()], dtype=float)
    sids = df['subject_id'].tolist()

    rows = []
    for i in range(len(df)):
        a = A_list[i]
        b = B_list[i]
        r = R[i]
        n = a.size
        ta = float(a[:3].sum()); ma = float(a[3:12].sum()); la = float(a[12:].sum())
        tb = float(b[:3].sum()); mb = float(b[3:12].sum()); lb = float(b[12:].sum())
        row = {'anch0': np.nan, 'Lk': np.nan, 'Mk': np.nan, 'cH': np.nan}
        sa = ta + ma + la; sb = tb + mb + lb
        # ANCHOR: one option all-1s, the other all-0s. Record whether the
        # subject picked the all-0 option (engaged anti-validity subjects
        # do this with prob ~1 - eps/2; guessers 0.5).
        if n >= 12 and sa == float(n) and sb == 0.0:
            row['anch0'] = 1.0 if r == 1.0 else 0.0
        elif n >= 12 and sb == float(n) and sa == 0.0:
            row['anch0'] = 1.0 if r == 0.0 else 0.0
        # Top-stack (H) option: exactly the three v=1.00 cues endorsed.
        aH = (n >= 12) and (ta == 3.0) and (ma == 0.0) and (la == 0.0)
        bH = (n >= 12) and (tb == 3.0) and (mb == 0.0) and (lb == 0.0)
        if aH != bH:
            h = 0.0 if aH else 1.0
            opp = b if aH else a
            ot = float(opp[:3].sum()); om = float(opp[3:12].sum()); ol = float(opp[12:].sum())
            if ot == 0.0:
                if om == 0.0 and 8.0 <= ol <= 12.0:
                    # 0.50-ladder kill cell (3x1.00 vs 8-12x0.50)
                    row['Lk'] = 1.0 if r == h else 0.0
                elif ol == 0.0 and 6.0 <= om <= 9.0:
                    # 0.60-ladder kill cell (3x1.00 vs 6-9x0.60)
                    row['Mk'] = 1.0 if r == h else 0.0
                elif om == 0.0 and ol == 5.0:
                    # L-control (3x1.00 vs 5x0.50)
                    row['cH'] = 1.0 if r == h else 0.0
                elif ol == 0.0 and om == 4.0:
                    # M-control (3x1.00 vs 4x0.60)
                    row['cH'] = 1.0 if r == h else 0.0
        rows.append(row)

    tdf = pd.DataFrame(rows)
    tdf['sid'] = sids

    stats = []
    for sid, g in tdf.groupby('sid', sort=False):
        Lv = g['Lk'].dropna()
        Mv = g['Mk'].dropna()
        sL = float(Lv.mean()) if len(Lv) > 0 else 0.5
        sM = float(Mv.mean()) if len(Mv) > 0 else 0.5
        an = g['anch0'].dropna()
        ok_a = (float(an.sum()) == float(len(an))) if len(an) > 0 else True
        cv = g['cH'].dropna()
        ok_c = (float(cv.mean()) <= 0.375) if len(cv) > 0 else True
        stats.append({'sid': sid, 'sL': sL, 'sM': sM, 'keep': bool(ok_a and ok_c)})

    st = pd.DataFrame(stats)
    if st.shape[0] == 0:
        return 0.0
    scr = st[st['keep']]

    # Multi-subject (pooled) case: cross-ladder consistency = Pearson
    # correlation across engaged subjects between the 0.50-ladder and
    # 0.60-ladder top-stack choice rates.
    if scr.shape[0] >= 4:
        x = scr['sL'].to_numpy(dtype=float)
        y = scr['sM'].to_numpy(dtype=float)
        if x.std() > 1e-9 and y.std() > 1e-9:
            with np.errstate(all='ignore'):
                r = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(r):
                return r

    # Single-subject / degenerate case: within-subject cross-ladder
    # agreement, bounded in [-1, 1] and tight under both theories.
    d = np.abs(st['sL'].to_numpy(dtype=float) - st['sM'].to_numpy(dtype=float))
    return float(np.clip(1.0 - 2.0 * float(d.mean()), -1.0, 1.0))
```

**Observed (real) value:** 0.1963 (var=0.0601)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0667 (var=0.0536) (Δ vs real -0.1296)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.3684 (var=0.0180)
- pi_5: 0.9529 (var=0.0191)
- pi_1: 0.9048 (var=0.0084)
- pi_2: 0.8735 (var=0.0093)
- pi_3: 0.2390 (var=0.0208)
- pi_4: 0.8613 (var=0.0179)
- pi_6: 0.1113 (var=0.0229)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell patterns are (top-cue count on H option, mid-cue count on L option,
    # low-cue count on L option), where H = the option whose 1s sit exclusively
    # on the v=1.00 cues and L = the option whose 1s sit exclusively on the
    # v<=0.60 cues.
    # TWIN-KILL cells (r = 1/v_l, the gamma=1 flip point for HEAI):
    #   4x1.00 vs 8x0.50  (trials 3, 9, 13, 19)   and
    #   3x1.00 vs 5x0.60  (trials 7, 10, 17, 20)  -> 32 rows/subject.
    # HIGH-RATIO cells (r = 3.0 on ladder 1, r = 2.0 on ladder 2):
    #   4x1.00 vs 12x0.50 (trials 5, 15)          and
    #   3x1.00 vs 6x0.60  (trials 8, 18)          -> 16 rows/subject.
    TWIN_PATTERNS = {(4.0, 0.0, 8.0), (3.0, 5.0, 0.0)}
    HIGH_PATTERNS = {(4.0, 0.0, 12.0), (3.0, 6.0, 0.0)}

    def _counts(ratings):
        try:
            arr = np.asarray(list(ratings), dtype=float).ravel()
        except Exception:
            return None
        if arr.size != 22:
            return None
        return (float(arr[0:4].sum()), float(arr[4:10].sum()), float(arr[10:22].sum()))

    def _is_H(c):
        # H option: all its 1s are on the four v=1.00 cues.
        return c[0] > 0 and c[1] == 0.0 and c[2] == 0.0

    def _is_L(c):
        # L option: no top-cue 1s, at least one 1 on the v<=0.60 cues.
        return c[0] == 0.0 and (c[1] > 0 or c[2] > 0)

    twin_hits = []
    high_hits = []

    responses = data["response"].values
    a_list = data["option_a_ratings"].values
    b_list = data["option_b_ratings"].values

    for resp, ra, rb in zip(responses, a_list, b_list):
        ca = _counts(ra)
        cb = _counts(rb)
        if ca is None or cb is None:
            continue
        try:
            r = int(float(resp))
        except Exception:
            continue
        if _is_H(ca) and _is_L(cb):
            h_cnt, l_cnt, h_is_a = ca, cb, True
        elif _is_H(cb) and _is_L(ca):
            h_cnt, l_cnt, h_is_a = cb, ca, False
        else:
            continue
        pattern = (float(h_cnt[0]), float(l_cnt[1]), float(l_cnt[2]))
        chose_h = (r == 0) if h_is_a else (r == 1)
        val = 1.0 if chose_h else 0.0
        if pattern in TWIN_PATTERNS:
            twin_hits.append(val)
        elif pattern in HIGH_PATTERNS:
            high_hits.append(val)

    if len(twin_hits) == 0 or len(high_hits) == 0:
        return 0.0

    p_twin = float(np.mean(twin_hits))
    p_high = float(np.mean(high_hits))
    # Signed psychometric rise between the twin-kill ratio region and the
    # high-ratio region. SCWI (step function pinned in [2.14, 2.35] ladder 1
    # and [1.76, 1.89] ladder 2) produces a LARGE rise (~0.55-0.60); HEAI
    # (bimodal gamma, shallow mode straddling the gamma=1 flip) produces a
    # SMALL rise (~0.20).
    return p_high - p_twin
```

**Observed (real) value:** 0.2963 (var=0.0806)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2919 (var=0.0413) (Δ vs real -0.0044)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1713 (var=0.0509)
- pi_3_1: 0.4531 (var=0.0906)
- pi_1: 0.0181 (var=0.0129)
- pi_2: 0.0131 (var=0.0090)
- pi_3: 0.3713 (var=0.0215)
- pi_4: 0.1175 (var=0.0341)
- pi_6: 0.3438 (var=0.0509)

### Experiment 11
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def _to_vec(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks], dtype=float)
        return np.asarray(x, dtype=float).ravel()

    A = np.array([_to_vec(x) for x in data['option_a_ratings']], dtype=float)
    B = np.array([_to_vec(x) for x in data['option_b_ratings']], dtype=float)

    r = np.asarray(data['response'])
    try:
        resp = r.astype(float)
    except Exception:
        resp = np.array([0.0 if str(v).strip().upper() in ('0', 'A') else 1.0 for v in r],
                        dtype=float)

    subj = np.asarray(data['subject_id'])

    n_feat = A.shape[1]
    val = np.array([0.9] * 6 + [0.85] * 11 + [0.5] * 5 + [1.0] * 6, dtype=float)
    if val.size < n_feat:
        val = np.concatenate([val, np.full(n_feat - val.size, 0.5)])
    val = val[:n_feat]

    hi = np.isclose(val, 0.9)
    mid = np.isclose(val, 0.85)

    A_hi = A[:, hi].sum(axis=1)
    A_mid = A[:, mid].sum(axis=1)
    B_hi = B[:, hi].sum(axis=1)
    B_mid = B[:, mid].sum(axis=1)
    A_other = A.sum(axis=1) - A_hi - A_mid
    B_other = B.sum(axis=1) - B_hi - B_mid

    # AL (near-knife, equal-count) cells: one option carries exactly 3 endorsements
    # on the 0.90-validity experts, the other exactly 3 on the 0.85-validity
    # experts, and every other expert is silent for both options.
    al = ((A_other == 0) & (B_other == 0)
          & (((A_hi == 3) & (A_mid == 0) & (B_hi == 0) & (B_mid == 3))
             | ((A_hi == 0) & (A_mid == 3) & (B_hi == 3) & (B_mid == 0))))

    subjects = np.unique(subj)

    if subjects.size > 1:
        # Population statistic: mean signed excess of anti-validity choices on
        # the AL cells (per subject: 2*#anti-choices - #AL-trials, in [-6, +6]).
        excess = []
        for s in subjects:
            m = al & (subj == s)
            k = int(m.sum())
            if k == 0:
                continue
            # anti-validity-preferred option = the one whose 1s sit on the
            # LOWER-validity (0.85) experts (a rating of 1 is a defect).
            anti = np.where(A_mid[m] == 3, 1.0 - resp[m], resp[m])
            excess.append(2.0 * float(anti.sum()) - float(k))
        if len(excess) == 0:
            return 0.0
        return float(np.mean(excess))
    else:
        # Single-subject slice: the population excess has no single-subject
        # analogue, so return the subject's position-bias baseline (mean A
        # response). By the design's mirror-pair symmetry this is pinned near
        # 0.5 with only binomial noise under BOTH theories, i.e. it is the
        # stable per-subject quantity against which the population-level
        # excess is defined.
        return float(np.mean(resp))
```

**Observed (real) value:** 2.8800 (var=0.0015)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 1.1200 (var=0.0012) (Δ vs real -1.7600)
**Other theories' values on this metric (for reference):**
- pi_3_1: 1.4400 (var=0.0007)
- pi_6: 0.3200 (var=0.0016)
- pi_1: -4.3200 (var=0.0012)
- pi_2: 0.2000 (var=0.0010)
- pi_3: 1.0800 (var=0.0009)
- pi_4: 1.7600 (var=0.0014)
- pi_5: 0.7200 (var=0.0008)

### Experiment 12
**Design**
  A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    df = data.reset_index(drop=True)

    # --- Design constants for this experiment --------------------------
    # 12 experts: validities [0.9 x 4, 0.6 x 8].
    VALID = np.array([0.9] * 4 + [0.6] * 8)
    N_F = VALID.size

    # Reference anti-validity defect magnitudes: a rating of 1 is treated
    # as a defect with weight -d_j. The reference ratio d(0.9)/d(0.6)
    # = 1.58 sits between SCWI's whole-box ratio range (1.562-1.651 over
    # tau in [0.15, 0.42]) and CNNAI's median ratio (1.5^(1+0.12) ~ 1.58),
    # so the per-trial signed margins fed to the fit are theory-neutral.
    D_HI, D_LO = 1.0, 0.633
    D_REF = np.where(VALID > 0.75, D_HI, D_LO)

    # Grid for the per-subject sensitivity (softmax inverse temperature)
    # maximum-likelihood fit, and the soft-threshold ramp applied to it.
    # The ramp [3.0, 4.2] targets exactly the region where the two
    # theories' beta distributions are disjoint: SCWI's deterministic
    # regime has beta ~ U[3.8, 6.0] (34% of subjects), while CNNAI's
    # engaged beta is lognormal(median 2.14, log-sd 0.20) clipped to
    # [0.9, 4.5] with 98% of engaged subjects below 3.2.
    GRID = np.linspace(0.0, 10.0, 501)
    RAMP_LO, RAMP_HI = 3.0, 4.2

    def _vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        try:
            arr = np.asarray(raw, dtype=float).ravel()
        except Exception:
            return None
        return arr if arr.size > 0 else None

    def _subject_score(sub):
        ms = []
        ys = []
        for a_cell, b_cell, resp in zip(
            sub['option_a_ratings'].values,
            sub['option_b_ratings'].values,
            sub['response'].values,
        ):
            A = _vec(a_cell)
            B = _vec(b_cell)
            if A is None or B is None or A.size == 0 or A.size != B.size:
                continue
            if A.size == N_F:
                d = D_REF
            else:
                d = np.full(A.size, 0.8)
            w = -d
            va = float(np.dot(w, A))
            vb = float(np.dot(w, B))
            m = abs(vb - va)
            if m < 1e-9 or m > 2.0:
                # identical pairs (ties) and the all-ones-vs-all-zeros
                # anchor screens carry no usable sensitivity info
                # (both theories put them at ~0.5 / lapse rate), and the
                # ~9-unit anchor margin would corrupt the fit.
                continue
            b_pred = vb > va
            r = float(resp)
            y = 1.0 if ((r > 0.5) == b_pred) else 0.0
            ms.append(m)
            ys.append(y)
        if len(ms) < 10:
            return 0.0
        m = np.asarray(ms, dtype=float)            # (n_trials,)
        y = np.asarray(ys, dtype=float)[:, None]   # (n_trials, 1)  <-- broadcast fix
        z = np.outer(m, GRID)                      # (n_trials, n_grid)
        # Bernoulli log-likelihood of P(y=1) = sigmoid(beta * margin):
        # ll(beta) = -sum_i [ y_i * log(1+e^{-z}) + (1-y_i) * log(1+e^{z}) ]
        ll = -(y * np.logaddexp(0.0, -z)
               + (1.0 - y) * np.logaddexp(0.0, z)).sum(axis=0)
        bhat = float(GRID[int(np.argmax(ll))])
        g = (bhat - RAMP_LO) / (RAMP_HI - RAMP_LO)
        return float(min(max(g, 0.0), 1.0))

    subjects = pd.unique(df['subject_id'])
    if len(subjects) == 0:
        return 0.0
    vals = []
    for sid in subjects:
        sub = df[df['subject_id'] == sid]
        vals.append(_subject_score(sub))
    return float(np.mean(vals))
```

**Observed (real) value:** 0.0913 (var=0.0729)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0200 (var=0.0196) (Δ vs real -0.0713)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0000 (var=0.0000)
- pi_3_1: 0.3587 (var=0.2287)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0823 (var=0.0589)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Compressive Anti-Validity Integration with a Moderate Deterministic Tail (CAVI). Subjects integrate all binary expert ratings additively into a per-option subjective DEFECT LOAD, comprehending a rating of 1 as a defect (sign inversion of the rating direction). The per-cue defect magnitude is supralinear in the instructed validity, d_j = v_j + tau*v_j^2 with tau in [0.35, 0.65], further modulated by a context elasticity: within a coarse validity context (the mean endorsed validity rounded to 0.1), cue j's defect is scaled by (v_j / v_ref)^eta, where eta has a small stable per-subject component and a larger per-context redraw (latent cross-context consistency r ~ 0.20 by construction). The option's subjective defect load is then COMPRESSED: D = (sum_j d_j*(v_j/v_ref)^eta * x_j)^kappa with kappa in [0.55, 0.85], so small stacks keep near-linear sharp margins while large stacks of endorsements grow sub-proportionally aversive; the option value is -D. Choice is a softmax over beta*(D_B - D_A) with a symmetric lapse, per-trial multiplicative attention diffusion (sd 0.05), and a three-part population: ~14% true guessers (beta 0.02-0.10), ~70% engaged (lognormal beta, median ~2.0, log-sd 0.35, clipped [0.8, 6.0]), and ~16% deterministic tail (beta uniform [4.0, 6.5], lapse 0.02-0.04, tau drawn from the upper sub-range and kappa near-linear, i.e. the confidently inverted subjects are also the least compressive). This single mechanism simultaneously yields the anti-validity signatures (negative Exp-1 slope, positive Exp-2/3 dissociations), the near-knife kill-cell composites of Exps 5-8, the flattened high-ratio psychometric rise of Exp 10 (compression pulls both twin-kill and high-ratio cells toward the noise scale), the moderate cross-ladder consistency of Exp 9 (context-keyed eta redraws), a small nonzero deterministic bloc for Exp 12's g, and a stronger near-knife AL excess on Exp 11 (high tau, low compression, hot tail).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Compressive Anti-Validity Integration with a Moderate Deterministic
    # Tail (CAVI).
    #
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Mechanism:
    #   1. SIGN INVERSION: a rating of 1 is comprehended as a DEFECT. The
    #      per-cue defect magnitude is supralinear in the instructed
    #      validity: d_j = v_j + tau * v_j^2, tau in [0.35, 0.65].
    #   2. CONTEXT ELASTICITY: within a coarse validity context (mean
    #      endorsed validity rounded to 0.1), cue j's defect is scaled by
    #      (v_j / v_ref)^eta. eta = stable per-subject component (sd 0.10)
    #      + per-context redraw (sd 0.20), so the latent cross-context
    #      consistency is r = 0.01/(0.01+0.04) ~ 0.20 by construction.
    #      The context key is the bucketed MEAN endorsed validity (not the
    #      max), so different validity ladders (e.g. the 0.50- and
    #      0.60-ladders of the cross-ladder experiment) draw independent
    #      etas while cells of one ladder share one draw.
    #   3. WITHIN-OPTION DIMINISHING SENSITIVITY: the option's subjective
    #      defect load is D = S^kappa with kappa in [0.55, 0.85], where
    #      S = sum_j d_j * (v_j/v_ref)^eta * x_j. Small stacks keep
    #      near-linear sharp margins; large stacks of endorsements grow
    #      sub-proportionally aversive. Option value = -D.
    #   4. POPULATION MIXTURE: ~14% true guessers (beta 0.02-0.10), ~70%
    #      engaged (lognormal beta, median 2.0, log-sd 0.35, clipped
    #      [0.8, 6.0], lapse 0.03-0.09), ~16% deterministic tail
    #      (beta uniform [4.0, 6.5], lapse 0.02-0.04). The tail also draws
    #      tau from the upper sub-range [0.55, 0.65] and kappa near-linear
    #      [0.78, 0.85]: the confidently inverted subjects comprehend the
    #      inversion most strongly and compress least.
    #   5. Choice = softmax(beta * values) with symmetric lapse and a
    #      per-trial multiplicative attention diffusion (sd 0.05) on the
    #      cue defect weights.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CAVI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    ts = _seed("tau_seed")
    ks = _seed("kappa_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    es = _seed("epsilon_seed")
    ebs = min(max(_seed("eta_base_seed"), 1e-6), 1.0 - 1e-6)
    ss = _seed("subject_seed")

    # --- Population mixture ---
    if u < 0.14:
        # True guessers (~14%): beta ~ 0 makes the softmax core nearly
        # uniform on ordinary trials; elevated lapse.
        t = u / 0.14
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * es              # 0.12 .. 0.25
        tau = 0.50
        kappa = 0.75
        eta_base = 0.00
    elif u < 0.84:
        # Engaged bulk (~70%).
        tau = 0.35 + 0.30 * ts                 # 0.35 .. 0.65
        kappa = 0.62 + 0.23 * ks                # 0.62 .. 0.85
        beta = math.exp(math.log(2.0) + 0.35 * float(ndtri(bs)))
        beta = min(max(beta, 0.8), 6.0)
        epsilon = 0.03 + 0.06 * es             # 0.03 .. 0.09
        eta_base = 0.06 + 0.10 * float(ndtri(ebs))
        eta_base = min(max(eta_base, -0.15), 0.25)
    else:
        # Deterministic tail (~16%): hot, low-lapse, strongly inverted and
        # least compressive.
        tau = 0.55 + 0.10 * ts                 # 0.55 .. 0.65
        kappa = 0.78 + 0.07 * ks                # 0.78 .. 0.85
        beta = 4.0 + 2.5 * bs                   # 4.0 .. 6.5
        epsilon = 0.02 + 0.02 * es              # 0.02 .. 0.04
        eta_base = 0.06 + 0.10 * float(ndtri(ebs))
        eta_base = min(max(eta_base, -0.15), 0.25)

    # --- Context: bucketed mean endorsed validity ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw centered low (total engaged eta ~ N(0.12, 0.22)),
    # keeping the engaged bulk below the kill-design flip knives while the
    # stable/redraw variance ratio fixes latent consistency at r ~ 0.20.
    eta_ctx = eta_base + 0.06 + 0.20 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.30), 0.60)

    # --- Defect weights: supralinear anti-validity, context-normalized ---
    d = val + tau * np.square(val)
    w = d * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion (sd 0.05).
    att = np.exp(0.05 * np.random.randn(n_features))
    w = w * att

    # Raw defect loads, then within-option diminishing sensitivity.
    s_a = float(np.dot(w, ra))
    s_b = float(np.dot(w, rb))
    d_a = s_a ** kappa if s_a > 0.0 else 0.0
    d_b = s_b ** kappa if s_b > 0.0 else 0.0

    # Option values are the (negative) compressed defect loads.
    scores = np.array([-d_a, -d_b])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- regime_seed: [0, 1]
- tau_seed: [0, 1]
- kappa_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- eta_base_seed: [0, 1]
- subject_seed: [0, 1]
- validities: validities

`rationale`: CAVI implements the arbiter's prescribed mechanism family, engineered as a stronger competitor to CNNAI rather than a return of SCWI. (1) SIGN INVERSION + SUPRALINEAR ANTI-VALIDITY: d_j = v_j + tau*v_j^2 with tau in [0.35, 0.65] (wider and higher than SCWI's [0.25, 0.45]) directly amplifies the near-knife 0.90-vs-0.85 discrimination that Exp 11's AL cells demand, while the population spread of tau keeps the Exp-8 kill cells (whose 5x0.90-vs-7x0.70 flip knife sits at tau ~ 0.645) straddled rather than crossed, holding that composite near the real 0.60. (2) WITHIN-OPTION COMPRESSION D = S^kappa: this is the decisive new lever. It flattens exactly the high-ratio psychometrics that every prior theory overshot — Exp 10's rise (SCWI 0.37, CNNAI 0.34, real 0.296) drops toward ~0.29 because both twin-kill and high-ratio margins are pulled toward the noise scale, and Exp 8's overshoot-prone cells soften toward 0.60 — while small stacks keep near-linear margins, so the anti-validity signatures of Exps 1-4 survive. Working backwards from the Exp-12 g metric (real 0.091, which every prior theory missed at 0.000 except pi_5): the per-subject MLE beta fit rescales by the local margin slope kappa*S^(kappa-1)*margin-ratio ~ 0.75, so the 16% deterministic tail (beta 4.0-6.5, kappa near-linear 0.78-0.85, lapse 0.02-0.04) lands fitted bhat in the ramp region [3.0, 4.2], producing g ~ 0.06-0.10 — the small-but-nonzero near-ceiling bloc that refutes both CNNAI's 0.00 and SCWI-style overshoots. (3) POPULATION MIXTURE: 14% guessers / 70% engaged (lognormal beta median 2.0, hotter and tighter than SCWI's uniform 1.1-1.7) / 16% tail. The hotter engaged beta lifts the undershooting adherence composites of Exps 5-7 (SCWI's 0.553 on Exp 5 vs real 0.684) toward 0.63-0.66, and sharpens the Exp-2 disagreement-vs-agreement dissociation toward the real 0.274 (agree trials carry large anti-validity margins -> strong anti-follow; disagree trials sit near the compressed noise scale -> near-chance following). The tail's tau is drawn from the upper sub-range because the confidently inverted subjects are also the ones who read the validity instructions hardest — this is what pushes the Exp-11 AL excess up (tail subjects hit ~0.74 anti-choice on AL cells) without overheating the engaged bulk. (4) CONTEXT ELASTICITY keyed on the bucketed MEAN endorsed validity (not the max, which would re-couple the Exp-9 ladders that both share max validity 1.0): stable sd 0.10 vs redraw sd 0.20 fixes latent cross-context consistency at r ~ 0.20, matching the observed 0.196 between SCWI's 0.37 and CNNAI's 0.11, and the low center (total ~N(0.12, 0.22)) keeps the engaged bulk below the Exp-8/10 flip knives. (5) Per-trial attention diffusion (sd 0.05) supplies within-context stochasticity that averages out within subjects, softening near-knife cells without adding between-subject variance. Net expected profile: Exp 1 ~ -0.19, Exp 2 ~ 0.22, Exp 3 ~ 0.13, Exp 4 ~ 0.16, Exp 5 ~ 0.64, Exp 6 ~ 0.64, Exp 7 ~ 0.66, Exp 8 ~ 0.59, Exp 9 ~ 0.20, Exp 10 ~ 0.29, Exp 11 ~ 1.2-1.5 (vs SCWI's 1.08 and CNNAI's 0.32), Exp 12 ~ 0.06-0.09 (vs 0.000 for both) — improving on the running best simultaneously on Exps 2, 5, 6, 11 and 12 while holding or improving every other experiment, which should clear the accept gate.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: CAVI is a faithful implementation of the arbiter's prescribed family (sign-inverted supralinear defect weights, within-option compression, three-part population mixture, context-keyed elasticity), and it clears the gate. The fit is genuinely excellent on the psychometric-shape metrics: Exp 1 (-0.180 vs -0.175), Exp 3 (0.140 vs 0.123), Exp 4 (0.186 vs 0.163), and — the round's headline win — Exp 10's rise (0.292 vs 0.296, where every prior theory overshot or undershot). The compression lever works exactly as intended on the high-ratio cells. However, the same compression is doing collateral damage everywhere margins are SMALL-to-MODERATE, and the candidate systematically undershoots the anti-validity adherence bloc: Exp 5 (0.456 vs 0.684), Exp 6 (0.474 vs 0.630), Exp 7 (0.529 vs 0.646), Exp 8 (0.522 vs 0.604, just below the arbiter's [0.55, 0.65] window), Exp 9 (0.067 vs 0.196, below the [0.15, 0.30] window), Exp 11 (1.12 vs 2.88 — far below the arbiter's [+2.4, +3.2] target, and even below the proposer's own 1.2-1.5 prediction), and Exp 12 (0.020 vs 0.091, below the [0.06, 0.15] window). The diagnosis is mechanical, not conceptual: a pure power map D = S^kappa compresses margins PROPORTIONALLY at every scale. For S > 1, the local margin slope is kappa*S^(kappa-1) < 1, so even 'small' stacks like Exp 11's AL cells (S ≈ 3.6-3.9, margin ≈ 0.28 uncompressed) get their margin roughly halved (≈ 0.14) — the theory's claim that 'small stacks keep near-linear sharp margins' is false for the power form whenever S > 1. The same halving softens the Exp 5-7 adherence cells and deflates the Exp 12 per-subject MLE beta fits (the metric fits a LINEAR-margin softmax, so the fitted bhat is the true beta scaled down by the compression slope, pushing the 16% deterministic tail below the [3.0, 4.2] ramp). Separately, the Exp 9 cross-ladder correlation (0.067) falls short of the by-construction r ≈ 0.20 because the eta-space consistency does not transfer linearly to choice-rate space: near-saturated per-ladder choice rates compress the correlation, so the stable/redraw variance split needs to be richer in the stable component than 0.20 to land at an observed 0.196.
Rationale: The candidate was accepted, so build on it — but it is not shippable yet: it misses the arbiter's explicit target windows on four experiments (8, 9, 11, 12) and undershoots real data by 0.12-0.23 on Exps 5-7. All failures point at ONE miscalibration, so the fix is a minor in-family adjustment, not a redesign: (1) Replace the pure power compression with an OFFSET/affine form that is near-identity at small S and compressive only at large S — e.g. D = (1 + S)^kappa - 1, or a two-regime map D = S for S < s0 and s0*(S/s0)^kappa for S >= s0 with s0 ≈ 2-3. This preserves the Exp 10 win (high-ratio cells have S ≈ 5-12 and stay compressed) while restoring near-linear margins on the AL cells (Exp 11 should rise from 1.12 toward the 2.4-2.9 range), the Exp 5-7 adherence composites (toward 0.60-0.68), and Exp 8's kill composite (into [0.55, 0.65]). (2) For Exp 12's g: the deterministic tail's effective slope is beta * kappa*S^(kappa-1); with the offset compression the tail's fitted bhat recovers, but also push the tail's kappa to [0.92, 1.0] (fully linear for the confidently inverted) and consider widening tail beta to [4.5, 7.5] so a larger share of tail subjects land bhat inside the [3.0, 4.2] ramp; target g ≈ 0.06-0.10. (3) For Exp 9: raise the stable share of eta — e.g. stable sd 0.15 with redraw sd 0.15 (eta-space r ≈ 0.5) — so that after choice-rate saturation the observed cross-ladder correlation lands near 0.20; verify the bucketed-mean key still separates the 0.50- and 0.60-ladders (it does: buckets ≈ 0.6 vs 0.7). (4) Keep everything else fixed: the supralinear tau range, the 14/70/16 mixture proportions, the lapse ranges, the attention diffusion, and the Exp 1-4 parameterization are all performing at or near the real values and should not be touched. Expected net effect: Exp 11 → ~2.4-2.9, Exps 5-7 → ~0.60-0.68, Exp 8 → ~0.57-0.62, Exp 9 → ~0.18-0.22, Exp 12 → ~0.06-0.09, with Exps 1-4 and 10 held (re-check Exp 10 after the compression change: the offset form slightly REDUCES compression at the twin-kill cells (S ≈ 4-5), which may nudge the rise up from 0.29 — if it drifts above 0.36, lower s0 or kappa slightly to compensate).

**Outcome of this advice:** iter 1 candidate loss=0.1319 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Compressive Anti-Validity Integration with a Moderate Deterministic Tail (CAVI, iter 2). Subjects integrate all binary expert ratings into a per-option subjective DEFECT LOAD, comprehending a rating of 1 as a defect (sign inversion). The per-cue defect magnitude is supralinear in the instructed validity, d_j = v_j + tau*v_j^2, modulated by a context elasticity (v_j/v_ref)^eta keyed on the bucketed mean endorsed validity. The within-option diminishing sensitivity is now an OFFSET (two-regime) map rather than a pure power: D = S for S < s0 (s0 = 3.0) and D = s0*(S/s0)^kappa for S >= s0 — near-identity (linear, sharp margins) at small-to-moderate stacks, compressive only for large endorsement stacks. Choice is softmax(beta*(D_B - D_A)) with symmetric lapse, per-trial attention diffusion (sd 0.05), and a three-part population: ~14% true guessers, ~70% engaged (lognormal beta, median ~2.0), and ~16% deterministic tail that is now nearly linear (kappa 0.92-1.0) and hotter (beta 4.5-7.5) — the confidently inverted subjects compress least, which both sharpens near-knife anti-validity adherence and lands their linear-margin fitted sensitivity inside the near-ceiling ramp.",
  "predict": "def predict(parameters, state, history):\n    # Compressive Anti-Validity Integration with a Moderate Deterministic\n    # Tail (CAVI), iter 2 -- offset-compression recalibration per the\n    # iter-1 critic diagnosis. Mechanism family UNCHANGED (sign-inverted\n    # supralinear anti-validity defect weights, context-keyed elasticity,\n    # within-option diminishing sensitivity, three-part population).\n    # Changes vs the accepted iter-1 base:\n    #   (A) PURE POWER COMPRESSION -> TWO-REGIME OFFSET MAP:\n    #       D = S for S < s0 (s0 = 3.0) and D = s0*(S/s0)^kappa for\n    #       S >= s0. The pure power form compressed margins\n    #       PROPORTIONALLY at every scale (local slope\n    #       kappa*S^(kappa-1) < 1 even at the AL cells' S ~ 3.6-3.9),\n    #       roughly halving exactly the small-to-moderate margins that\n    #       Exps 5-8 and 11 need. The offset map is the IDENTITY below\n    #       s0 and compressive only for large stacks, so near-knife and\n    #       adherence cells recover their linear sharpness while the\n    #       high-ratio cells (S ~ 5-12) stay compressed (Exp 10's rise\n    #       is preserved).\n    #   (B) DETERMINISTIC TAIL MADE NEARLY LINEAR AND HOTTER:\n    #       kappa 0.78-0.85 -> 0.92-1.0, beta [4.0, 6.5] -> [4.5, 7.5].\n    #       The Exp-12 g metric fits a LINEAR-margin softmax, so the\n    #       tail's fitted bhat ~ beta * local compression slope; with\n    #       kappa ~ 1 the slope is ~0.9-1.0 and a larger share of the\n    #       tail lands inside the [3.0, 4.2] ramp, producing\n    #       g ~ 0.06-0.15 (real 0.091) instead of 0.02.\n    #   (C) STABLE ETA SHARE RICHENED: eta_base sd 0.10 -> 0.15 with the\n    #       per-context redraw sd cut 0.20 -> 0.15 (eta-space latent\n    #       consistency r ~ 0.5). The iter-1 r ~ 0.20 in eta space\n    #       compressed to an observed cross-ladder correlation of only\n    #       0.067 because near-saturated per-ladder choice rates shrink\n    #       the correlation; the richer stable component restores the\n    #       observed Exp-9 value toward ~0.2. The bucketed-mean context\n    #       key is unchanged (it still separates the 0.50- and\n    #       0.60-ladders: buckets ~0.6 vs ~0.7).\n    #   (D) Everything else -- tau range [0.35, 0.65], the 14/70/16\n    #       mixture, lapse ranges, attention diffusion (sd 0.05), and\n    #       the softmax -- is untouched, per the critic's instruction.\n    import hashlib\n    import math\n    import numpy as np\n    from scipy.special import ndtri\n\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"CAVI expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    n_features = stim.shape[1]\n    ra, rb = stim[0], stim[1]\n\n    def _to_vec(raw):\n        if raw is None:\n            return None\n        if isinstance(raw, str):\n            s = raw.strip().strip('[]()')\n            toks = [t for t in s.replace(',', ' ').split() if t != '']\n            if not toks:\n                return None\n            return np.array([float(t) for t in toks])\n        arr = np.asarray(raw, dtype=float).ravel()\n        return arr if arr.size > 0 else None\n\n    val = _to_vec(parameters.get(\"validities\"))\n    if val is None or val.size != n_features:\n        # Defensive fallback: uniform mid-range validities.\n        val = np.full(n_features, 0.75)\n    val = np.clip(val, 1e-3, 1.0)\n\n    def _seed(name):\n        try:\n            x = float(parameters[name])\n        except (KeyError, TypeError, ValueError):\n            x = 0.5\n        if not np.isfinite(x):\n            x = 0.5\n        return min(max(x, 0.0), 1.0)\n\n    u = _seed(\"regime_seed\")\n    ts = _seed(\"tau_seed\")\n    ks = _seed(\"kappa_seed\")\n    bs = min(max(_seed(\"beta_seed\"), 1e-6), 1.0 - 1e-6)\n    es = _seed(\"epsilon_seed\")\n    ebs = min(max(_seed(\"eta_base_seed\"), 1e-6), 1.0 - 1e-6)\n    ss = _seed(\"subject_seed\")\n\n    # --- Population mixture ---\n    if u < 0.14:\n        # True guessers (~14%): beta ~ 0 makes the softmax core nearly\n        # uniform on ordinary trials; elevated lapse.\n        t = u / 0.14\n        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10\n        epsilon = 0.12 + 0.13 * es              # 0.12 .. 0.25\n        tau = 0.50\n        kappa = 0.75\n        eta_base = 0.00\n    elif u < 0.84:\n        # Engaged bulk (~70%).\n        tau = 0.35 + 0.30 * ts                 # 0.35 .. 0.65\n        kappa = 0.62 + 0.23 * ks                # 0.62 .. 0.85\n        beta = math.exp(math.log(2.0) + 0.35 * float(ndtri(bs)))\n        beta = min(max(beta, 0.8), 6.0)\n        epsilon = 0.03 + 0.06 * es             # 0.03 .. 0.09\n        # Richer stable component (sd 0.15): after choice-rate\n        # saturation the observed cross-ladder correlation lands near\n        # the real ~0.20 (iter-1's sd 0.10 delivered only 0.067).\n        eta_base = 0.06 + 0.15 * float(ndtri(ebs))\n        eta_base = min(max(eta_base, -0.25), 0.35)\n    else:\n        # Deterministic tail (~16%): hot, low-lapse, strongly inverted\n        # and now NEARLY LINEAR (kappa 0.92-1.0) -- the confidently\n        # inverted subjects compress least, so their linear-margin\n        # fitted sensitivity (Exp 12) recovers into the [3.0, 4.2]\n        # ramp and their near-knife adherence (Exps 5-8, 11) sharpens.\n        tau = 0.55 + 0.10 * ts                 # 0.55 .. 0.65\n        kappa = 0.92 + 0.08 * ks                # 0.92 .. 1.00\n        beta = 4.5 + 3.0 * bs                   # 4.5 .. 7.5\n        epsilon = 0.02 + 0.02 * es              # 0.02 .. 0.04\n        eta_base = 0.06 + 0.15 * float(ndtri(ebs))\n        eta_base = min(max(eta_base, -0.25), 0.35)\n\n    # --- Context: bucketed mean endorsed validity ---\n    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))\n    if active.size == 0:\n        # Nothing endorsed by either option: no evidence either way.\n        return np.full(2, 0.5)\n\n    v_ref = float(np.mean(val[active]))\n    bucket = float(np.round(v_ref, 1))\n    key = f\"{bucket:.1f}\"\n    h = int(hashlib.md5(f\"{ss:.9f}|{key}\".encode(\"utf-8\")).hexdigest()[:14], 16)\n    uc = ((h % 999983) + 0.5) / 999984.0\n    uc = min(max(uc, 1e-6), 1.0 - 1e-6)\n\n    # Per-context redraw (sd cut 0.20 -> 0.15): total engaged eta still\n    # ~ N(0.12, 0.21), centered below the kill-design flip knives, but\n    # the stable/redraw variance ratio now fixes latent consistency at\n    # r ~ 0.5 in eta space, which after saturation yields an observed\n    # cross-ladder correlation near the real 0.196.\n    eta_ctx = eta_base + 0.06 + 0.15 * float(ndtri(uc))\n    eta_ctx = min(max(eta_ctx, -0.30), 0.60)\n\n    # --- Defect weights: supralinear anti-validity, context-normalized ---\n    d = val + tau * np.square(val)\n    w = d * np.power(val / v_ref, eta_ctx)\n\n    # Per-trial multiplicative attention diffusion (sd 0.05).\n    att = np.exp(0.05 * np.random.randn(n_features))\n    w = w * att\n\n    # Raw defect loads.\n    s_a = float(np.dot(w, ra))\n    s_b = float(np.dot(w, rb))\n\n    # --- OFFSET (two-regime) diminishing sensitivity ---\n    # Identity below s0 (small stacks keep their full linear margin);\n    # compressive power law above s0 (large stacks grow sub-\n    # proportionally aversive). Continuous at s0.\n    S0 = 3.0\n\n    def _compress(s):\n        if s <= 0.0:\n            return 0.0\n        if s < S0:\n            return s\n        return S0 * (s / S0) ** kappa\n\n    d_a = _compress(s_a)\n    d_b = _compress(s_b)\n\n    # Option values are the (negative) compressed defect loads.\n    scores = np.array([-d_a, -d_b])\n\n    # Numerically stable softmax over the two option values.\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n\n    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # Guard against float drift.\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "regime_seed": "[0, 1]",
    "tau_seed": "[0, 1]",
    "kappa_seed": "[0, 1]",
    "beta_seed": "[0, 1]",
    "epsilon_seed": "[0, 1]",
    "eta_base_seed": "[0, 1]",
    "subject_seed": "[0, 1]",
    "validities": "validities"
  },
  "rationale": "This is the critic-prescribed minimal-diff recalibration of the ACCEPTED iter-1 CAVI base; the mechanism family, parameter names, and everything the critic said to keep (tau range [0.35, 0.65], the 14/70/16 mixture, lapse ranges, attention diffusion sd 0.05, the bucketed-mean context key) are untouched. Four edits: (1) OFFSET COMPRESSION. The iter-1 diagnosis was exactly right: a pure power map D = S^kappa has local slope kappa*S^(kappa-1) < 1 at EVERY scale above 1, so even 'small' stacks like the AL cells (S ~ 3.6-3.9) had their ~0.31 margin roughly halved, and the same halving softened the Exps 5-8 adherence composites and deflated the Exp-12 linear-margin MLE fits. The two-regime map D = S for S < s0 = 3.0, s0*(S/s0)^kappa above, is the identity at small-to-moderate S: the AL margin stays ~0.30 for the tail and ~0.20 (slope ~0.65) for the engaged bulk, the Exp 5 cells (e.g. 2x1.00 vs 5x0.50, margin 0.75) recover sigmoid(beta*0.5) ~ 0.73 adherence, and the Exp 8 kill cells (margins 0.09-0.35 at tau below the 0.645 knife) sharpen toward ~0.57-0.60 — while the Exp-10 high-ratio cells (S ~ 5-12) remain compressed, holding the round's headline win (estimated rise ~0.30 vs real 0.296; the twin-kill ladder-1 cell sits at the tau = 0.5 knife where its margin is ~0 either way). (2) TAIL NEARLY LINEAR AND HOTTER: kappa [0.92, 1.0] and beta [4.5, 7.5] mean the tail's effective linear-margin slope is beta*~0.95, so its Exp-12 fitted bhat lands in the [3.0, 4.2] ramp (g ~ 0.10-0.15 vs the iter-1 0.020 and real 0.091), and its near-knife anti-validity choices sharpen (AL anti-rate ~0.85, Exp 5-8 adherence ~0.9), lifting the Exps 5-8 composites and the Exp-11 excess (from 1.12 toward ~1.6-1.9 — still short of the real 2.88, but the largest in-family gain available without distorting the well-fit Exps 1-4). (3) STABLE ETA SHARE DOUBLED (sd 0.15 stable vs 0.15 redraw, latent r ~ 0.5): the iter-1 observed cross-ladder correlation (0.067) fell far below the by-construction 0.20 because near-saturated per-ladder choice rates compress eta-space consistency into choice space; the richer stable component (plus the tail's stable hot beta/tau, which are also cross-ladder traits) restores the observed Exp-9 correlation toward the real 0.196 while the unchanged bucketed-mean key keeps the 0.50- and 0.60-ladders on independent redraws (buckets 0.6 vs 0.7). (4) No other changes. Expected net profile: Exps 1-4 essentially held (Exp 1 ~ -0.18 to -0.19, Exp 3 ~ 0.14, Exp 4 improves toward ~0.17 since sharper anti-validity subjects follow the tally winner less on conflict trials, Exp 2 improves toward ~0.24 as restored agree-trial margins strengthen the anti-follow), Exp 10 held near 0.29-0.33, and simultaneous gains on Exps 5 (0.46 -> ~0.68), 6 (0.47 -> ~0.60), 7 (0.53 -> ~0.63), 8 (0.52 -> ~0.58), 9 (0.07 -> ~0.15-0.25), 11 (1.12 -> ~1.6-1.9), and 12 (0.02 -> ~0.10-0.14) — a strict aggregate improvement over the iter-1 loss of 0.1319, which should clear the accept gate."
}
```
```

## Usage

```json
{
  "input_tokens": 62912,
  "output_tokens": 25854,
  "reasoning_tokens": 21803,
  "provider": "Sail Research"
}
```
