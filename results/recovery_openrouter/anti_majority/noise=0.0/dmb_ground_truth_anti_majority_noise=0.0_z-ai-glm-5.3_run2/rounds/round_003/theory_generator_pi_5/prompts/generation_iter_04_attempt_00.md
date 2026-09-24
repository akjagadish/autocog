# generation_iter_04_attempt_00

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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace PAHI with a new theory, e.g. 'Heterogeneous-Elasticity Anti-Validity Integration (HEAI)': subjects integrate all cues additively with a comprehension inversion (rating of 1 = defect), per-cue anti-weight w_j = -v_j^gamma, where gamma is a free per-subject parameter drawn from a broad, mildly sublinear-to-linear-centered population distribution (approximately gamma ~ 1.15, SD ~ 0.25, truncated to [0.7, 1.7]) — critically with a population mean elasticity BELOW 1.4, outside PAHI's falsified box. Add (i) a guesser subpopulation (~15-20% of subjects with beta ~ 0 and elevated lapse) as in SCWI, (ii) a continuum of inverse temperatures beta spanning ~0.5-6 with a heavy right tail, and (iii) a symmetric lapse epsilon in [0.03, 0.25]. This combination reproduces: kill composites of ~0.60-0.66 on Experiments 1/2/7/8 (majority of subjects with gamma < ~1.35 follow the high-validity stack, guessers dilute toward 0.5, steep-gamma tail dilutes slightly below SCWI's 0.65-0.71 — matching the observed 0.60-0.68 and the larger observed variances); the Exp 4 dissociation near 0.27 (heterogeneous beta/gamma produces the observed large between-subject variance of ~0.09 that SCWI's three-regime mixture underpredicts); and the Exp 3/5/6 slopes and conflict rates near the observed values. The new theory differs from PAHI primarily by relocating the population elasticity from the falsified [1.4, 2.4] range to the empirically supported near-linear range, and differs from SCWI by replacing the fixed three-regime (guesser/mid/deterministic) mixture with a continuous per-subject heterogeneity in the weight-map shape itself, which is the lever needed to capture the residual gaps (overpredicted kill composites, underpredicted Exp 4 dissociation and between-subject variance) without disturbing the signs that SCWI gets right.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.920)

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


---

### `pi_3_1` (overall score: 0.897)

**Description**
Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI) -- corrected three-regime
    # population, iteration 7 (minimal-diff on the accepted iter-6 base).
    # Mechanism is UNCHANGED: subjective validity of cue j = a_j * v_j
    # (per-cue attention factor, fixed for the whole run), signed cue
    # weight w_j = -(v'_j + tau * v'_j^2) (a rating of 1 is a defect; the
    # anti-weight grows SUPRALINEARLY in the subjective instructed
    # validity), softmax over beta * values plus a symmetric lapse
    # epsilon. History is ignored (no trial-by-trial feedback).
    #
    # Population edits on top of the accepted iter-6 base (mechanism
    # untouched), per the iter-6 critic diagnosis that the remaining
    # gaps are pure mixture-calibration problems:
    #   * FRACTIONS 13/57/30 -> 16/50/34: the mid regime (57%) was the
    #     underperformer on Exp 5 and contributes almost nothing on the
    #     Exp-4 Probe-B conflict cells, so its mass is redistributed to
    #     BOTH tails -- the guesser tail lifts Exp 4 (guessers contribute
    #     0.5 on conflict cells vs mid's ~0.10) and the deterministic
    #     tail lifts Exp 5/6 (det ~0.9 vs mid's ~0.53).
    #   * MID BETA raised a small step, 1.6-2.2 -> 1.8-2.4 (NOT a swing
    #     back to iter-5's 1.8-2.8, which failed without real guessers;
    #     with 16% pinned at 0.5 the Exp-4 floor is higher).
    #   * WITHIN-REGIME SPREAD WIDENED for between-subject variance:
    #     deterministic beta 4.2-5.6 -> 3.8-6.0, mid epsilon 0.06-0.10
    #     -> 0.05-0.11, guesser epsilon 0.15-0.25 -> 0.12-0.28. These
    #     widen the per-subject hit-rate distribution without materially
    #     moving the population means.
    #   * Attention jitter stays exactly at the validated (0.90, 1.10);
    #     tau stays at [0.15, 0.42]; the beta~0 guesser implementation
    #     (the validated decoupling lever) is untouched.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
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

    att = _to_vec(parameters.get("attention"))
    if att is None or att.size != n_features:
        # Defensive fallback: no attention distortion.
        att = np.ones(n_features)
    att = np.clip(att, 0.6, 1.4)

    # --- Population mixture over (beta, epsilon) from a single uniform
    # seed. Three regimes: TRUE guessers (beta ~ 0 -> uniform core on
    # every trial), a mid anti-validity regime, and a deterministic
    # anti-validity regime. Iter-7 fractions: 16% / 50% / 34%.
    try:
        u = float(parameters["beta_seed"])
    except (TypeError, ValueError):
        u = 0.5
    u = min(max(u, 0.0), 1.0)
    if u < 0.16:
        # True guessing subpopulation (~16%): beta ~ 0 makes the
        # softmax core essentially uniform on every trial, so these
        # subjects sit at ~0.5 on every diagnostic cell.
        t = u / 0.16
        beta = 0.02 + 0.08 * t         # 0.02 .. 0.10
        epsilon = 0.12 + 0.16 * t      # 0.12 .. 0.28
    elif u < 0.66:
        # Mid subpopulation (~50%): moderate beta (small step up from
        # the iter-6 1.6-2.2 to target the Exp-5 cells where mid
        # subjects sit too close to chance), low lapse.
        t = (u - 0.16) / 0.50
        beta = 1.8 + 0.6 * t           # 1.8 .. 2.4
        epsilon = 0.11 - 0.06 * t      # 0.11 .. 0.05
    else:
        # Deterministic subpopulation (~34%): high beta with widened
        # within-regime spread, minimal lapse. Fraction (not beta
        # ceiling) guards Exp 6.
        t = (u - 0.66) / 0.34
        beta = 3.8 + 2.2 * t           # 3.8 .. 6.0
        epsilon = 0.04 - 0.02 * t      # 0.04 .. 0.02

    tau = float(parameters["tau"])

    # Subjective validities (instructed validity x per-cue attention),
    # then anti-validity weights: the anti-weight of expert j is
    # v'_j + tau * v'_j^2, supralinear in the subjective validity, so
    # the endorsements of the most trusted experts are avoided most
    # strongly while low-validity experts are nearly ignored.
    v_eff = val * att
    w = -(v_eff + tau * np.square(v_eff))

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
- beta_seed: [0, 1]
- tau: [0.15, 0.42]
- attention: [(0.90, 1.10)] * n_features
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1747 (var=0.0108) vs this=-0.2228 (var=0.0055)
- Experiment 2: real=0.2737 (var=0.0930) vs this=0.1713 (var=0.0139)
- Experiment 3: real=0.1227 (var=0.0008) vs this=0.1444 (var=0.0038)
- Experiment 4: real=0.1625 (var=0.0119) vs this=0.1388 (var=0.0265)
- Experiment 5: real=0.6837 (var=0.0643) vs this=0.6506 (var=0.0203)
- Experiment 6: real=0.6300 (var=0.0703) vs this=0.6646 (var=0.0194)
- Experiment 7: real=0.6462 (var=0.0395) vs this=0.6975 (var=0.0214)
- Experiment 8: real=0.6044 (var=0.0653) vs this=0.7078 (var=0.0349)


---

### `pi_4` (overall score: 0.388)

**Description**
Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI). Subjects integrate all cues additively into a per-option subjective value, but a rating of 1 is treated as evidence AGAINST the option (comprehension inversion, s = -1 for ~99% of subjects). The anti-weight of cue j is a free power law of the instructed validity, |w_j| = v_j^gamma, with gamma a free per-subject parameter drawn from a heterogeneous population distribution (gamma ~ 1.4-2.4, mildly supralinear on average). Choice is a softmax over beta * s * sum_j v_j^gamma * rating_j with a symmetric lapse epsilon. Because gamma, beta, and epsilon are drawn broadly per subject, the population reproduces both the mean-level signatures (negative Exp-1 slope, positive Exp-2 dissociation, positive Exp-3 slope, low Exp-4 conflict-following) and the large between-subject variance seen in the real data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = s * v_j**gamma per unit
    # rating, where s = -1 for the (vast majority of) comprehension-inverted
    # subjects and +1 for the rare pro-validity subject; gamma is a free
    # power-law exponent applied to the instructed validity. Option values
    # are the cue-weighted sums; choice is a numerically stable softmax over
    # beta * values, mixed with a symmetric lapse epsilon. History is
    # ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PAHI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s_txt = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s_txt.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    sign_seed = float(parameters["sign_seed"])

    # Heterogeneous comprehension inversion: with population probability
    # 0.99 a subject inverts the rating direction (a 1 counts AGAINST the
    # option); the remaining subjects read ratings pro-validity.
    s = -1.0 if sign_seed < 0.99 else 1.0

    # Power-law anti-validity cue weights: the anti-weight of expert j is
    # v_j**gamma (on average mildly supralinear in the instructed validity).
    w = s * np.power(val, gamma)

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
- beta: [1.5, 4.0]
- gamma: [1.4, 2.4]
- epsilon: [0.06, 0.20]
- sign_seed: [0, 1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1747 (var=0.0108) vs this=-0.1883 (var=0.0047)
- Experiment 2: real=0.2737 (var=0.0930) vs this=0.2669 (var=0.0115)
- Experiment 3: real=0.1227 (var=0.0008) vs this=0.1448 (var=0.0005)
- Experiment 4: real=0.1625 (var=0.0119) vs this=0.1696 (var=0.0044)
- Experiment 5: real=0.6837 (var=0.0643) vs this=0.2281 (var=0.0125)
- Experiment 6: real=0.6300 (var=0.0703) vs this=0.2267 (var=0.0083)
- Experiment 7: real=0.6462 (var=0.0395) vs this=0.2169 (var=0.0147)
- Experiment 8: real=0.6044 (var=0.0653) vs this=0.2078 (var=0.0291)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0760 -> ACCEPTED
- iter 2: loss=0.0694 -> ACCEPTED
- iter 3: loss=0.0671 -> ACCEPTED
- iter 4: loss=0.0815 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.0671 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.1678 (var=0.0076) (Δ vs real +0.0068)
  - iter 2: -0.1785 (var=0.0060) (Δ vs real -0.0038)
  - iter 3: -0.1667 (var=0.0081) (Δ vs real +0.0080)
  - iter 4 (most recent): -0.1767 (var=0.0091) (Δ vs real -0.0020)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0023 (var=0.0008)
- pi_2: 0.1952 (var=0.0019)
- pi_3: -0.1877 (var=0.0014)
- pi_4: -0.1883 (var=0.0047)
- pi_3_1: -0.2228 (var=0.0055)

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
  - iter 1: 0.1331 (var=0.0134) (Δ vs real -0.1406)
  - iter 2: 0.1300 (var=0.0184) (Δ vs real -0.1437)
  - iter 3: 0.1950 (var=0.0272) (Δ vs real -0.0787)
  - iter 4 (most recent): 0.1950 (var=0.0258) (Δ vs real -0.0787)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0006 (var=0.0060)
- pi_1: -0.6981 (var=0.0507)
- pi_3: 0.1862 (var=0.0095)
- pi_4: 0.2669 (var=0.0115)
- pi_3_1: 0.1713 (var=0.0139)

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
  - iter 1: 0.1079 (var=0.0031) (Δ vs real -0.0148)
  - iter 2: 0.1271 (var=0.0027) (Δ vs real +0.0044)
  - iter 3: 0.1331 (var=0.0030) (Δ vs real +0.0104)
  - iter 4 (most recent): 0.1238 (var=0.0032) (Δ vs real +0.0011)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1378 (var=0.0002)
- pi_2: -0.1263 (var=0.0006)
- pi_1: -0.0888 (var=0.0010)
- pi_4: 0.1448 (var=0.0005)
- pi_3_1: 0.1444 (var=0.0038)

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
  - iter 1: 0.1958 (var=0.0191) (Δ vs real +0.0333)
  - iter 2: 0.1992 (var=0.0228) (Δ vs real +0.0367)
  - iter 3: 0.2304 (var=0.0250) (Δ vs real +0.0679)
  - iter 4 (most recent): 0.2121 (var=0.0197) (Δ vs real +0.0496)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0090)
- pi_3: 0.1608 (var=0.0029)
- pi_1: 0.6796 (var=0.0042)
- pi_4: 0.1696 (var=0.0044)
- pi_3_1: 0.1388 (var=0.0265)

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
  - iter 1: 0.5669 (var=0.0241) (Δ vs real -0.1169)
  - iter 2: 0.5306 (var=0.0418) (Δ vs real -0.1531)
  - iter 3: 0.5700 (var=0.0485) (Δ vs real -0.1138)
  - iter 4 (most recent): 0.5169 (var=0.1074) (Δ vs real -0.1669)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5531 (var=0.0069)
- pi_4: 0.2281 (var=0.0125)
- pi_1: 0.8350 (var=0.0129)
- pi_2: 0.1300 (var=0.0103)
- pi_3_1: 0.6506 (var=0.0203)

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
  - iter 1: 0.5654 (var=0.0343) (Δ vs real -0.0646)
  - iter 2: 0.5642 (var=0.0392) (Δ vs real -0.0658)
  - iter 3: 0.5908 (var=0.0726) (Δ vs real -0.0392)
  - iter 4 (most recent): 0.5529 (var=0.1007) (Δ vs real -0.0771)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2267 (var=0.0083)
- pi_3: 0.5687 (var=0.0053)
- pi_1: 0.8783 (var=0.0092)
- pi_2: 0.1263 (var=0.0072)
- pi_3_1: 0.6646 (var=0.0194)

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
  - iter 1: 0.5988 (var=0.0296) (Δ vs real -0.0475)
  - iter 2: 0.6478 (var=0.0449) (Δ vs real +0.0016)
  - iter 3: 0.5931 (var=0.0721) (Δ vs real -0.0531)
  - iter 4 (most recent): 0.5469 (var=0.0950) (Δ vs real -0.0994)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.6975 (var=0.0214)
- pi_4: 0.2169 (var=0.0147)
- pi_1: 0.8762 (var=0.0079)
- pi_2: 0.1344 (var=0.0070)
- pi_3: 0.6412 (var=0.0050)

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
  - iter 1: 0.6267 (var=0.0367) (Δ vs real +0.0222)
  - iter 2: 0.6222 (var=0.0684) (Δ vs real +0.0178)
  - iter 3: 0.5911 (var=0.0919) (Δ vs real -0.0133)
  - iter 4 (most recent): 0.5900 (var=0.0986) (Δ vs real -0.0144)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2078 (var=0.0291)
- pi_3_1: 0.7078 (var=0.0349)
- pi_1: 0.8611 (var=0.0123)
- pi_2: 0.1200 (var=0.0071)
- pi_3: 0.6056 (var=0.0099)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

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

`rationale`: This is a distribution-only edit of the ACCEPTED iter-2 HEAI base (mechanism, guesser fraction, epsilon ranges, seeds, and code structure unchanged), implementing the iter-2 critic's diagnosis — whose iter-1 advice produced an ACCEPTED loss improvement, so its model of the metrics deserves weight. The critic showed that the wide unimodal N(1.12, 0.40) dumps its added spread into the ambiguous gamma ~ 1.2-1.6 band, where subjects sit at ~chance on every diagnostic cell: that raised variances (as predicted) but capped the means on both sides simultaneously — Exp 5 fell 0.567 -> 0.531 against 0.684, Exp 2 stayed stuck at 0.130 against 0.274, and the 1.8 truncation cap made top-cue dominance on Exp 2's margin-1 cells structurally impossible. The edit: (1) GAMMA MADE BIMODAL with minimal middle mass — a shallow majority (75% of engaged, N(1.04, 0.22) trunc [0.60, 1.42]) cleanly below the kill thresholds (~1.31-1.34), and a steep minority (25%, N(2.10, 0.26) trunc [1.65, 2.75]) cleanly above ~1.8 and extending toward the gamma ~ 2-3 range the critic identified as necessary for Exp 2. Mixture arithmetic on the established simulation anchors (PAHI's all-steep population: Exp 2 = 0.267, Exp 8 = 0.208; iter-2's shallow population: Exp 2 = 0.130, Exp 8 = 0.622) predicts Exp 2 rises to ~0.20-0.22 while Exp 8 holds near 0.62 — the steep minority is precisely the mass that lowers Exp 8 from shallow's overshoot region while lifting Exp 2. (2) The shallow mode's thin upper shoulder (1.32-1.42, ~10% of the mode) is retained deliberately: subjects just above the kill thresholds miss Exp 8 but still hit the higher-threshold Exp 5/7 ladder cells (flip points 1.58-2.07), which is what reconciles Exp 8's lower target mean (0.604) with Exp 5/6/7's higher ones (0.63-0.68) — a reconciliation neither a clean bimodal nor the old unimodal could achieve. (3) BETA MODE-LINKED AND HOTTER (shallow median 2.3 -> ~2.6; steep median ~3.3, both heavy-tailed): hotter shallow beta sharpens engaged adherence on Exp 5/6/7 (the pi_3_1 vs pi_3 contrast shows hot beta is the lever that lifted Exp 5 from 0.553 to 0.651), and hotter steep beta maximizes each steep subject's Exp 2 dissociation contribution while pushing steep Exp 8/Exp 4 hits toward the floor. Expected net movement vs iter-2: Exp 2 0.130 -> ~0.21, Exp 5 0.531 -> ~0.59, Exp 6 0.564 -> ~0.63, Exp 4 0.199 -> ~0.18, with Exp 1/3/8 approximately held and Exp 7 giving back ~0.03-0.05 (the one deliberate trade, repurchased many times over on Exp 2/5/6). Bimodality also widens the between-subject variance on the kill composites (Exp 6/7 variances move toward the observed 0.04-0.07). The population mean elasticity (~1.30) remains below the arbiter's 1.4 ceiling, all population structure continues to be derived inside predict from uniform seeds, and the declared parameter boxes are untouched — the candidate stays squarely inside the prescribed HEAI family while replacing the falsified unimodal shape with the separated-subpopulation structure the data demand.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: HEAI is a faithful implementation of the arbiter's prescribed family (additive anti-validity integration, per-subject power-law elasticity with population mean below 1.4, guesser subpopulation, heavy-tailed beta, symmetric lapse), and it was accepted by the gate. The signs are right everywhere: Exp 1 slope (-0.168 vs -0.175), Exp 3 slope (0.108 vs 0.123), Exp 4 conflict-following (0.196 vs 0.163), Exp 7 (0.599 vs 0.646), Exp 8 kill composite (0.627 vs 0.604). However, there is a systematic residual pattern across the suite: (a) Exp 2 dissociation is badly underpredicted in both mean (0.133 vs 0.274) and especially between-subject variance (0.013 vs 0.093 — a 7x gap); (b) Exp 5 adherence is underpredicted (0.567 vs 0.684) and Exp 6 mildly so (0.565 vs 0.630); (c) between-subject variance is underpredicted by roughly 2-3x on Exp 5 (0.024 vs 0.064), Exp 6 (0.034 vs 0.070) and Exp 8 (0.037 vs 0.065). The diagnosis is that the heterogeneity in the weight-map shape is not wide enough: with gamma ~ N(1.15, 0.25) truncated [0.7, 1.7], roughly three quarters of engaged subjects cluster in a narrow band around the kill-design count-ratio thresholds (~1.31-1.34 on Exp 8; ~1.32 on the (2, 5x0.5) Exp-5 ladder cell; ~0.81 on the (4, 7x0.5) Exp-5 cell), so most subjects sit on the same side of each threshold and the population behaves too homogeneously. The (4-perfect-cues vs 7x0.5) cells in Exp 5 are the clearest single failure: the flip threshold is gamma ~ 0.81, so nearly all engaged subjects (gamma > 0.81) abandon the high-validity stack there, dragging the Exp 5 composite down to 0.567 where humans show 0.684.
Rationale: The candidate is in-family and accepted, but the fit leaves large, systematic, and mechanically diagnosable gaps that can be closed with parameter-distribution adjustments inside the prescribed HEAI family — no mechanism change needed. Concretely: (1) WIDEN the gamma distribution: raise the SD from 0.25 to roughly 0.35-0.45 and extend the truncation range to approximately [0.6, 1.8], keeping the mean near 1.10-1.15 (still safely below the arbiter's 1.4 ceiling). This is the single lever that addresses the dominant failure — between-subject variance underprediction on Exp 2/5/6/8 — because it spreads subjects across both sides of every count-ratio threshold instead of clustering them just below it. Back-of-envelope: with SD ~0.4, P(gamma < 0.81) ~ 0.19 recovers top-stack following on the Exp-5 (4, 7x0.5) cells, P(gamma < 1.32) ~ 0.67 keeps the Exp-8 composite near 0.60-0.63 (currently well calibrated at 0.627 — do NOT reduce the guesser fraction or epsilon, which would push Exp 8 and Exp 4 too high), and the wider spread should lift the Exp 5 composite toward ~0.65-0.69 and roughly double the kill-metric variances. (2) Modestly hotter beta: raise the lognormal median from ~1.9 toward ~2.3-2.8 while keeping the heavy right tail and the [0.5, 6] clip. This sharpens engaged subjects' adherence on Exp 5/6/7 (means currently 0.04-0.12 too low) and simultaneously pulls the Exp 4 conflict-following rate down from 0.196 toward the observed 0.163 (hotter anti-validity integration means less tally-following on conflict probes). Watch Exp 1: the slope may become slightly more negative (currently -0.168 vs real -0.175; pi_3_1's steeper variants reach -0.22), so if it overshoots, compensate via the gamma widening rather than cooling beta. (3) The Exp 2 variance gap (0.013 vs 0.093) should be attacked primarily through the gamma widening — subjects near the bottom of the widened range (gamma ~ 0.6-0.8) have nearly flat anti-weights and behave tally-like, producing strongly positive per-subject dissociation, while steep-gamma subjects sit near zero — exactly the bimodality-in-continuous-form the real variance implies. If after widening, the Exp 2 mean is still short of ~0.27, allow the gamma lower bound to extend slightly further down (toward 0.5) rather than adding any new mechanism. All of these are distribution-shape recalibrations of the exact knobs the arbiter prescribed (gamma population, beta population, guesser fraction, lapse), so the candidate remains squarely inside the HEAI family.

**Outcome of this advice:** iter 1 candidate loss=0.0760 -> iter 2 candidate loss=0.0694 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 candidate is faithful to the prescribed HEAI family (mechanism untouched, mean elasticity 1.12 < 1.4, guessers 16%, heavy-tailed beta, symmetric lapse) and the gate accepted it (loss 0.0760 -> 0.0694). My iter-1 advice was partially validated: between-subject variances moved substantially toward the real values on Exp 5 (0.024 -> 0.042 vs 0.064), Exp 6 (0.034 -> 0.039 vs 0.070), Exp 8 (0.037 -> 0.068 vs 0.065, now essentially exact), and Exp 7's mean is now spot-on (0.648 vs 0.646). Exp 1 (-0.179 vs -0.175), Exp 3 (0.127 vs 0.123), and Exp 8 (0.622 vs 0.604) are all in excellent shape. HOWEVER, the widening backfired on exactly the mean it was supposed to fix: Exp 5 dropped from 0.567 to 0.531 against an observed 0.684, Exp 6 is flat at 0.564 vs 0.630, Exp 2 is stuck at 0.130 vs 0.274 (variance 0.018 vs 0.093, still a 5x gap), and Exp 4 drifted the wrong way (0.196 -> 0.199 vs 0.163). The mechanical reason is now clear: a wide UNIMODAL gamma centered at 1.12 dumps its added spread into the threshold band (gamma ~ 1.2-1.6), which is exactly where subjects sit at chance on the diagnostic cells. The Exp 5 included cells flip at gamma ~ 1.32/1.58/1.81/2.0 and the Exp 6 cells at ~1.17-1.48; subjects in the middle band contribute ~0.5 everywhere, which raises variance (as I predicted) but caps the means on BOTH sides simultaneously. Meanwhile the hard truncation cap at gamma = 1.8 makes TTB-like behavior on Exp 2's margin-1 cells structurally impossible (the top cue there needs gamma ~ 2-3 to dominate three lower cues), which is why Exp 2 never moved despite two rounds of tuning. The data demand SEPARATED subpopulations: a shallow majority cleanly below ~1.2 (drives Exp 5/6/8 top-stack adherence and the Exp 8 high-stack composite) and a steep minority cleanly above ~1.8 (drives the Exp 2 dissociation mean/variance and the low Exp 4 conflict-following rate) — with as little mass as possible wasted in the ambiguous middle. The current unimodal shape cannot deliver both; a bimodal continuous population can.
Rationale: verdict=regenerate

**Outcome of this advice:** iter 2 candidate loss=0.0694 -> iter 3 candidate loss=0.0671 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The iter-3 bimodal edit was accepted (loss 0.0694 -> 0.0671) and my iter-2 diagnosis was validated in its main predictions: Exp 2 rose 0.130 -> 0.195 (target 0.274), Exp 6 0.564 -> 0.591 (target 0.630) with variance now essentially exact (0.073 vs 0.070), and Exp 8 held at 0.591 (target 0.604). Exp 1 (-0.167 vs -0.175) and Exp 3 (0.133 vs 0.123) remain excellent. The separated-subpopulation structure is clearly the right shape. However, four systematic gaps remain, and they are jointly diagnosable: (1) Exp 5 is stuck at 0.570 vs 0.684 — the worst mean gap in the suite. The mechanism is identifiable: of the 14 included trials, 2 flip at gamma ~ 0.585 ((2 vs 3x0.5) cells) and 2 flip at gamma ~ 0.81 ((4 perfect vs 7x0.5) cells); with the shallow mode at N(1.04, 0.22), only ~0.2% and ~15% of shallow subjects fall below those thresholds, so nearly the entire engaged population abandons the high-validity stack on 4 of 14 trials. pi_3_1 (a shallower population) reaches 0.651 on this metric — the shallow mode is simply not shallow enough. (2) Exp 2 mean is still 0.08 short and its variance is 0.027 vs 0.093 — a 3.4x gap that mixture arithmetic pins on the steep mode being too small and/or not extreme enough per-subject (the proposer's own anchors: shallow population ~0.13, all-steep ~0.267; with 25% steep the mixture saturates near 0.20). (3) Exp 4 drifted the WRONG way, 0.199 -> 0.230 vs target 0.163. The culprit is the shallow-beta hike (median 2.3 -> 2.6): at gamma ~1.0 the anti-weights over Exp 4's validities [1.0, 0.95, 0.65, 0.5x3] are nearly flat, so a hot shallow subject follows the raw tally near-deterministically on the conflict probes. My iter-1 suggestion of beta ~2.3-2.8 overshot; the shallow median needs to come back down. (4) Exp 7 gave back its iter-2 spot-on fit (0.648 -> 0.593 vs 0.646), consistent with the shallow mode narrowing and centering at 1.04 — lower-gamma mass is what follows SCWI's high-validity stack on those cells.
Rationale: regenerate — but this is a small-step distribution recalibration on top of the accepted iter-3 base, not a redesign; the bimodal architecture should be kept. Three coordinated edits, all inside the prescribed HEAI family: (1) LOWER THE SHALLOW MODE CENTER from 1.04 to ~0.87-0.92 (keep SD ~0.22-0.25 and the thin upper shoulder to ~1.42). This is the single biggest lever: it puts ~35-45% of shallow subjects below the 0.81 threshold that governs the (4, 7x0.5) Exp-5 cells and some mass below 0.585, lifting Exp 5 from 0.570 toward 0.65+; it simultaneously raises Exp 7 back toward 0.646 (lower gamma = more SCWI-like on those cells) and nudges Exp 8 up from 0.591 toward 0.604 (kill thresholds 1.31-1.34, more mass below them). Watch Exp 3 (may dip slightly from 0.133 toward 0.123 — fine) and Exp 1 (may drift toward -0.15; accept a small miss there rather than re-heating beta). (2) RAISE THE STEEP FRACTION from 25% to ~32-35% of engaged, and consider centering the steep mode slightly higher (~2.2-2.4) so per-subject dissociation is more extreme. This attacks the two largest remaining gaps at once: Exp 2 mean (mixture arithmetic: 0.55*0.13 + 0.30*0.7 + 0.16*0 ≈ 0.27, on target) and especially Exp 2 variance (~0.07-0.08 vs the observed 0.093 — the steep pocket is the only variance source big enough), while pulling Exp 4 back DOWN toward 0.163 (steep subjects' integrators strongly prefer the tally loser, offsetting the shallow mode's tally-following). The cost on Exp 5/6/8 is small (the steep mode contributes little to those composites) and is more than repaid by edit (1). (3) COOL THE SHALLOW BETA back to a lognormal median ~2.2-2.4 (keep the heavy tail and clip). My iter-1 push toward 2.3-2.8 overshot on Exp 4: at near-flat weights, hotter shallow beta means MORE tally-following on the conflict probes, which is why Exp 4 rose 0.199 -> 0.230. Keep the steep beta hot (median ~3.3) — that is what maximizes each steep subject's Exp-2 contribution. Net expected movement: Exp 5 0.570 -> ~0.65, Exp 2 0.195 -> ~0.26 with var 0.027 -> ~0.07, Exp 4 0.230 -> ~0.17, Exp 7 0.593 -> ~0.64, with Exp 1/3/6/8 approximately held. One process note: before committing, verify the Exp-4 and Exp-1 side effects of each edit separately — this round showed that the beta hike and the gamma re-centering interact, and the proposer's a-priori mixture arithmetic under-predicted the Exp-4 regression. The population mean elasticity (~1.2 after these edits) stays below the arbiter's 1.4 ceiling, and no mechanism, guesser fraction, or lapse change is involved.

**Outcome of this advice:** iter 3 candidate loss=0.0671 -> iter 4 candidate loss=0.0815 -> the gate marked it REJECTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-4 candidate was REJECTED by the gate (loss 0.0671 -> 0.0815), so my iter-3 advice must be judged a failure and I will not repeat it. The per-experiment decomposition shows exactly why. The three coordinated edits interacted destructively: (a) raising the steep fraction from 25% to 33% of engaged subjects added ~8 percentage points of subjects who sit on the ANTI-SCWI side of the Exp-5/6/7 diagnostic cells (steep gamma > ~1.8 prefers the low-validity stack wherever SCWI prefers the high stack), dragging Exp 5 from 0.570 down to 0.517 (target 0.684), Exp 6 from 0.591 to 0.553 (target 0.630), and Exp 7 from 0.593 to 0.547 (target 0.646); (b) cooling the shallow beta (2.6 -> 2.3) softened shallow adherence everywhere, compounding those mean drops and pushing between-subject variances on Exp 5/6/7/8 well ABOVE their targets (0.107/0.101/0.095/0.099 vs 0.064/0.070/0.040/0.065); (c) the edit that was supposed to pay for all this — the steep enlargement lifting Exp 2 — did NOT materialize: Exp 2 is exactly unchanged (0.1950, var 0.026 vs targets 0.274, 0.093). The mixture arithmetic failed because the two other edits cancelled it: lowering the shallow mode 1.04 -> 0.89 flattens shallow weights and erodes the shallow subjects' own positive dissociation contribution (~-0.03), offsetting the +0.04 expected from the larger steep pocket. The good news in the wreckage: Exp 1 (-0.177 vs -0.175) and Exp 3 (0.124 vs 0.123) are essentially perfect, Exp 4 moved in the right direction (0.230 -> 0.212 toward 0.163), confirming the shallow-beta-cooling diagnosis, and the bimodal architecture itself remains validated (it produced the only accepted gains in this loop). The failure was in which knobs were turned, not in the family.
Rationale: verdict=regenerate. The base is the accepted iter-3 candidate; the next edit should be a DIFFERENT set of levers than the rejected iter-4 ones (do not re-lower the shallow mode to 0.89, do not re-raise the steep fraction to 33%). The suite's two stubborn gaps need to be attacked through per-subject extremity of the steep mode and lapse calibration, not through mixture proportions: (1) REVERT the steep fraction to ~25% of engaged (this alone should recover most of the Exp 5/6/7 mean drop) and keep the shallow mode in the 0.95-1.05 band with truncation roughly [0.70, 1.42] — the iter-4 widening down to 0.55 is what exploded the variances past their targets. (2) Attack the Exp-2 gap (mean 0.195 vs 0.274, var 0.026 vs 0.093) via PER-STEEP extremity instead of steep mass: on Exp-2's margin-1 cells the steep subject's evidence margins are small (~0.15-0.3 in weight units), so at steep beta median 3.3 the per-subject follow probability is only sigmoid(~0.5-0.9) ~ 0.62-0.71 — far from saturated. Raise the steep beta lognormal median from ~3.3 to ~5-6 and widen its clip to ~10-12; optionally center the steep gamma slightly higher (~2.3-2.5) with truncation extended up to ~3.2. This raises each steep subject's dissociation toward its ceiling, lifting both the Exp-2 mean and (critically) the between-subject variance, while the 25% fraction keeps the Exp-5/6/7 cost at its iter-3 level. (3) Tighten the steep lapse: draw epsilon for steep subjects from ~[0.03, 0.08] instead of the shared [0.03, 0.17] — lapse directly caps per-subject |dissociation| and is a pure variance/mean amplifier for the steep pocket. (4) Modest adherence lift for Exp 5/6/7 that does NOT reheat the shallow beta (which iter 3 proved raises Exp 4): tighten the shallow epsilon upper bound from 0.17 to ~0.12, and optionally trim the guesser fraction from 16% to ~15% (bottom of the arbiter's prescribed 15-20% band). (5) Keep the shallow beta at the cooled ~2.3-2.4 median — the Exp-4 movement 0.230 -> 0.212 confirmed that direction; the steeper/hotter steep mode from (2) should pull Exp 4 the rest of the way toward 0.163 since steep integrators anti-follow the tally on the conflict probes. Process discipline: test edits (2) and (1)+(4) separately before combining — this round demonstrated that the shallow-gamma, steep-fraction, and beta levers interact, and the a-priori mixture arithmetic under-predicts the Exp-4/5 side effects. All changes are distribution-shape recalibrations inside the prescribed HEAI family (additive anti-validity power-law integration, bimodal continuous gamma with population mean ~1.2 < 1.4, guesser subpopulation, heavy-tailed beta, symmetric lapse); no mechanism change is proposed.

**Outcome of this advice:** iter 4 candidate loss=0.0815 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Heterogeneous-Elasticity Anti-Validity Integration (HEAI), steep-extremity refinement. Subjects integrate ALL expert ratings additively into a per-option subjective value, but a rating of 1 is comprehended as a DEFECT (evidence against the option) — a sign inversion of the rating direction. The anti-weight of expert j is a per-subject power law of the instructed validity, w_j = -v_j^gamma, where gamma ('validity elasticity') is drawn from a BIMODAL continuous population: a shallow majority (75% of engaged subjects, gamma ~ N(1.04, 0.22) truncated [0.70, 1.42]) that sits below the kill-design count-ratio thresholds (~1.31-1.34) and follows the high-validity stack, and a steep minority (25% of engaged subjects, gamma ~ N(2.30, 0.26) truncated [1.75, 3.10]) whose near-lexicographic top-cue dominance drives the Exp-2 dissociation. The critical refinement versus the accepted base: the Exp-2 gap (mean 0.195 vs 0.274, variance 0.026 vs 0.093) is attacked through PER-SUBJECT EXTREMITY of the steep pocket rather than steep mass — steep subjects now draw a much hotter inverse temperature (lognormal median ~5.5, clip to 12), a higher and wider gamma (center 2.30, up to 3.10), and a tighter lapse ([0.03, 0.08]) — because on Exp-2's margin-1 cells the steep subject's evidence margins are small (~0.15-0.3 weight units), so per-subject dissociation was far from saturated at the old beta median 3.3. Meanwhile the shallow mode stays at 1.04 (NOT re-lowered to 0.89, whose rejection the iter-4 gate verdict confirmed), its lower truncation is tightened 0.60 -> 0.70 (the iter-4 widening down to ~0.55 is what exploded the variances), its beta is cooled to median ~2.3 (continuing the validated Exp-4 correction 0.230 -> 0.212 toward 0.163), and its lapse is tightened to [0.03, 0.12] to lift Exp-5/6/7 adherence WITHOUT reheating beta. The guesser subpopulation is trimmed to 15% (bottom of the arbiter's 15-20% band). Population mean elasticity ~1.32-1.36, below the arbiter's 1.4 ceiling.",
  "predict": "def predict(parameters, state, history):\n    # Heterogeneous-Elasticity Anti-Validity Integration (HEAI), iter 5.\n    # Stimulus: array-like of shape (2, n_features); row 0 = option A,\n    # row 1 = option B; entries are binary expert ratings.\n    # Mechanism (UNCHANGED from the accepted iter-3 base): every cue\n    # contributes signed evidence w_j = -v_j**gamma per unit rating\n    # (a rating of 1 is a defect; the anti-weight is a per-subject\n    # power law of the instructed validity). Choice is a numerically\n    # stable softmax over beta * option values, mixed with a symmetric\n    # lapse epsilon. History is ignored (no trial-by-trial feedback).\n    #\n    # Population edits on top of the accepted iter-3 base (mechanism\n    # untouched), implementing the iter-4 critic's diagnosis of the\n    # REJECTED iter-4 attempt (whose levers -- shallow mode lowered to\n    # 0.89, steep fraction raised to 33% -- are explicitly NOT repeated):\n    #   * STEEP EXTREMITY, NOT STEEP MASS: the Exp-2 gap is attacked via\n    #     per-subject extremity of the steep pocket. Steep fraction stays\n    #     at 25% of engaged. Steep beta lognormal median raised\n    #     exp(1.20) ~ 3.3 -> exp(1.70) ~ 5.5, clip [1.2, 8.0] ->\n    #     [2.0, 12.0] (on Exp-2's margin-1 cells the steep evidence\n    #     margins are only ~0.15-0.3 weight units, so the old beta left\n    #     per-subject dissociation at sigmoid(~0.5-0.9) ~ 0.62-0.71,\n    #     far from saturation). Steep gamma re-centered 2.10 -> 2.30,\n    #     truncation [1.65, 2.75] -> [1.75, 3.10]. Steep lapse tightened\n    #     to [0.03, 0.08] (lapse directly caps per-subject |dissociation|).\n    #   * SHALLOW MODE HELD at N(1.04, 0.22), lower truncation tightened\n    #     0.60 -> 0.70 (the iter-4 widening down to ~0.55 exploded the\n    #     between-subject variances past their targets).\n    #   * SHALLOW BETA COOLED to lognormal median exp(0.83) ~ 2.3 (from\n    #     2.6): the iter-4 Exp-4 movement 0.230 -> 0.212 confirmed this\n    #     direction; at near-flat shallow weights, hotter beta means MORE\n    #     tally-following on the Exp-4 conflict probes.\n    #   * LAPSES TIGHTENED: shallow epsilon [0.03, 0.17] -> [0.03, 0.12]\n    #     (lifts Exp-5/6/7 adherence without reheating beta); guesser\n    #     epsilon unchanged.\n    #   * GUESSER FRACTION trimmed 16% -> 15% (bottom of the arbiter's\n    #     prescribed 15-20% band).\n    # Population mean elasticity ~ 1.32-1.36 < 1.4 (arbiter ceiling).\n    import math\n    import numpy as np\n    from scipy.special import ndtri\n\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"HEAI expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    n_features = stim.shape[1]\n\n    def _to_vec(raw):\n        if raw is None:\n            return None\n        if isinstance(raw, str):\n            s = raw.strip().strip('[]()')\n            toks = [t for t in s.replace(',', ' ').split() if t != '']\n            if not toks:\n                return None\n            return np.array([float(t) for t in toks])\n        arr = np.asarray(raw, dtype=float).ravel()\n        return arr if arr.size > 0 else None\n\n    val = _to_vec(parameters.get(\"validities\"))\n    if val is None or val.size != n_features:\n        # Defensive fallback: uniform mid-range validities.\n        val = np.full(n_features, 0.75)\n    val = np.clip(val, 1e-3, 1.0)\n\n    def _seed(name):\n        try:\n            x = float(parameters[name])\n        except (KeyError, TypeError, ValueError):\n            x = 0.5\n        if not np.isfinite(x):\n            x = 0.5\n        return min(max(x, 0.0), 1.0)\n\n    u = _seed(\"regime_seed\")\n    # Keep CDF seeds strictly inside (0, 1) so ndtri stays finite.\n    gs = min(max(_seed(\"gamma_seed\"), 1e-6), 1.0 - 1e-6)\n    bs = min(max(_seed(\"beta_seed\"), 1e-6), 1.0 - 1e-6)\n    es = _seed(\"epsilon_seed\")\n\n    if u < 0.15:\n        # Guesser subpopulation (~15%): beta ~ 0 makes the softmax core\n        # essentially uniform on every trial; lapse is elevated.\n        t = u / 0.15\n        beta = 0.02 + 0.08 * t          # 0.02 .. 0.10\n        epsilon = 0.12 + 0.13 * es      # 0.12 .. 0.25\n        gamma = 1.12                    # irrelevant at beta ~ 0\n    else:\n        z_b = float(ndtri(bs))\n        if gs < 0.75:\n            # SHALLOW majority (~75% of engaged): gamma ~ N(1.04, 0.22)\n            # truncated to [0.70, 1.42]. Cleanly below the kill-design\n            # thresholds (~1.31-1.34), with a thin upper shoulder that\n            # keeps Exp 8's mean/variance calibrated.\n            t = gs / 0.75\n            tt = min(max(t, 1e-4), 1.0 - 1e-4)\n            gamma = 1.04 + 0.22 * float(ndtri(tt))\n            gamma = min(max(gamma, 0.70), 1.42)\n\n            # Cooled shallow beta: lognormal median ~2.3, heavy tail.\n            beta = math.exp(0.83 + 0.70 * z_b)\n            beta = min(max(beta, 0.6), 7.0)\n\n            # Tightened shallow lapse (was 0.03 .. 0.17).\n            epsilon = 0.03 + 0.09 * es  # 0.03 .. 0.12\n        else:\n            # STEEP minority (~25% of engaged): gamma ~ N(2.30, 0.26)\n            # truncated to [1.75, 3.10]. Cleanly above ~1.8 so the top\n            # cue can dominate three lower cues (the Exp 2 lever), now\n            # extending higher (up to 3.10) for per-subject extremity.\n            t = (gs - 0.75) / 0.25\n            tt = min(max(t, 1e-4), 1.0 - 1e-4)\n            gamma = 2.30 + 0.26 * float(ndtri(tt))\n            gamma = min(max(gamma, 1.75), 3.10)\n\n            # Much hotter steep beta: lognormal median ~5.5, clip 12.\n            # On Exp-2's margin-1 cells the steep evidence margins are\n            # only ~0.15-0.3 weight units, so saturation requires\n            # beta * margin >> 1.\n            beta = math.exp(1.70 + 0.60 * z_b)\n            beta = min(max(beta, 2.0), 12.0)\n\n            # Tight steep lapse: lapse directly caps per-subject\n            # |dissociation| on Exp 2 (mean AND variance amplifier).\n            epsilon = 0.03 + 0.05 * es  # 0.03 .. 0.08\n\n    # Anti-validity power-law cue weights: a rating of 1 counts AGAINST\n    # the option, with magnitude v_j**gamma.\n    w = -np.power(val, gamma)\n\n    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])\n\n    # Numerically stable softmax over the two option values.\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # Guard against float drift.\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "regime_seed": "[0, 1]",
    "gamma_seed": "[0, 1]",
    "beta_seed": "[0, 1]",
    "epsilon_seed": "[0, 1]",
    "validities": "validities"
  },
  "rationale": "This is a distribution-only minimal-diff edit of the ACCEPTED iter-3 HEAI base (mechanism, seeds, code structure, and parameter boxes unchanged), implementing the iter-4 critic's diagnosis — the first critic round whose candidate was REJECTED, so its post-mortem of WHY the iter-4 levers failed is the most informative signal available, and its explicit 'do-not-repeat' list is followed exactly: the shallow mode is NOT re-lowered to 0.89 (that dragged Exp 5 to 0.517, Exp 6 to 0.553, Exp 7 to 0.547), and the steep fraction is NOT re-raised to 33% (that added anti-SCWI mass on the Exp-5/6/7 cells). The five coordinated edits, all inside the prescribed HEAI family: (1) STEEP EXTREMITY replaces steep mass as the Exp-2 lever. The iter-4 wreckage showed that enlarging the steep pocket to 33% costs ~5pp on each of Exp 5/6/7 while the Exp-2 gain was cancelled by the simultaneous shallow flattening — Exp 2 stayed at exactly 0.1950. The mechanical reason, per the critic: on Exp-2's margin-1 cells a steep subject's evidence margin is only ~0.15-0.3 weight units, so at steep beta median 3.3 the per-subject follow probability is sigmoid(~0.5-0.9) ~ 0.62-0.71 — unsaturated. Raising the steep beta lognormal median to ~5.5 (clip 12), re-centering steep gamma at 2.30 with truncation to 3.10, and tightening the steep lapse to [0.03, 0.08] pushes each steep subject's dissociation toward its ceiling, lifting BOTH the Exp-2 mean (toward ~0.24-0.26) and its between-subject variance (the 0.026-vs-0.093 gap is a 3.4x deficit that only per-subject extremity in the steep pocket can close), while the 25% fraction holds the Exp-5/6/7 cost at its accepted iter-3 level. (2) The hotter, more extreme steep integrators also pull Exp 4 down toward 0.163 (steep subjects anti-follow the tally on the conflict probes), complementing edit (4). (3) SHALLOW MODE HELD at N(1.04, 0.22) with lower truncation tightened 0.60 -> 0.70: the iter-4 widening to ~0.55 is what exploded the Exp 5/6/7/8 variances to 0.095-0.107 against targets of 0.040-0.070; the thin upper shoulder to 1.42 is retained (it reconciles Exp 8's lower target mean with Exp 5/6/7's higher ones). (4) SHALLOW BETA COOLED to median ~2.3 and SHALLOW LAPSE TIGHTENED to [0.03, 0.12]: the iter-4 Exp-4 movement 0.230 -> 0.212 (target 0.163) validated the beta-cooling direction, and the tightened lapse recovers the Exp-5/6/7 adherence that cooling would otherwise soften — lapse reduction sharpens adherence on the ladder cells without the tally-following side effect that reheating beta produces on the Exp-4 conflict probes. (5) GUESSER FRACTION trimmed 16% -> 15%, the bottom of the arbiter's prescribed band, a small additional adherence lift on every composite. Expected net movement vs the accepted iter-3 base: Exp 2 0.195 -> ~0.25 with variance 0.026 -> ~0.06-0.08, Exp 4 0.230 -> ~0.17-0.19, Exp 5 0.570 -> ~0.60-0.63, Exp 6 0.591 -> ~0.62, Exp 7 0.593 -> ~0.62-0.64, with Exp 1 (-0.167), Exp 3 (0.133), and Exp 8 (0.591) approximately held (the shallow population is nearly unchanged; only its beta/lapse are modestly recalibrated in compensating directions). The population mean elasticity (~1.32-1.36 including guessers) remains below the arbiter's 1.4 ceiling, all population structure continues to be derived inside predict from uniform seeds, and the declared parameter boxes are untouched."
}
```
```

## Usage

```json
{
  "input_tokens": 43003,
  "output_tokens": 5845,
  "reasoning_tokens": 2001,
  "provider": "Sail Research"
}
```
