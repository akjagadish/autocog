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
- THEORY 1 = `pi_3_1`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace HEAI (pi_5) with a context-referenced, trait-free anti-validity theory. Sketch — Context-Normalized Noisy Anti-Validity Integration (CNNAI): (1) Keep the empirically well-supported core shared by both theories: all cues integrated additively into a per-option value, a rating of 1 is a defect (sign inversion), softmax plus symmetric lapse, a modest guesser subpopulation. (2) Replace the stable per-subject weight map (whether SCWI's fixed tau or HEAI's lifetime gamma) with a CONTEXT-REFERENCED weight: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean/max validity of the cues actually displayed in the current trial set (or block) and eta is an elasticity that is RE-DRAWN per context/block from a broad unimodal distribution centered near 0.3–0.6 (roughly lognormal, wide). Because eta is redrawn when the validity context changes (e.g., between the 0.50-ladder and 0.60-ladder of Exp 1), the theory predicts cross-ladder consistency near r ≈ 0.15–0.25 — matching the observed 0.196 — while still producing strong within-context anti-validity behavior (negative Exp-3 slope, low Exp-6 tally-following). Range normalization also naturally produces intermediate psychometric rises (Exp 2's D ≈ 0.25–0.35) because the effective weight ratio compresses toward the context's validity spread rather than following a fixed power law. (3) Add per-trial multiplicative attention noise on cue weights (not fixed per-cue attention), which inflates between-subject variance on diagnostic composites toward the observed 0.06–0.09 without shifting means — fixing the systematic variance underprediction of both current theories. (4) Retain a heterogeneous beta/epsilon mixture but let it be uncorrelated with eta, so no latent trait couples behavior across contexts. This theory is a stronger competitor to SCWI than HEAI because it keeps SCWI's empirically successful anti-validity integration while explaining the two things SCWI gets wrong: near-zero cross-context consistency and elevated between-subject variance. It differs from both predecessors precisely where they were falsified: neither a fixed quadratic map nor a fixed power-law trait, but a context-normalized, redrawn-per-context gradient with per-trial attention diffusion.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.927)

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


---

### `pi_3_1` (overall score: 0.827)

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
- Experiment 9: real=0.1963 (var=0.0601) vs this=0.3684 (var=0.0180)
- Experiment 10: real=0.2963 (var=0.0806) vs this=0.4531 (var=0.0906)


---

### `pi_5` (overall score: 0.649)

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


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1070 -> ACCEPTED
- iter 2: loss=0.1032 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.1032 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.1790 (var=0.0088) (Δ vs real -0.0043)
  - iter 2 (most recent): -0.1813 (var=0.0063) (Δ vs real -0.0067)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0023 (var=0.0008)
- pi_2: 0.1952 (var=0.0019)
- pi_3: -0.1877 (var=0.0014)
- pi_4: -0.1883 (var=0.0047)
- pi_3_1: -0.2228 (var=0.0055)
- pi_5: -0.1898 (var=0.0057)

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
  - iter 1: 0.1862 (var=0.0254) (Δ vs real -0.0875)
  - iter 2 (most recent): 0.1394 (var=0.0250) (Δ vs real -0.1344)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0006 (var=0.0060)
- pi_1: -0.6981 (var=0.0507)
- pi_3: 0.1862 (var=0.0095)
- pi_4: 0.2669 (var=0.0115)
- pi_3_1: 0.1713 (var=0.0139)
- pi_5: 0.1550 (var=0.0256)

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
  - iter 1: 0.1391 (var=0.0016) (Δ vs real +0.0164)
  - iter 2 (most recent): 0.1445 (var=0.0005) (Δ vs real +0.0218)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1378 (var=0.0002)
- pi_2: -0.1263 (var=0.0006)
- pi_1: -0.0888 (var=0.0010)
- pi_4: 0.1448 (var=0.0005)
- pi_3_1: 0.1444 (var=0.0038)
- pi_5: 0.1165 (var=0.0040)

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
  - iter 1: 0.1904 (var=0.0230) (Δ vs real +0.0279)
  - iter 2 (most recent): 0.1725 (var=0.0185) (Δ vs real +0.0100)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0090)
- pi_3: 0.1608 (var=0.0029)
- pi_1: 0.6796 (var=0.0042)
- pi_4: 0.1696 (var=0.0044)
- pi_3_1: 0.1388 (var=0.0265)
- pi_5: 0.1979 (var=0.0276)

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
  - iter 1: 0.6000 (var=0.0314) (Δ vs real -0.0837)
  - iter 2 (most recent): 0.5525 (var=0.0185) (Δ vs real -0.1312)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5531 (var=0.0069)
- pi_4: 0.2281 (var=0.0125)
- pi_1: 0.8350 (var=0.0129)
- pi_2: 0.1300 (var=0.0103)
- pi_3_1: 0.6506 (var=0.0203)
- pi_5: 0.6019 (var=0.0710)

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
  - iter 1: 0.5796 (var=0.0327) (Δ vs real -0.0504)
  - iter 2 (most recent): 0.5229 (var=0.0137) (Δ vs real -0.1071)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2267 (var=0.0083)
- pi_3: 0.5687 (var=0.0053)
- pi_1: 0.8783 (var=0.0092)
- pi_2: 0.1263 (var=0.0072)
- pi_3_1: 0.6646 (var=0.0194)
- pi_5: 0.7004 (var=0.0351)

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
  - iter 1: 0.5569 (var=0.0327) (Δ vs real -0.0894)
  - iter 2 (most recent): 0.5509 (var=0.0090) (Δ vs real -0.0953)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.6975 (var=0.0214)
- pi_4: 0.2169 (var=0.0147)
- pi_1: 0.8762 (var=0.0079)
- pi_2: 0.1344 (var=0.0070)
- pi_3: 0.6412 (var=0.0050)
- pi_5: 0.6056 (var=0.0640)

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
  - iter 1: 0.5900 (var=0.0332) (Δ vs real -0.0144)
  - iter 2 (most recent): 0.5578 (var=0.0296) (Δ vs real -0.0467)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2078 (var=0.0291)
- pi_3_1: 0.7078 (var=0.0349)
- pi_1: 0.8611 (var=0.0123)
- pi_2: 0.1200 (var=0.0071)
- pi_3: 0.6056 (var=0.0099)
- pi_5: 0.6122 (var=0.0654)

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
  - iter 1: 0.4412 (var=0.0444) (Δ vs real +0.2449)
  - iter 2 (most recent): 0.1635 (var=0.0422) (Δ vs real -0.0328)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.3684 (var=0.0180)
- pi_5: 0.9529 (var=0.0191)
- pi_1: 0.9048 (var=0.0084)
- pi_2: 0.8735 (var=0.0093)
- pi_3: 0.2390 (var=0.0208)
- pi_4: 0.8613 (var=0.0179)

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
  - iter 1: 0.1994 (var=0.0635) (Δ vs real -0.0969)
  - iter 2 (most recent): 0.1988 (var=0.0693) (Δ vs real -0.0975)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1713 (var=0.0509)
- pi_3_1: 0.4531 (var=0.0906)
- pi_1: 0.0181 (var=0.0129)
- pi_2: 0.0131 (var=0.0090)
- pi_3: 0.3713 (var=0.0215)
- pi_4: 0.1175 (var=0.0341)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Context-Normalized Noisy Anti-Validity Integration (CNNAI), recalibrated. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity (deterministically redrawn whenever the endorsed-cue set changes, hashed per subject and context). The stable cross-context component of eta is deliberately tiny (sd 0.10) and the per-context redraw large (sd 0.50), so behavior is steeply anti-validity within a context but nearly independent across validity contexts (cross-ladder r ~ 0.2). Response extremity (beta) is also nearly homogeneous across engaged subjects (lognormal sd 0.25), so no hidden stable trait couples contexts. Per-trial multiplicative attention diffusion (lognormal sd 0.30 per cue weight) supplies within-context stochasticity that inflates between-subject composite variance without shifting means. Choice is softmax(beta * values) plus a symmetric lapse epsilon, with a 13% true-guesser subpopulation (beta ~ 0) uncorrelated with eta.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Context-Normalized Noisy Anti-Validity Integration (CNNAI),
    # iter 2 -- minimal-diff recalibration of the accepted iter-1 base
    # per the iter-1 critic diagnosis. Mechanism family UNCHANGED:
    # additive anti-validity integration, context-referenced elasticity
    # w_j = -v_j * (v_j / v_ref)^eta, per-trial attention diffusion,
    # softmax + symmetric lapse, guesser mixture. Only the population
    # calibration constants move:
    #   (A) decouple the ladders: stable eta component sd 0.25 -> 0.10,
    #       per-context redraw sd 0.36 -> 0.50, engaged beta lognormal
    #       sd 0.45 -> 0.25 (clip tightened to [1.0, 5.0]) so stable
    #       response extremity no longer correlates contexts;
    #   (B) restore between-subject variance: per-trial attention
    #       diffusion sd 0.15 -> 0.30, engaged epsilon widened to
    #       0.02 .. 0.17;
    #   (C) shift the eta center modestly upward 0.19 -> 0.26 (still
    #       below the ~0.31-0.34 kill knife) and trim the guesser
    #       fraction 16% -> 13% to protect the Exp-1/Exp-3 guardrails.
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
        eta_base = 0.26                  # irrelevant at beta ~ 0
    else:
        # Engaged subpopulation (~87%). TINY stable elasticity
        # component (the only cross-context coupling, sd 0.10 so
        # cross-ladder consistency stays near r ~ 0.2), nearly
        # homogeneous beta (lognormal sd 0.25 so response extremity
        # does not act as a hidden trait), widened lapse.
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.26 + 0.10 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.35), 0.85)

        beta = math.exp(0.88 + 0.25 * float(ndtri(bs)))  # median ~2.4
        beta = min(max(beta, 1.0), 5.0)
        epsilon = 0.02 + 0.15 * ps       # 0.02 .. 0.17

    # --- Context: the set of cues endorsed on this trial. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    # Deterministic per-(subject, context) elasticity redraw: the same
    # cell (including A/B mirrors, which share the active set) always
    # elicits the same eta within a subject; different cells elicit
    # independent etas. The redraw sd is now 0.50 (up from 0.36) so the
    # per-context steepness dominates the tiny stable component.
    key = ",".join(str(int(i)) for i in active)
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    eta_ctx = eta_base + 0.50 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.90), 2.40)

    # Context-normalized anti-validity weights: v_ref is the mean
    # validity of the cues actually displayed (endorsed) this trial.
    v_ref = float(np.mean(val[active]))
    w = -val * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion on cue weights
    # (sd raised 0.15 -> 0.30 to inflate between-subject composite
    # variance toward the observed 0.06-0.09 without shifting means).
    att = np.exp(0.30 * np.random.randn(n_features))
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

`rationale`: This is a minimal-diff recalibration of the accepted CNNAI base, implementing all three of the critic's prescriptions while leaving the mechanism family untouched. (A) Breaking the hidden cross-ladder coupling: the iter-1 Exp-9 failure (r = 0.441 vs real 0.196) was diagnosed as stable per-subject components dominating the ladder composites once the per-cell context draws average out. I therefore shrink the stable elasticity component (eta_base sd 0.25 -> 0.10), enlarge the per-context redraw (sd 0.36 -> 0.50), and — critically — narrow the engaged beta distribution (lognormal sd 0.45 -> 0.25, clip [1.0, 5.0]) so heterogeneous response extremity can no longer act as a latent trait correlating the two ladders. The only remaining coupling is the tiny eta_base, pinning cross-ladder r near the observed ~0.2. (B) Restoring between-subject variance: the per-trial attention diffusion sd is raised 0.15 -> 0.30 and the engaged epsilon range widened to 0.02-0.17; both add stochasticity on near-knife cells (where the composites live) without shifting the population means, targeting the ~2x variance underprediction on Exps 2/5/6/8. (C) Lifting the steepness composites: the eta center moves 0.19 -> 0.26, still below the ~0.31-0.34 kill knife, so more contexts sit above the kill flip (Exps 2/5/6/7 and the Exp-10 rise should move toward their real values) while ~30% of contexts still fall below gamma = 1 for the Exp-10 twin cells. As the critic advised, the guesser fraction is trimmed 16% -> 13% to protect the Exp-1/Exp-3 guardrails against the added steepness rather than reversing the eta shift. Everything else — the additive anti-validity core, the context hashing, the softmax/lapse structure, the parameter set — is byte-identical to the accepted base.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate faithfully implements the prescribed CNNAI family (additive anti-validity integration, context-referenced elasticity, per-trial attention diffusion, guesser mixture) and its aggregate loss (0.1070) made it the accepted base. Per-experiment, however, there is a clear and systematic pattern of residual error rather than random noise. (1) The headline mechanism — near-zero cross-context consistency — FAILS: Exp 9 cross-ladder r = 0.441 vs real 0.196, worse than plain SCWI (pi_3: 0.239). Diagnosis: the subject-level ladder composites sL and sM average over many cells, so the per-cell context redraw (sd 0.36) averages out within each ladder, leaving the STABLE per-subject components (eta_base sd 0.25, and especially the heterogeneous beta with lognormal sd 0.45) to dominate between-subject differences. Stable beta is a hidden trait: high-beta subjects are more extreme on BOTH ladders, producing spurious cross-ladder correlation even with independent etas. (2) Between-subject variances are systematically underpredicted by roughly a factor of 2 on the diagnostic composites: Exp 2 var 0.025 vs 0.093, Exp 5 0.031 vs 0.064, Exp 6 0.033 vs 0.070, Exp 8 0.033 vs 0.065. The per-trial attention diffusion (sd 0.15) is too weak, and the averaging-out of per-cell eta draws suppresses exactly the variance the redraw was supposed to create. (3) The anti-validity steepness composites all undershoot in the same direction: Exp 2 (0.186 vs 0.274), Exp 5 (0.600 vs 0.684), Exp 6 (0.580 vs 0.630), Exp 7 (0.557 vs 0.646), Exp 10 rise (0.199 vs 0.296). The proposer deliberately centered eta at 0.19, below the kill-knife at ~0.31–0.34; the real data want more probability mass above the knife. Exps 1, 3, 4, 8 are essentially on target (slopes/composites within 0.01–0.03 of real), confirming the shared anti-validity core is sound.
Rationale: Regenerate, staying strictly inside the CNNAI family, with three targeted recalibrations. (A) Break the hidden cross-ladder coupling: shrink the stable elasticity component (eta_base sd 0.25 → ~0.10) and enlarge the per-context redraw (sd 0.36 → ~0.50), AND narrow the engaged beta distribution (lognormal sd 0.45 → ~0.25, or cap beta heterogeneity) so stable response extremity no longer correlates the two ladders. This should pull Exp 9 from 0.44 toward the observed 0.196 without touching the anti-validity core. (B) Restore between-subject variance: raise the per-trial attention diffusion (sd 0.15 → ~0.30) and/or let epsilon vary more across engaged subjects; the composites need var ≈ 0.06–0.09, not 0.03. Note the tension and its resolution: increasing per-context eta spread raises within-context variance while LOWERING cross-context correlation — both desired — whereas the current parameterization does neither because the context noise averages out over cells. (C) Shift the eta center modestly upward (0.19 → ~0.25–0.28, still below the 0.31–0.34 knife) so more contexts sit above the kill flip, lifting Exps 2/5/6/7 and the Exp 10 rise toward their real values while keeping enough mass below gamma = 1 for the twin cells. Watch Exp 3 (already slightly high at 0.139 vs 0.123) and Exp 1 slope as guardrails; if they overshoot, compensate by trimming the guesser fraction slightly (16% → ~13%) rather than reversing the eta shift. All changes are parameter recalibrations within the prescribed mechanism — no family switch needed.

**Outcome of this advice:** iter 1 candidate loss=0.1070 -> iter 2 candidate loss=0.1032 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The iter-2 recalibration did exactly what my iter-1 critique asked on its primary target, and the gate accepted it (loss 0.1070 -> 0.1032). Exp 9 cross-ladder r moved from 0.441 to 0.164 (real 0.196) — the headline CNNAI prediction of near-zero cross-context consistency is now essentially on target, and Exp 4 improved to within 0.01 of real. The narrowed beta, tiny stable eta component, and trimmed guesser fraction should be kept. However, the rest of the profile regressed or stagnated, and the pattern is diagnostic rather than noisy. (1) All anti-validity steepness composites UNDERSHOOT and got worse or flat vs iter 1: Exp 2 0.139 vs real 0.274 (error grew from -0.088 to -0.134), Exp 5 0.553 vs 0.684, Exp 6 0.523 vs 0.630, Exp 7 0.551 vs 0.646, Exp 8 0.558 vs 0.604, Exp 10 rise 0.199 vs 0.296 (unchanged despite the eta center moving 0.19 -> 0.26). (2) Between-subject variances got WORSE despite doubling the attention diffusion: Exp 5 var 0.031 -> 0.019 (real 0.064), Exp 6 0.033 -> 0.014 (real 0.070), Exp 2 stuck at 0.025 (real 0.093), Exp 8 0.030 (real 0.065). My iter-1 part-(B) prescription (inflate variance via per-trial diffusion) was wrong in mechanism, and I withdraw it: per-trial multiplicative noise is independent across trials, so it averages out within a subject's composite — it adds within-subject binomial-like noise, not between-subject dispersion. The same averaging-out logic dooms the steepness composites: the candidate redraws eta per endorsed-cue INDEX set, so each subject's composite averages over ~10-20 independent per-cell eta draws (many of which land near or below the kill knife, flipping or flattening individual cells), concentrating every subject near the population mean. This single design choice — per-cell context hashing — explains both failures simultaneously: diluted mean steepness AND suppressed between-subject variance, and it is also a drift from the arbiter sketch, which specified eta redrawn 'per context/block (e.g., between the 0.50-ladder and 0.60-ladder of Exp 1)', i.e., at the ladder/block level, not per trial cell.
Rationale: Regenerate, staying strictly inside the CNNAI family, with one structural recalibration plus parameter retuning. (A) COARSEN THE CONTEXT KEY: replace the hash over the endorsed-cue index set with a hash over the sorted set of distinct VALIDITY VALUES among the endorsed cues (equivalently, a bucketed v_ref / ladder identifier). All cells within one validity ladder then share a single per-subject eta draw: high-eta subjects are steep on every cell of that ladder, low-eta subjects shallow on every cell — restoring both the mean steepness composites (Exps 2/5/6/7/8/10) and the between-subject variance (target 0.06-0.09) in one move, because the composite is now driven by one shared draw instead of an average of many. Different ladders (Exp 9's {1.00,0.50} vs {1.00,0.60} cells; Exp 10's two ladders) remain distinct contexts, so cross-ladder decoupling and the hard-won Exp 9 r ~ 0.16-0.20 are preserved. (B) RETUNE: since pooling removes the dilution, keep the eta center at ~0.26 — do NOT raise it further; Exp 1 slope (-0.181 vs -0.175) and Exp 3 (0.145 vs 0.123) are already slightly steep and will steepen more under pooling, so be ready to trim the center toward ~0.22 if those guardrails overshoot. Reduce the per-trial attention diffusion back to sd ~0.15-0.20: at 0.30 it flattens psychometric composites (a contributor to the Exp 2 undershoot) and, as established, cannot create between-subject variance anyway. Keep the iter-2 constants that worked: narrow engaged beta (lognormal sd 0.25, clip [1,5]), tiny stable eta_base sd 0.10, 13% guessers, widened epsilon. (C) Verify against the variance targets explicitly: Exp 2 var 0.093, Exp 5 0.064, Exp 6 0.070, Exp 8 0.065 — if the shared-per-ladder eta still underdelivers, widen the per-context redraw sd (0.50 -> 0.60-0.70) rather than re-inflating beta or eta_base, since those were the coupling channels that broke Exp 9 in iter 1. All changes are within the prescribed CNNAI mechanism; no family switch.

**Outcome of this advice:** iter 2 candidate loss=0.1032 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Context-Normalized Noisy Anti-Validity Integration (CNNAI), ladder-pooled context key. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity. The CONTEXT is now the validity LADDER: eta is redrawn per (subject, distinct-validity-set-of-endorsed-cues), so all cells sharing a validity ladder (e.g., all {1.00, 0.50} stack comparisons) inherit ONE shared per-subject eta draw, while different ladders (e.g., {1.00, 0.50} vs {1.00, 0.60}) draw independently. This makes each subject's within-ladder composite steep and dispersed (driven by a single shared draw rather than an average of many independent per-cell draws), restoring both mean steepness and between-subject variance on the diagnostic composites, while preserving near-zero cross-ladder consistency (Exp 9 r ~ 0.2). The stable cross-context component of eta remains tiny (sd 0.10), engaged beta is nearly homogeneous (lognormal sd 0.25), per-trial multiplicative attention diffusion (sd 0.18) supplies within-context stochasticity, and choice is softmax(beta * values) plus a symmetric lapse epsilon with a 13% true-guesser subpopulation uncorrelated with eta.",
  "predict": "def predict(parameters, state, history):\n    # Context-Normalized Noisy Anti-Validity Integration (CNNAI),\n    # iter 3 -- minimal-diff structural recalibration of the accepted\n    # iter-2 base per the iter-2 critic diagnosis. Mechanism family\n    # UNCHANGED: additive anti-validity integration, context-referenced\n    # elasticity w_j = -v_j * (v_j / v_ref)^eta, per-trial attention\n    # diffusion, softmax + symmetric lapse, guesser mixture. Changes:\n    #   (A) CONTEXT KEY COARSENED: the per-(subject, context) eta redraw\n    #       is now keyed on the SORTED SET OF DISTINCT VALIDITY VALUES\n    #       among the endorsed cues (the ladder identifier), NOT the\n    #       endorsed-cue index set. All cells within one validity ladder\n    #       share a single per-subject eta draw, so each subject's\n    #       ladder composite is driven by one shared draw instead of an\n    #       average of ~10-20 independent per-cell draws -- restoring\n    #       mean steepness (Exps 2/5/6/7/8/10) and between-subject\n    #       variance in one move, while different ladders (Exp 9's\n    #       {1.00,0.50} vs {1.00,0.60}; Exp 10's two ladders) remain\n    #       independent contexts, preserving cross-ladder decoupling.\n    #   (B) RETUNE: eta center trimmed 0.26 -> 0.23 (pooling steepens\n    #       the Exp-1/Exp-3 guardrails, which were already slightly\n    #       steep); per-context redraw sd widened 0.50 -> 0.60 to hit\n    #       the variance targets (0.06-0.09) now that the shared draw\n    #       actually reaches the composites; per-trial attention\n    #       diffusion reduced 0.30 -> 0.18 (it cannot create\n    #       between-subject variance and was flattening the Exp-2\n    #       psychometric composite).\n    #   (C) KEPT from iter-2: narrow engaged beta (lognormal sd 0.25,\n    #       clip [1.0, 5.0]), tiny stable eta_base sd 0.10, 13%\n    #       guessers, widened engaged epsilon 0.02-0.17.\n    import hashlib\n    import math\n    import numpy as np\n    from scipy.special import ndtri\n\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"CNNAI expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    n_features = stim.shape[1]\n    ra, rb = stim[0], stim[1]\n\n    def _to_vec(raw):\n        if raw is None:\n            return None\n        if isinstance(raw, str):\n            s = raw.strip().strip('[]()')\n            toks = [t for t in s.replace(',', ' ').split() if t != '']\n            if not toks:\n                return None\n            return np.array([float(t) for t in toks])\n        arr = np.asarray(raw, dtype=float).ravel()\n        return arr if arr.size > 0 else None\n\n    val = _to_vec(parameters.get(\"validities\"))\n    if val is None or val.size != n_features:\n        # Defensive fallback: uniform mid-range validities.\n        val = np.full(n_features, 0.75)\n    val = np.clip(val, 1e-3, 1.0)\n\n    def _seed(name):\n        try:\n            x = float(parameters[name])\n        except (KeyError, TypeError, ValueError):\n            x = 0.5\n        if not np.isfinite(x):\n            x = 0.5\n        return min(max(x, 0.0), 1.0)\n\n    u = _seed(\"regime_seed\")\n    es = _seed(\"eta_base_seed\")\n    bs = min(max(_seed(\"beta_seed\"), 1e-6), 1.0 - 1e-6)\n    ps = _seed(\"epsilon_seed\")\n    ss = _seed(\"subject_seed\")\n\n    # --- Population mixture (uncorrelated with eta). ---\n    if u < 0.13:\n        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on\n        # every trial; elevated lapse. These subjects sit at ~0.5 on\n        # every diagnostic cell and dilute population means/slopes.\n        t = u / 0.13\n        beta = 0.02 + 0.08 * t            # 0.02 .. 0.10\n        epsilon = 0.12 + 0.13 * ps       # 0.12 .. 0.25\n        eta_base = 0.23                  # irrelevant at beta ~ 0\n    else:\n        # Engaged subpopulation (~87%). TINY stable elasticity\n        # component (the only cross-context coupling, sd 0.10 so\n        # cross-ladder consistency stays near r ~ 0.2), nearly\n        # homogeneous beta (lognormal sd 0.25 so response extremity\n        # does not act as a hidden trait), widened lapse.\n        zs = min(max(es, 1e-6), 1.0 - 1e-6)\n        eta_base = 0.23 + 0.10 * float(ndtri(zs))\n        eta_base = min(max(eta_base, -0.35), 0.85)\n\n        beta = math.exp(0.88 + 0.25 * float(ndtri(bs)))  # median ~2.4\n        beta = min(max(beta, 1.0), 5.0)\n        epsilon = 0.02 + 0.15 * ps       # 0.02 .. 0.17\n\n    # --- Context: the validity LADDER of the cues endorsed this trial. ---\n    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))\n    if active.size == 0:\n        # Nothing endorsed by either option: no evidence either way.\n        return np.full(2, 0.5)\n\n    # COARSENED context key: the sorted set of DISTINCT VALIDITY VALUES\n    # among the endorsed cues (a ladder identifier). All cells sharing\n    # one validity ladder (including A/B mirrors, which share the\n    # endorsed validity set) inherit the SAME per-subject eta draw, so\n    # a subject's whole ladder is steep or shallow together; different\n    # ladders draw independently. The redraw sd is 0.60 so the\n    # per-ladder steepness dominates the tiny stable component and\n    # produces between-subject composite variance near 0.06-0.09.\n    ladder_vals = sorted(set(float(np.round(v, 4)) for v in val[active]))\n    key = \",\".join(f\"{v:.4f}\" for v in ladder_vals)\n    h = int(hashlib.md5(f\"{ss:.9f}|{key}\".encode(\"utf-8\")).hexdigest()[:14], 16)\n    uc = ((h % 999983) + 0.5) / 999984.0\n    eta_ctx = eta_base + 0.60 * float(ndtri(uc))\n    eta_ctx = min(max(eta_ctx, -0.90), 2.40)\n\n    # Context-normalized anti-validity weights: v_ref is the mean\n    # validity of the cues actually displayed (endorsed) this trial.\n    v_ref = float(np.mean(val[active]))\n    w = -val * np.power(val / v_ref, eta_ctx)\n\n    # Per-trial multiplicative attention diffusion on cue weights\n    # (sd reduced 0.30 -> 0.18: it supplies within-context stochasticity\n    # without shifting means, but at 0.30 it was flattening the Exp-2\n    # psychometric composite and cannot create between-subject\n    # variance, which now comes from the shared per-ladder eta draw).\n    att = np.exp(0.18 * np.random.randn(n_features))\n    w = w * att\n\n    scores = np.array([float(np.dot(w, ra)), float(np.dot(w, rb))])\n\n    # Numerically stable softmax over the two option values.\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n\n    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # Guard against float drift.\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "subject_seed": "[0, 1]",
    "regime_seed": "[0, 1]",
    "eta_base_seed": "[0, 1]",
    "beta_seed": "[0, 1]",
    "epsilon_seed": "[0, 1]",
    "validities": "validities"
  },
  "rationale": "This is a minimal-diff edit of the accepted iter-2 CNNAI base implementing the iter-2 critic's single structural diagnosis. (A) COARSENED CONTEXT KEY: the iter-2 failures — steepness composites undershooting (Exp 2: 0.139 vs 0.274; Exp 5: 0.553 vs 0.684; Exp 6: 0.523 vs 0.630; Exp 7: 0.551 vs 0.646; Exp 8: 0.558 vs 0.604; Exp 10 rise: 0.199 vs 0.296) AND between-subject variance collapsing (Exp 5 var 0.019 vs 0.064; Exp 6 0.014 vs 0.070; Exp 2 0.025 vs 0.093; Exp 8 0.030 vs 0.065) — share one root cause: keying the eta redraw on the endorsed-cue INDEX set makes each subject's composite an average over ~10-20 independent per-cell draws, concentrating every subject near the population mean. Replacing the index-set hash with a hash over the sorted set of DISTINCT VALIDITY VALUES among the endorsed cues (the ladder identifier, exactly the 'per context/block' granularity the arbiter sketch specified) makes each subject's within-ladder composite driven by ONE shared draw: high-eta subjects are steep on every cell of that ladder, low-eta subjects shallow on every cell — restoring mean steepness and between-subject dispersion simultaneously. Different ladders (Exp 9's {1.00,0.50} vs {1.00,0.60}; Exp 10's two ladders) remain distinct contexts, so the hard-won Exp 9 cross-ladder r ~ 0.16-0.20 is preserved. (B) RETUNE per the critic's explicit guidance: eta center trimmed 0.26 -> 0.23 because pooling steepens the already-slightly-steep Exp-1/Exp-3 guardrails; per-context redraw sd widened 0.50 -> 0.60 (the critic's designated variance lever, chosen over re-inflating beta or eta_base, which were the coupling channels that broke Exp 9 in iter 1); per-trial attention diffusion reduced 0.30 -> 0.18 since the critic correctly withdrew the iter-1 variance prescription — independent per-trial noise averages out within a subject's composite and at 0.30 was flattening the Exp-2 psychometric rise. (C) KEPT the validated iter-2 constants: narrow engaged beta (lognormal sd 0.25, clip [1.0, 5.0]), tiny stable eta_base sd 0.10, 13% guessers, widened epsilon 0.02-0.17. Everything else — the additive anti-validity core, the v_ref range normalization, the softmax/lapse structure, the parameter set — is byte-identical to the accepted base."
}
```
```

## Usage

```json
{
  "input_tokens": 50747,
  "output_tokens": 5756,
  "reasoning_tokens": 2592,
  "provider": "Sail Research"
}
```
