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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace pi_6 with 'Consensus-Relative Edge-Distrust Tallying' (CREDT). Keep the shared, well-supported architecture (sign votes × validity weights × low-gain softmax × lapse) — that scaffold is endorsed by every experiment. Replace the weight function with three components: (1) MID-BAND PLATEAU WITHOUT A CLIFF: all cues with stated validity in [v_lo, v_hi] (v_lo ≈ 0.55, v_hi ≈ 0.96) receive approximately equal positive weight w_p ≈ 1 — pure tally behavior over diagnostic cues (recovers Exps 2-6, 10). There is NO absolute 0.98/0.99 discontinuity; adjacent high validities are near-exchangeable. (2) CHANCE-FLANK: a small negative weight for v near 0.5, w(0.5) = -kappa_lo with kappa_lo in [0.15, 0.25] — the one BSST component the data confirm (Exp 7: observed -0.168). (3) THE NOVEL PIECE — OUTLIER-GAP DISTRUST: the subjective weight of the top expert(s) is modulated by the gap between the highest stated validity and the next-highest validity in the panel (computable from parameters['validities']). Small gap (dense ladder, e.g., 0.99 atop 0.98/0.95/0.90 as in Exp 2) → the top cue keeps ~full plateau weight (predicts follow-99 ≈ 0.58-0.62, matching observed 0.599). Large gap (isolated outlier, e.g., 0.99 atop 0.75 as in Exps 8/9) → its weight collapses toward zero or slightly negative via w_top = w_p - rho * sigmoid((gap - g0)/s), with g0 ≈ 0.05-0.10, rho allowing the asymptote to reach [-0.6, +0.1] (predicts Exp 8 strong-side rate ≈ 0.08-0.20 near the lapse floor, and Exp 9 anti-99 ≈ 0.52-0.60, matching observed 0.580). Suggested parameters: w_p ∈ [0.8, 1.2], v_lo ∈ [0.52, 0.58], v_hi ∈ [0.95, 0.97], kappa_lo ∈ [0.15, 0.25], g0 ∈ [0.04, 0.12], s ∈ [0.02, 0.06], rho ∈ [0.9, 1.7], beta ∈ [0.30, 0.50], epsilon ∈ [0.03, 0.08]. This theory differs from BSST by making distrust of certainty RELATIVE to the panel consensus rather than an absolute cliff, and differs from FDEI by (i) keeping chance cues anti-diagnostic rather than annihilated and (ii) capping the influence of isolated super-experts rather than giving them the largest positive weight. It should simultaneously recover the tally core (Exps 2-6), the chance-flank negativity (Exp 7), the isolated-outlier distrust (Exps 1, 8, 9), and 98/90 exchangeability (Exp 10), making it a genuine task-invariant competitor to FDEI.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.686)

**Description**
Validity-Weighted Tallying (graded-evidence regime): people integrate evidence from ALL binary expert ratings, casting each feature a sign vote (+1 win / -1 loss / 0 tie) weighted by w_j = validity_j^gamma, summing into a zero-sum evidence score, with choice a softmax (inverse temperature beta) over the two scores plus an independent uniform lapse (epsilon). The theoretical claim added this round is about where humans sit in this family: in the UNSATURATED softmax regime. Because the evidence scores are integer-scale sign tallies, beta >= 3 already drives tanh(beta*m) ~ 1 for every unit margin, making the model behaviorally deterministic and pinning it to the tallying corner; the graded dose-response in tally margin that both experiments demand (Exp 1 metric at 44% of the deterministic-tally value; Exp 2's small positive b_d) lives at beta ~ 0.2-0.3, where p(A) is approximately linear in the tally margin. Gamma stays in a mild, provably sub-lexicographic band so that no subject leaps into the TTB regime.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Validity-Weighted Tallying.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a vote sign(a_j - b_j) for A's score and
    # the opposite for B's score; votes are weighted by w_j =
    # validity_j ** gamma. gamma = 0 -> unweighted Tallying;
    # gamma -> large -> lexicographic TTB (top cue dominates).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Validity-based weights. Since validities are in [0.5, 1.0],
    # gamma = 0 gives all-ones weights (pure Tallying); increasing
    # gamma increasingly privileges high-validity cues, with the
    # top cue dominating as gamma grows (TTB limit).
    w = np.power(val, gamma)

    # Weighted evidence scores. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable softmax over the two weighted scores.
    # When all cues tie (s_a == 0) the softmax is exactly uniform,
    # which is the correct behavior for an undiscriminating tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- gamma: [0.05, 0.7]
- beta: [0.03, 0.55]
- epsilon: [0.0, 0.08]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.2180 (var=0.0140) vs this=-0.2182 (var=0.0156)
- Experiment 2: real=0.0771 (var=0.0031) vs this=0.0861 (var=0.0026)
- Experiment 3: real=0.6170 (var=0.0034) vs this=0.6456 (var=0.0096)
- Experiment 4: real=0.5133 (var=0.0042) vs this=0.6062 (var=0.0085)
- Experiment 5: real=-0.1683 (var=0.0124) vs this=0.2958 (var=0.0606)
- Experiment 6: real=0.1180 (var=0.0074) vs this=0.4190 (var=0.0107)
- Experiment 7: real=0.5796 (var=0.0066) vs this=0.3831 (var=0.0085)
- Experiment 8: real=0.4625 (var=0.0103) vs this=0.6062 (var=0.0141)
- Experiment 9: real=0.5600 (var=0.0052) vs this=0.5220 (var=0.0036)
- Experiment 10: real=0.5988 (var=0.0026) vs this=0.6206 (var=0.0104)


---

### `pi_4` (overall score: 0.562)

**Description**
Flattened-Diagnosticity Evidence Integration (the extreme-compression corner of the Bayesian cue-counting family): people integrate ALL binary expert ratings into one evidence score, where each discriminating cue casts a sign vote weighted by w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta — chance cues (v=0.5) are annihilated exactly, but among genuinely diagnostic cues the subjective weight is a HEAVILY flattened (delta ~ 0.15) version of the normative log-odds, and choice is a low-gain softmax (2*beta ~ 0.6) plus a small uniform lapse. The new theoretical claim is that the compression exponent delta is itself the structural margin-compression device: shrinking delta equalizes weights, which GROWS count-dominated conflict margins (where cues split across options) while SHRINKING alignment-dominated cumulative margins (where several strong cues pile onto one side). This differential action — not any global gain knob, and not trial-level normalization — is what lets a single small beta simultaneously produce moderate TTB-leaning conflict-trial behavior (Exp 1), a slightly positive tally-vs-top-cue regression slope (Exp 2), graded ~0.62 tally-following (Exp 3), and near-chance |d|=1 agreement driven by the cancellation between super-expert matches and super-expert reversals (Exp 4).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Diagnosticity-Weighted Evidence Integration (Bayesian cue-counting).
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by the cue's log-likelihood-ratio diagnosticity
    # llr_j = log(v_j / (1 - v_j)), rectified by an attention floor
    # tau and compressed by an exponent delta:
    #   w_j = (max(llr_j - tau, 0))^delta
    # A v = 0.5 expert has llr = 0 -> exactly zero evidence.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    delta = float(parameters["delta"])
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Normative diagnosticity: log-likelihood-ratio of each expert.
    # Clip away from v = 1.0 to keep the log finite; v = 0.5 maps to 0.
    v = np.clip(val, 0.5, 1.0 - 1e-9)
    llr = np.log(v / (1.0 - v))

    # Limited-attention floor: diagnosticity below tau is discarded;
    # the remainder is compressed by the exponent delta. delta = 1 is
    # the purely normative Bayesian weight; delta < 1 flattens the
    # diagnosticity gradient toward tallying over informative cues
    # (while v = 0.5 cues stay annihilated at any delta, since
    # llr(0.5) = 0 <= tau); delta > 1 sharpens it toward
    # lexicographic/TTB-like use of the best experts.
    w = np.power(np.maximum(llr - tau, 0.0), delta)

    # Weighted evidence score. B's score is the mirror of A's because
    # every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable low-temperature softmax. When all attended
    # cues tie (s_a == 0) the softmax is exactly uniform, which is the
    # correct behavior for undiscriminating evidence.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- delta: [0.1, 0.2]
- tau: [0.0, 0.04]
- beta: [0.27, 0.37]
- epsilon: [0.0, 0.05]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.2180 (var=0.0140) vs this=-0.2188 (var=0.0064)
- Experiment 2: real=0.0771 (var=0.0031) vs this=0.0881 (var=0.0019)
- Experiment 3: real=0.6170 (var=0.0034) vs this=0.6485 (var=0.0027)
- Experiment 4: real=0.5133 (var=0.0042) vs this=0.5192 (var=0.0023)
- Experiment 5: real=-0.1683 (var=0.0124) vs this=0.0308 (var=0.0233)
- Experiment 6: real=0.1180 (var=0.0074) vs this=0.7670 (var=0.0039)
- Experiment 7: real=0.5796 (var=0.0066) vs this=0.3191 (var=0.0050)
- Experiment 8: real=0.4625 (var=0.0103) vs this=0.5400 (var=0.0192)
- Experiment 9: real=0.5600 (var=0.0052) vs this=0.3985 (var=0.0055)
- Experiment 10: real=0.5988 (var=0.0026) vs this=0.6719 (var=0.0085)


---

### `pi_6` (overall score: 0.529)

**Description**
Broad-Spectrum Skepticism Tallying (BSST), low-gain/held-cliff calibration. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, where the subjective weight w(v) is a wide inverted-U over STATED validity with skepticism on both flanks: (i) chance experts (v = 0.5) carry a small negative weight (~0.23 trusted-cue units) — a coin-flipper's endorsement is mildly anti-diagnostic; (ii) the entire band v ~ [0.55, 0.95] is a flat trust plateau (~0.9), making mid-band experts nearly exchangeable and generating moderate unweighted-tally behavior; (iii) distrust of claimed certainty is CONCENTRATED in a very sharp cliff between 0.98 and 0.99: w(0.98) stays mildly positive while w(0.99) plunges to ~ -1.1 trusted-cue units. The calibration claim added this round is about the GAIN-FLANK DECOUPLING: the softmax gain beta can be lowered (0.34) while the chance-flank magnitude kappa_lo is raised (0.23) so that the product beta*kappa_lo (~0.077) — the quantity that pins the chance-cue signature — is held invariant, but the product beta*|w(0.99)| (~0.37) is allowed to FALL because kappa_hi is held at its iter-2 value rather than deepened. This is the one combination never tried in nine iterations: every prior beta reduction (iters 5, 9) was bundled with a kappa_hi increase that exactly cancelled the gain on the 0.99-driven cells.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Broad-Spectrum Skepticism Tallying (BSST), sharp-late-cliff edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by a wide inverted-U 'skepticism' function of the
    # expert's STATED validity v:
    #   v <= v_rise :  w = -kappa_lo * (v_rise - v)/(v_rise - 0.5)
    #                  (chance experts: linear lobe, w(0.5) = -kappa_lo;
    #                   a coin-flipper's endorsement is mildly
    #                   anti-diagnostic noise)
    #   v >  v_rise :  w = w_plateau - (w_plateau + kappa_hi)
    #                        * sigmoid((v - v_c)/s_hi)
    #                  (flat trust plateau over the mid-band -- the
    #                   logistic is ~0 for every v <= 0.95, so all
    #                   mid-band experts get the full plateau weight;
    #                   then an extremely SHARP distrust cliff packed
    #                   between 0.98 and 0.99: w(0.98) stays mildly
    #                   positive, w(0.99) plunges to ~ -1.1 trusted-cue
    #                   units, saturating at -kappa_hi as v -> 1)
    # Votes are summed into a zero-sum evidence score; choice is a
    # low-gain softmax with inverse temperature beta plus a uniform
    # lapse epsilon.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    v_rise = float(parameters["v_rise"])
    w_plateau = float(parameters["w_plateau"])
    v_c = float(parameters["v_c"])
    s_hi = float(parameters["s_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_hi = float(parameters["kappa_hi"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)
    w = np.empty(n_features, dtype=float)

    # (i) Low flank: chance experts carry a small negative weight,
    # linearly reaching -kappa_lo at v = 0.5 and 0 at v_rise.
    lo = v < v_rise
    denom_lo = max(v_rise - 0.5, 1e-6)
    w[lo] = -kappa_lo * np.clip(v_rise - v[lo], 0.0, None) / denom_lo

    # (ii)+(iii) Flat trust plateau followed by an extremely sharp
    # distrust cliff concentrated between v = 0.98 and v = 0.99.
    # In the mid-band the logistic term is ~0, so w ~ w_plateau
    # (nearly equal weights -> tally-like counting); at 0.98 the
    # weight is only mildly discounted, at 0.99 it is strongly
    # anti-diagnostic, and it saturates at -kappa_hi as v -> 1.
    hi = ~lo
    # Numerically stable logistic.
    x = (v[hi] - v_c) / s_hi
    sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                   np.exp(x) / (1.0 + np.exp(x)))
    w[hi] = w_plateau - (w_plateau + kappa_hi) * sig

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- v_rise: [0.515, 0.54]
- w_plateau: [0.88, 0.93]
- v_c: [0.9855, 0.9875]
- s_hi: [0.0025, 0.0033]
- kappa_lo: [0.22, 0.235]
- kappa_hi: [1.68, 1.72]
- beta: [0.33, 0.35]
- epsilon: [0.05, 0.07]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.2180 (var=0.0140) vs this=-0.1973 (var=0.0076)
- Experiment 2: real=0.0771 (var=0.0031) vs this=0.0956 (var=0.0023)
- Experiment 3: real=0.6170 (var=0.0034) vs this=0.4811 (var=0.0035)
- Experiment 4: real=0.5133 (var=0.0042) vs this=0.3850 (var=0.0041)
- Experiment 5: real=-0.1683 (var=0.0124) vs this=-0.1350 (var=0.0257)
- Experiment 6: real=0.1180 (var=0.0074) vs this=0.4360 (var=0.0065)
- Experiment 7: real=0.5796 (var=0.0066) vs this=0.6756 (var=0.0042)
- Experiment 8: real=0.4625 (var=0.0103) vs this=0.4213 (var=0.0156)
- Experiment 9: real=0.5600 (var=0.0052) vs this=0.8115 (var=0.0019)
- Experiment 10: real=0.5988 (var=0.0026) vs this=0.2988 (var=0.0058)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.7165 -> ACCEPTED
- iter 2: loss=0.1676 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.1676 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    rows_a = data["option_a_ratings"].tolist()
    rows_b = data["option_b_ratings"].tolist()
    resp = data["response"].tolist()
    n_rows = len(rows_a)
    contribs = []
    for i in range(n_rows):
        ca = rows_a[i]
        cb = rows_b[i]
        if isinstance(ca, str):
            ca = ca.replace("[", " ").replace("]", " ").replace(",", " ").split()
        if isinstance(cb, str):
            cb = cb.replace("[", " ").replace("]", " ").replace(",", " ").split()
        a = [int(float(v)) for v in ca]
        b = [int(float(v)) for v in cb]
        r = int(resp[i])
        n = len(a)
        if n == 0 or n != len(b):
            continue
        ttb = -1
        for j in range(n):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb < 0:
            continue
        aw = 0
        bw = 0
        for j in range(n):
            if a[j] > b[j]:
                aw = aw + 1
            elif b[j] > a[j]:
                bw = bw + 1
        if aw == bw:
            if r == ttb:
                contribs.append(0.5)
            else:
                contribs.append(-0.5)
        else:
            if bw > aw:
                tal = 1
            else:
                tal = 0
            if tal == ttb:
                continue
            if r == ttb:
                contribs.append(1.0)
            else:
                contribs.append(-1.0)
    if len(contribs) == 0:
        return 0.0
    total = 0.0
    for c in contribs:
        total = total + c
    return float(total / len(contribs))
```

**Observed (real) value:** -0.2180 (var=0.0140)
**Previous candidate values (this loop):**
  - iter 1: -0.2672 (var=0.0083) (Δ vs real -0.0492)
  - iter 2 (most recent): -0.2757 (var=0.0056) (Δ vs real -0.0577)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6130 (var=0.0149)
- pi_2: -0.4425 (var=0.0124)
- pi_3: -0.2182 (var=0.0156)
- pi_4: -0.2188 (var=0.0064)
- pi_5: -0.4125 (var=0.0044)
- pi_6: -0.1973 (var=0.0076)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dual-dissociation regression metric.
    # Per trial, compute:
    #   d = tally margin  = (#features A wins) - (#features B wins)
    #   s = top-cue direction = sign(A[0] - B[0])   (feature 0 has the
    #       highest validity 0.92, and every designed pair discriminates on it)
    # Regress chose-A on [1, d, s] via OLS and return b_d - b_s.
    # Tallying: choice probability depends only on d  -> b_d > 0, b_s = 0 -> metric > 0.
    # TTB: choice probability depends only on s       -> b_d = 0, b_s > 0 -> metric < 0.
    # The two theories push this scalar in opposite directions, and the
    # partial regression cleanly removes the design's d-s correlation.
    df = data.reset_index(drop=True)

    A = np.stack([np.asarray(list(r), dtype=float) for r in df["option_a_ratings"].values])
    B = np.stack([np.asarray(list(r), dtype=float) for r in df["option_b_ratings"].values])

    # Tally margin: strict feature-wise wins; ties count for neither side.
    d = (A > B).sum(axis=1).astype(float) - (B > A).sum(axis=1).astype(float)

    # Direction of the highest-validity (top) cue, which always discriminates
    # in this design: +1 if it favors A, -1 if it favors B.
    s = np.sign(A[:, 0] - B[:, 0])

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = 1.0 - df["response"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(y)), d, s])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b_d = float(coef[1])
    b_s = float(coef[2])

    # Tallying's signature (slope in d) minus TTB's signature (slope in s).
    return b_d - b_s
```

**Observed (real) value:** 0.0771 (var=0.0031)
**Previous candidate values (this loop):**
  - iter 1: 0.1107 (var=0.0021) (Δ vs real +0.0336)
  - iter 2 (most recent): 0.1304 (var=0.0016) (Δ vs real +0.0533)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1196 (var=0.0023)
- pi_1: -0.3725 (var=0.0076)
- pi_3: 0.0861 (var=0.0026)
- pi_4: 0.0881 (var=0.0019)
- pi_5: 0.0789 (var=0.0021)
- pi_6: 0.0956 (var=0.0023)

### Experiment 3
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-following rate on decisive trials.

    For every trial, compute the raw (unweighted) tally margin
    d = (# cues where A wins) - (# cues where B wins).
    On trials with d != 0, check whether the subject's choice agrees
    with the option favored by the raw cue count (A if d > 0, B if
    d < 0). The metric is the fraction of such agreement over all
    decisive trials (d = 0 trials are excluded because the tallying
    heuristic itself is indifferent there).
    """
    df = data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return 0.5

    a_lists = df['option_a_ratings'].tolist()
    b_lists = df['option_b_ratings'].tolist()
    resp = df['response'].to_numpy(dtype=float)

    d = np.empty(n, dtype=float)
    for i in range(n):
        a = np.asarray(a_lists[i], dtype=float)
        b = np.asarray(b_lists[i], dtype=float)
        d[i] = float(np.sum(a > b) - np.sum(b > a))

    decisive = d != 0.0
    if not np.any(decisive):
        return 0.5

    dd = d[decisive]
    rr = resp[decisive]
    # Option favored by the raw tally: A (response 0) if d > 0,
    # otherwise B (response 1).
    agree = np.where(dd > 0.0, rr == 0.0, rr == 1.0)
    return float(np.mean(agree))
```

**Observed (real) value:** 0.6170 (var=0.0034)
**Previous candidate values (this loop):**
  - iter 1: n/a (var=n/a)
  - iter 2 (most recent): 0.6007 (var=0.0025) (Δ vs real -0.0163)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6456 (var=0.0096)
- pi_2: 0.8389 (var=0.0095)
- pi_1: 0.6944 (var=0.0071)
- pi_4: 0.6485 (var=0.0027)
- pi_5: 0.4996 (var=0.0026)
- pi_6: 0.4811 (var=0.0035)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Hard-ceiling test on the |d|=1 family (FACT 1 of the design).

    Metric = proportion of choices that agree with the UNWEIGHTED tally
    count-winner, restricted to trials whose raw tally margin is exactly
    +/-1 (the 8 trial types t1-t8; 48 of the 96 trials per subject).

    Tallying (any beta >= 1, any epsilon) predicts ~1 - epsilon/2 here,
    i.e. >= ~0.85 for any moderate lapse. The validity-weighted
    competitor is structurally capped: its choice probability on a |d|=1
    trial is (1-eps)*sigmoid(2*beta*m) with m = sum_j v_j^gamma*sign diff;
    the mean of |m| over the 8 |d|=1 trial types is <= 1 for every
    gamma in [0.05, 0.7] (equality only at gamma=0), and beta <= 0.55,
    so by concavity of tanh its mean agreement on this family is
    <= 0.5 + tanh(0.55)/2 = 0.750 over its ENTIRE parameter box.
    """
    def _mat(col):
        vals = []
        for v in col:
            if isinstance(v, np.ndarray):
                vals.append(v)
            else:
                vals.append(list(v))
        return np.asarray(vals, dtype=float)

    A = _mat(data["option_a_ratings"])
    B = _mat(data["option_b_ratings"])
    resp = np.asarray(data["response"], dtype=float)

    if A.ndim != 2 or A.shape[0] == 0 or A.shape != B.shape:
        return 0.5

    # Raw (unweighted) tally margin per trial.
    d = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # The |d| = 1 family: 8 unique pairs x 6 reps = 48 trials/subject.
    mask = np.abs(d) == 1
    if mask.sum() < 10:
        # Defensive fallback: use all non-tied trials.
        mask = d != 0
    if mask.sum() == 0:
        return 0.5

    dm = d[mask]
    rm = resp[mask]
    # Count-winner is A when d>0 (response 0), B when d<0 (response 1).
    consistent = np.where(dm > 0, rm == 0, rm == 1)
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.5133 (var=0.0042)
**Previous candidate values (this loop):**
  - iter 1: n/a (var=n/a)
  - iter 2 (most recent): 0.5058 (var=0.0033) (Δ vs real -0.0075)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8387 (var=0.0140)
- pi_3: 0.6062 (var=0.0085)
- pi_1: 0.6825 (var=0.0042)
- pi_4: 0.5192 (var=0.0023)
- pi_5: 0.3950 (var=0.0023)
- pi_6: 0.3850 (var=0.0041)

### Experiment 5
**Design**
  A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Chance-cue sensitivity on "chance-only" (Z-family) trials.
    # Design validities: [0.95, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5]
    # -> features 0-3 are diagnostic, features 4-7 are the four v=0.5 experts.
    # A Z-family trial is one where ALL diagnostic cues tie, so the pair is
    # discriminated ONLY by the 50%-accurate experts. On such a trial let
    # k = net number of chance cues won by A (signed). The competing theory
    # (llr-rectified weights) annihilates v=0.5 cues exactly, so p(A)=0.5
    # regardless of k -> subjects match the k-implied side at chance (metric 0).
    # My theory gives v=0.5 cues weight 0.5^gamma, so choices should follow
    # sign(k) above chance (metric > 0).
    DIAG = [0, 1, 2, 3]
    CHANCE = [4, 5, 6, 7]

    a_ratings = data["option_a_ratings"].apply(lambda x: np.asarray(list(x), dtype=int))
    b_ratings = data["option_b_ratings"].apply(lambda x: np.asarray(list(x), dtype=int))
    responses = data["response"].to_numpy()

    correct = []
    for av, bv, r in zip(a_ratings, b_ratings, responses):
        if av.shape[0] != 8 or bv.shape[0] != 8:
            continue
        # Keep only trials where every diagnostic cue ties (chance-only trials)
        if not np.array_equal(av[DIAG], bv[DIAG]):
            continue
        # Net chance-cue advantage for A
        k = int(np.sign(av[CHANCE] - bv[CHANCE]).sum())
        if k == 0:
            continue
        chose_a = (int(r) == 0)  # response: 0 = A, 1 = B
        followed = (k > 0 and chose_a) or (k < 0 and not chose_a)
        correct.append(1.0 if followed else 0.0)

    if len(correct) == 0:
        return 0.0
    # 2 * (accuracy - 0.5): 0 = chance-level following of chance cues,
    # positive = subjects systematically follow the 50%-accurate experts.
    return 2.0 * (float(np.mean(correct)) - 0.5)
```

**Observed (real) value:** -0.1683 (var=0.0124)
**Previous candidate values (this loop):**
  - iter 1: -0.1217 (var=0.0218) (Δ vs real +0.0467)
  - iter 2 (most recent): -0.1983 (var=0.0180) (Δ vs real -0.0300)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2958 (var=0.0606)
- pi_4: 0.0308 (var=0.0233)
- pi_1: 0.1625 (var=0.0145)
- pi_2: 0.7617 (var=0.0356)
- pi_5: -0.1167 (var=0.0195)
- pi_6: -0.1350 (var=0.0257)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Strong-Side Choice Rate on Multi-Weak Conflict trials (SSR).

    Selects the five conflict cells in which a high-validity side (99%
    experts, or 75% experts for R3) faces TWO OR MORE opposing weak
    experts: K2 (1x99 vs 2 weak), K3 (1x99 vs 3 weak), M3 (2x99 vs 3
    weak), M4 (2x99 vs 4 weak), R3 (2x75 vs 3 weak) -- 5 unique pairs
    x 2 mirrors x 4 reps = 40 trials per subject. Returns the
    proportion of trials on which the subject chose the strong side.
    """
    hits = 0
    n = 0

    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a, b, r in zip(a_col, b_col, r_col):
        a = [int(x) for x in a]
        b = [int(x) for x in b]
        if len(a) < 8 or len(b) < 8:
            continue

        # Validity-class vote balances (features 0-1 = 0.99 experts,
        # 2-3 = 0.75 experts, 4-7 = weak experts).
        d_hi = (a[0] + a[1]) - (b[0] + b[1])
        d_mid = (a[2] + a[3]) - (b[2] + b[3])
        d_lo = (a[4] + a[5] + a[6] + a[7]) - (b[4] + b[5] + b[6] + b[7])

        # K2/K3/M3/M4: super experts alone on one side (no 75s involved)
        # opposing >= 2 weak experts.
        if d_mid == 0 and d_hi != 0 and abs(d_lo) >= 2:
            target_is_a = d_hi > 0
        # R3: two 75% experts vs three weak experts.
        elif d_hi == 0 and abs(d_mid) == 2 and abs(d_lo) == 3:
            target_is_a = d_mid > 0
        else:
            # Excludes K1 (|d_lo|=1), R1, W2/W3 (no strong side),
            # W4a (|d_mid|=1), G1 (mixed 99+75 side), X1 (all tied).
            continue

        n += 1
        chose_a = (int(r) == 0)
        if chose_a == target_is_a:
            hits += 1

    if n == 0:
        return 0.5
    return hits / n

```

**Observed (real) value:** 0.1180 (var=0.0074)
**Previous candidate values (this loop):**
  - iter 1: n/a (var=n/a)
  - iter 2 (most recent): 0.6350 (var=0.0077) (Δ vs real +0.5170)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7670 (var=0.0039)
- pi_3: 0.4190 (var=0.0107)
- pi_1: 0.8475 (var=0.0076)
- pi_2: 0.1180 (var=0.0091)
- pi_5: 0.2870 (var=0.0027)
- pi_6: 0.4360 (var=0.0065)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Super-expert contrarianism index.

    Over all trials in which the feature-0 expert (the 99%-validity
    expert) discriminates between the two options (a[0] != b[0]),
    compute the proportion of choices that go AGAINST the option
    endorsed by that expert.

    Advocated theory (inverted trust): the 99% expert's endorsement is
    anti-diagnostic (w(0.99) ~ -1 trusted-cue unit), so subjects
    predominantly choose AGAINST it -> index ~0.8-0.9.

    Competing theory (flattened diagnosticity): the 99% expert gets the
    LARGEST positive weight, so subjects predominantly FOLLOW it ->
    index ~0.2-0.35.
    """
    n_anti = 0
    n_sel = 0
    for ra, rb, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(ra, dtype=int).ravel()
        b = np.asarray(rb, dtype=int).ravel()
        if a.size == 0 or b.size == 0 or a[0] == b[0]:
            continue  # expert 0 does not discriminate on this trial
        # Option index endorsed by the 99%-validity (feature-0) expert.
        endorsed = 0 if a[0] == 1 else 1
        r = int(resp)
        n_sel += 1
        if r != endorsed:
            n_anti += 1
    if n_sel == 0:
        return 0.5
    return float(n_anti) / float(n_sel)

```

**Observed (real) value:** 0.5796 (var=0.0066)
**Previous candidate values (this loop):**
  - iter 1: n/a (var=n/a)
  - iter 2 (most recent): 0.3462 (var=0.0035) (Δ vs real -0.2333)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7902 (var=0.0033)
- pi_4: 0.3191 (var=0.0050)
- pi_1: 0.1662 (var=0.0114)
- pi_2: 0.2569 (var=0.0106)
- pi_3: 0.3831 (var=0.0085)
- pi_6: 0.6756 (var=0.0042)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(follow the 98% expert | 98%-vs-90% conflict trials).

    A 'D-cell' trial is one where the top expert (v=0.98, feature 0) and the
    second expert (v=0.90, feature 1) endorse OPPOSITE options, while the
    third expert (v=0.75, feature 2) is tied. On these trials the metric is
    the fraction of choices that follow the option endorsed by the 98%
    expert. FDEI (compressed diagnosticity) treats 0.98 and 0.90 as nearly
    exchangeable, so this should be ~0.50 (coin flip). SITT (inverted trust)
    assigns the 98% expert a large NEGATIVE weight, so subjects should
    systematically choose AGAINST the 98% endorsement (~0.05-0.10).
    """
    if data is None or len(data) == 0:
        return 0.5

    follow_flags = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.ndim != 1 or a.shape[0] < 3 or a.shape != b.shape:
            continue

        d0 = a[0] - b[0]  # 98% expert's relative endorsement of A
        d1 = a[1] - b[1]  # 90% expert's relative endorsement of A
        d2 = a[2] - b[2]  # 75% expert must be tied (no interference)

        # Need a genuine 98-vs-90 conflict with the 75% expert tied.
        if d0 == 0 or d1 == 0 or d2 != 0:
            continue
        if np.sign(d0) == np.sign(d1):
            continue

        endorsed_a = d0 > 0          # 98% expert endorses option A
        chose_a = (int(row["response"]) == 0)
        follow_flags.append(1.0 if (endorsed_a == chose_a) else 0.0)

    if len(follow_flags) == 0:
        return 0.5
    return float(np.mean(follow_flags))
```

**Observed (real) value:** 0.4625 (var=0.0103)
**Previous candidate values (this loop):**
  - iter 1: n/a (var=n/a)
  - iter 2 (most recent): 0.4088 (var=0.0125) (Δ vs real -0.0538)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5400 (var=0.0192)
- pi_5: 0.0550 (var=0.0029)
- pi_1: 0.8688 (var=0.0163)
- pi_2: 0.6587 (var=0.0158)
- pi_3: 0.6062 (var=0.0141)
- pi_6: 0.4213 (var=0.0156)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Anti-super-expert choice rate on the 10 pre-registered 'cliff conflict'
    trials (S1/S1m, S2/S2m, D1/D1m, D2/D2m, V1/V1m).

    On each of these trials exactly one option is endorsed by the
    99%-validity experts (features 0 and 1) while the rival option is
    endorsed only by experts of validity <= 0.98, and the 99%-endorsed
    side has no 0.90-class support (this excludes the G1/G1m agreement
    anchors).  The metric is the fraction of these trials on which the
    subject chooses the option NOT endorsed by the 99% experts.

    BSST (sharp distrust cliff between 0.98 and 0.99, w(0.99) ~ -1.1
    trusted-cue units, w(0.98) ~ +0.6, w(0.90) ~ +0.9, w(0.50) ~ -0.23):
    the 99%-endorsed side is disfavored on every one of these cells, with
    predicted anti-99% choice rates of roughly S1 .86-.90, S2 .90-.95,
    D1 .72-.79, D2 .85-.94, V1 .70-.76  ->  overall ~0.84.

    FDEI (delta ~ 0.15 compression makes 0.98 and 0.99 experts nearly
    exchangeable, w99 - w98 in [-0.21, +0.15]): it is locked near 0.50 on
    the D cells, mildly PRO-99% on the S cells (the 99-side also carries
    a tiny validity edge), and strongly pro-99% on V1 (three chance cues
    are annihilated exactly)  ->  overall anti-99% rate ~0.30-0.53,
    center ~0.44.  Sign-disjoint from BSST with a ~0.40 mean gap.
    """

    def as_tuple(cell):
        try:
            return tuple([int(x) for x in cell])
        except Exception:
            return None

    # key: (A ratings, B ratings) -> favored option (0 = A, 1 = B),
    # where 'favored' is the side NOT endorsed by the 99% experts.
    TRIALS = {
        # S1 / S1m  (A or B wins both 0.99; rival wins both 0.90)
        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0), (0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0)): 1,
        ((0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0), (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0)): 0,
        # S2 / S2m  (one side wins 0.99+0.99+0.98; rival 0.90+0.90+0.55)
        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1), (0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1)): 1,
        ((0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1), (1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1)): 0,
        # D1 / D1m  (one side wins a 0.99; rival wins a 0.98)
        ((0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1)): 0,
        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)): 1,
        # D2 / D2m  (one side wins both 0.99; rival wins both 0.98)
        ((0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0), (1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0)): 0,
        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0), (0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0)): 1,
        # V1 / V1m  (one side wins a 0.99; rival wins all three 0.50)
        ((1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1)): 1,
        ((0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0)): 0,
    }

    hits = []
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a, b, r in zip(a_col, b_col, r_col):
        ka = as_tuple(a)
        kb = as_tuple(b)
        if ka is None or kb is None:
            continue
        fav = TRIALS.get((ka, kb))
        if fav is None:
            continue
        try:
            ri = int(r)
        except Exception:
            continue
        hits.append(1.0 if ri == fav else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5600 (var=0.0052)
**Previous candidate values (this loop):**
  - iter 1: n/a (var=n/a)
  - iter 2 (most recent): 0.4155 (var=0.0059) (Δ vs real -0.1445)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8115 (var=0.0019)
- pi_4: 0.3985 (var=0.0055)
- pi_1: 0.1815 (var=0.0131)
- pi_2: 0.5685 (var=0.0047)
- pi_3: 0.5220 (var=0.0036)
- pi_5: 0.7830 (var=0.0038)

### Experiment 10
**Design**
  A=[0, 0, 1, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Follow-the-99%-expert rate": across all trials where the single
    # 0.99-validity expert (feature index 0 in the fixed validity vector
    # [0.99, 0.98, 0.95, 0.90, 0.85, 0.75, 0.65, ., ., .]) discriminates
    # between A and B, compute the proportion of choices that go to the
    # option the 0.99 expert rated 1 (i.e., the option the top expert
    # "endorsed").
    #
    # FDEI: w(0.99) is the LARGEST positive weight in the whole vector
    # (its compression never inverts the validity ordering), so the
    # 0.99-favored option is followed above chance on essentially every
    # such trial -> metric ~ 0.60-0.65.
    # BSST: w(0.99) is strongly NEGATIVE (~ -1.1 trusted-cue units, the
    # distrust cliff), so the 0.99-favored option is chosen BELOW chance
    # on every such trial (the background cues are balanced/weak in this
    # design) -> metric ~ 0.15-0.35.
    # The sign of (metric - 0.5) alone separates the two theories on
    # every trial in the 0.99-discriminating set.

    TOP = 0  # index of the 0.99-validity expert, fixed by the design

    resp = data['response'].to_numpy()
    follows = []

    for i in range(len(data)):
        a_raw = data['option_a_ratings'].iloc[i]
        b_raw = data['option_b_ratings'].iloc[i]
        try:
            av = np.asarray(list(a_raw), dtype=int).ravel()
            bv = np.asarray(list(b_raw), dtype=int).ravel()
        except Exception:
            continue
        if av.size <= TOP or bv.size <= TOP:
            continue

        d = int(av[TOP]) - int(bv[TOP])
        if d == 0:
            continue  # 0.99 expert tied on this trial -> uninformative

        favored_a = d > 0          # option the 0.99 expert endorsed
        chose_a = (int(resp[i]) == 0)
        follows.append(1.0 if (chose_a == favored_a) else 0.0)

    if len(follows) == 0:
        return 0.5  # neutral fallback (should not occur: 8 trials x 4 reps)

    return float(np.mean(follows))
```

**Observed (real) value:** 0.5988 (var=0.0026)
**Previous candidate values (this loop):**
  - iter 1: n/a (var=n/a)
  - iter 2 (most recent): 0.6300 (var=0.0062) (Δ vs real +0.0312)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6719 (var=0.0085)
- pi_6: 0.2988 (var=0.0058)
- pi_1: 0.8525 (var=0.0127)
- pi_2: 0.6687 (var=0.0078)
- pi_3: 0.6206 (var=0.0104)
- pi_5: 0.1444 (var=0.0034)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Consensus-Relative Edge-Distrust Tallying (CREDT). People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, where the subjective weight w(v) of an expert is a function of its stated validity RELATIVE TO THE PANEL CONSENSUS, with three components. (1) MID-BAND PLATEAU WITHOUT A CLIFF: every expert with stated validity in [v_lo, v_hi] (~[0.55, 0.96]) receives approximately equal positive weight w_p ~ 1 — pure tallying over diagnostic cues, with NO absolute discontinuity anywhere; adjacent high validities (0.98, 0.99) are near-exchangeable in themselves. (2) CHANCE-FLANK: experts near v = 0.5 carry a small NEGATIVE weight (w(0.5) = -kappa_lo ~ -0.22) — a coin-flipper's endorsement is mildly anti-diagnostic noise, not zero information. (3) OUTLIER-GAP DISTRUST: the weight of the top expert(s) — those holding the panel's maximum validity, and only when that maximum exceeds the plateau's upper edge v_hi — is discounted by rho * sigmoid((gap - g0)/s_gap), where gap is the distance from the top validity to the next-highest DISTINCT validity in the panel. A dense ladder (0.99 atop 0.98) leaves the top expert at near-full plateau weight — the claimed edge is corroborated by the consensus and simply counted; an isolated super-expert (0.99 atop 0.75, gap ~ 0.24) has its influence collapsed toward zero or slightly negative — a claim of certainty nobody else corroborates is distrusted. The evidence score feeds a low-gain softmax (beta ~ 0.4-0.5) plus a small uniform lapse (epsilon ~ 0.05). Distrust of certainty is RELATIVE to the panel, not an absolute 0.98/0.99 cliff (BSST's failure), and not a monotone flattened-diagnosticity gradient that amplifies isolated super-experts (FDEI's failure).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Consensus-Relative Edge-Distrust Tallying (CREDT).
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j); votes are
    # weighted by a subjective trust function of the expert's STATED
    # validity v, evaluated RELATIVE to the panel:
    #   (i) MID-BAND PLATEAU, NO CLIFF: all v in [v_lo, v_hi] get
    #       approximately equal positive weight w_p (pure tallying
    #       over diagnostic cues; adjacent high validities are
    #       near-exchangeable -- there is no absolute 0.98/0.99 step).
    #  (ii) CHANCE-FLANK: v near 0.5 gets a small NEGATIVE weight,
    #       reaching -kappa_lo at v = 0.5 exactly (a coin-flipper's
    #       endorsement is mildly anti-diagnostic) and rising
    #       linearly to the plateau at v_lo.
    # (iii) OUTLIER-GAP DISTRUST: the top expert(s) -- those holding
    #       the panel's maximum validity, and only when that maximum
    #       exceeds the plateau's upper edge v_hi -- are discounted by
    #       rho * sigmoid((gap - g0)/s_gap), where gap is the distance
    #       from the top validity to the next-highest DISTINCT
    #       validity in the panel. Dense ladder -> gap ~ 0 -> top
    #       keeps ~full plateau weight (the edge is corroborated by
    #       the consensus). Isolated super-expert -> large gap ->
    #       its weight collapses toward zero or slightly negative.
    # Choice is a low-gain softmax (inverse temperature beta) over
    # the two mirrored scores, plus an independent uniform lapse
    # (epsilon).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    w_p = float(parameters["w_p"])
    v_lo = float(parameters["v_lo"])
    v_hi = float(parameters["v_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    rho = float(parameters["rho"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)

    # ---- Panel structure: top validity and its gap to the
    # next-highest DISTINCT validity in the panel. ----
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    # If every expert shares the top validity, the panel offers no
    # corroboration ladder; treat the next rung as chance (0.5).
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second

    # ---- (i) + (ii): plateau with a chance flank. ----
    w = np.full(n_features, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        w[flank] = -kappa_lo + (w_p + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom

    # ---- (iii): outlier-gap distrust of the top expert(s). ----
    # Fires only when the panel's maximum validity rises above the
    # trust plateau's upper edge (a genuine claim of near-certainty).
    if v_top > v_hi:
        x = (gap - g0) / s_gap
        # Numerically stable logistic via numpy tanh (no math module
        # dependency; stable for any x).
        sig = 0.5 * (1.0 + np.tanh(0.5 * x))
        w_top = w_p - rho * float(sig)
        # All experts tied at the panel maximum share the distrust
        # (e.g., a pair of isolated 0.99 super-experts).
        w[v >= v_top - 1e-9] = w_top

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a], dtype=float)

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- w_p: [0.9, 1.1]
- v_lo: [0.52, 0.56]
- v_hi: [0.95, 0.96]
- kappa_lo: [0.19, 0.25]
- g0: [0.09, 0.12]
- s_gap: [0.02, 0.035]
- rho: [1.0, 1.4]
- beta: [0.38, 0.50]
- epsilon: [0.04, 0.07]
- validities: validities

`rationale`: This is the minimal-diff repair the critic prescribed: the architecture, equations, and parameter names of the ACCEPTED CREDT base are unchanged; only (a) the crash bug and (b) the calibration box were touched. (a) BUG FIX: the outlier-gap branch called math.exp() while only numpy was in scope, so every experiment whose panel maximum exceeds v_hi (Exps 3, 4, 6, 7, 8, 9, 10 -- exactly the experiments that exercise the theory's novel component) crashed to n/a. The branch now computes the logistic through the everywhere-stable identity sigmoid(x) = 0.5*(1 + tanh(x/2)) using numpy, so the gap-driven distrust actually runs on all ten experiments for the first time. (b) RECALIBRATION against the critic's four anchor points, staying inside the arbiter's prescribed box: (1) gap ~ 0.01 (Exp 10's 0.99-atop-0.98 ladder, Exp 9's 0.99-atop-0.98): with g0 in [0.09, 0.12] and s_gap in [0.02, 0.035], the sigmoid sits deep in its foot at gap 0.01 (x ~ -3 to -5), so the top expert keeps ~full plateau weight and follow-99 lands at ~0.58-0.62 (observed 0.599), with the tiny residual discount (w_top ~ 0.95-0.99) supplying the slight anti-99 tilt Exp 9 needs (observed 0.560). (2) gap ~ 0.08 (Exp 8's 0.98 atop 0.90): x ~ -1 to -1.6, sigmoid ~ 0.15-0.30, so with rho in [1.0, 1.4] the discount is ~0.15-0.4 trusted-cue units -- a MILD discount that makes the 98% expert slightly weaker than the 90% expert, yielding follow-98 ~ 0.44-0.48 (observed 0.4625), not the strong anti-follow that destroyed pi_6 there. (3) gap ~ 0.24 (Exp 6's 0.99 atop 0.75, and Exp 4/7-style isolated super-experts): x ~ +4 to +7, sigmoid ~ 1, so w_top ~ w_p - rho reaches ~ -0.3 to 0.0 -- near-full collapse. In Exp 6's K2/K3/M3/M4 cells a lone (or paired) collapsed super-expert then loses to 2+ ordinary plateau cues, driving the strong-side rate to ~0.08-0.15 (observed 0.118); in Exp 4's |d|=1 family the discounted top cue plus the negative chance flank shrinks exactly the margins that pi_3 over-committed on (pi_3: 0.606 vs observed 0.513). (4) CHANCE-FLANK deepening: kappa_lo is raised to [0.19, 0.25] and beta to [0.38, 0.50] per the critic's note that iter-1's -0.1217 (observed -0.1683) was under-negative -- the product beta*kappa_lo, which pins the Z-trial signature, rises from ~0.09 to ~0.10-0.12, moving Exp 5 toward -0.16 to -0.20 while the lapse (epsilon ~ 0.05) keeps it off the floor. v_hi is kept at [0.95, 0.96] so that Exps 1/2/5 (top validities 0.92-0.95) never trigger the distrust branch and retain their already-good iter-1 fits (Exp 1: -0.267 vs -0.218; Exp 2: 0.111 vs 0.077; both within noise). Every declared range lies inside the arbiter's suggested box (g0 <= 0.12, s_gap >= 0.02, rho within [0.9, 1.7], beta within [0.30, 0.50], kappa_lo within [0.15, 0.25], epsilon within [0.03, 0.08]). No mechanism was added or removed: sign votes x plateau x chance flank x gap-driven top-cue distrust x low-gain softmax x lapse, exactly as prescribed.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate's simulation results reveal a critical implementation failure rather than a theoretical one. The candidate produced scorable values on only 3 of 10 experiments (Exp 1: -0.2672 vs -0.2180; Exp 2: 0.1107 vs 0.0771; Exp 5: -0.1217 vs -0.1683 — all reasonable fits), and 'n/a' on Experiments 3, 4, 6, 7, 8, 9, 10. The pattern of which experiments failed is diagnostic: the three experiments that succeeded are exactly those whose panel-maximum validity sits at or below the plateau's upper edge (Exp 2's top validity is 0.92; Exp 5's is 0.95; Exp 1's top is evidently ≤ v_hi), while every experiment whose top validity is ≥ 0.98 — precisely the experiments that exercise the OUTLIER-GAP DISTRUST branch — returned n/a. The mechanism is clear: the outlier-gap branch calls math.exp(), but `math` is never imported in the predict function (only numpy is used elsewhere). Whenever v_top > v_hi, the code hits `math.exp(-x)` / `math.exp(x)` and raises NameError, crashing the simulation. The three 'successful' experiments never execute the novel branch, so what was scored is just the shared tally/plateau/chance-flank scaffold — essentially pi_3-like behavior. The accept-gate loss (0.7165) therefore reflects a candidate whose defining component (the relative distrust of isolated super-experts, the entire point of CREDT per the arbiter's recommendation) has never been tested against the data. The fits on Exps 1, 2, 5 are encouraging for the scaffold, and Exp 5's negative chance-cue sensitivity (-0.1217 vs observed -0.1683) confirms the chance-flank works directionally (slightly under-negative; kappa_lo at the upper end of its range would help), but none of this validates the theory's novel claim.
Rationale: Regenerate — but this is a bug fix plus recalibration, not a redesign; the CREDT mechanism family is correct per the arbiter's prescription and must be kept intact. (1) CRITICAL BUG: replace `math.exp` with `np.exp` (or add `import math`, or use a fully numpy-vectorized stable sigmoid) in the outlier-gap branch. This single fix will make the model scorable on Experiments 3, 4, 6, 7, 8, 9, 10 — the seven experiments that actually test the theory's novel component and on which it currently cannot be evaluated. (2) After the fix, verify the gap-sigmoid calibration against the four anchor points: gap ≈ 0.01 (Exp 10: follow-99 must be ≈ 0.60, so the discount at gap 0.01 must be near zero — this argues for s_gap at the low end of [0.02, 0.06] and g0 ≈ 0.05-0.08 so the sigmoid is still in its foot at gap 0.01); gap ≈ 0.08 (Exp 8: follow-98 ≈ 0.46, i.e., only a mild discount of the 0.98 expert relative to the 0.90 expert — check that w_top - w(0.90) combined with beta yields ~0.46, not a strong anti-follow); gap ≈ 0.24 (Exp 6: strong-side rate ≈ 0.118, requiring near-full collapse of the 0.99 weight — rho at the upper end, ~1.3-1.7, so the asymptote reaches ≤ 0); and Exp 7 (anti-99 ≈ 0.58) and Exp 9 (anti-99 ≈ 0.56) — inspect the actual next-highest validity in those panels and confirm the fitted sigmoid delivers the required discount at those specific gaps; if Exp 7's gap is small (~0.09) while its required discount is large, the tension with Exp 8 (gap 0.08, mild discount) must be resolved within the family, e.g., by sharpening s_gap and shifting g0 so the transition sits between the two experiments' gaps, or by noting Exp 7's second rung is lower than assumed. (3) Minor: Exp 5 came out at -0.12 vs observed -0.168 — once re-run, push kappa_lo toward 0.22-0.25 and/or beta toward 0.45-0.50 to deepen the chance-cue negativity. Do not change the architecture: sign votes × plateau weights × chance flank × gap-driven top-cue distrust × low-gain softmax × lapse is exactly what the arbiter prescribed and the three scored experiments support it; the candidate simply has never actually run its core mechanism on the data.

**Outcome of this advice:** iter 1 candidate loss=0.7165 -> iter 2 candidate loss=0.1676 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The bug fix worked: all ten experiments are now scorable and the loss dropped from 0.7165 to 0.1676. Seven of ten experiments fit well — Exps 1 (-0.276 vs -0.218), 2 (0.130 vs 0.077), 3 (0.601 vs 0.617), 4 (0.506 vs 0.513), 5 (-0.198 vs -0.168), 8 (0.409 vs 0.463), 10 (0.630 vs 0.599) are all within |Δ| ≤ 0.06. But now that the outlier-gap branch actually runs, it fails precisely on the experiments that test it hardest. (A) EXP 6 is catastrophic: 0.635 vs observed 0.118 (Δ +0.52). The diagnosis is identifiable from the arithmetic: the model can only produce a strong-side rate this high if the 'weak' experts (v ≈ 0.50-0.52) are receiving NEGATIVE chance-flank weights, so their endorsements are inverted — 2-4 weak cues endorsing B then push the score toward A, and in the R3 cell two plateau-weighted 0.75 cues (w ≈ +1) crush three inverted chance cues, giving p(strong) ≈ 0.9. Real subjects follow the raw endorsement count (pi_2, pure tallying, hits 0.118 exactly). Note the proposer's own rationale predicted 0.08-0.15 here assuming the weak cues were plateau cues — the simulation contradicts that, confirming the weak validities sit below the effective v_lo and land in the negative flank. (B) EXP 7: 0.346 vs 0.580 (Δ -0.23) — the model net-FOLLOWS the 99% expert; the gap discount (rho ≤ 1.4, w_p up to 1.1, so w_top ≥ -0.5 at best, likely ≈ -0.2) is too weak against background plateau cues aligned with the 99. (C) EXP 9: 0.416 vs 0.560 (Δ -0.14) — the dense ladder (0.99 atop 0.98, gap 0.01) correctly leaves the distrust off, but then the D-cells sit at 0.5 and the V1 cell is strongly PRO-99 because the negative chance flank adds ~+0.66 to the 99-side, dragging the overall anti-99 rate below 0.5. (D) Minor regressions vs iter 1 on Exps 1 and 2 (over-negative / over-tally) and a slight overshoot on Exp 5 (-0.198 vs -0.168) from the kappa_lo/beta increase I recommended last round — that advice was accepted and helped overall, but the flank depth now needs to settle back toward the middle rather than push further.
Rationale: Regenerate — the architecture is right and gate-accepted, but three experiments (6, 7, 9) mis-fit in a way that points at specific, fixable miscalibrations inside the CREDT family. Do NOT change the family (sign votes × plateau × chance flank × gap distrust × low-gain softmax × lapse); make these in-family repairs: (1) FIX EXP 6 FIRST (Δ +0.52, the dominant loss term). Push v_lo to the bottom of its range (0.52) so that any weak-but-diagnostic cue above 0.50 joins the plateau; if Exp 6's weak experts are exactly v = 0.50 (so no v_lo ≥ 0.52 can include them), then GATE the chance flank by the outlier-distrust signal: when the gap-sigmoid is near 1 (isolated super-expert), distrust generalizes to the whole panel and chance cues should be counted (weight → ~0 or a positive fraction of w_p) rather than inverted — e.g., effective kappa = kappa_lo · (1 − sig_gap). Exp 5's panel (top 0.95, no outlier, branch off) retains the full negative flank, preserving its -0.168; Exp 6's panel reverts to endorsement counting, which is what its data (and pi_2's exact 0.118) show. This is a gating/conditioning fix squarely within the 'consensus-relative' brief, not a new mechanism. (2) SHARPEN AND STRENGTHEN the gap discount: s_gap → ~0.02, g0 → 0.11-0.12, rho → 1.5-1.7 (top of the arbiter box), w_p → 0.9. Then: gap 0.01 → essentially no discount (Exp 10 stays ≈ 0.60); gap 0.08 → w(0.98) ≈ 0.65-0.7 vs w(0.90) = 0.9, giving follow-98 ≈ 0.45-0.47 (Exp 8); gap 0.24 → w(0.99) ≈ -0.7 to -0.8, which — once the chance cues stop being inverted — drives Exp 6's K/M cells toward the lapse floor and lifts Exp 7's anti-99 from 0.35 toward the observed 0.58. (3) EXP 9 (Δ -0.14): after (1)-(2), re-check; if the D-cells still sit at 0.5, add a small SMOOTH, gap-independent discount of the panel-max cue whenever v_top > v_hi (Δw ≈ 0.2-0.3, no cliff — adjacent validities remain near-exchangeable), which moves the D-cells toward 0.56 while spending Exp 10's current +0.03 slack. (4) Settle kappa_lo ≈ 0.19-0.21 and beta ≈ 0.40-0.45 — between the iter-1 and iter-2 settings — to re-center Exps 1, 2, 5 (my iter-1 push to raise them was accepted and helped, but it has now slightly overshot; take an intermediate value rather than reversing direction). (5) Re-verify Exps 3 and 4 after retuning, since they are currently excellent (Δ ≤ 0.016) and the rho/v_lo changes touch their panels too; do not sacrifice them for Exp 9.

**Outcome of this advice:** iter 2 candidate loss=0.1676 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Consensus-Relative Edge-Distrust Tallying (CREDT), consensus-gated edition. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score. The subjective weight of an expert is determined by its stated validity RELATIVE TO THE PANEL CONSENSUS, via four components: (1) MID-BAND PLATEAU: all validities in [v_lo, v_hi] (~[0.53, 0.96]) get approximately equal positive weight w_p ~ 0.9 — pure tallying over diagnostic cues, no absolute discontinuity anywhere. (2) CHANCE FLANK: experts near v = 0.5 carry a small negative weight (w(0.5) = -kappa_lo ~ -0.2) — a coin-flipper's endorsement is mildly anti-diagnostic. (3) CONSENSUS GATING (new): the chance flank is itself modulated by the panel's outlier structure — when the panel's top expert is an ISOLATED outlier (large validity gap to the next rung), distrust generalizes from the super-expert to the validity scale itself, and the subject falls back to DEMOCRATIC TALLYING: chance cues are counted at plateau weight rather than inverted. A panel with no outlier (or a dense, corroborated ladder) keeps the full negative chance flank. (4) DUAL-COMPONENT TOP DISCOUNT (new): the panel-max expert's weight is discounted by (a) a SMOOTH EXTREMITY term, linear in how far v_top exceeds the trust ceiling v_hi (distrust of the claim's extremity itself, no cliff), plus (b) the GAP term rho*sigmoid((gap - g0)/s), which collapses the top expert's influence only when its edge is uncorroborated by the next rung. Choice is a low-gain softmax (beta ~ 0.43) over the mirrored scores plus a uniform lapse (epsilon ~ 0.05).",
  "predict": "def predict(parameters, state, history):\n    # Consensus-Relative Edge-Distrust Tallying (CREDT),\n    # consensus-gated edition.\n    # Stimulus: array of shape (2, n_features); row 0 = option A,\n    # row 1 = option B; entries are binary expert ratings (0/1).\n    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a\n    # subjective trust function of the expert's STATED validity v,\n    # evaluated RELATIVE to the panel:\n    #   (i) MID-BAND PLATEAU, NO CLIFF: all v in [v_lo, v_hi] get\n    #       approximately equal positive weight w_p.\n    #  (ii) CHANCE FLANK: v near 0.5 gets a small NEGATIVE weight,\n    #       reaching -kappa_lo at v = 0.5, rising linearly to the\n    #       plateau at v_lo.\n    # (iii) CONSENSUS GATING: the chance flank is modulated by the\n    #       panel's outlier signal sig_gap. When the panel's top\n    #       expert is an isolated outlier (sig_gap -> 1), distrust\n    #       generalizes to the validity scale itself and the subject\n    #       falls back to democratic tallying: chance cues are\n    #       counted at plateau weight instead of inverted.\n    #  (iv) DUAL-COMPONENT TOP DISCOUNT: the panel-max expert's\n    #       weight is discounted by (a) a smooth extremity term,\n    #       linear in (v_top - v_hi) — distrust of the claim's\n    #       extremity itself, no cliff — plus (b) the gap term\n    #       rho * sigmoid((gap - g0)/s_gap), which collapses the\n    #       top expert only when its edge over the next rung is\n    #       large (uncorroborated). A dense ladder (gap ~ 0.01)\n    #       leaves the top expert at near-plateau weight.\n    # Choice is a low-gain softmax (inverse temperature beta) over\n    # the two mirrored scores, plus a uniform lapse (epsilon).\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {val.shape[0]} != n_features {n_features}.\"\n        )\n\n    w_p = float(parameters[\"w_p\"])\n    v_lo = float(parameters[\"v_lo\"])\n    v_hi = float(parameters[\"v_hi\"])\n    kappa_lo = float(parameters[\"kappa_lo\"])\n    g0 = float(parameters[\"g0\"])\n    s_gap = float(parameters[\"s_gap\"])\n    rho = float(parameters[\"rho\"])\n    d_w = float(parameters[\"d_w\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    a, b = stim[0], stim[1]\n\n    # Feature-wise votes: +1 where A wins the feature, -1 where B\n    # wins, 0 on ties (a tied cue is uninformative for the pair).\n    votes = np.sign(a - b)\n\n    v = np.clip(val, 0.5, 1.0)\n\n    # ---- Panel structure: top validity and its gap to the\n    # next-highest DISTINCT validity in the panel. ----\n    v_top = float(v.max())\n    below = v[v < v_top - 1e-9]\n    # If every expert shares the top validity, the panel offers no\n    # corroboration ladder; treat the next rung as chance (0.5).\n    v_second = float(below.max()) if below.size > 0 else 0.5\n    gap = v_top - v_second\n\n    outlier = v_top > v_hi\n    if outlier:\n        x = (gap - g0) / s_gap\n        # Numerically stable logistic via the identity\n        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).\n        sig_gap = 0.5 * (1.0 + np.tanh(0.5 * x))\n    else:\n        sig_gap = 0.0\n\n    # ---- (i) + (ii) + (iii): plateau with a consensus-gated\n    # chance flank. ----\n    w = np.full(n_features, w_p, dtype=float)\n    flank = v < v_lo\n    if np.any(flank):\n        denom = max(v_lo - 0.5, 1e-6)\n        w_flank = -kappa_lo + (w_p + kappa_lo) * np.clip(\n            v[flank] - 0.5, 0.0, None\n        ) / denom\n        # Consensus gating: with an isolated super-expert in the\n        # panel (sig_gap -> 1), the validity scale itself is\n        # distrusted and chance cues are counted (weight -> w_p)\n        # rather than inverted; with no outlier (sig_gap = 0) the\n        # full negative flank is retained.\n        w[flank] = (1.0 - sig_gap) * w_flank + sig_gap * w_p\n\n    # ---- (iv): dual-component discount of the panel-max expert. ----\n    # Fires only when the panel's maximum validity rises above the\n    # trust plateau's upper edge (a genuine claim of near-certainty).\n    if outlier:\n        # (a) smooth extremity discount, linear in (v_top - v_hi):\n        # distrust grows with how extreme the claim itself is,\n        # independent of the ladder below it (no cliff — adjacent\n        # validities remain near-exchangeable).\n        # (b) gap discount: an uncorroborated edge over the next\n        # rung collapses the top expert's influence.\n        w_top = (\n            w_p\n            - d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)\n            - rho * float(sig_gap)\n        )\n        # All experts tied at the panel maximum share the distrust\n        # (e.g., a pair of isolated 0.99 super-experts).\n        w[v >= v_top - 1e-9] = w_top\n\n    # Weighted evidence score. B's score is the mirror of A's\n    # because every vote is zero-sum across the two options.\n    s_a = float(np.dot(w, votes))\n    scores = np.array([s_a, -s_a], dtype=float)\n\n    # Numerically stable low-gain softmax. When all cues tie\n    # (s_a == 0) the softmax is exactly uniform.\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n\n    # Independent lapse: with probability epsilon output a uniform\n    # pick over the two options.\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # guard against float drift\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "w_p": "[0.87, 0.93]",
    "v_lo": "[0.52, 0.54]",
    "v_hi": "[0.955, 0.965]",
    "kappa_lo": "[0.18, 0.22]",
    "g0": "[0.11, 0.12]",
    "s_gap": "[0.02, 0.025]",
    "rho": "[1.45, 1.65]",
    "d_w": "[0.30, 0.40]",
    "beta": "[0.41, 0.45]",
    "epsilon": "[0.04, 0.06]",
    "validities": "validities"
  },
  "rationale": "Minimal-diff repair of the ACCEPTED CREDT base, implementing all four in-family fixes the critic prescribed; the architecture (sign votes x plateau x chance flank x relative top-cue distrust x low-gain softmax x lapse) is untouched. (1) EXP 6 FIX (the dominant +0.52 error): the diagnosis was that the weak (v=0.5) experts land in the negative chance flank, so their endorsements INVERT — 2-4 weak cues endorsing B push the score toward A. The flank is now CONSENSUS-GATED: w_flank_eff = (1-sig_gap)*w_flank + sig_gap*w_p. In Exp 6 (0.99x2 atop 0.75, gap 0.24, sig_gap ~ 1) the weak cues revert to plateau weight — democratic tallying, which is what its data (and pi_2's exact 0.118) show — while the top expert collapses to w_top ~ 0.9 - 0.26 - 1.55 ~ -0.9. Hand-simulating the five cells (K2/K3/M3/M4/R3) with the box midpoints gives strong-side rates of ~0.11/0.07/0.04/0.03/0.32, average ~0.11 vs observed 0.118. Exp 5's panel (top 0.95 < v_hi ~ 0.96, sig_gap = 0) retains the full negative flank, preserving its -0.17. (2) SHARPENED GAP DISCOUNT per the critic: s_gap ~ 0.02, g0 ~ 0.115, rho ~ 1.55, w_p ~ 0.9. The transition now sits BETWEEN corroborated ladders and isolated outliers: gap 0.01 (Exps 9/10) -> sig ~ 0 -> top keeps near-plateau weight; gap 0.08 (Exp 8) -> sig ~ 0.17 -> w(0.98) ~ 0.46 vs w(0.90) = 0.9, follow-98 ~ 0.41-0.44 (observed 0.4625); gap 0.14 (Exp 7-style 0.99-atop-0.85 panels) -> sig ~ 0.75 -> w(0.99) ~ -0.5, which hand-simulation of Exp 7's counted rows puts at anti-99 ~ 0.55-0.60 (observed 0.580, up from 0.346); gap 0.24 -> full collapse (Exp 6). (3) EXP 9 FIX (critic item 3): a SMOOTH, gap-independent extremity discount d_w*(v_top - v_hi)/(1 - v_hi) ~ 0.26 at v_top = 0.99 is added to the panel-max cue whenever v_top > v_hi — no cliff, adjacent validities remain near-exchangeable (0.98 in Exp 8 is not the panel max there, so only the 0.99 panels feel it). This moves Exp 9's D/S cells to ~0.56-0.62 and the overall anti-99 rate to ~0.51 (from 0.416, observed 0.560) while spending only ~0.03 of Exp 10's slack (follow-99 ~ 0.56-0.58 vs observed 0.599). (4) RECALIBRATION per critic item 4: kappa_lo ~ 0.20 and beta ~ 0.43 — intermediate between the iter-1 and iter-2 settings — re-center Exps 1, 2, 5 (Exp 5 hand-sim: -2*(sigma(0.86*0.2*|k|)-0.5) ~ -0.17 vs observed -0.168; Exp 1 ~ -0.25 vs -0.218; Exp 2 ~ +0.11 vs +0.077). All ranges remain inside the arbiter's prescribed box (g0 <= 0.12, s_gap >= 0.02, rho <= 1.7, beta in [0.30, 0.50], kappa_lo in [0.15, 0.25], epsilon in [0.03, 0.08]); d_w is the critic's own suggested addition. Every declared parameter is used by predict. The expected net effect is a large gain on Exps 6 (0.635 -> ~0.11) and 7 (0.346 -> ~0.55) and a moderate gain on Exp 9 (0.416 -> ~0.51), at the cost of small (< 0.05) shifts on the seven currently well-fit experiments."
}
```
```

## Usage

```json
{
  "input_tokens": 32837,
  "output_tokens": 64381,
  "reasoning_tokens": 60970,
  "provider": "Sail Research"
}
```
