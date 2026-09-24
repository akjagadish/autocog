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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace pi_6 with the following stronger competitor to pi_7: 'Anchored Anti-Validity Frugality with Validity-Respecting Leak.' Core claims: (1) RETAIN the one-reason, anti-validity cascade (consult the weakest advertised expert first) as the population-central hierarchy, and RETAIN a typeless population — no discrete per-subject component draws, no primacy/recency types (this is what the real data demand: zero primacy types in Exps 2 and 16, binomial-floor between-subject variance in Exps 1, 5, 12, 13). (2) Replace pi_7's fixed kappa*A^3 momentary diffusion with GAP-PROPORTIONAL ANCHORING: the per-trial attention noise on cue j is scaled by the LOCAL validity gaps adjacent to j in the subjective order (noise_sd = lambda / (gap_to_next_weaker + gap_to_next_stronger)), so large-gap designs (Exps 6, 8, 15) anchor the decider less diffusely than pi_7 predicts — directly fixing pi_7's overprediction of follow strength there — while small-gap, high-ambiguity designs (Exps 9, 10) remain fully diffuse. (3) Add a graded VALIDITY-RESPECTING LEAK: a small continuous component of subjective weight proportional to +validity, scaled by reversal coherence and validity-gap magnitude (leak = rho_rev * mean_gap / scale), which pulls a fraction of trials toward the standard TTB decider. This is what generates the strong ANTI-reversed-cascade behavior observed in Exps 9 and 10 (observed -0.31/-0.29 vs pi_7's -0.15/-0.21): when the reversed cascade's first cue is a mid- or high-validity expert, the leak flips the decider toward the strongest discriminator, producing the observed below-chance reversed-cascade following that BOTH current theories underpredict. (4) Attenuate and design-scale the follow probability: p_f = (1-eps)*sigmoid(beta) + eps/2 with (beta, eps) drawn per subject but with beta modulated by design ambiguity (anchored designs -> higher effective beta), targeting p_f in [0.60, 0.75] with the lower end on ambiguous designs — fixing the Exp 6/8/15 overpredictions. (5) Keep exact validity ties anchoring the order, keep softmax-plus-lapse noise, keep history-free prediction. This theory differs from pi_6 by abolishing discrete types entirely (its fatal flaw) and differs from pi_7 by replacing the design-invariant cubic diffusion with gap-proportional anchoring plus a validity-respecting leak, which are precisely the two mechanisms needed to capture the residual misses (Exps 6, 8, 9, 10, 15) while preserving everything pi_7 already gets right.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_7` (overall score: 0.894)

**Description**
Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity (momentary-diffusion variant). People are one-reason decision makers: cues are consulted one at a time in descending subjective weight and the first strictly discriminating cue decides; no integration across cues (flat psychometric profile over tally margins). The population-central hierarchy is anti-validity (distrust-the-weakest, w = -validity). Each subject's weights are a continuous perturbation, w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), where (i) salience_j is a positional gradient favoring early-listed experts with gamma drawn asymmetrically from [-0.05, +0.12] — a graded tilt, never a discrete primacy rule, bounded so it can flip nothing on the anchored designs that pin the anti-validity core; (ii) binding ambiguity A = 1 - |Spearman(position, validity)| enters CUBED, a steeply convex, gate-free gradient; and (iii) the idiosyncratic distortion is split into a small stable part (zeta_j, fixed per subject) and a large MOMENTARY part (xi_j, re-drawn every trial): under high binding ambiguity the consultation order is not merely idiosyncratic but diffuse — attention fluctuates from trial to trial, which keeps populations tight around chance-level conformity instead of spreading subjects into fixed opposing deciders. All distortion applies only when all advertised validities are distinct (exact ties anchor the distrust order). Noise is softmax(beta) over the binary winner score plus an independent lapse epsilon (implied follow p_f mean ~0.71, SD ~0.08); no discriminating cue yields exactly 0.5; validity ties are broken by a free per-subject tie-break (~80% early position); history is ignored.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity
    # (momentary-diffusion variant).
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective weights (continuous, per subject):
    #     w_j = -val_j + gamma * salience_j
    #            + A^3 * (sigma_h * zeta_j + kappa * xi_j)
    #
    # - ANTI-VALIDITY CORE: -val_j (distrust the weakest advertised
    #   expert; consult ascending validity).
    # - POSITIONAL SALIENCE: salience_j = (n-1-j)/(n-1) - 1/2, favoring
    #   early-listed experts; gamma is drawn ASYMMETRICALLY from
    #   [-0.05, +0.12]: the negative clamp sits above the anchored-design
    #   flip threshold (-0.0625) and the positive cap below the primacy
    #   type threshold (+0.144), so the tilt remains a graded perturbation
    #   that can never manufacture spurious pure types while retaining a
    #   partial early-position lean on anchored designs.
    # - BINDING AMBIGUITY: A = 1 - |Spearman(display position, validity)|,
    #   now entering CUBED. The steeply convex gradient keeps
    #   low/moderate-ambiguity designs anchored (A^3 ~ 0.01-0.02) while
    #   fully diffusing high-ambiguity all-distinct designs (A^3 ~ 0.3-0.6).
    # - DIFFUSE HETEROGENEITY: the idiosyncratic distortion is split into
    #   (a) a SMALL STABLE part, sigma_h * zeta_j (zeta fixed per subject,
    #   zero-centered), and (b) a LARGE MOMENTARY part, kappa * xi_j,
    #   where xi is RE-DRAWN EVERY TRIAL. Momentary consultation diffusion
    #   is what makes high-ambiguity populations tight around chance-level
    #   conformity: with fixed per-subject orders, scrambling MAXIMIZES
    #   between-subject follow-rate dispersion, whereas the observed
    #   population SD (~0.064 on the conformity-dispersion experiment)
    #   sits barely above the binomial floor. Both parts are applied ONLY
    #   when all advertised validities are distinct (exact ties anchor the
    #   distrust order).
    # NO reversal re-encoding, NO sigmoid gates, NO discrete component
    # draws.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon (implied follow probability
    # p_f = (1-eps)*sigmoid(beta) + eps/2, mean ~0.71, SD ~0.08).
    # No discriminating cue -> exactly 0.5. History is ignored.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    sigma_h = float(parameters["sigma_h"])
    kappa = float(parameters["kappa"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # NO silent zeros fallback: a shape mismatch means the entire
        # heterogeneity mechanism would be silently disabled, so fail
        # loudly instead (per the arbiter's diagnosis request).
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- positional-salience gradient (early-listed experts favored) ----
    # Centered so the tilt is a pure graded perturbation: +0.5 at the
    # first listed expert, -0.5 at the last, linear in between.
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- continuous subjective weights ----
    w = -val + gamma * sal

    # ---- diffuse idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the distrust order (no distortion);
    # all-distinct lists degrade with the CUBE of binding ambiguity.
    # The distortion is split into a small STABLE per-subject part
    # (sigma_h * zeta) and a large MOMENTARY per-trial part (kappa * xi,
    # xi re-drawn fresh on every call), so that high-ambiguity designs
    # show tight, chance-centered conformity rather than fixed
    # idiosyncratic deciders that spread the population apart.
    distinct = (np.unique(val).shape[0] == n_features)
    if distinct and ambiguity > 0.0:
        amb3 = ambiguity * ambiguity * ambiguity
        xi = np.random.uniform(-1.0, 1.0, size=n_features)
        w = w + amb3 * (sigma_h * zeta + kappa * xi)

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter (population-consistent: ~80% of
    # subjects break ties by early display position).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p_f = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.71
    # with per-subject SD ~0.08 over the sampled (beta, epsilon) ranges.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.5, 1.9]
- epsilon: [0.08, 0.24]
- gamma: [-0.05, 0.12]
- sigma_h: [0.6, 1.2]
- kappa: [8, 12]
- zeta: [(-1, 1)] * n_features
- tie_break: [0, 1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3633 (var=0.0060) vs this=0.2898 (var=0.0117)
- Experiment 2: real=0.0358 (var=0.0181) vs this=0.1029 (var=0.0537)
- Experiment 3: real=0.4533 (var=0.0207) vs this=0.4400 (var=0.0168)
- Experiment 4: real=0.7787 (var=0.0600) vs this=0.9093 (var=0.1091)
- Experiment 5: real=0.6133 (var=0.0052) vs this=0.7217 (var=0.0070)
- Experiment 6: real=0.3400 (var=0.0288) vs this=0.4450 (var=0.0288)
- Experiment 7: real=-0.3107 (var=0.0225) vs this=-0.1520 (var=0.0136)
- Experiment 8: real=-0.2889 (var=0.0205) vs this=-0.2122 (var=0.0106)
- Experiment 9: real=0.4280 (var=0.0052) vs this=0.3867 (var=0.0066)
- Experiment 10: real=116.5556 (var=0.0041) vs this=119.2111 (var=0.0025)
- Experiment 11: real=0.4897 (var=0.0022) vs this=0.5056 (var=0.0039)
- Experiment 12: real=0.5324 (var=0.0041) vs this=0.5586 (var=0.0062)
- Experiment 13: real=0.1500 (var=0.0143) vs this=0.2758 (var=0.0171)
- Experiment 14: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 15: real=0.1541 (var=0.0151) vs this=0.0890 (var=0.0148)
- Experiment 16: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)


---

### `pi_6` (overall score: 0.729)

**Description**
Attenuated Frugality with Position-Validity Mixture Heterogeneity. People are one-reason decision makers: cues are consulted one at a time and the first strictly discriminating cue decides, with no integration across cues (flat psychometric profile over tally margins, matching the margin-0 signatures of the knife-edge experiments). The subjective cue hierarchy is NOT universal: each subject draws a single consultation order from a graded mixture over a small candidate set: (1) anti-validity (distrust-the-weakest: consult the lowest advertised validity first) as the population-central tendency with a constant base weight; (2) position-primacy (consult experts in reading order, earliest-listed first); (3) position-recency (consult later-listed experts first, a minority variant); and (4) a residual standard-TTB component (descending validity). The position components' mixture weights grow smoothly and linearly with binding ambiguity A = 1 - |Spearman(display position, advertised validity)|, with no sigmoid gates of any kind: when the validity-to-position binding is unambiguous (monotone validity lists), the mixture collapses onto the anti-validity order (plus the small TTB residual), producing strong, consistent one-reason contrasts; when the binding is ambiguous (non-monotone lists), the position components enter, the pooled deciders disperse, and hierarchy-dissociation contrasts collapse toward chance. On top of the discrete mixture, a per-cue idiosyncratic weight distortion (zeta, fixed per subject, zero-centered) is applied, smoothly scaled by ambiguity times reversal-coherence and only when all advertised validities are distinct, so that all-distinct ambiguous designs degrade grain-by-grain while exact validity ties anchor the discrete rule mixture. Response discipline is attenuated and heterogeneous: the decider-follow probability p_f = (1-eps)*sigmoid(beta) + eps/2 spans roughly 0.55-0.85 across subjects with mean ~0.73 and SD ~0.06, so pooled contrasts are diluted both by mixture heterogeneity (disagreeing deciders) and by moderate, variable following. Validity ties are broken by a free per-subject tie-break; noise enters as softmax(beta) over the binary winner score plus an independent lapse epsilon; history is ignored (no feedback in this task).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Attenuated Frugality with Position-Validity Mixture Heterogeneity.
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective hierarchy: each subject draws ONE consultation order from
    # a graded mixture over a small candidate set:
    #   - ANTI-VALIDITY (central):  w = -validity  (distrust the weakest
    #     advertised expert; consult ascending validity)
    #   - POSITION-PRIMACY:         w = -position  (reading order;
    #     earliest-listed expert first)
    #   - POSITION-RECENCY:         w = +position  (later-listed first;
    #     minority variant)
    #   - RESIDUAL TTB:             w = +validity  (descending validity)
    # The position components' weights grow SMOOTHLY and LINEARLY with
    # binding ambiguity
    #     A = 1 - |Spearman(display position, advertised validity)|
    # with no sigmoid gates: at A ~ 0 (monotone validity lists) the mixture
    # collapses onto anti-validity plus the small TTB residual; at high A
    # the position components enter and pooled deciders disperse, pulling
    # hierarchy-dissociation contrasts toward chance.
    #
    # Per-cue idiosyncratic distortion (zeta, zero-centered, fixed per
    # subject) is applied only when all advertised validities are DISTINCT,
    # smoothly scaled by ambiguity x reversal-coherence, so all-distinct
    # ambiguous designs degrade grain-by-grain while exact validity ties
    # anchor the discrete rule mixture.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    sigma_h = float(parameters["sigma_h"])
    tie_break = float(parameters["tie_break"])
    comp = float(parameters["comp"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # Graceful fallback: no per-cue distortion.
        zeta = np.zeros(n_features)

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    # 0 when the validity list is monotone in display position (the
    # validity-to-position binding is unambiguous -> the mixture
    # concentrates on the anti-validity distrust order); growing with
    # position-validity conflict (ambiguous binding -> position-anchored
    # reading orders enter the mixture).
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined -> maximal
            # idiosyncrasy.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----
    # Used ONLY as a smooth (gate-free) multiplier on the per-cue zeta
    # distortion: idiosyncratic scrambling is strongest when the binding is
    # ambiguous AND the list structure is reversal-coherent enough that
    # small perturbations reshuffle the subjective order.
    if n_features > 1:
        ranks_v = avg_ranks(val)
        ranks_r = avg_ranks(val[::-1])
        dv = ranks_v - ranks_v.mean()
        dr = ranks_r - ranks_r.mean()
        denom2 = np.sqrt(float((dv * dv).sum()) * float((dr * dr).sum()))
        if denom2 > 1e-12:
            rho_rev = float((dv * dr).sum()) / denom2
        else:
            rho_rev = 0.0
        coherence = abs(rho_rev)
    else:
        coherence = 0.0

    # ---- graded mixture weights over the candidate subjective orders ----
    # Smooth in ambiguity; NO sigmoid gates. The per-subject draw `comp` ~
    # U(0,1) selects one component via the cumulative normalized weights.
    w_anti = 0.45                  # anti-validity: constant central mass
    w_rec = 0.06 * ambiguity       # position-recency: minority variant
    w_prim = 0.80 * ambiguity      # position-primacy: reading-order default
    w_ttb = 0.04                   # residual standard TTB
    z_tot = w_anti + w_rec + w_prim + w_ttb
    c_anti = w_anti / z_tot
    c_rec = c_anti + w_rec / z_tot
    c_prim = c_rec + w_prim / z_tot

    if comp < c_anti:
        # Distrust-the-weakest: anti-validity subjective weights.
        w_base = -val
    elif comp < c_rec:
        # Position-recency: later-listed experts consulted first.
        w_base = np.arange(n_features, dtype=float)
    elif comp < c_prim:
        # Position-primacy: earlier-listed experts consulted first.
        w_base = -np.arange(n_features, dtype=float)
    else:
        # Residual standard TTB: descending advertised validity.
        w_base = val

    # ---- per-cue idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the discrete rule mixture (no per-cue
    # noise); all-distinct lists degrade grain-by-grain, smoothly scaled
    # by ambiguity x reversal-coherence.
    distinct = (np.unique(val).shape[0] == n_features)
    zeta_scale = sigma_h * ambiguity * coherence if distinct else 0.0

    w = w_base + zeta_scale * zeta

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter: 0 -> earlier feature position
    # first, 1 -> later feature position first among equally weighted cues.
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.73 with
    # per-subject SD ~0.06 over the sampled (beta, epsilon) ranges:
    # attenuated and variable, but high enough to reproduce the strong
    # margin-0 one-reason signatures of the monotone-validity designs.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.85, 1.8]
- epsilon: [0.10, 0.22]
- sigma_h: [0.4, 1.4]
- zeta: [(-1, 1)] * n_features
- tie_break: {0, 1}
- comp: [0, 1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3633 (var=0.0060) vs this=0.3008 (var=0.0206)
- Experiment 2: real=0.0358 (var=0.0181) vs this=0.0396 (var=0.0357)
- Experiment 3: real=0.4533 (var=0.0207) vs this=0.4244 (var=0.0833)
- Experiment 4: real=0.7787 (var=0.0600) vs this=0.7893 (var=0.3919)
- Experiment 5: real=0.6133 (var=0.0052) vs this=0.5400 (var=0.0198)
- Experiment 6: real=0.3400 (var=0.0288) vs this=0.4583 (var=0.1237)
- Experiment 7: real=-0.3107 (var=0.0225) vs this=-0.1993 (var=0.0321)
- Experiment 8: real=-0.2889 (var=0.0205) vs this=-0.2433 (var=0.0122)
- Experiment 9: real=0.4280 (var=0.0052) vs this=0.4400 (var=0.0202)
- Experiment 10: real=116.5556 (var=0.0041) vs this=184.6889 (var=0.0053)
- Experiment 11: real=0.4897 (var=0.0022) vs this=0.4649 (var=0.0080)
- Experiment 12: real=0.5324 (var=0.0041) vs this=0.5900 (var=0.0093)
- Experiment 13: real=0.1500 (var=0.0143) vs this=0.2417 (var=0.0078)
- Experiment 14: real=0.0000 (var=0.0000) vs this=0.1600 (var=0.2544)
- Experiment 15: real=0.1541 (var=0.0151) vs this=-0.1977 (var=0.1728)
- Experiment 16: real=0.0000 (var=0.0000) vs this=0.3000 (var=0.2100)


---

### `pi_4` (overall score: 0.680)

**Description**
Weakest-Expert-First Frugality (value-based anti-validity TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude/tally information. However, the subjective cue hierarchy is anti-correlated in VALUE with the instructed validities: subjects systematically consult the cue with the LOWEST advertised validity first, then the next-lowest, and so on (equivalently, subjective weights w_j = -validity_j). Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope). When no cue discriminates, the model guesses uniformly. Validity ties are broken by a free per-subject tie-breaking parameter (early-position vs late-position preference among equally weak cues). Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice, giving an implied cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 that spans roughly 0.54 to 0.86 across subjects with a mean near 0.70 — high enough to produce the steep contrasts in the non-monotone-validity experiments while moderate values cover the monotone ones.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Weakest-Expert-First Frugality (value-based anti-validity TTB).
    # One-reason decision making: cues are consulted one at a time and
    # the first strictly discriminating cue decides. The subjective
    # hierarchy is anti-correlated in VALUE with the instructed
    # validities: the cue with the LOWEST advertised validity is
    # consulted FIRST, then the next-lowest, etc. (subjective weights
    # w_j = -validity_j). This is a VALUE-based sort (ascending
    # validity), not a positional reversal of the validity list, so in
    # experiments where the validity list is not monotone in feature
    # position the two hierarchies genuinely differ. No integration
    # across cues occurs -> flat psychometric profile across tally
    # margins. Noise: softmax (inverse temperature beta) over the
    # binary winner score, plus an independent lapse epsilon mixing in
    # a uniform choice. History is ignored (no feedback in this task).
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

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    tie_break = float(parameters["tie_break"])

    # Weakest-expert-first hierarchy: sort cues by ASCENDING validity
    # value. Ties in validity are broken by the free tie_break
    # parameter: 0 -> earlier feature position first (stable),
    # 1 -> later feature position first among equally weak cues.
    # np.lexsort uses the LAST key as primary: primary = val ascending,
    # secondary = position (forward or reversed).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, val))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges spans ~[0.54, 0.86] across
    # subjects with a mean near 0.70.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.5, 1.5]
- epsilon: [0.06, 0.18]
- tie_break: {0, 1}
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3633 (var=0.0060) vs this=0.3205 (var=0.0077)
- Experiment 2: real=0.0358 (var=0.0181) vs this=0.0400 (var=0.0527)
- Experiment 3: real=0.4533 (var=0.0207) vs this=0.3751 (var=0.0163)
- Experiment 4: real=0.7787 (var=0.0600) vs this=0.8467 (var=0.0843)
- Experiment 5: real=0.6133 (var=0.0052) vs this=0.6883 (var=0.0074)
- Experiment 6: real=0.3400 (var=0.0288) vs this=0.3633 (var=0.0488)
- Experiment 7: real=-0.3107 (var=0.0225) vs this=-0.1453 (var=0.0099)
- Experiment 8: real=-0.2889 (var=0.0205) vs this=-0.1644 (var=0.0172)
- Experiment 9: real=0.4280 (var=0.0052) vs this=0.3193 (var=0.0078)
- Experiment 10: real=116.5556 (var=0.0041) vs this=588.7556 (var=0.0036)
- Experiment 11: real=0.4897 (var=0.0022) vs this=0.5215 (var=0.0033)
- Experiment 12: real=0.5324 (var=0.0041) vs this=0.7076 (var=0.0076)
- Experiment 13: real=0.1500 (var=0.0143) vs this=0.2950 (var=0.0089)
- Experiment 14: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 15: real=0.1541 (var=0.0151) vs this=-0.0330 (var=0.0125)
- Experiment 16: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    margins = []
    matches = []
    for a_cell, b_cell, resp in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float).ravel()
        b = np.asarray(b_cell, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] != validities.shape[0]:
            continue
        diff = a - b

        # Take-The-Best cascade: first discriminating cue in validity order.
        ttb_winner = -1
        for j in cue_order:
            if diff[j] > 0:
                ttb_winner = 0
                break
            if diff[j] < 0:
                ttb_winner = 1
                break
        if ttb_winner < 0:
            # Full tie (trial 13): no TTB winner, uninformative for both theories.
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # Tally margin of the TTB-favored option (positive = tally agrees with TTB).
        if ttb_winner == 0:
            margin = a_wins - b_wins
        else:
            margin = b_wins - a_wins

        r = int(resp)
        match = 1.0 if r == ttb_winner else 0.0
        margins.append(margin)
        matches.append(match)

    if not margins:
        return 0.0

    x = np.asarray(margins, dtype=float)
    y = np.asarray(matches, dtype=float)

    # (1) LEVEL: P(choosing the TTB-favored option) on trials where the
    # tally does NOT favor it (tally majority against, or exact tally tie).
    low = x <= 0
    level = float(np.mean(y[low])) if np.any(low) else 0.0

    # (2) SLOPE: OLS slope of the match indicator on the tally margin,
    # across all trials with a TTB winner (margins -3, -2, -1, 0, +1).
    vx = float(np.var(x))
    if len(x) > 1 and vx > 0:
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
    else:
        slope = 0.0

    return level - slope
```

**Observed (real) value:** 0.3633 (var=0.0060)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8479 (var=0.0111)
- pi_2: 0.0323 (var=0.0195)
- pi_3: 0.4099 (var=0.0092)
- pi_3_1: 0.2933 (var=0.0063)
- pi_3_2: 0.3314 (var=0.0076)
- pi_4: 0.3205 (var=0.0077)
- pi_5: 0.2898 (var=0.0076)
- pi_6: 0.3008 (var=0.0206)
- pi_7: 0.2898 (var=0.0117)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_matrix(col):
    # col: pandas Series whose cells are lists/tuples/arrays of binary ratings
    return np.array([list(r) for r in col], dtype=float)


def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    A = _to_matrix(data['option_a_ratings'])
    B = _to_matrix(data['option_b_ratings'])
    if A.ndim != 2 or A.shape != B.shape or A.shape[0] == 0:
        return 0.0

    # Tally margin: number of features where A beats B minus number where B beats A.
    margin = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # Signed choice: +1 if the subject chose A (response == 0), -1 if B (response == 1).
    resp = pd.to_numeric(data['response'], errors='coerce').to_numpy(dtype=float)
    s = 1.0 - 2.0 * resp

    # Alignment between the signed choice and the tally margin.
    # Tallying: E[s | margin] = (1-eps) * tanh(beta*margin/2)  =>  E[s*margin] > 0,
    # scaling like beta * E[margin^2] / 2 for small beta and saturating at mean(|margin|)
    # for large beta (max value 28/16 = 1.75 on this design).
    # TTB: E[s | trial] = (1-eps)*tanh(beta/2)*ttb_sign, and on this stimulus set
    # sum over the 16 trial types of ttb_sign * margin = -2 (mean -0.125), so the
    # TTB value is slightly NEGATIVE for any beta > 0.
    return float(np.mean(s * margin))
```

**Observed (real) value:** 0.0358 (var=0.0181)
**Other theories' values on this metric (for reference):**
- pi_2: 1.3208 (var=0.1532)
- pi_1: -0.1621 (var=0.0229)
- pi_3: 0.0608 (var=0.0436)
- pi_3_1: 0.1208 (var=0.0447)
- pi_3_2: 0.0708 (var=0.0422)
- pi_4: 0.0400 (var=0.0527)
- pi_5: 0.0462 (var=0.0368)
- pi_6: 0.0396 (var=0.0357)
- pi_7: 0.1029 (var=0.0537)

### Experiment 3
**Design**
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Reversed-cascade follow rate (centered).
    #
    # For every trial we determine the winner implied by a one-reason
    # cascade that consults cues in ASCENDING true validity (i.e., the
    # 0.5-validity expert first, ..., the 0.9-validity expert last) --
    # exactly the subjective hierarchy implied by full misbinding
    # (kappa = 1). Full-tie trials carry no directional prediction and
    # are dropped. The metric is the signed follow rate:
    #   +1  -> subject always picks the reversed-cascade winner
    #    0  -> chance
    #   -1  -> subject always picks the opposite option
    #
    # Predicted values:
    #   Misbinding TTB: p(follow) = (1-eps)*sigmoid(beta) + eps/2 ~ 0.60
    #                   on EVERY discriminating trial  ->  metric ~ +0.21
    #   Tallying:       on the 7 trials where the tally winner OPPOSES
    #                   the lowest-validity cue (trials 1,2,3,5,6,7,15)
    #                   p(follow) < 0.5 for ANY beta > 0; on the 5
    #                   zero-margin trials (4,8,11,12,13) p(follow)
    #                   = 0.5 EXACTLY for any (beta, eps); only the 3
    #                   agreeing trials (9,10,14) push above 0.5.
    #                   Hence the Tallying metric is strictly NEGATIVE
    #                   for every parameterization in its ranges
    #                   (approaching 0 only as beta->0 or eps->0.5).
    validities = np.array([0.90, 0.80, 0.70, 0.60, 0.50])
    # Ascending true validity == descending subjective validity under
    # full misbinding: consult index 4 first, then 3, 2, 1, 0.
    cue_order = np.argsort(validities, kind="stable")

    n_used = 0
    follow = 0
    for a_raw, b_raw, r in zip(data["option_a_ratings"],
                               data["option_b_ratings"],
                               data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != b.shape or a.size != validities.size:
            continue
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            # Full tie (trial 16): no directional prediction.
            continue
        n_used += 1
        if int(r) == winner:
            follow += 1

    if n_used == 0:
        return 0.0
    return 2.0 * (follow / n_used) - 1.0
```

**Observed (real) value:** 0.4533 (var=0.0207)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1947 (var=0.0151)
- pi_2: -0.1742 (var=0.0091)
- pi_1: -0.4209 (var=0.0150)
- pi_3_1: 0.4276 (var=0.0099)
- pi_3_2: 0.3960 (var=0.0138)
- pi_4: 0.3751 (var=0.0163)
- pi_5: 0.4636 (var=0.0174)
- pi_6: 0.4244 (var=0.0833)
- pi_7: 0.4400 (var=0.0168)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast metric built from two parameter-free anchors:
    #
    # (1) Margin-0 cells with a reversed-cascade decider (T5, T6, T11, T16).
    #     Tallying predicts p(A) = 0.5 EXACTLY on any tie trial, for any
    #     (beta, epsilon), so the signed follow-the-decider score has
    #     expectation exactly 0. The reversed-hierarchy model follows its
    #     decider with q = (1-eps)*sigmoid(beta) + eps/2 in [0.57, 0.64],
    #     so its expected score is 2q-1 in [0.14, 0.28].
    #
    # (2) Disagreement cells (T1, T2, T10, T14): nonzero tally margin whose
    #     sign OPPOSES the reversed decider. Tallying follows the margin
    #     with p >= 0.5 for any beta > 0 (expected signed score 2p-1 >= 0);
    #     the reversed model follows the decider, i.e. follows the tally
    #     with 1-q <= 0.43 (expected signed score 1-2q < 0).
    #
    # metric = mean(follow-decider score on margin-0 cells)
    #        - mean(follow-tally score on disagreement cells)
    #
    # Tallying: 0 - (2p-1) <= 0 (typically around -0.4 to -0.9).
    # Reversed: (2q-1) - (1-2q) = 4q-2 in [0.28, 0.56] > 0.
    # The two theories land on opposite sides of zero with a large gap.
    s1 = []
    s2 = []
    n_rows = len(data)
    for i in range(n_rows):
        row = data.iloc[i]
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        n = len(a)

        # Tally margin: strict feature-wise wins for A minus for B.
        margin = 0
        for j in range(n):
            if a[j] > b[j]:
                margin += 1
            elif b[j] > a[j]:
                margin -= 1

        # Reversed-hierarchy decider: first strictly discriminating
        # feature scanning from the LAST index down (ascending true
        # validity, since validities descend by index).
        dec = 0
        for j in range(n - 1, -1, -1):
            if a[j] > b[j]:
                dec = 1
                break
            if b[j] > a[j]:
                dec = -1
                break
        if dec == 0:
            # Identical options: no decider, no information.
            continue

        resp = int(row["response"])
        chose_a = (resp == 0)

        if margin == 0:
            # Knife-edge cell: tally is exactly 0.5 under Tallying.
            follow_dec = (dec == 1 and chose_a) or (dec == -1 and not chose_a)
            s1.append(1.0 if follow_dec else -1.0)
        elif (margin > 0 and dec == -1) or (margin < 0 and dec == 1):
            # Disagreement cell: tally direction opposes the decider.
            follow_tally = (margin > 0 and chose_a) or (margin < 0 and not chose_a)
            s2.append(1.0 if follow_tally else -1.0)

    m1 = float(np.mean(s1)) if len(s1) > 0 else 0.0
    m2 = float(np.mean(s2)) if len(s2) > 0 else 0.0
    return m1 - m2
```

**Observed (real) value:** 0.7787 (var=0.0600)
**Other theories' values on this metric (for reference):**
- pi_2: -0.6893 (var=0.0556)
- pi_3: 0.3867 (var=0.0818)
- pi_1: -1.3413 (var=0.2259)
- pi_3_1: 0.8027 (var=0.0699)
- pi_3_2: 0.6893 (var=0.0735)
- pi_4: 0.8467 (var=0.0843)
- pi_5: 0.8960 (var=0.0975)
- pi_6: 0.7893 (var=0.3919)
- pi_7: 0.9093 (var=0.1091)

### Experiment 5
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Decider-follow rate on the margin-0 DISCRIMINATING trials (the
    # "subjective-hierarchy ladder": T4, T7-T11 in the design).
    # On these trials the tally is exactly balanced, so Tallying is
    # pinned to p = 0.50 toward either side for ANY (beta, epsilon)
    # (softmax over equal tallies is uniform, lapse is uniform).
    # Misbinding TTB instead follows its (reversed-hierarchy) first
    # discriminating cue with p = (1-eps)*sigmoid(beta) + eps/2,
    # which is >= 0.63 over the stated parameter ranges. The metric
    # is the proportion of choices that agree with the misbound
    # decider on exactly those trials: ~0.71 under misbinding TTB,
    # ~0.50 under Tallying.
    val = np.array([0.7, 0.9, 0.6, 0.95, 0.75, 0.6], dtype=float)
    # kappa = 1 misbinding: subjective weights are the validity list
    # bound in reverse to feature positions.
    w = val[::-1]
    cue_order = np.argsort(-w, kind="stable")  # f2, f4, f1, f5, f3, f0

    a = np.array([list(x) for x in data['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int).ravel()

    if a.shape[0] == 0:
        return 0.5

    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Margin-0 trials that still have at least one discriminating cue
    # (excludes the full tie T16, where both theories say 0.5).
    mask = (a_wins == b_wins) & ((a_wins + b_wins) > 0)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return 0.5

    follow = []
    for i in idx:
        decider = None
        for j in cue_order:
            if a[i, j] > b[i, j]:
                decider = 0
                break
            if b[i, j] > a[i, j]:
                decider = 1
                break
        if decider is None:
            continue  # shouldn't happen on discriminating trials
        follow.append(1 if resp[i] == decider else 0)

    if len(follow) == 0:
        return 0.5
    return float(np.mean(follow))
```

**Observed (real) value:** 0.6133 (var=0.0052)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.7178 (var=0.0092)
- pi_2: 0.5072 (var=0.0084)
- pi_1: 0.1383 (var=0.0146)
- pi_3: 0.6261 (var=0.0070)
- pi_3_2: 0.6633 (var=0.0068)
- pi_4: 0.6883 (var=0.0074)
- pi_5: 0.6311 (var=0.0430)
- pi_6: 0.5400 (var=0.0198)
- pi_7: 0.7217 (var=0.0070)

### Experiment 6
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Knife-edge signed contrast on the four margin-0 discriminating cells.

    Tallying predicts p(A) = 0.50 EXACTLY on every margin-0 cell, for any
    (beta, epsilon): equal tallies give an exactly uniform softmax. The
    reversed-hierarchy one-reason model, in contrast, lets a single cue
    decide these same cells, producing a systematic ~0.70/0.30 split whose
    sign is set by which option wins on the cascade's first cue.

    Metric = mean p(choose A | T11, T13) - mean p(choose A | T8, T15),
    where T11/T13 are the margin-0 cells whose cascade-first cue favors A
    and T8/T15 are the margin-0 cells whose cascade-first cue favors B
    (under the design's validities [0.95, 0.85, 0.75, 0.65, 0.6, 0.6]
    reversed -> f5 consulted first).

    Tallying: expectation exactly 0 for ANY parameter values.
    Misbinding TTB: expectation ~ +/-(2p-1) in [0.33, 0.49].
    """
    def to_key(v):
        return tuple([int(x) for x in v])

    a_keys = data["option_a_ratings"].apply(to_key)
    b_keys = data["option_b_ratings"].apply(to_key)
    chose_a = 1.0 - data["response"].astype(float)  # response 0 = A

    # Margin-0 (knife-edge) cells, identified by their rating vectors.
    # Cascade-first cue (f5) favors A here:
    hi_cells = [
        ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)),  # T11: 1-1 split, f5->A
        ((0, 0, 0, 0, 1, 1), (1, 1, 0, 0, 0, 0)),  # T13: 2-2 split, f5->A
    ]
    # Cascade-first cue (f5) favors B here:
    lo_cells = [
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),  # T8:  1-1 split, f5->B
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),  # T15: 3-3 split, f5->B
    ]

    def cell_pa(ak, bk):
        mask = (a_keys == ak) & (b_keys == bk)
        if int(mask.sum()) == 0:
            return np.nan
        return float(chose_a[mask].mean())

    hi = [cell_pa(ak, bk) for ak, bk in hi_cells]
    lo = [cell_pa(ak, bk) for ak, bk in lo_cells]
    hi = [v for v in hi if not np.isnan(v)]
    lo = [v for v in lo if not np.isnan(v)]
    if not hi or not lo:
        return 0.0

    return float(np.mean(hi) - np.mean(lo))
```

**Observed (real) value:** 0.3400 (var=0.0288)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0367 (var=0.0470)
- pi_3_1: 0.3933 (var=0.0286)
- pi_1: -0.7033 (var=0.0409)
- pi_3: 0.2250 (var=0.0465)
- pi_3_2: 0.3950 (var=0.0319)
- pi_4: 0.3633 (var=0.0488)
- pi_5: 0.4983 (var=0.0354)
- pi_6: 0.4583 (var=0.1237)
- pi_7: 0.4450 (var=0.0288)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Knife-edge decider-direction contrast at balanced tally margins.
    #
    # For every trial we compute (a) the tally margin m = (#features A wins)
    # - (#features B wins), and (b) the winner implied by the advocated
    # misbinding-TTB cascade (subjective hierarchy = instructed validities
    # reversed end-to-end: cue order f4 -> f0 -> f2 -> f1 -> f3 -> f5).
    #
    # We keep only trials with |m| in {0, 3} (excluding the |m|=1 cells,
    # whose decider directions are unbalanced across margins) and where the
    # cascade has a discriminating cue. In this subset the misbinding-A
    # cells are {T1 (m=-3), T3 (m=0), T5 (m=+3), T9 (m=0), T10 (m=0)} and
    # the misbinding-B cells are {T6 (m=+3), T7 (m=0), T8 (m=-3), T11 (m=0),
    # T12 (m=0)} -- perfectly mirror-balanced in margin.
    #
    # Metric = P(choose A | cascade says A) - P(choose A | cascade says B).
    #
    # Under Tallying, p(A) = (1-eps)*sigmoid(beta*m) + eps/2, which is
    # exactly symmetric in m. Within each group the margins are balanced
    # (-3/+3 pairs and 0s), so BOTH group means are exactly 0.5 for ANY
    # (beta, epsilon) -- the metric is pinned to 0 parameter-free.
    #
    # Under misbinding TTB, p(A) = p_follow on cascade-A cells and
    # 1 - p_follow on cascade-B cells, so the metric equals 2*p_follow - 1,
    # which over the advocated parameter ranges lies in ~[0.27, 0.44]
    # (mean ~0.36) regardless of margin.

    df = data.reset_index(drop=True)
    a_lists = df['option_a_ratings'].tolist()
    b_lists = df['option_b_ratings'].tolist()
    resp = df['response'].to_numpy()

    # Design validities and the kappa=1 reversed subjective hierarchy.
    val = np.array([0.6, 0.95, 0.6, 0.85, 0.65, 0.9])
    w = val[::-1]
    cue_order = np.argsort(-w, kind='stable').tolist()  # -> [4, 0, 2, 1, 3, 5]

    a_side = []  # trials where the misbound cascade favors A
    b_side = []  # trials where the misbound cascade favors B

    for i in range(len(df)):
        av = [int(v) for v in a_lists[i]]
        bv = [int(v) for v in b_lists[i]]
        n = len(av)

        margin = (sum(1 for j in range(n) if av[j] > bv[j])
                  - sum(1 for j in range(n) if bv[j] > av[j]))
        if abs(margin) == 1:
            # Unbalanced decider directions at |m|=1; excluded so the
            # Tallying expectation stays exactly 0.
            continue

        winner = -1
        for j in cue_order:
            if j >= n:
                continue
            if av[j] > bv[j]:
                winner = 0
                break
            if bv[j] > av[j]:
                winner = 1
                break
        if winner == -1:
            continue  # full-tie trial (T16): no discriminating cue

        chose_a = 1.0 if resp[i] == 0 else 0.0
        if winner == 0:
            a_side.append(chose_a)
        else:
            b_side.append(chose_a)

    pa = float(np.mean(a_side)) if len(a_side) > 0 else 0.5
    pb = float(np.mean(b_side)) if len(b_side) > 0 else 0.5
    return float(pa - pb)
```

**Observed (real) value:** -0.3107 (var=0.0225)
**Other theories' values on this metric (for reference):**
- pi_3_2: 0.3613 (var=0.0180)
- pi_2: -0.0100 (var=0.0118)
- pi_1: 0.0160 (var=0.0107)
- pi_3: 0.2040 (var=0.0089)
- pi_3_1: 0.4007 (var=0.0109)
- pi_4: -0.1453 (var=0.0099)
- pi_5: -0.2327 (var=0.0163)
- pi_6: -0.1993 (var=0.0321)
- pi_7: -0.1520 (var=0.0136)

### Experiment 8
**Design**
  A=[1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

# Design-time constants for this experiment.
VALIDITIES = np.array([0.6, 0.9, 0.8, 0.95, 0.65, 0.75, 0.6])


def metric(data: pd.DataFrame) -> float:
    """Cue-following deviation from chance on tally-tied (margin-0) trials.

    For every trial whose tally margin is zero (A and B win on equal numbers
    of features), find the first strictly discriminating cue under the
    reversed-hierarchy cue order (the competing theory's subjective
    hierarchy). Record whether the subject's choice followed that cue.
    Tallying predicts exactly chance (0 deviation) on every tied trial,
    for ANY (beta, epsilon). One-reason misbinding predicts a systematic
    follow rate of p = (1-eps)*sigmoid(beta) + eps/2, i.e. a deviation of
    roughly +0.12 to +0.25 above chance.
    """
    # Reversed-hierarchy subjective weights and cue order (stable ties).
    w = VALIDITIES[::-1].astype(float)
    cue_order = np.argsort(-w, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    resp_col = data["response"].tolist()

    follows = []
    for a_raw, b_raw, r_raw in zip(a_col, b_col, resp_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.ndim != 1:
            continue

        # Tally margin: number of features A wins minus number B wins.
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin != 0:
            # Only tied tallies carry the parameter-free signature.
            continue

        # First discriminating cue under the reversed hierarchy.
        decider = None
        for j in cue_order:
            if a[j] > b[j]:
                decider = 0  # cue favors A
                break
            if b[j] > a[j]:
                decider = 1  # cue favors B
                break
        if decider is None:
            # Full tie (T16): both theories say 0.5; no information.
            continue

        resp = int(r_raw)
        follows.append(1.0 if resp == decider else 0.0)

    if len(follows) == 0:
        return 0.0

    # Deviation of the tied-trial cue-follow rate from chance.
    return float(np.mean(follows)) - 0.5
```

**Observed (real) value:** -0.2889 (var=0.0205)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0011 (var=0.0175)
- pi_3_2: 0.1878 (var=0.0164)
- pi_1: 0.3422 (var=0.0196)
- pi_3: 0.0944 (var=0.0115)
- pi_3_1: 0.2311 (var=0.0115)
- pi_4: -0.1644 (var=0.0172)
- pi_5: -0.2356 (var=0.0097)
- pi_6: -0.2433 (var=0.0122)
- pi_7: -0.2122 (var=0.0106)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dissociation metric: fraction of choices matching the advocated
    # (Reversed-Hierarchy Frugality) cascade winner, computed ONLY on
    # trials where the two theories' cue hierarchies yield OPPOSITE
    # predictions. On those trials the advocated theory predicts a
    # match rate of p = (1-eps)*sigmoid(beta) + eps/2 ~ 0.62-0.80,
    # while the competing theory predicts the complementary rate
    # ~ 1-p ~ 0.20-0.40, so the two theories produce visibly
    # different numbers on this scalar.
    val = np.array([0.9, 0.8, 0.95, 0.7, 0.6, 0.65])
    n = len(val)

    # Advocated hierarchy: subjective weights = reversed validity list.
    w_adv = val[::-1]  # [0.65, 0.6, 0.7, 0.95, 0.8, 0.9]
    order_adv = np.argsort(-w_adv, kind="stable")  # f3, f5, f4, f2, f0, f1

    # Competing hierarchy: ascending advertised validity.
    order_comp = np.lexsort((np.arange(n), val))  # f4, f5, f3, f1, f0, f2

    def cascade_winner(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1  # no discriminating cue

    matches = []
    for ra, rb, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = [int(v) for v in ra]
        b = [int(v) for v in rb]
        r = int(resp)

        wa = cascade_winner(a, b, order_adv)
        wc = cascade_winner(a, b, order_comp)

        # Keep only trials that genuinely dissociate the two theories:
        # both cascades terminate and they pick different options.
        if wa < 0 or wc < 0 or wa == wc:
            continue

        matches.append(1.0 if r == wa else 0.0)

    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4280 (var=0.0052)
**Other theories' values on this metric (for reference):**
- pi_3_2: 0.6607 (var=0.0064)
- pi_4: 0.3193 (var=0.0078)
- pi_1: 0.8787 (var=0.0117)
- pi_2: 0.5133 (var=0.0067)
- pi_3: 0.6033 (var=0.0089)
- pi_3_1: 0.7033 (var=0.0066)
- pi_5: 0.3993 (var=0.0160)
- pi_6: 0.4400 (var=0.0202)
- pi_7: 0.3867 (var=0.0066)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # ---------- design constants ----------
    NF = 6
    VAL = [0.9, 0.65, 0.6, 0.8, 0.95, 0.7]
    # Weakest-Expert-First cascade: cues in ASCENDING validity
    adv_order = sorted(range(NF), key=lambda j: VAL[j])      # [2,1,5,3,0,4]
    # Reversed-Hierarchy (misbinding) cascade: DESCENDING val[::-1]
    rev = VAL[::-1]
    comp_order = sorted(range(NF), key=lambda j: -rev[j])    # [1,5,2,0,4,3]

    # ---------- robust cell parsers ----------
    def parse_vec(x):
        try:
            if isinstance(x, np.ndarray):
                v = np.asarray(x, dtype=int).ravel()
                return v if v.size == NF else None
            if isinstance(x, (list, tuple)):
                v = np.asarray(list(x), dtype=float).ravel()
                if v.size != NF:
                    return None
                return np.rint(v).astype(int)
            s = str(x).strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            elif s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            toks = s.replace(",", " ").replace(";", " ").split()
            if len(toks) == 1:
                t0 = toks[0].strip("'\"")
                if len(t0) == NF and all(ch in "01" for ch in t0):
                    return np.asarray([int(ch) for ch in t0], dtype=int)
                return None
            vals = []
            for t in toks:
                vals.append(int(round(float(t.strip("'\"")))))
            v = np.asarray(vals, dtype=int)
            return v if v.size == NF else None
        except Exception:
            return None

    def parse_resp(x):
        try:
            if isinstance(x, (bool, np.bool_)):
                return int(x)
            if isinstance(x, (int, np.integer)):
                r = int(x)
                return r if r in (0, 1) else None
            if isinstance(x, (float, np.floating)):
                if np.isnan(x):
                    return None
                r = int(round(float(x)))
                return r if r in (0, 1) else None
            s = str(x).strip().upper()
            if s == "A":
                return 0
            if s == "B":
                return 1
            r = int(round(float(s)))
            return r if r in (0, 1) else None
        except Exception:
            return None

    # ---------- column resolution ----------
    cols = list(data.columns)
    lmap = {}
    for c in cols:
        lmap[str(c).lower()] = c

    def resolve(names, keys):
        for nm in names:
            if nm in lmap:
                return lmap[nm]
        for c in cols:
            lc = str(c).lower()
            if any(k in lc for k in keys):
                return c
        return None

    a_col = resolve(["option_a_ratings", "optiona_ratings", "ratings_a",
                     "a_ratings", "option_a"], ["a_rat", "option_a", "ratings_a"])
    b_col = resolve(["option_b_ratings", "optionb_ratings", "ratings_b",
                     "b_ratings", "option_b"], ["b_rat", "option_b", "ratings_b"])
    r_col = resolve(["response", "choice", "chosen", "answer", "pick",
                     "selected"], ["respon", "choic", "chose", "select", "answer", "pick"])
    s_col = resolve(["subject_id", "subjectid", "subject", "subj",
                     "participant"], ["subject", "subj", "participant"])
    if a_col is None or b_col is None or r_col is None:
        return 0.0

    def cascade(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # ---------- per-subject cascade-follow rates ----------
    n_rows = int(len(data))
    if s_col is not None:
        try:
            subjects = [str(v) for v in list(data[s_col])]
        except Exception:
            subjects = ["s%d" % i for i in range(n_rows)]
    else:
        subjects = ["s%d" % i for i in range(n_rows)]

    a_cells = list(data[a_col])
    b_cells = list(data[b_col])
    r_cells = list(data[r_col])

    stats = {}
    for i in range(n_rows):
        a = parse_vec(a_cells[i])
        b = parse_vec(b_cells[i])
        resp = parse_resp(r_cells[i])
        if a is None or b is None or resp is None:
            continue
        wa = cascade(a, b, adv_order)
        if wa is None:
            continue  # complete tie: both theories guess 50/50
        wc = cascade(a, b, comp_order)
        if wc != wa:
            continue  # keep only trials where both hierarchies dictate the
                       # same option, so the follow rate is theory-neutral
        rec = stats.get(subjects[i])
        if rec is None:
            rec = [0, 0]
            stats[subjects[i]] = rec
        rec[1] += 1
        if resp == wa:
            rec[0] += 1

    rates = []
    for k in sorted(stats.keys()):
        f, t = stats[k]
        if t > 0:
            rates.append(f / float(t))
    if not rates:
        return 0.0

    x = np.asarray(rates, dtype=float)
    n = int(x.size)
    m = float(x.mean())
    if n > 1:
        dm = np.abs(x[:, None] - x[None, :])
        g = float(dm.sum() / (n * (n - 1.0)))  # Gini mean difference (ordered pairs)
    else:
        g = 0.0

    # ---------- population-level pairwise conformity-dispersion index ----------
    # Every ordered subject pair (i, j) contributes the anchor excess of its
    # first member, (x_i - 0.5), plus half the pair's follow-rate distance,
    # 0.5 * |x_i - x_j|.  Summed over all n^2 ordered pairs this equals
    #   n^2 * (mean(x) - 0.5) + 0.5 * n(n-1) * GiniMD(x).
    # For a single-subject slice it collapses to that subject's own excess
    # conformity (x - 0.5): the one diagonal pair contributes no distance.
    index = (float(n) ** 2) * (m - 0.5) + 0.5 * float(n) * (float(n) - 1.0) * g
    return float(index)
```

**Observed (real) value:** 116.5556 (var=0.0041)
**Other theories' values on this metric (for reference):**
- pi_4: 588.7556 (var=0.0036)
- pi_3_2: 536.1222 (var=0.0024)
- pi_1: -680.7000 (var=0.0054)
- pi_2: 105.3778 (var=0.0026)
- pi_3: 332.3111 (var=0.0024)
- pi_3_1: 567.5556 (var=0.0025)
- pi_5: 184.2778 (var=0.0153)
- pi_6: 184.6889 (var=0.0053)
- pi_7: 119.2111 (var=0.0025)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Anti-misbound decider-follow rate on hierarchy-dissociating cells.

    For each trial we compute the option picked by the competitor's
    hierarchy (weakest-advertised-expert-first: cues sorted by ASCENDING
    validity, ties by display position) and the option picked by the
    advocated theory's anti-misbound re-encoding (w = -val[::-1],
    consulted in descending subjective weight, which is equivalent to
    sorting cues by ASCENDING val[::-1], ties by display position), using
    the experiment's validities [0.80, 0.55, 0.90, 0.55, 0.80, 0.65].

    On this design the two hierarchies' first discriminating cues point to
    OPPOSITE options on 13 of the 16 unique pairs (T1-T13) and agree on
    T14/T15 (T16 is a full tie). The deciders are invariant to either
    theory's free tie_break parameter because no trial opposes two cues
    tied within a hierarchy group. The metric is the proportion of
    choices that follow the anti-misbound decider on those 13
    dissociating cells only.

    Predicted values:
      - Weakest-Expert-First (competing): subjects follow their own
        (central) decider with p_f ~ 0.70, hence follow the anti-misbound
        decider with ~0.30.
      - Advocated: ~95% of subjects are anti-misbinders (m_am =
    ambiguity x coherence-gate ~ 0.95 for this arrangement), so the
        follow rate toward the anti-misbound decider is ~0.70.
    The two theories therefore sit on opposite sides of 0.50 with a
    ~0.40 gap, while per-subject values are tight (each subject
    contributes 13 cells x 6 reps = 78 dissociating trials).
    """
    VAL = [0.80, 0.55, 0.90, 0.55, 0.80, 0.65]
    n = len(VAL)

    # Competitor hierarchy: ascending advertised validity, position ties.
    comp_order = sorted(range(n), key=lambda j: (VAL[j], j))

    # Anti-misbound hierarchy: w = -val[::-1] consulted in DESCENDING
    # subjective weight == ascending val[::-1], position ties.
    rev = VAL[::-1]
    mis_order = sorted(range(n), key=lambda j: (rev[j], j))

    def decider(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    hits = []
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = [int(v) for v in a_raw]
            b = [int(v) for v in b_raw]
        except Exception:
            continue
        if len(a) != n or len(b) != n:
            continue
        cd = decider(a, b, comp_order)
        md = decider(a, b, mis_order)
        # Keep only cells where the two hierarchies genuinely dissociate.
        if cd is None or md is None or cd == md:
            continue
        hits.append(1.0 if int(r) == int(md) else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4897 (var=0.0022)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4262 (var=0.0087)
- pi_4: 0.5215 (var=0.0033)
- pi_1: 0.3595 (var=0.0039)
- pi_2: 0.4626 (var=0.0037)
- pi_3: 0.5290 (var=0.0024)
- pi_3_1: 0.5915 (var=0.0016)
- pi_3_2: 0.5703 (var=0.0038)
- pi_6: 0.4649 (var=0.0080)
- pi_7: 0.5056 (var=0.0039)

### Experiment 12
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np

    # Reference hierarchy: the WEAKEST-EXPERT-FIRST (ascending-validity)
    # cascade for the design validity vector [0.75, 0.9, 0.65, 0.8, 0.85, 0.6]:
    # consult order f5 (0.60) -> f2 (0.65) -> f0 (0.75) -> f3 (0.80)
    # -> f4 (0.85) -> f1 (0.90). On every diagnostic trial below, the first
    # cue in this order that discriminates A from B defines the reference
    # side, and the metric measures how often the subject's choice lands
    # on that side.
    val = [0.75, 0.9, 0.65, 0.8, 0.85, 0.6]
    order = sorted(range(6), key=lambda j: val[j])

    # Diagnostic multi-cue trials: T3, T4, T7, T8, T9, T10, T11
    # (7 unique pairs x 6 repetitions = 42 trials per subject).
    # These are the cells on which the two theories' hierarchies diverge:
    # the reference decider is always the low-validity expert (f2 at 0.65
    # or f3 at 0.80), while the competing theory's anti-misbound component
    # (and its per-cue zeta distortion of either component) lands its
    # decider on a different, higher-validity expert for a large fraction
    # of subjects.
    diag = set()
    diag.add(((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 1, 1)))  # T3
    diag.add(((1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1)))  # T4 (mirror of T3)
    diag.add(((1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0)))  # T7
    diag.add(((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0)))  # T8 (mirror of T7)
    diag.add(((1, 1, 0, 0, 1, 1), (1, 1, 1, 1, 0, 1)))  # T9
    diag.add(((1, 1, 1, 1, 0, 1), (1, 1, 0, 0, 1, 1)))  # T10 (mirror of T9)
    diag.add(((0, 1, 0, 0, 0, 0), (1, 0, 1, 0, 1, 0)))  # T11

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    hits = []
    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        try:
            a = tuple([int(v) for v in a_raw])
            b = tuple([int(v) for v in b_raw])
            r = int(r_raw)
        except Exception:
            continue
        if len(a) != 6 or len(b) != 6:
            continue
        if (a, b) not in diag:
            continue
        # Reference decider: first strictly discriminating cue in the
        # ascending-validity (weakest-expert-first) consult order.
        side = None
        for j in order:
            if a[j] != b[j]:
                side = 0 if a[j] > b[j] else 1
                break
        if side is None:
            continue
        hits.append(1.0 if r == side else 0.0)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5324 (var=0.0041)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7076 (var=0.0076)
- pi_5: 0.5895 (var=0.0200)
- pi_1: 0.3476 (var=0.0042)
- pi_2: 0.7433 (var=0.0076)
- pi_3: 0.5343 (var=0.0067)
- pi_3_1: 0.5810 (var=0.0066)
- pi_3_2: 0.5752 (var=0.0040)
- pi_6: 0.5900 (var=0.0093)
- pi_7: 0.5586 (var=0.0062)

### Experiment 13
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Follow-the-anti-misbound-decider rate on the four trial types where the
    # anti-misbound hierarchy (subjective weights w = -validities[::-1], i.e.
    # consult {f1,f3} -> f5 -> f2 -> {f0,f4}) is pitted head-to-head against
    # the position-primacy hierarchy, and where the competitor's anti-validity
    # and TTB components split away from its primacy mass: T1, T2, T9, T10
    # (24 of the 96 trials per subject).
    #
    # Deciders are recomputed inside the metric with EXACTLY the same
    # conventions as both theories' predict functions (w = -val[::-1],
    # np.lexsort((secondary, -w))), and a trial is included only if the
    # decider is identical under BOTH tie-break conventions (verified to
    # hold on all four types: the f1/f3 and f0/f4 weight ties never
    # internally oppose here), so the classification is exact and
    # tie-break independent. Zeta distortion is OFF in both theories for
    # this design (validity ties present), so per-subject deciders are
    # deterministic given the mixture component.
    #
    # Predicted values:
    #   ADVOCATED (m_am = 0.912 * sigmoid((0.727-0.45)/0.08) = 0.884):
    #     88.4% anti-misbound subjects follow this decider at p_f ~ 0.73;
    #     11.6% anti-validity subjects oppose it on T1/T2 and follow it on
    #     T9/T10 -> 0.50. Population ~ 0.70, per-subject SD ~ 0.13.
    #   COMPETING (primacy 0.573, anti-val 0.353, recency 0.043, TTB 0.031):
    #     primacy subjects OPPOSE the misbound decider on all four types
    #     (their f0 decider points the other way) -> 1-p_f ~ 0.27;
    #     anti-validity and TTB subjects sit at 0.50 (oppose on T1/T2 via
    #     f5, follow on T9/T10 via f1); recency -> 0.27 (f5/f4 point
    #     against on all four). Population ~ 0.36, per-subject SD ~ 0.15.
    # Gap ~ 0.34 with tight per-subject spreads on both sides: Welch t at
    # N=25 ~ 8-9, far beyond alpha = 0.01. Values near 0.7 support the
    # anti-misbound mixture; values near 0.35 support the position mixture.
    n_features = 6
    val = np.array([0.65, 0.90, 0.50, 0.80, 0.50, 0.90], dtype=float)

    # Anti-misbound subjective weights (identical to the advocated theory's
    # w_base = -val[::-1]).
    w_misbound = -val[::-1]

    def decider(a, b, w, later_first=False):
        # Consult cues in DESCENDING subjective weight; same lexsort
        # convention as both predict functions.
        n = len(w)
        pos = np.arange(n)
        secondary = (n - 1 - pos) if later_first else pos
        order = np.lexsort((secondary, -w))
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # The four head-to-head trial types: T1, T2, T9, T10.
    trials = [
        ([0, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1]),  # T1
        ([1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 0]),  # T2 (mirror of T1)
        ([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1]),  # T9
        ([1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1]),  # T10 (mirror of T9)
    ]

    target = {}
    for a_list, b_list in trials:
        a = np.asarray(a_list, dtype=int)
        b = np.asarray(b_list, dtype=int)
        d_early = decider(a, b, w_misbound, later_first=False)
        d_late = decider(a, b, w_misbound, later_first=True)
        # Include only if the decider is tie-break independent.
        if d_early is None or d_late is None or d_early != d_late:
            continue
        key = (tuple([int(v) for v in a]), tuple([int(v) for v in b]))
        target[key] = int(d_early)

    if not target:
        return 0.5

    hits = []
    for ra, rb, resp in zip(data["option_a_ratings"],
                            data["option_b_ratings"],
                            data["response"]):
        a = tuple([int(v) for v in np.asarray(ra).ravel()])
        b = tuple([int(v) for v in np.asarray(rb).ravel()])
        key = (a, b)
        if key in target:
            hits.append(1.0 if int(resp) == target[key] else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1500 (var=0.0143)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6542 (var=0.0400)
- pi_6: 0.2417 (var=0.0078)
- pi_1: 0.4908 (var=0.0063)
- pi_2: 0.1350 (var=0.0085)
- pi_3: 0.4050 (var=0.0097)
- pi_3_1: 0.2858 (var=0.0074)
- pi_3_2: 0.3150 (var=0.0106)
- pi_4: 0.2950 (var=0.0089)
- pi_7: 0.2758 (var=0.0171)

### Experiment 14
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Primacy-type excess vs validity-respecting-type excess.

    Two cell families are selected by their discriminating-cue sets
    (feature positions where the two options' ratings differ):

    Family 1 ("early-cue cells"; design trials T1-T4, T7-T8, T13, T15):
        sets {1,4}, {2,4}, {1,5}, {1,3}, {1,3,4}. The earliest-listed
        discriminating expert (f1 or f2) favors the opposite option from
        every later discriminating expert, so a position-primacy reader
        and an anti-validity (weakest-first) reader pick opposite options.

    Family 2 ("weak-cue cells"; design trials T5-T6, T9-T10):
        sets {3,4}, {3,5}. The earliest-listed discriminating expert is
        the low-validity f3, opposing f4/f5, so primacy and anti-validity
        readers disagree here as well, but a descending-validity (TTB-like
        or anti-misbound) reader now sides with the anti-validity reader.

    Per subject: X1 = rate of choosing the earliest-cue option on
    Family 1, X2 = same on Family 2. Score = +1 if the subject follows
    the earliest-listed expert on BOTH families (pure position-primacy
    consultation order), -1 if it follows it on Family 1 but reverses on
    Family 2 (validity-respecting order), 0 otherwise. The metric is the
    mean score: the excess of primacy-type subjects over
    validity-respecting-type subjects.
    """
    fam1 = {frozenset((1, 4)), frozenset((2, 4)), frozenset((1, 5)),
            frozenset((1, 3)), frozenset((1, 3, 4))}
    fam2 = {frozenset((3, 4)), frozenset((3, 5))}

    def _resp(v):
        if isinstance(v, str):
            return 0 if v.strip().upper() in ("A", "0") else 1
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    scores = []
    for _, sdf in data.groupby("subject_id"):
        f1, f2 = [], []
        for _, row in sdf.iterrows():
            try:
                a = np.asarray(row["option_a_ratings"], dtype=int).ravel()
                b = np.asarray(row["option_b_ratings"], dtype=int).ravel()
            except (TypeError, ValueError):
                continue
            if a.size == 0 or a.size != b.size:
                continue
            diff = np.nonzero(a != b)[0]
            if diff.size == 0:
                continue
            dset = frozenset(int(j) for j in diff)
            in1 = dset in fam1
            in2 = dset in fam2
            if not (in1 or in2):
                continue
            j0 = int(diff[0])  # earliest-listed discriminating expert
            fav0 = 0 if a[j0] == 1 else 1  # option favored by that cue
            # earliest cue must oppose every later discriminating cue
            ok = True
            for j in diff[1:]:
                if (0 if a[int(j)] == 1 else 1) == fav0:
                    ok = False
                    break
            if not ok:
                continue
            r = _resp(row["response"])
            if r not in (0, 1):
                continue
            follow = 1.0 if r == fav0 else 0.0
            if in1:
                f1.append(follow)
            else:
                f2.append(follow)
        if not f1 or not f2:
            continue
        x1 = float(np.mean(f1))
        x2 = float(np.mean(f2))
        if x1 <= 0.5:
            s = 0.0
        elif x2 > 0.5:
            s = 1.0
        elif x2 < 0.5:
            s = -1.0
        else:
            s = 0.0
        scores.append(s)

    if not scores:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.1600 (var=0.2544)
- pi_5: -0.3600 (var=0.2304)
- pi_1: -1.0000 (var=0.0000)
- pi_2: -0.1400 (var=0.2004)
- pi_3: 0.0200 (var=0.0196)
- pi_3_1: 0.0000 (var=0.0000)
- pi_3_2: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_7: 0.0000 (var=0.0000)

### Experiment 15
**Design**
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Count-vs-Primacy planned contrast for one-reason binary-feature choice.
    #
    # For every trial we compute, directly from the rating vectors:
    #   diff[j] = +1 if expert j rates A=1,B=0 (favors A), -1 if B=1,A=0,
    #   n      = number of discriminating experts,
    #   d      = sum(diff) = (#A-favoring cues) - (#B-favoring cues),
    #   e      = diff of the EARLIEST (lowest-index) discriminating cue (+1 -> A).
    #
    # The trial weight is  w = d/n - e , i.e. the count-based choice
    # direction minus the primacy-based choice direction, and the metric is
    # the mean of w * y with y = +1 for choosing A, -1 for choosing B.
    #
    # Under the advocated momentary-diffusion theory the first consulted
    # discriminating cue is ~uniform over the discriminators, so
    # E[y] ~ s * d/n  (s = 2*p_f - 1) and the metric concentrates near
    # +s * 0.26 ~ +0.11 (tight across subjects: only binomial noise plus a
    # small s-spread; full-tie trials carry no weight and the single-cue
    # control trial gets w = 0 automatically).
    #
    # Under the competing discrete position-mixture theory the dominant
    # primacy component (mass ~0.54-0.59 at this design's ambiguity, with
    # anti/recency/TTB minorities) makes E[y] ~ s * e on essentially every
    # cell, so the metric is strongly negative (population ~ -0.28), with
    # large between-subject spread from the fixed per-subject type draw.
    # The contrast is validity-vector robust: the earliest-discriminator and
    # count structures are fixed by the trial list, and the sign of the gap
    # holds for either advertised-validity arrangement.
    df = data
    vals = []
    a_col = df["option_a_ratings"]
    b_col = df["option_b_ratings"]
    r_col = df["response"]
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.ndim != 1 or a.size == 0:
            continue
        diff = a - b
        disc = np.where(diff != 0)[0]
        n = disc.size
        if n == 0:
            # Full tie (no discriminating expert): no structural signal.
            continue
        d = float(diff.sum())            # count margin: A-cues minus B-cues
        e = float(diff[disc[0]])        # earliest discriminator: +1 -> A, -1 -> B
        w = d / float(n) - e            # count direction minus primacy direction
        y = 1.0 if int(resp) == 0 else -1.0   # response 0 = chose A
        vals.append(w * y)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.1541 (var=0.0151)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0890 (var=0.0148)
- pi_6: -0.1977 (var=0.1728)
- pi_1: 0.5590 (var=0.0378)
- pi_2: 0.4101 (var=0.0246)
- pi_3: -0.0365 (var=0.0123)
- pi_3_1: -0.0876 (var=0.0103)
- pi_3_2: -0.0896 (var=0.0137)
- pi_4: -0.0330 (var=0.0125)
- pi_5: 0.0927 (var=0.0619)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-fixed advertised validities (positions f0..f5).
    VALIDITIES = np.array([0.80, 0.50, 0.85, 0.60, 0.90, 0.95])
    # A subject whose choice rate on primacy-vs-anti conflict trials exceeds
    # this threshold is classified as a position-primacy (earliest-listed-
    # expert) decider. 0.55 sits well above the anti-followers' ceiling
    # (~0.27-0.36 incl. the competing theory's momentary diffusion) and well
    # below the primacy-followers' level (~0.73).
    THRESHOLD = 0.55
    MIN_CONFLICT_TRIALS = 10

    def to_arr(x):
        if isinstance(x, str):
            s = x.strip().strip("[]()")
            return np.array([float(v) for v in s.replace(",", " ").split()])
        return np.asarray(x, dtype=float).ravel()

    df = data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return 0.0

    subj = df["subject_id"].values
    resp = np.asarray(df["response"].values, dtype=float)
    a_col = df["option_a_ratings"].values
    b_col = df["option_b_ratings"].values

    is_conflict = np.zeros(n, dtype=bool)
    prim_dir = np.zeros(n, dtype=int)

    for i in range(n):
        a = to_arr(a_col[i])
        b = to_arr(b_col[i])
        if a.size != b.size or a.size != VALIDITIES.size:
            continue
        disc = a != b
        if not disc.any():
            continue
        d_idx = np.where(disc)[0]
        # Primacy decider: earliest-listed discriminating expert.
        e = int(d_idx[0])
        # Anti-validity decider: weakest advertised discriminating expert.
        anti = int(d_idx[np.argmin(VALIDITIES[d_idx])])
        pdir = 0 if a[e] > b[e] else 1
        adir = 0 if a[anti] > b[anti] else 1
        if pdir != adir:
            is_conflict[i] = True
            prim_dir[i] = pdir

    subjects = pd.unique(df["subject_id"])
    flags = []
    for s in subjects:
        m = (subj == s) & is_conflict
        cnt = int(m.sum())
        if cnt < MIN_CONFLICT_TRIALS:
            flags.append(0.0)
            continue
        rate = float(np.mean(resp[m] == prim_dir[m]))
        flags.append(1.0 if rate > THRESHOLD else 0.0)

    if not flags:
        return 0.0
    return float(np.mean(flags))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3000 (var=0.2100)
- pi_7: 0.0000 (var=0.0000)
- pi_1: 0.9600 (var=0.0384)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.5200 (var=0.2496)
- pi_3_1: 0.8400 (var=0.1344)
- pi_3_2: 0.7800 (var=0.1716)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0800 (var=0.0736)


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
```json
{
  "theory": "Anchored Anti-Validity Frugality with Validity-Respecting Leak. People are one-reason decision makers: experts are consulted one at a time and the first strictly discriminating expert decides; there is no integration across cues (flat psychometric profile over tally margins). The population-central consultation hierarchy is ANTI-VALIDITY (distrust-the-weakest: consult the lowest advertised validity first), and the population is TYPELESS: every subject is a continuous perturbation around that core (no discrete per-subject component draws, no primacy/recency types), which is what the data demand (zero primacy types in the type-classification experiments, binomial-floor between-subject variance on the tight designs). Three graded, design-scaled mechanisms modulate the core. (1) GAP-PROPORTIONAL ANCHORING: the per-trial attention noise on expert j is scaled by the local validity gaps adjacent to j in the subjective order — noise_sd_j = kappa * A^3 * (ref_gap / (gap_to_next_weaker_j + gap_to_next_stronger_j)) — so experts isolated by large validity gaps act as anchors (little momentary diffusion) while experts crowded in dense validity clusters fluctuate heavily; the whole distortion is gated by binding ambiguity A = 1 - |Spearman(position, validity)| so that anchored (monotone) designs stay anchored and small-gap, high-ambiguity designs remain fully diffuse. (2) VALIDITY-RESPECTING LEAK: on a per-trial Bernoulli draw with design-level probability q = clip(rho_leak * reversal_coherence * mean_validity_gap, 0, 0.10), the decider consults in standard TTB (descending validity) order instead — a small continuous component of subjective weight proportional to +validity that pulls a fraction of trials toward the strongest discriminator. When a reversed cascade's first cue is a mid- or low-validity expert, this leak flips the decider toward the strongest discriminating expert, generating below-chance reversed-cascade following. (3) DESIGN-SCALED RESPONSE DISCIPLINE: the follow probability p_f = (1-eps)*sigmoid(beta_eff) + eps/2 with beta_eff = beta*(1 + boost*(1-A)) — anchored designs get a higher effective beta (tighter following), ambiguous designs sit at the lower end, targeting p_f in [0.60, 0.75]. Exact validity ties anchor the order (no diffusion); noise is softmax over the binary winner score plus an independent lapse; no discriminating cue yields exactly 0.5; history is ignored.",
  "predict": "def predict(parameters, state, history):\n    # Anchored Anti-Validity Frugality with Validity-Respecting Leak.\n    #\n    # One-reason decision making: experts are consulted one at a time in\n    # DESCENDING subjective weight; the first strictly discriminating expert\n    # decides; no integration across cues.\n    #\n    # Subjective weights (continuous, per subject, TYPELESS population):\n    #     core:      w_j = -val_j + gamma * salience_j\n    #     diffusion: w_j += kappa * A^3 * gap_ratio_j * (xi_j + sigma_h*zeta_j)\n    #                (xi re-drawn EVERY TRIAL; only when all validities\n    #                are distinct -- exact ties anchor the order)\n    #     leak:      with per-trial probability\n    #                q = clip(rho_leak * coherence * mean_gap, 0, 0.10)\n    #                the trial instead uses the standard TTB order\n    #                (descending validity) -- a graded validity-respecting\n    #                component that pulls a fraction of trials toward the\n    #                strongest discriminator.\n    #\n    # GAP-PROPORTIONAL ANCHORING: gap_ratio_j = ref_gap / (gap to the next\n    # weaker + gap to the next stronger neighbor of j in validity space).\n    # Experts isolated by large validity gaps are ANCHORS (small momentary\n    # diffusion); experts in dense validity clusters fluctuate heavily.\n    # The whole distortion is gated by binding ambiguity A (cubed), so\n    # monotone anchored designs stay anchored while small-gap,\n    # high-ambiguity designs remain fully diffuse.\n    #\n    # Response discipline: p_f = (1-eps)*sigmoid(beta_eff) + eps/2 with\n    # beta_eff = beta * (1 + boost*(1-A)): anchored designs follow the\n    # decider more tightly, ambiguous designs sit at the lower end,\n    # targeting p_f in [0.60, 0.75].\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    n_features = stim.shape[1]\n\n    val = np.asarray(parameters[\"validities\"], dtype=float).ravel()\n    if val.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {val.shape[0]} != n_features {n_features}.\"\n        )\n\n    beta = float(parameters[\"beta\"])\n    boost = float(parameters[\"boost\"])\n    epsilon = float(parameters[\"epsilon\"])\n    gamma = float(parameters[\"gamma\"])\n    kappa = float(parameters[\"kappa\"])\n    sigma_h = float(parameters[\"sigma_h\"])\n    rho_leak = float(parameters[\"rho_leak\"])\n    tie_break = float(parameters[\"tie_break\"])\n\n    zeta = np.asarray(parameters[\"zeta\"], dtype=float).ravel()\n    if zeta.shape[0] != n_features:\n        raise ValueError(\n            f\"zeta length {zeta.shape[0]} != n_features {n_features}.\"\n        )\n\n    # ---- helper: average-tie ranks ----\n    def avg_ranks(x):\n        n = x.shape[0]\n        srt = np.argsort(x, kind=\"stable\")\n        ranks = np.empty(n, dtype=float)\n        i = 0\n        while i < n:\n            j = i\n            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:\n                j += 1\n            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank\n            i = j + 1\n        return ranks\n\n    # ---- positional-salience gradient (early-listed experts favored) ----\n    if n_features > 1:\n        sal = ((n_features - 1 - np.arange(n_features, dtype=float))\n               / (n_features - 1.0)) - 0.5\n    else:\n        sal = np.zeros(1, dtype=float)\n\n    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----\n    if n_features > 1:\n        ranks = avg_ranks(val)\n        pos_r = np.arange(n_features, dtype=float)\n        vp = pos_r - pos_r.mean()\n        vr = ranks - ranks.mean()\n        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))\n        if denom > 1e-12:\n            rho = float((vp * vr).sum()) / denom\n        else:\n            rho = 0.0\n        ambiguity = 1.0 - abs(rho)\n    else:\n        ambiguity = 0.0\n\n    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----\n    if n_features > 1:\n        ranks_v = avg_ranks(val)\n        ranks_r = avg_ranks(val[::-1])\n        dv = ranks_v - ranks_v.mean()\n        dr = ranks_r - ranks_r.mean()\n        denom2 = np.sqrt(float((dv * dv).sum()) * float((dr * dr).sum()))\n        if denom2 > 1e-12:\n            rho_rev = float((dv * dr).sum()) / denom2\n        else:\n            rho_rev = 0.0\n        coherence = abs(rho_rev)\n    else:\n        coherence = 0.0\n\n    # ---- mean adjacent validity gap over UNIQUE validities ----\n    uvals = np.unique(val)\n    if uvals.size > 1:\n        mean_gap = float(np.mean(np.diff(np.sort(uvals))))\n    else:\n        mean_gap = 0.0\n\n    distinct = (uvals.size == n_features)\n\n    # ---- gap-proportional anchoring ratios ----\n    # gap_ratio_j = ref_gap / (gap to next weaker + gap to next stronger).\n    # Large local gaps -> ratio << 1 -> the expert anchors the order;\n    # dense clusters -> ratio up to the cap -> the expert diffuses.\n    ref_gap = 0.10\n    gap_ratio = np.ones(n_features, dtype=float)\n    if distinct and n_features > 1:\n        srt = np.argsort(val, kind=\"stable\")\n        sv = val[srt]\n        gs = np.ones(n_features, dtype=float)\n        for r in range(n_features):\n            g = 0.0\n            if r > 0:\n                g += sv[r] - sv[r - 1]          # gap to next weaker\n            if r < n_features - 1:\n                g += sv[r + 1] - sv[r]          # gap to next stronger\n            gs[srt[r]] = max(g, 0.05)           # numerical floor\n        gap_ratio = np.minimum(ref_gap / gs, 2.0)\n\n    # ---- validity-respecting leak probability (design-level, continuous) ----\n    leak_q = float(np.clip(rho_leak * coherence * mean_gap, 0.0, 0.10))\n\n    # ---- per-trial subjective hierarchy ----\n    pos = np.arange(n_features)\n    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)\n\n    if np.random.random() < leak_q:\n        # VALIDITY-RESPECTING LEAK: this trial consults in standard TTB\n        # order (descending advertised validity). When a reversed\n        # cascade's first cue is a mid- or low-validity expert, this\n        # flips the decider toward the strongest discriminator.\n        w = val.copy()\n    else:\n        # ANTI-VALIDITY CORE + graded positional salience.\n        w = -val + gamma * sal\n        # GAP-PROPORTIONAL MOMENTARY DIFFUSION (distinct lists only;\n        # exact validity ties anchor the distrust order).\n        if distinct and ambiguity > 0.0:\n            amb3 = ambiguity * ambiguity * ambiguity\n            xi = np.random.normal(0.0, 1.0, size=n_features)\n            w = w + amb3 * kappa * gap_ratio * (xi + sigma_h * zeta)\n\n    # Consult experts in DESCENDING subjective weight; ties broken by the\n    # free per-subject tie_break (~80% early display position).\n    cue_order = np.lexsort((secondary, -w))\n\n    a, b = stim[0], stim[1]\n    winner = None\n    for j in cue_order:\n        if a[j] > b[j]:\n            winner = 0\n            break\n        if b[j] > a[j]:\n            winner = 1\n            break\n\n    n_opts = 2\n    if winner is None:\n        # No discriminating cue: pure guess (lapse mixing leaves a\n        # uniform unchanged).\n        return np.ones(n_opts) / float(n_opts)\n\n    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])\n\n    # Design-scaled response discipline: anchored designs (low ambiguity)\n    # get a higher effective inverse temperature, so the decider-follow\n    # probability p_f = (1-eps)*sigmoid(beta_eff) + eps/2 sits in\n    # [0.60, 0.75] with the lower end on ambiguous designs.\n    beta_eff = beta * (1.0 + boost * (1.0 - ambiguity))\n\n    # Numerically stable softmax over the binary winner score.\n    z = beta_eff * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))",
  "policy": "def policy(probs):\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # guard against float drift\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "beta": "[0.4, 1.4]",
    "boost": "[0.2, 0.5]",
    "epsilon": "[0.08, 0.20]",
    "gamma": "[-0.04, 0.06]",
    "kappa": "[6, 11]",
    "sigma_h": "[0.1, 0.35]",
    "rho_leak": "[0.25, 0.75]",
    "zeta": "[(-1, 1)] * n_features",
    "tie_break": "[0, 1]",
    "validities": "validities"
  },
  "rationology": null,
  "rationale": "This theory implements the arbiter's prescribed family — Anchored Anti-Validity Frugality with Validity-Respecting Leak — while explicitly preserving everything the ACCEPTED base (pi_7, score 0.894) already gets right, per the loss trajectory (pi_6's discrete-type mixture was rejected at 0.729; its fatal flaw was discrete per-subject component draws that produced spurious primacy types and huge between-subject variance). (1) TYPELESS ANTI-VALIDITY CORE: w = -validity + a narrow graded positional salience (gamma now in [-0.04, +0.06], mean ~0.01 instead of pi_7's +0.035) — the narrower gamma keeps the type-classification experiments at exactly 0 primacy types while removing pi_7's small over-tilt toward early-position/high-validity experts that inflated its tally-alignment miss on the monotone six-cue experiment (0.103 vs real 0.036). (2) GAP-PROPORTIONAL ANCHORING replaces pi_7's design-invariant kappa*A^3 uniform diffusion: the per-expert momentary noise is scaled by ref_gap/(gap_to_next_weaker + gap_to_next_stronger), so validity-isolated experts anchor the consultation order while dense validity clusters diffuse — this adds diffusion precisely at the weakest experts (which the anti-validity order consults first) on the small-gap ambiguous designs where pi_7 slightly over-follows, while keeping the anchored monotone designs (where pi_7's A^3 gate already yields near-zero noise) untouched. I retain the ambiguity-cubed gate on the diffusion (the arbiter's examples of fully diffuse designs are exactly the high-ambiguity ones), because removing it entirely would inject heavy diffusion into the monotone distinct designs that pi_7 fits well. (3) VALIDITY-RESPECTING LEAK: a per-trial Bernoulli (NOT a per-subject type) with design-level probability q = rho_leak * reversal_coherence * mean_gap, capped at 0.10, switches that trial's hierarchy to standard TTB. This is the mechanism that pulls a fraction of trials toward the strongest discriminator: it attenuates the over-following of the anti-decider on the knife-edge ladder designs (where pi_7 over-predicts follow strength by ~0.11), and when a reversed cascade's first cue is a mid- or low-validity expert it flips the decider toward the strongest discriminator, producing the below-chance reversed-cascade following both prior theories under-predict. The cap at 0.10 protects the pure anti-validity-follow experiments (e.g., the ascending-validity follow design, real 0.453) from being dragged toward TTB, and the coherence x gap scaling keeps the leak near zero on designs whose validity structure gives the anti-order no traction. (4) DESIGN-SCALED RESPONSE DISCIPLINE: beta_eff = beta*(1 + boost*(1-A)) raises the follow probability on anchored designs (where the data demand tight following, p_f ~0.73) and lowers it on ambiguous designs (p_f ~0.68), jointly with the leak targeting the [0.60, 0.75] corridor the arbiter specified; this combination cancels the leak's cost on the anchored anti-follow designs while amplifying its benefit on the ambiguous over-following ones. (5) Exact validity ties anchor the order (no diffusion), softmax-plus-lapse noise is retained, no-discriminating-cue trials are exactly 0.5, and prediction remains history-free. Net effect relative to the accepted base: reduced follow-strength overprediction on the knife-edge/ladder designs, reduced tally-alignment bias on monotone designs, stronger opposition to reversed cascades whose first cue is not the strongest expert, and unchanged behavior on the type-classification and tie-anchored designs where pi_7 already matches the real data."
}
```
```

## Usage

```json
{
  "input_tokens": 37964,
  "output_tokens": 66739,
  "reasoning_tokens": 62836,
  "provider": "Sail Research"
}
```
