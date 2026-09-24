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
- THEORY 1 = `pi_9`
- THEORY 2 = `pi_8_1`
- The recommendation below acts on THEORY 2 (= `pi_8_1`).

Replace pi_8_1 with a 'Small-Gap Composition-Gated Outlier Distrust' (SG-CGD) theory: (i) KEEP the shared, well-validated machinery verbatim — sign-vote tallying, trust plateau [v_lo, v_hi] with ceiling ~0.981 and shallow within-plateau tilt, consensus-gated chance flank with partial-ceiling promotion (kappa_gate*w_p), low-gain softmax plus uniform lapse. (ii) Outlier distrust: bounded, saturating shrinkage toward a floor with the peaked kernel centered at g_star ~ 0.05 (BODT's location, confirmed by Exps 14/15), with a RIGHT WALL steep enough that distrust has essentially receded by gap ~0.09 (s_r ~ 0.025) — this fixes Exps 1/2, where observed following at gap 0.09 (0.72 double-99 cells; 0.46 follow index) demands w_top ~ 0 to +0.25. (iii) Add a small POSITIVE 'edge credibility' ramp for well-separated outliers: for gap >= ~0.09, w_top rises toward the plateau value (a 99% expert sitting far above a coherent 0.85-0.90 rung is treated as genuinely more informative, not fraudulent), capturing Exp 1's 0.72 and Exp 18's 0.55 at gap 0.14. (iv) Composition amplification exists ONLY as a narrow near-kernel centered g_near ~ 0.03-0.04 (width ~0.015), gated by a chance-fraction logistic with c_half ~ 0.40-0.45, and is STRUCTURALLY INERT for gap >= ~0.07 — this reproduces the c_frac modulation at gap 0.05 (Exp 15: 0.29 at c_frac 0.55; Exp 14: 0.45 at c_frac 0.30) while guaranteeing no amplification at gaps 0.09-0.14 (Exps 1, 18). (v) NO far-fraud route below gap ~0.15; if any far route is retained at all, center it at g_far >= 0.18 with very sharp walls (EOBOT's placement), so it is closed on every panel tested here. (vi) Nudge the uniform-skepticism floor / base distrust slightly upward relative to both current calibrations to close the Exp 9 residual (observed anti-follow 0.58 vs predicted 0.40-0.45). Calibration targets across all 18 experiments: Exp 1 ~0.70, Exp 2 ~0.46, Exp 14 ~0.45, Exp 15 ~0.29, Exp 16 ~0.57, Exp 17 ~0.52, Exp 18 ~0.55, Exp 9 ~0.55-0.60, with all shared-machinery experiments (3-8, 10-13) held within their current ~0.01-0.03 agreement. This new theory is a direct, falsifiable competitor to EOBOT: it agrees with EOBOT on the small-gap composition gate but predicts the opposite sign of EOBOT's relocated peak at gap 0.09, where the incumbent is now known to over-distrust.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_9` (overall score: 0.803)

**Description**
Edge-Only Bounded Outlier Tallying (EOBOT), gate-widening calibration edition. Subjects integrate every binary expert rating as a sign vote (+1/-1/0) into a zero-sum evidence score with subjective weights computed RELATIVE to the panel: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) with a shallow within-plateau tilt; (ii) a small negative chance flank at v near 0.5, consensus-gated with a PARTIAL promotion cap kappa_gate*w_p when an above-ceiling claim destabilizes the scale; (iii) outlier distrust is a BOUNDED, SATURATING shrinkage toward a floor driven by the outlier's uncorroborated EDGE over the next rung — a peaked kernel RELOCATED to gap ~0.088 (so dense ladders and gap-0.05 singleton outliers on trusted panels are taken near face value, while gap-0.09 outliers are deeply distrusted), sitting on a uniform-skepticism floor e_min; (iv) the panel's chance-fraction AMPLIFIES distrust through two split, independently gated kernels: comp_near (its own narrow kernel centered at gap ~0.036, so majority/partial-chance panels deepen distrust only in the small-gap regime) and comp_far (a sharp logistic at gap ~0.18, so a far outlier floating over a chance-laden panel is treated as rigged while a gap-0.14 outlier is followed at face value). This round is a pure calibration claim on the loop-validated structure: the composition gate threshold c_half is LOWERED to [0.34, 0.38] so partially-chance-laden gap-0.05 singleton-99 panels (Exp 12, c_frac ~0.375) actually enter the composition regime (with c_near trimmed to hold the already-perfect double-99 cell, Exp 13, as its gate widens), kappa_lo is interpolated to its empirically bracketed midpoint for the chance-flank sensitivity cell (Exp 5), and e_min is nudged upward for the dense-ladder cell (Exp 9). Choice is a low-gain softmax plus a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Edge-Only Bounded Outlier Tallying (EOBOT), gate-widening edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU with shallow tilt: all v in [v_lo, v_hi]
    #       (v_hi just above 0.98, so a 98% expert stays INSIDE the
    #       plateau) get approximately equal weight w_p, with a
    #       within-band tilt so higher-validity cues slightly
    #       outrank lower ones.
    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small
    #       negative weight (-kappa_lo at v=0.5, rising linearly to
    #       the tilted plateau at v_lo). When an above-ceiling claim
    #       is present the flank is gated and chance cues are
    #       PROMOTED to a partial ceiling kappa_gate*w_p (counted,
    #       but weaker than genuine mid-band experts).
    # (iii) EDGE-ONLY BOUNDED OUTLIER DISTRUST, RELOCATED PEAK: the
    #       discount of the panel-max expert is a saturating
    #       shrinkage toward a floor,
    #         w_top_mod = w_floor + (w_p - w_floor)*exp(-rho_e*E(gap))
    #       with E a PEAKED kernel centered at g_star ~ 0.088 sitting
    #       on a uniform-skepticism floor e_min: near zero distrust
    #       on dense ladders AND on gap-0.05 singleton outliers
    #       (steep left wall s_l), deep distrust at gap ~0.09 (peak),
    #       receding by gap ~0.14 (right wall s_r).
    #  (iv) SPLIT COMPOSITION AMPLIFICATION, both gated by
    #       comp_gate(c_frac) with threshold c_half ~ 0.36 (LOWERED
    #       this round so partially-chance-laden gap-0.05 panels
    #       enter the regime):
    #         comp_near = c_near * sig_gap * L(gap) * comp_gate
    #       with L a NARROW kernel centered at g_near ~ 0.036 (fires
    #       only in the small-gap regime; inert at gap >= 0.09), and
    #         comp_far  = c_far * sig_gap * far(gap) * comp_gate
    #       with far() a SHARP logistic centered at g_far ~ 0.183:
    #       closed at gap 0.14 (a 0.99 outlier over a 0.85 rung is
    #       followed even on a majority-chance panel), fully open by
    #       gap ~0.24 (a far outlier over a chance-laden panel is a
    #       fraud signal).
    # A small bounded extremity term d_w*(v_top - v_hi)/(1 - v_hi)
    # is retained. All experts tied at the panel maximum share
    # w_top (double-99 panels double the anti-vote). Choice is a
    # low-gain softmax (inverse temperature beta) over the two
    # mirrored scores, plus a uniform lapse (epsilon).
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
    kappa_gate = float(parameters["kappa_gate"])
    tilt = float(parameters["tilt"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    lambda_ext = float(parameters["lambda_ext"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    e_min = float(parameters["e_min"])
    w_floor = float(parameters["w_floor"])
    rho_e = float(parameters["rho_e"])
    d_w = float(parameters["d_w"])
    g_near = float(parameters["g_near"])
    s_near = float(parameters["s_near"])
    c_near = float(parameters["c_near"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    c_far = float(parameters["c_far"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    def _sigmoid(x):
        # Numerically stable logistic via the identity
        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).
        return 0.5 * (1.0 + np.tanh(0.5 * x))

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

    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981
    # a 0.98-topped panel stays inside the trusted plateau; only
    # 99%+ claims enter the distrust branch.
    outlier = v_top > v_hi

    if outlier:
        sig_gap = _sigmoid((gap - g0) / s_gap)
    else:
        sig_gap = 0.0

    # ---- (ii): consensus gate for the chance flank. ----
    if outlier:
        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)
    else:
        gate = 0.0

    # ---- (i) + (ii): tilted plateau with a consensus-gated
    # chance flank whose promotion is capped at the PARTIAL
    # ceiling kappa_gate * w_p. ----
    w = np.full(n_features, w_p, dtype=float)
    span = max(v_hi - v_lo, 1e-6)
    plateau = (v >= v_lo) & (v <= v_hi)
    if np.any(plateau):
        # Shallow within-plateau validity gradient: higher-validity
        # experts slightly outrank lower ones while the band
        # remains 'approximately equal'.
        w[plateau] = w_p * (
            1.0 - tilt * (v_hi - v[plateau]) / span
        )
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        # Flank rises linearly to the TILTED plateau value at v_lo,
        # keeping the weight profile continuous at the band edge.
        w_lo_eff = w_p * (1.0 - tilt)
        w_flank = -kappa_lo + (w_lo_eff + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # With an above-ceiling claim in the panel, chance cues are
        # counted toward a partial ceiling kappa_gate*w_p rather
        # than inverted (and rather than fully trusted).
        w[flank] = (1.0 - gate) * w_flank + gate * kappa_gate * w_p

    # ---- (iii) + (iv): relocated edge-only bounded distrust with
    # the split, gate-widened composition amplification. ----
    if outlier:
        # (a) Peaked kernel RELOCATED to g_star ~ 0.088, sitting on
        # the uniform-skepticism floor e_min: near-zero distrust on
        # dense ladders (gap ~ 0.01) and on gap-0.05 singleton
        # outliers (steep left wall s_l), deep distrust at the
        # gap ~ 0.09 peak, receding right flank (s_r).
        width = s_l if gap < g_star else s_r
        E_kernel = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        E = max(E_kernel, e_min)

        # (b) BOUNDED, SATURATING shrinkage toward the floor:
        # w_top_mod in [w_floor, w_p] for every E.
        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_e * E))

        # (c) Split composition amplification. The chance-fraction
        # gate threshold c_half is LOWERED (~0.36) so PARTIALLY
        # chance-laden panels (c_frac ~ 0.375) enter the regime:
        #   comp_near fires only in the small-gap regime via its own
        #   narrow kernel L (centered g_near ~ 0.036), deepening the
        #   anti-vote for an uncorroborated near-rung outlier on a
        #   chance-laden scale; it is inert at gap >= 0.09.
        #   comp_far is the far-fraud route: a SHARP logistic at
        #   g_far ~ 0.183, closed at gap 0.14, fully open by 0.24.
        c_frac = float(np.mean(flank))
        comp_gate = _sigmoid((c_frac - c_half) / s_c)
        L = float(np.exp(-0.5 * ((gap - g_near) / s_near) ** 2))
        comp_near = c_near * float(sig_gap) * L * float(comp_gate)
        far = _sigmoid((gap - g_far) / s_far)
        comp_far = c_far * float(sig_gap) * float(far) * float(comp_gate)

        # (d) Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_top_mod - ext - comp_near - comp_far
        # All experts tied at the panel maximum share the distrust
        # (e.g., a pair of 0.99 super-experts doubles the anti-vote).
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
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.55]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.19, 0.22]
- kappa_gate: [0.66, 0.76]
- tilt: [0.08, 0.12]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- lambda_ext: [0.30, 0.42]
- g_star: [0.084, 0.092]
- s_l: [0.011, 0.015]
- s_r: [0.027, 0.033]
- e_min: [0.12, 0.16]
- w_floor: [-0.58, -0.52]
- rho_e: [1.15, 1.45]
- d_w: [0.08, 0.14]
- g_near: [0.032, 0.040]
- s_near: [0.016, 0.022]
- c_near: [1.9, 2.3]
- c_half: [0.34, 0.38]
- s_c: [0.05, 0.07]
- g_far: [0.178, 0.188]
- s_far: [0.018, 0.022]
- c_far: [4.2, 5.2]
- beta: [0.38, 0.46]
- epsilon: [0.035, 0.055]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.2180 (var=0.0140) vs this=-0.2460 (var=0.0055)
- Experiment 2: real=0.0771 (var=0.0031) vs this=0.1077 (var=0.0028)
- Experiment 3: real=0.6170 (var=0.0034) vs this=0.5933 (var=0.0037)
- Experiment 4: real=0.5133 (var=0.0042) vs this=0.5262 (var=0.0035)
- Experiment 5: real=-0.1683 (var=0.0124) vs this=-0.1667 (var=0.0190)
- Experiment 6: real=0.1180 (var=0.0074) vs this=0.1180 (var=0.0024)
- Experiment 7: real=0.5796 (var=0.0066) vs this=0.4013 (var=0.0051)
- Experiment 8: real=0.4625 (var=0.0103) vs this=0.4462 (var=0.0158)
- Experiment 9: real=0.5600 (var=0.0052) vs this=0.5470 (var=0.0060)
- Experiment 10: real=0.5988 (var=0.0026) vs this=0.5844 (var=0.0106)
- Experiment 11: real=0.0206 (var=0.0003) vs this=0.0375 (var=0.0004)
- Experiment 12: real=0.4525 (var=0.0028) vs this=0.4096 (var=0.0045)
- Experiment 13: real=0.2900 (var=0.0036) vs this=0.2996 (var=0.0045)
- Experiment 14: real=0.5716 (var=0.0029) vs this=0.5497 (var=0.0027)
- Experiment 15: real=0.4830 (var=0.0041) vs this=0.4778 (var=0.0036)
- Experiment 16: real=0.5474 (var=0.0024) vs this=0.5687 (var=0.0055)
- Experiment 17: real=0.7212 (var=0.0108) vs this=0.4618 (var=0.0063)
- Experiment 18: real=0.4635 (var=0.0041) vs this=0.3261 (var=0.0028)


---

### `pi_8` (overall score: 0.756)

**Description**
Bounded Outlier-Distrust Tallying (BODT). Subjects integrate all binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, with subjective weights computed RELATIVE to the panel rather than as a monotone function of stated validity: (i) a trust plateau over [v_lo, v_hi] (ceiling just above 0.98) gives all mid-band and 98% experts approximately equal weight; (ii) a small negative chance flank at v near 0.5, consensus-gated so chance cues are counted rather than inverted when an above-ceiling outlier destabilizes the scale; (iii) distrust of an above-ceiling outlier is a BOUNDED, SATURATING shrinkage of its weight toward a floor, peaked in the outlier's uncorroborated edge over the next rung (maximal near gap ~0.05: spurious precision; small but nonzero on dense ladders; receding for large edges on coherent panels); (iv) strongly negative distrust reserved for the composition-amplified regime — an outlier floating above a majority-chance panel. Choice is a low-gain softmax plus a uniform lapse. This round is a pure calibration claim: the only two levers whose slopes have been verified against reliably-signed residuals across the loop (the composition-route amplitude c_amp on Exp6, and the chance-flank magnitude kappa_lo on Exp5) are moved to their anti-oscillation midpoints; every structural parameter is held at its accepted value.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Bounded Outlier-Distrust Tallying (BODT), engaged-shoulder edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU: all v in [v_lo, v_hi] (v_hi just above
    #       0.98, so a 98% expert stays INSIDE the plateau) get
    #       approximately equal positive weight w_p.
    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small
    #       negative weight (-kappa_lo at v=0.5, rising linearly to
    #       the plateau at v_lo). When an above-ceiling claim is
    #       present, the flank is gated: gate = lambda_ext +
    #       (1-lambda_ext)*sig_gap, so chance cues are COUNTED
    #       (weight -> w_p) rather than inverted.
    # (iii) BOUNDED OUTLIER DISTRUST (the core BODT claim): the
    #       discount of the panel-max expert is a SATURATING
    #       SHRINKAGE toward a floor, not an unbounded sign flip:
    #         w_top_mod = w_floor + (w_p - w_floor) * exp(-rho_m * P)
    #       where P is a PEAKED bump in the rung gap (maximal at
    #       gap ~ g_star ~ 0.05 -- spurious precision; a SMALL but
    #       nonzero discount on dense ladders gap ~ 0.01 via the
    #       widened left width s_l; receding for large edges on
    #       coherent panels via the right width s_r, WIDENED so
    #       the shoulder covers the moderate-gap regime gap ~ 0.09).
    #       Because of the exponential saturation, w_top_mod never
    #       leaves [w_floor, w_p].
    #  (iv) COMPOSITION-AMPLIFIED ROUTE: an additive discount
    #         comp = c_amp * sig_gap * far(gap) * gate(c_frac)
    #       where far(gap) is a logistic gate whose center g_far is
    #       LOW enough (~0.10-0.12) to partially engage on
    #       chance-laden moderate-gap panels (gap ~ 0.09 with
    #       c_frac > c_half), while remaining essentially closed at
    #       gap ~ 0.05 and fully open only for far outliers
    #       (gap >= ~0.15) on majority-chance panels.
    # A small bounded extremity term d_w*(v_top - v_hi)/(1 - v_hi)
    # is retained. Choice is a low-gain softmax (inverse temperature
    # beta) over the two mirrored scores, plus a uniform lapse
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
    lambda_ext = float(parameters["lambda_ext"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    w_floor = float(parameters["w_floor"])
    rho_m = float(parameters["rho_m"])
    d_w = float(parameters["d_w"])
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    c_amp = float(parameters["c_amp"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    def _sigmoid(x):
        # Numerically stable logistic via the identity
        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).
        return 0.5 * (1.0 + np.tanh(0.5 * x))

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

    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981
    # a 0.98-topped panel stays inside the trusted plateau; only
    # 99%+ claims enter the distrust branch.
    outlier = v_top > v_hi

    if outlier:
        sig_gap = _sigmoid((gap - g0) / s_gap)
    else:
        sig_gap = 0.0

    # ---- (ii): consensus gate for the chance flank. ----
    if outlier:
        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)
    else:
        gate = 0.0

    # ---- (i) + (ii): plateau with a consensus-gated chance flank. ----
    w = np.full(n_features, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        w_flank = -kappa_lo + (w_p + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # With an above-ceiling claim in the panel, chance cues are
        # counted (weight -> w_p) rather than inverted.
        w[flank] = (1.0 - gate) * w_flank + gate * w_p

    # ---- (iii) + (iv): bounded outlier distrust, with a
    # composition-amplified route that now partially engages on
    # chance-laden MODERATE-gap panels. ----
    if outlier:
        # (a) Peaked bump in the uncorroborated edge: maximal near
        # gap ~ g_star (spurious precision), a SMALL nonzero
        # discount on dense ladders (widened s_l), receding for
        # large edges on coherent panels (widened right shoulder
        # s_r covers the moderate-gap regime gap ~ 0.09).
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))

        # (b) BOUNDED, SATURATING shrinkage toward the floor:
        # w_top_mod in [w_floor, w_p] for every P; even at peak
        # distrust the outlier is only demoted toward the floor
        # (~ -0.25..-0.30 plateau units), never unboundedly flipped.
        w_top_mod = w_floor + (w_p - w_floor) * float(np.exp(-rho_m * P))

        # (c) Composition-amplified route: a logistic far gate
        # centered at g_far ~ 0.10-0.12, so it PARTIALLY engages on
        # chance-laden moderate-gap panels (gap ~ 0.09, c_frac >
        # c_half) and is fully open only for far outliers
        # (gap >= ~0.15) on majority-chance panels.
        c_frac = float(np.mean(flank))
        comp_gate = _sigmoid((c_frac - c_half) / s_c)
        far = _sigmoid((gap - g_far) / s_far)
        comp = c_amp * float(sig_gap) * float(far) * float(comp_gate)

        # (d) Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_top_mod - ext - comp
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
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.54]
- v_hi: [0.9805, 0.9825]
- kappa_lo: [0.26, 0.28]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- lambda_ext: [0.35, 0.45]
- g_star: [0.046, 0.050]
- s_l: [0.018, 0.024]
- s_r: [0.045, 0.055]
- w_floor: [-0.30, -0.25]
- rho_m: [0.9, 1.2]
- d_w: [0.15, 0.22]
- g_far: [0.10, 0.12]
- s_far: [0.025, 0.035]
- c_half: [0.32, 0.35]
- s_c: [0.05, 0.07]
- c_amp: [2.5, 2.8]
- beta: [0.34, 0.39]
- epsilon: [0.05, 0.07]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.2180 (var=0.0140) vs this=-0.2522 (var=0.0047)
- Experiment 2: real=0.0771 (var=0.0031) vs this=0.1105 (var=0.0025)
- Experiment 3: real=0.6170 (var=0.0034) vs this=0.6393 (var=0.0026)
- Experiment 4: real=0.5133 (var=0.0042) vs this=0.5350 (var=0.0045)
- Experiment 5: real=-0.1683 (var=0.0124) vs this=-0.1750 (var=0.0140)
- Experiment 6: real=0.1180 (var=0.0074) vs this=0.1115 (var=0.0018)
- Experiment 7: real=0.5796 (var=0.0066) vs this=0.5133 (var=0.0036)
- Experiment 8: real=0.4625 (var=0.0103) vs this=0.4550 (var=0.0133)
- Experiment 9: real=0.5600 (var=0.0052) vs this=0.5455 (var=0.0066)
- Experiment 10: real=0.5988 (var=0.0026) vs this=0.5800 (var=0.0079)
- Experiment 11: real=0.0206 (var=0.0003) vs this=0.0576 (var=0.0002)
- Experiment 12: real=0.4525 (var=0.0028) vs this=0.4387 (var=0.0066)
- Experiment 13: real=0.2900 (var=0.0036) vs this=0.3475 (var=0.0040)
- Experiment 14: real=0.5716 (var=0.0029) vs this=0.4855 (var=0.0025)
- Experiment 15: real=0.4830 (var=0.0041) vs this=0.5137 (var=0.0034)
- Experiment 16: real=0.5474 (var=0.0024) vs this=0.4048 (var=0.0041)
- Experiment 17: real=0.7212 (var=0.0108) vs this=0.3964 (var=0.0050)
- Experiment 18: real=0.4635 (var=0.0041) vs this=0.3811 (var=0.0037)


---

### `pi_3` (overall score: 0.613)

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
- Experiment 11: real=0.0206 (var=0.0003) vs this=0.0148 (var=0.0004)
- Experiment 12: real=0.4525 (var=0.0028) vs this=0.5867 (var=0.0071)
- Experiment 13: real=0.2900 (var=0.0036) vs this=0.6114 (var=0.0059)
- Experiment 14: real=0.5716 (var=0.0029) vs this=0.6150 (var=0.0059)
- Experiment 15: real=0.4830 (var=0.0041) vs this=0.3044 (var=0.0057)
- Experiment 16: real=0.5474 (var=0.0024) vs this=0.6538 (var=0.0058)
- Experiment 17: real=0.7212 (var=0.0108) vs this=0.6915 (var=0.0149)
- Experiment 18: real=0.4635 (var=0.0041) vs this=0.6723 (var=0.0044)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0626 -> ACCEPTED
- iter 2: loss=0.0715 -> REJECTED
- iter 3: loss=0.0622 -> ACCEPTED
- iter 4: loss=0.0771 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.0622 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.2557 (var=0.0082) (Δ vs real -0.0377)
  - iter 2: -0.2497 (var=0.0079) (Δ vs real -0.0318)
  - iter 3: -0.2437 (var=0.0060) (Δ vs real -0.0257)
  - iter 4 (most recent): -0.2208 (var=0.0086) (Δ vs real -0.0028)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6130 (var=0.0149)
- pi_2: -0.4425 (var=0.0124)
- pi_3: -0.2182 (var=0.0156)
- pi_4: -0.2188 (var=0.0064)
- pi_5: -0.4125 (var=0.0044)
- pi_6: -0.1973 (var=0.0076)
- pi_7: -0.2530 (var=0.0065)
- pi_8: -0.2522 (var=0.0047)
- pi_8_1: -0.2380 (var=0.0059)
- pi_9: -0.2460 (var=0.0055)

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
  - iter 1: 0.1074 (var=0.0019) (Δ vs real +0.0303)
  - iter 2: 0.1017 (var=0.0024) (Δ vs real +0.0247)
  - iter 3: 0.1074 (var=0.0022) (Δ vs real +0.0303)
  - iter 4 (most recent): 0.1126 (var=0.0025) (Δ vs real +0.0355)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1196 (var=0.0023)
- pi_1: -0.3725 (var=0.0076)
- pi_3: 0.0861 (var=0.0026)
- pi_4: 0.0881 (var=0.0019)
- pi_5: 0.0789 (var=0.0021)
- pi_6: 0.0956 (var=0.0023)
- pi_7: 0.1225 (var=0.0025)
- pi_8: 0.1105 (var=0.0025)
- pi_8_1: 0.1174 (var=0.0015)
- pi_9: 0.1077 (var=0.0028)

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
  - iter 1: 0.6081 (var=0.0059) (Δ vs real -0.0089)
  - iter 2: 0.6207 (var=0.0051) (Δ vs real +0.0037)
  - iter 3: 0.6359 (var=0.0027) (Δ vs real +0.0189)
  - iter 4 (most recent): 0.6022 (var=0.0038) (Δ vs real -0.0148)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6456 (var=0.0096)
- pi_2: 0.8389 (var=0.0095)
- pi_1: 0.6944 (var=0.0071)
- pi_4: 0.6485 (var=0.0027)
- pi_5: 0.4996 (var=0.0026)
- pi_6: 0.4811 (var=0.0035)
- pi_7: 0.6267 (var=0.0043)
- pi_8: 0.6393 (var=0.0026)
- pi_8_1: 0.6463 (var=0.0039)
- pi_9: 0.5933 (var=0.0037)

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
  - iter 1: 0.5371 (var=0.0043) (Δ vs real +0.0238)
  - iter 2: 0.4783 (var=0.0026) (Δ vs real -0.0350)
  - iter 3: 0.5292 (var=0.0042) (Δ vs real +0.0158)
  - iter 4 (most recent): 0.4838 (var=0.0058) (Δ vs real -0.0296)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8387 (var=0.0140)
- pi_3: 0.6062 (var=0.0085)
- pi_1: 0.6825 (var=0.0042)
- pi_4: 0.5192 (var=0.0023)
- pi_5: 0.3950 (var=0.0023)
- pi_6: 0.3850 (var=0.0041)
- pi_7: 0.5425 (var=0.0051)
- pi_8: 0.5350 (var=0.0045)
- pi_8_1: 0.5279 (var=0.0036)
- pi_9: 0.5262 (var=0.0035)

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
  - iter 1: -0.1375 (var=0.0254) (Δ vs real +0.0308)
  - iter 2: -0.1292 (var=0.0174) (Δ vs real +0.0392)
  - iter 3: -0.1758 (var=0.0220) (Δ vs real -0.0075)
  - iter 4 (most recent): -0.1608 (var=0.0209) (Δ vs real +0.0075)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2958 (var=0.0606)
- pi_4: 0.0308 (var=0.0233)
- pi_1: 0.1625 (var=0.0145)
- pi_2: 0.7617 (var=0.0356)
- pi_5: -0.1167 (var=0.0195)
- pi_6: -0.1350 (var=0.0257)
- pi_7: -0.1425 (var=0.0147)
- pi_8: -0.1750 (var=0.0140)
- pi_8_1: -0.1600 (var=0.0245)
- pi_9: -0.1667 (var=0.0190)

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
  - iter 1: 0.1250 (var=0.0016) (Δ vs real +0.0070)
  - iter 2: 0.1160 (var=0.0025) (Δ vs real -0.0020)
  - iter 3: 0.1260 (var=0.0017) (Δ vs real +0.0080)
  - iter 4 (most recent): 0.1170 (var=0.0015) (Δ vs real -0.0010)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7670 (var=0.0039)
- pi_3: 0.4190 (var=0.0107)
- pi_1: 0.8475 (var=0.0076)
- pi_2: 0.1180 (var=0.0091)
- pi_5: 0.2870 (var=0.0027)
- pi_6: 0.4360 (var=0.0065)
- pi_7: 0.1235 (var=0.0028)
- pi_8: 0.1115 (var=0.0018)
- pi_8_1: 0.1280 (var=0.0020)
- pi_9: 0.1180 (var=0.0024)

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
  - iter 1: 0.3800 (var=0.0044) (Δ vs real -0.1996)
  - iter 2: 0.3733 (var=0.0047) (Δ vs real -0.2062)
  - iter 3: 0.3693 (var=0.0033) (Δ vs real -0.2102)
  - iter 4 (most recent): 0.4480 (var=0.0041) (Δ vs real -0.1316)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7902 (var=0.0033)
- pi_4: 0.3191 (var=0.0050)
- pi_1: 0.1662 (var=0.0114)
- pi_2: 0.2569 (var=0.0106)
- pi_3: 0.3831 (var=0.0085)
- pi_6: 0.6756 (var=0.0042)
- pi_7: 0.5787 (var=0.0053)
- pi_8: 0.5133 (var=0.0036)
- pi_8_1: 0.4542 (var=0.0043)
- pi_9: 0.4013 (var=0.0051)

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
  - iter 1: 0.4575 (var=0.0216) (Δ vs real -0.0050)
  - iter 2: 0.4575 (var=0.0165) (Δ vs real -0.0050)
  - iter 3: 0.4163 (var=0.0167) (Δ vs real -0.0463)
  - iter 4 (most recent): 0.4888 (var=0.0196) (Δ vs real +0.0262)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5400 (var=0.0192)
- pi_5: 0.0550 (var=0.0029)
- pi_1: 0.8688 (var=0.0163)
- pi_2: 0.6587 (var=0.0158)
- pi_3: 0.6062 (var=0.0141)
- pi_6: 0.4213 (var=0.0156)
- pi_7: 0.4813 (var=0.0186)
- pi_8: 0.4550 (var=0.0133)
- pi_8_1: 0.4813 (var=0.0124)
- pi_9: 0.4462 (var=0.0158)

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
  - iter 1: 0.5220 (var=0.0063) (Δ vs real -0.0380)
  - iter 2: 0.7055 (var=0.0053) (Δ vs real +0.1455)
  - iter 3: 0.5290 (var=0.0066) (Δ vs real -0.0310)
  - iter 4 (most recent): 0.6400 (var=0.0067) (Δ vs real +0.0800)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8115 (var=0.0019)
- pi_4: 0.3985 (var=0.0055)
- pi_1: 0.1815 (var=0.0131)
- pi_2: 0.5685 (var=0.0047)
- pi_3: 0.5220 (var=0.0036)
- pi_5: 0.7830 (var=0.0038)
- pi_7: 0.5715 (var=0.0054)
- pi_8: 0.5455 (var=0.0066)
- pi_8_1: 0.5560 (var=0.0071)
- pi_9: 0.5470 (var=0.0060)

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
  - iter 1: 0.5975 (var=0.0083) (Δ vs real -0.0012)
  - iter 2: 0.5881 (var=0.0077) (Δ vs real -0.0106)
  - iter 3: 0.6031 (var=0.0083) (Δ vs real +0.0044)
  - iter 4 (most recent): 0.5238 (var=0.0084) (Δ vs real -0.0750)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6719 (var=0.0085)
- pi_6: 0.2988 (var=0.0058)
- pi_1: 0.8525 (var=0.0127)
- pi_2: 0.6687 (var=0.0078)
- pi_3: 0.6206 (var=0.0104)
- pi_5: 0.1444 (var=0.0034)
- pi_7: 0.6094 (var=0.0057)
- pi_8: 0.5800 (var=0.0079)
- pi_8_1: 0.5725 (var=0.0067)
- pi_9: 0.5844 (var=0.0106)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Theory-discriminant index for the 10-expert panel
    validities = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50].

    Both theories reduce to a FIXED weight vector over the 10 experts on
    this panel. At each theory's parameter-box midpoint we compute its
    predicted p(choose A) for the observed trial, take the difference
    d_t = p_CREDT(t) - p_FDEI(t), and average d_t * y_t over all trials,
    where y_t = +1 if the subject chose A and -1 if B.

    E[metric] = (1/T) * sum_t d_t * (2*p_true(t) - 1).
    If CREDT generated the data this is strongly POSITIVE (~ +0.045);
    if FDEI generated the data it is NEGATIVE (~ -0.022). The contrast
    is driven by the cells where the boxes are disjoint: the Z-family
    (chance-cue ladders: CREDT 0.59-0.74 vs FDEI exactly 0.50), the
    0.99-expert trials (CREDT ~0.28-0.36 vs FDEI ~0.55-0.60), the
    X-family sign flips, and K1 (four trusted experts vs four
    coin-flippers: CREDT ~0.50 vs FDEI ~0.78).
    """
    validities = np.array([0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50])

    # ---------- CREDT weights at parameter-box midpoints ----------
    w_p, v_lo, v_hi, kappa_lo = 0.90, 0.53, 0.9815, 0.20
    g0, s_gap = 0.043, 0.013
    rho, d_w, lambda_ext = 1.36, 0.40, 0.40
    g_star, s_l, s_r = 0.048, 0.012, 0.033
    c_half, s_c, c_amp = 0.36, 0.06, 1.29
    beta_c, eps_c = 0.42, 0.05

    v = np.clip(validities, 0.5, 1.0)
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second
    outlier = v_top > v_hi
    if outlier:
        sig_gap = 0.5 * (1.0 + np.tanh(0.5 * ((gap - g0) / s_gap)))
    else:
        sig_gap = 0.0
    gate = (lambda_ext + (1.0 - lambda_ext) * sig_gap) if outlier else 0.0

    w_c = np.full(10, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        w_fl = -kappa_lo + (w_p + kappa_lo) * np.clip(v[flank] - 0.5, 0.0, None) / (v_lo - 0.5)
        w_c[flank] = (1.0 - gate) * w_fl + gate * w_p
    if outlier:
        c_frac = float(flank.mean())
        C = c_amp * 0.5 * (1.0 + np.tanh(0.5 * (c_frac - c_half) / s_c))
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        w_top = w_p - d_w * (v_top - v_hi) / (1.0 - v_hi) - rho * (P + C * sig_gap)
        w_c[v >= v_top - 1e-9] = w_top

    # ---------- FDEI weights at parameter-box midpoints ----------
    delta, tau = 0.15, 0.02
    beta_f, eps_f = 0.32, 0.025
    vv = np.clip(validities, 0.5, 1.0 - 1e-9)
    llr = np.log(vv / (1.0 - vv))
    w_f = np.power(np.maximum(llr - tau, 0.0), delta)

    def _p_choose_a(s, beta, eps):
        # low-gain softmax over [s, -s] plus uniform lapse
        z = beta * s
        p_core = 1.0 / (1.0 + np.exp(-z))
        return (1.0 - eps) * p_core + eps * 0.5

    total = 0.0
    n = 0
    for a_cell, b_cell, resp in zip(data['option_a_ratings'],
                                    data['option_b_ratings'],
                                    data['response']):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        votes = np.sign(a - b)
        s_c = float(np.dot(w_c, votes))
        s_f = float(np.dot(w_f, votes))
        p_c = _p_choose_a(s_c, beta_c, eps_c)
        p_f = _p_choose_a(s_f, beta_f, eps_f)
        d_t = p_c - p_f
        y = 1.0 if int(resp) == 0 else -1.0
        total += d_t * y
        n += 1

    if n == 0:
        return 0.0
    return float(total / n)
```

**Observed (real) value:** 0.0206 (var=0.0003)
**Previous candidate values (this loop):**
  - iter 1: 0.0289 (var=0.0003) (Δ vs real +0.0084)
  - iter 2: 0.0307 (var=0.0003) (Δ vs real +0.0101)
  - iter 3: 0.0284 (var=0.0003) (Δ vs real +0.0078)
  - iter 4 (most recent): 0.0289 (var=0.0003) (Δ vs real +0.0083)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0758 (var=0.0002)
- pi_4: -0.0398 (var=0.0003)
- pi_1: -0.0446 (var=0.0003)
- pi_2: 0.0431 (var=0.0003)
- pi_3: 0.0148 (var=0.0004)
- pi_5: -0.0288 (var=0.0001)
- pi_6: -0.0227 (var=0.0002)
- pi_8: 0.0576 (var=0.0002)
- pi_8_1: 0.0464 (var=0.0004)
- pi_9: 0.0375 (var=0.0004)

### Experiment 12
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Super-expert deference rate.

    Among all trials on which the highest-validity expert (feature 0, the
    99% expert) discriminates between option A and option B, return the
    fraction of choices that FOLLOW that expert's recommendation (i.e.
    chose the option the 99% expert rated 1).

    FDEI (flattened-diagnosticity integration) gives the 0.99 expert its
    largest positive weight, so it predicts deference well above chance
    (~0.57 overall, ~0.69 on isolated super-expert trials, but BELOW 0.5
    when a 0.94+0.65+0.50 coalition opposes it).

    CREDT, on this panel (0.99 sitting 0.05 above the 0.94 rung with a
    partially chance-laden scale), assigns the panel-max expert a large
    NEGATIVE weight, so it predicts systematic ANTI-deference (~0.21).
    """
    df = data.reset_index(drop=True)

    def _first_rating(cell):
        # Cells may be list / tuple / np.ndarray of 0/1 ints, or a string
        # serialization like "[1, 0, 1, ...]". Robustly extract element 0.
        if isinstance(cell, str):
            s = cell.strip().strip("[]")
            parts = s.replace(",", " ").split()
            return int(float(parts[0]))
        seq = list(cell)
        return int(seq[0])

    a_col = df["option_a_ratings"]
    b_col = df["option_b_ratings"]
    r_col = df["response"]

    n = 0
    follow = 0
    for i in range(len(df)):
        try:
            a0 = _first_rating(a_col.iloc[i])
            b0 = _first_rating(b_col.iloc[i])
        except Exception:
            continue
        # Skip trials where the 99% expert does not discriminate.
        if a0 not in (0, 1) or b0 not in (0, 1) or a0 == b0:
            continue
        # Option endorsed by the 99% expert: 0 = A, 1 = B.
        winner = 0 if a0 == 1 else 1
        resp = int(r_col.iloc[i])
        n += 1
        if resp == winner:
            follow += 1

    if n == 0:
        return 0.5
    return float(follow) / float(n)
```

**Observed (real) value:** 0.4525 (var=0.0028)
**Previous candidate values (this loop):**
  - iter 1: 0.4696 (var=0.0054) (Δ vs real +0.0171)
  - iter 2: 0.4567 (var=0.0040) (Δ vs real +0.0042)
  - iter 3: 0.4804 (var=0.0044) (Δ vs real +0.0279)
  - iter 4 (most recent): 0.3946 (var=0.0057) (Δ vs real -0.0579)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6017 (var=0.0039)
- pi_7: 0.2367 (var=0.0031)
- pi_1: 0.8358 (var=0.0153)
- pi_2: 0.6479 (var=0.0042)
- pi_3: 0.5867 (var=0.0071)
- pi_5: 0.1121 (var=0.0017)
- pi_6: 0.2667 (var=0.0041)
- pi_8: 0.4387 (var=0.0066)
- pi_8_1: 0.4658 (var=0.0052)
- pi_9: 0.4096 (var=0.0045)

### Experiment 13
**Design**
  A=[1, 1, 1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Double-99 endorsement avoidance" metric.
    #
    # On every trial, identify whether BOTH top-validity experts (the two
    # tied 0.99 experts, features 0 and 1 of the fixed panel
    # [0.99, 0.99, 0.94, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]) cast the SAME
    # vote (both favor A, or both favor B). On those trials, record
    # whether the subject chose the option the 0.99 pair endorsed.
    # The metric is the proportion of such choices.
    #
    # This isolates the single structural disagreement between CREDT and
    # BODT on this panel: the DEPTH of distrust of a 0.99 claim sitting
    # gap=0.05 above its next rung on a chance-laden panel.
    #   CREDT: w_top ~ -1.7 per 99% expert  -> the endorsement is worth
    #          roughly two trusted cues AGAINST -> p(pick endorsed) ~ 0.16
    #          (box ~[0.12, 0.19]).
    #   BODT:  w_top ~ -0.14 per 99% expert -> the endorsement is worth
    #          roughly nothing -> p(pick endorsed) ~ 0.40 (box ~[0.35, 0.47]).
    # The two predicted boxes are disjoint by ~0.2 in probability units.
    # Mirror trials (e.g., trials 1/2, 3/4, 6/7, 8/9, 16/17) put the 99-pair
    # on opposite sides, so pure A/B side bias cancels.
    # Works identically on the pooled DataFrame and on a single subject's
    # slice (each subject has 14 qualifying trial types x 4 reps = 56 rows).
    validities = [0.99, 0.99, 0.94, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]
    val = np.asarray(validities, dtype=float)
    v_top = float(val.max())
    top_idx = np.where(val >= v_top - 1e-9)[0]

    picks = []
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a_cell, b_cell, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_cell, dtype=float).ravel()
        b = np.asarray(b_cell, dtype=float).ravel()
        if a.shape[0] != val.shape[0] or b.shape[0] != val.shape[0]:
            continue
        diffs = a[top_idx] - b[top_idx]
        # Both top experts must vote, and vote together.
        if np.all(diffs > 0):
            endorsed = 0  # both 0.99 experts favor A
        elif np.all(diffs < 0):
            endorsed = 1  # both 0.99 experts favor B
        else:
            # split votes or ties among the top experts -> not a
            # double-endorsement trial
            continue
        resp = int(r)
        picks.append(1.0 if resp == endorsed else 0.0)

    if len(picks) == 0:
        return 0.5
    return float(np.mean(picks))
```

**Observed (real) value:** 0.2900 (var=0.0036)
**Previous candidate values (this loop):**
  - iter 1: 0.3357 (var=0.0052) (Δ vs real +0.0457)
  - iter 2: 0.2146 (var=0.0040) (Δ vs real -0.0754)
  - iter 3: 0.3196 (var=0.0054) (Δ vs real +0.0296)
  - iter 4 (most recent): 0.1929 (var=0.0028) (Δ vs real -0.0971)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0639 (var=0.0012)
- pi_8: 0.3475 (var=0.0040)
- pi_1: 0.8289 (var=0.0104)
- pi_2: 0.5486 (var=0.0043)
- pi_3: 0.6114 (var=0.0059)
- pi_4: 0.7611 (var=0.0029)
- pi_5: 0.1286 (var=0.0030)
- pi_6: 0.3032 (var=0.0042)
- pi_8_1: 0.2904 (var=0.0028)
- pi_9: 0.2996 (var=0.0045)

### Experiment 14
**Design**
  A=[1, 1, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def _to_vec(x):
        if isinstance(x, str):
            import ast
            x = ast.literal_eval(x)
        return np.asarray(x, dtype=float).ravel()

    follows = []
    for a_raw, b_raw, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        try:
            a = _to_vec(a_raw)
            b = _to_vec(b_raw)
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        # The 0.99-validity expert is feature index 0 (displayed first).
        d0 = float(a[0]) - float(b[0])
        if d0 == 0.0:
            # Top expert tied between options: endorsement undefined.
            continue
        endorsed_a = d0 > 0.0
        try:
            chose_a = (int(resp) == 0)
        except Exception:
            continue
        follows.append(1.0 if (endorsed_a == chose_a) else 0.0)

    if len(follows) == 0:
        return 0.5
    return float(np.mean(follows))
```

**Observed (real) value:** 0.5716 (var=0.0029)
**Previous candidate values (this loop):**
  - iter 1: 0.5592 (var=0.0030) (Δ vs real -0.0124)
  - iter 2: 0.5826 (var=0.0029) (Δ vs real +0.0111)
  - iter 3: 0.5582 (var=0.0038) (Δ vs real -0.0134)
  - iter 4 (most recent): 0.4897 (var=0.0034) (Δ vs real -0.0818)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4855 (var=0.0025)
- pi_7: 0.3326 (var=0.0039)
- pi_1: 0.8437 (var=0.0114)
- pi_2: 0.7321 (var=0.0085)
- pi_3: 0.6150 (var=0.0059)
- pi_4: 0.6305 (var=0.0031)
- pi_5: 0.1179 (var=0.0015)
- pi_6: 0.2871 (var=0.0027)
- pi_8_1: 0.5024 (var=0.0027)
- pi_9: 0.5497 (var=0.0027)

### Experiment 15
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anti-follow rate of the 99%-expert endorsements.

    On every trial where the two top-validity (0.99) experts cast a
    net nonzero vote (i.e., they jointly endorse one option), compute
    the probability that the subject chooses the NON-endorsed option.
    Under CREDT (negative subjective weight for the uncorroborated
    0.99 outlier on this chance-laden panel) this rate is well above
    0.5 (predicted ~0.56-0.65); under BODT (bounded distrust, w_top
    near zero) it is below 0.5 (predicted ~0.38-0.48). The two
    predicted boxes are disjoint.
    """

    def _to_vec(cell):
        if isinstance(cell, np.ndarray):
            return cell
        if isinstance(cell, (list, tuple)):
            return np.asarray(cell, dtype=float)
        # Fallback: parse string-encoded sequences like "[1, 0, ...]"
        s = str(cell).strip().replace("[", "").replace("]", "")
        return np.asarray([float(v) for v in s.split(",") if v.strip() != ""], dtype=float)

    a = np.vstack([_to_vec(c) for c in data["option_a_ratings"]])
    b = np.vstack([_to_vec(c) for c in data["option_b_ratings"]])
    resp = np.asarray(data["response"], dtype=float)

    # Net vote of the two 0.99-validity experts (features 0 and 1,
    # fixed by this experiment's validity vector).
    net = np.sign(a[:, 0] - b[:, 0]) + np.sign(a[:, 1] - b[:, 1])

    mask = net != 0
    if not np.any(mask):
        return 0.5

    net_m = net[mask]
    r = np.rint(resp[mask]).astype(int)
    endorsed_a = net_m > 0  # True -> 0.99 experts endorse A

    # 'against' = subject picked the option the 0.99 experts did NOT endorse
    against = np.where(endorsed_a, r == 1, r == 0)

    return float(np.mean(against))
```

**Observed (real) value:** 0.4830 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1: 0.4230 (var=0.0043) (Δ vs real -0.0600)
  - iter 2: 0.4085 (var=0.0029) (Δ vs real -0.0744)
  - iter 3: 0.4156 (var=0.0046) (Δ vs real -0.0674)
  - iter 4 (most recent): 0.4170 (var=0.0043) (Δ vs real -0.0659)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6411 (var=0.0036)
- pi_8_1: 0.4474 (var=0.0052)
- pi_1: 0.1615 (var=0.0132)
- pi_2: 0.2174 (var=0.0047)
- pi_3: 0.3044 (var=0.0057)
- pi_4: 0.2544 (var=0.0028)
- pi_5: 0.7807 (var=0.0039)
- pi_6: 0.6574 (var=0.0055)
- pi_8: 0.5137 (var=0.0034)
- pi_9: 0.4778 (var=0.0036)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Global outlier-endorsement follow index, side-balanced.

    Uses every trial on which the 0.99-validity outlier expert (feature
    index 0 in this design) actually discriminates the two options (his
    rating is 1 for exactly one of A/B). On those trials we ask: did the
    subject choose the side the outlier endorsed? The index averages the
    follow rate over outlier-endorses-A trials and outlier-endorses-B
    trials separately, so pure display/side bias cancels.

    These are trials 1-8 and 20-24 of the design (R0-R5, R3m, R5m, V1,
    V2, V1m, K1, K2), 13 trial types x 3 reps = 39 trials per subject.

    BODT (deep, composition-amplified outlier distrust: w(0.99) ~ -1.8
    to -3.5 in chance-cue units) predicts subjects systematically REJECT
    the outlier-endorsed side: index ~ 0.32 (centers .20/.24/.29/.34/.39/
    .45 on R0-R5, .34/.46 on the mirrors, ~.21 on the V cells, .42/.36 on
    K1/K2). CREDT (shallow distrust: w(0.99) ~ -0.7 to -1.3, coin-flippers
    near-exchangeable with experts) predicts the outlier's endorsement
    still carries the day on balance: index ~ 0.51 (R-ladder crosses 0.5
    at k~2). The two predicted values are separated by ~0.19 in mean
    against a per-subject SD of ~0.09-0.11 (binomial noise over 39 trials
    plus parameter heterogeneity), giving Welch t of order 6-8 at N=25.
    """
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    n_follow_outlier_a = 0
    n_outlier_a = 0
    n_follow_outlier_b = 0
    n_outlier_b = 0

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        try:
            a = [int(v) for v in a_cell]
            b = [int(v) for v in b_cell]
        except (TypeError, ValueError):
            continue
        if len(a) == 0 or len(a) != len(b):
            continue
        # The 0.99-validity outlier expert occupies feature index 0 in
        # this experiment's panel [0.99, 0.85, 0.80, 0.75, 0.50 x5].
        if a[0] == b[0]:
            # Outlier's endorsement does not discriminate this pair.
            continue
        r = int(resp)
        if a[0] == 1:
            # Outlier endorses A.
            n_outlier_a += 1
            if r == 0:
                n_follow_outlier_a += 1
        else:
            # Outlier endorses B.
            n_outlier_b += 1
            if r == 1:
                n_follow_outlier_b += 1

    if n_outlier_a == 0 and n_outlier_b == 0:
        return 0.5

    rate_a = (float(n_follow_outlier_a) / float(n_outlier_a)) if n_outlier_a > 0 else 0.5
    rate_b = (float(n_follow_outlier_b) / float(n_outlier_b)) if n_outlier_b > 0 else 0.5
    # Side-balanced average: cancels any constant A-vs-B display bias.
    return 0.5 * (rate_a + rate_b)
```

**Observed (real) value:** 0.5474 (var=0.0024)
**Previous candidate values (this loop):**
  - iter 1: 0.5679 (var=0.0069) (Δ vs real +0.0204)
  - iter 2: 0.5604 (var=0.0078) (Δ vs real +0.0129)
  - iter 3: 0.5605 (var=0.0047) (Δ vs real +0.0131)
  - iter 4 (most recent): 0.5374 (var=0.0074) (Δ vs real -0.0101)
**Other theories' values on this metric (for reference):**
- pi_8_1: 0.1801 (var=0.0041)
- pi_7: 0.4646 (var=0.0065)
- pi_1: 0.8708 (var=0.0110)
- pi_2: 0.7154 (var=0.0058)
- pi_3: 0.6538 (var=0.0058)
- pi_4: 0.6016 (var=0.0074)
- pi_5: 0.1555 (var=0.0017)
- pi_6: 0.2763 (var=0.0042)
- pi_8: 0.4048 (var=0.0041)
- pi_9: 0.5687 (var=0.0055)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Double-99 endorsement-following rate on the disputed panel configuration.
    #
    # The design's single structural disagreement is the weight assigned to
    # the 0.99 experts on a majority-chance panel with a 0.09 edge over the
    # 0.90 rung: EOBOT's relocated bounded kernel yields only a MILD anti-vote
    # (w(0.99) ~ -0.28, about half a chance cue against), while BODT's open
    # far-fraud composition route yields a deep anti-vote (w(0.99) ~ -1.9,
    # about three chance cues against). The cleanest behavioral readout of
    # that single lever is: on trials where ONE option wins BOTH 0.99 experts
    # (features 0 and 1) and the other wins neither, how often does the
    # subject follow the double-99 endorsement?
    #
    # These are exactly trials 1, 2 (M1), 11, 12 (V3), 19-23 (D-family),
    # 24, 25 (K1) -- 11 unique trials x 3 reps = 33 observations per subject.
    # Predicted values: EOBOT ~ 0.45 (masked cells near-tie, D-ladder
    # followed, unanimous panel near chance), BODT ~ 0.10 (strong anti-
    # following everywhere). The w(0.99) boxes are disjoint by >= 0.8 weight
    # units, so no parameter setting in either box can close the gap, while
    # every shared parameter (w_p, w_ch, beta, epsilon) enters both theories
    # identically and thus largely cancels in the contrast.

    resp_col = data["response"].to_numpy()
    a_col = data["option_a_ratings"].to_numpy()
    b_col = data["option_b_ratings"].to_numpy()

    follow = []
    for i in range(len(data)):
        a = np.asarray(a_col[i], dtype=float).ravel()
        b = np.asarray(b_col[i], dtype=float).ravel()
        if a.size < 2 or b.size < 2 or a.size != b.size:
            continue

        # Sign votes on the two 0.99-expert features (design fixes features
        # 0 and 1 as the tied panel-maximum 0.99 experts).
        v0 = np.sign(a[0] - b[0])
        v1 = np.sign(a[1] - b[1])

        # Keep only trials where a single option wins BOTH 99% experts
        # (the double-99 endorsement cells: M1, V3, D2-D5, K1 and mirrors).
        if v0 == 0 or v1 == 0 or v0 != v1:
            continue

        chose_a = (int(resp_col[i]) == 0)
        if v0 > 0:
            # A holds the double-99 endorsement.
            follow.append(1.0 if chose_a else 0.0)
        else:
            # B holds the double-99 endorsement (mirror trials).
            follow.append(0.0 if chose_a else 1.0)

    if not follow:
        return float("nan")
    return float(np.mean(follow))
```

**Observed (real) value:** 0.7212 (var=0.0108)
**Previous candidate values (this loop):**
  - iter 1: 0.6139 (var=0.0035) (Δ vs real -0.1073)
  - iter 2: 0.6236 (var=0.0082) (Δ vs real -0.0976)
  - iter 3: 0.6188 (var=0.0053) (Δ vs real -0.1024)
  - iter 4 (most recent): 0.6073 (var=0.0083) (Δ vs real -0.1139)
**Other theories' values on this metric (for reference):**
- pi_9: 0.4618 (var=0.0063)
- pi_8_1: 0.0976 (var=0.0019)
- pi_1: 0.8418 (var=0.0119)
- pi_2: 0.7291 (var=0.0086)
- pi_3: 0.6915 (var=0.0149)
- pi_4: 0.7200 (var=0.0069)
- pi_5: 0.1764 (var=0.0018)
- pi_6: 0.2176 (var=0.0030)
- pi_7: 0.2091 (var=0.0039)
- pi_8: 0.3964 (var=0.0050)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def _to_mat(col):
        vals = list(col)
        rows = [np.asarray(v, dtype=float).ravel() for v in vals]
        return np.array(rows, dtype=float)

    a = _to_mat(data['option_a_ratings'])
    b = _to_mat(data['option_b_ratings'])
    resp = np.asarray(data['response'], dtype=float).ravel()

    # The two 99%-validity experts occupy features 0 and 1 of the fixed
    # validity vector [0.99, 0.99, 0.90, 0.85, 0.80, 0.75, 0.70, 0.5, 0.5, 0.5].
    # Net vote of the double-99 endorsement on this trial:
    #   +2 / +1 : the 99(s) favor A,  -2 / -1 : favor B,  0 : tied/split.
    e = (a[:, 0] - b[:, 0]) + (a[:, 1] - b[:, 1])

    mask = e != 0
    if not np.any(mask):
        return 0.0

    e_m = e[mask]
    r_m = resp[mask]  # 0 = chose A, 1 = chose B

    # "Follow" = the chosen option is the one the 99% experts endorse.
    follow = np.where(e_m > 0, 1.0 - r_m, r_m)

    return float(np.mean(follow))
```

**Observed (real) value:** 0.4635 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1: 0.3992 (var=0.0026) (Δ vs real -0.0643)
  - iter 2: 0.4488 (var=0.0030) (Δ vs real -0.0147)
  - iter 3: 0.4229 (var=0.0033) (Δ vs real -0.0405)
  - iter 4 (most recent): 0.4179 (var=0.0029) (Δ vs real -0.0456)
**Other theories' values on this metric (for reference):**
- pi_8_1: 0.4960 (var=0.0032)
- pi_9: 0.3261 (var=0.0028)
- pi_1: 0.8240 (var=0.0131)
- pi_2: 0.7483 (var=0.0039)
- pi_3: 0.6723 (var=0.0044)
- pi_4: 0.7141 (var=0.0034)
- pi_5: 0.0501 (var=0.0007)
- pi_6: 0.1819 (var=0.0018)
- pi_7: 0.3120 (var=0.0025)
- pi_8: 0.3811 (var=0.0037)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
SG-CGD (Small-Gap Composition-Gated Outlier Distrust), revert-and-keep recalibration. Subjects integrate every binary expert rating as a sign vote into a zero-sum evidence score with subjective weights computed relative to the panel: a tilted trust plateau over [v_lo, v_hi] (ceiling just above 0.98), a consensus-gated chance flank with partial-ceiling promotion, and an outlier branch for above-ceiling claims consisting of (a) a bounded, saturating base distrust peaked at gap ~0.088 sitting on a shallow floor, (b) a near-kernel composition amplification (narrow Gaussian at gap ~0.035, chance-fraction gate) that deepens distrust only of small-edge outliers on chance-laden scales, (c) a far-fraud route at gap ~0.185, and (d) a composition-gated edge-credibility ramp that promotes a well-separated 99% expert back toward plateau trust when the surrounding scale is unreliable. Choice is a low-gain softmax plus uniform lapse. This round is a pure calibration claim: it KEEPS the iter-2 edits that the gate's decomposition showed delivered the gap-0.09 pair (shallower distrust floor w_floor ~ -0.20, gentled rho_e ~ 1.15, sharper high-threshold credibility gate), REVERTS the near-route gate sharpening that blew up Exps 9/13/15 (c_half back above 0.44, s_c re-widened, sig_gap throttle RESTORED in comp_near), and RECALIBRATES c_near upward so the gap-0.05 chance-laden distrust magnitude is preserved under the lifted floor.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # SG-CGD: Small-Gap Composition-Gated Outlier Distrust with a
    # composition-gated edge-credibility ramp (revert-and-keep edition).
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a
    # subjective trust function of the expert's STATED validity v,
    # evaluated RELATIVE to the panel:
    #   (i) TRUST PLATEAU with shallow tilt: all v in [v_lo, v_hi]
    #       (v_hi just above 0.98, so a 98% expert stays INSIDE the
    #       plateau) get approximately equal weight w_p, with a
    #       within-band tilt so higher-validity cues slightly
    #       outrank lower ones.
    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small
    #       negative weight (-kappa_lo at v=0.5, rising linearly to
    #       the tilted plateau at v_lo). When an above-ceiling claim
    #       is present the flank is gated and chance cues are
    #       PROMOTED to a partial ceiling kappa_gate*w_p.
    # (iii) BOUNDED BASE DISTRUST of the panel-max expert: a
    #       saturating shrinkage toward a SHALLOW floor,
    #         w_base = w_floor + (w_p - w_floor)*exp(-rho_e*E)
    #       with E = max(peaked gap kernel, e_min); the kernel peaks
    #       at gap ~0.088 (steep left wall s_l, wide right shoulder
    #       s_r) so a coherent-ladder outlier at gap 0.09-0.14 is
    #       demoted to roughly zero weight, never deeply flipped.
    #  (iv) NEAR-KERNEL COMPOSITION AMPLIFICATION (small-gap gate):
    #         comp_near = c_near * sig_gap * L(gap) * G(c_frac)
    #       L is a NARROW Gaussian centered g_near ~ 0.035 (inert for
    #       gap >= 0.07); G is a chance-fraction logistic. The
    #       sig_gap factor is ESSENTIAL and retained: it throttles
    #       the route on dense ladders (gap ~ 0.01), which protects
    #       the cliff-conflict cells from spurious anti-votes.
    #  (v) FAR-FRAUD ROUTE: comp_far = c_far * sig_gap * F(gap) *
    #       G_far(c_frac), F a SHARP logistic at g_far ~ 0.185 with
    #       its own chance gate: closed on every gap 0.09-0.14 panel
    #       tested here, open for far outliers over chance-laden
    #       panels (gap >= ~0.18).
    # (vi) COMPOSITION-GATED EDGE-CREDIBILITY RAMP: for gap >= ~0.075,
    #       cred = c_cred * R(gap) * Q(c_frac), capped at the plateau
    #       value. On a chance-laden panel a well-separated 99%
    #       expert is the only credible voice and is promoted back
    #       toward plateau trust; on a coherent-ladder panel the ramp
    #       stays closed and residual kernel distrust persists.
    # Choice is a low-gain softmax (inverse temperature beta) over
    # the two mirrored scores, plus a uniform lapse (epsilon).
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
    tilt = float(parameters["tilt"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_gate = float(parameters["kappa_gate"])
    lambda_ext = float(parameters["lambda_ext"])
    g0 = float(parameters["g0"])
    s_gap = float(parameters["s_gap"])
    g_star = float(parameters["g_star"])
    s_l = float(parameters["s_l"])
    s_r = float(parameters["s_r"])
    e_min = float(parameters["e_min"])
    w_floor = float(parameters["w_floor"])
    rho_e = float(parameters["rho_e"])
    d_w = float(parameters["d_w"])
    g_near = float(parameters["g_near"])
    s_near = float(parameters["s_near"])
    c_near = float(parameters["c_near"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    g_far = float(parameters["g_far"])
    s_far = float(parameters["s_far"])
    c_far = float(parameters["c_far"])
    c_half_far = float(parameters["c_half_far"])
    s_c_far = float(parameters["s_c_far"])
    c_cred = float(parameters["c_cred"])
    g_edge = float(parameters["g_edge"])
    s_edge = float(parameters["s_edge"])
    c_cred_half = float(parameters["c_cred_half"])
    s_cred = float(parameters["s_cred"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    def _sigmoid(x):
        # Numerically stable logistic via the identity
        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).
        return 0.5 * (1.0 + np.tanh(0.5 * x))

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

    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981 a
    # 0.98-topped panel stays INSIDE the trusted plateau; only 99%+
    # claims enter the distrust branch.
    outlier = v_top > v_hi

    if outlier:
        sig_gap = _sigmoid((gap - g0) / s_gap)
    else:
        sig_gap = 0.0

    # ---- (ii): consensus gate for the chance flank. ----
    if outlier:
        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)
    else:
        gate = 0.0

    # ---- (i) + (ii): tilted plateau with a consensus-gated
    # chance flank whose promotion is capped at the PARTIAL
    # ceiling kappa_gate * w_p. ----
    w = np.full(n_features, w_p, dtype=float)
    span = max(v_hi - v_lo, 1e-6)
    plateau = (v >= v_lo) & (v <= v_hi)
    if np.any(plateau):
        # Shallow within-plateau validity gradient: higher-validity
        # experts slightly outrank lower ones while the band remains
        # 'approximately equal'.
        w[plateau] = w_p * (
            1.0 - tilt * (v_hi - v[plateau]) / span
        )
    flank = v < v_lo
    if np.any(flank):
        denom = max(v_lo - 0.5, 1e-6)
        # Flank rises linearly to the TILTED plateau value at v_lo,
        # keeping the weight profile continuous at the band edge.
        w_lo_eff = w_p * (1.0 - tilt)
        w_flank = -kappa_lo + (w_lo_eff + kappa_lo) * np.clip(
            v[flank] - 0.5, 0.0, None
        ) / denom
        # With an above-ceiling claim in the panel, chance cues are
        # counted toward a partial ceiling kappa_gate*w_p rather
        # than inverted (and rather than fully trusted).
        w[flank] = (1.0 - gate) * w_flank + gate * kappa_gate * w_p

    # ---- (iii)-(vi): outlier distrust / credibility architecture. ----
    if outlier:
        # Chance fraction of the panel (share of experts below the
        # trust band) drives both composition gates.
        c_frac = float(np.mean(flank))

        # (iii) Bounded base distrust: saturating shrinkage toward a
        # shallow floor, driven by a peaked kernel in the
        # uncorroborated edge (peak at g_star, steep left wall,
        # wide right shoulder), sitting on a uniform-skepticism floor.
        width = s_l if gap < g_star else s_r
        E_kernel = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        E = max(E_kernel, e_min)
        w_base = w_floor + (w_p - w_floor) * float(np.exp(-rho_e * E))

        # (iv) Near-kernel composition amplification: fires only in
        # the small-gap regime (L centered g_near, narrow width),
        # only on chance-laden scales (G threshold c_half), and is
        # THROTTLED by sig_gap so dense ladders (gap ~ 0.01, where
        # sig_gap ~ 0.08) are protected from spurious anti-votes.
        L = float(np.exp(-0.5 * ((gap - g_near) / s_near) ** 2))
        G_near = _sigmoid((c_frac - c_half) / s_c)
        comp_near = c_near * float(sig_gap) * L * float(G_near)

        # (v) Far-fraud route: sharp logistic at g_far with its own
        # chance gate; closed on every gap 0.09-0.14 panel tested
        # here, open for far outliers over chance-laden panels.
        F_far = _sigmoid((gap - g_far) / s_far)
        G_far = _sigmoid((c_frac - c_half_far) / s_c_far)
        comp_far = c_far * float(sig_gap) * float(F_far) * float(G_far)

        # (vi) Composition-gated edge-credibility ramp: for
        # well-separated outliers (gap >= ~g_edge) on chance-laden
        # scales, the panel-max expert is promoted back toward
        # plateau trust (capped at w_p); on coherent ladders the
        # ramp stays closed and residual kernel distrust persists.
        R_cred = _sigmoid((gap - g_edge) / s_edge)
        Q_cred = _sigmoid((c_frac - c_cred_half) / s_cred)
        cred = c_cred * float(R_cred) * float(Q_cred)

        # Small bounded extremity term.
        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)

        w_top = w_base - ext - comp_near - comp_far + cred
        # The outlier is at most as trusted as a plateau expert.
        w_top = min(w_top, w_p)
        # All experts tied at the panel maximum share the same
        # weight (a pair of 0.99 super-experts doubles the vote).
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
- w_p: [0.88, 0.92]
- v_lo: [0.52, 0.55]
- v_hi: [0.9805, 0.9825]
- tilt: [0.09, 0.13]
- kappa_lo: [0.19, 0.22]
- kappa_gate: [0.66, 0.74]
- lambda_ext: [0.30, 0.42]
- g0: [0.041, 0.045]
- s_gap: [0.012, 0.014]
- g_star: [0.084, 0.092]
- s_l: [0.013, 0.017]
- s_r: [0.040, 0.050]
- e_min: [0.13, 0.17]
- w_floor: [-0.24, -0.17]
- rho_e: [1.05, 1.30]
- d_w: [0.08, 0.12]
- g_near: [0.032, 0.038]
- s_near: [0.014, 0.018]
- c_near: [3.5, 4.0]
- c_half: [0.44, 0.48]
- s_c: [0.085, 0.105]
- g_far: [0.180, 0.190]
- s_far: [0.018, 0.022]
- c_far: [20, 26]
- c_half_far: [0.57, 0.63]
- s_c_far: [0.06, 0.08]
- c_cred: [1.15, 1.40]
- g_edge: [0.070, 0.080]
- s_edge: [0.018, 0.022]
- c_cred_half: [0.57, 0.63]
- s_cred: [0.05, 0.06]
- beta: [0.38, 0.46]
- epsilon: [0.035, 0.055]
- validities: validities

`rationale`: This is the critic-prescribed surgical revert-and-keep on the ACCEPTED iter-1 SG-CGD base; the mechanism code is byte-identical and only eight parameter boxes move. The gate's decomposition of the rejected iter-2 candidate showed the edit split cleanly in two halves, and I keep exactly the half that paid.

(1) KEEP (the half that worked in iter 2): the shallower distrust floor (w_floor -0.28 -> ~-0.20), gentled saturation (rho_e 1.6 -> ~1.15), and the retuned credibility ramp (c_cred ~1.15-1.40, c_cred_half ~0.60, s_cred ~0.055). These delivered Exp 18 (-0.064 -> -0.015 error), Exp 16 (+0.020 -> +0.013), Exp 12 (+0.017 -> +0.004), Exp 17 (-0.107 -> -0.098) while holding Exp 14. Arithmetic check at the gap-0.09 peak: w_base = -0.20 + 1.10*exp(-1.15) ~ +0.15, so the coherent-ladder outlier (Exp 18, c_frac 0.30, ramp closed) sits at w_top ~ +0.09 (near-zero, follow ~0.46), while the majority-chance panel (Exp 17, c_frac 0.727, Q_cred ~ 0.91) is promoted to w_top ~ 0.88 (capped at w_p).

(2) REVERT (the half that backfired): the near-route gate sharpening. c_half returns to [0.44, 0.48], s_c re-widens to [0.085, 0.105], s_near stays narrow [0.014, 0.018], and — decisively — the sig_gap throttle REMAINS in comp_near (it was never removed from this base). The iter-2 post-mortem is unambiguous: dropping sig_gap let comp_near fire on dense ladders (gap ~ 0.01, where sig_gap ~ 0.08 would have throttled it to ~0.04), which is what blew Exp 9 from 0.522 to 0.706; the sharpened gate simultaneously over-opened gap-0.05 panels (Exp 13 -> 0.215). With the throttle restored and the gate reverted, comp_near on a dense ladder is ~0.04 regardless of c_near, so Exp 9 returns to ~0.52 (the small residual drift from the w_floor lift, ~-0.02, is the accepted cost of the keep-half).

(3) RECALIBRATE c_near (2.75 -> ~3.8): the kept w_floor lift raises w_base by ~+0.08 on every panel where the kernel is small — including the gap-0.05 cells, where E = e_min ~ 0.15. Without compensation, Exps 13/15 would drift toward over-following. Raising c_near restores the iter-1 gap-0.05 distrust magnitude: at Exp 13's panel (gap 0.05, c_frac 0.556) comp_near ~ 3.8*0.63*0.64*0.73 ~ 1.12 vs iter-1's 0.92, giving w_top ~ -0.44 vs iter-1's -0.33 — this also closes iter-1's residual there (real 0.290 vs 0.336). At Exp 14's coherent panel (c_frac 0.30) the gate stays closed (G ~ 0.16), so w_top is unchanged at ~ +0.43 and its near-perfect fit is preserved; Exp 12 (c_frac 0.375) gets slightly more distrust, moving 0.470 toward the real 0.4525.

(4) Exp 7 diagnostic (as requested, before editing): the panel is an 8-feature ladder topped by the 99% expert with a chance cluster, i.e. gap ~ 0.01-0.09 at c_frac ~ 0.375. On that panel L(gap 0.09) ~ 0.003 (comp_near ~ 0 under ANY gate threshold) and Q_cred(0.375) ~ 0.02 (ramp closed); at gap 0.01 the sig_gap throttle (~0.08) pins comp_near ~ 0.04. So w_top lands at +0.1 (gap 0.09) or +0.64 (gap 0.01) — both routes are structurally closed, confirming the critic's diagnosis that no gate placement reaches this cell. The remaining lever, e_min, is now bounded: with the kept w_floor ~ -0.20, w_base can never drop below -0.20, short of the ~ -0.3 needed; and raising e_min binds the pinned dense-ladder cells (Exps 9/10, both currently within 0.012 and 0.001 of real). Per the critic's own framing this cell is upside, not prerequisite — I accept the ~0.18 residual rather than sacrifice four locked cells on a lever that arithmetic shows cannot reach the target anyway.

(5) All shared-machinery boxes (w_p, v_lo, v_hi, tilt, kappa_lo, kappa_gate, lambda_ext, g0, s_gap, beta, epsilon) and the far-fraud route are pinned at the accepted iter-1 values, so the no-outlier experiments (1-5, 8) are mechanically unchanged and Exps 6/10/11 move only through the outlier-branch recalibration, all within guardrails. Expected movement vs the accepted base: Exp 17 0.614 -> ~0.63, Exp 18 0.399 -> ~0.45, Exp 13 0.336 -> ~0.30, Exp 15 recovering toward its iter-1 level or better, Exp 12 0.470 -> ~0.46, with Exps 3-6/8/10/11/14/16 held within ~0.01-0.03 — a strict-improvement edit aimed cleanly below the 0.0626 floor.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a genuine step forward and was correctly accepted by the gate: it is the only theory in the pool that simultaneously fits the gap-0.09 pair in the right direction (Exp 17: 0.614 vs real 0.721; Exp 18: 0.399 vs real 0.464 — compare pi_8_1 at 0.098/0.496 and pi_9 at 0.462/0.326, each of which sacrifices one of the two). The composition-gated edge-credibility ramp is doing real work, and the shared-machinery experiments (3, 6, 8, 10, 11) are essentially locked in (|diff| <= 0.009). The proposer's documented deviation from the arbiter's g_star ~ 0.05 kernel placement is empirically vindicated: with a 0.05-centered kernel receding by 0.09, Exp 18 (real 0.464 at c_frac 0.30) could not be fit without destroying Exp 14's coherent-panel gradient. However, the fit is not yet shippable. Decomposing the aggregate loss, a single cell dominates: Experiment 7 (anti-follow of the 99% expert: real 0.580, candidate 0.380, |diff| = 0.20) contributes roughly 60% of the total squared error by itself. The remaining misses cluster into two coherent, in-family-fixable groups: (a) under-following at gap 0.09 — Exp 17 (-0.107) and Exp 18 (-0.064) both need w_top nudged upward; (b) under-distrusting at gap 0.05 on chance-laden panels — Exp 13 (0.336 vs 0.290), Exp 15 (follow 0.577 vs real 0.517), and mildly Exp 12 (0.470 vs 0.453) and Exp 9 (0.522 vs 0.560). Crucially, groups (a) and (b) are separable by gap (0.09 vs 0.05) and by c_frac, so they can be tuned without fighting each other.
Rationale: Regenerate — but this is a tuning pass on an accepted base, not a redesign; keep the SG-CGD architecture exactly as prescribed. Concretely: (1) ATTACK EXP 7, the dominant residual. The panel is a dense validity ladder with a small top gap and several chance cues, so the near-kernel composition route is the only in-family lever that can reach it without touching the shared machinery: raise c_near (from [2.5, 3.0] toward [3.2, 3.8]) and widen s_near modestly (toward [0.017, 0.022]) so L(gap ~ 0.01-0.02) is no longer attenuated; verify Exp 14 (gap 0.05, coherent, gate closed) and Exps 3-8 hold. (2) Deepen small-gap chance-laden distrust to close Exps 13/15: lower c_half toward [0.40, 0.44] — this closes the gate further on coherent panels (protecting Exp 14) while opening it on c_frac ~ 0.55-0.60 panels; the c_near raise in (1) reinforces this. (3) Raise gap-0.09 following for Exps 17/18: increase c_cred (toward [1.1, 1.45]) and/or lower c_cred_half toward [0.45, 0.50] so the ramp fires more decisively on Exp 17's c_frac 0.727 panel, and lift w_floor from ~-0.28 toward ~-0.18 (or reduce rho_e slightly) so the residual kernel distrust at gap 0.09 on Exp 18's coherent ladder is shallower. Exp 16 (gap 0.14, currently +0.021 high) tolerates a small w_floor raise but is the guardrail — do not overshoot it past ~0.60. (4) Leave the far-fraud route, consensus gate, softmax, and lapse untouched. Expected effect: Exp 7 from 0.38 toward ~0.50-0.55, Exps 13/15/12/9 each closed by half or more, Exps 17/18 to within ~0.04, with all currently-good cells held. Since the loop guarantees the base never regresses, this targeted pass is low-risk and addresses the two largest error pools directly.

**Outcome of this advice:** iter 1 candidate loss=0.0626 -> iter 2 candidate loss=0.0715 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The gate rejected this candidate (loss 0.0626 -> 0.0715), and the per-experiment decomposition shows exactly why. The tuning pass had two halves, and they fared oppositely. The half aimed at the gap-0.09 pair WORKED and should be kept: the shallower w_floor/rho_e plus the sharper, higher-threshold credibility gate moved Exp 18 from -0.064 to -0.015 error, Exp 16 from +0.020 to +0.013, Exp 12 from +0.017 to +0.004, and Exp 17 from -0.107 to -0.098, with Exp 14 held. The half aimed at the small-gap composition cells BACKFIRED: Exp 9 blew up from -0.038 to +0.146 error (0.522 -> 0.706 vs real 0.560), Exp 13 overshot from +0.046 to -0.075 (0.336 -> 0.215 vs real 0.290), Exp 15 moved AWAY from target (0.423 -> 0.409 vs real 0.483), and Exps 4/5 drifted. Most damning, Exp 7 — the single cell motivating the near-route edits and ~60% of the residual error — is essentially UNMOVED (0.380 -> 0.373 vs real 0.580). Two very different near-route calibrations (iter 1: sig_gap-throttled; iter 2: unthrottled with a sharpened gate) have now produced the same Exp 7 value, which is diagnostic: the route is effectively closed on that panel under any threshold in the arbiter's prescribed 0.40-0.45 window, so its chance fraction evidently sits below the gate (or its gap attenuates L beyond rescue), while the sharpened gate simultaneously over-opens the panels it should not touch (Exps 9/13). The proposer's pre-submission arithmetic ('w_top ~ -0.1 on chance-laden dense ladders', 'Exp 13 essentially unchanged') was demonstrably wrong; a per-experiment (gap, c_frac, L, G, w_top) audit table would have caught the Exp 9/13 overshoots before submission.
Rationale: Regenerate. My iter-1 advice — lower c_half toward 0.40-0.44, raise c_near, widen s_near — was implemented in overshooting form and REJECTED by the gate, so I am not repeating it. The next pass should be a surgical revert-and-keep on the accepted iter-1 base, plus a diagnosed (not blind) attempt at Exp 7. Concretely: (1) KEEP verbatim the w_floor lift (~-0.20), gentled rho_e (~1.15), and the credibility-ramp retuning (c_cred ~1.1-1.4, c_cred_half ~0.60, s_cred ~0.055) — these delivered Exps 18/16/12/17 and are the round's real progress. (2) REVERT the near-route gate sharpening: c_half back to ~[0.44, 0.48], s_c back to ~[0.08, 0.11], and recalibrate c_near so the gap-0.05 chance-laden magnitude returns to iter-1 levels (Exp 13 ~0.33, Exp 15 recovering toward 0.48). On the sig_gap drop: the +0.18 Exp 9 jump is the signature of the un-throttled route firing on a panel sig_gap used to protect, so either restore the sig_gap factor or verify with the panel numbers that the reverted gate holds Exp 9 closed; do not keep it dropped unconditionally. (3) PIN the shared-machinery parameters (w_p, v_lo, v_hi, tilt, kappa_lo, kappa_gate, lambda_ext, g0, s_gap, beta, epsilon) to the accepted iter-1 fitted values rather than re-fitting them, so Exps 3-6/8/10/11 stop drifting (Exp 4 slipped 0.059 with no intended mechanism change). (4) For Exp 7, stop tweaking the gate — two failed attempts prove the threshold placement is not the issue. Before editing anything, compute and report the panel's actual (gap, c_frac) and the resulting (L, G, comp_near, w_top). If c_frac is below ~0.30, the composition route cannot reach Exp 7 by construction and the only in-family lever left is the uniform-skepticism floor e_min (which binds exactly where the kernel is ~0, i.e. dense ladders): raise e_min toward [0.30, 0.45] and compensate the gap-0.05 cells with a reduced c_near — a measured trade of a few hundredths on Exps 10/14/12 (guardrails: hold Exp 10 within 0.03 of 0.599, Exp 14 within 0.03 of 0.572, Exp 16 <= ~0.58) for a large Exp 7 gain. If instead the diagnostic shows Exp 7's gap is ~0.09 with near-zero c_frac, the fix belongs in the kernel depth / cred-ramp low-c_frac behavior, and the near route should be left alone. The revert-and-keep package alone should land near loss ~0.055-0.058 (below the accepted 0.0626), so the Exp 7 attempt is upside, not a prerequisite for beating the gate.

**Outcome of this advice:** iter 2 candidate loss=0.0715 -> iter 3 candidate loss=0.0622 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The revert-and-keep strategy worked: the gate accepted this candidate (0.0626 -> 0.0622), and the per-experiment decomposition confirms the surgical split. The kept half (shallower w_floor ~-0.20, gentled rho_e ~1.15, sharper high-threshold credibility gate) held its gains: Exp 18 at -0.041 (from -0.064 at iter 1), Exp 16 at +0.013, Exp 14 held at -0.013, Exp 5 is now essentially perfect (-0.008), and Exps 6/10/11 remain locked within 0.008. The reverted half successfully un-did the iter-2 damage: Exp 9 recovered from 0.706 to 0.529 (real 0.560), Exp 13 from 0.215 to 0.320 (real 0.290). This validates the iter-2 diagnosis that the sig_gap throttle and the gate placement, not the kept edits, caused the blowup. However, the candidate is only a marginal improvement, and the loss remains dominated by three persistent pools: Exp 7 (-0.210, ~0.044 squared error, roughly 40% of the total alone), Exp 17 (-0.102, essentially unmoved across all three iterations at 0.614/0.624/0.619 vs real 0.721), and Exp 15 (-0.067, also stuck at 0.423/0.409/0.416 vs real 0.483). Two structural facts emerge from the trajectory: (a) Exp 17 appears capped by the w_top <= w_p clamp — the arbiter-prescribed 'rises toward the plateau value' ceiling — since three different credibility-ramp calibrations all land at ~0.62; (b) Exp 7 has resisted two near-route recalibrations, and the proposer's own panel arithmetic (correctly, in my check) shows the bounded-saturating form with a shallow floor cannot push dense-ladder w_top negative. BUT the proposer declined the one lever my iter-2 advice actually named for this case — the uniform-skepticism floor e_min — on the grounds that it cannot fully reach w_top ~ -0.3. That argument proves too much: partial movement is valuable on the dominant error pool, and my own sign audit of the e_min lever shows it is favorable almost everywhere it binds. e_min binds exactly where the gap kernel is ~0: dense ladders (Exps 7, 9, 10) and gap-0.05 cells (Exps 12, 13, 15). Raising e_min from ~0.15 toward ~0.35 lowers w_base on all of those panels, and the current residuals on those very cells are signed in the anti-follow direction: Exp 7 needs more anti-follow (0.369 vs 0.580), Exp 9 needs more (+0.031), Exp 12 needs more (+0.028), Exp 13 needs more (+0.030), Exp 15 needs more (+0.067), and Exp 16 (gap 0.14, wide right shoulder, E~0.5) also benefits slightly (+0.013 too much following). The only cells it hurts are Exp 10 (currently +0.004, with ~0.03 guardrail slack) and Exp 14 (currently -0.013, ~0.02 slack) — small, bounded costs against a potential halving of the Exp 7 pool. Cells where the kernel is at peak (Exps 17, 18, gap ~0.09) are mechanically unaffected by e_min since E = max(kernel, e_min) is kernel-dominated there.
Rationale: verdict: regenerate

**Outcome of this advice:** iter 3 candidate loss=0.0622 -> iter 4 candidate loss=0.0771 -> the gate marked it REJECTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The gate rejected this candidate (0.0622 -> 0.0771), and the per-experiment decomposition shows a clean, interpretable pattern: the e_min lever WORKS on its primary target but was pushed roughly twice as deep as the data allow. Gains: Exp 7 moved 0.369 -> 0.448 (real 0.580) — the largest single-iteration movement on the dominant residual pool in the entire loop — and Exp 8 (-0.046 -> +0.026), Exp 1 (-0.026 -> -0.003), Exp 3, and Exp 6 all improved. But the uniform floor is flat in gap, so it also slammed every cell where the kernel is small: Exp 9 overshot from -0.031 to +0.080 (0.529 -> 0.640 vs real 0.560), Exp 12 from +0.028 to -0.058, Exp 13 from +0.030 to -0.097 (0.320 -> 0.193 vs real 0.290), Exp 14 from -0.013 to -0.082, and Exp 10 from +0.004 to -0.075. Exp 15 — the other named target — is essentially unmoved (-0.067 -> -0.066), confirming the floor does not reach the gap-0.05 chance-laden cells the way the proposer's audit claimed. The pre-submission panel audit was again materially wrong (predicted Exp 12 ~0.45, got 0.395; Exp 13 ~0.27-0.29, got 0.193; predicted aggregate 0.050-0.055, got 0.0771). Critically, a linear interpolation between the iter-3 and iter-4 calibrations says the optimum sits near d ~ 0.5: the signed errors at the two endpoints bracket the real values on Exps 7, 9, 12, 13 (iter-3 undershoots, iter-4 overshoots on each), so a half-depth floor should land near zero on several of them simultaneously while keeping roughly half the Exp 7 gain. The direction my iter-3 critique prescribed was correct; the proposer implemented it at roughly double the specified depth (e_min ~0.59 vs the recommended 0.30-0.45) and added an unrequested w_floor dip that deepened exactly the cells that then overshot.
Rationale: Regenerate — but this is a re-depth of the same lever, not a new direction. (1) HALVE THE EDIT: raise e_min from the accepted iter-3 value (~0.15) only to [0.32, 0.40] (midpoint ~0.36, inside the 0.30-0.45 window my iter-3 rationale actually specified — the current [0.54, 0.64] is beyond it), and REVERT w_floor fully to the accepted iter-3 value (~-0.20), dropping the compensating dip entirely. The dip contributed ~0.045 of extra weight reduction on precisely the gap-0.05 cells that overshot (Exps 9/12/13/14); removing it retains most of the pure-e_min effect on Exp 7's dense ladder while cutting the collateral damage roughly in half. Interpolating the two endpoint calibrations, this should land Exp 9 near ~0.56 (real 0.560), Exp 12 near ~0.44, Exp 13 near ~0.26, Exp 14 near ~0.53, Exp 10 near ~0.56, while keeping Exp 7 near ~0.41 — a net squared-error improvement of roughly 0.015-0.018 by endpoint arithmetic, comfortably below the 0.0622 gate. (2) PIN every other parameter to the accepted iter-3 values — do not touch the near-route gate (c_half, s_c, sig_gap throttle), the credibility ramp, the kernel placement, or any shared machinery; two failed attempts have established those are calibrated. (3) Do not chase Exp 15 or Exp 17 in this pass: Exp 15 has been stuck at ~0.42 across four calibrations and the near-route route to it is known to blow up Exps 9/13; Exp 17 is capped by the arbiter-prescribed w_top <= w_p clamp. Accept both as structural residuals this round. (4) Before submitting, verify the arithmetic with the actual panel numbers (gap, c_frac, E, w_top per experiment) — the last two rejections were both accompanied by pre-submission audits that mispredicted the gap-0.05 cells by 0.05-0.10; if the audit and the iter-3/iter-4 endpoint interpolation disagree, trust the interpolation and choose the shallower e_min. The guardrails: Exp 10 must stay >= ~0.56, Exp 14 >= ~0.52, Exp 9 within ~0.04 of 0.560.

**Outcome of this advice:** iter 4 candidate loss=0.0771 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "SG-CGD (Small-Gap Composition-Gated Outlier Distrust), half-depth uniform-skepticism edition. Subjects integrate every binary expert rating as a sign vote into a zero-sum evidence score with subjective weights computed relative to the panel: a tilted trust plateau over [v_lo, v_hi] (ceiling just above 0.98), a consensus-gated chance flank with partial-ceiling promotion, and an outlier branch for above-ceiling claims consisting of (a) a bounded, saturating base distrust peaked at gap ~0.088 sitting on a uniform-skepticism floor, (b) a near-kernel composition amplification (narrow Gaussian at gap ~0.035, chance-fraction gate, sig_gap-throttled) that deepens distrust only of small-edge outliers on chance-laden scales, (c) a far-fraud route at gap ~0.185, and (d) a composition-gated edge-credibility ramp that promotes a well-separated 99% expert back toward plateau trust when the surrounding scale is unreliable. The single theoretical claim updated this round: the uniform-skepticism floor e_min — the residual distrust applied to a panel-max expert even when the gap kernel is silent (dense ladders, gap ~0.05 cells) — sits at a MODERATE depth (~0.36), roughly halfway between the near-zero floor of the accepted calibration and the deep floor of the rejected overshoot. This floor is what generates the empirically required mild anti-follow of 99% experts on dense validity ladders (Exp 7) without unbounded sign flips. Choice is a low-gain softmax plus uniform lapse.",
  "predict": "def predict(parameters, state, history):\n    # SG-CGD: Small-Gap Composition-Gated Outlier Distrust with a\n    # composition-gated edge-credibility ramp, half-depth-floor edition.\n    # Stimulus: array of shape (2, n_features); row 0 = option A,\n    # row 1 = option B; entries are binary expert ratings (0/1).\n    # Each feature j casts a sign vote sign(a_j - b_j), weighted by a\n    # subjective trust function of the expert's STATED validity v,\n    # evaluated RELATIVE to the panel:\n    #   (i) TRUST PLATEAU with shallow tilt: all v in [v_lo, v_hi]\n    #       (v_hi just above 0.98, so a 98% expert stays INSIDE the\n    #       plateau) get approximately equal weight w_p, with a\n    #       within-band tilt so higher-validity cues slightly\n    #       outrank lower ones.\n    #  (ii) CHANCE FLANK + CONSENSUS GATE: v near 0.5 gets a small\n    #       negative weight (-kappa_lo at v=0.5, rising linearly to\n    #       the tilted plateau at v_lo). When an above-ceiling claim\n    #       is present the flank is gated and chance cues are\n    #       PROMOTED to a partial ceiling kappa_gate*w_p.\n    # (iii) BOUNDED BASE DISTRUST of the panel-max expert: a\n    #       saturating shrinkage toward a SHALLOW floor,\n    #         w_base = w_floor + (w_p - w_floor)*exp(-rho_e*E)\n    #       with E = max(peaked gap kernel, e_min); the kernel peaks\n    #       at gap ~0.088 (steep left wall s_l, wide right shoulder\n    #       s_r). The UNIFORM-SKEPTICISM FLOOR e_min is the single\n    #       lever moved this round: raised from ~0.15 to ~0.36\n    #       (half the rejected overshoot depth), so dense-ladder\n    #       and gap-0.05 outliers carry a moderate residual\n    #       distrust even where the gap kernel is silent.\n    #  (iv) NEAR-KERNEL COMPOSITION AMPLIFICATION (small-gap gate):\n    #         comp_near = c_near * sig_gap * L(gap) * G(c_frac)\n    #       L is a NARROW Gaussian centered g_near ~ 0.035 (inert for\n    #       gap >= 0.07); G is a chance-fraction logistic. The\n    #       sig_gap factor is ESSENTIAL and retained: it throttles\n    #       the route on dense ladders (gap ~ 0.01), which protects\n    #       the cliff-conflict cells from spurious anti-votes.\n    #  (v) FAR-FRAUD ROUTE: comp_far = c_far * sig_gap * F(gap) *\n    #       G_far(c_frac), F a SHARP logistic at g_far ~ 0.185 with\n    #       its own chance gate: closed on every gap 0.09-0.14 panel\n    #       tested here, open for far outliers over chance-laden\n    #       panels (gap >= ~0.18).\n    # (vi) COMPOSITION-GATED EDGE-CREDIBILITY RAMP: for gap >= ~0.075,\n    #       cred = c_cred * R(gap) * Q(c_frac), capped at the plateau\n    #       value. On a chance-laden panel a well-separated 99%\n    #       expert is the only credible voice and is promoted back\n    #       toward plateau trust; on a coherent-ladder panel the ramp\n    #       stays closed and residual kernel distrust persists.\n    # Choice is a low-gain softmax (inverse temperature beta) over\n    # the two mirrored scores, plus a uniform lapse (epsilon).\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {val.shape[0]} != n_features {n_features}.\"\n        )\n\n    w_p = float(parameters[\"w_p\"])\n    v_lo = float(parameters[\"v_lo\"])\n    v_hi = float(parameters[\"v_hi\"])\n    tilt = float(parameters[\"tilt\"])\n    kappa_lo = float(parameters[\"kappa_lo\"])\n    kappa_gate = float(parameters[\"kappa_gate\"])\n    lambda_ext = float(parameters[\"lambda_ext\"])\n    g0 = float(parameters[\"g0\"])\n    s_gap = float(parameters[\"s_gap\"])\n    g_star = float(parameters[\"g_star\"])\n    s_l = float(parameters[\"s_l\"])\n    s_r = float(parameters[\"s_r\"])\n    e_min = float(parameters[\"e_min\"])\n    w_floor = float(parameters[\"w_floor\"])\n    rho_e = float(parameters[\"rho_e\"])\n    d_w = float(parameters[\"d_w\"])\n    g_near = float(parameters[\"g_near\"])\n    s_near = float(parameters[\"s_near\"])\n    c_near = float(parameters[\"c_near\"])\n    c_half = float(parameters[\"c_half\"])\n    s_c = float(parameters[\"s_c\"])\n    g_far = float(parameters[\"g_far\"])\n    s_far = float(parameters[\"s_far\"])\n    c_far = float(parameters[\"c_far\"])\n    c_half_far = float(parameters[\"c_half_far\"])\n    s_c_far = float(parameters[\"s_c_far\"])\n    c_cred = float(parameters[\"c_cred\"])\n    g_edge = float(parameters[\"g_edge\"])\n    s_edge = float(parameters[\"s_edge\"])\n    c_cred_half = float(parameters[\"c_cred_half\"])\n    s_cred = float(parameters[\"s_cred\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    def _sigmoid(x):\n        # Numerically stable logistic via the identity\n        # sigmoid(x) = 0.5 * (1 + tanh(x / 2)).\n        return 0.5 * (1.0 + np.tanh(0.5 * x))\n\n    a, b = stim[0], stim[1]\n\n    # Feature-wise votes: +1 where A wins the feature, -1 where B\n    # wins, 0 on ties (a tied cue is uninformative for the pair).\n    votes = np.sign(a - b)\n\n    v = np.clip(val, 0.5, 1.0)\n\n    # ---- Panel structure: top validity and its gap to the\n    # next-highest DISTINCT validity in the panel. ----\n    v_top = float(v.max())\n    below = v[v < v_top - 1e-9]\n    # If every expert shares the top validity, the panel offers no\n    # corroboration ladder; treat the next rung as chance (0.5).\n    v_second = float(below.max()) if below.size > 0 else 0.5\n    gap = v_top - v_second\n\n    # Outlier = a claim ABOVE the trust ceiling. With v_hi ~ 0.981 a\n    # 0.98-topped panel stays INSIDE the trusted plateau; only 99%+\n    # claims enter the distrust branch.\n    outlier = v_top > v_hi\n\n    if outlier:\n        sig_gap = _sigmoid((gap - g0) / s_gap)\n    else:\n        sig_gap = 0.0\n\n    # ---- (ii): consensus gate for the chance flank. ----\n    if outlier:\n        gate = lambda_ext + (1.0 - lambda_ext) * float(sig_gap)\n    else:\n        gate = 0.0\n\n    # ---- (i) + (ii): tilted plateau with a consensus-gated\n    # chance flank whose promotion is capped at the PARTIAL\n    # ceiling kappa_gate * w_p. ----\n    w = np.full(n_features, w_p, dtype=float)\n    span = max(v_hi - v_lo, 1e-6)\n    plateau = (v >= v_lo) & (v <= v_hi)\n    if np.any(plateau):\n        # Shallow within-plateau validity gradient: higher-validity\n        # experts slightly outrank lower ones while the band remains\n        # 'approximately equal'.\n        w[plateau] = w_p * (\n            1.0 - tilt * (v_hi - v[plateau]) / span\n        )\n    flank = v < v_lo\n    if np.any(flank):\n        denom = max(v_lo - 0.5, 1e-6)\n        # Flank rises linearly to the TILTED plateau value at v_lo,\n        # keeping the weight profile continuous at the band edge.\n        w_lo_eff = w_p * (1.0 - tilt)\n        w_flank = -kappa_lo + (w_lo_eff + kappa_lo) * np.clip(\n            v[flank] - 0.5, 0.0, None\n        ) / denom\n        # With an above-ceiling claim in the panel, chance cues are\n        # counted toward a partial ceiling kappa_gate*w_p rather\n        # than inverted (and rather than fully trusted).\n        w[flank] = (1.0 - gate) * w_flank + gate * kappa_gate * w_p\n\n    # ---- (iii)-(vi): outlier distrust / credibility architecture. ----\n    if outlier:\n        # Chance fraction of the panel (share of experts below the\n        # trust band) drives both composition gates.\n        c_frac = float(np.mean(flank))\n\n        # (iii) Bounded base distrust: saturating shrinkage toward a\n        # shallow floor, driven by a peaked kernel in the\n        # uncorroborated edge (peak at g_star, steep left wall,\n        # wide right shoulder), sitting on the uniform-skepticism\n        # floor e_min (RAISED to ~0.36 this round: half the rejected\n        # overshoot depth, per the endpoint interpolation).\n        width = s_l if gap < g_star else s_r\n        E_kernel = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))\n        E = max(E_kernel, e_min)\n        w_base = w_floor + (w_p - w_floor) * float(np.exp(-rho_e * E))\n\n        # (iv) Near-kernel composition amplification: fires only in\n        # the small-gap regime (L centered g_near, narrow width),\n        # only on chance-laden scales (G threshold c_half), and is\n        # THROTTLED by sig_gap so dense ladders (gap ~ 0.01, where\n        # sig_gap ~ 0.08) are protected from spurious anti-votes.\n        L = float(np.exp(-0.5 * ((gap - g_near) / s_near) ** 2))\n        G_near = _sigmoid((c_frac - c_half) / s_c)\n        comp_near = c_near * float(sig_gap) * L * float(G_near)\n\n        # (v) Far-fraud route: sharp logistic at g_far with its own\n        # chance gate; closed on every gap 0.09-0.14 panel tested\n        # here, open for far outliers over chance-laden panels.\n        F_far = _sigmoid((gap - g_far) / s_far)\n        G_far = _sigmoid((c_frac - c_half_far) / s_c_far)\n        comp_far = c_far * float(sig_gap) * float(F_far) * float(G_far)\n\n        # (vi) Composition-gated edge-credibility ramp: for\n        # well-separated outliers (gap >= ~g_edge) on chance-laden\n        # scales, the panel-max expert is promoted back toward\n        # plateau trust (capped at w_p); on coherent ladders the\n        # ramp stays closed and residual kernel distrust persists.\n        R_cred = _sigmoid((gap - g_edge) / s_edge)\n        Q_cred = _sigmoid((c_frac - c_cred_half) / s_cred)\n        cred = c_cred * float(R_cred) * float(Q_cred)\n\n        # Small bounded extremity term.\n        ext = d_w * (v_top - v_hi) / max(1.0 - v_hi, 1e-6)\n\n        w_top = w_base - ext - comp_near - comp_far + cred\n        # The outlier is at most as trusted as a plateau expert.\n        w_top = min(w_top, w_p)\n        # All experts tied at the panel maximum share the same\n        # weight (a pair of 0.99 super-experts doubles the vote).\n        w[v >= v_top - 1e-9] = w_top\n\n    # Weighted evidence score. B's score is the mirror of A's\n    # because every vote is zero-sum across the two options.\n    s_a = float(np.dot(w, votes))\n    scores = np.array([s_a, -s_a], dtype=float)\n\n    # Numerically stable low-gain softmax. When all cues tie\n    # (s_a == 0) the softmax is exactly uniform.\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n\n    # Independent lapse: with probability epsilon output a uniform\n    # pick over the two options.\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # guard against float drift\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "w_p": "[0.88, 0.92]",
    "v_lo": "[0.52, 0.55]",
    "v_hi": "[0.9805, 0.9825]",
    "tilt": "[0.09, 0.13]",
    "kappa_lo": "[0.19, 0.22]",
    "kappa_gate": "[0.66, 0.74]",
    "lambda_ext": "[0.30, 0.42]",
    "g0": "[0.041, 0.045]",
    "s_gap": "[0.012, 0.014]",
    "g_star": "[0.084, 0.092]",
    "s_l": "[0.013, 0.017]",
    "s_r": "[0.040, 0.050]",
    "e_min": "[0.32, 0.40]",
    "w_floor": "[-0.24, -0.17]",
    "rho_e": "[1.05, 1.30]",
    "d_w": "[0.08, 0.12]",
    "g_near": "[0.032, 0.038]",
    "s_near": "[0.014, 0.018]",
    "c_near": "[3.5, 4.0]",
    "c_half": "[0.44, 0.48]",
    "s_c": "[0.085, 0.105]",
    "g_far": "[0.180, 0.190]",
    "s_far": "[0.018, 0.022]",
    "c_far": "[20, 26]",
    "c_half_far": "[0.57, 0.63]",
    "s_c_far": "[0.06, 0.08]",
    "c_cred": "[1.15, 1.40]",
    "g_edge": "[0.070, 0.080]",
    "s_edge": "[0.018, 0.022]",
    "c_cred_half": "[0.57, 0.63]",
    "s_cred": "[0.05, 0.06]",
    "beta": "[0.38, 0.46]",
    "epsilon": "[0.035, 0.055]",
    "validities": "validities"
  },
  "rationale": "This is the critic-prescribed half-depth re-edit of the single e_min lever on the ACCEPTED iter-3 base. The mechanism code is byte-identical to the running-best candidate; exactly ONE parameter box moves (e_min: [0.13, 0.17] -> [0.32, 0.40], midpoint ~0.36, inside the critic's 0.30-0.45 window and roughly half the rejected iter-4 depth of ~0.59). The rejected iter-4 candidate's w_floor dip is NOT present here — w_floor stays at the accepted iter-3 box [-0.24, -0.17] — and every other parameter (near-route gate, credibility ramp, kernel placement, shared machinery, far-fraud route) is pinned to the accepted values, exactly as the iter-4 feedback instructs after two failed attempts established those are calibrated.\n\nPre-submission panel audit with the actual panel numbers, as requested (using box midpoints: w_p=0.90, w_floor=-0.205, rho_e=1.175, e_min=0.36):\n\n- Dense ladders (gap ~0.01, kernel ~0, E = e_min = 0.36): w_base = -0.205 + 1.105*exp(-1.175*0.36) = -0.205 + 1.105*0.655 = +0.52. At the iter-3 floor (e_min=0.15) the same arithmetic gives +0.74, so the edit lowers dense-ladder w_top by ~0.22 per 99% expert. Exp 7 (anti-follow, real 0.580, iter-3 0.369): the two 99% experts' combined vote drops by ~0.44 trusted-cue units on the discriminating trials, moving anti-follow toward ~0.42-0.46 — roughly half of iter-4's gain (0.448) retained, as the endpoint interpolation predicts. Exp 9 (real 0.560, iter-3 0.529): the small extra distrust moves it to ~0.55-0.57, i.e. ON target rather than iter-4's 0.640 overshoot. Exp 10 (real 0.599, iter-3 0.603): follow drops to ~0.56-0.57, inside the >=0.56 guardrail with margin.\n\n- Gap-0.05 cells (kernel ~0, E = e_min): w_base moves +0.74 -> +0.52. Exp 13 (c_frac 0.556, G~0.77, comp_near ~1.15): w_top -0.50 -> -0.71, follow 0.320 -> ~0.26 (real 0.290) — the endpoint interpolation's landing zone. Exp 12 (c_frac 0.375, G~0.19, comp_near ~0.29): w_top ~0.38 -> ~0.17, follow 0.480 -> ~0.44 (real 0.4525). Exp 14 (coherent, c_frac 0.30, G~0.16): w_top 0.43 -> 0.23, follow 0.558 -> ~0.52-0.53, at the >=0.52 guardrail edge — the accepted cost. Exp 15 (c_frac ~0.60, gate fully open, comp_near ~1.4): the extra ~0.22 of floor distrust adds anti-follow on a cell that has been stuck at 0.42 vs real 0.483 — a small favorable drift, not chased.\n\n- Kernel-dominated cells (gap ~0.09, E = kernel ~ 1): e_min is mechanically inert. Exps 17/18 are unchanged from the accepted base (0.619/0.423), as are Exps 16 (gap 0.14, kernel 0.51 > 0.36) and 6 (far-route dominated). No-outlier experiments (1-5, 8) are structurally untouched.\n\nWhere my audit and the critic's endpoint interpolation could disagree (Exps 13/14, where I project slightly past the interpolation's landing zones), I follow the critic's explicit instruction to trust the interpolation and choose the shallower side: e_min's box TOP is 0.40, not 0.45, so the deepest sampled subject stays at the interpolation's upper edge. Expected movement vs the accepted base: Exp 7 0.369 -> ~0.42-0.45, Exp 9 0.529 -> ~0.56, Exp 12 0.480 -> ~0.44, Exp 13 0.320 -> ~0.26-0.28, with Exps 3-6/8/10/11/16/17/18 held within ~0.01-0.04 — a projected aggregate loss near ~0.050-0.055, comfortably below the 0.0622 gate floor. Exps 15 and 17 remain accepted structural residuals this round per the feedback."
}
```
```

## Usage

```json
{
  "input_tokens": 65064,
  "output_tokens": 8530,
  "reasoning_tokens": 3253,
  "provider": "Sail Research"
}
```
