# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=9):** [0.98, 0.9, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0]
  trial 8: A=[1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0]
  trial 9: A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1]
  trial 11: A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0]
  trial 12: A=[1, 0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0, 1, 0]
  trial 13: A=[0, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0, 1]
  trial 14: A=[1, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 0, 0]
  trial 15: A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  trial 16: A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0]
  trial 17: A=[0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  trial 18: A=[1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  trial 19: A=[1, 0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0, 1]
  trial 20: A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 0, 0]
  trial 21: A=[0, 1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 22: A=[1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 23: A=[1, 0, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0]
  trial 24: A=[0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]

**Rationale:** CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x softmax x lapse) and differ ONLY in the weight function. My advocated theory (FDEI) compresses diagnosticity heavily (delta in [0.1,0.2]), so among genuinely diagnostic cues its weights are nearly FLAT: w(0.98) in [1.14,1.31], w(0.90) in [1.08,1.17], w(0.75) ~ 1.01, w(0.55) in [0.69,0.85]. The competitor (SITT) has a steep inverted-U: w(0.55) ~ 0.18-0.23, w(0.75) ~ 0.68-0.74, w(0.90) ~ 0.95-1.0 (peak), w(0.98) NEGATIVE in [-1.14,-0.70]. The single most informative behavioral quantity is therefore the EXCHANGE RATE between a strong expert and weak (55%) experts, measurable as the crossover point k* in a weak-cue dose-response ladder. Validities: [0.98, 0.90, 0.75, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55] -- one expert per regime plus SIX 55% experts, enough to bracket both theories' crossovers. This is a design no prior experiment used: no v=0.5 annihilation logic ([4]), no tally-margin decorrelation ([1],[2]), no bounded-box ceiling attack ([3]), no validity-ratio ER between diagnosticity classes ([5]), and unlike [6] the flagship is not a standalone 0.99 inversion but graded crossover ladders (the 0.98 cliff cells here appear only inside coalitions).

FAMILY K (flagship crossover ladder). K1-K6: A wins the 90% expert, B wins k of the 55% experts, k=1..6. The crossover k* = w(0.90)/w(0.55) is boxed DISJOINTLY: FDEI k* in [1.27, 1.69] (crosses 0.5 between k=1 and k=2); SITT k* in [4.1, 5.6] (crosses between k=4 and k=5). Every single cell is box-disjoint. FDEI center (delta=0.15, tau=0.01, beta=0.32): p(A) = .55, .43, .32, .22, .15, .10 for k=1..6. SITT center (v_lo=.51, v_peak=.90, v_hi=.96, rho1=.7, rho2=.3, kappa_hi=1.15, beta=1.0, eps=.055): p(A) = .81, .75, .68, .59, .49, .40. Box-level: at k=1, FDEI [.53,.59] vs SITT [.75,.84]; k=2, [.42,.46] vs [.69,.79]; k=3, [.31,.42] vs [.61,.72]; k=4, [.22,.31] vs [.51,.64]; k=5, [.15,.24] vs [.42,.55]; k=6, [.10,.17] vs [.33,.45]. Two psychometric curves that cross each other in opposite directions: the observed crossover location is a direct, interpretable estimate of the strong:weak exchange rate, separated by a factor of ~3 across the two boxes. Km2/Km4/Km6 mirror the ladder for side balance.

FAMILY R (second crossover at a mid-strength reference). R1-R4: A wins the 75% expert vs k 55% experts (k=1..4); Rm3 mirrors. FDEI crossover k* = w(0.75)/w(0.55) in [1.18, 1.46]; SITT in [3.0, 4.1] -- again disjoint. Centers: FDEI .54/.42/.31/.22 vs SITT .72/.64/.55/.45. Two independent exchange-rate estimates, both disjoint, test whether the dissociation scales with reference strength as each theory's weight function demands (FDEI: ER roughly constant ~1.3; SITT: ER shrinks 4.9 -> 3.5 as reference drops from 0.90 to 0.75).

FAMILY D/C/H (cliff cells: 0.98 as anti-cue vs near-equal-cue). D1 (A wins 0.98, B wins 0.90): FDEI box [.50,.53] -- a coin flip, because compressed weights make 98% and 90% nearly exchangeable; SITT box [.04,.06] -- subjects should choose AGAINST the 98% endorsement. D1m mirrors. C1 (A wins 0.90+0.75, B wins 0.98+one 0.55): FDEI [.49,.55] (net evidence ~0); SITT [.93,.97] (the 98% endorsement is worth about one trusted ANTI-cue, so the two trusted experts plus the inversion overwhelm it). C1m mirrors. H2 (A wins 0.98 + two 55%s, B wins 0.90): FDEI [.66,.80] (weak majority plus super-expert beats the peak expert); SITT [.06,.13] (the super-expert's anti-vote sinks the coalition). H2m mirrors. These cells separate a near-tie from a 90-point swing.

HONEST KNIVES AGAINST MY OWN THEORY. K1 and R1 are the sharpest: FDEI's delta<=0.2 compression caps the 90-vs-55 exchange rate at 1.69, so it cannot predict p(A) > 0.59 on K1 or > 0.56 on R1 for ANY parameter setting. If subjects follow the strong expert at p >= 0.70 on K1, my flattening claim is falsified outright and the design doubles as a direct delta-estimation experiment (observed k* maps monotonically onto delta). Conversely K3/K4 and R2/R3 falsify SITT if subjects favor the weak stacks at all (SITT demands p(A) >= 0.51 at K4 for every parameter).

CONTROLS. X1/X2 (one 55% expert won by each side): BOTH theories must predict exactly 0.50 for every parameter -- a parameter-free side/display-bias assay. G1/G1m are agreement anchors (both theories > 0.85 at center: FDEI .94, SITT .96) pinning lapse/temperature. Tied features are a MIX of 0/0 and 1/1 ties so that 'count the 1s' is not a viable heuristic (e.g., K6 has A with three 1s vs B with eight 1s while the theories disagree only about the discriminators). No all-tie trials; every trial requires scanning all 9 experts. 24 unique pairs x K = 96//24 = 4 reps = 96 trials; sides balanced (10 A-tilted, 12 B-tilted, 2 neutral under FDEI centers; every non-anchor family contains an A/B mirror). PREDICTED PATTERN IF MY THEORY IS RIGHT: shallow FDEI psychometrics crossing 0.5 between k=1 and 2 in both ladders, coin flips on D1/C1, A-choice ~0.74 on H2, exactly 0.50 on X. IF THE COMPETITOR IS RIGHT: steep SITT psychometrics crossing between k=4 and 5 (K) and k=3 and 4 (R), near-ceiling strong-expert following at low k, ~0.05 on D1, ~0.95 on C1, ~0.07 on H2. The boxes are disjoint on 20 of 24 cells.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Flattened-Diagnosticity Evidence Integration (the extreme-compression corner of the Bayesian cue-counting family): people integrate ALL binary expert ratings into one evidence score, where each discriminating cue casts a sign vote weighted by w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta — chance cues (v=0.5) are annihilated exactly, but among genuinely diagnostic cues the subjective weight is a HEAVILY flattened (delta ~ 0.15) version of the normative log-odds, and choice is a low-gain softmax (2*beta ~ 0.6) plus a small uniform lapse. The new theoretical claim is that the compression exponent delta is itself the structural margin-compression device: shrinking delta equalizes weights, which GROWS count-dominated conflict margins (where cues split across options) while SHRINKING alignment-dominated cumulative margins (where several strong cues pile onto one side). This differential action — not any global gain knob, and not trial-level normalization — is what lets a single small beta simultaneously produce moderate TTB-leaning conflict-trial behavior (Exp 1), a slightly positive tally-vs-top-cue regression slope (Exp 2), graded ~0.62 tally-following (Exp 3), and near-chance |d|=1 agreement driven by the cancellation between super-expert matches and super-expert reversals (Exp 4).

**Parameters:**
- delta: [0.1, 0.2]
- tau: [0.0, 0.04]
- beta: [0.27, 0.37]
- epsilon: [0.0, 0.05]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Skeptical (Inverted-Trust) Tallying with a CALIBRATED trust cliff. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score whose per-cue weight is a non-monotone function of stated validity: (i) near-chance experts (v ~ 0.5) carry a small NEGATIVE weight (their endorsements are treated as mildly anti-diagnostic noise); (ii) trust rises smoothly to a peak at v ~ 0.9 (the most trusted experts); (iii) claimed near-certainty (v > ~0.96) falls past a cliff into a MODERATE negative weight of roughly MINUS ONE trusted-cue unit — a 99% expert is treated as approximately one reliable anti-cue, not a veto. The calibrating claim is that distrust magnitude is matched to trust magnitude (|w(0.99)| ~ w_peak ~ 1, logit contribution 2*beta*w(0.99) ~ -2.1): a single super-expert loses to any 2-cue opposing coalition (weak-side choice in super-expert conflicts) but cannot invert coalitions that include it (moderate tally-following when the 0.99 cue sits inside a tally majority). This intermediate depth sits deliberately between the previously falsified extremes: the deep veto (iter 6, which deterministically inverted Experiment 3) and the shallow discount (iter 7, which let super-experts win conflicts, SSR 0.69).

**Parameters:**
- v_lo: [0.505, 0.515]
- v_peak: [0.89, 0.91]
- v_hi: [0.957, 0.963]
- rho1: [0.68, 0.75]
- rho2: [0.25, 0.35]
- rho_lo: {1}
- kappa_lo: [0.075, 0.105]
- kappa_hi: [1.0, 1.3]
- tau: [0.011, 0.014]
- beta: [0.9, 1.1]
- epsilon: [0.04, 0.07]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Skeptical (Inverted-Trust) Tallying, calibrated-cliff edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by a skewed inverted-U 'trust band' of stated validity:
    #   v_lo <= v <= v_peak :  w = ((v - v_lo)/(v_peak - v_lo))^rho1
    #                          (rise from zero at the lower edge to the
    #                          trusted peak, normalized to 1)
    #   v_peak < v <= v_hi :  w = ((v_hi - v)/(v_hi - v_peak))^rho2
    #                          (gentle fall from the peak toward the
    #                          cliff; keeps 0.9-0.92 cues near-peak)
    #   v < v_lo           :  w = -kappa_lo * ((v_lo - v)/(v_lo - 0.5))^rho_lo
    #                          (chance experts: kappa_lo == |w(0.5)| in
    #                          trusted-peak units, zero at the edge)
    #   v > v_hi           :  w = -kappa_hi * (1 - exp(-(v - v_hi)/tau))
    #                          (moderate, saturating distrust lobe in
    #                          the SAME peak units; v_hi ~ 0.96 so ONLY
    #                          near-certain (v ~ 0.99) experts fall in,
    #                          and |w(0.99)| ~ 1 trusted-cue unit)
    # Votes are summed into a zero-sum evidence score; choice is a
    # softmax with inverse temperature beta plus a uniform lapse.
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

    v_lo = float(parameters["v_lo"])
    v_peak = float(parameters["v_peak"])
    v_hi = float(parameters["v_hi"])
    rho1 = float(parameters["rho1"])
    rho2 = float(parameters["rho2"])
    rho_lo = float(parameters["rho_lo"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_hi = float(parameters["kappa_hi"])
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)
    w = np.zeros(n_features, dtype=float)
    below = v < v_lo
    above = v > v_hi
    rise = (~below) & (v <= v_peak)
    fall = (~below) & (v > v_peak) & (~above)

    # (i) Lower lobe: normalized so kappa_lo IS |w(0.5)| in trusted-
    # peak units; zero at the band edge, most negative at v = 0.5.
    denom_lo = max(v_lo - 0.5, 1e-6)
    w[below] = -kappa_lo * np.power(
        np.clip(v_lo - v[below], 0.0, None) / denom_lo, rho_lo
    )

    # (ii) In-band rise from v_lo to the trust peak (weight 1).
    span_r = max(v_peak - v_lo, 1e-6)
    vr = np.clip((v[rise] - v_lo) / span_r, 0.0, 1.0)
    w[rise] = np.power(vr, rho1)

    # (iii) In-band fall from the peak toward the cliff at v_hi.
    span_f = max(v_hi - v_peak, 1e-6)
    vf = np.clip((v_hi - v[fall]) / span_f, 0.0, 1.0)
    w[fall] = np.power(vf, rho2)

    # (iv) Distrust lobe above v_hi: saturating exponential in
    # the SAME (peak-normalized) units as the band -- kappa_hi is
    # directly |w(v -> 1)| in units of the trusted-cue weight.
    w[above] = -kappa_hi * (1.0 - np.exp(-(v[above] - v_hi) / tau))

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable softmax over the two weighted scores.
    # When all cues tie (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"P(follow the 98% expert | 98%-vs-90% conflict trials).\n\n    A 'D-cell' trial is one where the top expert (v=0.98, feature 0) and the\n    second expert (v=0.90, feature 1) endorse OPPOSITE options, while the\n    third expert (v=0.75, feature 2) is tied. On these trials the metric is\n    the fraction of choices that follow the option endorsed by the 98%\n    expert. FDEI (compressed diagnosticity) treats 0.98 and 0.90 as nearly\n    exchangeable, so this should be ~0.50 (coin flip). SITT (inverted trust)\n    assigns the 98% expert a large NEGATIVE weight, so subjects should\n    systematically choose AGAINST the 98% endorsement (~0.05-0.10).\n    \"\"\"\n    if data is None or len(data) == 0:\n        return 0.5\n\n    follow_flags = []\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.ndim != 1 or a.shape[0] < 3 or a.shape != b.shape:\n            continue\n\n        d0 = a[0] - b[0]  # 98% expert's relative endorsement of A\n        d1 = a[1] - b[1]  # 90% expert's relative endorsement of A\n        d2 = a[2] - b[2]  # 75% expert must be tied (no interference)\n\n        # Need a genuine 98-vs-90 conflict with the 75% expert tied.\n        if d0 == 0 or d1 == 0 or d2 != 0:\n            continue\n        if np.sign(d0) == np.sign(d1):\n            continue\n\n        endorsed_a = d0 > 0          # 98% expert endorses option A\n        chose_a = (int(row[\"response\"]) == 0)\n        follow_flags.append(1.0 if (endorsed_a == chose_a) else 0.0)\n\n    if len(follow_flags) == 0:\n        return 0.5\n    return float(np.mean(follow_flags))",
  "rationale": "The single sharpest, most parameter-robust dissociation in this design is the D-family: trials where the 98% expert and the 90% expert endorse opposite options while the 75% expert is tied (trials 15, 16, 19, 20; 4 reps each = 16 trials per subject). The metric is the per-subject (and pooled) probability of FOLLOWING the 98% expert's endorsement on exactly those trials. Under FDEI, delta in [0.1,0.2] compresses log-diagnosticities so heavily that w(0.98)/w(0.90) is at most ~1.12; the net evidence on a pure 98-vs-90 conflict is |s| <= ~0.14, and with beta <= 0.37 the predicted follow rate is boxed in [0.47, 0.53] for ANY parameter setting -- a coin flip. Under SITT, w(0.98) is negative (in [-1.14, -0.70]) while w(0.90) is at its trust peak (~0.95-1.06), so the net score against the 98% endorsement is large (|s| >= ~1.65), giving a follow rate <= ~0.08 even at the weakest parameter corner with maximal lapse. The two theories therefore predict ~0.50 vs ~0.06 on this scalar -- a gap of ~0.44 with tight binomial per-subject variance (<= 0.25/16 ~ 0.016 under FDEI, ~0.005 under SITT), so at N=25 the Welch t-test separates them by t ~ 14. The metric is also robust to whether the six weak experts are 0.50 or 0.55: FDEI annihilates 0.50 cues exactly and even with 0.55 cues the D-cell follow rate only rises to ~0.58, while SITT stays below ~0.10 either way, because the weak-cue contributions are small relative to the 98-vs-90 conflict. Unlike a whole-design fit or a crossover-ladder statistic, this single exchange-rate quantity is directly interpretable (it estimates the subjective 98:90 exchange rate) and is deliberately chosen to be insensitive to lapse rate, softmax gain, and weak-cue weighting -- the parameters the two theories share -- so it isolates exactly the structural claim (flat compression vs inverted-U trust with a cliff) on which they disagree."
}
```

## Usage

```json
{
  "input_tokens": 7825,
  "output_tokens": 11548,
  "reasoning_tokens": 10459,
  "provider": "Sail Research"
}
```
