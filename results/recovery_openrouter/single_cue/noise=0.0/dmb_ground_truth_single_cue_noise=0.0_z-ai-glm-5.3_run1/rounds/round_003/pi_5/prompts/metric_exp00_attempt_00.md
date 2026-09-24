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
**Validities (n_features=8):** [0.99, 0.95, 0.9, 0.8, 0.7, 0.99, 0.99, 0.99]

**Trial pairs (n=19):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 13: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 15: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 17: A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 19: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]

**Rationale:** CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x softmax x lapse) and differ ONLY in the weight function's shape across validity. My advocated theory makes three sign-structured claims the competitor cannot mimic: (i) claimed near-certainty is ANTI-diagnostic: w(0.99) is negative (~ -1 trusted-cue unit, box |w(0.99)| in [0.86, 1.24]) while the competitor gives 0.99 its LARGEST positive weight (w in [1.17, 1.38]); (ii) trust peaks at v~0.90 and FALLS by 0.95 (my w(0.95)/w(0.90) ~ 0.5-0.7) while the competitor is monotone (though its delta~0.15 flattening makes 0.95 vs 0.90 nearly a tie, ratio 1.05); (iii) near-chance experts carry a small NEGATIVE weight (w(0.50) = -0.075 to -0.105) while the competitor annihilates v=0.5 EXACTLY (llr=0 for every parameter) and gives v=0.51 up to +0.53 cue units where my theory gives ~0. Validities: [0.99, 0.95, 0.90, 0.80, 0.70, 0.51, 0.50, 0.50] -- one expert in each regime. 19 unique pairs x K = 96//19 = 5 = 95 trials.

FAMILY S/X (super-expert inversion + parameter-free flatness knife). S1 (A wins only the 0.99 expert): my box p(A) in [0.08, 0.20] (center 0.13); competitor box [0.62, 0.73] (center 0.68) -- DISJOINT with a 40-point gap, the flagship cell: I predict subjects choose AGAINST the 99% expert's endorsement, the competitor predicts they follow it. S1m mirrors. X1 (A wins 0.99, B wins one 0.50 cue) and X3 (A wins 0.99 plus one 0.50 cue): the competitor's w(0.50)=0 exactly, so it must predict p(S1) = p(X1) = p(X3) EXACTLY for every (delta, tau, beta, epsilon) -- a parameter-free triple equality. My theory predicts a strict ordering X3 (0.12) < S1 (0.13) < X1 (0.15) at center, with box gaps of 0.05-0.13 per step, because a chance cue won by B is anti-diagnostic evidence FOR A. Observed equality across S1/X1/X3 falsifies my chance-cue anti-weight; any reliable deviation from the triple equality falsifies the competitor outright.

FAMILY P (peak-at-0.90 non-monotonicity). P1 (A wins 0.95, B wins 0.90): my box [0.23, 0.38] (center 0.31 -- the 0.90 expert beats the 0.95 expert); competitor box [0.47, 0.55] (center 0.51 -- its flattened weights make the two nearly exchangeable). Disjoint. P1m mirrors. Note p(P1) > 0.56 would falsify BOTH theories, an honest boundary.

FAMILY D (calibrated cliff depth vs super-expert coalitions). D4 (A wins 0.99+0.90, B wins 0.80): my box [0.10, 0.25] (center 0.17 -- the super-expert's anti-vote nearly cancels the 0.90 endorsement, so a single 0.80 expert wins); competitor box [0.62, 0.87] (center 0.69). Sign-flip cell. D4m mirrors. D1m (A wins 0.90, B wins 0.99): mine [0.93, 0.97] vs competitor [0.42, 0.50] -- the competitor can never exceed chance when a lone 0.99 expert opposes a 0.90 expert, I predict near-certainty. D5 (A wins 0.99+0.90+0.80, B wins 0.70) tests the 'cannot invert coalitions that include it' claim: I follow the 3-vs-1 tally only MODERATELY (center 0.58, box [0.44, 0.71]) while the competitor is confident (center 0.81, box [0.74, 0.94]) -- a confidence dissociation even where choices agree; p(D5) >= 0.85 strains my calibrated cliff depth. D2 (A wins 0.99+0.51, B wins 0.90) vs D1-structure: the competitor predicts D2 > D1 by up to +0.15 (the 0.51 cue is worth up to 0.53 units to it); I predict |D2 - D1| <= 0.02 (w(0.51) ~ 0), and my box on D2 is [0.04, 0.07] vs competitor [0.48, 0.65] -- disjoint.

FAMILY C (chance-cue anti-weight). C2 (A wins both 0.50 cues, nothing else discriminates): the competitor must predict EXACTLY 0.50 (all weights zero -> uniform softmax, lapse uniform) -- parameter-free; I predict 0.42 (box [0.38, 0.43]), i.e., a majority of 50%-accurate endorsements pushes choice BELOW chance. C3 (A wins 0.51+0.50+0.50, a 3-0 cue sweep): my box [0.35, 0.41]; competitor box [0.50, 0.60] -- disjoint, and at center a 20-point sign-direction flip (I choose B despite A winning every discriminating cue). C3m mirrors. Any p > 0.50 on C2/C3 falsifies my anti-diagnostic sign; any reliable p != 0.50 on C2 falsifies the competitor with no free parameters.

FAMILY F/T (delta-flattening vs v_lo annihilation). F1 (A wins 0.80; B wins 0.70+0.51+0.50): the competitor's flattened weights make the 0.51 cue worth over half a 0.80 cue, so its box predicts B (center 0.41, box [0.42, 0.52] for p(A)); my w(0.51)~0 and w(0.50)<0, so I predict A (center 0.64, box [0.57, 0.69]) -- disjoint, center sign flip despite a 1-vs-3 cue count. F1m mirrors. T1 (A wins 0.90; B wins 0.51+0.50+0.50): my box [0.85, 0.92] (the weak stack is anti-diagnostic, amplifying A) vs competitor [0.55, 0.70] -- disjoint; p(T1) < 0.80 is a knife against MY theory's chance-cue distrust magnitude, included deliberately.

ANCHORS. G1/G1m (0.95+0.90+0.80+0.70 vs 0.51+0.50+0.50): both theories choose the trusted stack at high confidence (mine ~0.95, competitor ~0.89), pinning lapse/temperature. PREDICTED PATTERN IF MY THEORY IS RIGHT (center params): S1 .13, X3 .12, S1 .13, X1 .15 (ordered), P1 .31, D4 .17, D1m .93, D5 .58, D2 .04, C2 .42, C3 .38, F1 .64, T1 .88, G1 .95 -- super-expert endorsements followed inverts, chance-cue majorities inverted, 0.90 > 0.95. IF THE COMPETITOR IS RIGHT: S1 = X1 = X3 = .68 exactly, P1 .51, D4 .69, D1m .47, D5 .81, D2 .60, C2 exactly .50, C3 .58, F1 .41, T1 .57, G1 .89 -- super-experts followed, chance cues invisible, near-flat validity gradient. The boxes are disjoint on 11 of 19 cells and every family contains a parameter-free equality or exact-0.50 prediction for the competitor that my theory breaks in a signed, graded, pre-registered direction. Sides are balanced (9 A-tilted / 10 B-tilted under my theory; every family has an A/B mirror); no all-tie trials; every trial requires scanning all 8 experts.

**Computed schedule:** 19 unique pairs × 5 reps = 95 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Super-expert contrarianism index.\n\n    Over all trials in which the feature-0 expert (the 99%-validity\n    expert) discriminates between the two options (a[0] != b[0]),\n    compute the proportion of choices that go AGAINST the option\n    endorsed by that expert.\n\n    Advocated theory (inverted trust): the 99% expert's endorsement is\n    anti-diagnostic (w(0.99) ~ -1 trusted-cue unit), so subjects\n    predominantly choose AGAINST it -> index ~0.8-0.9.\n\n    Competing theory (flattened diagnosticity): the 99% expert gets the\n    LARGEST positive weight, so subjects predominantly FOLLOW it ->\n    index ~0.2-0.35.\n    \"\"\"\n    n_anti = 0\n    n_sel = 0\n    for ra, rb, resp in zip(\n        data[\"option_a_ratings\"], data[\"option_b_ratings\"], data[\"response\"]\n    ):\n        a = np.asarray(ra, dtype=int).ravel()\n        b = np.asarray(rb, dtype=int).ravel()\n        if a.size == 0 or b.size == 0 or a[0] == b[0]:\n            continue  # expert 0 does not discriminate on this trial\n        # Option index endorsed by the 99%-validity (feature-0) expert.\n        endorsed = 0 if a[0] == 1 else 1\n        r = int(resp)\n        n_sel += 1\n        if r != endorsed:\n            n_anti += 1\n    if n_sel == 0:\n        return 0.5\n    return float(n_anti) / float(n_sel)\n",
  "rationale": "The single sharpest, most robust knife between the two theories in this design is the sign of the weight on the 99%-validity expert (feature 0). My advocated theory assigns w(0.99) ~ -1 trusted-cue unit (anti-diagnostic endorsement), while the competitor assigns it its largest positive weight (~1.16-1.36 cue units). Feature 0 discriminates on 9 of the 19 unique trial types (trials 1, 2, 3, 4, 7, 8, 15, 16, 17), i.e. 45 of the 95 trials per subject, so the metric averages over a large, pre-registered, sign-consistent family rather than a single cell.\n\nPredicted values (center parameters, lapse included):\n- Advocated theory: per-trial anti-follow rates ~0.87 (S1/S1m, trials 1-2), ~0.85-0.90 (trials 3-4, 7-8), ~0.93 (trial 15, the 0.90-vs-0.99 cell), ~0.96 (trial 17), ~0.42 (trial 16) -> pooled index ~0.84, with per-subject SD ~0.05-0.07 (45 Bernoulli trials plus modest parameter spread; the parameter boxes keep every S1-type anti-follow rate inside [0.79, 0.90]).\n- Competing theory: anti-follow ~0.19-0.31 on the S1/D-type cells, ~0.40-0.52 on the near-tie cells -> pooled index ~0.33, per-subject SD ~0.07.\n\nThe mean gap (~0.5) is enormous relative to the between-subject spread, so Welch's t-test at N=25 separates the theories by many standard errors (t > 20 expected). The metric is a plain behavioral proportion (no free parameters, no fitting to either model), it is well defined both on the pooled DataFrame and on any single subject's slice (each subject sees all 19 trial types x 5 reps, so ~45 qualifying trials per subject), and it is robust to the exact validities of features 5-7 because it depends only on feature 0, whose validity (0.99) is fixed by the design. Under my theory the index is high (subjects are contrarians toward the near-certain expert); under the competitor it is low (subjects follow the near-certain expert). A value near 0.5 would be evidence against both."
}
```

## Usage

```json
{
  "input_tokens": 7619,
  "output_tokens": 14901,
  "reasoning_tokens": 13926,
  "provider": "Sail Research"
}
```
