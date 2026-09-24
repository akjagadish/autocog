# metric_exp01_attempt_00

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
**Validities (n_features=12):** [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
  trial 14: A=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
  trial 18: A=[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]
  trial 20: A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

**Rationale:** STRUCTURAL SETUP. GSP and RSD-vgp share the identical anti-endorsement core: p(A) = 0.5 + (1-eps)(2m-1)(sigmoid(beta*E) - 0.5) with E = sum_j v_j^gamma (b_j - a_j), RAW weights, and nearly identical m ([0.75,1] vs [0.74,1]) and epsilon boxes. RSD's two-component gamma mixture is algebraically exactly Uniform[0,2] (both components have density 0.5). The theories are therefore the SAME model with different parameter SUPPORTS, and only three regions can ever dissociate them: (i) gamma in (1.5, 2] -- 25% of RSD subjects, 0% of GSP; (ii) beta in [1.1, 1.8) -- 46.1% of GSP subjects (ln(1.8/1.1)/ln(3.2/1.1) = 0.461), 0% of RSD; (iii) beta in (3.2, 6.0] -- 52.3% of RSD subjects, 0% of GSP. This design makes all three boundaries decisive using a NEW architecture built on EXACT-UNIT cues, unlike prior designs whose gamma-invariant probes relied on 0.99-validity cues (2% weight drift) and whose beta readout required A_eff normalization via a saturated anchor.

KEY ALGEBRAIC LEVER (new, exact, parameter-free). For BOTH theories, on EVERY trial, p - 0.5 = A_eff * (sigmoid(beta*E) - 0.5) exactly, where A_eff = (1-eps)(2m-1) > 0. Therefore the RATIO of choice deviations across cells with the same evidence sign is exactly A_eff-free AND lapse-free: r12 = (p_{E=2} - 0.5)/(p_{E=1} - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5), a pure function of beta alone. I construct cells with EXACTLY integer evidence E = 1, 2, 3 using cues of validity 1.00 (weight = 1 for every gamma in [0,2], perfectly gamma-invariant) plus exactly-canceling shared endorsements, so E is an exact integer for every subject of either theory at every gamma.

ARCHITECTURE (12 experts): three 1.00 cues (exact unit weights, gamma-invariant), then 0.50 x3, 0.55 x2, 0.60 x2, 0.65, 0.70. S(gamma) ranges over [5.99, 12], so the unanimous cell saturates every subject (beta*S >= 1.1*5.99 = 6.59, sigmoid >= 0.9986), reading out A_eff = 2*(p_U - 0.5).

DISSOCIATION 1 -- BETA SUPPORT VIA THE EXACT RATIO LADDER (headline). Cells B1/B2/B3 differ only on unit cues with exactly canceling shared endorsements: E = -1, -2, -3 exactly, for every gamma. Per-subject r12 = (sigmoid(2b)-0.5)/(sigmoid(b)-0.5): GSP support [1.082, 1.604] with 46.1% of subjects ABOVE 1.3216 (all beta < 1.8); RSD support [1.005, 1.3216] with a HARD ceiling at 1.3216 (beta >= 1.8) and 52.3% at or below 1.082 (beta >= 3.2). The regions are DISJOINT on both tails: any subject with r12 > 1.33 categorically falsifies RSD while GSP requires ~46% of subjects there; conversely any subject with r12 <= 1.082 falsifies GSP while RSD requires ~52% there. The corroborating ratio r13 = (sigmoid(3b)-0.5)/(sigmoid(b)-0.5) gives an independent second readout (GSP [1.085, 1.860], 46% above 1.384; RSD ceiling 1.384). No prior design exploited this exact cancellation-based ratio signature.

DISSOCIATION 2 -- GAMMA CEILING VIA THE FLIP LADDER. Flip cells pit the exact unit pair (A endorsed on two 1.00 cues, weight exactly 2 for all gamma) against low-validity blocks whose total weight W(gamma) crosses 2 at a designed flip point g*: E(gamma) = W(gamma) - 2, strictly decreasing. Flip points: F0 = 0.585 (3x0.50), F1 = 1.31 (0.50, 0.55, 0.60, 0.70), F2 = 1.47 (0.55, 0.60, 0.65, 0.70), F3 = 1.65 (0.50, 0.50, 0.55, 0.60, 0.70), F4 = 1.91 (0.50 x3, 0.55, 0.60, 0.70). Because m >= 0.74 > 0.5 makes p >= 0.5 whenever E >= 0, and E_F3(1.5) = +0.165 > 0 and E_F4(1.5) = +0.519 > 0, NO GSP subject (gamma <= 1.5) can truly prefer the low-validity-denser side on F3/F4 -- a hard floor at exactly 50%. RSD REQUIRES (2-g*)/2 = 17.5% of subjects below 0.5 on F3 and 4.5% on F4. Predicted population inversion rates (fraction of subjects with true p < 0.5): F0: 0.610 (GSP) / 0.707 (RSD); F1: 0.127 / 0.345; F2: 0.020 / 0.265; F3: 0.000 / 0.175; F4: 0.000 / 0.045. The inversion-rate curve vs flip point is piecewise-linear falling to zero AT 1.5 with a hard edge (GSP) vs continuing linearly to 2.0 (RSD); F2/F3 tightly bracket the 1.5 edge. Within-subject, an RSD inverter of F3 necessarily inverts F2, F1, F0 (monotone in gamma), and per-subject gamma estimates fill (1.5, 2.0] with 25% mass under RSD vs a hard edge at 1.5 under GSP. Inversion depth: on F1 an RSD subject at gamma=2, beta=6 has E = -0.60, p(sparse side) ~ 0.17; on F3 at gamma=2, p ~ 0.23.

CONTROLS. All beta-axis cells are CONFLICT-STYLE (both options carry endorsements; shared endorsements on identical validities cancel exactly), ruling out any 'pick the empty bar' heuristic; Uc (9 endorsements vs 3, |E| >= 3.44) is a near-saturated conflict twin of the unanimous cell, so p_U ~ p_Uc verifies that the A_eff anchor is not an empty-bar artifact. C2 is a 2-2 COUNT TIE (A endorsed on the two unit cues, B on two 0.50 cues): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (GSP mid-box p ~ 0.74, RSD ~ 0.82) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2 are identical-vector ties where both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by sparse-option side (ladder, U), unit-cue side (B-cells), or low-validity side (C2).

POOLED MID-BOX PREDICTIONS (GSP at gamma=0.75, beta=2.0, A_eff=0.70 vs RSD at gamma=1.0, beta=3.0, A_eff=0.70), p(sparse side): U 0.85/0.85; B1 0.77/0.82; B2 0.84/0.85; B3 0.85/0.85; F0 0.42/0.28; F1 0.71/0.67; F2 0.73/0.74; F3 0.75/0.75; F4 0.76/0.76; C2 0.74/0.82; ties 0.50/0.50. Pooled differences are modest (largest: F0 ~ 0.14, C2 ~ 0.08, B1 ~ 0.05) -- honestly acknowledged: the arbiter is SUBJECT-LEVEL population structure, requiring N >= 60-80 and hierarchical Bayesian fitting (per-subject gamma, beta, A_eff with population priors given by each theory, plus simulation-based model recovery), as in the strongest prior designs.

DECISIVE QUANTITATIVE SIGNATURES: (a) per-subject exact ratio r12 > 1.33 (equivalently hierarchical beta-hat < 1.8): ~46% of subjects (GSP) vs 0% hard (RSD); (b) r12 <= 1.082 (beta-hat >= 3.2): ~52% (RSD) vs ~0% (GSP); (c) any single subject with pooled F3 (or F4) rate credibly below 0.5 categorically falsifies GSP (E_F3(1.5) = +0.165 > 0 makes p >= 0.5 for every GSP parameter), while RSD requires 17.5% of subjects below 0.5 on F3; (d) inversion-rate survival curve over flip points 0.585/1.31/1.47/1.65/1.91: hard zero past 1.5 with a kink (GSP) vs linear decline to 2.0 with 25% of gamma-hat mass in (1.5, 2] (RSD); (e) within-subject monotone flip pattern (inverting F3 implies inverting F2, F1, F0); (f) per-subject 2D (gamma-hat, beta-hat) histogram with mass split across the three exclusive support regions [beta<1.8] vs [gamma>1.5] vs [beta>3.2]; (g) C2 low-side preference > 50% and ties at exactly 50% (shared controls, the former ruling out counting).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Graded Sparse-Preference with Heterogeneous Attenuation (GSP), unnormalized-evidence variant. Subjects integrate ALL cues additively with near-flat validity weighting under an anti-endorsement (sparse-preferring) reading: the evidence for option A over B is E = sum_j v_j^gamma (b_j - a_j), where each endorsement (rating = 1) carried by an option counts AGAINST it, so the sparser option accumulates evidence. The cue weights are used RAW (v_j^gamma, unnormalized) so the evidence scale preserves the ABSOLUTE endorsement-count margin of the trial independent of the experiment's feature count — the cross-experiment scale separation (moderate pooled preference on 5-cue shallow-margin trials vs strong preference on 8-cue margin-2 conflicts) that per-experiment normalization provably erases. The anti-endorsement reading is graded per subject: each subject carries a continuous polarity weight m in [0.75, 1.0] (per-trial probability of responding with the inverted polarity), a log-uniformly dispersed inverse temperature beta, and a lapse epsilon in [0.02, 0.12]: p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with identical rating vectors yielding exactly 50%. The population is a continuum of attenuations concentrated on the inverted side (no empty middle, no mirror cluster), with mean effective attenuation (1-eps)(2m-1) ~ 0.70 matching the observed extremity anchors.

**Parameters:**
- gamma: [0, 1.5]
- log_beta: [0.095, 1.163]
- epsilon: [0.02, 0.12]
- m: [0.75, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Graded Sparse-Preference with Heterogeneous Attenuation (GSP),
    # UNNORMALIZED-evidence variant.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GSP expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    # log-uniform inverse temperature: beta = exp(log_beta),
    # log_beta ~ U[ln 1.1, ln 3.2]  =>  beta log-uniform over [1.1, 3.2],
    # calibrated to the RAW (unnormalized) evidence scale below.
    # Lower bound raised from 0.8 to 1.1 (the interpolation lever);
    # upper bound capped at 3.2 (guardrail for Exp4/Exp6 overshoot).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization by
    # sum(w). This preserves the absolute endorsement-margin scale:
    # the evidence magnitude reflects the raw count margin of the
    # trial (a margin-2 conflict carries twice the evidence of a
    # margin-1 conflict) independent of n_features.
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. The sparser option accumulates
    # evidence. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic of the inverted evidence.
    # (both branches exponentiate a non-positive argument, so no
    # overflow is possible)
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED polarity (sparse-preferring), with probability
    # (1 - m) with the conventional polarity. This is the expectation of
    # a per-trial Bernoulli polarity draw against a continuous,
    # strongly-inverted per-subject weight m in [0.75, 1.0].
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp). When choosing between two products described by binary expert ratings, subjects read an endorsement (rating = 1) as evidence AGAINST the option carrying it, so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. The population is a continuum of graded polarity (no bimodal trait clusters): each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, and — the new discriminative dimension — validity-gated precision: a per-subject validity exponent gamma dispersed over [0, 2] with a substantial steep-gamma subpopulation, and a steep log-uniformly dispersed raw inverse temperature beta over [1.8, 6.0], calibrated so that essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) exceeds 9.5. The resulting subject family is 'steep slope x moderate asymptote': p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69. This decomposition is what the data jointly demand: the slope-sensitive diagnostics (psychometric slope posterior, evidence-usage slopes, mixture log-likelihood ratios) require steep normalized slopes, while the extremity/consistency diagnostics (saturated sparse-choice rates) require moderate asymptotic preference — a combination no single shallow-slope, high-extremity or steep-slope, high-extremity parameterization can produce.

**Parameters:**
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RSD-vgp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component mixture ----
    # gamma_u ~ U[0, 1]. With probability 0.65 the subject draws gamma
    # uniformly from [0, 1.3] (near-flat validity weighting); with
    # probability 0.35 from [1.3, 2.0] (the steep-gating subpopulation
    # that validity-conflict cells discriminate).
    u = float(parameters["gamma_u"])
    if u < 0.65:
        gamma = 1.3 * (u / 0.65)
    else:
        gamma = 1.3 + 0.7 * ((u - 0.65) / 0.35)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 1.8, ln 6.0]  =>  beta
    # log-uniform over [1.8, 6.0]. Calibrated against the RAW evidence
    # scale below so that every subject's effective normalized slope
    # beta * sum(v^gamma) comfortably exceeds 9.5 on any plausible
    # validity vector (min ~ 1.8 * 7.4 ~ 13).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
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
[0] rationale: The two theories share an identical likelihood family p(A) = (1-eps)[m*sigmoid(beta*E) + (1-m)(1-sigmoid(beta*E))] + eps/2 with E = sum_j v_j^gamma (b_j - a_j); they differ ONLY in the population prior over (gamma, log beta, eps, m). Hence single-cell choice rates confound the shared likelihood with heterogeneous subject parameters (A_eff, gamma, beta), which is why any pooled-rate contrast is small relative to between-subject spread. The Neyman-Pearson-optimal discriminator of two mixtures over the same likelihood family is the per-subject marginal likelihood ratio, so I compute exactly that: for each subject, the likelihood of their 96 choices is marginalized by grid quadrature over each theory's exact prior (GSP: gamma~U[0,1.5], log_beta~U[ln1.1,ln3.2], eps~U[0.02,0.12], m~U[0.75,1]; RSD: gamma~U[0,2] — its two-component mixture is algebraically uniform — log_beta~U[ln1.8,ln6.0], eps~U[0.03,0.11], m~U[0.74,1]), and the metric is the mean log evidence ratio (log Bayes factor). This is a symmetric, pre-registered two-model comparison: it returns large positive values if GSP generated the data and large negative values if RSD did — it assumes neither is true. The design makes it powerful: the gamma-invariant 0.99-probe ladder, the flip ladder (k=3 vs top pair), the C2 count-tie, and the saturated U cells jointly pin down each subject's (beta, gamma, A_eff), so subjects whose parameters fall outside a theory's support are heavily penalized. GSP places 46% of its mass at beta < 1.8 (plus 10% at eps in (0.11,0.12]) where RSD has zero density, and RSD places ~64% of its mass in the union {beta > 3.2} (52%), {gamma > 1.5} (25%) where GSP has zero density; each such subject contributes roughly +0.5 to +2 (GSP-only region) or -0.5 to -2 (RSD-only region) nats, while overlap subjects cluster near the log prior-density ratio (~+0.2). Expected per-subject means are approximately +0.5 (GSP) vs -0.7 (RSD) with between-subject SD ~0.5-0.8, giving Welch t ~ 4-6 at N=25 — comfortably below alpha=0.01. The metric is well-defined on any single subject's slice (their personal log Bayes factor) and on the pooled frame (mean over subjects), so the between-subject variance is always computable.
metric_source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Per-subject log marginal-likelihood ratio (log Bayes factor)
    # GSP vs RSD-vgp, averaged over subjects.
    #
    # Both theories share the SAME likelihood family:
    #   p(A) = (1-eps)*[ m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E)) ] + eps/2
    #   E = sum_j v_j^gamma (b_j - a_j)   (raw, anti-endorsement)
    # They differ ONLY in the population prior over (gamma, log beta, eps, m):
    #   GSP: gamma~U[0,1.5], log_beta~U[ln1.1,ln3.2], eps~U[0.02,0.12], m~U[0.75,1]
    #   RSD: gamma~U[0,2] (the 2-component mixture is exactly uniform),
    #        log_beta~U[ln1.8,ln6.0], eps~U[0.03,0.11], m~U[0.74,1]
    # So the optimal discriminator of simulated populations is the per-subject
    # evidence ratio, computed by exact grid marginalization of the likelihood
    # under each theory's prior. Positive = favors GSP, negative = favors RSD.

    VALS = np.array([1.0, 0.99, 0.99, 0.99, 0.85, 0.8,
                     0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])

    def log_evidence(D, kA, kB, gam, lnb, eps, mgr):
        # D: (T, F) evidence direction (b - a) per unique trial
        # kA, kB: (T,) counts of A-choices and B-choices per unique trial
        W = np.power(VALS[None, :], gam[:, None])            # (G, F) weights
        Ev = D @ W.T                                        # (T, G) evidence
        B = np.exp(lnb)                                     # (B,) betas
        x = Ev[:, :, None] * B[None, None, :]               # (T, G, B)
        sig = 0.5 * (1.0 + np.tanh(0.5 * x))                # stable logistic
        s = sig.reshape(D.shape[0], -1)                      # (T, G*B)
        chunks = []
        for e in eps:
            for mm in mgr:
                c1 = (1.0 - e) * (2.0 * mm - 1.0)
                c0 = (1.0 - e) * (1.0 - mm) + 0.5 * e
                p = c0 + c1 * s                             # p(A) on each trial
                ll = np.sum(kA[:, None] * np.log(p)
                            + kB[:, None] * np.log1p(-p), axis=0)
                chunks.append(ll)
        LL = np.concatenate(chunks)                         # (G*B*E*M,)
        mx = float(np.max(LL))
        # equal grid weights in the (gamma, log beta, eps, m) measure
        return mx + float(np.log(np.mean(np.exp(LL - mx))))

    def subject_llr(sub):
        # compress the 96 trials into unique (A,B) pairs with choice counts
        agg = {}
        for a, b, r in zip(sub["option_a_ratings"].tolist(),
                           sub["option_b_ratings"].tolist(),
                           sub["response"].tolist()):
            ka = tuple((int(v) for v in a))
            kb = tuple((int(v) for v in b))
            key = (ka, kb)
            if key not in agg:
                agg[key] = [0, 0]
            agg[key][0] += 1
            if int(r) == 0:          # response 0 == chose A
                agg[key][1] += 1
        D = []
        nA = []
        nB = []
        for (ka, kb), (n, na) in agg.items():
            D.append(np.asarray(kb, dtype=float) - np.asarray(ka, dtype=float))
            nA.append(float(na))
            nB.append(float(n - na))
        D = np.vstack(D)
        kA = np.array(nA)
        kB = np.array(nB)

        # GSP prior
        lg = log_evidence(D, kA, kB,
                          np.linspace(0.0, 1.5, 19),
                          np.linspace(np.log(1.1), np.log(3.2), 19),
                          np.linspace(0.02, 0.12, 5),
                          np.linspace(0.75, 1.0, 6))
        # RSD-vgp prior (gamma mixture is exactly U[0,2])
        lr = log_evidence(D, kA, kB,
                          np.linspace(0.0, 2.0, 25),
                          np.linspace(np.log(1.8), np.log(6.0), 19),
                          np.linspace(0.03, 0.11, 5),
                          np.linspace(0.74, 1.0, 6))
        return lg - lr

    data = data.reset_index(drop=True)
    llrs = []
    for _, sub in data.groupby("subject_id", sort=False):
        llrs.append(subject_llr(sub))
    if len(llrs) == 0:
        return 0.0
    llrs = np.clip(np.array(llrs, dtype=float), -6.0, 6.0)
    return float(np.mean(llrs))
outcome: self_sim=0.1970 (var=0.2752) adversary_sim=-0.1509 (var=0.6258) welch_t=+1.833 p=0.07402 (N=25, alpha=0.01) -> reject

[1] rationale: WHY THE PREVIOUS METRIC FAILED: the log Bayes factor is the sufficient statistic for classification, but on the log-odds scale its between-subject SD (0.52-0.79) dwarfed the mean gap (0.35): weakly identified subjects produce heavy-tailed log-ratios that inflate variance without adding signal, giving t=1.83. The fix is to keep the same Bayesian machinery but (a) reduce it to the genuinely discriminative marginal and (b) map it onto a BOUNDED scale so noise cannot blow up the variance. KEY REDUCTION: both theories share the exact likelihood p(A) = 0.5 + A_eff*(sigmoid(beta*E)-0.5) with E = sum_j v_j^gamma (b_j - a_j) and A_eff = (1-eps)(2m-1) (eps and m enter ONLY through A_eff -- algebraically exact), so each subject's 96 choices identify just (ln beta, gamma, A_eff), and the theories differ only in the prior over this 3-tuple. The decisive prior difference is the NON-OVERLAPPING beta supports: GSP draws ln beta ~ U[ln1.1, ln3.2] (46% of its mass below ln 1.8, where RSD has zero density), RSD draws ln beta ~ U[ln1.8, ln6.0] (52% of its mass above ln 3.2, where GSP has zero density). THE METRIC: per subject, compute the exact grid posterior over (ln beta, gamma, A_eff) under a prior that is the equal-weight MIXTURE of the two theories' parameter boxes (perfectly symmetric -- neither theory is favored; weakly identified subjects of either population fall back to the same near-zero baseline). The per-subject score is posterior mass in GSP's exclusive low-beta region minus mass in RSD's exclusive high-beta region, bounded in [-1, 1]. WHY IT DISCRIMINATES: GSP's low-beta subpopulation (46%) is precisely the WELL-identified one -- low beta puts the 0.99-probe, P85/P75 and F2/F3 cells on the mid-slope of the psychometric function (Fisher information for ln beta ~ 8-14), so their posteriors concentrate below ln 1.8 and they score ~ +0.5 to +0.8. RSD's high-beta subpopulation (52%) is pushed back below ln 3.2 by the low-validity cells (F2: |E| up to 0.87, F3, P75, P85), which even a beta=6 subject cannot fit with beta <= 3.2 (log-likelihood penalty ~1.5-3 nats), so they score ~ -0.45 to -0.7. Overlap-beta subjects of both populations score near 0 and cancel. Expected: GSP ~ +0.3 (SD ~0.33), RSD ~ -0.26 (SD ~0.30), Welch t ~ 5-6 at N=25, p < 1e-5. ROBUSTNESS: because the score is a bounded posterior probability rather than an unbounded log-ratio, it degrades gracefully under misspecified noise -- even if the true posterior width were twice my Fisher-information estimate (sigma ~0.9 instead of ~0.5), the mean gap only shrinks to ~0.4 with SD ~0.25 (t ~ 5); if posteriors were twice as sharp as estimated the gap grows to ~0.85. The contrast fails only if subjects' beta were completely unidentifiable, which the design's mid-slope cells preclude. The metric works identically on a single subject's slice (their personal score) and on the pooled frame (mean score), so the between-subject variance is always computable, and it is symmetric by construction: it returns large positive values iff the population contains GSP-only low-beta subjects, large negative values iff it contains RSD-only high-beta subjects -- exactly the population-support difference the experiment was built to arbitrate.
metric_source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # ------------------------------------------------------------------
    # Posterior mass in the beta-support region EXCLUSIVE to GSP
    # (raw inverse temperature beta in [1.1, 1.8)) minus posterior mass
    # in the region EXCLUSIVE to RSD-vgp (beta in (3.2, 6.0]), averaged
    # over subjects.
    #
    # Both theories share the exact per-trial likelihood
    #     p(A) = 0.5 + A_eff * ( sigmoid(beta * E) - 0.5 ),
    #     E = sum_j v_j^gamma (b_j - a_j),   A_eff = (1-eps)(2m-1),
    # so a subject's 96 choices only identify (ln beta, gamma, A_eff),
    # and the two theories differ ONLY in the population prior over
    # these parameters. The decisive, non-overlapping prior support is:
    #     GSP-only:  ln beta in [ln 1.1, ln 1.8)   (46% of GSP mass, 0% RSD)
    #     RSD-only:  ln beta in (ln 3.2, ln 6.0]   (52% of RSD mass, 0% GSP)
    # For each subject we compute the Bayesian posterior over
    # (ln beta, gamma, A_eff) under a SYMMETRIC prior (equal mixture of
    # the two theories' parameter boxes -- no theory is favored by
    # construction), and report the difference between the posterior
    # mass falling in GSP's exclusive low-beta region and the mass
    # falling in RSD's exclusive high-beta region. Bounded in [-1, 1].
    #
    # Expected values: ~ +0.25 to +0.35 under GSP (the 46% low-beta
    # subpopulation is well identified by the gamma-invariant 0.99-probe
    # ladder and mid-slope flip cells), ~ -0.25 under RSD (its 52%
    # high-beta subpopulation is pinned by the low-validity probe/flip
    # cells F2/F3/P75/P85, which even beta=6 subjects cannot fit with
    # beta <= 3.2).
    # ------------------------------------------------------------------

    VALS = np.array([1.0, 0.99, 0.99, 0.99, 0.85, 0.80] + [0.75] * 7)

    C_LO = np.log(1.8)   # edge: GSP-exclusive  <->  shared beta support
    C_HI = np.log(3.2)   # edge: shared support  <->  RSD-exclusive beta

    n_b, n_g, n_a = 57, 21, 13
    lnb = np.linspace(np.log(1.1), np.log(6.0), n_b)   # union support
    gam = np.linspace(0.0, 2.0, n_g)                    # union support
    ave = np.linspace(0.42, 0.98, n_a)                  # union A_eff
    betas = np.exp(lnb)

    # ---- symmetric mixture prior: 0.5 * GSP box + 0.5 * RSD box -----
    tb = 0.5 * (lnb[1] - lnb[0])
    tg = 0.5 * (gam[1] - gam[0])
    ta = 0.5 * (ave[1] - ave[0])
    gsp_mask = (
        (lnb <= np.log(3.2) + tb)[:, None, None]
        & (gam <= 1.5 + tg)[None, :, None]
        & (ave >= 0.44 - ta)[None, None, :]
    )
    rsd_mask = (
        (lnb >= np.log(1.8) - tb)[:, None, None]
        & (ave >= 0.4272 - ta)[None, None, :]
    )
    n_gsp = float(gsp_mask.sum())
    n_rsd = float(rsd_mask.sum())
    prior = 0.5 * gsp_mask / n_gsp + 0.5 * rsd_mask / n_rsd  # (b, g, a)

    # validity weights per gamma
    W = np.power(VALS[None, :], gam[:, None])  # (n_g, F)

    def subject_score(sub: pd.DataFrame) -> float:
        # aggregate the subject's trials into unique (A, B) pairs
        agg = {}
        for a, b, r in zip(
            sub["option_a_ratings"].tolist(),
            sub["option_b_ratings"].tolist(),
            sub["response"].tolist(),
        ):
            key = (
                tuple((int(v) for v in a)),
                tuple((int(v) for v in b)),
            )
            if key in agg:
                cnt = agg[key]
            else:
                cnt = agg[key] = [0, 0]
            cnt[0] += 1
            if int(r) == 0:      # response 0 == chose A
                cnt[1] += 1
        keys = list(agg.keys())
        D = np.array(
            [np.asarray(kb, dtype=float) - np.asarray(ka, dtype=float)
             for (ka, kb) in keys],
            dtype=float,
        )                                   # (T, F) evidence direction
        kA = np.array([agg[k][1] for k in keys], dtype=float)
        kB = np.array([agg[k][0] - agg[k][1] for k in keys], dtype=float)

        E = D @ W.T                              # (T, n_g)
        x = E[:, :, None] * betas[None, None, :]  # (T, n_g, n_b)
        sig = 0.5 * (1.0 + np.tanh(0.5 * x))      # stable logistic

        LL = np.empty((n_b, n_g, n_a))
        for ia in range(n_a):
            p = 0.5 + ave[ia] * (sig - 0.5)       # (T, n_g, n_b)
            p = np.clip(p, 1e-9, 1.0 - 1e-9)
            ll = (kA[:, None, None] * np.log(p)
                  + kB[:, None, None] * np.log1p(-p))
            LL[:, :, ia] = ll.sum(axis=0).T        # (n_b, n_g)

        LL = LL.reshape(n_b, -1)
        w = prior.reshape(n_b, -1) * np.exp(LL - LL.max())
        tot = w.sum()
        if not np.isfinite(tot) or tot <= 0.0:
            return 0.0
        w = w.sum(axis=1) / tot                   # marginal over (gamma, A)
        return float(w[lnb < C_LO].sum() - w[lnb > C_HI].sum())

    data = data.reset_index(drop=True)
    scores = [
        subject_score(sub)
        for _, sub in data.groupby("subject_id", sort=False)
    ]
    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))
outcome: self_sim=-0.3373 (var=0.0348) adversary_sim=-0.4150 (var=0.0691) welch_t=+1.205 p=0.2347 (N=25, alpha=0.01) -> reject

[2] rationale: WHY THE PREVIOUS ATTEMPTS FAILED AND WHAT CHANGES. Both theories share the identical likelihood family p(A) = (1-eps)[m*sigmoid(beta*E) + (1-m)(1-sigmoid(beta*E))] + eps/2 with E = sum_j v_j^gamma (b_j - a_j); they differ only in the population prior over (gamma, log beta, eps, m). Attempt [0] (raw per-subject log Bayes factor) had the right direction but SD 0.52/0.79 vs a gap of only 0.35; attempt [1] (posterior mass in exclusive beta regions) collapsed to means -0.34/-0.42 -- a sign pattern impossible under correct specification, which pinpoints the real culprit: both metrics assumed low-expert validities of 0.75x6, while the flip-ladder architecture (verified: the stated flip points t* = 0.685/1.17/1.545/1.855 reproduce exactly only with 0.55-valid low cues and T = 1 + 0.99^gamma, and the design reserves experts 8-13 -- exactly six positions -- for the ladder while expert 7 serves the P75 probe) indicates the simulated validities are [1.0, 0.99, 0.99, 0.99, 0.85, 0.80, 0.75, 0.55x6]. With 0.75 assumed, every flip trial's evidence is badly wrong, so GSP subjects with gamma in (0.685, 1.5] (54% of the population) appear to flip F3/F4 in ways no grid parameter can explain, scrambling the posterior. THE METRIC. Per subject, compute the exact grid marginal likelihood of their 96 choices under each theory's prior box (GSP: gamma U[0,1.5], ln beta U[ln1.1,ln3.2], eps U[0.02,0.12], m U[0.75,1]; RSD: gamma U[0,2] -- its two-component mixture is algebraically uniform -- ln beta U[ln1.8,ln6], eps U[0.03,0.11], m U[0.74,1]), with the ambiguous low validity of experts 8-13 marginalized over {0.55, 0.65, 0.75} identically in both marginals (a shared nuisance prior that cancels in the ratio -- the statistic adapts to whichever value generated the data, eliminating the misspecification that sank both prior attempts). The per-subject score is tanh(log Bayes factor), bounded in (-1, 1). WHY THE GAP IS LARGE AND THE VARIANCE SMALL. The unbounded log BF failed on variance, not signal: RSD's strongly-identified subjects produce log-BF tails of -3 to -7 (RSD variance 0.63). tanh maps those to ~-1, capping the between-subject variance at O(1) while leaving moderate evidence nearly linear. The mean separation is driven by three uncompensable support differences: (i) RSD's gamma in (1.5, 2] subpopulation (25%) inverts the k=5/k=6 flip cells (E_5 = 5*0.55^gamma - (1+0.99^gamma) < 0), which NO GSP parameter can fit (E_5 >= +0.058 for all gamma <= 1.5) -- worth 1-7 nats per such subject; (ii) RSD's beta in (3.2, 6] subpopulation (52%) is forced above GSP's ceiling on the mid-slope k=2/k=3 cells (~-0.35 nats each); (iii) GSP's beta in [1.1, 1.8) subpopulation (46%) sits below RSD's floor, worth ~+0.3-0.6 nats. Overlap subjects of both populations contribute the same log prior-volume ratio (+0.22), so they shift both means equally without touching the gap. Expected per-subject tanh scores: ~ +0.3 under GSP (SD ~0.3, noise-dominated) vs ~ -0.35 under RSD (SD ~0.45), giving Welch t ~ 5-7 at N = 25. The metric is symmetric (it rewards whichever population generated the data), well-defined on any single subject's slice (their personal bounded log Bayes factor), so the between-subject variance is always computable, and it is not trivially true for either theory -- it is the Neyman-Pearson-optimal discriminator of the two mixtures, evaluated with the correct stimulus model.
metric_source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # ------------------------------------------------------------------
    # Per-subject BOUNDED log Bayes factor, GSP vs RSD-vgp:
    #     score_i = tanh( log m_GSP(data_i) - log m_RSD(data_i) )
    #     metric  = mean_i score_i
    #
    # Both theories share the exact per-trial likelihood
    #     p(A) = (1-eps)*[m*s + (1-m)*(1-s)] + eps/2,  s = sigmoid(beta*E)
    #     E   = sum_j v_j^gamma * (b_j - a_j)   (raw anti-endorsement)
    # and differ ONLY in the population prior over (gamma, ln beta, eps, m):
    #     GSP: gamma~U[0,1.5], ln beta~U[ln1.1,ln3.2], eps~U[0.02,0.12], m~U[0.75,1]
    #     RSD: gamma~U[0,2]  , ln beta~U[ln1.8,ln6.0], eps~U[0.03,0.11], m~U[0.74,1]
    # Each marginal is computed by grid quadrature over the theory's exact
    # box (uniform grid = uniform prior).
    #
    # TWO FIXES vs the previously failed attempts:
    # (1) The low validity of experts 8-13 is a design ambiguity (0.55x6
    #     vs 0.75x6). It is now marginalized over {0.55, 0.65, 0.75}
    #     IDENTICALLY in both marginals -- a shared nuisance prior that
    #     cancels in the ratio, so the statistic adapts to whichever
    #     value generated the data instead of being misspecified on the
    #     48 flip-ladder trials.
    # (2) The per-subject log Bayes factor is mapped through tanh(),
    #     bounding every contribution to (-1, 1). The unbounded version
    #     failed because a few strongly-identified subjects produced
    #     log-BF tails of -3..-6 that inflated between-subject variance
    #     (0.63 on the adversary side) without adding proportional
    #     mean separation; tanh compresses exactly those tails while
    #     leaving moderate evidence nearly linear.
    #
    # Why the gap is large: the likelihood family is shared, so the Bayes
    # factor is the optimal discriminator of the two mixtures. GSP places
    # 46% of its mass at beta < 1.8 (RSD: 0%); RSD places 25% of its mass
    # at gamma in (1.5, 2] (GSP: 0%), and those subjects INVERT the k=5/k=6
    # flip cells (E_5 = 5*0.55^gamma - (1+0.99^gamma) < 0), which no
    # parameter in GSP's box can reproduce (E_5 >= +0.058 for all
    # gamma <= 1.5) -- an uncompensable misfit worth several nats. GSP
    # subjects, in contrast, are either inside RSD's box (contribute the
    # log prior-volume ratio, +0.22) or at beta in [1.1,1.8) where RSD's
    # beta floor forces only mild, partially compensable misfit.
    # ------------------------------------------------------------------

    V_HI = np.array([1.0, 0.99, 0.99, 0.99, 0.85, 0.80, 0.75])  # experts 1-7
    V_LOS = np.array([0.55, 0.65, 0.75])                        # experts 8-13

    GRID = {
        "G": (np.linspace(0.0, 1.5, 21),
              np.linspace(np.log(1.1), np.log(3.2), 17),
              np.linspace(0.02, 0.12, 4),
              np.linspace(0.75, 1.0, 6)),
        "R": (np.linspace(0.0, 2.0, 31),
              np.linspace(np.log(1.8), np.log(6.0), 17),
              np.linspace(0.03, 0.11, 4),
              np.linspace(0.74, 1.0, 6)),
    }

    def log_marginal(D, kA, kB, key):
        gam, lnb, eps_g, m_g = GRID[key]
        # (c0, c1) coefficients for every (eps, m) grid point:
        # p(A) = c0 + c1 * sigmoid(beta * E)
        E_, M_ = np.meshgrid(eps_g, m_g, indexing="ij")
        c0 = ((1.0 - E_) * (1.0 - M_) + 0.5 * E_).ravel()
        c1 = ((1.0 - E_) * (2.0 * M_ - 1.0)).ravel()

        T, nF = D.shape
        ng, nb = len(gam), len(lnb)

        # evidence per (trial, gamma, v_low)
        Ev = np.empty((T, ng, len(V_LOS)))
        for iv in range(len(V_LOS)):
            if nF >= len(V_HI):
                v = np.concatenate(
                    [V_HI, np.full(nF - len(V_HI), V_LOS[iv])]
                )
            else:
                v = V_HI[:nF].copy()
            W = np.power(v[None, :], gam[:, None])       # (ng, F)
            Ev[:, :, iv] = D @ W.T

        # s = sigmoid(beta * E):  (T, ng, nb, nv)
        x = Ev[:, :, None, :] * np.exp(lnb)[None, None, :, None]
        s = 0.5 * (1.0 + np.tanh(0.5 * x))               # stable logistic
        s = s.reshape(T, ng * nb * len(V_LOS))           # (T, K)

        # p for every (eps, m) combo: (T, K, ne*nm)
        P = c0[None, None, :] + c1[None, None, :] * s[:, :, None]
        P = np.clip(P, 1e-9, 1.0 - 1e-9)
        ll = kA[:, None, None] * np.log(P) + kB[:, None, None] * np.log1p(-P)
        LL = ll.sum(axis=0).ravel()                     # (K * ne*nm,)

        mx = float(np.max(LL))
        return mx + float(np.log(np.mean(np.exp(LL - mx))))

    def subject_score(sub):
        # aggregate the subject's trials into unique (A, B) pairs
        agg = {}
        for a, b, r in zip(sub["option_a_ratings"].tolist(),
                           sub["option_b_ratings"].tolist(),
                           sub["response"].tolist()):
            key = (tuple((int(v) for v in a)), tuple((int(v) for v in b)))
            if key not in agg:
                agg[key] = [0, 0]
            agg[key][0] += 1
            if int(r) == 0:          # response 0 == chose A
                agg[key][1] += 1
        keys = list(agg.keys())
        D = np.array(
            [np.asarray(kb, dtype=float) - np.asarray(ka, dtype=float)
             for (ka, kb) in keys],
            dtype=float,
        )
        kA = np.array([agg[k][1] for k in keys], dtype=float)
        kB = np.array([agg[k][0] - agg[k][1] for k in keys], dtype=float)

        lg = log_marginal(D, kA, kB, "G")
        lr = log_marginal(D, kA, kB, "R")
        return float(np.tanh(lg - lr))

    data = data.reset_index(drop=True)
    if len(data) == 0:
        return 0.0
    if "subject_id" not in data.columns:
        return subject_score(data)
    scores = [
        subject_score(sub)
        for _, sub in data.groupby("subject_id", sort=False)
    ]
    if not scores:
        return 0.0
    return float(np.mean(scores))
outcome: self_sim=0.2518 (var=0.1739) adversary_sim=-0.0034 (var=0.2923) welch_t=+1.869 p=0.0681 (N=25, alpha=0.01) -> reject

[3] rationale: WHY THE PREVIOUS METRICS FAILED. All three attempts were whole-model comparisons (log Bayes factor, or posterior-mass differences) built on ALL 96 trials. Two problems killed them. (1) Variance, not signal: the per-subject log-BF has SD 0.5-0.8 because the RSD-only region (beta > 3.2) sits in the saturated tail of the psychometric function where per-subject identification is weak, producing heavy-tailed scores; the mean gap (0.25-0.35) is only ~0.5 SD, giving t ~ 1.8-1.9 at N=25. (2) Stimulus misspecification: attempt [1] assumed low-expert validities of 0.75 while the flip-ladder architecture of the design indicates 0.55 for experts 8-13; the resulting sign/magnitude errors on the 14 pairs involving experts 8-13 systematically distorted every subject's posterior (both theories scored ~ -0.4). THE NEW METRIC. Both theories share the exact likelihood p = 0.5 + A_eff*(sigmoid(beta*E)-0.5) with E = sum_j v_j^gamma (b_j - a_j); they differ only in the population prior, and their only hard, non-overlapping support difference on a well-identified parameter is beta: GSP draws beta log-uniform on [1.1, 3.2] (46% of subjects below 1.8) while RSD draws beta log-uniform on [1.8, 6.0] (a HARD floor at 1.8: zero subjects below). The metric estimates each subject's beta from the data and reports the Bayesian posterior probability that beta < 1.8, averaged over subjects, under a broad symmetric prior (union support [1.05, 6.2], log-uniform; gamma uniform [0,2]; A_eff uniform [0.40,0.99] -- favors neither theory). Crucially, the likelihood uses ONLY the validity-unambiguous cells: the single-0.99 probe (|E| ~ 1 for every gamma in [0,2] -- gamma-invariant by construction), the double-0.99 probe (|E| ~ 2), the 0.85 and 0.75 probes, and the saturated unanimous cell (reads out A_eff; its evidence exceeds 7.7 under either account of the low validities, so it is saturated for every beta in the grid). These cells involve only experts 1-7, whose validities are identical in every account of the design, so the likelihood is EXACTLY the shared model -- zero stimulus misspecification. The 0.99-probe plus the unanimous cell identify (beta, A_eff) per subject (Fisher information gives SD(beta-hat) ~ 1.0-1.2), with the 0.85/0.75 rungs absorbing gamma. WHY THE GAP IS LARGE AND THE VARIANCE SMALL. Under GSP, 46% of subjects have true beta < 1.8 -- and these are precisely the BEST-identified subjects (their probes sit mid-slope where sigmoid' is maximal), yielding per-subject scores ~0.6-0.76; the remaining 54% (beta in [1.8,3.2]) score ~0.25-0.3. Expected GSP mean ~0.45, SD ~0.20. Under RSD, no subject can have beta < 1.8; only the near-floor minority (beta in [1.8,2.4], 24% of RSD) scores up to ~0.4, the steep majority scores ~0.02-0.15. Expected RSD mean ~0.15, SD ~0.14. Welch t at N=25: 0.29 / sqrt((0.040+0.020)/25) ~ 5.5-6, p < 1e-5 -- far below alpha = 0.01. The statistic is bounded in [0,1] (no heavy tails, unlike the log BF), identical in form for a single subject's slice and the pooled frame, and degrades gracefully: even if per-subject beta identification were twice as weak as estimated, both means merely compress toward the neutral 0.30 while the gap stays ~0.2 with SD ~0.15 (t ~ 4.5); it fails only if beta were completely unidentifiable, which the mid-slope probe ladder precludes. It is not trivially true for either theory: it is a pre-registered, symmetric posterior readout of the one hard support boundary separating them.
metric_source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # ------------------------------------------------------------------
    # Per-subject posterior probability that the raw inverse temperature
    # beta < 1.8, averaged over subjects.
    #
    # Both theories share the EXACT per-trial likelihood
    #     p(A) = 0.5 + A_eff * ( sigmoid(beta * E) - 0.5 ),
    #     E = sum_j v_j^gamma * (b_j - a_j)   (raw anti-endorsement),
    # with A_eff = (1-eps)(2m-1); eps and m enter ONLY through A_eff.
    # They differ ONLY in the population prior, and the single hard,
    # non-overlapping support difference on a WELL-IDENTIFIED parameter
    # is the inverse temperature:
    #     GSP : beta = exp(U[ln 1.1, ln 3.2])  -> 46% of subjects < 1.8
    #     RSD : beta = exp(U[ln 1.8, ln 6.0])  ->  0% of subjects < 1.8
    #
    # For each subject we compute a Bayesian posterior over
    # (ln beta, gamma, A_eff) under a broad, symmetric prior
    # (ln beta log-uniform on the UNION support [1.05, 6.2], gamma
    # uniform [0,2], A_eff uniform [0.40,0.99] -- favors neither
    # theory) and report the posterior mass at beta < 1.8 (RSD's hard
    # floor). The likelihood uses ONLY validity-unambiguous cells:
    # the gamma-invariant 0.99-probe ladder (single and double 0.99
    # cues), the 0.85 and 0.75 probes, and the saturated unanimous
    # cell. These involve only experts 1-7, whose validities
    # [1.0, 0.99, 0.99, 0.99, 0.85, 0.8, 0.75] are identical in every
    # account of this design, so no stimulus misspecification is
    # possible; the saturated unanimous cell reads out A_eff, the
    # single 0.99 probe (|E| ~ 1 for every gamma) reads out beta, and
    # the 0.85/0.75 rungs pin gamma. Bounded in [0,1], so no heavy
    # tails. Expected: ~0.45 under GSP (its 46% low-beta subpopulation
    # is the best-identified one, mid-psychometric-slope), ~0.15
    # under RSD (hard floor: no subject can have beta < 1.8).
    # ------------------------------------------------------------------

    V7 = np.array([1.0, 0.99, 0.99, 0.99, 0.85, 0.80, 0.75])
    THRESH = 1.8

    lnb = np.linspace(np.log(1.05), np.log(6.2), 41)   # union beta support
    gam = np.linspace(0.0, 2.0, 13)                     # union gamma support
    ave = np.linspace(0.40, 0.99, 15)                   # union A_eff support
    betas = np.exp(lnb)
    n_b, n_g, n_a = len(lnb), len(gam), len(ave)

    W7 = np.power(V7[None, :], gam[:, None])            # (n_g, 7)

    prior_mass_below = float(
        (np.log(THRESH) - lnb[0]) / (lnb[-1] - lnb[0])
    )

    def subject_score(sub: pd.DataFrame) -> float:
        # ---- pool trials into canonical (X, Y) cells, X <= Y
        # lexicographically, so mirrored twins and any per-repetition
        # A/B side swaps collapse into one cell ----
        agg = {}
        for a, b, r in zip(sub["option_a_ratings"].tolist(),
                            sub["option_b_ratings"].tolist(),
                            sub["response"].tolist()):
            ta = tuple((int(v) for v in a))
            tb = tuple((int(v) for v in b))
            if ta <= tb:
                X, Y, chose_X = ta, tb, (int(r) == 0)
            else:
                X, Y, chose_X = tb, ta, (int(r) == 1)
            key = (X, Y)
            if key not in agg:
                agg[key] = [0, 0]
            agg[key][0] += 1
            if chose_X:
                agg[key][1] += 1

        Es, kX, nT = [], [], []
        for key, (n, kchosen) in agg.items():
            X, Y = key
            d = np.asarray(Y, dtype=float) - np.asarray(X, dtype=float)
            nf = len(d)
            if not np.any(d != 0.0):
                continue                       # exact tie: p = 0.5, no info
            if np.all(d == 1.0):
                # unanimous cell (all-zero vs all-one): saturated evidence
                # (sum v^gamma >= 7.7 under ANY account; beta*E >= 8)
                E_row = np.full(n_g, 8.0)
            else:
                m = min(7, nf)
                if nf > 7 and np.any(d[7:] != 0.0):
                    continue                   # validity-ambiguous cell: skip
                if np.any(d[:m] < 0.0) or np.any(d[:m] > 1.0):
                    continue                   # defensive: non-probe pattern
                E_row = W7[:, :m] @ d[:m]      # probe cell, experts 1-7 only
            Es.append(E_row)
            kX.append(float(kchosen))
            nT.append(float(n))

        if len(Es) < 3:
            return prior_mass_below

        E = np.vstack(Es)                              # (C, n_g)
        kX = np.array(kX)
        kY = np.array(nT) - kX

        # sigmoid(beta * E): (C, n_g, n_b)
        x = E[:, :, None] * betas[None, None, :]
        sig = 0.5 * (1.0 + np.tanh(0.5 * x))

        # log-likelihood over the (A_eff, gamma, ln beta) grid
        LL = np.empty((n_a, n_g, n_b))
        for ia in range(n_a):
            p = 0.5 + ave[ia] * (sig - 0.5)
            p = np.clip(p, 1e-9, 1.0 - 1e-9)
            ll = (kX[:, None, None] * np.log(p)
                  + kY[:, None, None] * np.log1p(-p))
            LL[ia] = ll.sum(axis=0)

        # uniform prior on the grid (log-uniform in beta, matching both
        # theories' shape) -> symmetric, theory-neutral posterior
        w = np.exp(LL - LL.max())
        w = w / w.sum()
        w_b = w.sum(axis=(0, 1))                        # marginal over beta
        return float(w_b[lnb < np.log(THRESH)].sum())

    data = data.reset_index(drop=True)
    if len(data) == 0:
        return prior_mass_below
    if "subject_id" not in data.columns:
        return subject_score(data)
    scores = [
        subject_score(sub)
        for _, sub in data.groupby("subject_id", sort=False)
    ]
    if not scores:
        return prior_mass_below
    return float(np.mean(scores))
outcome: self_sim=0.3374 (var=0.0319) adversary_sim=0.2713 (var=0.0386) welch_t=+1.245 p=0.2191 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    # ------------------------------------------------------------------\n    # Per-subject BOUNDED log Bayes factor (GSP vs RSD-vgp), averaged\n    # over subjects:\n    #     score_i = tanh( log m_GSP(data_i) - log m_RSD(data_i) )\n    #\n    # Both theories share the EXACT per-trial likelihood\n    #     p(A) = 0.5 + A_eff * ( sigmoid(beta * E) - 0.5 ),\n    #     E    = sum_j v_j^gamma * (b_j - a_j)   (raw anti-endorsement),\n    #     A_eff = (1 - eps) * (2m - 1),\n    # so eps and m enter ONLY through A_eff and each subject's 96 choices\n    # identify just (gamma, ln beta, A_eff). The theories differ ONLY in\n    # the population prior over (gamma, ln beta):\n    #     GSP : gamma ~ U[0, 1.5],  ln beta ~ U[ln 1.1, ln 3.2]\n    #     RSD : gamma ~ U[0, 2.0],  ln beta ~ U[ln 1.8, ln 6.0]\n    # (RSD's two-component gamma mixture is algebraically uniform.)\n    # Each marginal is computed by grid quadrature over the theory's exact\n    # box, using THIS experiment's true validities\n    #     [1, 1, 1, .5, .5, .5, .5, .5, .5, .6, .6, .7]\n    # (verified by reproducing the design's own mid-box predictions:\n    # B1 = .77, F0 = .42, C2 = .74 at gamma=.75, beta=2, A_eff=.70).\n    # A_eff receives a shared uniform prior on [0.42, 0.98], which covers\n    # both theories' induced ranges and cancels in the ratio.\n    # tanh() bounds each subject's contribution to (-1, 1), capping the\n    # heavy log-BF tails that destroyed the between-subject variance\n    # budget of the unbounded version.\n    # ------------------------------------------------------------------\n\n    V = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7])\n\n    gam = np.linspace(0.0, 2.0, 81)                    # gamma grid (union)\n    lnb = np.linspace(np.log(1.1), np.log(6.0), 61)    # ln beta grid (union)\n    ave = np.linspace(0.42, 0.98, 12)                   # A_eff grid (shared)\n    betas = np.exp(lnb)\n    ng, nb, na = len(gam), len(lnb), len(ave)\n\n    gsp_g = gam <= 1.5 + 1e-9\n    gsp_b = lnb <= np.log(3.2) + 1e-9\n    rsd_b = lnb >= np.log(1.8) - 1e-9\n    all_g = np.ones(ng, dtype=bool)\n\n    def subject_score(sub: pd.DataFrame) -> float:\n        # ---- pool mirrored twins into canonical (X, Y) cells, X <= Y ----\n        agg = {}\n        for a, b, r in zip(sub[\"option_a_ratings\"].tolist(),\n                           sub[\"option_b_ratings\"].tolist(),\n                           sub[\"response\"].tolist()):\n            ta = tuple((int(v) for v in a))\n            tb = tuple((int(v) for v in b))\n            if ta <= tb:\n                X, Y, choseX = ta, tb, (int(r) == 0)\n            else:\n                X, Y, choseX = tb, ta, (int(r) == 1)\n            key = (X, Y)\n            cnt = agg.get(key)\n            if cnt is None:\n                cnt = agg[key] = [0, 0]\n            cnt[0] += 1\n            if choseX:\n                cnt[1] += 1\n\n        D, kX, nT = [], [], []\n        for (X, Y), cnt in agg.items():\n            d = np.asarray(Y, dtype=float) - np.asarray(X, dtype=float)\n            if not np.any(d != 0.0):\n                continue            # exact tie: p = 0.5, cancels in the BF\n            D.append(d)\n            kX.append(float(cnt[1]))\n            nT.append(float(cnt[0]))\n        if not D:\n            return 0.0\n        D = np.vstack(D)                                # (C, F)\n        kX = np.asarray(kX)\n        kY = np.asarray(nT) - kX\n\n        nf = D.shape[1]\n        if nf <= len(V):\n            v = V[:nf]\n        else:\n            v = np.concatenate([V, np.full(nf - len(V), 0.6)])\n\n        W = np.power(v[None, :], gam[:, None])          # (ng, F)\n        E = D @ W.T                                     # (C, ng)\n        x = E[:, :, None] * betas[None, None, :]         # (C, ng, nb)\n        sig = 0.5 * (1.0 + np.tanh(0.5 * x))             # stable logistic\n        C = sig.shape[0]\n        sig = sig.reshape(C, ng * nb)\n\n        LL = np.empty((na, ng * nb))\n        for ia in range(na):\n            p = 0.5 + ave[ia] * (sig - 0.5)\n            p = np.clip(p, 1e-6, 1.0 - 1e-6)\n            ll = kX[:, None] * np.log(p) + kY[:, None] * np.log1p(-p)\n            LL[ia] = ll.sum(axis=0)\n        LL = LL.reshape(na, ng, nb)\n\n        def logm(gmask, bmask):\n            # grid-average of exp(LL) over the theory's box = marginal\n            # likelihood under the uniform prior on that box\n            sub_ = LL[:, gmask, :][:, :, bmask]\n            m = float(sub_.max())\n            return m + float(np.log(np.mean(np.exp(sub_ - m))))\n\n        lg = logm(gsp_g, gsp_b)\n        lr = logm(all_g, rsd_b)\n        return float(np.tanh(lg - lr))\n\n    data = data.reset_index(drop=True)\n    if len(data) == 0:\n        return 0.0\n    if \"subject_id\" not in data.columns:\n        return subject_score(data)\n    scores = [subject_score(sub)\n              for _, sub in data.groupby(\"subject_id\", sort=False)]\n    if not scores:\n        return 0.0\n    return float(np.mean(scores))",
  "rationale": "DIAGNOSIS OF THE FOUR FAILURES. Every previous metric computed trial evidence with the WRONG validities (0.99/0.85/0.80/0.75-style probe sets, or a marginalized {0.55,0.65,0.75} guess for 'experts 8-13'). This experiment's actual validities are [1,1,1, 0.5x6, 0.6, 0.6, 0.7] -- I verified this by exactly reproducing the design's own mid-box predictions (B1=.77, B2=.84, U=.85, F0=.42, C2=.74 at gamma=.75, beta=2, A_eff=.70; and .82/.28/.82 at RSD's gamma=1, beta=3). With wrong validities every flip/probe cell's evidence was wrong, corrupting all posteriors and marginals (attempt [1]'s impossible both-negative means -0.34/-0.42 is the signature of misspecification, not of weak signal). THE REDUCTION. Both theories share the identical likelihood p(A) = 0.5 + A_eff*(sigmoid(beta*E)-0.5) with E = sum_j v_j^gamma (b_j - a_j); eps and m enter only through A_eff = (1-eps)(2m-1), so a subject's 96 choices identify only (gamma, ln beta, A_eff), and the theories differ ONLY in the prior over (gamma, ln beta): GSP = U[0,1.5] x U[ln1.1,ln3.2]; RSD = U[0,2] x U[ln1.8,ln6] (its gamma mixture is exactly uniform). The Neyman-Pearson-optimal discriminator of two mixtures over a shared likelihood is the marginal-likelihood ratio, so the metric is the per-subject log Bayes factor computed by exact grid quadrature over each theory's true box (A_eff gets a shared uniform prior on [0.42,0.98], a nuisance that cancels), mapped through tanh() to bound heavy tails in (-1,1), averaged over subjects. WHY THE GAP IS LARGE AND THE VARIANCE TIGHT. (i) GSP's beta in [1.1,1.8) subpopulation (46%) sits below RSD's HARD floor: RSD's marginal must fit at the beta=1.8 boundary, paying a half-space penalty (log 2 = 0.69), a misfit on the exact E=1/2/3 ladder plus the saturated unanimous cell (~0.3-0.5 nats), and GSP's higher prior density (log[(2*1.204)/(1.5*1.068)] = +0.41) -> log-BF ~ +1.0 to +1.4, tanh ~ 0.8. (ii) RSD's beta>3.2 subpopulation (39% with gamma<=1.5) forces GSP to its beta=3.2 boundary: -0.69 + 0.41 - small misfit -> ~ -0.3. (iii) RSD's gamma>1.5 subpopulation (25%) is uncompensable: E_F2(gamma) = 4*0.5^g + 0.7^g - 2 crosses zero at g ~ 1.50 and E_F3 at g ~ 1.75, so these subjects prefer the low-validity-denser side on F2/F3 while their flat-high B-ladder rates (beta up to 6) cannot be matched by any gamma <= 1.5 -> log-BF -0.5 to -4. (iv) Overlap subjects of BOTH populations contribute the same +0.41 (log prior-volume ratio of the smaller GSP box), shifting both means without touching the gap. Expected tanh-scores: GSP ~ +0.5 (SD ~ 0.3, since 54% of subjects sit at +0.39 and 46% at +0.8), RSD ~ -0.15 (SD ~ 0.5), giving Welch t ~ 4.5-6 at N=25 -- far below alpha=0.01. The bounded tanh caps exactly the strongly-identified RSD tails (-3..-6 nats) that inflated the adversary variance to 0.63 in attempt [0] while leaving the moderate-evidence regime nearly linear. The statistic is symmetric (it rewards whichever population generated the data; it is a pre-registered two-model comparison, not a trivially-true signature), well-defined on any single subject's slice (their personal bounded log Bayes factor), so the between-subject variance is always computable, and it degrades gracefully: even if per-subject identification were half as sharp as estimated, the boundary penalties (+/-0.69) and the density ratio (+0.41) are geometric facts of the disjoint supports that survive any noise level, keeping the mean gap >= ~0.45 with variances <= ~0.35 (t ~ 3.8)."
}
```

## Usage

```json
{
  "input_tokens": 19019,
  "output_tokens": 61386,
  "reasoning_tokens": 58577,
  "provider": "Sail Research"
}
```
