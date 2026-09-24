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
**Validities (n_features=12):** [0.99, 0.95, 0.88, 0.85, 0.82, 0.8, 0.78, 0.76, 0.74, 0.72, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 3: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** ADVOCATED: GSP (unnormalized raw-evidence variant). COMPETITOR: RFI-delta. Both theories share the anti-endorsement sparse-preference core and the same per-trial mixture algebra, so no direction dissociation exists; the design must attack their three structural differences quantitatively. (1) EVIDENCE SCALE. GSP's raw weights give every subject an effective slope (in normalized-evidence units) s = beta_raw * S(gamma), while RFI's normalized weights give s = beta <= 8. I choose 12 validities [0.99, 0.95, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.52, 0.52] with S(0)=12, S(1)=9.33, S(1.5)=8.33, so GSP's slope support [1.1*8.33, 3.2*12] = [9.16, 38.4] is DISJOINT from RFI's [3, 8]. Behavioral consequence, read out on a 7-rung matched mid-block margin ladder (Slow: single 0.52-cue, L1..L6: nested cues 4-9 at 0.85-0.74, U: unanimous; rung evidence E_norm = 0.045-0.083 / 0.083-0.094 / 0.167-0.183 / 0.25-0.27 / 0.33-0.35 / 0.50-0.51 / 1.0, gamma-invariant for both theories): every GSP subject's psychometric transition midpoint sits at E_norm <= 0.109 (at or below the single mid-cue rung), while every RFI subject's sits at E_norm >= 0.125 (between L1 and L4). Per-subject half-plane test on the decisive margin-2 rung: rise = p(U) - p(L2) satisfies rise <= 0.556*(p(L2) - 0.5) for EVERY GSP subject (sigma_L2 >= sigma(9.16*0.167) = 0.821) and rise >= 0.626*(p(L2) - 0.5) for EVERY RFI subject (sigma_L2 <= sigma(8*0.179) = 0.807, sigma_U tied to the same beta) -- disjoint half-planes in the (p(L2), rise) plane. Pooled: GSP ~ 0.65/0.71/0.80/0.84/0.85/0.85/0.85 (flat above margin 2; rise L2->U ~ 0.05), RFI ~ 0.56/0.58/0.67/0.73/0.78/0.83/0.86 (monotone climb; pooled rise L2->U >= 0.10). Single-cue ceiling: no RFI subject exceeds p = 0.647 on L1 or 0.635 on Slow; GSP reaches 0.95+. (2) GAMMA CEILING. RFI caps gamma at 1; GSP allows up to 1.5. Flip cells F1/F2/F3 pit a single top cue (0.99 / 0.95 / 0.88) against the low pair (0.52, 0.52); the weighted evidence E(gamma) = 2*0.52^gamma - v_top^gamma is positive for all gamma <= 1 (RFI sign-robustly prefers the top-cue option, p in [0.48, 0.534], slip and lapse cannot cross 0.5) but flips sign at gamma = 1.07 / 1.15 / 1.32. Any subject preferring the low-pair option on any F-cell, in the monotone flip order F1 before F2 before F3, falsifies RFI categorically while remaining inside GSP's box (acknowledged adversarial risk: GSP does not guarantee steep-gamma mass, so absence of flips is also GSP-compatible -- the F-ladder is a one-sided weapon; the ladder signatures above are the two-sided core). (3) POPULATION ARCHITECTURE. RFI: 94-95% majority with per-subject attenuation (1-eps)(1-2*delta) in [0.699, 0.84] (p_U in [0.817, 0.92]) plus ~5% conventional minority with NEGATIVE attenuation (p_U in [0.08, 0.18]) and an empty middle band; GSP: continuous m in [0.75, 1] gives attenuation in [0.44, 0.98], never negative (p_core >= 0.5 whenever E >= 0), with stated mean effective attenuation ~0.70 -- i.e., GSP predicts substantial subject mass BELOW RFI's 0.699 majority floor and ABOVE its 0.84 majority cap (up to p_U = 0.99, past RFI's hard delta >= 0.08 cap of 0.92). Fitted per-subject attenuation from the 56 ladder trials is the arbiter: any subject with negative attenuation falsifies GSP (RFI requires ~5%); any subject with attenuation < 0.699 or > 0.84 falsifies RFI's narrow majority box. Controls: C2 (A endorsed on the top pair 0.99+0.95, B on the low pair 0.52+0.52; 2-2 count tie) -- both theories prefer the low-validity-endorsed option but GSP confidently (p up to 0.94) while RFI is capped at 0.64, simultaneously ruling out pure count-based strategies; T0/T1 identical-vector ties verify both theories' exact-50% prediction and catch key bias; every informative trial has a mirrored twin (A/B swapped) so side bias cancels, with analysis pooled by sparse-option side (ladder) or top-cue-option side (F-cells, C2). 24 unique pairs x K = 4 = 96 trials exactly: per subject 56 ladder observations (8 per pooled rung), 24 flip observations (8 per pooled flip cell), 8 C2, 8 ties. N >= 40-60 subjects with hierarchical per-subject psychometric fitting. DECISIVE QUANTITATIVE SIGNATURES: (a) per-subject (p(L2), rise) half-plane disjointness: rise <= 0.556*(p2-0.5) for all subjects (GSP) vs rise >= 0.626*(p2-0.5) (RFI); (b) per-subject transition-midpoint support E_mid <= 0.109 (GSP) vs >= 0.125 (RFI) -- disjoint slope supports [9.16, 38.4] vs [3, 8]; (c) flat-above-margin-2 ladder with pooled rise L2->U ~ 0.05 (GSP) vs monotone climb with rise >= 0.10 (RFI); (d) single-cue rungs: p(L1) pooled ~0.71 vs ~0.58, no RFI subject above 0.647; (e) attenuation histogram: continuous [0.44, 0.98] with mass below 0.699 and/or above 0.84, no negative values (GSP) vs bimodal with empty middle, tight majority band, ~5% negative-attenuation minority (RFI); (f) any F-cell flip (monotone in flip point 1.07/1.15/1.32) categorically falsifies RFI's gamma <= 1; (g) ties at exactly 50% (shared control); (h) C2 magnitude gap (GSP up to 0.94, RFI <= 0.64).

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
**Description:** Red-Flag Integration with Trait Polarity and Attentional Slip (RFI-delta). When choosing between two products described by binary expert ratings, a large majority of subjects read a rating of 1 as a 'red flag' — evidence AGAINST the option carrying it — so the sparser option is perceived as higher quality; a small minority reads polarity conventionally. Polarity is a stable PER-SUBJECT trait: subject i carries a fixed sign s_i ∈ {+1, -1} drawn once, with P(s_i = -1) = pi at the population level (pi near the arbiter's ceiling, ~0.94). On top of the trait, each subject occasionally suffers a small attentional slip: on any given trial, with probability delta (≈0.10), the subject momentarily reverts to the opposite (instructed-mapping) polarity for that trial alone. The slip is structured exactly like a lapse — a transient per-trial reversion against a stable dominant trait — not a reopening of per-trial polarity mixing, and it is kept small (delta ≤ 0.12). Within a trial, all cues are integrated additively with near-flat validity weighting: evidence for A over B is E = s · Σ_j w_j (a_j − b_j), w_j ∝ v_j^gamma. Choice is a softmax over [E, −E] with per-subject inverse temperature beta plus an independent lapse epsilon to uniform; identical rating vectors yield exactly 50%. The slip attenuates each subject's effective extremity by (1 − 2·delta) and the population's effective anti-endorsement signal to q_eff = pi(1−delta) + (1−pi)delta ≈ 0.85, which simultaneously pulls the pooled point estimates back from saturation toward the observed moderate magnitudes and compresses the between-subject mixture variance 4·pi(1−pi)·((1−2·delta)·x)^2 toward the tight real values — while preserving the falsifiable subject-level bimodality signature (two attenuated polarity clusters, not one homogeneous middle).

**Parameters:**
- pi: [0.94, 0.95]
- polarity_u: [0, 1]
- delta: [0.08, 0.12]
- gamma: [0, 1]
- beta: [3, 8]
- epsilon: [0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Red-Flag Integration with Trait Polarity + attentional slip (RFI-delta).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RFI expects a (2, n_features) stimulus; got shape {stim.shape}."
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
    pi = float(parameters["pi"])
    u = float(parameters["polarity_u"])
    delta = float(parameters["delta"])

    # Trait polarity: drawn ONCE per subject (both parameters are fixed
    # for the entire subject run). s = -1 -> red-flag reading (rating of 1
    # is evidence AGAINST the carrying option); s = +1 -> conventional
    # reading (endorsement favors the carrying option). At the population
    # level P(s = -1) = E[pi] with pi in [0.94, 0.95], i.e. an
    # anti-endorsement majority near the arbiter's ceiling and a small
    # (~5%) conventional minority.
    s = -1.0 if u < pi else 1.0

    # Near-flat validity weighting: w_j = v_j^gamma, gamma in [0, 1].
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), absorbed by beta; it keeps the evidence scale O(1) for any
    # gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Signed additive evidence for A over B. With s = -1 each endorsement
    # carried by A counts AGAINST A and each endorsement carried by B
    # counts FOR A (the sparser option accumulates evidence).
    E = s * float(np.sum(w * (a - b)))

    # Softmax over [E, -E] with max-subtraction for numerical stability.
    # Identical rating vectors give E = 0 -> exactly 50/50.
    scores = np.array([E, -E])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_trait = e / e.sum()

    # Attentional slip: with probability delta the subject momentarily
    # reverts to the OPPOSITE polarity mapping for this trial alone.
    # The opposite-polarity distribution is the mirror of p_trait, so the
    # per-trial choice distribution is a (1-delta)/delta mixture of the
    # two mirrors. This is the expectation of a per-trial Bernoulli slip
    # against a stable dominant trait (lapse-structured, small delta),
    # NOT per-trial polarity mixing: the trait sign is still fixed per
    # subject and the slip only attenuates each subject's extremity by
    # (1 - 2*delta).
    p_slip = p_trait[::-1]
    p_core = (1.0 - delta) * p_trait + delta * p_slip

    # Independent lapse to uniform choice.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Low-evidence ladder preference: P(choose the all-zero \"sparse\" option\n    # on trials where exactly one option has NO endorsements and the other\n    # has exactly 1 or 2 endorsements). In this design these are the\n    # Slow / L1 / L2 rungs (unique trials 1-6), i.e. 24 of the 96 trials\n    # per subject (8 obs per rung).\n    #\n    # GSP (raw, unnormalized evidence): a single mid-validity endorsement\n    # carries raw evidence 0.5^gamma .. 0.85^gamma in [0.35, 1.0], multiplied\n    # by beta in [1.1, 3.2]  ->  logit inputs 0.4 .. 3.2  ->  strong sparse\n    # preference even at the lowest rungs (pooled ~0.74).\n    # RFI (per-experiment normalized evidence): the same rungs carry\n    # normalized evidence only ~0.05-0.18, times beta <= 8 and further\n    # attenuated by the slip (1-2*delta) and the ~5% polarity minority\n    # -> weak preference (pooled ~0.60).\n    # The gap (~0.14) is large relative to the per-subject SD (~0.11),\n    # and the statistic is a simple proportion, so it is well defined both\n    # on the pooled DataFrame and on each single subject's slice.\n\n    def _vec(x):\n        if isinstance(x, np.ndarray):\n            return x.astype(float)\n        return np.asarray(list(x), dtype=float)\n\n    a_raw = data[\"option_a_ratings\"].tolist()\n    b_raw = data[\"option_b_ratings\"].tolist()\n    resp = data[\"response\"].to_numpy()\n\n    vals = []\n    for i in range(len(data)):\n        a = _vec(a_raw[i])\n        b = _vec(b_raw[i])\n        ca = float(np.sum(a))\n        cb = float(np.sum(b))\n        r = int(resp[i])\n        # Exactly one option is endorsed-free; the other carries 1-2 cues.\n        if ca == 0.0 and 1.0 <= cb <= 2.0:\n            # A is the sparse option; response 0 means A was chosen.\n            vals.append(1.0 if r == 0 else 0.0)\n        elif cb == 0.0 and 1.0 <= ca <= 2.0:\n            # B is the sparse option; response 1 means B was chosen.\n            vals.append(1.0 if r == 1 else 0.0)\n        # All other trials (3+ cue rungs, unanimous, flip cells, C2, ties)\n        # are deliberately excluded: they carry little slope information\n        # beyond what these low rungs already provide.\n\n    if len(vals) == 0:\n        return 0.5\n    return float(np.mean(vals))",
  "rationale": "The two theories are behaviorally near-identical in their qualitative structure (both anti-endorsement, both mixture-attenuated), so the discriminator must be quantitative: the EVIDENCE SCALE. GSP integrates raw (unnormalized) validity weights, so a single mid-validity endorsement carries evidence 0.5^gamma..0.85^gamma (~0.35-1.0) times beta in [1.1, 3.2], giving logit inputs of 0.4-3.2 even at the lowest rungs of the margin ladder. RFI normalizes weights by their sum (~8.3-12 here), so the same rungs carry normalized evidence of only ~0.05-0.18, times beta <= 8, further attenuated by the slip (1-2*delta in [0.70, 0.84]) and diluted by the ~5.5% conventional-polarity minority. The metric is the proportion of choosing the all-zero (sparse) product on the low-evidence rungs of the ladder: trials where one option has no endorsements and the other has exactly 1 or 2 (the Slow rung with the single 0.5-validity cue, the L1 rung with the single 0.85 cue, and the L2 rung with the 0.85+0.82 pair; unique trials 1-6, 24 observations per subject). Expected values: GSP pooled ~0.74 (per-subject range roughly 0.59-0.97, SD ~0.11 including binomial noise over 24 trials); RFI pooled ~0.60 (majority cluster ~0.61 with SD ~0.03, ~5.5% polarity-flipped subjects near 0.39, total SD ~0.12). This yields a Welch t of roughly 4-4.5 at N=25 (p ~ 1e-4), comfortably below alpha=0.01. Higher rungs (3+ endorsements, unanimous) are excluded because GSP subjects saturate there (sigma ~ 1 regardless of slope), so they add only attenuation-driven between-subject noise, not slope signal; flip/C2 cells are excluded because they are one-sided gamma weapons with poor per-subject reliability. The statistic is a plain proportion over pre-identified trial types, so it is identically computable on the pooled DataFrame and on each subject's slice, and it uses only scalar-safe operations (no grouping/hashing of the list-valued rating columns)."
}
```

## Usage

```json
{
  "input_tokens": 7957,
  "output_tokens": 47158,
  "reasoning_tokens": 45912,
  "provider": "Sail Research"
}
```
