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
**Validities (n_features=12):** [0.98, 0.95, 0.88, 0.85, 0.82, 0.8, 0.78, 0.76, 0.74, 0.72, 0.62, 0.62]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
  trial 19: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** ADVOCATED: RFI-delta. COMPETITOR: GSP (unnormalized-evidence variant). The two theories share the anti-endorsement sparse-preference core, so no single cell separates them in direction. The decisive structural difference is the EVIDENCE SCALE: RFI NORMALIZES its cue weights (w = v^gamma / sum(v^gamma), so evidence lives in [-1, +1] with slope = beta <= 8), while GSP uses RAW weights (effective slope in normalized units = beta_raw * S(gamma), where S(gamma) = sum(v_j^gamma)). I therefore choose 12 validities [0.98, 0.95, 0.88, 0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.62, 0.55] with S(0)=12, S(1)=9.45, S(1.5)=8.46. Since GSP's beta_raw >= 1.1, its effective slope is >= 1.1*8.46 = 9.3 > 8 for EVERY gamma in its box [0, 1.5]: the two theories' effective-slope sets are DISJOINT ([3,8] vs [9.3, 38.4]). Behavioral consequence: GSP saturates by endorsement margin 2, while RFI's confidence must keep climbing all the way to the unanimous trial. The design is a margin ladder (option A endorsed on the top-k cues, k = 1,2,3,4,6,8,12, vs an empty B; nested top-k sets make the rung ordering gamma-invariant for both theories), plus conflict-style replications with nonempty bars (tied cues cancel exactly in both), a count-tie validity-conflict cell, a gamma probe, and identical-vector ties. (1) SATURATION DISSOCIATION (headline): on the margin-2 cells (difference carried by the two top cues), RFI's normalized evidence is capped at E_norm = 1.93/9.45 = 0.204, so even at beta=8, delta=0.08, eps=0 its p(sparse) <= 0.78, while its unanimous p >= 0.816 — EVERY RFI majority subject shows a within-subject rise p(k=12) - p(k=2) >= ~0.11 (typically 0.15-0.25), and the ~5% minority shows the mirrored fall <= -0.11. GSP's raw margin-2 evidence is >= 1.895 for all gamma, so sigma >= 0.890 and p(k=2) >= 0.672 for every GSP subject, with rise <= (2m-1)(1-eps)(1-sigma2) <= 0.283*(p2 - 0.5) <= 0.08 at RFI's boundary p2 = 0.78. The (p2, rise) regions are therefore mathematically disjoint: RFI = {p2 in [0.55,0.78], rise >= 0.11} plus a mirrored low cluster {p2 <= 0.40, rise <= -0.11}; GSP = {p2 >= 0.67, rise <= 0.283*(p2-0.5)}. Pooled mid-box predictions: RFI ladder p(sparse) ~ 0.59/0.67/0.73/0.78/0.83/0.86/0.88 for k=1/2/3/4/6/8/12 (rise k2->k12 ~ 0.17); GSP ~ 0.75/0.83/0.85/0.85/0.85/0.85/0.85 (rise ~ 0.02) — GSP is FLAT above margin 2, RFI keeps climbing. (2) SINGLE-CUE CEILING: on k=1 trials RFI's E_norm <= 0.104, so p(sparse) <= 0.665 for every RFI subject (pooled ~0.59); GSP's raw single-cue evidence (~1) with beta_raw up to 3.2 allows p up to 0.95 (pooled ~0.75). Subjects or populations confident on single-cue trials falsify RFI. (3) POPULATION ARCHITECTURE: RFI predicts a bimodal per-subject distribution on extreme cells (94-95% anti-endorsement majority with p(k=12) in [0.82, 0.92], ~5% conventional minority at [0.08, 0.18], EMPTY middle band); GSP's continuous m in [0.75, 1] predicts p(k=12) in [0.72, 0.98] with NO subject below 0.5 on any positive-evidence cell (m >= 0.75 > 0.5 makes p_core >= 0.5 for E >= 0). Even one subject with a below-0.5 sparse rate on E>0 cells falsifies GSP while RFI requires ~5% of them; conversely, population mass in (0.72, 0.82) or (0.92, 1) falsifies RFI's narrow per-subject box. Requires N >= 40-60 subjects; per subject, pooling L2/L2c and mirrors gives 16 margin-2 observations and 8 unanimous observations, enough for a hierarchical (p2, p12) scatter. (4) GAMMA PROBE (G1): A endorsed only on cue 1 (0.98), B on cues 11-12 (0.62, 0.55). A is the count-sparser option; the weighted evidence E = 0.62^gamma + 0.55^gamma - 0.98^gamma is positive for all gamma <= 1 (RFI's box: at gamma=1, +0.19) but flips sign at gamma ~ 1.41, inside GSP's gamma box (at gamma=1.5, -0.074). RFI sign-robustly prefers the count-sparser option (p ~ 0.51-0.53); GSP's steep-gamma subjects prefer the count-denser one. Pooled p(count-sparser) < 0.47 falsifies RFI while GSP survives. (5) COUNT-TIE CONTROL (C2): A endorsed on the two top cues, B on the two bottom cues — a 2-2 count tie, so pure counting predicts 50%; both theories prefer the low-validity-endorsed (effectively sparser) option B, but RFI's E_norm <= 0.08 caps p(B) <= ~0.60 while GSP reaches up to 0.95 (pooled ~0.69 vs ~0.54). This cell simultaneously rules out count-based strategies and adds a second low-margin scale probe. (6) CONFLICT-STYLE REPLICATIONS: L2c and L6c replicate the margin-2 and margin-6 rungs with both bars nonempty (shared endorsements on tied cues cancel exactly in both theories), controlling any 'pick the empty bar' heuristic and doubling the observations on the decisive margin-2 cell. (7) TIES: T0/T1 are identical-vector trials where both theories predict exactly 50% (key-bias check). Every informative trial has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by sparse-option side. 24 unique pairs x K = 4 = 96 trials exactly. DECISIVE QUANTITATIVE SIGNATURES: (a) pooled rise p(k=12) - p(k=2) ~ 0.17 (RFI) vs ~ 0.02 (GSP), and per-subject rise >= 0.11 (RFI, both mirrored clusters) vs GSP's bound rise <= 0.283*(p2 - 0.5) — disjoint regions in the (p2, rise) plane; (b) p(k=1) <= 0.665 for all subjects (RFI) vs substantial mass above 0.7 (GSP); (c) bimodal subject distribution with empty middle and ~5% sub-0.5 minority (RFI) vs continuous [0.72, 0.98] with a hard >= 0.5 floor (GSP); (d) p(count-sparser) >= 0.5 on G1 with a flip below 0.5 falsifying RFI's gamma <= 1; (e) monotone climbing ladder through k=12 (RFI) vs flat-above-k=2 saturation (GSP); (f) ties at exactly 50% (shared control).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Experiment constants (fixed at design time)\n    VALIDITIES = np.array([0.98, 0.95, 0.88, 0.85, 0.82, 0.80,\n                           0.78, 0.76, 0.74, 0.72, 0.62, 0.62])\n    # weights used only to break COUNT ties (which option is \"effectively sparser\")\n    TIE_W = np.power(VALIDITIES, 0.75)\n\n    # Contrast weights keyed by (n_endorsements_of_sparser_option,\n    # n_endorsements_of_denser_option). Negative on low-margin cells,\n    # positive on high-margin cells, and EXACTLY sum-zero over the design\n    # so that any global shift of a subject's sparse-choice rate (lapses,\n    # graded attenuation m, slip delta) cancels out of the statistic.\n    W_PAIR = {\n        (0, 1): -0.38,   # single-cue vs empty\n        (0, 2): -0.42,  # margin-2 ladder rung\n        (4, 6): -0.42,  # margin-2 conflict replication\n        (0, 3): -0.25,  # margin-3 rung\n        (0, 4): -0.06,  # margin-4 rung\n        (1, 2):  0.09,  # gamma probe (count 1 vs 2)\n        (0, 6):  0.26,  # margin-6 rung\n        (2, 8):  0.26,  # margin-6 conflict replication\n        (0, 8):  0.45,  # margin-8 rung\n        (0, 12): 0.58,  # unanimous rung\n    }\n    W_TIE = {2: -0.11}  # count-tie validity-conflict cell\n\n    df = data.reset_index(drop=True)\n    n = len(df)\n\n    row_w = np.zeros(n)\n    row_y = np.zeros(n)\n    row_ok = np.zeros(n, dtype=bool)\n\n    for i in range(n):\n        a = np.asarray(df.at[i, \"option_a_ratings\"], dtype=float)\n        b = np.asarray(df.at[i, \"option_b_ratings\"], dtype=float)\n        ca = int(round(float(np.sum(a))))\n        cb = int(round(float(np.sum(b))))\n        resp = int(df.at[i, \"response\"])\n\n        if ca == cb:\n            if np.array_equal(a, b):\n                continue  # identical vectors: both theories say 50/50\n            wa = float(np.sum(TIE_W * a))\n            wb = float(np.sum(TIE_W * b))\n            if wa == wb:\n                continue\n            sparse = 0 if wa < wb else 1  # lower weighted endorsement mass\n            w = float(W_TIE.get(ca, 0.0))\n        else:\n            sparse = 0 if ca < cb else 1  # fewer endorsements = sparser\n            key = (min(ca, cb), max(ca, cb))\n            w = float(W_PAIR.get(key, 0.0))\n\n        row_w[i] = w\n        row_y[i] = 1.0 if resp == sparse else 0.0\n        row_ok[i] = True\n\n    sids = df[\"subject_id\"].values\n    vals = []\n    for sid in pd.unique(sids):\n        mask = (sids == sid) & row_ok\n        idx = np.where(mask)[0]\n        if len(idx) == 0:\n            continue\n        w = row_w[idx]\n        y = row_y[idx]\n        denom = float(np.sum(np.abs(w)))\n        if denom <= 0.0:\n            continue\n        # per-subject polarity orientation: the RFI trait mixture contains a\n        # small mirrored minority; orienting by each subject's dominant\n        # polarity maps both clusters onto the same signed contrast.\n        rate = float(np.mean(y))\n        s_hat = 1.0 if rate >= 0.5 else -1.0\n        vals.append(s_hat * float(np.sum(w * y)) / denom)\n\n    if len(vals) == 0:\n        return 0.0\n    return float(np.mean(vals))",
  "rationale": "SATURATION-SHAPE CONTRAST INDEX. The two theories share the anti-endorsement core and are matched on average extremity, so raw preference rates cannot separate them. The decisive difference is the EVIDENCE SCALE: RFI normalizes its cue weights (evidence in [-1,1], slope beta<=8, so margin-2 evidence is only ~0.19 and even the unanimous trial only reaches ~1.0), whereas GSP uses raw weights (effective slope beta_raw*sum(v^gamma) >= 9.3), so GSP's softmax is essentially saturated by endorsement margin 2 while RFI's sparse-choice probability keeps climbing with margin. My metric is a fixed linear contrast on the per-trial indicator 'chose the sparser option', with weights that are NEGATIVE on low-margin cells (single-cue, margin-2 ladder and conflict replications, count-tie validity conflict) and POSITIVE on high-margin cells (margin 6, 8, unanimous), with the noise-appropriate magnitudes (each cell's weight is proportional to the predicted RFI-minus-GSP probability gap divided by its Bernoulli variance). Three robustness features are built in: (1) the weights sum EXACTLY to zero over the 88 informative trials, so any global shift of a subject's response rate -- GSP's graded attenuation m, its lapse epsilon, RFI's slip delta and lapse epsilon -- cancels identically; this removes the dominant between-subject nuisance directions (I verified the residual sensitivities to GSP's m and RFI's beta are <0.3 in raw units vs a sampling sd of ~1.3); (2) each subject's contrast is multiplied by the sign of their overall sparse-preference rate, which maps RFI's ~5% conventional-polarity minority onto the same signed value as the majority (for a sign-flipped subject the centered contrast flips sign exactly, so orientation recovers it) while every GSP subject (m>=0.75) is oriented +1; (3) mirrored A/B twins and identical-vector ties are handled symmetrically. Expected values: RFI-delta ~ +0.09 (pooled per-subject mean; per-subject sd ~0.05, dominated by binomial noise over 96 trials), GSP ~ +0.02 (per-subject sd ~0.05): the two distributions are separated by ~1.4 per-subject sd, giving Welch t ~ 5 at N=25 (p < 0.001). The sign of every low-margin weight is guaranteed by the disjoint effective-slope sets: even at the extreme corners, GSP's per-subject margin-2 sigmoid (>=0.889) exceeds RFI's maximum (0.836), while at high margins both theories sit at their ceilings (~0.85), so the contrast direction cannot reverse under any admissible parameter draw."
}
```

## Usage

```json
{
  "input_tokens": 8102,
  "output_tokens": 64701,
  "reasoning_tokens": 63055,
  "provider": "Sail Research"
}
```
