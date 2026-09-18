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
**Validities (n_features=7):** [0.97, 0.94, 0.72, 0.7, 0.68, 0.6, 0.54]

**Trial pairs (n=18):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0]
  trial 12: A=[0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 14: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 18: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]

**Rationale:** I advocate WC-ST (validity-blind Weber-scaled flag counting with an accelerating Weibull threshold, a severity tiebreak on exact count ties, and discrete expert triage) against PW-FMS (validity-tilted flag mass + strong worst-flag severity, divisive compression, graded near-tie cascade). Both models agree on polarity ('1' = warning flag; the less-bad option wins) and both use a worst-flag tiebreak at exact count ties, so direction-only tests are useless. They disagree on exactly three things, and every pair below targets one of them.

VALIDITIES [0.97, 0.94, 0.72, 0.70, 0.68, 0.60, 0.54]. Two very high experts, three mid experts safely ABOVE WC-ST's 0.65 triage cutoff, and exactly two droppable experts (0.60, 0.54). PW-FMS mean-centred weights at gamma=1 are w=[1.469, 1.409, 0.969, 0.929, 0.889, 0.729, 0.609] (top:bottom ~2.4:1, and the top-vs-mid gap of 0.50 is what drives its severity term); at gamma=0.5 they are w=[1.234, 1.204, 0.984, 0.964, 0.944, 0.864, 0.804]. All 9 base pairs appear in both A/B orders (18 unique pairs, K=5 -> 90 trials), so side bias cancels exactly. Pairs P1-P8 carry NO flags on the droppable experts, so triage is disabled there by construction and the magnitude tests are uncontaminated; only P9 is triage-sensitive.

CLASS 1 - CUE-CONTROLLED MAGNITUDE LADDER (P1 0v1, P2 1v2, P3 2v3, P4 3v4, P5 2v4; trials 1-10 and their mirrors). The ladder is strictly NESTED and the single discriminating flag is ALWAYS expert 3 (0.72); in P2-P5 both options carry expert 1's flag, so PW-FMS's severity term cancels EXACTLY and its numerator |D| is constant (=w3) across P2-P4 for every gamma and lambda. Only the compression denominator grows, so PW-FMS is committed to a shallow monotone decline: ~0.86, 0.77, 0.72, 0.70 (gamma .6-1, lam 1.1, beta 3.6, c .2, eps .29) - never near chance. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, which through the Weibull (theta ~.40, kappa ~6) plus lapse gives ~0.85, 0.74, 0.53, 0.505 - a CLIFF to chance. A ~0.20 model gap sits on P3 and P4. A flat-ish plateau at 2v3/3v4 is impossible for WC-ST at any theta/kappa (its psychometric function is strictly increasing in s and cue identity is held fixed); a collapse to chance there is impossible for PW-FMS at any comp_c because compression is divisive, not threshold-like. P5 (2v4) is the ratio control: same smaller count as P3 and the same base counts as P4, but ratio 1:2. WC-ST says P5 (s=0.436, ~0.79) must be indistinguishable from P2 (s=0.408) and far ABOVE P4 (same raw difference is irrelevant, only ratio matters); PW-FMS says P5 (0.81) sits only slightly above P2 (0.77) and only ~0.11 above P4. The signature {P2 ~ P5} >> {P3 > P4 ~ chance} is uniquely WC-ST; a linear/Gaussian margin rule or PW-FMS both predict P4 ~ P2.

CLASS 2 - COMPOSITION INVARIANCE AT MATCHED COUNTS (P6: A=[1,1,0,0,0,0,0] vs B=[0,0,1,1,1,0,0]; P7: A=[0,0,0,1,1,0,0] vs B=[1,1,1,0,0,0,0]). Both are 2-vs-3 with the fewer-flag option always A, so WC-ST is MATHEMATICALLY FORCED to give P6 = P7 = P3 (~0.53, weak A preference), with no triage escape because experts 6-7 carry no flags. PW-FMS reads them completely differently: in P6 the two flags sit on the 97%/94% experts, so mass (2.878 vs 2.787) plus a strong severity difference (w1-w3 = 0.50) gives D = +0.64 at gamma=1, lam=1.1 -> it REVERSES and picks the THREE-flag option, p(A) ~ 0.33; in P7 the same logic runs the other way (D = -2.62) -> p(A) ~ 0.85. That is a ~0.52 within-subject spread across two pairs WC-ST must call identical, and it is the single cleanest falsification test in the design. Equally, if P6 = P7 = P3 to within noise, no gamma/severity setting of PW-FMS can produce it.

CLASS 3 - EQUAL-COUNT TIE MAGNITUDE AND DISPERSION (P8: A=[0,0,1,0,0,0,0] vs B=[0,0,0,1,0,0,0]). Counts tie and the weighted masses (0.969 vs 0.929) nearly cancel, so PW-FMS enters its cascade, finds equal counts, and applies its STRONG worst-flag repulsion (tie_bias ~ 0.07) -> p(A) ~ 0.20, TIGHT across subjects (tie_bias in [0, 0.14]). WC-ST applies the same lexicographic rule but with a broad, mildly-repelled subject-level tie_bias in [0.05, 0.45] -> mean p(A) ~ 0.33 with LARGE between-subject dispersion. So P8 tests the magnitude and the variance of the tiebreak, not its direction, and it simultaneously identifies tie_bias for both models.

CLASS 4 - SEVERITY-VS-COUNT / TRIAGE (P9: A=[0,0,0,0,0,1,1], two flags on the 60%/54% experts, vs B=[1,0,0,0,0,0,0], one flag on the 97% expert). This is PW-FMS's own best pair and is included to keep the design fair: its severity term (sev_a - sev_b = 0.729-1.469) overwhelms the count difference, giving D = -0.95 at gamma=1 -> it picks the TWO-flag option A at ~0.78, while low-gamma subjects fall into the count fallback and pick B at ~0.78 (a gamma-driven bimodality). WC-ST is validity-blind: 84% of subjects count 1-vs-2 and pick B (p(A) ~ 0.26), while the triage subgroup discards the two sub-0.65 experts, reads 0-vs-1, and REVERSES to p(A) ~ 0.85 - a small, discrete upper mode rather than a continuum. The predicted group means differ by ~0.2-0.4 and, more informatively, the shape of the between-subject distribution differs (a small discrete ~16% mode for WC-ST vs a broad gamma-continuum split for PW-FMS), and WC-ST's reversers on P9 must be exactly the subjects who show NO composition effect on P6/P7.

PREDICTIONS. WC-ST: near-ceiling P1 (~0.85), ~0.74 P2, ~0.79 P5, and near chance on P3 and especially P4; exact equality of P3, P6 and P7 with no composition effect and no reversal; moderate, widely dispersed rejection of the top-flag holder on P8; mostly B on P9 with a small discrete triage mode. PW-FMS: a shallow monotone ladder (~0.86 -> 0.70, never approaching chance) with P4 close to P2; a huge P6-vs-P7 composition gap including a direction reversal on P6; tight, extreme (~0.20) responding on P8; and a gamma-graded, severity-driven preference for the two-low-flag option on P9.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** **Weber Counting with Severity Tiebreak and Expert Triage (WC-ST).**

A binary expert rating of '1' is read as a *warning flag*, not an endorsement, so an option's badness grows with the number of 1s it carries and the subject picks the option that seems to carry FEWER flags. Three mechanistic claims, plus one heterogeneity claim:

1. **Negative polarity, validity-blind numerosity.** The decision variable is the raw *count* of flags per option, not a validity-weighted mass. Which experts contributed the flags is irrelevant to the count; the stated validities are acknowledged but never converted into cue weights. This is what makes the model look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs, with a single mechanism.

2. **Scalar variability (Weber noise) on the counts.** Perceived numerosity is noisy with noise that scales with magnitude, so discriminability depends on the RATIO-scaled difference, not the raw difference:  s = (c_hi - c_lo) / sqrt(c_hi^2 + c_lo^2 + 1). The small additive constant 1 keeps 0-vs-1 easy. Crucially, the psychometric function over s is *threshold-like* (Weibull with slope exponent kappa > 1), i.e. accelerating near zero: comparisons whose count ratio falls below the subject's Weber threshold theta stay essentially at chance, while comparisons above it jump quickly to a lapse-limited ceiling. This is what produces near-chance 2-vs-3 (~0.53) at the same time as high 1-vs-2 (~0.73), 2-vs-4 (~0.78) and near-ceiling 0-vs-1 / 1-vs-4 / 0-vs-5 (~0.85) — a pattern that neither a flat sign-only rule (FC-LEX) nor a linear/Gaussian margin rule can generate.

3. **Severity tiebreak on exact count ties.** When the two options carry exactly the same number of flags the counting mechanism is silent. The subject then asks which single flag is *worst*: the most reliable expert that discriminates decides, and the option carrying THAT expert's flag is (usually) rejected. The strength/direction of this fallback is a subject-level bias tie_bias = P(choose the option flagged by the top discriminating expert), sitting below 0.5 (mildly repelled). Because this rule keys on the top discriminating expert in exactly the same way for every equal-count pair, it predicts *no* differential across equal-count pairs that share the same top discriminating expert.

4. **Discrete expert triage (minority).** A minority of subjects do not count everyone: they apply an all-or-none trust filter, ignoring the one or two least-valid experts entirely before counting (never a graded weighting — gradedness is ruled out by the equal-count data). Triage subjects therefore *reverse* on pairs whose flag surplus lives entirely on the discarded experts (e.g. 2 high-validity flags vs 3 low-validity flags), which injects genuine bimodal between-subject dispersion exactly where the human data show it, and creates the small positive asymmetry among equal-count pairs whose tie is broken by triage.

WC-ST is a sharp competitor to sign-only flag counting: they agree on polarity and validity-blindness but disagree about magnitude invariance. WC-ST predicts a monotone *ratio-driven* accuracy gradient (0v1 > 1v2 > 2v3 > 3v4 at constant difference; 0v1 < 0v2 < 0v4 at constant smaller count), whereas the sign-only rule predicts one flat plateau.

**Parameters:**
- theta: [0.34, 0.46]
- kappa: [4.5, 7.5]
- epsilon: [0.12, 0.48]
- tie_bias: [0.05, 0.45]
- ignore_prone: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------------- 1. parse stimulus ----------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n = a.size

    # ---------------- 2. validities ----------------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n:
        val = np.array([0.9]) if n == 1 else np.linspace(0.9, 0.55, n)

    # ---------------- 3. subject-level parameters ----------------
    theta = float(parameters["theta"])
    kappa = float(parameters["kappa"])
    eps = float(np.clip(float(parameters["epsilon"]), 0.0, 1.0))
    tie_bias = float(np.clip(float(parameters["tie_bias"]), 0.0, 1.0))
    ip = float(np.clip(float(parameters["ignore_prone"]), 0.0, 1.0))
    theta = max(theta, 1e-3)
    kappa = max(kappa, 0.5)

    # ---------------- 4. discrete expert triage (minority of subjects) ----
    # most subjects count every expert; a minority discard the 1 (or 2)
    # least-valid experts outright before counting.
    if ip < 0.84:
        n_drop = 0
    elif ip < 0.96:
        n_drop = 1
    else:
        n_drop = 2
    mask = np.ones(n, dtype=bool)
    if n_drop > 0 and n > 2:
        order_asc = np.argsort(val, kind="stable")   # ascending validity
        dropped = 0
        for idx in order_asc:
            if dropped >= n_drop or (n - dropped) <= 2:
                break
            if val[idx] < 0.65:                       # only near-chance experts
                mask[idx] = False
                dropped += 1
    af = a[mask]
    bf = b[mask]
    if af.size == 0:
        af, bf, mask = a, b, np.ones(n, dtype=bool)

    # ---------------- 5. Weber-noisy numerosity comparison --------------
    ca = float(np.sum(af))
    cb = float(np.sum(bf))

    if abs(ca - cb) > 1e-9:
        c_hi = max(ca, cb)
        c_lo = min(ca, cb)
        denom = np.sqrt(c_hi * c_hi + c_lo * c_lo + 1.0)
        s = (c_hi - c_lo) / denom               # ratio-scaled evidence
        x = (s / theta) ** kappa
        x = float(np.clip(x, 0.0, 50.0))
        core = 1.0 - 0.5 * np.exp(-x)           # Weibull psychometric fn
        core = float(np.clip(core, 0.5, 1.0))
        winner = 0 if ca < cb else 1            # fewer flags is chosen
        p_core = np.empty(2, dtype=float)
        p_core[winner] = core
        p_core[1 - winner] = 1.0 - core
    else:
        # ------------- 6. severity tiebreak on exact count ties ----------
        idxs = np.where(mask)[0]
        owner = None
        order = idxs[np.argsort(-val[idxs], kind="stable")]
        for j in order:
            if a[j] > b[j]:
                owner = 0
                break
            if b[j] > a[j]:
                owner = 1
                break
        if owner is None:                        # fall back to full cue set
            order_all = np.argsort(-val, kind="stable")
            for j in order_all:
                if a[j] > b[j]:
                    owner = 0
                    break
                if b[j] > a[j]:
                    owner = 1
                    break
        if owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.empty(2, dtype=float)
            p_core[owner] = tie_bias             # option carrying the worst flag
            p_core[1 - owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------------- 7. attention lapse ----------------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Partially Weighted Flag Mass with Worst-Flag Severity (PW-FMS).**

Subjects read a binary expert rating of '1' as a *warning flag* rather than an endorsement, and choose the option carrying the smaller *badness*. Five claims:

1. **Negative polarity.** Badness grows with the flags an option carries; the option with the smaller badness is preferred. This single mechanism makes the population look 'anti-TTB' on dominance/agreement pairs and 'pro-TTB' on conflict pairs.

2. **Partial validity weighting on a subject-level continuum.** Each cue gets w_j = 1 + gamma*(r_j - mean(r)) with r_j = (v_j-0.5)/0.5 and gamma a subject-level tilt running from 0 (pure equal-weight counter) to strongly validity-tilted. Because the tilt is *centred*, its behavioural consequence scales with the design's validity spread, which is why the same population looks like a counter in one design and like a weigher in another.

3. **Mild divisive compression, not a threshold.** Evidence s = |D| / (1 + c*(mass_a + mass_b)) fed through a plain logistic plus lapse, so discriminability declines *shallowly and monotonically* with base flag count rather than collapsing to chance. The compression denominator is the plain flag mass — the perceived *quantity* of complaints — because that is what sets the scale against which a difference is judged.

4. **Worst-flag severity, and it is strong.** An option is judged not only by *how much* flag mass it carries but also by *how bad its single worst flag is*. The comparative badness signal is D = (mass_a - mass_b) + lam*(sev_a - sev_b), where sev_i = max_j w_j x_ij is the weight of the most reliable expert who flagged option i (0 if unflagged) and lam is a subject-level severity weight, typically above the point of mere cancellation. Severity is a salience term, not a re-weighting of the tally — it enters the numerator of the evidence only, never the compression scale, and it is never gated on the count difference.

5. **A graded-evidence cascade at near-ties, terminating in strong worst-flag repulsion.** When |D| falls inside a subject-specific resolution band tau the weighted comparison is untrustworthy and the subject falls back lexicographically: (i) if the raw flag *counts* differ, take the option with fewer flags (same compression/logistic applied to counts); (ii) if the counts are also equal, ask which single flag is worst and be *strongly* repelled by the option carrying the top discriminating expert's flag (tie_bias ~ 0.07).

6. **Heterogeneous evidence sensitivity is the primary source of population dispersion.** The subject-level inverse temperature beta is broadly distributed, not narrowly clustered: some subjects convert badness differences into choices almost deterministically while others remain shallow even on large gaps. This spread — not lapse spread — is what generates the large between-subject variance in per-subject accuracy observed on the count-driven designs, and it simultaneously softens near-ceiling agreement/dominance contrasts (low-beta tail) while lifting count-driven accuracy (high-beta tail).

**Parameters:**
- gamma: [0.0, 1.25]
- severity: [0.55, 1.65]
- beta: [1.8, 5.4]
- comp_c: [0.12, 0.28]
- tau: [0.10, 0.55]
- epsilon: [0.18, 0.40]
- tie_bias: [0.0, 0.14]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------------- 1. parse stimulus ----------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n = a.size

    # ---------------- 2. design validities ----------------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n:
        val = np.array([0.9]) if n == 1 else np.linspace(0.90, 0.55, n)

    # ---------------- 3. subject-level parameters ----------------
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    c = float(parameters["comp_c"])
    tau = float(parameters["tau"])
    lam = float(parameters["severity"])
    eps = float(np.clip(float(parameters["epsilon"]), 0.0, 1.0))
    tie_bias = float(np.clip(float(parameters["tie_bias"]), 0.0, 1.0))
    gamma = max(gamma, 0.0)
    beta = max(beta, 0.0)
    c = max(c, 0.0)
    tau = max(tau, 0.0)
    lam = max(lam, 0.0)

    # ---------------- 4. partially weighted flag mass ----------------
    r = np.clip((val - 0.5) / 0.5, 0.0, 1.0)     # normalised validity strength
    rbar = float(np.mean(r))
    w = 1.0 + gamma * (r - rbar)                 # centred tilt: mean weight == 1
    w = np.clip(w, 0.05, None)                   # weights stay positive

    mass_a = float(np.dot(w, a))
    mass_b = float(np.dot(w, b))

    # ---------- 4b. worst-flag severity ----------
    fa = w * a
    fb = w * b
    sev_a = float(np.max(fa)) if np.any(a > 0.5) else 0.0
    sev_b = float(np.max(fb)) if np.any(b > 0.5) else 0.0

    # signed badness difference (A - B); severity enters the NUMERATOR only
    D = (mass_a - mass_b) + lam * (sev_a - sev_b)
    scale = mass_a + mass_b                      # compression scale = plain flag mass

    def _logistic_diff(diff, sc):
        s = abs(diff) / (1.0 + c * max(sc, 0.0))
        z = float(np.clip(beta * s, -50.0, 50.0))
        core = 1.0 / (1.0 + np.exp(-z))          # P(choose the LESS bad option)
        core = float(np.clip(core, 0.0, 1.0))
        win = 0 if diff < 0 else 1               # less badness is chosen
        out = np.empty(2, dtype=float)
        out[win] = core
        out[1 - win] = 1.0 - core
        return out

    if abs(D) >= tau and abs(D) > 1e-9:
        # ------------- 5. divisive compression + logistic on badness -------
        p_core = _logistic_diff(D, scale)
    else:
        # ------------- 6. near-tie cascade -------------------------------
        ca = float(np.sum(a))
        cb = float(np.sum(b))
        if abs(ca - cb) > 1e-9:
            # (i) badness unresolvable -> fall back on raw flag counting
            p_core = _logistic_diff(ca - cb, ca + cb)
        else:
            # (ii) counts tie too -> worst discriminating flag, strongly repellent
            owner = None
            order = np.argsort(-val, kind="stable")
            for j in order:
                if a[j] > b[j]:
                    owner = 0
                    break
                if b[j] > a[j]:
                    owner = 1
                    break
            if owner is None:
                p_core = np.ones(2) / 2.0
            else:
                p_core = np.empty(2, dtype=float)
                p_core[owner] = tie_bias         # option carrying the worst flag
                p_core[1 - owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------------- 7. attention lapse ----------------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    hits_no_top = []   # fewer-flag option does NOT carry the top expert's flag (P7-type)\n    hits_top = []      # fewer-flag option DOES carry the top expert's flag (P6-type)\n\n    for _, row in data.iterrows():\n        try:\n            a = [int(v) for v in row['option_a_ratings']]\n            b = [int(v) for v in row['option_b_ratings']]\n            resp = int(row['response'])\n        except Exception:\n            continue\n        if len(a) != len(b) or len(a) < 1:\n            continue\n        ca = sum(a)\n        cb = sum(b)\n        # matched-composition contrast: 2-flag vs 3-flag pairs\n        if sorted([ca, cb]) != [2, 3]:\n            continue\n        # top-validity expert (index 0) must discriminate the two options\n        if a[0] == b[0]:\n            continue\n        # which side has fewer flags?\n        if ca < cb:\n            fewer_side = 0\n            fewer_vec = a\n        else:\n            fewer_side = 1\n            fewer_vec = b\n        chose_fewer = 1.0 if resp == fewer_side else 0.0\n        if fewer_vec[0] == 1:\n            hits_top.append(chose_fewer)\n        else:\n            hits_no_top.append(chose_fewer)\n\n    if len(hits_top) == 0 or len(hits_no_top) == 0:\n        return 0.0\n\n    return float(np.mean(hits_no_top) - np.mean(hits_top))\n",
  "rationale": "This isolates the Class-2 composition-invariance test. Only four unique trials qualify (P6: [1,1,0,0,0,0,0] vs [0,0,1,1,1,0,0] and P7: [0,0,0,1,1,0,0] vs [1,1,1,0,0,0,0], both A/B orders, 5 reps each = 20 trials/subject, 10 per cell). Both pairs are 2-vs-3 flag comparisons that differ ONLY in which experts carry the flags, so side bias cancels exactly by mirroring and base-rate/lapse/threshold parameters cancel in the difference.\n\nWC-ST is validity-blind: the decision variable is the raw count, so 2-vs-3 yields exactly the same Weber evidence (s = 1/sqrt(14) = 0.267) regardless of which experts the flags sit on, and no expert here falls below the 0.65 triage cutoff, so triage subjects behave identically too. WC-ST therefore predicts this contrast is 0 for every subject, up to binomial noise on 10+10 trials (per-subject var ~0.05). All subject-level heterogeneity (theta, kappa, epsilon) cancels in the difference, which keeps the per-subject variance tight and purely sampling-driven.\n\nPW-FMS cannot be 0 unless gamma=0: with tilted weights plus its strong worst-flag severity term, the fewer-flag option that carries the 0.97 expert's flag becomes the LESS attractive one (D = +0.09 + lam*0.50 > 0 at gamma=1, so it reverses and picks the three-flag option, p(fewer) ~ 0.33), while in the mirrored composition the same machinery gives D = -2.62 and p(fewer) ~ 0.85. Predicted contrast ~ +0.3 to +0.5 (gamma-graded), versus ~0.00 for WC-ST. The direction and magnitude gap is large and the WC-ST cell is essentially noise-free per subject."
}
```

## Usage

```json
{
  "input_tokens": 13290,
  "output_tokens": 4553
}
```
