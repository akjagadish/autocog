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
**Validities (n_features=6):** [0.97, 0.93, 0.7, 0.68, 0.66, 0.55]

**Trial pairs (n=18):**
  trial 1: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  trial 10: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 12: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 14: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 16: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 17: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 18: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]

**Rationale:** GOAL. PW-FMS (advocated) and WC-ST (competitor) agree on polarity ('1' = warning flag, fewer/less-bad wins) and on the use of a worst-flag tiebreak at exact count ties, so direction-only tests are useless. They disagree on exactly three things, and every pair here targets one of them:

(i) HOW DISCRIMINABILITY SCALES WITH BASE COUNT. PW-FMS uses divisive compression, s = |D|/(1 + c*(mass_a+mass_b)), i.e. a SHALLOW monotone decline. WC-ST uses ratio-scaled Weber evidence pushed through an accelerating Weibull, i.e. a CLIFF to chance once the count ratio drops below theta.
(ii) COMPOSITION SENSITIVITY. PW-FMS's decision variable is validity-tilted mass PLUS a strong worst-flag severity term; WC-ST's is the validity-blind raw count. So two pairs with the identical count comparison but different validity composition must be identical for WC-ST and can differ enormously (even reverse) for PW-FMS.
(iii) THE SOURCE OF BETWEEN-SUBJECT REVERSALS. WC-ST's ONLY reversal mechanism is discrete triage of experts with validity < 0.65. I therefore chose validities so that exactly one expert (0.55) is droppable, and I keep that expert UNFLAGGED in every composition pair 

VALIDITIES [0.97, 0.93, 0.70, 0.68, 0.66, 0.55]. Two high experts, three mid experts safely above the 0.65 triage threshold, one droppable low expert. PW-FMS mean-centred weights: gamma=0.6 -> [1.27,1.22,0.94,0.92,0.89,0.76]; gamma=1.0 -> [1.44,1.36,0.90,0.86,0.82,0.60]; gamma=1.25 -> [1.55,1.45,0.88,0.83,0.78,0.50].

CLASS 1 \EEMPTY- CONSTANT-CUE MAGNITUDE LADDER (base pairs P1 0v1, P2 1v2, P3 2v3, P4 3v4, P5 4v5; trials 1-10). These are strictly NESTED and the single discriminating flag is ALWAYS expert 3 (0.70): only the shared background count grows. Because both options carry expert 1's flag in P2-P5, PW-FMS's severity term cancels exactly and D = -w3 is CONSTANT across the whole ladder for every gamma and every lambda; only the compression denominator grows. Predicted PW-FMS accuracies (gamma .6, lam 1.1, beta 3.6, c .2, eps .29): 0.86, 0.79, 0.75, 0.72, 0.70 \u2014 a shallow ~0.15 drop that never approaches chance. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, 0.154, giving (theta .40, kappa 6, eps .30): 0.85, 0.74, 0.53, 0.505, 0.50 \u2014 a cliff. The models differ by ~0.20 on P3-P5. A flat-ish plateau at 2v3/3v4/4v5 is impossible for WC-ST at ANY theta/kappa (its psychometric function is strictly increasing in s and the design holds cue identity fixed); a collapse to chance there is impossible for PW-FMS at any comp_c because compression is divisive, not threshold-like.

CLASS 2 \u2014 COMPOSITION CONTRASTS AT MATCHED COUNTS (P6 = pair 11: A=[1,1,0,0,0,0] vs B=[0,0,1,1,1,0], a 2v3; P7 = pair 13: A=[1,0,0,0,0,0] vs B=[0,0,0,1,1,0], a 1v2). WC-ST is mathematically forced to give P6 the SAME accuracy as P3 (both 2v3, s=0.267 -> ~0.53) and P7 the SAME accuracy as P2 (both 1v2, s=0.408 -> ~0.74); no expert can be triaged because expert 6 carries no flag. PW-FMS, by contrast, loads A with the two most reliable flags: at gamma>=0.75 the mass difference plus the strong severity term (Delta sev = w1-w3 = 0.32..0.68) flips the sign of D. On P6, D = -0.27+1.1*0.32 = +0.09 at gamma=0.6 (falls in the tau band -> count fallback -> A at ~0.75) but D = +0.81 at gamma=1.0 and +1.26 at gamma=1.25 -> decisive choice of the THREE-flag option B (p(A) ~ 0.29). Same structure on P7 (D = -0.16 at gamma .6 but +0.40 at gamma 1.0, +0.74 at gamma 1.25). So PW-FMS predicts (a) a large within-subject P3-vs-P6 and P2-vs-P7 gap and (b) genuinely BIMODAL between-subject responding on P6/P7, with the reversers being precisely the subjects who are most accurate on the low-ratio ladder items. WC-ST predicts zero P3/P6 and P2/P7 differences and zero dispersion beyond binomial + lapse, because its one reversal mechanism (triage) is disabled by construction here.

CLASS 3 \u2014 EQUAL-COUNT PAIRS: TIE CASCADE vs SEVERITY TIEBREAK + TRIAGE (P8 = pair 15: A=[1,0,0,0,0,1] vs B=[0,1,0,0,1,0], 2v2; P9 = pair 17: A=[0,0,1,0,0,0] vs B=[0,0,0,1,0,0], 1v1). Both pairs are decided by expert 1 / expert 3 respectively, and in BOTH the raw counts tie. PW-FMS: masses and severities nearly cancel (|D| <= 0.05 across the whole gamma range on both), so it enters its near-tie cascade, counts tie too, and it applies its STRONG worst-flag repulsion tie_bias ~ 0.07 -> p(option holding the top discriminating flag) ~ 0.19 after lapse, and crucially the SAME value on P8 and P9 with little between-subject spread. WC-ST: p = tie_bias, uniform in [0.05,0.45] -> mean ~0.33 with wide dispersion on P9, but on P8 its triage subgroup (~16%) drops expert 6, reads 1-vs-2, and REVERSES to p(A) ~ 0.79. So WC-ST uniquely predicts an upper mode above 0.5 on P8 that is absent on P9; PW-FMS uniquely predicts a tight, extreme, P8=P9 invariant rejection of the worst-flag holder. P9 also cleanly identifies tie_bias for both models.

Every base pair appears in both A/B orders (18 unique pairs, K=5 -> 90 trials) so side bias cancels exactly. P1 (0v1) anchors the lapse rate for both models.

SUMMARY OF DISCRIMINATING PREDICTIONS. PW-FMS: shallow monotone ladder (~0.86 -> ~0.70, never near chance); large composition gaps P3>>P6 and P2>>P7 with bimodal reversals on P6/P7 driven by gamma/severity; tight, extreme, identical responding on the two equal-count pairs. WC-ST: near-ceiling P1, ~0.74 P2, and chance from P3 onward; exact equality of P3 with P6 and of P2 with P7 with no reversals; tie_bias-driven moderate responding on P9 plus a triage-driven high mode on P8 only.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    if data is None or len(data) == 0:\n        return 0.5\n\n    hits = 0\n    total = 0\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()\n            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()\n        except Exception:\n            continue\n        if a.size == 0 or a.size != b.size:\n            continue\n        ca = float(np.sum(a))\n        cb = float(np.sum(b))\n        if abs(ca - cb) != 1.0:\n            continue\n        # strictly nested pairs only (the difference is a single shared cue)\n        nested = bool(np.all(a <= b + 1e-9)) or bool(np.all(b <= a + 1e-9))\n        if not nested:\n            continue\n        # high base-count region only: the smaller option already carries >= 2 flags\n        if min(ca, cb) < 2.0:\n            continue\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        if resp not in (0, 1):\n            continue\n        better = 0 if ca < cb else 1  # fewer warning flags is the 'correct' option\n        total += 1\n        if resp == better:\n            hits += 1\n\n    if total == 0:\n        return 0.5\n    return float(hits) / float(total)",
  "rationale": "This isolates the constant-cue magnitude ladder at high base counts: the nested pairs 2-vs-3, 3-vs-4 and 4-vs-5 (design trials 5-10), where the single discriminating flag is always expert 3 (0.70) and only the shared background count grows. The two theories make maximally divergent predictions here.\n\nPW-FMS (advocated): the worst-flag severity term cancels exactly (both options carry expert 1's flag in every one of these pairs), so the numerator is the constant |D| = w3 (~0.9 for any gamma). Only the divisive compression denominator 1 + c*(mass_a+mass_b) grows, and it does so shallowly, so accuracy declines gently from ~0.75 to ~0.70 and never approaches chance. Pooled over 30 trials/subject, expected value ~0.68-0.72.\n\nWC-ST (competitor): the evidence is the ratio-scaled Weber quantity s = 1/sqrt(c_hi^2+c_lo^2+1) = 0.267, 0.196, 0.154 for these three pairs, all far below every admissible Weber threshold theta in [0.34,0.46] with an accelerating Weibull (kappa 4.5-7.5). The psychometric function is pinned essentially at chance: expected value ~0.50-0.53, with no triage escape (expert 6, the only droppable one, is never flagged here) and no tie-branch involvement.\n\nSo the predicted gap is ~0.18 in the mean. Crucially, both sides are tight per subject: WC-ST's value is squeezed against the 0.5 floor (variance is essentially binomial), and PW-FMS's spread comes only from beta/eps acting on a fixed, moderate evidence level, pooled over 30 trials. Averaging over 6 unique pairs x 5 reps and over both A/B orders also cancels side bias and cuts binomial noise. The metric is well defined on a single subject's slice and on the pooled frame alike."
}
```

## Usage

```json
{
  "input_tokens": 13095,
  "output_tokens": 5868
}
```
