# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** **Partially Weighted Flag Mass with Worst-Flag Severity (PW-FMS).**

Subjects read a binary expert rating of '1' as a *warning flag* rather than an endorsement, and choose the option carrying the smaller *badness*. Five claims:

1. **Negative polarity.** Badness grows with the flags an option carries; the option with the smaller badness is preferred. This single mechanism makes the population look 'anti-TTB' on dominance/agreement pairs and 'pro-TTB' on conflict pairs.

2. **Partial validity weighting on a subject-level continuum.** Each cue gets w_j = 1 + gamma*(r_j - mean(r)) with r_j = (v_j-0.5)/0.5 and gamma a subject-level tilt running from 0 (pure equal-weight counter) to strongly validity-tilted. Because the tilt is *centred*, its behavioural consequence scales with the design's validity spread, which is why the same population looks like a counter in one design and like a weigher in another.

3. **Mild divisive compression, not a threshold.** Evidence s = |D| / (1 + c*(mass_a + mass_b)) fed through a plain logistic plus lapse, so discriminability declines *shallowly and monotonically* with base flag count rather than collapsing to chance. The compression denominator is the plain flag mass — the perceived *quantity* of complaints — because that is what sets the scale against which a difference is judged.

4. **Worst-flag severity, and it is strong.** An option is judged not only by *how much* flag mass it carries but also by *how bad its single worst flag is*. The comparative badness signal is D = (mass_a - mass_b) + lam*(sev_a - sev_b), where sev_i = max_j w_j x_ij is the weight of the most reliable expert who flagged option i (0 if unflagged) and lam is a subject-level severity weight, typically above the point of mere cancellation. Severity is a salience term, not a re-weighting of the tally — it enters the numerator of the evidence only, never the compression scale, and it is never gated on the count difference.

5. **A graded-evidence cascade at near-ties, terminating in strong worst-flag repulsion.** When |D| falls inside a subject-specific resolution band tau the weighted comparison is untrustworthy and the subject falls back lexicographically: (i) if the raw flag *counts* differ, take the option with fewer flags (same compression/logistic applied to counts); (ii) if the counts are also equal, ask which single flag is worst and be *strongly* repelled by the option carrying the top discriminating expert's flag (tie_bias ~ 0.07).

6. **Heterogeneous evidence sensitivity is the primary source of population dispersion.** The subject-level inverse temperature beta is broadly distributed, not narrowly clustered: some subjects convert badness differences into choices almost deterministically while others remain shallow even on large gaps. This spread — not lapse spread — is what generates the large between-subject variance in per-subject accuracy observed on the count-driven designs, and it simultaneously softens near-ceiling agreement/dominance contrasts (low-beta tail) while lifting count-driven accuracy (high-beta tail).

**Rationale:** MINIMAL DIFF: `predict` and `policy` are byte-identical to the accepted iter-8 base (stimulus parsing, centred weights w_j = 1 + gamma*(r_j - rbar), ungated numerator-only worst-flag severity, divisive compression on plain flag mass, logistic, near-tie cascade with count then worst-flag steps, lapse). The ONLY change is one parameter range: `beta` [2.5, 4.8] -> [1.8, 5.4]. Mean is held essentially constant (3.65 -> 3.60); only the spread grows (half-width 1.15 -> 1.80). Nothing is bundled: gamma, severity, comp_c, tau, epsilon and tie_bias are all untouched, no gate is reintroduced, no weight convexity, no band change.

WHY THIS KNOB. This is exactly the critic's iteration-8 primary recommendation, and it is the one lever explicitly pre-announced in iteration 7 that has never been tried. Every other lever in the model has been exhausted or shown to be counterproductive: tau (3 pushes, 1 accepted / 2 rejected), weight convexity (rejected), a count gate on severity (rejected), epsilon widening (rejected, delivered zero dispersion), tie_bias (already did its work at ~0.07), and a further severity step (moved its own target the wrong way last round). Beta *spread* is untouched.

MECHANISTIC EXPECTATION. The residual signature is diagnostic of a misspecified spread, not a misspecified mean: Exp2 (-0.677 vs -0.634), Exp3 (-0.603 vs -0.577) and Exp4 want LESS determinism while Exp7 (0.723 vs 0.773) and Exp8 (0.155 vs 0.166) want MORE. A mean shift in either direction is a wash; a spread increase at constant mean helps both ends simultaneously because the metrics are non-linear in beta:
- Low-beta tail (beta ~1.8-2.5): on the agreement/dominance pairs of Exps 2/3/4 the logistic is currently saturated, so these subjects pull the near-ceiling contrast magnitudes down, moving Exp2 from -0.677 toward -0.634 and Exp3 from -0.603 toward -0.577 without touching the sign structure.
- High-beta tail (beta ~4.8-5.4): the count-driven metrics of Exp7 and Exp8 are far from ceiling (the compression denominator 1 + c*(mass_a+mass_b) is large at high base counts), so extra sensitivity translates almost linearly into hits; Exp7 should move toward 0.74-0.76 and the easy-minus-hard gap on Exp8 toward 0.16.
Because the losses on the saturated experiments are concave in beta while the gains on the unsaturated ones are near-linear, a mean-preserving spread should net a small improvement on the point estimates in addition to the variance gain.

VARIANCE. This is the only untried route to the persistent 2-5x between-subject variance shortfall (Exp7 0.008 vs 0.033; Exp8 0.014 vs 0.079; Exp5 0.073 vs 0.133; Exp1 0.008 vs 0.026). Lapse spread was tested and inert because eps only compresses everything toward 0.5 multiplicatively near the metric midpoint; beta spread instead diversifies where each subject sits on the psychometric curve for every scored pair, which is precisely where the observed dispersion lives.

GUARDRAILS CHECKED BY REASONING. Exp1's scored conflict pairs have large |D| already near ceiling for mid-beta subjects, so the low- and high-beta tails roughly offset and 0.773 should hold in 0.76-0.79. Exp5's crossover pair routes through the count step for tilted subjects and its outcome is sign-determined, not magnitude-determined, so the 0.5167 mean should hold in 0.50-0.55 while its variance rises. Exp4's dominance leg is pinned near 0 regardless of beta, so only the conflict leg moves, and it moves by less than the guardrail width. Exp6's differential is set by band membership and tie_bias, both beta-independent, so it should be unchanged (it remains the one large residual, and the critic agrees lam is no longer the lever for it).

NOT BUNDLED. All three rejections in this loop came from pairing the targeted fix with a second knob move, so this ships the beta re-range alone.

**Parameters:**
  - `gamma`: `[0.0, 1.25]`
  - `severity`: `[0.55, 1.65]`
  - `beta`: `[1.8, 5.4]`
  - `comp_c`: `[0.12, 0.28]`
  - `tau`: `[0.10, 0.55]`
  - `epsilon`: `[0.18, 0.40]`
  - `tie_bias`: `[0.0, 0.14]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


### slot 2 — `pi_5` — KILLED ✗

**Description:** **Weber Counting with Severity Tiebreak and Expert Triage (WC-ST).**

A binary expert rating of '1' is read as a *warning flag*, not an endorsement, so an option's badness grows with the number of 1s it carries and the subject picks the option that seems to carry FEWER flags. Three mechanistic claims, plus one heterogeneity claim:

1. **Negative polarity, validity-blind numerosity.** The decision variable is the raw *count* of flags per option, not a validity-weighted mass. Which experts contributed the flags is irrelevant to the count; the stated validities are acknowledged but never converted into cue weights. This is what makes the model look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs, with a single mechanism.

2. **Scalar variability (Weber noise) on the counts.** Perceived numerosity is noisy with noise that scales with magnitude, so discriminability depends on the RATIO-scaled difference, not the raw difference:  s = (c_hi - c_lo) / sqrt(c_hi^2 + c_lo^2 + 1). The small additive constant 1 keeps 0-vs-1 easy. Crucially, the psychometric function over s is *threshold-like* (Weibull with slope exponent kappa > 1), i.e. accelerating near zero: comparisons whose count ratio falls below the subject's Weber threshold theta stay essentially at chance, while comparisons above it jump quickly to a lapse-limited ceiling. This is what produces near-chance 2-vs-3 (~0.53) at the same time as high 1-vs-2 (~0.73), 2-vs-4 (~0.78) and near-ceiling 0-vs-1 / 1-vs-4 / 0-vs-5 (~0.85) — a pattern that neither a flat sign-only rule (FC-LEX) nor a linear/Gaussian margin rule can generate.

3. **Severity tiebreak on exact count ties.** When the two options carry exactly the same number of flags the counting mechanism is silent. The subject then asks which single flag is *worst*: the most reliable expert that discriminates decides, and the option carrying THAT expert's flag is (usually) rejected. The strength/direction of this fallback is a subject-level bias tie_bias = P(choose the option flagged by the top discriminating expert), sitting below 0.5 (mildly repelled). Because this rule keys on the top discriminating expert in exactly the same way for every equal-count pair, it predicts *no* differential across equal-count pairs that share the same top discriminating expert.

4. **Discrete expert triage (minority).** A minority of subjects do not count everyone: they apply an all-or-none trust filter, ignoring the one or two least-valid experts entirely before counting (never a graded weighting — gradedness is ruled out by the equal-count data). Triage subjects therefore *reverse* on pairs whose flag surplus lives entirely on the discarded experts (e.g. 2 high-validity flags vs 3 low-validity flags), which injects genuine bimodal between-subject dispersion exactly where the human data show it, and creates the small positive asymmetry among equal-count pairs whose tie is broken by triage.

WC-ST is a sharp competitor to sign-only flag counting: they agree on polarity and validity-blindness but disagree about magnitude invariance. WC-ST predicts a monotone *ratio-driven* accuracy gradient (0v1 > 1v2 > 2v3 > 3v4 at constant difference; 0v1 < 0v2 < 0v4 at constant smaller count), whereas the sign-only rule predicts one flat plateau.

**Rationale:** The running-best theory (FC-LEX) nailed five metrics but failed badly on the 2-vs-3 count comparison (predicted 0.837, observed 0.517) because its accuracy is magnitude-blind: every unequal-count pair gets the same flat h. The graded-weight alternative (PWEA) fixed that cell only by over-rotating into validity-weighted evidence, which then produced a large spurious asymmetry between equal-count pairs (Exp 6: +0.54 vs observed 0.067). WC-ST resolves the dilemma with the arbiter's prescribed Weber-counting family plus two specific refinements that the arithmetic of the data demands.

(i) *Threshold-shaped (Weibull), not Gaussian, psychometric function.* I checked analytically that a plain Phi() over the ratio-scaled count difference cannot simultaneously give acc(2,3)~0.52 and acc(1,2)~0.73: those two comparisons differ by only a factor 1.5 in s, so any near-linear link forces them together, collapsing Exp 1 to chance. An accelerating Weibull with kappa~6 expands that factor into a threshold non-linearity: s=0.267 (2v3) sits below threshold and stays near chance, while s>=0.41 (1v2, 2v4) is well above it and s>=0.6 (0v1, 1v4, 1v5, 0v5) is lapse-limited at ceiling. This single change is what makes the whole set of six metrics simultaneously reachable.

(ii) *Severity (top-expert) tiebreak rather than a pure side bias.* Both Exp-6 target pairs are exact count ties whose top discriminating expert is feature 0, so the same tiebreak applies to both and the Exp-6 contrast comes out at ~0 — the cue-independence the arbiter demanded is achieved behaviourally without giving up the low tie-hit rate (~0.25) that Exp 3 needs for its rev cell (a coin-flip tiebreak would cost Exp 3 about 0.04).

(iii) *Discrete expert triage in a small minority (~16% of subjects).* This is the arbiter's point-4 heterogeneity, implemented as an all-or-none discard of the one or two least-valid experts (never graded weighting, which Exp 2/Exp 6 rule out). Triage subjects reverse on Exp 5's [1,1,0,0,0] vs [0,0,1,1,1] pair (the flag surplus lives entirely on the discarded 0.56/0.52 experts), pushing Exp 5 down to ~0.50 and injecting the bimodal between-subject dispersion the data show (var=0.133), while also generating the small positive Exp-6 asymmetry (~+0.09 vs observed +0.067) and sharpening Exp 3 toward -0.55.

Hand-computation over the six designs at the centre of the declared ranges gives: Exp1 ~0.76, Exp2 ~-0.62, Exp3 ~-0.55, Exp4 ~0.69, Exp5 ~0.50, Exp6 ~0.09, against observed 0.774 / -0.634 / -0.577 / 0.660 / 0.517 / 0.067 — every cell within ~0.05, versus FC-LEX's 0.32 miss on Exp 5. The theory is also experiment-invariant by construction: nothing keys on a specific design, only on flag counts, count ratios, and validity rank order, so it makes crisp falsifiable predictions for the ratio-vs-difference ladders the arbiter proposed.

**Parameters:**
  - `theta`: `[0.34, 0.46]`
  - `kappa`: `[4.5, 7.5]`
  - `epsilon`: `[0.12, 0.48]`
  - `tie_bias`: `[0.05, 0.45]`
  - `ignore_prone`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** **Sign-Dominant Flag Difference with Interference-Free Configural Dominance and a Weighting Sub-population (SD-FD/CD-M v2).**

A binary expert rating of '1' is read as a *warning flag*, not an endorsement, so an option's badness grows with the flags it carries and subjects pick the option that seems *less* flagged. Six claims:

1. **Negative polarity (domain-invariant).** Badness is monotone in flags; the less-flagged option is chosen. This single fact makes the population look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs (Exps 2–4, 7).

2. **Sign-dominant, magnitude-invariant difference detection.** The primary decision variable is the *flag-count difference*, and the psychometric function is almost completely saturating in |Δ| (t_c = sign(Δ)·|Δ|^rho with rho≈0.08). A one-flag advantage is nearly as diagnostic as a four-flag advantage: Exp 1 (Δ=1,2,3), Exp 7 (Δ=1) and Exp 2's conflict cell (Δ=2,3,4) all sit near 0.77. No divisive compression by total mass, no Weber threshold.

3. **Configural dominance is a distinct, INTERFERENCE-FREE read.** When one option's flag set is a subset of the other's, the options differ on a single side and the comparison is solved perceptually rather than by tallying. Crucially the shared flags are *aligned* in such a comparison, so they do not interfere: nested pairs get a fixed multiplicative gain (1+nu) and **no base-count cost at all**. This predicts a flat nested plateau at ~0.87 that is invariant to base count — 0-vs-1 dominance (Exp 4), 0-vs-5 (Exps 2, 3) and nested Δ=1 at base 2–4 (Exp 9, 0.872) are all equally accurate — while non-nested Δ=1 pairs at the same base counts sit much lower (Exp 7, 0.773). The previous version multiplied the nested boost on top of a full-gain base-0 term, which over-saturated dominance/agreement pairs (Exps 2/3/4) and forced the boost to be too large to hold Exp 9 up.

4. **A small additive base-count interference cost, applying only to NON-nested comparisons.** When the two flag sets cross-cut, each extra shared flag on the table costs a few points of discriminability (gain = 1 − kappa·min(base,6), saturating, never collapsing to chance). This supplies Exp 8's and Exp 10's small positive easy-minus-hard gaps.

5. **A bimodal counter/weigher mixture whose expression is design-scaled.** About half the population are validity-blind counters (phi = 0); the rest blend in a validity-weighted flag mass (weights u_j = r_j/mean(r), r_j = 2v_j − 1) with weight phi ∈ [0.5, 0.95]. Because the weights are proportional (not centred), the weigher's override strength scales with the design's validity spread: decisive reversal when one expert dominates (Exp 5), mild perturbation in graded designs (Exps 8, 10), negligible on agreement pairs (Exps 2, 3). Dispersion is bimodal on composition contrasts and unimodal (sensitivity/lapse driven) on the magnitude ladder.

6. **Count-ties are solved by everyone the same way, with load-graded repulsion.** When the raw flag counts are exactly equal the count read is silent for counters and weighers alike; all subjects fall back on 'who complained loudest' and are repelled by the option carrying the top discriminating expert's flag. The repulsion *weakens as more flags are already on the table* (p(choose the top-flagged option) rises with the shared flag load), because with many complaints on both sides the identity of the loudest complaint is less salient. This yields the small POSITIVE differential between heavy and light equal-count pairs seen in Exp 6 (+0.067) without any new mechanism.

**Rationale:** Minimal-diff edit on the ACCEPTED base (loss 0.0501), staying entirely inside the SD-FD/CD-M family. Three code lines changed plus range retunes, each targeting one of the critic's three systematic mismatches.

(1) **Re-balance nested gain vs baseline sensitivity by removing the base-count cost from nested trials.** The critic diagnosed that nu was doing too much and beta too little: non-nested count pairs were under-accurate (Exp1 0.732 vs 0.774) while nested large-Δ pairs were over-saturated (Exp2 -0.691 vs -0.634, Exp3 -0.609 vs -0.577). Diagnosing the arithmetic: the old gain was (1+nu)·(1−kappa·base), so a base-0 nested pair (0-vs-1, 0-vs-5 in Exps 2/3/4) got the FULL (1+nu)≈2.2 multiplier and reached ~0.95 accuracy, whereas the data imply dominance accuracy is only ~0.86–0.88 — essentially the SAME as nested Δ=1 at base 2–4 (Exp9 = 0.872). The principled fix is that in a dominance read the shared flags are aligned and therefore do not interfere: nested → gain = (1+nu) with no base cost, non-nested → gain = 1−kappa·base. This makes the nested plateau flat in base count (as the Exp4 vs Exp9 comparison demands) and lets nu shrink to ~0.35 while beta rises to ~2.55. Net effect: non-nested pairs (Exps 1, 7, and the conflict cells of 2/3) gain ~+0.04, base-0 nested pairs (agreement/rev cells of Exps 2/3, dominance base of Exp4) lose ~0.07, and Exp9 is held at ~0.87. Hand-computing the cells gives Exp1≈0.78, Exp2≈-0.63, Exp3≈-0.55, Exp4≈+0.67, Exp7≈0.77, Exp9≈0.87 — all closer to the observed values than the base.

(2) **kappa raised to [0.04, 0.13]** as instructed, which now bites only on the cross-cutting pairs that constitute Exp8's hard cell (2-vs-3, 3-vs-4 at base 2–3) while leaving the nested easy (0,1) trial untouched, so the easy-minus-hard gap widens from 0.09 toward ~0.13–0.15 without dragging Exp9/Exp10.

(3) **Load-graded tie repulsion (one added line + one parameter).** Exp6 was structurally pinned at ~0 because both target pairs are exact count ties broken by the same top expert. Following the critic's suggestion, tie repulsion now weakens with the shared flag load (tb = tie_bias + tie_slope·min(count,4)): the 2-flag-each pair gets ~0.41 and the 1-flag-each pair ~0.35, giving a small positive differential (~+0.06 vs real +0.067) with realistic binomial spread. This also nudges Exp3's equal-count rev trial upward, further softening that contrast in the right direction.

(4) **Widened heterogeneity ranges** (beta [1.9,3.2], epsilon [0.04,0.32]) to lift the under-dispersed between-subject variance on the count/polarity contrast metrics (Exps 1/2/3/4/8) without touching the weigher fraction, which Exps 5/10 already calibrate well.

**Parameters:**
  - `beta`: `[1.9, 3.2]`
  - `rho`: `[0.0, 0.16]`
  - `nu`: `[0.12, 0.55]`
  - `kappa`: `[0.04, 0.13]`
  - `mu`: `[0.40, 0.65]`
  - `epsilon`: `[0.04, 0.32]`
  - `tie_bias`: `[0.14, 0.42]`
  - `tie_slope`: `[0.0, 0.13]`
  - `w_prone`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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
    val = np.clip(val, 0.5, 1.0)

    # ---------------- 3. subject-level parameters ----------------
    beta = max(float(parameters["beta"]), 0.0)
    rho = float(np.clip(float(parameters["rho"]), 0.0, 1.0))
    nu = max(float(parameters["nu"]), 0.0)
    kappa = max(float(parameters["kappa"]), 0.0)
    mu = max(float(parameters["mu"]), 0.0)
    eps = float(np.clip(float(parameters["epsilon"]), 0.0, 1.0))
    tie_bias = float(np.clip(float(parameters["tie_bias"]), 0.0, 1.0))
    tie_slope = max(float(parameters["tie_slope"]), 0.0)
    wp = float(np.clip(float(parameters["w_prone"]), 0.0, 1.0))

    # bimodal mixture: ~52% pure counters (phi = 0), rest weighers (phi in [0.5,0.95])
    W_THRESH = 0.52
    if wp < W_THRESH:
        phi = 0.0
    else:
        phi = 0.50 + 0.45 * (wp - W_THRESH) / (1.0 - W_THRESH)
    phi = float(np.clip(phi, 0.0, 1.0))

    # ---------------- 4. flag counts and weighted flag mass ------------
    r = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)
    rbar = float(np.mean(r))
    if rbar <= 1e-9:
        u = np.ones(n)
    else:
        u = r / rbar                      # mean-1 proportional validity weights

    ca = float(np.sum(a))
    cb = float(np.sum(b))
    d = ca - cb                            # >0 : A carries more warning flags
    ma = float(np.dot(u, a))
    mb = float(np.dot(u, b))
    base = min(ca, cb)                     # shared flag load already on the table

    # configural dominance: one flag set is a subset of the other
    nested = bool(np.all(a <= b + 1e-9) or bool(np.all(b <= a + 1e-9)))

    # ---------------- 5. decision ----------------
    if abs(d) > 1e-9:
        # sign-dominant (saturating) count evidence, positive => choose A
        t_c = -np.sign(d) * (abs(d) ** rho)
        # validity-weighted flag-mass evidence, positive => choose A
        t_m = mu * (mb - ma)
        V = (1.0 - phi) * t_c + phi * t_m

        if nested:
            # dominance read: shared flags are aligned, so they do not interfere
            gain = 1.0 + nu
        else:
            # cross-cutting sets: small additive interference from shared flags
            gain = max(0.25, 1.0 - kappa * min(base, 6.0))
        z = float(np.clip(beta * gain * V, -50.0, 50.0))
        pA = 1.0 / (1.0 + np.exp(-z))
        p_core = np.array([pA, 1.0 - pA], dtype=float)
    else:
        # ---------- 6. exact count tie: load-graded worst-flag repulsion ----------
        owner = None
        order = np.argsort(-val, kind="stable")
        for j in order:
            if a[j] > b[j] + 1e-9:
                owner = 0
                break
            if b[j] > a[j] + 1e-9:
                owner = 1
                break
        if owner is None:
            p_core = np.ones(2) / 2.0
        else:
            # repulsion weakens as more flags are already on the table
            tb = float(np.clip(tie_bias + tie_slope * min(ca, 4.0), 0.0, 0.5))
            p_core = np.empty(2, dtype=float)
            p_core[owner] = tb             # option carrying the top discriminating flag
            p_core[1 - owner] = 1.0 - tb

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------------- 7. attention lapse ----------------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
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
