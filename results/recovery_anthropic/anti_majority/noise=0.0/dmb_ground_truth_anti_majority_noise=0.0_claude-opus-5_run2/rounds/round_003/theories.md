# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** **Flag Counting with Lexicographic Tiebreak (FC-LEX).**

Subjects in this task do not read a binary expert rating of '1' as an endorsement; they read it as a *warning flag*. And, crucially, they do not weight those flags by the validities they were told about in the instructions. On every trial the subject performs a single, extremely cheap operation: count how many 1s each option carries, and reject the option carrying more flags.

Three claims:

1. **Negative polarity.** A '1' is a strike against an option (a flag/complaint/warning), so an option's *badness* is monotone in its number of 1s. The option with the strictly smaller flag count is chosen.

2. **Validity-blind, magnitude-blind counting.** All experts are counted equally: the validities are acknowledged in the instructions but are not converted into cue weights. Furthermore, only the *sign* of the count difference matters. Choice probability is therefore a single flat accuracy h = (1-eps)*sigmoid(beta) + eps/2 on every trial where the counts differ, regardless of whether the difference is 1 flag or 5 flags, and regardless of which experts contributed the flags. There is no psychometric function over a validity-weighted margin: a 1-vs-0 comparison and a 1-vs-5 comparison produce the same accuracy.

3. **Lexicographic tiebreak on equal counts.** When the two options carry exactly the same number of flags the counting rule is silent. The subject then falls back to a one-reason rule: locate the highest-validity expert who discriminates and let that single expert decide. Because the subject is unsure whether that expert's '1' should attract or repel (the counting rule just told them 1s are bad, but the tiebreak is invoked precisely because badness is balanced), the direction of this tiebreak is a free subject-level bias tie_bias, which sits near chance and is mildly repelled-by-1 on average. Equal-count pairs therefore yield near-chance performance, not the strongly graded reversals a weighted-mass model predicts.

The theory is deliberately non-compensatory in *magnitude* while being fully compensatory in *counting*: it agrees with a weighted-evidence account on every design tested so far (because in those designs the option with the single top-validity 1 also happens to carry the fewest flags), but it diverges sharply on equal-count pairs such as [1,0,0,0,0] vs [0,0,0,0,1], on pairs where more 1s coincides with smaller weighted mass, and on margin ladders, where FC-LEX predicts a flat accuracy plateau conditional on the sign of the count difference.

**Rationale:** I implemented exactly the mechanism family the arbiter prescribed (negative-polarity, validity-blind flag counting with a sign-only choice rule and a lexicographic tiebreak on equal counts), and then chose parameter ranges by solving the metric algebra analytically for each of the four experiments.

Working through the designs: (a) Exp-1's six TTB/tally conflict trials all have the TTB-favoured option carrying strictly FEWER 1s, so FC-LEX reproduces TTB there and the metric equals the flat accuracy h. (b) In Exp-2 every 'agreement' trial (|margin|>=3) has the TTB-favoured option carrying MORE 1s, so FC-LEX rejects it (chose_ttb = 1-h), while every qualifying 'conflict' trial has the TTB-favoured option carrying FEWER 1s (chose_ttb = h); the metric is therefore 1-2h. (c) Exp-4 gives h-(1-h)=2h-1 because the dominance pair [1,0,0,0,0] vs [0,0,0,0,0] is rejected by flag counting while the conflict pair [1,0,0,0,0] vs [0,1,1,1,1] is accepted. (d) Exp-3 reduces to 0.75 - 1.75h + 0.25*q, where q is the equal-count tiebreak hit rate contributed by the [1,0,0,0,0] vs [0,0,0,1,0] pair — the one place in the whole design set where a pure counting rule must hand over to the one-reason tiebreak.

Setting beta in [1.3, 2.7] and epsilon in [0.05, 0.25] gives a pooled flat accuracy h ~ 0.815, yielding predicted values 0.815 (Exp-1, real 0.774), -0.63 (Exp-2, real -0.634), +0.63 (Exp-4, real +0.660); tie_bias uniform on [0.10, 0.70] (mean 0.40, i.e. the top-cue '1' mildly repels on tie trials) gives Exp-3 ~ -0.58 against a real -0.577. All four signature contrasts are captured with three free parameters and a single mechanism.

Why this is a genuinely different and better-motivated theory than what came before: pi_1 (TTB) fails Exp-2/3/4 because it has no polarity claim; pi_2 (tallying) has positive polarity and gets every sign backwards; pi_3 (PWEA) fits, but it does so with a graded validity-weighted margin, which is an untested extra commitment. FC-LEX matches pi_3's fit on all existing designs while making a strictly cheaper and sharply falsifiable claim — accuracy depends only on the SIGN of an unweighted flag count, so it predicts a flat plateau where PWEA predicts a monotone psychometric function, and near-chance responding where PWEA predicts strong graded reversals on equal-count pairs. That contrast is directly testable in the next round of designs, which is the point of surfacing it now.

**Parameters:**
  - `beta`: `[1.3, 2.7]`
  - `epsilon`: `[0.05, 0.25]`
  - `tie_bias`: `[0.10, 0.70]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
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
    n_features = a.size

    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])
    tie_bias = float(parameters["tie_bias"])
    eps = min(max(eps, 0.0), 1.0)
    tie_bias = min(max(tie_bias, 0.0), 1.0)

    # ---------- 2. validity-blind flag count (negative polarity) ----------
    count_a = float(np.sum(a))
    count_b = float(np.sum(b))
    d = count_a - count_b   # positive => A carries MORE warning flags => reject A

    if abs(d) > 1e-9:
        # Sign-only rule: magnitude of the count difference is ignored.
        winner = 1 if d > 0 else 0   # fewer flags wins
        s = 1.0 / (1.0 + np.exp(-beta))   # sigmoid(beta * sign) for the winner
        p_core = np.zeros(2, dtype=float)
        p_core[winner] = s
        p_core[1 - winner] = 1.0 - s
    else:
        # ---------- 3. lexicographic tiebreak on equal flag counts ----------
        val = parameters.get("validities", None)
        try:
            val = np.asarray(val, dtype=float).ravel()
        except Exception:
            val = np.array([])
        if val.size != n_features:
            if n_features == 1:
                val = np.array([0.9])
            else:
                val = np.linspace(0.9, 0.55, n_features)

        order = np.argsort(-val, kind="stable")
        top1_owner = None   # index of the option carrying the '1' on the top discriminating cue
        for j in order:
            if a[j] > b[j]:
                top1_owner = 0
                break
            if b[j] > a[j]:
                top1_owner = 1
                break

        if top1_owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.zeros(2, dtype=float)
            p_core[top1_owner] = tie_bias
            p_core[1 - top1_owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------- 4. attention lapse ----------
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


### slot 2 — `pi_5` — SURVIVED ✓

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

### `pi_6` → slot 1 (via `new_theory`)

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
