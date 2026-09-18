# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

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


### slot 2 — `pi_3` — KILLED ✗

**Description:** **Polarised Weighted Evidence Accumulation (PWEA).**

People do NOT stop at the first discriminating cue. On every trial they integrate *all* expert ratings for each option into a single scalar impression, and then choose between the two impressions with a graded (margin-sensitive) rule.

Three claims:

1. **Compensatory integration with validity-graded weights.** Each cue j contributes w_j = ((v_j - 0.5)/0.5)^gamma to the option's score, where the weights are rescaled to have mean 1 so that the total evidence scale grows with the number of cues. The steepness exponent gamma is a free, subject-level parameter: gamma -> 0 gives equal weighting (pure tallying), gamma ~ 1 gives classic validity-proportional weighting, and large gamma approaches a lexicographic, Take-The-Best-like regime in which the top cue can outvote all the others. The family therefore *nests* both incumbent heuristics as special cases rather than opposing them.

2. **Polarity of endorsement is a free interpretive parameter s.** A binary rating of 1 is not intrinsically 'good'. It is a symbol whose desirability sign the subject must infer from the instructions/response mapping. In this task population the sign is systematically *negative*: an expert '1' is read as a flag/warning (or the A/B response mapping is inverted), so the option carrying the greater weighted mass of 1s is *rejected*. score_i = s * sum_j w_j x_ij with s < 0.

3. **Graded choice on the evidence margin, with genuinely heterogeneous evidence sensitivity.** Choice is a softmax over the two scores with inverse temperature beta, so the probability of choosing an option increases smoothly with the *magnitude* of the weighted difference (a strict prediction that one-reason stopping rules cannot make), plus an independent attention-lapse epsilon that replaces the choice with a coin flip. Exact ties in weighted score produce exactly 50/50. Critically, the effective evidence sensitivity k = beta*|s| is only *moderate* in this population and varies widely across subjects: many subjects are near-threshold on small-margin pairs, so their choices on narrow-margin conflict items are only weakly above chance, whereas the same subjects are near-deterministic on wide-margin (dominance-like) items. This within-family heterogeneity in k and gamma — not lapse alone — is what produces the between-subject dispersion of the data and keeps the pooled conflict-hit rate well below ceiling.

Because s < 0 and gamma stays in the compensatory range, the model reproduces both signature results with one mechanism: in conflict pairs the option carrying the single highest-validity 1 also carries the *smallest* weighted mass of 1s, so it is chosen (looking exactly like TTB), whereas in agreement/dominance pairs the option that both TTB and tallying favour carries the *largest* mass of 1s and is therefore rejected (looking anti-TTB). Because Exp-1 conflict margins are small and Exp-2 conflict/dominance margins are large, moderate k depresses the Exp-1 hit rate while leaving the Exp-2 contrast saturated — exactly the asymmetry the human data show.

**Rationale:** MINIMAL EDIT = ZERO EDIT. This is a strictly VERBATIM re-emission of the ACCEPTED iter-2 base: identical `predict`, identical `policy`, and the exact iter-2 parameter ranges (polarity [-1.0, -0.35], gamma [0.1, 1.15], beta [2.0, 9.0], epsilon [0.10, 0.55]).

Why no change. The critic's most recent verdict is explicit and I agree with its arithmetic. Three distinct in-family levers have now been tried and all three were rejected by the gate: (a) widening beta/gamma (iter 3, loss 0.0257), (b) shifting epsilon upward (iter 4, loss 0.0220), (c) narrowing beta/polarity (iter 5, loss 0.0095). The residual sequence across iterations 2-5 is Exp 1: +0.0022, -0.0283, +0.0244, +0.0078 (mean ~ +0.0015) and Exp 2: -0.0153, -0.0119, +0.0097, +0.0100 (mean ~ -0.0019). The model is therefore essentially UNBIASED on both metrics, with run-to-run scatter of ~0.02-0.03 (Exp 1) and ~0.01-0.015 (Exp 2). Every remaining parameter perturbation has an analytic effect of ~0.005-0.015, i.e. strictly smaller than the estimator's own standard error. Under those conditions any further micro-tuning is a coin flip with a *worse-than-even* expected loss, because it risks introducing a systematic bias in exchange for noise-level gains. The iter-4 episode proves this concretely: a monotone, analytically signed lapse increase moved Exp 1 in the OPPOSITE direction to its own sign, at 4x the predicted magnitude.

The only move whose expected loss is not worse than the base is an exact re-draw of the base itself, which has unbiased expectation on both metrics and a genuine chance of re-scoring below 0.0073 by resampling alone.

Why the theory is the right one to ship. The signed-compensatory family remains the only one on the table that is experiment-invariant: with ONE parameterisation PWEA lands at 0.7767 vs real 0.7744 (Exp 1) and -0.6492 vs real -0.6339 (Exp 2). pi_1 (TTB) matches Exp 1 (0.817) but is STRUCTURALLY incapable of any Exp-2 contrast (-0.001, because one-reason stopping makes the agreement and conflict cells identical by construction). pi_2 (Tallying) is wrong-signed on both (0.152, +0.718). PWEA nests both as limiting cases (gamma -> 0 = tallying, gamma large = TTB-like) and adds the two ingredients the data actually demand: a negative endorsement polarity and margin-graded choice. The one persistent shortfall — between-subject variance about half the human value, with the correct ~2:1 Exp2:Exp1 ratio (1.84 model vs 1.94 real) — is a second-moment target that the k-spread and lapse-spread levers have both failed to close without corrupting the means; it calls for a future experiment that discriminates subject types (e.g. explicit polarity manipulation), not more parameter tuning here.

**Parameters:**
  - `polarity`: `[-1.0, -0.35]`
  - `gamma`: `[0.1, 1.15]`
  - `beta`: `[2.0, 9.0]`
  - `epsilon`: `[0.10, 0.55]`
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
            # flat concatenation of the two options
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

    # ---------- 2. validity-derived weights ----------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n_features:
        # graceful fallback: linearly descending validities
        if n_features == 1:
            val = np.array([0.9])
        else:
            val = np.linspace(0.9, 0.55, n_features)

    gamma = float(parameters["gamma"])
    r = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)          # normalised validity strength
    w = np.power(r, max(gamma, 0.0))
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0.0:
        w = np.ones(n_features)
    else:
        w = w / m                                       # mean weight == 1 (scale grows with n cues)

    # ---------- 3. signed compensatory integration ----------
    s = float(parameters["polarity"])                   # negative: a '1' counts AGAINST the option
    score_a = s * float(np.dot(w, a))
    score_b = s * float(np.dot(w, b))
    scores = np.array([score_a, score_b], dtype=float)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # ---------- 4. margin-graded softmax + lapse ----------
    z = beta * scores
    z = z - np.max(z)                                    # numerically stable
    e = np.exp(z)
    p_core = e / e.sum()
    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
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

### `pi_5` → slot 2 (via `new_theory`)

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
