# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with well-spread validities (0.90, 0.80, 0.70, 0.60, 0.55) so that the validity ordering is unambiguous and TTB's cue cascade is well defined. The design contains three diagnostic trial classes plus controls, each presented in both A/B mirror forms to remove side bias.

(1) CONFLICT trials (pairs 1-6): the highest discriminating cue favors one option while the majority of remaining cues favors the other. E.g. A=[1,0,0,0,0] vs B=[0,1,1,1,1]: TTB stops on expert 1 and picks A with probability ~sigmoid(beta), whereas Tallying counts 1 vs 4 and picks B with probability ~sigmoid(3*beta) \u2014 the two theories predict opposite responses, and Tallying's tally margin (3) is large so its prediction is nearly deterministic for any moderate beta. Pairs 3-4 use a smaller tally margin (1 vs 2), so Tallying predicts a *weaker* (margin-1 softmax) preference for the option TTB rejects; the graded difference across margin sizes is a quantitative signature that separates the models beyond mere direction.

(2) TALLY-TIE trials (pairs 7-10): the feature-wise wins are exactly balanced, so Tallying is forced to 50/50 regardless of beta, while TTB still discriminates decisively. Pairs 7-8 tie 2-2 with the top cue discriminating (TTB confident); pairs 9-10 tie 1-1 with all three high-validity cues equal so TTB is driven by the low-validity expert 4. If subjects show systematic (non-chance) preferences on these trials, Tallying cannot account for it at any parameter setting.

(3) CONTROL trials (pairs 11-14) where both theories agree, providing clean leverage on beta and epsilon so that the diagnostic trials are interpreted with well-identified noise parameters.

Pair 5-6 additionally makes the top cue non-discriminating (both options rated 1 by expert 1), forcing TTB down to expert 2 while Tallying reads a 1-vs-3 margin the other way \u2014 testing whether people truly stop at the first *discriminating* cue rather than integrating.

Advocated theory (TTB) predicts: choice of the option favored by the highest discriminating cue on ALL of pairs 1-10, with roughly constant accuracy independent of how many lower cues oppose it, and above-chance systematic choice on the tally-tie pairs. Tallying predicts the opposite direction on pairs 1-6 with strength graded by tally margin, and exact chance on pairs 7-10.

[1] I advocate TALLYING and design for a quantitative dissociation that the previous experiment (which emphasized direction reversals only) did not isolate: the *graded* dependence of choice probability on the tally margin versus TTB's flat, margin-independent confidence.

Six experts with clearly ordered, well-spread validities (0.90, 0.82, 0.74, 0.66, 0.58, 0.53) make TTB's cue cascade unambiguous (cue order = feature order). Every pair appears in both A/B mirror forms (16 unique pairs, K=6 -> 96 trials) so side bias cancels.

(1) GRADED-AGREEMENT trials (pairs 1-6). Both models favour the same option, but the tally margin is +1 (3 wins vs 2), +3 (4 vs 1), and +5 (5 vs 0). TTB always stops at expert 1 and therefore predicts EXACTLY the same accuracy, (1-eps)*sigmoid(beta)+eps/2, in all three conditions; its binary score never changes. Tallying predicts monotonically increasing accuracy, sigmoid(1*beta) < sigmoid(3*beta) < sigmoid(5*beta). Because these conditions also pin down beta and epsilon, a reliable accuracy gradient across margin 1 -> 3 -> 5 cannot be absorbed by any TTB parameter setting: TTB is forced to attribute all three to one sigmoid(beta). This is the core quantitative signature, and it is measured on trials where the two theories agree in DIRECTION, so it is not contaminated by any direction-reversal strategy shift.

(2) CONFLICT trials (pairs 7-12). Expert 1 favours one option while the remaining experts favour the other, with tally margins of -1 (2 vs 3), -3 (1 vs 4) and -4 (1 vs 5). TTB predicts choice of the expert-1 option with the SAME probability in all three; Tallying predicts choice of the opposite option with strength graded by margin. Direction and gradient are both diagnostic here, and the -1 case is the crucial one: Tallying predicts a weak reversal (sigmoid(beta)) exactly matching in magnitude what it predicts for the +1 agreement pairs.

(3) DESCENT trial (pairs 13-14): experts 1 and 2 both rate the two products 1, so TTB must descend to expert 3 and choose that option, while the tally is 1 vs 3 the other way. This tests whether people truly stop at the first discriminating cue or integrate across the three low-validity experts.

(4) TALLY-TIE trial (pairs 15-16): A wins on experts 1-2, B wins on experts 3-4 (2-2 tie). Tallying is forced to exactly 50/50 for any parameters; TTB predicts a decisive expert-1-based preference. This is the competing theory's unique prediction and provides a hard falsification test of my advocated model; including it makes the design fair rather than stacked.

Predictions if Tallying is right: accuracy rises steeply with |tally margin| in both the agreement and conflict blocks, choices reverse against expert 1 on pairs 7-12 and 13-14, and choices sit at chance on pairs 15-16. If TTB is right: flat accuracy across all margins, expert-1 option chosen throughout pairs 1-12, expert-3 option on 13-14, and clearly above-chance systematic choice on the tally-tie pairs.

[2] I advocate TAKE-THE-BEST (TTB) against Polarised Weighted Evidence Accumulation (PWEA). PWEA makes two commitments that TTB flatly denies: (i) a rating of '1' is read as a NEGATIVE flag (polarity s<0), so the option carrying the greater validity-weighted mass of 1s is REJECTED; and (ii) choice probability scales smoothly with that weighted mass margin. TTB instead says the option holding the '1' on the highest discriminating expert is chosen, with a single, margin-independent accuracy (1-eps)*sigmoid(beta) on every discriminating trial.

Crucially, the two earlier experiments in this series pitted TTB against Tallying, using only 'conflict' pairs of the form A=[1,0,0,0,0] vs B=[0,1,1,1,1]. On exactly those pairs PWEA MIMICS TTB (the top-cue option also carries the least mass of 1s), so nothing so far can separate them. This design therefore concentrates on the trial class where PWEA and TTB predict OPPOSITE choices: pairs where the option favoured by the top expert also carries MORE 1s.

Five experts with well-spread, strictly ordered validities (0.93, 0.84, 0.72, 0.62, 0.54) make TTB's cue cascade unambiguous. gamma=1 weights (mean-1 normalised) are w=[1.87, 1.48, 0.96, 0.52, 0.17]. All 8 base pairs are shown in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials) so any side bias cancels.

CLASS I - POLARITY REVERSAL, LARGE MARGIN (pairs 1-2 and mirrors 9-10):
[1,1,1,1,1] vs [0,0,0,0,0] (mass 5.00 vs 0) and [1,1,1,0,0] vs [0,0,0,1,1] (4.31 vs 0.70). TTB picks the all-/mostly-1 option essentially deterministically; PWEA picks the OPPOSITE option essentially deterministically for every admissible parameter setting (any gamma, any s<0, beta>=2). These are the hard falsification trials: they cannot be reconciled with PWEA by any parameter tuning, and equally, a reliable preference for the sparse-1 option would kill TTB outright.

CLASS II - GRADED REVERSAL (pairs 3, 4, 6 and mirrors): [1,0,0,0,0] vs [0,0,0,0,0] (PWEA margin 1.87), [0,0,0,0,1] vs [0,0,0,0,0] (margin 0.17 at gamma=1), and [1,0,0,0,0] vs [0,0,0,1,0] (margin 1.35). TTB predicts the SAME accuracy toward the option with the single 1 (or with the higher-validity 1) in all three, because only the sign on the first discriminating cue matters. PWEA predicts the opposite direction with strength graded ~10:1 between the top-cue and bottom-cue trials. So the design delivers a within-subject direction test AND a magnitude-gradient test on the same trial family: flat, above-chance, top-cue-consistent accuracy across margins is uniquely TTB; a margin-graded reversal is uniquely PWEA.

CLASS III - CASCADE / NEAR-TIE TRIAL (pair 7 and mirror): [1,1,0,0,0] vs [1,0,1,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and decisively picks A. PWEA's weighted masses are 3.35 vs 3.52 at gamma=1 (and reverse to favour A at gamma<=0.5): across essentially the whole admissible gamma range PWEA is pinned near 50/50 here, whereas TTB predicts full, undiminished accuracy. Systematic choice of A on this pair is diagnostic of one-reason stopping and cannot be produced by PWEA at any evidence sensitivity.

CLASS IV - AGREEMENT / CALIBRATION (pairs 5 and 8, plus mirrors): [1,0,0,0,0] vs [0,1,1,1,1] and [1,0,0,0,0] vs [0,1,1,0,0]. Here both theories favour the same option (the classic TTB-vs-Tallying conflict items on which PWEA mimics TTB). They pin down beta and epsilon for BOTH models independently of the diagnostic classes, so the reversal trials are interpreted against well-identified noise parameters, and they guarantee the design is not simply stacked toward one family.

PREDICTIONS. If TTB is right: on every pair the option bearing the '1' on the highest discriminating expert is chosen at a single, roughly constant above-chance rate, including the mostly-1 dominance pairs (Class I), the bottom-cue pairs (Class II), and the cascade pair (Class III). If PWEA is right: choices REVERSE on Classes I-II (sparse-1 option chosen), near-deterministically on Class I and only weakly on the small-margin Class II items, sit at chance on Class III, and agree with TTB only on Class IV.

[3] I advocate Polarised Weighted Evidence Accumulation (PWEA) against Take-The-Best (TTB). The two prior TTB-vs-PWEA designs contrasted a few isolated reversal pairs; this design instead builds a *continuous margin ladder* on which TTB is mathematically forced to a single flat accuracy while PWEA must produce a monotone psychometric function with a parameter-robust crossing point, plus a gamma-graded pair and a robust weighted-tie pair.

Validities (0.95, 0.80, 0.70, 0.62, 0.55) are strictly ordered so TTB's cue cascade is unambiguous. At gamma=1 the mean-1 normalised weights are w=[2.01, 1.34, 0.89, 0.54, 0.22]; at gamma=0.1 they are nearly flat, w=[1.10, 1.05, 1.01, 0.96, 0.88]. Every base pair appears in both A/B orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

LADDER (base pairs 1-4, all of the form A=[1,0,0,0,0] versus a B with progressively more low-validity 1s): B = [0,0,0,0,0] / [0,1,0,0,0] / [0,1,1,0,0] / [0,1,1,1,1]. On ALL FOUR, expert 1 discriminates in favour of A, so TTB predicts exactly the same accuracy (1-eps)*sigmoid(beta) toward A on each; nothing in TTB can make these four differ. PWEA's signed weighted-mass margin (A minus B) is +2.01, +0.67, -0.22, -0.98 at gamma=1 and +1.10, +0.04, -0.92, -2.80 at gamma=0.1: monotone in the same order for the whole admissible gamma range, and with s<0 this yields a monotonically INCREASING probability of choosing A across pairs 1->4, crossing 50% between pair 2 and pair 3 for every gamma in [0.1, 1.15]. A four-point graded curve with a sign flip located between pairs 2 and 3 is uniquely PWEA; flat above-chance A-choice is uniquely TTB. Pair 4 is the classic 'conflict' item on which the two families agree in direction, so it also pins down beta/epsilon for both models.

WEIGHT-vs-COUNT GRADIENT (pairs 6 and 7): [1,0,0,0,0] vs [0,0,0,0,1] and [0,0,1,0,0] vs [0,0,0,1,0]. Both have one 1 per option, so TTB again predicts identical accuracy toward the higher-validity-1 option (A) in both. PWEA predicts the opposite direction (B) with margins in an approximately 5:1 ratio (1.79 vs 0.36 at gamma=1), collapsing toward chance for both as gamma->0. These items therefore separate direction (PWEA vs TTB) and simultaneously identify gamma, which no TTB parameter can mimic.

CASCADE REVERSAL (pair 5): [1,1,1,0,0] vs [1,0,0,1,1]. Expert 1 does not discriminate, so TTB descends to expert 2 and picks A decisively. PWEA's margin (w2+w3-w4-w5) is positive for every gamma (1.47 at gamma=1, 0.22 at gamma=0.1), so PWEA chooses B, strongly if gamma is near 1 and weakly if gamma is near 0. This tests one-reason stopping against signed integration on a trial where the top cue is uninformative.

ROBUST WEIGHTED TIE (pair 8): [0,1,0,0,1] vs [0,0,1,1,0]. Counts are equal and the weighted masses nearly cancel across the whole gamma range (|margin| <= 0.19 for gamma in [0.1, 1.15], and it even changes sign), so PWEA is pinned essentially at chance for any beta and |s|. TTB, however, stops at expert 2 and picks A with its full, undiminished accuracy. Systematic A-choice here would falsify PWEA; near-chance responding here combined with decisive responding on pairs 1 and 5 would falsify TTB, because TTB's accuracy must be constant across all discriminating trials.

PREDICTIONS. PWEA: choice of A rises monotonically across ladder pairs 1->4 with the crossover between 2 and 3; strong B-preference on 6, weak B-preference on 7 (ratio ~5:1); B-preference on 5; chance on 8; overall dispersion driven by subject-level k=beta*|s| and gamma. TTB: a single, roughly constant above-chance preference for A on every one of pairs 1-8 (and for the mirrored option in the mirrors), with no dependence on weighted mass, cue count, or margin.

[4] I advocate FC-LEX (validity-blind flag counting with a lexicographic tiebreak) against PWEA (validity-weighted, margin-graded signed integration). Both models agree that a '1' is a warning flag, so direction alone is often non-diagnostic; the decisive dissociations are (a) MAGNITUDE INVARIANCE and (b) COMPOSITION INVARIANCE of the count rule, plus (c) the equal-count class.

Validities are deliberately skewed: (0.97, 0.66, 0.60, 0.56, 0.52). At gamma=1 the mean-1-normalised PWEA weights are w=[2.90, 0.99, 0.62, 0.37, 0.12] (a ~24:1 top-to-bottom ratio); at gamma=0.1 they are nearly flat (~[1.19,1.04,1.02,1.00,0.94]). This skew makes PWEA's weighted mass diverge sharply from raw flag counts for moderate/high gamma, while low gamma collapses PWEA onto tallying (which is direction-identical to counting) - so the design simultaneously separates the families and identifies gamma.

CLASS 1 - FLAT PLATEAU / MARGIN LADDER (pairs 1-2, 5-6, 7-8). A=[0,0,0,0,0] against B with 1, 4, or 5 top-loaded flags. All three have a count difference in the same direction, so FC-LEX predicts EXACTLY the same accuracy (1-eps)*sigmoid(beta)+eps/2 on all of them - no parameter of FC-LEX can make them differ. PWEA's weighted margins are 2.90, 1.80, 5.00 at gamma=1 and 1.19, 4.00, 5.19 at gamma=0.1: for every admissible gamma the [0,1,1,1,1] and [1,1,1,1,1] items carry a much larger margin than a single low-weight flag, so PWEA must show a monotone psychometric gradient.

CLASS 2 - COMPOSITION INVARIANCE (pairs 1-2 vs 3-4). Both are 0-flags-versus-1-flag, but the flag sits on the 97% expert in one case and on the 52% expert in the other. FC-LEX: identical accuracy (counting is validity-blind). PWEA at gamma=1: margin 2.90 versus 0.12, i.e. near-deterministic versus essentially chance - a ~24:1 evidence ratio on trials that FC-LEX must treat as identical. This is the single cleanest quantitative test in the design, because direction is the same for both models, so no strategy-reversal confound can explain a gradient.

CLASS 3 - COUNT-vs-WEIGHT REVERSAL (pairs 9-10: [1,0,0,0,0] vs [0,1,1,1,1]; pairs 11-12: [1,1,0,0,0] vs [0,0,1,1,1]). Here more flags coincides with LESS weighted mass. FC-LEX chooses the fewer-flag option (A in 9 and 11) at its usual flat rate. PWEA with gamma near 1 chooses the OPPOSITE option (masses 2.90 vs 1.80 and 3.89 vs 1.11), while with gamma near 0.1 it agrees with FC-LEX. A reliable preference for the many-flags option here is impossible for FC-LEX at any parameter setting and pins gamma for PWEA.

CLASS 4 - EQUAL-COUNT TIEBREAK (pairs 13-14: [1,0,0,0,0] vs [0,0,0,0,1]; pairs 15-16: [0,0,1,0,0] vs [0,0,0,1,0]). Counts are tied, so the counting rule is silent and FC-LEX falls back on the top discriminating expert with a subject-level bias tie_bias near chance - and, critically, the SAME p(top-flag option) in both pairs, because both are decided by whichever expert discriminates first. PWEA instead predicts a strongly graded reversal: margin 2.78 for pair 13 versus 0.25 for pair 15 at gamma=1 (a >10:1 ratio), collapsing toward chance for both as gamma->0. So Class 4 gives a within-subject magnitude contrast on trials where FC-LEX is committed to invariance.

All 8 base pairs appear in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly, and pairs 7-8 (dominance) plus pairs 1-2 provide clean identification of beta/epsilon for FC-LEX and of beta*|s|/epsilon for PWEA.

PREDICTIONS. FC-LEX: one flat above-chance accuracy toward the fewer-flag option across pairs 1-12 regardless of margin size (1 vs 4 vs 5 flags) and regardless of which expert carries the flag; near-chance, magnitude-invariant, subject-idiosyncratic responding on pairs 13-16. PWEA: monotone accuracy growth with weighted margin in Class 1, a large accuracy gap between top-cue and bottom-cue single-flag items in Class 2, direction reversals in Class 3 for moderate/high gamma, and a strongly graded (not invariant) split between pairs 13-14 and 15-16.

[5] I advocate PWEA (validity-weighted, margin-graded signed integration) against FC-LEX (validity-blind flag counting with a single lexicographic tiebreak). Both models agree that a '1' is a warning flag, so raw direction is often non-diagnostic; the decisive tests are (a) FC-LEX's magnitude invariance, (b) its composition invariance, and (c) the fact that FC-LEX must use ONE subject-level tie_bias, expressed in the frame of the top discriminating expert, on every equal-count pair.

Validities (0.90, 0.86, 0.84, 0.60, 0.52) are deliberately clustered high-high-high / low-low. At gamma=1 the mean-1-normalised PWEA weights are w=[1.639, 1.475, 1.393, 0.410, 0.082] (a 20:1 top-to-bottom ratio); at gamma=0.1 they flatten to ~[1.09, 1.08, 1.07, 0.95, 0.81]. Crucially the *three-cue clustering* means several of my diagnostic contrasts survive the entire admissible gamma range, so they are not just gamma-identification items.

CLASS 1 - THE TIEBREAK-FRAME CONTRADICTION (base pairs 1-3, the novel core of this design). All three have EQUAL flag counts, so FC-LEX's counting rule is silent and it must fall back on the highest discriminating expert, giving p(option holding the top-cue '1') = tie_bias, the SAME number on all three pairs for a given subject.
  P1: [1,0,0,0,1] vs [0,1,1,0,0]. Top discriminating expert is #1, owned by A. But A's weighted mass (1.639+0.082=1.72) is SMALLER than B's (1.475+1.393=2.87), so PWEA chooses A. This inequality holds for every gamma in [0.1,1.15] (at gamma=0.1: 1.70 vs 1.93; at gamma=1.15: 0.80 vs 1.32 in raw units), so it is parameter-robust.
  P2: [1,0,0,0,0] vs [0,0,0,0,1]. Top discriminating expert is again #1, again owned by A, but now A carries the larger mass (1.639 vs 0.082), so PWEA chooses B decisively.
  Therefore PWEA predicts OPPOSITE directions on P1 and P2 while FC-LEX is mathematically forced to the identical p(A)=tie_bias on both. No FC-LEX parameter setting can produce this reversal; it is the single cleanest falsification test in the design.
  P3: [0,1,0,0,0] vs [0,0,1,0,0]. Weighted masses are 1.475 vs 1.393 - essentially tied for all gamma - so PWEA is pinned at ~50/50 for any beta and |s|, whereas FC-LEX descends to expert #2 (owner A) and predicts p(A)=tie_bias, which may be as extreme as 0.10 or 0.70. Near-chance responding here combined with decisive, oppositely-signed responding on P1 and P2 is uniquely PWEA.

CLASS 2 - COMPOSITION INVARIANCE (P4 vs P5). Both are 0-flags-versus-1-flag, so FC-LEX predicts identical accuracy (validity-blind counting). PWEA's margin is 1.639 (flag on the 90% expert) versus 0.082 (flag on the 52% expert) at gamma=1 - a 20:1 evidence ratio, i.e. near-deterministic versus near-chance. Direction is the same for both models here, so a reliable gradient cannot be attributed to any strategy reversal.

CLASS 3 - MARGIN LADDER / PLATEAU TEST (P5, P4, P6: margins 0.08, 1.64, 5.00 with the count difference always in the same direction). FC-LEX predicts one flat plateau accuracy (1-eps)*sigmoid(beta)+eps/2 across all three; PWEA must produce a monotone psychometric function. P6 ([0,0,0,0,0] vs [1,1,1,1,1]) also anchors beta*|s| and epsilon for PWEA and beta/epsilon for FC-LEX.

CLASS 4 - COUNT-vs-WEIGHT REVERSAL (P7: [1,0,0,0,0] vs [0,0,0,1,1]; P8: [1,1,0,0,0] vs [0,0,1,1,1]). Here MORE flags coincide with LESS weighted mass (1.639 vs 0.492; 3.114 vs 1.885 at gamma=1). FC-LEX must choose the fewer-flag option A at its flat rate; PWEA chooses B for every gamma above about 0.35 and agrees with FC-LEX only as gamma->0.1. Any reliable preference for the many-flag option is impossible for FC-LEX at any parameter setting, and the graded difference between P7 and P8 further pins gamma.

All 8 base pairs appear in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

PREDICTIONS. PWEA: A chosen on P1 but B chosen on P2 (a within-subject sign flip that FC-LEX's single tie_bias cannot express); chance on P3; steeply graded accuracy across P5 < P4 < P6; a large P4-vs-P5 composition gap; reversals toward the many-flag option on P7-P8 for moderate/high gamma. FC-LEX: one flat above-chance rate toward the fewer-flag option on P4-P8 irrespective of margin size or which expert carries the flag, and the SAME, magnitude-invariant p(top-cue-flag option)=tie_bias on all three equal-count pairs P1-P3.

[6] I advocate FC-LEX (sign-only, validity-blind flag counting with a lexicographic tiebreak) against WC-ST (Weber-scaled flag counting with a threshold-like Weibull psychometric plus discrete expert triage). The two models agree completely on POLARITY (a '1' is a warning flag, fewer flags wins) and on VALIDITY-BLINDNESS of the count, and they use the identical severity/lexicographic tiebreak on exact ties. So direction-only tests are useless here. The whole design is built on the ONE place they disagree: FC-LEX says only the SIGN of the count difference matters (one flat plateau accuracy on every unequal-count trial), while WC-ST says discriminability is ratio-scaled, s=(c_hi-c_lo)/sqrt(c_hi^2+c_lo^2+1), pushed through an accelerating Weibull, so accuracy must collapse to chance when the count ratio is small.

Six experts with well-spread validities (0.92, 0.86, 0.78, 0.68, 0.60, 0.54). Exactly two experts fall below the 0.65 'triage' threshold (experts 5 and 6), so WC-ST's minority triage mechanism is well defined and testable. All 8 base pairs appear in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

CLASS 1 - CONSTANT-DIFFERENCE MARGIN LADDER (P1 0-vs-1, P2 1-vs-2, P3 2-vs-3, P4 3-vs-4). Every pair has a count difference of exactly ONE flag in the same direction (A always carries fewer). FC-LEX is mathematically forced to the SAME accuracy (1-eps)*sigmoid(beta)+eps/2 on all four; no FC-LEX parameter can make them differ. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, which through the Weibull (theta in [0.34,0.46], kappa in [4.5,7.5]) gives roughly 1.00, 0.84, 0.54, 0.51 before lapses: a steep monotone collapse from near-ceiling to essentially chance. A reliable within-subject accuracy drop from 0-vs-1 to 3-vs-4 falsifies FC-LEX; a flat plateau across the ladder cannot be produced by WC-ST at any theta/kappa, because its psychometric function is strictly increasing in s.

CLASS 2 - CONSTANT-SMALLER-COUNT CONTRAST (P3 2-vs-3 versus P5 2-vs-4). Same smaller count, different difference. FC-LEX: identical accuracy. WC-ST: s = 0.267 vs 0.436, i.e. near-chance versus ~0.90 core. This is a second, orthogonal magnitude test that shares direction between models, so no strategy-reversal confound can explain a gradient.

CLASS 3 - TRIAGE REVERSAL (P6: A=[1,1,0,0,0,0] two flags on the 92%/86% experts vs B=[0,0,0,1,1,1] three flags on the 68%/60%/54% experts). Counting subjects choose A (2<3); WC-ST triage subjects discard experts 5-6 and read 2-vs-1, choosing B, and n_drop=1 subjects read 2-vs-2 and fall to the severity tiebreak, which also pushes them toward B. FC-LEX must choose A at its usual flat rate for every subject. Bimodal between-subject dispersion, or a group mean pulled to/below chance on P6 while P1 stays near ceiling, is uniquely WC-ST.

CLASS 4 - EQUAL-COUNT PAIRS (P7: [0,0,0,0,1,1] vs [1,1,0,0,0,0], 2-vs-2 with the flags on opposite ends of the validity scale; P8: [1,0,0,0,0,0] vs [0,1,0,0,0,0], 1-vs-1 control). Both models are silent on counting and fall back on the top discriminating expert, so P8 cleanly identifies tie_bias for both. P7 is diagnostic only for WC-ST's triage subgroup (dropping experts 5-6 turns it into 0-vs-2, a decisive A choice), whereas FC-LEX must give exactly the same p(top-flag option)=tie_bias on P7 and P8. A systematically more extreme response on P7 than P8, in a subset of subjects, is a triage signature.

Ladder pairs P1-P3 carry no flags on the droppable experts (triage-immune), and P4-P5 are constructed so that triage preserves the direction of the count difference, so the magnitude ladder is not confounded by the triage mechanism.

PREDICTIONS. FC-LEX: one flat, above-chance accuracy toward the fewer-flag option on P1-P6 regardless of whether the comparison is 0-vs-1 or 3-vs-4, plus the same magnitude-invariant tie_bias on P7 and P8. WC-ST: a steep monotone ratio-driven gradient (P1 near ceiling > P5 > P2 >> P3 ~ P4 ~ chance), and a bimodal, partly reversed response on P6 with an accompanying P7-vs-P8 asymmetry.

[7] I advocate WC-ST (Weber-scaled flag counting with a threshold-like Weibull, severity tiebreak, and discrete expert triage) against FC-LEX (sign-only, validity-blind flag counting with a lexicographic tiebreak). The two models are identical in polarity (a '1' is a warning flag; the option with fewer flags is chosen), identical in validity-blindness of the count, and identical in their tiebreak on exact ties. So direction tests are worthless. Every pair here targets the single point of disagreement: FC-LEX says only the SIGN of the count difference matters (one flat plateau accuracy on every unequal-count trial, for any beta/eps), whereas WC-ST says discriminability is RATIO-scaled, s=(c_hi-c_lo)/sqrt(c_hi^2+c_lo^2+1), pushed through an accelerating Weibull, so accuracy must fall to chance when the count ratio is small and must be INVARIANT across pairs with equal ratio but very different raw differences.

Eight experts, well spread (0.93, 0.88, 0.82, 0.75, 0.70, 0.62, 0.57, 0.52). Exactly the two lowest (0.57, 0.52 at indices 6 and 7) are the ones WC-ST's triage subjects can discard (val<0.65 and only the 1-2 least valid are dropped), so triage is sharply defined. All 9 base pairs appear in both A/B mirror orders (18 unique pairs, K=5 -> 90 trials), so side bias cancels exactly.

CLASS 1 - CONSTANT-DIFFERENCE LADDER (P1 0v1, P2 1v2, P3 2v3, P4 3v4; all flags confined to experts 1-6 so triage cannot touch them). The count difference is exactly ONE flag in the same direction on all four, so FC-LEX is mathematically forced to the identical accuracy (1-eps)*sigmoid(beta)+eps/2 on all four. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, giving core accuracies of about 1.00, 0.84, 0.54, 0.51 for mid-range theta=0.40, kappa=6: a steep, monotone collapse from ceiling to chance. A within-subject accuracy drop from 0v1 to 3v4 is impossible for FC-LEX at any parameters; a flat plateau is impossible for WC-ST because its psychometric function is strictly increasing in s.

CLASS 2 - RATIO INVARIANCE AT GROWING RAW DIFFERENCE (P2 1v2, P5 2v4, P6 3v6; ratios all 1:2, raw differences 1, 2, 3). WC-ST's s = 0.408, 0.436, 0.442 -- essentially identical, so it predicts these three to be statistically indistinguishable and all clearly ABOVE P3 (2v3, difference 1, s=0.267) and P4 (3v4, difference 1, s=0.196), even though P3/P4 involve larger counts and P4 has the same raw difference as P2. FC-LEX predicts all five of P2-P6 equal. A linear/Gaussian margin account would predict P6>P5>P2 and P4~P2. Only WC-ST predicts the exact pattern {P2 ~ P5 ~ P6} >> {P3 > P4 ~ chance}. This is the core quantitative signature and is measured on trials where both theories agree in DIRECTION, so no strategy-reversal confound can explain a gradient.

CLASS 3 - TRIAGE REVERSAL (P7: A=[1,1,0,0,0,0,0,0], two flags on the 93%/88% experts, vs B=[0,0,0,0,0,1,1,1], three flags on the 62%/57%/52% experts). Plain counters (both models) choose A because 2<3. WC-ST triage subjects who discard experts 7 and 8 read 2-vs-1 and choose B at s=0.408 (~0.84 core); subjects who discard only expert 8 read 2-vs-2, fall to the severity tiebreak keyed on expert 1 (owned by A) and, since tie_bias<0.5, are also pushed toward B. FC-LEX must choose A at its flat plateau rate for every subject. Bimodal between-subject dispersion on P7, or a group mean pulled toward/below chance on P7 while P1 stays near ceiling, is uniquely WC-ST.

CLASS 4 - EQUAL-COUNT PAIRS (P8: [1,0,...] vs [0,1,0,...], a clean 1-vs-1 tie; P9: A=[0,0,0,0,0,0,1,1] (2 flags on the two droppable experts) vs B=[1,1,0,0,0,0,0,0] (2 flags on the top two experts)). Both models are silent on counting and fall back on the top discriminating expert, so P8 identifies tie_bias for both models simultaneously. FC-LEX is then committed to |p-0.5| being EXACTLY the same on P9 as on P8 (the tiebreak is invoked identically and keys on expert 1 in both). WC-ST agrees for non-triage subjects, but triage subjects read P9 as 0-vs-2 (s=0.894, ceiling) and choose A decisively. A reliably more extreme, less variable response on P9 than P8 -- concentrated in the same subjects who reverse on P7 -- is a triage signature that FC-LEX cannot produce with its single tie_bias.

PREDICTIONS. WC-ST: near-ceiling on P1; ~0.84-0.91 and mutually indistinguishable on P2, P5, P6; near chance on P3 and especially P4; partly reversed / bimodal on P7; and a P9-vs-P8 asymmetry in the triage subgroup. FC-LEX: one flat above-chance accuracy toward the fewer-flag option on P1-P7 irrespective of whether the comparison is 0v1, 3v4 or 3v6 and irrespective of which experts carry the flags, plus the same magnitude-invariant tie_bias-driven deviation on P8 and P9.

[8] GOAL. PW-FMS (advocated) and WC-ST (competitor) agree on polarity ('1' = warning flag, fewer/less-bad wins) and on the use of a worst-flag tiebreak at exact count ties, so direction-only tests are useless. They disagree on exactly three things, and every pair here targets one of them:

(i) HOW DISCRIMINABILITY SCALES WITH BASE COUNT. PW-FMS uses divisive compression, s = |D|/(1 + c*(mass_a+mass_b)), i.e. a SHALLOW monotone decline. WC-ST uses ratio-scaled Weber evidence pushed through an accelerating Weibull, i.e. a CLIFF to chance once the count ratio drops below theta.
(ii) COMPOSITION SENSITIVITY. PW-FMS's decision variable is validity-tilted mass PLUS a strong worst-flag severity term; WC-ST's is the validity-blind raw count. So two pairs with the identical count comparison but different validity composition must be identical for WC-ST and can differ enormously (even reverse) for PW-FMS.
(iii) THE SOURCE OF BETWEEN-SUBJECT REVERSALS. WC-ST's ONLY reversal mechanism is discrete triage of experts with validity < 0.65. I therefore chose validities so that exactly one expert (0.55) is droppable, and I keep that expert UNFLAGGED in every composition pair 

VALIDITIES [0.97, 0.93, 0.70, 0.68, 0.66, 0.55]. Two high experts, three mid experts safely above the 0.65 triage threshold, one droppable low expert. PW-FMS mean-centred weights: gamma=0.6 -> [1.27,1.22,0.94,0.92,0.89,0.76]; gamma=1.0 -> [1.44,1.36,0.90,0.86,0.82,0.60]; gamma=1.25 -> [1.55,1.45,0.88,0.83,0.78,0.50].

CLASS 1 \EEMPTY- CONSTANT-CUE MAGNITUDE LADDER (base pairs P1 0v1, P2 1v2, P3 2v3, P4 3v4, P5 4v5; trials 1-10). These are strictly NESTED and the single discriminating flag is ALWAYS expert 3 (0.70): only the shared background count grows. Because both options carry expert 1's flag in P2-P5, PW-FMS's severity term cancels exactly and D = -w3 is CONSTANT across the whole ladder for every gamma and every lambda; only the compression denominator grows. Predicted PW-FMS accuracies (gamma .6, lam 1.1, beta 3.6, c .2, eps .29): 0.86, 0.79, 0.75, 0.72, 0.70 \u2014 a shallow ~0.15 drop that never approaches chance. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, 0.154, giving (theta .40, kappa 6, eps .30): 0.85, 0.74, 0.53, 0.505, 0.50 \u2014 a cliff. The models differ by ~0.20 on P3-P5. A flat-ish plateau at 2v3/3v4/4v5 is impossible for WC-ST at ANY theta/kappa (its psychometric function is strictly increasing in s and the design holds cue identity fixed); a collapse to chance there is impossible for PW-FMS at any comp_c because compression is divisive, not threshold-like.

CLASS 2 \u2014 COMPOSITION CONTRASTS AT MATCHED COUNTS (P6 = pair 11: A=[1,1,0,0,0,0] vs B=[0,0,1,1,1,0], a 2v3; P7 = pair 13: A=[1,0,0,0,0,0] vs B=[0,0,0,1,1,0], a 1v2). WC-ST is mathematically forced to give P6 the SAME accuracy as P3 (both 2v3, s=0.267 -> ~0.53) and P7 the SAME accuracy as P2 (both 1v2, s=0.408 -> ~0.74); no expert can be triaged because expert 6 carries no flag. PW-FMS, by contrast, loads A with the two most reliable flags: at gamma>=0.75 the mass difference plus the strong severity term (Delta sev = w1-w3 = 0.32..0.68) flips the sign of D. On P6, D = -0.27+1.1*0.32 = +0.09 at gamma=0.6 (falls in the tau band -> count fallback -> A at ~0.75) but D = +0.81 at gamma=1.0 and +1.26 at gamma=1.25 -> decisive choice of the THREE-flag option B (p(A) ~ 0.29). Same structure on P7 (D = -0.16 at gamma .6 but +0.40 at gamma 1.0, +0.74 at gamma 1.25). So PW-FMS predicts (a) a large within-subject P3-vs-P6 and P2-vs-P7 gap and (b) genuinely BIMODAL between-subject responding on P6/P7, with the reversers being precisely the subjects who are most accurate on the low-ratio ladder items. WC-ST predicts zero P3/P6 and P2/P7 differences and zero dispersion beyond binomial + lapse, because its one reversal mechanism (triage) is disabled by construction here.

CLASS 3 \u2014 EQUAL-COUNT PAIRS: TIE CASCADE vs SEVERITY TIEBREAK + TRIAGE (P8 = pair 15: A=[1,0,0,0,0,1] vs B=[0,1,0,0,1,0], 2v2; P9 = pair 17: A=[0,0,1,0,0,0] vs B=[0,0,0,1,0,0], 1v1). Both pairs are decided by expert 1 / expert 3 respectively, and in BOTH the raw counts tie. PW-FMS: masses and severities nearly cancel (|D| <= 0.05 across the whole gamma range on both), so it enters its near-tie cascade, counts tie too, and it applies its STRONG worst-flag repulsion tie_bias ~ 0.07 -> p(option holding the top discriminating flag) ~ 0.19 after lapse, and crucially the SAME value on P8 and P9 with little between-subject spread. WC-ST: p = tie_bias, uniform in [0.05,0.45] -> mean ~0.33 with wide dispersion on P9, but on P8 its triage subgroup (~16%) drops expert 6, reads 1-vs-2, and REVERSES to p(A) ~ 0.79. So WC-ST uniquely predicts an upper mode above 0.5 on P8 that is absent on P9; PW-FMS uniquely predicts a tight, extreme, P8=P9 invariant rejection of the worst-flag holder. P9 also cleanly identifies tie_bias for both models.

Every base pair appears in both A/B orders (18 unique pairs, K=5 -> 90 trials) so side bias cancels exactly. P1 (0v1) anchors the lapse rate for both models.

SUMMARY OF DISCRIMINATING PREDICTIONS. PW-FMS: shallow monotone ladder (~0.86 -> ~0.70, never near chance); large composition gaps P3>>P6 and P2>>P7 with bimodal reversals on P6/P7 driven by gamma/severity; tight, extreme, identical responding on the two equal-count pairs. WC-ST: near-ceiling P1, ~0.74 P2, and chance from P3 onward; exact equality of P3 with P6 and of P2 with P7 with no reversals; tie_bias-driven moderate responding on P9 plus a triage-driven high mode on P8 only.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "I advocate WC-ST (validity-blind Weber-scaled flag counting with an accelerating Weibull threshold, a severity tiebreak on exact count ties, and discrete expert triage) against PW-FMS (validity-tilted flag mass + strong worst-flag severity, divisive compression, graded near-tie cascade). Both models agree on polarity ('1' = warning flag; the less-bad option wins) and both use a worst-flag tiebreak at exact count ties, so direction-only tests are useless. They disagree on exactly three things, and every pair below targets one of them.\n\nVALIDITIES [0.97, 0.94, 0.72, 0.70, 0.68, 0.60, 0.54]. Two very high experts, three mid experts safely ABOVE WC-ST's 0.65 triage cutoff, and exactly two droppable experts (0.60, 0.54). PW-FMS mean-centred weights at gamma=1 are w=[1.469, 1.409, 0.969, 0.929, 0.889, 0.729, 0.609] (top:bottom ~2.4:1, and the top-vs-mid gap of 0.50 is what drives its severity term); at gamma=0.5 they are w=[1.234, 1.204, 0.984, 0.964, 0.944, 0.864, 0.804]. All 9 base pairs appear in both A/B orders (18 unique pairs, K=5 -> 90 trials), so side bias cancels exactly. Pairs P1-P8 carry NO flags on the droppable experts, so triage is disabled there by construction and the magnitude tests are uncontaminated; only P9 is triage-sensitive.\n\nCLASS 1 - CUE-CONTROLLED MAGNITUDE LADDER (P1 0v1, P2 1v2, P3 2v3, P4 3v4, P5 2v4; trials 1-10 and their mirrors). The ladder is strictly NESTED and the single discriminating flag is ALWAYS expert 3 (0.72); in P2-P5 both options carry expert 1's flag, so PW-FMS's severity term cancels EXACTLY and its numerator |D| is constant (=w3) across P2-P4 for every gamma and lambda. Only the compression denominator grows, so PW-FMS is committed to a shallow monotone decline: ~0.86, 0.77, 0.72, 0.70 (gamma .6-1, lam 1.1, beta 3.6, c .2, eps .29) - never near chance. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, which through the Weibull (theta ~.40, kappa ~6) plus lapse gives ~0.85, 0.74, 0.53, 0.505 - a CLIFF to chance. A ~0.20 model gap sits on P3 and P4. A flat-ish plateau at 2v3/3v4 is impossible for WC-ST at any theta/kappa (its psychometric function is strictly increasing in s and cue identity is held fixed); a collapse to chance there is impossible for PW-FMS at any comp_c because compression is divisive, not threshold-like. P5 (2v4) is the ratio control: same smaller count as P3 and the same base counts as P4, but ratio 1:2. WC-ST says P5 (s=0.436, ~0.79) must be indistinguishable from P2 (s=0.408) and far ABOVE P4 (same raw difference is irrelevant, only ratio matters); PW-FMS says P5 (0.81) sits only slightly above P2 (0.77) and only ~0.11 above P4. The signature {P2 ~ P5} >> {P3 > P4 ~ chance} is uniquely WC-ST; a linear/Gaussian margin rule or PW-FMS both predict P4 ~ P2.\n\nCLASS 2 - COMPOSITION INVARIANCE AT MATCHED COUNTS (P6: A=[1,1,0,0,0,0,0] vs B=[0,0,1,1,1,0,0]; P7: A=[0,0,0,1,1,0,0] vs B=[1,1,1,0,0,0,0]). Both are 2-vs-3 with the fewer-flag option always A, so WC-ST is MATHEMATICALLY FORCED to give P6 = P7 = P3 (~0.53, weak A preference), with no triage escape because experts 6-7 carry no flags. PW-FMS reads them completely differently: in P6 the two flags sit on the 97%/94% experts, so mass (2.878 vs 2.787) plus a strong severity difference (w1-w3 = 0.50) gives D = +0.64 at gamma=1, lam=1.1 -> it REVERSES and picks the THREE-flag option, p(A) ~ 0.33; in P7 the same logic runs the other way (D = -2.62) -> p(A) ~ 0.85. That is a ~0.52 within-subject spread across two pairs WC-ST must call identical, and it is the single cleanest falsification test in the design. Equally, if P6 = P7 = P3 to within noise, no gamma/severity setting of PW-FMS can produce it.\n\nCLASS 3 - EQUAL-COUNT TIE MAGNITUDE AND DISPERSION (P8: A=[0,0,1,0,0,0,0] vs B=[0,0,0,1,0,0,0]). Counts tie and the weighted masses (0.969 vs 0.929) nearly cancel, so PW-FMS enters its cascade, finds equal counts, and applies its STRONG worst-flag repulsion (tie_bias ~ 0.07) -> p(A) ~ 0.20, TIGHT across subjects (tie_bias in [0, 0.14]). WC-ST applies the same lexicographic rule but with a broad, mildly-repelled subject-level tie_bias in [0.05, 0.45] -> mean p(A) ~ 0.33 with LARGE between-subject dispersion. So P8 tests the magnitude and the variance of the tiebreak, not its direction, and it simultaneously identifies tie_bias for both models.\n\nCLASS 4 - SEVERITY-VS-COUNT / TRIAGE (P9: A=[0,0,0,0,0,1,1], two flags on the 60%/54% experts, vs B=[1,0,0,0,0,0,0], one flag on the 97% expert). This is PW-FMS's own best pair and is included to keep the design fair: its severity term (sev_a - sev_b = 0.729-1.469) overwhelms the count difference, giving D = -0.95 at gamma=1 -> it picks the TWO-flag option A at ~0.78, while low-gamma subjects fall into the count fallback and pick B at ~0.78 (a gamma-driven bimodality). WC-ST is validity-blind: 84% of subjects count 1-vs-2 and pick B (p(A) ~ 0.26), while the triage subgroup discards the two sub-0.65 experts, reads 0-vs-1, and REVERSES to p(A) ~ 0.85 - a small, discrete upper mode rather than a continuum. The predicted group means differ by ~0.2-0.4 and, more informatively, the shape of the between-subject distribution differs (a small discrete ~16% mode for WC-ST vs a broad gamma-continuum split for PW-FMS), and WC-ST's reversers on P9 must be exactly the subjects who show NO composition effect on P6/P7.\n\nPREDICTIONS. WC-ST: near-ceiling P1 (~0.85), ~0.74 P2, ~0.79 P5, and near chance on P3 and especially P4; exact equality of P3, P6 and P7 with no composition effect and no reversal; moderate, widely dispersed rejection of the top-flag holder on P8; mostly B on P9 with a small discrete triage mode. PW-FMS: a shallow monotone ladder (~0.86 -> 0.70, never approaching chance) with P4 close to P2; a huge P6-vs-P7 composition gap including a direction reversal on P6; tight, extreme (~0.20) responding on P8; and a gamma-graded, severity-driven preference for the two-low-flag option on P9.",
  "validities": [
    0.97,
    0.94,
    0.72,
    0.7,
    0.68,
    0.6,
    0.54
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 25174,
  "output_tokens": 20553
}
```
