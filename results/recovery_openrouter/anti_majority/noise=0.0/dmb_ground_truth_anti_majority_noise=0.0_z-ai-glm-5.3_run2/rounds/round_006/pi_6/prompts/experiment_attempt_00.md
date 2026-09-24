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
**Description:** Context-Normalized Noisy Anti-Validity Integration (CNNAI), knife-avoidance recalibration. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity CONTEXT BUCKETS (the mean endorsed validity rounded to 0.1), so all trials sharing a bucket inherit one shared per-subject draw. The elasticity decomposes into a small stable per-subject component (eta_base ~ N(0.02, 0.09)) and a larger per-bucket redraw (~N(0.10, 0.19)), total hard-clipped to [-0.25, 0.70]; the variance ratio fixes latent cross-context consistency at r ~ 0.18 by construction. The engaged eta distribution (mean ~0.12, sd ~0.21) is deliberately centered well BELOW the kill-design knife region (eta ~ 0.31-0.34 where high-validity stacks flip to low-validity stacks), so the engaged bulk behaves SCWI-like in effective curvature while margins on the below-knife side stay large. A modest per-trial multiplicative attention diffusion (sd 0.05) supplies within-context stochasticity, choice is softmax(beta * values) with nearly homogeneous engaged beta (lognormal median ~2.14, log-sd 0.20) plus a narrow symmetric lapse (0.03-0.09), and a 13% true-guesser subpopulation is retained, uncorrelated with eta.

**Parameters:**
- subject_seed: [0, 1]
- regime_seed: [0, 1]
- eta_base_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Context-Normalized Noisy Anti-Validity Integration (CNNAI),
    # iter 6 -- knife-avoidance recalibration of the accepted iter-5 base
    # per the iter-5 critic diagnosis. Mechanism family UNCHANGED:
    # additive anti-validity integration, context-referenced elasticity
    # w_j = -v_j * (v_j / v_ref)^eta with v_ref = mean validity of the
    # endorsed cues, per-trial attention diffusion, softmax + symmetric
    # lapse, guesser mixture. Changes vs the accepted iter-5 base:
    #   (A) ETA CENTER LOWERED 0.23 -> 0.12 (eta_base ~ N(0.02, 0.09) plus
    #       per-context redraw ~ N(0.10, 0.19), total clipped [-0.25, 0.70]).
    #       The kill composites (Exps 5/6/8) are near-knife statistics:
    #       their flip knives sit at eta ~ 0.30-0.34, and on the
    #       below-knife side the score margins GROW as eta decreases
    #       (e.g. Exp 5's 2x1.00-vs-5x0.50 cell: margin 0.14 at eta 0.23
    #       vs 0.33 at eta 0.10). Centering at 0.12 both moves engaged
    #       mass below the knives (P(eta < 0.33) ~ 0.83) and enlarges
    #       the margins, lifting the undershooting composites.
    #   (B) ATTENTION DIFFUSION CUT 0.10 -> 0.05: at 0.10 the per-trial
    #       margin noise (sd ~ 0.25 on a 12-cue cell) was comparable to
    #       the near-knife signals (~0.15-0.35), damping below-knife
    #       subjects to p ~ 0.55-0.6 -- the main softening channel on
    #       Exps 5/6/8. Per-trial noise averages out within a subject
    #       (iter-2 critic), so cutting it lifts means without touching
    #       between-subject variance.
    #   (C) CONTEXT KEY COARSENED to the mean endorsed validity rounded
    #       to 0.1 (a bucketed v_ref / ladder identifier). The critic's
    #       'rounded MAX endorsed validity' variant was NOT used: both
    #       Exp-9 ladders ({1.00,0.50} and {1.00,0.60} cells) have max
    #       validity 1.0, so a max-based key would map them to the SAME
    #       draw and re-couple them -- recreating the iter-1 r = 0.44
    #       failure the critic itself warns against. Bucketed mean keeps
    #       them distinct (0.6 vs 0.7) while merging within-experiment
    #       fragmentation (Exp 2: ~2-3 keys instead of ~6; Exp 5: 2 keys),
    #       so per-subject composites stop averaging over many
    #       independent eta draws.
    #   (D) EXP-9 COUPLING CHANNELS CLOSED: engaged epsilon narrowed
    #       0.02-0.17 -> 0.03-0.09 (stable lapse was the dominant
    #       residual cross-ladder trait), eta_base sd cut 0.13 -> 0.09
    #       (latent r = 0.0081/(0.0081+0.0361) ~ 0.18), beta median
    #       trimmed 2.34 -> 2.14 with log-sd 0.25 -> 0.20 (guards Exps
    #       1/3/10 against the noise cut, and removes the last sizeable
    #       stable-trait correlation channel).
    #   (E) v_ref STAYS MEAN-REFERENCED (not max): the weight ratio
    #       w(v_h)/w(v_l) = (v_h/v_l)^(1+eta) is invariant to v_ref, and
    #       max-referencing multiplies every margin by v_max^(-eta) <= 1,
    #       which would DEFLATE exactly the near-knife margins the
    #       composites need. The critic's goal (steeper high-validity
    #       region) is achieved via (A)+(B) instead.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CNNAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. These subjects sit at ~0.5 on
        # every diagnostic cell and dilute population means/slopes.
        t = u / 0.13
        beta = 0.02 + 0.08 * t            # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps       # 0.12 .. 0.25
        eta_base = 0.10                  # irrelevant at beta ~ 0
    else:
        # Engaged subpopulation (~87%). Small stable elasticity
        # component (sd 0.09) paired with a larger per-context redraw
        # (sd 0.19): the variance ratio fixes latent cross-context
        # consistency at r ~ 0.18 by construction. The stable lapse
        # range is NARROW (0.03-0.09) so it can no longer act as a
        # hidden coupling channel across contexts, and beta is nearly
        # homogeneous (lognormal sd 0.20).
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.76 + 0.20 * float(ndtri(bs)))  # median ~2.14
        beta = min(max(beta, 0.9), 4.5)
        epsilon = 0.03 + 0.06 * ps       # 0.03 .. 0.09

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    # v_ref = mean validity of the cues actually displayed (endorsed)
    # this trial; the context key is v_ref rounded to one decimal, so
    # all cells of one validity ladder share a single per-subject eta
    # draw while different ladders (e.g. Exp 9's 0.50-ladder at bucket
    # 0.6 vs its 0.60-ladder at bucket 0.7) draw independently.
    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw, centered low so the engaged bulk sits below
    # the kill knives (~0.31-0.34) with large below-knife margins:
    # engaged eta ~ N(0.12, 0.21), P(eta < 0.33) ~ 0.83.
    eta_ctx = eta_base + 0.10 + 0.19 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion on cue weights,
    # sd cut 0.10 -> 0.05: at 0.10 the margin noise on near-knife cells
    # was comparable to the signal and was the main softening channel
    # on the Exp 5/6/8 composites.
    att = np.exp(0.05 * np.random.randn(n_features))
    w = w * att

    scores = np.array([float(np.dot(w, ra)), float(np.dot(w, rb))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Kneed Compressive Anti-Validity Integration with a Moderate Deterministic Tail (CAVI-K), beta-homogenization refinement. Subjects integrate all binary expert ratings into a per-option subjective DEFECT LOAD, comprehending a rating of 1 as a defect (sign inversion). Per-cue defect magnitude is supralinear in the instructed validity, d_j = v_j + tau*v_j^2 (tau in [0.35, 0.65]), modulated by a context elasticity (v_j/v_ref)^eta keyed on the bucketed mean endorsed validity, with a zero-centered stable component (sd 0.10) plus a per-context redraw (sd 0.20), fixing latent cross-context consistency at r ~ 0.20. The defect load is compressed by a two-regime map with the knee at 4.45 — above every near-knife small stack (AL cells, 2-top-cue adherence cells, which keep full raw margins) and below the large diagnostic stacks (which grow sub-proportionally aversive under kappa in [0.55, 0.68]). Population: ~14% true guessers, ~70% engaged with a HOMOGENIZED lognormal beta (median 2.65, log-sd 0.29, clip [0.8, 6.0]) — the tightened spread removes both the saturating hot-engaged bloc (which was driving the Exp-9 cross-ladder correlation to 0.388 via beta-saturation, not latent consistency) and the soft low-beta tail (which was diluting anti-validity adherence on Exps 2/5/6/8/11) — and ~16% deterministic tail (beta 4.0-6.5, near-linear kappa, pinned eta). Choice is softmax(beta*(D_B - D_A)) with symmetric lapse and per-trial attention diffusion (sd 0.035).

**Parameters:**
- regime_seed: [0, 1]
- tau_seed: [0, 1]
- kappa_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- eta_base_seed: [0, 1]
- subject_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Kneed Compressive Anti-Validity Integration (CAVI-K), iter 5.
    # Minimal single-lever diff on the accepted iter-4 base, applying
    # the iter-4 critic diagnosis. Changes vs the accepted base:
    #   (A) ENGAGED BETA LOG-SD TRIMMED 0.35 -> 0.29 with the median
    #       nudged 2.5 -> 2.65 (clip unchanged [0.8, 6.0]). The Exp-9
    #       trajectory isolated the failure: at a CONSTANT eta split
    #       (0.10/0.20), raising the beta median 2.0 -> 2.5 swung the
    #       cross-ladder correlation 0.067 -> 0.388, i.e. the correlation
    #       is driven by the saturating HOT END of the engaged beta
    #       distribution, not by latent context consistency. Trimming the
    #       log-sd removes that bloc (pulling Exp 9 back toward its
    #       [0.15, 0.30] window) while ALSO removing the soft low-beta
    #       tail that was diluting anti-validity adherence on
    #       Exps 2/5/6/8/11 — the shared undershoot of all four prior
    #       iterations. The small median nudge (2.5 -> 2.65, well short
    #       of the critic's 3.0+ that the AL-sigmoid arithmetic rules
    #       out) targets Exp 11's remaining shortfall (2.24 vs the
    #       [2.4, 3.2] window).
    #   (B) ENGAGED KAPPA UPPER BOUND 0.70 -> 0.68 (range now
    #       [0.55, 0.68]): a single small pre-emptive guardrail step.
    #       Removing the low-beta tail raises mean adherence broadly,
    #       which will push Exp 10's rise (0.350 at iter 4, window
    #       [0.22, 0.36]) toward its ceiling; slightly stronger
    #       compression on the large diagnostic stacks holds it inside
    #       the window. This is deliberately NOT the iter-3 confounded
    #       trim (which fought a simultaneous knee raise) — the knee
    #       stays at 4.45 and only the upper bound moves by 0.02.
    #   Everything else is held verbatim from the accepted iter-4 base:
    #   knee 4.45 (above the max AL stack 4.28 for every tau), tau
    #   [0.35, 0.65], the 0.10/0.20 eta split with pinned tail eta,
    #   attention diffusion 0.035, the 14/70/16 mixture, and all lapse
    #   ranges — each either in-window or explicitly validated at
    #   iter 4.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CAVI-K expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    ts = _seed("tau_seed")
    ks = _seed("kappa_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    es = _seed("epsilon_seed")
    ebs = min(max(_seed("eta_base_seed"), 1e-6), 1.0 - 1e-6)
    ss = _seed("subject_seed")

    # --- Population mixture ---
    if u < 0.14:
        # True guessers (~14%).
        t = u / 0.14
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * es              # 0.12 .. 0.25
        tau = 0.50
        kappa = 0.75
        eta_base = 0.00
    elif u < 0.84:
        # Engaged bulk (~70%): HOMOGENIZED beta (median 2.65, log-sd
        # 0.29), kappa acting only above the knee, zero-centered stable
        # elasticity. The tightened beta spread removes both the
        # saturating hot bloc (Exp 9) and the soft low-beta tail
        # (adherence dilution on Exps 2/5/6/8/11).
        tau = 0.35 + 0.30 * ts                 # 0.35 .. 0.65
        kappa = 0.55 + 0.13 * ks                # 0.55 .. 0.68
        beta = math.exp(math.log(2.65) + 0.29 * float(ndtri(bs)))
        beta = min(max(beta, 0.8), 6.0)
        epsilon = 0.03 + 0.06 * es             # 0.03 .. 0.09
        eta_base = 0.10 * float(ndtri(ebs))
        eta_base = min(max(eta_base, -0.15), 0.15)
    else:
        # Deterministic tail (~16%): hot, low-lapse, strongly inverted,
        # least compressive, PINNED stable elasticity (no ndtri draw) so
        # the tail cannot act as a hidden cross-context coupling channel.
        tau = 0.55 + 0.10 * ts                 # 0.55 .. 0.65
        kappa = 0.78 + 0.07 * ks                # 0.78 .. 0.85
        beta = 4.0 + 2.5 * bs                   # 4.0 .. 6.5
        epsilon = 0.02 + 0.02 * es              # 0.02 .. 0.04
        eta_base = 0.05                         # pinned

    # --- Context: bucketed mean endorsed validity ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw, zero-centered; stable/redraw sd ratio 0.10/0.20
    # fixes latent cross-context consistency at r ~ 0.20 by construction.
    eta_ctx = eta_base + 0.20 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.30), 0.50)

    # --- Defect weights: supralinear anti-validity, context-normalized ---
    d = val + tau * np.square(val)
    w = d * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion (sd 0.035, the
    # iter-3-proven level that lets Exp-12 fitted bhat reach the ramp).
    att = np.exp(0.035 * np.random.randn(n_features))
    w = w * att

    # Raw defect loads.
    s_a = float(np.dot(w, ra))
    s_b = float(np.dot(w, rb))

    # TWO-REGIME (offset) compression: identity below the knee, power
    # compression above. Knee 4.45 > max AL stack 4.28 (at tau = 0.65),
    # so near-knife small stacks keep full raw margins for every tau;
    # the large Exp-10/12 stacks (4.7-11.5) are compressed.
    S_KNEE = 4.45

    def _compress(s):
        if s <= S_KNEE:
            return s
        return S_KNEE * ((s / S_KNEE) ** kappa)

    d_a = _compress(s_a)
    d_b = _compress(s_b)

    scores = np.array([-d_a, -d_b])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design exploits the core structural difference between Take The Best (TTB) and Tallying: TTB's choice depends ONLY on the single highest-validity discriminating cue, while Tallying depends on the total count of cue wins. I fix 5 experts with well-spread validities [0.95, 0.85, 0.75, 0.65, 0.55] and construct a family of critical trials where the top cue favors A but B wins on more (or all) of the remaining cues, with the tally margin parametrically varied. KEY QUANTITATIVE DISSOCIATION: across the four trial types with tally margins of B+4, B+3, B+2, and exact tie (pairs 1, 3, 5, 7), TTB predicts an IDENTICAL probability of choosing A on every one of them — P(A) = sigmoid(beta)*(1-epsilon) + epsilon/2, completely flat as a function of tally margin, because in all four cases the top cue (v=0.95) discriminates in favor of A. Tallying instead predicts a strictly graded, monotonically decreasing P(A) as the tally margin against A grows: 1/(1+e^{3*beta}), 1/(1+e^{2*beta}), 1/(1+e^{beta}), and exactly 0.5 respectively. So the signature is FLAT-vs-GRADED psychometric slope over tally margin at fixed TTB evidence. A second, orthogonal quantitative probe: pairs 11, 13, 15 hold the tally structure roughly constant but shift WHICH cue is the top discriminating one (cue 2 at v=0.85, cue 3 at v=0.75). TTB predicts the same confident A-choice regardless of which cue discriminates (its score is always 1 vs 0), whereas any validity-weighting or evidence-accumulation account would predict graded confidence. Pair 13 is especially diagnostic: the second cue favors A, all others tie, so TTB predicts a confident A choice while Tallying predicts exact 50/50 guessing. Pairs 9/10 are agreement controls (both theories predict the same confident choice) to calibrate shared noise parameters; pairs 17/18 are full-tie trials where both theories must guess, providing a pure lapse-rate baseline. All critical pairs are mirrored (A/B swapped) to absorb side/response biases. With 18 unique pairs, K = 96//18 = 5 repetitions yields 90 trials per subject. Because there is no feedback, subjective cue order stays anchored to the instructed validities, keeping TTB's cue cascade well-defined. If subjects' P(A) is flat across the margin series, TTB is supported; if it declines systematically with the tally margin (approaching 50% at tie), Tallying is supported; a mixed pattern (e.g., partial decline) would favor neither in its pure form.

[1] This design inverts the logic of the classic TTB-vs-Tallying conflict: instead of holding the top cue fixed and varying the tally margin (already explored), I hold the TALLY MARGIN fixed and parametrically vary WHICH cues the losing-side wins on. I use 5 experts with strictly ordered validities [0.90, 0.80, 0.70, 0.60, 0.55] so TTB's cascade order is unambiguous. CORE DISSOCIATION (pairs 1-7): every one of these pairs has the identical tally structure — A wins exactly 3 cues, B wins exactly 2 (margin +1 for A) — so Tallying predicts the exact same P(A) = sigmoid(beta) on all seven trials, a perfectly FLAT psychometric function over the series. TTB instead predicts a STEP FUNCTION: on pairs 1-3 the top cue (v=0.90) is one of B's wins, so TTB confidently picks B (P(A) = 1 - sigmoid(beta)); on pairs 4-7 the top cue favors A, so TTB confidently picks A (P(A) = sigmoid(beta)). The predicted discontinuity sits exactly between pair 3 (B wins cues {1,5}) and pair 4 (B wins cues {2,3}) — TTB specifies the step's location precisely, while Tallying specifies zero slope. This is a double dissociation in slope: the previously explored design had TTB flat / Tallying graded; this one has Tallying flat / TTB discontinuous, so the two designs jointly triangulate both theories. Pairs 8-14 are exact A/B mirrors of 1-7 to absorb side/response bias (Tallying: flat sigma(-beta); TTB: mirrored step). SECOND PROBE — cascade depth at graded tally (pairs 15-17): the top k cues are forced to tie so TTB's deciding cue moves down the validity ladder (cue 2, then cue 3, then cue 4) while it always picks B with the SAME confidence 1 - sigmoid(beta), because TTB's binary score is validity-invariant. Tallying, in contrast, predicts a strictly graded decline in P(A) — sigma(2*beta), sigma(beta), and exactly 0.5 — because the tally margin shrinks from +2 to +1 to 0 as ties remove A's winning cues. Pair 17 is the sharpest single trial: Tallying predicts exactly 50/50 guessing (tally 1-1), TTB predicts a confident B choice from cue 4 alone. Pairs 18-20 mirror these. CONTROLS: pair 21 (A wins all cues) is an agreement trial calibrating the shared sigmoid(beta) confidence asymptote; pair 22 (identical vectors) is a full-tie trial where both theories must guess, giving a pure lapse-rate baseline; pairs 23-24 are extreme single-cue-vs-all-others conflicts (top cue alone vs. four opposing cues) providing maximum-separation anchor points where the theories disagree most strongly in magnitude (Tallying: sigma(-3*beta) vs TTB: sigmoid(beta)). With 24 unique pairs, K = 96 // 24 = 4 repetitions yields exactly 96 trials. No feedback keeps subjective cue order anchored to instructed validities. DECISION RULE: if P(A) across pairs 1-7 is flat at the level fit on margin+1 trials, Tallying is supported; if it shows a sharp step exactly between pairs 3 and 4, TTB is supported; a graded monotone drift favoring neither step nor flatness would falsify both pure forms.

[2] ADVERSARIAL LOGIC. The advocated theory (SCWI) and the competitor (Tallying) have a deep structural relationship on binary cues: because SCWI's signed value difference between options is sum_j w_j*(a_j - b_j) with w_j = -(v_j + tau*v_j^2), tied cues cancel exactly, and SCWI effectively computes an ANTI-VALIDITY-WEIGHTED tally: it prefers the option that LOSES on more (and more heavily weighted) cues. Tallying prefers the option that WINS on more cues, ignoring validity entirely. Two things follow. (1) SIGN: on every trial with a nonzero tally margin the two theories prefer OPPOSITE options. (2) SLOPE: Tallying's evidence is the raw count margin; SCWI's evidence is the validity-weighted (inverted) margin, so the two theories order trials differently. The design exploits both, with a parameter-free kill shot: on any trial where the feature-wise counts are TIED (a_wins = b_wins), Tallying's softmax is exactly uniform for ANY beta and ANY epsilon (lapse is symmetric), pinning P(A) = 0.500 exactly. SCWI is not so pinned: a count tie with unequal validity composition yields a nonzero weighted margin. PROBE A (pairs 1-8, count ties): I construct 8 count-tie trials whose SCWI weighted margin M = sum|w| over B's winning cues minus sum|w| over A's winning cues sweeps from -0.92 to +0.92 (computed at tau=0.35; the sweep is robust across tau in [0.25,0.45]). Tallying predicts EXACTLY 0.500 on all eight, a horizontal line with zero free parameters. SCWI predicts a monotone psychometric sweep of P(A) from ~0.25 up to ~0.75, with the specific ordering implied by the supralinear anti-validity weights (e.g., pair 3, where A wins the two top experts and B wins the two worst, yields P(A)~0.25; pair 4, the cue-label swap, yields ~0.75). Any reliable deviation from 0.5 on these trials falsifies Tallying outright; the graded M-ordering then tests SCWI quantitatively. PROBE B (pairs 9-15, tally margin fixed at +1 for A: A wins exactly 3 cues, B wins 2): Tallying predicts a FLAT P(A) = sigmoid(beta)*(1-eps)+eps/2 on all seven, while SCWI predicts a strictly graded function of which cues A wins — from P(A)~0.11 when A wins the three most trusted experts (its 'defect load' is maximal) to P(A)~0.49 when A wins the three least trusted. This is a flat-vs-graded slope dissociation at fixed tally evidence, and every SCWI prediction sits BELOW 0.5 while Tallying's sits ABOVE 0.5: a double (sign + slope) dissociation. PROBE D (pairs 16-19) mirrors four Probe-B trials (A/B swapped) to absorb side and response bias; Tallying stays flat at sigmoid(-beta), SCWI stays graded above 0.5. ANCHORS (pairs 20-23): extreme trials where A wins all cues (pair 20), B wins all (21), or A wins four low-validity cues against B's single top cue (22/23). Tallying approaches deterministic choice of the tally winner (sigmoid(5*beta), sigmoid(3*beta)); SCWI approaches deterministic choice of the tally LOSER, with confidence tracking the weighted margin M rather than the count margin — plotting P(A) against count margin gives a rising function under Tallying and a falling function under SCWI. Pair 24 is a full tie (identical vectors) where both theories predict exactly 0.5, providing a pure lapse/side-bias baseline. With 24 unique pairs, K = 96//24 = 4 repetitions (96 trials). DECISION RULE: (i) if P(A) on pairs 1-8 is reliably different from 0.5 in either direction, Tallying is falsified (it cannot produce this for any parameter values); (ii) if P(A) on pairs 9-15 is flat, Tallying is supported and SCWI is falsified; (iii) if P(A) across pairs 1-15 is a monotone function of the anti-validity weighted margin M (increasing in M, crossing 0.5 near M=0, with the tally loser preferred), SCWI is supported; (iv) a preference for the tally WINNER that is nonetheless graded by validity weights would falsify both pure forms (it would indicate pro-validity weighted integration).

[3] ADVERSARIAL LOGIC (the 'transpose' of the count-tie trick). On binary cues, Tallying's evidence is the integer count margin m = a_wins - b_wins, while SCWI's evidence is the anti-validity weighted margin D = -sum_j (v_j + tau*v_j^2)*s_j (s_j = +1 if A wins cue j, -1 if B wins). The already-explored design pinned TALLYING at 0.5 (count ties) and let SCWI vary. I do the exact inverse: I engineer WEIGHTED TIES (D ~= 0, robustly across the full tau range [0.25, 0.45]) at NONZERO tally margins, so SCWI is pinned near 0.5 while Tallying is graded — and then hold D at a large fixed negative value while m varies, so SCWI is flat-low while Tallying rises. This yields a two-probe double slope dissociation in the opposite assignment from the prior design.

VALIDITY ENGINEERING. With validities [1.00, 0.95, 0.65, 0.54, 0.51, 0.50], a weighted tie between a cue set X and cue set Y requires sum_X v + tau*sum_X v^2 = sum_Y v + tau*sum_Y v^2. I choose sets where the constant and tau-coefficient nearly cancel simultaneously: e.g., the four low cues {3,4,5,6} (sum v = 2.20, sum v^2 = 1.224) vs the two top cues {1,2} (1.95, 1.9025): residual c(tau) = 0.25 - 0.678*tau, i.e., |c| <= 0.080 for ALL tau in [0.25, 0.45]. Similar cancellations: {4,5} vs {2} (|c| <= 0.058) and {3,6} vs {1} (|c| <= 0.068).

PROBE A (pairs 1-6, weighted ties at nonzero tally margin): pairs 1-2 have m = +1 and pair 3 has m = +2, all with |D| <= 0.08 across the entire SCWI parameter box. SCWI therefore predicts P(A) = 0.500 +/- 0.032 on ALL SIX (including mirrors) for ANY beta in [1.1,1.7], tau in [0.25,0.45], epsilon in [0.05,0.15] — a horizontal line with essentially no free parameters. Tallying predicts a graded ladder: P(A) = sigmoid(m*beta)*(1-eps)+eps/2, with lapse-corrected logits exactly proportional to m (the m=+1 and m=+2 trials must sit in 1:2 logit ratio, symmetric under mirroring). Any reliable, mirror-symmetric departure from 0.5 that grows with |m| falsifies SCWI outright; perfect flatness at 0.5 falsifies Tallying (which must exceed 0.5 at m>0 for any parameters).

PROBE B (pairs 7-12, fixed weighted evidence, varying tally margin): three trials with D pinned in a narrow confident band — pair 7: m=+1, D in [-1.30,-1.13]; pair 8: m=+2, D in [-1.18,-1.02]; pair 9: m=+3, D in [-1.30,-1.26]. SCWI predicts a FLAT-LOW P(A) ~ 0.15-0.26 on all three (mirrors ~0.74-0.85), because its evidence barely changes. Tallying predicts P(A) >= 0.5 always (m>0), strictly increasing in m: ~0.86 -> 0.96 -> 0.97 at beta=2. Huge vertical separation plus opposite slope sign. Note this is the inverse hold of the prior design's fixed-margin probe (there: m fixed, D varied; here: D fixed, m varies).

MINIMAL PAIR (pairs 3 vs 9): identical stimuli except cue 2 changes from a B-win to a tie. Tallying: m goes +2 -> +3, P(A) barely moves (0.96 -> 0.97). SCWI: D jumps from -0.01 to -1.28, P(A) collapses 0.50 -> 0.18. A single-cue manipulation that one theory treats as negligible and the other as decisive.

ANCHORS AND CONTROLS: pairs 13/14 (one option wins all six cues, m = +/-6, |D| ~ 5.0-5.6) are maximum-separation trials — Tallying ~0.97 vs SCWI ~epsilon/2 (~0.05). Pairs 15/16 are full ties (identical vectors, one all-0 and one mixed) where BOTH theories predict exactly 0.500, providing a pure lapse/side-bias baseline and verifying that tie values (0 vs 1) are inert. All critical pairs are mirrored to absorb side/response bias. 16 unique pairs, K = 96//16 = 6 repetitions = 96 trials.

DECISION RULES: (i) If P(A) on pairs 1-6 deviates from 0.5 symmetrically and grows with |m| (logit ratio ~1:2 for m=1 vs m=2), Tallying is supported and SCWI is falsified — SCWI cannot exceed 0.532 or dip below 0.468 there for any allowed parameters. (ii) If pairs 1-6 are flat at 0.5 while pairs 7-12 sit flat at ~0.2 (mirrors ~0.8) and pair 13 shows P(A) ~ 0.05 (anti-following the all-1s option), SCWI is supported and Tallying is falsified — Tallying cannot go below 0.5 - eps/2 at m>0. (iii) Flat 0.5 on Probe A combined with a Tallying-like increase on Probe B (or any preference for the tally winner graded by validity weights) falsifies both pure forms. Bonus: pair 9 also separates TTB (which picks B via cue 1) from Tallying, so a TTB-like subject is identifiable rather than confounded with SCWI.

[4] ADVERSARIAL LOGIC. SCWI and PAHI are structurally near-identical: both are anti-validity additive integrators (a rating of 1 counts AGAINST an option, tied cues cancel exactly, choice is a softmax plus symmetric lapse). The ONLY substantive difference is the validity-to-weight mapping: SCWI uses w(v) = v + tau*v^2 with tau in [0.25, 0.45]; PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]. Two mathematical facts drive the whole design. (1) UNIFORM STEEPNESS ORDERING: SCWI's log-elasticity is (1+2*tau*v)/(1+tau*v) <= 1.9/1.45 = 1.31 everywhere on v in [0.5, 1], while PAHI's elasticity is gamma >= 1.4. Therefore for ANY two validities, PAHI's relative weight w(v_hi)/w(v_lo) strictly exceeds SCWI's: PAHI is a uniformly steeper validity gradient. (2) LEVEL CROSSING: at v = 1.0 PAHI's weight is EXACTLY 1 (gamma-invariant) versus SCWI's 1.25-1.45, while at v = 0.5 SCWI's weight is 0.56-0.61 versus PAHI's 0.19-0.38. Consequence: on 'concentration' trials pitting a few high-validity endorsements against many low-validity endorsements, the preference flips at a knife-edge count ratio that differs between the theories, and integer ratios can be placed in the gap between the two knife-edges. THE KILL ZONE: with validities exactly 1.00 (two cues) and 0.50 (eight cues), let d = (number of 0.5-cue 1s on the low-cue option) - (number of 0.5-cue 1s on the top-cue option). SCWI prefers the top-cue option H iff d > 2(1+tau)/(0.5+0.25*tau), i.e., d in (4.44, 4.74) over the whole tau box -> flips strictly between d=4 and d=5. PAHI prefers H iff d > 2/0.5^gamma = 2^(gamma+1) in (5.28, 10.55) -> flips at d=6 or LATER, never before 5.28. So at d=5 the two theories make OPPOSITE predictions for every admissible parameter value: SCWI P(H) in [0.54, 0.62] (its evidence is 0.5-0.75*tau > 0 since tau < 2/3), PAHI P(H) in [0.04, 0.46] (its evidence is 5*0.5^gamma - 2 < 0 since gamma > log2(2.5) = 1.32). Note v = 1.0 exactly is essential: with v = 0.99, SCWI's flip condition tau < 0.389 fails inside its own box, destroying the parameter-free separation. PRIMARY INSTRUMENT (pairs 1-12): a d-sweep from 3 to 8, fully mirrored. SCWI predicts a psychometric curve crossing 0.5 strictly between d=4 (evidence = -tau < 0 always) and d=5, then saturating (P(H) ~ 0.67/0.77/0.84 at d=6/7/8; at d=8 its evidence is exactly +2.0 for ALL tau - a parameter-free anchor). PAHI predicts a crossing at d* = 2^(gamma+1) in (5, 10.6): at or beyond d=6 for gamma >= 1.59, and for gamma > 2 the curve never exceeds 0.5 within the observable range d <= 8. Three parameter-free statistics separate them: (i) the SIGN of P(H) at d=5 (16 trials per subject: pairs 3, 9, 13, 14 x 4 reps); (ii) the CROSSING LOCATION (SCWI: 4-5 for every subject, since the tau range is too narrow to move it; PAHI: >= 5.3, heterogeneous across subjects - SCWI predicts essentially zero between-subject variance in crossing point, PAHI predicts large variance); (iii) the saturation level at d=8 (SCWI >= 0.84; PAHI can sit below 0.5). SECONDARY KILLS at other validity mixtures, so the dissociation is not an artifact of the 1.0/0.5 pair alone: pair 15/16 (G): {1.0, 1.0, 0.85} vs seven 0.5-cues - SCWI margin 0.65-0.97*tau in [0.21, 0.41] favors the high option, PAHI margin 2+0.85^gamma-7*0.5^gamma in [0.14, 1.34] favors the low option; P(H) in [0.55, 0.66] vs [0.10, 0.46]. Pair 17/18 (S): two 0.85-cues vs four 0.5-cues - SCWI margin 0.3-0.445*tau > 0 always (tau < 0.674) favors the 0.85 option, PAHI margin 0.5^gamma*(1.7^gamma - 2) > 0 always (gamma > 1.305) favors the 0.5 option. Pair 19/20 (M2): two 1.0-cues vs {0.85, 0.85, 0.5} - SCWI margin 0.2-0.305*tau > 0 always (tau < 0.656), PAHI margin 2-2*0.85^gamma-0.5^gamma in [0.03, 0.48] > 0 always. GAMMA-LADDER (pairs 21/22 and d=6/7/8): pair W ({1.0, 0.85} vs five 0.5-cues) splits PAHI at gamma ~= 1.47; d=6 splits at 1.59; d=7 at 1.81; d=8 at 2.0. SCWI predicts uniformly high P(H) on all of W/d6/d7/d8 (>= 0.55/0.67/0.77/0.84). If PAHI is correct, the observed flips across this ladder must recover a single consistent gamma AND that gamma must place the d=5 choice below 0.5; if SCWI is correct, all ladder points sit high while d=5 sits above 0.5. ROBUSTNESS VARIANT (pairs 13/14): d=5 realized with an extra 1 on BOTH options (3 vs 6 total 1s instead of 2 vs 5) - identical difference, different totals; both theories predict behavior identical to pair 3, ruling out raw count-of-1s heuristics. CONTROLS: pair 23 (all-1 vs all-0) is a maximum-separation anti-validity anchor where both theories predict P(all-1 option) ~= epsilon/2, calibrating the lapse; pair 24 (identical mixed vectors) is a full tie where both predict exactly 0.500, calibrating side/response bias. All critical pairs are mirrored to absorb side bias. 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials. DECISION RULES: (i) If the d=5 composite H-choice rate exceeds 0.5 (with d=4 below and d=6 above), SCWI is supported and PAHI is falsified - PAHI cannot exceed 0.463 there for any (beta, gamma, epsilon) in its box. (ii) If the d=5 rate is below 0.5, or the psychometric crossing sits at d >= 6 (or never), PAHI is supported and SCWI is falsified - SCWI cannot cross later than 4.74 low-cue units. (iii) If the crossing point varies widely across subjects (some never crossing by d=8), that specifically favors PAHI's heterogeneous gamma over SCWI's narrow tau. (iv) If the secondary kills G/S/M2 flip sign relative to the primary kill (e.g., high option preferred at 1.0/0.5 but not at 0.85/0.5), BOTH pure forms are falsified, since each theory's weight-shape prediction is sign-consistent across all these mixtures by construction.

[5] ADVERSARIAL LOGIC. PAHI (advocated) and SCWI (competitor) are structurally near-identical anti-validity additive integrators: a rating of 1 counts AGAINST an option, tied cues cancel, choice is a softmax over the anti-load difference plus a symmetric lapse. The ONLY substantive difference is the validity-to-weight map: PAHI uses w(v)=v^gamma with gamma in [1.4, 2.4]; SCWI uses w(v)=v+tau*v^2 with tau in [0.25, 0.45]. Three mathematical facts drive the design. (1) UNIFORM STEEPNESS ORDERING: SCWI's log-local elasticity is (1+2*tau*v)/(1+tau*v), which lies in [1.11, 1.31] for ALL v in [0.5, 1] and ALL tau in its box; PAHI's elasticity is exactly gamma >= 1.4. The elasticity ranges are DISJOINT, so for every v < 1, PAHI's top-to-low weight ratio w(1)/w(v) strictly exceeds SCWI's. (2) CROSSING-WINDOW DISJOINTNESS AT r=2: fix counts of 2 top cues (v=1.00, where PAHI's weight is exactly 1, gamma-invariant - a fixed 'currency') versus 4 low cues at validity v. The choice flips at v* solving 2*w(v) = w(1). SCWI: v* in [0.550, 0.576] across its ENTIRE parameter box - a 0.026-wide, effectively parameter-free window. PAHI: v* = 0.5^(1/gamma) in [0.610, 0.749]. The windows are disjoint, so each subject's measured flip validity classifies the theories with zero free parameters. (3) FLATNESS AT r=2.5: at counts 2:5, SCWI's flip validity v* in [0.449, 0.477] lies BELOW the floor 0.5, so SCWI must prefer the top-cue option for EVERY v in [0.5, 1] and every parameter - a parameter-free flat-H prediction. PAHI flips at 0.4^(1/gamma) in [0.520, 0.683] and must prefer the LOW option at v=0.50 for every gamma - parameter-free flat-L at the floor. Opposite pinned predictions. WHERE THIS SITS RELATIVE TO PRIOR DESIGNS: experiment [4] swept the COUNT RATIO at fixed validities (1.0 vs 0.5) and used a single knife-edge (d=5). This design is its TRANSPOSE: it fixes the count ratio and sweeps the LOW-CUE VALIDITY, making the statistic the CROSSING VALIDITY (its location, its between-subject spread, and its cross-ratio consistency), plus amplified stacked kills at validity mixtures (1.0/0.60/0.50 and 1.0/0.55) that [4] never touched. Only one cell (2x1.00 vs 5x0.50) coincides with [4]'s d=5 anchor, retained deliberately as a cross-experiment replication cell. INSTRUMENT: 28 experts - 4 at 1.00, 5 at 0.65, 5 at 0.60, 9 at 0.55, 5 at 0.50. Two ladders at r=2 (2:4) and r=2.5 (2:5) over v in {0.50, 0.55, 0.60, 0.65}, two stacked kills (4x1.00 vs 4x0.60+5x0.50; 4x1.00 vs 9x0.55, i.e. r=2.25 sitting inside the (2.11, 2.32) gap at v=0.55), all mirrored, plus tie / lapse / agreement controls. 24 unique pairs x K = 96//24 = 4 = 96 trials. KILL CELLS (parameter-free sign separation; SCWI evidence is linear in tau so endpoint checks suffice, PAHI evidence is monotone decreasing in gamma on every kill cell): R2-60 (2x1.00 vs 4x0.60): SCWI evidence +0.148..+0.26, P(top) in [0.535, 0.603]; PAHI evidence -0.044..-0.826, P(top) <= 0.487. This cell is the r=2 crossing detector: SCWI's crossing (<=0.576) is below 0.60, PAHI's (>=0.610) is above. R25-50: SCWI +0.163..+0.313, P >= 0.538; PAHI -0.106..-1.053, P <= 0.468. STK (4x1.00 vs 4x0.60+5x0.50): SCWI +0.311..+0.573, P in [0.572, 0.742]; PAHI -0.149..-1.879, P <= 0.456. STK2 (4x1.00 vs 9x0.55): SCWI +0.375..+0.630, P in [0.588, 0.733]; PAHI -0.103..-1.856, P <= 0.469. R2-55 is a REVERSE kill: SCWI sits exactly at its crossing (evidence -0.156..+0.002, P(top) in [0.464, 0.501], i.e. pinned at chance) while PAHI is confidently opposite (evidence -0.268..-1.047, P(top) <= 0.420). R25-55: SCWI P in [0.62, 0.73]; PAHI prefers the low option for 87% of its gamma range. GAMMA-MAPPING RUNGS: R2-65 (SCWI [0.61, 0.70]; PAHI splits at gamma=1.61, 79% prefer the low option), R25-60 (SCWI [0.70, 0.82]; PAHI splits at gamma=1.79, 61% low), R25-65 (SCWI [0.76, 0.88]; PAHI splits at gamma=2.13, 73% top). POPULATION SHAPE AND HETEROGENEITY SIGNATURES: plotting P(top-option) versus low validity at r=2, SCWI crosses 0.5 inside (0.55, 0.60) for EVERY subject with near-zero between-subject spread (the whole tau box moves v* by only 0.026), whereas PAHI crosses in (0.60, 0.75) with crossing = 0.5^(1/gamma) spread over 0.14 validity units - a fivefold variance dissociation in crossing location. At r=2.5, SCWI never crosses (flat above 0.5 down to v=0.50) while PAHI crosses inside (0.52, 0.68). Moreover PAHI's two crossings must be jointly consistent with a single gamma: v*_{2.5} = 0.8^(1/gamma) * v*_2, i.e. v*_{2.5} in [0.85, 0.91] x v*_2; SCWI has no observable r=2.5 crossing at all. CONTROLS: pair 21 (identical mixed vectors) pins side/response bias at exactly 0.500 for both theories; pair 22 (all-1 vs all-0) calibrates the lapse (both predict P(all-1 option) ~ epsilon/2, confirming anti-validity processing); pairs 23/24 are confident agreement cells (e.g. one 0.50-cue 1 vs two 1.00-cue 1s) calibrating the shared softmax/lapse on the anti-validity side. DECISION RULES: (i) SCWI is supported, and PAHI falsified, if P(top) > 0.5 on all four kill cells (especially STK/STK2 >= 0.57, which PAHI cannot exceed at 0.456/0.469 for any admissible parameters), the r=2 crossing lies in (0.55, 0.60) with tight between-subject clustering, r=2.5 is flat-H all the way down to v=0.50, and R2-55 sits at ~0.50. (ii) PAHI is supported, and SCWI falsified, if P(top) < 0.5 on the kill cells (STK/STK2 <= 0.47, below SCWI's 0.572/0.588 floor), the r=2 crossing lies at or above 0.61 (low-option choices persist through v=0.60 and top-option choices emerge only at 0.65+), an r=2.5 crossing is observed inside (0.52, 0.68) and is consistent with the r=2 crossing through a single gamma, R2-55 is clearly below 0.5, and crossing locations vary widely across subjects. (iii) If subjects prefer options with MORE high-validity 1s (pro-validity integration), or crossings appear below 0.55, or the profile fits neither map, both pure forms are falsified.

[6] ADVERSARIAL LOGIC. SCWI (advocated) and PAHI (competing) agree on everything except the validity-to-weight map: SCWI uses w(v) = v + tau*v^2 with tau in [0.15, 0.42]; PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]. Both are anti-validity additive integrators (a rating of 1 is a defect, tied cues cancel, softmax plus symmetric lapse), so the ONLY exploitable wedge is the SHAPE of w(v). The key mathematical fact: the local log-elasticity of SCWI's map, (1+2*tau*v)/(1+tau*v), lies in [1.13, 1.30] for ALL v in [0.5, 1] and ALL tau in its box, while PAHI's elasticity is exactly gamma >= 1.4. The ranges are DISJOINT, so for every validity pair (v_hi, v_lo) SCWI's weight ratio w(v_hi)/w(v_lo) is strictly SMALLER than PAHI's. Experiments [4] and [5] exploited this at the v = 1.00 anchor (where PAHI's weight is pinned at exactly 1). This design is the TRANSPOSE into the mid-validity region, using NO validity at 1.00: the knife-edge becomes a RATIONAL COUNT RATIO. For a trial pitting a endorsements on v_hi-cues against b endorsements on v_lo-cues, both theories prefer the high-validity-stack option iff their weight ratio is below b/a. SCWI ratio windows (over its full tau box): (0.75, 0.50): [1.552, 1.630]; (0.80, 0.50): [1.667, 1.766]; (0.85, 0.50): [1.783, 1.907]. PAHI windows (over gamma in [1.4, 2.4]): 1.5^gamma = [1.764, 2.646]; 1.6^gamma = [1.931, 3.090]; 1.7^gamma = [2.102, 3.573]. Each inter-window gap contains a simple rational: 5/3 = 1.667 and 7/4 = 1.75 in (1.630, 1.764); 9/5 = 1.80 in (1.766, 1.931); 2 in (1.907, 2.102). Placing the count ratio exactly at these rationals yields KILL CELLS with OPPOSITE sign predictions for EVERY admissible parameter value of BOTH theories.

KILL CELLS (evidence for the stated option, positive for all parameters): B3 = 6x0.75 vs 10x0.50 (ratio 5/3): SCWI prefers the 0.75-stack with evidence 0.5 - 0.875*tau in [0.13, 0.37]; PAHI prefers the 0.50-stack with evidence 0.5^gamma*(6*1.5^gamma - 10) in [0.22, 1.11]. B4 = 4x0.75 vs 7x0.50 (ratio 7/4): SCWI evidence 0.5 - 0.5*tau in [0.29, 0.43]; PAHI evidence in [0.02, 0.68]. K3 = 5x0.80 vs 9x0.50 (ratio 9/5): SCWI evidence 0.5 - 0.95*tau in [0.10, 0.36]; PAHI evidence in [0.25, 1.23]. A3 = 4x0.85 vs 8x0.50 (ratio 2): SCWI evidence 0.6 - 0.89*tau in [0.23, 0.47]; PAHI evidence in [0.16, 1.19]. Intuition: PAHI's steep power law makes mid-validity endorsements relatively heavy, so for PAHI a 1 on a 0.75-cue outweighs 5/3 endorsements on 0.50-cues; SCWI's near-linear-quadratic map makes them equivalent-or-lighter, so SCWI needs the count ratio to reach ~1.75-2.1 before flipping. POPULATION PREDICTIONS (SCWI mixture 16% guessers / 50% mid / 34% deterministic; PAHI with beta in [1.5,4], gamma in [1.4,2.4], epsilon in [0.06,0.2]): kill composite (48 trials/subject, pairs 1-12) SCWI = 0.62-0.72 (point ~0.66); PAHI = 0.15-0.45 (point ~0.25). Every non-guesser SCWI subject sits >= 0.55; every PAHI subject sits <= 0.50.

LADDER / FLIP-LOCATION DISSOCIATION. Bracketing rungs pin each theory's crossing point, per subject, at two independent validity pairs. On the 0.75/0.50 ladder: SCWI's flip ratio lies in (3/2, 5/3) for EVERY subject (low-stack at r = 3/2 [pair 13], high-stack at r = 5/3 [pairs 1-4]); PAHI's flip ratio is >= 1.764, hence at or above the 7/4 rung (low-stack at 5/3 AND at 7/4 [pairs 5-6], high-stack at r = 2 only if gamma < 1.71 [pairs 15-16]). On the 0.85/0.50 ladder: SCWI flips in (7/4, 2) (low-stack at 7/4 [pairs 17-18], high-stack at 2 [pairs 9-12]); PAHI flips above 2 (low-stack at 2, high-stack at 9/4 only if gamma < 1.53 [pairs 19-20]). A systematic ONE-RUNG SHIFT in the psychometric crossing over count ratio, replicated across two validity mixtures -- a quantitative signature SCWI's tau-box cannot cross into and PAHI's gamma-box cannot cross out of.

GAMMA-RUNGS (PAHI-specific consistency test). B5 (6x0.75 vs 12x0.50) splits PAHI exactly at gamma = 1.71; A4 (4x0.85 vs 9x0.50) splits PAHI at gamma = 1.53. SCWI predicts the HIGH-stack on both with huge margins (1.34-1.44 and 0.83-1.00): population P(high-stack) ~ 0.85 vs PAHI's ~0.30-0.40. A PAHI subject's rung and kill choices must be jointly consistent with a single gamma (e.g., high-stack on A4 implies gamma < 1.53, which forces high-stack on B5); SCWI predicts high-stack on all rungs uniformly, with essentially zero between-subject spread in crossing location (the whole tau box moves the 0.75-ladder crossing ratio by only 0.078, all inside the same rung gap), whereas PAHI's crossings spread over ~0.9 ratio units.

POPULATION-MIXTURE PROBE (novel; untouched by experiments [0]-[5]). SCWI's beta_seed mixture places 16% of subjects in a true-guesser regime (beta in [0.02, 0.10], epsilon in [0.12, 0.28]). On the all-1-vs-all-0 anchors (pairs 21-22) the total anti-load is ~23.5 (SCWI units) or ~17 (PAHI units), so these guessers sit at P(choose all-1) in [0.22, 0.35] -- and at ~0.5 on every kill and rung. PAHI has NO such regime: every subject has beta >= 1.5 and epsilon <= 0.20, so P(choose all-1) <= epsilon/2 <= 0.10 for EVERY subject. Statistic: fraction of subjects choosing the all-1 option on >= 3 of 8 anchor trials: SCWI ~ 0.08-0.13; PAHI ~ 0.01 (and no PAHI subject can exceed ~0.12 in true probability). SCWI additionally predicts a TRIMODAL per-subject anchor distribution (masses near 0.02, 0.05, and 0.3) with guesser-flagged subjects also at chance on kills/rungs; PAHI predicts a unimodal-low continuum.

CONTROLS. Pairs 13-14 (B2: 4x0.75 vs 6x0.50, r = 3/2) and 17-18 (A2: 4x0.85 vs 7x0.50, r = 7/4) are agreement cells where BOTH theories prefer the low-stack (SCWI evidence 0.75*tau in [0.11, 0.32] and 1.14*tau - 0.1 in [0.07, 0.38]; PAHI evidence 0.40-0.87 and 0.53-1.31): they calibrate the shared softmax/lapse on the anti-validity side and identify non-inverted TTB subjects (who pick the high-stack because the top discriminating cue favors it -- note TTB also picks the high-stack on the kills, so B2/A2 and the anchors are what exclude it) and pro-validity integrators (who pick all-1 on anchors). Pair 24 (CAL: one 0.85-endorsement vs one 0.50-endorsement) is a single-cue anti-validity calibration (both theories prefer the 0.50-1 option, P ~ 0.75-0.85). Pair 23 (TIE: identical mixed vectors) pins side/response bias at exactly 0.500 for both. All critical pairs are mirrored; B3 and A3 include cue-rotation variants (different low-cue subsets) to average within-subject per-cue attention (drawn from [0.90, 1.10]); kills that use every cue of a class (K3, B5) average attention by construction, and the population-level sign predictions are immune to attention because it is symmetric with mean 1. Worst-case attention analysis: flipping B3 requires ~8% adverse class-mean attention (~2 SD of a 10-cue class mean); B4/B5/A4 margins (0.29-1.44) require 15-30% and cannot flip.

DESIGN. 29 experts (6 at 0.85, 5 at 0.80, 6 at 0.75, 12 at 0.50); 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials. Per subject: 48 kill trials, 32 rung trials, 8 anchor trials, 4 tie trials, 4 calibration trials.

DECISION RULES. (i) SCWI is supported, and PAHI falsified, if the kill composite P(high-stack) > 0.55 with per-subject high-stack majorities (PAHI cannot exceed 0.50 on any kill for any admissible parameters), the rung composite >= 0.75, the 0.75-ladder crossing lies in (1.5, 1.667) and the 0.85-ladder crossing in (1.75, 2) with tight between-subject clustering, and the anchor-tail fraction (subjects >= 3/8 all-1 choices, also near chance on kills) is ~0.10-0.15 where PAHI predicts ~0.01. (ii) PAHI is supported, and SCWI falsified, if all four kills show low-stack preference (composite < 0.45), the rung pattern is jointly consistent with a single gamma per subject (A4 high-stack => B5 high-stack; A4 low-stack with gamma > 1.53 => correspondingly graded B5), crossings sit at or above 5/3 (0.75-ladder) and at or above 2 (0.85-ladder) with wide between-subject spread, and the anchor distribution is unimodal-low with no subject above ~0.12. (iii) BOTH pure forms are falsified if subjects prefer the option with MORE high-validity endorsements (pro-validity: all-1 chosen on anchors, low-stack rejected on kills), or if a flat anti-tallying pattern appears (low-stack on EVERY cell including the B5/A4 rungs, which both theories reject), or if the profile fits neither map.

[7] ADVERSARIAL LOGIC. PAHI (advocated) and SCWI (competing) are structurally near-identical anti-validity additive integrators: a rating of 1 counts AGAINST an option, tied cues cancel exactly, and choice is a softmax over the anti-load difference plus a symmetric lapse. The ONLY exploitable wedge is the validity-to-weight map: PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]; SCWI uses w(v) = v + tau*v^2 with tau in [0.15, 0.42]. The exact weight-RATIO windows w(v_h)/w(v_l) over the FULL parameter boxes, for three validity pairs forming a triangle (none explored in [0]-[6], which used 1.00/0.50, 1.00/0.55-0.65, and 0.75-0.85/0.50): (0.90, 0.70): SCWI [1.321, 1.369] vs PAHI [1.422, 1.828]; (0.70, 0.50): SCWI [1.439, 1.497] vs PAHI [1.602, 2.243]; (0.90, 0.50): SCWI [1.900, 2.050] vs PAHI [2.277, 4.098]. In every pair the windows are DISJOINT and each gap contains a rational count ratio: 7/5 = 1.40, 14/9 = 1.556, 11/5 = 2.20. A trial pitting a endorsements on high-validity cues against b on low-validity cues (all other cues 0 for both options) is therefore a parameter-free SIGN KILL at those ratios: SCWI prefers the high-validity stack H (its ratio is below b/a for EVERY tau) while PAHI prefers the low stack L (its ratio is above b/a for EVERY gamma). KILL MARGINS: P1-K (5x0.90 vs 7x0.70): SCWI H by 0.14-0.31, PAHI L by 0.10-1.35; P2-K (9x0.70 vs 14x0.50): SCWI H by 0.32-0.56, PAHI L by 0.11-0.60; P3-K (5x0.90 vs 11x0.50): SCWI H by 0.45-0.81, PAHI L by 0.15-1.80. Population kill composite (18 trials/subject): SCWI ~0.55-0.79 (16% guessers at 0.50, non-guessers 0.55-0.88 with beta 1.8-6.0); PAHI ~0.15-0.45 (margins against H, beta 1.5-4.0, epsilon 0.06-0.20).

NOVELTY VS PRIOR DESIGNS. (1) The (0.90, 0.70) pair is the CLOSEST validity pair ever pitted: the two maps' absolute levels nearly coincide (SCWI 1.02-1.24 vs 0.77-0.91; PAHI 0.78-0.86 vs 0.36-0.60), so only relative steepness discriminates — a pure shape test with no level confound. (2) IMPLIED-EXPONENT TRIANGULATION (headline statistic): for a subject whose choices flip from L to H at count ratio r* on pair (v_h, v_l), define gamma_hat = ln(r*)/ln(v_h/v_l). PAHI: gamma_hat = gamma in [1.4, 2.4] at EVERY pair, jointly consistent with a single gamma, with large between-subject spread (SD ~0.29). SCWI: gamma_hat in [1.08, 1.25] at every pair (P1 [1.11, 1.25]; P2 [1.08, 1.20]; P3 [1.09, 1.22]) — strictly below PAHI's floor of 1.4 for ANY admissible parameters, with tight spread (SD ~0.05). The exponent LEVEL is thus a parameter-free classifier replicated three times, plus a fivefold between-subject variance dissociation. (3) A within-subject CONSISTENCY WEB: SCWI predicts H on every cell with r >= 1.5 (all six gamma-rungs and all three kills) for every subject regardless of tau — a step function crossing in (1.25, 1.40) at 0.9/0.7, (1.40, 1.556) at 0.7/0.5, (1.75, 2.20) at 0.9/0.5. PAHI predicts L on all kills and brackets, with rungs splitting at gamma = 1.52 (P2-C), 1.61 (P1-C), 1.66 (P2-D), 2.06 (P2-E), 2.13 (P3-E), 2.23 (P1-E); e.g., a PAHI subject choosing H on P1-E but L on P2-E has gamma in (2.06, 2.23), which FORCES L on P3-E, P2-D, P1-C and all kills; H on P1-C forces H on P2-C/D/E and P3-E. SCWI cannot generate any such covariance — it predicts H on all rungs unconditionally.

RUNGS AND BRACKETS. P1-B (r = 4/3) and P3-B (r = 2) are SCWI tau-rungs splitting at tau = 0.213 and tau = 0.323 (with the ordering constraint H on P1-B => H on P3-B); PAHI is confidently L on both (margins 0.18-0.88), so 'H on P1-B' alone is nearly a kill for low-tau SCWI. P1-A (r = 1.25), P2-A (r = 1.40), P3-A (r = 1.75) are agreement-L brackets (both theories prefer the low stack, margins 0.10-1.78); P1-F (r = 2) is the agreement-H bracket. TIE (identical vectors) pins side/response bias at exactly 0.500 for both theories; ANCHOR (all-1 vs all-0, total anti-load ~21 SCWI / ~12 PAHI units) calibrates the lapse and screens out pro-validity integrators and PAHI's rare s = +1 subjects, who choose the all-1 option. ATTENTION ROBUSTNESS: kill stacks use 5-14 cues per side; flipping a kill requires class-mean attention distortions of 2-8% (1-3 SD of the class-mean of SCWI's per-cue attention in [0.90, 1.10]); P2-K/P3-K need > 2 SD, and population-level sign predictions are immune because attention is symmetric with mean 1. No feedback keeps weights anchored to the instructed validities; every critical trial requires comparing many cues, satisfying the min-RT viewing requirement.

DESIGN. 28 experts (5 x 0.90, 9 x 0.70, 14 x 0.50); 32 unique pairs x K = 96//32 = 3 repetitions = 96 trials (18 kill, 12 tau-rung, 36 gamma-rung, 24 bracket, 3 tie, 3 anchor trials). All critical pairs are mirrored to absorb side bias.

DECISION RULES. (i) SCWI is supported, and PAHI falsified, if the kill composite P(H) >= 0.55 with per-subject H-majorities (PAHI cannot exceed 0.50 on any kill for any admissible parameters), H-majority on all six gamma-rungs, rung-implied exponents clustering in [1.05, 1.30] at all three pairs with between-subject SD < 0.08, and a coherent P1-B/P3-B tau pattern. (ii) PAHI is supported, and SCWI falsified, if the kill composite P(H) <= 0.45 (SCWI's population floor is ~0.55), rung choices are jointly consistent with a single gamma in [1.4, 2.4] per subject (any violation of the consistency web — e.g., H on P1-E with L on P2-C — falsifies PAHI), implied exponents >= 1.4 at all three pairs with wide between-subject spread, and ANCHOR choices at the lapse floor. (iii) BOTH are falsified if the all-1 option is chosen well above lapse on ANCHOR (pro-validity integration), if kills and rungs sit flat at 0.5 (pure guessing), or if implied exponents fall in the empty band (1.25, 1.40) or fit neither map.

[8] ADVERSARIAL LOGIC. SCWI (advocated) and HEAI (competing) are near-identical anti-validity additive integrators: a rating of 1 is a defect, tied cues cancel, softmax plus symmetric lapse, 16% guessers. Moreover, HEAI's SHALLOW mode (75% of engaged, gamma ~ N(1.04, 0.22)) is almost exactly degenerate with SCWI's map w(v) = v + tau*v^2 (tau in [0.15, 0.42], implied power-exponent 1.10-1.23 at every validity pair): their weight-ratio windows overlap at every mixture I could construct, so no single-cell sign kill between them exists. The honest adversarial move is therefore to target HEAI's DISTINCTIVE claim — the BIMODAL gamma, specifically the STEEP minority (25% of engaged = 21% of population, gamma in [1.65, 2.75], median 2.10) — whose validity gradient is far steeper than anything in SCWI's box, and to exploit the resulting POPULATION and DISTRIBUTIONAL dissociations, which SCWI's unimodal near-linear-quadratic map cannot reproduce for any parameters.

MATHEMATICAL SETUP. With 3 cues at v=1.00 (where HEAI's weight is exactly 1, gamma-invariant) pitted against b cues at v=0.50, an anti-validity integrator prefers the TOP-STACK option H (the 3 high-validity 1s) iff 3*w(1.00) < b*w(0.50). SCWI's margin (defect_L - defect_H) = (0.5b - 3) + tau*(0.25b - 3): NEGATIVE (prefers the low stack) for b <= 6, and POSITIVE for every b >= 8 with margin 1 - tau in [0.58, 0.85] at b=8 up to EXACTLY 3.0 at b=12 (parameter-free anchor: the tau-term cancels). No SCWI parameter value, no attention draw (class-mean attention shifts the margin by SD ~0.12 at worst), and no beta/epsilon can make an engaged SCWI subject prefer the low stack at b >= 8. HEAI's flip occurs at gamma* = log2(b/3): b=8 -> 1.415, b=9 -> 1.585, b=10 -> 1.737, b=11 -> 1.874, b=12 -> 2.0. EVERY steep subject (gamma >= 1.65) prefers the LOW stack at b=8 and b=9 with large margins (e.g. at gamma=2.10, b=9: margin 0.90), and flips to H only as b/3 exceeds 2^gamma. A second, independent ladder at v=0.60 (3x1.00 vs b cues, gamma* = ln(b/3)/ln(5/3)) replicates the instrument at a different validity mixture: SCWI's flip ratio there is in [1.76, 1.89], so SCWI prefers H at b >= 6 (margin 0.25-0.47) for all tau, while steep subjects flip at gamma* = 1.357 (b=6), 1.658 (b=7), 1.921 (b=8), 2.151 (b=9).

PREDICTED DISSOCIATIONS. (S1) KILL COMPOSITE (cells L2, L3, M2, M3 + mirrors, gamma* <= 1.66, 32 trials/subject): SCWI population P(H) ~ 0.80; HEAI ~ 0.62-0.66 (shallow confidently H, steep confidently L, guessers 0.5). (S2) SUBJECT-LEVEL KILL, parameter-free: on L3 (3x1.00 vs 9x0.50) SCWI's margin is >= 1.19 for EVERY admissible (tau, attention, regime) — no engaged SCWI subject can prefer the low stack. HEAI predicts ~21% of subjects prefer it with P ~ 0.8-0.9. The fraction of subjects with an L-majority on the kill composite is ~0-3% under SCWI vs ~22-25% under HEAI, and under HEAI those subjects form a distinct LOW MODE in the between-subject distribution (bimodal), whereas SCWI predicts only guesser mass near 0.5 and engaged mass >= 0.6. (S3) FLIP-LOCATION CEILING: SCWI's implied flip exponent is bounded in [1.10, 1.23] at (1.0, 0.5) — BELOW the lowest kill rung (gamma* = 1.357 at M2). So under SCWI, NO subject's psychometric flip (low-stack -> top-stack) lies inside the observable ladder; under HEAI, ~21% of subjects flip inside [1.4, 2.2], with the flip point estimating their personal gamma. (S4) CROSS-LADDER CONSISTENCY WEB: a HEAI steep subject's flips on the 0.50-ladder (gamma* = log2(b/3)) and the 0.60-ladder (gamma* = ln(b/3)/ln(5/3)) must recover the SAME gamma and be monotone in gamma*; e.g. L at L3 (gamma* 1.585) plus H at M5 (gamma* 2.151) implies gamma in (1.585, 2.151), which forces L at L2, M2, M3 and H at L6 only if gamma < 2.0. SCWI predicts no flips at all, hence no such covariance; L-choices across kill cells are uncorrelated noise under SCWI and strongly correlated (latent steep class) under HEAI — a testable correlation difference at matched marginal rates. (S5) PARAMETER-FREE ANCHOR L6 (3x1.00 vs 12x0.50): SCWI's margin is exactly 3.0 for every tau (the tau-coefficient 0.25*12 - 3 vanishes), so P(H) >= 1 - epsilon for every engaged subject; HEAI's steep subjects with gamma > 2.0 (65% of steep) still prefer the low stack there. POPULATION PSYCHOMETRICS over gamma*: SCWI is flat-high and saturated early (0.80 -> 0.92 from gamma* = 1.415 to 2.15, rising only through margin growth); HEAI is graded and rises late (0.64 -> 0.80), with the shallow mode's thin upper shoulder (gamma in [1.32, 1.42]) adding a small extra L-mass at L2 that SCWI cannot generate.

CONTROLS AND EXCLUSION OF THIRD PARTIES. L-C (3x1.00 vs 5x0.50, r = 5/3) and M-C (3x1.00 vs 4x0.60, r = 4/3) sit BELOW every theory's flip ratio: both SCWI and HEAI (including all steep subjects) confidently prefer the LOW-validity stack there. These controls (i) anchor the bottom of the psychometric function, and (ii) exclude confounded strategies that mimic the kill predictions: a TTB subject follows the top cue and picks the top stack on EVERY cell (kills AND controls — the controls expose it); a 'fewer-1s' heuristic likewise picks the 3-1 option everywhere including controls; a tallying/more-1s subject picks the low stack on kills (mimicking steep-HEAI) but is exposed by the ANCHOR (all-1 vs all-0), where both target theories predict P(all-1) ~ epsilon/2 while tallying predicts ~1. TIE (identical mixed vectors) pins side/response bias at exactly 0.500 for both theories. ANCHOR also calibrates the shared lapse and screens pro-validity integrators. ATTENTION ROBUSTNESS: kill stacks use 3 vs 6-12 cues per side, so SCWI's per-cue attention jitter (class-mean SD 0.02-0.03) shifts margins by ~0.05-0.12 against margins >= 0.25 (M2, worst case) — no flip; the L3+ cells are z >= 10 safe. All critical pairs are mirrored to absorb side bias. Every trial requires comparing large multi-cue stacks (satisfying the min-RT viewing requirement), and with 24 unique pairs, K = 96 // 24 = 4 repetitions = 96 trials per subject (32 kill, 40 rung, 16 control, 4 tie, 4 anchor).

DECISION RULES. (i) SCWI is supported, and HEAI falsified, if the kill composite sits at ~0.78-0.85 with NO subject (beyond guesser noise at ~0.5) showing an L-majority on L3/L4/L5/L6 or M3/M4/M5, no psychometric flips inside the ladder, L6 at ~0.92, and a unimodal-high plus guesser-tail subject distribution. HEAI's steep mode (21% of the population confidently anti-following at gamma* <= 1.6) cannot be produced by SCWI at any parameter values. (ii) HEAI is supported, and SCWI falsified, if ~20% of subjects confidently prefer the low stack on the low-gamma* kills, their flips lie inside the ladder and are jointly consistent with a single gamma across BOTH ladders (monotone in gamma*, gamma_hat agreeing between the 0.50- and 0.60-ladders), the kill composite sits at ~0.62-0.68, and the between-subject kill distribution is bimodal with an empty band between the shallow cluster (flips below 1.4) and the steep cluster (flips in [1.65, 2.4]). Any subject with an L-majority on L3 alone falsifies SCWI outright, since SCWI's margin there is >= 1.19 across its entire parameter and attention box. (iii) BOTH are falsified if the all-1 option is chosen well above lapse on ANCHOR (pro-validity integration or tallying), if the top stack is preferred on the L-C/M-C controls (TTB or fewer-1s heuristic), or if flips appear below gamma* = 1.3 or fit neither a power law nor the quadratic map.

[9] ADVERSARIAL LOGIC. HEAI (advocated) and SCWI (competing) are mechanistically identical anti-validity additive integrators: a rating of 1 is a defect, tied cues cancel exactly, choice is a softmax plus symmetric lapse, and both carry a 16% true-guesser subpopulation. The only wedge is the validity-to-weight map. Every prior experiment ([4]-[8]) attacked this wedge from the STEEP side (the old gamma-box [1.4, 2.4] and HEAI's steep mode), using count ratios at or above each theory's flip window. The unexploited territory is HEAI's SHALLOW mode: 75% of engaged subjects with gamma ~ N(1.04, 0.22) on [0.60, 1.42] - i.e. mostly SUBLINEAR or barely supralinear. SCWI's implied power exponent is pinned by its tau-box to [1.10, 1.23] at the (1.00, 0.50) pair (ln[(1+tau)/(0.5+0.25tau)]/ln 2) and [1.09, 1.22] at (1.00, 0.60). Hence ~60% of shallow subjects have validity gradients SHALLOWER than ANY admissible SCWI subject: they prefer the high-validity stack at count ratios where SCWI, for every (tau, attention, beta, epsilon), prefers the low-validity stack. No prior design sampled this ratio region: [4] and [8] placed rungs at r >= 2.5-2.67, [5] swept low-cue validity at fixed r, and their kills all targeted gamma >= 1.4.

INSTRUMENT: THE RECIPROCAL-RATIO LADDER. With a endorsements on v=1.00 cues (where HEAI's weight is EXACTLY 1, gamma-invariant) against b endorsements on v_l cues, an anti-validity integrator prefers the top-stack option H iff w(1)/w(v_l) < b/a = r. SCWI's flip ratio is (1+tau)/(v_l + tau*v_l^2): [2.14, 2.35] at v_l=0.50, [1.76, 1.89] at v_l=0.60. HEAI's flip ratio is (1/v_l)^gamma, so the cell with b = a/v_l (r = 2.0 on ladder 1: 4x1.00 vs 8x0.50; r = 5/3 on ladder 2: 3x1.00 vs 5x0.60) flips EXACTLY at gamma = 1 - the shallow mode's median. These TWIN KILLS (pairs 3, 9 and 7, 10, plus mirrors) are parameter-free for SCWI: its margin is 2*tau (ladder 1) and 1.2*tau (ladder 2), strictly positive for ALL tau in [0.15, 0.42], so EVERY engaged SCWI subject prefers the low stack (per-trial P(H) <= 0.37 even at the tau=0.15 edge with mid beta=1.8). HEAI instead splits its shallow mode at gamma=1: the 43% with gamma < 1 prefer H with margins 0.3-1.3 at beta median 2.6 (P(H) per trial 0.6-0.95), the steep mode prefers L, guessers sit at 0.5.

QUANTITATIVE POPULATION PREDICTIONS, P(H): L1a (r=1.5): both <= 0.12 (parameter-free agreement-L: flip needs gamma < 0.585, below HEAI's 0.60 truncation, and SCWI margin 1+2.5tau > 0). L1b (r=1.75): SCWI <= 0.15, HEAI ~0.16. L1c/K2rot (r=2.0): SCWI 0.20-0.33, HEAI ~0.36-0.40 - HEAI ABOVE SCWI's ceiling. L1d (r=2.25): SCWI ~0.50 (its internal tau-rung, splitting at tau=0.286 with margins <= 0.24), HEAI ~0.42-0.46. L1f (r=3.0, mode separator): SCWI 0.86-0.92 (margin 2-tau in [1.58, 1.85], every engaged subject >= 0.9), HEAI 0.66-0.72 - HEAI BELOW SCWI's floor. L2a (r=4/3): both <= 0.12 (agreement-L; HEAI flip needs gamma < 0.563, outside its box). L2b/K4rot (r=5/3): SCWI 0.18-0.30, HEAI ~0.38. L2c (r=2.0): SCWI 0.85-0.90, HEAI 0.55-0.60. So the two theories' psychometric curves CROSS TWICE with opposite signs in different ratio regions: HEAI exceeds SCWI exactly where the shallow mode's sub-linearity bites (r = 1/v_l), and SCWI exceeds HEAI where HEAI's steep mode (21% of the population, confidently L everywhere) and guessers cap the composite. SCWI cannot be simultaneously above 0.84 at r=3.0/L2c and below... it is PINNED: P(H) <= 0.33 on the twin kills and >= 0.84 on L1f for ALL parameters. Observing twin-kill >= 0.35 OR L1f <= 0.80 falsifies SCWI outright; observing L1f >= 0.84 with twin-kill <= 0.30 falsifies HEAI (whose L1f ceiling is 0.71 even if every shallow subject were deterministic).

SUBJECT-LEVEL (PRIMARY) STATISTICS. (S1) H-majority (>= 60% H choices) on the 32-trial twin-kill composite (pairs 3, 7, 9, 10 + mirrors, 4 reps each): HEAI predicts 20-27% of subjects (shallow gamma <= ~0.9); SCWI predicts < 3-5% (only guesser binomial flukes - an engaged SCWI subject's per-trial P(H) never exceeds 0.37, so a 32-trial majority is vanishingly unlikely). (S2) L-majority at the mode separator L1f (8 trials): HEAI ~20-22% (the steep mode; gamma* = log2(3) = 1.585 lies in the EMPTY inter-mode band (1.42, 1.65), so HEAI's per-subject distribution there is bimodal with an empty middle - shallow >= 0.9, steep <= 0.1); SCWI ~0-2%, unimodal-high. (S3) FLIP LOCATION on ladder 1: SCWI's every engaged subject steps L (r=2.0) -> chance (2.25) -> H (3.0), flip confined to [2.14, 2.35] with near-zero between-subject spread; HEAI's flips spread over r = 2^gamma in [1.52, 2.68] (graded across the 1.75/2.0/2.25 rungs for roughly half of engaged subjects), and 21% of subjects NEVER flip (steep: L even at r=3.0, since 1.65 > 1.585). (S4) CROSS-LADDER CONSISTENCY WEB: a shallow subject's two flip ratios must satisfy r2* = r1*^(ln(5/3)/ln 2) = r1*^0.737 with a single gamma in [0.6, 1.4]; e.g. H at BOTH twin kills implies gamma < 1; H at L1c but L at L2b violates any single gamma and falsifies HEAI's per-subject homogeneity. Under SCWI, H at either twin kill is itself near-impossible, so SCWI generates no such covariance. (S5) STEP-SIZE profile: SCWI concentrates essentially its entire rise into single inter-rung intervals (ladder 2: P(2.0) - P(5/3) ~ 0.65; ladder 1: P(3.0) - P(2.25) ~ 0.4); HEAI's rise is spread and capped (ladder 2 difference ~0.19; ladder 1 rise ~0.26 with ceiling 0.70).

CONFOUNDS AND THIRD PARTIES. Bottom anchors L1a/L2a are parameter-free agreement-L cells for BOTH theories (all shallow gamma >= 0.60 > 0.585/0.563, all SCWI margins >= 0.83): a TTB subject (follows the top discriminating cue) and a fewer-1s heuristic both pick the 4-1 top stack there and are exposed; a tallying/more-1s subject picks the low stack on every ladder cell, mimicking anti-validity, but is exposed by the ANCHOR (all-1 vs all-0), where both target theories predict P(choose all-1) ~ epsilon/2 while tallying predicts ~1; pro-validity integrators are likewise exposed. TIE pairs pin side/response bias at exactly 0.500. All critical pairs are mirrored. SCWI's per-cue attention (U[0.9, 1.1]) is defeated by cue rotation: the twin kills use four different cue subsets spanning all three validity classes (top cues 0-3 vs 1-3; 0.50-cues 10-17 vs 14-21; 0.60-cues 4-8 vs 5-9), so an H-majority on the composite requires ~2 SD adverse class-mean attention on three independent classes simultaneously PLUS tau at its floor; the joint probability is < 2-3%. L1f's SCWI margin (1.58-1.85) needs > 5 SD to flip. HEAI has no attention parameter at all. No feedback keeps weights anchored to the instructed validities; every critical trial requires comparing large multi-cue stacks (4-12 cues per side), satisfying the min-RT viewing requirement.

DESIGN. 22 experts: 4 at 1.00, 6 at 0.60, 12 at 0.50. 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials (40 ladder-1, 24 ladder-2, 16 rotation, 8 tie, 8 anchor trials per subject).

DECISION RULES. (i) HEAI is supported, and SCWI falsified, if >= 15% of subjects show H-majority on the twin-kill composite (SCWI ceiling ~5%), psychometric flips appear inside r in (1.6, 2.1) and are jointly consistent across both ladders with a single gamma-hat in [0.6, 1.4], ~20% of subjects show the steep signature (L at every rung including r=3.0, producing the bimodal empty-middle distribution at L1f), and the population profile shows P(H|twin kill) >= 0.35 with P(H|r=3.0) <= 0.80 - a joint pattern outside SCWI's closed parameter box. (ii) SCWI is supported, and HEAI falsified, if the twin-kill composite <= 0.30 with no H-majority subjects beyond guesser noise, L1f >= 0.84 and L2c >= 0.85, every subject's flip confined to (2.0, 3.0) on ladder 1 and (5/3, 2.0) on ladder 2 with near-zero between-subject spread, and the whole rise concentrated in single inter-rung steps. (iii) BOTH are falsified if the top stack is chosen on the bottom anchors (TTB / fewer-1s / pro-validity), the all-1 option is chosen well above lapse on the ANCHOR (tallying / pro-validity), everything sits flat at 0.5 (pure guessing), flips appear below r = 1.6 (gamma < 0.68, outside HEAI's truncation and below SCWI's window), or the profile fits neither map.

[10] ADVERSARIAL LOGIC. SCWI (advocated) and CNNAI (competing) are near-identical anti-validity additive integrators: a rating of 1 is a defect, tied cues cancel exactly, softmax plus symmetric lapse, plus a true-guesser subpopulation (16% vs 13%). Moreover, CNNAI's effective validity exponent is 1+eta with eta ~ N(0.12, 0.21) per context bucket (clipped [-0.25, 0.70], exponent in [0.75, 1.70]), which is CENTERED EXACTLY ON SCWI's tau-implied exponent (w(v)=v+tau*v^2, tau in [0.15, 0.42], implies power exponent 1.09-1.25 at every validity pair). The engaged bulk of CNNAI was deliberately calibrated to behave SCWI-like in curvature, so NO single-cell sign kill exists at the population median. The two real wedges are: (W1) CONTEXT-DEPENDENCE: CNNAI's weights are w_j = -v_j*(v_j/v_ref)^eta with v_ref = mean validity of the cues endorsed on the CURRENT trial, and its eta is REDRAWN independently for every coarse bucket of v_ref (rounded to 0.1), with cross-bucket consistency fixed at r ~ 0.18 by construction (stable component sd 0.09, bucket redraw sd 0.19). SCWI's weights are strictly context-free (per-cue attention is fixed across trials; tied cues cancel exactly). (W2) EXPONENT SPREAD: CNNAI's per-subject, per-bucket exponent has sd 0.21; SCWI's tau-box confines the implied exponent to a 0.13-wide band (flip-ratio window at (0.90, 0.55) of only [1.716, 1.832], plus small attention jitter), roughly a 3-5 fold variance difference.

THE CONTEXT-ONLY KILL (primary, parameter-free against SCWI). Because tied cues (rated 1 by BOTH options) cancel exactly in both theories, I can manipulate CNNAI's context variable v_ref WITHOUT touching the discriminating evidence: add tied endorsements on filler experts whose validities shift the mean endorsed validity across a bucket boundary. SCWI's per-trial choice probability on such a pair is provably IDENTICAL to the untwinmed version for EVERY admissible (tau, attention, beta, epsilon, regime) -- the filler cues enter both options' scores equally. CNNAI instead (i) redraws eta independently (new bucket key -> fresh hash draw) and (ii) rescales every margin by v_ref^(-eta). Near CNNAI's knife this is a large effect: the per-bucket eta redraw (sd 0.19) maps to a between-bucket logit sd of roughly 1.5 at the near-knife rungs (margin slope ~2.6/eta-unit, beta ~2.14), so a subject at p=0.5 in one bucket sits anywhere from ~0.2 to ~0.8 in another. Concretely: pairs L2/L3/L4 (rungs r=1.667, 1.75, 1.833 of a 0.90-vs-0.55 count-ratio ladder, all with v_ref ~ 0.64, bucket 0.6) are twinned by T1/T2/T3 (the IDENTICAL conflicts plus tied 1.00-validity fillers, v_ref ~ 0.76, bucket 0.8) and by U2/U1 (same conflicts plus fewer 1.00-fillers, v_ref ~ 0.72, bucket 0.7). PRIMARY STATISTIC: per-subject near-knife composite (rungs 1.667+1.75+1.833; 18 trials per bucket) across the three buckets, versus a split-half of the SAME bucket as the noise baseline. SCWI: across-bucket variance = within-bucket split-half variance (ratio ~ 1), cross-bucket rate correlation at the binomial noise ceiling (~0.8 observed), ladder-in-bucket-0.6 predicts bucket-0.8 twin choices at ~0.75-0.85 accuracy. CNNAI: across-bucket variance >> split-half variance (latent ratio ~ 20-40x), cross-bucket correlation ~ 0.15-0.25, transfer accuracy ~ 0.55. Any reliable excess of across-bucket over within-bucket inconsistency, or any single subject whose per-trial preference SIGN flips between twinned contexts beyond binomial noise, falsifies SCWI outright -- SCWI cannot produce it for any parameter values. Conversely, perfect cross-bucket consistency falsifies CNNAI, whose bucket-keyed eta redraw is hard-wired.

THE LADDER (secondary: exponent spread and population shape). All six ladder rungs (r = 1.5, 1.667, 1.75, 1.833, 2.0, 2.25 at validity pair 0.90/0.55, a=4-6 endorsements on 0.90-cues vs b=6-11 on 0.55-cues) share bucket 0.6 via 0.50-validity tied fillers, so CNNAI behaves as a FIXED-exponent integrator within the ladder and each subject's flip location estimates 1+eta cleanly. SCWI: every subject's flip lies in the window (1.66, 1.90) (core window [1.716, 1.832] widened only by rare adverse class-mean attention), between-subject flip sd ~ 0.05-0.07, and NO engaged subject's per-trial P(L-stack) at r=2.0 exceeds ~0.32 (evidence >= 0.455 for all tau; weakest mid-regime bound sigmoid(-1.8*0.455) with lapse). CNNAI: flips spread over (1.45, 2.31) with sd ~ 0.18; ~9% of engaged subjects (eta > 0.41) prefer the L-stack at r=2.0 and ~35% prefer the H-stack at r=1.667 (SCWI: 0% -- knife tau = 0.055, below the tau floor). Population psychometric over r (approximate): SCWI ~ 0.20, 0.26, 0.46, 0.64, 0.78, 0.85; CNNAI ~ 0.29, 0.42, 0.51, 0.61, 0.70, 0.79. CNNAI's curve is systematically FLATTER: rise P(2.0)-P(1.667) ~ 0.28 vs SCWI ~ 0.50; CNNAI sits ABOVE SCWI at the bottom rungs (P(H|1.667) >= ~0.38 vs <= ~0.32) and BELOW at the top (P(H|2.0) <= ~0.72 vs >= ~0.75). A minor CNNAI-only signature: bucket-0.8 twins are slightly SOFTER (margins scaled by (0.64/0.76)^eta, ~2-4 points toward 0.5 for the eta>0.4 tail) -- SCWI predicts exactly zero change.

CONTROLS AND THIRD PARTIES. F1 twinning L3 within bucket 0.6 (k=3 vs k=6 tied 0.50-fillers, v_ref 0.639 vs 0.615) is a negative control: BOTH theories predict identical behavior, so any filler-count sensitivity (display complexity, total-1s heuristics) is exposed and would invalidate the context interpretation; note any difference-based heuristic is immune to tied cues by construction. U1 (bucket 0.7, j=2 fillers) vs T2 (bucket 0.8, j=4) matches filler counts more closely than either matches the ladder, so a monotone filler-count heuristic cannot mimic the bucket-discrete eta redraw (which predicts subject-specific, sign-random bucket differences, not monotone drift). AL (3x0.90 vs 3x0.55, r=1) and AH (2x0.90 vs 5x0.55 + one tied 0.50, r=2.5) are parameter-free agreement cells (both theories confidently prefer the low-validity stack at r=1 and the high-validity stack at r=2.5 for every admissible parameter) that calibrate the softmax/lapse asymptotes and expose TTB and fewer-1s heuristics (both pick the 0.90-stack on AL, where both target theories pick the 0.55-stack). TIE (identical mixed vectors) pins side/response bias at exactly 0.500 for both theories. ANCHOR (all-1 vs all-0) calibrates the lapse and exposes tallying/pro-validity integration (both target theories predict P(all-1) ~ epsilon/2; tallying predicts ~1). All critical pairs are mirrored to absorb side bias; mirrors leave v_ref (and hence CNNAI's bucket key) unchanged. Attention robustness: SCWI's per-cue attention cancels exactly on tied cues and is constant across twinned contexts, so the context kill is immune to it; the ladder uses fixed cue subsets so attention acts as a fixed per-subject recalculation, and class-mean attention (sd ~ 0.03) moves flip ratios by only ~0.05. Every trial requires comparing 8-23 active cues, satisfying the min-RT viewing requirement. 28 experts (6 at 0.90, 11 at 0.55, 5 at 0.50, 6 at 1.00); 32 unique pairs x K = 96//32 = 3 repetitions = 96 trials (54 ladder, 24 bucket-0.8 transfer, 12 bucket-0.7 transfer, 6 filler control, 12 agreement anchors, 3 tie, 3 anchor trials per subject).

DECISION RULES. (i) SCWI is supported, and CNNAI falsified, if per-subject choice rates on twinned conflicts are statistically indistinguishable across buckets (across-bucket variance = split-half variance; cross-bucket correlation at the noise ceiling; ladder-to-twin transfer accuracy >= ~0.75), every structured subject's flip lies inside (1.66, 1.90) with sd < 0.08, the population psychometric is steep (rise P(2.0)-P(1.667) >= 0.45 with P(H|1.667) <= 0.32), and bucket-0.8 twins show zero systematic softening. (ii) CNNAI is supported, and SCWI falsified, if across-bucket inconsistency substantially and reliably exceeds within-bucket inconsistency (variance ratio >> 1; cross-bucket correlation ~ 0.2; transfer accuracy ~ chance), flips spread with sd ~ 0.18 including subjects preferring the L-stack at r >= 2.0 and the H-stack at r = 1.667, and the population psychometric is flatter (P(H|1.667) >= 0.38, rise <= 0.35) with slight bucket-0.8 softening. A single subject whose preference reliably flips sign between twinned contexts falsifies SCWI outright. (iii) BOTH are falsified if the all-1 option is chosen well above lapse on ANCHOR (tallying/pro-validity), the 0.90-stack is chosen on AL (TTB/fewer-1s), F1 differs from L3 (filler-count heuristic), context effects are monotone in filler count rather than bucket-discrete, or everything sits flat at 0.5 (pure guessing).

[11] ADVERSARIAL LOGIC. CNNAI (advocated) and SCWI (competing) are near-degenerate anti-validity additive integrators, and CNNAI's own documentation concedes this: its engaged eta is centered at 0.12 precisely so that 'the engaged bulk behaves SCWI-like in effective curvature' while sitting below the knife region eta~0.31-0.34. At the validity pair (0.90, 0.60), SCWI's weight ratio w(0.90)/w(0.60) = (0.9+0.81*tau)/(0.6+0.36*tau) is confined by tau in [0.15, 0.42] to [1.562, 1.651] (implied power exponent 1.10-1.24), while CNNAI's ratio is 1.5^(1+eta) with eta ~ N(0.12, 0.21) clipped [-0.25, 0.70]; the engaged bulk overlaps and no single-cell population sign kill exists. Experiment [10] already took the context-redraw wedge (bucket twinning and transfer). This design attacks the two flanks CNNAI's knife-avoidance recalibration left un-fortified and that no prior experiment instrumented. FLANK 1 -- THE DETERMINISTIC BLOC (primary, distributional kill). SCWI hard-codes a trimodal response mixture: 16% true guessers (beta 0.02-0.10), 50% mid (beta 1.8-2.4, eps 0.05-0.11), and 34% DETERMINISTIC (beta 3.8-6.0, eps 0.02-0.04). CNNAI hard-codes a homogeneous engaged beta: lognormal median 2.14, log-sd 0.20, hard-clipped to [0.9, 4.5] -- P(beta>3.2) ~ 2%, P(beta>4) ~ 0.1% -- with eps 0.03-0.09. Instrument: the M-cell, where one option carries a single extra 0.60-cue endorsement and the other 11 cues are tied (both options show 11-12 ones). The choice margin is exactly |w(0.60)|: 0.654-0.751 for SCWI (only 15% variation over the whole tau box) and 0.54-0.62 for CNNAI (eta-robust). Both theories prefer the defect-free option, so the cell measures CONFIDENCE, not direction. Per-trial p: SCWI deterministic regime 0.89-0.99; SCWI mid 0.71-0.86; CNNAI engaged 0.68-0.86 (essentially nobody above 0.87, since even beta=4.5 gives only ~0.90 and that beta has ~0.01% prior mass). Pre-registered statistic: the fraction of subjects choosing the defect-free option on >= 22 of 24 M-trials, gated on S-cell anti-validity consistency (>= 8/12; see screens below) to exclude count-heuristics. SCWI predicts 0.22-0.35 (0.34 x P(>=22/24 | p~0.95) plus a small mid leak; even the worst corner of its box -- tau=0.15, beta=3.8, eps=0.04 -- yields >= 0.20). CNNAI predicts 0.005-0.03 (its beta ceiling makes a sustained 0.917 rate essentially impossible; guesser binomial flukes add ~0.0001). The boxes are disjoint by ~7x. Crucially, the population MEANS are nearly identical (SCWI ~0.79, CNNAI ~0.73): a mean-level test -- the kind CNNAI's calibration was built to survive -- is underpowered here by construction; the dissociation lives in the upper tail and in the trimodal-vs-bimodal shape of the per-subject M-rate distribution (SCWI: modes near 0.50 / 0.78 / 0.95; CNNAI: 0.50 / ~0.76 with only a thin continuous shoulder). FLANK 2 -- THE SUBLINEAR TAIL (secondary; level-and-step kill on a count-ratio ladder). SCWI's map is everywhere steeper than linear; CNNAI places 28% of its engaged mass at eta < 0, i.e., SUBLINEAR anti-validity weighting (w(0.90)/w(0.60) < 1.5: high-validity defects weighted less than proportionally). The rung at exactly the linear knife -- 4x0.90 vs 6x0.60 endorsements (ratio 3/2) -- pins the two: every engaged SCWI subject prefers the low-validity stack (margin 0.16-0.45 across the entire tau box, never flipping), while CNNAI's eta<0 subjects (~24% of the population) prefer the HIGH-validity stack (margins up to 0.36). Population P(H-stack): SCWI 0.30-0.40 vs CNNAI 0.42-0.47. The adjacent rung at 5/3 (3x0.90 vs 5x0.60) sits just above SCWI's whole flip window [1.562, 1.651]: ALL engaged SCWI subjects flip between the two adjacent rungs (step +0.26 at the box center, range +0.15..+0.34), whereas CNNAI rises only +0.10..+0.18 (25% of engaged still prefer the L-stack at 5/3, i.e., eta>0.26; 14% at 7/4). Rungs 7/4 (4 vs 7) and 2/1 (4 vs 8) complete the ladder: SCWI ~0.72 / ~0.81 vs CNNAI ~0.65 / ~0.80. SINGLE-CONTEXT CONSTRUCTION (conservative by design). Every critical trial endorses all 12 experts (each cue is 1 for at least one option), so CNNAI's v_ref = 0.70 exactly and its bucket key is '0.7' on EVERY trial: one eta draw per subject for the whole experiment. CNNAI therefore runs in its most SCWI-like fixed configuration -- no bucket redraws, no v_ref rescaling differences -- so any observed dissociation is conservative and cannot be attributed to CNNAI's context stochasticity. This deliberately does NOT repeat [10]'s twinning/transfer logic; the wedges here (beta-mixture shape; sublinear curvature tail) are untouched by [0]-[10]. MARGIN RUNGS AND SCREENS. S-cells (one exclusive 0.90-endorsement vs one exclusive 0.60-endorsement, 10 tied cues; margin w90-w60 = 0.37-0.49 SCWI, 0.22-0.54 CNNAI): both theories prefer the low-validity-endorsed option (population ~0.72 vs ~0.67) -- and the 1-count is exactly tied (11-11), so Tallying sits at exactly 0.500 while TTB and pro-validity integrators pick the high-validity option: a triple screen that simultaneously verifies the anti-validity direction and gates the bloc statistic against 'fewer-1s' heuristics (which are at chance on tally-tied S-cells). L-cells (single 0.90 defect, margin 1.02-1.24) and XL-cells (two 0.60 defects, margin 1.31-1.50) are confident agreement anchors calibrating the softmax ceiling (both theories 0.85-0.93). ANCHOR (all-1 vs all-0): both theories predict P(choose all-1) ~ eps/2 (population ~0.10 including guessers); tallying and pro-validity integration predict ~1 -- an exclusion screen. TIE (identical vectors): exactly 0.500 for both -- side-bias calibration. Every critical pair is mirrored (mirroring leaves v_ref and the bucket key unchanged). ATTENTION ROBUSTNESS. M-cells rotate the defect cue across four different 0.60-experts, and knife cells rotate cue subsets, so SCWI's fixed per-cue attention (U[0.9, 1.1]) enters margins multiplicatively and averages out (class-mean sd ~0.03 over rotations); a deterministic-regime SCWI subject's M-cell p cannot drop below ~0.88. CNNAI's per-trial attention diffusion (sd 0.05) shifts margins by ~0.03 against margins of 0.54-0.75 -- negligible. The bloc kill requires only a sustained high rate, not a sign flip, so attention cannot sabotage it from either side. DESIGN. 12 experts: 4 at 0.90, 8 at 0.60. 32 unique pairs x K = 96//32 = 3 = 96 trials per subject: 24 M (bloc), 12 S (direction + screens), 6 L + 6 XL (agreement anchors), 12 K(3/2) + 12 K(5/3) + 6 K(7/4) + 6 K(2/1) (knife ladder), 6 ANCHOR, 6 TIE. Every trial requires scanning 9-11 tied cues to locate the discriminating endorsements, satisfying the min-RT viewing requirement. APPROXIMATE POPULATION PREDICTIONS (integrated over each theory's stated parameter boxes): M-cell mean 0.79 (SCWI) vs 0.73 (CNNAI); M-bloc fraction (>=22/24, S-gated) 0.22-0.35 vs 0.005-0.03; M-rate distribution trimodal vs bimodal; S-cell 0.72 vs 0.67; L 0.88 vs 0.85; XL 0.93 vs 0.90; K(3/2) P(H-stack) 0.30-0.40 vs 0.42-0.47; K(5/3) ~0.59 both, step from K(3/2) +0.26 vs +0.15; K(7/4) 0.66-0.80 (point 0.72) vs ~0.65; K(2/1) ~0.81 vs ~0.80; ANCHOR P(all-1) ~0.10 both; TIE 0.500 both. DECISION RULES. (i) SCWI is supported and CNNAI falsified if the S-gated M-bloc fraction is >= 0.18 with a visible near-ceiling mode (~0.9+) in the per-subject M-rate distribution -- CNNAI cannot exceed ~0.03-0.06 for ANY admissible parameters (beta capped at 4.5 with 98% of engaged subjects below 3.2) -- together with K(3/2) <= 0.40 and a ladder step K(5/3)-K(3/2) >= 0.20. (ii) CNNAI is supported and SCWI falsified if the M-bloc fraction is <= 0.07 with a unimodal engaged M-rate distribution and M-mean <= 0.75, K(3/2) >= 0.42 (above SCWI's whole-box ceiling), and step <= 0.16. (iii) BOTH are falsified if the all-1 option is chosen well above 0.15 on ANCHOR (tallying/pro-validity), TIE deviates from 0.5 (side bias), the high-validity-endorsed option wins the S-cells (TTB/pro-validity), or the M-rate distribution fits neither shape (e.g., mass only at 0.5 and 1.0). TERTIARY (supportive only, weak): the between-subject correlation of K(3/2) H-rate with M-rate among non-guessers is NEGATIVE under SCWI (a single tau raises both M-confidence and the L-stack margin at the linear knife) and POSITIVE under CNNAI (a single low eta both enlarges w(0.60) and flips the linear knife toward the H-stack); both mechanisms are attenuated by beta/binomial noise, so this is corroborative, not decisive.

[12] ADVERSARIAL LOGIC. CAVI-K (advocated) and CNNAI (competing) are near-degenerate anti-validity additive integrators: a rating of 1 is a defect, tied cues cancel exactly, both use the SAME bucketed-mean-endorsed-validity context key with a per-bucket eta redraw, both have softmax + symmetric lapse + a true-guesser subpopulation, and their effective validity exponents overlap almost everywhere (CAVI-K's w(v)=v+tau*v^2 implies exponent ~1.1-1.3; CNNAI's 1+eta is centered at 1.12). Experiments [10] and [11] already took the context-redraw and beta-mixture wedges. The ONE hard structural difference no prior design has touched is CAVI-K's TWO-REGIME COMPRESSIVE LOAD MAP: defect loads are identity below a knee at 4.45 and power-compressed above it with kappa in [0.55,0.68] (engaged) and [0.78,0.85] (deterministic tail) -- ALWAYS strictly below 1. CNNAI is exactly linear in the ratings. Because compress() is strictly increasing it can never flip a preference; it only shrinks margins. So the dissociation is a CONFIDENCE-vs-LOAD profile, and it can be engineered to be parameter-free on both sides.

THE INSTRUMENT: THE TIED-FILLER LOAD LADDER AT SINGLE-VALIDITY ACTIVE SETS. On a trial where one option carries k extra 1s (defects) on cues of validity v and both options share f tied 1s on other cues of the SAME validity v: (i) every active cue has v_j = v_ref, so (v_j/v_ref)^eta = 1 for EVERY eta in BOTH theories -- the entire context/elasticity machinery (bucket redraws, stable components, pinned tail eta) is exactly inert; (ii) tied cues cancel exactly in CNNAI's score difference, and the bucket key is identical at every rung, so CNNAI's per-trial choice probability is provably IDENTICAL at every filler count (up to mean-zero per-trial attention noise on the discriminating cues only) -- a horizontal line with zero free parameters; (iii) in CAVI-K the tied cues add weight f*w to BOTH options' raw defect loads, and once the loads cross 4.45 the power compression shrinks the margin like s^(kappa-1): as f grows, engaged confidence falls from sigmoid(beta*k*w) toward 0.5, for EVERY admissible (tau, kappa, beta, epsilon, eta) and every non-guesser subject. The direction of choice never changes; only its confidence. CNNAI is pinned flat; CAVI-K is pinned decreasing.

VALIDITY ENGINEERING. 26 experts: 9 at 0.90 (family H), 17 at 0.60 (family L). Per-cue defect weight w_H = 0.9+0.81*tau in [1.18, 1.43] over the whole tau box (engaged [0.35,0.65], deterministic [0.55,0.65]); w_L = 0.6+0.36*tau in [0.726, 0.834]. PARAMETER-FREE FLAT ZONE: family H rungs f<=2 give max load 3*w_H = 4.28 < 4.45 and family L rungs f<=4 give max load 5*w_L = 4.17 < 4.45 for EVERY admissible tau (even a 2-SD adverse class-mean attention draw cannot cross the knee), so CAVI-K is exactly flat there too. PARAMETER-FREE COMPRESSED ZONE: H f=8 gives min load 8*1.18 = 9.47 and L f=16 gives min load 16*0.726 = 11.6, both far above 4.45 for the entire box. The knee is crossed between f=2 and f=3 in family H and between f=4 and f=6 in family L.

HEADLINE MINIMAL PAIR (count-matched, load-mismatched): H-f4 vs L-f4 have the IDENTICAL count structure -- option A shows 5 ones, option B shows 4, one extra defect for A, same display complexity -- but H-f4's load (5*w_H in [5.9, 7.15]) is above the knee while L-f4's (5*w_L in [3.63, 4.17]) is below it. CAVI-K predicts a confidence DROP in H-f4 but ZERO change in L-f4; CNNAI predicts zero change in both; any raw-count, total-1s, or search-difficulty heuristic treats the two cells identically and is thereby excluded. Symmetrically, going from the bottom rung to f=4 adds MORE ones in family L (1->5 per option) than in family H (3->5), so a difficulty account predicts a LARGER drop in L -- the exact opposite of CAVI-K's prediction.

PREDICTED POPULATION RATES, P(choose the defect-free option), midpoint parameters (engaged tau=.5, kappa=.615, beta=2.65, eps=.06; deterministic tau=.6, kappa=.815, beta=5.25, eps=.03; mixture 14/70/16; CNNAI engaged beta=2.14, eps=.06, mixture 13/87): CAVI-K family H: f=0: 0.87, f=2: 0.87 (flat -- sub-knee), f=4: 0.81, f=8: 0.78. Family L: f=0: 0.82, f=4: 0.82 (flat -- sub-knee), f=8: 0.73, f=16: 0.70. k=2 probe (two defects, family L): f=0: 0.90, f=16: 0.80. CNNAI: family H flat at 0.80, family L flat at 0.72, k=2 flat at 0.85 -- at EVERY rung, for every parameter. In LOG-ODDS the dissociation is large: CAVI-K's engaged per-subject logit falls by ~1.2 (family L: 2.65*0.78 -> 2.65*0.32) to ~2.0 (family H: 3.46 -> 1.50) across the ladder; CNNAI's falls by exactly 0.00. Per-subject drop distribution: CAVI-K predicts a negative slope for EVERY non-guesser (large for the 70% engaged, small for the 16% deterministic tail whose kappa=0.78-0.85 compresses weakly -- a bimodal drop-magnitude distribution plus a zero-slope guesser mass at 0.5); CNNAI predicts zero slope for every subject. KNEE LOCALIZATION: CAVI-K's drop onset sits at ~4 total ones on the defect option in family H and ~6-7 in family L -- the same WEIGHTED load (~4.45 in (v+tau*v^2) units) at different raw counts, a quantitative signature no count-based heuristic can reproduce. MARGIN SCALING: the L2 rungs (k=2 at f=0 and f=15 fillers) show the logit drop scales with the raw margin (compressed margin ~= kappa*(s/4.45)^(kappa-1)*k*w), replicating the drop at a second confidence level; CNNAI stays flat there too.

CONTROLS AND THIRD PARTIES. TIE trials (identical vectors, one all-1 and one mixed) pin side/response bias at exactly 0.500 for both theories. ANCHOR (all-1 vs all-0, mirrored) calibrates the lapse and screens tallying and pro-validity integration (both target theories predict P(choose all-1) ~ 0.09 including the guesser mass at 0.5; tallying/pro-validity predicts ~1). The sub-knee rungs are an internal artifact screen: any drop appearing at H f<=2 or L f<=4 falsifies the compression interpretation (and both theories), since those loads are below the knee for the entire parameter box -- this catches general dilution/complexity effects. Mirroring every critical pair absorbs side bias and leaves v_ref and the bucket key unchanged. All 26 expert rows are displayed on every trial, so display complexity is constant across rungs; the discriminating 1 is embedded among many fillers at high rungs, so every trial genuinely requires comparing the two rating vectors (min-RT requirement). Attention robustness: per-cue attention is mean-1 multiplicative; flipping a sub-knee rung across the knee requires a 4-7% class-mean distortion (2-4 SD), and the population-level slope predictions are immune because attention is symmetric. No feedback keeps the instructed validities anchored.

DESIGN. 26 experts (9 at 0.90, 17 at 0.60); 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials per subject (8 trials per rung per family, 32 ladder trials per family, 8 k=2 trials, 8 tie, 8 anchor).

DECISION RULES. (i) CAVI-K is supported, and CNNAI falsified, if P(defect-free option) shows a reliable negative slope over filler load within both families (population-level and/or a majority of non-guesser subjects), with the sub-knee rungs flat (H f<=2, L f<=4), the drop onset located between the sub-knee and above-knee rungs at load-matched positions across families (~4 ones in H, ~6-7 in L), the H-f4-vs-L-f4 count-matched contrast showing a drop only in H, and the k=2 rungs replicating the drop at higher base confidence. CNNAI cannot produce ANY of these slopes for any admissible parameters: tied cues cancel exactly and its context machinery is inert by construction. (ii) CNNAI is supported, and CAVI-K falsified, if confidence is flat at every rung in both families (slope indistinguishable from 0 beyond binomial noise): CAVI-K's compression with kappa < 1 in every regime is hard-wired, and the top-rung loads (9.5-14.2) exceed the knee 4.45 across its entire parameter box, so it cannot stay flat. (iii) BOTH are falsified if the all-1 option is chosen well above ~0.15 on ANCHOR (tallying/pro-validity), TIE deviates from 0.5 (side bias), the drop is monotone in RAW filler count or total ones rather than weighted load (count/difficulty heuristic -- exposed by the H-f4/L-f4 count-matched contrast and the cross-family onset locations), or the drop appears already in the sub-knee rungs (dilution artifact).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "ADVERSARIAL LOGIC. CNNAI (advocated) and CAVI-K (competing) are near-degenerate in ordinary mixed-validity trials: both are anti-validity additive integrators (a 1 is a defect, tied cues cancel), both use the identical bucketed-mean-endorsed-validity context key with a per-bucket eta redraw, both have softmax + symmetric lapse + a ~13-14% guesser subpopulation, and their effective validity exponents overlap almost everywhere (CNNAI's 1+eta ~ N(1.12, 0.21) vs CAVI-K's tau-implied ~1.26-1.42 plus zero-centered eta noise). Experiments [10]-[12] already took the context-redraw, beta-bloc, and compression-knee wedges. The ONE hard structural difference no prior design has isolated is WHERE each theory's supralinearity lives. CNNAI's curvature is ENTIRELY context-relative: w_j = -v_j*(v_j/v_ref)^eta, so when every cue endorsed on a trial shares a single validity v, v_ref = v and (v_j/v_ref)^eta = 1 for EVERY eta \u2014 CNNAI's per-defect weight collapses to c(v) = v EXACTLY, with zero free parameters: the entire eta machinery (bucket redraws, stable component, knife-avoidance centering at 0.12) is provably inert. CAVI-K's supralinearity is context-INVARIANT: d_j = v_j + tau*v_j^2, and the tau*v^2 term never vanishes in any context \u2014 tau in [0.35,0.65] (engaged) and [0.55,0.65] (deterministic tail) is always engaged. THE INSTRUMENT: a validity ladder of pure count-discrimination trials. On each trial, option A endorses 2 cues and option B endorses 1 cue, ALL from a single validity family v in {1.00, 0.90, 0.75, 0.60, 0.50} (all other experts rated 0 by both options). Both theories prefer B (fewer defects); the measurement is CONFIDENCE as a function of v. Per-trial core logit = beta*c(v). Within-subject logit ratios cancel beta and lapse, isolating c(v). KILL RATIOS (parameter-free): c(1.0)/c(0.5) = 2.000 EXACTLY for every engaged CNNAI subject (any beta, any eta, any bucket draw, any attention realization \u2014 attention is mean-1 multiplicative and averages over 16 trials/family), versus (1+tau)/(0.5+0.25*tau) in [2.298, 2.490] for EVERY non-guesser CAVI-K subject (monotone increasing in tau; deterministic tail >= 2.43). The band (2.05, 2.25) is EMPTY for both theories \u2014 a per-subject and population classifier with no free parameters. Mid rungs replicate the gap: c(0.9)/c(0.5) = 1.80 (CNNAI) vs [2.02, 2.15] (CAVI-K), empty band (1.85, 2.00); c(0.75)/c(0.5) = 1.50 vs [1.61, 1.68], empty band (1.53, 1.60). SUB-KNEE GUARANTEE: maximum raw defect load on any trial is 2*(1+0.65) = 3.30 < 4.45, safe by >= 24% even under a +4-SD adverse attention draw, so CAVI-K's two-regime compression is ALSO exactly inert (identity regime) \u2014 the dissociation isolates the weight map alone, with compression, eta redraws, bucket keys, and v_ref rescaling all neutralized in BOTH theories. POPULATION PREDICTIONS (fully specified mixtures, integrating each theory's stated parameter distributions; P(choose the fewer-defect option) on the (2,1) ladder): CNNAI ~ [0.69, 0.73, 0.77, 0.80, 0.82] for v = [0.50, 0.60, 0.75, 0.90, 1.00]; CAVI-K ~ [0.79, 0.83, 0.86, 0.88, 0.89]. The curves are separated by 0.07-0.10 at EVERY rung with non-overlapping population ranges (CNNAI's ceiling at v=0.50 is ~0.74 even if all engaged subjects sat at their 84th-percentile beta; CAVI-K's floor is ~0.76): CAVI-K is pinned more confident everywhere because its c(0.5) >= 0.5875 > 0.50 and its beta median 2.65 > 2.14 are hard-coded. SECONDARY STATISTICS from the same trials: (S1) DETERMINISTIC-TAIL BLOC: on the v=0.60 (2,1) rung (16 trials/subject), CAVI-K's 16% deterministic tail (beta 4.0-6.5, eps 0.02-0.04) sits at per-trial p ~ 0.96-0.99, so ~13-17% of subjects choose B on >= 15/16 trials; CNNAI's homogeneous beta (98% below 3.2, clip 4.5) caps engaged per-trial p at ~0.91, predicting a bloc fraction of ~0.01. (S2) LAPSE/CEILING CELL: the (2,0) pair at v=1.00 (A shows two 1.00-cue 1s, B shows none) drives both theories near ceiling (CNNAI pop ~0.89, CAVI-K ~0.91), pinning each subject's lapse for the ratio correction, and gives a Delta=2 replication of the v-ordering. (S3) RATIO DISPERSION: CNNAI's engaged subjects ALL have ratio exactly 2.00 (between-subject sd ~ 0 from the map; only binomial/attention noise), whereas CAVI-K's tau spread induces sd ~ 0.06 plus a visible deterministic-tail mode near 2.45. CONTROLS AND THIRD-PARTY SCREENS: the ladder's RISING profile itself screens validity-blind strategies \u2014 an unweighted counter (c(v) constant) predicts a FLAT ladder (ratio 1.0); a pro-validity integrator prefers the option with MORE high-validity 1s (P < 0.5 at every rung) and is exposed by the ANCHOR (all-1 vs all-0), where both target theories predict P(choose all-1) ~ 0.09 (guessers at 0.5) while tallying/pro-validity predicts ~1; TIE (identical mixed vectors) pins side/response bias at exactly 0.500 for both; every critical pair is mirrored (swapping A/B leaves counts, v_ref, and the bucket key unchanged), absorbing side bias. ROBUSTNESS: per-trial attention diffusion (CNNAI sd 0.05, CAVI-K sd 0.035) is mean-1 multiplicative on weights and averages out over 16 trials/family (second-order softening ~1-2%, applied to both theories' quoted numbers); no feedback keeps weights anchored to the instructed validities; every trial requires scanning all 25 expert rows to locate and compare the sparse 1s, satisfying the min-RT viewing requirement. NOVELTY: [10] probed context redraws (mixed contexts), [11] probed the beta bloc and a mixed-validity knife ladder, [12] probed above-knee compression via tied-filler load. This design neutralizes ALL of those mechanisms and attacks the one remaining hard difference \u2014 the context-invariant quadratic term \u2014 in the exact regime (single-validity active sets, sub-knee loads) where CNNAI is parameter-free linear and CAVI-K is pinned quadratic. DESIGN: 25 experts (5 at each of 1.00, 0.90, 0.75, 0.60, 0.50); 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials (16 ladder trials per validity family, 8 ceiling/lapse trials, 4 tie, 4 anchor). DECISION RULES: (i) CNNAI is supported and CAVI-K falsified if the population (2,1) ladder matches [0.69, 0.73, 0.77, 0.80, 0.82] (+/- 0.045), the lapse-corrected hierarchical weight estimate satisfies c_hat(1.0)/c_hat(0.5) ~ 2.00 with CI excluding 2.25 and c_hat(0.9)/c_hat(0.5) ~ 1.80 below 2.0, the v=0.60 bloc fraction is <= 0.05, and between-subject ratio dispersion is near zero \u2014 CAVI-K cannot produce a ratio below 2.298 for ANY admissible (tau, kappa, beta, epsilon, eta) in any regime. (ii) CAVI-K is supported and CNNAI falsified if the ladder matches [0.79, 0.83, 0.86, 0.88, 0.89], the ratio is >= 2.28 with c_hat(0.9)/c_hat(0.5) >= 2.0, and ~12-18% of subjects show the near-ceiling bloc on the v=0.60 rung \u2014 CNNAI's exact-linear map (c(v) = v with zero free parameters in these contexts) cannot produce ratio 2.3, and its homogeneous beta cannot produce the bloc. (iii) BOTH are falsified if the ladder is FLAT (validity-blind counting), P(fewer-defect option) < 0.5 (pro-validity), the all-1 option is chosen well above 0.15 on ANCHOR (tallying/pro-validity), TIE deviates from 0.5 (side bias), or the recovered weight profile fits neither a line through the origin nor the quadratic family.",
  "validities": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.9,
    0.9,
    0.9,
    0.9,
    0.9,
    0.75,
    0.75,
    0.75,
    0.75,
    0.75,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 32274,
  "output_tokens": 55188,
  "reasoning_tokens": 50065,
  "provider": "Sail Research"
}
```
