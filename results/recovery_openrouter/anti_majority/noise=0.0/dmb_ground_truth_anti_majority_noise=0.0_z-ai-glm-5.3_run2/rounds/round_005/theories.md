# Round 5 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3_1` — KILLED ✗

**Description:** Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

**Rationale:** Minimal-diff edit on the accepted iter-6 base (loss 0.0707); the SCWI mechanism, softmax, lapse form, weight form, beta~0 guesser implementation, (0.90, 1.10) attention jitter, and tau box [0.15, 0.42] are all untouched. I apply the iter-6 critic's diagnosis that the remaining gaps (Exp 5 at 0.609 vs 0.684, Exp 4 at 0.110 vs 0.163) are pure mixture-calibration problems, with three targeted edits. (1) PRIMARY LEVER -- FRACTIONS 13/57/30 -> 16/50/34. The mid regime at 57% is the underperformer: the critic's own analysis of the iter-6 forecast miss (Exp 5 predicted 0.65-0.66, observed 0.609) localizes the error to the mid regime sitting at ~0.52-0.55 on the Exp-5 diagnostic cells rather than the forecast ~0.55-0.65, and at 57% of the population that dominates the mean. Redistributing mid mass to BOTH tails attacks the two largest gaps simultaneously: each 1pp moved to guessers adds ~+0.004 to Exp 4 (guessers contribute 0.5 vs mid's ~0.10 on the Probe-B conflict cells), and each 1pp moved to deterministic adds ~+0.003 to Exp 5 (det ~0.9 vs mid's ~0.53). Expected by the critic's regime arithmetic: Exp 4 0.110 -> ~0.13-0.14, Exp 5 0.609 -> ~0.64-0.66, Exp 6 0.635 -> ~0.64-0.66. I take the deterministic fraction to 34% (within the critic's 16/50/34 recipe) while noting its own guard: if simulation showed Exp 6 above 0.66 the det fraction should be capped at 32% with the extra mass as guessers -- the 34% choice keeps Exp 6's expected band [0.61, 0.66] because the guesser increase (13% -> 16%) partially offsets the det increase (30% -> 34%) on the Exp-6 cells (guessers ~0.5, det ~0.9, so the paired move is nearly mean-neutral there while both tails gain on their target metrics). (2) SMALL MID-BETA STEP: 1.6-2.2 -> 1.8-2.4, deliberately NOT a swing back to iter-5's failed 1.8-2.8. Iter-5's mid-beta raise failed in a mixture WITHOUT real guessers; with 16% of subjects pinned at exactly 0.5, the Exp-4 floor is higher, and a modest raise targets the Exp-5 cells where mid subjects sit too close to chance with only a second-order Exp-4 cost. This also nudges Exp 2 (the agree/disagree dissociation, 0.183 vs 0.274) slightly upward via more deterministic agree-trial anti-following, though I continue to accept Exp 2 as near this family's structural ceiling per four consecutive critic iterations and do not trade the recoverable Exp-4/5 gains for it. (3) VARIANCE VIA WITHIN-REGIME SPREAD: deterministic beta widened 4.2-5.6 -> 3.8-6.0, mid epsilon 0.06-0.10 -> 0.05-0.11, guesser epsilon 0.15-0.25 -> 0.12-0.28. The regime MEANS are now correctly separated (validated by the iter-6 variance gains: Exp 1 var 0.0086 vs target 0.0108), so the remaining 3-4x variance deficit on Exps 2/5/6 (0.017/0.022/0.020 vs 0.093/0.064/0.070) most plausibly comes from the within-regime boxes being too tight; widening them fattens the per-subject hit-rate distribution on near-threshold trials without materially moving the population means. Expected aggregate movement: Exp 4 +0.02-0.03, Exp 5 +0.03-0.05, Exp 6 roughly held or +0.01, Exp 1 slope roughly held (guesser increase flattens, det increase steepens -- offsetting), Exp 3 held in [0.12, 0.16] (its iter-6 value 0.142 is nearly perfect), and between-subject variance up on Exps 5/6 toward the critic's 1.5x target. Net, the two largest mean gaps shrink and the variance deficit improves, so the aggregate loss should land strictly below the 0.0707 floor. Consistent with the critic's stop condition, if this calibration iteration fails to beat 0.0707, the loop should ship iter 6 rather than oscillate further.

**Parameters:**
  - `beta_seed`: `[0, 1]`
  - `tau`: `[0.15, 0.42]`
  - `attention`: `[(0.90, 1.10)] * n_features`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI) -- corrected three-regime
    # population, iteration 7 (minimal-diff on the accepted iter-6 base).
    # Mechanism is UNCHANGED: subjective validity of cue j = a_j * v_j
    # (per-cue attention factor, fixed for the whole run), signed cue
    # weight w_j = -(v'_j + tau * v'_j^2) (a rating of 1 is a defect; the
    # anti-weight grows SUPRALINEARLY in the subjective instructed
    # validity), softmax over beta * values plus a symmetric lapse
    # epsilon. History is ignored (no trial-by-trial feedback).
    #
    # Population edits on top of the accepted iter-6 base (mechanism
    # untouched), per the iter-6 critic diagnosis that the remaining
    # gaps are pure mixture-calibration problems:
    #   * FRACTIONS 13/57/30 -> 16/50/34: the mid regime (57%) was the
    #     underperformer on Exp 5 and contributes almost nothing on the
    #     Exp-4 Probe-B conflict cells, so its mass is redistributed to
    #     BOTH tails -- the guesser tail lifts Exp 4 (guessers contribute
    #     0.5 on conflict cells vs mid's ~0.10) and the deterministic
    #     tail lifts Exp 5/6 (det ~0.9 vs mid's ~0.53).
    #   * MID BETA raised a small step, 1.6-2.2 -> 1.8-2.4 (NOT a swing
    #     back to iter-5's 1.8-2.8, which failed without real guessers;
    #     with 16% pinned at 0.5 the Exp-4 floor is higher).
    #   * WITHIN-REGIME SPREAD WIDENED for between-subject variance:
    #     deterministic beta 4.2-5.6 -> 3.8-6.0, mid epsilon 0.06-0.10
    #     -> 0.05-0.11, guesser epsilon 0.15-0.25 -> 0.12-0.28. These
    #     widen the per-subject hit-rate distribution without materially
    #     moving the population means.
    #   * Attention jitter stays exactly at the validated (0.90, 1.10);
    #     tau stays at [0.15, 0.42]; the beta~0 guesser implementation
    #     (the validated decoupling lever) is untouched.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

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

    att = _to_vec(parameters.get("attention"))
    if att is None or att.size != n_features:
        # Defensive fallback: no attention distortion.
        att = np.ones(n_features)
    att = np.clip(att, 0.6, 1.4)

    # --- Population mixture over (beta, epsilon) from a single uniform
    # seed. Three regimes: TRUE guessers (beta ~ 0 -> uniform core on
    # every trial), a mid anti-validity regime, and a deterministic
    # anti-validity regime. Iter-7 fractions: 16% / 50% / 34%.
    try:
        u = float(parameters["beta_seed"])
    except (TypeError, ValueError):
        u = 0.5
    u = min(max(u, 0.0), 1.0)
    if u < 0.16:
        # True guessing subpopulation (~16%): beta ~ 0 makes the
        # softmax core essentially uniform on every trial, so these
        # subjects sit at ~0.5 on every diagnostic cell.
        t = u / 0.16
        beta = 0.02 + 0.08 * t         # 0.02 .. 0.10
        epsilon = 0.12 + 0.16 * t      # 0.12 .. 0.28
    elif u < 0.66:
        # Mid subpopulation (~50%): moderate beta (small step up from
        # the iter-6 1.6-2.2 to target the Exp-5 cells where mid
        # subjects sit too close to chance), low lapse.
        t = (u - 0.16) / 0.50
        beta = 1.8 + 0.6 * t           # 1.8 .. 2.4
        epsilon = 0.11 - 0.06 * t      # 0.11 .. 0.05
    else:
        # Deterministic subpopulation (~34%): high beta with widened
        # within-regime spread, minimal lapse. Fraction (not beta
        # ceiling) guards Exp 6.
        t = (u - 0.66) / 0.34
        beta = 3.8 + 2.2 * t           # 3.8 .. 6.0
        epsilon = 0.04 - 0.02 * t      # 0.04 .. 0.02

    tau = float(parameters["tau"])

    # Subjective validities (instructed validity x per-cue attention),
    # then anti-validity weights: the anti-weight of expert j is
    # v'_j + tau * v'_j^2, supralinear in the subjective validity, so
    # the endorsements of the most trusted experts are avoided most
    # strongly while low-validity experts are nearly ignored.
    v_eff = val * att
    w = -(v_eff + tau * np.square(v_eff))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Context-Normalized Noisy Anti-Validity Integration (CNNAI), knife-avoidance recalibration. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity CONTEXT BUCKETS (the mean endorsed validity rounded to 0.1), so all trials sharing a bucket inherit one shared per-subject draw. The elasticity decomposes into a small stable per-subject component (eta_base ~ N(0.02, 0.09)) and a larger per-bucket redraw (~N(0.10, 0.19)), total hard-clipped to [-0.25, 0.70]; the variance ratio fixes latent cross-context consistency at r ~ 0.18 by construction. The engaged eta distribution (mean ~0.12, sd ~0.21) is deliberately centered well BELOW the kill-design knife region (eta ~ 0.31-0.34 where high-validity stacks flip to low-validity stacks), so the engaged bulk behaves SCWI-like in effective curvature while margins on the below-knife side stay large. A modest per-trial multiplicative attention diffusion (sd 0.05) supplies within-context stochasticity, choice is softmax(beta * values) with nearly homogeneous engaged beta (lognormal median ~2.14, log-sd 0.20) plus a narrow symmetric lapse (0.03-0.09), and a 13% true-guesser subpopulation is retained, uncorrelated with eta.

**Rationale:** This is a minimal-diff edit of the accepted iter-5 CNNAI base (loss 0.0913) addressing the iter-5 critic's four prescriptions, with two deliberate deviations that I justify below. The mechanism family, softmax/lapse structure, guesser mixture, and parameter set are unchanged; only the eta distribution, the context key, the attention diffusion, beta, and epsilon move. (1) EXP 5/6/7/8 MEANS (the worst residuals: Exp 5 at 0.503 vs real 0.684) ARE FIXED BY LOWERING THE ETA CENTER AND CUTTING NOISE, NOT BY MAX-REFERENCING. I did not adopt the critic's v_ref = max swap because the weight ratio w(v_h)/w(v_l) = (v_h/v_l)^(1+eta) is algebraically invariant to v_ref -- max-referencing changes only the absolute scale, multiplying every score margin by v_max^(-eta) <= 1, which would DEFLATE the near-knife margins (0.15-0.35 in weight units) that drive exactly the composites we need to lift. Instead I lowered the engaged eta center from 0.23 to 0.12 (eta_base ~ N(0.02,0.09) + bucket redraw ~ N(0.10,0.19)): the kill flip knives sit at eta ~ 0.30-0.34, and on the below-knife side margins GROW as eta decreases (Exp 5's 2x1.00-vs-5x0.50 cell: margin 0.14 at eta 0.23, 0.33 at eta 0.10), so centering at 0.12 both puts P(eta < 0.33) ~ 0.83 of engaged mass on the SCWI-following side and enlarges its signal. Working the actual cells numerically (per the critic's methodological demand -- no analytic F(knife) arguments), this config yields Exp 5 ~ 0.63-0.65 (from 0.503), Exp 6 ~ 0.59-0.61 (from 0.558), Exp 8 ~ 0.60-0.64 (from 0.562), Exp 7 ~ 0.62-0.64 (from 0.607). The attention diffusion cut (0.10 -> 0.05) removes the dominant damping channel: at 0.10, per-trial margin noise (sd ~ 0.25 on a 12-cue kill cell) exceeded the signal. (2) EXP 9's OVERSHOOTING r (0.352 vs 0.196) IS FIXED THROUGH THE COUPLING CHANNELS: engaged epsilon narrowed to 0.03-0.09 (the stable lapse was the last large cross-ladder trait), eta_base sd cut 0.13 -> 0.09 (latent r = 0.09^2/(0.09^2+0.19^2) ~ 0.18, on target given the observed +0.1 latent-to-empirical gap), and beta log-sd trimmed 0.25 -> 0.20. (3) THE CONTEXT KEY IS THE BUCKETED MEAN ENDORSED VALIDITY (rounded to 0.1), NOT THE CRITIC'S ROUNDED MAX: both Exp-9 ladders have max endorsed validity 1.0, so a max-based key would hand them the SAME eta draw and re-couple them -- the exact iter-1 r = 0.44 failure the critic elsewhere warns against. Bucketed mean keeps the ladders distinct (0.6 vs 0.7) while coarsening within-experiment fragmentation (Exp 2 collapses from ~6 keys to ~2-3; Exp 5 to 2), so per-subject composites are driven by shared draws -- the critic's own iter-2 diagnosis of the chronic variance underprediction (Exp 2 var 0.009 vs 0.093). (4) GUARDRAILS: beta median trimmed 2.34 -> 2.14 hedges Exps 1/3 against the noise cut, and the lowered eta center directly pulls Exp 10's rise down (twin-cell margins shrink toward the eta=0 knife, raising p_twin) -- projected ~0.31-0.33 from 0.341, toward the real 0.296 -- while Exp 4's slight undershoot (0.142 vs 0.163) closes because flatter weights track the count margin more. Every composite claim above was checked by direct margin arithmetic on the actual experimental cells including attention noise, lapse, and the 13% guessers, per the critic's methodological note.

**Parameters:**
  - `subject_seed`: `[0, 1]`
  - `regime_seed`: `[0, 1]`
  - `eta_base_seed`: `[0, 1]`
  - `beta_seed`: `[0, 1]`
  - `epsilon_seed`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Kneed Compressive Anti-Validity Integration with a Moderate Deterministic Tail (CAVI-K), beta-homogenization refinement. Subjects integrate all binary expert ratings into a per-option subjective DEFECT LOAD, comprehending a rating of 1 as a defect (sign inversion). Per-cue defect magnitude is supralinear in the instructed validity, d_j = v_j + tau*v_j^2 (tau in [0.35, 0.65]), modulated by a context elasticity (v_j/v_ref)^eta keyed on the bucketed mean endorsed validity, with a zero-centered stable component (sd 0.10) plus a per-context redraw (sd 0.20), fixing latent cross-context consistency at r ~ 0.20. The defect load is compressed by a two-regime map with the knee at 4.45 — above every near-knife small stack (AL cells, 2-top-cue adherence cells, which keep full raw margins) and below the large diagnostic stacks (which grow sub-proportionally aversive under kappa in [0.55, 0.68]). Population: ~14% true guessers, ~70% engaged with a HOMOGENIZED lognormal beta (median 2.65, log-sd 0.29, clip [0.8, 6.0]) — the tightened spread removes both the saturating hot-engaged bloc (which was driving the Exp-9 cross-ladder correlation to 0.388 via beta-saturation, not latent consistency) and the soft low-beta tail (which was diluting anti-validity adherence on Exps 2/5/6/8/11) — and ~16% deterministic tail (beta 4.0-6.5, near-linear kappa, pinned eta). Choice is softmax(beta*(D_B - D_A)) with symmetric lapse and per-trial attention diffusion (sd 0.035).

**Rationale:** This is a minimal single-lever diff of the accepted iter-4 base implementing the iter-4 critic's diagnosis, with one small guardrail step added. (1) PRIMARY LEVER — ENGAGED BETA LOG-SD 0.35 -> 0.29, median 2.5 -> 2.65, clip unchanged. The critic's key datum: with the eta split held constant at 0.10/0.20 across iters 1 and 4, the Exp-9 cross-ladder correlation swung 0.067 -> 0.388 purely from the beta-median raise — the correlation is driven by the saturating HOT END of the engaged beta distribution, not by latent context consistency. Trimming the log-sd removes that saturating bloc (pulling Exp 9 back toward its [0.15, 0.30] window) without touching the eta geometry that iter 2 showed is dangerously saturating (0.15/0.15 exploded Exp 9 to 0.595). Simultaneously, removing the soft low-beta tail raises mean anti-validity adherence on the persistently undershooting Exps 2/5/6/8/11 — the shared deficit of all four prior iterations that no compression-geometry edit ever fixed. The small median nudge 2.5 -> 2.65 (not the 3.0+ the AL-sigmoid arithmetic rules out: beta 2.5 -> engaged AL anti-rate 0.66, beta 3.2 -> 0.71) targets Exp 11's remaining shortfall (2.24 vs the [2.4, 3.2] window). (2) GUARDRAIL — ENGAGED KAPPA UPPER BOUND 0.70 -> 0.68, a single 0.02 step. Removing the low-beta tail raises adherence broadly, which will push Exp 10's rise (0.350 at iter 4, window [0.22, 0.36]) toward its ceiling; marginally stronger compression on the large diagnostic stacks (which are the only ones above the 4.45 knee) holds it inside the window. This is deliberately NOT the iter-3 confounded kappa trim that fought a simultaneous knee raise — the knee stays at 4.45 and only the upper bound moves. (3) HELD VERBATIM: knee 4.45 (calibrated by computed stack arithmetic: above the max AL stack 4.28 for every tau), tau [0.35, 0.65], the 0.10/0.20 eta split, the pinned tail eta, attention diffusion 0.035 (the proven Exp-12 fix), the 14/70/16 mixture, and all lapse ranges — each either in-window or explicitly validated at iter 4. Expected profile: Exp 9 -> ~0.15-0.28, Exp 11 -> ~2.3-2.6, Exps 5/6 -> ~0.55-0.60, Exp 8 -> ~0.53-0.57, Exp 2 -> ~0.18-0.22, Exp 10 held ~0.30-0.35, Exps 1/3/4/12 near their iter-4 values (Exp 1's slope may steepen slightly from removing soft subjects; the log-sd trim keeps the hot shoulder thin enough to stay above -0.23) — which should strictly beat the 0.1231 base.

**Parameters:**
  - `regime_seed`: `[0, 1]`
  - `tau_seed`: `[0, 1]`
  - `kappa_seed`: `[0, 1]`
  - `beta_seed`: `[0, 1]`
  - `epsilon_seed`: `[0, 1]`
  - `eta_base_seed`: `[0, 1]`
  - `subject_seed`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```
