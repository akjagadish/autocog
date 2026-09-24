# Round 7 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

**Description:** SACNAI — Salience-Amplified Context-Normalized Anti-Validity Integration with a Hot Tail. Subjects integrate all binary expert ratings additively into a per-option value, but comprehend a rating of 1 as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j/v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity buckets (mean endorsed validity rounded to 0.1), decomposing into a small stable subject component and a larger per-bucket redraw (latent cross-context r ~ 0.18). NEW — top-tier salience amplifier: on each trial, the highest-validity tier AMONG THE CURRENTLY ENDORSED experts (the 'most prestigious expert present') receives an additional multiplicative aversion sigma_eff = 1 + (sigma-1)*exp(-gap/0.10), where gap is the validity gap between the top endorsed tier and the next endorsed tier. The aversion is therefore strongest exactly on near-knife top-tier contrasts (e.g. 3x0.90 vs 3x0.85), where the top expert's prestige must break an almost tied count contest, and vanishes on coarse contrasts (gap >= 0.2, e.g. 0.90-vs-0.70 kill stacks), where it would act as a mere scale factor or flip established stack preferences. A modest per-trial multiplicative attention diffusion (sd 0.045) supplies within-context stochasticity. Choice is softmax(beta * value) with a symmetric lapse. The population is a three-part mixture decoupled from any load/compression machinery: ~13% true guessers (beta ~ 0, elevated lapse), ~72% engaged (beta lognormal, median 2.0, log-sd 0.25, clip [0.8, 4.5]; narrow lapse 0.03-0.09), and ~15% hot tail (beta ~ U[3.4, 5.2], lapse 0.02-0.05, eta pinned at the engaged median) — sized so the fitted-sensitivity ramp fraction lands near the observed ~0.09 while the winsorized ladder slope stays near 1.4. No compressive load map and no context-invariant quadratic term: the model is exactly linear in the ratings within a validity family, so tied-filler load ladders are exactly flat.

**Rationale:** SACNAI keeps the empirically validated CNNAI core (additive anti-validity integration, context-referenced elasticity w_j = -v_j*(v_j/v_ref)^eta with the bucketed context key and closed coupling channels, attention diffusion 0.045, softmax + symmetric lapse) and adds exactly the two components the arbiter diagnosed as missing. (1) TOP-TIER SALIENCE AMPLIFIER: the single largest residual error of CNNAI was the near-knife top-tier contrast (the 3x0.90-vs-3x0.85 AL cells, observed excess 2.88 vs predicted 0.32, i.e. observed anti-rate ~0.86 vs predicted ~0.58). A per-subject sigma in [1.25, 1.65] applied to the highest-validity tier among currently endorsed experts, with strength keyed to the endorsed-tier validity gap via exp(-gap/0.10), roughly doubles the margin exactly on those cells (engaged margin ~0.23 -> ~0.97, pooled anti-rate ~0.83, excess ~2.7) while leaving coarse-contrast designs untouched: on the 0.90-vs-0.70 kill stacks (gap 0.2) sigma_eff ~ 1.06, which moves the near-knife 5x0.90-vs-7x0.70 cell toward the knife and pulls the kill-composite down from CNNAI's overshooting 0.70 toward the observed 0.60; on single-validity ladders (one endorsed tier) the amplifier is off, so the tied-filler load ladder stays exactly flat (linear core, no knee) and ladder slopes are driven only by beta. Crucially, keying to the TOP ENDORSED tier (not the experiment's global top tier) is what makes the mechanism bite on the AL cells, where the contested experts are 0.90/0.85 even though the experiment also contains 1.0 cues. (2) HOT TAIL: CNNAI scored exactly 0 on the fitted-sensitivity ramp (observed 0.09) because its engaged beta never exceeds 4.5 in effective sensitivity. A 15% hot subpopulation (beta ~ U[3.4, 5.2], lapse 0.02-0.05, eta pinned at the engaged median so it amplifies the core sign pattern without shifting curvature) supplies the ramp mass; the engaged median is trimmed 2.14 -> 2.0 so the winsorized ladder slope stays near the observed ~1.4 under the 3.5 cap, and the tail simultaneously strengthens the TTB-dissociation contrast (0.165 -> toward 0.27). The falsifiable signatures versus CNNAI are the hot-tail ramp fraction and the near-knife top-tier adherence; versus CAVI-K they are the exact flatness of the tied-filler load ladder and the linear single-validity ladder, since SACNAI contains no compressive load map and no context-invariant quadratic term. Expected net movement: large gains on the AL-cell excess, the ramp fraction, and the kill composite, small costs on the conflict-set tally-following and twin-kill rise from the hotter tail — a clearly favorable trade under the aggregate loss.

**Parameters:**
  - `subject_seed`: `[0, 1]`
  - `regime_seed`: `[0, 1]`
  - `eta_base_seed`: `[0, 1]`
  - `beta_seed`: `[0, 1]`
  - `epsilon_seed`: `[0, 1]`
  - `sigma_seed`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # SACNAI: Salience-Amplified Context-Normalized Anti-Validity
    # Integration with a Hot Tail.
    #
    # CORE (retained from CNNAI, validated by Exps 1 & 2): a rating of 1
    # is a defect; per-option value = sum_j w_j * rating_j with
    # w_j = -v_j * (v_j / v_ref)^eta, v_ref = mean validity of the
    # endorsed cues; eta is drawn per bucketed-mean-endorsed-validity
    # context (small stable subject component sd 0.09 + larger
    # per-bucket redraw sd 0.19 -> latent cross-context r ~ 0.18);
    # per-trial multiplicative attention diffusion (sd 0.045);
    # softmax(beta * value) with symmetric lapse. NO compressive load
    # map (knee empirically dead) and NO context-invariant quadratic
    # term. The model is exactly linear in the ratings within a
    # validity family, so tied-filler load ladders are exactly flat.
    #
    # NEW (1) TOP-TIER SALIENCE AMPLIFIER: on each trial, the
    # highest-validity tier among the currently ENDORSED experts
    # receives an extra multiplicative aversion
    #     sigma_eff = 1 + (sigma - 1) * exp(-gap / 0.10),
    # where gap = (top endorsed validity) - (next endorsed validity).
    # Rank-based 'most-prestigious-expert aversion': strongest exactly
    # on near-knife top-tier contrasts (gap ~ 0.05, e.g. 3x0.90 vs
    # 3x0.85 cells), where it roughly doubles the value margin; a
    # near-common scale factor on moderate gaps (0.10); and essentially
    # off on coarse contrasts (gap >= 0.20, e.g. the 0.90-vs-0.70 and
    # 1.00-vs-0.50 kill stacks), where amplification would flip
    # established stack preferences. Tied (both-endorsed) top cues
    # cancel exactly in the value difference, so the amplifier only
    # bites on discriminating prestige.
    #
    # NEW (2) POPULATION (decoupled from any falsified machinery):
    # ~13% true guessers (beta ~ 0, elevated lapse), ~72% engaged
    # (beta lognormal median 2.0, log-sd 0.25, clip [0.8, 4.5]; narrow
    # lapse 0.03-0.09; eta centered at 0.12, safely below the kill
    # knives at eta ~ 0.31-0.34), and ~15% hot tail (beta ~ U[3.4, 5.2],
    # lapse 0.02-0.05, eta pinned at the engaged median) -- sized so
    # the fitted-sensitivity ramp fraction lands near the observed
    # ~0.09 while the winsorized single-validity ladder slope stays
    # near ~1.4 (engaged median trimmed to 2.0 to offset the tail's
    # contribution under the 3.5 winsorization cap).
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SACNAI expects a (2, n_features) stimulus; got shape {stim.shape}."
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
    gs = _seed("sigma_seed")

    # Per-subject salience-amplifier strength, U[1.25, 1.65]
    # (modest center ~1.45; the gap key does the fine calibration).
    sigma = 1.25 + 0.40 * gs

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. These subjects sit at ~0.5 on
        # every diagnostic cell and dilute population means/slopes.
        t = u / 0.13
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps             # 0.12 .. 0.25
        eta_base, redraw_mu, redraw_sd = 0.12, 0.10, 0.19  # irrelevant
    elif u < 0.28:
        # Hot tail (~15%): sharp, low-lapse subjects whose elasticity
        # is pinned at the engaged median, so they amplify the CORE
        # sign pattern without shifting the effective curvature.
        # Calibrated (rather than the nominal 4-7 box) so that the
        # fitted-sensitivity ramp fraction lands near the observed
        # ~0.09 while the winsorized ladder slope stays near ~1.4.
        beta = 3.4 + 1.8 * bs                  # 3.4 .. 5.2
        epsilon = 0.02 + 0.03 * ps             # 0.02 .. 0.05
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.10
    else:
        # Engaged bulk (~72%). Small stable elasticity component
        # (sd 0.09) paired with a larger per-context redraw (sd 0.19):
        # the variance ratio fixes latent cross-context consistency
        # at r ~ 0.18 by construction. Narrow stable lapse so it
        # cannot act as a hidden cross-context coupling channel.
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.693 + 0.25 * float(ndtri(bs)))  # median ~2.0
        beta = min(max(beta, 0.8), 4.5)
        epsilon = 0.03 + 0.06 * ps             # 0.03 .. 0.09
        redraw_mu, redraw_sd = 0.10, 0.19

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Engaged eta ~ N(0.12, 0.21): centered well below the kill knives
    # (~0.31-0.34) with large below-knife margins. Hot-tail eta is
    # pinned at the engaged median (0.12) with a tighter redraw.
    eta_ctx = eta_base + redraw_mu + redraw_sd * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # --- Top-tier salience amplifier (rank-based prestige aversion). ---
    # Keyed to the validity gap between the top endorsed tier and the
    # next endorsed tier: sigma_eff = 1 + (sigma-1)*exp(-gap/0.10).
    # gap ~ 0.05 (near-knife, e.g. 0.90 vs 0.85): strong amplification
    # (~1.27 at sigma center), roughly doubling the margin on those
    # cells. gap ~ 0.10: mild (~1.17). gap >= 0.20 (coarse kill
    # stacks like 0.90-vs-0.70 or 1.00-vs-0.50): essentially off
    # (~1.06 or less), so established stack preferences and
    # single-validity ladder slopes are preserved. Single-family
    # trials (only one endorsed tier) have no contrast tier: off.
    tiers = np.unique(val[active])
    if tiers.size >= 2:
        gap = float(tiers[-1] - tiers[-2])
        k_gap = math.exp(-gap / 0.10)
        sig_eff = 1.0 + (sigma - 1.0) * k_gap
        if sig_eff > 1.0 + 1e-9:
            top_mask = val >= (tiers[-1] - 1e-9)
            w = np.where(top_mask, w * sig_eff, w)

    # Per-trial multiplicative attention diffusion on cue weights
    # (sd 0.045): per-trial noise averages out within a subject, so
    # it supplies within-context stochasticity without damping
    # population means.
    att = np.exp(0.045 * np.random.randn(n_features))
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


### slot 2 — `pi_6` — KILLED ✗

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

### `pi_9` → slot 2 (via `new_theory`)

**Description:** PQSAI (recalibrated) — Prestige-Quantity Salience Anti-validity Integration with Load Dilution and a Bimodal Salience Mixture. Subjects integrate binary expert ratings additively into a per-option value but comprehend a rating of 1 as a DEFECT (sign inversion), with context-normalized anti-weights w_j = -v_j*(v_j/v_ref)^eta (v_ref = mean endorsed validity; eta drawn per bucketed-mean-endorsed-validity context, latent cross-context r ~ 0.18). The top endorsed validity tier receives an additional multiplicative aversion sigma_eff = 1 + (sigma_i - 1)*exp(-gap/0.10)*D, where D = 1/(1 + (n_disc/n0)^1.1) is a FLAT load-dilution factor (exponent softened 1.5 -> 1.1, n0 raised 6 -> ~10): prestige aversion is strong on small knife-edge contests (n_disc = 6, D ~ 0.62-0.68) but retains SUBSTANTIAL amplification even on large stacks (n_disc = 13-14, D ~ 0.40-0.45), so the population sits between the pure linear core and full undiluted salience exactly where the real data lie. The engaged population is bimodal in salience: ~50% prestige-sensitive (sigma_i ~ U[1.6, 2.1]) and ~50% linear counters (sigma_i ~ U[1.0, 1.15]), generating large between-subject variance on near-knife composites. Per-trial attention diffusion scales with load, sd = s0 + s1*sqrt(n_active), attenuating within-subject reliability on complex multi-tier trials. Softmax(beta*value) with symmetric lapse; ~13% true guessers; ~12% hot tail; exactly linear within a validity family (tied-filler load ladders flat).

**Rationale:** This is a minimal-diff, in-family recalibration of the accepted PQSAI base, applying exactly the critic's five-point diagnosis. The iter-1 candidate failed its two flagship targets in OPPOSITE directions on the same knob — Exp 11 at 1.52 vs real 2.88 (amplification too weak at n_disc = 6) and Exp 16 at 0.686 vs real 0.495 (amplification switched off at n_disc = 13-14, collapsing onto the pure core). Both failures indicate the dilution curve D(n_disc) decays too steeply, not that the mechanism is wrong. (1) FLATTEN THE CURVE: exponent 1.5 -> 1.1 and dilution_n0 6 -> ~10 (exposed [7, 13]). New operating points D(6) ~ 0.62-0.68 (was ~0.50) and D(13) ~ 0.40-0.45 (was ~0.15-0.24), inside the critic's target windows. At n_disc = 6, gap = 0.05, a sensitive subject (sigma ~ 1.95) now gets sigma_eff ~ 1.35-1.40 — margin on the 3v3 AL cells roughly 3*(0.90*1.38 - 0.85) ~ 1.1-1.3, i.e., a half-to-full recovery of SACNAI's undiluted amplification, which is what the real 2.88 (between the core's 0.32 and SACNAI's 3.60) requires; expected Exp 11 ~ 2.3-2.8 (was 1.52). At n_disc = 13-14 the same subjects retain sigma_eff ~ 1.25-1.30 — essentially SACNAI-strength amplification for the sensitive half and core behavior for the linear half — landing Exp 16's population mean between the core (0.686) and SACNAI (0.339), i.e., near the real 0.495 (was 0.686). (2) STRENGTHEN THE SENSITIVE MODE: sigma_sensitive box widened [1.4, 1.9] -> [1.6, 2.1], fraction toward 0.5 ([0.4, 0.6]); the target 2.88 sits roughly halfway between the delivered 1.52 and SACNAI's 3.60, and the half-strength recovery is delivered by the combination of (1) and (2) rather than by either alone. (3) GAP SCALE UNCHANGED AT 0.10: coarse-gap stacks (Exps 5/8/10, gap >= 0.15-0.5) keep exp(-gap/0.10) <= 0.14-0.22 so the amplifier stays essentially off there, preserving the validated core fits; the extra small-cell amplification comes from n0/exponent/sigma, not from widening the gap window. (4) EXP 2 VARIANCE: after flattening, the sensitive mode reaches sigma_eff ~ 1.25 on the gap-0.10, n_disc ~ 5-6 cells vs ~ 1.02 for linear counters — a much larger mode separation than iter-1's 1.12-vs-1.01 — which, together with the per-subject spread in sigma_sensitive and dilution_n0, should lift Exp 2's between-subject variance from 0.019 toward the real 0.093 while also raising the mean (0.188 -> ~0.24) toward 0.2737. (5) EXP 9 OVER-CONSISTENCY: noise_s1 raised (default 0.014 -> 0.022, range [0.012, 0.03]) so the 24-feature complex trials (n_active up to ~15, sd ~ 0.10-0.13) acquire enough within-subject stochasticity to attenuate the cross-ladder correlation from 0.347 toward the real 0.196; noise_s0 kept low ([0.02, 0.04]) so Exp 13's flat ladder (single-family, amplifier structurally off, symmetric noise) and Exp 14's slope (n_active = 3, sd ~ 0.065) are preserved. Expected net effect vs the running best: large gains on Exp 11 (error 1.36 -> ~0.2) and Exp 16 (0.19 -> ~0.05), moderate gains on Exps 15, 2, and 9, with the validated core experiments (1, 3, 4, 6, 7, 12, 13, 14) essentially untouched because their cells either have coarse gaps (amplifier off) or single-family ladders (amplifier structurally off); the only mild risk is a small overshoot on Exp 15 (toward ~0.60-0.65 vs real 0.576) and a slight softening on Exp 5, both far outweighed by the Exp 11/16 corrections.

**Parameters:**
  - `subject_seed`: `[0, 1]`
  - `regime_seed`: `[0, 1]`
  - `eta_base_seed`: `[0, 1]`
  - `beta_seed`: `[0, 1]`
  - `epsilon_seed`: `[0, 1]`
  - `mode_seed`: `[0, 1]`
  - `sigma_sensitive`: `[1.6, 2.1]`
  - `sigma_sensitive_fraction`: `[0.4, 0.6]`
  - `dilution_n0`: `[7, 13]`
  - `noise_s0`: `[0.02, 0.04]`
  - `noise_s1`: `[0.012, 0.03]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # PQSAI (recalibrated iter 2): Prestige-Quantity Salience
    # Anti-validity Integration with Load Dilution and a Bimodal
    # Salience Mixture.
    #
    # CHANGES VS THE ACCEPTED ITER-1 BASE (minimal diff, per the
    # critic's five-point recalibration — all in-family):
    #   (A) FLATTER DILUTION CURVE: exponent 1.5 -> 1.1 and
    #       dilution_n0 raised (default 6 -> 10, exposed range
    #       [7, 13]). New operating points: D(n_disc=6) ~ 0.62-0.68
    #       (was ~0.50) and D(n_disc=13) ~ 0.40-0.45 (was ~0.15-0.24).
    #       This simultaneously raises Exp 11 toward 2.88 (small-n_disc
    #       cells keep much more amplification) and pulls Exp 16 down
    #       from the pure-core 0.686 toward 0.495 (large-stack cells
    #       now RETAIN substantial amplification, landing the
    #       population between core and full SACNAI).
    #   (B) STRONGER SENSITIVE MODE: sigma_sensitive box widened to
    #       [1.6, 2.1] (default 1.95) and sensitive fraction toward
    #       0.5 (range [0.4, 0.6]). The 3v3 AL cells now receive
    #       sigma_eff ~ 1.35-1.40 on sensitive subjects — roughly a
    #       half-to-full recovery of SACNAI's undiluted amplification,
    #       which is what the real 2.88 (vs SACNAI's 3.60 and the
    #       core's 0.32) requires.
    #   (C) GAP SCALE KEPT AT 0.10: coarse-gap stacks (Exps 5/8/10)
    #       keep exp(-gap/0.10) <= 0.14-0.22, so the amplifier stays
    #       essentially off there and the validated core predictions
    #       are preserved.
    #   (D) EXP 2 VARIANCE: the widened sensitive box (up to 2.1) plus
    #       the flatter dilution enlarges the mode separation on
    #       gap-0.10, n_disc ~ 5-6 cells (sensitive sigma_eff ~ 1.25 vs
    #       linear ~ 1.02), injecting the missing between-subject
    #       spread without any new mechanism.
    #   (E) EXP 9 OVER-CONSISTENCY: noise_s1 raised (default 0.014 ->
    #       0.022, range [0.012, 0.03]) to attenuate within-subject
    #       reliability on the 24-feature complex trials and pull the
    #       cross-ladder correlation down toward 0.196; noise_s0 kept
    #       low (range [0.02, 0.04]) so Exp 13's flat ladder and
    #       Exp 14's slope are preserved.
    #
    # Everything else — the shared CNNAI/SACNAI core (defect sign,
    # context-normalized anti-validity weights, bucketed per-context
    # eta with latent r ~ 0.18, 13% guessers, 12% hot tail, exact
    # within-family linearity) — is retained verbatim from the
    # accepted base.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PQSAI expects a (2, n_features) stimulus; got shape {stim.shape}."
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

    def _pos(name, default, lo, hi):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = default
        if not np.isfinite(x):
            x = default
        return min(max(x, lo), hi)

    u = _seed("regime_seed")
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")
    ms = _seed("mode_seed")

    # --- PQSAI-specific knobs (recalibrated) ---
    n0 = _pos("dilution_n0", 10.0, 2.0, 20.0)          # dilution scale (raised)
    s0 = _pos("noise_s0", 0.03, 0.0, 0.2)              # noise intercept (kept low)
    s1 = _pos("noise_s1", 0.022, 0.0, 0.1)             # load-scaled slope (raised)
    sig_s = _pos("sigma_sensitive", 1.95, 1.0, 3.0)    # sensitive-mode sigma (raised)
    frac = _pos("sigma_sensitive_fraction", 0.52, 0.05, 0.95)  # toward 0.5

    # --- Bimodal salience mixture (applies to all non-guessers) ---
    if ms < frac:
        # Prestige-sensitive subject: sigma ~ U[1.6, 2.1] (drawn per
        # subject via the sigma_sensitive parameter).
        sigma = sig_s
    else:
        # Linear counter: sigma ~ U[1.0, 1.15].
        t = (ms - frac) / max(1e-9, 1.0 - frac)
        sigma = 1.0 + 0.15 * t

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. The amplifier is irrelevant
        # here, so pin sigma at 1.
        t = u / 0.13
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps             # 0.12 .. 0.25
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.0
        sigma = 1.0
    elif u < 0.25:
        # Hot tail (~12%): sharp, low-lapse subjects whose elasticity
        # is pinned at the engaged median, so they amplify the CORE
        # sign pattern without shifting the effective curvature.
        beta = 3.4 + 1.8 * bs                  # 3.4 .. 5.2
        epsilon = 0.02 + 0.03 * ps             # 0.02 .. 0.05
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.10
    else:
        # Engaged bulk (~75%). Small stable elasticity component
        # (sd 0.09) + larger per-context redraw (sd 0.19): the variance
        # ratio fixes latent cross-context consistency at r ~ 0.18.
        # Narrow stable lapse so it cannot act as a hidden
        # cross-context coupling channel.
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.693 + 0.25 * float(ndtri(bs)))  # median ~2.0
        beta = min(max(beta, 0.8), 4.5)
        epsilon = 0.03 + 0.06 * ps             # 0.03 .. 0.09
        redraw_mu, redraw_sd = 0.10, 0.19

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Engaged eta ~ N(0.12, 0.21): centered well below the kill
    # knives (~0.31-0.34). Hot-tail eta is pinned at the engaged
    # median (0.12) with a tighter redraw.
    eta_ctx = eta_base + redraw_mu + redraw_sd * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # --- Prestige aversion with QUANTITY DILUTION (flattened) ---
    # The top endorsed tier is amplified by
    #     sigma_eff = 1 + (sigma - 1) * exp(-gap/0.10) * D,
    #     D = 1 / (1 + (n_disc/n0)^1.1),
    # where n_disc counts the cues on which the two options differ.
    # FLATTENED curve (exponent 1.1, n0 ~ 10): small knife-edge
    # contests (3v3, n_disc = 6, D ~ 0.62-0.68) keep most of the
    # amplification -> strong prestige aversion (Exp 11 AL cells,
    # Exp 15 L-cells); large stacks (n_disc = 13-14, D ~ 0.40-0.45)
    # retain SUBSTANTIAL amplification, landing the population
    # between the linear core and full undiluted salience (Exp 16).
    # Single-family trials have no contrast tier: off, so
    # within-family load ladders stay exactly flat.
    tiers = np.unique(val[active])
    if tiers.size >= 2:
        gap = float(tiers[-1] - tiers[-2])
        n_disc = int(np.count_nonzero(ra != rb))
        D = 1.0 / (1.0 + (n_disc / n0) ** 1.1)
        sig_eff = 1.0 + (sigma - 1.0) * math.exp(-gap / 0.10) * D
        if sig_eff > 1.0 + 1e-9:
            top_mask = val >= (tiers[-1] - 1e-9)
            w = np.where(top_mask, w * sig_eff, w)

    # --- Load-scaled attention diffusion ---
    # sd = s0 + s1*sqrt(n_active): small-stack cells stay crisp while
    # complex multi-tier trials acquire within-subject stochasticity
    # (attenuating cross-ladder consistency on 24-feature designs).
    # Per-trial noise averages out within a subject, so it supplies
    # within-context stochasticity without damping population means.
    sd_att = s0 + s1 * math.sqrt(float(active.size))
    att = np.exp(sd_att * np.random.randn(n_features))
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
