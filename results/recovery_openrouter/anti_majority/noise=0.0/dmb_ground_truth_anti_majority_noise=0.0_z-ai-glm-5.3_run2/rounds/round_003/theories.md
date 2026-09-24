# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3_1` — SURVIVED ✓

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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI). Subjects integrate all cues additively into a per-option subjective value, but a rating of 1 is treated as evidence AGAINST the option (comprehension inversion, s = -1 for ~99% of subjects). The anti-weight of cue j is a free power law of the instructed validity, |w_j| = v_j^gamma, with gamma a free per-subject parameter drawn from a heterogeneous population distribution (gamma ~ 1.4-2.4, mildly supralinear on average). Choice is a softmax over beta * s * sum_j v_j^gamma * rating_j with a symmetric lapse epsilon. Because gamma, beta, and epsilon are drawn broadly per subject, the population reproduces both the mean-level signatures (negative Exp-1 slope, positive Exp-2 dissociation, positive Exp-3 slope, low Exp-4 conflict-following) and the large between-subject variance seen in the real data.

**Rationale:** Minimal-diff edit addressing the critic's diagnosis on the ACCEPTED base (loss 0.0872): the mechanism family is correct (all four metric signs right, three of four beats SCWI), but every metric was systematically TOO EXTREME in magnitude (Exp1 -0.2283 vs -0.1747; Exp2 0.3600 vs 0.2737; Exp3 0.1581 vs 0.1227; Exp4 0.1737 vs 0.1625), and the between-subject variances were 5-10x too small (Exp2 var 0.0107 vs 0.0930; Exp1 0.0007 vs 0.0108). Both problems point at the same knobs, so I change ONLY the parameter ranges — the predict/policy code is untouched. (1) SOFTEN the steepness: gamma shifts from [2.0, 2.4] (mean 2.2) to [1.4, 2.4] (mean 1.9), lowering the average supralinear exponent so the anti-validity value gap on TTB/tally-agreement trials narrows; this pulls the Exp-2 dissociation down from 0.36 toward the critic's target interval [0.24, 0.31] while moving Exp 1 toward -0.17, Exp 3 toward 0.12, and Exp 4 toward 0.16 — all in the correct direction, since all four overshoots were signed in the same 'too deterministic' direction. (2) WIDEN heterogeneity to reproduce the between-subject variance: the per-subject parameter draw IS the population distribution, so widening the ranges directly widens the per-subject metric spread. The new gamma range has sd ~0.29 (critic suggested 0.3-0.5) and the new beta range [1.5, 4.0] has sd ~0.72 (critic suggested ~0.8); the broad beta spread in particular makes some subjects near-deterministic and others noisy, which is what generates real var ~0.09 on the Exp-2 dissociation and var ~0.01 on the Exp-1 slope. Epsilon widens modestly to [0.06, 0.20] (mean 0.13 vs 0.085 before), adding a further uniform softening that acts on all trials roughly equally — a secondary lever per the critic, used gently so as not to undo the gamma correction. (3) The inversion rate stays pinned at 0.99 via sign_seed: the critic explicitly endorsed this data-driven deviation from the arbiter's example (a 10-15% pro-validity subgroup would push Exp-4 conflict-following well above the observed 0.1625), so it is retained unchanged. No mechanism changes, no experiment-specific machinery, no history use — one task-invariant power-law anti-validity integrator with softmax-plus-lapse, now calibrated.

**Parameters:**
  - `beta`: `[1.5, 4.0]`
  - `gamma`: `[1.4, 2.4]`
  - `epsilon`: `[0.06, 0.20]`
  - `sign_seed`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = s * v_j**gamma per unit
    # rating, where s = -1 for the (vast majority of) comprehension-inverted
    # subjects and +1 for the rare pro-validity subject; gamma is a free
    # power-law exponent applied to the instructed validity. Option values
    # are the cue-weighted sums; choice is a numerically stable softmax over
    # beta * values, mixed with a symmetric lapse epsilon. History is
    # ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PAHI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s_txt = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s_txt.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    sign_seed = float(parameters["sign_seed"])

    # Heterogeneous comprehension inversion: with population probability
    # 0.99 a subject inverts the rating direction (a 1 counts AGAINST the
    # option); the remaining subjects read ratings pro-validity.
    s = -1.0 if sign_seed < 0.99 else 1.0

    # Power-law anti-validity cue weights: the anti-weight of expert j is
    # v_j**gamma (on average mildly supralinear in the instructed validity).
    w = s * np.power(val, gamma)

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


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Heterogeneous-Elasticity Anti-Validity Integration (HEAI), bimodal-population refinement. Subjects integrate ALL expert ratings additively into a per-option subjective value, but a rating of 1 is comprehended as a DEFECT (evidence against the option) — a sign inversion of the rating direction. The anti-weight of expert j is a per-subject power law of the instructed validity, w_j = -v_j^gamma, where gamma ('validity elasticity') is drawn from a BIMODAL continuous population: a shallow majority (75% of engaged subjects, gamma ~ N(1.04, 0.22) truncated [0.60, 1.42]) that sits cleanly below the kill-design count-ratio thresholds (~1.31-1.34) and therefore follows the high-validity stack, and a steep minority (25% of engaged subjects, gamma ~ N(2.10, 0.26) truncated [1.65, 2.75]) that sits cleanly above them, whose near-lexicographic top-cue dominance drives the Exp-2 dissociation and keeps the Exp-8/Exp-4 composites from overshooting. Inverse temperature is mode-linked (shallow: lognormal median ~2.6; steep: lognormal median ~3.3; both heavy-tailed), a 16% guesser subpopulation (beta ~ 0) and a symmetric lapse epsilon are retained unchanged. The population mean elasticity (~1.30) stays below the arbiter's 1.4 ceiling. The key change from the falsified wide-unimodal shape is that almost no mass is wasted in the ambiguous 1.2-1.6 band where subjects sit at chance on the diagnostic cells: the shallow mode's thin upper shoulder (1.32-1.42) is retained deliberately, because subjects just above the kill thresholds miss Exp 8 while still hitting the higher-threshold Exp 5/7 ladder cells — the lever that lets Exp 8's mean and variance stay calibrated without capping Exp 5/6/7.

**Rationale:** This is a distribution-only edit of the ACCEPTED iter-2 HEAI base (mechanism, guesser fraction, epsilon ranges, seeds, and code structure unchanged), implementing the iter-2 critic's diagnosis — whose iter-1 advice produced an ACCEPTED loss improvement, so its model of the metrics deserves weight. The critic showed that the wide unimodal N(1.12, 0.40) dumps its added spread into the ambiguous gamma ~ 1.2-1.6 band, where subjects sit at ~chance on every diagnostic cell: that raised variances (as predicted) but capped the means on both sides simultaneously — Exp 5 fell 0.567 -> 0.531 against 0.684, Exp 2 stayed stuck at 0.130 against 0.274, and the 1.8 truncation cap made top-cue dominance on Exp 2's margin-1 cells structurally impossible. The edit: (1) GAMMA MADE BIMODAL with minimal middle mass — a shallow majority (75% of engaged, N(1.04, 0.22) trunc [0.60, 1.42]) cleanly below the kill thresholds (~1.31-1.34), and a steep minority (25%, N(2.10, 0.26) trunc [1.65, 2.75]) cleanly above ~1.8 and extending toward the gamma ~ 2-3 range the critic identified as necessary for Exp 2. Mixture arithmetic on the established simulation anchors (PAHI's all-steep population: Exp 2 = 0.267, Exp 8 = 0.208; iter-2's shallow population: Exp 2 = 0.130, Exp 8 = 0.622) predicts Exp 2 rises to ~0.20-0.22 while Exp 8 holds near 0.62 — the steep minority is precisely the mass that lowers Exp 8 from shallow's overshoot region while lifting Exp 2. (2) The shallow mode's thin upper shoulder (1.32-1.42, ~10% of the mode) is retained deliberately: subjects just above the kill thresholds miss Exp 8 but still hit the higher-threshold Exp 5/7 ladder cells (flip points 1.58-2.07), which is what reconciles Exp 8's lower target mean (0.604) with Exp 5/6/7's higher ones (0.63-0.68) — a reconciliation neither a clean bimodal nor the old unimodal could achieve. (3) BETA MODE-LINKED AND HOTTER (shallow median 2.3 -> ~2.6; steep median ~3.3, both heavy-tailed): hotter shallow beta sharpens engaged adherence on Exp 5/6/7 (the pi_3_1 vs pi_3 contrast shows hot beta is the lever that lifted Exp 5 from 0.553 to 0.651), and hotter steep beta maximizes each steep subject's Exp 2 dissociation contribution while pushing steep Exp 8/Exp 4 hits toward the floor. Expected net movement vs iter-2: Exp 2 0.130 -> ~0.21, Exp 5 0.531 -> ~0.59, Exp 6 0.564 -> ~0.63, Exp 4 0.199 -> ~0.18, with Exp 1/3/8 approximately held and Exp 7 giving back ~0.03-0.05 (the one deliberate trade, repurchased many times over on Exp 2/5/6). Bimodality also widens the between-subject variance on the kill composites (Exp 6/7 variances move toward the observed 0.04-0.07). The population mean elasticity (~1.30) remains below the arbiter's 1.4 ceiling, all population structure continues to be derived inside predict from uniform seeds, and the declared parameter boxes are untouched — the candidate stays squarely inside the prescribed HEAI family while replacing the falsified unimodal shape with the separated-subpopulation structure the data demand.

**Parameters:**
  - `regime_seed`: `[0, 1]`
  - `gamma_seed`: `[0, 1]`
  - `beta_seed`: `[0, 1]`
  - `epsilon_seed`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Heterogeneous-Elasticity Anti-Validity Integration (HEAI), iter 3.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Mechanism (UNCHANGED from the accepted iter-2 base): every cue
    # contributes signed evidence w_j = -v_j**gamma per unit rating
    # (a rating of 1 is a defect; the anti-weight is a per-subject
    # power law of the instructed validity). Choice is a numerically
    # stable softmax over beta * option values, mixed with a symmetric
    # lapse epsilon. History is ignored (no trial-by-trial feedback).
    #
    # Population edits on top of the accepted iter-2 base (mechanism
    # untouched), implementing the iter-2 critic's diagnosis that a
    # wide UNIMODAL gamma dumps its spread into the ambiguous
    # threshold band (gamma ~ 1.2-1.6) where subjects sit at ~chance
    # on every diagnostic cell, capping the means on BOTH sides:
    #   * GAMMA MADE BIMODAL: the engaged population is now a mixture
    #     of a SHALLOW majority (75%: N(1.04, 0.22) trunc [0.60, 1.42])
    #     and a STEEP minority (25%: N(2.10, 0.26) trunc [1.65, 2.75]),
    #     with minimal mass in between. The shallow mode sits cleanly
    #     below the kill-design count-ratio thresholds (~1.31-1.34 on
    #     Exp 8; ~1.32+ on the Exp 5 ladder), so it drives high-stack
    #     adherence on Exp 5/6/7/8; its thin upper shoulder (1.32-1.42,
    #     ~10% of the mode) deliberately remains to keep Exp 8's mean
    #     and between-subject variance calibrated. The steep mode sits
    #     cleanly above ~1.8 (critic's prescription; extends to 2.75 so
    #     the top cue can dominate three lower cues), which is what
    #     lifts the Exp 2 dissociation mean/variance and holds Exp 4
    #     down. Population mean elasticity ~ 1.30 < 1.4 (arbiter box).
    #   * BETA MODE-LINKED AND HOTTER: shallow subjects draw lognormal
    #     beta, median exp(0.95) ~ 2.6 (up from ~2.3), sigma 0.70,
    #     clip [0.6, 7.0]; steep subjects draw a hotter lognormal,
    #     median exp(1.20) ~ 3.3, sigma 0.60, clip [1.2, 8.0] (the
    #     steep 'confident lexicographic' subjects are also the
    #     sharpest). Hotter shallow beta sharpens engaged adherence on
    #     Exp 5/6/7; hotter steep beta maximizes the Exp 2 contribution
    #     per steep subject while pushing steep Exp 8/Exp 4 hits down.
    #   * Guesser fraction (16%), epsilon ranges, seeds, and everything
    #     else are untouched.
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HEAI expects a (2, n_features) stimulus; got shape {stim.shape}."
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

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    # Keep CDF seeds strictly inside (0, 1) so ndtri stays finite.
    gs = min(max(_seed("gamma_seed"), 1e-6), 1.0 - 1e-6)
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    es = _seed("epsilon_seed")

    if u < 0.16:
        # Guesser subpopulation (~16%): beta ~ 0 makes the softmax core
        # essentially uniform on every trial; lapse is elevated.
        t = u / 0.16
        beta = 0.02 + 0.08 * t          # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * es      # 0.12 .. 0.25
        gamma = 1.12                    # irrelevant at beta ~ 0
    else:
        if gs < 0.75:
            # SHALLOW majority (~75% of engaged): gamma ~ N(1.04, 0.22)
            # truncated to [0.60, 1.42]. Cleanly below the kill-design
            # thresholds (~1.31-1.34), with a thin upper shoulder that
            # keeps Exp 8's mean/variance calibrated.
            t = gs / 0.75
            tt = min(max(t, 1e-4), 1.0 - 1e-4)
            gamma = 1.04 + 0.22 * float(ndtri(tt))
            gamma = min(max(gamma, 0.60), 1.42)

            # Hot shallow beta: lognormal median ~2.6, heavy tail.
            z_b = float(ndtri(bs))
            beta = math.exp(0.95 + 0.70 * z_b)
            beta = min(max(beta, 0.6), 7.0)
        else:
            # STEEP minority (~25% of engaged): gamma ~ N(2.10, 0.26)
            # truncated to [1.65, 2.75]. Cleanly above ~1.8 so the top
            # cue can dominate three lower cues (the Exp 2 lever).
            t = (gs - 0.75) / 0.25
            tt = min(max(t, 1e-4), 1.0 - 1e-4)
            gamma = 2.10 + 0.26 * float(ndtri(tt))
            gamma = min(max(gamma, 1.65), 2.75)

            # Steep subjects are also sharper: lognormal median ~3.3.
            z_b = float(ndtri(bs))
            beta = math.exp(1.20 + 0.60 * z_b)
            beta = min(max(beta, 1.2), 8.0)

        # Symmetric lapse for engaged subjects (unchanged).
        epsilon = 0.03 + 0.14 * es      # 0.03 .. 0.17

    # Anti-validity power-law cue weights: a rating of 1 counts AGAINST
    # the option, with magnitude v_j**gamma.
    w = -np.power(val, gamma)

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
