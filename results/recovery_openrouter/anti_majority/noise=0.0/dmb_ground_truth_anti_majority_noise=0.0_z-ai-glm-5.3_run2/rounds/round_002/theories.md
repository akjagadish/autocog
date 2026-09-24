# Round 2 — Theories

**Verdict:** `new_model` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

**Rationale:** Diagnosis of prior failures: TTB (pi_1) predicts a FLAT Experiment-1 slope (~0 vs real -0.175) because only the top discriminating cue is used, and a strongly negative Experiment-2 metric (-0.70 vs real +0.274) because it always anti-follows the tally when it conflicts with the top cue. Tallying (pi_2) predicts a POSITIVE Experiment-1 slope (+0.20 vs real -0.175) and exactly zero on Experiment-2 (it follows the tally on every margin-1 trial regardless of TTB agreement). Both fail because they fix the evidence sign to 'rating 1 = good'. The arbiter's prescribed family — signed, validity-weighted integration of all cues — fixes this: with an inverted sign, the top-cue-favored option (which carries more 1s) is systematically AVOIDED, producing the negative Experiment-1 slope. Two refinements inside that family, both sanctioned by the arbiter's 'optionally free per-cue weights': (1) I do NOT use a symmetric 50/50 mixture over s = +1/-1, because pooling pro- and anti-signed subjects would cancel BOTH signatures (a pro-signed subpopulation contributes a positive Exp-1 slope and a non-positive Exp-2 metric); the data require a predominantly anti-signed population, so the inversion is structural and graded heterogeneity enters through beta/tau/epsilon instead. (2) A purely proportional inversion w_j = -v_j caps the Experiment-2 metric at ~+0.19 (deterministically it actually yields ~0, since the anti-validity subject anti-follows the tally on every margin-1 trial; only softmax noise generates positive values, and disagree trials have systematically smaller value gaps than agree trials). Adding the supralinear validity-squared term tau*v_j^2 — stronger inversion for more trusted experts — raises the disagreement-trial follow rate toward the observed +0.27 while keeping the Experiment-1 slope in range. Quantitative checks at the parameter-range centers (beta=1.4, tau=0.35, epsilon=0.10): Experiment-1 slope ~= -0.20 (real -0.1747) and Experiment-2 metric ~= +0.27 (real +0.2737), versus TTB (0.00, -0.70) and Tallying (+0.20, -0.00). The mechanism is experiment-invariant: it consumes only the instructed validities vector (any length) via the symbolic 'validities' parameter, uses all cues additively, and contains no experiment-specific shapes or stopping rule, so it transfers to any n_features/validity profile in this domain.

**Parameters:**
  - `beta`: `[1.1, 1.7]`
  - `tau`: `[0.25, 0.45]`
  - `epsilon`: `[0.05, 0.15]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = -(v_j + tau * v_j^2)
    # per unit rating (anti-validity weighting, supralinear in the
    # instructed validity). Option values are the cue-weighted sums;
    # choice is a numerically stable softmax over beta * values, mixed
    # with a symmetric lapse. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])

    # Signed (anti-validity) cue weights: a rating of 1 is evidence
    # AGAINST an option, and the inversion strength grows supralinearly
    # with the instructed validity of the expert.
    w = -(val + tau * np.square(val))

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


### slot 2 — `pi_4` — SURVIVED ✓

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

### `pi_3_1` → slot 1 (via `new_model`)

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
