# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Red-Flag Integration with Trait Polarity and Attentional Slip (RFI-delta). When choosing between two products described by binary expert ratings, a large majority of subjects read a rating of 1 as a 'red flag' — evidence AGAINST the option carrying it — so the sparser option is perceived as higher quality; a small minority reads polarity conventionally. Polarity is a stable PER-SUBJECT trait: subject i carries a fixed sign s_i ∈ {+1, -1} drawn once, with P(s_i = -1) = pi at the population level (pi near the arbiter's ceiling, ~0.94). On top of the trait, each subject occasionally suffers a small attentional slip: on any given trial, with probability delta (≈0.10), the subject momentarily reverts to the opposite (instructed-mapping) polarity for that trial alone. The slip is structured exactly like a lapse — a transient per-trial reversion against a stable dominant trait — not a reopening of per-trial polarity mixing, and it is kept small (delta ≤ 0.12). Within a trial, all cues are integrated additively with near-flat validity weighting: evidence for A over B is E = s · Σ_j w_j (a_j − b_j), w_j ∝ v_j^gamma. Choice is a softmax over [E, −E] with per-subject inverse temperature beta plus an independent lapse epsilon to uniform; identical rating vectors yield exactly 50%. The slip attenuates each subject's effective extremity by (1 − 2·delta) and the population's effective anti-endorsement signal to q_eff = pi(1−delta) + (1−pi)delta ≈ 0.85, which simultaneously pulls the pooled point estimates back from saturation toward the observed moderate magnitudes and compresses the between-subject mixture variance 4·pi(1−pi)·((1−2·delta)·x)^2 toward the tight real values — while preserving the falsifiable subject-level bimodality signature (two attenuated polarity clusters, not one homogeneous middle).

**Rationale:** This is a minimal-diff edit implementing the iteration-9 critic's recalibration exactly, on top of the RFI-delta architecture that produced the best rejected attempt of the loop (loss 0.0484). The iteration-9 verdict VALIDATED the polarity-slip mechanism on precisely the residuals that six prior parameter-only excursions could not touch: it collapsed between-subject variance 2-4x across the board (Exp 1: 0.056->0.016 vs real 0.012; Exp 4: 0.296->0.071 vs 0.054) — the first candidate to break the binary trait-mixture variance floor — and moved Exp 6, the base's dominant point error, from -0.806 to -0.714 (real -0.676). The gate still rejected it because the pooled points oversaturated on the tight-variance experiments (Exp 1 +0.083, Exp 4 -0.053). The critic's diagnosis is that both remaining residuals respond to the same knob, and the arithmetic dictates which: at equal effective anti-endorsement signal q_eff, high-pi + high-delta strictly dominates low-pi + low-delta on BOTH run-to-run wobble (sd of the realized fraction scales with sqrt(pi(1-pi))) and mixture variance (4*pi*(1-pi)*((1-2*delta)*x)^2). So this edit makes exactly four changes to the iter-9 candidate's calibration and nothing else: (1) HOLD pi at [0.94, 0.95] — the arbiter's ceiling and the variance-optimal corner (sqrt(0.052) vs sqrt(0.102) wobble). (2) RAISE delta to [0.08, 0.12] (midpoint ~0.10, up from ~0.07), dropping q_eff to ~0.85 (2q-1 ~ 0.70 vs the current 0.766). This pulls Exp 1 from ~0.24 toward ~0.19-0.20, Exp 4 from -0.576 toward ~-0.52, Exp 2 from 0.146 up toward ~0.17 (real 0.181), Exp 3 from -0.745 toward ~-0.71 (real -0.717), and Exp 6 from -0.714 toward ~-0.67 (real -0.676) — five of six experiments move toward target; the one cost is slight Exp 5 attenuation (~0.56 -> ~0.55 vs real 0.594), the cheapest trade available since Exp 5's current error is only -0.030. The same move compresses every between-subject variance by the factor (1-2*delta)^2 (~0.61 vs the current ~0.74). (3) Cool beta to [3, 8]: the Exp 5/6 variance residuals (4-5x real) are attributable to the beta spread pinning high-beta subjects at structural extremes while moderate-beta subjects sit partway in; a modestly cooler, tighter band attacks that without re-entering the near-chance regime that blew up iter 1. (4) Keep gamma in [0, 1] and epsilon in [0, 0.08] verbatim — gamma excursions failed twice (iters 5-6) and epsilon changes never paid. The slip stays small (delta <= 0.12, well under the ~0.15 concession threshold the critic set): the trait-sign draw, additive near-flat evidence rule, and softmax/lapse skeleton are untouched, and the subject-level bimodality prediction is retained in attenuated form. Expected signature: Exp 1 ~0.19, Exp 2 ~0.17, Exp 3 ~-0.71, Exp 4 ~-0.52, Exp 5 ~0.55-0.58, Exp 6 ~-0.68, with variances at or below iter-9's 0.016/0.016/0.177/0.071/0.169/0.194 band — which should land strictly below the 0.0393 loss floor. Per the critic's binding stopping guidance, if this recalibration is also rejected, the loop should ship the iter-3 base; but the slip mechanism is the first structural gain since iter 3 and this calibration shifts its expected loss below the base's, making it the highest-expected-value remaining move.

**Parameters:**
  - `pi`: `[0.94, 0.95]`
  - `polarity_u`: `[0, 1]`
  - `delta`: `[0.08, 0.12]`
  - `gamma`: `[0, 1]`
  - `beta`: `[3, 8]`
  - `epsilon`: `[0, 0.08]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Red-Flag Integration with Trait Polarity + attentional slip (RFI-delta).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RFI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    pi = float(parameters["pi"])
    u = float(parameters["polarity_u"])
    delta = float(parameters["delta"])

    # Trait polarity: drawn ONCE per subject (both parameters are fixed
    # for the entire subject run). s = -1 -> red-flag reading (rating of 1
    # is evidence AGAINST the carrying option); s = +1 -> conventional
    # reading (endorsement favors the carrying option). At the population
    # level P(s = -1) = E[pi] with pi in [0.94, 0.95], i.e. an
    # anti-endorsement majority near the arbiter's ceiling and a small
    # (~5%) conventional minority.
    s = -1.0 if u < pi else 1.0

    # Near-flat validity weighting: w_j = v_j^gamma, gamma in [0, 1].
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), absorbed by beta; it keeps the evidence scale O(1) for any
    # gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Signed additive evidence for A over B. With s = -1 each endorsement
    # carried by A counts AGAINST A and each endorsement carried by B
    # counts FOR A (the sparser option accumulates evidence).
    E = s * float(np.sum(w * (a - b)))

    # Softmax over [E, -E] with max-subtraction for numerical stability.
    # Identical rating vectors give E = 0 -> exactly 50/50.
    scores = np.array([E, -E])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_trait = e / e.sum()

    # Attentional slip: with probability delta the subject momentarily
    # reverts to the OPPOSITE polarity mapping for this trial alone.
    # The opposite-polarity distribution is the mirror of p_trait, so the
    # per-trial choice distribution is a (1-delta)/delta mixture of the
    # two mirrors. This is the expectation of a per-trial Bernoulli slip
    # against a stable dominant trait (lapse-structured, small delta),
    # NOT per-trial polarity mixing: the trait sign is still fixed per
    # subject and the slip only attenuates each subject's extremity by
    # (1 - 2*delta).
    p_slip = p_trait[::-1]
    p_core = (1.0 - delta) * p_trait + delta * p_slip

    # Independent lapse to uniform choice.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Sparse-Option Preference / Polarity-Inverted Evidence Integration (SPI). When choosing between two options described by binary expert ratings, a substantial share of subjects systematically mis-map cue polarity: an endorsement (rating = 1) is treated as evidence AGAINST an option, so the option with FEWER 1s (the sparser option) is perceived as higher quality — consistent with a 'fewer red flags' reading of the display or a rarity-implies-quality inference. Subjects integrate ALL cues additively with near-uniform weights: the evidence for option A over B is E = sum_j w_j * (b_j - a_j), where w_j = v_j^gamma with gamma small (validity plays at most a mild role). Choice is a softmax over [E, -E] with inverse temperature beta, plus an independent lapse epsilon to uniform. A per-subject polarity-mixing weight rho captures heterogeneity: with probability 1-rho the subject responds according to the inverted (sparse-preferring) polarity, with probability rho according to the conventional polarity. This mixture attenuates the extreme pure-sparse predictions toward the moderately negative values observed. The theory is sharply distinguished from both incumbents: unlike Take-The-Best it uses all cues and inverts the evidence sign (so it is not pinned at ~0 on the depth/conflict-invariance and margin-slope metrics where TTB is structurally stuck), and unlike power-weighted integration it places humans at the FLAT end of the weighting ladder with inverted polarity, producing the negative signatures that steep-gamma PWEI cannot generate at any parameter setting.

**Rationale:** The arbiter diagnosed the mechanistic failure precisely: TTB (pi_1) is structurally pinned near 0 on Experiments 3 and 4 (its constant-confidence invariant forces 2*p_HIGH - p_DEPTH5 - p_REV = 0 and slope = 0 exactly), while the observed values are strongly NEGATIVE (-0.717, -0.523). PWEI (pi_3) with steep gamma produces strongly POSITIVE values on those same metrics (1.05, 0.55) and cannot go negative at any parameter setting, because its evidence sign always favors the denser, top-cue-supported option. The only mechanism family that generates the observed negative signatures while still matching the mildly TTB-positive values on Experiments 1 (+0.158) and 2 (0.181 = P(tally winner), i.e. siding AGAINST the tally majority) is a polarity inversion with near-flat weighting: prefer the option with FEWER endorsements, integrating all cues. I implemented exactly the prescribed SPI: E = sum_j v_j^gamma * (b_j - a_j), gamma in [0,2] (near-tallying), softmax with beta, lapse epsilon, plus the suggested per-subject polarity mixing weight rho. Hand-verification at a representative parameter point (gamma=1, beta=10, epsilon=0.2, rho=0.1): Experiment 1 dissociation metric ≈ +0.17 (observed 0.158) — on conflict trials the sparse option usually coincides with the TTB winner because its single endorsement sits on the top cue, while tally-tie trials 7/8 slightly reverse; Experiment 2 P(tally winner on conflicts) ≈ 0.21 (observed 0.1814) — SPI sides with the TTB/sparser option against the majority; Experiment 3 invariance contrast ≈ -0.75 (observed -0.717) — p_HIGH is pushed to the lapse floor (subjects pick the all-zeros option against TTB), while p_DEPTH5 and p_REV stay above 0.5 (the sparse option happens to be the TTB winner there); Experiment 4 slope of choosing the TTB winner on the PWEI margin ≈ -0.56 (observed -0.523) — larger weighted margins toward the dense TTB winner map to LOWER choice probability. All four predictions land near the observed values simultaneously, which no incumbent can do: TTB is stuck at (0.35, 0.15, 0.00, 0.01) and PWEI at (0.25, 0.26, 1.05, 0.55). The rho mixture is essential rather than cosmetic: pure inverted polarity overshoots (e.g., Exp 3 near -0.9 to -1.6 at high beta), and the 5-35% conventional-polarity minority attenuates the pooled statistics into the observed band, exactly the attenuation the arbiter noted (pure prediction ~0.35 vs observed 0.158 on the dissociation metric). The theory remains experiment-invariant: it has one mechanism (inverted all-cue integration) whose signatures — negative margin slopes, negative invariance contrasts, anti-tally conflict choices — are exactly the pattern present in every experiment's data.

**Parameters:**
  - `gamma`: `[0, 2]`
  - `beta`: `[0.5, 20]`
  - `epsilon`: `[0.05, 0.35]`
  - `rho`: `[0, 0.35]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Sparse-option Preference / Polarity-Inverted Evidence Integration (SPI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SPI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    rho = float(parameters["rho"])

    # Near-uniform validity weighting: w_j = v_j^gamma, gamma in [0, 2].
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), absorbed by beta; it keeps the evidence scale O(1) for any
    # gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Polarity-INVERTED evidence for A over B: each cue where B carries
    # the endorsement (1) counts FOR A; each cue where A carries the
    # endorsement counts AGAINST A. Sparse options accumulate evidence.
    E = float(np.sum(w * (b - a)))

    # Softmax over [E, -E] with max-subtraction for numerical stability.
    # Inverted polarity: positive E favors A (the sparser option).
    scores_inv = np.array([E, -E])
    z = beta * (scores_inv - scores_inv.max())
    e = np.exp(z)
    p_inv = e / e.sum()

    # Conventional polarity (endorsement favors the option carrying it):
    # mirror of the inverted distribution.
    scores_norm = -scores_inv
    z2 = beta * (scores_norm - scores_norm.max())
    e2 = np.exp(z2)
    p_norm = e2 / e2.sum()

    # Per-subject polarity mixture: (1 - rho) inverted, rho conventional.
    p_core = (1.0 - rho) * p_inv + rho * p_norm

    # Independent lapse to uniform choice.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Graded Sparse-Preference with Heterogeneous Attenuation (GSP), unnormalized-evidence variant. Subjects integrate ALL cues additively with near-flat validity weighting under an anti-endorsement (sparse-preferring) reading: the evidence for option A over B is E = sum_j v_j^gamma (b_j - a_j), where each endorsement (rating = 1) carried by an option counts AGAINST it, so the sparser option accumulates evidence. The cue weights are used RAW (v_j^gamma, unnormalized) so the evidence scale preserves the ABSOLUTE endorsement-count margin of the trial independent of the experiment's feature count — the cross-experiment scale separation (moderate pooled preference on 5-cue shallow-margin trials vs strong preference on 8-cue margin-2 conflicts) that per-experiment normalization provably erases. The anti-endorsement reading is graded per subject: each subject carries a continuous polarity weight m in [0.75, 1.0] (per-trial probability of responding with the inverted polarity), a log-uniformly dispersed inverse temperature beta, and a lapse epsilon in [0.02, 0.12]: p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with identical rating vectors yielding exactly 50%. The population is a continuum of attenuations concentrated on the inverted side (no empty middle, no mirror cluster), with mean effective attenuation (1-eps)(2m-1) ~ 0.70 matching the observed extremity anchors.

**Rationale:** This is the minimal-diff edit the iteration-4 critique prescribed: the running-best (iter-4, loss 0.0470) GSP base is re-emitted verbatim except for ONE knob — the raw-beta box. The log-uniform inverse temperature is raised from [0.8, 3.0] to [1.1, 3.2] (log_beta in [ln 1.1, ln 3.2] = [0.095, 1.163]). Everything else — the raw unnormalized w_j = v_j^gamma evidence scale, the additive inverted-evidence sigmoid, the continuous graded m in [0.75, 1.0], the lapse epsilon in [0.02, 0.12], the exact-50% ties, the policy — is unchanged, per the critique's explicit instructions (no m/eps widening, which the gate refuted in iter-3; no renormalization, which iter-4's acceptance validated as the key unlock). Why this interpolation: the residuals are bracketed by scored iterations. On Exp1, iter-3's hotter effective 5-cue beta (~[5,16] normalized) overshot (0.183 vs real 0.158) while iter-4's current box undershoots (0.147); the new lower bound of 1.1 gives an effective normalized ~[3.9, 11.4], squarely between them, targeting ~0.158. On Exp5 (largest remaining miss, 0.503 vs real 0.594), shifting log-uniform mass off the coldest decile steepens the mid-cell psychometric, and iter-3 empirically confirmed hotter settings move Exp5 toward real (0.559). Exp2 (0.206 vs real 0.181) and Exp8 (0.317 vs real 0.347) both improve as margin-2 and weakest high-contrast trials saturate further. Exp7 is attenuation-set, not beta-set, so it stays ~0.69. Guardrail applied directly: the critique warned that if the upper bound of 3.8 pushes Exp4 past ~-0.56 (currently near-perfect at -0.5238) or Exp6 past ~-0.72 (already 0.016 beyond real at -0.6921), the upper bound should be capped at 3.2. Since hotter beta monotonically deepens both of these negative signatures and Exp6 is ALREADY past real, I adopt the 3.2 cap preemptively — the interpolation is primarily a lower-bound correction, and the modest upper-bound trim (3.0 -> 3.2 is still a slight raise, preserving most of the mean-preserving between-subject dispersion gain on Exp3/4/5 where variance remains ~40-60% under real) while protecting the two experiments where the base is already at or past its target. If this interpolation fails to beat 0.0470, the loop has converged and iter-4 ships as-is; its worst residual (Exp5, Δ -0.092) is within the spread other viable theories show.

**Parameters:**
  - `gamma`: `[0, 1.5]`
  - `log_beta`: `[0.095, 1.163]`
  - `epsilon`: `[0.02, 0.12]`
  - `m`: `[0.75, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Graded Sparse-Preference with Heterogeneous Attenuation (GSP),
    # UNNORMALIZED-evidence variant.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GSP expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    # log-uniform inverse temperature: beta = exp(log_beta),
    # log_beta ~ U[ln 1.1, ln 3.2]  =>  beta log-uniform over [1.1, 3.2],
    # calibrated to the RAW (unnormalized) evidence scale below.
    # Lower bound raised from 0.8 to 1.1 (the interpolation lever);
    # upper bound capped at 3.2 (guardrail for Exp4/Exp6 overshoot).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization by
    # sum(w). This preserves the absolute endorsement-margin scale:
    # the evidence magnitude reflects the raw count margin of the
    # trial (a margin-2 conflict carries twice the evidence of a
    # margin-1 conflict) independent of n_features.
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. The sparser option accumulates
    # evidence. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic of the inverted evidence.
    # (both branches exponentiate a non-positive argument, so no
    # overflow is possible)
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED polarity (sparse-preferring), with probability
    # (1 - m) with the conventional polarity. This is the expectation of
    # a per-trial Bernoulli polarity draw against a continuous,
    # strongly-inverted per-subject weight m in [0.75, 1.0].
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```
