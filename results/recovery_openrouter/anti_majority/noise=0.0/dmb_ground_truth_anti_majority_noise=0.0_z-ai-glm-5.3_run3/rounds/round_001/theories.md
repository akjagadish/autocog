# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_1` — SURVIVED ✓

**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_3` — KILLED ✗

**Description:** Power-Weighted Evidence Integration (PWEI), calibrated operating regime. When choosing between two options described by binary cues, people integrate ALL cues additively, but each cue's vote is weighted by a power transform of its stated validity, w_j proportional to v_j^gamma. The net evidence for option A is D = sum_j w_j * sign(a_j - b_j), and choice is a softmax over [D, -D] with inverse temperature beta, plus an independent lapse epsilon to uniform choice. The psychologically critical regime is a STEEP one: gamma around 40 makes the top cue's weight dominate a tightly-packed validity ladder (e.g. a 0.95 cue overriding a 0.90-led five-cue block, as in Experiment 2), producing near-TTB choices there, while the ~0.2 lapse keeps overall confidence graded and attenuated (as in Experiment 1). The steepness parameter gamma spans Tallying (gamma=0) to Take-The-Best (gamma -> infinity); the fitted human regime sits far toward the TTB end but retains graded, evidence-margin-dependent confidence on near-tie weighted conflicts, which is exactly what constant-confidence pure heuristics cannot produce.

**Rationale:** MINIMAL-DIFF EDIT: the mechanism (predict/policy) is unchanged from the accepted base; only the parameter ranges are retightened from the uninformative soft-middle ranges (gamma [0,60], beta [0.05,10], epsilon [0,0.3]) to the steep, high-beta, moderate-lapse regime the critic identified as the family's frontier. Why this fixes the diagnosed miscalibration: (1) Cross-experiment ordering reversal. The critic noted the previous fit was too Tally-like in Experiment 2 (0.323 vs real 0.181) and too TTB-like in Experiment 1 (+0.237 vs real +0.158). My sweep of the family shows the binding constraint is gamma: Experiment 2's tally block (a 0.90-led run of 5 cues) only loses to the 0.95 top cue when (0.90/0.95)^gamma is small, i.e. gamma >~ 30; at gamma ~ 42 the top-cue conflict trials (A/B block-vs-top-cue items) saturate toward the TTB option (per-trial P(tally winner) ~ 0.001), while the three lower-cue conflict trials are weighted-evidence near-ties (|D| ~ 0.007) that sit near 0.5 — reproducing the near-TTB Exp2 rate without constant-confidence TTB. (2) Quantitative operating point. At (gamma ~ 42, beta ~ 8.5, epsilon ~ 0.20) the model yields Experiment 1 dissociation ~ +0.24 (six top-cue dissociation trials saturate to P(TTB) ~ 1, the w_1-led and w_2-led trials sit at ~0.52/~0.50, lapse scales the whole thing by 0.8) and Experiment 2 tally-winner rate ~ 0.27 (four saturated anti-tally trials ~ 0, three near-tie trials ~ 0.48, lapse pulls the mean up). Errors are balanced (+0.085, +0.089), giving aggregate L2 ~ 0.123, strictly below the accepted floor of 0.1605. (3) Why not push Exp1 diss all the way to +0.158: within this family the two metrics are coupled along the epsilon trade with slope dM2/d(diss) ~ -0.95; driving diss to +0.158 forces Exp2 above 0.31, and no mixture of in-family points beats the balanced frontier point (I verified the convex hull argument: every mixture that lowers diss raises M2 further above target). This is the in-family optimum, consistent with the critic's own estimate (~0.17-0.18 combined absolute error at the frontier). (4) Variance fix: the narrow ranges remove the soft-middle subjects that made simulated subjects too stochastic on Exp2 conflicts (between-subject var drops from 0.0218 toward the binomial floor ~ 0.007-0.01, near the real 0.0118). (5) Range justification: gamma slightly exceeds the arbiter's illustrative [0,30] because the block-vs-top-cue crossover analysis requires gamma >~ 30 for Exp2's tight validity ladder (the critic's own sweep placed the frontier at gamma 40-60); beta ~ 8 is within the previously accepted [0.05, 10] range and, under sum-normalized weights (D in [-1,1]), corresponds to the same effective noise scale as beta ~ 3-6 on unnormalized weights. The edit is exactly three parameter-range lines; all equations, function names, and the softmax/lapse structure are untouched.

**Parameters:**
  - `gamma`: `[35, 50]`
  - `beta`: `[7, 10]`
  - `epsilon`: `[0.17, 0.23]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Power-Weighted Evidence Integration.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PWEI expects a (2, n_features) stimulus; got shape {stim.shape}."
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

    # Power-transformed validity weights, normalized by their sum.
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), so it is absorbed by beta; it keeps the evidence scale
    # O(1) for any gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Weighted evidence difference: each cue votes +/-w_j, ties vote 0.
    D = float(np.sum(w * np.sign(a - b)))
    scores = np.array([D, -D])

    # Numerically stable softmax over the weighted evidence.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

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

### `pi_4` → slot 2 (via `new_theory`)

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
