# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

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


### slot 2 — `pi_2` — KILLED ✗

**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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


## Replacement

### `pi_4` → slot 2 (via `new_theory`)

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
