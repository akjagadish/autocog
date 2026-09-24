# Round 2 — Theories

**Verdict:** `new_model` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3_1` — KILLED ✗

**Description:** Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

**Rationale:** The arbiter's core diagnosis is correct and I keep it: the reversal signature (kappa = 1) is at full strength and only the noise regime was miscalibrated — the previous candidate's implied cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 averaged only ~0.60, which is exactly why the reversed-follow metrics came in low. The mechanism code (reversed validity vector, stable descending-subjective-validity argsort, first strictly discriminating cue decides, softmax-with-lapse noise, uniform guess on full ties) is therefore kept structurally identical; only the noise ranges are recalibrated.

However, I set the ranges by a joint calibration over ALL FOUR metrics rather than the arbiter's literal [0.9, 1.5] x [0.03, 0.10]. Writing each metric in terms of the single follow probability p makes the algebra transparent. In this design family the reversed-cascade winner always opposes the instructed-validity TTB winner, so: (a) Experiment 1's metric is level - slope with slope = 0 (the theory's flat psychometric profile) and level = P(pick TTB winner | tally margin <= 0) = 1 - p, so Exp1 = 1 - p; (b) Experiment 3's reversed-follow rate is 2p - 1; (c) Experiment 4's contrast is (2p - 1) - (1 - 2p) = 4p - 2; (d) Experiment 2's alignment stays small and positive, scaling with (2p - 1). The real data therefore pin p from three directions: Exp1 wants p ~ 0.64, Exp3 wants p ~ 0.73, Exp4 wants p ~ 0.695. The least-squares optimum over these (weighting each metric's sensitivity: 1, 2, and 4 respectively) is p* ~ 0.70. The arbiter's literal ranges imply E[p] ~ 0.75, which overshoots Experiment 4 badly (metric = 4p - 2 would reach ~1.0 vs real 0.78) and pushes Experiment 1 down to ~0.25 vs real 0.363. So I keep the arbiter's direction — beta roughly doubled from [0.45, 0.55], epsilon midpoint lowered from 0.13 to 0.11, p raised from 0.60 to ~0.70 — but land at the joint optimum rather than 0.73+.

Concretely, with beta ~ U[0.8, 1.2] and epsilon ~ U[0.08, 0.14]: E[sigmoid(beta)] = 0.731, E[epsilon] = 0.11, so E[p] = 0.89 * 0.731 + 0.055 = 0.71. Expected metric values: Exp1 = 1 - p ~ 0.30 (real 0.363; previous candidate 0.41), Exp2 ~ +0.05-0.08 (real 0.036; previous 0.061), Exp3 = 2p - 1 ~ 0.41 (real 0.453; previous 0.195), Exp4 = 4p - 2 ~ 0.80 (real 0.779; previous 0.387). The two large errors of the running-best (Exp3 off by 0.26, Exp4 off by 0.39) shrink to under 0.07, while the two already-good experiments (Exp1, Exp2) degrade only marginally — a clear net improvement in aggregate loss. kappa stays fixed at {1.0} because the theory claims full misbinding and the observed reversal signature shows no attenuation; only choice consistency was underestimated. Parameter heterogeneity across subjects (the beta/epsilon ranges) is retained so the between-subject variance of the metrics remains human-like rather than deterministic.

**Parameters:**
  - `beta`: `[0.8, 1.2]`
  - `epsilon`: `[0.08, 0.14]`
  - `kappa`: `{1.0}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality (misbinding TTB). One-reason
    # decision making: cues are consulted one at a time and the first
    # strictly discriminating cue decides. The subjective hierarchy is
    # ANTI-correlated with the instructed validities (kappa = 1: the
    # communicated validity list is misbound to feature positions,
    # highest validity bound to the last-presented feature), so the
    # cascade runs in ascending true validity. No integration across
    # cues occurs -> flat psychometric profile across tally margins.
    # Noise: softmax (inverse temperature beta) over the binary winner
    # score, plus an independent lapse epsilon mixing in a uniform
    # choice. History is ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # Subjective validities: mixture of the instructed vector and its
    # end-to-end reversal. kappa = 1 -> fully reversed hierarchy
    # (full positional misbinding), per the theory's fixed claim.
    kappa = float(parameters["kappa"])
    w = (1.0 - kappa) * val + kappa * val[::-1]

    # Descending SUBJECTIVE validity; stable argsort so ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-w, kind="stable")

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_2` — SURVIVED ✓

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

### `pi_3_2` → slot 1 (via `new_model`)

**Description:** Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

**Rationale:** The decision rule (kappa=1 reversed TTB cascade, uniform guess on ties, softmax+lapse output) is unchanged — the arbiter is right that the misfit is purely in the noise magnitudes. But I calibrated the new noise regime from the algebra of each metric rather than adopting the suggested ranges verbatim.

Deriving each experiment's implied follow probability p = (1-eps)*sigmoid(beta) + eps/2 from the observed values and the previous ACCEPTED candidate's (pi_3_1, p_bar≈0.703) predictions:

- Exp 1: on every margin<=0 trial the reversed cascade winner OPPOSES the TTB winner, so metric = 1 - p; real 0.3633 -> p ≈ 0.637 (pi_3_1 gave 0.2933, i.e. p too high).
- Exp 2: E[s*margin] = (2p-1)*C with C≈0.29 (recovered from pi_3: 0.0608 at p=0.606 and pi_3_1: 0.1208 at p=0.703); real 0.0358 -> p ≈ 0.56.
- Exp 3: metric = 2p-1 exactly; real 0.4533 -> p ≈ 0.727 (pi_3_1 already near-perfect at 0.4276).
- Exp 4: metric = 4p-2; real 0.7787 -> p ≈ 0.695 (pi_3_1 near-perfect at 0.8027).
- Exp 5: metric ≈ p + 0.015; real 0.6133 -> p ≈ 0.60 (pi_3_1 overshot at 0.7178).
- Exp 6: metric = 2p-1; real 0.34 -> p ≈ 0.67.

The per-experiment optima range from 0.56 to 0.73; minimizing the (per-experiment normalized) L2 distance over p gives an optimum at p_bar ≈ 0.68, versus the current 0.703. This lowers the predicted Exp 1 level (0.29 -> ~0.36), Exp 2 value (0.121 -> ~0.07), Exp 5 follow rate (0.718 -> ~0.69) and Exp 6 contrast (0.39 -> ~0.36) while only marginally degrading the excellent Exp 3/Exp 4 fits — a net aggregate-loss improvement.

Why I deviate from the arbiter's literal ranges (beta [0.1,0.9], eps [0.10,0.40]): those ranges imply p_bar ≈ 0.59 (e.g. beta=0.1, eps=0.25 gives p≈0.52; beta=0.9, eps=0.25 gives p≈0.66), which would drag Exp 3 down to ~0.18 (real 0.4533) and Exp 4 down to ~0.36 (real 0.7787) — exactly the two experiments where the arbiter itself says the reversed-hierarchy direction is strongly confirmed and should not be degraded. My ranges follow the arbiter's *direction* (lower, wider beta; higher epsilon; more between-subject heterogeneity, with per-subject p spanning ~[0.62, 0.75]) but center the follow rate at ~0.68, the value the cross-experiment algebra actually demands. kappa stays fixed at 1.0: for monotone validity lists any kappa in [0.85, 1] yields the identical fully-reversed cue order, so freeing it adds nothing, and values below 0.5 would flip the hierarchy and destroy the confirmed Exp 2/5/6 fits.

**Parameters:**
  - `beta`: `[0.6, 1.2]`
  - `epsilon`: `[0.08, 0.18]`
  - `kappa`: `{1.0}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality (misbinding TTB). One-reason
    # decision making: cues are consulted one at a time and the first
    # strictly discriminating cue decides. The subjective hierarchy is
    # ANTI-correlated with the instructed validities (kappa = 1: the
    # communicated validity list is misbound to feature positions,
    # highest validity bound to the last-presented feature), so the
    # cascade runs in ascending true validity. No integration across
    # cues occurs -> flat psychometric profile across tally margins.
    # Noise: softmax (inverse temperature beta) over the binary winner
    # score, plus an independent lapse epsilon mixing in a uniform
    # choice. History is ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # Subjective validities: mixture of the instructed vector and its
    # end-to-end reversal. kappa = 1 -> fully reversed hierarchy
    # (full positional misbinding), per the theory's fixed claim.
    kappa = float(parameters["kappa"])
    w = (1.0 - kappa) * val + kappa * val[::-1]

    # Descending SUBJECTIVE validity; stable argsort so ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-w, kind="stable")

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges averages ~0.68 with per-subject
    # values spanning ~[0.62, 0.75].
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```
