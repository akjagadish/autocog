# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3_2` — SURVIVED ✓

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

**Description:** Weakest-Expert-First Frugality (value-based anti-validity TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude/tally information. However, the subjective cue hierarchy is anti-correlated in VALUE with the instructed validities: subjects systematically consult the cue with the LOWEST advertised validity first, then the next-lowest, and so on (equivalently, subjective weights w_j = -validity_j). Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope). When no cue discriminates, the model guesses uniformly. Validity ties are broken by a free per-subject tie-breaking parameter (early-position vs late-position preference among equally weak cues). Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice, giving an implied cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 that spans roughly 0.54 to 0.86 across subjects with a mean near 0.70 — high enough to produce the steep contrasts in the non-monotone-validity experiments while moderate values cover the monotone ones.

**Rationale:** The arbiter diagnosed the fatal flaw of the pi_3 family: it implemented the anti-validity hierarchy by POSITIONALLY reversing the communicated validity list. In the six monotone-validity experiments (where validity decreases with feature index) positional reversal coincides with ascending-validity sorting, so pi_3_2 succeeded there; but in the two non-monotone experiments (presented Experiments 7 and 8) positional reversal wrongly promotes mid- and high-validity cues to the top of the hierarchy (f4 at 0.65 in Experiment 7; f3 at 0.95 in Experiment 8), producing strongly POSITIVE metric values where the real data are strongly NEGATIVE (-0.31 and -0.29). The fix is to compute the anti-correlation on validity VALUES: sort cues by ascending true validity. I verified analytically that this flips both signatures: in Experiment 7 the metric becomes (7-14q)/12 (e.g., -0.245 at q=0.70, vs real -0.311, where pi_3_2 predicted +0.36); in Experiment 8 the single margin-0 cell T11 has the weakest cue f0 favoring A while the positional-reversal decider is f5 favoring B, so the metric becomes 0.5-q (e.g., -0.20 at q=0.70, vs real -0.289, where pi_3_2 predicted +0.19). Meanwhile, in every monotone-validity experiment the ascending-value order is IDENTICAL to the positional reversal, so the already-good predictions of pi_3_2 on Experiments 1-6 are preserved exactly (Experiment 1 metric = 1-q; Experiment 3 = 2q-1; Experiment 4 = 4q-2; Experiment 5 = q; Experiment 6 = 2q-1). I then solved for the implied follow probability q each experiment demands (0.64, 0.58, 0.73, 0.70, 0.61, 0.67, 0.76, 0.79 respectively) and set the noise ranges beta in [0.5, 1.5], epsilon in [0.06, 0.18] so that the mean implied q = E[(1-eps)*sigmoid(beta)+eps/2] lands at approximately 0.70 — the value that minimizes the aggregate squared error across all eight per-experiment constraints (q=0.70 yields RMS error ~0.06, better than both 0.68 and 0.72) — while the per-subject span [0.54, 0.86] covers the heterogeneity the arbiter requested. The tie_break parameter implements the arbiter's suggested free Bernoulli over early-vs-late position among equally weak cues; I verified it is inert for all eight metrics' diagnostic cells (tied cues never jointly determine a scored cell), so it adds subject heterogeneity without distorting predictions. The model retains the empirically validated one-reason, margin-blind structure (flat across tally margins, uniform guess on full ties) that Tallying (pi_2) lacked, while replacing the positional mechanism with the value-based one that produces the correct negative signs in the two non-monotone experiments.

**Parameters:**
  - `beta`: `[0.5, 1.5]`
  - `epsilon`: `[0.06, 0.18]`
  - `tie_break`: `{0, 1}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Weakest-Expert-First Frugality (value-based anti-validity TTB).
    # One-reason decision making: cues are consulted one at a time and
    # the first strictly discriminating cue decides. The subjective
    # hierarchy is anti-correlated in VALUE with the instructed
    # validities: the cue with the LOWEST advertised validity is
    # consulted FIRST, then the next-lowest, etc. (subjective weights
    # w_j = -validity_j). This is a VALUE-based sort (ascending
    # validity), not a positional reversal of the validity list, so in
    # experiments where the validity list is not monotone in feature
    # position the two hierarchies genuinely differ. No integration
    # across cues occurs -> flat psychometric profile across tally
    # margins. Noise: softmax (inverse temperature beta) over the
    # binary winner score, plus an independent lapse epsilon mixing in
    # a uniform choice. History is ignored (no feedback in this task).
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

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    tie_break = float(parameters["tie_break"])

    # Weakest-expert-first hierarchy: sort cues by ASCENDING validity
    # value. Ties in validity are broken by the free tie_break
    # parameter: 0 -> earlier feature position first (stable),
    # 1 -> later feature position first among equally weak cues.
    # np.lexsort uses the LAST key as primary: primary = val ascending,
    # secondary = position (forward or reversed).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, val))

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

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges spans ~[0.54, 0.86] across
    # subjects with a mean near 0.70.
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
