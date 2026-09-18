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

**Description:** Confidence-Capped Cascade Integration proposes that people translate each communicated cue validity into an absolute diagnosticity signal and inspect cues from most to least valid. A discriminating cue controls the decision immediately when its diagnosticity exceeds a subject-specific confidence threshold. Earlier weak cues do not force premature stopping: search continues until a sufficiently diagnostic cue is encountered. If no discriminating cue reaches threshold, the decision maker switches to compensatory integration, summing all discriminating cues with weights determined by their stated validities. Thus, the theory produces one-cue dominance when reliable cues are available but graded multi-cue sensitivity when every cue is weak. Subject heterogeneity arises from differences in confidence threshold, sensitivity to stated validity, integration weighting, response temperature, and lapse rate.

**Rationale:** The model replaces equal-weight tallying with a genuine validity-gated cascade while avoiding the rigidity of strict Take The Best. In both reported experiments, the highly valid early discriminating cues will generally cross the low-to-moderate absolute confidence threshold, so later agreeing or conflicting cues are ignored. This yields top-cue control in Experiment 1 and approximately margin-invariant response reliability in Experiment 2, matching the observed metrics. The mechanism remains distinguishable from strict TTB across experiments: lowering absolute validities toward 0.5 causes cues to fall below threshold, activating validity-weighted integration and producing graded effects of additional evidence. Because the gate depends on absolute log diagnosticity rather than cue rank alone, holding ranks and signs fixed while changing absolute validities predicts a qualitative transition from one-cue to multi-cue processing. The narrow lapse range and moderately high response temperatures preserve robust majority-level dominance without making behavior deterministic, while all cognitive parameters vary at the subject level.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `confidence_threshold`: `[0.05, 0.2]`
  - `validity_sensitivity`: `[0.75, 1.5]`
  - `integration_exponent`: `[0.5, 2.5]`
  - `beta`: `[4.0, 16.0]`
  - `epsilon`: `[0.0, 0.15]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Confidence-Capped Cascade Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got shape {validities.shape}."
        )

    threshold = float(parameters["confidence_threshold"])
    sensitivity = float(parameters["validity_sensitivity"])
    integration_exponent = float(parameters["integration_exponent"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # A validity is the communicated probability that a cue points toward
    # the superior option. Its log odds are therefore an absolute measure
    # of the cue's diagnosticity. Clipping handles validity exactly equal
    # to 0.5 or 1 without numerical singularities.
    v = np.clip(validities, 0.5, 1.0 - 1e-9)
    log_diagnosticity = np.log(v / (1.0 - v))
    subjective_confidence = sensitivity * log_diagnosticity

    # Stable sorting preserves the displayed feature order when two cues
    # have equal communicated validity.
    cue_order = np.argsort(-validities, kind="stable")
    differences = stim[0] - stim[1]

    decision_evidence = None
    for j in cue_order:
        if differences[j] == 0.0:
            continue
        if subjective_confidence[j] >= threshold:
            # The first sufficiently diagnostic discriminating cue caps
            # search. Positive evidence favors A and negative favors B.
            decision_evidence = (
                np.sign(differences[j]) * subjective_confidence[j]
            )
            break

    if decision_evidence is None:
        # When no cue is individually convincing, combine all available
        # weak evidence. The exponent controls how sharply integration
        # privileges validity differences.
        weights = np.power(
            np.maximum(log_diagnosticity, 1e-12), integration_exponent
        )
        decision_evidence = float(
            np.sum(weights * np.sign(differences))
        )

    # Symmetric utilities ensure exact indifference when integrated evidence
    # is zero. Positive evidence favors action 0 (A), negative favors 1 (B).
    logits = beta * np.array(
        [0.5 * decision_evidence, -0.5 * decision_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_4` → slot 2 (via `new_theory`)

**Description:** Noisy Priority Race Search proposes that communicated cue validities establish persistent but imperfect subjective cue priorities rather than a universally fixed ordering. Each subject transforms validity log-odds into priority, adds a stable cue-specific representation bias, and has a serial-position prior favoring earlier displayed cues. On each trial, independent retrieval fluctuations perturb these priorities. Cues are inspected in the resulting stochastic order, tied cues are skipped, and search stops at the first discriminating cue. Only that cue determines the intended response; evidence from uninspected or lower-priority cues is never integrated. Small validity gaps can therefore produce stable individual departures and trial-level reversals of the stated ordering, whereas large validity gaps produce nearly deterministic lexicographic search. Response execution is separately subject to temperature-controlled errors and occasional lapses.

**Rationale:** This is a minimal calibration edit that leaves the pure first-discriminating-cue priority race unchanged. Raising the retrieval-noise floor from 0.02 to 0.04 removes the most deterministic priority-race corner while preserving strong lexicographic dominance. Narrowing response temperature to 1.6–2.0 prevents near-errorless subjects from crossing Experiment 3's 90% consistency threshold, while still yielding roughly 83–88% execution fidelity—comfortably above the majority criterion in Experiment 1. The lapse range is reduced to 0–0.05 so this calibration reflects priority competition and bounded execution fidelity rather than broad generic guessing. Together, these changes should reduce the categorical between-subject variance in Experiments 2 and 3 and modestly increase Experiment 4's reversal rate toward 0.162 without introducing any compensatory aggregation.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `priority_bias`: `[(-0.08, 0.08)] * n_features`
  - `validity_sensitivity`: `[1.0, 5.0]`
  - `serial_priority`: `[0.15, 0.45]`
  - `retrieval_noise`: `[0.04, 0.08]`
  - `response_temperature`: `[1.6, 2.0]`
  - `lapse_rate`: `[0.0, 0.05]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Noisy Priority Race Search expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    persistent_bias = np.asarray(parameters["priority_bias"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )
    if persistent_bias.ndim != 1 or persistent_bias.size != n_features:
        raise ValueError(
            f"priority_bias must have length {n_features}; got {persistent_bias.shape}."
        )

    validity_sensitivity = float(parameters["validity_sensitivity"])
    serial_priority = float(parameters["serial_priority"])
    retrieval_noise = float(parameters["retrieval_noise"])
    beta = float(parameters["response_temperature"])
    lapse_rate = float(parameters["lapse_rate"])

    # Persistent subjective priority representation. Validity log odds encode
    # communicated diagnosticity. The serial term supplies a weak display-order
    # prior and resolves equal-validity cues without assuming exact equality in
    # their subjective representations.
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-9)
    validity_signal = np.log(v / (1.0 - v))
    serial_signal = (n_features - 1) - np.arange(n_features, dtype=np.float64)
    priority = (
        validity_sensitivity * validity_signal
        + serial_priority * serial_signal
        + persistent_bias
    )

    differences = stim[0] - stim[1]
    discriminating = np.flatnonzero(differences != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # If retrieval shocks are iid Gumbel, the probability that a particular
    # discriminating cue is reached first is a softmax over its priority. Cues
    # that tie on the current trial may be inspected but cannot terminate
    # search, so they do not enter the race among stopping cues.
    race_logits = priority[discriminating] / max(retrieval_noise, 1e-12)
    race_logits = race_logits - np.max(race_logits)
    first_prob = np.exp(race_logits)
    first_prob /= first_prob.sum()

    # Marginalize over which single cue wins the priority race. This is not
    # evidence integration: in every latent search realization exactly one
    # first discriminating cue determines the intended choice.
    favors_a = differences[discriminating] > 0.0
    intended_a = float(np.sum(first_prob[favors_a]))

    # Conditional on a stopping cue, response temperature permits execution
    # errors of equal size in either direction.
    if beta >= 0.0:
        follow_probability = 1.0 / (1.0 + np.exp(-beta))
    else:
        exp_beta = np.exp(beta)
        follow_probability = exp_beta / (1.0 + exp_beta)

    core_a = (
        intended_a * follow_probability
        + (1.0 - intended_a) * (1.0 - follow_probability)
    )
    p_a = (1.0 - lapse_rate) * core_a + lapse_rate * 0.5
    p_a = float(np.clip(p_a, 0.0, 1.0))

    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
