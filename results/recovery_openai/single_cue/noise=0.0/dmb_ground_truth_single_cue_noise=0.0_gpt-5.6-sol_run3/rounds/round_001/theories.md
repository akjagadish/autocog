# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Communicated-Reliability Accumulation theory proposes that people integrate every discriminating expert rating after translating communicated validity into subjective diagnostic weight. The weight is a mildly superlinear log-odds transformation plus a modest common reliability floor: w_j = logit(v_j)^gamma + c. The log-odds component privileges more reliable experts, while the positive floor represents a minimal contribution assigned to any expert judged better than chance. Net evidence favoring A is E = sum_j w_j(A_j-B_j). Consequently, sufficiently numerous weaker experts can overturn a stronger cue, and concordant cue-number margins generate graded confidence. Choice sensitivity varies moderately across people, while only very rare lapses produce validity-independent guessing. Choice history is ignored because the task provides no outcome feedback.

**Rationale:** This is a minimal calibration of the accepted iteration-8 model: only the gamma and reliability_floor ranges change. Raising the common floor from [0.04, 0.10] to [0.10, 0.14] should extend the accepted improvement in Experiment 2 by making each additional concordant cue contribute a stronger common increment, thereby sharpening the tally-margin profile. Gamma is moved only mildly above one, to [1.02, 1.05], so the most reliable cues regain relative leverage on Experiment 1 conflict trials and the TTB-winner rate can move back toward 0.403. This is much narrower than the previously rejected strongly superlinear intervention and retains the successful affine floor. Beta and epsilon are left unchanged to isolate the two-dimensional calibration recommended by the critic and avoid reintroducing saturation, excessive lapses, or unnecessary Experiment 1 variance.

**Parameters:**
  - `beta`: `[0.24, 0.58]`
  - `epsilon`: `[0.0, 0.005]`
  - `gamma`: `[1.02, 1.05]`
  - `reliability_floor`: `[0.10, 0.14]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Communicated-Reliability Accumulation.
    # state has shape (2, n_features), with rows for options A and B.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Reliability accumulation expects shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Clipping only protects the log-odds transform at its mathematical
    # endpoints. A modest common floor strengthens accumulation by cue number,
    # while mild superlinearity preserves the leverage of highly valid cues.
    v = np.clip(validities, 0.5, 1.0 - 1e-12)
    log_odds = np.log(v) - np.log1p(-v)
    gamma = float(parameters["gamma"])
    reliability_floor = float(parameters["reliability_floor"])
    reliability_weights = np.power(log_odds, gamma) + reliability_floor

    a = stimulus[0]
    b = stimulus[1]
    evidence_for_a = float(np.dot(reliability_weights, a - b))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Symmetric logits make their difference beta * evidence_for_a, so the
    # core probability of A is sigmoid(beta * evidence_for_a).
    logits = np.array(
        [0.5 * beta * evidence_for_a, -0.5 * beta * evidence_for_a],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probabilities = (1.0 - epsilon) * p_core + epsilon * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.ones(len(probabilities), dtype=np.float64) / len(probabilities)
    else:
        probabilities /= total
    return int(np.random.choice(len(probabilities), p=probabilities))
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

**Description:** Compressed-Reliability Consensus theory holds that people represent every discriminating expert primarily as one categorical vote and encode communicated reliability through a separate, bounded comparison of the two endorsing groups. For each option, attention is biased moderately toward its more reliable endorsers, but their reliabilities are summarized as a softmax-weighted mean rather than accumulated expert by expert. Net evidence is therefore the full signed tally plus an identity-sensitive contrast between the two bounded group summaries. Agreement additionally amplifies confidence as the absolute tally margin grows, with amplification saturating after a decisive coalition has formed. This representation permits expert identity to determine equal-tally choices, preserves substantial influence for near-chance experts, and allows sufficiently numerous weak experts to override the strongest expert. Reliability sensitivity, consensus amplification, choice sensitivity, and lapse rate vary across subjects. Because the task provides no correctness feedback, choices are reversal-symmetric and independent of history.

**Rationale:** The accepted iteration-5 model is retained except for the reliability-combination block. Instead of adding a reliability bonus from every discriminating expert, the revision preserves the complete signed tally and adds a bounded contrast between option-level reliability summaries. Within each side, a fixed moderate softmax temperature of 4 emphasizes more reliable endorsers without becoming a hard maximum; taking a weighted mean keeps each side's reliability signal bounded. This should increase strongest-expert leverage in ordinary conflicts and equal tallies without allowing several medium-validity bonuses to accumulate. It can also flatten Experiment 2's margin-consistency gradient because reliability can support tally winners even at small margins. The six-expert baseline coalition remains fully represented in Experiment 4, while the accepted margin-five consensus cap, choice-sensitivity range, lapse range, and all hierarchical parameter variation remain unchanged.

**Parameters:**
  - `reliability_sensitivity`: `[0.65, 1.05]`
  - `consensus_amplification`: `[0.03, 0.07]`
  - `choice_sensitivity`: `[0.13, 0.30]`
  - `lapse_rate`: `[0.0, 0.08]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Compressed-Reliability Consensus model. History is intentionally ignored
    # because the task supplies no outcome feedback from which to recalibrate.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # Every discriminating expert contributes a full baseline vote through the
    # signed tally. Reliability is encoded separately as a bounded summary of
    # each endorsing side, preventing identity increments from accumulating
    # without bound as a coalition grows.
    v = np.clip(validities, 0.5, 1.0)
    reliability_code = np.tanh(3.0 * (v - 0.5)) / np.tanh(1.5)

    difference = stimulus[0] - stimulus[1]
    a_endorsers = difference > 0.0
    b_endorsers = difference < 0.0
    tally_margin = float(np.sum(a_endorsers) - np.sum(b_endorsers))

    def side_reliability(mask):
        values = reliability_code[mask]
        if values.size == 0:
            return 0.0
        attention_logits = 4.0 * values
        attention_logits = attention_logits - np.max(attention_logits)
        attention = np.exp(attention_logits)
        attention /= attention.sum()
        return float(np.dot(attention, values))

    reliability_contrast = (
        side_reliability(a_endorsers) - side_reliability(b_endorsers)
    )
    alpha = float(parameters["reliability_sensitivity"])
    weighted_balance = tally_margin + alpha * reliability_contrast

    # Agreement is represented separately by the unweighted vote margin.
    # It changes confidence but does not erase which experts supplied the
    # votes. Leaving margin-one and tied tallies unamplified preserves
    # uncertainty in sparse conflicts and identity effects in equal tallies.
    # Amplification saturates at the level reached by a margin-five coalition.
    excess_consensus = max(abs(tally_margin) - 1.0, 0.0)
    consensus_gain = float(parameters["consensus_amplification"])
    confidence_multiplier = 1.0 + consensus_gain * min(excess_consensus ** 1.5, 8.0)
    evidence_for_a = weighted_balance * confidence_multiplier

    sensitivity = float(parameters["choice_sensitivity"])
    logits = np.array(
        [0.5 * sensitivity * evidence_for_a,
         -0.5 * sensitivity * evidence_for_a],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + lapse * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(probabilities.size, 1.0 / probabilities.size)
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))
```
