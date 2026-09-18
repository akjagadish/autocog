# Round 2 — Theories

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


### slot 2 — `pi_4` — KILLED ✗

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


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Accessibility-Gated Validity Cascade: Communicated validities create a stable, descending cue hierarchy rather than a noisy priority race. On each trial, however, a cue can control choice only if its identity, stated validity, and displayed value are successfully accessible. Accessibility varies stably across people and cues and declines with list length and absolute display position, with a modest accelerating cost for cues near the end of long displays. Search follows the advertised-validity hierarchy among accessible cues and stops at the first accessible cue that discriminates. Lower-ranked evidence is never integrated. If no discriminating cue is accessible, the decision maker guesses. Retrieval failures are distinct from response execution errors: after forming an intended choice, the person can accidentally reverse it or lapse to a random response.

**Rationale:** The model preserves the experiment-invariant success of lexicographic choice while replacing pi_4's problematic noisy priority race. Validity gaps never directly generate order reversals: whenever two cues are accessible, the one with higher advertised validity is consulted first. Deviations arise because a cue can be unavailable, and that failure is systematically related to cognitive accessibility. Early high-validity cues are normally retrieved, supporting dominance in Experiments 1, 3, and 4. Occasional retrieval and execution failures permit imperfect consistency and the modest majority override in Experiment 5. Most importantly, the late-position cost makes the high-validity cue at the end of the 18-cue display less accessible than an early high-validity cue, allowing the negative endpoint contrast in Experiment 6 without claiming that larger validity gaps make the validity ordering noisier. Because every latent decision still stops on one accessible cue, the model does not acquire graded tally-margin sensitivity in Experiment 2. The parameter ranges keep baseline accessibility high, cue-specific variation modest, and execution noise bounded, preventing the mechanism from mimicking arbitrary cue rankings.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `cue_accessibility_offsets`: `[(-0.15, 0.15)] * n_features`
  - `baseline_accessibility`: `[3.2, 3.8]`
  - `list_length_gradient`: `[0.035, 0.055]`
  - `position_gradient`: `[0.015, 0.035]`
  - `late_position_cost`: `[0.003, 0.0055]`
  - `execution_flip`: `[0.09, 0.13]`
  - `lapse_rate`: `[0.0, 0.02]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Accessibility-Gated Validity Cascade expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but "
            f"n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    cue_offsets = np.asarray(
        parameters["cue_accessibility_offsets"], dtype=np.float64
    )
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got "
            f"shape {validities.shape}."
        )
    if cue_offsets.ndim != 1 or cue_offsets.size != n_features:
        raise ValueError(
            f"cue_accessibility_offsets must have length {n_features}; "
            f"got shape {cue_offsets.shape}."
        )

    baseline = float(parameters["baseline_accessibility"])
    list_gradient = float(parameters["list_length_gradient"])
    position_gradient = float(parameters["position_gradient"])
    late_position_cost = float(parameters["late_position_cost"])
    execution_flip = float(parameters["execution_flip"])
    lapse_rate = float(parameters["lapse_rate"])

    # Absolute zero-based display position. The quadratic component is a
    # restrained terminal-display cost: it has little effect on early cues
    # but makes cues near the end of long lists appreciably harder to encode
    # and retrieve. Cue offsets represent stable identity/complexity effects.
    positions = np.arange(n_features, dtype=np.float64)
    access_logits = (
        baseline
        - list_gradient * max(n_features - 1, 0)
        - position_gradient * positions
        - late_position_cost * positions * positions
        + cue_offsets
    )

    # Numerically stable logistic transformation.
    access_prob = np.empty(n_features, dtype=np.float64)
    positive = access_logits >= 0.0
    access_prob[positive] = 1.0 / (1.0 + np.exp(-access_logits[positive]))
    exp_x = np.exp(access_logits[~positive])
    access_prob[~positive] = exp_x / (1.0 + exp_x)
    access_prob = np.clip(access_prob, 1e-9, 1.0 - 1e-9)

    differences = stim[0] - stim[1]
    discriminating = differences != 0.0
    if not np.any(discriminating):
        return np.array([0.5, 0.5], dtype=np.float64)

    # Advertised validity fixes the search hierarchy. Stable sorting means
    # equal-validity cues retain their displayed order. Accessibility changes
    # whether a cue participates, never its priority relative to another cue.
    cue_order = np.argsort(-validities, kind="stable")

    # Marginalize exactly over independent trial-level accessibility events.
    # reach_probability is the probability that every earlier discriminating
    # cue in the validity hierarchy was inaccessible. Accessible tying cues do
    # not terminate search and therefore leave this quantity unchanged.
    reach_probability = 1.0
    intended_a = 0.0
    intended_b = 0.0

    for j in cue_order:
        if not discriminating[j]:
            continue

        stop_probability = reach_probability * access_prob[j]
        if differences[j] > 0.0:
            intended_a += stop_probability
        else:
            intended_b += stop_probability

        reach_probability *= 1.0 - access_prob[j]

    # If every discriminating cue is inaccessible, no evidence is available.
    intended_a += 0.5 * reach_probability
    intended_b += 0.5 * reach_probability

    intended_total = intended_a + intended_b
    if not np.isfinite(intended_total) or intended_total <= 0.0:
        intended_a = 0.5
    else:
        intended_a /= intended_total

    # Execution reversal is separate from retrieval failure. A final lapse
    # replaces the executed response with a random choice.
    executed_a = (
        intended_a * (1.0 - execution_flip)
        + (1.0 - intended_a) * execution_flip
    )
    p_a = (1.0 - lapse_rate) * executed_a + lapse_rate * 0.5
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
