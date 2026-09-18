# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

**Rationale:** This is a minimal extension of the accepted recency-overwrite model. The cue scan and overwrite equations are unchanged. The only mechanistic addition is a discriminating-cue-count fatigue gate that begins after the second reason, together with a broader and slightly lower lapse range. Two-reason trials therefore retain the strong overwrite needed for Experiment 2, whereas long reason sequences are pulled modestly toward uncertainty, reducing the excessive Experiment-1 contrast. Lower average lapse simultaneously strengthens anti-TTB behavior in Experiment 2. Broad subject-level ranges for fatigue and lapse should also increase the previously underestimated between-subject variability. The gate is task-invariant and depends only on processing load, not experiment identity or stimulus-specific switches.

**Parameters:**
  - `validities`: `validities`
  - `cue_attention`: `[0.95, 1.0]`
  - `overwrite_rate`: `[0.95, 1.0]`
  - `update_fatigue`: `[0.06, 0.17]`
  - `lapse_rate`: `[0.15, 0.41]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recency-Gated Reason Overwrite expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["cue_attention"])
    overwrite = float(parameters["overwrite_rate"])
    fatigue = float(parameters["update_fatigue"])
    lapse = float(parameters["lapse_rate"])

    # Stable sorting represents the displayed high-to-low validity scan;
    # ties retain their display order.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]

    # Exact marginal distribution over the provisional state:
    # A selected, B selected, or no discriminating cue encoded yet.
    p_a, p_b, p_uncommitted = 0.0, 0.0, 1.0
    discriminating_count = 0

    for j in cue_order:
        direction = diff[j]
        if direction == 0:
            continue

        discriminating_count += 1
        old_a, old_b, old_u = p_a, p_b, p_uncommitted

        if direction > 0:
            # An attended A cue initializes an uncommitted decision and
            # overwrites a conflicting B decision with the overwrite rate.
            p_a = old_a + attention * overwrite * old_b + attention * old_u
            p_b = old_b * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)
        else:
            p_b = old_b + attention * overwrite * old_a + attention * old_u
            p_a = old_a * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)

    # If no discriminating cue was encoded, either response is an equal guess.
    p_core = np.array([
        p_a + 0.5 * p_uncommitted,
        p_b + 0.5 * p_uncommitted
    ], dtype=np.float64)

    # Processing the first reason and its immediate successor is reliable.
    # Additional discriminating updates progressively weaken expression of
    # the retained choice without disrupting the crucial second-cue reversal.
    excess_updates = max(0, discriminating_count - 2)
    retention = (1.0 - fatigue) ** excess_updates
    p_core = retention * p_core + (1.0 - retention) * np.array([0.5, 0.5])

    # Independent execution/inattention lapse.
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
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

**Description:** Interference-Limited Retrospective Reason Sampling with Length-Calibrated Accessibility. After viewing all cues, a decision maker retrospectively retrieves one or, subject to stable working-memory capacity, occasionally two discriminating reasons. In two-reason conflicts, retrieval has a strong saturated terminal-recency floor, producing reliable last-reason capture without importing subject heterogeneity from longer conflicts. Around four reasons, stable variation in recency remains expressed. When conflicts exceed four reasons, nonlinear source interference flattens recency while accessibility approaches a validity-biased rather than uniform asymptote. Choice follows the signed, validity-sensitive evidence supplied by the retrieved reasons, with a small response lapse.

**Rationale:** This is a minimal accessibility edit to the accepted model. First, two-cue conflicts receive a strong but saturated recency gradient. This should lower Experiment 2 first-cue agreement and its between-subject variance without narrowing the broad recency distribution responsible for Experiment 4's accurately reproduced heterogeneity. Second, overload now selectively increases validity-based accessibility while cubic interference continues to suppress temporal recency. Long conflicts therefore approach validity-biased retrieval rather than uniform, tally-like sampling, which should reduce Experiment 1's excessive TTB–tally contrast. The capacity limit, four-cue recency regime, evidence mapping, continuation mechanism, lapse, and all parameter ranges are otherwise unchanged.

**Parameters:**
  - `validities`: `validities`
  - `memory_capacity`: `{1, 2}`
  - `recency_bias`: `[0.2, 4.6]`
  - `validity_access`: `[0.05, 0.35]`
  - `interference_rate`: `[0.6, 1.4]`
  - `second_reason_rate`: `[0.02, 0.12]`
  - `evidence_floor`: `[0.35, 0.75]`
  - `evidence_gain`: `[3.0, 7.0]`
  - `lapse_rate`: `[0.03, 0.17]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Retrospective reason sampling expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = int(parameters["memory_capacity"])
    recency = float(parameters["recency_bias"])
    validity_access = float(parameters["validity_access"])
    interference = float(parameters["interference_rate"])
    continuation = float(parameters["second_reason_rate"])
    evidence_floor = float(parameters["evidence_floor"])
    evidence_gain = float(parameters["evidence_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_indices = np.flatnonzero(diff != 0)
    m = int(cue_indices.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Reliability is represented as instructed log-odds. Clipping only
    # prevents infinite evidence for a validity communicated as exactly 1.
    v = np.clip(validities[cue_indices], 0.500001, 0.999)
    reliability = np.log(v / (1.0 - v))
    directions = np.sign(diff[cue_indices])  # +1 favors A, -1 favors B

    # Cues are displayed from highest to lowest validity. Ordinal position
    # among discriminating cues therefore indexes retrospective recency.
    ordinal_recency = np.arange(m, dtype=float)

    # Interference is negligible through four reasons but grows cubically
    # thereafter, rapidly flattening accessibility in long conflicts.
    overload = float(max(0, m - 4))
    effective_recency = recency / (1.0 + interference * overload * overload * overload)

    # A two-reason conflict is represented as a compact terminal episode:
    # its recency gradient is strong and saturates across subjects. Stable
    # recency differences remain fully expressed in four-reason conflicts.
    if m == 2:
        effective_recency = 3.4 + 0.15 * np.tanh(recency - 2.0)

    # Under overload, interference removes temporal-source information but
    # does not make all reasons equally accessible. Retrieval instead tends
    # toward a modestly validity-biased asymptote.
    effective_validity_access = (
        validity_access + 0.55 * overload / (1.0 + overload)
    )

    log_access = (
        effective_recency * ordinal_recency
        + effective_validity_access * reliability
    )
    log_access -= np.max(log_access)
    access = np.exp(log_access)
    first_probs = access / access.sum()

    def probability_a(retrieved_positions):
        # Every retrieved reason supplies substantial directional evidence;
        # instructed validity additionally scales that evidence and resolves
        # conflicts when more than one reason is retrieved.
        positions = list(retrieved_positions)
        evidence = float(np.sum(
            directions[positions]
            * (evidence_floor + reliability[positions])
        ))
        x = float(np.clip(evidence_gain * evidence, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Exact marginalization over the first retrospectively retrieved reason.
    p_a_one = 0.0
    for i in range(m):
        p_a_one += float(first_probs[i]) * probability_a((i,))

    # Capacity is an upper bound: even capacity-two subjects often terminate
    # retrieval after one sufficient reason. If retrieval continues, the
    # second reason is sampled without replacement from the same accessibility
    # distribution and both reasons are integrated.
    if capacity >= 2 and m >= 2 and continuation > 0.0:
        p_a_two = 0.0
        for i in range(m):
            remaining_mass = float(access.sum() - access[i])
            if remaining_mass <= 0.0:
                continue
            for j in range(m):
                if j == i:
                    continue
                p_ordered = float(first_probs[i]) * float(access[j] / remaining_mass)
                p_a_two += p_ordered * probability_a((i, j))
        p_a_core = (1.0 - continuation) * p_a_one + continuation * p_a_two
    else:
        p_a_core = p_a_one

    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
