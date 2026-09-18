# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_1` — KILLED ✗

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


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** Accessibility-Gated Stochastic Lexicographic Search: People maintain a stable attentional priority over cues based primarily on visual position and learned accessibility. Communicated validities modify that priority only when the validity-to-position mapping is simple enough to encode and maintain. Mapping simplicity is assessed from how closely validity rank follows display order, while a subject-specific mapping-fidelity parameter limits the influence of this information. On each choice, attention follows a noisy priority ordering and search stops at the first cue that discriminates. Thus, stable priorities guide decisions without imposing an identical deterministic cue order on every trial. When validity and position are aligned, validity information gives the accessible first cue a sufficiently large advantage to produce nearly invariant one-cue choices. Under a nonmonotonic mapping, weak validity gating and modest attentional fluctuations distribute search across accessible cues, producing systematic choices that are approximately orthogonal to validity-ranked search without requiring cancellation between strongly polarized subjects. A small lapse probability captures response errors.

**Rationale:** This is a minimal in-family edit to the accepted candidate. The mapping-simplicity gate, priority equation, validity contribution, positional contribution, and lapse process are retained. Only the deterministic sorting block is replaced by the marginal choice probabilities implied by a Plackett-Luce/Gumbel cue ordering: among currently discriminating cues, softmax priority determines which one is encountered first. Accessibility-weight dispersion is also narrowed from [0,1] to [0.48,0.52]. These changes directly address the excessive Experiment 2 between-subject variance by preventing each subject's arbitrary accessibility draw from fixing a highly polarized cue order. They do not turn the model into globally random responding: visual position and stable latent priorities continue to structure each choice. In Experiment 1, the validity-position alignment gives cue 1 a large priority advantage, so stochastic search remains nearly lexicographic; the selected temperature permits only a small increase in lower-cue influence as conflict grows, moving the contrast toward the observed small positive value. In Experiment 2, the weak mapping gate and balanced mirrored schedule make stochastic accessibility-based search approximately orthogonal to the validity-ranked winner within subjects, reducing reliance on cross-subject cancellation.

**Parameters:**
  - `validities`: `validities`
  - `accessibility_weights`: `[(0.48, 0.52)] * n_features`
  - `position_strength`: `[0.02, 0.15]`
  - `validity_strength`: `[3.0, 5.0]`
  - `mapping_fidelity`: `[0.8, 1.0]`
  - `complexity_power`: `[3.0, 8.0]`
  - `attention_temperature`: `[0.30, 0.40]`
  - `epsilon`: `[0.02, 0.15]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Accessibility-Gated Stochastic Lexicographic Search.
    # The subject has a stable, subject-specific cue priority. Validity
    # affects this priority strongly only when validity rank is easy to map
    # onto visual position. A Plackett-Luce attention process samples the
    # cue order, and search stops at the first discriminating cue.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(parameters["accessibility_weights"], dtype=float)

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != n_features {n_features}."
        )

    if n_features == 1:
        position = np.zeros(1, dtype=float)
        validity_rank_score = np.ones(1, dtype=float)
        mapping_simplicity = 1.0
    else:
        # Earlier positions have greater default visual accessibility.
        position = np.arange(n_features, dtype=float) / float(n_features - 1)

        # Bounded validity-rank representation: 1 for the most valid cue and
        # 0 for the least valid. Stable sorting makes ties favor earlier cues.
        validity_order = np.argsort(-validities, kind="stable")
        ranks = np.empty(n_features, dtype=float)
        ranks[validity_order] = np.arange(n_features, dtype=float)
        validity_rank_score = 1.0 - ranks / float(n_features - 1)

        # A descending validity sequence is easy to map onto display position.
        # Compute rank-position coherence directly with NumPy so prediction
        # does not depend on an external scipy namespace being initialized.
        x = position - np.mean(position)
        y = validity_rank_score - np.mean(validity_rank_score)
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if not np.isfinite(denom) or denom <= 0.0:
            mapping_simplicity = 0.0
        else:
            rho = float(np.sum(x * y) / denom)
            mapping_simplicity = float(np.clip(-rho, 0.0, 1.0))

    complexity_power = float(parameters["complexity_power"])
    mapping_fidelity = float(parameters["mapping_fidelity"])
    mapping_gate = mapping_fidelity * (mapping_simplicity ** complexity_power)
    mapping_gate = float(np.clip(mapping_gate, 0.0, 1.0))

    position_strength = float(parameters["position_strength"])
    validity_strength = float(parameters["validity_strength"])

    # When the mapping is easy, communicated validity reshapes accessibility.
    # When it is difficult, stable learned accessibility dominates instead.
    learned_component = (1.0 - mapping_gate) * accessibility
    positional_component = position_strength * (1.0 - position)
    validity_component = validity_strength * mapping_gate * validity_rank_score
    priority = learned_component + positional_component + validity_component

    a, b = stim[0], stim[1]
    discriminating = np.flatnonzero(a != b)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Under a Plackett-Luce/Gumbel ordering, the probability that a cue is the
    # first discriminating cue is its softmax weight among discriminating cues.
    attention_temperature = float(parameters["attention_temperature"])
    logits = priority[discriminating] / attention_temperature
    logits = logits - np.max(logits)
    cue_probs = np.exp(logits)
    cue_probs /= cue_probs.sum()

    p_core = np.zeros(2, dtype=np.float64)
    for j, cue_prob in zip(discriminating, cue_probs):
        if a[j] > b[j]:
            p_core[0] += cue_prob
        else:
            p_core[1] += cue_prob

    epsilon = float(parameters["epsilon"])
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_4` → slot 1 (via `new_theory`)

**Description:** Accessibility-Anchored Fixed Lexicographic Search. Each person constructs one stable cue hierarchy at the beginning of a task and reuses it across trials. Priority is dominated by display position, with earlier cues receiving a large accessibility advantage. Subject-specific long-run accessibility or salience offsets can alter priorities locally, while communicated validity provides only a weak additive adjustment. The strength of that adjustment is expressed as a narrowly distributed ratio to positional anchoring, preventing independent parameter draws from causing unstable population-wide cue-order reversals. Thus validity can resolve competition between nearby positions but cannot promote a late cue merely because it has the highest instructed validity. Choice follows the first discriminating cue in this fixed hierarchy. Rare search slips skip that cue and consult the next discriminating cue, while bounded response imprecision and a small lapse rate generate occasional choices against the selected cue. The hierarchy is not resampled trial by trial and does not depend on the number of conflicting later cues.

**Rationale:** This is a minimal edit to the accepted iteration-1 model. The independent validity-weight parameter is replaced by a narrowly bounded validity-to-position ratio, with validity_weight computed as validity_position_ratio times position_strength. The ratio is only slightly above the accepted model's central ratio, implementing the critic's requested small interpolation toward iteration 2 rather than repeating iteration 2's larger threshold-crossing adjustment. Correlating the two strengths prevents independent draws from producing extreme ratios while retaining the accepted accessibility dispersion and all choice-process parameters. The modest increase should allow the exceptionally valid second-position cue to compete somewhat more often in Experiment 2, reducing excessive negative validity agreement. Because positional costs still accumulate linearly, the adjustment remains too weak to elevate cue 8 over several early cues in Experiment 3. The unchanged fixed hierarchy, rare skip process, response precision, and lapse preserve the near-zero conflict-count and rank-distance effects in Experiments 1 and 4.

**Parameters:**
  - `validities`: `validities`
  - `accessibility_weights`: `[(0.0, 0.03)] * n_features`
  - `position_strength`: `[0.12, 0.16]`
  - `validity_position_ratio`: `[3.80, 3.90]`
  - `order_perturbation`: `[0.0, 0.03]`
  - `response_precision`: `[1.45, 1.85]`
  - `epsilon`: `[0.01, 0.04]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Accessibility-Anchored Fixed Lexicographic Search.
    # Parameters are fixed for the subject, so the resulting hierarchy is
    # stable across all trials. History is intentionally ignored because no
    # outcome feedback is available to support cue-order learning.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(
        parameters["accessibility_weights"], dtype=float
    )

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != "
            f"n_features {n_features}."
        )

    # Position imposes a linear cost for moving attention to later cues.
    # Raw validity differences are deliberately not converted to ranks:
    # communicated validity is only a weak local adjustment and therefore
    # cannot receive an extra boost when its ordering matches the display.
    position_strength = float(parameters["position_strength"])
    validity_position_ratio = float(parameters["validity_position_ratio"])
    validity_weight = validity_position_ratio * position_strength
    positions = np.arange(n_features, dtype=float)
    centered_validity = validities - float(np.mean(validities))

    priority = (
        -position_strength * positions
        + accessibility
        + validity_weight * centered_validity
    )

    # Stable sorting supplies a deterministic tie rule favoring earlier
    # positions. Since all components are subject-level constants, this order
    # is fixed rather than redrawn on every trial.
    cue_order = np.argsort(-priority, kind="stable")

    a, b = stim[0], stim[1]
    discriminating_order = [int(j) for j in cue_order if a[j] != b[j]]
    if len(discriminating_order) == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def cue_winner(j):
        return 0 if a[j] > b[j] else 1

    first_winner = cue_winner(discriminating_order[0])

    # A rare search-control slip skips the first discriminating cue. This is
    # a tightly constrained perturbation of a fixed hierarchy, not a newly
    # sampled Plackett-Luce ordering.
    order_perturbation = float(parameters["order_perturbation"])
    order_perturbation = float(np.clip(order_perturbation, 0.0, 1.0))
    if len(discriminating_order) >= 2:
        second_winner = cue_winner(discriminating_order[1])
    else:
        second_winner = first_winner

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - order_perturbation
    p_selected[second_winner] += order_perturbation

    # Conditional on the selected cue, response precision controls accidental
    # reversal of its recommendation. The stable logistic form avoids an
    # unconstrained trial-wise attentional softmax.
    response_precision = float(parameters["response_precision"])
    if response_precision >= 0.0:
        response_reliability = 1.0 / (1.0 + np.exp(-response_precision))
    else:
        e = np.exp(response_precision)
        response_reliability = e / (1.0 + e)

    p_core = (
        response_reliability * p_selected
        + (1.0 - response_reliability) * p_selected[::-1]
    )

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
