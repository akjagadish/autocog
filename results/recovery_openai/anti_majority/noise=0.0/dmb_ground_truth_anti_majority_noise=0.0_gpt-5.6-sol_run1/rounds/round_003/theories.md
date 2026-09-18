# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Competitive Source-Binding Cascade. Communicated validities establish an ordinal, usually lexicographic cue hierarchy, but successful use of a cue requires binding its validity label, display identity, and current recommendation into one retrievable representation. Binding fidelity is determined jointly rather than by independent cue accessibility. In particular, nearby discriminating cues with similar compressed validities and the same recommendation form a competitive cluster. Members of such a cluster mutually inhibit one another and can become collectively difficult to identify, so adding redundant support can reduce rather than increase use of the highest-validity cue. When the leading cue is not retrieved, search falls through the validity hierarchy, often allowing isolated, salient opposing cues to control choice. Validities are represented nonlinearly, with moderately diagnostic labels especially vulnerable to source-binding uncertainty. Display position contributes modest primacy and a terminally localized recency boost but no obligatory late-position cost. Thus, pure terminal position can be beneficial while dense terminal redundancy remains harmful because competition overwhelms recency. In unusually long displays, memory load additionally produces a zero-mean early-to-late binding contrast: early sources receive modest interference while late sources retain greater temporal distinctiveness. Stable individual differences govern validity compression, the location and width of maximal binding uncertainty, competitive interference, serial-position biases, long-display positional contrast, and response reversal.

**Rationale:** This is an isolated positional edit to the accepted iteration-6 model. The uncertainty profile, crowding equation, intermediate-validity competition, and ordinary recency kernel are unchanged, preserving the close Experiment-6 result and strong negative Experiment-8 redundancy gradient. The only new term is a bounded, zero-mean early-to-late binding contrast multiplied by excess display load. Its multiplier is exactly zero for displays of 24 or fewer cues, so it cannot alter Experiments 1–6 or 8; it reaches full strength only in the 32-feature display of Experiment 7. There it modestly reduces early-target binding and improves late-target binding without changing mean support across display positions. Subject-level variation in the new coefficient should also raise Experiment-7 variance while leaving the excessive Experiment-8 variance untouched. This directly addresses the latest critic diagnosis without repeating the rejected global centering and amplification of recency.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `validity_compression`: `[0.85, 1.15]`
  - `baseline_binding`: `[2.4, 3.6]`
  - `binding_uncertainty`: `[2.8, 4.5]`
  - `uncertainty_center`: `[0.52, 0.64]`
  - `uncertainty_width`: `[0.14, 0.22]`
  - `interference_strength`: `[1.6, 2.6]`
  - `intermediate_competition`: `[0.8, 1.5]`
  - `crowding_exponent`: `[2.0, 2.5]`
  - `similarity_bandwidth`: `[0.16, 0.4]`
  - `neighborhood_width`: `[1.5, 3.5]`
  - `primacy_strength`: `[0.0, 0.08]`
  - `recency_strength`: `[1.4, 2.4]`
  - `long_display_position_strength`: `[0.45, 1.05]`
  - `execution_flip`: `[0.08, 0.18]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive Source-Binding Cascade expects shape (2, n_features); got {stim.shape}."
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

    compression = float(parameters["validity_compression"])
    baseline = float(parameters["baseline_binding"])
    binding_uncertainty = float(parameters["binding_uncertainty"])
    uncertainty_center = float(parameters["uncertainty_center"])
    uncertainty_width = float(parameters["uncertainty_width"])
    interference = float(parameters["interference_strength"])
    intermediate_competition = float(parameters["intermediate_competition"])
    crowd_exponent = float(parameters["crowding_exponent"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    neighborhood_width = float(parameters["neighborhood_width"])
    primacy = float(parameters["primacy_strength"])
    recency = float(parameters["recency_strength"])
    long_display_position = float(parameters["long_display_position_strength"])
    execution_flip = float(parameters["execution_flip"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    discriminating = directions != 0.0
    if not np.any(discriminating):
        return np.array([0.5, 0.5], dtype=np.float64)

    # Compress communicated validity onto a bounded subjective diagnosticity
    # scale. Ranking remains based on the communicated values, while distances
    # used in source competition are subjective and compressed.
    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_strength = np.power(objective_strength, compression)

    positions = np.arange(n_features, dtype=np.float64)
    serial_scale = max(0.12 * max(n_features - 1, 1), 1.0)
    primacy_profile = np.exp(-positions / serial_scale)
    recency_profile = np.exp(-(n_features - 1.0 - positions) / serial_scale)

    # An additional zero-mean positional contrast emerges only under memory
    # loads exceeding 24 cues. It is therefore absent in the 18- and 24-cue
    # displays while reaching full strength at 32 cues. Centering preserves
    # average binding support rather than globally increasing accessibility.
    long_display_load = float(np.clip((n_features - 24.0) / 8.0, 0.0, 1.0))
    if n_features > 1:
        long_position_profile = 2.0 * positions / (n_features - 1.0) - 1.0
    else:
        long_position_profile = np.zeros(n_features, dtype=np.float64)

    # Binding uncertainty is concentrated around a subject-specific region of
    # moderately diagnostic labels. This permits the .8177 region to be more
    # confusable than near-equal .55 labels without changing validity rank.
    uncertainty_scale = max(uncertainty_width, 1e-9)
    uncertainty_profile = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_scale
        )
    )

    # Compute display-contingent competitive crowding. Only cues that actually
    # discriminate on this trial enter the competition. Same-direction cues
    # inhibit one another most when they are nearby and have similar compressed
    # validities. Consequently, redundant cues can jointly suppress retrieval.
    crowding = np.zeros(n_features, dtype=np.float64)
    active = np.flatnonzero(discriminating)
    sim_scale = max(similarity_bandwidth, 1e-9)
    pos_scale = max(neighborhood_width, 1e-9)

    for j in active:
        competitors = active[active != j]
        if competitors.size == 0:
            continue
        same_direction = directions[competitors] == directions[j]
        competitors = competitors[same_direction]
        if competitors.size == 0:
            continue

        position_similarity = np.exp(
            -np.abs(positions[competitors] - positions[j]) / pos_scale
        )
        validity_similarity = np.exp(
            -np.abs(
                subjective_strength[competitors] - subjective_strength[j]
            ) / sim_scale
        )
        crowding[j] = float(np.sum(position_similarity * validity_similarity))

    # Moderately diagnostic sources are especially vulnerable when they must
    # be distinguished from a cluster of similar sources. This selectively
    # strengthens competition among grouped 70% cues while allowing a slightly
    # shallower global density penalty for very high-validity terminal clusters.
    competition_cost = (
        interference
        * np.power(crowding, crowd_exponent)
        * (1.0 + intermediate_competition * uncertainty_profile)
    )
    binding_logits = (
        baseline
        - binding_uncertainty * uncertainty_profile
        - competition_cost
        + primacy * primacy_profile
        + recency * recency_profile
        + long_display_position * long_display_load * long_position_profile
    )

    # Stable logistic transformation to cue-binding probabilities.
    binding_prob = np.empty(n_features, dtype=np.float64)
    nonnegative = binding_logits >= 0.0
    binding_prob[nonnegative] = 1.0 / (
        1.0 + np.exp(-binding_logits[nonnegative])
    )
    exp_logits = np.exp(binding_logits[~nonnegative])
    binding_prob[~nonnegative] = exp_logits / (1.0 + exp_logits)
    binding_prob = np.clip(binding_prob, 1e-9, 1.0 - 1e-9)

    # Validity-guided cascade: the first successfully bound discriminating cue
    # controls the intended response. Stable sorting uses display order only to
    # resolve exact validity ties. We marginalize over residual binding outcomes
    # after the display-level competitive costs have been computed.
    cue_order = np.argsort(-validities, kind="stable")
    reach_probability = 1.0
    intended_a = 0.0
    intended_b = 0.0

    for j in cue_order:
        if not discriminating[j]:
            continue
        stop_probability = reach_probability * binding_prob[j]
        if differences[j] > 0.0:
            intended_a += stop_probability
        else:
            intended_b += stop_probability
        reach_probability *= 1.0 - binding_prob[j]

    # If no discriminating source is successfully bound, the person guesses.
    intended_a += 0.5 * reach_probability
    intended_b += 0.5 * reach_probability
    total = intended_a + intended_b
    if not np.isfinite(total) or total <= 0.0:
        intended_a = 0.5
    else:
        intended_a /= total

    # A separate execution process can reverse the intended binary response.
    p_a = (
        intended_a * (1.0 - execution_flip)
        + (1.0 - intended_a) * execution_flip
    )
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
