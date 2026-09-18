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

**Description:** Conflict-Gated Redundancy-Discounted Attention (CGRDA) proposes that people integrate all discriminating cues, but cue influence depends jointly on instructed validity, serial position, directional redundancy, and the current conflict configuration. Instructed validity and early presentation increase a cue's baseline attentional weight. Successive endorsements of an option have diminishing marginal impact because repeated evidence is treated as partially redundant. When the numbers of cues favoring the two options are closely balanced, attention is reoriented toward the final discriminating cue, producing a conflict-gated recency contribution. Thus, an early compact set of reliable cues can defeat a larger but redundant set, while the last cue can reverse the decision in balanced conflicts. Primacy, redundancy discounting, recency strength, and conflict sensitivity are stable subject-level traits.

**Rationale:** Experiment 1 contains imbalanced conflicts in which the first-discriminating side is represented by earlier and more valid cues but often loses the raw tally. Primacy and instructed-validity weighting favor that compact early evidence, while redundancy discounting prevents five or seven same-direction endorsements from contributing as five or seven independent observations. Because the tallies are imbalanced, the conflict gate largely suppresses recency, preserving above-chance first-cue agreement. Experiment 2 instead consists of balanced conflicts whose final discriminating cue opposes the highest-validity discriminating cue. Exact balance fully opens the recency gate, allowing endpoint evidence to overturn the initial validity-weighted impression and generate below-chance first-cue alignment. Continuous subject-level variation in primacy, recency, validity sensitivity, and redundancy discounting can produce the larger observed between-subject variance in Experiment 2 without requiring trial-by-trial feedback. Unlike Tallying, the model's cue contributions are context dependent; unlike Take The Best, every discriminating cue contributes. It also makes trial-level predictions under independent manipulations of tally imbalance, repeated endorsements, cue validity, and order. In particular, reversing display order changes the recency term while leaving validity weights attached to cue identities, separating order-based conflict resolution from genuine validity use.

**Parameters:**
  - `validities`: `validities`
  - `primacy`: `[0.8, 1.4]`
  - `redundancy_discount`: `[0.25, 1.25]`
  - `validity_sensitivity`: `[0.3, 1.5]`
  - `recency_strength`: `[1.15, 1.65]`
  - `balance_sensitivity`: `[2.5, 5.0]`
  - `beta`: `[0.8, 1.8]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CGRDA expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive differences favor B and negative differences favor A.
    diff = stim[1] - stim[0]
    signs = np.sign(diff)
    discriminating = np.flatnonzero(signs != 0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    primacy = float(parameters["primacy"])
    redundancy_discount = float(parameters["redundancy_discount"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    recency_strength = float(parameters["recency_strength"])
    balance_sensitivity = float(parameters["balance_sensitivity"])
    beta = float(parameters["beta"])

    # Convert communicated validities to reliability evidence. Normalizing
    # preserves validity rank without making the score depend arbitrarily
    # on the absolute log-odds scale selected by an experiment.
    clipped = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(clipped / (1.0 - clipped))
    reliability /= max(float(np.max(reliability)), 1e-12)
    reliability = np.maximum(reliability, 0.05)

    count_a = 0
    count_b = 0
    integrated = 0.0
    total_weight = 0.0

    for j in discriminating:
        direction = float(signs[j])
        if direction > 0:
            count_b += 1
            repetition = count_b
        else:
            count_a += 1
            repetition = count_a

        validity_weight = reliability[j] ** validity_sensitivity
        position_weight = np.exp(-primacy * float(j))
        novelty_weight = float(repetition) ** (-redundancy_discount)
        weight = validity_weight * position_weight * novelty_weight

        integrated += direction * weight
        total_weight += weight

    # Put integrated evidence on a common bounded scale across feature
    # counts and experiments.
    core_evidence = integrated / max(total_weight, 1e-12)

    # Recency is strongest at exact tally balance and falls rapidly as one
    # side acquires more endorsements. The final discriminating cue, not
    # necessarily the least-valid cue, determines the direction of this
    # conflict-resolution contribution.
    tally_imbalance = abs(count_b - count_a)
    conflict_gate = np.exp(-balance_sensitivity * float(tally_imbalance))
    last_direction = float(signs[int(discriminating[-1])])
    recency_evidence = recency_strength * conflict_gate * last_direction

    choice_evidence = core_evidence + recency_evidence
    utilities = np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])

    x = beta * utilities
    x = x - np.max(x)
    probs = np.exp(x)
    probs /= probs.sum()
    return probs.astype(np.float64)
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


## Replacement

### `pi_4` → slot 2 (via `new_theory`)

**Description:** Dominance-Calibrated Saturating Coalitions (DCSC) proposes that people classify the communicated validity profile before evaluating individual options. A cue activates a dominance regime when it accounts for a sufficiently large share of total diagnosticity and the profile as a whole is concentrated rather than broadly distributed across similarly informative cues. In that regime, the highest-validity discriminating cue anchors the decision, while the remaining cues provide only a small, order-invariant adjustment. Without sufficient profile-level dominance, people enter a compensatory regime and accumulate signed reliability evidence from every discriminating cue. Reliability differences are compressed, and repeated support for the same option has diminishing marginal impact, representing perceived redundancy among concordant experts. Thus, several lower-validity cues can collectively defeat the best discriminating cue, but neither early nor late serial position is intrinsically favored. Evidential balance can induce additional deliberation, modeled as a modest increase in choice consistency rather than a directional recency rule. Stable individual differences concern the dominance threshold, sensitivity to profile concentration, compression and saturation of reliability evidence, leakage of coalition evidence into dominance decisions, deliberation under conflict, and response sensitivity.

**Rationale:** This is a minimal extension of the accepted DCSC gate. The strongest-cue share is retained, but a normalized Herfindahl concentration term is added so profiles with similar top shares are no longer treated as equivalent. A profile with one dominant cue and a generally weak remainder receives a high dominance score, whereas a profile retaining substantial distributed lower-cue diagnosticity remains compensatory. The revised threshold is calibrated to the expanded score rather than increasing dominance globally. This should preserve Experiment 1's stable dominance while moving Experiment 2 into the existing compensatory branch, whose compressed, weakly saturated coalition evidence permits lower cues to overturn the highest discriminating cue. All order-invariant evidence, anchoring, leakage, saturation, conflict-deliberation, and response equations are otherwise unchanged, protecting the exact absence of an Experiment-4 positional crossover. Moving diffuse profiles into compensation also gives the existing balance-dependent deliberation mechanism behavioral leverage in Experiment 3 without adding recency.

**Parameters:**
  - `validities`: `validities`
  - `dominance_threshold`: `[0.285, 0.305]`
  - `gap_sensitivity`: `[70.0, 110.0]`
  - `profile_shape_weight`: `[1.8, 2.2]`
  - `validity_compression`: `[0.02, 0.18]`
  - `redundancy_saturation`: `[0.08, 0.25]`
  - `anchor_strength`: `[0.8, 1.2]`
  - `dominance_leak`: `[0.02, 0.10]`
  - `conflict_deliberation`: `[0.30, 0.60]`
  - `beta`: `[2.8, 4.8]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"DCSC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive signs favor B and negative signs favor A. Presentation
    # position never enters the evidence weights.
    signs = np.sign(stim[1] - stim[0])
    discriminating = np.flatnonzero(signs != 0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    dominance_threshold = float(parameters["dominance_threshold"])
    gap_sensitivity = float(parameters["gap_sensitivity"])
    profile_shape_weight = float(parameters["profile_shape_weight"])
    validity_compression = float(parameters["validity_compression"])
    redundancy_saturation = float(parameters["redundancy_saturation"])
    anchor_strength = float(parameters["anchor_strength"])
    dominance_leak = float(parameters["dominance_leak"])
    conflict_deliberation = float(parameters["conflict_deliberation"])
    beta = float(parameters["beta"])

    # Communicated validities are converted to nonnegative diagnostic
    # strengths. The small exponent compresses differences in the
    # compensatory regime without reversing the communicated ordering.
    diagnosticity = np.clip(2.0 * validities - 1.0, 1e-6, 1.0)
    cue_weights = diagnosticity ** validity_compression

    # Dominance depends both on the strongest cue's diagnosticity share
    # and on concentration across the complete profile. The normalized
    # Herfindahl term distinguishes a genuinely concentrated profile from
    # one in which the strongest cue has a similar share but substantial
    # diagnosticity remains broadly distributed among lower cues.
    profile_shares = diagnosticity / max(float(np.sum(diagnosticity)), 1e-12)
    dominance_concentration = float(np.max(profile_shares))
    profile_hhi = float(np.sum(profile_shares ** 2))
    uniform_hhi = 1.0 / float(n_features)
    shape_concentration = (profile_hhi - uniform_hhi) / max(
        1.0 - uniform_hhi, 1e-12
    )
    shape_concentration = float(np.clip(shape_concentration, 0.0, 1.0))
    dominance_score = (
        dominance_concentration
        + profile_shape_weight * shape_concentration
    )
    gate_logit = gap_sensitivity * (
        dominance_score - dominance_threshold
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    dominance_gate = 1.0 / (1.0 + np.exp(-gate_logit))

    # Aggregate reliability separately for the two coalitions. The
    # exponential transform gives diminishing marginal impact to repeated
    # concordant evidence and is invariant to the order in which cues are
    # displayed.
    support_b = float(np.sum(cue_weights[(signs > 0)]))
    support_a = float(np.sum(cue_weights[(signs < 0)]))
    rho = max(redundancy_saturation, 1e-8)
    bounded_b = -np.expm1(-rho * support_b) / rho
    bounded_a = -np.expm1(-rho * support_a) / rho

    # Express coalition evidence in approximately cue-vote units so the
    # response scale remains comparable across validity profiles.
    mean_discriminating_weight = float(np.mean(cue_weights[discriminating]))
    coalition_evidence = (bounded_b - bounded_a) / max(
        mean_discriminating_weight, 1e-12
    )

    # The anchor is the most valid cue that actually discriminates. Ties
    # in communicated validity are resolved without using serial position:
    # their signed evidence is averaged, so tied opposing anchors cancel.
    disc_weights = diagnosticity[discriminating]
    best_weight = float(np.max(disc_weights))
    best_mask = np.isclose(disc_weights, best_weight, rtol=1e-10, atol=1e-12)
    anchor_direction = float(np.mean(signs[discriminating][best_mask]))
    anchor_evidence = anchor_strength * anchor_direction

    # In dominance mode coalition evidence only leaks weakly into the
    # anchored judgment. In integration mode it is fully compensatory.
    dominance_evidence = anchor_evidence + dominance_leak * coalition_evidence
    choice_evidence = (
        dominance_gate * dominance_evidence
        + (1.0 - dominance_gate) * coalition_evidence
    )

    # Balanced conflict can prompt more careful implementation of the
    # selected rule, but it supplies no directional or last-cue evidence.
    count_b = int(np.sum(signs > 0))
    count_a = int(np.sum(signs < 0))
    balance = np.exp(-abs(count_b - count_a))
    beta_eff = beta * (1.0 + conflict_deliberation * balance)

    utilities = np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])
    logits = beta_eff * utilities
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs.astype(np.float64)
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
