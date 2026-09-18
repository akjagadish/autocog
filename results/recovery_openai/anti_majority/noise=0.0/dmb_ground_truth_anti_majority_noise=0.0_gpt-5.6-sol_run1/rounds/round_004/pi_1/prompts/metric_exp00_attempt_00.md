# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=32):** [0.55, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.92, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

**Rationale:** The six base conditions hold both the amount and direction of evidence constant while manipulating only the spatial density of a same-direction cue cluster. Expert 17 (index 16) has the uniquely highest validity, 92%, and always favors A. Six 90%-valid experts also favor A, while an isolated 55%-valid expert favors B. Across conditions, the six redundant A cues move symmetrically away from Expert 17: their offsets are ±{1,2,3}, ±{1,3,5}, ±{2,4,6}, ±{2,5,8}, ±{3,7,11}, and ±{5,10,15}. This symmetry approximately balances early-versus-late positional effects in the 32-cue display, and the number, validities, and recommendations of all discriminating cues remain fixed. Take The Best always stops at the 92% cue and therefore predicts one condition-invariant probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 of choosing A. The Competitive Source-Binding Cascade instead predicts a pronounced spacing gradient. In the densest condition, the seven nearby, similarly valid A sources form a mutually inhibiting cluster; even the leading 92% source receives crowding from both sides, permitting the isolated B source to control many choices. As spacing increases, positional similarity and hence crowding decline exponentially, so successful retrieval of at least one A source should increase and A choices should rise, potentially crossing from below to above chance. This is an adversarial redundancy effect: distributing exactly the same supporting evidence improves its use, whereas TTB is indifferent to both redundancy and spacing. Exact A/B reversals balance response-side preferences and should invert the competing theory's spacing profile without changing its magnitude. The 12 unique pairs are repeated eight times, yielding 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Competitive Source-Binding Cascade. Communicated validities establish an ordinal, usually lexicographic cue hierarchy, but successful use of a cue requires binding its validity label, display identity, and current recommendation into one retrievable representation. Binding fidelity is determined jointly rather than by independent cue accessibility. In particular, nearby discriminating cues with similar compressed validities and the same recommendation form a competitive cluster. Members of such a cluster mutually inhibit one another and can become collectively difficult to identify, so adding redundant support can reduce rather than increase use of the highest-validity cue. When the leading cue is not retrieved, search falls through the validity hierarchy, often allowing isolated, salient opposing cues to control choice. Validities are represented nonlinearly, with moderately diagnostic labels especially vulnerable to source-binding uncertainty. Display position contributes modest primacy and a terminally localized recency boost but no obligatory late-position cost. Thus, pure terminal position can be beneficial while dense terminal redundancy remains harmful because competition overwhelms recency. In unusually long displays, memory load additionally produces a zero-mean early-to-late binding contrast: early sources receive modest interference while late sources retain greater temporal distinctiveness. Stable individual differences govern validity compression, the location and width of maximal binding uncertainty, competitive interference, serial-position biases, long-display positional contrast, and response reversal.

**Parameters:**
- n_features: n_features
- validities: validities
- validity_compression: [0.85, 1.15]
- baseline_binding: [2.4, 3.6]
- binding_uncertainty: [2.8, 4.5]
- uncertainty_center: [0.52, 0.64]
- uncertainty_width: [0.14, 0.22]
- interference_strength: [1.6, 2.6]
- intermediate_competition: [0.8, 1.5]
- crowding_exponent: [2.0, 2.5]
- similarity_bandwidth: [0.16, 0.4]
- neighborhood_width: [1.5, 3.5]
- primacy_strength: [0.0, 0.08]
- recency_strength: [1.4, 2.4]
- long_display_position_strength: [0.45, 1.05]
- execution_flip: [0.08, 0.18]

**`predict source code`:**
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

**`policy source code`:**
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


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Choice congruence with the uniquely most valid cue (Expert 17, index 16),\n    # contrasted between the two sparsest and four densest cue arrangements.\n    congruent_sparse = []\n    congruent_dense = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        diff = a - b\n\n        if diff.size <= 16 or diff[16] == 0:\n            continue\n\n        leading_direction = np.sign(diff[16])\n        leading_response = 0 if leading_direction > 0 else 1\n        congruent = float(int(row[\"response\"]) == leading_response)\n\n        # Recover the arrangement from the six other cues supporting the same\n        # option as Expert 17. Their mean distance from index 16 is exactly one\n        # of 2, 3, 4, 5, 7, or 10 in this design.\n        indices = np.arange(diff.size)\n        supporters = indices[(np.sign(diff) == leading_direction) & (indices != 16)]\n        if supporters.size == 0:\n            continue\n        spread = float(np.mean(np.abs(supporters - 16)))\n\n        if spread > 6.0:\n            congruent_sparse.append(congruent)\n        else:\n            congruent_dense.append(congruent)\n\n    if len(congruent_sparse) == 0 or len(congruent_dense) == 0:\n        return 0.0\n\n    return float(np.mean(congruent_sparse) - np.mean(congruent_dense))",
  "rationale": "This is the within-subject increase in following the uniquely highest-validity cue when its six redundant supporters are sparse (mean offsets 7 or 10) rather than dense/intermediate (offsets 2\u20135). Take The Best predicts the same probability of following Expert 17 in every arrangement, so its expected contrast is zero even when beta, lapse rate, or overall response accuracy differ across subjects. The competing cascade predicts substantially greater congruence in sparse displays because reduced source competition restores retrieval of Expert 17 or another same-direction high-validity cue. Pooling 32 sparse and 64 dense observations per subject yields a stable estimate, while coding choices relative to Expert 17 and including exact A/B reversals cancels fixed response-side preferences."
}
```

## Usage

```json
{
  "input_tokens": 7802,
  "output_tokens": 2259
}
```
