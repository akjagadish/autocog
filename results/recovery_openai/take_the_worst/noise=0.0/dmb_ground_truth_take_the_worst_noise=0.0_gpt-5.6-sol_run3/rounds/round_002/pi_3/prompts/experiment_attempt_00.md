# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Accessibility-Gated Stochastic Lexicographic Search: People maintain a stable attentional priority over cues based primarily on visual position and learned accessibility. Communicated validities modify that priority only when the validity-to-position mapping is simple enough to encode and maintain. Mapping simplicity is assessed from how closely validity rank follows display order, while a subject-specific mapping-fidelity parameter limits the influence of this information. On each choice, attention follows a noisy priority ordering and search stops at the first cue that discriminates. Thus, stable priorities guide decisions without imposing an identical deterministic cue order on every trial. When validity and position are aligned, validity information gives the accessible first cue a sufficiently large advantage to produce nearly invariant one-cue choices. Under a nonmonotonic mapping, weak validity gating and modest attentional fluctuations distribute search across accessible cues, producing systematic choices that are approximately orthogonal to validity-ranked search without requiring cancellation between strongly polarized subjects. A small lapse probability captures response errors.

**Parameters:**
- validities: validities
- accessibility_weights: [(0.48, 0.52)] * n_features
- position_strength: [0.02, 0.15]
- validity_strength: [3.0, 5.0]
- mapping_fidelity: [0.8, 1.0]
- complexity_power: [3.0, 8.0]
- attention_temperature: [0.30, 0.40]
- epsilon: [0.02, 0.15]

**`predict source code`:**
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

**`policy source code`:**
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


## COMPETING THEORY
**Description:** Accessibility-Anchored Fixed Lexicographic Search. Each person constructs one stable cue hierarchy at the beginning of a task and reuses it across trials. Priority is dominated by display position, with earlier cues receiving a large accessibility advantage. Subject-specific long-run accessibility or salience offsets can alter priorities locally, while communicated validity provides only a weak additive adjustment. The strength of that adjustment is expressed as a narrowly distributed ratio to positional anchoring, preventing independent parameter draws from causing unstable population-wide cue-order reversals. Thus validity can resolve competition between nearby positions but cannot promote a late cue merely because it has the highest instructed validity. Choice follows the first discriminating cue in this fixed hierarchy. Rare search slips skip that cue and consult the next discriminating cue, while bounded response imprecision and a small lapse rate generate occasional choices against the selected cue. The hierarchy is not resampled trial by trial and does not depend on the number of conflicting later cues.

**Parameters:**
- validities: validities
- accessibility_weights: [(0.0, 0.03)] * n_features
- position_strength: [0.12, 0.16]
- validity_position_ratio: [3.80, 3.90]
- order_perturbation: [0.0, 0.03]
- response_precision: [1.45, 1.85]
- epsilon: [0.01, 0.04]

**`predict source code`:**
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

**`policy source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The highest-validity expert (feature 1, validity .95) always favors one product, while q = 1 through 6 lower-validity experts favor the other product; all remaining features tie. Each configuration is paired with its A/B mirror to control response and display-side biases. Take The Best therefore follows feature 1 on every trial and predicts the same choice probability at every value of q: apart from orientation, its probability is constant at (1-epsilon)*sigmoid(beta)+epsilon/2. Tallying predicts an exact 50/50 choice when q=1 (one cue win per option), then increasingly strong preference for the lower-cue majority as q rises, with tally margins q-1 and probabilities governed by sigmoid(beta*(q-1)). Thus the key quantitative signature is a flat, high-validity-cue preference under TTB versus a direction reversal and monotonic evidence-strength gradient under Tallying. The 12 unique mirrored pairs repeat eight times each, yielding 96 trials.

[1] This balanced-evidence design targets a parameter-robust dissociation. Every option pair contains exactly three cues favoring A, three favoring B, and two ties. Tallying therefore predicts P(A)=0.5 on every trial, regardless of beta or epsilon. In each of the first six pairs, however, the highest-validity cue that discriminates favors A; in the corresponding A/B mirror it favors B. Take The Best consequently predicts its ranked-cue winner with probability (1-epsilon)*sigmoid(beta)+epsilon/2. The first discriminating cue occurs at validity rank 1, 2, or 3 across conditions, while the tally remains fixed at 3-3. Thus systematic choice of the validity-ranked winner, invariant across these ranks, supports Take The Best, whereas chance-level aggregate and within-pair choices support Tallying. Validity rank is deliberately nonaligned with display position, reducing the possibility that a simple left-to-right feature-order strategy mimics Take The Best. All 12 unique pairs are diagnostic and repeat eight times for 96 trials.

[2] This design creates a parameter-robust directional dissociation while also testing a quantitative position gradient. The validity-to-display mapping is deliberately nonmonotonic and has a positive rank-position correlation, so the competing model's mapping-simplicity term is clipped to zero and communicated validity cannot alter its attentional priorities. On every trial, the globally most valid cue (feature 8, validity .95) favors one option, while exactly three lower-validity cues favor the other. Take The Best therefore always selects the feature-8 winner and predicts the same oriented choice probability in every condition: (1-epsilon)*sigmoid(beta)+epsilon/2, which remains above .50 throughout its parameter range. In the competing stochastic-search model, validity is gated out and the first discriminating cue is selected primarily by accessibility and display position. Because three cues support the opposing option and only one supports the validity-ranked winner, it predicts the validity-ranked winner on only about 20-30% of trials across its parameter range. Moreover, as the three opposing cues shift from early positions (features 1-3) toward later positions (features 5-7), their positional advantage decreases, producing a modest increase in choices of the feature-8 winner; TTB predicts complete invariance across this manipulation. A/B mirrors remove product-key bias. The 12 unique pairs repeat eight times for 96 trials.

[3] This design tests a rank-distance signature that Take The Best cannot produce. The ten validities decrease monotonically with display position, making mapping simplicity exactly maximal for Accessibility-Gated Stochastic Lexicographic Search. Each trial has exactly two discriminating experts: the higher-validity expert favors one product and the lower-validity expert favors the other, while all remaining experts tie. The rank distance between those two experts is manipulated at four levels (1, 2, 4, or 7 ranks), with three different cue pairs at each distance. Take The Best always follows the higher-validity discriminating expert and therefore predicts exactly the same oriented choice probability at every rank distance: (1-epsilon)*sigmoid(beta)+epsilon/2. In contrast, stochastic lexicographic search can inspect either discriminating expert. Because aligned validity information creates a priority difference proportional to their rank separation, it predicts a monotonic increase in choice of the higher-ranked expert's product as rank distance grows: adjacent conflicts should produce appreciable stochastic choice, whereas seven-rank conflicts should be nearly deterministic. Replicating each distance at different absolute positions separates rank-distance effects from idiosyncratic accessibility of a particular expert. Every pair has an A/B mirror to control product-side and response-key biases. The 24 unique pairs repeat four times, giving 96 trials.

[4] The validity sequence is globally anti-aligned with display position: validity ranks increase strongly toward later positions, so the competing theory's rank-position correlation is positive and its mapping-simplicity term is clipped to zero. Consequently, Accessibility-Gated Stochastic Lexicographic Search must ignore communicated validity in this experiment. Each diagnostic trial pits an earlier, low-validity cue against the immediately following, much higher-validity cue. Under the advocated theory, the 0.45-0.47 validity increase is large enough to reverse the local positional ordering for every allowed parameter combination, including the accessibility offsets. It therefore follows the later, high-validity cue and predicts approximately 0.78-0.86 choice probability for its product. The competing theory instead gives the later cue no validity advantage; with only a small one-position accessibility difference, it predicts choices near chance and, on average, a slight preference for the earlier low-validity cue (roughly 0.45-0.53 for the high-validity-cue winner across parameter extremes). Thus the critical signature is systematic selection of locally validity-promoted later cues rather than near-chance choices. Four adjacent-cue conflicts establish that the effect generalizes across the display rather than depending on one expert. Each conflict has an A/B mirror, controlling product-side and response-key biases. The eight unique pairs repeat 12 times each for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design exploits the competing theories' sharply different treatment of cues below the first two positions. The validity sequence has rank-score order [7,6,0,1,2,3,4,5], giving rank-position correlation -1/6 and mapping simplicity 1/6. Consequently, the advocated theory's validity gate is at most (1/6)^3 \u2248 .0046 and attention is governed mainly by weak positional accessibility plus stochastic cue-order sampling. All eight cues discriminate on every trial. Features 1 and 2 always support the same option, five cues support its opponent, and one additional supporting cue is moved systematically from feature 3 through feature 8. The advocated theory therefore integrates the Plackett-Luce weights of three supporting versus five opposing cues and predicts that the option supported by features 1 and 2 will generally be chosen less than 50% of the time, approximately .38-.48 across conditions and parameter extremes. It additionally predicts a graded decline as the third supporting cue moves later in the display. In the fixed-search theory, features 1 and 2 are robustly the top two cues: their large validity and positional advantages cannot be reversed by the allowed accessibility offsets. Because both the first cue and the cue consulted after a search slip support the same option, all six conditions receive exactly the same prediction, approximately .80-.86 choice probability for that option, regardless of where the third supporting cue appears. Thus the theories predict opposite majority choices as well as a quantitative position gradient versus invariance. A/B mirrors control product and response-key biases. The 12 unique pairs repeat eight times for 96 trials.",
  "validities": [
    0.95,
    0.94,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 5530,
  "output_tokens": 3759
}
```
