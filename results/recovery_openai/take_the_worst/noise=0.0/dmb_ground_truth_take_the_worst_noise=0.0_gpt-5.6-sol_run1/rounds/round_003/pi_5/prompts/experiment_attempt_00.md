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
**Description:** Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. When the display contains a partially tied leading-validity tier, confidence in diffuse pooling is selectively attenuated at loads 3–4 because the tied priority structure interferes with forming a unitary low-load comparison. At intermediate load, people retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool graded signed evidence across all diagnostic members of that tier. Under high load, whole-tier poolers obtain a modest coherence benefit when compressing a partially tied leading tier, strengthening—but not discretizing—the tier's graded mean evidence. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.

**Parameters:**
- validities: validities
- low_center: [4.3, 4.8]
- high_center: [7.0, 7.35]
- transition_width: [0.9, 1.3]
- high_transition_width: [0.3, 0.5]
- partial_tie_acceleration: [0.9, 1.4]
- all_equal_delay: [0.9, 1.3]
- recency_compression_coupling: [0.25, 0.45]
- recency_strength: [4.8, 6.8]
- validity_strength: [0.2, 0.65]
- tie_rule: {0, 1}
- partial_tie_low_attenuation: [0.35, 0.6]
- low_evidence_gain: [0.6, 0.85]
- high_evidence_gain: [0.4, 0.65]
- partial_tier_pooling_gain: [1.45, 1.8]
- beta: [4.8, 6.2]
- epsilon: [0.0, 0.06]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Load-Indexed Format Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    load = float(active.size)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    def stable_sigmoid(x):
        x = float(x)
        if x >= 0.0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)

    # Observable load gradually opens sequential retrieval and, later,
    # validity-tier compression. The low transition begins slightly earlier
    # so retrieval can contribute on three-cue conflicts.
    low_width = float(parameters["transition_width"])
    entered_retrieval = stable_sigmoid(
        (load - float(parameters["low_center"])) / low_width
    )

    global_best = float(np.max(validities))
    global_top_tier = np.flatnonzero(np.isclose(
        validities, global_best, rtol=1e-9, atol=1e-12
    ))
    partial_top_tie = 1.0 if 1 < global_top_tier.size < n_features else 0.0
    all_equal = 1.0 if global_top_tier.size == n_features else 0.0

    # Recency strength and compression timing are stably correlated within a
    # person. Strong-recency subjects retain retrieval longer; validity-led
    # subjects compress earlier. Distinct partial tiers facilitate chunking,
    # whereas an all-equal display supplies no validity boundary to compress.
    recency_strength = float(parameters["recency_strength"])
    recency_profile = (recency_strength - 5.8) / 1.0
    effective_high_center = (
        float(parameters["high_center"])
        + float(parameters["recency_compression_coupling"]) * recency_profile
        - partial_top_tie * float(parameters["partial_tie_acceleration"])
        + all_equal * float(parameters["all_equal_delay"])
    )
    compressed = stable_sigmoid(
        (load - effective_high_center)
        / float(parameters["high_transition_width"])
    )

    w_low = 1.0 - entered_retrieval
    w_high = entered_retrieval * compressed
    w_mid = entered_retrieval * (1.0 - compressed)

    # Low-load format: diffuse equal-weight comparison. Normalization makes
    # the evidence the diagnostic-cue majority margin rather than display size.
    low_evidence = float(np.mean(signs[active]))

    # A partially tied leading tier selectively lowers confidence in diffuse
    # pooling at loads 3-4, without changing strictly ordered low-load displays.
    low_gain_multiplier = 1.0
    if partial_top_tie > 0.0 and 3.0 <= load <= 4.0:
        low_gain_multiplier = float(parameters["partial_tie_low_attenuation"])

    # Construct communicated-validity tiers. Cues tied in communicated
    # validity have the same rank; display position remains separately visible.
    tier_rank = np.asarray(
        [np.sum(validities > validities[j] + 1e-12) for j in range(n_features)],
        dtype=float,
    )
    rank_scale = max(1.0, float(n_features - 1))
    normalized_rank = tier_rank / rank_scale
    if n_features > 1:
        display_position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        display_position = np.zeros(1, dtype=float)

    # Intermediate format: expected evidence from sequential retrieval.
    # Positional accessibility is mildly attenuated at three-cue load and
    # reaches its original strength by load six, preserving overload gradients.
    recency_load_scale = 0.85 + 0.15 * np.clip((load - 3.0) / 3.0, 0.0, 1.0)
    retrieval_log_access = (
        recency_load_scale * recency_strength * display_position[active]
        - float(parameters["validity_strength"]) * normalized_rank[active]
    )
    retrieval_log_access -= np.max(retrieval_log_access)
    retrieval_weights = np.exp(retrieval_log_access)
    retrieval_weights /= retrieval_weights.sum()
    mid_evidence = float(np.dot(retrieval_weights, signs[active]))

    # High-load format: compress to the best communicated-validity tier that
    # is diagnostic on this trial. Stable tie_rule=0 uses its first displayed
    # member; tie_rule=1 pools graded evidence over the complete active tier.
    best_validity = float(np.max(validities[active]))
    top_tier = active[np.isclose(
        validities[active], best_validity, rtol=1e-9, atol=1e-12
    )]

    tie_rule = int(parameters["tie_rule"])
    if tie_rule == 0:
        first_member = int(np.min(top_tier))
        high_evidence = float(signs[first_member])
    else:
        high_evidence = float(np.mean(signs[top_tier]))

    # At overload, coherent retrieval of an entire partially tied leading tier
    # gives whole-tier poolers a bounded confidence benefit. The underlying
    # evidence remains its graded mean rather than being collapsed to its sign.
    high_gain = float(parameters["high_evidence_gain"])
    if tie_rule == 1 and partial_top_tie > 0.0 and load >= 6.0:
        high_gain *= float(parameters["partial_tier_pooling_gain"])

    beta = float(parameters["beta"])

    def evidence_probabilities(evidence, gain=1.0):
        logits = np.array([
            0.5 * beta * float(gain) * float(evidence),
            -0.5 * beta * float(gain) * float(evidence),
        ], dtype=float)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    # Format recruitment is part of representation, not response noise.
    # Stable format-specific gains calibrate confidence without weakening the
    # intermediate positional profile that is needed at overload loads.
    p_core = (
        w_low * evidence_probabilities(
            low_evidence,
            low_gain_multiplier * float(parameters["low_evidence_gain"])
        )
        + w_mid * evidence_probabilities(mid_evidence)
        + w_high * evidence_probabilities(high_evidence, high_gain)
    )

    # Ordinary lapses occur only after each format has generated evidence.
    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total
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


## COMPETING THEORY
**Description:** Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

**Parameters:**
- validities: validities
- capacity: [4.5, 5.5]
- gate_slope: [2.0, 5.0]
- beta: [1.55, 2.05]
- epsilon: [0.0, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Comparison expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    discriminating = np.flatnonzero(differences != 0)
    load = float(discriminating.size)
    if load == 0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["capacity"])
    gate_slope = float(parameters["gate_slope"])

    # Probability that exhaustive equal-weight integration is recruited.
    # This stable logistic gate approaches one below capacity and zero above it.
    gate_logit = gate_slope * (capacity - load)
    if gate_logit >= 0.0:
        exp_neg = np.exp(-gate_logit)
        p_integrate = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(gate_logit)
        p_integrate = exp_pos / (1.0 + exp_pos)

    # Manageable-load strategy: all discriminating cues receive equal weight.
    tally_margin = float(np.sum(np.sign(differences)))
    if tally_margin > 0.0:
        tally_scores = np.array([1.0, 0.0])
    elif tally_margin < 0.0:
        tally_scores = np.array([0.0, 1.0])
    else:
        tally_scores = np.array([0.0, 0.0])

    # Overload strategy: validity-ranked compression to the first diagnostic cue.
    cue_order = np.argsort(-validities, kind="stable")
    lex_scores = np.array([0.0, 0.0])
    for cue in cue_order:
        if differences[cue] > 0.0:
            lex_scores = np.array([1.0, 0.0])
            break
        if differences[cue] < 0.0:
            lex_scores = np.array([0.0, 1.0])
            break

    beta = float(parameters["beta"])

    def noisy_strategy_choice(scores):
        logits = beta * np.asarray(scores, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    # Response noise follows, rather than determines, strategy recruitment.
    p_tally = noisy_strategy_choice(tally_scores)
    p_lex = noisy_strategy_choice(lex_scores)
    p_core = p_integrate * p_tally + (1.0 - p_integrate) * p_lex

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Eight validity-ranked cues create a controlled tally-margin series while holding Take The Best's decision basis constant. In the first eight pairs, the highest-validity cue always favors A, so TTB predicts the same A-choice probability on every pair, regardless of the other seven cues. Across those pairs, however, A wins m = 1,...,8 cues, producing tally differences of -6, -4, -2, 0, 2, 4, 6, and 8. Tallying therefore predicts strong B choices for m=1 to 3, exact guessing at m=4, and increasingly confident A choices for m=5 to 8. In particular, the m=4 pair gives a sharp dissociation: TTB favors A while Tallying predicts exactly 0.5 for any beta and epsilon. The final eight pairs swap A and B, preserving the theoretical contrasts while controlling response-side bias. Thus TTB predicts a flat choice function conditional on the top cue's direction, whereas Tallying predicts a steep monotonic function of the number of cue wins. With 16 unique pairs, each is repeated six times for 96 total trials.

[1] This design orthogonally contrasts the direction of the overall tally with the direction of the first validity-ranked discriminating cue. Every pair contains exactly three discriminating cues. In trials 1-6, A wins two cues and B wins one, so Tallying predicts the same A-choice probability on every trial: epsilon/2 + (1-epsilon)*sigmoid(beta). However, the first discriminating cue occurs successively at validity ranks 1 through 6 and alternates between favoring A and favoring B. Take The Best therefore predicts alternating high and low A-choice probabilities even though the tally is unchanged. Trials 7-12 swap A and B, balancing response side and reversing both predictions. Across all 12 pairs, tally direction and TTB direction are balanced: the theories agree on six pairs and directly conflict on six. Thus evidence for Tallying is a two-level choice pattern determined solely by the 2-versus-1 cue majority, invariant to stopping rank and the top discriminating cue; TTB instead predicts choices grouped by that first discriminating cue. With 12 unique pairs, each pair is repeated eight times for 96 trials.

[2] This design targets the advocated theory's capacity gate by varying the number of discriminating cues from 3 through 8, spanning the capacity range of 4.5-5.5. In the first six pairs, equal-weight tallying favors A while the highest-validity discriminating cue always favors B; the remaining six pairs are exact A/B reversals. Tally margins are held constant within parity: A wins by one cue at loads 3, 5, and 7, and by two cues at loads 4, 6, and 8. Consequently, the competing Tallying theory predicts load-invariant A-choice probabilities within each parity series—sigmoid(beta) for odd loads and sigmoid(2*beta) for even loads—and always favors the cue-majority option. In contrast, Capacity-Gated Comparison predicts a systematic crossover: strong majority-consistent choices at loads 3-4, an intermediate transition near load 5, and strong highest-validity-cue-consistent choices at loads 6-8. Its post-strategy noise does not depend on tally margin, so the predicted load gradient appears in both parity series. The mirrored pairs balance response side and should exhibit the same crossover toward B rather than A. With 12 unique pairs repeated eight times, the experiment contains 96 trials.

[3] This design tests a sharp prediction produced by exact tally ties at and above the proposed capacity boundary. Every pair has an equal number of cues favoring A and B, so Tallying predicts P(A)=0.50 exactly on every trial, independently of beta and epsilon. Discriminating-cue load is varied across 4, 6, and 8 cues. Within each load there are two structurally distinct exemplars, followed by exact A/B reversals. In each unreversed pair, the highest-validity discriminating cue favors A; in each reversal it favors B. Capacity-Gated Comparison therefore predicts near-chance choice at load 4 when integration is usually recruited, but a marked lexicographic preference at loads 6 and 8 as overload recruits the highest-validity cue. Given its parameter ranges, the model predicts roughly P(lex-consistent choice)=0.72-0.89 at load 6 and 0.80-0.89 at load 8, whereas Tallying remains exactly at 0.50. Thus the critical signature is a load-dependent, response-side-symmetric departure from chance toward the highest-validity discriminating cue. Twelve unique pairs repeated eight times yield 96 trials.

[4] This design exploits equal-validity ties to create a qualitative and quantitative dissociation between stable lexicographic tie-breaking and rank integration. Experts 1-3 share the highest validity. All seven cues discriminate, every unreversed pair has the same 4-versus-3 majority favoring A, and Expert 1 always favors A. Thus both branches of Capacity-Gated Comparison—equal-weight tallying and the stable validity-ranked lexicographic search—select A on every unreversed pair. It therefore predicts a flat P(A)=(1-epsilon)*sigmoid(beta)+epsilon/2, approximately 0.80-0.89, regardless of how evidence is distributed among the other tied top experts. Precision-Annealed Rank Integration instead gives Experts 1-3 exactly equal and overwhelmingly dominant attention because they share rank zero, while the next active expert has rank three. The first two pairs have all three top experts favoring A and should produce near-certain A choices; the next two have a 2-to-1 top-tier majority for A and should produce moderately strong A choices; the final two have Experts 2 and 3 opposing Expert 1, so their 2-to-1 top-tier majority favors B and should reverse choice despite the overall 4-to-3 majority for A. Across the stated parameter ranges, the competitor should produce an approximately descending profile from above 0.95, through roughly 0.77-0.87, to roughly 0.14-0.25 P(A), whereas the advocated theory remains flat near 0.80-0.89. Two tail arrangements within each top-tier composition test the competitor's smaller residual rank-weighting effects. The final six pairs are exact A/B reversals, balancing response side. Twelve unique pairs repeated eight times yield 96 trials.

[5] All trials have exactly five discriminating cues: Experts 1, 2, 4, 6, and 8 (validity ranks 0, 1, 3, 5, and 7). In every unreversed pair, three cues favor A, the highest-validity cue favors B, and exactly one additional cue favors B. That second B-favoring cue moves progressively from Expert 2 to Expert 4, Expert 6, and Expert 8. Prospective Capacity-Gated Comparison therefore predicts exactly the same A-choice probability for all four unreversed pairs: load, tally direction and margin, and the first validity-ranked discriminating cue are identical. Its integration branch selects A, its lexicographic branch selects B, and only the fixed load-5 gate determines their mixture. Precision-Annealed Rank Integration instead predicts a strictly increasing A-choice gradient as the second B cue moves down ranks 1, 3, 5, and 7, because the opposing evidence loses attention mass while all other properties remain fixed. At representative parameter midpoints, the approximate unreversed profile is P(A)=0.12, 0.31, 0.46, and 0.55, rather than the competitor's flat profile. Exact A/B reversals should produce the complementary decreasing profile while controlling response bias. The non-discriminating cues vary in their shared displayed values across exemplars but cannot affect either model. Eight unique pairs repeated 12 times yield 96 trials.

[6] This design isolates the competing theory's late high-load format transition while placing all trials above the advocated theory's capacity boundary. There are two exemplars each at discriminating-cue loads 6, 7, and 8, followed by exact A/B reversals. In every unreversed pair, the highest-validity active expert favors A, whereas the most recent active experts favor B. At loads 6 and 7 the equal-weight tally also favors B; at load 8 it is tied. Prospective Capacity-Gated Comparison should already be predominantly lexicographic: it may show some residual B-majority influence at load 6, but should strongly favor A by load 7 and then change very little from load 7 to load 8. Across its parameter ranges, its approximate unreversed prediction is P(A)=0.65-0.89 at load 6, 0.77-0.89 at load 7, and 0.80-0.89 at load 8. Load-Indexed Format Arbitration instead assigns substantial weight to recency-dominated sequential retrieval at loads 6-7, yielding B-directed evidence because the late diagnostic cues all favor B. Its narrow high-load transition then increasingly compresses the display to the uniquely highest-validity active cue, which favors A. It therefore predicts a substantially steeper increase in A choice from load 6 through load 8, with especially diagnostic disagreement at load 6 and a larger load-7-to-load-8 change than the advocated theory. The two exemplars at each load alter active positions while preserving this early-validity-versus-late-recency conflict. Exact reversals require complementary choice profiles and control response-side bias. Twelve unique pairs repeated eight times yield 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design creates matched low-load confidence and recency dissociations while making Prospective Capacity-Gated Comparison exactly invariant. Experts 1 and 2 form a distinctive partially tied leading-validity tier. Each unreversed trial has either three or four discriminating cues, the highest-validity active cue favors A, and the equal-weight majority also favors A. Consequently, both branches of the competing theory select A on every unreversed trial, predicting the same P(A) throughout (approximately 0.80-0.89 over its parameter ranges), regardless of load, unanimity, or the losing cue's display position. Within each fixed active-cue set, Load-Indexed Format Arbitration instead predicts a graded profile. The unanimous condition has evidence +1 in every format and should produce the strongest A preference. Moving to a narrow A majority reduces normalized diffuse evidence to 1/3 at load 3 or 1/2 at load 4; moreover, the partially tied leading tier selectively attenuates low-format confidence at these loads. Among narrow-majority trials, placing the sole B-favoring cue late rather than at an intermediate position should further reduce A choice because sequential retrieval has a strong recency gradient. Thus the critical predicted ordering is unanimous > intermediate-position loser > latest-position loser, especially at load 4 where retrieval recruitment is greater. The competing theory predicts no such within-set ordering because tally direction and lexicographic direction are unchanged. Exact A/B reversals provide response-side balance. There are 24 unique pairs, yielding four repetitions each for 96 trials.",
  "validities": [
    0.91,
    0.91,
    0.82,
    0.75,
    0.69,
    0.63,
    0.57,
    0.52
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
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
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
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
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      0,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      0,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      0,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
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
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      0,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      0,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      0,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
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
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 6855,
  "output_tokens": 3961
}
```
