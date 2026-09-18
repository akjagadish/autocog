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
**Description:** Validity-Ordered Competitive Evidence Buffer with Conflict-Normalized Readout. Decision makers inspect cues in descending order of instructed validity and encode every discriminating cue as signed evidence rather than as a provisional choice or directional chunk. Each newly encoded reason causes graded displacement of all earlier traces in a capacity-limited working-memory buffer. Displacement becomes stronger as the buffer exceeds its stable subject-specific capacity, but old evidence is never categorically erased. At response, the decision maker sums all surviving validity-scaled traces. Readout normalization depends on the balance of accessible evidence: it is strongest when substantial trace mass supports both options and weak when the accessible buffer contains a coherent majority. Readout uncertainty additionally grows quadratically with capacity overload and receives a count-based attenuation beginning explicitly at seven discriminating reasons, contracting long conflicts toward chance without imposing arrangement-specific effects. Stable individual differences in effective capacity, displacement rate, validity sensitivity, response gain, and conflict normalization generate heterogeneous recency and integration behavior.

**Parameters:**
- validities: validities
- buffer_capacity: [3.0, 3.8]
- displacement_rate: [2.1, 2.9]
- validity_sensitivity: [0.0, 0.08]
- response_gain: [2.0, 6.5]
- integration_normalization: [0.02, 0.34]
- overflow_noise: [0.75, 1.50]
- lapse_rate: [0.03, 0.15]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive evidence buffer expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = float(parameters["buffer_capacity"])
    displacement = float(parameters["displacement_rate"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    response_gain = float(parameters["response_gain"])
    integration_normalization = float(parameters["integration_normalization"])
    overflow_noise = float(parameters["overflow_noise"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]

    # Inspection follows instructed validity, independently of physical
    # feature position. Stable sorting only resolves exact validity ties.
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Convert instructed reliability into evidence strength. Normalization
    # removes arbitrary differences in the overall validity scale between
    # experiments while preserving all within-experiment validity ratios.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    positive_reliability = reliability[reliability > 0.0]
    reliability_scale = float(np.mean(positive_reliability))
    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:
        reliability_scale = 1.0

    normalized_reliability = np.clip(reliability / reliability_scale, 1e-4, 20.0)
    cue_strength = np.power(normalized_reliability, validity_sensitivity)
    cue_strength = np.clip(cue_strength, 1e-6, 30.0)

    # The buffer contains separate reason traces. A new discriminating reason
    # interferes continuously with every old trace. Displacement is present
    # even below capacity and increases smoothly under overload; there is no
    # categorical slot deletion and no compression into directional runs.
    traces = []
    for j in discriminating:
        current_load = float(len(traces))
        overload_before_entry = max(0.0, current_load + 1.0 - capacity)
        load_factor = 1.0 + overload_before_entry / max(capacity, 1e-6)
        retention = np.exp(-displacement * load_factor / max(capacity, 1e-6))
        retention = float(np.clip(retention, 0.0, 1.0))

        if traces:
            traces = [retention * trace for trace in traces]

        direction = float(np.sign(diff[j]))  # positive favors A
        traces.append(direction * float(cue_strength[j]))

    accessible_evidence = float(np.sum(traces))

    # Preserve the accepted quadratic capacity-overload uncertainty through
    # six reasons. Beginning explicitly at seven discriminating reasons, add
    # a modest count-based attenuation that is not smeared by capacity.
    overflow = max(0.0, float(m) - capacity)
    long_sequence_excess = max(0.0, float(m) - 6.0)
    convex_multiplier = 1.0 + 0.45 * long_sequence_excess
    readout_sd = np.sqrt(
        1.0 + overflow_noise * overflow * overflow * convex_multiplier
    )

    # Normalize integrated evidence only when accessible traces substantially
    # support both options. Squaring the balance ratio leaves coherent
    # majorities largely intact. The effect remains absent for two reasons.
    positive_mass = float(np.sum([trace for trace in traces if trace > 0.0]))
    negative_mass = float(-np.sum([trace for trace in traces if trace < 0.0]))
    total_mass = positive_mass + negative_mass
    if m <= 2 or total_mass <= 0.0 or not np.isfinite(total_mass):
        conflict_load = 0.0
    else:
        balance = 2.0 * min(positive_mass, negative_mass) / total_mass
        balance = float(np.clip(balance, 0.0, 1.0))
        conflict_load = np.sqrt(max(0.0, float(m) - 2.0)) * balance * balance
    integration_divisor = 1.0 + integration_normalization * conflict_load

    decision_variable = (
        response_gain * accessible_evidence
        / (readout_sd * integration_divisor)
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a_core = 1.0 / (1.0 + np.exp(-decision_variable))
    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)

    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
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
**Description:** Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

**Parameters:**
- validities: validities
- cue_attention: [0.95, 1.0]
- overwrite_rate: [0.95, 1.0]
- update_fatigue: [0.06, 0.17]
- lapse_rate: [0.15, 0.41]

**`predict source code`:**
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

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design creates a within-subject quantitative dissociation between Take The Best (TTB) and Tallying. The validities impose an unambiguous cue order. In each base pair, TTB favors A because the highest-validity cue that discriminates favors A; therefore, apart from common beta/epsilon noise, TTB predicts the same probability of choosing its winner across all eight base conditions. By contrast, the signed Tallying margins for A are -5, -3, -1, 0, +2, +3, +7, and -4, respectively. Tallying consequently predicts a strong monotonic probability gradient, including choices opposite TTB in five conditions, exact 50/50 responding in the zero-margin condition, and varying confidence where the rules agree. Every base pair is accompanied by its A/B reversal, balancing response side while preserving the diagnostic evidence structure. With 16 unique pairs, each can be repeated six times for exactly 96 trials.

[1] This is a concentrated cancellation test. In every unique pair, A and B win on exactly the same number of discriminating features: seven base pairs contain one win per option, and the eighth contains three wins per option. Tallying therefore predicts exactly P(A)=P(B)=0.50 on every trial, independently of beta and epsilon. Nevertheless, each pair has a uniquely defined first discriminating cue in validity order. That cue favors A in the eight base pairs and B in their exact reversals. Take The Best therefore predicts a systematic preference for the designated winner, with probability (1-epsilon)*sigmoid(beta)+epsilon/2, rather than chance. The first discriminating cue ranges from validity rank 1 through rank 7, while one condition includes six conflicting discriminations, demonstrating that the tally cancellation prediction is invariant to both cue rank and the amount of opposing evidence. Exact A/B reversals balance response side. With 16 unique pairs repeated six times, the experiment has exactly 96 trials.

[2] This design holds the Tallying margin exactly constant while manipulating the number and recency order of discriminating reasons. In each of the four base pairs, A wins exactly one more feature than B: the tallies are 2–1, 3–2, 4–3, and 5–4. Tallying therefore predicts the identical base-trial probability P(A) = (1-epsilon)*sigmoid(beta) + epsilon/2 at every discriminating-cue count. In contrast, cues are encountered in descending validity order, and the final discriminating cue always favors B. Recency-Gated Reason Overwrite consequently predicts B rather than the tally winner A. Moreover, the base sequences all end in an A,A,B motif, keeping the terminal overwrite structure comparable, while the number of discriminating cues increases from 3 to 5 to 7 to 9. Its fatigue mechanism predicts that the deviation toward the final-cue winner B will progressively contract toward 0.50; each two-cue increase applies an additional retention factor of (1-update_fatigue)^2, approximately 0.69–0.88 over the allowed fatigue range. Exact A/B reversals balance response side and produce the mirror-image predictions. Thus, Tallying predicts a count-invariant preference for the feature-majority option, whereas the advocated theory predicts an opposing last-reason preference with a systematic fatigue gradient. There are eight unique pairs, each repeated 12 times for exactly 96 trials.

[3] This is a fixed-evidence permutation test. Every trial has four discriminating cues, with exactly two favoring A and two favoring B. Tallying therefore predicts P(A)=0.50 on all six trial types, independently of beta and epsilon. The trials exhaust all validity-ordered permutations of two A-favoring and two B-favoring reasons, so the total evidence and processing-fatigue factor are held constant while reason order changes. Recency-Gated Reason Overwrite instead predicts that the fourth (lowest-validity) cue controls the modal response: AABB, ABAB, and BAAB should favor B, whereas BBAA, BABA, and ABBA should favor A. It also predicts a secondary quantitative ordering: sequences ending with two concordant reasons (AABB and BBAA) should generally express the terminal preference more strongly than sequences requiring a single final overwrite, because a missed final update is less consequential when the preceding reason already supports the same option. Thus, systematic terminal-cue preferences—and especially terminal-run strength differences—contradict Tallying's exact chance prediction. The six conditions form three exact A/B-reversal pairs, balance response side, and are each repeated 16 times for 96 trials.

[4] This is a fixed-length, fixed-tally permutation test targeting the competing theory's overload-induced, validity-biased accessibility. All trials contain 10 discriminating cues, with five favoring each option, so discriminating-cue count, fatigue, and total directional tally are held constant. In the two base trials, the final two cues both favor B. The first base sequence is AAAAABBBBB: A owns the five most valid cues, whereas B owns the five least valid cues. The second is BBBAAAAABB: B owns the three most valid cues and the final two cues, while A owns the middle five. Recency-Gated Reason Overwrite predicts similar B preferences in both base conditions because two consecutive terminal B reasons almost completely overwrite any earlier state; with ten reasons, both receive the identical fatigue factor (1-update_fatigue)^8. Even at the lowest attention and overwrite rates, two terminal overwrites leave less than about 1% of a fully conflicting prior state, so the different prefixes should have little effect. In contrast, the retrospective-sampling theory predicts a large permutation effect. At m=10, cubic interference nearly eliminates its ordinal-recency gradient and retrieval approaches a validity-biased asymptote. It should therefore favor A in AAAAABBBBB, where A possesses all five high-validity reasons, but favor B in BBBAAAAABB, where B possesses the three strongest reasons as well as the terminal reasons. Thus, the critical result is not merely overall accuracy but the within-subject contrast: the advocated theory predicts nearly equal terminal-option choice rates across the two base permutations, whereas the competitor predicts a substantial shift, potentially crossing 0.50. Exact A/B reversals balance response side. Four unique pairs repeated 24 times yield 96 trials.

[5] This is a nested conflict-length crossover test. In every base trial, the discriminating cues occur in descending-validity order as an initial run favoring A followed by a terminal run favoring B. The 2-cue condition uses the nearly matched 69% and 68% experts (AB), and the 4-cue condition uses the nearly matched 70%–67% experts (AABB). The advocated retrospective-sampling theory should therefore favor terminal option B at both short lengths: its two-reason terminal-recency floor is especially strong, and recency remains fully expressed at four reasons. The 7- and 8-cue conditions add highly valid early A reasons and low-validity terminal B reasons. Once conflict exceeds four reasons, the advocated theory's cubic interference sharply flattens recency while accessibility becomes validity-biased. It should consequently cross over to A for AAAABBB and AAAABBBB, aided by A's ownership of the 99% and 95% experts (and by a 4–3 reason advantage at length 7). The competing overwrite theory predicts no such sign reversal: its final attended cue and redundant terminal B run should preserve a B preference at every length. Its update-fatigue mechanism predicts only a smooth contraction of that B preference toward 0.50 as length increases. Thus, the key quantitative signature is an approximately B, B, A, A profile across 2, 4, 7, and 8 discriminating reasons under retrospective sampling, versus B preferences throughout with gradually declining magnitude under overwrite. The shorter conditions deliberately use experts with similar validities, ensuring that their expected B preference is driven by recency rather than a large reliability difference. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield exactly 96 trials.

[6] This is a switch-density-by-length test with a protected two-reason suffix. At each discriminating-cue count (6, 8, or 10), the low-switch and high-switch base trials contain equal numbers of A- and B-favoring cues and end with two B-favoring cues. The low-switch sequences are AAABBB, AAAABBBB, and AAAAABBBBB; the matched high-switch sequences are AABABB, ABABAABB, and ABABABAABB. Recency-Gated Reason Overwrite predicts nearly equal B-choice rates for the two patterns at each length: two terminal B reasons reduce any effect of the prefix to at most roughly (1-cue_attention*overwrite_rate)^2 before common fatigue and lapse transformations. It predicts primarily a shared contraction toward 0.50 as length increases because both members of each pair have the same discriminating-cue count. Directional-Run Chunking instead predicts a growing segmentation contrast. The low-switch sequences retain a coherent terminal B chunk, whereas the high-switch sequences contain 3, 5, and 7 directional boundaries. Its quadratic excess-switch interference should make the high-switch conditions progressively approach chance, especially at lengths 8 and 10, while the corresponding low-switch conditions continue to favor B. Thus the critical signature is approximately zero low-versus-high contrast at each length under overwrite, versus an increasingly large low-switch B advantage under chunking. Exact A/B reversals balance response side. The 12 unique pairs are repeated eight times for exactly 96 trials.

[7] This fixed-length scan-order conflict test exploits the theories' different definitions of cue order. The experts are displayed in index order, but their validities imply the descending-validity order 3, 4, 5, 6, 2, 1. In every base trial, the displayed sequence consists of one A run followed by a terminal B run: AAAAAB, AAAABB, AAABBB, or AABBBB. Directional-Run Chunking therefore predicts choices favoring B. It additionally predicts a nonlinear suffix-length profile: the singleton B suffix should be weakest, adding a second B cue should produce a marked increase through run completion and suffix replacement, and terminal runs of two through four cues should be relatively similar because run confidence saturates rapidly. Length, discriminating-cue count, and switch count are fixed at six, six, and one, respectively. In contrast, the competing theory scans in descending validity, where Experts 2 and 1 are always the final two attended discriminating cues and both favor A on every base trial. Two near-certain terminal overwrites should thus yield a strong, approximately condition-invariant A preference; common six-cue fatigue only attenuates that preference. The decisive dissociation is therefore opposite modal choices, supplemented by a singleton-versus-repeated-suffix increase and a two-to-four-cue plateau under chunking versus a nearly flat profile under overwrite. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times produce 96 trials.

[8] This is a below-capacity trace-survival test using three discriminating reasons. The critical base sequences are ABA and BBA in descending-validity order; both share the same final B,A suffix and therefore end in favor of A, but their first reason has opposite direction. Under Recency-Gated Reason Overwrite, the shared B,A suffix almost completely eliminates the influence of the first reason. The exact core-probability difference between ABA and BBA is cue_attention*(1-cue_attention*overwrite_rate)^2. After the common three-reason fatigue and lapse transformations, the allowed parameter ranges bound the predicted A-choice contrast below approximately 0.008. Thus, the advocated theory predicts virtually identical A-choice rates for these two trials. The competitive buffer instead retains a graded trace of the first reason because three reasons do not exceed its minimum capacity. Before readout normalization, changing the first reason from B to A changes accessible evidence by 2*s1*exp(-2*displacement_rate/buffer_capacity), approximately 0.29–0.66 times the first cue's validity strength. It therefore predicts reliably more support for A in ABA than in BBA, potentially including a modal-choice crossover for some parameter values. BAB and AAB are exact option reversals, providing the corresponding B-choice contrast while balancing response side. Experts 1 and 5 are nondiscriminating but establish a broad advertised-validity range; the diagnostic reasons are Experts 2–4. Four unique pairs repeated 24 times produce exactly 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This is a fixed-length prefix-coherence test with a protected two-reason suffix. In descending-validity order, the four base sequences are AAAA, BAAA, ABAA, and BBAA; thus every base trial has four discriminating reasons and ends with two reasons favoring A. Recency-Gated Reason Overwrite predicts almost identical A-choice rates across these conditions. If q = 1 - cue_attention*overwrite_rate, two terminal A updates reduce any prefix-dependent difference by q^2; over the allowed ranges q^2 is at most about 0.0095, and common fatigue and lapse transformations reduce the observable contrast further. The Competitive Evidence Buffer instead preserves graded traces of both prefix reasons. It predicts P(A): AAAA > BAAA > ABAA > BBAA. Reversing the earliest reason reduces accessible A evidence by twice its surviving trace; reversing the second reason has a larger effect because that trace undergoes one fewer displacement step. Reversing both produces the largest reduction. Conflict normalization reinforces this gradient because the one- and two-B conditions contain increasingly substantial accessible evidence for both options, whereas AAAA is completely coherent. The critical quantitative signature is therefore a monotonic coherence gradient, including a larger AAAA-versus-ABAA contrast than AAAA-versus-BAAA, against the overwrite theory's essentially flat profile. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.",
  "validities": [
    0.95,
    0.8,
    0.65,
    0.52
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 6763,
  "output_tokens": 4574
}
```
