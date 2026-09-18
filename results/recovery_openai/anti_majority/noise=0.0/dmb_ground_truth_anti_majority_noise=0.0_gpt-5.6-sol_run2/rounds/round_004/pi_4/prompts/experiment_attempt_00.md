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
**Description:** Semantic-Polarity Smooth Integration theory: people first interpret what the marked binary state means rather than assuming that 1 is intrinsically desirable. Each subject forms a stable posterior belief that a mark is adverse or suspicious. Conditional on either the adverse interpretation or its residual beneficial alternative, the subject smoothly integrates two continuous signals: the difference in total marks and a partially count-residualized validity-weighted difference. The count signal normally dominates, while the weighted signal supplies graded sensitivity to which experts carry the marks on every trial. The two semantic interpretations generate opposite choice distributions, which are averaged according to the subject's polarity confidence. A small lapse process captures stimulus-independent errors.

**Parameters:**
- validities: validities
- count_weight: [0.38, 1.05]
- validity_weight: [0.7, 1.8]
- partial_centering: [0.1, 0.3]
- adverse_polarity_confidence: [0.78, 1.0]
- lapse_rate: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Semantic-Polarity Smooth Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # Positive differences mean that A contains more marked states than B.
    count_difference = float(np.sum(a - b))
    partial_centering = float(parameters["partial_centering"])
    effective_validities = validities - partial_centering * float(np.mean(validities))
    weighted_difference = float(np.dot(effective_validities, a - b))

    # There is no count gate: validity contributes continuously on every
    # discriminating trial. Partial centering limits the count component
    # embedded in the weighted sum without removing it completely.
    integrated_markedness = (
        float(parameters["count_weight"]) * count_difference
        + float(parameters["validity_weight"]) * weighted_difference
    )

    adverse_logits = np.array(
        [-0.5 * integrated_markedness, 0.5 * integrated_markedness],
        dtype=float,
    )
    adverse_logits = adverse_logits - np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # The beneficial interpretation reverses the semantic polarity while
    # preserving the same smooth evidence-integration computation.
    beneficial_probs = adverse_probs[::-1]
    polarity_confidence = float(parameters["adverse_polarity_confidence"])
    core_probs = (
        polarity_confidence * adverse_probs
        + (1.0 - polarity_confidence) * beneficial_probs
    )

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs):
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
**Description:** Capacity-Calibrated Dual-Channel Integration theory: people continuously integrate a rapid representation of the relative number of marked ratings with a validity-sensitive representation of where those marks occur. Neither representation is encoded in raw additive units. The count channel expresses the tally margin relative to the proportion of displayed cues that discriminate and then passes it through a smooth saturating response, so large margins remain decisive without producing indefinitely increasing evidence. A stable discrimination-scaling exponent governs how strongly sparse cue sets dilute count evidence. The validity channel represents count-residualized reliability evidence using softly standardized units and divisively normalizes it according to display size, the number of discriminating cues, and cue conflict. Partial rather than full reliability-energy standardization preserves psychologically meaningful differences between sharply separated and shallow validity profiles without allowing low-dispersion profiles to become artificially amplified. Validity remains available on every trial, including count ties, but there are no special tie, audit, or selected-margin rules. A stable allocation parameter controls the balance between channels. Capacity also contracts the behavioral expression of individual differences in allocation and precision as displays become large, yielding appreciable heterogeneity in compact tasks but more homogeneous behavior in very high-dimensional tasks. Stable semantic-polarity confidence determines whether marked states are interpreted as adverse or beneficial, while stable response precision and lapse rate govern stochasticity.

**Parameters:**
- validities: validities
- count_allocation: [0.58, 0.86]
- count_saturation_scale: [0.6, 1.2]
- count_discrimination_exponent: [0.20, 0.45]
- integration_capacity: [7.0, 14.0]
- validity_sensitivity: [2.2, 4.5]
- reliability_standardization_exponent: [0.05, 0.30]
- display_compression: [0.35, 1.1]
- discrimination_compression: [0.3, 1.0]
- conflict_compression: [0.45, 1.35]
- response_precision: [3.0, 6.0]
- adverse_polarity_confidence: [0.84, 1.0]
- lapse_rate: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Calibrated Dual-Channel Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    discriminating = np.abs(difference)
    n_discriminating = float(np.sum(discriminating))
    count_difference = float(np.sum(difference))

    if n_discriminating <= 0.0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["integration_capacity"])

    # Large displays reduce the behavioral expression of stable parameter
    # differences. This is a continuous capacity effect rather than a switch.
    heterogeneity_expression = capacity / (capacity + float(n_features))
    raw_allocation = float(parameters["count_allocation"])
    allocation = 0.5 + heterogeneity_expression * (raw_allocation - 0.5)

    raw_precision = float(parameters["response_precision"])
    effective_precision = 1.0 + heterogeneity_expression * (raw_precision - 1.0)

    # The tally margin is represented relative to the fraction of the display
    # that actually discriminates. Smooth saturation makes margins near the
    # extremes robust while compressing the difference between moderate and
    # very large raw margins. The exponent below one-half weakens count
    # evidence selectively when only a sparse subset of cues discriminates.
    discrimination_fraction = n_discriminating / float(n_features)
    count_denominator = float(parameters["count_saturation_scale"]) * (
        max(discrimination_fraction, 1.0 / float(n_features))
        ** float(parameters["count_discrimination_exponent"])
    )
    relative_count = count_difference / max(count_denominator, 1e-12)

    count_load_factor = np.sqrt(capacity / (capacity + float(n_features)))
    count_signal = (
        2.0
        * allocation
        * count_load_factor
        * np.tanh(relative_count)
    )

    # Remove the validity component that merely repeats total mark count.
    centered_validities = validities - float(np.mean(validities))
    raw_validity = float(np.dot(centered_validities, difference))

    # Soft standardization preserves differences in validity-profile
    # dispersion. An exponent below one prevents shallow profiles with small
    # reliability energy from being disproportionately amplified.
    reliability_energy = float(
        np.sqrt(np.dot(centered_validities ** 2, discriminating))
    )
    reliability_exponent = float(parameters["reliability_standardization_exponent"])
    validity_denominator = max(reliability_energy, 0.025) ** reliability_exponent
    standardized_validity = raw_validity / validity_denominator

    # Conflict is maximal when discriminating cues split evenly and declines
    # smoothly as their count majority becomes coherent.
    conflict = 1.0 - min(1.0, abs(count_difference) / n_discriminating)
    display_load = float(n_features) / (float(n_features) + capacity)
    discrimination_load = n_discriminating / (n_discriminating + capacity)

    normalization = (
        1.0
        + float(parameters["display_compression"]) * display_load
        + float(parameters["discrimination_compression"]) * discrimination_load
        + float(parameters["conflict_compression"]) * conflict
    )
    validity_signal = (
        (1.0 - allocation)
        * float(parameters["validity_sensitivity"])
        * standardized_validity
        / normalization
    )

    markedness_evidence = effective_precision * (count_signal + validity_signal)

    # Under the adverse interpretation, positive markedness evidence means A
    # carries the greater burden and therefore favors response B.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    beneficial_probs = adverse_probs[::-1]
    polarity = float(parameters["adverse_polarity_confidence"])
    core_probs = polarity * adverse_probs + (1.0 - polarity) * beneficial_probs

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This 7-cue design creates a direct quantitative dissociation. In each base pair, the highest-validity cue that discriminates favors A, while the equal-weight tally either favors B or is tied; the final six pairs swap A and B to eliminate response-side confounds. For the first three base pairs, cue 1 favors the TTB option, but that option loses the overall tally by margins of 1, 3, and 5. For the next three, cue 1 ties and cue 2 favors the TTB option, while the tally is respectively tied or favors the other option by margins of 2 and 4. TTB predicts the same probability of choosing its designated winner in every condition because its binary winner score is always 1 versus 0, regardless of search depth or tally margin. In contrast, Tallying predicts exactly 0.5 choice probability in the tally-tie condition and increasingly strong preference against the TTB winner as the opposing tally margin grows. Thus the critical signature is an approximately flat, validity-ordered choice profile under TTB versus a graded tally-margin profile under Tallying. The 12 unique pairs yield exactly 96 trials when repeated eight times.

[1] This 6-cue design makes Tallying parameter-free on every trial: each option wins on exactly one feature, while all remaining features tie, so the two tallies are always equal. Tallying therefore predicts P(A)=P(B)=0.5 regardless of beta or epsilon. In contrast, Take The Best selects the option favored by the highest-validity discriminating cue. The first discriminating cue occurs at ranks 1 through 5 across the eight base pairs, and the lower-ranked discriminating cue always favors the other option. TTB predicts the same above-chance probability for its designated winner at every search depth, namely (1-epsilon)*sigmoid(beta)+epsilon/2. The final eight pairs reverse A and B, balancing response side while preserving the cue-rank structure. Thus, systematic preference for the validity-ordered winner supports TTB, whereas chance-level choice on every pair—including no validity-ordered preference at any search depth—is uniquely predicted by Tallying. Sixteen unique pairs repeated six times yield exactly 96 trials.

[2] This 7-feature, 16-pair design produces two complementary dissociations. In the eight unequal-count trials, one option has 1, 3, 5, or 7 more marked ratings than the other, with each configuration also presented after swapping A and B. Because the features are binary, the tally margin exactly equals the difference in the number of 1s. Tallying therefore predicts an increasingly strong preference for the option with more 1s as the absolute margin increases. Count-Gated Markedness Minimization predicts the opposite ordering: an increasingly strong preference for the option with fewer 1s, while ignoring validity at this stage. The remaining eight trials are four equal-count configurations and their A/B reversals. Tallying is parameter-free on these trials and predicts exactly 0.5 because each option wins equally many features. The markedness theory instead consults the highest-validity discriminating feature and favors the option with a 0 on that feature. The decisive feature is systematically placed at validity ranks 1 through 4, while the more valid preceding features tie. Thus, support for the advocated theory requires both an anti-positive, margin-graded preference on unequal-count trials and a validity-ordered anti-mark preference on equal-count trials; Tallying predicts the reverse choice polarity in the first set and chance in the second. Sixteen unique pairs repeated six times yield exactly 96 trials.

[3] This design targets the competitor's count-gating discontinuity rather than merely contrasting positive versus negative interpretations of 1. There are four matched triplets, indexed by the validity rank (1 through 4) of the highest-validity discriminating expert. Within every base triplet, that expert gives B a 1 and A a 0. Lower-validity A-favoring ratings are then added one at a time, producing tally/count margins for A of 0, +1, and +2 while leaving the highest-validity discriminating rating unchanged. Tallying predicts P(A)=0.5 at margin 0 and a monotonic increase above 0.5 at margins +1 and +2. Count-Gated Markedness Minimization instead favors A at margin 0 because its validity tiebreak treats B's mark on the best discriminating expert as adverse; as soon as one additional low-validity A mark creates a nonzero count difference, the count gate activates and the theory reverses to favoring B, increasingly strongly at margin +2. Thus, adding a nominally favorable low-validity rating causes no reversal under Tallying but a qualitative A-to-B reversal under the competitor. The same construction at four validity ranks shows that the effect is tied to the proposed gate rather than one expert's screen location. Every pair is also presented with A and B exchanged, balancing response side. The 24 unique pairs are repeated four times for 96 trials.

[4] This 11-feature design isolates validity sensitivity while holding the marked-count difference constant. In each of six base pairs, A has exactly one more mark than B, so Count-Gated Markedness Minimization always activates its count stage and predicts the same probability of choosing B; validity cannot affect its prediction. Across the six pairs, however, marks are progressively transferred onto higher-validity experts for B and onto lower-validity experts for A. The raw validity-weighted difference dot(validities, A-B) consequently decreases in equal steps from +0.50 to -1.60: +0.50, +0.08, -0.34, -0.76, -1.18, and -1.60. Because the count difference remains +1, partial centering subtracts the same quantity in every condition and preserves this strictly decreasing profile. The smooth-integration theory therefore predicts a monotonic decline in preference for B and eventually a reversal toward A. Indeed, over its full parameter ranges, its integrated markedness is necessarily positive in the first condition but necessarily negative in the last condition, whereas the advocated theory favors B equally strongly at both endpoints and throughout the series. The remaining six pairs swap A and B, balancing response side while preserving the diagnostic profile when choices are coded toward the option with fewer marks. Twelve unique pairs repeated eight times yield exactly 96 trials.

[5] This 11-feature design isolates smooth validity integration within equal-count trials, directly targeting the competitor's validity-tiebreak gate. In every base pair, A and B each contain exactly five marks, and the highest-validity discriminating expert (validity 0.99) marks A but not B. Count-Gated Markedness Minimization therefore makes exactly the same prediction across all six base conditions: its count stage ties, its tiebreak always selects the 0.99 expert, and it favors B with a condition-invariant probability. In contrast, lower-validity marks are redistributed so that the validity-weighted difference dot(validities, A-B) declines from +1.49 through +0.89, +0.29, -0.09, -0.51, to -0.94. Because the count difference is zero, both the count term and the partial-centering correction vanish in Semantic-Polarity Smooth Integration; its evidence is exactly validity_weight times this graded weighted difference. It therefore predicts a monotonic decline in choices of B, near indifference around the zero crossing, and a reversal toward A in the negative-weight conditions. The six additional pairs exchange A and B, balancing response side while preserving the flat-versus-graded dissociation when choices are coded toward the option left unmarked by the 0.99 expert. Twelve unique pairs repeated eight times yield 96 trials.

[6] This 15-feature design tests the advocated theory's discrepancy-evoked validity audit while making its tie-escalation bonus negligible because tie capacity is exp(-(15-7)) = exp(-8). Each of eight matched sets contains an equal-count trial and a margin-one trial. In every equal-count trial, A's marks are concentrated on low-validity experts and B's marks on high-validity experts, producing a strongly negative count-residualized validity signal. The corresponding margin-one trial differs only by changing the mean-validity expert (validity 0.75, exactly equal to the mean of all validities) from a 0 to a 1 for A. This adds one A mark while leaving residual validity exactly unchanged. Smooth Integration therefore necessarily predicts that this manipulation increases markedness evidence for A, by count_weight + validity_weight*0.75*(1-partial_centering), and hence increases choices of B under the predominant adverse interpretation. The advocated theory predicts the opposite direction: the one-mark discrepancy activates an audit gain of at least 0.9 on the unchanged negative residual. Across the eight sets, residual validity ranges from -2.58 to -3.08. Even in the weakest case, the largest possible added count signal is about 2.20, whereas the smallest audit penalty is 0.9*2.58 = 2.32; the negligible 15-cue tie bonus cannot offset this. Thus the advocated theory necessarily predicts lower B choice probability on the margin-one trial than on its matched tie trial, whereas the competing theory necessarily predicts higher B choice probability. Seven leave-one-pair-out variants provide a quantitative residual-strength gradient in addition to the full configuration. All 16 base trials are also A/B reversed, yielding 32 unique pairs repeated three times for exactly 96 trials.

[7] This 37-feature design targets the competitor's discrepancy-audit decay between count margins 1 and 5 while holding count-residualized validity evidence fixed. The validities consist of 16 high/low pairs symmetric around the mean validity of 0.75, plus five exactly mean-validity experts. In every matched pair of trials, high-validity experts mark A and low-validity experts mark B, producing positive residual-validity evidence that A carries the more suspicious marks. The margin-one trial has one additional A mark from a 0.75 expert; the matched margin-five trial activates all five 0.75 experts. Because mean-validity experts have zero centered validity, this manipulation raises the A-minus-B count margin from 1 to 5 without changing residual validity. Four configurations omit different symmetric expert pairs, yielding residual-validity values 7.38, 7.34, 7.30, and 7.26. Semantic-Polarity Smooth Integration necessarily predicts increased choice of B from margin 1 to margin 5: its evidence change is 4*[count_weight + 0.75*(1-partial_centering)*validity_weight], which is strictly positive throughout its parameter ranges. The competing theory necessarily predicts the opposite change. At margin 1 its audit activation is 1, whereas at margin 5 it is 5*exp(-4*audit_decay), at most 5*exp(-2.8)=0.304. Even at the weakest residual value, the minimum resulting audit loss is 0.9*(1-0.304)*7.26=4.55 evidence units. The largest possible increase from the bounded count signal and extreme-count tail is approximately 4.42, so competitor evidence for B must decrease. Thus the critical within-configuration signature is a margin-one-to-margin-five increase in B choices under smooth integration versus a decrease under residual auditing. Every trial is also A/B reversed. The 16 unique pairs repeated six times yield exactly 96 trials.

[8] This 19-feature design isolates the advocated theory's discrimination-fraction scaling and count saturation while making the competing theory exactly condition-invariant. The validity profile has mean 0.75. In every base trial, A has exactly one more mark than B, and the validity-weighted difference is exactly 0.75. The first, mean-validity expert favors A; additional discriminating experts are introduced in equal-validity opposing pairs, with one member favoring A and the other favoring B. Consequently, the number of discriminating cues increases across conditions as 1, 3, 5, 7, 11, 15, 17, and 19, while both the count difference (+1) and weighted difference (0.75) remain fixed. Semantic-Polarity Smooth Integration therefore predicts exactly the same choice probability in all eight conditions for every parameter setting: partial centering also remains constant because the count difference is fixed. In the advocated theory, centered-validity evidence is exactly zero because the lone unpaired cue equals the profile mean and every added pair cancels. Its prediction is thus driven by the count channel. As the discriminating fraction rises from 1/19 to 1, the denominator of relative count evidence increases, reducing the saturated count signal. It therefore predicts a strictly decreasing probability of choosing B—the option with fewer marks—across the eight base conditions, with the largest contrast between the sparse and fully discriminating displays. The final eight trials exchange A and B, balancing response side while preserving the diagnostic pattern when choices are coded toward the option with fewer marks. Sixteen unique pairs repeated six times yield exactly 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This 17-feature design isolates the competing theory's saturating count representation while eliminating its validity channel. The validity profile has mean 0.60 and is partitioned into reliability-balanced groups: one five-expert group and six two-expert groups, each with average validity exactly 0.60. Within a trial, every expert in a group favors the same option. Consequently, each group's centered-validity contributions cancel, so dot(validities-mean(validities), A-B)=0 on every trial. All 17 experts discriminate in every condition, holding discrimination fraction, conflict normalization, display load, and discrimination load fixed apart from the count-margin term itself. Across the six base trials, A's mark-count advantage is 1, 3, 5, 7, 9, or 13. Semantic-Polarity Smooth Integration therefore has integrated markedness exactly d*[count_weight + 0.60*(1-partial_centering)*validity_weight], where d is the count margin: latent evidence grows linearly across the entire series, predicting progressively more choices of B, the less-marked option, especially beyond margin 3 for subjects not already at their polarity ceiling. Capacity-Calibrated Dual-Channel Integration instead has zero validity signal and count evidence proportional to tanh(d/count_saturation_scale). Because all cues discriminate, this response is already at least tanh(3/1.2)=0.987 at margin 3 and changes by less than 1.4% of its unsaturated scale between margins 3 and 13. It therefore predicts an abrupt margin-1-to-margin-3 change followed by an almost flat choice profile, with bounded evidence at large margins. The critical quantitative dissociation is thus continued large-margin growth under smooth additive integration versus near-complete saturation after margin 3 under dual-channel integration. Every base pair is also A/B reversed to balance response side. Twelve unique pairs repeated eight times yield exactly 96 trials.",
  "validities": [
    0.95,
    0.55,
    0.5,
    0.5,
    0.5,
    0.7,
    0.5,
    0.69,
    0.51,
    0.68,
    0.52,
    0.67,
    0.53,
    0.66,
    0.54,
    0.65,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
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
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
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
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
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
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
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
      0,
      0,
      1,
      1,
      1,
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
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      1,
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
      1,
      1,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      1,
      1,
      1,
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
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      1,
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
      1,
      1,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
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
      0,
      0,
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
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
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
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
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
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
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
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 6777,
  "output_tokens": 5044
}
```
