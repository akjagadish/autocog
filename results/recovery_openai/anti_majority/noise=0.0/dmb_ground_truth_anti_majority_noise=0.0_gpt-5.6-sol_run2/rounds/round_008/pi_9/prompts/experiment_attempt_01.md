# experiment_attempt_01

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
**Description:** Channel-Specific Coherence Integration theory: binary counts and cue reliabilities are encoded by distinct cognitive channels before being combined. The count channel rapidly computes the difference between option-level numbers of marked ratings. It depends only on that signed margin, so adding mutually canceling cues does not dilute count evidence. Its response is smooth and bounded, reflecting saturation of an approximate magnitude representation rather than unlimited linear accumulation. The reliability channel instead binds each discriminating cue's identity to its communicated validity. It first removes the validity component redundant with the count margin and then measures the signed coherence of the remaining cue-validity contributions. A reliability residual supported by mutually consistent contributions receives high gain, whereas the same residual embedded in opposing contributions receives lower gain because maintaining the required cue-identity bindings is difficult. This loss is determined by evidence coherence and communicated validity-profile contrast, not by feature count, diagnostic-cue count, or a generic load threshold. Stable subject differences govern relative allocation to the two channels and response precision. A stable semantic-polarity belief determines whether marks are interpreted as adverse or beneficial, and a small lapse process captures stimulus-independent choices.

**Parameters:**
- validities: validities
- reliability_style: [0.0, 1.0]
- count_gain: [1.9, 3.0]
- count_scale: [2.8, 4.8]
- reliability_floor: [0.08, 0.30]
- coherent_binding_gain: [0.25, 0.78]
- coherence_power: [0.65, 1.65]
- profile_contrast_gain: [0.0, 0.45]
- response_precision: [0.68, 1.48]
- adverse_polarity_confidence: [0.80, 1.0]
- lapse_rate: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Channel-Specific Coherence Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    count_difference = float(np.sum(difference))

    # Stable integration style allocates expression strength between two
    # separately computed channels. The smooth transformation allows broad
    # subject heterogeneity without imposing discrete strategy classes.
    raw_style = float(parameters["reliability_style"])
    style = 0.5 + 0.5 * np.tanh(2.2 * (2.0 * raw_style - 1.0)) / np.tanh(2.2)
    count_allocation = 1.30 - 0.60 * style
    reliability_allocation = 0.35 + 1.35 * style

    # The count channel reads only the option-level mark difference. Hence
    # mutually canceling added cues have exactly no effect. Saturation is
    # gradual and smooth rather than a hard margin rule or load-dependent
    # divisive compression.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        float(parameters["count_gain"])
        * count_allocation
        * np.tanh(count_difference / count_scale)
    )

    # Reliability is represented as evidence orthogonal to ordinary tallying.
    # Centering ensures that a common shift of all communicated validities
    # cannot masquerade as additional count evidence.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    absolute_support = float(np.sum(np.abs(signed_components)))

    # Coherence compares the net residual with all reliability evidence that
    # had to be identity-bound. It is one for uniformly directed support and
    # approaches zero when a fixed residual is embedded in opposing groups.
    # It contains no explicit count-margin, display-size, or cue-count term.
    if absolute_support > 1e-12:
        coherence = abs(reliability_residual) / absolute_support
    else:
        coherence = 0.0
    coherence = float(np.clip(coherence, 0.0, 1.0))

    # Communicated profiles with meaningful reliability contrast make identity
    # bindings more distinguishable. This is a global property of the stated
    # profile, not a trial-level overload proxy.
    profile_mean = float(np.mean(validities))
    profile_spread = float(np.sqrt(np.mean((validities - profile_mean) ** 2)))
    profile_range = float(np.max(validities) - np.min(validities))
    if profile_range > 1e-12:
        profile_contrast = np.clip(2.0 * profile_spread / profile_range, 0.0, 1.0)
    else:
        profile_contrast = 0.0

    coherence_power = float(parameters["coherence_power"])
    coherent_access = coherence ** coherence_power
    profile_factor = (
        1.0
        + float(parameters["profile_contrast_gain"])
        * profile_contrast
    )

    # A nonzero floor permits reliability sensitivity even for diffuse fields,
    # while the binding term produces selective attenuation when opposing
    # reliability contributions proliferate.
    reliability_gain = (
        float(parameters["reliability_floor"])
        + float(parameters["coherent_binding_gain"])
        * coherent_access
        * profile_factor
    )
    reliability_signal = (
        reliability_allocation
        * reliability_gain
        * reliability_residual
    )

    # The independently summarized channels meet only after their respective
    # transformations. Precision is stable within subject and introduces
    # persistent heterogeneity in the expression of both summaries.
    markedness_evidence = float(parameters["response_precision"]) * (
        count_signal + reliability_signal
    )

    # Positive markedness indicates that A carries more adverse evidence and
    # therefore favors B under the dominant adverse-mark interpretation.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # Semantic polarity is stable across trials. The alternative beneficial
    # interpretation reverses the complete markedness representation and thus
    # preserves exact A/B reversal symmetry.
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
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Relative-Evidence Conflict Arbitration theory: people construct two density-invariant summaries of a binary-feature display. A signed count accumulator represents only the difference in the numbers of marks carried by the options. It grows approximately linearly across ordinary margins and saturates only at relatively large margins, so adding equal numbers of opposing marks cannot dilute count evidence. In parallel, communicated validities are centered around their profile mean and bound to cue directions. Access to this reliability residual is governed by its absolute magnitude and signed coherence: consistently directed reliability contributions become accessible, whereas additions containing genuinely opposing contributions reduce access by increasing gross support without increasing the residual. Accessibility never depends directly on display size, the number of discriminating cues, or active-validity dispersion. A shallow, band-limited calibration discounts intermediate reliability magnitudes but recovers for genuinely strong residuals, representing temporary ambiguity in mapping moderately accumulated reliability evidence onto a categorical preference rather than generic overload. The two summaries are combined by relative-evidence arbitration. Congruent summaries receive modest mutual reinforcement. During conflict, balanced channels inhibit confidence, while sufficiently asymmetric conflicts recruit a winner-enhancement process favoring whichever signed summary is stronger. Thus pure count problems show robust margin growth, but structured count-reliability conflicts can be compressed or reversed without invoking density overload. After arbitration, a mild bounded-confidence transformation imposes diminishing decisional returns on repeatedly accumulated coherent evidence without altering the raw channel ratio used to resolve conflicts. Stable but moderate differences in channel balance and response precision coexist with stable semantic polarity and occasional lapses.

**Parameters:**
- validities: validities
- channel_balance: [0.0, 1.0]
- count_gain: [0.92, 1.18]
- count_scale: [7.0, 11.5]
- reliability_floor: [0.14, 0.32]
- coherence_gain: [0.66, 1.00]
- coherence_power: [1.00, 1.70]
- magnitude_half_saturation: [0.08, 0.28]
- reliability_gain: [0.72, 1.16]
- reliability_notch_center: [1.6, 2.2]
- reliability_notch_width: [0.45, 0.80]
- reliability_notch_depth: [0.12, 0.26]
- congruent_synergy: [0.0, 0.14]
- conflict_compression: [0.35, 0.95]
- winner_enhancement: [0.18, 0.62]
- arbitration_curvature: [0.65, 1.35]
- decision_bound: [2.8, 4.8]
- response_precision: [0.82, 1.22]
- adverse_polarity_confidence: [0.84, 0.98]
- lapse_rate: [0.0, 0.10]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Relative-Evidence Conflict Arbitration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    count_difference = float(np.sum(difference))

    # Stable channel balance produces moderate individual differences without
    # creating discrete strategy classes or trial-dependent attention shifts.
    raw_balance = float(parameters["channel_balance"])
    centered_balance = np.tanh(1.6 * (2.0 * raw_balance - 1.0)) / np.tanh(1.6)
    count_allocation = 1.0 - 0.14 * centered_balance
    reliability_allocation = 1.0 + 0.22 * centered_balance

    # The count accumulator reads only the signed mark margin. Its large scale
    # gives approximately linear growth through ordinary margins and delays
    # saturation. Equal and opposite added marks leave it exactly unchanged.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        count_allocation
        * float(parameters["count_gain"])
        * count_scale
        * np.tanh(count_difference / count_scale)
    )

    # Reliability information is the component of validity-weighted evidence
    # not already represented by an undifferentiated count. Centering also
    # prevents the common validity level from acting as a second tally.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    gross_reliability_support = float(np.sum(np.abs(signed_components)))

    # Coherence is determined solely by the geometry of signed reliability
    # contributions. Consistent contributions have coherence one; opposing
    # additions increase gross support and reduce coherence even if the net
    # residual is unchanged. No cue-count or display-density term appears.
    if gross_reliability_support > 1e-12:
        coherence = abs(reliability_residual) / gross_reliability_support
    else:
        coherence = 0.0
    coherence = float(np.clip(coherence, 0.0, 1.0))

    residual_magnitude = abs(reliability_residual)
    half_saturation = float(parameters["magnitude_half_saturation"])
    magnitude_access = residual_magnitude / (
        residual_magnitude + half_saturation + 1e-12
    )

    accessibility = (
        float(parameters["reliability_floor"])
        + float(parameters["coherence_gain"])
        * (coherence ** float(parameters["coherence_power"]))
        * magnitude_access
    )

    # Intermediate reliability magnitudes receive a shallow ambiguity cost,
    # but accessibility recovers for genuinely strong residuals. This smooth
    # notch depends only on signed reliability magnitude, not display density.
    notch_center = float(parameters["reliability_notch_center"])
    notch_width = float(parameters["reliability_notch_width"])
    notch = np.exp(
        -0.5 * ((residual_magnitude - notch_center) / notch_width) ** 2
    )
    magnitude_calibration = (
        1.0 - float(parameters["reliability_notch_depth"]) * notch
    )

    reliability_signal = (
        reliability_allocation
        * float(parameters["reliability_gain"])
        * accessibility
        * magnitude_calibration
        * reliability_residual
    )

    c_abs = abs(count_signal)
    r_abs = abs(reliability_signal)
    total_abs = c_abs + r_abs

    if total_abs <= 1e-12:
        integrated_signal = 0.0
    elif count_signal * reliability_signal > 0.0:
        # Agreement is mildly superadditive but remains a function of relative
        # signed support, not the number of displayed or active features.
        agreement_balance = 4.0 * c_abs * r_abs / (total_abs * total_abs + 1e-12)
        integrated_signal = (count_signal + reliability_signal) * (
            1.0
            + float(parameters["congruent_synergy"]) * agreement_balance
        )
    elif count_signal * reliability_signal < 0.0:
        # Balanced conflict compresses confidence. Asymmetric conflict recruits
        # winner enhancement, permitting either robust count choices or
        # reliability-driven reversals according to relative evidence alone.
        conflict_balance = 4.0 * c_abs * r_abs / (
            total_abs * total_abs + 1e-12
        )
        relative_dominance = abs(c_abs - r_abs) / (total_abs + 1e-12)
        dominance_gate = relative_dominance ** float(
            parameters["arbitration_curvature"]
        )

        compressed_sum = (count_signal + reliability_signal) / (
            1.0
            + float(parameters["conflict_compression"])
            * conflict_balance
        )
        winner_signal = count_signal if c_abs >= r_abs else reliability_signal
        winner_bonus = (
            float(parameters["winner_enhancement"])
            * conflict_balance
            * dominance_gate
            * winner_signal
        )
        integrated_signal = compressed_sum + winner_bonus
    else:
        integrated_signal = count_signal + reliability_signal

    # Arbitration uses the unbounded channel summaries above, preserving the
    # raw count-to-reliability ratio in structured conflicts. Only the final
    # decisional representation has diminishing returns, so repeated coherent
    # contributions cannot increase confidence without limit.
    decision_bound = float(parameters["decision_bound"])
    integrated_signal = decision_bound * np.tanh(
        integrated_signal / decision_bound
    )

    markedness_evidence = (
        float(parameters["response_precision"]) * integrated_signal
    )

    # Positive markedness means A carries the greater adverse burden and thus
    # favors B. Semantic polarity is fixed across the entire subject run.
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
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
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

[9] This 17-feature design isolates the competing theory's saturating count representation while eliminating its validity channel. The validity profile has mean 0.60 and is partitioned into reliability-balanced groups: one five-expert group and six two-expert groups, each with average validity exactly 0.60. Within a trial, every expert in a group favors the same option. Consequently, each group's centered-validity contributions cancel, so dot(validities-mean(validities), A-B)=0 on every trial. All 17 experts discriminate in every condition, holding discrimination fraction, conflict normalization, display load, and discrimination load fixed apart from the count-margin term itself. Across the six base trials, A's mark-count advantage is 1, 3, 5, 7, 9, or 13. Semantic-Polarity Smooth Integration therefore has integrated markedness exactly d*[count_weight + 0.60*(1-partial_centering)*validity_weight], where d is the count margin: latent evidence grows linearly across the entire series, predicting progressively more choices of B, the less-marked option, especially beyond margin 3 for subjects not already at their polarity ceiling. Capacity-Calibrated Dual-Channel Integration instead has zero validity signal and count evidence proportional to tanh(d/count_saturation_scale). Because all cues discriminate, this response is already at least tanh(3/1.2)=0.987 at margin 3 and changes by less than 1.4% of its unsaturated scale between margins 3 and 13. It therefore predicts an abrupt margin-1-to-margin-3 change followed by an almost flat choice profile, with bounded evidence at large margins. The critical quantitative dissociation is thus continued large-margin growth under smooth additive integration versus near-complete saturation after margin 3 under dual-channel integration. Every base pair is also A/B reversed to balance response side. Twelve unique pairs repeated eight times yield exactly 96 trials.

[10] This 37-feature design isolates the advocated theory's steep display-load attenuation of validity evidence while eliminating count evidence in both theories. The validity profile is symmetric around 0.75: 18 high-validity experts (0.99 to 0.82), one mean-validity expert, and 18 low-validity experts (0.51 to 0.68). In every trial A and B have exactly the same number of marks, so the count difference is zero. Across six base conditions, A is marked by the first m high-validity experts and B by their symmetrically matched low-validity experts, for m = 1, 2, 4, 7, 11, and 18. The resulting count-residualized validity differences are +0.48, +0.94, +1.80, +2.94, +4.18, and +5.58. All pairs are also A/B reversed. Smooth Integration therefore predicts a pronounced monotonic increase in choices of the option whose marks are concentrated on low-validity experts. At the strongest endpoint its evidence is at least 0.7*5.58 = 3.906; even with its lowest adverse-polarity confidence and highest lapse rate, it predicts approximately P(validity-favored option) >= 0.74. The advocated theory applies a 37-feature attention factor no larger than 1/(1+(37/16)^5), leaving endpoint evidence below approximately 0.22 even under maximal validity attention and response precision. It consequently predicts P(validity-favored option) below approximately 0.555 and an almost flat, near-indifferent profile. Thus the diagnostic result is a large graded validity effect under Smooth Integration versus severe compression toward chance under the advocated load-capacity theory. Twelve unique pairs repeated eight times yield exactly 96 trials.

[11] This 15-feature design creates a conflict between a fixed validity residual and count margins of 0, 1, 2, 3, 4, and 5. The validity profile has mean 0.75. In every base trial, A is marked by four low-validity experts (0.55, 0.60, 0.65, 0.70), while B is marked by their four high-validity counterparts (0.95, 0.90, 0.85, 0.80). This produces a constant centered-validity residual of -1.00 and an initially tied count. Successive conditions add marks to A on up to five exactly mean-validity experts. These marks increase the A-minus-B count margin without changing the competitor's centered-validity signal. Under Semantic-Polarity Smooth Integration, markedness evidence is exactly -validity_weight + d*[count_weight + 0.75*(1-partial_centering)*validity_weight], where d is the count margin. Its evidence therefore increases linearly at every step, with a slope between 0.7475 and 2.265; it is necessarily positive by margin 2 and rises by at least 2.99 evidence units from margin 1 to margin 5. The competing theory instead predicts an abrupt reversal from the negative validity signal at margin 0 to positive markedness at margin 1: even its weakest first-margin count component exceeds its strongest possible opposing validity component at 15 features. Thereafter its tanh count accumulator is nearly saturated. From margin 1 to margin 5, its unscaled count signal can increase by no more than about 0.16, while its validity signal remains exactly fixed. Thus the critical quantitative signature is a strong, sustained margin-1-to-margin-5 increase in choices of B under Smooth Integration versus an early margin-1 step followed by an almost flat profile under Dual-Timescale Margin Integration. All six pairs are also A/B reversed. The 12 unique pairs repeated eight times yield exactly 96 trials.

[12] This 32-feature design holds both count evidence and the competing theory's validity-weighted evidence exactly constant while selectively degrading the advocated theory's reliability organization and then inducing diagnostic overload. The validity profile consists of 16 high/low pairs symmetric around mean 0.75. In every base trial, A and B have equal numbers of marks, and the first two high-validity experts favor A while their low-validity partners favor B, producing a constant centered-validity residual of +0.94. Successive conditions add four discriminating cues: both members of one symmetric validity pair favor A, and both members of the next symmetric pair favor B. Each addition has zero net count and zero net centered-validity contribution. Thus the number of discriminating cues increases from 4 to 8, 12, 16, 20, 24, 28, and 32 while count_difference remains 0 and dot(validities,A-B) remains +0.94. Semantic-Polarity Smooth Integration consequently predicts exactly the same B-choice probability in all eight conditions for every parameter setting. Resource-Adaptive Diagnostic Field Integration instead predicts a graded decline in B choices. Its reliability residual stays +0.94, but organization falls from 1.00 at four discriminating cues to about 0.18 at 32 cues because the added evidence is internally canceling. Reliability amplification therefore declines, and beyond roughly 20 discriminating cues the overload-dependent divisor further compresses this diffuse evidence. The clustered high-versus-low validity profile keeps the intended organization mechanism accessible rather than confounding it with a uniformly graded profile. The critical signature is therefore a flat profile under Smooth Integration versus decreasing preference for B—the option favored by the fixed reliability residual—as discriminating, mutually canceling cues accumulate, with additional attenuation after 20 cues. Every base pair is also A/B reversed. Sixteen unique pairs repeated six times yield exactly 96 trials.

[13] This 35-feature design isolates the competing theory's diagnostic-overload compression of count evidence while holding every input to Semantic-Polarity Smooth Integration exactly constant. The validity profile has mean 0.75: seven mean-validity experts plus fourteen high/low pairs symmetric around 0.75. In every base trial, the seven mean-validity experts mark A, giving A a fixed count advantage of 7 and a validity-weighted difference of 5.25. Successive conditions activate two additional symmetric pairs: both members of one pair mark A and both members of the other pair mark B. Each four-cue addition contributes zero net count and zero centered-validity evidence. Thus the number of discriminating cues rises from 7 to 11, 15, 19, 23, 27, 31, and 35, while count_difference remains +7, dot(validities,A-B) remains 5.25, and the competitor's reliability residual remains exactly zero. Semantic-Polarity Smooth Integration therefore predicts exactly the same B-choice probability in all eight conditions for every parameter setting. Resource-Adaptive Diagnostic Field Integration has no reliability signal here, but once diagnostic demand exceeds its span of approximately 20–23 cues, overload divisively compresses its large-margin count signal. At 35 discriminating cues, even the weakest overload-compression setting produces a several-fold count divisor; attenuation of precision heterogeneity cannot offset it. The competitor consequently predicts a relatively stable profile through sparse conditions followed by a pronounced decline toward indifference in choices of B, the option with fewer marks, across the dense conditions. This late overload knee, versus exact invariance under the advocated theory, is the critical quantitative dissociation. Every base pair is also A/B reversed. Sixteen unique pairs repeated six times yield exactly 96 trials.

[14] This 35-feature, 2-by-4 factorial design isolates the advocated theory's margin-selective diagnostic-overload compression. The validity profile contains seven mean-validity experts (0.75) and fourteen high/low pairs symmetric around 0.75. Base conditions cross A-minus-B count margins of 1, 3, 5, and 7 with sparse versus dense diagnostic fields. In sparse trials, only the required number of mean-validity experts favors A. In dense trials, 28 additional cues discriminate: seven complete high/low pairs favor A and seven complete high/low pairs favor B. These additions contribute zero net count and zero centered-validity residual. Consequently, within each count margin, Channel-Specific Coherence Integration predicts exactly identical sparse and dense choice probabilities: its count channel depends only on the signed margin, and its reliability signal is exactly zero. Resource-Adaptive Diagnostic Field Integration instead predicts a diagnostic-density penalty whose magnitude increases sharply with count margin. Dense trials contain 29, 31, 33, or 35 discriminating cues and therefore exceed its diagnostic span, but its powered margin activation makes overload compression relatively weak at margin 1 and progressively stronger at margins 3, 5, and 7. The critical signature is thus a density-by-margin interaction: little sparse-to-dense attenuation at margin 1 but a pronounced movement toward indifference at larger margins. The competing theory predicts a zero density effect at every margin, although it may show ordinary count saturation across margins. All eight base trials are also A/B reversed, balancing response side. Sixteen unique pairs repeated six times yield exactly 96 trials.

[15] This 18-feature design isolates active reliability-contrast dilution while holding the advocated theory's two evidence channels exactly constant. The validity profile contains one high-validity expert (0.95), one low-validity expert (0.55), and sixteen mean-validity experts (0.75), giving a profile mean of 0.75. In every base condition, the 0.95 expert marks A and the 0.55 expert marks B. Thus count difference is zero, centered-validity residual is +0.40, absolute reliability support is 0.40, and coherence is exactly 1. Successive conditions activate zero, one, ..., seven pairs of mean-validity experts, with one member of each pair favoring A and the other favoring B. These additions preserve zero count difference and contribute exactly zero centered-validity evidence. Channel-Specific Coherence Integration therefore predicts exactly the same choice probability in all eight conditions for every parameter setting: its count signal remains zero, while reliability residual, support, coherence, and the fixed profile-contrast factor are unchanged. Resource-Adaptive Diagnostic Field Integration instead predicts decreasing preference for B, the option favored by the positive reliability residual. Although its residual and organization remain fixed, the standard deviation of centered validities among discriminating cues falls from 0.20 with 2 active cues to about 0.071 with 16 active cues. Its contrast salience consequently falls from about 0.667 to 0.414, reducing diagnostic reliability attention. All conditions remain below the model's 20–23-cue diagnostic span, so this tests active-contrast dilution rather than overload. The critical quantitative dissociation is exact invariance under the advocated theory versus a smooth movement toward indifference under the competing theory as mean-validity, mutually canceling discriminators are added. Eight A/B-reversed trials balance response side, producing 16 unique pairs repeated six times for exactly 96 trials.

[16] This 11-feature, 2-by-2-by-2 design targets the advocated theory's conflict arbitration rather than its previously tested reliability-calibration notch. The validity profile contains four high/low pairs (0.90/0.60) and three mean-validity experts (0.75), so its mean is exactly 0.75. High-versus-low paired discriminators generate coherent centered-validity residuals of either +0.60 or -0.60 when two pairs are active and +1.20 or -1.20 when all four pairs are active. Every nonzero reliability contribution within a trial has the same sign, so reliability coherence is exactly 1. Independently, either one or all three mean-validity experts favor A, producing A-minus-B count margins of +1 or +3 without changing the reliability residual. The eight base trials therefore cross count margin (1 versus 3), reliability magnitude (0.60 versus 1.20), and count-reliability relation (congruent versus conflicting). Channel-Specific Coherence Integration is additively separable on its latent markedness scale: E(d,R)=C(d)+K*R. Consequently, the latent contrast between +R and -R must be identical at count margins 1 and 3, and the margin-1-to-margin-3 count increment must be identical for positive and negative residuals. Relative-Evidence Conflict Arbitration predicts violations of both restrictions. Congruent channels receive synergy, whereas opposing channels undergo balance-dependent compression and winner enhancement. These effects should be strongest for the margin-1, magnitude-1.20 conflict, where count and reliability are comparatively balanced, and weaker at margin 3, where count evidence is more dominant. The diagnostic outcome is thus a localized conflict kink or reversal and a nonzero count-margin-by-reliability-sign interaction after fitting the common response link, not merely an overall validity effect. All eight pairs are also A/B reversed, balancing response side and constraining semantic polarity. Sixteen unique pairs repeated six times yield exactly 96 trials.

[17] This 24-feature design isolates magnitude-dependent reliability access and the competitor's intermediate-magnitude calibration notch while eliminating count evidence and reliability-coherence changes. The validity profile consists of 12 high/low pairs symmetric around 0.75. In each base condition, an increasing prefix of pairs discriminates: the higher-validity member marks A and the lower-validity member marks B. Consequently, A and B always have equal mark counts, every nonzero centered-validity contribution has the same positive sign, and reliability coherence is exactly 1. The centered-validity residual increases through 0.10, 0.20, 0.40, 0.60, 0.90, 1.20, 1.60, 2.00, 2.48, 2.96, 3.44, and 3.92. Channel-Specific Coherence Integration therefore predicts an exactly proportional latent reliability signal: for each subject, reliability_signal/R is constant across all 12 conditions, with only the common response mapping producing bounded observable probabilities. Relative-Evidence Conflict Arbitration necessarily violates that proportionality. Its magnitude-access term strongly increases reliability_signal/R between small and moderate residuals, its notch selectively discounts residuals around 1.6–2.2, and access recovers beyond that band. It thus predicts a curved profile with accelerating per-unit sensitivity at low magnitudes, localized attenuation around the intermediate conditions, and recovery at high magnitudes. The critical analysis fits each theory's common polarity, lapse, precision, and bounded response mapping and tests the residual-magnitude profile: a single linear latent trajectory supports the advocated theory, whereas the competitor predicts magnitude-specific curvature even though count margin and coherence never change. Every base pair is also A/B reversed. The 24 unique pairs are each repeated four times, yielding exactly 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This 48-feature, 4-by-2 design tests a quantitative interaction between reliability-residual magnitude and reliability coherence while eliminating count evidence. The validity profile contains 24 identical high/low pairs (0.80/0.70), with mean validity 0.75. A coherently oriented pair has the high-validity expert mark A and the low-validity expert mark B, contributing +0.10 to the centered-validity residual; reversing that pair contributes -0.10. Every active pair gives each option one mark, so the count difference is exactly zero throughout. Residual magnitudes are 0.10, 0.20, 0.40, and 0.80. In coherent conditions, respectively 1, 2, 4, or 8 pairs contribute positively, giving coherence 1. In matched diffuse conditions, 2n pairs contribute positively and n pairs negatively, preserving residual n*0.10 but producing gross support 3n*0.10 and hence coherence exactly 1/3. Channel-Specific Coherence Integration predicts multiplicative separability on its latent evidence scale: at every residual magnitude, the diffuse-to-coherent reliability-signal ratio is exactly [floor + coherent_gain*(1/3)^power*profile_factor]/[floor + coherent_gain*profile_factor], independent of residual magnitude. Relative-Evidence Conflict Arbitration violates this invariance because its coherence-dependent component is multiplied by magnitude access R/(R+half_saturation), while its reliability floor is not. It therefore predicts relatively little coherence attenuation at R=0.10 but progressively greater proportional attenuation as R rises toward 0.80; its calibration term cannot remove this within-magnitude ratio because it multiplies both coherence conditions equally. The critical test fits each theory's common polarity, lapse, precision, and response bound and asks whether the latent diffuse/coherent ratio is constant across residual magnitude, as advocated, or systematically decreases with magnitude, as the competitor predicts. The final eight trials swap A and B, balancing response side. Sixteen unique pairs repeated six times yield exactly 96 trials.",
  "validities": [
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7,
    0.8,
    0.7
  ],
  "trial_a_ratings": [
    [
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
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
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
      0,
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
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
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
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
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
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0,
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
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      0,
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
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
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
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
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
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0,
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
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      0,
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
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
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
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
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
      0,
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
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 11717,
  "output_tokens": 5917
}
```
