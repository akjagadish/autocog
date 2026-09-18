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
**Description:** Latent Validity-Construal Tally theory proposes that participants first encode each discriminating expert as one categorical vote and form an unweighted vote margin. Communicated validities are represented separately as monotonic diagnostic-strength codes whose signed sum is compressed into bounded reliability evidence. Crucially, participants differ in their interpretation of the communicated validity scale. The dominant latent construal treats lower stated values as safer or more diagnostic, reversing the direction of reliability evidence. Small subgroups either disregard the validity information or interpret it conventionally. Because reliability evidence is bounded, the dominant reversed construal can overturn a one-vote tally or resolve an exact tally, but it cannot overturn a decisive many-versus-few coalition. When reliability opposes a one-vote tally, its influence is selectively attenuated if that tally advantage arises from a compact, proportionally coherent coalition; reliability remains nearly unchanged in diffuse many-versus-many conflicts. Reliability-based commitment applies only when validity overturns a nonzero tally; exact-tally decisions remain graded rather than receiving the same categorical amplification. In exact tallies, reliability is further gated by its relative salience: the absolute signed validity imbalance is compared with the total encoded validity strength on both sides. Diffuse conflicts between two substantial coalitions receive little reliability influence, whereas endpoint contrasts involving a near-chance expert remain salient. The raw tally is also smoothly saturated, reducing excessive confidence differences between moderate and large margins while retaining its dominance in decisive coalitions. The model is reversal-symmetric and history-independent because no outcome feedback is available.

**Parameters:**
- construal_quantile: [0, 1]
- orientation_strength: [1.9, 2.4]
- reliability_accumulation: [1.8, 3.2]
- exact_tie_salience_threshold: [0.45, 0.60]
- exact_tie_salience_slope: [12, 18]
- conflict_consensus_floor: [0.35, 0.60]
- evidence_scale: [2.3, 2.9]
- choice_sensitivity: [1.55, 2.15]
- construal_commitment: [2.1, 2.5]
- lapse_rate: [0.0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Latent Validity-Construal Tally model. History is intentionally ignored:
    # choices without correctness feedback cannot reveal expert accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # +1 denotes an expert uniquely endorsing A, -1 an expert uniquely
    # endorsing B, and zero a nondiscriminating expert.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    tally_margin = float(np.sum(signed_votes))

    # The communicated scale is encoded monotonically before its psychological
    # direction is selected. Near-chance experts retain a categorical vote but
    # supply little reliability evidence. Extreme values are compressed.
    v = np.clip(validities, 0.5, 1.0)
    validity_code = np.tanh(3.0 * (v - 0.5)) / np.tanh(1.5)
    signed_validity_sum = float(np.dot(signed_votes, validity_code))

    # Accumulation is cue-by-cue but saturates, so validity can settle a weak
    # conflict without growing enough to defeat a large tally coalition.
    accumulation = float(parameters["reliability_accumulation"])
    bounded_reliability = float(np.tanh(accumulation * signed_validity_sum))

    # In an exact tally, distinguish a concentrated diagnostic imbalance from
    # a diffuse conflict between two substantial endorsing coalitions. The
    # normalization is used only at exact ties, leaving every nonzero-tally
    # prediction and its successful margin profile unchanged.
    if abs(tally_margin) < 1e-12:
        total_validity_strength = float(
            np.dot(np.abs(signed_votes), validity_code)
        )
        if total_validity_strength > 1e-12:
            relative_salience = np.clip(
                abs(signed_validity_sum) / total_validity_strength, 0.0, 1.0
            )
        else:
            relative_salience = 0.0

        threshold = float(parameters["exact_tie_salience_threshold"])
        slope = float(parameters["exact_tie_salience_slope"])
        raw_gate = 1.0 / (
            1.0 + np.exp(-slope * (relative_salience - threshold))
        )
        endpoint_gate = 1.0 / (
            1.0 + np.exp(-slope * (1.0 - threshold))
        )
        bounded_reliability *= float(raw_gate / endpoint_gate)

    # The continuous latent index implements a task-invariant population
    # mixture. Nearly all participants reverse the instructed scale; very small
    # groups ignore it or use its conventional direction.
    construal_quantile = float(parameters["construal_quantile"])
    if construal_quantile < 0.99:
        validity_orientation = -1.0
    elif construal_quantile < 0.998:
        validity_orientation = 0.0
    else:
        validity_orientation = 1.0

    orientation_strength = float(parameters["orientation_strength"])
    reliability_evidence = (
        validity_orientation * orientation_strength * bounded_reliability
    )

    # In a nonzero one-vote conflict, coalition structure regulates only the
    # opposing reliability signal. A compact proportional majority protects
    # its tally, whereas a diffuse many-versus-many margin leaves validity
    # almost unchanged. Larger tally margins and aligned reliability are not
    # modified, preserving the established margin-confidence profile.
    discriminating_count = int(np.count_nonzero(signed_votes))
    if (
        discriminating_count > 0
        and 0.5 < abs(tally_margin) < 1.5
        and reliability_evidence * tally_margin < 0.0
    ):
        relative_consensus = abs(tally_margin) / float(discriminating_count)
        coherent_conflict = 1.0 / (
            1.0 + np.exp(-24.0 * (relative_consensus - 0.25))
        )
        conflict_floor = float(parameters["conflict_consensus_floor"])
        conflict_gate = 1.0 - (1.0 - conflict_floor) * coherent_conflict
        reliability_evidence *= float(conflict_gate)

    # Smoothly saturating the raw tally reduces excessive margin-dependent
    # confidence while preserving its sign and its dominance for large margins.
    tally_evidence = 3.8 * np.tanh(tally_margin / 3.8)
    total_evidence = tally_evidence + reliability_evidence

    # A reliability-based construal that reverses a weak nonzero tally is
    # treated as a categorical resolution. Exact-tally reliability choices are
    # left graded, avoiding excessive amplification in balanced coalitions.
    reliability_controls = False
    if abs(total_evidence) > 1e-12 and abs(reliability_evidence) > 1e-12:
        if abs(tally_margin) >= 1e-12:
            if np.sign(total_evidence) != np.sign(tally_margin):
                reliability_controls = True

    evidence_scale = float(parameters["evidence_scale"])
    normalized_evidence = float(np.tanh(total_evidence / evidence_scale))
    decision_variable = (
        float(parameters["choice_sensitivity"]) * normalized_evidence
    )
    if reliability_controls:
        decision_variable *= float(parameters["construal_commitment"])

    # Positive evidence favors A. Construct symmetric logits to preserve exact
    # invariance under swapping the two displayed options.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size, 1.0 / probabilities.size, dtype=np.float64
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))
```


## COMPETING THEORY
**Description:** Uncertain-Credibility Normalized Voting theory proposes that people do not treat instructed expert validities as precisely known reliabilities. In the absence of feedback, each communicated validity is shrunk toward a shared, moderately credible prototype. Consequently, every discriminating expert supplies one dominant categorical vote, while the incompletely trusted deviation of that expert's validity from the prototype supplies only a residual adjustment. Extreme communicated deviations are compressed smoothly because participants regard unusually high or low precision claims as especially uncertain, while still retaining limited distinctions among the most credible experts. These residual adjustments are accumulated cue by cue, preserving expert identity, but their total is divisively normalized and increasingly bounded once a clear tally coalition exists. Residual reliability is also contextually gated: it is strongly discounted in exact tally ties, where conflicting reliability claims provide no categorical direction, while otherwise retaining its ordinary residual weight without a special margin-one amplification. Before saturation, total evidence undergoes a moderately convex magnitude transformation, making small vote margins comparatively uncertain while preserving confidence for large coalitions. The resulting evidence is passed through a saturating response transformation, so additional agreement has diminishing effects once evidence is decisive. Between-person differences arise in reliability shrinkage, response sensitivity, and lapse rate. Because choices reveal no correctness information, the representation is not updated from choice history.

**Parameters:**
- reliability_shrinkage: [0.58, 0.84]
- choice_sensitivity: [2.20, 2.70]
- lapse_rate: [0.0, 0.12]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Uncertain-Credibility Normalized Voting. History is intentionally ignored:
    # without outcome feedback, previous choices cannot identify expert accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # A difference of +1 is an endorsement of A and -1 an endorsement of B.
    # Nondiscriminating experts contribute neither a vote nor reliability evidence.
    signed_endorsements = np.sign(stimulus[0] - stimulus[1])
    discriminating = signed_endorsements != 0.0
    tally_evidence = float(np.sum(signed_endorsements))

    # Map validity onto above-chance diagnostic strength. The reference value
    # zero corresponds to a common validity of .75. Lower-tail deviations are
    # winsorized, while upper-tail deviations are smoothly compressed so that
    # very credible experts remain distinguishable without receiving full weight.
    v = np.clip(validities, 0.5, 1.0)
    communicated_strength = 2.0 * (v - 0.5)
    raw_deviation = communicated_strength - 0.5
    lower_bounded = np.maximum(raw_deviation, -0.25)
    upper_excess = np.maximum(lower_bounded - 0.25, 0.0)
    reliability_deviation = (
        np.minimum(lower_bounded, 0.25)
        + 0.13 * np.tanh(upper_excess / 0.10)
    )

    # Strong shrinkage leaves only a small fraction of communicated deviations.
    # Deviations are nevertheless summed expert by expert, unlike a summary of
    # the two endorsing groups. Divisive normalization bounds their joint impact.
    shrinkage = float(parameters["reliability_shrinkage"])
    retained_precision = 1.0 - shrinkage
    active_deviations = reliability_deviation[discriminating]
    active_signs = signed_endorsements[discriminating]

    if active_deviations.size == 0:
        reliability_adjustment = 0.0
    else:
        signed_sum = float(np.dot(active_signs, active_deviations))
        deviation_mass = float(np.sum(np.abs(active_deviations)))
        coalition_protection = 0.12 * max(abs(tally_evidence) - 1.0, 0.0) ** 2
        reliability_adjustment = (
            4.0 * retained_precision * signed_sum
            / (1.0 + retained_precision * deviation_mass + coalition_protection)
        )

        # Reliability is least trusted when the categorical tally is exactly
        # tied. Outside exact ties it receives its ordinary residual weight;
        # there is no separate amplification at a one-vote margin.
        absolute_tally = abs(tally_evidence)
        tie_gate = 1.0 - 0.94 * np.exp(-((absolute_tally / 0.35) ** 2))
        reliability_adjustment *= tie_gate

    total_evidence_for_a = tally_evidence + reliability_adjustment

    # Slightly stronger convexity compresses low evidence further. The revised
    # scale is calibrated to residual-shifted large-margin evidence rather than
    # to a raw tally of exactly five.
    response_curvature = 1.28
    saturation_scale = 4.50
    curved_evidence = np.sign(total_evidence_for_a) * (
        abs(total_evidence_for_a) / saturation_scale
    ) ** response_curvature
    normalized_evidence = np.tanh(curved_evidence)
    sensitivity = float(parameters["choice_sensitivity"])
    decision_variable = sensitivity * normalized_evidence

    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * p_core + lapse * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size, 1.0 / probabilities.size, dtype=np.float64
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This 16-pair, 96-trial design creates both directional and quantitative dissociations. In each of the first eight pairs, the highest-validity discriminating cue favors A; the remaining eight pairs are exact A/B reversals. Therefore Take The Best predicts the same choice probability for its lexicographic winner on every trial: epsilon/2 + (1-epsilon)*sigmoid(beta), regardless of which cue is first discriminating or how many lower-validity cues oppose it. Pairs 1-4 manipulate the rank of the first discriminating cue while producing equal tallies (one cue for each option). Tallying must predict exactly 0.5 on these pairs, whereas Take The Best predicts a systematic preference. In pairs 5-8, the lexicographic winner loses the tally by respective score gaps of 1, 2, 3, and 5. Take The Best continues to predict an invariant preference for that winner, while Tallying predicts the opposite option with monotonically increasing confidence as the gap grows. Exact reversals balance response side and preserve every theoretical contrast. With 16 unique pairs, each is repeated six times.

[1] This design isolates a quantitative confidence signature without relying on opposite directional choices. The highest-validity expert (feature 5, validity 0.95) discriminates on every trial, so Take The Best always stops immediately and predicts the same probability for its winner on all eight pairs: epsilon/2 + (1-epsilon)*sigmoid(beta). The tally winner is always the same option as the Take-The-Best winner, but its feature-win margin is systematically manipulated across four conditions: 4-3 (gap 1), 5-2 (gap 3), 6-1 (gap 5), and 7-0 (gap 7), with two tied features in every condition. Tallying therefore predicts winner probabilities epsilon/2 + (1-epsilon)*sigmoid(beta*d) for d in {1,3,5,7}, producing a strictly increasing choice-consistency gradient except at practical saturation. Take The Best cannot produce this within-subject margin gradient because its binary score difference is always one. Each condition has an exact A/B reversal, balancing response side while preserving the theoretical contrast. There are eight unique pairs, so each is repeated 12 times for approximately 96 trials.

[2] Each base pair gives A one feature win and B two feature wins; the final eight pairs are exact A/B reversals. Tallying therefore predicts exactly the same B-choice probability on all eight base pairs because its tally margin is always 2-1, and the same mirrored A-choice probability on all reversals. Communicated-Reliability Accumulation instead distinguishes the identities of the winning experts. At representative parameter values (gamma about 1.035 and reliability_floor about 0.12), net evidence for A across the eight base pairs is approximately 2.71, 1.80, 1.00, 0.43, -0.10, -0.34, -0.71, and -1.40. Thus it predicts a strong, ordered choice gradient and a crossover from choosing the one-cue minority A to choosing the two-cue majority B. This ordering and crossover are robust over the advocated parameter ranges, whereas no setting of Tallying's beta or epsilon can produce stimulus-specific differences among pairs having the same tally margin. Exact reversals balance response side and should mirror the gradient. With 16 unique pairs, each pair is repeated six times for 96 trials.

[3] This six-pair design makes tally margin increase while communicated-reliability evidence decreases and ultimately reverses direction. In the first three pairs, A's tally margins are respectively 1, 3, and 5: A wins 1-0, 3-0, and 6-1 features. Tallying therefore predicts a strictly increasing A-choice gradient, with probabilities epsilon/2 + (1-epsilon)*sigmoid(beta*d) for d in {1,3,5}. Reliability accumulation predicts the opposite quantitative pattern because the larger tallies are built from progressively weaker experts. At representative parameters (gamma=1.035 and reliability_floor=0.12), its evidence for A is approximately 3.18, 2.44, and -1.04. These evidence values remain ordered across the specified parameter ranges, and the third remains negative: there, Tallying favors A by a five-feature margin, whereas reliability accumulation favors B because the 95%-valid expert outweighs six weak experts. The final three pairs are exact A/B reversals, balancing response side and mirroring both theories' predictions. Thus the critical signatures are an increasing choice-consistency gradient under Tallying versus a decreasing gradient and directional crossover under reliability accumulation. With six unique pairs, each is repeated 16 times for 96 trials.

[4] This eight-pair design holds the signed tally at exactly zero while varying coalition size and composition. In the four base pairs, A is endorsed by one 95%-valid expert plus respectively 0, 1, 2, or 3 near-chance (51%) experts; B is endorsed by respectively 1, 2, 3, or 4 experts of 73.5% validity. Because both sides have equal numbers of endorsers within every pair, the advocated model's common reliability floor cancels. Across its parameter range, its evidence for A is approximately 1.99–2.09, 1.01–1.10, 0.02–0.11, and -0.96–-0.88. It therefore predicts a steep decline in A choices and a robust directional crossover on the fourth condition, where four moderately reliable experts cumulatively outweigh the high expert and three weak experts. Compressed-Reliability Consensus instead assigns every condition a tied tally and compares bounded group summaries. Softmax attention keeps A's group summary dominated by the 95% expert: its reliability contrast remains positive at approximately 0.295, 0.273, 0.252, and 0.232. It consequently predicts A above chance in all four conditions, with only a shallow decline and no crossover. Thus the critical dissociation is a large accumulation-driven gradient ending below chance versus a near-flat, uniformly above-chance consensus pattern. The final four pairs are exact A/B reversals, balancing response side and requiring both patterns to mirror. Eight unique pairs are each repeated 12 times for 96 trials.

[5] This design targets both a robust confidence dissociation and a secondary composition-sensitive gradient. Every base pair gives A seven discriminating endorsements and B two, producing the same tally margin of five. Five of A's endorsements always come from 50%-valid experts. Compressed-Reliability Consensus treats these as full categorical votes and applies its saturated margin-five consensus multiplier; it therefore predicts substantially above-chance A choice. Even at its least confident parameter settings, its predicted A probability is approximately 0.67 or greater. Communicated-Reliability Accumulation gives each 50%-valid expert only the reliability floor, so these five endorsements contribute total evidence of only 5c = 0.50–0.70. The remaining two-versus-two coalitions were selected to have approximately equal summed transformed log-odds weights throughout the accumulation model's narrow gamma range. Consequently, accumulation predicts a nearly invariant and much weaker A preference, approximately 0.53–0.60 across base pairs. The four conditions also increase the dispersion of A's two non-chance reliabilities while preserving their accumulated weight: from 69%/80% to 51%/90%, opposed by two experts near 75%. Softmax compression increasingly emphasizes A's high-reliability member, moving the advocated model's reliability contrast from slightly negative to positive and producing an increasing choice-confidence gradient. Accumulation predicts essentially no such condition gradient because the transformed-weight sums are matched. Exact reversals balance response side and require mirrored predictions. There are eight unique pairs, each repeated 12 times for 96 trials.

[6] This 12-pair design isolates expert-by-expert residual accumulation from bounded group summarization while holding the categorical tally fixed. In each of the six base pairs, A is endorsed by m high-validity (95%) experts and B by m-1 near-chance (51%) experts, for m = 2,...,7. Thus A's tally margin is always exactly one, and both options always have at least one endorser. Compressed-Reliability Consensus predicts the same response probability in every base condition: because each coalition is homogeneous, its softmax-weighted summaries are always the 95% and 51% reliability codes, respectively; the reliability contrast is therefore constant, the tally is always +1, and consensus amplification is inactive. Across its parameter ranges, it predicts an approximately 0.55-0.64 A-choice probability with no coalition-size gradient. Uncertain-Credibility Normalized Voting instead preserves each expert's identity and accumulates residual deviations cue by cue. Every added 95% endorsement for A contributes a positive residual, while every added 51% endorsement for B also shifts residual evidence toward A because that weak B vote is discounted relative to the shared prototype. Divisive normalization slows but does not eliminate this accumulation. Its total evidence for A rises monotonically from approximately 1.54-2.17 at m=2 to 2.58-3.52 at m=7, yielding a substantial within-subject increase in A-choice consistency. The critical signature is therefore a strong monotonic coalition-size gradient under the advocated theory versus an exactly flat profile under the competing theory, despite an identical one-vote tally and identical within-side reliability summaries. The final six pairs are exact A/B reversals, balancing response side and requiring the gradient to mirror. With 12 unique pairs, each is repeated eight times for 96 trials.

[7] This 12-pair design targets the competing theory's exact-tally reliability gate. Every trial has exactly one expert endorsing A and one endorsing B, so the categorical tally is zero. In the six base pairs, A's endorser has validity 0.60, 0.80, or 1.00, whereas B's endorser always has validity 0.50. Each reliability contrast appears in two nuisance contexts: all nondiscriminating experts rate both products 0 or all rate both products 1. Compressed-Reliability Consensus leaves reliability contrasts active in exact ties and therefore predicts an ordered increase in A choices as A's endorser rises from 60% to 80% to 100%. Its softmax group summaries reduce to the two individual endorsers, making the two nuisance contexts exactly equivalent. Uncertain-Credibility Normalized Voting multiplies reliability evidence by only 0.06 in an exact tally tie and then further compresses the resulting tiny evidence through its convex response transformation. It consequently predicts choices extremely close to chance at all three reliability levels and an almost negligible gradient; at 60% versus 50%, lower-tail winsorization makes its signed reliability adjustment exactly zero. Thus the critical quantitative signature is a visible reliability-contrast gradient, reaching a meaningful above-chance preference in the 100%-versus-50% condition, under the advocated theory versus an essentially flat chance-level profile under the competing theory. The final six pairs are exact A/B reversals, balancing response side and requiring the gradient to mirror. With 12 unique pairs, each is repeated eight times for 96 trials.

[8] This eight-pair design targets the theories' opposite use of communicated validity while holding A's categorical tally advantage fixed at 7 endorsements versus 6. The first seven experts have validity .95 and the last seven validity .50. Across the four base pairs, the high-validity endorsement imbalance favoring A is +7, +3, -3, and -6. Under Uncertain-Credibility Normalized Voting, the corresponding signed residual-deviation sums are approximately +4.07, +1.60, -2.10, and -3.96. Its total evidence for A therefore decreases from approximately 2.58–3.53 to 1.62–1.99, then to approximately -0.30–0.19, and finally to -1.50–-0.55 over the reliability-shrinkage range. It predicts a steep conventional-validity gradient: strong A choice in the first conditions, near-chance choice in the third, and a B preference in the fourth despite A's one-vote tally advantage. The competing theory's dominant latent construal reverses validity direction. Its bounded reliability signal is nearly saturated whenever the high-validity imbalance is nonzero; hence it predicts the opposite gradient—B choices for the +7 and +3 conditions because reversed reliability overturns A's weak tally, but confident A choices for the -3 and -6 conditions because reversed reliability aligns with A's tally. The conflicts involve 13 discriminating experts, so the competing theory's compact-coalition attenuation is essentially inactive. Thus the theories predict oppositely signed condition gradients and opposite majority choices at both ends, not merely different overall accuracy. The final four pairs are exact A/B reversals, balancing response side and requiring mirrored effects. Eight unique pairs are each repeated 12 times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This eight-pair design isolates the advocated theory's exact-tally relative-salience gate while holding both the categorical tally and the signed validity imbalance constant. In each of the four base pairs, A and B receive equal numbers of endorsements. A always has one unmatched 100%-valid endorser and B one unmatched 50%-valid endorser, producing the same signed validity sum favoring A. Conditions 2-4 add one, two, or three canceling pairs of 100%-valid endorsers\u2014one for A and one for B per pair. These additions leave the tally at zero and leave the signed validity sum unchanged, but increase total encoded validity strength. Consequently, relative salience under the advocated model falls exactly from 1 to 1/3, 1/5, and 1/7. For the dominant reversed-validity construal, the endpoint condition should therefore produce a substantial B preference despite A owning the 100%-valid endorsement, approximately 0.72-0.79 B choice across typical parameter combinations. Adding one balanced high-validity pair sharply gates this reliability evidence, and the 1/5 and 1/7 conditions should be close to chance. This predicts a pronounced nonlinear endpoint-versus-diffuse gradient even though raw reliability imbalance is identical. Uncertain-Credibility Normalized Voting instead applies its 0.06 exact-tie gate in every condition and then convexly compresses the already tiny residual evidence. It should remain extremely close to chance throughout, with at most a minute conventional preference for A and no sizable endpoint effect. Thus the critical dissociation is a strong reversed-validity choice in the concentrated 1-versus-1 endpoint condition followed by collapse toward chance under diffuse balanced coalitions, versus an essentially flat chance-level profile under the competing theory. The final four pairs are exact A/B reversals, balancing response side. Eight unique pairs are each repeated 12 times for 96 trials.",
  "validities": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.5
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
      0
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
      1,
      1,
      0,
      1,
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
      1
    ],
    [
      0,
      0,
      1,
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
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      1
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
      1
    ],
    [
      0,
      0,
      1,
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
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0,
      1,
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
      0
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
      1,
      1,
      0,
      1,
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
  "input_tokens": 7679,
  "output_tokens": 2819
}
```
