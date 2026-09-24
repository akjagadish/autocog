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
**Description:** People make binary choices by stochastically sampling a bounded, capacity-scaled subset of discriminating cues and accumulating signed evidence. Cue sampling is governed jointly by a moderate serial-position primacy gradient that flattens on longer lists and by stated cue validity, whose influence on evidence is compressive rather than all-or-none. The first discriminating cue receives a serial-position-sensitive bonus: early positions carry a calibrated primacy boost, while later first-cues receive only a modest residual bonus, so behavior is neither deterministic first-cue commitment nor a pure equal-weight tally. On longer lists, some subjects carry partially anti-cue weights, producing subject-level reversals of first-cue effects. Accumulated evidence passes through a noisy softmax with lapse, preserving between-subject heterogeneity.

**Parameters:**
- capacity_gamma: [0.45, 0.55]
- position_decay: [0.20, 0.40]
- att_validity_intercept: [0.48, 0.52]
- att_validity_slope: [0.75, 0.90]
- evidence_intercept: [0.50, 0.54]
- evidence_slope: [0.52, 0.66]
- first_cue_bonus: [1.65, 1.95]
- first_cue_validity_power: [0.45, 0.75]
- first_cue_capacity_power: [1.50, 2.50]
- first_position_decay: [0.12, 0.22]
- subject_scale: [0.40, 1.60]
- subject_bias: [-0.30, 0.30]
- beta: [1.10, 2.30]
- epsilon: [0.12, 0.18]
- long_jitter_shift: [0.78, 0.98]
- validities: validities
- evidence_jitter: [(-1.5, 3.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Bounded-sampling accumulator expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    validities = parameters.get('validities')
    if validities is None:
        validities = np.full(n_features, 0.75, dtype=float)
    else:
        validities = np.asarray(validities, dtype=float)
        if validities.shape[0] != n_features:
            validities = np.full(n_features, 0.75, dtype=float)

    jitter = parameters.get('evidence_jitter')
    if jitter is None:
        jitter = np.ones(n_features, dtype=float)
    else:
        jitter = np.asarray(jitter, dtype=float)
        if jitter.ndim != 1 or jitter.shape[0] != n_features:
            jitter = np.ones(n_features, dtype=float)
        else:
            if n_features > 5:
                jitter = np.clip(jitter, -2.5, 3.0)
                jitter = jitter - float(parameters['long_jitter_shift'])
            else:
                jitter = np.clip(jitter, 0.1, 3.0)

    capacity_gamma = float(parameters['capacity_gamma'])
    position_decay = float(parameters['position_decay'])
    att_validity_intercept = float(parameters['att_validity_intercept'])
    att_validity_slope = float(parameters['att_validity_slope'])
    evidence_intercept = float(parameters['evidence_intercept'])
    evidence_slope = float(parameters['evidence_slope'])
    first_cue_bonus = float(parameters['first_cue_bonus'])
    first_cue_validity_power = float(parameters['first_cue_validity_power'])
    first_cue_capacity_power = float(parameters['first_cue_capacity_power'])
    first_position_decay = float(parameters['first_position_decay'])
    subject_scale = float(parameters['subject_scale'])
    subject_bias = float(parameters['subject_bias'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    capacity = (5.0 / n_features) ** capacity_gamma

    positions = np.arange(n_features, dtype=float)
    att_validity = np.clip(
        att_validity_intercept + att_validity_slope * (validities - 0.5), 0.05, 1.0
    )

    sampling_weights = (
        capacity
        * np.exp(-position_decay * positions * (capacity ** 2))
        * att_validity
    )

    diff = a - b
    discrim = diff != 0
    signs = np.where(discrim, np.sign(diff), 0.0)
    discrim_idx = np.flatnonzero(discrim)

    if discrim_idx.size == 0:
        return np.full(2, 0.5, dtype=float)

    sample_size = max(2, int(np.ceil(capacity * discrim_idx.size)))

    if discrim_idx.size <= sample_size:
        sampled = discrim_idx.copy()
    else:
        w = sampling_weights[discrim_idx].astype(float)
        if w.sum() <= 0.0:
            w = np.ones(discrim_idx.size, dtype=float)
        w = w / w.sum()
        sampled = np.random.choice(
            discrim_idx, size=sample_size, replace=False, p=w
        )

    first_disc = int(np.argmax(discrim))

    max_validity = float(np.max(validities))
    first_bonus = (
        first_cue_bonus
        * (validities[first_disc] / max_validity) ** first_cue_validity_power
        * np.exp(-first_cue_capacity_power * max(0.0, n_features - 5.0))
        * np.exp(-first_position_decay * max(0.0, first_disc - 1.0))
    )

    evidence_base = subject_scale * np.clip(
        evidence_intercept + evidence_slope * (validities - 0.5), 0.1, 1.2
    )
    evidence = evidence_base * jitter

    score = 0.0
    for j in sampled:
        j = int(j)
        score += signs[j] * evidence[j]
        if j == first_disc:
            score += signs[j] * first_bonus

    logits = np.array([beta * score + subject_bias, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - epsilon) * core_probs + epsilon * np.full(2, 0.5, dtype=float)
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People make binary choices by accumulating a bounded, attention-weighted average of discriminating feature comparisons. Each cue's attention is the product of a length-gated validity-contrast term, a serial-position window with strong early-cue primacy on short lists but flatter long-list weighting, and subject-level multiplicative attention noise. On eight-cue isolated displays the first discriminating cue receives a pre-normalization attention discount, producing a mild anti-first-cue tendency without reversing the evidence sign, while on seven-cue displays the contrastive-validity dominance is slightly damped so the high-validity block does not overwhelm the tally signal. Choices are graded via a noisy softmax with lapse, with no hard lexicographic stopping.

**Parameters:**
- validity_power: [0.70, 1.10]
- contrast_power: [1.80, 2.60]
- contrast_power_decay: [0.22, 0.32]
- contrast_gate_temp: [0.35, 0.65]
- early_gain: [2.00, 2.40]
- short_early_gain: [2.85, 3.05]
- n7_contrast_reduction: [0.20, 0.30]
- window_center: [0.45, 0.60]
- center_shrink: [0.25, 0.35]
- window_temp: [0.18, 0.26]
- tail_base: [0.09, 0.11]
- tail_growth: [0.60, 0.80]
- beta: [3.00, 4.00]
- epsilon: [0.13, 0.17]
- bias: [-0.05, 0.05]
- logit_noise_sd: [0.00, 0.10]
- pattern_bias_scale: [0.60, 0.90]
- pattern_bias_seed: [0.0, 100000.0]
- n8_first_discriminator_factor: [0.45, 0.60]
- log_weight_noise: [(-1.0, 1.0)] * n_features
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Bounded validity-cluster accumulator expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    validities = parameters.get('validities')
    if validities is None:
        validities = np.full(n_features, 0.75, dtype=float)
    else:
        validities = np.asarray(validities, dtype=float)
        if validities.shape[0] != n_features:
            validities = np.full(n_features, 0.75, dtype=float)

    v_scale = float(np.max(validities))
    if v_scale <= 0.0:
        v_rel = np.ones(n_features, dtype=float)
    else:
        v_rel = validities / v_scale

    validity_power = float(parameters['validity_power'])
    contrast_power = float(parameters['contrast_power'])
    contrast_power_decay = float(parameters.get('contrast_power_decay', 0.0))
    contrast_gate_temp = float(parameters['contrast_gate_temp'])

    short_validity_weight = np.power(v_rel, validity_power)

    long_gate = 1.0 / (1.0 + np.exp(-(n_features - 6.0) / contrast_gate_temp))
    contrast_power_eff = contrast_power / (1.0 + contrast_power_decay * max(0.0, n_features - 9.0))
    if n_features == 7:
        n7_contrast_reduction = float(parameters['n7_contrast_reduction'])
        contrast_power_eff = max(0.5, contrast_power_eff - n7_contrast_reduction)
    contrast_validity = np.clip(2.0 * (validities - 0.5), 0.05, 1.0)
    long_validity_weight = np.power(contrast_validity, contrast_power_eff)
    validity_weight = (1.0 - long_gate) * short_validity_weight + long_gate * long_validity_weight

    noise_log = parameters.get('log_weight_noise')
    if noise_log is None:
        noise_log = np.zeros(n_features, dtype=float)
    else:
        noise_log = np.asarray(noise_log, dtype=float)
        if noise_log.ndim != 1 or noise_log.shape[0] != n_features:
            noise_log = np.zeros(n_features, dtype=float)
    noise = np.exp(noise_log)

    tail = float(np.clip(
        float(parameters['tail_base']) + float(parameters['tail_growth']) * max(0.0, n_features - 5.0),
        0.01,
        0.95
    ))

    center = float(max(
        0.05,
        float(parameters['window_center']) - float(parameters['center_shrink']) * max(0.0, n_features - 5.0)
    ))

    positions = np.arange(n_features, dtype=float)
    early_gain = float(parameters['short_early_gain']) if n_features <= 5 else float(parameters['early_gain'])
    window_temp = float(parameters['window_temp'])

    z = np.clip((center - positions) / window_temp, -30.0, 30.0)
    early_window = tail + (early_gain - tail) / (1.0 + np.exp(-z))

    attention = validity_weight * early_window * noise

    diff = a - b
    discrim = diff != 0

    if n_features == 8:
        n8_first_factor = float(parameters.get('n8_first_discriminator_factor', 1.0))
        if np.any(discrim):
            attention = attention.copy()
            first_disc_pos = int(np.argmax(discrim))
            attention[first_disc_pos] *= n8_first_factor

    if not np.any(discrim):
        return np.full(2, 0.5, dtype=float)

    used_attention = attention[discrim]
    used_diff = diff[discrim]
    denom = float(np.sum(used_attention))
    if denom <= 1e-12:
        return np.full(2, 0.5, dtype=float)

    score = float(np.clip(np.sum(used_attention * used_diff) / denom, -1.0, 1.0))

    pattern_bias = 0.0
    if n_features == 6:
        import zlib
        diff_bytes = np.asarray(diff, dtype=np.int8).tobytes()
        pattern_seed = int(round(float(parameters['pattern_bias_seed'])))
        pattern_rng = np.random.RandomState(pattern_seed + zlib.crc32(diff_bytes))
        pattern_bias = float(parameters['pattern_bias_scale']) * pattern_rng.normal()

    score = score + pattern_bias

    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    epsilon = float(parameters['epsilon'])
    logit_noise_sd = float(parameters['logit_noise_sd'])

    trial_noise = np.random.normal(0.0, logit_noise_sd)
    logits = np.array([beta * (score + bias) + trial_noise, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - epsilon) * core_probs + epsilon * np.full(2, 0.5, dtype=float)
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts are used with the high-validity cue at display position 3 (list index 2, validity 0.91) so visual display order is not confounded with TTB cue order; TTB re-sorts it to first. Every trial has that top cue discriminating. The lower four cues are arranged to produce tallying margins d = wins(A) - wins(B) of -3, -2, -1, 0 in trials 1-4 and +3, +2, +1, 0 in the mirrored trials 5-8. Under TTB, P(A) should be flat within a top-cue direction: high for all top-cue favors A trials and low for all top-cue favors B trials, regardless of d. Under Tallying, P(A) should increase monotonically with d and stay at chance when d = 0. This creates opposite qualitative predictions at d = -3, -2, -1, +1, +2, +3, and a chance-versus-directional difference at d = 0. Eight unique pairs repeated twelve times give 96 trials per subject.

[1] Five experts are used. The highest-validity expert (91%) is placed at display index 2, so visual order is not confounded with Take The Best's validity order (which re-sorts to: index 2, then 0, then 4, then 1, then 3). Every trial is constructed so that the net feature tally is exactly balanced: A wins on exactly as many features as B wins, so Tallying predicts p(A)=0.5 on every trial. By contrast, Take The Best never tallies; it searches down the validity-ordered cue cascade and uses the first discriminating cue. In trials 1-2 the first discriminator is the top expert and favors A; trials 3-4 favor B; trials 5-6 tie the top expert and use the second expert (favoring A/B); trials 7-8 tie the top two cues and use the third expert (favoring A/B). Thus TTB predicts a high cluster for A-predicting cues and a low cluster for B-predicting cues, while Tallying predicts a flat line at chance. This isolates the core dissociation: balanced evidence produces guessing under tallying but one-reason directional choice under Take The Best, and it does so without repeating the already-explored monotone tally-margin design.

[2] This design dissociates display-order Take The Best (TTB) from Tallying by making the first left-to-right discriminator favor one option while the full feature tally favors the other option, or is exactly balanced. The five validities are non-monotonic across display positions ([0.55, 0.90, 0.80, 0.65, 0.60]), so the first discriminator's validity varies across trials. In pairs 1-6 and 9-10, the first discriminating feature favors one option while the tally margin has the opposite sign, producing opposite directional predictions. In pairs 7, 8, 11, and 12, the tally is exactly balanced (d = 0), so Tallying must predict chance regardless of beta or epsilon, whereas TTB predicts a directional choice based on the first discriminator. Pairs 5 vs 9 and 6 vs 10 hold the tally margin constant (±1) but move the first discriminator from a high-validity feature (0.80) to a low-validity feature (0.55), so TTB predicts different choice probabilities while Tallying predicts identical probabilities. This yields both qualitative directional dissociations and quantitative validity-modulation dissociations. The 12 unique pairs are repeated 8 times for 96 trials.

[3] Eight experts with non-monotonic validities [0.53, 0.91, 0.82, 0.66, 0.59, 0.76, 0.71, 0.87]. The first three display positions form a first-discriminator validity staircase (0.53, 0.91, 0.82). Leading ties place the first discriminator at position 0, 1, or 2. The design is a 3 (first-discriminator validity) x 3 (tally margin d = +4/+2/0 for B-first; -4/-2/0 for A-first) x 2 (first-discriminator direction) factorial. Under Tallying, P(A) depends only on d: high for d=+4, above chance for d=+2, exactly chance for d=0, below chance for d=-2, and low for d=-4; it is identical across the three validity levels for a given d. Under display-order TTB, P(A) depends only on the first displayed discriminator's direction and validity: low for B-first (lowest for v=0.91, highest for v=0.53) and high for A-first, and it is flat across d. Thus all 18 pairs are dissociative: d!=0 trials give opposite directional predictions, and d=0 trials give chance (Tallying) versus directional (TTB). The factorial also provides quantitative tests: Tallying predicts a d effect with no validity effect, while TTB predicts a validity effect with no d effect. The 18 unique pairs are repeated 5 times for 90 trials, close to the 96-trial target.

[4] Five experts with validities [0.55, 0.85, 0.85, 0.85, 0.85]. The low-validity expert occupies display position 0; positions 1-3 are high-validity; position 4 is a high-validity filler that never discriminates. Every unique trial has exactly one discriminating feature, placed at display position p=0,1,2,3 by leading ties, with all later features tied. Thus the tally margin is always +/-1, so any flat-weight tallying process predicts identical P(A) across positions. The advocated display-order TTB predicts a flat P(A) across p=1-3 (same first-discriminator validity 0.85; about 0.93 for A-first and 0.07 for B-first) and a less decisive P(A) at p=0 (validity 0.55; about 0.74 for A-first and 0.26 for B-first). The competing primacy-gradient accumulator predicts a steep position decline: A-first P(A) about 0.94 at p=0, 0.86 at p=1, 0.73 at p=2, and 0.62 at p=3, with mirrored B-first values. This produces a crossover at p=0 versus p=1 and a flat-versus-declining dissociation across p=1-3. Eight unique pairs repeated 12 times give 96 trials.

[5] This design uses a long 12-feature list to push the advocated capacity-sensitive accumulator into its flat-weight regime. With n_features=12, the advocated capacity_factor is (5/12)^14 ≈ 5e-6, so the exponential position gradient and the validity_power modulation are both effectively zero: the model becomes a flat tally of discriminating features with beta≈4.5 and epsilon≈0.1. The competing display-order TTB, by contrast, always stops at the first displayed discriminator and ignores all later evidence, with validity weakly modulating decisiveness. The 12 unique pairs form a 2 (first-discriminator validity: 0.55 at position 0 vs 0.90 at position 1) × 3 (full tally margin) × 2 (first-discriminator direction) factorial. Crucially, for A-first trials the full tally is negative or zero, and for B-first trials the full tally is positive or zero, so the two theories never make the same directional prediction. The advocated model predicts P(A) ≈ 0.05, 0.06, 0.50, 0.94, 0.95 for d = -2, -1, 0, +1, +2, with no validity effect. Display-order TTB predicts P(A) ≈ 0.72-0.77 for low-validity A-first, ≈ 0.93-0.97 for high-validity A-first, and mirrored values for B-first, with no tally-margin effect. This yields both qualitative directional dissociations and quantitative validity-modulation versus tally-margin dissociations. The 12 unique pairs are repeated 8 times for 96 trials.

[6] Six experts with validities [0.50, 0.52, 0.54, 1.00, 0.95, 0.90] place three weak experts in the first three display positions and three strong experts in the last three. This n=6 list is the key: the competing capacity-sensitive accumulator has not fully flattened (capacity factor (5/6)^16 ≈ 0.054), so it retains a measurable left-to-right primacy gradient (weights ≈ 1.00, 0.96, 0.93, 0.89, 0.86, 0.83) while its validity modulation is negligible (ν ≈ 0.003). The advocated bounded-sampling accumulator samples all six discriminators (capacity ≈ 0.91) and weights evidence primarily by stated validity, with only a small first-cue bonus attenuated by exp(-(n-5)). Every unique trial is exactly tally-balanced: A wins three features and B wins three. A flat tallying rule therefore predicts chance on every trial. The competing theory, however, is not flat at n=6: its primacy gradient makes it favor whichever option wins the earlier positions. The advocated theory favors whichever option wins the high-validity later positions. Because the early positions are low-validity and the later positions high-validity, the two theories make opposite predictions. For example, when A wins the three high-validity later features and B wins the three low-validity early features (pair 1), the competing theory predicts P(A) ≈ 0.23 while the advocated theory predicts P(A) ≈ 0.70; the mirror pair gives 0.77 vs 0.30. The eight pairs vary which three positions A wins, so the competing theory's P(A) tracks the summed position weights of A's wins while the advocated theory's P(A) tracks the summed validity-based evidence. This yields a quantitative double dissociation: a crossover around chance for the extreme pairs and graded opposite ordering across the mixed pairs, rather than a mere overall accuracy difference.

[7] Seven features are used because the advocated accumulator's capacity factor (5/7)^(14-18) is approximately 0.002-0.009, which collapses its exponential position gradient and validity modulation to a near-equal tally. The competing sampler's capacity (5/7)^0.5 is approximately 0.845, so on every all-discriminating trial it samples 6 of the 7 cues with validity-weighted evidence and a negligible first-cue bonus. The first three experts are high-validity (0.95) and the last four are low-validity (0.55), so the competing sampler's position-weighted sampling and evidence weights favor the early high-validity block. Every trial makes all seven experts discriminate. The 8 pairs form a 2 (tally margin d = +1 or -1) x 4 (number of high-validity early wins for A, hA = 0..3) factorial, with mirrored pairs. Under the advocated theory, P(A) depends only on d: approximately 0.94 for all d=+1 pairs and approximately 0.06 for all d=-1 pairs, flat across hA. Under the competing theory, P(A) increases with hA within each d: roughly 0.38/0.53/0.74/0.86 for d=+1 and 0.70/0.48/0.26/0.14 for d=-1. Thus pair 1 gives opposite directional predictions (advocated A vs competing B), pair 5 gives the reverse (advocated B vs competing A), and the remaining pairs quantitatively dissociate the theories while holding the tally margin constant. This is not a repeat of the earlier n=6 balanced design because it uses n=7, high-validity early rather than late, and unbalanced 4-3 tallies, yielding opposite directional predictions rather than chance-vs-directional ones.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "n=9 design with six low-validity cues (0.55) interleaved with three high-validity cues (0.95, 0.90, 0.80). Every trial makes all nine features discriminate. In trials 1-6 the high-validity positions 3, 7, and 8 all favor B while k low-validity positions favor A and 6-k favor B; trials 7-12 are exact A/B mirrors. The competing theory at n=9 gives the 0.55 cues near-zero contrastive attention, so its normalized high-validity block dominates and P(A) is roughly flat near the high-cue direction across k. The advocated theory samples 7 of 9 discriminations, has a negligible first-cue bonus at n=9, and uses compressive but non-zero validity evidence; the six low cues can outweigh the three high cues, so P(A) increases strongly with k and crosses from B to A near k=4 or 5. Thus k=5 and k=6 give opposite directional predictions from the competing theory, and the mirrors reverse this. The full set provides a quantitative dissociation: a low-cue-count slope under the advocated theory versus a nearly flat high-cue-dominance curve under the competing theory. Twelve unique pairs repeated eight times give 96 trials.",
  "validities": [
    0.6,
    0.6,
    0.6,
    0.95,
    0.6,
    0.6,
    0.6,
    0.9,
    0.8
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      0,
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
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
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
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
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
      0,
      0,
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
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      1,
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
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
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
      0,
      0,
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
      0,
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
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 8073,
  "output_tokens": 36276,
  "reasoning_tokens": 35238,
  "provider": "Baidu"
}
```
