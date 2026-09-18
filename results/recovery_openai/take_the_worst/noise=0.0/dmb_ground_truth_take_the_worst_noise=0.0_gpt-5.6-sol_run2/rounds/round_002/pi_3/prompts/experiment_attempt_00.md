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
**Description:** Rank-Budget Arbitration theory proposes that people initially represent advertised cue validities as an ordered hierarchy and try to make a lexicographic choice using the highest-ranked discriminating cue. Reliable access to that hierarchy is capacity-limited, however. The probability that rank retrieval fails increases smoothly with the observable representational load imposed by the number of available cues and the number of cues that must be compared on the current trial. When rank retrieval succeeds, choice approximates Take The Best and lower-ranked tally direction has little influence. When retrieval fails, cue ranks are compressed into a coarse common category, and the person pools the directions of all discriminating cues with approximately equal weight. Strategy arbitration occurs anew on every trial rather than assigning a person permanently to one heuristic. Individual capacity determines the transition location, gate steepness determines how abrupt it is, evidence sensitivity controls response reliability, and lapses produce occasional uninformed choices. Thus, lexicographic and tally-like rules are limiting regimes of a single adaptive representational mechanism.

**Parameters:**
- n_features: n_features
- validities: validities
- capacity: [9.7, 9.9]
- gate_steepness: [2.0, 2.4]
- evidence_sensitivity: [1.0, 5.0]
- lapse: [0.0, 0.15]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Rank-Budget Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    diff = stim[0] - stim[1]
    directions = np.sign(diff)
    discriminating = np.flatnonzero(directions != 0)

    # Successful rank retrieval: use only the most valid discriminating cue.
    if discriminating.size == 0:
        lexical_delta = 0.0
    else:
        cue_order = np.argsort(-validities, kind="stable")
        lexical_delta = 0.0
        for cue in cue_order:
            if directions[cue] != 0:
                lexical_delta = float(directions[cue])
                break

    # Failed rank retrieval: cue ranks are compressed and directions are pooled.
    # The sign transform prevents larger tallies from receiving mechanically
    # greater response precision than lexicographic evidence.
    pooled_delta = float(np.sign(np.sum(directions)))

    # Besides cue-set size, simultaneous discriminations add comparison load.
    comparison_fraction = float(discriminating.size) / float(max(n_features, 1))
    observable_load = float(n_features) + 0.25 * comparison_fraction

    capacity = float(parameters["capacity"])
    steepness = float(parameters["gate_steepness"])
    gate_logit = steepness * (observable_load - capacity)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    pooling_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    sensitivity = float(parameters["evidence_sensitivity"])

    def binary_softmax(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    p_lexical = binary_softmax(lexical_delta)
    p_pooled = binary_softmax(pooled_delta)
    p_core = ((1.0 - pooling_probability) * p_lexical
              + pooling_probability * p_pooled)

    lapse = float(parameters["lapse"])
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
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Ecological Encoding Hysteresis theory claims that decision makers learn how to encode an entire block of comparisons. They begin with a sparse-ecology prior favoring maintenance of the advertised validity hierarchy. After each comparison, they update a slowly changing estimate of the proportion of cues that discriminate. When discrimination remains sparse and ties are common, cue ranks remain cognitively useful and an ordered-search task set persists: the highest-validity discriminating cue controls choice, even on an occasional dense trial. When comparisons repeatedly contain many simultaneous discriminations, maintaining separate cue ranks becomes inefficient. People then compress cue directions into a relational gist and approximately pool them with equal weights. A steep contextual transition and a ceiling on compression implement hysteresis-like persistence rather than trial-by-trial load arbitration. Individual differences in the contextual threshold, prior strength, compression ceiling, sensitivity, and lapses produce heterogeneous behavior near the transition. Thus, identical diagnostic comparisons can elicit lexical choices in sparse filler blocks but majority choices in saturated filler blocks.

**Parameters:**
- n_features: n_features
- validities: validities
- context_prior: [0.50, 0.60]
- prior_strength: [1.0, 3.0]
- context_threshold: [0.68, 0.71]
- context_steepness: [35.0, 55.0]
- pooling_ceiling: [0.68, 0.76]
- evidence_sensitivity: [1.6, 3.0]
- lapse: [0.0, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Ecological Encoding Hysteresis expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Reconstruct the slowly accumulated block ecology. Each observation is
    # the fraction of cues that discriminate in a comparison; its complement
    # is tie prevalence. A prior prevents one unusual early trial from
    # immediately replacing the initial ordered-search task set.
    density_sum = 0.0
    n_observed = 0
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        density_sum += float(np.mean(a_i != b_i))
        n_observed += 1

    # Viewing the current display supplies another ecological observation,
    # but its effect shrinks as evidence about the block accumulates.
    current_density = float(np.mean(stim[0] != stim[1]))
    density_sum += current_density
    n_observed += 1

    prior_density = float(parameters["context_prior"])
    prior_strength = float(parameters["prior_strength"])
    context_density = (
        prior_strength * prior_density + density_sum
    ) / (prior_strength + float(n_observed))

    # Saturated ecologies trigger a compressed relational encoding. The
    # ceiling allows some ordered processing to survive even in dense blocks.
    threshold = float(parameters["context_threshold"])
    steepness = float(parameters["context_steepness"])
    gate_logit = steepness * (context_density - threshold)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_activation = 1.0 / (1.0 + np.exp(-gate_logit))
    pooling_probability = (
        float(parameters["pooling_ceiling"]) * compression_activation
    )

    directions = np.sign(stim[0] - stim[1])

    # Ordered encoding: consult cues by advertised validity and stop at the
    # first discrimination.
    lexical_delta = 0.0
    cue_order = np.argsort(-validities, kind="stable")
    for cue in cue_order:
        if directions[cue] != 0.0:
            lexical_delta = float(directions[cue])
            break

    # Compressed encoding: retain all cue directions but discard fine-grained
    # rank and magnitude information. Sign normalization puts its response
    # scale on the same footing as the lexical representation.
    pooled_delta = float(np.sign(np.sum(directions)))

    sensitivity = float(parameters["evidence_sensitivity"])

    def choice_probs(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_ordered = choice_probs(lexical_delta)
    p_compressed = choice_probs(pooled_delta)
    p_core = (
        (1.0 - pooling_probability) * p_ordered
        + pooling_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Seven experts with sharply separated advertised validities establish an unambiguous cue hierarchy. The eight structural contrasts are each presented in both A/B orientations, controlling side bias and yielding 16 unique pairs (six repetitions each, for 96 trials). For the first orientation of each contrast, TTB always selects A; for the reversed orientation it always selects B. Critically, TTB predicts the same choice probability on every trial because its binary winner score is invariant to cue rank and to all evidence below the first discriminating cue. Tallying instead predicts distinct patterns. Contrast 1 has one win per option and therefore forces exact 50/50 tallying despite a clear TTB choice. Contrasts 2-4 oppose the highest-validity cue with lower-validity majorities whose tally margins favor the other option by 1, 3, and 5, producing an increasingly strong reversal under tallying but no change under TTB. Contrasts 5-6 make the highest one or two cues tie, so TTB proceeds to the second- or third-ranked cue and still chooses A, whereas tallying favors B by margins 1 and 2. Contrasts 7-8 make both theories choose A but vary the tally margin from 1 to 4; tallying predicts a quantitative confidence gradient while TTB again predicts a constant choice probability. Thus the design tests both categorical conflict and the central quantitative dissociation: tally-margin sensitivity versus TTB invariance.

[1] This design tests the advocated theory's permutation invariance rather than repeating a tally-margin gradient. Every trial in the first block contains exactly five feature wins for A, three for B, and two ties, so Tallying assigns all eight trials the identical score difference of +2 and predicts the same A-choice probability: (1-epsilon)*sigmoid(2*beta)+epsilon/2. However, the same win/loss/tie multiset is permuted across validity ranks. On trials 1-4 the first discriminating cue favors A, whereas on trials 5-8 it favors B; the first discriminator occurs at ranks 1, 2, or 3 across exemplars. TTB therefore predicts two sharply separated probability clusters within a set that Tallying predicts to be homogeneous. Trials 9-16 reverse A and B for exact side balance: Tallying gives every reversed trial the same score difference of -2, while TTB's winner also reverses. Thus the key quantitative diagnostic is within-block heterogeneity: systematic choice changes based on the highest-ranked discriminating cue support TTB, whereas stable choice probabilities under validity-rank permutations support Tallying. Six repetitions of each of the 16 unique pairs yield 96 trials.

[2] Use 10 experts, placing the observable cue load just above the competing theory's narrow capacity range of 9.7–9.9. Across eight structural conditions, the highest-validity discriminating cue always favors A, while the equally weighted pooled direction always favors B. The number of discriminating cues increases from 3 to 10, with all remaining cues tied. Take The Best therefore predicts exactly the same A-choice probability in every structural condition: Expert 1 is always the first discriminator, and all later evidence is ignored. Rank-Budget Arbitration instead predicts a monotonic decline in A choices as the number of discriminating cues increases. Its observable load rises from 10.075 to 10.250, increasing the probability of switching from the A-favoring lexicographic rule to the B-favoring pooled rule. Importantly, pooled evidence has the same signed value (-1) throughout because that theory sign-compresses the tally, so its predicted gradient is specifically attributable to comparison load rather than tally magnitude. Odd and even conditions alternate pooled margins of one and two where possible, helping distinguish this load gradient from ordinary sensitivity to tally margin. Every condition is also presented with A and B reversed, controlling side bias while preserving the predicted gradient toward the lexicographic winner for TTB and away from it as load rises for Rank-Budget Arbitration. The 16 unique pairs are each repeated six times, yielding 96 trials.

[3] This design holds both cue-set size and trialwise comparison load exactly constant: all 10 cues discriminate on every trial, so observable load is always 10.25. With capacity 9.7–9.9 and gate steepness 2.0–2.4, Rank-Budget Arbitration predicts a pooling probability of approximately 0.67–0.79 on every trial. In each of the first eight pairs, the highest-validity cue favors A, so Take The Best predicts one constant A-choice probability regardless of all other ratings. Rank-Budget Arbitration instead predicts three probability plateaus according to the sign-compressed pooled direction. In pairs 1–3, lower-ranked majorities favor B, so pooling conflicts with the lexical cue and A choices should fall below 0.5 because pooling is more probable than rank retrieval. In pairs 4–5, the pooled tally ties, producing an intermediate A-choice probability between 0.5 and the lexical-choice probability. In pairs 6–8, pooling and the lexical cue both favor A, producing the highest A-choice probability. The different tally margins within the conflict class (−8, −6, and −2) and agreement class (+2, +6, and +10) should not produce gradients under the advocated theory because pooled evidence is sign-compressed; they form predicted within-class plateaus. Pairs 9–16 are exact A/B reversals. Thus the diagnostic signature is a reversed three-level conflict/tie/agreement pattern at identical comparison load, versus TTB's single probability level for all eight structures in each orientation. Six repetitions of each of the 16 pairs yield 96 trials.

[4] Use nine experts, placing every trial below Rank-Budget Arbitration's capacity range while making the entire experienced ecology maximally dense. All nine cues discriminate on every trial, so Rank-Budget Arbitration's observable load is fixed at 9.25. Given capacity 9.7–9.9 and gate steepness 2.0–2.4, it predicts only about 0.17–0.29 pooling, with no history or serial-position effect. In each base pair, the highest-validity cue favors A, but exactly five of nine cues favor B; hence lexicographic and pooled representations conflict. Rank-Budget Arbitration should therefore continue choosing the lexicographic winner A on substantially more than half of base trials (approximately 0.58 or higher across its parameter ranges, including lapses). Ecological Encoding Hysteresis instead accumulates a discrimination density of 1.0 on every trial. Despite its sparse prior, its context estimate crosses the 0.68–0.71 threshold within the first few dense comparisons and approaches saturated compression of 0.68–0.76. It consequently predicts a rapid early shift toward the majority winner B and an asymptotic A-choice probability of roughly 0.26–0.40. Thus the primary dissociation is a categorical reversal after context acquisition: persistent lexicographic-majority A choices under Rank-Budget Arbitration versus majority B choices under Ecological Encoding Hysteresis. A secondary diagnostic is serial position: only the advocated theory predicts an early transition, while the competitor is stationary because every trial has identical load. Eight lower-cue permutations prevent reliance on a single memorized display while preserving all theoretically relevant quantities; eight exact A/B reversals balance response side. Each of the 16 pairs is repeated six times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Use 10 experts, so Rank-Budget Arbitration is above its capacity range on every trial, while keeping the experienced discrimination ecology sparse overall. Each structural pair has the highest-validity cue favoring A and every other discriminating cue favoring B; thus lexicographic choice selects A, whereas sign-compressed pooling selects B. Five of eight base structures have only 3/10 discriminating cues, and the remaining structures have 5/10, 7/10, and 9/10. Including exact A/B reversals gives 16 unique pairs and an average discrimination density of only 0.45. Consequently, after the short initial transient, Ecological Encoding Hysteresis estimates a context far below its 0.68\u20130.71 threshold and predicts persistent lexicographic choices even on the occasional dense comparisons. Rank-Budget Arbitration instead responds to each trial's observable load. Its pooling probability is approximately 0.59\u20130.71 at density 0.3 and rises to approximately 0.66\u20130.78 at density 0.9; because it exceeds 0.5 throughout, it predicts more choices of the pooled-majority winner than of the lexical winner at every density, with a monotonic strengthening as the number of discriminations rises. The principal dissociation is therefore an asymptotic choice reversal: majority-winner choices under Rank-Budget Arbitration versus highest-validity-cue choices under Ecological Encoding Hysteresis. A secondary quantitative diagnostic is the density gradient in lexical-winner choice after context has stabilized: Rank-Budget Arbitration predicts a decline from 3 to 9 discriminations, whereas Ecological Encoding Hysteresis predicts nearly invariant lexical responding because an individual display receives vanishing weight in the accumulated sparse context. The five low-density permutations prevent one display from dominating and stabilize the sparse ecology; exact reversals balance response side. Each of the 16 pairs is repeated six times, yielding 96 trials. Primary analyses should emphasize later trials, while early trials provide an additional check for the history-dependent transient predicted only by Ecological Encoding Hysteresis.",
  "validities": [
    0.95,
    0.9,
    0.84,
    0.79,
    0.74,
    0.69,
    0.64,
    0.6,
    0.56,
    0.52
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
      0
    ],
    [
      0,
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
      0,
      0,
      0,
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
      0,
      0,
      1,
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
      1,
      1,
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
      0,
      0,
      1
    ],
    [
      0,
      1,
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
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
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
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
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
      0,
      0,
      0,
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
      0,
      0,
      1,
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
      1,
      1,
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
      0,
      0,
      1
    ],
    [
      0,
      1,
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
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 5499,
  "output_tokens": 2932
}
```
