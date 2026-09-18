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
**Description:** Precision-Annealed Rank Integration theory: people always combine the signed evidence supplied by every discriminating cue. Comparison load does not recruit a different heuristic; instead, it continuously changes the precision with which cue priorities can be represented. At low load, the attention distribution is diffuse and discriminating cues receive nearly equal weight, approximating tallying. As the number of discriminating cues exceeds a stable representational capacity, attention is progressively concentrated toward cues higher in the communicated-validity ordering. Attention is implemented as a Gibbs distribution over validity ranks, with its concentration increasing smoothly with load. The load curve is mildly powered while anchored at high load, reducing premature concentration without weakening overload attention. The highest-validity cue retains its original Gibbs attention mass, while the conditional distribution over remaining cues is only mildly flattened, preserving collective tail evidence without excessively reallocating attention toward the lowest-validity cues. The normalized weighted evidence is then translated into a choice through ordinary logistic response noise. Stable individual differences generate heterogeneous but gradual crossovers without categorical switching between tallying and Take The Best.

**Parameters:**
- validities: validities
- capacity: [5.3, 5.8]
- transition_width: [0.6, 0.95]
- concentration_gain: [1.15, 1.55]
- schedule_power: [1.15, 1.35]
- tail_flatten: [0.75, 0.95]
- beta: [4.2, 5.4]
- epsilon: [0.0, 0.06]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Precision-Annealed Rank Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    signs = np.sign(differences)
    discriminating = np.flatnonzero(signs != 0.0)
    load = float(discriminating.size)

    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    # A cue's rank is the number of cues having strictly greater communicated
    # validity. Thus equally valid cues receive exactly equal attention.
    ranks = np.asarray(
        [np.sum(validities > validities[j]) for j in range(n_features)],
        dtype=float,
    )

    capacity = float(parameters["capacity"])
    transition_width = float(parameters["transition_width"])
    concentration_gain = float(parameters["concentration_gain"])
    schedule_power = float(parameters["schedule_power"])

    # Smoothly increasing attention concentration. Powering the curve relative
    # to a load-7 anchor reduces premature low-load concentration while leaving
    # the overload concentration of the accepted model approximately intact.
    def stable_sigmoid(value):
        if value >= 0.0:
            return 1.0 / (1.0 + np.exp(-value))
        ex = np.exp(value)
        return ex / (1.0 + ex)

    x = (load - capacity) / transition_width
    concentration_fraction = stable_sigmoid(x)
    anchor_fraction = stable_sigmoid((7.0 - capacity) / transition_width)
    concentration = concentration_gain * anchor_fraction * (
        concentration_fraction / anchor_fraction
    ) ** schedule_power

    active_ranks = ranks[discriminating]
    log_attention = -concentration * active_ranks
    log_attention -= np.max(log_attention)
    attention = np.exp(log_attention)
    attention /= attention.sum()

    # Preserve the Gibbs mass assigned to the highest-validity active cue(s),
    # but flatten only the conditional distribution over the remaining cues.
    top_mask = active_ranks == np.min(active_ranks)
    tail_mask = ~top_mask
    if np.any(tail_mask):
        top_mass = float(np.sum(attention[top_mask]))
        tail_flatten = float(parameters["tail_flatten"])
        tail_logits = tail_flatten * log_attention[tail_mask]
        tail_logits -= np.max(tail_logits)
        tail_distribution = np.exp(tail_logits)
        tail_distribution /= tail_distribution.sum()

        adjusted_attention = np.zeros_like(attention)
        adjusted_attention[top_mask] = attention[top_mask]
        adjusted_attention[tail_mask] = (1.0 - top_mass) * tail_distribution
        attention = adjusted_attention

    # Positive evidence favors A and negative evidence favors B. Normalizing
    # attention keeps response precision comparable across different loads.
    evidence = float(np.dot(attention, signs[discriminating]))

    beta = float(parameters["beta"])
    logits = np.array([0.5 * beta * evidence, -0.5 * beta * evidence], dtype=float)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5], dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "All trials have exactly five discriminating cues: Experts 1, 2, 4, 6, and 8 (validity ranks 0, 1, 3, 5, and 7). In every unreversed pair, three cues favor A, the highest-validity cue favors B, and exactly one additional cue favors B. That second B-favoring cue moves progressively from Expert 2 to Expert 4, Expert 6, and Expert 8. Prospective Capacity-Gated Comparison therefore predicts exactly the same A-choice probability for all four unreversed pairs: load, tally direction and margin, and the first validity-ranked discriminating cue are identical. Its integration branch selects A, its lexicographic branch selects B, and only the fixed load-5 gate determines their mixture. Precision-Annealed Rank Integration instead predicts a strictly increasing A-choice gradient as the second B cue moves down ranks 1, 3, 5, and 7, because the opposing evidence loses attention mass while all other properties remain fixed. At representative parameter midpoints, the approximate unreversed profile is P(A)=0.12, 0.31, 0.46, and 0.55, rather than the competitor's flat profile. Exact A/B reversals should produce the complementary decreasing profile while controlling response bias. The non-discriminating cues vary in their shared displayed values across exemplars but cannot affect either model. Eight unique pairs repeated 12 times yield 96 trials.",
  "validities": [
    0.95,
    0.89,
    0.83,
    0.77,
    0.7,
    0.63,
    0.57,
    0.52
  ],
  "trial_a_ratings": [
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
      0,
      1,
      0,
      0,
      1,
      1,
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
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
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
      1,
      1,
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
      1,
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
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      1,
      1,
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
      1,
      1,
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
      1,
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
      1,
      1
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
      0,
      1,
      0,
      0,
      1,
      1,
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
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 5127,
  "output_tokens": 3867
}
```
