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
**Description:** Conflict-Gated Redundancy-Discounted Attention (CGRDA) proposes that people integrate all discriminating cues, but cue influence depends jointly on instructed validity, serial position, directional redundancy, and the current conflict configuration. Instructed validity and early presentation increase a cue's baseline attentional weight. Successive endorsements of an option have diminishing marginal impact because repeated evidence is treated as partially redundant. When the numbers of cues favoring the two options are closely balanced, attention is reoriented toward the final discriminating cue, producing a conflict-gated recency contribution. Thus, an early compact set of reliable cues can defeat a larger but redundant set, while the last cue can reverse the decision in balanced conflicts. Primacy, redundancy discounting, recency strength, and conflict sensitivity are stable subject-level traits.

**Parameters:**
- validities: validities
- primacy: [0.8, 1.4]
- redundancy_discount: [0.25, 1.25]
- validity_sensitivity: [0.3, 1.5]
- recency_strength: [1.15, 1.65]
- balance_sensitivity: [2.5, 5.0]
- beta: [0.8, 1.8]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CGRDA expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive differences favor B and negative differences favor A.
    diff = stim[1] - stim[0]
    signs = np.sign(diff)
    discriminating = np.flatnonzero(signs != 0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    primacy = float(parameters["primacy"])
    redundancy_discount = float(parameters["redundancy_discount"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    recency_strength = float(parameters["recency_strength"])
    balance_sensitivity = float(parameters["balance_sensitivity"])
    beta = float(parameters["beta"])

    # Convert communicated validities to reliability evidence. Normalizing
    # preserves validity rank without making the score depend arbitrarily
    # on the absolute log-odds scale selected by an experiment.
    clipped = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(clipped / (1.0 - clipped))
    reliability /= max(float(np.max(reliability)), 1e-12)
    reliability = np.maximum(reliability, 0.05)

    count_a = 0
    count_b = 0
    integrated = 0.0
    total_weight = 0.0

    for j in discriminating:
        direction = float(signs[j])
        if direction > 0:
            count_b += 1
            repetition = count_b
        else:
            count_a += 1
            repetition = count_a

        validity_weight = reliability[j] ** validity_sensitivity
        position_weight = np.exp(-primacy * float(j))
        novelty_weight = float(repetition) ** (-redundancy_discount)
        weight = validity_weight * position_weight * novelty_weight

        integrated += direction * weight
        total_weight += weight

    # Put integrated evidence on a common bounded scale across feature
    # counts and experiments.
    core_evidence = integrated / max(total_weight, 1e-12)

    # Recency is strongest at exact tally balance and falls rapidly as one
    # side acquires more endorsements. The final discriminating cue, not
    # necessarily the least-valid cue, determines the direction of this
    # conflict-resolution contribution.
    tally_imbalance = abs(count_b - count_a)
    conflict_gate = np.exp(-balance_sensitivity * float(tally_imbalance))
    last_direction = float(signs[int(discriminating[-1])])
    recency_evidence = recency_strength * conflict_gate * last_direction

    choice_evidence = core_evidence + recency_evidence
    utilities = np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])

    x = beta * utilities
    x = x - np.max(x)
    probs = np.exp(x)
    probs /= probs.sum()
    return probs.astype(np.float64)
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Eight experts have strictly ordered, widely separated validities. Every pair creates a conflict between Take The Best and Tallying: the highest-validity discriminating expert favors one option, while the majority of discriminating experts favors the other. The design orthogonally balances whether A or B is favored by the decisive expert and manipulates the opposing tally margin. In trials where Expert 1 discriminates, the tally margins are 2, 4, and 6; where Expert 1 ties and Expert 2 is the first discriminating expert, the margins are 1, 3, and 5. Take The Best therefore predicts the same choice probability for the option favored by the first discriminating expert on every trial: epsilon/2 + (1-epsilon)*sigmoid(beta), invariant to both cue rank and tally margin. Tallying predicts that this option is chosen below chance, with probability epsilon/2 + (1-epsilon)*sigmoid(-beta*m), and predicts a systematic decrease as the opposing margin m increases. Thus, the critical dissociation is both directional and quantitative: invariant support for the top-cue winner under TTB versus increasingly strong rejection of it under Tallying. Swapping A and B for each evidence pattern controls response-side preferences. The 12 unique pairs yield exactly 96 trials with eight repetitions per pair.

[1] This design targets the parameter-free prediction of Tallying on exactly balanced evidence. On every trial, A and B win on the same number of discriminating features, so Tallying predicts P(A)=P(B)=0.5 for every beta and epsilon. Nevertheless, each pair has a uniquely defined first discriminating expert in descending-validity order. Take The Best therefore predicts choice of that expert's favored option with probability epsilon/2 + (1-epsilon)*sigmoid(beta), which is strictly above chance throughout the stated parameter ranges. The six underlying evidence patterns vary both the rank of the first discriminating expert (1, 3, or 5) and the total number of discriminating experts (2, 4, 6, or 8), while preserving an exactly tied tally. Take The Best predicts the same above-chance preference across these manipulations, whereas Tallying remains exactly at chance. Each pattern is paired with an A/B reversal, controlling response-side bias. Thus the critical statistic is the probability of choosing the option favored by the first discriminating expert: exactly 0.5 under Tallying versus a common value above 0.5 under Take The Best. The 12 unique pairs are repeated eight times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "The design exploits a sharp conflict-gated crossover unique to CGRDA while holding Take The Best's decisive cue constant. Expert 5 has the highest validity (0.95) and always discriminates in favor of the designated TTB winner. Each family contains an exactly balanced 2-versus-2 trial and a matched 3-versus-1 trial differing only in Expert 1's direction. On balanced trials, the final discriminating expert (Expert 6) opposes Expert 5. TTB therefore chooses Expert 5's option with the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta) on every trial. CGRDA instead predicts the Expert-6 option on every balanced trial throughout its parameter ranges: tally balance fully opens the recency gate, and recency_strength is at least 1.15 while normalized core evidence is bounded in magnitude by 1. On matched imbalanced trials, reversing Expert 1 creates three endorsements for the TTB winner versus one final endorsement for its opponent. The imbalance of two suppresses recency by at least exp(-5), while the early Expert-1 endorsement and two additional concordant endorsements make CGRDA favor the TTB winner. Thus, the critical statistic is the probability of choosing the Expert-5 winner: TTB predicts invariance between balanced and imbalanced conditions, whereas CGRDA predicts a crossover from below chance on balanced trials to above chance on imbalanced trials. Moving the second pro-TTB endorsement across Experts 2, 3, and 4 provides an additional serial-position manipulation: TTB remains invariant, while CGRDA's integrated evidence changes with primacy. Exact A/B reversals control response-side bias. The 12 unique pairs are each repeated eight times, yielding 96 trials.",
  "validities": [
    0.68,
    0.66,
    0.64,
    0.62,
    0.95,
    0.52
  ],
  "trial_a_ratings": [
    [
      0,
      1,
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
      0,
      1
    ],
    [
      1,
      1,
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
      1
    ],
    [
      0,
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
      1
    ],
    [
      1,
      0,
      1,
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
      1
    ],
    [
      0,
      0,
      0,
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
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
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
      1
    ],
    [
      1,
      1,
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
      0,
      1
    ],
    [
      0,
      0,
      1,
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
      1
    ],
    [
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
      1
    ],
    [
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
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
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
  "input_tokens": 4016,
  "output_tokens": 3950
}
```
