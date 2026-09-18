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
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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
**Description:** Communicated-Reliability Accumulation theory proposes that people integrate every discriminating expert rating after translating communicated validity into subjective diagnostic weight. The weight is a mildly superlinear log-odds transformation plus a modest common reliability floor: w_j = logit(v_j)^gamma + c. The log-odds component privileges more reliable experts, while the positive floor represents a minimal contribution assigned to any expert judged better than chance. Net evidence favoring A is E = sum_j w_j(A_j-B_j). Consequently, sufficiently numerous weaker experts can overturn a stronger cue, and concordant cue-number margins generate graded confidence. Choice sensitivity varies moderately across people, while only very rare lapses produce validity-independent guessing. Choice history is ignored because the task provides no outcome feedback.

**Parameters:**
- beta: [0.24, 0.58]
- epsilon: [0.0, 0.005]
- gamma: [1.02, 1.05]
- reliability_floor: [0.10, 0.14]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Communicated-Reliability Accumulation.
    # state has shape (2, n_features), with rows for options A and B.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Reliability accumulation expects shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Clipping only protects the log-odds transform at its mathematical
    # endpoints. A modest common floor strengthens accumulation by cue number,
    # while mild superlinearity preserves the leverage of highly valid cues.
    v = np.clip(validities, 0.5, 1.0 - 1e-12)
    log_odds = np.log(v) - np.log1p(-v)
    gamma = float(parameters["gamma"])
    reliability_floor = float(parameters["reliability_floor"])
    reliability_weights = np.power(log_odds, gamma) + reliability_floor

    a = stimulus[0]
    b = stimulus[1]
    evidence_for_a = float(np.dot(reliability_weights, a - b))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Symmetric logits make their difference beta * evidence_for_a, so the
    # core probability of A is sigmoid(beta * evidence_for_a).
    logits = np.array(
        [0.5 * beta * evidence_for_a, -0.5 * beta * evidence_for_a],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probabilities = (1.0 - epsilon) * p_core + epsilon * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.ones(len(probabilities), dtype=np.float64) / len(probabilities)
    else:
        probabilities /= total
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This 16-pair, 96-trial design creates both directional and quantitative dissociations. In each of the first eight pairs, the highest-validity discriminating cue favors A; the remaining eight pairs are exact A/B reversals. Therefore Take The Best predicts the same choice probability for its lexicographic winner on every trial: epsilon/2 + (1-epsilon)*sigmoid(beta), regardless of which cue is first discriminating or how many lower-validity cues oppose it. Pairs 1-4 manipulate the rank of the first discriminating cue while producing equal tallies (one cue for each option). Tallying must predict exactly 0.5 on these pairs, whereas Take The Best predicts a systematic preference. In pairs 5-8, the lexicographic winner loses the tally by respective score gaps of 1, 2, 3, and 5. Take The Best continues to predict an invariant preference for that winner, while Tallying predicts the opposite option with monotonically increasing confidence as the gap grows. Exact reversals balance response side and preserve every theoretical contrast. With 16 unique pairs, each is repeated six times.

[1] This design isolates a quantitative confidence signature without relying on opposite directional choices. The highest-validity expert (feature 5, validity 0.95) discriminates on every trial, so Take The Best always stops immediately and predicts the same probability for its winner on all eight pairs: epsilon/2 + (1-epsilon)*sigmoid(beta). The tally winner is always the same option as the Take-The-Best winner, but its feature-win margin is systematically manipulated across four conditions: 4-3 (gap 1), 5-2 (gap 3), 6-1 (gap 5), and 7-0 (gap 7), with two tied features in every condition. Tallying therefore predicts winner probabilities epsilon/2 + (1-epsilon)*sigmoid(beta*d) for d in {1,3,5,7}, producing a strictly increasing choice-consistency gradient except at practical saturation. Take The Best cannot produce this within-subject margin gradient because its binary score difference is always one. Each condition has an exact A/B reversal, balancing response side while preserving the theoretical contrast. There are eight unique pairs, so each is repeated 12 times for approximately 96 trials.

[2] Each base pair gives A one feature win and B two feature wins; the final eight pairs are exact A/B reversals. Tallying therefore predicts exactly the same B-choice probability on all eight base pairs because its tally margin is always 2-1, and the same mirrored A-choice probability on all reversals. Communicated-Reliability Accumulation instead distinguishes the identities of the winning experts. At representative parameter values (gamma about 1.035 and reliability_floor about 0.12), net evidence for A across the eight base pairs is approximately 2.71, 1.80, 1.00, 0.43, -0.10, -0.34, -0.71, and -1.40. Thus it predicts a strong, ordered choice gradient and a crossover from choosing the one-cue minority A to choosing the two-cue majority B. This ordering and crossover are robust over the advocated parameter ranges, whereas no setting of Tallying's beta or epsilon can produce stimulus-specific differences among pairs having the same tally margin. Exact reversals balance response side and should mirror the gradient. With 16 unique pairs, each pair is repeated six times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This six-pair design makes tally margin increase while communicated-reliability evidence decreases and ultimately reverses direction. In the first three pairs, A's tally margins are respectively 1, 3, and 5: A wins 1-0, 3-0, and 6-1 features. Tallying therefore predicts a strictly increasing A-choice gradient, with probabilities epsilon/2 + (1-epsilon)*sigmoid(beta*d) for d in {1,3,5}. Reliability accumulation predicts the opposite quantitative pattern because the larger tallies are built from progressively weaker experts. At representative parameters (gamma=1.035 and reliability_floor=0.12), its evidence for A is approximately 3.18, 2.44, and -1.04. These evidence values remain ordered across the specified parameter ranges, and the third remains negative: there, Tallying favors A by a five-feature margin, whereas reliability accumulation favors B because the 95%-valid expert outweighs six weak experts. The final three pairs are exact A/B reversals, balancing response side and mirroring both theories' predictions. Thus the critical signatures are an increasing choice-consistency gradient under Tallying versus a decreasing gradient and directional crossover under reliability accumulation. With six unique pairs, each is repeated 16 times for 96 trials.",
  "validities": [
    0.95,
    0.8,
    0.72,
    0.68,
    0.6,
    0.54,
    0.52,
    0.51,
    0.505
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
      0
    ],
    [
      0,
      0,
      1,
      1,
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
      1,
      1,
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
  "input_tokens": 3646,
  "output_tokens": 4740
}
```
