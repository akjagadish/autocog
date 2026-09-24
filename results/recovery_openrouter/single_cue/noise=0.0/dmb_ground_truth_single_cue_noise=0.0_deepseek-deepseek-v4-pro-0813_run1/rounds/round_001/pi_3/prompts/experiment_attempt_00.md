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
**Description:** People do not stop after the first discriminating cue, nor do they weight every cue equally. Instead, they form a continuous decision value by comparing the two options on every feature, weighting each signed feature comparison by a monotonically increasing function of the advertised cue validity. Cues with validity near chance are either ignored or heavily discounted, while highly valid cues receive much larger weight. Ties on a feature contribute zero evidence. The resulting validity-weighted evidence sum is passed through a softmax with inverse temperature beta, plus a small uniform lapse probability epsilon.

**Parameters:**
- beta: [1.0, 50.0]
- epsilon: [0.0, 0.25]
- offset: [0.0, 0.5]
- gamma: [0.5, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-weighted additive integration.
    # state is array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored because validities are
    # advertised in the instructions and no trial-by-trial feedback is given.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-weighted integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1:
        validities = validities.reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {stim.shape[1]}."
        )

    offset = float(parameters["offset"])
    gamma = float(parameters["gamma"])

    # Monotone diagnosticity weight: w_j = max(validity_j - offset, 0)^gamma.
    # For offset = 0.5 and gamma = 1 this is a validity-gain weighting,
    # giving chance cues zero weight.
    weights = np.maximum(validities - offset, 0.0) ** gamma

    # Signed feature-wise evidence: positive favors A, negative favors B.
    # Feature ties contribute zero.
    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over [A, B].
    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - logits.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## COMPETING THEORY
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Eight experts are used. Expert 0 is advertised as 90% accurate; the remaining seven are clearly lower (75% down to 50%). Every critical trial makes Expert 0 discriminate: it favors one option. The seven lower experts are arranged so that k = 0,1,...,7 of them favor the opposite option. Take The Best stops at Expert 0 and therefore predicts a constant probability of choosing the top-cued option across all k. Tallying counts all eight experts; the tally margin for the top-cued option is 8 - 2k, so its predicted choice probability is a decreasing sigmoid in k, crossing chance at k=4 and reversing for k>4. Two mirrored blocks are included (Expert 0 favors A in one block, B in the other) to double observations and control for response bias. This yields a strong quantitative dissociation: a flat psychometric function over k under Take The Best versus a negatively sloped sigmoid under Tallying. The 16 unique pairs are repeated 6 times for roughly 96 trials.

[1] Six experts with validities 0.90, 0.80, 0.70, 0.60, 0.55, 0.50 define a fixed TTB cue order. The 16 unique pairs form a cue-locus × tally-margin design. In A-first trials (A0-A7) the first discriminating cue in validity order favors A; in B-first trials (B0-B7) it favors B. Higher-validity cues are tied on r=1 and r=2 trials so the first discriminator is not always the top expert, extending the already-explored design. Lower cues are arranged to produce tally margins (#A wins - #B wins) of -4, -3, -2, 0, +2, +3, +4. TTB stops at the first discriminator, so it predicts a single p(A) for all A-first trials and a single p(A) for all B-first trials, independent of the lower-cue margin. Tallying counts all six cues, so it predicts p(A) = sigmoid(beta * margin), increasing monotonically with the margin and equal to chance at margin 0. Thus TTB predicts a flat step function in margin within each first-discriminator direction, whereas Tallying predicts a graded psychometric function; zero-margin trials additionally oppose TTB's directional prediction to Tallying's chance prediction. The 16 unique pairs are repeated K=6 times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Six experts with validities 0.92, 0.84, 0.76, 0.68, 0.58, 0.50. The design holds the Tallying score constant within two blocks while varying validity-weighted evidence. The first eight trials are zero tally-margin pairs: A wins exactly one feature and B wins exactly one feature, so Tallying gives a 1-1 tie and predicts p(A)=0.5 for every pair. The discriminating feature pair is varied from (A wins the 50% expert, B wins the 92% expert) to the reverse, so the advocated validity-weighted evidence increases monotonically from w5-w0 to w0-w5, predicting a psychometric curve from below to above chance. The last eight trials are constant +2 tally-margin pairs: A wins 4 features, B wins 2, so Tallying predicts the same p(A)=sigmoid(2*beta) for all eight. The four A-win features are chosen along a dominance chain (each adjacent pair swaps one A-win from a lower-validity to a higher-validity feature), so the validity-weighted evidence strictly increases for any monotone validity-weighting function. With offset=0.5, gamma=1 the +2 block runs from -0.24 (weighted favors B despite A winning 4 of 6 features) to +1.12. Thus Tallying predicts two flat lines (0.5 and a single >0.5 value), while the advocated theory predicts two increasing psychometric functions of weighted evidence. The 16 unique pairs are repeated K=6 for about 96 trials.",
  "validities": [
    0.92,
    0.84,
    0.76,
    0.68,
    0.58,
    0.5
  ],
  "trial_a_ratings": [
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
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
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
      0
    ],
    [
      1,
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
      0
    ],
    [
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
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1
    ],
    [
      1,
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
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
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
      0
    ],
    [
      1,
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
      0
    ],
    [
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
      1,
      0,
      0
    ],
    [
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
      1
    ],
    [
      1,
      1,
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
      0
    ],
    [
      0,
      1,
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
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
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
  "input_tokens": 3586,
  "output_tokens": 25562,
  "reasoning_tokens": 19123,
  "provider": "CoreWeave"
}
```
