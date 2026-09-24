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
**Description:** On each choice, a decision maker probabilistically selects between two comparison strategies. With probability q, centered near 0.25, the choice is made by a one-reason stopping rule: cues are inspected in descending validity order and the first discriminating cue determines the preferred option, with no further cues consulted. With probability 1 - q, the choice is made by full feature-win tallying: each option receives one point for every feature on which it strictly beats the other option, ties contribute nothing, and the option with the larger tally is preferred. Each strategy's preference is passed through its own softmax choice rule, giving nearly deterministic but not perfectly errorless responding, and there is no additional uniform lapse. The mixture probability q is a stable subject-level parameter applied independently on every trial, so the model recovers pure Take The Best when q = 1, pure Tallying when q = 0, and an intermediate strategy mixture when q is near 0.25.

**Parameters:**
- q: [0.22, 0.28]
- beta_tally: [5.44, 5.50]
- beta_ttb: [5.22, 5.28]
- epsilon: {0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    def _softmax(scores, beta):
        z = beta * (scores - np.max(scores))
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    n_features = stim.shape[1]
    a = stim[0]
    b = stim[1]

    validities = np.asarray(parameters['validities'], dtype=float)
    if validities.shape[0] == n_features:
        cue_order = np.argsort(-validities, kind='stable').tolist()
    else:
        cue_order = list(range(n_features))

    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        p_ttb = np.ones(2) / 2.0
    else:
        ttb_scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        p_ttb = _softmax(ttb_scores, float(parameters['beta_ttb']))

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    p_tally = _softmax(
        np.array([a_wins, b_wins]),
        float(parameters['beta_tally'])
    )

    q = float(parameters['q'])
    p = q * p_ttb + (1.0 - q) * p_tally

    epsilon = float(parameters['epsilon'])
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2) / 2.0

    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People choose between two options through a stable, subject-level mixture of three comparison routes: (1) forward lexicographic scanning, inspecting cues in descending validity order and stopping at the first discriminating cue; (2) backward lexicographic scanning, inspecting cues in ascending validity order and stopping at the first discriminating cue in that reversed order; and (3) feature-win tallying, where each option receives one point per feature on which it strictly beats the other and ties contribute nothing. The forward and backward lexicographic weights jointly act as a subject-level polarity parameter for first-validity-cue effects: forward weight supports a positive first-discriminator effect, while backward weight supports a negative one. Each route's preferred option is passed through its own softmax choice rule, so responding is noisy but not controlled by a uniform lapse.

**Parameters:**
- w_forward: [0.22, 0.32]
- w_backward: [0.42, 0.54]
- beta_forward: [0.08, 0.20]
- beta_backward: [1.30, 1.70]
- beta_tally: [1.80, 2.60]
- epsilon: {0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Preferred validity vector comes from the experiment.  If the
    # parameter resolver ever leaves it unresolved, fall back to a
    # descending validity vector so the model is still scorable.
    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.ones(2, dtype=float) / 2.0

        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design makes the highest-validity cue disagree with the majority of the remaining cues on every trial. All six validities are ordered so feature 1 is the top TTB cue; every critical pair has that top cue discriminate in one direction while at least three of the five lower-validity cues discriminate in the opposite direction. Take The Best stops at the top cue and predicts the top-cue winner; Tallying counts all feature-wise wins and predicts the opposite majority. The 32 unique pairs contain 16 top-cue-favors-A cases and 16 top-cue-favors-B cases to balance any position bias. For a top-A pair where m lower cues favor B (here m = 3, 4, or 5), TTB predicts choice A with probability about 1/(1+exp(-beta)) before lapse, while Tallying predicts choice B with probability about 1/(1+exp(-beta*(m-1))), so the theories are separated in direction at moderate and high beta. The 32 unique pairs are repeated K = 3 times for 96 total trials.

[1] This design isolates a second-cue stopping signature. The highest-validity cue (feature 0, validity 0.90) is tied on every trial, so Take The Best skips it and stops at the second cue (feature 1, validity 0.78). Feature 1 always discriminates, so TTB never consults the three lower cues; it predicts the same choice probability for all feature-1-favors-A trials and the opposite for all feature-1-favors-B trials. Tallying, by contrast, treats the top tie as uninformative and counts wins on all remaining cues. The three lower cues are varied so that Tallying's tally difference d = A_wins - B_wins takes values -2, -1, and 0 for feature-1-A trials, and 0, +1, and +2 for feature-1-B trials. Thus TTB predicts a flat choice curve across lower-cue patterns within each direction, whereas Tallying predicts a logistic curve in d, producing opposite choices on the d=-1/-2 and d=+1/+2 patterns and guessing on d=0. The 32 unique pairs are balanced over top-tie value (0/0 vs 1/1) and second-cue direction, and are repeated K=3 for 96 total trials.

[2] Six features with a spread of validities. Feature 0 (validity 0.90) is tied on every trial, so Take The Best skips it and stops at feature 1 (validity 0.80), which always discriminates. The four lower features are arranged to set the tally difference d = A_wins - B_wins to -2, -1, 0, +1, or +2. Critically, for each d there are trials where feature 1 favors A and trials where feature 1 favors B. Pure Tallying depends only on d and therefore must predict identical choice probabilities for the two feature-1 directions within each d. The TTB/Tallying mixture instead predicts a within-d split of about q≈0.25: e.g. at d=0, pure Tallying predicts 0.5 for all trials, while the mixture predicts ≈0.62 when feature 1 favors A and ≈0.38 when it favors B; at d=+1, pure Tallying predicts near 1.0, while the mixture predicts ≈1.0 vs ≈0.75. This within-d split cannot be produced by any beta or epsilon in pure Tallying. The 32 unique pairs are repeated K=3 for 96 total trials.

[3] This design isolates a signature of pure Tallying that the TTB/Tallying mixture cannot reproduce. Pure Tallying depends only on the tally difference d = A_wins - B_wins. When d = 0, its softmax over equal tallies is exactly uniform and the lapse term also preserves uniformity, so it must predict P(A) = 0.5 for every such pair, regardless of which individual cue discriminates. The mixture model instead adds a TTB component (q ≈ 0.25) that stops at the first discriminating cue in descending-validity order. By tying the top two or top three cues on every trial, the first discriminator is pushed to feature 2 or feature 3. All 32 unique pairs are constructed to have d = 0, so the Tallying component of the mixture also gives 0.5, but the TTB component shifts choice probabilities to about 0.62 when the first discriminator favors A and about 0.38 when it favors B (using q ≈ 0.25 and beta_ttb ≈ 5.25). The critical contrast is therefore first-discriminator direction: pure Tallying predicts 0.5 vs 0.5 (difference 0), while the mixture predicts approximately 0.62 vs 0.38 (difference about 0.25). The design balances first-discriminator direction, top-tie value (0/0 vs 1/1), and discriminator position (feature 2 vs feature 3) across the 32 unique pairs, which are repeated K = 3 times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design isolates a direction-of-lexicographic-scanning dissociation. All 32 unique pairs hold the feature-win tally difference at d = 0, so both theories' tallying routes are neutral and cannot drive the choice. On every trial the highest-validity cue (feature 0, validity 0.90) discriminates, and the lowest-validity cue (feature 5, validity 0.50) discriminates in the opposite direction. The advocated TTB/Tally mixture has only a descending lexicographic route, so its TTB component stops at feature 0; with q near 0.25 it predicts P(A) \u2248 0.62 when feature 0 favors A and \u2248 0.38 when feature 0 favors B. The competing model instead has a strong backward (ascending) lexicographic route (weight \u2248 0.42-0.54, beta \u2248 1.3-1.7) plus a weak forward route and neutral tallying, so on the same conflict trials it predicts the mirror pattern: P(A) \u2248 0.32-0.40 when feature 0 favors A/feature 5 favors B, and \u2248 0.60-0.68 when feature 0 favors B/feature 5 favors A. The middle four features are varied with equal A/B wins or ties to keep d = 0 while generating 32 unique pairs, 16 in each top-cue direction, repeated K = 3 for 96 trials.",
  "validities": [
    0.9,
    0.82,
    0.74,
    0.66,
    0.58,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
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
      0
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
      1,
      1,
      0,
      0
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
      1,
      1,
      0
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
      1,
      0,
      1,
      1,
      0
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
      1,
      1,
      0,
      1,
      0
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
      1,
      1,
      1,
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
      1,
      0,
      1,
      1,
      1,
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
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
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
      1
    ],
    [
      0,
      1,
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
      1
    ],
    [
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
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
      0,
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
      0,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
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
      0,
      1
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
      1,
      1,
      1,
      1
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
      1,
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
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
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
      1
    ],
    [
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
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
      0,
      1
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
      0,
      0,
      1,
      0,
      1
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
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
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
      0,
      0,
      0,
      1,
      0,
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
      0
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
      1,
      1,
      0,
      0
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
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
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
      0,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
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
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1,
      0
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
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
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
  "input_tokens": 4760,
  "output_tokens": 19259,
  "reasoning_tokens": 13323,
  "provider": "CoreWeave"
}
```
