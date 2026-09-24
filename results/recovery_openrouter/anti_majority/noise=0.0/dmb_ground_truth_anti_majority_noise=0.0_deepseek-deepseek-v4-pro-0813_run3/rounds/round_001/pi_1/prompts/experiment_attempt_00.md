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
**Description:** People invert the meaning of binary expert ratings: a rating of 1 is treated as a defect or costly negative signal, and a rating of 0 is treated as a clean or favorable signal. Across all features, they tally how often each option is defect-free relative to the other option, and choose the option with the larger defect-free tally. Ties lead to guessing. Choice probability is a noisy softmax over the two defect tallies with inverse temperature beta, plus an independent lapse process that produces uniform guessing with probability epsilon. The heuristic uses all features equally, ignores validities and past trials, and differs from Take The Best by aggregating rather than stopping at the first discriminating cue, and from standard Tallying by reversing the valence of the binary ratings.

**Parameters:**
- beta: [1.0, 5.0]
- epsilon: [0.30, 0.40]

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Inverse-tallying expects a (2, n_features) stimulus; got shape {stim.shape}.')

    a, b = stim[0], stim[1]

    # Defect-free wins: A has a 0 where B has a 1.
    a_wins_defects = float(np.sum(b > a))
    # Defect-free wins: B has a 0 where A has a 1.
    b_wins_defects = float(np.sum(a > b))

    scores = np.array([a_wins_defects, b_wins_defects])

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Numerically stable softmax over inverse-cue tallies.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Six features are used. The highest-validity expert (0.90) is always the first cue TTB tests. I hold that cue's discriminating direction constant across blocks and vary the lower five cues so Tallying's net evidence changes while TTB sees the same first discriminating cue. In the top-A block the top cue is 1 vs 0 and lower-cue net wins take values +5, +3, +1, -1, -3, -5; in the top-B block the top cue is reversed. TTB predicts identical p(A) for all six top-A pairs and identical p(A) for all six top-B pairs because the first discriminating cue is unchanged; Tallying predicts a graded logistic change because its tally advantage is 1 + lower_net for top-A pairs and lower_net - 1 for top-B pairs. At lower_net = -1 in the top-A block and lower_net = +1 in the top-B block, Tallying is exactly tied while TTB commits to A or B. At lower_net = -3/-5 in the top-A block and +3/+5 in the top-B block, the two theories make opposite directional predictions. The critical signature is flat TTB choice probabilities across the six lower-net levels versus a monotone logistic gradient for Tallying. Twelve unique pairs repeated K = 8 gives 96 trials.

[1] Twelve unique pairs form a 4 × 3 factorial. Six features have validities [0.92, 0.84, 0.76, 0.68, 0.60, 0.55], so TTB's cue cascade is feature 0 → 1 → 2 → 3 → 4 → 5. The first factor is the direction of the first discriminating cue: feature 0 favors A; feature 0 favors B; feature 0 ties and feature 1 favors A; feature 0 ties and feature 1 favors B. The second factor is the tally difference (+2, 0, -2), implemented by the remaining lower features. Tallying ignores which features discriminate, so it predicts three choice-probability levels that depend only on tally difference and are identical across the four first-cue conditions. TTB ignores all lower features once a discriminating cue is found, so it predicts only two choice-probability levels (high for first-cue A, low for first-cue B) and no effect of tally difference. This yields opposite directional predictions on conflict trials: top A with tally -2 (pair 3), top B with tally +2 (pair 4), top tie/second A with tally -2 (pair 9), and top tie/second B with tally +2 (pair 10). It is the complement of the already-explored design: instead of holding the top cue constant and varying lower net, it holds tally difference constant across cue-order conditions and varies the first discriminating cue. With 12 unique pairs repeated K=8, total trials = 96.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Take The Best stops at the first discriminating cue in validity order and ignores all lower cues. Inverse tallying aggregates all cues with reversed valence (0 = favorable) and chooses the option with more zeros. This design holds the inverse-tallying score at exactly zero for every trial: each option has exactly one 1 and one 0 in the two discriminating positions, and all other positions tie. Thus inverse tallying predicts p(A)=0.5 for every pair, regardless of its beta and epsilon. TTB, however, consults cues in validity order and stops at the first discriminating cue. By shifting the position of that first discriminating cue from feature 0 to feature 5 and varying its direction (A=1/B=0 vs A=0/B=1), TTB predicts p(A) to be high when the first cue favors A and low when it favors B, regardless of the lower balancing cue. This yields a strong quantitative dissociation: a flat 0.5 psychometric function across all 12 pairs for inverse tallying versus a step function of first-cue direction for TTB. Unlike the already-explored designs, which held the top cue constant while varying lower-cue net (experiment 0) or varied first-cue direction and tally difference with standard Tallying (experiment 1), this design uses the inverse-tallying tie prediction on every trial and varies the first-cue position across all six possible positions.",
  "validities": [
    0.9,
    0.84,
    0.78,
    0.72,
    0.66,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
    [
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
      0,
      0
    ],
    [
      0,
      0,
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
      0
    ],
    [
      0,
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
      0
    ],
    [
      0,
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
      0,
      1
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
      0
    ],
    [
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
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
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
      1,
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
      0,
      0
    ],
    [
      0,
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
      1,
      0
    ],
    [
      0,
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
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 3754,
  "output_tokens": 23984,
  "reasoning_tokens": 21661,
  "provider": "CoreWeave"
}
```
