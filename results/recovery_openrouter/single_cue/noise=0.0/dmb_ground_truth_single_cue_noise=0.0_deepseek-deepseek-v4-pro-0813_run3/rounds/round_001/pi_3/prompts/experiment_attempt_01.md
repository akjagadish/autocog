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
**Description:** People use a validity-graded compensatory integration rule. On every choice they inspect all features, compute a signed option advantage by summing each discriminating feature's contribution, and then choose via a softmax over that advantage plus an independent lapse. Unlike Take The Best there is no early stopping. Unlike pure Tallying, each feature's contribution is weighted by a validity-sensitive multiplier. The multiplier is anchored around equal weighting, w_j = 1 + kappa * (v_j - mean(v)), normalized so the average weight is exactly 1. Because kappa has a symmetric prior centered at zero, the population-level expected weight vector is the equal-weight vector, but individual subjects may slightly overweight or underweight higher-validity cues. Tied features contribute nothing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- kappa: [-0.10, 0.10]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Validity-graded compensatory integration.
    # state is the current pair of option feature vectors, shape (2, n_features).
    # Row 0 = option A, row 1 = option B. History is irrelevant because there is
    # no trial-by-trial feedback and the rule is applied independently each trial.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-graded integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Validity-sensitive weights around a strict equal-weight anchor.
    # The centered validities sum to zero, so the raw weights sum to n_features.
    # Normalizing by their mean keeps the scale of the advantage score directly
    # comparable to an unweighted signed tally when kappa is small.
    kappa = float(parameters["kappa"])
    centered_validities = validities - np.mean(validities)
    weights = 1.0 + kappa * centered_validities
    weights = weights / np.mean(weights)

    # Discriminating features contribute according to their weights; ties
    # have difference zero and thus contribute nothing.
    advantage = float(np.dot(weights, stim[0] - stim[1]))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Stable softmax over the signed advantage versus a zero-advantage boundary.
    # This reduces to logistic choice in the weighted advantage.
    scores = np.array([advantage, 0.0])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
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
[0] This design uses six binary cues with widely spread validities. It places TTB's first discriminating cue at position 0, 1, or 2 in the validity-sorted order, while packing the later cues to oppose or tie the tallying score. In half the unique pairs the first discriminating cue favors A and in half it favors B. TTB therefore predicts p(A) is uniformly high for all six TTB-A trials and uniformly low for all six TTB-B trials, because later cues are never consulted after the first discriminating cue. Tallying instead ignores validity and counts all features: in the six oppose pairs the tally winner is opposite to TTB (count differences of 1 vs 5, 1 vs 4, and 1 vs 3), producing a crossover; in the six tie pairs tallying predicts 0.5 regardless of beta. The 12 unique pairs are each repeated 8 times for roughly 96 trials per subject. This gives a quantitative dissociation: TTB choice probability should be invariant within first-cue winner type, whereas tallying choice probability should move from near 0 to 0.5 to near 1 across the tally-difference levels.

[1] Eight experts with validities spread from .50 to .92; the highest validity is placed at display index 4 so TTB must re-sort and cannot be mimicked by left-to-right scanning. In every unique pair the top cue discriminates: trials 1-8 have top cue A, trials 9-16 top cue B. TTB therefore predicts a constant p(A) within each block (sigmoid(beta) for A-top, sigmoid(-beta) for B-top, plus lapse), because lower cues are never consulted. Tallying instead counts all eight cues, so p(A) = sigmoid(beta*d) with d = #A wins - #B wins. The d sequence is +8,+6,+4,+2,0,-2,-4,-6 for A-top and -8,-6,-4,-2,0,+2,+4,+6 for B-top. This yields a full psychometric curve for Tallying within each TTB-constant block, including d=0 pairs where Tallying must guess but TTB remains confident, and crossover pairs where the top cue and the majority of cues point in opposite directions. Mirrored blocks control for A/B response bias. 16 unique pairs x 6 repeats = 96 trials.

[2] This design isolates the validity-weighting term of the advocated theory against pure Tallying. Eight experts have validities in two clusters (0.50-0.56 and 0.94-1.00), so the centered validity sums are -0.88 for the low cluster and +0.88 for the high cluster. The critical trials are eight equal-tally (d=0) pairs: A and B each win on exactly four experts, so Tallying is forced to predict p(A)=0.5 on every one of them. Across these pairs I vary which four experts A wins on, producing a validity-weighted advantage D = kappa*S with S = +1.76, +1.00, +0.84, +0.16, -0.16, -0.84, -1.00, -1.76. Validity-graded integration therefore predicts a monotone psychometric curve across S (p(A) > 0.5 for positive S when kappa > 0, reversed when kappa < 0), whereas Tallying predicts a flat line at 0.5. Four additional d=+/-1 pairs hold the tally difference constant while flipping whether A's wins are on high- or low-validity experts (S=+/-1.57); Tallying predicts identical p(A) within each d, while the advocated theory predicts a validity-composition shift. The 12 unique pairs are repeated 8 times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design dissociates validity-graded compensatory integration from pure Tallying by crossing tally difference d with a validity-composition score S. Six low-validity (0.5) and six high-validity (1.0) experts give centered validities of -0.25 and +0.25. For each pair define d = #A wins - #B wins and S = 2 * sum of centered validities of features A wins. Tallying depends only on d, so it predicts a flat p(A)=0.5 for all eight d=0 pairs, one common value for the two d=+1 pairs, and one common value for the two d=-1 pairs. Validity-graded integration predicts D = d + kappa*S, so within each d block p(A) should vary with S. The d=0 pairs span S = +3, +2.5, +2, +1, -1, -2, -2.5, -3 (the +/-2.5 pairs include tied experts), producing a psychometric curve where Tallying is flat at 0.5. The d=+1 pairs have S=+3 and -3 (D=1.3 vs 0.7 at kappa=0.1), and the d=-1 pairs have S=+2.5 and -2.5 (D=-0.75 vs -1.25), so Tallying predicts identical p(A) within each d block while the advocated theory predicts a validity-composition shift. This is a quantitative dissociation: Tallying yields a three-level step function in d, whereas the advocated theory yields a continuous S effect within each d level. Because kappa may vary across subjects, the d=0 S curve also predicts overdispersion in individual choice proportions that Tallying cannot produce.",
  "validities": [
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
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
      0,
      1,
      1,
      1,
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
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
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
      1,
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
      1,
      1,
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
      0,
      0,
      0,
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
      1,
      1,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
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
      1,
      1,
      1,
      1,
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
      1,
      1,
      1,
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
      0,
      1,
      1,
      1,
      1,
      1,
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
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
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
      0,
      1,
      1,
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
      0,
      1,
      0,
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
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
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
      0,
      0,
      0,
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
      1,
      1,
      1,
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
  "input_tokens": 3935,
  "output_tokens": 32499,
  "reasoning_tokens": 20385,
  "provider": "CoreWeave"
}
```
