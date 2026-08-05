# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

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
**Description:** Skeptical Tallying: Decision-makers primarily rely on a Tallying heuristic, counting the number of features where one option strictly dominates the other. However, when the tallies are tied (or closely matched), they do not simply guess. Instead, they exhibit skepticism toward the highest-validity cues—perhaps viewing them as redundant, overly salient, or manipulated—and systematically break ties by favoring options that possess more lower-validity features. This is modeled by augmenting the tally score with a secondary component that weights features inversely to their stated validity.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Skeptical Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying (count of strict feature-wise wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Secondary mechanism: Tie-breaking favoring lower-validity features
    # Features are weighted by (1 - validity) so that lower validity cues provide a larger bonus.
    tie_breaker_a = np.sum(a * (1.0 - val))
    tie_breaker_b = np.sum(b * (1.0 - val))
    
    # Combine tally with the tie-breaker.
    # gamma controls the strength of the tie-breaker relative to a strict tally win.
    score_a = a_wins + gamma * tie_breaker_a
    score_b = b_wins + gamma * tie_breaker_b
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically dissociates Take The Best (TTB) from Tallying. By using five features with strictly decreasing validities, TTB relies entirely on the first feature that discriminates between the two options. Tallying, however, simply counts the number of features favoring each option, ignoring validities. The trial pairs pit a single higher-validity cue against multiple lower-validity cues, causing TTB and Tallying to make diametrically opposite predictions on most trials. Some trials also feature ties for Tallying where TTB makes a clear prediction, further separating the models' choice probabilities.

[1] This design systematically pits Tallying against Take The Best (TTB) by creating trials where the highest-validity discriminating cue favors one option, but a larger number of lower-validity cues favors the other. In Tallying, all features are weighted equally, so the option with more winning features is chosen. In TTB, only the single most valid discriminating feature matters. The design includes trials with strong opposition (e.g., 4 cues to 1), moderate opposition (e.g., 3 cues to 1), and trials where Tallying predicts a tie while TTB makes a deterministic prediction. This provides a robust quantitative dissociation across the parameter space.

[2] This design systematically dissociates the Weighted Additive (WADD) rule from Tallying by contrasting the number of winning features against the sum of their validities. Tallying ignores validities entirely, counting only the number of features favoring each option. WADD, however, computes a weighted sum where weights are derived from validities (scaled by an exponent gamma). The trial set includes 'opposition' trials where one option is supported by fewer but highly valid features, while the other is supported by a greater number of low-validity features, causing the models to predict opposite choices. It also includes 'tie' trials where both options win on an equal number of features (Tallying predicts a 50/50 guess) but the features differ in validity, leading WADD to predict a strong preference.

[3] This design systematically dissociates Tallying from the Weighted Additive (WADD) rule by contrasting the sheer quantity of winning features against the sum of their validities. Tallying ignores validities, relying entirely on the count of features favoring each option. WADD computes a weighted sum where validities (scaled by gamma) determine the weights. By using a bimodal validity distribution (two highly valid features and three low-validity features), we create 'opposition' trials where one option wins on the two highly valid features while the other wins on the three low-validity features. Tallying predicts a win for the option with three features, whereas WADD strongly favors the option with the two highly valid features (provided gamma > 0). We also include 'tie' trials where both options win on an equal number of features, leading Tallying to predict a 50/50 guess, while WADD predicts a strong preference based on validity differences.

[4] This design systematically dissociates Standard Tallying from Skeptical Tallying by focusing on trials where the primary Tallying mechanism results in a tie. Under Standard Tallying, when both options win on an equal number of features, the model predicts a 50/50 guess regardless of the features' validities. In contrast, Skeptical Tallying breaks these ties by weighting the features inversely to their stated validity, favoring the option that possesses more lower-validity features. By constructing trials where option A wins on low-validity features and option B wins on high-validity features, Standard Tallying predicts indifference, whereas Skeptical Tallying predicts a clear preference for option A. Additional trials vary the magnitude of the tie-breaker difference to capture the effect across the parameter space.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Standard Tallying from Skeptical Tallying, we exploit the latter's 'tie-breaker' mechanism, which effectively functions as an additive bonus that scales inversely with feature validity. For Standard Tallying, all feature wins are weighted equally (weight = 1), and validities are ignored. For Skeptical Tallying, a feature win contributes 1 + gamma * (1 - validity). Thus, Skeptical Tallying assigns strictly higher weight to wins on low-validity features. We use 7 features (four high validity 0.95, three low validity 0.50). In 'tie' trials (e.g., A wins on 2 high-validity features, B wins on 2 low-validity features), Standard Tallying predicts a 50/50 guess, while Skeptical Tallying strongly prefers B. In 'opposition' trials (A wins 4 high-validity features, B wins 3 low-validity features), Standard Tallying deterministicly prefers A (4 wins vs 3), whereas Skeptical Tallying can flip its preference to B if the skepticism parameter gamma is sufficiently high, because the cumulative low-validity bonus outweighs the single extra win for A.",
  "validities": [
    0.95,
    0.95,
    0.95,
    0.95,
    0.5,
    0.5,
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
      0
    ],
    [
      1,
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
      0,
      0
    ],
    [
      1,
      0,
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
      0,
      1,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
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
      1,
      0
    ],
    [
      0,
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
      0,
      0,
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
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3282,
  "candidates_token_count": 636,
  "total_token_count": 7869
}
```
