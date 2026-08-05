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
**Description:** Decision-makers use the 'Take-The-Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They search through the available features in descending order of their validities. The first feature that discriminates between the two options determines the choice, and all subsequent features are completely ignored. If no features discriminate between the options, the decision-maker guesses randomly. Response noise is incorporated as an overall lapse rate (epsilon), representing trials where the subject guesses uniformly at random instead of applying the heuristic.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(-val, kind='mergesort')
    
    # Default to guessing if no feature discriminates
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Decision-makers use a Compensatory Weighted Additive (WADD) strategy where they integrate all available cues, weighting each by its validity centered around chance (validity - 0.5). This linear integration allows multiple weaker cues to appropriately accumulate and potentially override a single strong cue, avoiding the extreme dominance that log-odds weighting can cause. Choices are made probabilistically via a softmax function over the accumulated evidence, with an independent lapse rate accounting for random guesses.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Extract validities
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Linear weighting centered around chance (0.5) to prevent extreme weights
    weights = val - 0.5
    
    # Weighted additive sums
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Tallying, we design trials where the option favored by the most valid cue is opposed by the majority of lower-validity cues. TTB will strictly follow the highest-validity discriminating cue, ignoring the total count of winning cues. In contrast, Tallying ignores validities entirely and simply counts the number of cues each option wins. We include trials where TTB chooses one option while Tallying chooses the other, as well as trials where Tallying is indifferent (predicts a 50/50 guess) but TTB has a strong preference.

[1] To quantitatively dissociate Tallying (which gives equal weight to all cues and chooses the option with the most feature wins) from Take The Best (which relies solely on the most valid discriminating cue), we design trials where the two models make completely opposed predictions or where one model predicts a tie while the other strongly prefers one option. By using 5 features with cascading validities, we construct cases where the highest-validity cue points to one option, but a majority of lower-validity cues point to the other. Additionally, we include trials where the top cues tie, forcing TTB to rely on a mid-validity cue, while Tallying counts the remaining features to favor the opposite option. This creates a clear double dissociation in choice probabilities.

[2] To quantitatively dissociate WADD from Tallying, we use a 5-feature design with a steep drop-off in validities. Tallying simply counts the number of features each option wins, ignoring the validities entirely. WADD weights each feature by its validity. We include trials where WADD strongly prefers an option with fewer winning features because it wins on the highest validity cues (e.g., A wins on the top 2 features, B wins on the bottom 3 features). We also include trials where Tallying predicts indifference (a 2-vs-2 tie in feature wins), but WADD has a strong preference due to the specific validities of the winning features. This ensures a broad range of predicted choice probabilities that sharply separate the two models.

[3] To quantitatively dissociate Tallying from Weighted Additive (WADD), we use a 5-feature design with an extreme drop-off in validities (one cue at 1.0, and the rest at 0.5). Tallying ignores validities entirely and simply counts the number of features each option wins. WADD evaluates options by their validity-weighted sums. This allows us to create trials where Tallying strictly prefers an option (because it wins on three low-validity cues) while WADD strongly prefers the other (because it wins on the single highest-validity cue plus one low-validity cue). We also include trials where WADD is indifferent but Tallying has a strong preference, and vice versa. This full spectrum of dissociations provides a sharp contrast between the two models' predictions.

[4] To strongly dissociate Take The Best (TTB) and Tallying, we use 5 features with cascading validities. We create trials where the highest-validity cue points to one option, but a majority of lower-validity cues point to the other. This pits TTB's lexicographic stopping rule against Tallying's unweighted feature counting. We include variations where the top cues tie, forcing TTB to rely on a mid-validity cue while Tallying still counts the remaining features to favor the opposite option. We also include trials where Tallying predicts a tie but TTB finds a discriminating cue, resulting in a strong preference for TTB while Tallying guesses.

[5] To cleanly dissociate Tallying from Take The Best (TTB) and avoid repeating previous monotonic 5-feature designs, we employ a 6-feature design with unordered validities. TTB must internally sort the features by validity and decide based on the first discriminating cue, whereas Tallying simply sums the unweighted feature wins, completely ignoring validities. We design trials where the single highest-validity cue favors one option, but a large majority of lower-validity cues favor the other. We also include trials where the highest valid cues tie, forcing TTB to drop to secondary cues while Tallying counts the remaining features. Finally, we include trials where Tallying predicts a strict tie (equal number of feature wins) but TTB has a strong preference, creating a robust quantitative double dissociation.

[6] To quantitatively dissociate Weighted Additive (WADD) from Tallying, we use a 5-feature design with linearly decreasing validities. WADD centers validities around 0.5 to compute weights (0.4, 0.3, 0.2, 0.1, 0.0), whereas Tallying simply counts the number of winning features for each option, completely ignoring validities. We design trials where WADD predicts indifference but Tallying strongly prefers one option (e.g., A wins on the highest validity cue while B wins on three lower validity cues). We also include trials where Tallying predicts a tie (equal number of feature wins) but WADD strongly prefers one option due to higher validity weights. Finally, we include trials where the models make strictly opposing predictions, creating a robust quantitative double dissociation between the compensatory validity-weighted integration of WADD and the unweighted feature counting of Tallying.

[7] To quantitatively dissociate Tallying from WADD, we use a 6-feature design with two highly valid cues and four weakly valid cues. WADD translates these into weights (val - 0.5): [0.5, 0.4, 0.1, 0.1, 0.1, 0.1]. Tallying simply counts the number of winning features, completely ignoring the validities. We design trials where WADD strongly prefers an option because it wins on the top two features, while Tallying prefers the other option because it wins on the four weaker features. We also include trials where Tallying predicts a tie (equal number of feature wins) but WADD has a strict preference, and trials where WADD predicts a tie (equal weighted sum) but Tallying has a strict preference. This ensures a robust double dissociation between unweighted feature counting and compensatory validity-weighted integration.

[8] To quantitatively dissociate the Compensatory Weighted Additive (WADD) strategy from the non-compensatory Take-The-Best (TTB) heuristic, we use a 5-feature design with unordered, cascaded validities. TTB relies solely on the highest-validity discriminating feature, ignoring all others. In contrast, WADD integrates all features using weights derived from their validities centered around chance (val - 0.5). We design trials where the single best discriminating feature favors one option (which TTB will choose), but a combination of several lower-validity features favors the other option (which WADD will choose). We also include trials where WADD predicts a strict tie (the sum of lower weights exactly equals the highest weight) while TTB has a strong preference, creating a robust quantitative double dissociation.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Take-The-Best (TTB) and Weighted Additive (WADD) strategies, we use a 5-feature design with unordered validities. TTB relies solely on the highest-validity discriminating feature, whereas WADD integrates all features using weights derived from their validities centered around chance (val - 0.5). We design trials where the single best discriminating feature favors one option (which TTB will choose), but a combination of several lower-validity features strongly favors the other option (which WADD will choose). We also include trials where WADD predicts a strict tie (the sum of lower weights exactly equals the highest weight) while TTB has a strong preference, and trials where top cues tie, forcing TTB to drop to secondary cues while WADD continues to integrate all evidence.",
  "validities": [
    0.6,
    0.8,
    0.6,
    0.9,
    0.7
  ],
  "trial_a_ratings": [
    [
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
      0
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      1,
      0,
      0,
      1
    ],
    [
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
      1,
      1
    ],
    [
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
      0
    ],
    [
      0,
      1,
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
  "prompt_token_count": 3538,
  "candidates_token_count": 447,
  "total_token_count": 6046
}
```
