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
**Description:** People employ a non-compensatory Take-The-Best (TTB) heuristic to make decisions under uncertainty. Instead of integrating all available features, decision-makers rank cues by their subjective validity and inspect them sequentially in descending order. The search stops at the very first cue that discriminates between the two options, and the choice is determined entirely by this single cue. If no cues discriminate after inspecting all of them, the decision-maker guesses randomly. Response stochasticity is modeled simply through a lapse rate (epsilon), rather than a softmax, reflecting a constant probability of execution error or random guessing regardless of the cue.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Rank cues by validity in descending order
    order = np.argsort(val)[::-1]
    
    epsilon = float(parameters["epsilon"])
    
    p_a = 0.5
    p_b = 0.5
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_a = 1.0 - epsilon
            p_b = epsilon
            break
        elif stim[1, idx] > stim[0, idx]:
            p_a = epsilon
            p_b = 1.0 - epsilon
            break
            
    return np.array([p_a, p_b])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People make choices by integrating all available feature information, weighting each cue by its subjective validity. Unlike Tallying (which weights all cues equally) or Take The Best (which stops at the first discriminating cue), the Weighted Additive (WADD) model computes an overall expected value for each option by summing the products of the feature values and their validities. Choice probabilities are then generated via a softmax function over these weighted sums, allowing for graded sensitivity to both the number of supporting features and their relative importance.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: compute the weighted sum of features for each option
    score_a = np.sum(stim[0] * val)
    score_b = np.sum(stim[1] * val)
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
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically pits Take The Best (TTB) against Tallying by creating configurations where the single most valid discriminating cue points to one option, while the sheer number of winning cues points to the other option (or results in a tie). For TTB, choice is dictated entirely by the first cue in the validity hierarchy that differs between A and B, ignoring the rest. For Tallying, all cues are weighted equally, and the option with the most 'wins' is chosen. By including trials where a single high-validity cue is outweighed by multiple lower-validity cues, we can quantitatively dissociate the two theories.

[1] This design uses 5 features with strictly descending validities to cleanly separate Tallying from Take The Best (TTB). Trials are constructed such that the highest-validity discriminating cue often points in the opposite direction of the simple majority of features. In these adversarial trials, TTB's 'one-reason' stopping rule predicts choice based exclusively on the top discriminating cue, while Tallying integrates all cues equally, leading to either the opposite choice or a predicted tie (when wins are balanced). A mix of adversarial, tie-breaking, and agreement trials ensures a robust quantitative dissociation of the two models across the parameter space.

[2] This design quantitatively dissociates the Weighted Additive (WADD) model from Tallying by contrasting the sheer quantity of winning cues against their cumulative validity. Tallying ignores specific validity magnitudes and counts feature-wise wins, predicting ties when wins are matched and favoring the option with more wins regardless of cue strength. WADD integrates exact validities, allowing a smaller number of high-validity features to outweigh a larger number of low-validity features. The trial set includes adversarial conditions where Tallying and WADD make strictly opposing predictions (e.g., 3 low-validity wins vs. 2 high-validity wins), tie-breaking conditions where Tallying predicts indifference but WADD has a strong preference, and agreement trials to balance the design and ensure robust parameter estimation.

[3] This design quantitatively dissociates Tallying (which simply counts the number of winning features) from Weighted Additive (WADD) (which integrates both the number of features and their exact validities). By employing a skewed distribution of validities (two high, three low), we can construct adversarial trials where one option wins on more features but the other option has a higher weighted sum. We also include tie-breaking trials where Tallying predicts indifference (equal number of wins) but WADD predicts a strict preference due to validity differences.

[4] This design quantitatively dissociates the Weighted Additive (WADD) model from Take The Best (TTB) by contrasting the highest-validity discriminating cue against the combined weight of multiple lower-validity cues. TTB employs a non-compensatory stopping rule, making its choice based exclusively on the single most valid cue that discriminates between the options. WADD, on the other hand, is fully compensatory, integrating all features weighted by their validities. By using a steep but surmountable validity hierarchy (e.g., 0.95 vs. 0.70, 0.65, 0.60, 0.55), we construct adversarial trials where the single best cue strongly points to one option, but the sum of the remaining cues outweighs it, leading WADD to choose the opposite option. We also include agreement trials and partial-conflict trials to ensure robust parameter estimation for both models.

[5] This design aims to strongly dissociate Take The Best (TTB) from the Weighted Additive (WADD) model by exploiting the compensatory nature of WADD versus the non-compensatory stopping rule of TTB. We use a set of 5 features with a relatively flat validity hierarchy (0.90, 0.85, 0.80, 0.75, 0.70). In many trials, the single most valid discriminating cue points to Option A, but the sum of the remaining lower-validity cues strongly points to Option B. TTB will invariably choose the option favored by the first discriminating cue in the hierarchy, ignoring the rest. WADD integrates all cues, allowing the sheer number of lower-validity cues to outweigh the single best cue. The trial set includes adversarial configurations, partial agreement, and full agreement trials to ensure robust parameter estimation for both models.

[6] This design quantitatively dissociates the Weighted Additive (WADD) model from the Rank-Weighted Additive model by exploiting the difference between cardinal validities and ordinal ranks. We use a set of validities with a highly non-linear spacing: one very high validity (0.95) and four clustered lower validities (0.60, 0.58, 0.56, 0.54). For WADD, the sum of two or three of the lower validities can easily outweigh or be outweighed by the top validity depending on the exact cardinal values. However, in the Rank model, the validities are transformed to strictly linear integer weights (5, 4, 3, 2, 1). This creates adversarial trials where WADD predicts one choice based on the exact sum of probabilities, while the Rank model predicts the opposite choice because the ordinal ranks dictate a different balance of power.

[7] This design systematically dissociates the Weighted Additive (WADD) model from the Rank-Weighted Additive model by leveraging a specific distribution of validities: one very high validity and four spaced-out lower validities. While WADD integrates the exact cardinal validities, the Rank model converts them into strictly ordinal integer weights (5, 4, 3, 2, 1). By carefully pairing options, we create adversarial scenarios where the sum of specific cardinal validities favors one option (e.g., 0.98 + 0.51 = 1.49 vs 0.75 + 0.65 = 1.40), but the sum of their ordinal ranks favors the other (e.g., 5 + 1 = 6 vs 4 + 3 = 7). Additional trials where the models predict ties versus strict preferences ensure a robust quantitative dissociation across the parameter space.

[8] This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take-The-Best (TTB) heuristic using a steep but surmountable validity hierarchy. TTB makes a decision based entirely on the highest-validity cue that discriminates between the two options. WADD evaluates the weighted sum of all cues. By pairing options where the single highest-validity cue favors Option A but the sum of multiple lower-validity cues heavily favors Option B, we create adversarial trials where the two models make strictly opposing predictions. We also include tie-breaking trials for WADD (where weighted sums are equal but TTB finds a discriminating cue) and agreement trials, ensuring a robust test of choice stochasticity and integration mechanisms.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates Take-The-Best (TTB) from the Weighted Additive (WADD) model by using a steep but surmountable validity hierarchy. TTB makes decisions based solely on the highest-validity discriminating cue, ignoring all others. WADD integrates all cues, allowing the sum of several lower-validity cues to outweigh the single highest-validity cue. By presenting trials where the top cue points to one option but the sum of the remaining cues heavily favors the other, we create adversarial conditions. We also include agreement trials and close-call trials where the models either align or predict differing choice probabilities, ensuring robust parameter estimation and clear dissociation of the two theories.",
  "validities": [
    0.92,
    0.71,
    0.68,
    0.62,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
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
      0
    ],
    [
      1,
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0
    ],
    [
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
      0
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
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
      1
    ]
  ],
  "trial_b_ratings": [
    [
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
      0
    ],
    [
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1
    ],
    [
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
      0
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
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
      0
    ],
    [
      1,
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
  "prompt_token_count": 3484,
  "candidates_token_count": 560,
  "total_token_count": 5573
}
```
