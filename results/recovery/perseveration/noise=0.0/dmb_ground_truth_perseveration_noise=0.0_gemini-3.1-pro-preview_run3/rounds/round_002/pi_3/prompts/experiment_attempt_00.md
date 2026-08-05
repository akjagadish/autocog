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
**Description:** Decision-makers use a Weighted Additive (WADD) strategy, integrating all available features weighted by their validities, but they are subject to significant spatial/positional biases (e.g., a baseline preference for Option A over Option B) and high levels of task disengagement (lapse rate). In environments where subjects ignore features, the positional bias and lapse rate dominate the choice, leading to choices that appear completely orthogonal to standard heuristic predictions like TTB or Tallying (yielding exact 0.5 consistency).

**Parameters:**
- beta: [0.0, 10.0]
- bias_A: [-20.0, 20.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive (WADD) scores
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    
    beta = float(parameters["beta"])
    bias_a = float(parameters["bias_A"])
    epsilon = float(parameters["epsilon"])
    
    # Incorporate spatial/positional bias for Option A
    logits = np.array([beta * score_a + bias_a, beta * score_b])
    
    # Numerically stable softmax
    logits = logits - np.max(logits)
    p_core = np.exp(logits) / np.sum(np.exp(logits))
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Subjects exhibit complete disengagement from the multi-attribute decision task. Rather than evaluating the options based on their features and the validities of those features, subjects ignore all stimulus information and simply guess at random on every trial, choosing Option A or Option B with equal probability.

**Parameters:**
(none)

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The model completely ignores the state and history, 
    # reflecting total task disengagement.
    return np.array([0.5, 0.5])
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
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, the experimental design pits the highest-validity discriminating cue against the sheer number of winning cues. In TTB, the option with the higher value on the most valid discriminating cue is chosen, ignoring all other cues. In Tallying, the option that has a higher value on more cues is chosen, ignoring cue validities completely. The trials are constructed such that one option wins on the single most valid discriminating cue, while the other option wins on a larger number of less valid cues. This leads to completely opposite predictions between the two models.

[1] This design systematically pits the single most valid discriminating cue against the overall number of winning cues. In Take The Best (TTB), only the highest validity feature that discriminates between the two options is considered, completely ignoring the sheer volume of supporting features. In contrast, Tallying ignores feature validities and simply counts the number of features favoring each option. The trial set includes pairs where TTB strongly prefers one option while Tallying prefers the other (due to a larger number of lower-validity wins), as well as pairs where Tallying predicts a tie (equal number of winning features) but TTB makes a deterministic prediction based on the highest-validity cue.

[2] To quantitatively dissociate Take The Best (TTB) from a Weighted Additive (WADD) strategy that is subject to strong spatial biases and lapses, we design trials where the single highest-validity discriminating cue strongly opposes the sum of the remaining lower-validity cues. In TTB, the choice is deterministically driven by the first discriminating cue in descending order of validity, completely ignoring the cumulative weight of all other cues. In WADD, the sum of the lower-validity cues can easily outweigh the single highest-validity cue. By symmetrically alternating whether Option A or Option B possesses the highest-validity cue or the bulk of the lower-validity cues, we can also separate true WADD integration from a simple Option A positional bias.

[3] This design dissociates WADD (with its potential for spatial bias and feature integration) from Take The Best (TTB). We include 'compensatory' trials where the highest-validity cue points to one option while the sum of the remaining cues points to the other, separating TTB's non-compensatory one-reason decision making from WADD's integration. Crucially, we also include 'tie' trials where both options have identical feature profiles; TTB must guess (predicting exactly 50/50), whereas the advocated WADD model will reveal any underlying spatial/positional bias (e.g., a baseline preference for Option A).

[4] To quantitatively dissociate the Advocated Theory (pure random guessing / complete disengagement) from the Competing Theory (WADD with spatial bias and lapse rate), this design includes three types of trials: (1) Extreme dominance trials where Option A or Option B has all positive features, maximizing the WADD difference. (2) Compensatory trials where the highest-validity cue opposes the sum of lower-validity cues, testing sensitivity to weighted feature integration. (3) Identical-profile (tie) trials where both options have the exact same features. The Advocated Theory rigidly predicts exactly 50/50 choice probabilities across all trial types. In contrast, the Competing Theory will predict strong deviations from 50/50 on dominance and compensatory trials (via the beta parameter) and will reveal any baseline positional preference on the identical-profile trials (via the bias_A parameter).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Advocated Theory (WADD with spatial bias and lapse rate) from the Competing Theory (pure random guessing), we must expose variations in choice probabilities that random guessing cannot capture. The Competing Theory rigidly predicts a 50/50 choice split on every trial, regardless of the stimuli. The Advocated Theory predicts that choices will track the weighted sums of features (modulated by sensitivity 'beta') and may exhibit a baseline preference for Option A (via 'bias_A'). We include extreme dominance trials (where one option has all positive features and the other has none) to elicit maximum deviation from 50/50. We also include compensatory trials to test sensitivity to feature integration, and identical-profile (tie) trials where both options have the same features. In tie trials, random guessing still predicts exactly 0.5, whereas the Advocated Theory will reveal any underlying spatial bias.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2557,
  "candidates_token_count": 479,
  "total_token_count": 3624
}
```
