# experiment_attempt_01

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
**Description:** Thresholded Tallying (Soft Subset Tallying): Decision makers use an equal-weight heuristic, but first filter the available information by cue validity. Instead of a hard threshold, they use a soft, probabilistic inclusion criterion where features with validity near or above a subjective threshold are more likely to be considered. These softly filtered features are then tallied to form a preference. This differentiable thresholding allows smooth transitions between pure Tallying (all cues included) and stricter subset tallying, reflecting boundedly rational cue selection that is easier to optimize during learning or fitting.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [-1.0, 1.0]
- k: [1.0, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Tallying expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    
    # Filter features: soft threshold using a logistic sigmoid
    mask = 1.0 / (1.0 + np.exp(-k * (validities - theta)))
    
    # Tallying: count the number of positive features among the softly kept ones
    scores = np.sum(stim * mask, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Weighted Additive (WADD), we use a set of four features with a steep drop-off in validity for the best cue, but where the sum of the lower validities exceeds the highest validity. In 'compensatory' trials, one option is endorsed by the single most valid cue, while the other option is endorsed by multiple less valid cues. TTB, being a non-compensatory 'one-reason' heuristic, will invariably choose the option favored by the single best discriminating cue. WADD, which integrates all information, will choose the option favored by the sum of the lesser cues, provided their combined weight is greater. We also include 'non-compensatory' trials where both models agree, serving as a baseline for accuracy and attention.

[1] To quantitatively dissociate Weighted Additive (WADD) from Take The Best (TTB), we use a five-feature design where the validity of the cues decreases progressively. The trials are constructed such that the single most valid discriminating cue always favors one option (which TTB will choose, ignoring all other cues), while the sum of the validities of the remaining discriminating cues strongly favors the other option (which WADD will choose, as it integrates all available information). By varying which cue is the highest discriminating one (e.g., cue 1, cue 2, or cue 3) and having the remaining cues compensate for it, we ensure that TTB and WADD make diametrically opposed predictions across a variety of comparisons, completely separating the non-compensatory 'one-reason' stopping rule from the compensatory weighted-sum rule.

[2] To quantitatively dissociate Tallying (Equal Weight) from Weighted Additive (WADD), we utilize a 5-feature design where the sum of the validities of the two most valid cues exceeds the sum of the remaining three less valid cues. In the critical dissociation trials, one option is endorsed by the two highly valid cues, while the competing option is endorsed by the three lower-validity cues. Tallying ignores cue validities and simply counts the positive features, thus strictly preferring the option with three endorsements. WADD, however, computes a weighted sum and will prefer the option with the two highly valid cues. We also include trials where both models agree, and trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strong preference, allowing for a precise estimation of the models' parameters.

[3] This design quantitatively dissociates the Weighted Additive (WADD) model from the Tallying (Equal Weight) model. By using a 5-feature design with a steep drop-off in validities between the top two features and the bottom three, we create critical dissociation trials where one option is supported by the two highly valid features (WADD preference) while the other option is supported by the three lower-validity features (Tallying preference). We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference due to validity weighting, and baseline trials where both models agree.

[4] This design dissociates pure Tallying from Thresholded Tallying by exploiting a sharp difference in cue validities. The 6 features are split into a high-validity group (cues 1-3) and a low-validity group (cues 4-6). Pure Tallying ignores these validities and simply counts the total number of positive features. Thresholded Tallying, however, applies a soft filter that down-weights or eliminates the low-validity cues. By pitting options with many low-validity endorsements against options with fewer high-validity endorsements, pure Tallying will choose the option with the higher total count, whereas Thresholded Tallying will choose the option with the higher count among the high-validity cues. Ties in Tallying are also broken by Thresholded Tallying based on the distribution of high-validity cues.

[5] To quantitatively dissociate Thresholded Tallying from standard Tallying (Equal Weight), we use a 5-feature design with a clear gap in validities: three high-validity features and two low-validity features. Standard Tallying ignores validities entirely and counts all positive features equally. Thresholded Tallying, however, applies a soft threshold that down-weights or eliminates the low-validity features, effectively tallying only the high-validity ones. By pitting options with fewer high-validity features against options with a greater total number of features (which are mostly low-validity), we create clear dissociation trials. For instance, an option with 2 high-validity and 0 low-validity features will be preferred by Thresholded Tallying over an option with 1 high-validity and 2 low-validity features, whereas standard Tallying will prefer the latter due to its higher unweighted sum.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Thresholded Tallying from Pure Tallying (Equal Weight), we use a 6-feature design with a clear bimodal distribution of validities: three high-validity features (0.85-0.95) and three low-validity features (0.50-0.60). Pure Tallying ignores validities and counts all positive features equally. Thresholded Tallying applies a soft threshold that filters out low-validity features, effectively tallying only the high-validity ones. By pitting options with fewer high-validity features against options with a greater total number of features (which are predominantly low-validity), we create stark dissociations. For example, an option with 2 high-validity and 0 low-validity features will be chosen by Thresholded Tallying over an option with 0 high-validity and 3 low-validity features, while Pure Tallying predicts the opposite. We also include tie-breaking trials where Pure Tallying predicts equal preference but Thresholded Tallying strongly favors the option with high-validity cues.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.6,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
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
      1,
      0,
      0,
      1,
      1,
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
      0,
      1,
      1,
      0,
      0,
      1
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
      0,
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
      1,
      1
    ],
    [
      1,
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
      0
    ],
    [
      1,
      0,
      0,
      0,
      1,
      1
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
      1,
      1,
      0
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
      1,
      1,
      1
    ],
    [
      0,
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
  "prompt_token_count": 3146,
  "candidates_token_count": 724,
  "total_token_count": 5593
}
```
