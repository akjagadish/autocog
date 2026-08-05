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
**Description:** Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Evidence-Dependent Noise: Decision-makers evaluate options using a single compensatory mechanism (Weighted Additive) where features are weighted by a subjective transformation of their validities. However, the decision process is subject to evidence-dependent noise: the temperature of the softmax choice rule scales with the total conflict between the options (defined as the total weighted evidence of features that differ between the two options). This ensures that trials with higher evidence magnitudes or greater feature conflict naturally generate higher decision noise. This single-mechanism approach preserves log-odds linearity while explaining why extremeness flattens across trials with varying evidence magnitudes.

**Parameters:**
- gamma: [0.0, 5.0]
- beta: [0.1, 20.0]
- theta: [0.0, 10.0]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus expects shape (2, n_features); got {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective feature weighting
    weights = val ** gamma
    
    # Calculate weighted sums of evidence for each option
    scores = np.dot(stim, weights)
    
    # Calculate conflict: total weight of features where the options differ
    diff = np.abs(stim[0] - stim[1])
    conflict = np.dot(diff, weights)
    
    # Effective beta scales inversely with conflict (higher conflict = more noise)
    beta_eff = beta / (1.0 + theta * conflict)
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta_eff * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The experiment is designed to strongly dissociate Take The Best (TTB) from Tallying. TTB decides solely based on the single most valid cue that discriminates between the options, ignoring the sheer number of cues that support either option. Tallying, conversely, simply counts the number of cues favoring each option, ignoring cue validities. The trial pairs pit a single high-validity cue against multiple lower-validity cues. In these configurations, TTB predicts that the option winning on the single best cue will be chosen, whereas Tallying predicts that the option with a higher total number of winning cues will be chosen. Some trials also feature ties on the most valid cues to ensure TTB's cue search cascade is tested.

[1] This design dissociates Tallying from Take The Best (TTB) by pitting the highest-validity discriminating cue against the sheer quantity of supporting cues. TTB decides solely based on the first cue in the validity hierarchy that discriminates between the two options. In contrast, Tallying ignores validity weights entirely and simply counts the number of features on which each option is superior. The trial pairs contain strict oppositions where one option wins on the single highest-validity discriminating cue while the other option wins on multiple lower-validity cues, leading to divergent predictions. Control trials where both models agree are also included to measure baseline choice noise.

[2] This design dissociates the Weighted Additive (WADD) theory from Tallying by manipulating the distribution of cue validities across the options. WADD integrates all features by weighting them according to a subjective transformation of their validities, whereas Tallying ignores validities entirely and simply counts the number of features favoring each option. The trial set includes pairs where one option is supported by a few high-validity cues while the other is supported by a larger number of low-validity cues, leading to opposing predictions. Additionally, several trials present an equal number of winning features for both options (a tie for Tallying), where WADD strongly prefers the option with higher-validity features.

[3] This design systematically dissociates Tallying from Weighted Additive (WADD) by contrasting the quantity of winning features against their magnitude (validity). Tallying relies strictly on the count of feature-wise wins, predicting indifference (50/50) when the number of winning features is tied, and favoring the option with more winning features otherwise. WADD, conversely, integrates the subjective validities. We include trials where the option with fewer winning features has much higher validities (WADD prefers the option with fewer wins, Tallying prefers the option with more), as well as 'tie' trials where both options win on an equal number of features but one has higher validities (Tallying predicts exactly 50/50, WADD predicts a strong preference).

[4] This design dissociates Weighted Additive (WADD) from Validity-Thresholded Tallying by exploiting the non-monotonic preference reversals predicted by the thresholding mechanism. We use five cues with linearly decreasing validities. In key trials, Option A is supported by the single most valid cue and the lowest valid cues, while Option B is supported by the intermediate valid cues. Because Thresholded Tallying applies a hard validity cutoff and then tallies the remaining cues equally, varying the threshold parameter tau leads to wild, non-monotonic swings in preference (e.g., A wins -> Tie -> B wins -> Tie -> A wins). In contrast, WADD integrates all cues with graded weights, predicting a consistent preference for the option with the higher weighted sum. This structural difference makes it impossible for WADD's smooth weighting (via gamma) to mimic the discrete, threshold-driven choice patterns of Thresholded Tallying.

[5] This design dissociates Validity-Thresholded Tallying from Weighted Additive (WADD) by leveraging the discrete, non-monotonic choice patterns produced by a hard validity threshold compared to smooth exponential weighting. By using five cues with a linear spread of validities, we construct trials where the highest valid cue is pitted against various combinations of lower valid cues. For example, when Option A is supported by cue 1 and 4, and Option B by cues 2 and 3, Thresholded Tallying predicts a non-monotonic sequence of preferences (e.g., A -> Tie -> B -> Tie) as the validity threshold is lowered. WADD, in contrast, applies a continuous convex/concave transformation to the validities via the gamma parameter, which cannot capture these discrete step-like preference reversals. The chosen trial pairs maximize the divergence between a step-function tallying approach and a continuous weighted-sum approach.

[6] To quantitatively dissociate pure WADD from a mixture of WADD and Probabilistic Tallying, we must target the core difference: the mixture model assumes that on a subset of trials, subjects ignore validities entirely and rely on a soft count of winning features. By generating a stimulus set where the WADD evidence difference and the Tallying evidence difference are completely orthogonal, we can test whether choice probabilities scale consistently with a single unified evidence metric (pure WADD) or show systematic deviations on trials where Tallying pulls the probability in a different direction. If human subjects truly use a unified WADD process with a single non-linear validity transformation (gamma) and noise parameter (beta), the choice probabilities will be perfectly predicted by the WADD score differences, forcing the mixture model to estimate w_wadd near 1 and rendering the Tallying component superfluous.

[7] To quantitatively dissociate pure WADD from the Strategy Selection (WADD + Probabilistic Tallying) mixture model, this design manipulates the magnitude of the Tallying evidence (the difference in the number of winning features) while systematically varying the WADD evidence (the difference in weighted sums). In pure WADD, choice probabilities are determined solely by a non-linear scaling of validities (gamma) and a noise parameter (beta). In the mixture model, the addition of a probabilistic Tallying component means that on a subset of trials, choices are driven by a softmax over the raw count of winning features. By including trials where the Tallying difference is extreme (e.g., 4 vs 1) but the WADD difference is either small or points in the opposite direction, and comparing them to trials where the Tallying difference is zero (ties) but the WADD difference is similar, we force the mixture model to predict a specific distortion in the psychometric function. Pure WADD cannot capture this independent sensitivity to feature-count differences because its beta parameter must symmetrically scale the single WADD evidence axis.

[8] To quantitatively dissociate pure Weighted Additive (WADD) from Evidence-Dependent Noise, we must orthogonalize the net evidence difference between the options and the total feature conflict. Pure WADD predicts that choice probabilities are determined solely by the difference in subjectively weighted evidence. The Competing Theory posits that the choice temperature scales with total conflict (the sum of weights of features that differ between the options). By using pairs of features with identical validities (e.g., 0.8 and 0.8), we can add 'canceling' evidence to both options. This leaves the net evidence difference perfectly unchanged for ANY subjective validity transformation (gamma), but strictly increases the total conflict. Pure WADD predicts identical choice probabilities across these conflict levels, whereas Evidence-Dependent Noise predicts probabilities will regress toward 0.5 on high-conflict trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate pure Weighted Additive (WADD) from Evidence-Dependent Noise, we must orthogonalize the net evidence difference between the options and the total feature conflict. Pure WADD predicts that choice probabilities are determined solely by the difference in subjectively weighted evidence. The Competing Theory posits that the choice temperature scales with total conflict (the sum of weights of features that differ between the options). By using pairs of features with identical validities (e.g., 0.9 and 0.9), we can add 'canceling' evidence to both options. This leaves the net evidence difference perfectly unchanged for ANY subjective validity transformation (gamma), but strictly increases the total conflict. Pure WADD predicts identical choice probabilities across these conflict levels, whereas Evidence-Dependent Noise predicts probabilities will regress toward 0.5 on high-conflict trials.",
  "validities": [
    0.9,
    0.9,
    0.7,
    0.7,
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
      0,
      1,
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
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
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
      0,
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
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
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
      1,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3829,
  "candidates_token_count": 691,
  "total_token_count": 7404
}
```
