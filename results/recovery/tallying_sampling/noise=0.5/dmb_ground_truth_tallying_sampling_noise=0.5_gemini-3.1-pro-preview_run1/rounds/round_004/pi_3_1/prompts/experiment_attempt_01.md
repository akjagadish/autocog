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
**Description:** Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

**Parameters:**
- beta: [0.0, 3.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weights for all features, ignoring validities.
    # The score for each option is simply the sum of its positive features.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Cancellation and Focus (Ratio Rule)

**Parameters:**
- beta: [0.0, 5.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Cancellation: eliminate features shared by both options.
    # Focus only on unique positive features.
    unique_a = np.sum((stim[0] > stim[1]).astype(float))
    unique_b = np.sum((stim[1] > stim[0]).astype(float))
    
    scores = np.array([unique_a, unique_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if unique_a == 0 and unique_b == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Ratio rule over unique features to provide a distinct alternative to Tallying's softmax.
        s_beta = np.zeros_like(scores)
        for i in range(len(scores)):
            if scores[i] > 0:
                s_beta[i] = scores[i] ** beta
            else:
                s_beta[i] = 0.0 if beta > 0 else 1.0
        p_core = s_beta / np.sum(s_beta)
        
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD), we must create trials where the single most valid discriminating feature opposes the compensatory sum of all other valid features. By using 5 features with a descending spread of validities, we can design 'compensatory' trials where one option wins on the highest-validity cue but loses on the sum of several lower-validity cues. TTB will consistently choose the option with the highest-validity cue, ignoring the rest, whereas WADD will integrate across all cues, leading it to choose the option with the greater overall weighted sum. We also include agreement trials to establish baseline performance.

[1] To cleanly dissociate WADD from TTB, we employ a 5-feature environment with linearly spaced validities. We design 'compensatory' trials where one option excels on the single most valid cue (which TTB relies upon exclusively), while the other option excels on multiple lower-validity cues such that their combined weighted sum is strictly greater (which WADD prefers). This creates a sharp quantitative divergence in predictions. We also include agreement trials (where the top cue and the sum of remaining cues align) to measure baseline performance and estimate noise parameters.

[2] To quantitatively dissociate Tallying (Equal Weights) from WADD (Weighted Additive), we use a 5-feature environment with a steep drop-off in validities. We design 'opposition' trials where one option has fewer positive features but those features are of very high validity, while the other option has more positive features of lower validity. Tallying will strictly prefer the option with more positive features, whereas WADD will prefer the option with the higher weighted sum. We also include 'indifference' trials where Tallying predicts a tie but WADD has a strong preference, and 'agreement' trials where both models prefer the same option, to establish baseline performance and noise levels.

[3] To quantitatively dissociate WADD from Tallying, we use a 5-feature environment with a set of validities that allows for strong divergence between weighted sums and simple feature counts. We include 'opposition' trials where one option possesses fewer total positive features but those features have much higher validities (Tallying prefers the option with more features; WADD prefers the option with higher validities). We also include 'indifference' trials where both options have the same number of positive features but different weighted sums (Tallying predicts a tie; WADD predicts a strict preference). Finally, agreement trials are included to assess baseline performance and estimate noise parameters.

[4] To quantitatively dissociate pure Tallying from a Mixture of Heuristics (Tallying + Take-The-Best), we design trials where the two strategies diverge. Tallying relies solely on the unweighted sum of positive features, while Take-The-Best (TTB) relies exclusively on the single highest-validity discriminating feature. By creating 'opposition' trials where one option has more positive features but lacks the highest-validity feature, Tallying will strongly prefer the feature-rich option, whereas the Mixture model will show a dampened preference (or even a reversal) due to the TTB component pulling in the opposite direction. We also include 'indifference' trials where both options have an equal number of positive features; here pure Tallying predicts a 50/50 split, but the Mixture model predicts a systematic preference for the option possessing the highest-validity feature.

[5] To quantitatively dissociate the Mixture of Heuristics (Tallying + Take-The-Best) from pure Tallying, we exploit trials where the two strategies make divergent predictions. Pure Tallying relies exclusively on the unweighted sum of positive features, whereas the Mixture model is influenced by the single highest-validity discriminating feature (the TTB component). We use 'indifference' trials where both options have an equal number of positive features but differ on the highest-validity cue; pure Tallying predicts a 50/50 split, but the Mixture model predicts a systematic preference. We also use 'opposition' trials where one option has more positive features (favored by Tallying) but the other option has the highest-validity feature (favored by TTB); here, pure Tallying strongly prefers the feature-rich option, while the Mixture model shows a dampened preference or reversal.

[6] To quantitatively dissociate pure Tallying from a Mixture of Heuristics (Tallying + Take-The-Best), we create specific 'indifference' and 'opposition' trials. On indifference trials, both options have the same number of positive features, but one option possesses the highest-validity feature. Pure Tallying predicts a strict 50/50 choice, whereas the Mixture model predicts a bias towards the option with the highest-validity feature. On opposition trials, one option has more positive features but lacks the highest-validity feature; pure Tallying predicts a strong preference for the feature-rich option, while the Mixture model predicts a weakened preference due to the conflicting TTB component. Agreement trials are included to assess baseline performance and noise.

[7] To quantitatively dissociate the Mixture of Heuristics (Tallying + Take-The-Best) from pure Tallying, we use a 5-feature environment with a distinct validity hierarchy. We design 'indifference' trials where both options have the same number of positive features, but Option A possesses the highest-validity feature (or the highest discriminating feature). Pure Tallying predicts a 50/50 choice on these trials, whereas the Mixture model predicts a systematic preference for Option A due to the TTB component. We also design 'opposition' trials where Option B has more positive features (favored by Tallying) but Option A has the highest-validity feature (favored by TTB). Pure Tallying will strongly prefer Option B, while the Mixture model will show a dampened preference or even a reversal. Agreement trials are included to establish baseline performance.

[8] To quantitatively dissociate Tallying from Cancellation and Focus, we exploit how they process shared features. Tallying computes the unweighted sum of positive features for each option and applies a softmax rule based on the difference in sums. Therefore, Tallying predicts identical choice probabilities for any trial pairs that have the same difference in total positive features, regardless of how many features the options share. Cancellation and Focus, however, first eliminates shared features and then applies a ratio rule to the remaining unique positive features. We design trials where the difference in total positive features is held constant (e.g., Option A has exactly 1 more positive feature than Option B), but the number of shared features varies. On trials where Option B's only positive features are shared with Option A, Cancellation predicts a deterministic choice for A (ratio = 1.0), whereas Tallying predicts a soft preference. On trials where the options share no features, Cancellation applies its ratio rule to a mixture of unique features, leading to a much weaker preference for A, while Tallying's prediction remains unchanged.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Tallying from Cancellation and Focus, we manipulate the number of shared features while keeping the difference in total positive features constant. Tallying computes the unweighted sum of positive features and applies a softmax rule based on the difference in these sums, meaning it predicts the same choice probabilities for any trial pairs with the same difference in total features. Cancellation and Focus, however, eliminates shared features and applies a ratio rule to the remaining unique positive features. We design trials where Option A always has exactly 1 more positive feature than Option B. In some trials (e.g., A has 4, B has 3, with 3 shared), B has no unique positive features, so Cancellation predicts a deterministic choice for A (1 vs 0 unique), while Tallying predicts a soft preference. In other trials (e.g., A has 3, B has 2, with 0 shared), Cancellation applies a ratio rule to a mixture of unique features (3 vs 2), predicting a much weaker preference for A, whereas Tallying's prediction remains identical to the first case.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      0,
      0
    ],
    [
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
      0
    ],
    [
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
      0
    ],
    [
      0,
      1,
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
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0
    ],
    [
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3530,
  "candidates_token_count": 505,
  "total_token_count": 4867
}
```
