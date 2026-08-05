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
**Description:** Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

**Parameters:**
- beta: [0.01, 20.0]
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
**Description:** Mixture of Heuristics: Subjects probabilistically switch between a compensatory, equal-weighting strategy (Tallying) and a frugal, lexicographic strategy (Take-The-Best). By mixing these two heuristics with a prior bias towards Tallying, the model captures the dominant tendency to count features while allowing for occasional validity-maximizing behavior, avoiding the unrealistic assumption of full compensatory validity weighting (WADD).

**Parameters:**
- beta_tally: [0.01, 20.0]
- beta_ttb: [0.01, 20.0]
- p_tally: [0.4, 0.85]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Tallying prediction
    scores_tally = np.sum(stim, axis=1)
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Take-The-Best prediction
    val = np.asarray(parameters["validities"], dtype=float)
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
        p_ttb_core = np.ones(2) / 2.0
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # Mixture
    p_tally_weight = float(parameters["p_tally"])
    p_combined = p_tally_weight * p_tally_core + (1.0 - p_tally_weight) * p_ttb_core
    
    # Lapse
    epsilon = float(parameters["epsilon"])
    n_opts = p_combined.shape[0]
    
    return (1.0 - epsilon) * p_combined + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD), we must create trials where the single most valid discriminating feature opposes the compensatory sum of all other valid features. By using 5 features with a descending spread of validities, we can design 'compensatory' trials where one option wins on the highest-validity cue but loses on the sum of several lower-validity cues. TTB will consistently choose the option with the highest-validity cue, ignoring the rest, whereas WADD will integrate across all cues, leading it to choose the option with the greater overall weighted sum. We also include agreement trials to establish baseline performance.

[1] To cleanly dissociate WADD from TTB, we employ a 5-feature environment with linearly spaced validities. We design 'compensatory' trials where one option excels on the single most valid cue (which TTB relies upon exclusively), while the other option excels on multiple lower-validity cues such that their combined weighted sum is strictly greater (which WADD prefers). This creates a sharp quantitative divergence in predictions. We also include agreement trials (where the top cue and the sum of remaining cues align) to measure baseline performance and estimate noise parameters.

[2] To quantitatively dissociate Tallying (Equal Weights) from WADD (Weighted Additive), we use a 5-feature environment with a steep drop-off in validities. We design 'opposition' trials where one option has fewer positive features but those features are of very high validity, while the other option has more positive features of lower validity. Tallying will strictly prefer the option with more positive features, whereas WADD will prefer the option with the higher weighted sum. We also include 'indifference' trials where Tallying predicts a tie but WADD has a strong preference, and 'agreement' trials where both models prefer the same option, to establish baseline performance and noise levels.

[3] To quantitatively dissociate WADD from Tallying, we use a 5-feature environment with a set of validities that allows for strong divergence between weighted sums and simple feature counts. We include 'opposition' trials where one option possesses fewer total positive features but those features have much higher validities (Tallying prefers the option with more features; WADD prefers the option with higher validities). We also include 'indifference' trials where both options have the same number of positive features but different weighted sums (Tallying predicts a tie; WADD predicts a strict preference). Finally, agreement trials are included to assess baseline performance and estimate noise parameters.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate pure Tallying from a Mixture of Heuristics (Tallying + Take-The-Best), we design trials where the two strategies diverge. Tallying relies solely on the unweighted sum of positive features, while Take-The-Best (TTB) relies exclusively on the single highest-validity discriminating feature. By creating 'opposition' trials where one option has more positive features but lacks the highest-validity feature, Tallying will strongly prefer the feature-rich option, whereas the Mixture model will show a dampened preference (or even a reversal) due to the TTB component pulling in the opposite direction. We also include 'indifference' trials where both options have an equal number of positive features; here pure Tallying predicts a 50/50 split, but the Mixture model predicts a systematic preference for the option possessing the highest-validity feature.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55
  ],
  "trial_a_ratings": [
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
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
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
      0,
      0,
      1
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
      1,
      0
    ],
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2903,
  "candidates_token_count": 468,
  "total_token_count": 4096
}
```
