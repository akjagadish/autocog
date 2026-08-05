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
**Description:** Take The Best (TTB) with Extreme Noise Theory: People use a non-compensatory 'one-reason' decision heuristic, comparing options by consulting cues one at a time in order of descending validity. The first cue that discriminates between the options determines the choice, and lower-validity cues are ignored. However, subjects experience extreme decision conflict or distraction, leading to a very high lapse rate (guessing) that pulls choice probabilities almost entirely toward chance (0.50), masking the underlying deterministic TTB process in aggregate data.

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind='stable').tolist()
    
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
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax over the binary TTB scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with high uniform lapse (guessing)
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Tallying (Equal Weights) Theory with High Noise: People often ignore the varying validities or importances of different pieces of information, instead simply counting the number of positive features (or cues) favoring each option. However, when faced with difficult compensatory tradeoffs where one option has the best cue but the other has many weaker cues, subjects may experience high decision conflict and rely heavily on guessing. Thus, the choice process is characterized by a high degree of noise, pulling choice probabilities very close to chance (0.50).

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.2, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Convert stimulus to a numpy array of shape (2, n_features)
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying rule: sum the unweighted positive features for each option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallied scores with max-subtraction for stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse (guessing) distribution
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically pits Take The Best (TTB) against Weighted Additive (WADD) by creating 'compensatory' environments where the most valid discriminating cue points to option A, but a coalition of less valid cues points to option B. TTB relies solely on the highest-validity discriminating cue and will consistently choose A in these critical trials. In contrast, WADD integrates all features weighted by their validities, allowing the sheer number of lower-validity cues favoring B to outweigh the single high-validity cue favoring A. We also include non-compensatory trials where both models agree, ensuring that differences in model fit are driven by the specific cue-conflict structure rather than mere noise or random responding.

[1] This design systematically contrasts Weighted Additive (WADD) and Take The Best (TTB) by varying the distribution of less valid cues when the single most valid cue favors one option. In TTB, only the top discriminating cue dictates the choice, so the subject will consistently choose the option with the highest validity advantage. WADD, on the other hand, integrates all cues according to their validity weights. By creating 'compensatory' trials where the option lacking the highest-validity cue is supported by multiple lower-validity cues, we can produce stark opposite predictions between the two models. Non-compensatory trials are included as controls to ensure baseline adherence to cue validities.

[2] This design systematically pits Tallying with High Noise against Weighted Additive (WADD) by using 5 features with a steep drop-off in validities. In the critical trial (A: [1,1,0,0,0], B: [0,0,1,1,1]), Tallying predicts a preference for B because it has more positive features (3 vs 2), whereas WADD predicts a preference for A because the two high-validity features outweigh the three low-validity features. Furthermore, Tallying with High Noise predicts that across all trials, choices will be heavily regressed toward 0.50 (guessing) due to high decision conflict and noise, while WADD predicts strong, deterministic choices (probabilities near 1.0 or 0.0) whenever there is a clear difference in the weighted sums.

[3] This design uses 6 features to create a quantitative dissociation between Weighted Additive (WADD) and Tallying with High Noise, avoiding the previously explored 5-feature steep drop-off design. By employing three high-validity cues and three low-validity cues, we can construct trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strong, deterministic preference for the option with high-validity features. Furthermore, we include critical dissociation trials where Tallying strictly prefers the option with more positive features (e.g., 4 vs 3), but WADD prefers the option with fewer, higher-validity features. Under the competing Tallying with High Noise theory, choices will be weakly distributed around 0.50 across all trials, whereas WADD predicts strong preferences that scale with the weighted sum differences.

[4] This design pits Tallying against Take The Best (TTB) by systematically varying the number of positive features versus the presence of the highest-validity discriminating feature. In these trials, the option with the single highest-validity discriminating cue (favored by TTB) always has fewer total positive cues than the alternative (favored by Tallying). Because the advocated theory (Tallying with High Noise) predicts choices driven by the unweighted sum of features, it will predict preferences for the option with more positive cues, albeit with small margins due to high noise. In contrast, the competing theory (TTB with Extreme Noise) predicts preferences in the exact opposite direction based strictly on the highest-validity cue, though also heavily regressed toward chance.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design systematically dissociates Take The Best (TTB) with Extreme Noise from Tallying with High Noise using 6 features. By constructing trials where the single highest-validity discriminating cue points to one option while the sheer number of positive cues points to the other, we force the two theories to predict preferences in opposite directions. Because both theories predict high noise (choices near 0.50), the key dissociation lies in the *direction* of the slight deviations from chance. TTB will consistently lean toward the option with the highest-validity discriminating cue, whereas Tallying will lean toward the option with the higher unweighted sum of positive features. We also include a trial where Tallying predicts a perfect tie (equal number of positive features) but TTB predicts a strict preference, further distinguishing the underlying mechanisms.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55,
    0.51
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
      0,
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
      0
    ],
    [
      0,
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
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
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
      1
    ],
    [
      1,
      0,
      1,
      1,
      1,
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
      1
    ],
    [
      1,
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
  "prompt_token_count": 2914,
  "candidates_token_count": 495,
  "total_token_count": 4805
}
```
