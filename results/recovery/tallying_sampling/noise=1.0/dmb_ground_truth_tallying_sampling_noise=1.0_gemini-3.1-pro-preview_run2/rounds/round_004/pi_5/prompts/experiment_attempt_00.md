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
**Description:** Weighted Additive (WADD) with Extreme Noise Theory: Subjects attempt to integrate all available cues by weighting them according to their provided validities. However, the cognitive demand of integrating multiple conflicting fictitious validities is overwhelming. This results in extreme decision conflict and distraction, leading to a near-total reliance on guessing (a very high lapse rate). Consequently, choice probabilities are pulled almost entirely toward chance (0.50), masking the underlying compensatory process in the aggregate behavioral data.

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.95, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Weighted sum per option (dot product with per-feature weights)
    scores = stim @ (validities * w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores with max-subtraction for stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with extreme uniform lapse (guessing)
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
**Description:** Pure Random Guessing (Complete Cognitive Overload): Subjects completely ignore the cues and validities due to the high cognitive demand and artificial nature of the fictitious task. This results in choices that are strictly driven by a 50/50 binomial coin flip, without any underlying compensatory or non-compensatory signal.

**Parameters:**
- guessing_rate: {0.5}

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    p = float(parameters['guessing_rate'])
    return np.array([p, 1.0 - p])
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
[0] This design systematically pits Take The Best (TTB) against Weighted Additive (WADD) by creating 'compensatory' environments where the most valid discriminating cue points to option A, but a coalition of less valid cues points to option B. TTB relies solely on the highest-validity discriminating cue and will consistently choose A in these critical trials. In contrast, WADD integrates all features weighted by their validities, allowing the sheer number of lower-validity cues favoring B to outweigh the single high-validity cue favoring A. We also include non-compensatory trials where both models agree, ensuring that differences in model fit are driven by the specific cue-conflict structure rather than mere noise or random responding.

[1] This design systematically contrasts Weighted Additive (WADD) and Take The Best (TTB) by varying the distribution of less valid cues when the single most valid cue favors one option. In TTB, only the top discriminating cue dictates the choice, so the subject will consistently choose the option with the highest validity advantage. WADD, on the other hand, integrates all cues according to their validity weights. By creating 'compensatory' trials where the option lacking the highest-validity cue is supported by multiple lower-validity cues, we can produce stark opposite predictions between the two models. Non-compensatory trials are included as controls to ensure baseline adherence to cue validities.

[2] This design systematically pits Tallying with High Noise against Weighted Additive (WADD) by using 5 features with a steep drop-off in validities. In the critical trial (A: [1,1,0,0,0], B: [0,0,1,1,1]), Tallying predicts a preference for B because it has more positive features (3 vs 2), whereas WADD predicts a preference for A because the two high-validity features outweigh the three low-validity features. Furthermore, Tallying with High Noise predicts that across all trials, choices will be heavily regressed toward 0.50 (guessing) due to high decision conflict and noise, while WADD predicts strong, deterministic choices (probabilities near 1.0 or 0.0) whenever there is a clear difference in the weighted sums.

[3] This design uses 6 features to create a quantitative dissociation between Weighted Additive (WADD) and Tallying with High Noise, avoiding the previously explored 5-feature steep drop-off design. By employing three high-validity cues and three low-validity cues, we can construct trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strong, deterministic preference for the option with high-validity features. Furthermore, we include critical dissociation trials where Tallying strictly prefers the option with more positive features (e.g., 4 vs 3), but WADD prefers the option with fewer, higher-validity features. Under the competing Tallying with High Noise theory, choices will be weakly distributed around 0.50 across all trials, whereas WADD predicts strong preferences that scale with the weighted sum differences.

[4] This design pits Tallying against Take The Best (TTB) by systematically varying the number of positive features versus the presence of the highest-validity discriminating feature. In these trials, the option with the single highest-validity discriminating cue (favored by TTB) always has fewer total positive cues than the alternative (favored by Tallying). Because the advocated theory (Tallying with High Noise) predicts choices driven by the unweighted sum of features, it will predict preferences for the option with more positive cues, albeit with small margins due to high noise. In contrast, the competing theory (TTB with Extreme Noise) predicts preferences in the exact opposite direction based strictly on the highest-validity cue, though also heavily regressed toward chance.

[5] This design systematically dissociates Take The Best (TTB) with Extreme Noise from Tallying with High Noise using 6 features. By constructing trials where the single highest-validity discriminating cue points to one option while the sheer number of positive cues points to the other, we force the two theories to predict preferences in opposite directions. Because both theories predict high noise (choices near 0.50), the key dissociation lies in the *direction* of the slight deviations from chance. TTB will consistently lean toward the option with the highest-validity discriminating cue, whereas Tallying will lean toward the option with the higher unweighted sum of positive features. We also include a trial where Tallying predicts a perfect tie (equal number of positive features) but TTB predicts a strict preference, further distinguishing the underlying mechanisms.

[6] This design systematically pits Weighted Additive (WADD) against Take The Best (TTB) under conditions of extreme noise. Both the advocated WADD and competing TTB theories assume high lapse rates (guessing) that pull choice probabilities toward 0.50. However, the models diverge on the direction of the slight deviations from chance. By creating 'compensatory' trials where the single highest-validity discriminating cue points to one option, but the weighted sum of several lower-validity cues points to the other, we force the two theories to predict opposite preferences. TTB will always lean slightly toward the option favored by the single best cue, whereas WADD will lean slightly toward the option with the higher total weighted value.

[7] This design tests a fundamental quantitative dissociation between Take The Best (TTB) and Weighted Additive (WADD) under extreme noise. TTB predicts that the choice probability (deviation from 0.50) is invariant to *which* cue discriminates the options, because the winning option always receives a deterministic score of 1.0 vs 0.0 before the softmax and lapse are applied. In contrast, WADD's pre-softmax difference is proportional to the weighted sum difference. By including compensatory trials (where a higher cue opposes multiple lower cues), WADD is forced to assign exponentially decaying weights to lower cues to mimic TTB's choices. Consequently, on 'pure discrimination' trials that tie on all top cues and only discriminate on the lowest cues, WADD must predict choice probabilities that are virtually indistinguishable from 0.50. TTB, however, predicts the exact same choice probability for these low-cue discrimination trials as it does for the high-cue trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate Weighted Additive (WADD) with Extreme Noise from Pure Random Guessing, we must look for any systematic deviation from 0.50 that correlates with the weighted sum of features. Pure Random Guessing predicts exactly 50/50 choices across all trials, regardless of the stimulus. WADD with Extreme Noise predicts that choices will be heavily regressed toward 0.50, but will still retain a small, monotonic signal favoring the option with the higher weighted sum. By including trials with extreme differences (e.g., all positive vs. all negative cues) as well as moderate and reversed differences, we can test whether the small deviations from chance are systematic (supporting WADD) or non-existent (supporting Pure Guessing).",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
    [
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
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1
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
      1,
      1,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3094,
  "candidates_token_count": 480,
  "total_token_count": 4494
}
```
