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
**Description:** Take-The-Best (TTB) heuristic: People employ a non-compensatory, lexicographic decision process. They evaluate features sequentially in descending order of their subjective validity. The very first feature that discriminates between the two options (i.e., one option has a positive feature value while the other does not) determines the choice, and all lower-validity features are strictly ignored. If no feature discriminates, they guess. Response noise is modeled via a softmax over the resulting binary preference and an independent random lapse rate.

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
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Evaluate features sequentially
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to allow for noise
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Epsilon-greedy lapse
    n_opts = p_core.shape[0]
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
**Description:** Two-Stage Take-The-Best and WADD Heuristic: Decision makers employ a boundedly rational, two-stage process. In the first stage, they act strictly non-compensatory by checking only the single most valid cue (like Take-The-Best). If this primary cue discriminates between the options, the decision is made immediately. However, if the most valid cue is tied, they fall back to a compensatory process, computing a Weighted Additive (WADD) score of the remaining cues to break the tie. This decouples the primary cue's overriding influence from the secondary cues' collective weighting.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Two-stage expects a (2, n_features) stimulus.")

    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    best_cue = order[0]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    def apply_noise(scores):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    
    # Stage 1: Check the single most valid cue
    if stim[0, best_cue] != stim[1, best_cue]:
        scores = np.zeros(2)
        if stim[0, best_cue] > stim[1, best_cue]:
            scores[0] = 1.0
        else:
            scores[1] = 1.0
        return apply_noise(scores)
    
    # Stage 2: Fallback to WADD of the remaining cues if tied
    remaining_cues = order[1:]
    if len(remaining_cues) > 0:
        scores = np.sum(stim[:, remaining_cues] * validities[remaining_cues], axis=1)
    else:
        scores = np.zeros(2)
        
    return apply_noise(scores)
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
[0] The design contrasts Tallying (which simply counts the number of features favoring each option) against Weighted Additive (WADD) (which weights each feature by its validity). By selecting a set of 5 features with two highly valid experts and three lower-validity experts, we can create trials where one option wins on a larger number of features (favored by Tallying) but the other option wins on the fewer, but more highly-weighted features (favored by WADD). The inclusion of agreement trials and Tallying-tie trials provides baseline checks and full dissociation.

[1] To quantitatively dissociate WADD from Tallying, we use a 5-feature design with a steep drop-off in validities (two highly valid experts, three barely valid experts). This allows us to construct 'compensatory failure' trials where one option is favored by a larger number of low-validity features (winning the Tallying count) but the other option is favored by fewer, but higher-validity features (winning the WADD score). We also include trials where Tallying predicts a tie (equal number of winning features) but WADD strongly predicts one option due to the validity differences, as well as agreement trials to ensure basic task engagement.

[2] This design quantitatively dissociates Take-The-Best (TTB) from the Weighted Additive (WADD) model. We use a 5-feature design with a spread of validities. The core of the dissociation lies in 'compensatory' trials where the single most valid discriminating feature favors one option (dictating the TTB choice), but a coalition of multiple lower-validity features favors the other option (dictating the WADD choice). By varying which feature is the most valid discriminator and how many lower-validity features oppose it, we can firmly separate lexicographic single-reason decision making from compensatory weighted summation.

[3] To quantitatively dissociate Weighted Additive (WADD) from Take-The-Best (TTB), we employ a 5-feature design with linearly decreasing validities. The core dissociation relies on 'compensatory' trials where the single most valid discriminating feature favors one option (dictating the TTB choice), but a coalition of multiple lower-validity features favors the other option (dictating the WADD choice). We vary which feature is the most valid discriminator by tying the higher-validity features on some trials, ensuring that TTB's lexicographic stopping rule is tested at multiple points in the hierarchy, while WADD consistently integrates all available information.

[4] To quantitatively dissociate Take-The-Best (TTB) from Tallying, this design uses 5 features with descending validities. TTB decides based solely on the first discriminating feature (highest validity), whereas Tallying counts the total number of positive features, ignoring validities entirely. The core dissociation trials pit the most valid cue against a numerical majority of lower-validity cues (e.g., Option A has only the most valid feature, while Option B has three lower-validity features). We also include trials where the highest validities are tied, forcing TTB to look further down the hierarchy while Tallying continues to count overall totals. Finally, trials where Tallying predicts a tie (equal number of positive features) but TTB predicts a clear winner provide further discriminatory power.

[5] To quantitatively dissociate Tallying from Take-The-Best (TTB), we employ a 5-feature design with descending validities. Tallying completely ignores validities, choosing the option with the most positive features, while TTB ignores the number of features, choosing the option favored by the single most valid discriminating feature. The core dissociation trials pit the most valid cue against a numerical majority of lower-validity cues (e.g., Option A has only the most valid feature, while Option B has three lower-validity features). We also include trials where the highest validities are tied, forcing TTB to look further down the hierarchy while Tallying continues to count overall totals. Finally, trials where Tallying predicts a tie (equal number of positive features) but TTB predicts a clear winner provide further discriminatory power.

[6] This design quantitatively dissociates Take-The-Best (TTB) from Weighted Additive (WADD) using a 5-feature task with a gentle linear decrease in validities. TTB decides solely based on the first discriminating feature, ignoring all others. WADD, however, integrates all features weighted by their validity. The core dissociation trials pit the single most valid cue against a coalition of two or three slightly less valid cues. Because the validity drop-off is gentle, the sum of the weights of the lower-validity cues easily outweighs the single highest-validity cue for WADD, whereas TTB inflexibly follows the single best cue. Agreement trials are also included to measure baseline accuracy and noise.

[7] This design quantitatively dissociates the Weighted Additive (WADD) theory from the Take-The-Best (TTB) heuristic by pitting the single most valid feature against a coalition of less valid features. The validities are chosen such that the highest validity feature is strong, but can be mathematically outweighed by the sum of the remaining features across all WADD weighting schemes (log-odds, raw validity, and validity-minus-half). TTB, being strictly non-compensatory, will invariably choose the option favored by the single most valid discriminating feature, whereas WADD will integrate the information and choose the option favored by the coalition of lower-validity features.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Take-The-Best (TTB) from the Two-Stage TTB+WADD heuristic, we must examine trials where the single most valid cue is tied. When the most valid cue discriminates, both models make identical predictions. However, when the most valid cue is tied, TTB proceeds lexicographically to the second most valid cue, ignoring all lower validities. In contrast, the Two-Stage model abandons the lexicographic rule entirely and computes a Weighted Additive (WADD) score of *all* remaining cues. The critical dissociation trials tie the most valid cue, give the second most valid cue to one option (dictating the TTB choice), and give a coalition of lower-validity cues to the other option (dictating the Two-Stage WADD choice).",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.6
  ],
  "trial_a_ratings": [
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
      1,
      1,
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
      1,
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
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      1,
      1,
      1
    ],
    [
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
      0
    ],
    [
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      1
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
  "prompt_token_count": 3463,
  "candidates_token_count": 449,
  "total_token_count": 5513
}
```
