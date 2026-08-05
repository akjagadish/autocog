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
**Description:** People employ a probabilistic 'Take The Best' (TTB) heuristic to make binary choices. They order features by their subjective validity, which is informed by objective validities but subject to random fluctuations (noise). They compare the options sequentially based on this subjective ordering. The first feature that discriminates between the two options determines the choice, and all subsequent features are ignored. This introduces stochasticity into the cue hierarchy, explaining why choices sometimes deviate from strict reliance on the single highest-validity cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- sigma: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    
    # Add Gaussian noise to validities to model subjective fluctuations in cue hierarchy
    noisy_validities = validities + np.random.normal(0, sigma, size=validities.shape)
    
    # Order features by noisy validity in descending order
    order = np.argsort(noisy_validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Iterate through features in order of validity
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    # If no features discriminate, both options are tied
    if scores[0] == 0.0 and scores[1] == 0.0:
        scores = np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
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
**Description:** Random Choice / Minimal Effort: In the absence of trial-by-trial feedback and when faced with complex multi-attribute binary arrays, participants largely abandon systematic cue-based strategies. Instead, they resort to minimal effort processing, which manifests as random guessing or behavior dominated by an extremely high lapse rate. Any residual systematicity is extremely weak, resulting in choice probabilities that are consistently very close to 0.5 across all experimental conditions.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.9, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Minimal effort evaluation (e.g., simple tallying of 1s)
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Dominated by a extremely high lapse rate (epsilon near 1.0)
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design aims to dissociate Tallying (which counts the number of features favoring each option, ignoring validities and magnitudes) from Weighted Additive (WADD) (which computes the weighted sum of feature values). By using five features with a steep validity drop-off, we can create pairs where a few high-validity features outweigh several lower-validity features (WADD preference) versus pairs where the sheer number of lower-validity features dictates the choice (Tallying preference). For instance, in Trial 1, Option A has the two best features (WADD wins) while Option B has the three worst features (Tallying wins).

[1] This design quantitatively dissociates Weighted Additive (WADD) from Tallying using 5 features. The validities are chosen such that the sum of the top two validities (0.9 + 0.8 = 1.7) is slightly greater than the sum of the bottom three validities (0.6 + 0.55 + 0.5 = 1.65). This enables critical trials where Option A has the top two features and Option B has the bottom three: WADD prefers Option A (higher weighted sum), whereas Tallying prefers Option B (higher count of winning features, 3 vs 2). Additional trials include cases where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers one option due to validity differences, as well as baseline trials where both models agree.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) while distinguishing from previously explored designs, this experiment uses a 6-feature environment with a distinct cluster of high validities and a cluster of low validities. By making the top three validities very close (0.90, 0.88, 0.86), TTB's subjective cue hierarchy (which includes noise) will frequently swap their order, whereas WADD integrates them stably. We design trials where the highest-validity feature points to one option, but a combination of the slightly-lower high-validity features and multiple low-validity features points to the other. This setup not only tests compensatory vs. non-compensatory choice but also probes the stochastic nature of TTB's cue hierarchy.

[3] This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic by pitting a single highly valid feature against multiple moderately valid features. We use five features where the highest validity is 0.95 and the others range from 0.80 to 0.65. In critical trials, Option B excels on the single most predictive feature, while Option A is supported by a combination of the remaining features. Because the sum of the lower validities far exceeds the highest validity, WADD strongly prefers Option A. In contrast, TTB (even with validity noise) will primarily rely on the highest-validity feature and thus prefer Option B. By varying which subset of lower-validity features Option A possesses, we can map out the precise compensatory threshold predicted by WADD but absent in TTB.

[4] To quantitatively dissociate Take The Best (TTB) from Take The First (Left-to-Right Lexicographic), we arrange the feature validities in ascending order from left to right. The leftmost feature has the lowest validity, while the rightmost feature has the highest validity. Take The First strictly evaluates features from left to right, thus heavily relying on the lowest-validity features if they discriminate. In contrast, Take The Best reorders features by their subjective validities (with noise) and will predominantly evaluate the rightmost, highest-validity feature first. By pitting the leftmost features against the rightmost features, the two models make completely opposite predictions.

[5] To quantitatively dissociate Take The First (Left-to-Right Lexicographic) from Take The Best (TTB), we place the highest validity features in the middle of the array, with lower validities on the left and right edges. Take The First strictly evaluates features from left to right, meaning it will base its decisions on the leftmost features (indices 0 and 1) whenever they discriminate, regardless of their low validity. Take The Best, however, sorts features by their subjective validities and will jump straight to the middle features (indices 2 and 3). By designing trials where the leftmost features favor one option and the middle features favor the other, we can cleanly separate the two models. This avoids the confound of a strictly ascending validity array where left-to-right could be mistaken for reverse-validity search.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate Take The Best (TTB) from the Random Choice / Minimal Effort model, we employ a 5-feature design with a steep validity hierarchy (0.95 to 0.55). The critical trials pit the single most valid feature against all remaining features. TTB predicts a strong preference for the option with the highest-validity feature, as it sequentially evaluates and stops at the first discriminating cue. In contrast, the Random Choice model predicts choices that are either perfectly random (due to a near 1.0 lapse rate) or slightly leaning toward the option with more positive features (due to minimal effort tallying). The stark difference in predicted choice probabilities (strong systematic preference vs. ~0.5) will clearly adjudicate between the models.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
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
      0,
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
  "prompt_token_count": 3195,
  "candidates_token_count": 440,
  "total_token_count": 4660
}
```
