# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Heuristic Decision Making domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Choose `rating_max` (>= 1) to set the upper bound of each rating value: rating_max=1 gives binary ratings (but note that Tallying ≡ Equal-Weight on binary features), while rating_max >= 2 gives cardinal ratings that dissociate magnitude-based heuristics (EW, WADD) from sign-based ones (Tallying). Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, rating_max]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities, rating_max, and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see a set of ratings for each product. Each rating comes from an expert and is an integer between 0 and an upper bound fixed for this experiment (higher = more positive). The experts differ in how well they typically predict product quality, and their accuracies are communicated before the experiment begins; they are always shown in the same order. On every trial, decide which product you believe is of higher quality and press F for product A or J for product B. There is no time limit, but please respond as accurately as you can. There is no feedback during the task: try to make the best inference you can from the ratings and the experts' accuracies. Press SPACE when you are ready to begin.

## ADVOCATED THEORY
**Description:** Soft Take-The-Best with Power-Law Decay: Decision-makers evaluate options by ranking features according to their subjective validity and integrating evidence across all cues. However, instead of the importance of cues decaying exponentially with their rank, it decays according to a power law. This 'fatter tail' enables a strong accumulation of evidence on multiple lower-ranked cues to compensate for a loss on top-ranked cues, capturing compensatory behavior in scenarios where many weak cues oppose a few strong ones.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- alpha: [0.0, 3.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Soft TTB expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(v)[::-1]
    
    alpha = float(parameters["alpha"])
    # Power-law decay based on rank (rank 1, 2, 3...)
    weights = 1.0 / ((np.arange(len(v)) + 1.0) ** alpha)
    
    # Binary wins on ordered features
    a_wins = (a[order] > b[order]).astype(float)
    b_wins = (b[order] > a[order]).astype(float)
    
    score_a = np.sum(a_wins * weights)
    score_b = np.sum(b_wins * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Weighted Tallying heuristic: People evaluate options by comparing them feature-by-feature, but instead of just counting the number of winning features (as in regular Tallying), they weight each win by the feature's subjective validity or importance. This heuristic discards cardinal magnitudes (the size of the difference between feature values is ignored), making it robust to extreme outlier values that would skew an additive model. However, unlike unweighted Tallying, it incorporates the known validities of the cues, allowing more important features to break ties or even override a larger count of less important features.

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
        raise ValueError(f"Weighted Tallying expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate strict wins for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Weight the wins by the validities
    score_a = np.sum(a_wins * v)
    score_b = np.sum(b_wins * v)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
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
[0] To dissociate Tallying from WADD, we use cardinal features (rating_max=5) and vary both the number of winning features and the magnitude of those wins. Tallying ignores both feature validities and the magnitude of differences, counting only the number of features where one option strictly exceeds the other. WADD computes a validity-weighted sum of the cardinal ratings. The trials are designed such that in some cases, one option wins on a majority of features by a small margin (favored by Tallying), while the other option wins on fewer, high-validity features by a large margin (favored by WADD).

[1] To rigorously dissociate WADD from Tallying, this design uses four cardinal features (max rating 10) and a steep validity gradient. Tallying ignores both magnitude and validities, relying solely on the number of winning features. The trial pairs are constructed to pit these mechanisms against each other: in some trials, one option wins on a single high-validity feature by a large magnitude (dominating the WADD score) while the other option wins on all remaining features by small margins (dominating the Tallying score). We also include trials that produce a perfect tie for one model but a strong preference for the other, isolating the specific computational mechanisms (weighted sum vs. unweighted count) driving choice.

[2] To dissociate Tallying from Equal-Weight (EQW), this design uses cardinal features (rating_max = 10). Tallying computes preferences based strictly on the number of winning features (ignoring the magnitude of the difference), whereas EQW sums the raw feature values across all attributes. We construct trials where one option wins on a single feature by a massive margin (dominating the EQW sum) while the other option wins on multiple features by small margins (dominating the Tallying count). We also include trials that produce a perfect tie for one heuristic but a strong preference for the other.

[3] To cleanly dissociate Equal-Weight (EQW) from Tallying, we exploit their core difference: EQW sums feature magnitudes, whereas Tallying counts the number of strictly winning features regardless of magnitude. We use cardinal features (rating_max = 10) to create scenarios where one option wins on a single feature by a massive margin (dominating the EQW sum) while the other option wins on multiple features by small margins (dominating the Tallying count). We also include trials that produce a perfect tie for one heuristic but a strong preference for the other, isolating the specific computational mechanisms (summing magnitudes vs. counting signs) driving choice.

[4] To dissociate unweighted Tallying from Weighted Tallying, we use 5 features with a steep validity gradient. Tallying simply counts the number of winning features, ignoring validities. Weighted Tallying weights each win by its validity. We include trials where one option wins on a majority of low-validity features (favored by Tallying) while the other option wins on fewer high-validity features (favored by Weighted Tallying). We also include trials that produce a perfect tie in Tallying (equal number of winning features) but a strong preference in Weighted Tallying due to validity differences.

[5] To dissociate Weighted Tallying from unweighted Tallying, we use binary features (rating_max=1) with a set of validities that clearly distinguish feature counts from validity-weighted counts. Tallying simply counts the number of winning features, ignoring validities. Weighted Tallying weights each win by its validity. We include trials where one option wins on a larger number of low-validity features (favored by Tallying) while the other option wins on fewer, high-validity features (favored by Weighted Tallying). We also include trials that produce a perfect tie in Tallying (equal number of winning features) but a strong preference in Weighted Tallying due to validity differences.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- rating_max: Upper bound (inclusive) of each rating value. Ratings are integers in [0, rating_max]. Choose rating_max=1 for binary ratings (note: Tallying and Equal-Weight are behaviorally identical on binary features — if you want to dissociate magnitude-based from sign-based heuristics, choose rating_max >= 2 for cardinal ratings).
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is an integer in [0, rating_max]. `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate Soft Take-The-Best (with Power-Law Decay) from Weighted Tallying, we exploit how they assign importance to features. Both models ignore the magnitude of feature differences and focus purely on 'wins'. However, Weighted Tallying weights each win linearly by the raw validity of the feature, whereas Soft TTB weights each win by a power-law transformation of the feature's rank (e.g., 1, 1/2^a, 1/3^a). By clustering the top validities very close to each other (e.g., 0.95, 0.90, 0.85, 0.80), Weighted Tallying treats them as nearly equal, meaning a combination of the 2nd and 3rd most valid features easily outweighs the 1st. In contrast, Soft TTB enforces a strict rank-based drop-off regardless of the raw validity values, meaning the 1st feature can still dominate the 2nd and 3rd combined. The trials pit a single high-rank win against multiple slightly lower-rank wins to strongly dissociate the predictions of the two models.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.8,
    0.5
  ],
  "rating_max": 1,
  "trial_a_ratings": [
    [
      1,
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
      0
    ],
    [
      1,
      1,
      0,
      0,
      0
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
      0,
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
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3221,
  "candidates_token_count": 537,
  "total_token_count": 6112
}
```
