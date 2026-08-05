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
**Description:** Weighted Additive Strategy (WADD): Decision makers use a fully compensatory strategy, integrating all available information by computing a weighted sum of feature values for each option, where each feature is weighted by its explicit validity. The option with the highest overall value is chosen.

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
        raise ValueError("WADD expects a (2, n_features) state.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: Dot product of feature vectors and validities
    scores = stim @ validities
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Tallying (Equal Weight Heuristic): People compare two options by ignoring the explicit validities of the cues and simply counting the number of positive features (or cues present) for each option. The option with the higher count (tally) is chosen. This is a strongly compensatory strategy that avoids the cognitive overhead of multiplying features by validities, yet allows multiple lower-validity cues to jointly overrule a single high-validity cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) state.")
        
    # Tallying: simple sum of positive cues per option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tally scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] To cleanly dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit the compensatory nature of WADD versus the non-compensatory, lexicographic nature of TTB. The experiment uses 5 features with descending validities. In the critical trials, one option possesses the single best discriminating cue (which dictates TTB's choice), while the competing option possesses multiple lower-validity cues whose combined weight exceeds the single best cue (which dictates WADD's choice). By including trials where the 'best' cue is offset by 2, 3, or 4 lower-validity cues, we can robustly measure the degree of compensation subjects exhibit. Control trials where both models agree are also included to ensure basic task engagement.

[1] To cleanly dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic, this design uses 4 features with non-descending validities. By scrambling the order of validities (e.g., placing the most valid cue second), we ensure subjects must actively process the validity numbers rather than relying on left-to-right reading order. The trials include critical compensatory cases where the option possessing the single best discriminating cue is pitted against an option possessing multiple lower-validity cues whose combined weight exceeds the best cue. In these cases, TTB and WADD make opposite predictions. We also include trials where the best cue is tied, forcing TTB to look at the second-best cue, which is then outweighed by the remaining cues for WADD, as well as control trials where both models agree.

[2] This design aims to cleanly dissociate Tallying (Equal Weight) from the Weighted Additive (WADD) model. We use 5 features with a steep drop-off in validities (two highly valid cues, three moderately valid cues). Critical trials pit an option with fewer, but highly valid, features against an option with more, but weakly valid, features. In these cases, Tallying predicts the subject will choose the option with the higher count of positive features, whereas WADD predicts the subject will choose the option with the higher validity-weighted sum. We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference, and agreement trials to ensure basic task engagement.

[3] This design specifically dissociates the compensatory Weighted Additive (WADD) model from the Tallying (Equal Weight) heuristic. By utilizing a set of 5 features with a steep drop-off in validities (two highly valid cues, three less valid cues), we create scenarios where the two models make conflicting predictions. Critical trials pit an option with fewer, but highly valid, features against an option with more, but weakly valid, features. In these cases, Tallying predicts the subject will choose the option with the higher count of positive features, whereas WADD predicts the subject will choose the option with the higher validity-weighted sum. We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference based on validity differences, as well as agreement trials to ensure basic task engagement.

[4] While Tallying and MCD produce identical choice probabilities under standard softmax (because the difference in their scores is mathematically equivalent for binary features), this design systematically varies the number of shared features (both shared '1's and shared '0's) across trials while holding the relative difference constant. By doing so, we can test for deviations from the standard softmax assumption—such as differences in reaction time, confidence, or decision noise—that would indicate whether subjects are computing holistic sums (which scale with shared '1's) or performing dimension-wise cancellations (which ignore shared features).

[5] MCD and Tallying yield mathematically identical choice probabilities under a standard softmax choice rule for binary features, because the difference in their scores (MCD's pairwise cancellations vs. Tallying's independent sums) is always equal. To empirically dissociate them, we must look beyond standard choice probabilities to secondary measures such as reaction times, confidence, or decision noise, which are often influenced by the absolute magnitude of the scores. By expanding the feature space to 8 features, we can systematically vary the number of shared positive features (shared '1's) and shared negative features (shared '0's) across trials while holding the absolute difference in scores constant. If subjects use Tallying, cognitive load and perceived value might scale with the total number of '1's (e.g., comparing 7 vs 5 feels different than 3 vs 1). If subjects use MCD, shared features are immediately canceled out, making trials with many shared '1's cognitively identical to trials with many shared '0's.

[6] To effectively dissociate Tallying (Equal Weight) from the Weighted Additive Strategy (WADD), we utilize an environment with 5 features characterized by a steep drop-off in validities. We pit options with fewer, but highly valid, features against options with more, but weakly valid, features. In these critical trials, Tallying dictates that the decision-maker should choose the option with the higher absolute count of positive features, while WADD predicts a preference for the option with the higher validity-weighted sum. We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference, to further highlight WADD's sensitivity to explicit validities.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To cleanly dissociate the Weighted Additive Strategy (WADD) from Tallying (Equal Weight), this design uses 5 features with a steep drop-off in validities. We construct critical trials that pit options with fewer, but highly valid, features against options with more, but weakly valid, features. In these cases, WADD predicts a preference for the option with the higher validity-weighted sum, whereas Tallying predicts a preference for the option with the higher absolute count of positive features. We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference based on validity differences, as well as agreement trials to ensure basic task engagement.",
  "validities": [
    0.9,
    0.8,
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
    ],
    [
      1,
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
      0,
      1,
      1,
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
      0,
      0,
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
      1,
      0,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
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
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
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
      0,
      0,
      1,
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
  "prompt_token_count": 3047,
  "candidates_token_count": 694,
  "total_token_count": 6818
}
```
