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
**Description:** People use a non-compensatory, lexicographic heuristic called 'Take The Best' (TTB) to choose between options. They search through features in descending order of their subjective validity and stop at the first feature that discriminates between the two options (i.e., one option has a positive rating and the other does not). The choice is based entirely on this single discriminating cue, completely ignoring all remaining features, regardless of how many lower-validity cues might favor the alternative.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    chosen = -1
    for idx in order:
        if a[idx] > b[idx]:
            chosen = 0
            break
        elif b[idx] > a[idx]:
            chosen = 1
            break
            
    # Deterministic choice based on the first discriminating cue
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # If all features tie, guess randomly
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse rate for noise
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Softmax-Weighted Additive Model: Subjects use a fully compensatory integration process where they calculate a weighted sum of all features, but they subjectively scale the stated cue validities using a softmax transformation. By exponentiating and normalizing the validities with a scaling parameter, differences between high and low validity cues are amplified without shrinking the overall magnitudes of the weights. This allows a purely compensatory, additive mechanism to rival and approximate strict non-compensatory behavior (like Take-The-Best) when the scaling parameter is sufficiently large, while still integrating all available information.

**Parameters:**
- gamma: [0.0, 1000.0]
- beta: [1.0, 50.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Model expects a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax transformation of validities to amplify importance of top cues
    # without shrinking the overall magnitudes of the weights to zero.
    z_v = gamma * validities
    z_v = z_v - np.max(z_v)
    e_v = np.exp(z_v)
    weights = e_v / np.sum(e_v)
    
    # Compensatory integration (Weighted Additive)
    scores = stim @ weights

    # Softmax decision rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Blended with a uniform lapse rate
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
[0] This design contrasts Tallying (which simply counts the number of features favoring each option, ignoring validities) with WADD (which computes a validity-weighted sum). The validities are deliberately highly skewed ([0.95, 0.85, 0.60, 0.55, 0.50]). In critical trials (e.g., A=[0,0,1,1,1] vs B=[1,1,0,0,0]), Tallying strongly favors A because it wins on 3 out of 5 features. In contrast, WADD strongly favors B because the two features it wins on have much higher validities (0.95 + 0.85 = 1.80) than the three features A wins on (0.60 + 0.55 + 0.50 = 1.65). The design also includes pairs where both models agree, and pairs where Tallying predicts a tie but WADD has a clear preference, providing a rich set of quantitative dissociations.

[1] This design systematically dissociates the Weighted Additive rule (WADD) and the Tallying heuristic using a 6-feature setup with a wide spread of validities. It includes trials where the models make opposite predictions (e.g., Tallying favors the option with more positive features, while WADD favors the option with fewer but more valid features), as well as trials where one model predicts a tie while the other predicts a strong preference. This enables a precise quantitative evaluation of whether subjects rely on feature validities (WADD) or simply count the number of winning features (Tallying).

[2] This design aims to quantitatively dissociate Tallying from the Weighted Additive (WADD) rule. Validities are chosen to be highly dispersed, distinguishing the feature-counting nature of Tallying from the validity-weighting nature of WADD. Crucial trials pit an option with fewer, but highly valid, features against an option with more, but weakly valid, features. This produces strong divergent predictions (e.g., Tallying prefers A while WADD prefers B). Additionally, trials where Tallying predicts a tie but WADD predicts a preference are included, alongside agreement trials, covering the full spectrum of possible heuristic dynamics.

[3] This design quantitatively dissociates WADD from Tallying by systematically varying the distribution of features. Validities are spread across a wide range ([0.95, 0.85, 0.75, 0.60, 0.55, 0.50]). Critical trials pit an option with many low-validity features against an option with fewer high-validity features, leading to opposite choices. Additionally, trials where both models agree and trials where Tallying predicts a tie but WADD predicts a strong preference are included to fully map the decision space.

[4] This design specifically targets the dissociation between Take The Best (TTB) and Weighted Additive (WADD) models by creating choice environments where a single highly valid cue points to one option, but a constellation of slightly less valid cues points to the other. Validities are set close together ([0.90, 0.85, 0.80, 0.75, 0.70]) so that WADD will frequently allow the combination of multiple lower-ranked cues to outweight the single highest-validity cue. TTB, on the other hand, strictly follows a lexicographic stopping rule, choosing based only on the first discriminating cue and completely ignoring the quantity of opposing cues. The trial set includes extreme dissociations (where one option has only the best cue and the other has all remaining cues), subtle dissociations (where the compensatory sum just barely edges out the top cue), and agreement trials to ensure baseline performance checks.

[5] This design systematically dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic. Validities are chosen such that the highest validity cue strongly favors one option, but the sum of the remaining lower-validity cues can easily outweigh it. Critical trials pit an option that wins on the single most valid cue against an option that wins on multiple lower-validity cues, leading TTB to choose the former and WADD to choose the latter. The design also includes trials where the highest validity cue is tied, forcing TTB to look at the second-best cue, which is then outweighed by the remaining cues under WADD. Agreement trials are included as a baseline.

[6] This design specifically targets the dissociation between standard Take-The-Best (TTB) with a uniform lapse rate and Probabilistic Take-The-Best (PTTB) with validity noise. The advocated theory (TTB) relies solely on the ordinal ranking of validities: it will always check the highest-ranked cue first, regardless of how close it is to the second-best cue. The competing theory (PTTB) adds Gaussian noise to the validities before sorting, meaning that cues with similar validities will frequently swap ranks in the subject's mind. By using a set of validities with one large gap (e.g., 0.95 vs 0.85) and several small gaps (e.g., 0.85, 0.83, 0.81), we can create trials where the two models make identical predictions (when the discriminating cue is separated by a large gap) and trials where they diverge significantly (when the discriminating cue is only marginally better than the opposing cues, leading PTTB to frequently sample the opposing cues first).

[7] This design specifically targets the dissociation between standard Take-The-Best (TTB) and Probabilistic Take-The-Best (PTTB) by manipulating the validity distance between the best discriminating cue and the next best cues. Standard TTB predicts a constant choice probability (1 - epsilon) for the option favored by the highest-validity discriminating cue, regardless of how close the next-best cue's validity is. In contrast, PTTB adds trial-by-trial noise to validities before sorting, meaning rank inversions are highly likely for cues with similar validities but exceedingly rare for cues with disparate validities. By pairing validities that are tightly clustered (e.g., 0.92, 0.89, 0.86) with those that are distant (0.65, 0.55), we create 'small gap' trials where PTTB predicts a much softer preference than TTB, and 'large gap' trials where PTTB and TTB both predict strong preferences.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To mimic TTB's strict non-compensatory choices on top-tier cues (e.g., cue 1 vs cues 2-5), Softmax-WADD must use a very large scaling parameter (gamma) to ensure the top weight outweighs the sum of all lower weights. However, doing so exponentially squashes the weights of lower-tier cues to near-zero. Consequently, when choices depend exclusively on these lower-tier cues (e.g., cue 4 vs cue 5), Softmax-WADD predicts near-random guessing because the score differences become vanishingly small and cannot be salvaged even by the maximum inverse-temperature (beta) parameter. TTB, in contrast, applies the same lexicographic rule uniformly, predicting strong preferences regardless of whether the discrimination happens at the top or bottom of the validity hierarchy.",
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
      0,
      0,
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
      1,
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
      1,
      1,
      0,
      1,
      0
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
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
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
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3622,
  "candidates_token_count": 523,
  "total_token_count": 11761
}
```
