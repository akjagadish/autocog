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
**Description:** Strategy Mixture Theory with Decoupled Determinism: Decision-makers probabilistically choose between Tallying and Take-The-Best on any given trial. Because Tallying operates on raw integer count differences (which can be large) while TTB operates on binary 1/0 differences, the two heuristics have different intrinsic scales. To account for this, the model uses independent inverse temperatures (beta_tally and beta_ttb) for each strategy. This allows the decision process to independently calibrate the determinism of each heuristic, avoiding artificial scale distortions when balancing the two strategies via the mixture weight.

**Parameters:**
- beta_tally: [0.1, 20.0]
- beta_ttb: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_tally_weight: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    p_tally_weight = float(parameters["p_tally_weight"])
    
    # --- Tallying Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # --- Take-The-Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Mixture & Noise ---
    p_mix = p_tally_weight * p_tally + (1.0 - p_tally_weight) * p_ttb
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Weighted Additive (WADD) Theory with Non-linear Subjective Weights: People evaluate options by considering all available features, weighting each by a subjective transformation of its validity. Rather than using raw validities, people subjectively distort them via an exponent (gamma). A gamma > 1 disproportionately emphasizes higher-validity cues (shifting behavior towards Take The Best), whereas a gamma < 1 flattens the weights (shifting behavior towards Tallying). The subjective value of an option is the weighted sum of its features, and choices are made probabilistically via a softmax over these values, with a lapse rate.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 6.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Weighted additive sum for each option
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design strictly pits Take The Best (TTB) against Tallying by creating a complete dissociation between the two models. In every trial, one option is favored by the highest-validity discriminating cue (which TTB relies on exclusively), while the other option possesses a greater total number of positive cues (which Tallying simply counts). Consequently, TTB and Tallying will predict exactly opposite choices for every single trial. This allows for a robust quantitative evaluation of whether subjects rely on a single dominant cue or sum across all available cues.

[1] This design systematically varies the Tallying difference (number of features favoring A minus number of features favoring B) while holding the Take The Best (TTB) prediction constant. By including trials where Tallying strongly opposes TTB, weakly opposes TTB, ties, weakly agrees, and strongly agrees, we can test whether subjects' choice probabilities scale with the Tallying count difference (as predicted by Tallying with a softmax choice rule) or remain a flat step-function determined solely by the highest-validity cue (as predicted by TTB).

[2] This design pits WADD against Tallying by manipulating the distribution of feature validities. Tallying ignores validities and simply counts the number of winning features for each option, predicting a tie when the counts are equal and a strict preference when one option has more winning features. WADD, however, weights each feature by a subjective transformation of its validity. We include 'Tallying Tie' trials where Tallying predicts random guessing (50/50), but WADD strongly predicts the option with higher-validity features. We also include 'Conflict' trials where one option wins on more features (favored by Tallying) but the other option wins on fewer, higher-validity features (favored by WADD, depending on the gamma parameter). This creates a quantitative dissociation where WADD predicts a graded probability of choice based on the weighted sum, whereas Tallying predicts either a step-function preference based on counts or a complete tie.

[3] This design isolates the effect of cue validities to cleanly dissociate Tallying (which ignores validities and only counts feature wins) from WADD (which weights features by a subjective transformation of their validities). We use four features with a wide spread of validities. The trials include 'Tallying Ties' where each option wins on the same number of features but differs in validity, forcing Tallying to predict random guessing while WADD predicts a strong preference. We also include 'Conflict' trials where one option wins on more features (favored by Tallying) while the other option wins on fewer but higher-validity features (favored by WADD, especially for higher values of gamma). This provides a rigorous quantitative test of whether subjects completely ignore validities as Tallying predicts.

[4] To decisively dissociate WADD from the Tallying/TTB Strategy Mixture, we introduce 'Dominance Violation' trials. In these critical trials, Option A possesses the single highest-validity cue (winning TTB) AND a greater total number of winning cues (winning Tallying). Because both strategies in the Mixture model agree, the Mixture model strictly predicts Option A. However, WADD with a non-linear validity transformation (gamma > 1) will predict Option B if Option B's fewer cues are collectively strong enough (e.g., the second and third highest validities) to outweigh Option A's combination of one top cue and several weak cues. This creates a qualitative divergence that no mixture of Tallying and TTB can explain. We also include standard conflict trials to calibrate the mixture weights and gamma.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design tests for a qualitative signature of the Strategy Mixture Theory that WADD cannot capture: the decoupling of determinism between TTB and Tallying. By varying the feature count difference (e.g., Option A having 1 vs 4 features, or 2 vs 3 features) while holding the TTB winner constant, WADD strictly predicts a monotonic change in choice probabilities because the weighted sum difference must change. In contrast, the Strategy Mixture Theory can predict perfectly flat (plateaued) choice probabilities across these trials if the Tallying inverse temperature (beta_tally) is high, because Tallying becomes deterministic as long as the count difference is non-zero, regardless of its magnitude. This allows the mixture model to predict identical choice proportions for pairs with drastically different feature-sum differences.",
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
      1,
      1,
      1,
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
      1,
      0,
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
      1,
      1
    ],
    [
      0,
      0,
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
  "prompt_token_count": 3292,
  "candidates_token_count": 585,
  "total_token_count": 5684
}
```
