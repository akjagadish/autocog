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
**Description:** People integrate information from multiple cues by computing a weighted sum of the feature values, where the weights are a non-linear transformation of the cues' validities. This Weighted Additive (WADD) strategy is compensatory but incorporates a scaling parameter (gamma) that can exponentiate the validities. This allows the decision-maker to dynamically re-balance attention—either steepening the weights to behave more like Take The Best, or flattening them to behave more like Tallying—to match the empirical balance of conflict resolution.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 5.0]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Weighted Additive (WADD) heuristic with exponentiated validities.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Exponentiate validities to allow non-linear scaling of weights
    weights = val ** gamma
    
    # Compute the weighted sum of features for each option
    # using the scaled cue validities as weights.
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Heuristic Mixture (Strategy Selection) Theory: Decision-makers do not rely on a single strategy; instead, they probabilistically select between different heuristics on each trial. Specifically, they mix between a non-compensatory, lexicographic strategy (Take-The-Best) and a non-weighted, compensatory strategy (Tallying). A mixture parameter governs the probability of using TTB versus Tallying, allowing the model to capture intermediate choice behaviors by blending one-reason decision making with simple feature-counting.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.25]
- p_mix: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take-The-Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        ttb_scores = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        ttb_scores = np.array([0.0, 1.0])
    else:
        ttb_scores = np.array([0.0, 0.0])
        
    beta = float(parameters["beta"])
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Strategy Mixture
    p_mix = float(parameters["p_mix"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_mix * p_ttb + (1.0 - p_mix) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the highest-validity cue often favors one option while a greater number of lower-validity cues favor the other. TTB will strictly follow the single highest-validity cue that discriminates between the options, ignoring the rest. Tallying, however, will simply count the total number of winning features for each option, completely ignoring cue validities. By creating trials where the option with the highest validity cue has fewer total winning cues than the alternative, we force the two models to make opposing predictions. We also include trials where Tallying predicts a tie (equal number of winning cues) while TTB makes a deterministic prediction based on the most valid cue.

[1] To strongly dissociate Tallying from Take The Best (TTB), we use a 5-feature design with a randomized validity order. We construct specific trial types: (1) Tallying favors one option because it wins on more features, while TTB favors the other because it wins on the single highest-validity feature; (2) Tallying predicts a tie (equal number of winning features) while TTB makes a deterministic choice based on the most valid cue; (3) Extreme cases where Tallying strongly favors one option (winning 4 out of 5 features) while TTB favors the other based entirely on the 1 remaining high-validity feature. These conditions ensure maximum quantitative divergence between the models.

[2] To quantitatively dissociate WADD (with an exponentiated validity parameter) from Tallying, we use a 5-feature design with a broad spread of validities. Tallying strictly counts the number of features favoring each option, treating all features equally and predicting a tie when the counts are equal. WADD, in contrast, computes a weighted sum where the weights are a non-linear transformation of the validities. We construct trials where: (1) Tallying favors one option by a large margin (e.g., 3 or 4 winning features to 1) but WADD favors the other due to the non-linear scaling of a single high-validity feature; (2) Tallying predicts a perfect tie (2 wins vs 2 wins) while WADD strongly prefers the option with the highest validity cue; and (3) Tallying favors one option slightly but WADD flips its preference depending on the gamma parameter, allowing us to identify the specific degree of non-linear scaling used by the subject.

[3] To quantitatively dissociate Tallying (which strictly counts the number of winning features) from WADD (which computes a weighted sum with exponentiated validities), we use a 5-feature design with a specific spread of validities. Tallying ignores validity magnitudes entirely, predicting choices solely based on the number of features favoring each option, and predicting equal preference (a tie) when the counts are equal. WADD, depending on its gamma parameter, can scale validities non-linearly, behaving more like Take-The-Best (high gamma) or Equal Weight (low gamma). We include trials where Tallying predicts a tie but WADD has a strong preference, trials where Tallying favors one option (e.g., 3 wins vs 2 wins) but WADD favors the other due to high-validity features, and extreme cases where a single highest-validity feature is pitted against all other lower-validity features.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate WADD from a Heuristic Mixture of Take-The-Best (TTB) and Tallying, we exploit the non-linear compensatory nature of WADD. The Heuristic Mixture model can only produce choice probabilities that are linear interpolations between TTB (which strictly chooses based on the highest validity cue) and Tallying (which strictly counts the number of winning cues). We construct 'Tie-Breaker' trials where Tallying predicts a tie (equal number of winning cues for both options) and TTB strongly favors Option B. Consequently, the Mixture model MUST favor Option B (or predict a tie if p_mix = 0). However, WADD with a low gamma parameter (which flattens validities) will actually favor Option A, because the sum of multiple moderately-high validities for Option A exceeds the sum of the highest and lowest validities for Option B. This creates a stark qualitative and quantitative divergence where WADD predicts a preference reversal that the Mixture model mathematically cannot capture.",
  "validities": [
    0.95,
    0.85,
    0.8,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
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
      1,
      0
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
      1,
      1,
      1,
      1
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
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      1,
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
      1
    ],
    [
      1,
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
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3289,
  "candidates_token_count": 561,
  "total_token_count": 6513
}
```
