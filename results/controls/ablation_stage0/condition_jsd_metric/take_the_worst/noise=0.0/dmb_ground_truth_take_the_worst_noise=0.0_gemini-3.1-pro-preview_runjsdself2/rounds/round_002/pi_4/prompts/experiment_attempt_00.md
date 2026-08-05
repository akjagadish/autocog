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
**Description:** Probabilistic-Stopping Take-The-Best (TTB)

**Parameters:**
- alpha: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Descending validity; argsort is stable so validity ties break toward earlier feature index
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    p_A = 0.0
    p_B = 0.0
    p_continue = 1.0
    
    for j in cue_order:
        if a[j] > b[j]:
            p_A += p_continue * alpha
            p_continue *= (1.0 - alpha)
        elif b[j] > a[j]:
            p_B += p_continue * alpha
            p_continue *= (1.0 - alpha)
            
    # If search exhausts all cues or fails to stop, guess uniformly
    p_A += p_continue * 0.5
    p_B += p_continue * 0.5
    
    p_core = np.array([p_A, p_B])
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Weighted Additive (WADD) theory posits that decision-makers integrate all available cues by weighting them proportionally to their validities. The overall value for each option is the validity-weighted sum of its features. Choices are made probabilistically via a softmax over these aggregated values, incorporating response noise through an inverse temperature parameter and an independent lapse rate.

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
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
    
    # Calculate validity-weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) and Tallying, we use 4 features with descending validities. We design trial pairs where the highest-validity discriminating cue favors one option (which TTB will choose), but the sheer number of winning cues favors the other option or results in a tie (which Tallying will choose or guess uniformly). For example, if Option A wins only on the most valid cue and Option B wins on the remaining three cues, TTB strongly predicts A while Tallying strongly predicts B. Other trials contrast TTB's definitive choice with Tallying's exact tie.

[1] To quantitatively dissociate Tallying from Take The Best (TTB), we use 5 features with strictly descending validities. TTB relies solely on the highest-validity discriminating cue, ignoring all others. Tallying, by contrast, weights all cues equally and simply counts the number of winning features for each option. The trials are designed such that the most valid discriminating cue favors one option (which TTB will categorically choose), while the sheer number of winning cues favors the other option (which Tallying will choose). Additionally, we include trials where Tallying predicts a perfect tie (guessing) but TTB makes a strong deterministic prediction based on a single high-validity cue.

[2] To quantitatively dissociate WADD and Tallying, we use 5 features with a steep drop-off in validities. WADD integrates all cues weighted by their validity, allowing a small number of highly valid cues to outweigh a larger number of less valid cues. Tallying, on the other hand, ignores validities and simply counts the number of winning features for each option. We design trials where Option A wins on fewer but highly valid features (strongly favored by WADD), while Option B wins on more but less valid features (favored by Tallying). We also include trials where Tallying predicts a perfect tie (because both options win on the same number of features), whereas WADD strongly predicts one option due to the disparity in cue validities.

[3] To quantitatively dissociate Tallying (advocated theory) from Weighted Additive (WADD) (competing theory), we use a 5-feature design with a significant disparity in validities. Tallying completely ignores cue validities and simply counts the number of winning features for each option, whereas WADD integrates all features weighted by their respective validities. We design trials where one option wins on fewer but highly valid features (strongly favored by WADD), while the other option wins on a greater number of less valid features (favored by Tallying). Additionally, we include trials where Tallying predicts a perfect tie (because both options win on the exact same number of features), but WADD makes a deterministic prediction due to the differing validities of those features.

[4] To quantitatively dissociate Weighted Additive (WADD) theory from Probabilistic-Stopping Take-The-Best (TTB) theory, we use 5 features with a prominent highest-validity cue and a set of lower-validity cues whose combined weight can exceed the highest one. TTB makes decisions based primarily on the highest-validity discriminating cue and stops search probabilistically, often ignoring the remaining cues. WADD integrates all cues weighted by their validities. We design 'compensatory' trials where Option A possesses the single most valid cue (strongly favored by TTB) but Option B possesses multiple lower-validity cues whose sum outweighs Option A's single cue (strongly favored by WADD). We also include trials where both theories agree, to serve as a baseline for noise and overall performance.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Probabilistic-Stopping Take-The-Best (TTB) from Weighted Additive (WADD) theory, we use 5 features with linearly decreasing validities. TTB evaluates cues in descending order of validity and stops probabilistically upon finding a discriminating cue, meaning it generally favors the option with the highest-validity discriminating feature. WADD, however, computes a compensatory sum of all features weighted by their validities. We design a set of trials where Option A possesses the single most valid discriminating cue (which TTB will heavily favor), while Option B possesses multiple lower-validity cues whose combined weight strongly exceeds Option A's sum (which WADD will heavily favor). By varying which cue is the highest valid discriminator (e.g., the first vs. the second cue), we can also probe the probabilistic stopping mechanism of TTB, as its preference strength will degrade if it fails to stop at earlier cues.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
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
      1,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2839,
  "candidates_token_count": 469,
  "total_token_count": 4169
}
```
