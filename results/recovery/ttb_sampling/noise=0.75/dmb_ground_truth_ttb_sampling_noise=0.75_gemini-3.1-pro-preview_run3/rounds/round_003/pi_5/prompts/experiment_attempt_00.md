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
**Description:** Threshold-based Sequential Search integrates the fast-and-frugal nature of Take-The-Best with the compensatory evidence accumulation of Weighted Additive (WADD) models. Decision-makers search through cues in descending order of their validity, maintaining a running tally of the evidence (weighted by each cue's validity). Instead of stopping at the very first discriminating cue, search terminates only when the absolute accumulated evidence exceeds an internal confidence threshold. If all cues are exhausted without reaching this threshold, the option with the higher accumulated evidence is chosen. This allows for fast, one-reason decisions when a highly valid cue strongly favors one option, while enabling compensatory behavior when early cues provide weak or conflicting evidence.

**Parameters:**
- threshold: [0.0, 1.25]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    threshold = float(parameters["threshold"])
    
    # Sequential search with evidence accumulation
    for idx in order:
        diff = a[idx] - b[idx]
        evidence += diff * validities[idx]
        
        # Stop search if the confidence threshold is met or exceeded
        if abs(evidence) >= threshold:
            break
            
    # Convert accumulated evidence into discrete choice scores
    scores = np.zeros(2)
    if evidence > 0:
        scores[0] = 1.0
    elif evidence < 0:
        scores[1] = 1.0
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend in uniform lapse
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
**Description:** Strategy Mixture (Take-The-Best and Weighted Additive)

**Parameters:**
- beta_ttb: [0.1, 5.0]
- beta_wadd: [0.1, 10.0]
- mixture_ttb: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    # --- TTB (Take-The-Best) Process ---
    order = np.argsort(validities)[::-1]
    scores_ttb = np.zeros(2)
    for idx in order:
        if a[idx] > b[idx]:
            scores_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores_ttb[1] = 1.0
            break
            
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- WADD (Weighted Additive) Process ---
    w = np.asarray(parameters["weights"], dtype=float)
    scores_wadd = stim @ (validities * w)
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Mixture ---
    mix = float(parameters["mixture_ttb"])
    p_mix = mix * p_ttb + (1.0 - mix) * p_wadd
    
    # --- Lapse ---
    epsilon = float(parameters["epsilon"])
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
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
[0] To quantitatively dissociate Tallying from WADD, we use a 5-feature design with a skewed distribution of validities (two very high, three low). Tallying simply counts the number of features favoring each option, treating all features equally. WADD computes a weighted sum using the validities. By pitting an option (A) that wins on a small number of high-validity features against an option (B) that wins on a larger number of low-validity features, we create direct conflicts: Tallying prefers B due to the higher count of winning features, while WADD prefers A due to the higher cumulative validity of its fewer winning features. We also include pairs where the two heuristics agree or where Tallying predicts a tie but WADD predicts a strict preference, allowing us to estimate the noise parameters independently of the structural dissociation.

[1] To quantitatively dissociate WADD from Tallying, we use a 5-feature design with a steep drop-off in validities (two high, three low). Tallying counts the number of winning features irrespective of their validity, while WADD weights each feature by its validity. By pitting an option that wins on two high-validity features against one that wins on three low-validity features, we create a direct contradiction: Tallying prefers the option with three winning features, whereas WADD prefers the option with two high-validity features. We also include trials where Tallying predicts a tie but WADD predicts a strict preference, allowing for a precise estimation of choice noise and a robust comparison of the two models.

[2] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), we utilize a 5-feature design with a linearly decreasing set of validities. TTB processes features lexicographically, stopping at the most valid feature that discriminates between the two options. WADD, in contrast, integrates all features by computing a weighted sum. The key to dissociating them lies in 'compensatory' trials where the single most valid discriminating feature favors one option, but a coalition of less valid features favors the other option such that their combined weight exceeds the highest valid feature. We include trials where TTB and WADD strongly disagree (e.g., Option A wins on the highest validity cue only, while Option B wins on all remaining cues), trials where they agree (to anchor the noise parameters), and trials with ties on the top cue(s) to test TTB's reliance on secondary cues versus WADD's full-profile integration.

[3] To quantitatively dissociate the compensatory Weighted Additive (WADD) rule from the non-compensatory Take The Best (TTB) heuristic, we use a 5-feature design where the sum of lower-validity cues can outweigh a single higher-validity cue. The trials are designed such that the most valid discriminating cue often favors Option A (leading TTB to choose A), while a coalition of less valid cues favors Option B (leading WADD to choose B). We also include agreement trials to calibrate choice noise independently of the structural differences.

[4] To quantitatively dissociate the pure Take-The-Best (TTB) theory from a Strategy Mixture (TTB + WADD) theory, we use a 5-feature design with a relatively flat but strictly descending validity profile. TTB dictates that choices are driven entirely by the single most valid discriminating cue. The Mixture model allows for choices to be influenced by the Weighted Additive (WADD) value. By pitting an option that wins on the highest validity cue against an option that wins on multiple lower-validity cues, we create strong 'compensatory' conflicts. A pure TTB decision-maker will consistently choose the option favored by the top cue, whereas a Mixture decision-maker will exhibit choice probabilities that are softened or reversed by the WADD component. We include various degrees of conflict (e.g., top cue vs. 4 lower cues, top cue vs. 2 lower cues) and agreement trials to accurately estimate noise and mixture parameters.

[5] To quantitatively dissociate the Strategy Mixture (TTB + WADD) theory from pure Take-The-Best (TTB), we use a 5-feature design with a linearly descending validity profile. Pure TTB relies solely on the most valid discriminating cue, treating all subsequent cues as irrelevant. The Mixture model, however, integrates a Weighted Additive (WADD) component, meaning the sum of lower-validity cues can pull choice probabilities away from the TTB prediction. By creating 'compensatory conflict' trials where the highest valid cue favors Option A but a coalition of lower valid cues heavily favors Option B, pure TTB predicts a strong preference for A (modulated only by general noise), while the Mixture model predicts a softened preference for A or even a preference for B. We vary the strength of the WADD counter-evidence to provide a fine-grained test of the mixture component.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Threshold-based Sequential Search (TSS) theory from a Strategy Mixture (TTB + WADD), we exploit TSS's evidence accumulation mechanism which allows for early termination based on the sequence of cue agreements, rather than a fixed mixture of pure TTB and pure WADD. We use a 5-feature design with linearly descending validities. We construct a critical set of 'compensatory' trials where TTB always favors Option A and WADD always favors Option B. For a Mixture model, the choice probability for A should be roughly constant across these trials (dictated by the mixture weight). However, TSS will choose differently depending on how the evidence accumulates: if the top cues agree (e.g., A wins on cues 1 and 2 but loses on 3, 4, 5), TSS quickly hits its threshold and chooses A (acting like TTB). If the top cues disagree (e.g., A wins on cue 1, B wins on cue 2), the accumulated evidence drops, forcing TSS to evaluate further cues, ultimately choosing B (acting like WADD). This creates a stark, qualitative reversal in choice predictions that a simple probabilistic mixture of TTB and WADD cannot reproduce.",
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
    ],
    [
      0,
      0,
      0,
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
    ],
    [
      1,
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
  "prompt_token_count": 3497,
  "candidates_token_count": 599,
  "total_token_count": 5887
}
```
