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
**Description:** Weighted Additive (WADD) Theory with Subjective Validity Scaling: People compare options by computing a comprehensive value for each option, integrating all available cues by weighting each feature. However, instead of using raw objective validities, individuals apply a non-linear subjective scaling to these validities. This scaling allows a single highly valid cue to be disproportionately weighted to balance out multiple less valid cues, gracefully explaining why individuals exhibit near-indifference (choice proportions around 0.5) when faced with adversarial choices where one option excels on the best cue but the other excels on numerous lesser cues. Response noise is captured via a softmax over the options' weighted sums, along with an independent lapse rate.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 10.0]
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
        raise ValueError("Length of validities must match n_features.")
    
    # Apply non-linear subjective scaling to validities
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Compute weighted additive values for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
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
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Rank-Weighted Additive Theory: Individuals integrate all cues but weight them according to an exponential decay based solely on their rank-order of validity. This creates a 'soft' lexicographic rule that acts primarily like Take-The-Best, but allows multiple secondary cues to exert a small, non-zero compensatory pull on the decision. Response variability is captured via a softmax choice rule and a lapse rate.

**Parameters:**
- decay: [0.01, 1.0]
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
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
    
    # Ranks: 0 is highest validity
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(val))
    
    # Exponential decay based on rank
    decay = float(parameters["decay"])
    weights = decay ** ranks
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
[0] This design systematically pits Take The Best (TTB) against Tallying using a 5-feature environment. The validities are strictly descending. In each trial, Option A is designed to win on the highest-validity discriminating cue (leading TTB to choose A), while Option B is designed to win on a greater total number of less valid cues (leading Tallying to choose B). By varying which cue is the highest valid discriminator (e.g., cue 0, cue 1, or cue 2) and the number of cues Option B wins on, we can robustly dissociate the one-reason decision making of TTB from the unweighted cue-counting of Tallying.

[1] This design systematically pits Tallying against Take The Best (TTB) by contrasting the number of winning features with the validity of the single best discriminating feature. Across all trials, one option wins on the single highest-validity discriminating cue (leading TTB to choose it), while the other option wins on a strictly greater number of lower-validity cues (leading Tallying to choose it). By varying which cue is the highest valid discriminator and alternating whether Option A or Option B is favored by each theory, we can firmly dissociate the cue-counting mechanism of Tallying from the one-reason stopping rule of TTB.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) with subjective validity scaling, we vary the number of lower-validity cues that contradict the highest-validity discriminating cue. For TTB, choices are determined exclusively by the first discriminating cue, meaning choice probabilities will be identical across trials that share the same top discriminator, regardless of the number of opposing lower-validity cues. In contrast, WADD integrates all cues, so it predicts a parametric decrease in the choice probability for the option favored by the best cue as the number of opposing lower-validity cues increases.

[3] To quantitatively dissociate WADD from TTB, we hold the top discriminating cue constant while systematically varying the *validity* (rather than just the number) of the opposing cues. TTB bases its decision entirely on the first discriminating cue and ignores the magnitude of any lower-validity cues, predicting identical choice probabilities across these trials. Tallying also fails to distinguish between these trials since the number of opposing cues remains constant. WADD, however, integrates the subjective weights of all cues, predicting a parametric shift in choice probability depending on whether the opposing evidence comes from high-validity or low-validity secondary cues.

[4] This design tests the core assumption of the Dual Process Strategy Mixture (TTB + Tallying) against WADD by holding both the TTB-favored cue and the Tallying count differences constant across matched pairs, while varying the validities of the secondary cues. The Mixture theory predicts identical choice probabilities within these matched pairs because TTB only processes the top discriminating cue (which is identical) and Tallying only processes the number of winning cues (which is identical). In contrast, WADD integrates the subjective validities of all cues, predicting significant differences within the matched pairs because the secondary cues have different validities.

[5] This design isolates a critical qualitative dissociation between the Dual Process Strategy Mixture and WADD with subjective scaling. In key trials, Option A possesses the single highest-validity discriminating cue (causing TTB to favor A) AND a greater total number of winning cues (causing Tallying to favor A). Consequently, the Mixture model is strictly constrained to predict a preference for Option A. However, Option B is supported by multiple highly valid cues that are only marginally inferior to Option A's single top cue. Under WADD with a non-linear subjective validity scaling (e.g., gamma > 1), the combined weight of Option B's multiple strong cues surpasses Option A's single top cue and weak secondary cues, allowing WADD to predict a preference for Option B. This creates a scenario where the two theories predict choices in opposite directions.

[6] This design strictly dissociates Rank-Weighted Additive Theory from WADD with Subjective Validity Scaling by exploiting their fundamentally different weighting mechanisms. Rank-Weighted Additive determines weights strictly by the ordinal rank of validities (e.g., weight = decay^rank), meaning the weight ratio between rank 0 and 1 is identical to the ratio between rank 3 and 4. In contrast, WADD scales the cardinal validity values. By clustering the top validities closely together (e.g., 0.95, 0.94, 0.93) and placing the middle validities much higher than the bottom ones (e.g., 0.85 vs 0.55), we create scenarios where Option A possesses a slightly higher-ranked top cue but a much lower-ranked bottom cue compared to Option B. Rank-Weighted Additive's mathematical structure guarantees it will always prefer Option A (since the rank advantage at the top mathematically outweighs the rank disadvantage at the bottom for any decay < 1). Meanwhile, WADD will strongly prefer Option B because the cardinal penalty of losing the top cue is negligible compared to the massive cardinal gain from the middle cues over the bottom cues, holding true across a very wide range of non-linear gamma scalings.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates Rank-Weighted Additive Theory from WADD by testing the strict exponential decay assumption of the former. By isolating comparisons between adjacent ranks (Rank 0 vs 1, Rank 1 vs 2, Rank 2 vs 3) using validities that have a massive cardinal drop between the first and second cue but small drops thereafter (e.g., 0.95, 0.65, 0.60, 0.55), we expose the rigidity of the Rank-Weighted model. Rank-Weighted Theory dictates that the ratio of choice logit differences between successive adjacent rank comparisons must be exactly equal to the constant decay parameter 'd'. Therefore, it mathematically cannot accommodate a large difference in choice proportions for Rank 0 vs 1 alongside equal choice proportions for Rank 1 vs 2 and Rank 2 vs 3. WADD, relying on cardinal validities, naturally predicts these varying choice logit differences because it scales the actual objective validities.",
  "validities": [
    0.95,
    0.65,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      1,
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
  "prompt_token_count": 3358,
  "candidates_token_count": 503,
  "total_token_count": 6837
}
```
