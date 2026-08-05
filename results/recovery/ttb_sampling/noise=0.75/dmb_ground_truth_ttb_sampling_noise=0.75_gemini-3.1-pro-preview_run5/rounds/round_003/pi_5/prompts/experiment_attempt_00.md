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
**Description:** Sequential Evidence Accumulation. Decision-makers inspect features sequentially in decreasing order of validity. At each step, the validity-weighted difference between the options' features is added to an accumulated evidence tally. If the absolute value of this accumulated evidence exceeds a critical threshold, search stops immediately and a decision is made based on the current tally (mimicking Take-The-Best when the threshold is low). If all features are exhausted without crossing the threshold, a choice is made probabilistically based on the final accumulated tally (mimicking compensatory strategies like Weighted Additive when the threshold is high).

**Parameters:**
- threshold: [0.01, 1.5]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order (stable sort for ties)
    order = np.argsort(-validities, kind='stable')
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    
    # Sequential search and evidence accumulation
    for idx in order:
        diff = a[idx] - b[idx]
        evidence += validities[idx] * diff
        
        # Stop search if accumulated evidence exceeds the threshold
        if abs(evidence) >= threshold:
            break
            
    # Convert the final evidence into choice probabilities.
    # Evidence represents the accumulated advantage of Option A over Option B.
    scores = np.array([evidence, 0.0])
    
    # Softmax over the scores
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
**Description:** Probabilistic Strategy Selection (Mixture of TTB and Tallying)

**Parameters:**
- p_lex: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    p_lex = float(parameters["p_lex"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (Lexicographic)
    order = np.argsort(-validities, kind='stable')
    scores_ttb = np.zeros(2)
    for idx in order:
        if a[idx] > b[idx]:
            scores_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tal = np.array([a_wins, b_wins])
    
    # Softmax for TTB
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # Softmax for Tallying
    z_tal = beta * (scores_tal - scores_tal.max())
    e_tal = np.exp(z_tal)
    p_tal = e_tal / e_tal.sum()
    
    # Mixture of strategies
    p_core = p_lex * p_ttb + (1.0 - p_lex) * p_tal
    
    # Uniform lapse blended into the mixture
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
[0] To dissociate Tallying from Weighted Additive (WADD) models, we use a 5-feature design where the first two features have very high validities and the remaining three have low validities. Tallying simply counts the number of features on which an option is superior, ignoring validity magnitudes. WADD, however, integrates both the feature values and their validities. In the critical dissociation trials (Trials 1 and 2), one option wins on 3 low-validity features while the other wins on 2 high-validity features. Tallying strongly predicts the option with 3 wins, whereas WADD strongly predicts the option with 2 high-validity wins. We also include trials where Tallying predicts a tie (equal number of wins) but WADD has a strict preference, and agreement trials where both heuristics favor the same option, to serve as a baseline for overall choice consistency.

[1] To cleanly dissociate the Weighted Additive (WADD) model from the Tallying heuristic, we use a 5-feature design with a mix of very high and very low validities: [0.95, 0.80, 0.55, 0.50, 0.50]. Tallying ignores validity magnitudes and simply counts the number of features on which an option is superior. WADD integrates both the feature values and their validities. The design includes trials where Tallying predicts a strict preference but WADD predicts the opposite (e.g., one option wins on 3 low-validity features, while the other wins on 2 high-validity features), trials where Tallying predicts a tie but WADD has a strict preference (e.g., both options win on 2 features, but one wins on higher validity features), and agreement trials where both models predict the same choice.

[2] To quantitatively dissociate Take-The-Best (TTB) from the Weighted Additive (WADD) model, we construct a 5-feature design with linearly spaced validities. TTB is non-compensatory, meaning the single highest-validity cue that discriminates between the two options will strictly determine its choice, regardless of the values of all other cues. WADD is compensatory and integrates all cue values weighted by their validities, allowing multiple lower-validity cues to override a deficit on the highest-validity cue. The core dissociation trials pit a single advantage on the most predictive available feature against multiple advantages on less predictive features. For instance, in Trial 1, Option A is endorsed only by the highest-validity expert, while Option B is endorsed by the remaining four. TTB decisively chooses A, while WADD decisively chooses B. Trial 4 checks for the same compensatory logic but restricts the highest valid cue to a tie, making the second-highest cue the decisive one for TTB. Trial 6 acts as an agreement trial where both models prefer Option A, ensuring that subjects are paying attention and not just picking the option with more positive ratings.

[3] To quantitatively dissociate the Weighted Additive (WADD) model from Take-The-Best (TTB), we employ a 5-feature design with linearly descending validities. TTB employs a non-compensatory lexicographic stopping rule, meaning it will always choose the option endorsed by the highest-validity discriminating cue, completely ignoring all other cues. In contrast, WADD is fully compensatory and sums the validity-weighted values of all features. The core dissociation trials pit an option with a single advantage on the most predictive available feature against an option with multiple advantages on less predictive features. For example, when Option A is endorsed only by the highest-validity expert and Option B by all other experts, TTB decisively chooses A, whereas WADD decisively chooses B. We include variations where the highest-validity cue ties (forcing TTB to rely on the second-highest cue while WADD still integrates the rest) and agreement trials to ensure baseline task engagement.

[4] To dissociate the Probabilistic Strategy Selection (Mixture of TTB and Tallying) model from the Weighted Additive (WADD) model, we exploit the fact that the mixture model relies on two specific, distinct sub-strategies: a purely lexicographic one (TTB) and a purely unweighted compensatory one (Tallying). WADD, on the other hand, evaluates options using a single set of continuous, validity-scaled weights, which must behave as a linear, compensatory model. We construct a 5-feature design with descending validities. The trials are designed to create specific conflicts between TTB and Tallying, and to probe choice patterns that a linear model (WADD) struggles to fit simultaneously. For example, some trials pit the single highest-validity cue against multiple lower-validity cues (where TTB and Tallying disagree, causing the mixture model to predict an intermediate choice probability depending on p_lex). By varying the number of opposing low-validity cues across trials, we can observe whether the choice probabilities shift in a step-function manner (as predicted by the mixture model, since Tallying only counts wins) or in a graded manner (as predicted by WADD, which sums the continuous weights).

[5] To strictly dissociate the Weighted Additive (WADD) model from the Probabilistic Strategy Selection (Mixture of TTB and Tallying) model, we exploit the fact that a mixture model's predictions are bounded by the predictions of its constituent strategies. If both TTB and Tallying predict the same option (or if one predicts an option and the other predicts a tie), the mixture model MUST predict that option with a probability of at least 0.5. We design critical trials where Tallying predicts a tie (equal number of wins for both options) and TTB strictly prefers Option A (Option A wins on the single most valid discriminating cue). Consequently, the Mixture model must predict that Option A is chosen more often than Option B. However, by carefully selecting the validities, we ensure that the sum of the validities of Option B's winning cues is significantly greater than the sum for Option A. Thus, WADD strongly predicts Option B, creating a strict qualitative and quantitative dissociation.

[6] To strictly dissociate the Probabilistic Strategy Selection (Mixture) model from the Sequential Evidence Accumulation (Accumulator) model, we use a 7-feature design. The core dissociation exploits the fact that the Mixture model is bounded by its two constituent strategies: TTB and Tallying. In Trial 1, Option A wins on the single highest-validity feature (so TTB chooses A) and wins on three low-validity features, giving it 4 wins total (so Tallying chooses A). Consequently, the Mixture model MUST strongly predict Option A. However, Option B wins on three high-validity features whose sum exceeds the sum of Option A's four features. The Accumulator model, if its threshold is high enough to not stop at the first feature, will integrate all evidence and decisively choose Option B. Other trials reverse this or create scenarios where the Accumulator must choose one option regardless of threshold while the Mixture model can choose the other if Tallying dominates.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To strictly dissociate the Sequential Evidence Accumulation model from the Probabilistic Strategy Selection (Mixture) model, we exploit the fact that the Mixture model is strictly bounded by its two constituent heuristics: Take-The-Best (TTB) and Tallying. In Trials 1 and 2, Option B wins on the single highest-validity feature (thus TTB chooses B) and also wins on more features overall (thus Tallying chooses B). Consequently, the Mixture model MUST predict Option B. However, the sum of the validities for Option A's winning features is greater than that for Option B. The Sequential Evidence Accumulation model, if it has a sufficiently high threshold, will not stop at the first feature but will integrate all evidence, thereby decisively choosing Option A. This creates a scenario where the advocated model can predict a choice that is mathematically impossible for the competing mixture model.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.8,
    0.55,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      0,
      1,
      1,
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
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
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
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
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
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
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
  "prompt_token_count": 3921,
  "candidates_token_count": 552,
  "total_token_count": 7314
}
```
