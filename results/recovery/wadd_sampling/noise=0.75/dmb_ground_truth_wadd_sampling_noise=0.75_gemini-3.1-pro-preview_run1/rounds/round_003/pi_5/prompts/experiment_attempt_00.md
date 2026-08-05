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
**Description:** High-Lapse Tallying Theory: Subjects find the multi-attribute binary feature task cognitively demanding, leading to a very high rate of random guessing. However, when they do engage with the stimulus, they rely on a simple unweighted tally of positive features (Tallying) rather than integrating complex cue validities. This produces a very weak but non-zero behavioral signal that slightly favors options with a higher simple count of positive features.

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.7, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies for the engaged decision process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Mix with a high lapse rate (random guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Generalized Weighted Additive (WADD) Theory: Decision-makers evaluate options by computing a compensatory weighted sum of their features. However, instead of using raw cue validities as weights, individuals scale the validities non-linearly using a parameter gamma. This scaling allows the decision process to smoothly interpolate between Tallying (gamma=0, where all cues are weighted equally), standard WADD (gamma=1), and Take The Best (gamma -> infinity, where the most valid cue dominates). The final choice is made probabilistically via a softmax function over the computed option values, mixed with a random lapse rate.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 10.0]
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
        raise ValueError("Length of validities must match number of features.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design strictly dissociates Take The Best (TTB) from Tallying by pitting the highest-validity discriminating cue against the sheer number of winning cues. In each trial, one option wins on the most valid discriminating cue (thus chosen by TTB), while the other option wins on a strictly greater number of lower-validity cues (thus chosen by Tallying). By varying which cue is the highest discriminating one (1st, 2nd, or 3rd validities), we can thoroughly test whether subjects rely on a single reason or sum the features.

[1] This design quantitatively dissociates Tallying from Take The Best (TTB) by systematically pitting the highest-validity discriminating cue against the unweighted sum of lower-validity cues. We use 5 features with strictly decreasing validities. In every trial, one option is favored by TTB because it wins on the single most predictive cue that discriminates the options, whereas the other option is favored by Tallying because it wins on a strictly greater number of lower-validity cues. By shifting which cue is the highest discriminating one (1st, 2nd, or 3rd validity) across trials, we can rule out alternative heuristics like simple weighting and confirm whether subjects are purely counting features (Tallying) or adopting a one-reason stopping rule (TTB).

[2] This design strictly dissociates Generalized Weighted Additive (WADD) Theory from Tallying. Tallying ignores cue validities and simply counts the number of features favoring each option, treating all features equally. WADD, on the other hand, weights features by their non-linearly scaled validities (parameterized by gamma). The trials are carefully constructed to include: (1) pairs where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers one option due to validity differences; (2) pairs where Tallying prefers one option because it wins on more features, but WADD prefers the other because its fewer winning features have a higher combined validity (even at gamma=1); and (3) pairs that distinguish different ranges of the WADD gamma parameter, where Tallying and WADD (at gamma=1) agree, but WADD at higher gamma values (approaching Take The Best) flips its preference.

[3] To quantitatively dissociate Tallying from Generalized WADD, we must exploit the fact that Tallying strictly ignores cue validities (effectively fixing the WADD gamma parameter to 0). In this design, we pit the sheer number of winning features against the validities of those features. Several trials are constructed such that Tallying predicts a strict tie (equal number of winning features) while WADD predicts a strong preference for the option with the higher-validity feature. Other trials present a scenario where one option has fewer winning features but higher validities (benefiting WADD with gamma >= 1), while the other option has more winning features of lower validity (favored by Tallying). If subjects use Tallying, they will show indifference on the tied trials and a preference for the option with more winning features on the others, a pattern WADD can only fit by collapsing gamma to 0.

[4] To quantitatively dissociate the Generalized Weighted Additive (WADD) Theory from the Random Guessing Theory, we need to present trials where WADD predicts a systematic, varying preference across trials while Random Guessing predicts a flat 50% choice probability everywhere. We use four features with evenly spaced validities. The trials range from complete dominance (where WADD predicts near 100% preference) to partial conflicts (where WADD predicts moderate preferences depending on the gamma and beta parameters). If subjects are merely guessing randomly, all trials will yield ~50% choice rates. If they are evaluating features as WADD predicts, their choice probabilities will systematically track the difference in the non-linearly weighted feature sums.

[5] To quantitatively dissociate the Generalized Weighted Additive (WADD) Theory from the Random Guessing Theory, we use a 5-feature design with a wide spread of validities. Random Guessing predicts a flat 50% choice probability across all trials, regardless of the features presented. WADD predicts systematic deviations from 50% based on the weighted sum of the features. We include trials ranging from complete dominance (where WADD predicts near 100% preference for one option) to complex trade-offs (e.g., the highest validity cue versus multiple lower validity cues). If subjects are merely guessing, their choice rates will remain at 50% across all these diverse trial types. If they follow a compensatory or heuristic-like strategy captured by WADD, their choice probabilities will systematically track the differences in feature values and validities.

[6] To quantitatively dissociate Generalized WADD from High-Lapse Tallying, we exploit two main differences: the integration of cue validities and the base lapse rate. High-Lapse Tallying strictly counts positive features (ignoring validities) and assumes a high random guessing rate (epsilon >= 0.7), capping its maximum predicted choice probability around 65%. Generalized WADD accounts for validities (scaled non-linearly) and allows for much lower lapse rates, thus predicting extreme choice probabilities on dominant trials. We include trials where Tallying predicts a tie but WADD predicts a strong preference (e.g., A=[1,0,0,0] vs B=[0,1,0,0]), trials where Tallying prefers one option but WADD prefers the other due to high-validity cues (e.g., A=[1,0,0,0] vs B=[0,1,1,1]), and dominance trials to expose the difference in maximum choice probabilities (e.g., A=[1,1,1,0] vs B=[0,0,0,1]).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate High-Lapse Tallying from Generalized WADD, we exploit their differences in both feature weighting and base lapse rates. High-Lapse Tallying strictly counts positive features (ignoring validities) and assumes a high random guessing rate (capping its maximum choice probability around 65%). Generalized WADD accounts for validities (scaled non-linearly) and allows for much lower lapse rates, thus predicting extreme choice probabilities on dominant trials. We include: (1) trials where Tallying predicts a tie but WADD predicts a strong preference based on validity differences, (2) conflict trials where Tallying prefers one option because it has more positive features but WADD prefers the other due to high-validity cues, and (3) a pure dominance trial to expose the difference in maximum predicted choice probabilities due to the distinct epsilon bounds.",
  "validities": [
    0.95,
    0.85,
    0.6,
    0.55,
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
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
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
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0
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
  "prompt_token_count": 3370,
  "candidates_token_count": 493,
  "total_token_count": 4771
}
```
