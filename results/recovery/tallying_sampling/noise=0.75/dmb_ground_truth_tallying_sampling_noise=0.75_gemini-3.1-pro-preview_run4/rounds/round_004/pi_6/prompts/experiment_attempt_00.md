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
**Description:** Minimalist Heuristic (Trial-Specific Search): Decision-makers evaluate options sequentially using a one-reason lexicographic process, completely ignoring objective cue validities. Instead of maintaining a stable subjective cue hierarchy, individuals search through the available cues in a random order that varies from trial to trial. They stop at the first cue that discriminates between the two options and choose the favored option. Analytically, the probability of choosing an option is proportional to its share of the discriminating cues. This trial-by-trial stochasticity produces low per-subject determinism on symmetrically tied trials, matching human data.

**Parameters:**
- beta: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in the Minimalist heuristic
    _ = parameters["validities"]
    
    # Trial-specific random cue order is analytically equivalent to choosing 
    # proportional to the number of discriminating cues for each option.
    diff = stim[0] - stim[1]
    d_a = np.sum(diff > 0)
    d_b = np.sum(diff < 0)
    
    if d_a + d_b > 0:
        p_a = d_a / (d_a + d_b)
        p_b = d_b / (d_a + d_b)
    else:
        p_a = 0.5
        p_b = 0.5
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores [1, 0] or [0, 1] marginalized over the random cue order
    S = np.exp(beta) / (np.exp(beta) + 1.0)
    
    prob_a = p_a * S + p_b * (1.0 - S)
    prob_b = p_b * S + p_a * (1.0 - S)
    
    p_core = np.array([prob_a, prob_b])
    
    # Uniform lapse
    n_opts = len(p_core)
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
**Description:** Tallying (Equal Weighting): Decision-makers completely ignore the provided cue validities and simply count the number of positive features for each option. The option with the highest tally is chosen. If the tallies are equal, the decision-maker guesses randomly. This heuristic provides an extremely fast and frugal way to compare options, perfectly explaining chance-level performance in 1-on-1 single-cue comparisons (where tallies tie) and highly consistent choices when one option has strictly more positive features. The decision process is subject to significant response noise, reflecting the inherent stochasticity in human choice behavior.

**Parameters:**
- beta: [0.0, 1.5]
- epsilon: [0.2, 0.8]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in Tallying, but we read them to satisfy the parameter reference rule.
    _ = parameters["validities"]
    
    # Tallying: sum the features for each option (ignoring validities)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
    n_opts = len(p_core)
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
[0] To cleanly dissociate Take The Best (TTB) from Weighted Additive (WADD), we employ a design where a single highly valid cue favors one option, but multiple lower-validity cues favor the other. TTB, being a non-compensatory 'one-reason' heuristic, will consistently choose the option with the highest valid discriminating cue, ignoring all other features. In contrast, WADD integrates all features, allowing a deficit on the most valid cue to be compensated by an accumulation of smaller advantages on lower-validity cues. The trial pairs are constructed to pit the highest-validity discriminating cue against various combinations of lower-validity cues, ensuring the two models predict opposite choices across multiple contexts.

[1] Instead of simply pitting the single best cue against all others (which was previously explored), this design systematically introduces ties on the highest-validity cues to trigger the 'cue cascade'. TTB evaluates options sequentially and stops at the first discriminating cue, ignoring all remaining cues regardless of how many favor the alternative. By tying the top cues in various combinations, we force TTB to rely on the 2nd or 3rd cue, while WADD continues to integrate all features. This allows us to test if the non-compensatory threshold is absolute or if subjects are more likely to compensate when the discriminating cue is further down the validity hierarchy. WADD predicts a compensatory choice favoring the option with more lower-validity cues, while TTB predicts strict adherence to the first discriminating cue, even if it is only the 3rd most valid overall.

[2] To quantitatively dissociate the Strategy Mixture Model from a pure Weighted Additive (WADD) model, we exploit the fact that a pure WADD model predicts choice probabilities that are strictly a monotonic function of the difference in weighted sums (a single sigmoid curve). The Strategy Mixture Model, however, predicts that choices are a mixture of WADD and Take The Best (TTB). Therefore, if we plot the probability of choosing Option A against the WADD difference (WADD_A - WADD_B), the Mixture Model predicts that the points will split into distinct curves depending on which option TTB favors. We construct pairs that systematically vary the WADD difference while orthogonally manipulating the TTB prediction (e.g., pairs where WADD strongly favors B but TTB favors A, and vice versa). A pure WADD model cannot account for vertical shifts in choice probability at the same WADD difference caused by the non-compensatory TTB component.

[3] By fixing validities such that multiple pairs produce an exact tie in the objective Weighted Additive (WADD) score, we can sharply dissociate the Advocated WADD model (which features free weighting parameters per cue) from the Strategy Mixture Model (which restricts its WADD component to use exact objective validities). For these objective-WADD-tied pairs, the Strategy Mixture Model is strictly forced to predict a preference aligned with the Take The Best (TTB) heuristic, because the WADD component provides no directional preference. In contrast, the Advocated WADD model can accommodate choices that go against TTB on these 'tied' trials by allowing subjective weights to deviate from objective validities. We include a spectrum of objective WADD ties, small differences, and opposing TTB predictions to fully expose this parameterization difference.

[4] To dissociate the Advocated Non-linear Weighting Model from the Competing Weighted Additive (WADD) Model, we exploit the difference in their parameterizations. The Advocated model uses a single parameter, gamma, to non-linearly transform all objective validities simultaneously. This means the trade-offs between any single high-validity cue and multiple lower-validity cues are strictly coupled across the entire validity hierarchy. In contrast, the Competing model possesses an independent subjective weight parameter (w) for every feature, allowing it to decouple these trade-offs (e.g., treating the most valid cue non-compensatorily while treating the rest compensatorily). By systematically pitting single higher-validity cues against different combinations of lower-validity cues across the hierarchy, we test if the transitions between compensatory and non-compensatory choices follow the strict 1-dimensional manifold predicted by a single gamma parameter, or if they require the independent flexibility of the Competing model.

[5] To quantitatively dissociate the Advocated Theory (Weighted Additive with free per-feature weights) from the Competing Theory (Non-linear Weighting with a single gamma parameter), we exploit the strict structural constraints imposed by the single gamma parameter. In the Competing model, the subjective weights must strictly preserve the objective validity ordering (since x^gamma > y^gamma for any x > y and gamma > 0), and the trade-offs between cues are tightly coupled across the entire validity hierarchy. In contrast, the Advocated model possesses an independent weighting parameter for each feature, allowing it to easily decouple trade-offs (e.g., heavily discounting one specific high-validity cue while preserving the weights of others) and even invert the subjective importance of cues. We present direct 1-on-1 cue comparisons alongside complex 1-vs-many and 2-vs-many tradeoffs to test if subjects' choices adhere to the rigid 1-dimensional manifold predicted by a single gamma parameter, or if they exhibit the multidimensional flexibility captured by the Advocated model's free weights.

[6] To quantitatively dissociate Tallying from the Weighted Additive (WADD) model, we exploit Tallying's core assumption: it completely ignores cue validities and relies solely on the unweighted sum of positive features. We construct trials that pit a small number of high-validity features against a larger number of low-validity features. In these cases, Tallying strictly predicts a preference for the option with more features, while WADD predicts a preference for the option with fewer, higher-validity features. Furthermore, we include 'tally-tied' trials where both options have the same number of positive features but differ significantly in total validity. Tallying predicts exact indifference (chance-level choice) on these trials, whereas WADD predicts a strong preference for the option with higher-validity features. This structural divergence in predictions across multiple trial types provides a sharp, quantitative dissociation between the two models.

[7] To quantitatively dissociate the Advocated Theory (Weighted Additive with free per-feature weights) from the Competing Theory (Tallying), we exploit Tallying's core assumption: it completely ignores cue validities and relies solely on the unweighted sum of positive features. We construct trials that pit a small number of high-validity features against a larger number of low-validity features. In these cases, Tallying strictly predicts a preference for the option with more features, while WADD can predict a preference for the option with fewer, higher-validity features. Furthermore, we include multiple 'tally-tied' trials where both options have the same number of positive features but differ significantly in total validity. Tallying predicts exact indifference (chance-level choice) on these trials, whereas WADD predicts a strong preference for the option with higher-validity features. This structural divergence provides a sharp, quantitative dissociation.

[8] To quantitatively dissociate Tallying from the Minimalist Heuristic, we exploit the fact that while both models ignore validities and agree on the ordinal direction of preference (they both favor the option with more positive features), they scale their choice probabilities differently. Tallying's choice probability depends strictly on the absolute difference in tallies (tally_A - tally_B). In contrast, the Minimalist Heuristic's choice probability depends on the proportion of discriminating cues favoring the chosen option (d_A / (d_A + d_B)). By creating a series of trials where the absolute tally difference is held constant (e.g., always 1) but the total number of discriminating cues increases, Tallying predicts identical choice probabilities across these trials. The Minimalist model, however, predicts a decreasing preference strength as the proportion of favored cues dilutes (e.g., 1/1 vs 2/3 vs 3/5 vs 4/7).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Minimalist Heuristic from Tallying, we exploit the inverse of a previously explored relationship. While both models ignore cue validities, Tallying relies on the absolute difference in positive features (tally_A - tally_B), whereas the Minimalist Heuristic relies on the proportion of discriminating cues favoring the option (d_A / (d_A + d_B)). We construct sets of trials where the proportion of discriminating cues is strictly held constant (e.g., 2/3 or 3/4 or 1) while the absolute difference in tallies varies (e.g., d_A=2 vs d_B=1 gives a difference of 1; d_A=4 vs d_B=2 gives a difference of 2; d_A=6 vs d_B=3 gives a difference of 3). The Advocated Minimalist model predicts identical choice probabilities across trials within these sets because the relative share of favorable cues is unchanged. In contrast, the Competing Tallying model predicts that preference strength will increase monotonically as the absolute tally difference grows, despite the constant proportion.",
  "validities": [
    0.9,
    0.85,
    0.8,
    0.75,
    0.7,
    0.65,
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
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
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
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
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
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
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
      0,
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
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
      0,
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
      0,
      0,
      0,
      0,
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
  "prompt_token_count": 3950,
  "candidates_token_count": 803,
  "total_token_count": 6865
}
```
