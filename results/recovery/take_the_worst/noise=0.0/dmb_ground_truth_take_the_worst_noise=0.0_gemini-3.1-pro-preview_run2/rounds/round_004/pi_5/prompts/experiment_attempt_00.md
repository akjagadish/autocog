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
**Description:** Anti-Expertise Tallying: Decision-makers actively distrust or ignore high-validity 'expert' cues, perceiving them as overly salient, redundant, or manipulative. Instead of using them, subjects filter out features whose stated validity exceeds a certain personal threshold. After discarding these high-validity cues, subjects apply a standard Tallying heuristic (counting strict feature-wise wins) on the remaining lower-validity features to make their choice.

**Parameters:**
- threshold: [0.5, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Anti-Expertise Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Filter out features with validity above the threshold
    mask = val <= threshold
    
    # Fallback to all features if the threshold is so low that all features are discarded
    if not np.any(mask):
        mask = np.ones_like(val, dtype=bool)
        
    a_masked = a[mask]
    b_masked = b[mask]
    
    # Tallying on the remaining features
    a_wins = float(np.sum(a_masked > b_masked))
    b_wins = float(np.sum(b_masked > a_masked))
    
    scores = np.array([a_wins, b_wins])
    
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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Inverse Validity Weighting: Subjects actively distrust expert ratings, treating high-validity cues as manipulative or overly salient. Instead of relying on them, they use a weighted additive model where the subjective weight of each feature is inversely proportional to its stated validity (1 - validity). This mechanism leads subjects to consistently prefer options that are supported by lower-validity features over those supported by high-validity features, explaining the strong anti-expert choices observed in the experiments.

**Parameters:**
- beta: [0.1, 100.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Inverse Validity Weighting expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Primary mechanism: Weighted additive using inverse validities
    # Lower validity features receive higher subjective weights
    weights = 1.0 - val
    
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
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically dissociates Take The Best (TTB) from Tallying. By using five features with strictly decreasing validities, TTB relies entirely on the first feature that discriminates between the two options. Tallying, however, simply counts the number of features favoring each option, ignoring validities. The trial pairs pit a single higher-validity cue against multiple lower-validity cues, causing TTB and Tallying to make diametrically opposite predictions on most trials. Some trials also feature ties for Tallying where TTB makes a clear prediction, further separating the models' choice probabilities.

[1] This design systematically pits Tallying against Take The Best (TTB) by creating trials where the highest-validity discriminating cue favors one option, but a larger number of lower-validity cues favors the other. In Tallying, all features are weighted equally, so the option with more winning features is chosen. In TTB, only the single most valid discriminating feature matters. The design includes trials with strong opposition (e.g., 4 cues to 1), moderate opposition (e.g., 3 cues to 1), and trials where Tallying predicts a tie while TTB makes a deterministic prediction. This provides a robust quantitative dissociation across the parameter space.

[2] This design systematically dissociates the Weighted Additive (WADD) rule from Tallying by contrasting the number of winning features against the sum of their validities. Tallying ignores validities entirely, counting only the number of features favoring each option. WADD, however, computes a weighted sum where weights are derived from validities (scaled by an exponent gamma). The trial set includes 'opposition' trials where one option is supported by fewer but highly valid features, while the other is supported by a greater number of low-validity features, causing the models to predict opposite choices. It also includes 'tie' trials where both options win on an equal number of features (Tallying predicts a 50/50 guess) but the features differ in validity, leading WADD to predict a strong preference.

[3] This design systematically dissociates Tallying from the Weighted Additive (WADD) rule by contrasting the sheer quantity of winning features against the sum of their validities. Tallying ignores validities, relying entirely on the count of features favoring each option. WADD computes a weighted sum where validities (scaled by gamma) determine the weights. By using a bimodal validity distribution (two highly valid features and three low-validity features), we create 'opposition' trials where one option wins on the two highly valid features while the other wins on the three low-validity features. Tallying predicts a win for the option with three features, whereas WADD strongly favors the option with the two highly valid features (provided gamma > 0). We also include 'tie' trials where both options win on an equal number of features, leading Tallying to predict a 50/50 guess, while WADD predicts a strong preference based on validity differences.

[4] This design systematically dissociates Standard Tallying from Skeptical Tallying by focusing on trials where the primary Tallying mechanism results in a tie. Under Standard Tallying, when both options win on an equal number of features, the model predicts a 50/50 guess regardless of the features' validities. In contrast, Skeptical Tallying breaks these ties by weighting the features inversely to their stated validity, favoring the option that possesses more lower-validity features. By constructing trials where option A wins on low-validity features and option B wins on high-validity features, Standard Tallying predicts indifference, whereas Skeptical Tallying predicts a clear preference for option A. Additional trials vary the magnitude of the tie-breaker difference to capture the effect across the parameter space.

[5] To quantitatively dissociate Standard Tallying from Skeptical Tallying, we exploit the latter's 'tie-breaker' mechanism, which effectively functions as an additive bonus that scales inversely with feature validity. For Standard Tallying, all feature wins are weighted equally (weight = 1), and validities are ignored. For Skeptical Tallying, a feature win contributes 1 + gamma * (1 - validity). Thus, Skeptical Tallying assigns strictly higher weight to wins on low-validity features. We use 7 features (four high validity 0.95, three low validity 0.50). In 'tie' trials (e.g., A wins on 2 high-validity features, B wins on 2 low-validity features), Standard Tallying predicts a 50/50 guess, while Skeptical Tallying strongly prefers B. In 'opposition' trials (A wins 4 high-validity features, B wins 3 low-validity features), Standard Tallying deterministicly prefers A (4 wins vs 3), whereas Skeptical Tallying can flip its preference to B if the skepticism parameter gamma is sufficiently high, because the cumulative low-validity bonus outweighs the single extra win for A.

[6] This design dissociates 'Skeptical Tallying' from 'Anti-Expertise Tallying' by contrasting their mechanisms for handling high-validity cues. Skeptical Tallying retains all features in its primary tally but uses low-validity features as a tie-breaker bonus, meaning it will still strongly favor an option with a clear majority of high-validity wins. Conversely, Anti-Expertise Tallying applies a hard threshold, completely discarding features whose validities are deemed too high. By pitting an option with multiple high-validity wins against an option with fewer but lower-validity wins, Skeptical Tallying predicts a choice for the high-validity majority, while Anti-Expertise Tallying (depending on the threshold) will either predict a tie or a win for the low-validity option because the high-validity features have been filtered out.

[7] This design systematically dissociates 'Anti-Expertise Tallying' from 'Skeptical Tallying'. Skeptical Tallying always counts all features but uses low-validity features as a tie-breaker, so it strongly favors options with a strict majority of feature wins (even if they are high-validity) and breaks ties in favor of options with more low-validity features. Anti-Expertise Tallying, however, applies a hard threshold to filter out high-validity features altogether before tallying. The trials include 'opposition' trials where Option A wins on a majority of features that happen to be high-validity, causing Skeptical Tallying to choose A, while Anti-Expertise Tallying drops those features and chooses B. We also include 'tie' trials where both options have an equal number of wins, but the distribution of validities causes Skeptical Tallying's tie-breaker to favor Option A, whereas Anti-Expertise Tallying's filtering mechanism causes it to favor Option B.

[8] This design systematically dissociates Inverse Validity Weighting (IVW) from Anti-Expertise Tallying (AET) by exploiting their fundamentally different mechanisms for handling high-validity features. IVW retains all features but assigns them subjective weights inversely proportional to their stated validity (1 - validity). Consequently, a large number of high-validity features can still collectively outweigh a single low-validity feature if their summed inverse-validity weights are high enough. In contrast, AET applies a hard threshold, completely filtering out features whose validity exceeds the subject's personal limit before tallying the remaining features. By pitting an option with multiple high-validity wins against an option with fewer (but lower-validity) wins, IVW will often favor the option with numerous high-validity features due to the accumulated weight. However, AET will entirely discard those high-validity features for subjects with moderate-to-low thresholds, causing it to deterministically choose the option supported by the remaining low-validity features.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design systematically dissociates Inverse Validity Weighting (IVW) from Anti-Expertise Tallying (AET) by exploiting their different mechanisms for discounting high-validity features. IVW retains all features but assigns them subjective weights inversely proportional to their stated validity (1 - validity). Thus, a large number of high-validity features can collectively outweigh a single low-validity feature if their summed inverse-validity weights are sufficiently high. In contrast, AET applies a hard threshold, completely filtering out features whose validity exceeds the subject's personal limit before tallying the remaining features. By pitting an option with many high-validity wins against an option with fewer (but lower-validity) wins, IVW will favor the option with numerous high-validity features due to the accumulated inverse weight. However, AET will entirely discard those high-validity features for subjects with moderate-to-low thresholds, causing it to deterministically choose the option supported by the remaining low-validity features.",
  "validities": [
    0.9,
    0.9,
    0.9,
    0.85,
    0.85,
    0.85,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
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
      1,
      1,
      1,
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
      0
    ],
    [
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
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
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
      1,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
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
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
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
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      1,
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
      0,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3955,
  "candidates_token_count": 725,
  "total_token_count": 7787
}
```
