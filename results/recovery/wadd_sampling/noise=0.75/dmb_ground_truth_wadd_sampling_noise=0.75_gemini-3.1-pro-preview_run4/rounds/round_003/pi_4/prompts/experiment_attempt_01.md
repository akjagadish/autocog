# experiment_attempt_01

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
**Description:** Strategy Mixture Theory with Independent Scaling: Individuals use a probabilistic mixture of distinct heuristics (WADD, Tallying, and Take-The-Best), but because the internal evidence scales of these heuristics vary dramatically (log-odds sums vs. integer counts vs. binary indicators), each heuristic applies its own independent temperature parameter to properly calibrate its choice probabilities before mixing.

**Parameters:**
- beta_wadd: [0.1, 10.0]
- beta_tally: [0.1, 10.0]
- beta_ttb: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- w_wadd: [0.0, 1.0]
- w_tally: [0.0, 1.0]
- w_ttb: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, w)
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
    if winner is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    beta_wadd = float(parameters["beta_wadd"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    p_wadd = get_probs(scores_wadd, beta_wadd)
    p_tally = get_probs(scores_tally, beta_tally)
    p_ttb = get_probs(scores_ttb, beta_ttb)
    
    w1 = float(parameters["w_wadd"])
    w2 = float(parameters["w_tally"])
    w3 = float(parameters["w_ttb"])
    w_sum = w1 + w2 + w3 + 1e-9
    
    p_mix = (w1 * p_wadd + w2 * p_tally + w3 * p_ttb) / w_sum
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Rank-based Weighting Theory posits that decision-makers do not use complex mathematical transformations like log-odds to weigh evidence. Instead, they rely on the simple ordinal ranking of cue validities. Cues are weighted according to an inverse function of their rank (proportional to 1/rank^rho). By restricting the decay parameter rho to a moderate range, the theory maintains a highly compensatory mechanism that prevents over-reliance on the most valid cue, capturing the softer probability matching observed in human multi-attribute decision making.

**Parameters:**
- beta: [0.01, 15.0]
- epsilon: [0.0, 0.5]
- rho: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute ranks (1 is the highest validity)
    # Using stable sort for consistent tie-breaking if validities are equal
    order = np.argsort(-val, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Apply rank-based weighting
    rho = float(parameters["rho"])
    w = 1.0 / (ranks ** rho)
    
    # Compute weighted sum of features for each option
    scores = np.dot(stim, w)
    
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
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Tallying, we construct trials where the highest-validity discriminating cue favors one option, but a larger number of lower-validity cues favor the alternative. TTB will choose the option favored by the single best cue, while Tallying will simply count the total number of winning cues for each option and choose the one with the higher tally, ignoring cue validities. We also include trials where Tallying results in a tie (equal number of winning cues) while TTB makes a deterministic choice based on the higher-validity cue.

[1] To pit Tallying against Take The Best (TTB), we use a 5-feature design with strictly decreasing validities. The critical trials are designed such that the highest validity cue (Feature 1) favors one option, while a larger number of lower-validity cues favor the other option. TTB relies solely on the first discriminating cue and will therefore choose the option favored by Feature 1. In contrast, Tallying simply counts the number of winning features, ignoring validities, and will choose the option with more winning cues. We also include trials where Tallying results in a tie (equal number of winning cues for both options) but TTB makes a deterministic choice based on the highest validity cue, cleanly separating the predictions of the two models.

[2] To cleanly dissociate WADD (Weighted Additive Model) from Tallying, we use a 5-feature design with a steep distribution of validities. WADD integrates features by weighting them proportionally to their log-odds, allowing a single highly valid cue to outweigh several weak cues. Tallying, however, strictly counts the number of winning features for each option regardless of their validity. By pitting a single high-validity feature against multiple low-validity features, we create trials where WADD strongly prefers the option with the high-validity feature, while Tallying prefers the option with a greater quantity of winning features. We also include trials that reverse this polarity to ensure the dissociation holds symmetrically.

[3] To quantitatively dissociate Tallying from the Weighted Additive Model (WADD), we utilize a 5-feature design with a high variance in cue validities. WADD integrates features by weighting them according to their log-odds, allowing a single highly valid cue (e.g., 0.95) to completely override multiple lower-validity cues (e.g., 0.60, 0.55, 0.51). In contrast, Tallying simply counts the number of winning features for each option, treating all features equally regardless of validity. We construct trials where one option wins on a single high-validity feature while the other option wins on multiple low-validity features, leading to opposing predictions. We also include trials where Tallying predicts a tie (equal number of winning features) while WADD strongly prefers the option with the more valid features.

[4] To dissociate WADD from a mixture of Tallying and Take-The-Best (TTB), we construct critical trials where WADD predicts the opposite of both TTB and Tallying. By setting one extremely high validity cue (0.95), two highly valid cues (0.90, 0.85), and three weak cues (~0.5), we can create a situation where Option A wins on the two highly valid cues, while Option B wins on the extremely high validity cue AND the three weak cues. Here, WADD's log-odds integration strongly favors Option A. However, TTB favors Option B (due to the highest validity cue), and Tallying also favors Option B (4 winning cues vs 2). A mixture model will thus predict a much higher choice probability for Option B than the pure WADD model. We include symmetric versions and trials where heuristics partially agree to estimate mixture weights.

[5] To quantitatively dissociate the Strategy Mixture Theory from a pure Weighted Additive (WADD) model, we design a 5-feature experiment with validities that create distinct profiles of agreement and disagreement among WADD, Take-The-Best (TTB), and Tallying. In a pure WADD model, choice probabilities are exclusively determined by the difference in the options' log-odds sums. In contrast, the Mixture Theory incorporates TTB (which is highly sensitive to the single most valid discriminating cue) and Tallying (which is sensitive only to the unweighted count of winning cues), each with independent softmax temperatures. By including trials where WADD prefers one option while TTB and Tallying either tie or pull in opposite directions, the Mixture Theory predicts choice probabilities that systematically deviate from the strict log-odds differences predicted by pure WADD.

[6] To quantitatively dissociate Rank-based Weighting Theory from the Strategy Mixture Theory, we use a set of validities where the top three cues have high and closely clustered objective validities (e.g., 0.90, 0.88, 0.86), while the remaining cues have low validities (~0.55, 0.51). For a log-odds based mechanism like WADD (a core component of the Strategy Mixture), the top three cues have very similar, large evidence weights, meaning the 2nd and 3rd cues combined will easily outweigh the 1st cue. However, Rank-based Weighting relies strictly on the ordinal rank of the cues, enforcing a steep decay (e.g., proportional to 1/rank^rho) regardless of how close the actual validities are. By pitting the 1st-ranked cue (sometimes with weak cues) against the 2nd and 3rd-ranked cues combined, Rank-based Weighting predicts the 1st cue dominates (since 1 > 1/2 + 1/3), whereas WADD strongly favors the combination of the 2nd and 3rd cues. We include variations in the number of weak cues to manipulate Tallying, creating profiles where Strategy Mixture and Rank-based Weighting make opposite predictions.

[7] To quantitatively dissociate the Strategy Mixture Theory (WADD, Tallying, TTB) from the Rank-based Weighting Theory, we exploit their fundamentally different treatments of cue validities. By selecting a set of high, closely-spaced validities (e.g., 0.95 to 0.91), the log-odds weights used by WADD (a core component of the Strategy Mixture) remain relatively flat and large. In contrast, Rank-based Weighting ignores the absolute validity values and strictly enforces a steep decay based on ordinal rank (e.g., proportional to 1/rank). We design critical trials where Option A possesses the 1st-ranked cue and the 5th-ranked cue, while Option B possesses the 2nd, 3rd, and 4th-ranked cues. WADD heavily favors Option B because the sum of three large log-odds easily outweighs the sum of two. Tallying also favors Option B (3 cues vs 2). Thus, the Strategy Mixture Theory strongly predicts a preference for Option B. However, for a moderate to high decay parameter, Rank-based Weighting predicts that the 1st-ranked cue alone carries enough weight to make Option A the winner (e.g., 1 + 1/5 > 1/2 + 1/3 + 1/4). This creates a sharp divergence in the predicted choice probabilities between the two theories.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Strategy Mixture Theory from the Rank-based Weighting Theory, we exploit the non-linear transformation of validities into log-odds (used by WADD in the Mixture Theory) versus the strict ordinal rank decay (used by Rank-based Weighting). We select a top-heavy validity distribution where the first three cues are highly valid and closely spaced, while the remaining cues are weaker. In critical trials, Option A is supported by the 2nd and 3rd ranked cues, while Option B is supported by the 1st and 5th ranked cues. For WADD, the sum of the log-odds for cues 2 and 3 heavily outweighs the sum for cues 1 and 5, strongly favoring Option A. Tallying results in a tie (2 cues each), leaving the Strategy Mixture Theory to predominantly favor Option A (unless the TTB weight is overwhelmingly high). Conversely, the Rank-based Weighting Theory applies a steep decay based strictly on rank (e.g., 1/rank), meaning the 1st ranked cue combined with the 5th easily outweighs the 2nd and 3rd (e.g., 1 + 1/5 > 1/2 + 1/3), leading to a strict preference for Option B. This creates a sharp, qualitative divergence in predicted choice probabilities.",
  "validities": [
    0.88,
    0.85,
    0.82,
    0.7,
    0.6
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
      0,
      1
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
    ],
    [
      0,
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
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
    ],
    [
      1,
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
  "prompt_token_count": 4197,
  "candidates_token_count": 564,
  "total_token_count": 9452
}
```
