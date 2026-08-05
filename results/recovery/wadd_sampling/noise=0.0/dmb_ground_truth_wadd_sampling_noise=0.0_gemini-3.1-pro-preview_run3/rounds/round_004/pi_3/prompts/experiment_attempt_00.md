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
**Description:** Weighted Additive (WADD) Theory: Decision-makers evaluate options using a fully compensatory strategy. They multiply each feature's value by its corresponding cue validity and sum these products to form an overall subjective value for each option. The option with the higher weighted sum is chosen. This allows multiple lower-validity cues to collectively outweigh a single high-validity cue, capturing behavior that falls between pure Take The Best and pure Tallying. To account for empirical response noise, the decision process incorporates a moderate degree of stochasticity.

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.1, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
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
**Description:** Contingent Heuristic Theory (Difference-based Compensatory): Decision-makers conditionally apply decision strategies based on the relative strength of the best available evidence compared to the next best piece of evidence. They identify the most valid cue that discriminates between the two options and the second most valid discriminating cue. If the difference in validity between these two cues exceeds a subjective threshold (indicating the top cue is highly salient relative to the rest), they rely entirely on the top cue (Take-The-Best). If the difference is not sufficiently large, they fall back to a fully compensatory Weighted Additive (WADD) strategy.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- delta: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    cue_order = np.argsort(-val, kind='stable')
    
    ttb_winner = None
    v_max = 0.0
    v_next = 0.0
    found_first = False
    
    for cue in cue_order:
        if a[cue] != b[cue]:
            if not found_first:
                v_max = val[cue]
                ttb_winner = 0 if a[cue] > b[cue] else 1
                found_first = True
            else:
                v_next = val[cue]
                break
                
    delta = float(parameters["delta"])
    
    if ttb_winner is not None and (v_max - v_next) >= delta:
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
    else:
        score_a = np.sum(a * val)
        score_b = np.sum(b * val)
        scores = np.array([score_a, score_b])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
[0] This design systematically pits Take The Best (TTB) against Tallying using 5 features with descending validities. TTB decides solely based on the highest-validity cue that discriminates between the two options, while Tallying simply counts the number of features each option wins on, regardless of their validities. The trials are constructed such that in the critical pairs, one option wins on the single most valid discriminating cue (favored by TTB), while the other option wins on a larger number of less valid cues (favored by Tallying). Some agreement trials are also included to anchor the noise parameters.

[1] This design systematically dissociates Tallying from Take The Best (TTB) across a 5-feature space. TTB decides solely based on the highest-validity cue that discriminates between the two options. Tallying counts the number of features each option wins on, completely ignoring validities. By constructing trials where one option wins on the single most valid discriminating cue (favored by TTB) while the alternative option wins on a larger number of less-valid cues (favored by Tallying), we create strong quantitative dissociations. We also include agreement trials and trials where Tallying produces ties (forcing a guess) while TTB makes deterministic predictions, allowing precise estimation of the noise parameters.

[2] This design aims to dissociate the Weighted Additive (WADD) theory from Tallying by manipulating the distribution of feature validities. WADD computes a compensatory weighted sum using the exact validities, allowing a small number of highly valid cues to outweigh a larger number of less valid cues. Tallying, on the other hand, ignores cue validities entirely and simply counts the number of features favoring each option. By setting two high-validity features and three low-validity features, we create critical trials where one option is favored by WADD (winning on the two high-validity cues) while the other is favored by Tallying (winning on the three low-validity cues). We also include trials where Tallying predicts a tie (equal number of wins) but WADD makes a strong prediction, as well as agreement trials to anchor noise parameter estimates.

[3] To cleanly dissociate Tallying from the Weighted Additive (WADD) theory, this design uses a bimodal distribution of feature validities: two highly valid cues and three lower-validity cues. Tallying ignores validities and simply counts the number of features on which an option wins. WADD, however, computes a compensatory weighted sum, allowing the two high-validity cues to outweigh the three low-validity cues. We include critical dissociation trials where one option wins on the three low-validity cues (favored by Tallying) while the other option wins on the two high-validity cues (favored by WADD). We also include trials where Tallying predicts a tie (equal number of wins) but WADD makes a clear prediction, as well as agreement trials to anchor noise parameters.

[4] This design systematically pits the Weighted Additive (WADD) theory against Take The Best (TTB). TTB uses a non-compensatory lexicographic rule, deciding solely based on the highest-validity cue that discriminates between the options. In contrast, WADD uses a fully compensatory strategy, where multiple lower-validity cues can sum up to outweigh a single high-validity cue. By setting validities to [0.9, 0.75, 0.7, 0.65, 0.6], we create critical trials where one option wins on the single most valid cue (favored by TTB) while the alternative option wins on several less valid cues whose sum exceeds the highest validity (favored by WADD). Agreement trials are included to anchor noise parameters.

[5] This design systematically pits Take The Best (TTB) against the Weighted Additive (WADD) theory. TTB uses a fast-and-frugal lexicographic rule, making its decision based entirely on the highest-validity cue that discriminates between the two options. WADD, conversely, uses a fully compensatory strategy, multiplying cue values by their validities and summing them up, which allows multiple lower-validity cues to collectively outweigh a single high-validity cue. By setting validities to [0.90, 0.80, 0.70, 0.60, 0.50], we create critical dissociation trials where one option wins on the single most valid discriminating cue (favored by TTB) while the alternative option wins on several less valid cues whose sum exceeds the highest validity (favored by WADD). Agreement trials are also included to anchor noise parameters.

[6] To cleanly dissociate the pure Weighted Additive (WADD) theory from the Strategy Mixture theory, we must create a set of trials where the WADD value difference between options varies systematically while the predictions of Take The Best (TTB) and Tallying are orthogonal or directly opposed. In the Strategy Mixture theory, choice probabilities are a rigid linear combination of the three strategies. By designing trials where WADD predicts a strong preference but TTB and Tallying either predict a tie or strongly oppose WADD, we can observe whether choices follow the continuous compensatory differences predicted by pure WADD or the fixed mixture patterns predicted by the competing theory.

[7] To dissociate the Strategy Mixture Theory from the pure Weighted Additive (WADD) Theory, this design leverages trials where WADD predicts either a tie, a very small difference, or a preference that directly contradicts Take The Best (TTB) and Tallying. Pure WADD choices are driven entirely by the continuous weighted sum of features. In contrast, Strategy Mixture Theory posits a probabilistic combination of WADD, Tallying, and TTB. By presenting trials where WADD predicts near-indifference but TTB or Tallying makes a strong deterministic prediction (e.g., WADD scores are nearly equal, but TTB strongly favors option A), Strategy Mixture Theory will predict a noticeable shift in choice probability toward option A, whereas pure WADD will predict near 50/50 guessing.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design dissociates the pure Weighted Additive (WADD) theory from the Contingent Heuristic theory by systematically varying the validity gap between the best and second-best discriminating cues. WADD evaluates options via a fully compensatory sum across all trials. In contrast, the Contingent Heuristic falls back to WADD only when the validity gap is smaller than a subjective threshold (delta); when the gap exceeds this threshold, it abruptly switches to a non-compensatory Take-The-Best (TTB) strategy. By including critical trials with large validity gaps where the top cue favors option A but the sum of remaining cues strongly favors option B, we can observe whether choice probabilities reflect a continuous compensatory evaluation (WADD) or a discrete threshold-based switch to a single-cue strategy (Contingent).",
  "validities": [
    0.95,
    0.85,
    0.65,
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
      0,
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
      0,
      1,
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
      1,
      0,
      0
    ],
    [
      0,
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
      1,
      1,
      0
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
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3558,
  "candidates_token_count": 508,
  "total_token_count": 5885
}
```
