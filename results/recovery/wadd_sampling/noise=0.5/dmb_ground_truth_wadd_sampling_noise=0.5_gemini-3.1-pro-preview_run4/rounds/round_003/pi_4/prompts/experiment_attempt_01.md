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
**Description:** Strategy Mixture (Take-The-Best and Tallying with Tallying Bias): Decision-makers are heterogeneous in their strategy use, probabilistically switching between strategies. On any given choice, a subject has a probability 'p_ttb' of applying a one-reason lexicographic heuristic (Take-The-Best) and a probability '1 - p_ttb' of applying an equal-weight compensatory heuristic (Tallying). Empirical data suggests that people generally favor Tallying over Take-The-Best in these environments, so the probability of using Take-The-Best is bounded between 10% and 50%, ensuring the mixture slightly favors Tallying to better match observed aggregate choice probabilities.

**Parameters:**
- p_ttb: [0.1, 0.5]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # Tallying Prediction
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_tally_core
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
**Description:** Sequential Evidence Accumulation: Decision-makers inspect cues sequentially in order of validity, accumulating evidence for the favored option. The evidence contributed by each cue is its validity scaled by a non-linear parameter gamma. Search terminates when the absolute evidence difference reaches a threshold theta, or when all cues are exhausted. A choice is then made based on the accumulated evidence with softmax noise. This unified mechanism smoothly interpolates between Take-The-Best (low threshold), Tallying (high threshold, gamma=0), and Weighted Additive (high threshold, gamma>0).

**Parameters:**
- theta: [0.0, 3.0]
- gamma: [0.0, 1.0]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities
    v = np.power(val, gamma)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += v[j] * diff
            if abs(E) >= theta:
                break
                
    # E > 0 means A is favored; E < 0 means B is favored
    scores = np.array([E, 0.0])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e_vals = np.exp(z)
    p = e_vals / np.sum(e_vals)
    
    # Add lapse rate
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
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the cues have strictly descending validities. The crucial trials are those where the most valid discriminating cue favors one option (which TTB will choose), but a greater number of less-valid cues favor the other option (which Tallying will choose). We include cases where the highest-validity cue decides the choice, as well as cases where the highest-validity cue is tied and the second highest decides for TTB, but the remaining cues overpower it in Tallying. We also include trials where Tallying predicts a tie (equal number of wins) while TTB makes a deterministic prediction.

[1] This design uses 5 features with strictly descending validities to cleanly separate Tallying from Take The Best (TTB). The trials are constructed such that the highest-validity discriminating cue frequently points to one option (which TTB will select), while a numerical majority of lower-validity cues point to the alternative (which Tallying will select). We also include trials where Tallying predicts a tie (equal number of strict wins) but TTB makes a deterministic prediction based on the most valid cue. By spanning different magnitudes of tally differences (e.g., 4 vs 1, 3 vs 2) against the TTB predictions, we can quantitatively measure the degree to which subjects aggregate features versus stopping at the best cue.

[2] To quantitatively dissociate the Non-linear Weighted Additive (WADD) strategy from Tallying, this design uses 5 features with a broad spread of validities. Tallying relies purely on the count of strict feature-wise wins, ignoring both the magnitude of the validities and their non-linear scaling. In contrast, WADD computes an overall value by summing exponentiated validities. The trial pairs are constructed to pit the number of winning features against the magnitude of the validities. We include trials where Tallying predicts a clear win for one option (e.g., winning on 3 or 4 lower-validity features) while WADD strongly prefers the other (winning on 1 or 2 high-validity features, especially when gamma > 1). We also include trials where Tallying predicts a perfect tie (equal number of winning features), forcing it to guess, whereas WADD makes a deterministic prediction based on the specific validities of the winning features.

[3] This design dissociates Tallying from Non-linear Weighted Additive (WADD) models by varying the distribution of feature validities against the sheer count of winning features. Tallying purely counts the number of strictly winning features (ignoring validity magnitudes entirely), whereas WADD sums exponentiated validities. By using a steep drop-off in validities (e.g., two highly valid experts and three lower-validity experts), we can construct trials where Tallying predicts a clear win for the option with more winning features, but WADD predicts the opposite because the fewer features are highly valid. We also include trials where Tallying predicts a perfect tie (equal number of winning features) forcing it to guess, while WADD makes a deterministic prediction. Finally, trials where WADD's prediction flips depending on its non-linearity parameter (gamma) are included to further separate the models.

[4] This design quantitatively dissociates the Non-linear Weighted Additive (WADD) strategy from a Strategy Mixture of Take-The-Best (TTB) and Tallying. By using four features with a specific validity profile (one very high validity, two moderately high validities, and one low validity), we can create trials where the mixture model and WADD make opposite predictions. For example, when Option A has the two moderately high validities and Option B has the highest and lowest validities, Tallying predicts a tie, so the TTB+Tallying mixture model defaults to TTB's prediction (Option B). However, WADD calculates that the sum of the two moderately high validities exceeds the sum of the highest and lowest validities, thus firmly predicting Option A. Other trials vary the number of winning features to test WADD's non-linear scaling against the fixed linear combination of the mixture model.

[5] This design isolates the Strategy Mixture (Take-The-Best + Tallying) from the Non-linear Weighted Additive (WADD) model by exploiting how they resolve trade-offs between cue rank, cue validity magnitude, and the number of winning cues. The Mixture model heavily relies on Tallying (which simply counts winning features without weighting them by validity) and breaks ties using TTB (which strictly looks at the highest-ranked cue). By contrast, WADD computes a compensatory sum of non-linearly scaled validities. We use a validity profile with one very high validity (0.95), two closely following high validities (0.92, 0.88), and two low validities (0.60, 0.55). In crucial trials (e.g., A wins on cues 1 and 4; B wins on cues 2 and 3), Tallying sees a tie, so the Mixture model defaults to TTB's choice (Option A). However, WADD calculates that the sum of the two moderately high validities for B strictly exceeds the sum of the highest and a low validity for A, consistently predicting Option B regardless of its non-linearity parameter (gamma). Additional trials pit a Tallying advantage against a WADD magnitude advantage to further decouple the predictions across the parameter space.

[6] This design quantitatively dissociates the Sequential Evidence Accumulation (SEA) model from the Strategy Mixture (Take-The-Best + Tallying) model by exploiting their different tie-breaking and stopping rules. We use 6 features with a specific validity profile: one very high validity (0.95), two high validities (0.90, 0.85), and three low validities (0.55, 0.50, 0.50). In Trials 1 and 2, Tallying sees a perfect tie (3 vs 3), forcing the Mixture model to default entirely to TTB, which strictly follows the 0.95 cue. However, SEA (with a high threshold and gamma near 1) integrates all evidence and correctly calculates that the sum of the two high validities plus one low validity (0.90+0.85+0.55=2.30) outweighs the highest validity plus two low validities (0.95+0.50+0.50=1.95), leading to the opposite prediction. In Trials 3 and 4, one option has a massive Tallying advantage (5 vs 1) but loses on the most valid cue. The Mixture model overwhelmingly predicts the Tallying winner. In contrast, SEA with a low threshold (e.g., theta = 0.8) will terminate search immediately after the first cue, predicting the TTB winner and ignoring the remaining cues entirely.

[7] This design quantitatively dissociates the Strategy Mixture model (Take-The-Best + Tallying) from the Sequential Evidence Accumulation (SEA) model. We use 5 features with linearly spaced validities (0.9, 0.8, 0.7, 0.6, 0.5). The key dissociation occurs in trials where Tallying predicts a perfect tie (e.g., Option A wins on cues 1 and 4; Option B wins on cues 2 and 3). In the Strategy Mixture model, the tie in Tallying means the decision defaults entirely to TTB, which strictly selects Option A based on the highest validity cue. However, for the SEA model, the evidence sum for Option A (0.9 + 0.6) exactly equals Option B (0.8 + 0.7). Thus, SEA predicts a perfect tie (if the threshold is high enough to integrate all cues), regardless of its non-linearity parameter gamma. Additional trials pit a Tallying advantage (e.g., 4 wins to 1) against a TTB advantage to test SEA's early stopping rule (low threshold) against the Mixture's probabilistic weighting.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Strategy Mixture model (Take-The-Best + Tallying) from Sequential Evidence Accumulation (SEA), we exploit how the two models handle ties in Tallying and conflicting cue magnitudes. In Strategy Mixture, a Tallying tie causes the model to default entirely to the TTB prediction (scaled by p_ttb), yielding a moderate but robust preference for the option favored by the highest validity cue. In Trial 1 and 2, Option A wins the highest validity cue and a low validity cue, while Option B wins two very high validity cues. Tallying sees a perfect tie (2 vs 2), so the Mixture model strictly predicts Option A. However, SEA (with a high threshold to avoid early stopping) integrates all evidence and calculates that B's cues significantly outweigh A's, predicting Option B. If SEA uses a low threshold, it mimics TTB and strongly predicts A (unlike the Mixture's moderate prediction). Further trials pit a Tallying advantage against a TTB advantage to constrain the parameter space of both models.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
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
      1,
      0,
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
      0,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4324,
  "candidates_token_count": 435,
  "total_token_count": 9212
}
```
