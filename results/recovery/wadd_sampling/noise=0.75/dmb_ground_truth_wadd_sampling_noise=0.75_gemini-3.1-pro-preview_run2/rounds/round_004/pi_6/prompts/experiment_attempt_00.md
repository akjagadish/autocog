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
**Description:** Rank-Weighted Additive Model: Decision-makers assign subjective weights to cues based on their ordinal rank of validity rather than transforming raw validities via log-odds. The weight of each cue decays as an inverse power function of its rank (e.g., 1 / rank^k). This compensatory approach preserves monotonicity and naturally dampens the extreme predictions of pure log-odds WADD, successfully explaining intermediate choice proportions and avoiding the ordinal violations typical of non-compensatory heuristics like Take The Best.

**Parameters:**
- beta: [0.1, 3.0]
- decay_rate: [0.0, 2.0]
- epsilon: [0.0, 0.3]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    decay_rate = float(parameters["decay_rate"])
    epsilon = float(parameters["epsilon"])
    
    # Determine ranks of validities (highest validity gets rank 1)
    order = np.argsort(-val, kind="stable")
    ranks = np.zeros_like(val)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Calculate subjective weights based on rank
    w = 1.0 / (ranks ** decay_rate)
    
    # Compute weighted additive values for both options
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores = np.array([v_a, v_b])
    
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
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Strategy Mixture of TTB and WADD: Decision-makers draw from a repertoire of strategies, specifically mixing a non-compensatory heuristic (Take The Best) and a fully compensatory one (Weighted Additive). By probabilistically blending TTB (which selects the option with the highest valid cue) and WADD (which computes subjective value via log-odds weighted sums of all cues), the model naturally predicts the intermediate choice proportions observed on conflict trials where the two strategies make opposite predictions.

**Parameters:**
- beta_ttb: [0.01, 100.0]
- beta_wadd: [0.01, 100.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    p_ttb_weight = float(parameters["p_ttb"])
    
    # --- Take The Best (TTB) Prediction ---
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
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Weighted Additive (WADD) Prediction ---
    # Transform validities to log-odds to represent additive evidence
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores_wadd = np.array([v_a, v_b])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- Mixture ---
    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_wadd
    
    n_opts = p_mixed.shape[0]
    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the most valid cue often points to one option while the simple majority of cues (Tallying) points to the other. For instance, if Option A has a 1 on the highest validity cue but 0s on all others, and Option B has a 0 on the highest but 1s on the rest, TTB will confidently choose A, whereas Tallying will confidently choose B (since B wins on 3 out of 4 features). We also include trials where Tallying ties but TTB has a strict preference, further separating the models' predictions.

[1] This design uses 5 features with descending validities to systematically pit the Tallying heuristic against Take The Best (TTB). Tallying simply counts the number of features on which an option is strictly superior, completely ignoring feature validities. TTB, on the other hand, relies exclusively on the single most valid cue that discriminates between the options. The trial pairs are constructed such that the highest-validity discriminating cue frequently contradicts the simple majority of features, leading to opposite choices between the two models. Furthermore, we include trials where Tallying predicts a perfect tie (because both options win on an equal number of features) but TTB has a strict preference based on the highest valid cue, providing a robust quantitative dissociation of the two theories.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use a 4-feature design where the validities are chosen such that the log-odds of the lower-validity cues sum to more than the log-odds of the highest-validity cue. For instance, with validities [0.9, 0.8, 0.75, 0.6], the log-odds are approximately [2.20, 1.39, 1.10, 0.41]. When Option A possesses only the most valid cue (0.9) and Option B possesses the second and third most valid cues (0.8 and 0.75), TTB will deterministically choose Option A because it only considers the first discriminating cue. In contrast, WADD will choose Option B because the combined log-odds weight of the two lower cues (1.39 + 1.10 = 2.49) strictly outweighs the highest cue (2.20). By systematically varying whether the lower cues can overcome the highest discriminating cue, we can robustly separate the non-compensatory 'one-reason' logic of TTB from the compensatory, evidence-summing logic of WADD.

[3] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we employ a 5-feature design. The validities are carefully calibrated such that the log-odds of combinations of lower-validity cues exceed the log-odds of the single most valid discriminating cue. For instance, with validities [0.88, 0.78, 0.72, 0.68, 0.62], the log-odds weights are approximately [1.99, 1.26, 0.94, 0.75, 0.49]. In trial pairs where Option A possesses only the single highest discriminating cue, TTB will deterministically choose Option A. However, Option B is endowed with multiple lower-validity cues whose combined log-odds strictly outweigh Option A's single cue, causing WADD to select Option B. By systematically varying the dispersion of these compensating cues across different levels of the validity hierarchy, we can robustly separate the non-compensatory 'one-reason' logic of TTB from the compensatory, evidence-summing logic of WADD.

[4] This design quantitatively dissociates the Strategy Mixture Theory (TTB + Tallying) from the Weighted Additive (WADD) theory by exploiting the non-linear transformation of validities into log-odds. We use a 6-feature design with one very high validity cue (0.93), two high validity cues (0.90, 0.88), and three very low validity cues (0.52). Because WADD uses log-odds weights, the sum of the two high-validity cues (log-odds ~4.18) strictly outweighs the highest validity cue combined with all low-validity cues (log-odds ~2.82). In critical trials, Option A possesses the highest validity cue and several low validity cues, while Option B possesses the two high validity cues. TTB will choose A (because it has the single best cue). Tallying will also choose A (because A has more total cues). Consequently, any mixture of TTB and Tallying will strictly prefer A. However, WADD will strongly prefer B, providing a definitive qualitative and quantitative dissociation.

[5] To quantitatively dissociate the Weighted Additive (WADD) theory from the Strategy Mixture Theory (TTB + Tallying), we use a 5-feature design with three high-validity cues with very similar validities and two low-validity cues. Because WADD uses the log-odds of validities as weights, the combined weight of the second and third most valid cues strictly outweighs the single most valid cue combined with all low-validity cues. In critical trials, Option A possesses the highest validity cue alongside multiple low-validity cues, while Option B possesses only the second and third most valid cues. TTB will select Option A because it has the best cue. Tallying will also select Option A because it has more cues in total (e.g., 3 vs. 2). Consequently, any mixture of TTB and Tallying must strictly prefer Option A. In stark contrast, WADD will strongly prefer Option B, providing a definitive dissociation. Additional trials pit TTB and Tallying against each other in ways that align WADD with one or the other, ensuring robust parameter estimation across theories.

[6] To quantitatively dissociate the Strategy Mixture (TTB + WADD) from pure WADD, we exploit the variance in choice determinism across agreement and conflict trials. Pure WADD uses a single inverse temperature (beta) to scale log-odds differences into choice probabilities. If pure WADD tries to fit intermediate choice proportions on trials where TTB and WADD conflict, it must use a low beta, which incorrectly predicts low determinism on trials where both strategies strongly agree. The Strategy Mixture model, however, can simultaneously predict high determinism on agreement trials (where both TTB and WADD point the same way) and intermediate proportions on conflict trials (by mixing the two strategies via p_ttb). By varying the magnitude of the WADD log-odds difference independently of whether TTB agrees or conflicts, we create a dissociation that a single-process WADD model cannot accommodate.

[7] To quantitatively dissociate pure Weighted Additive (WADD) from the Strategy Mixture (TTB + WADD), we create a set of trials that induce a qualitative reversal in choice probabilities between the two models. Pure WADD predicts that choice probabilities are strictly monotonic with respect to the difference in log-odds weighted sums. In contrast, the Mixture model can violate this monotonicity because the Take The Best (TTB) component always selects the option with the highest discriminating cue, regardless of the overall weighted sum. We use 4 features with validities [0.90, 0.85, 0.80, 0.60]. In Trial 1, Option A has a strong WADD advantage (summing the three lower cues) while Option B wins on TTB. In Trial 2, Option A has a very weak WADD advantage but wins on TTB. Pure WADD must predict a higher probability of choosing A in Trial 1 than in Trial 2. The Mixture model, however, will likely predict a higher probability of choosing A in Trial 2 (where both WADD and TTB agree) than in Trial 1 (where TTB pulls the choice toward B). This ordinal violation provides a definitive signature to separate the theories.

[8] This design quantitatively and qualitatively dissociates the Strategy Mixture (TTB + WADD) from the Rank-Weighted Additive model by exploiting a mathematical rigidity in the Rank-Weighted model's inverse power law. In the Rank-Weighted model, the weight of cue `i` is `1 / i^k`. This strictly enforces that the drop in subjective weight from Rank 1 to Rank 2 is proportionally larger than the drop from Rank 2 to Rank 3. However, the Strategy Mixture model (via its WADD component) uses the log-odds of validities, which can violate this rank-based constraint. We set validities such that the log-odds drop from Rank 1 to Rank 2 is tiny (0.90 vs 0.88), but the drop from Rank 2 to Rank 3 is massive (0.88 vs 0.60). In Trial 1 (Cue 1 vs. Cues 2,3,4,5), WADD strongly prefers Option B because Cue 2 alone almost offsets Cue 1. In Trial 2 (Cue 2 vs. Cues 3,4,5), WADD strongly prefers Option A because the lower cues are too weak to offset Cue 2. It is mathematically impossible for the Rank-Weighted model to simultaneously prefer Option B in Trial 1 and Option A in Trial 2 for any value of `k`, providing a definitive signature of log-odds weighting over rank-based weighting.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design provides a strict qualitative dissociation between the Rank-Weighted Additive Model and the Strategy Mixture (TTB + WADD) theory. We use a 5-feature design with two extremely high validities (0.98, 0.95) and three very low validities (0.55, 0.53, 0.51). Because WADD relies on log-odds transformations, the log-odds of the top two cues (~3.89 and ~2.94) overwhelmingly dominate the bottom three cues (~0.20, ~0.12, ~0.04). In our critical trials, Option A possesses the highest-validity cue(s) while Option B possesses a greater number of low-validity cues. Consequently, both TTB (which only looks at the top discriminating cue) and WADD (which sums the massive log-odds) will strictly and strongly prefer Option A. Thus, any mixture of TTB and WADD must predict a strong preference for Option A. However, the Rank-Weighted Additive Model completely ignores the magnitude of the log-odds gap, assigning weights based solely on ordinal rank (e.g., 1, 1/2^k, 1/3^k, etc.). For low-to-moderate values of the decay parameter 'k', the sum of the ranks for Option B's multiple cues will exceed the rank weight of Option A's top cue(s), allowing the Rank-Weighted model to predict a preference for Option B. This constitutes a definitive ordinal violation that the Mixture model cannot accommodate under any parameterization.",
  "validities": [
    0.98,
    0.95,
    0.55,
    0.53,
    0.51
  ],
  "trial_a_ratings": [
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
      0,
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
    ]
  ],
  "trial_b_ratings": [
    [
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
      1,
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
  "prompt_token_count": 4688,
  "candidates_token_count": 557,
  "total_token_count": 8657
}
```
