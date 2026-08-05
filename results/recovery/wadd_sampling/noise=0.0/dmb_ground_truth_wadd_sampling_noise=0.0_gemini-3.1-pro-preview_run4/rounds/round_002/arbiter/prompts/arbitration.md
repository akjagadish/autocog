# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Decision-makers evaluate options by computing a weighted sum of their features, where the weights are a subjective power transformation of the cue validities. This Weighted Additive (WADD) model is fully compensatory, allowing the combined evidence of multiple lower-validity cues to potentially outweigh a single higher-validity cue, while still remaining sensitive to the varying diagnosticity of different features. The parameter governing the power transformation is constrained to prevent extreme exponentiation, ensuring the model remains compensatory rather than collapsing into a non-compensatory heuristic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective transformation of validities into weights
    w = val ** gamma
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Decision-makers assign importance to features based on their ordinal rank in validity rather than their exact cardinal values. This Rank-Based Weighting heuristic avoids the extreme sensitivity to numerical validity differences seen in purely compensatory models, while still acknowledging that some cues are more diagnostic than others. Feature weights are computed as a power transformation of their inverse rank (e.g., 1 / rank^gamma). By restricting gamma to lower values, the model maintains a strong compensatory nature, ensuring that multiple lower-ranked cues can outweigh a single higher-ranked cue. Combined with a lower softmax temperature upper bound, it prevents overly deterministic choices and captures the noisier human behavior in conflicting trade-offs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Rank-Based Weighting expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Determine the ordinal rank of each feature's validity (1 = highest validity)
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Weights are a power transformation of the inverse rank
    w = 1.0 / (ranks ** gamma)
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.8, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) model from Rank-Based Weighting, we use a 5-feature design where the top four validities are clustered closely together, while the fifth is much lower. In WADD, closely clustered validities result in similar weights, making the sum of multiple slightly-lower-validity features easily exceed a single higher-validity feature. In Rank-Based Weighting, weights are determined by inverse rank, which imposes a sharp, non-linear drop-off in importance regardless of the actual cardinal differences (e.g., the 1st feature is weighted twice as much as the 2nd feature, even if their validities are 0.95 and 0.90). This structural difference guarantees opposing predictions on trials where a single highest-validity feature is pitted against two slightly lower-validity features, or when comparing sums across different rank tiers.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of choices that align with the Rank-Based Weighting theory on two highly discriminative trials. In Trial 3, a single top-tier feature paired with the lowest-tier feature (A) competes against two mid-tier features (B). WADD, using cardinal validities, finds the mid-tier sum stronger and favors B; Rank-Based Weighting's steep drop-off makes the top-tier feature dominate, favoring A. In Trial 5, the combination of 3rd and 4th ranked features (A) competes against 2nd and 5th ranked features (B). WADD favors A (0.85+0.80 > 0.90+0.50), whereas Rank-Based Weighting favors B (1/2 + 1/5 > 1/3 + 1/4). By calculating the rate of 'Rank-Based choices' across these two trials, the metric will be robustly above 0.5 for Rank-Based Weighting and robustly below 0.5 for WADD, providing excellent discrimination with low between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    # Trial 3: A=[1, 0, 0, 0, 1]. WADD favors B, Rank-based favors A.
    is_t3 = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[4] == 1)
    
    # Trial 5: A=[0, 0, 1, 1, 0]. WADD favors A, Rank-based favors B.
    is_t5 = data['option_a_ratings'].apply(lambda x: x[2] == 1 and x[3] == 1)
    
    t3_data = data[is_t3]
    t5_data = data[is_t5]
    
    score = 0.0
    n = 0
    
    if len(t3_data) > 0:
        # Rank-based predicts choosing A (response == 0)
        score += (t3_data['response'] == 0).sum()
        n += len(t3_data)
        
    if len(t5_data) > 0:
        # Rank-based predicts choosing B (response == 1)
        score += (t5_data['response'] == 1).sum()
        n += len(t5_data)
        
    if n == 0:
        return 0.5
        
    return float(score / n)

### RESULTS
- Predicted under pi_3 (simulated): 0.2258 (var=0.0261)
- Predicted under pi_4 (simulated): 0.5574 (var=0.0120)
- Observed on real data: 0.4379 (var=0.0545)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.99, 0.97, 0.95, 0.93, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate Rank-Based Weighting from the Weighted Additive (WADD) model, we use a 5-feature design where the top four validities are clustered very closely together (e.g., 0.99, 0.97, 0.95, 0.93), while the fifth is much lower. In WADD, closely clustered validities result in similar cardinal weights, allowing the sum of two slightly-lower-validity features to easily exceed a single higher-validity feature. In Rank-Based Weighting, weights are determined by ordinal rank (e.g., 1/rank^gamma), which imposes a sharp, non-linear drop-off in importance regardless of the actual cardinal differences (e.g., the 1st feature is heavily favored over the 2nd feature, even though their validities are 0.99 and 0.97). This structural difference guarantees opposing predictions on trials where a single highest-validity feature is pitted against two slightly lower-validity features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject chooses the option containing the single highest-validity feature available between the two choices. In the Rank-Based Weighting model, weights drop off steeply according to rank (e.g., 1 vs 1/2 vs 1/3), making the highest-ranking feature dominant and leading to choices that align with it. In the Weighted Additive (WADD) model, the closely clustered top validities (e.g., 0.99, 0.97, 0.95) yield similar cardinal weights, allowing two slightly lower-validity features to easily outweigh a single higher-validity feature. Thus, Rank-Based Weighting will score highly on this metric, whereas WADD will score very low.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        diff = a - b
        
        # The features are ordered by validity (highest to lowest)
        # The first non-zero difference indicates which option has the highest validity feature
        idx = np.nonzero(diff)[0][0]
        
        # diff[idx] == 1 means A has it, diff[idx] == -1 means B has it
        highest_validity_is_b = 1 if diff[idx] == -1 else 0
        
        # Check if the subject's response (0 for A, 1 for B) aligns with the highest valid feature
        if row['response'] == highest_validity_is_b:
            matches.append(1.0)
        else:
            matches.append(0.0)
            
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_3 (simulated): 0.1765 (var=0.0096)
- Predicted under pi_4 (simulated): 0.4867 (var=0.0072)
- Observed on real data: 0.4083 (var=0.0157)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # The cues are ordered by validity (descending) from left to right.
    # Find the first cue where the two options differ.
    diff = a_mat - b_mat
    non_zero = diff != 0
    first_diff_idx = np.argmax(non_zero, axis=1)
    
    # Get the difference value at that first discriminating cue.
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    # TTB predicts A (0) if A > B, else B (1)
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # Calculate the proportion of responses matching the TTB prediction
    matches = (data['response'].values == ttb_pred)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2988 (var=0.0249)
**Predicted under pi_3:** 0.3183 (var=0.0105)
**Predicted under pi_4:** 0.3685 (var=0.0144)

### Experiment 4
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    # Extract options matrices
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying predictions: count wins for each option
    a_wins = (a_mat > b_mat).sum(axis=1)
    b_wins = (b_mat > a_mat).sum(axis=1)
    
    tally_preds = np.zeros(len(data)) - 1
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    # Take The Best (TTB) predictions: first discriminating cue
    ttb_preds = np.zeros(len(data)) - 1
    for i in range(len(data)):
        for j in range(a_mat.shape[1]):
            if a_mat[i, j] > b_mat[i, j]:
                ttb_preds[i] = 0
                break
            elif b_mat[i, j] > a_mat[i, j]:
                ttb_preds[i] = 1
                break
                
    # Identify trials where Tallying and TTB make strict, opposing predictions
    valid = (tally_preds != -1) & (ttb_preds != -1) & (tally_preds != ttb_preds)
    
    if not np.any(valid):
        return 0.5
        
    # Calculate the proportion of responses that align with Tallying
    responses = data['response'].values[valid]
    tally_preds_valid = tally_preds[valid]
    
    return float(np.mean(responses == tally_preds_valid))
```

**Observed (real) value:** 0.7037 (var=0.0287)
**Predicted under pi_3:** 0.8434 (var=0.0130)
**Predicted under pi_4:** 0.6711 (var=0.0160)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the options are [1, 0, 1, 0, 0] and [0, 1, 0, 1, 0]
    is_A_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    is_B_target = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    
    # Count how many times the target option [1, 0, 1, 0, 0] was chosen
    chose_target_when_A = (data.loc[is_A_target, 'response'] == 0).sum()
    chose_target_when_B = (data.loc[is_B_target, 'response'] == 1).sum()
    
    total_trials = is_A_target.sum() + is_B_target.sum()
    if total_trials == 0:
        return 0.5
        
    return float((chose_target_when_A + chose_target_when_B) / total_trials)
```

**Observed (real) value:** 0.6050 (var=0.0828)
**Predicted under pi_3:** 0.6775 (var=0.0220)
**Predicted under pi_4:** 0.6425 (var=0.0201)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the unweighted sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Extract the first (highest validity) feature for each option
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    b_first = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where one option has more positive features (higher sum)
    # but lacks the most important feature (first feature is 0).
    # This corresponds exactly to trials 1 and 3 in the design.
    mask_a_higher_but_lacks_first = (sum_a > sum_b) & (a_first == 0)
    mask_b_higher_but_lacks_first = (sum_b > sum_a) & (b_first == 0)
    
    mask = mask_a_higher_but_lacks_first | mask_b_higher_but_lacks_first
    
    if not mask.any():
        return 0.5
        
    filtered = data[mask]
    chose_a = filtered['response'] == 0
    
    a_is_higher = sum_a[mask] > sum_b[mask]
    
    # Calculate the proportion of times the subject chose the option with the higher unweighted sum
    chose_higher = (chose_a & a_is_higher) | (~chose_a & ~a_is_higher)
    
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.3713 (var=0.0726)
**Predicted under pi_3:** 0.3719 (var=0.0884)
**Predicted under pi_4:** 0.5062 (var=0.0350)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments, neither theory perfectly captures all the data, though both show distinct strengths. THEORY 2 (Rank-Based Weighting) is generally closer to the observed metrics in Experiments 1, 2, 4, and 5, demonstrating that human decision-makers often exhibit a non-linear, rank-based drop-off in feature importance rather than pure cardinal weighting. However, THEORY 1 (WADD) performs exceptionally well on Experiment 6 (predicting 0.3719 vs observed 0.3713) and slightly better on Experiment 3, suggesting that cardinal validities still play a crucial role when making trade-offs, especially when unweighted sums of features compete with the highest-validity cue. Because both theories succeed on disjoint sets of experiments, we need a new theory that bridges these mechanisms.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new 'Hybrid Cardinal-Rank Weighting' theory or a 'Strategy Selection' theory. Decision-makers might not rely purely on cardinal validities or pure ordinal ranks. Instead, they might use weights that are a function of both the cardinal validity and its ordinal rank (e.g., scaling cardinal validities by a rank-based discount factor), or they may engage in strategy selection where a mixture of compensatory WADD and non-compensatory/rank-based heuristics are used across the population. This new theory should capture the strict drop-offs seen in Exps 1, 2, and 4, while preserving the compensatory thresholding behavior that perfectly matched Exp 6."
}
```

## Usage

```json
{
  "prompt_token_count": 6632,
  "candidates_token_count": 355,
  "total_token_count": 8071
}
```
