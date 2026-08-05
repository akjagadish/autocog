# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Strategy Selection (WADD and TTB Mixture) with Compensatory Bias

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Selection expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    
    # Strategy 1: Compensatory WADD
    w_wadd = val ** gamma
    score_a = np.sum(w_wadd * a)
    score_b = np.sum(w_wadd * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # Strategy 2: Non-compensatory Take-The-Best (TTB)
    order = np.argsort(-val, kind="stable")
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Mixture of strategies
    p_core = alpha * p_wadd + (1.0 - alpha) * p_ttb
    
    epsilon = float(parameters["epsilon"])
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


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=10):** [0.99, 0.95, 0.65, 0.64, 0.63, 0.62, 0.61, 0.6, 0.59, 0.58]

**Trial pairs (n=4):**
  trial 1: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Rationale:** To strongly dissociate Strategy Selection (WADD + TTB) from Rank-Based Weighting, we use a 10-feature design with a sharp cardinal validity drop-off after the second feature, but a gradual decline thereafter. We pit Option B, which has the 2nd most valid feature and a few of the lowest valid features, against Option A, which has an equal number of middle-ranked features (e.g., 3rd, 4th, and 5th). Because Rank-Based Weighting relies on inverse ordinal rank (1/rank^gamma), it heavily discounts the 8th-10th features, making the rank-weight sum of the 3rd, 4th, and 5th features outweigh the 2nd plus the 9th and 10th. Thus, Rank-Based Weighting strictly prefers Option A. In contrast, the Advocated Theory's TTB component strictly prefers Option B (because it possesses the 2nd ranked feature, while A's highest is the 3rd). Furthermore, the WADD component also strongly prefers Option B because the cardinal validity of the 2nd feature (0.95) is so much higher than the 3rd (0.65) that it easily compensates for B's weaker trailing features. Consequently, both components of the Advocated Theory agree on Option B, leading to a perfect quantitative dissociation where the two theories predict choices in opposite directions across all valid parameter regimes.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pits Option B (which possesses the 2nd most valid feature but weaker trailing features) against Option A (which possesses several middle-ranked features). According to the design rationale, the Advocated Theory (Strategy Selection with WADD and TTB) strongly prefers Option B, whereas the Competing Theory (Rank-Based Weighting) strictly prefers Option A (due to the inverse-rank weighting heavily discounting the trailing features while making the middle-rank features sum to a larger value). Therefore, the overall proportion of Option B choices (response == 1) provides a direct, highly discriminative measure between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.7454 (var=0.0130)
- Predicted under pi_4 (simulated): 0.4475 (var=0.0036)
- Observed on real data: 0.5012 (var=0.0602)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=8):** [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=3):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates Rank-Based Weighting from Strategy Selection (WADD + TTB Mixture) by exploiting how each theory scales feature importance. We use an 8-feature environment with a smooth linear decline in validities. In the critical trials, Option A possesses a few of the highest-validity features, while Option B possesses a larger number of lower-validity features. Because Rank-Based Weighting uses an inverse power function of ordinal rank (1/rank^gamma), the sum of the weights for the top few features easily exceeds the sum for many lower-ranked features, leading to a strong preference for Option A. In contrast, the Strategy Selection model's WADD component uses the cardinal validities directly, allowing the sheer number of Option B's features (each with a validity >= 0.50) to overpower Option A's few features. Meanwhile, the TTB component in Strategy Selection may favor Option A, but the strong compensatory bias of WADD pulls the mixture toward Option B or creates a massive quantitative divergence in predicted choice probabilities compared to the near-deterministic Rank-Based predictions.

**Computed schedule:** 3 unique pairs × 32 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 2, Option A possesses the highest-validity feature, triggering the Take-The-Best (TTB) component of the Strategy Selection theory to deterministically favor A. In Trial 3, Option B possesses the highest-validity feature, causing TTB to completely flip and favor B. This non-compensatory behavior leads to a large drop in the probability of choosing A from Trial 2 to Trial 3 for the Strategy Selection model. In contrast, Rank-Based Weighting is purely compensatory and values features based on a smoothed rank transformation. The difference in the total weighted sums between Trial 2 and Trial 3 is much smaller in the Rank-Based model, resulting in a significantly smaller change in choice probabilities. By calculating the difference in the proportion of times Option A is chosen between Trial 2 and Trial 3, we isolate this sharp non-compensatory shift, robustly discriminating the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Trial 2: Option A has features 1 and 3 (indices 0 and 2)
    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 0 and x[2] == 1)
    
    # Trial 3: Option A has features 2 and 3 (indices 1 and 2)
    t3_mask = data['option_a_ratings'].apply(lambda x: x[0] == 0 and x[1] == 1 and x[2] == 1)
    
    t2_data = data[t2_mask]
    t3_data = data[t3_mask]
    
    if len(t2_data) == 0 or len(t3_data) == 0:
        return 0.0
        
    p_a_t2 = np.mean(t2_data['response'] == 0)
    p_a_t3 = np.mean(t3_data['response'] == 0)
    
    return float(p_a_t2 - p_a_t3)

### RESULTS
- Predicted under pi_5 (simulated): 0.2006 (var=0.0213)
- Predicted under pi_4 (simulated): 0.0737 (var=0.0203)
- Observed on real data: -0.0363 (var=0.0444)

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
**Predicted under pi_5:** 0.3904 (var=0.0104)
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
**Predicted under pi_5:** 0.6795 (var=0.0180)
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
**Predicted under pi_5:** 0.6867 (var=0.0154)
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
**Predicted under pi_5:** 0.3281 (var=0.0599)
**Predicted under pi_4:** 0.5062 (var=0.0350)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.4379 (var=0.0545)
**Predicted under pi_5:** 0.3811 (var=0.0209)
**Predicted under pi_4:** 0.5574 (var=0.0120)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.4083 (var=0.0157)
**Predicted under pi_5:** 0.3162 (var=0.0160)
**Predicted under pi_4:** 0.4867 (var=0.0072)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 8 experiments, neither Strategy Selection (pi_5) nor Rank-Based Weighting (pi_4) consistently captures human behavior. Strategy Selection overpredicts the preference for options with a single high-validity feature (Exp 1 and 2), while Rank-Based Weighting struggles in experiments where unweighted sums or specific compensatory trade-offs dominate (Exp 6 and 7). The observed data often sits between the extreme compensatory nature of WADD/Rank-Based models and the non-compensatory nature of TTB. This suggests that subjects might be using a simplified compensatory strategy that ignores low-quality information rather than weighting all features or relying on just one.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Validity Threshold Tallying' theory. In this theory, decision-makers simplify the choice environment by setting a threshold on cue validity. Cues with a validity above the threshold are given equal weight (unit weight / tallying), while cues below the threshold are ignored completely. This explains why small differences in high validities don't lead to strict preferences (unlike WADD) and why low-validity cues fail to compensate for fewer high-validity cues (unlike Rank-Based Weighting). The model should fit a threshold parameter and a softmax temperature to account for choice noise."
}
```

## Usage

```json
{
  "prompt_token_count": 7711,
  "candidates_token_count": 301,
  "total_token_count": 9007
}
```
