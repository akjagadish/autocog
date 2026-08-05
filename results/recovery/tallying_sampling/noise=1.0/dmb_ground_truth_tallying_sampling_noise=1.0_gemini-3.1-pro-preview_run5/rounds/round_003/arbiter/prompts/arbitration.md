# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
When faced with multi-attribute choices without correctness feedback and where no single option overwhelmingly dominates, subjects may lack the motivation or cognitive resources to systematically apply complex heuristics like Take-The-Best or Tallying. Instead, their behavior is best described by a Random Guessing baseline. In this model, subjects simply guess between the two options on every trial, potentially exhibiting a slight idiosyncratic bias toward one option over the other, but otherwise ignoring the feature validities and values entirely.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Pure random guessing baseline with an idiosyncratic side bias.
    # The stimulus and history are ignored.
    bias = float(parameters["bias"])
    
    # Return the probabilities for choosing Option A (index 0) and Option B (index 1)
    return np.array([bias, 1.0 - bias])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_5
First-Feature Heuristic (Left-to-Right Reading Bias) with High Lapse: Subjects minimize cognitive effort by inspecting only the first (leftmost) feature in the array. However, because this task is demanding and feedback is absent, subjects rely heavily on random guessing on the vast majority of trials, only occasionally applying the first-feature rule. This results in behavior that is near-random but retains a faint trace of the leftmost feature's influence.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"First-Feature Heuristic expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a_first = stim[0, 0]
    b_first = stim[1, 0]
    
    epsilon = float(parameters["epsilon"])
    
    if a_first > b_first:
        p_core = np.array([1.0, 0.0])
    elif b_first > a_first:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Blend deterministic first-feature rule with a random lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=3):** [0.9, 0.7, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0]  B=[0, 1, 1]
  trial 2: A=[1, 1, 0]  B=[0, 0, 1]
  trial 3: A=[0, 1, 1]  B=[1, 0, 0]
  trial 4: A=[0, 0, 1]  B=[1, 1, 0]
  trial 5: A=[1, 1, 1]  B=[1, 0, 0]
  trial 6: A=[0, 0, 0]  B=[0, 1, 1]

**Rationale:** To dissociate the Random Guessing baseline from the First-Feature Heuristic (with high lapse), we must manipulate the value of the first (leftmost) feature while varying the remaining features. Random Guessing predicts a constant choice probability for Option A across all trials, driven solely by an idiosyncratic side bias, regardless of any feature values. The First-Feature Heuristic predicts that choice probability for Option A will systematically increase when Option A has a 1 and Option B has a 0 on the first feature, decrease when the reverse is true, and drop exactly to 50% when the first features are tied. By including trials where the first feature strongly favors A, strongly favors B, or is tied, we can test whether the choice probabilities show this step-like function or remain perfectly flat.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the probability of choosing Option A between trials where Option A is superior on the first feature and trials where Option B is superior on the first feature. Under the Random Guessing baseline, subjects ignore the features entirely, so this difference should be zero on average. Under the First-Feature Heuristic, even with a high lapse rate, subjects will choose Option A slightly more often when it is favored by the first feature, resulting in a strictly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    b_first = data['option_b_ratings'].apply(lambda x: x[0])
    
    a_gt_b = (a_first > b_first)
    a_lt_b = (a_first < b_first)
    
    chose_a = (data['response'] == 0)
    
    if a_gt_b.sum() == 0 or a_lt_b.sum() == 0:
        return 0.0
        
    p_a_gt = chose_a[a_gt_b].mean()
    p_a_lt = chose_a[a_lt_b].mean()
    
    return float(p_a_gt - p_a_lt)

### RESULTS
- Predicted under pi_4 (simulated): -0.0050 (var=0.0095)
- Predicted under pi_5 (simulated): 0.0856 (var=0.0144)
- Observed on real data: 0.0050 (var=0.0121)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 8: A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 1]  B=[0, 1, 1, 0]

**Rationale:** To cleanly dissociate the First-Feature Heuristic (with high lapse) from a Random Guessing baseline, we must manipulate the value of the first (leftmost) feature while varying the other features. The First-Feature Heuristic predicts a higher probability of choosing Option A when Option A has a 1 and Option B has a 0 on the first feature, and a lower probability when the reverse is true. When the first feature is tied, it predicts equal preference (50/50). In contrast, the Random Guessing model predicts a completely flat choice probability across all trials, driven only by an idiosyncratic side bias. By including trials where the first feature strongly favors A, strongly favors B, or is tied, we can detect whether the choice probabilities systematically shift with the first feature or remain invariant.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric isolates the effect of the first feature by comparing the probability of choosing Option A when the first feature uniquely favors A versus when it uniquely favors B. Under the First-Feature Heuristic, this difference will be positive (even with high lapse rates, a faint trace remains). Under the Random Guessing model, this difference will be zero on average, as the first feature is ignored.

Source:
def metric(data: pd.DataFrame) -> float:
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    b_first = data['option_b_ratings'].apply(lambda x: x[0])
    choose_a = 1 - data['response']
    
    mask_a_better = (a_first == 1) & (b_first == 0)
    mask_b_better = (a_first == 0) & (b_first == 1)
    
    p_a_given_a_better = choose_a[mask_a_better].mean() if mask_a_better.sum() > 0 else 0.5
    p_a_given_b_better = choose_a[mask_b_better].mean() if mask_b_better.sum() > 0 else 0.5
    
    return float(p_a_given_a_better - p_a_given_b_better)

### RESULTS
- Predicted under pi_4 (simulated): -0.0037 (var=0.0165)
- Predicted under pi_5 (simulated): 0.1474 (var=0.0220)
- Observed on real data: -0.0341 (var=0.0154)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Calculate the difference between features
    diff = a_mat - b_mat
    
    # Find the index of the first discriminating cue (first non-zero difference)
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    
    ttb_preds = np.zeros(len(diff))
    for i in range(len(diff)):
        idx = first_diff_idx[i]
        if diff[i, idx] > 0:
            ttb_preds[i] = 0
        elif diff[i, idx] < 0:
            ttb_preds[i] = 1
        else:
            ttb_preds[i] = 0.5
            
    # Return the proportion of choices that align with the Take-The-Best prediction
    return float(np.mean(resp == ttb_preds))
```

**Observed (real) value:** 0.4883 (var=0.0032)
**Predicted under pi_4:** 0.4985 (var=0.0023)
**Predicted under pi_5:** 0.5231 (var=0.0025)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    valid_trials = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        if ttb_pred != -1:
            if row['response'] == ttb_pred:
                matches += 1
            valid_trials += 1
    return matches / valid_trials if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.4963 (var=0.0022)
**Predicted under pi_4:** 0.4925 (var=0.0018)
**Predicted under pi_5:** 0.5373 (var=0.0026)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A has 3 low-validity cues, B has 2 high-validity cues
    is_trial_1 = (a_str == '00111') & (b_str == '11000')
    # Trial 2: A has 2 high-validity cues, B has 3 low-validity cues
    is_trial_2 = (a_str == '11000') & (b_str == '00111')
    
    mask = is_trial_1 | is_trial_2
    if not mask.any():
        return 0.5
        
    # Tallying prefers the option with 3 positive features
    tally_correct_1 = is_trial_1 & (data['response'] == 0)
    tally_correct_2 = is_trial_2 & (data['response'] == 1)
    
    return float((tally_correct_1 | tally_correct_2).sum() / mask.sum())
```

**Observed (real) value:** 0.4833 (var=0.0088)
**Predicted under pi_4:** 0.4767 (var=0.0081)
**Predicted under pi_5:** 0.4342 (var=0.0079)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1 directly pits the two most valid features (Option A) 
    # against the three least valid features (Option B).
    # Tallying strictly prefers Option B (tally 3 vs 2).
    # WADD, on average, prefers Option A because the combined validity 
    # of the top 2 features outweighs the bottom 3.
    t1_mask = data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)
    if not t1_mask.any():
        return 0.5
    
    # Return the proportion of times Option A was chosen
    return float((data.loc[t1_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4500 (var=0.0206)
**Predicted under pi_4:** 0.4750 (var=0.0231)
**Predicted under pi_5:** 0.5317 (var=0.0250)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the sum of features for options A and B
    sum_a = data['option_a_ratings'].apply(lambda x: sum(x))
    sum_b = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Identify extreme trials where one option completely dominates the other
    diff = sum_a - sum_b
    extreme_mask = diff.abs() == 5
    
    if extreme_mask.sum() == 0:
        return 0.5
        
    # Determine the 'correct' choice (0 for A, 1 for B)
    correct_responses = (diff[extreme_mask] < 0).astype(int)
    actual_responses = data.loc[extreme_mask, 'response']
    
    # Calculate the proportion of choices that align with the dominating option
    return float((actual_responses == correct_responses).mean())
```

**Observed (real) value:** 0.5050 (var=0.0055)
**Predicted under pi_4:** 0.4708 (var=0.0099)
**Predicted under pi_5:** 0.5392 (var=0.0116)

### Experiment 8
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.9, 0.8, 0.6, 0.5])
    
    # Vectorized computation of WADD scores for Option A and B
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    score_A = A.dot(validities)
    score_B = B.dot(validities)
    diff = score_A - score_B
    
    # Filter out trials where options have identical WADD scores
    mask = np.abs(diff) > 1e-5
    if not np.any(mask):
        return 0.5
        
    # Predicted response: 0 (Option A) if score_A > score_B, else 1 (Option B)
    predicted = (diff[mask] < 0).astype(int)
    actual = data['response'].values[mask]
    
    # Return the proportion of choices consistent with the WADD rule
    return float(np.mean(predicted == actual))
```

**Observed (real) value:** 0.5015 (var=0.0036)
**Predicted under pi_4:** 0.5078 (var=0.0034)
**Predicted under pi_5:** 0.5370 (var=0.0048)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 8 experiments, the observed data overwhelmingly supports Theory 1 (pi_4), which posits that subjects are essentially engaging in random guessing with an idiosyncratic side bias. Theory 2 (pi_5), which suggested a faint first-feature heuristic, consistently overpredicted the influence of the first feature. The observed metric values in almost all experiments hovered very close to what would be expected by pure chance (e.g., ~0.5 for accuracy metrics, ~0 for difference metrics), matching Theory 1's predictions almost perfectly.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Since pure random guessing (Theory 1) explains the data so well, Theory 2 should be completely replaced. Propose a new theory that tests for a different, very faint signal amidst the noise\u2014such as a 'Tallying' heuristic (simply counting the number of positive features) but with an extremely high lapse rate (e.g., >95% random guessing). This will test if there is any residual sensitivity to the overall number of positive attributes that pure random guessing might be missing, without overpredicting the effect of any single feature."
}
```

## Usage

```json
{
  "prompt_token_count": 7103,
  "candidates_token_count": 262,
  "total_token_count": 8189
}
```
