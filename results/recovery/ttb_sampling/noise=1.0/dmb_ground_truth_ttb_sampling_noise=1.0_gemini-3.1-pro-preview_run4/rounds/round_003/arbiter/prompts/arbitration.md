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
Random Guessing: Due to task complexity, lack of trial-by-trial correctness feedback, or low motivation, subjects do not systematically evaluate the options using the provided validities or features. Instead, they make uniformly random choices on every trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Random Feature Heuristic (Minimalist): Subjects do not integrate multiple features or weight them by validity. Instead, on each trial, they randomly sample exactly one feature uniformly at random. They choose the option with the higher value on that sampled feature. If the options tie on that feature, they guess randomly. This acts as a highly noisy but boundedly rational alternative to pure guessing, predicting choices that slightly favor the option with more positive features overall. A lapse rate (epsilon) accounts for additional purely random choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    # The marginal probability of choosing A when picking one feature at random:
    # P(A) = 1/n * sum(a_i > b_i) + 0.5/n * sum(a_i == b_i)
    # For binary features, this simplifies exactly to 0.5 + 0.5 * (sum(a) - sum(b)) / n
    diff = np.sum(a) - np.sum(b)
    p_a_core = 0.5 + 0.5 * diff / n_features
    p_b_core = 1.0 - p_a_core
    
    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a_core + epsilon * 0.5
    p_b = (1.0 - epsilon) * p_b_core + epsilon * 0.5
    
    return np.array([p_a, p_b])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]
  trial 9: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates Random Guessing from the Random Feature Heuristic by systematically varying the difference in the total number of positive features between Option A and Option B. Random Guessing predicts a flat 50% choice probability across all trials, regardless of the feature values. The Random Feature Heuristic, however, predicts that choice probabilities will track the difference in the unweighted sum of features (since picking a random feature favors the option with more positive features). We use 5 features and construct trials with maximum differences (5 vs 0), moderate differences (4 vs 1, 3 vs 0, 5 vs 2), and ties (2 vs 2, 3 vs 3). If subjects are purely guessing, choice proportions will remain near 0.5 for all trial types. If they use a Random Feature Heuristic, their choice proportions will systematically diverge from 0.5 as the difference in the number of positive features increases.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric computes the proportion of trials where the subject chose the option with the strictly greater number of positive features (excluding ties). Under the Random Guessing theory, subjects ignore features entirely, so this proportion will be tightly centered around 0.5. Under the Random Feature Heuristic, subjects sample a feature and choose the option with a higher value on that feature, which statistically favors the option with more positive features overall, leading to a proportion significantly greater than 0.5. The exclusion of tie trials maximizes the expected contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    responses = data.loc[mask, 'response']
    a_more = sum_a[mask] > sum_b[mask]
    
    chose_max = ((responses == 0) & a_more) | ((responses == 1) & ~a_more)
    return float(chose_max.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.4986 (var=0.0024)
- Predicted under pi_5 (simulated): 0.6925 (var=0.0126)
- Observed on real data: 0.5006 (var=0.0047)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=7):** [0.95, 0.85, 0.75, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=9):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the Random Feature Heuristic from Random Guessing by systematically varying the difference in the total number of positive features between Option A and Option B across 7 features. Random Guessing predicts a flat 50% choice probability across all trials, regardless of the feature values. The Random Feature Heuristic, however, predicts that choice probabilities will track the difference in the unweighted sum of features. We construct a parametric gradient of trials with differences ranging from +7 to -7. If subjects are purely guessing, choice proportions will remain near 0.5 for all trial types. If they use a Random Feature Heuristic, their choice proportions will systematically and linearly diverge from 0.5 as the difference in the number of positive features increases.

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



### METRIC
Rationale:
The Random Feature Heuristic predicts that the probability of choosing Option A increases as the difference in the number of positive features between A and B increases. Random Guessing predicts a flat 50% probability regardless of the features. This metric computes the mean rate of choosing A when A has more positive features than B, minus the mean rate of choosing A when A has fewer positive features. For Random Guessing, this difference will be approximately 0. For the Random Feature Heuristic, it will be significantly positive.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff = sum_a - sum_b
    
    chose_a = 1 - data['response']
    
    pos_mean = chose_a[diff > 0].mean()
    neg_mean = chose_a[diff < 0].mean()
    
    if pd.isna(pos_mean) or pd.isna(neg_mean):
        return 0.0
        
    return float(pos_mean - neg_mean)

### RESULTS
- Predicted under pi_4 (simulated): 0.0206 (var=0.0141)
- Predicted under pi_5 (simulated): 0.2624 (var=0.0308)
- Observed on real data: 0.0112 (var=0.0080)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].apply(tuple)
    b_ratings = data['option_b_ratings'].apply(tuple)
    
    mask1 = (a_ratings == (0, 0, 0, 1, 1, 1)) & (b_ratings == (1, 1, 0, 0, 0, 0))
    mask2 = (a_ratings == (1, 1, 0, 0, 0, 0)) & (b_ratings == (0, 0, 0, 1, 1, 1))
    
    resp = []
    if mask1.any():
        resp.extend(data.loc[mask1, 'response'].tolist())
    if mask2.any():
        resp.extend((1 - data.loc[mask2, 'response']).tolist())
        
    if len(resp) == 0:
        return 0.5
    return float(np.mean(resp))
```

**Observed (real) value:** 0.4844 (var=0.0167)
**Predicted under pi_4:** 0.4867 (var=0.0202)
**Predicted under pi_5:** 0.4778 (var=0.0357)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    is_t1 = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    is_t2 = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    wadd_consistent = (is_t1 & (data['response'] == 0)) | (is_t2 & (data['response'] == 1))
    mask = is_t1 | is_t2
    
    if mask.sum() == 0:
        return 0.5
    return float(wadd_consistent.sum() / mask.sum())
```

**Observed (real) value:** 0.4800 (var=0.0103)
**Predicted under pi_4:** 0.5150 (var=0.0123)
**Predicted under pi_5:** 0.4625 (var=0.0074)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            matches.append(1.0 if ttb_pred == resp else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.4954 (var=0.0018)
**Predicted under pi_4:** 0.5096 (var=0.0020)
**Predicted under pi_5:** 0.4269 (var=0.0036)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(np.sum)
    b_sums = data['option_b_ratings'].apply(np.sum)
    wadd_choice = (b_sums > a_sums).astype(int)
    return float((data['response'] == wadd_choice).mean())
```

**Observed (real) value:** 0.5046 (var=0.0028)
**Predicted under pi_4:** 0.5193 (var=0.0019)
**Predicted under pi_5:** 0.5673 (var=0.0040)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    keys = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + '_' + data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    means = data.groupby(keys)['response'].mean()
    return float(np.var(means))
```

**Observed (real) value:** 0.0015 (var=0.0001)
**Predicted under pi_4:** 0.0003 (var=0.0001)
**Predicted under pi_5:** 0.0141 (var=0.0005)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.8, 0.75, 0.6, 0.5])
    
    matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a_ratings = np.array(row['option_a_ratings'])
        b_ratings = np.array(row['option_b_ratings'])
        
        score_a = np.dot(a_ratings, validities)
        score_b = np.dot(b_ratings, validities)
        
        if abs(score_a - score_b) > 1e-5:
            predicted_response = 0 if score_a > score_b else 1
            if row['response'] == predicted_response:
                matches += 1
            total += 1
            
    return matches / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5040 (var=0.0025)
**Predicted under pi_4:** 0.5040 (var=0.0025)
**Predicted under pi_5:** 0.5870 (var=0.0039)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 8 experiments, the observed metrics consistently sit exactly at the baseline expected by pure Random Guessing (Theory 1). For example, in Experiment 1, the proportion of choosing the option with more positive features is 0.5006. In Experiment 2, the difference in choice rate based on feature sum differences is 0.0112 (effectively zero). Experiment 8 shows a 0.5040 match with WADD. Theory 2 (Random Feature Heuristic) consistently predicts systematic deviations from 50% that never materialize in the real data. Therefore, Theory 1 is extremely accurate, and Theory 2 is fundamentally incorrect because subjects are not using the features at all to guide their decisions.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Random Feature Heuristic with a 'Spatial Bias / Option Preference' theory. Since the data overwhelmingly shows that subjects ignore the features entirely, any feature-based heuristic will fail. Instead, propose a theory where subjects ignore features but have an idiosyncratic, subject-level bias towards choosing Option A or Option B (e.g., a left/right spatial bias). This would explain the lack of feature-based variance while allowing for individual differences in baseline choice rates."
}
```

## Usage

```json
{
  "prompt_token_count": 7229,
  "candidates_token_count": 286,
  "total_token_count": 8365
}
```
