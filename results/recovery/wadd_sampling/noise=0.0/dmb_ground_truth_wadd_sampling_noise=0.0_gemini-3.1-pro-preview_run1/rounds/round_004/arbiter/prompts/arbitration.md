# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Raw Validity-Weighted Additive Strategy: Decision-makers do not rely on fast-and-frugal heuristics or non-linear rank transformations. Instead, they compute a simple compensatory weighted sum of the features, where the weights are directly proportional to the raw validities provided in the instructions. This naturally explains the ~0.50 split in conflict trials because the sum of a few high-validity cues roughly equals the sum of several lower-validity cues, bridging the gap between heuristic and rank-based models. A constrained temperature parameter ensures that small differences in these sums translate to intermediate choice probabilities rather than deterministic outcomes.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute simple weighted sums using raw validities as weights
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([score_a, score_b])
    
    # Convert scores to probabilities using softmax
    z = beta * scores
    z = z - np.max(z)
    p_core = np.exp(z)
    p_core = p_core / np.sum(p_core)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Non-linear Rank-Weighted Additive Strategy: Decision-makers evaluate options by integrating all available features, but weight them by a non-linear transformation of their ordinal rank of importance. By scaling the ranks by a power parameter gamma, the decision-maker can smoothly interpolate between pure Tallying (gamma=0, where all features are weighted equally) and steeper rank-based weighting (gamma>1). This flexibility allows the model to capture the observed ~0.50 split in conflict trials by balancing the sums of top versus bottom feature ranks appropriately.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # rankdata assigns rank 1 to the smallest value. 
    # Thus, higher validity gets a proportionally higher integer rank (weight).
    gamma = float(parameters["gamma"])
    weights = rankdata(val) ** gamma
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** This design dissociates the Raw Validity-Weighted Additive Strategy (WADD) from the Non-linear Rank-Weighted Additive Strategy by exploiting the disconnect between raw validity differences and ordinal rank differences. We use a set of validities with a large gap between the top two cues (0.95, 0.90) and the bottom three cues (0.60, 0.55, 0.50). WADD's compensatory sums rely strictly on these raw values, making it highly sensitive to the large gap. In contrast, the Rank-Weighted model enforces a rigid, evenly-spaced ordinal structure (ranks 5, 4, 3, 2, 1) scaled by a parameter gamma. Trial pairs are constructed such that WADD's predictions (e.g., strongly preferring an option with multiple lower-ranked cues whose raw validities sum to a higher value, or predicting a tie when raw sums match) are contradicted by the Rank-Weighted model, which can amplify the highest rank (gamma > 1) to dominate the choice or compress ranks (gamma < 1) to approximate tallying.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric contrasts the choice probabilities of the target option in Trial 5 versus Trial 2. Under the WADD (Advocated) theory, Trial 2 is a perfect tie in raw validity sums (A=1.50, B=1.50), resulting in a choice probability of exactly 0.5, while Trial 5 slightly favors A (A=1.50, B=1.40). Thus, WADD predicts a small, stable positive difference (~0.1). In contrast, the Rank-Weighted (Competing) theory applies an amplifying non-linear transformation to the ranks. For Trial 5, it heavily favors A for any gamma > 0 (as A has ranks 5 and 2 vs B's 4 and 1). For Trial 2, it actually penalizes A for gamma > 1 (as B has rank 5). Consequently, the Rank-Weighted model predicts a drastically larger difference in choice probabilities, creating a substantial and robust gap between the two theories' predictions.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # T5: A=[1, 0, 0, 1, 0], B=[0, 1, 0, 0, 1]
    t5_a_mask = (a_tuples == (1, 0, 0, 1, 0)) & (b_tuples == (0, 1, 0, 0, 1))
    t5_b_mask = (b_tuples == (1, 0, 0, 1, 0)) & (a_tuples == (0, 1, 0, 0, 1))
    
    t5_a_chosen = (t5_a_mask & (data['response'] == 0)).sum() + (t5_b_mask & (data['response'] == 1)).sum()
    t5_total = t5_a_mask.sum() + t5_b_mask.sum()
    p_t5 = t5_a_chosen / t5_total if t5_total > 0 else 0.5
    
    # T2: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 0]
    t2_a_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 0))
    t2_b_mask = (b_tuples == (0, 1, 1, 0, 0)) & (a_tuples == (1, 0, 0, 1, 0))
    
    t2_a_chosen = (t2_a_mask & (data['response'] == 0)).sum() + (t2_b_mask & (data['response'] == 1)).sum()
    t2_total = t2_a_mask.sum() + t2_b_mask.sum()
    p_t2 = t2_a_chosen / t2_total if t2_total > 0 else 0.5
    
    return float(p_t5 - p_t2)


### RESULTS
- Predicted under pi_6 (simulated): 0.1084 (var=0.0177)
- Predicted under pi_4 (simulated): 0.4653 (var=0.1143)
- Observed on real data: 0.0926 (var=0.2806)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.95, 0.6, 0.58, 0.56, 0.54, 0.52]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 6: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]

**Rationale:** This design dissociates the Non-linear Rank-Weighted Additive Strategy from the Raw Validity-Weighted Additive Strategy by exploiting the divergence between raw validities and their ordinal ranks. We use a set of validities with one dominant cue (0.95) and five tightly clustered, much weaker cues (0.60, 0.58, 0.56, 0.54, 0.52). The Raw Validity model evaluates options strictly by summing these raw values, meaning the dominant cue is easily outweighed by a combination of three or more weaker cues. In contrast, the Rank-Weighted model enforces a rigid ordinal spacing (ranks 6, 5, 4, 3, 2, 1) scaled by a parameter gamma. Depending on gamma, the Rank-Weighted model can either heavily amplify the top rank (causing it to dominate despite being outnumbered) or compress the ranks (approximating Tallying), leading to clear opposite predictions from the Raw Validity model across various trial types.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 4, Option A has features 2, 3, and 4 present, while Option B has features 1, 5, and 6. Under the Raw Validity model, the sum for A is 0.60+0.58+0.56 = 1.74, and for B is 0.95+0.54+0.52 = 2.01. Thus, the Raw Validity model strongly favors B. However, under the Rank-Weighted model, the ranks for A are 5+4+3=12, and for B are 6+2+1=9. For any gamma >= 0, the Rank model predicts a preference for A (or a tie if gamma=0). Measuring the proportion of times A is chosen in this specific trial cleanly discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 4 where A = [0, 1, 1, 1, 0, 0] and B = [1, 0, 0, 0, 1, 1]
    data_copy = data.copy()
    data_copy['A_str'] = data_copy['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t4 = data_copy[data_copy['A_str'] == '011100']
    if len(t4) == 0:
        return 0.5
        
    # response == 0 means option A was chosen
    return float((t4['response'] == 0).mean())

### RESULTS
- Predicted under pi_6 (simulated): 0.3387 (var=0.0152)
- Predicted under pi_4 (simulated): 0.8387 (var=0.0167)
- Observed on real data: 0.3375 (var=0.0894)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_choice(a, b):
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return -1

    ttb_choices = [get_ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    
    matches = (data['response'] == ttb_choices)
    return float(matches.mean())
```

**Observed (real) value:** 0.2758 (var=0.0200)
**Predicted under pi_6:** 0.2331 (var=0.0135)
**Predicted under pi_4:** 0.3375 (var=0.0395)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_ttb_match(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        for j in range(len(a)):
            if a[j] > b[j]:
                return resp == 0
            elif b[j] > a[j]:
                return resp == 1
        return False
        
    return float(data.apply(is_ttb_match, axis=1).mean())
```

**Observed (real) value:** 0.2825 (var=0.0090)
**Predicted under pi_6:** 0.2356 (var=0.0078)
**Predicted under pi_4:** 0.3419 (var=0.0265)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    wadd_aligned = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial type 1: A has the 2 high-validity features, B has the 3 low-validity features
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            wadd_aligned.append(1.0 if resp == 0 else 0.0)
        # Trial type 2: Flipped
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            wadd_aligned.append(1.0 if resp == 1 else 0.0)
            
    if len(wadd_aligned) == 0:
        return 0.5
    return float(np.mean(wadd_aligned))
```

**Observed (real) value:** 0.5083 (var=0.0801)
**Predicted under pi_6:** 0.6383 (var=0.0139)
**Predicted under pi_4:** 0.6683 (var=0.1086)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total
```

**Observed (real) value:** 0.3446 (var=0.0522)
**Predicted under pi_6:** 0.3931 (var=0.0116)
**Predicted under pi_4:** 0.3185 (var=0.0896)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    target_trials = {
        ('10000', '00110'),
        ('01001', '00110'),
        ('10001', '00111')
    }
    
    combined = list(zip(a_str, b_str))
    mask = [pair in target_trials for pair in combined]
    
    if sum(mask) == 0:
        return 0.0
        
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.3593 (var=0.0448)
**Predicted under pi_6:** 0.2772 (var=0.0105)
**Predicted under pi_4:** 0.4846 (var=0.0866)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_ratings = data['option_a_ratings'].apply(tuple)
    mask = (a_ratings == (1, 1, 0, 0, 0)) | (a_ratings == (1, 0, 1, 0, 0))
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.4667 (var=0.0658)
**Predicted under pi_6:** 0.3075 (var=0.0108)
**Predicted under pi_4:** 0.6967 (var=0.0834)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_trial_6(row):
        return tuple(row['option_a_ratings']) == (0, 1, 1, 0, 0) and tuple(row['option_b_ratings']) == (0, 0, 0, 1, 1)
        
    def is_trial_1_or_2(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        return (a == (1, 1, 0, 0, 1) and b == (1, 0, 1, 1, 0)) or (a == (0, 1, 0, 0, 1) and b == (0, 0, 1, 1, 0))
        
    mask_6 = data.apply(is_trial_6, axis=1)
    mask_12 = data.apply(is_trial_1_or_2, axis=1)
    
    if mask_6.sum() == 0 or mask_12.sum() == 0:
        return 0.0
        
    p_a_6 = (data[mask_6]['response'] == 0).mean()
    p_a_12 = (data[mask_12]['response'] == 0).mean()
    
    return float(p_a_6 - p_a_12)

```

**Observed (real) value:** 0.1250 (var=0.1604)
**Predicted under pi_6:** 0.1738 (var=0.0351)
**Predicted under pi_4:** 0.2538 (var=0.0866)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    t3_mask = a_str == '01011'
    val = data.loc[t3_mask, 'response'].mean()
    if pd.isna(val):
        return 0.5
    return float(val)
```

**Observed (real) value:** 0.4547 (var=0.0760)
**Predicted under pi_6:** 0.3158 (var=0.0124)
**Predicted under pi_4:** 0.6895 (var=0.0910)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (Raw Validity-Weighted Additive Strategy, pi_6) dominates Theory 2 (Non-linear Rank-Weighted Additive Strategy, pi_4) across almost all experiments. In the pivotal Experiments 1 and 2, Theory 1's predictions nearly perfectly match the observed data, whereas Theory 2 is off by massive margins (e.g., predicting 0.83 vs an observed 0.33 in Exp 2). However, Theory 1 still underpredicts the observed metric in Experiments 8 and 10 (~0.31 predicted vs ~0.46 observed), indicating that while raw validities are a much better basis for weighting than ordinal ranks, a simple linear sum might not capture all nuances of the decision process. Theory 2 is fundamentally misaligned with the data and needs to be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory to replace the Rank-Weighted model. A strong candidate would be a 'Log-Odds Weighted Additive Strategy' (Naive Bayes), where features are weighted by the log-odds of their validities (log(v / (1-v))). This non-linear transformation of raw validities amplifies differences between high validities (e.g., 0.95 vs 0.90) much more than middle validities (e.g., 0.60 vs 0.55), which could help close the gap in Experiments 8 and 10 while maintaining the excellent performance of Theory 1 in Experiments 1 and 2. Alternatively, consider a 'Mixture of Heuristics' theory where decision-makers probabilistically choose between Take-The-Best and Tallying on a trial-by-trial basis."
}
```

## Usage

```json
{
  "prompt_token_count": 8510,
  "candidates_token_count": 399,
  "total_token_count": 10091
}
```
