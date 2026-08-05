# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People integrate information from multiple cues by computing a weighted sum of the feature values, where the weights are a non-linear transformation of the cues' validities. This Weighted Additive (WADD) strategy is compensatory but incorporates a scaling parameter (gamma) that can exponentiate the validities. This allows the decision-maker to dynamically re-balance attention—either steepening the weights to behave more like Take The Best, or flattening them to behave more like Tallying—to match the empirical balance of conflict resolution.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    # Weighted Additive (WADD) heuristic with exponentiated validities.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Exponentiate validities to allow non-linear scaling of weights
    weights = val ** gamma
    
    # Compute the weighted sum of features for each option
    # using the scaled cue validities as weights.
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


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_6
Rank-Based Weighting with Normalized Directional Cues: Decision-makers evaluate cues based on their informational value (distance from 0.5 validity) rather than raw validity. They sort cues by this importance and assign weights that decay exponentially based on normalized ordinal rank (w = direction * alpha ^ normalized_rank). Normalizing ranks to the [0, 1] interval ensures that the decay profile is invariant to the total number of features in the environment, allowing the same cognitive strategy to scale robustly across contexts with varying dimensionality.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute cue importance (distance from 0.5) and direction
    importance = np.abs(val - 0.5)
    direction = np.sign(val - 0.5)
    
    # Calculate ranks (0 is highest importance, 1 is second highest, etc.)
    order = np.argsort(-importance, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(val), dtype=float)
    
    # Normalize ranks to [0, 1] so that decay is invariant to the number of features
    if len(val) > 1:
        normalized_ranks = ranks / (len(val) - 1)
    else:
        normalized_ranks = ranks
    
    alpha = float(parameters["alpha"])
    # Compute weights: direction * alpha^normalized_rank.
    weights = direction * (alpha ** normalized_ranks)
    
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


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Rationale:** This design quantitatively dissociates Weighted Additive (WADD) with exponentiated validities from Rank-Based Weighting (RBW) with Normalized Directional Cues by exploiting the mathematical properties of their weighting functions. RBW uses an exponentially decaying weight based on normalized ordinal rank (alpha^rank). Because exponential decay is strictly convex, RBW will ALWAYS assign a higher combined weight to the most extreme ranks (e.g., the 1st and 5th cues) than to intermediate ranks (e.g., the 2nd and 4th cues), regardless of the alpha parameter. In contrast, WADD scales the raw validities by an exponent (gamma). When gamma < 1, the WADD weighting function becomes concave with respect to the raw validities, assigning higher relative weight to intermediate validities compared to the extremes. By constructing trials where Option A has the highest and lowest validities and Option B has the intermediate validities, RBW is mathematically forced to predict Option A, while WADD will predict Option B for gamma < 1. We also include trials where both models can flip their preferences depending on their respective parameters, ensuring full identifiability of the scaling behaviors.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
WADD and RBW differ in their preference for Option B across trials due to their weighting functions. In Trial 4 (A=[1,0,1,0,0] vs B=[0,1,0,1,1]), WADD prefers B only when gamma < 1.9 (approx 37% of its parameter space), whereas RBW prefers B when alpha > 0.35 (approx 65% of its parameter space). In Trial 2 (A=[1,0,0,1,0] vs B=[0,1,1,0,0]), WADD prefers B when gamma < 1.0 (approx 20% of the space), while RBW mathematically never prefers B regardless of alpha, because the exponential decay of ranks makes the 1st and 4th cues always outweigh the 2nd and 3rd. By subtracting the rate of choosing B in Trial 2 from the rate of choosing B in Trial 4, we isolate this difference. RBW is expected to show a large positive difference because it switches from strongly preferring A in Trial 2 to often preferring B in Trial 4. WADD is expected to show a much smaller difference because its preferences shift less dramatically between these two trials. This metric effectively discriminates the theories while remaining robust to random choice noise.

Source:
def metric(data: pd.DataFrame) -> float:
    data['a_tuple'] = data['option_a_ratings'].apply(tuple)
    t4_mask = data['a_tuple'] == (1, 0, 1, 0, 0)
    t2_mask = data['a_tuple'] == (1, 0, 0, 1, 0)
    
    if not t4_mask.any() or not t2_mask.any():
        return 0.0
        
    t4_b_rate = data.loc[t4_mask, 'response'].mean()
    t2_b_rate = data.loc[t2_mask, 'response'].mean()
    
    return float(t4_b_rate - t2_b_rate)

### RESULTS
- Predicted under pi_3 (simulated): -0.0237 (var=0.0559)
- Predicted under pi_6 (simulated): 0.1713 (var=0.0692)
- Observed on real data: 0.1450 (var=0.1834)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design dissociates Rank-Based Weighting (RBW) from Weighted Additive (WADD) using a 5-feature environment. RBW evaluates cues based on their ordinal rank of importance and applies an exponentially decaying weight. Because this decay is strictly convex, RBW will consistently favor options possessing a combination of high-ranked and low-ranked cues over options with intermediate-ranked cues (e.g., ranks 0 and 3 vs. ranks 1 and 2), regardless of its alpha parameter. In contrast, WADD scales raw validities by an exponent gamma. For gamma < 1, WADD's weighting function becomes concave over raw validities, causing it to assign higher relative weight to intermediate validities and thus predicting the opposite choice. By including trials that pit extreme-ranked cues against intermediate ones, we force a qualitative divergence where WADD predicts preference reversals that RBW mathematically cannot produce.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric corrects the trial strings from previous attempts and isolates the unique structural divergence between the two models on Trials 3 and 5. For WADD, Option A in Trial 5 is almost always preferred more strongly than Option A in Trial 3 across its entire parameter space, meaning `P(A|T3) - P(A|T5)` is strictly negative or near zero. For RBW, a significant portion of its parameter space (steep decay, alpha < 0.2) produces a strong preference reversal where Option A is chosen in T3 but Option B is chosen in T5, yielding a large positive difference. By applying a ReLU function (max(0, diff)), we completely zero out the noise from the WADD subjects while capturing the strong positive signal from the RBW subjects. This ensures a clean, robust discrimination with minimal between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # T3 pits extreme cues against intermediate cues
    t3 = data[data['a_str'] == '10001']
    # T5 tests the tail drop-off
    t5 = data[data['a_str'] == '01110']
    
    p_a_t3 = 1.0 - t3['response'].mean() if len(t3) > 0 else 0.5
    p_a_t5 = 1.0 - t5['response'].mean() if len(t5) > 0 else 0.5
    
    # We apply a ReLU-like function to isolate the subset of subjects
    # who strongly prefer Option A in T3 over T5.
    # RBW produces this preference reversal for a significant portion of its prior (small alpha).
    # WADD mathematically cannot produce this reversal (it almost always prefers T5 over T3).
    diff = p_a_t3 - p_a_t5
    return float(diff if diff > 0 else 0.0)

### RESULTS
- Predicted under pi_3 (simulated): 0.0000 (var=0.0105)
- Predicted under pi_6 (simulated): 0.0000 (var=0.0423)
- Observed on real data: 0.0000 (var=0.0696)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        a_wins = sum(1 for i in range(4) if a[i] > b[i])
        b_wins = sum(1 for i in range(4) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if resp == ttb_winner:
                ttb_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_consistent / total)

```

**Observed (real) value:** 0.3520 (var=0.0355)
**Predicted under pi_3:** 0.4170 (var=0.0576)
**Predicted under pi_6:** 0.3293 (var=0.0683)

### Experiment 4
**Design**
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Fixed validities from the experimental design
    val = np.array([0.65, 0.95, 0.55, 0.75, 0.85])
    cue_order = np.argsort(-val, kind='stable').tolist()
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue  # Tallying predicts a tie, skip
            
        # TTB prediction
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        # Only consider trials where the two heuristics make STRICTLY OPPOSITE predictions
        if ttb_pred is not None and tally_pred != ttb_pred:
            matches.append(1.0 if row['response'] == tally_pred else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.6600 (var=0.0377)
**Predicted under pi_3:** 0.6763 (var=0.0203)
**Predicted under pi_6:** 0.6477 (var=0.0431)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_has_f0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_f0 = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    chose_f0 = (a_has_f0 & chose_a) | (b_has_f0 & chose_b)
    return float(chose_f0.mean())
```

**Observed (real) value:** 0.3862 (var=0.0372)
**Predicted under pi_3:** 0.4656 (var=0.0213)
**Predicted under pi_6:** 0.4367 (var=0.0371)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 3 where A=[1, 0, 1, 0, 0] and B=[0, 1, 0, 1, 0]
    # This is the only trial where the sum of features is 2 for both options.
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    mask = (a_sums == 2) & (b_sums == 2)
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.6225 (var=0.0889)
**Predicted under pi_3:** 0.8100 (var=0.0256)
**Predicted under pi_6:** 0.7225 (var=0.0303)

### Experiment 7
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify the tie-breaker trials (Trials 1 and 6) where Option A is [0, 1, 1, 0, 0]
    is_target = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (0, 1, 1, 0, 0))
    if not is_target.any():
        return 0.5
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[is_target, 'response'] == 0))
```

**Observed (real) value:** 0.6217 (var=0.0568)
**Predicted under pi_3:** 0.5975 (var=0.0257)
**Predicted under pi_6:** 0.5583 (var=0.0281)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10011') & (b_str == '01100')
    t2_mask = (a_str == '01100') & (b_str == '10011')
    
    t1_choices = data.loc[t1_mask, 'response'] == 0
    t2_choices = data.loc[t2_mask, 'response'] == 1
    
    total_target_choices = t1_choices.sum() + t2_choices.sum()
    total_trials = t1_mask.sum() + t2_mask.sum()
    
    if total_trials == 0:
        return 0.5
    return float(total_target_choices / total_trials)
```

**Observed (real) value:** 0.6025 (var=0.0898)
**Predicted under pi_3:** 0.4706 (var=0.0862)
**Predicted under pi_6:** 0.8375 (var=0.0188)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1: Target has cues [1, 1, 0, 0, 0], Opposing has [0, 0, 1, 1, 1]
    is_t1_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Identify Trial 2: Target has cue [1, 0, 0, 0, 0], Opposing has [0, 1, 1, 0, 0]
    is_t2_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    is_t2_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    
    # Calculate P(Target | T1)
    t1_target_choices = (is_t1_fwd & (data['response'] == 0)).sum() + (is_t1_rev & (data['response'] == 1)).sum()
    t1_total = is_t1_fwd.sum() + is_t1_rev.sum()
    p_t1 = float(t1_target_choices / t1_total) if t1_total > 0 else 0.5
    
    # Calculate P(Target | T2)
    t2_target_choices = (is_t2_fwd & (data['response'] == 0)).sum() + (is_t2_rev & (data['response'] == 1)).sum()
    t2_total = is_t2_fwd.sum() + is_t2_rev.sum()
    p_t2 = float(t2_target_choices / t2_total) if t2_total > 0 else 0.5
    
    # Return the simple difference between the two choice probabilities
    return p_t1 - p_t2

```

**Observed (real) value:** 0.2200 (var=0.1117)
**Predicted under pi_3:** 0.2816 (var=0.0715)
**Predicted under pi_6:** 0.1663 (var=0.0663)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    # Trial 2: Option A is [1, 0, 0, 0, 0], Option B is [0, 1, 0, 0, 1]
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    # Trial 4: Option A is [0, 1, 0, 1, 1], Option B is [1, 0, 1, 0, 0]
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 1))
    
    t2_resp = data.loc[t2_mask, 'response'].mean()
    t4_resp = data.loc[t4_mask, 'response'].mean()
    
    if pd.isna(t2_resp) or pd.isna(t4_resp):
        return 0.0
        
    # Response is 1 if Option B is chosen, 0 if Option A is chosen.
    return float(t4_resp - t2_resp)
```

**Observed (real) value:** 0.0211 (var=0.2810)
**Predicted under pi_3:** 0.0463 (var=0.2994)
**Predicted under pi_6:** -0.3021 (var=0.2941)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 1 (WADD with exponentiated validities, pi_3) and Theory 2 (Rank-based weighting, pi_6) show complementary strengths and weaknesses. Theory 2 performs exceptionally well on Experiment 1, correctly predicting the positive shift in metric, whereas Theory 1 predicts a negative shift. However, Theory 2 fails catastrophically on Experiment 10 (predicting -0.30 vs observed 0.02) and Experiment 8 (predicting 0.83 vs observed 0.60). Theory 1 is generally more robust across the entire suite of experiments, maintaining closer predictions to the observed data on average, but it lacks the structural flexibility to capture the specific rank-based or non-linear extreme-cue preference seen in Experiment 1. Because Theory 2's strict rank-based exponential decay creates rigid mathematical artifacts that lead to severe mispredictions in several environments, it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Log-Odds Weighting (Na\u00efve Bayes)' theory. Instead of rank-based exponential decay or simple exponentiation of raw validities, this theory posits that decision-makers intuitively convert cue validities into log-odds (i.e., weight = log(v / (1 - v))). This naturally provides a principled, non-linear transformation that heavily penalizes uninformative cues and strongly boosts extreme validities. It combines the continuous nature of WADD (avoiding the rigid artifacts of rank-ordering) with the steep extreme-cue preference of the rank-based model, potentially explaining both Experiment 1 and Experiment 10 simultaneously."
}
```

## Usage

```json
{
  "prompt_token_count": 9216,
  "candidates_token_count": 378,
  "total_token_count": 10909
}
```
