# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
People process feature arrays from right to left, exhibiting a recency or layout bias. Instead of consulting cues in order of their objective validities, subjects use a Right-to-Left Take-The-Best (R2L-TTB) heuristic. They evaluate features starting from the last index down to the first, choosing the option that wins on the first discriminating feature encountered in this reversed order. If no feature discriminates, they guess. Response noise is incorporated via a softmax over the binary winner score and a constant lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("R2L-TTB expects a (2, n_features) stimulus.")
    
    n_features = stim.shape[1]
    # Right-to-left cue order
    cue_order = list(range(n_features - 1, -1, -1))
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for choice probability
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
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_6
People integrate all available features but exhibit a strong spatial attention bias, weighting features more heavily the further to the right they appear. This Right-biased Compensatory Model evaluates options using a weighted additive sum where the subjective weight of each feature grows exponentially from left to right. This allows the model to capture behavior that appears strictly non-compensatory (like Right-to-Left Take-The-Best) when the growth factor is large, while still accommodating subjects who might integrate multiple cues in a compensatory manner with a heavy rightward bias. Decisions are made by applying a softmax choice rule over these weighted sums, alongside a baseline rate of random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Right-biased WADD expects a (2, n_features) stimulus.")
    
    n_features = stim.shape[1]
    a, b = stim[0], stim[1]
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Weights increase exponentially from left to right
    weights = np.array([gamma ** j for j in range(n_features)])
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    # Softmax for choice probability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.63, 0.6, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 9: A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 10: A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 11: A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 13: A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  trial 14: A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 15: A=[1, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
In R2L-TTB, the choice probability depends solely on a fixed score difference (1 vs 0) for the winner, regardless of how many features favor it. Thus, R2L-TTB predicts that the probability of choosing B in Trial 1 (where B wins on all features) should be identical to the probability of choosing A in Trial 11 (where A wins on only the leftmost feature, and all other features are tied). In contrast, the Right-biased WADD model produces a massive score difference for Trial 1 (sum of all weights) but a very small score difference for Trial 11 (only the weight of the leftmost feature, gamma^0 = 1). Therefore, WADD predicts a much higher accuracy for Trial 1 than Trial 11, while R2L-TTB predicts a difference of zero.

Source:
def metric(data: pd.DataFrame) -> float:
    # Trial 1: A=[0,0,0,0], B=[1,1,1,1] -> B dominates
    is_t1 = (data['option_a_ratings'].apply(tuple) == (0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (1, 1, 1, 1))
    p_b_t1 = (data[is_t1]['response'] == 1).mean()
    
    # Trial 11: A=[1,0,1,0], B=[0,0,1,0] -> A wins on the leftmost feature only
    is_t11 = (data['option_a_ratings'].apply(tuple) == (1, 0, 1, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 0))
    p_a_t11 = (data[is_t11]['response'] == 0).mean()
    
    if pd.isna(p_b_t1) or pd.isna(p_a_t11):
        return 0.0
        
    return float(p_b_t1 - p_a_t11)

### RESULTS
- Predicted under pi_4 (simulated): -0.0167 (var=0.0469)
- Predicted under pi_6 (simulated): -0.0067 (var=0.0422)
- Observed on real data: -0.0200 (var=0.0296)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=4):** [0.95, 0.74, 0.64, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 7: A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 9: A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 10: A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 11: A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 12: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 13: A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 14: A=[1, 0, 0, 1]  B=[1, 1, 1, 1]
  trial 15: A=[0, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The Competing Theory (Right-to-Left Take-The-Best) determines its choice based solely on the rightmost discriminating cue. It completely ignores all other cues, meaning its choice probability (driven by a constant softmax over [1,0]) is identical regardless of whether the remaining cues agree or disagree with the winner. The Advocated Theory (Right-biased Compensatory Model), however, integrates all cues. When the rightmost cue is opposed by other cues (Conflict trials), the net score difference between the options shrinks, reducing choice confidence compared to trials where all discriminating cues point in the same direction (Consistent trials). This metric calculates the difference in accuracy (relative to the R2L-TTB prediction) between Consistent and Conflict trials. The Competing Theory expects this difference to be exactly 0, while the Advocated Theory predicts a strictly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches_consistent = []
    matches_conflict = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        diff = a - b
        
        nonzero_indices = np.where(diff != 0)[0]
        if len(nonzero_indices) == 0:
            continue
            
        highest_idx = nonzero_indices[-1]
        
        if diff[highest_idx] == 1:
            r2l_winner = 0
            is_conflict = np.any(diff == -1)
        else:
            r2l_winner = 1
            is_conflict = np.any(diff == 1)
            
        match = 1 if row['response'] == r2l_winner else 0
        
        if is_conflict:
            matches_conflict.append(match)
        else:
            matches_consistent.append(match)
            
    if len(matches_consistent) == 0 or len(matches_conflict) == 0:
        return 0.0
        
    return float(np.mean(matches_consistent) - np.mean(matches_conflict))

### RESULTS
- Predicted under pi_4 (simulated): -0.0015 (var=0.0062)
- Predicted under pi_6 (simulated): 0.0005 (var=0.0085)
- Observed on real data: -0.0105 (var=0.0057)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_trial_3(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        t1 = (1, 0, 0, 1)
        t2 = (0, 1, 1, 1)
        return (a == t1 and b == t2) or (a == t2 and b == t1)
        
    mask = data.apply(is_trial_3, axis=1)
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    def chose_ttb_option(row):
        a = tuple(row['option_a_ratings'])
        # The TTB-favored option is the one with cue 0 == 1, i.e., (1, 0, 0, 1)
        if a == (1, 0, 0, 1):
            return 1.0 if row['response'] == 0 else 0.0
        else:
            return 1.0 if row['response'] == 1 else 0.0
            
    return float(subset.apply(chose_ttb_option, axis=1).mean())
```

**Observed (real) value:** 0.1933 (var=0.0304)
**Predicted under pi_4:** 0.1300 (var=0.0248)
**Predicted under pi_6:** 0.1000 (var=0.0144)

### Experiment 4
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_ratings = data['option_a_ratings'].apply(tuple)
    b_ratings = data['option_b_ratings'].apply(tuple)
    
    mask8 = (a_ratings == (0, 1, 0, 1)) & (b_ratings == (1, 1, 0, 0))
    mask11 = (a_ratings == (1, 1, 0, 1)) & (b_ratings == (1, 0, 1, 1))
    
    resp8 = data.loc[mask8, 'response']
    resp11 = data.loc[mask11, 'response']
    
    score8 = (resp8 == 1).mean() if len(resp8) > 0 else 0.5
    score11 = (resp11 == 0).mean() if len(resp11) > 0 else 0.5
    
    return float(score8 + score11)
```

**Observed (real) value:** 0.2067 (var=0.0540)
**Predicted under pi_4:** 0.3700 (var=0.0948)
**Predicted under pi_6:** 0.2300 (var=0.0532)

### Experiment 5
**Design**
  A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify specific trials by their feature string representations
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # High WADD ratio trials (Tallying difference is exactly 1)
    # Trial 4: A=[0,0,0,1], B=[0,1,0,1] -> WADD ratio B/A = 1.35 / 0.55 = 2.45
    t4 = data[(data['a_str'] == '0001') & (data['b_str'] == '0101')]
    # Trial 8: A=[0,0,1,1], B=[0,0,0,1] -> WADD ratio A/B = 1.33 / 0.55 = 2.41
    t8 = data[(data['a_str'] == '0011') & (data['b_str'] == '0001')]
    
    # Low WADD ratio trials (Tallying difference is exactly 1)
    # Trial 1: A=[1,0,1,0], B=[1,0,1,1] -> WADD ratio B/A = 2.28 / 1.73 = 1.31
    t1 = data[(data['a_str'] == '1010') & (data['b_str'] == '1011')]
    # Trial 12: A=[1,1,0,0], B=[1,1,0,1] -> WADD ratio B/A = 2.30 / 1.75 = 1.31
    t12 = data[(data['a_str'] == '1100') & (data['b_str'] == '1101')]
    
    p_b_t4 = t4['response'].mean() if len(t4) > 0 else 0.5
    p_a_t8 = 1.0 - t8['response'].mean() if len(t8) > 0 else 0.5
    
    p_b_t1 = t1['response'].mean() if len(t1) > 0 else 0.5
    p_b_t12 = t12['response'].mean() if len(t12) > 0 else 0.5
    
    high_ratio_acc = (p_b_t4 + p_a_t8) / 2.0
    low_ratio_acc = (p_b_t1 + p_b_t12) / 2.0
    
    return float(high_ratio_acc - low_ratio_acc)
```

**Observed (real) value:** -0.0067 (var=0.0177)
**Predicted under pi_4:** -0.0033 (var=0.0233)
**Predicted under pi_6:** -0.0250 (var=0.0173)

### Experiment 6
**Design**
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t9_mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0]) & \
              data['option_b_ratings'].apply(lambda x: list(x) == [0, 0, 0, 1])
              
    t10_mask = data['option_a_ratings'].apply(lambda x: list(x) == [0, 0, 1, 1]) & \
               data['option_b_ratings'].apply(lambda x: list(x) == [0, 1, 0, 0])
               
    resp_9 = data[t9_mask]['response'].mean()
    resp_10 = data[t10_mask]['response'].mean()
    
    if pd.isna(resp_9) or pd.isna(resp_10):
        return 0.0
        
    return float(resp_10 - resp_9)
```

**Observed (real) value:** -0.7133 (var=0.0434)
**Predicted under pi_4:** -0.6833 (var=0.0781)
**Predicted under pi_6:** -0.7333 (var=0.0844)

### Experiment 7
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t11_a, t11_b = (1, 1, 1, 0), (1, 0, 0, 1)
    t12_a, t12_b = (0, 0, 1, 1), (1, 1, 1, 0)
    
    scores = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t11_a and b == t11_b:
            scores.append(1.0 if resp == 1 else 0.0)
        elif a == t11_b and b == t11_a:
            scores.append(1.0 if resp == 0 else 0.0)
        elif a == t12_a and b == t12_b:
            scores.append(1.0 if resp == 0 else 0.0)
        elif a == t12_b and b == t12_a:
            scores.append(1.0 if resp == 1 else 0.0)
            
    if len(scores) == 0:
        return 0.5
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8367 (var=0.0203)
**Predicted under pi_4:** 0.8417 (var=0.0184)
**Predicted under pi_6:** 0.8567 (var=0.0272)

### Experiment 8
**Design**
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        if a == [0, 1, 1, 0] and b == [0, 0, 0, 1]:
            return True
        if a == [0, 0, 0, 1] and b == [0, 1, 1, 0]:
            return True
        return False
        
    mask = data.apply(is_target_trial, axis=1)
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    def r2l_choice(row):
        a = list(row['option_a_ratings'])
        if a == [0, 1, 1, 0]:
            return row['response'] == 1
        else:
            return row['response'] == 0
            
    return float(subset.apply(r2l_choice, axis=1).mean())
```

**Observed (real) value:** 0.8200 (var=0.0509)
**Predicted under pi_4:** 0.8233 (var=0.0382)
**Predicted under pi_6:** 0.7767 (var=0.0573)

### Experiment 9
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    r2l_match_count = 0
    conflict_count = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Standard TTB (Left-to-Right) winner
        l2r_winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                l2r_winner = 0
                break
            elif b[j] > a[j]:
                l2r_winner = 1
                break
                
        # R2L-TTB (Right-to-Left) winner
        r2l_winner = None
        for j in range(len(a)-1, -1, -1):
            if a[j] > b[j]:
                r2l_winner = 0
                break
            elif b[j] > a[j]:
                r2l_winner = 1
                break
                
        # Only look at trials where the two heuristics disagree
        if l2r_winner is not None and r2l_winner is not None and l2r_winner != r2l_winner:
            conflict_count += 1
            if resp == r2l_winner:
                r2l_match_count += 1
                
    if conflict_count == 0:
        return 0.5
        
    return r2l_match_count / conflict_count

```

**Observed (real) value:** 0.8208 (var=0.0190)
**Predicted under pi_4:** 0.8304 (var=0.0091)
**Predicted under pi_6:** 0.8633 (var=0.0089)

### Experiment 10
**Design**
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    match_l2r = 0
    total_diff = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        l2r_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                l2r_pred = 0
                break
            elif b[i] > a[i]:
                l2r_pred = 1
                break
                
        r2l_pred = None
        for i in range(len(a)-1, -1, -1):
            if a[i] > b[i]:
                r2l_pred = 0
                break
            elif b[i] > a[i]:
                r2l_pred = 1
                break
                
        if l2r_pred is not None and r2l_pred is not None and l2r_pred != r2l_pred:
            total_diff += 1
            if row['response'] == l2r_pred:
                match_l2r += 1
                
    if total_diff == 0:
        return 0.5
    return match_l2r / total_diff
```

**Observed (real) value:** 0.1933 (var=0.0236)
**Predicted under pi_4:** 0.2033 (var=0.0155)
**Predicted under pi_6:** 0.1208 (var=0.0105)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 1 (Right-to-Left Take-The-Best, pi_4) consistently outperforms Theory 2 (Right-biased Compensatory Model, pi_6). Experiments 8, 9, and 10 explicitly test Left-to-Right vs Right-to-Left evaluation and find strong alignment with the Right-to-Left heuristic. Theory 1 accurately captures the high rate of right-to-left choices. Theory 2's exponential weighting scheme attempts to mimic this non-compensatory pattern but struggles with exact proportions, often over- or under-predicting the effect sizes (e.g., in Exp 8 and 10). However, Theory 1's parameter ranges for noise (epsilon/beta) seem slightly too high, causing it to overpredict error rates in Exp 4. Overall, the Right-to-Left non-compensatory heuristic is a much better baseline than a compensatory exponential-weight model.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with a new theory that acknowledges the primacy of the rightmost features but introduces a different fallback mechanism than strict R2L-TTB. For instance, propose a 'Rightmost-Cue plus Tallying' theory: people first check the rightmost cue; if it discriminates, they use it. If it ties, instead of moving strictly one-by-one to the left (like R2L-TTB), they simply tally the remaining cues to make a quick decision. This provides a simpler cognitive mechanism for ties than sequential R2L evaluation and might better capture variance in trials where the rightmost cue ties."
}
```

## Usage

```json
{
  "prompt_token_count": 11636,
  "candidates_token_count": 374,
  "total_token_count": 13681
}
```
