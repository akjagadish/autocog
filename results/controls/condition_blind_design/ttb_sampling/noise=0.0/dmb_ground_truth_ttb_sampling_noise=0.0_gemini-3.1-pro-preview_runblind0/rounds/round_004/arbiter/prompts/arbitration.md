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
Decision-makers use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. Features are ranked by their validity, and options are compared on features one by one in descending order of validity. The choice is determined entirely by the first feature that discriminates between the options, ignoring all lower-validity cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Lexicographic evaluation
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_6
Hybrid TTB-WADD Theory: Decision-makers probabilistically alternate between a strict non-compensatory strategy (Take The Best) and a fully compensatory strategy (Weighted Additive Model / WADD) depending on the trial or internal state. The model computes the choice probabilities for both TTB (relying solely on the first discriminating cue) and WADD (summing the validity-weighted cue values) and blends them using a subject-level mixing parameter.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Hybrid model expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # --- TTB Probability ---
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    ttb_p = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_p = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_p = np.array([0.0, 1.0])
            break
            
    # --- WADD Probability ---
    score_a = np.sum(a * validities)
    score_b = np.sum(b * validities)
    wadd_scores = np.array([score_a, score_b])
    
    tau = float(parameters["tau"])
    z = (wadd_scores - np.max(wadd_scores)) / tau
    e = np.exp(z)
    wadd_p = e / np.sum(e)
    
    # --- Blend ---
    mix_rate = float(parameters["mix_rate"])
    p_core = mix_rate * ttb_p + (1.0 - mix_rate) * wadd_p
    
    # --- Lapse Rate ---
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.72, 0.86, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0]  B=[0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 1]  B=[0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 9: A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  trial 10: A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  trial 11: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 12: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 13: A=[1, 1, 0, 1]  B=[0, 1, 1, 0]
  trial 14: A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 15: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 16: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Under the Take The Best (TTB) model, the probability of choosing the favored option is determined entirely by the first discriminating cue and is thus constant across all trials where a discriminating cue exists (subject only to a constant lapse rate and softmax temperature on binary scores). Under the Hybrid TTB-WADD model, the choice probability is sensitive to the magnitude of the difference in weighted additive (WADD) scores. By computing the difference in accuracy between trials with a large WADD score difference (T5, T16, T8) and trials with a small WADD score difference (T11, T10, T13), we expect a metric near 0 for pure TTB and significantly greater than 0 for the Hybrid model.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    def get_acc(a_target, b_target, favored_if_a):
        mask1 = (a_tuples == a_target) & (b_tuples == b_target)
        mask2 = (a_tuples == b_target) & (b_tuples == a_target)
        acc = []
        if mask1.any():
            acc.extend((data.loc[mask1, 'response'] == (0 if favored_if_a else 1)).tolist())
        if mask2.any():
            acc.extend((data.loc[mask2, 'response'] == (1 if favored_if_a else 0)).tolist())
        return acc

    acc_small = []
    acc_small.extend(get_acc((1, 0, 0, 0), (0, 1, 0, 0), True))  # T11
    acc_small.extend(get_acc((0, 0, 1, 0), (1, 0, 0, 1), False)) # T10
    acc_small.extend(get_acc((1, 1, 0, 1), (0, 1, 1, 0), True))  # T13
    
    acc_large = []
    acc_large.extend(get_acc((1, 1, 0, 1), (0, 0, 0, 0), True))  # T5
    acc_large.extend(get_acc((1, 0, 1, 1), (0, 0, 0, 0), True))  # T16
    acc_large.extend(get_acc((1, 1, 1, 0), (0, 1, 0, 1), True))  # T8
    
    if not acc_small or not acc_large:
        return 0.0
        
    return float(np.mean(acc_large) - np.mean(acc_small))

### RESULTS
- Predicted under pi_3 (simulated): -0.0033 (var=0.0217)
- Predicted under pi_6 (simulated): 0.0356 (var=0.0180)
- Observed on real data: 0.0000 (var=0.0116)

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
Trial 6 is the only trial in the design where the Take-The-Best (TTB) strategy and the Weighted Additive (WADD) strategy predict different choices. TTB relies solely on the highest validity cue and prefers option A, while WADD integrates all cues and prefers option B because the sum of the lower validities outweighs the single highest one. The TTB model will almost exclusively choose A (yielding a metric near 0), whereas the Hybrid model, which mixes TTB and WADD, will choose B much more frequently.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify the critical trial where TTB and WADD diverge.
    # Trial 6: A=[1, 0, 0, 0], B=[0, 0, 1, 1]
    # TTB chooses A (cue 1 is 1 for A and 0 for B).
    # WADD chooses B (0.95 for A vs 0.64 + 0.55 = 1.19 for B).
    mask = (data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0])) & \
           (data['option_b_ratings'].apply(lambda x: list(x) == [0, 0, 1, 1]))
    
    if not mask.any():
        return 0.0
        
    # Return the proportion of times B was chosen (response == 1)
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1000 (var=0.0211)
- Predicted under pi_6 (simulated): 0.2767 (var=0.0496)
- Observed on real data: 0.1533 (var=0.0376)

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
    import numpy as np
    validities = np.array([0.95, 0.76, 0.92, 0.55])
    
    # Extract matrices of A and B features
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying differences
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_diff = a_wins - b_wins
    
    # WADD differences (expected without subject-specific weights)
    wadd_diff = np.sum((a_mat - b_mat) * validities, axis=1)
    
    # 1 if chose A, 0 if chose B
    chose_a = 1.0 - data['response'].values
    
    # We compare choice probabilities within trials where Tallying predicts the exact same difference.
    # For tally_diff == 1, WADD predicts some trials favor A more strongly than others.
    mask1_high = (tally_diff == 1) & (wadd_diff > 0.85)
    mask1_low = (tally_diff == 1) & (wadd_diff < 0.85)
    
    diff1 = 0.0
    if np.any(mask1_high) and np.any(mask1_low):
        diff1 = np.mean(chose_a[mask1_high]) - np.mean(chose_a[mask1_low])
        
    # For tally_diff == -1, WADD predicts some trials favor B more strongly than others.
    mask_m1_high = (tally_diff == -1) & (wadd_diff > -0.85)
    mask_m1_low = (tally_diff == -1) & (wadd_diff < -0.85)
    
    diff_m1 = 0.0
    if np.any(mask_m1_high) and np.any(mask_m1_low):
        diff_m1 = np.mean(chose_a[mask_m1_high]) - np.mean(chose_a[mask_m1_low])
        
    # Under Tallying, both diff1 and diff_m1 should be 0.
    # Under WADD, both diff1 and diff_m1 should be positive.
    return float(diff1 + diff_m1)
```

**Observed (real) value:** 0.1467 (var=0.0250)
**Predicted under pi_3:** 0.1900 (var=0.0422)
**Predicted under pi_6:** 0.1550 (var=0.0474)

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
    # Identify trials where Tallying predicts a tie but WADD predicts a strong preference.
    # Trial 8: A=[0, 1, 0, 1], B=[1, 1, 0, 0]
    # A wins on feature 4 (validity 0.55). B wins on feature 1 (validity 0.95).
    # Tallying sees 1 win for A and 1 win for B, predicting exactly 50% choice for B.
    # WADD sees B's win on the most important feature as outweighing A's win on the least important, predicting >50% choice for B.
    is_target = data['option_a_ratings'].apply(lambda x: list(x) == [0, 1, 0, 1]) & \
                data['option_b_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0])
    
    if is_target.sum() == 0:
        return 0.5
        
    return float(data.loc[is_target, 'response'].mean())
```

**Observed (real) value:** 0.8200 (var=0.0532)
**Predicted under pi_3:** 0.8733 (var=0.0284)
**Predicted under pi_6:** 0.7800 (var=0.0460)

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
    import pandas as pd
    import numpy as np
    
    ttb_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 3: A=[0, 1, 1, 0], B=[1, 0, 0, 0]
        # TTB chooses B (due to F1), WADD might choose A (due to F2+F3)
        if a == (0, 1, 1, 0) and b == (1, 0, 0, 0):
            if resp == 1:
                ttb_match += 1
            total += 1
        elif a == (1, 0, 0, 0) and b == (0, 1, 1, 0):
            if resp == 0:
                ttb_match += 1
            total += 1
            
        # Trial 11: A=[1, 0, 1, 0], B=[0, 1, 1, 1]
        # TTB chooses A (due to F1), WADD might choose B (due to F2+F4)
        elif a == (1, 0, 1, 0) and b == (0, 1, 1, 1):
            if resp == 0:
                ttb_match += 1
            total += 1
        elif a == (0, 1, 1, 1) and b == (1, 0, 1, 0):
            if resp == 1:
                ttb_match += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_match / total)

```

**Observed (real) value:** 0.8433 (var=0.0296)
**Predicted under pi_3:** 0.8117 (var=0.0236)
**Predicted under pi_6:** 0.7050 (var=0.0470)

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
    # Convert feature lists to tuples to allow element-wise comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify the two trials where WADD's compensatory nature opposes TTB's lexicographic rule
    # Trial 10: TTB chooses B (due to feature 2), WADD leans A (features 3 + 4 compensate for feature 2)
    is_trial_10 = (a_tuples == (0, 0, 1, 1)) & (b_tuples == (0, 1, 0, 0))
    # Trial 14: TTB chooses A (due to feature 2), WADD leans B (features 3 + 4 compensate for feature 2)
    is_trial_14 = (a_tuples == (1, 1, 0, 0)) & (b_tuples == (1, 0, 1, 1))
    
    # Calculate the proportion of choices that align with the WADD compensatory prediction
    wadd_choice_10 = (data.loc[is_trial_10, 'response'] == 0).mean()
    wadd_choice_14 = (data.loc[is_trial_14, 'response'] == 1).mean()
    
    # Handle edge cases where a subject might have missing data for these specific trials
    if pd.isna(wadd_choice_10): wadd_choice_10 = 0.5
    if pd.isna(wadd_choice_14): wadd_choice_14 = 0.5
    
    return float((wadd_choice_10 + wadd_choice_14) / 2.0)
```

**Observed (real) value:** 0.1333 (var=0.0156)
**Predicted under pi_3:** 0.1550 (var=0.0122)
**Predicted under pi_6:** 0.2700 (var=0.0457)

### Experiment 7
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Trial 14: A=[1, 1, 1, 1], B=[0, 0, 0, 0] -> TTB predicts A
    t14 = data[(data['A_tuple'] == (1, 1, 1, 1)) & (data['B_tuple'] == (0, 0, 0, 0))]
    # Trial 7: A=[0, 0, 1, 0], B=[0, 0, 1, 1] -> TTB predicts B
    t7 = data[(data['A_tuple'] == (0, 0, 1, 0)) & (data['B_tuple'] == (0, 0, 1, 1))]
    
    if len(t14) == 0 or len(t7) == 0:
        return 0.0
        
    p_A_14 = (t14['response'] == 0).mean()
    p_B_7 = (t7['response'] == 1).mean()
    
    return float(p_A_14 - p_B_7)
```

**Observed (real) value:** -0.0067 (var=0.0433)
**Predicted under pi_3:** 0.0233 (var=0.0245)
**Predicted under pi_6:** 0.0600 (var=0.0453)

### Experiment 8
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
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A'] = data['option_a_ratings'].apply(tuple)
    data['B'] = data['option_b_ratings'].apply(tuple)
    
    def prob_choose_1(op1, op2):
        mask1 = (data['A'] == op1) & (data['B'] == op2)
        mask2 = (data['A'] == op2) & (data['B'] == op1)
        
        choices_op1 = 0
        total = 0
        
        if mask1.sum() > 0:
            choices_op1 += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            choices_op1 += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
            
        return choices_op1 / total if total > 0 else np.nan

    # Trial 1: op1=(1,0,0,0), op2=(0,1,1,0)
    # Highest validity cue favors op1, but two lower cues favor op2.
    p_t1 = prob_choose_1((1,0,0,0), (0,1,1,0))
    
    # Trial 3: op1=(1,0,0,0), op2=(0,0,1,0)
    # Highest validity cue favors op1, only one lower cue favors op2.
    p_t3 = prob_choose_1((1,0,0,0), (0,0,1,0))
    
    # Trial 2: op1=(1,1,0,0), op2=(0,1,0,1)
    # Highest validity cue favors op1, lowest cue favors op2.
    p_t2 = prob_choose_1((1,1,0,0), (0,1,0,1))
    
    # Trial 13: op1=(1,1,0,0), op2=(0,1,0,0)
    # Highest validity cue favors op1, no cues favor op2.
    p_t13 = prob_choose_1((1,1,0,0), (0,1,0,0))
    
    val1 = (p_t3 - p_t1) if not np.isnan(p_t3 - p_t1) else 0.0
    val2 = (p_t13 - p_t2) if not np.isnan(p_t13 - p_t2) else 0.0
    
    return float(val1 + val2)
```

**Observed (real) value:** -0.0733 (var=0.0624)
**Predicted under pi_3:** 0.0600 (var=0.0786)
**Predicted under pi_6:** 0.0800 (var=0.0980)

### Experiment 9
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_target(a, b):
        return list(a) == [1, 0, 1, 0] and list(b) == [0, 1, 1, 0]
        
    def is_target_rev(a, b):
        return list(a) == [0, 1, 1, 0] and list(b) == [1, 0, 1, 0]

    fwd = data.apply(lambda row: is_target(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    rev = data.apply(lambda row: is_target_rev(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    
    n_fwd = fwd.sum()
    n_rev = rev.sum()
    
    if n_fwd + n_rev == 0:
        return 0.0
        
    chose_target_fwd = (data.loc[fwd, 'response'] == 0).sum()
    chose_target_rev = (data.loc[rev, 'response'] == 1).sum()
    
    return float(chose_target_fwd + chose_target_rev) / (n_fwd + n_rev)
```

**Observed (real) value:** 0.8733 (var=0.0317)
**Predicted under pi_3:** 0.8667 (var=0.0267)
**Predicted under pi_6:** 0.8000 (var=0.0411)

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
    import pandas as pd
    
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    b0 = data['option_b_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    b1 = data['option_b_ratings'].apply(lambda x: x[1])
    
    mask_A = (a0 > b0) & (a1 < b1)
    mask_B = (a0 < b0) & (a1 > b1)
    
    subset_A = data[mask_A]
    subset_B = data[mask_B]
    
    total_conflict = len(subset_A) + len(subset_B)
    if total_conflict == 0:
        return 0.5
        
    chose_f0_A = (subset_A['response'] == 0).sum()
    chose_f0_B = (subset_B['response'] == 1).sum()
    
    return float((chose_f0_A + chose_f0_B) / total_conflict)
```

**Observed (real) value:** 0.8300 (var=0.0186)
**Predicted under pi_3:** 0.8250 (var=0.0184)
**Predicted under pi_6:** 0.6550 (var=0.0461)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 1 (Take The Best, pi_3) consistently outperforms Theory 2 (Hybrid TTB-WADD, pi_6). In experiments 1, 5, 6, 7, 9, and 10, pi_3's predictions are remarkably close to the observed data, whereas pi_6 frequently underestimates the reliance on the single highest-validity cue. The observed data strongly supports a non-compensatory lexicographic approach. However, neither model perfectly captures the nuances in some edge cases (e.g., Experiment 8). Since pi_3 is already a very strong baseline, pi_6 should be replaced with a brand-new theory that acts as a more plausible competitor to pure TTB, rather than just tweaking the hybrid mixing weights.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory that remains within the non-compensatory or semi-compensatory family to better compete with TTB. For example, a 'Probabilistic Cue Search' theory where cues are evaluated sequentially but the order of evaluation is probabilistic (weighted by cue validities), or a 'Top-K Tallying' theory where decision-makers tally only the top 2 or 3 most valid cues rather than integrating all cues (WADD) or strictly stopping at the first discriminating cue (TTB). This would provide a more realistic departure from strict TTB than a full WADD hybrid."
}
```

## Usage

```json
{
  "prompt_token_count": 12333,
  "candidates_token_count": 326,
  "total_token_count": 13709
}
```
