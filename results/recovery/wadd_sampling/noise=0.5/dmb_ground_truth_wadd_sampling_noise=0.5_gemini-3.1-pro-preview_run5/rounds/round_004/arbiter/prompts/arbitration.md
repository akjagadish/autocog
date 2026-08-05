# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
WADD / Take-The-Best Strategy Mixture Theory with Calibrated Determinism: Decision-makers probabilistically choose between a fully compensatory Weighted Additive (WADD) strategy and a non-compensatory Take-The-Best (TTB) strategy on any given trial. By mixing WADD and TTB, the model captures both dominance violations (via WADD) and choice probability plateaus (via TTB). Independent inverse temperatures allow each strategy to scale properly. To match the high level of equivocation in human data across multiple experiments, the determinism of these strategies is strongly bounded, ensuring predictions remain appropriately probabilistic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_wadd = float(parameters["beta_wadd"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    p_wadd_weight = float(parameters["p_wadd_weight"])
    
    # --- WADD Prediction ---
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores_wadd = np.array([score_a, score_b])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- Take-The-Best (TTB) Prediction ---
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
        
    # --- Mixture & Noise ---
    p_mix = p_wadd_weight * p_wadd + (1.0 - p_wadd_weight) * p_ttb
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_6
Normalized Rank-Dependent Evidence Accumulation: Decision-makers use a unified compensatory process where the subjective weight of each feature decays exponentially based on its validity rank. To maintain consistent sensitivity across contexts with varying numbers of features, these rank-based weights are normalized to sum to 1.0 before evidence accumulation. This ensures the decision mechanism's determinism scales uniformly across different experimental environments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Assign exponentially decaying weights based on rank
    weights = np.zeros_like(val)
    for rank, cue_idx in enumerate(cue_order):
        weights[cue_idx] = gamma ** rank
        
    # Normalize weights to sum to 1.0
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
        
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add random lapse noise
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.98, 0.95, 0.94, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design introduces a strict quantitative dissociation between the Normalized Rank-Dependent model (Competing) and the WADD/TTB Strategy Mixture (Advocated). By structuring the validities with a steep drop-off between the top three and bottom two cues, we create critical 'Trap' trials (Pairs 1-4). In these trials, Option A wins on the highest available validity cue (winning TTB), ensuring the Rank-Dependent model strictly predicts P(A) >= 0.5 for all possible values of its exponential decay parameter (gamma). However, Option B's cues possess a higher sum of raw validities, meaning WADD strictly favors Option B. If subjects choose Option B more than 50% of the time, the Rank-Dependent model is mathematically falsified because it cannot produce P(A) < 0.5, whereas the Strategy Mixture model naturally captures this via a high WADD mixture weight. Additional conflict trials (Pairs 5-8) are included to calibrate the gamma and mixture parameters.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the proportion of Option B choices in 'Trap' trials (Trials 1-4, uniquely identified by both options having exactly 2 positive features). The Competing theory (Rank-Dependent) mathematically guarantees that Option A is favored in these trials regardless of its parameters, predicting a choice proportion for B of <= 0.5. The Advocated theory (WADD/TTB Mixture) allows Option B to be chosen > 50% of the time due to the WADD component strongly favoring B. This metric will clearly separate the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    is_trap = (data['option_a_ratings'].apply(sum) == 2) & (data['option_b_ratings'].apply(sum) == 2)
    trap_data = data[is_trap]
    if len(trap_data) == 0:
        return 0.0
    return float(trap_data['response'].mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.4221 (var=0.0154)
- Predicted under pi_6 (simulated): 0.2729 (var=0.0216)
- Observed on real data: 0.5550 (var=0.0143)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.94, 0.93, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design strictly dissociates the Advocated Theory (Normalized Rank-Dependent) from the Competing Theory (WADD/TTB Mixture) by exploiting the difference between rank-based weighting and raw-validity weighting. We use a validity distribution with a cluster of high validities and a cluster of low validities: [0.95, 0.94, 0.93, 0.55, 0.50]. In critical trials (e.g., A wins on rank 0 and 3; B wins on rank 1 and 2), the Rank-Dependent model mathematically MUST favor Option A for any decay parameter gamma < 1, because 1 + gamma^3 > gamma + gamma^2. However, the Competing Theory's WADD component strongly favors Option B because the sum of raw validities for B (0.94 + 0.93 = 1.87) vastly exceeds A (0.95 + 0.55 = 1.50). If subjects choose B, the Advocated theory is falsified as it cannot predict P(A) < 0.5 on these trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Measures the proportion of times Option A is chosen in critical trials (Trials 1 and 2). In these trials, the Advocated Theory (Rank-Dependent) mathematically MUST favor Option A for any decay parameter gamma < 1, because 1 + gamma^3 > gamma + gamma^2. The Competing Theory (WADD/TTB Mixture), however, contains a WADD component that strongly favors Option B due to the high sum of its raw validities (0.94 + 0.93 = 1.87) compared to A (0.95 + 0.55 = 1.50). This pulls the Competing Theory's choice probability towards B, creating a sharp divergence between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    is_b_01100 = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    is_a_trial1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0))
    is_a_trial2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1))
    
    mask = is_b_01100 & (is_a_trial1 | is_a_trial2)
    
    if mask.sum() == 0:
        return 0.5
        
    return float((data.loc[mask, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.5358 (var=0.0188)
- Predicted under pi_6 (simulated): 0.7258 (var=0.0345)
- Observed on real data: 0.3833 (var=0.0261)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            if b[i] > a[i]:
                return 1
        return -1
    
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if ttb_predict(a, b) == r:
            matches += 1
            
    return float(matches) / len(data) if len(data) > 0 else 0.0
```

**Observed (real) value:** 0.3747 (var=0.0051)
**Predicted under pi_5:** 0.5238 (var=0.0281)
**Predicted under pi_6:** 0.5878 (var=0.0519)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Calculate first feature for each option (most valid cue)
    first_a = data['option_a_ratings'].apply(lambda x: x[0])
    first_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where Tallying and TTB make strictly opposite predictions
    # Tallying decides by sum(A) vs sum(B)
    # TTB decides by A[0] vs B[0]
    opposing_mask = ((sum_a > sum_b) & (first_a < first_b)) | ((sum_a < sum_b) & (first_a > first_b))
    opposing_data = data[opposing_mask]
    
    if len(opposing_data) == 0:
        return 0.5
        
    # Determine what Tallying would choose: 0 (A) if sum_a > sum_b else 1 (B)
    tallying_choice = (opposing_data['option_a_ratings'].apply(sum) < opposing_data['option_b_ratings'].apply(sum)).astype(int)
    
    # Calculate the proportion of responses that match the Tallying prediction
    match_rate = (opposing_data['response'] == tallying_choice).mean()
    
    return float(match_rate)
```

**Observed (real) value:** 0.6400 (var=0.0095)
**Predicted under pi_5:** 0.5296 (var=0.0326)
**Predicted under pi_6:** 0.4379 (var=0.0642)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    wadd_match = 0
    n_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            
            if wadd_a > wadd_b:
                expected = 0
            elif wadd_b > wadd_a:
                expected = 1
            else:
                continue
                
            if row['response'] == expected:
                wadd_match += 1
            n_ties += 1
            
    if n_ties == 0:
        return 0.5
    return wadd_match / n_ties
```

**Observed (real) value:** 0.5863 (var=0.0190)
**Predicted under pi_5:** 0.6275 (var=0.0122)
**Predicted under pi_6:** 0.7762 (var=0.0149)

### Experiment 6
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target_pairs = [
        ( (1,0,0,0), (0,1,0,0) ),
        ( (1,0,1,0), (0,1,0,1) )
    ]
    
    chosen_has_f1 = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        is_target = False
        for p in target_pairs:
            if (a == p[0] and b == p[1]) or (a == p[1] and b == p[0]):
                is_target = True
                break
                
        if is_target:
            chose_a = (row['response'] == 0)
            chosen_opt = a if chose_a else b
            
            if chosen_opt[0] == 1:
                chosen_has_f1.append(1)
            else:
                chosen_has_f1.append(0)
                
    if len(chosen_has_f1) == 0:
        return 0.5
        
    return float(np.mean(chosen_has_f1))
```

**Observed (real) value:** 0.5317 (var=0.0228)
**Predicted under pi_5:** 0.6717 (var=0.0158)
**Predicted under pi_6:** 0.7467 (var=0.0264)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = tuple(a)
        b = tuple(b)
        if a == (1, 0, 0, 1, 1) and b == (0, 1, 1, 0, 0):
            wadd_choices += int(r == 1)
            total += 1
        elif a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 1, 1):
            wadd_choices += int(r == 0)
            total += 1
    return float(wadd_choices / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4700 (var=0.0206)
**Predicted under pi_5:** 0.3492 (var=0.0201)
**Predicted under pi_6:** 0.2367 (var=0.0179)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Response = 0 means option A was chosen
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    t1_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 1, 1))
    t2_mask = (data['A_tuple'] == (1, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 1, 1, 1))
    t9_mask = (data['A_tuple'] == (1, 1, 1, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 1))
    t10_mask = (data['A_tuple'] == (1, 1, 1, 1, 0)) & (data['B_tuple'] == (0, 0, 0, 0, 1))
    
    p_t1 = data.loc[t1_mask, 'chose_A'].mean()
    p_t2 = data.loc[t2_mask, 'chose_A'].mean()
    p_t9 = data.loc[t9_mask, 'chose_A'].mean()
    p_t10 = data.loc[t10_mask, 'chose_A'].mean()
    
    if pd.isna(p_t1) or pd.isna(p_t2) or pd.isna(p_t9) or pd.isna(p_t10):
        return 0.0
        
    return (p_t2 - p_t1) + (p_t10 - p_t9)

```

**Observed (real) value:** 0.1378 (var=0.0902)
**Predicted under pi_5:** 0.1111 (var=0.1141)
**Predicted under pi_6:** 0.1667 (var=0.0781)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1_a = (1, 0, 0, 0, 1, 1, 1)
    t1_b = (0, 1, 1, 1, 0, 0, 0)
    
    t2_a = (0, 1, 1, 1, 0, 0, 0)
    t2_b = (1, 0, 0, 0, 1, 1, 1)
    
    matches = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a == t1_a and b == t1_b:
            matches.append(1 if row['response'] == 0 else 0)
        elif a == t2_a and b == t2_b:
            matches.append(1 if row['response'] == 1 else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4500 (var=0.0325)
**Predicted under pi_5:** 0.5708 (var=0.0219)
**Predicted under pi_6:** 0.7108 (var=0.0402)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.9, 0.85, 0.8, 0.6, 0.55, 0.5])
    
    def get_wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * val) - np.sum(b * val)
        
    wadd_diffs = data.apply(get_wadd_diff, axis=1)
    
    set1_mask = wadd_diffs > 0.1
    set2_mask = wadd_diffs < 0.1
    
    chose_a = (data['response'] == 0).astype(float)
    
    p_a_set1 = chose_a[set1_mask].mean()
    p_a_set2 = chose_a[set2_mask].mean()
    
    if np.isnan(p_a_set1) or np.isnan(p_a_set2):
        return 0.0
        
    return float(p_a_set1 - p_a_set2)

```

**Observed (real) value:** 0.0767 (var=0.0235)
**Predicted under pi_5:** 0.0717 (var=0.0104)
**Predicted under pi_6:** 0.0546 (var=0.0136)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 1 (WADD/TTB Strategy Mixture) consistently outperforms Theory 2 (Normalized Rank-Dependent Evidence Accumulation). Theory 2 consistently predicts overly deterministic behavior in favor of the highest-ranked cues (e.g., Exps 2, 5, 6, 9), whereas the observed data is much closer to 0.5 or favors the option with a higher total number of features (Tallying). For instance, in Experiment 4, which pits Tallying against Take-The-Best, subjects favor Tallying 64% of the time, which Theory 2 fails to capture. While Theory 1 is better, it still systematically mispredicts the exact choice proportions, often underestimating the pull of Tallying. Theory 2's rank-based exponential decay is too extreme and falsified by the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Normalized Rank-Dependent theory with a 'Validity-Shrinkage / Equal-Weighting Theory'. In this new theory, decision-makers do not use raw validities or exponential ranks; instead, their subjective weights are a convex combination of the provided validities and equal weights (Tallying). This captures the strong empirical tendency of subjects to simply count the number of positive features (as seen in Exp 4) while still retaining some sensitivity to the explicitly stated cue validities. The model should compute subjective weights as w = alpha * validities + (1 - alpha) * (1 / n_features), calculate scores for each option, and apply a softmax choice rule with lapse noise."
}
```

## Usage

```json
{
  "prompt_token_count": 10129,
  "candidates_token_count": 366,
  "total_token_count": 11544
}
```
