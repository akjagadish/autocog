# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Rank-Weighted Exponential Integration: Decision-makers evaluate all cues simultaneously but weight them exponentially according to their validity rank (weight = alpha^(-rank)). This creates a non-compensatory profile that mimics Take-The-Best when cues are aligned, but naturally allows for compensatory overrides when multiple lower-ranked cues strongly align against the top cue, capturing regressions to chance in highly conflicting trial designs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine the rank of each cue (0 = highest validity, 1 = second highest, etc.)
    cue_order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(cue_order)
    ranks[cue_order] = np.arange(len(val))
    
    alpha = float(parameters["alpha"])
    # Exponentially decay weights based on rank
    weights = alpha ** (-ranks)
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Sequential Evidence Accumulation with Normalized Weights: Decision-makers inspect cues sequentially in descending order of validity. Each cue's difference updates a running evidence tally weighted by the cue's normalized log-odds validity. Normalizing the weights ensures that the accumulated evidence scales consistently across different experiments, making the latent decision threshold an invariant parameter. If the absolute evidence crosses this threshold, search stops and a choice is made immediately. If all cues are exhausted without crossing the threshold, the decision defaults to the accumulated tally.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds weights and normalize
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    weights = weights / np.sum(weights)
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        E += diff * weights[j]
        # Stop search if evidence crosses threshold (and is non-zero to skip ties)
        if abs(E) >= threshold and abs(E) > 1e-5:
            break
            
    # E > 0 favors option A, E < 0 favors option B
    scores = np.array([E, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.94, 0.93, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design contrasts Rank-Weighted Exponential Integration (Theory 1) with Sequential Evidence Accumulation using Normalized Log-Odds (Theory 2). By choosing validities where the top three cues have very similar, high validities (and thus similar log-odds weights) but the last two cues have very low validities, we expose the difference in weighting mechanisms. Theory 1 forces an exponential drop-off based strictly on rank (so Cue 1 is always substantially heavily weighted than Cue 2, regardless of their actual validity difference), whereas Theory 2 weights Cue 1 and Cue 2 almost equally because their log-odds are similar. In trials where Option A has Cue 1 and the bottom cues, while Option B has Cues 2 and 3, Theory 1 will consistently favor Option A across a wide range of alpha parameters, whereas Theory 2 (if its threshold allows integration) will strongly favor Option B. Varying the threshold and alpha across other trial configurations systematically maps the divergence between exponential rank decay and proportional log-odds accumulation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates Trial 1, where Option A has Cue 1 (the most valid cue) plus the two least valid cues, while Option B has Cues 2 and 3. Under Theory 1 (Rank-Weighted Exponential Integration), Option A is consistently favored regardless of the alpha parameter because the combination of rank 0, 3, and 4 always outweighs rank 1 and 2. Under Theory 2 (Sequential Evidence Accumulation with Normalized Log-Odds), the weights of Cues 1, 2, and 3 are nearly identical. Unless the decision threshold is extremely low (causing search to terminate immediately after Cue 1), the accumulation process will integrate Cues 2 and 3, leading to a strong preference for Option B. Thus, Theory 1 predicts a choice probability near 1.0 for the target option, while Theory 2 predicts a much lower probability.

Source:
def metric(data: pd.DataFrame) -> float:
    target_opt = (1, 0, 0, 1, 1)
    opp_opt = (0, 1, 1, 0, 0)
    
    a_is_target = data['option_a_ratings'].apply(tuple) == target_opt
    b_is_opp = data['option_b_ratings'].apply(tuple) == opp_opt
    mask1 = a_is_target & b_is_opp
    
    a_is_opp = data['option_a_ratings'].apply(tuple) == opp_opt
    b_is_target = data['option_b_ratings'].apply(tuple) == target_opt
    mask2 = a_is_opp & b_is_target
    
    valid_trials = data[mask1 | mask2]
    if len(valid_trials) == 0:
        return 0.5
        
    chose_target_when_A = (valid_trials['response'] == 0) & mask1
    chose_target_when_B = (valid_trials['response'] == 1) & mask2
    
    prob = (chose_target_when_A.sum() + chose_target_when_B.sum()) / len(valid_trials)
    return float(prob)

### RESULTS
- Predicted under pi_6 (simulated): 0.6917 (var=0.0220)
- Predicted under pi_5 (simulated): 0.4517 (var=0.0584)
- Observed on real data: 0.7200 (var=0.2016)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates Theory 1 (Sequential Evidence Accumulation with Normalized Log-Odds) from Theory 2 (Rank-Weighted Exponential Integration). We use a validity profile where the top three cues are highly valid but distinguishable, while the bottom two are weak. In Theory 1, the log-odds weights for the top three cues are relatively close, allowing the combination of cues 2 and 3 to outweigh cue 1. Thus, Theory 1's choice depends heavily on its evidence threshold (switching from TTB-like to WADD-like). In contrast, Theory 2 enforces an exponential drop-off based purely on rank. For many alpha values, cue 1 dominates cues 2 and 3 combined, but cues 2, 4, and 5 might outweigh cue 1 if alpha is low. By pitting the top cue against various combinations of lower-ranked cues, we create trials where Theory 1 dictates one choice while Theory 2 strictly dictates the opposite, or where the parameters of each theory trigger opposite switching behaviors.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates Trial 1: [1, 0, 0, 1, 1] vs [0, 1, 1, 0, 0]. In Theory 2 (Rank-Weighted Exponential Integration), the exponential penalty ensures that the top cue strictly dominates the combination of cues 2 and 3 for all valid values of alpha >= 1.0. Consequently, Theory 2 will almost never choose the target option [0, 1, 1, 0, 0] (except due to epsilon noise). In contrast, Theory 1 (Sequential Evidence Accumulation) uses normalized log-odds validities. The weight of cue 1 (0.393) is smaller than the combined weights of cues 2 and 3 (0.294 + 0.232 = 0.526). If Theory 1's decision threshold is low (<= 0.393), search stops at cue 1 and it rejects the target option. However, if the threshold is high (> 0.393, which occurs ~60% of the time since threshold ~ U(0,1)), it evaluates all cues and the accumulated evidence favors the target option. Thus, Theory 1 chooses the target option ~60% of the time (modulo noise), while Theory 2 chooses it ~0% of the time. This massive mean gap provides a perfect, highly significant discriminator.

Source:
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to strings for robust matching without hashing issues
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Isolate Trial 1: [1, 0, 0, 1, 1] vs [0, 1, 1, 0, 0]
    # The target option is [0, 1, 1, 0, 0], which lacks the top cue but has the next two.
    mask_a_target = (a_str == '01100') & (b_str == '10011')
    mask_b_target = (b_str == '01100') & (a_str == '10011')
    
    valid_mask = mask_a_target | mask_b_target
    if not valid_mask.any():
        return 0.5
        
    subset = data[valid_mask]
    mask_a_subset = mask_a_target[valid_mask]
    
    # Response is 0 if A is chosen, 1 if B is chosen
    # Track how often the subject chose the target option [0, 1, 1, 0, 0]
    chose_target = ((mask_a_subset) & (subset['response'] == 0)) | ((~mask_a_subset) & (subset['response'] == 1))
    
    return float(chose_target.mean())

### RESULTS
- Predicted under pi_6 (simulated): 0.3233 (var=0.0299)
- Predicted under pi_5 (simulated): 0.4817 (var=0.0476)
- Observed on real data: 0.5200 (var=0.2496)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    is_1 = a_tuples == (1, 0, 0, 0)
    is_3 = a_tuples == (0, 1, 1, 1)
    is_5 = a_tuples == (1, 1, 0, 0)
    is_6 = a_tuples == (0, 0, 1, 1)
    
    correct = 0
    total = 0
    
    if is_1.any():
        correct += (data.loc[is_1, 'response'] == 0).sum()
        total += is_1.sum()
    if is_3.any():
        correct += (data.loc[is_3, 'response'] == 1).sum()
        total += is_3.sum()
    if is_5.any():
        correct += (data.loc[is_5, 'response'] == 0).sum()
        total += is_5.sum()
    if is_6.any():
        correct += (data.loc[is_6, 'response'] == 1).sum()
        total += is_6.sum()
        
    if total == 0:
        return 0.5
    return float(correct / total)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.4534 (var=0.0254)
**Predicted under pi_5:** 0.5437 (var=0.0229)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        tally_match.append(row['response'] == tally_choice)
        
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5594 (var=0.0385)
**Predicted under pi_5:** 0.4383 (var=0.0189)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None and row['response'] == ttb_pred:
            ttb_matches += 1
            
    return float(ttb_matches / total)
```

**Observed (real) value:** 0.6950 (var=0.1026)
**Predicted under pi_6:** 0.4215 (var=0.0238)
**Predicted under pi_5:** 0.4765 (var=0.0151)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Predicted under pi_6:** 0.5196 (var=0.0218)
**Predicted under pi_5:** 0.5262 (var=0.0235)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    relevant_trials = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # Check if one option has [1, 0, 0] on the first 3 features and the other has [0, 1, 1]
        a_top3 = [a[0], a[1], a[2]]
        b_top3 = [b[0], b[1], b[2]]
        if a_top3 == [1, 0, 0] and b_top3 == [0, 1, 1]:
            relevant_trials += 1
            if row['response'] == 1:
                wadd_consistent += 1
        elif a_top3 == [0, 1, 1] and b_top3 == [1, 0, 0]:
            relevant_trials += 1
            if row['response'] == 0:
                wadd_consistent += 1
    if relevant_trials == 0:
        return 0.5
    return wadd_consistent / relevant_trials
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.4234 (var=0.0123)
**Predicted under pi_5:** 0.5022 (var=0.0539)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where A=[1, 0, 0, 1, 1] and B=[0, 1, 1, 0, 0]
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1))
    if not mask.any():
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5600 (var=0.2464)
**Predicted under pi_6:** 0.3038 (var=0.0292)
**Predicted under pi_5:** 0.4587 (var=0.0270)

### Experiment 9
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Option A is [0, 1, 1, 0, 0] and Option B is [1, 0, 0, 1, 0]
    # Convert lists to tuples to make them hashable/comparable
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    trial_1_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 0))
    trial_1_data = data[trial_1_mask]
    
    if len(trial_1_data) == 0:
        return 0.5
        
    # Return the proportion of times Option B (response == 1) was chosen
    return float(trial_1_data['response'].mean())
```

**Observed (real) value:** 0.4000 (var=0.2400)
**Predicted under pi_6:** 0.6125 (var=0.0222)
**Predicted under pi_5:** 0.4825 (var=0.0430)

### Experiment 10
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    anti_ttb_count = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        if sum(a) == sum(b):
            if a[0] > b[0]:
                ttb_choice = 0
            elif b[0] > a[0]:
                ttb_choice = 1
            else:
                continue
            if row['response'] != ttb_choice:
                anti_ttb_count += 1
            total += 1
    if total == 0:
        return 0.0
    return anti_ttb_count / total
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.4125 (var=0.0194)
**Predicted under pi_5:** 0.5392 (var=0.0486)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 2 (Sequential Evidence Accumulation with Normalized Weights) generally provides closer predictions to the observed data than Theory 1 (Rank-Weighted Exponential Integration), particularly in Exps 2, 7, 8, 9, and 10. However, both theories fail significantly on Experiment 5, where the observed data shows a strong adherence to Take-The-Best (TTB) choices (~69.5%), while both theories predict much lower TTB consistency (~42-47%). This suggests that neither an exponential integration nor a normalized sequential accumulation perfectly captures the robust TTB-like behavior observed in some contexts, mixed with compensatory behavior in others.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Heuristic Mixture: Take-The-Best and Tallying' theory. Instead of a single complex integration mechanism, this theory posits that decision-makers probabilistically employ one of two classic heuristics on any given trial (or across the population): strict Take-The-Best (TTB) and simple equal-weight Tallying (WADD without cue weights). With probability `p_ttb`, the agent searches cues in validity order and stops at the first discriminating cue. With probability `1 - p_ttb`, the agent counts the total number of positive cues for each option and chooses the one with the higher tally, guessing randomly on ties. This mixture naturally accounts for the high TTB adherence in Exp 5 while allowing compensatory Tallying to pull choice probabilities toward 0.5 in highly conflicting trials."
}
```

## Usage

```json
{
  "prompt_token_count": 8825,
  "candidates_token_count": 354,
  "total_token_count": 10121
}
```
