# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take The Best (TTB) posits that decision-makers consider features sequentially in order of their subjective validities and stop searching as soon as they find a feature that discriminates between the two options. The choice is then based entirely on this single cue, providing a non-compensatory heuristic alternative to compensatory models like WADD.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = False
    b_wins = False
    
    # Search for the first discriminating cue
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            a_wins = True
            break
        elif stim[1, idx] > stim[0, idx]:
            b_wins = True
            break
            
    if a_wins:
        p_core = np.array([1.0, 0.0])
    elif b_wins:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## THEORY 2 — pi_5
The Weighted Additive (WADD) model posits a fully compensatory decision strategy. Decision-makers evaluate all available cues for both options, weighting each cue by its subjective importance (operationalized as the log-odds of its validity). These weighted cues are integrated into a single compensatory score for each option. The option with the higher score is more likely to be chosen, with choices being probabilistic according to a softmax function over the scores. This provides a strong compensatory baseline to contrast with non-compensatory heuristics like Take The Best.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds to serve as weights
    validities = np.clip(validities, 1e-5, 1.0 - 1e-5)
    weights = np.log(validities / (1.0 - validities))
    
    # Calculate compensatory scores for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Convert scores to probabilities via softmax with max-subtraction for stability
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.63, 0.8, 0.58, 0.73, 0.68]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 2: A=[0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 5: A=[0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To maximally distinguish Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit the fundamental difference in their decision rules: TTB is purely non-compensatory and stops at the first discriminating feature, whereas WADD integrates all features using their log-odds validities as weights. We define a set of validities where the log-odds of the most valid feature is outweighed by the sum of the log-odds of several less valid features. By pitting the single highest-validity feature against a combination of lower-validity features, we create 'pure reversal' trials where TTB deterministically chooses the option favored by the single best cue, while WADD consistently prefers the option supported by the aggregate weight of the smaller cues. The validities are scrambled in their presentation order to ensure heuristics that rely on validity-ordering must actively re-sort them.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental trials were carefully designed such that the single most valid discriminating cue always favors Option A, while the sum of the remaining less valid cues always favors Option B. Thus, TTB universally predicts choosing Option A across all 8 trial types, whereas WADD universally predicts choosing Option B. The overall proportion of Option A choices perfectly separates the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    # The experimental design consists of 'pure reversal' trials where
    # Take The Best (TTB) deterministically prefers Option A (response == 0),
    # while Weighted Additive (WADD) consistently prefers Option B (response == 1).
    # We simply compute the proportion of times Option A was chosen.
    return float((data['response'] == 0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8731 (var=0.0061)
- Predicted under pi_5 (simulated): 0.2487 (var=0.0081)
- Observed on real data: 0.8450 (var=0.0103)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.75, 0.9, 0.65, 0.8, 0.7]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 2: A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Rationale:** To maximally distinguish the non-compensatory Take The Best (TTB) model from the compensatory Weighted Additive (WADD) model, we construct a set of validities where the log-odds weight of the highest validity feature is outweighed by the sum of the log-odds weights of several lower validity features. The validities are presented out of order to ensure heuristics that sort by validity must actively re-sort them. We design trials that pit the single most valid discriminating feature against combinations of less valid features, creating pure reversals where TTB deterministically chooses the option favored by the single best cue, while WADD chooses the option supported by the aggregate weight of the smaller cues. We also include control trials where both models agree.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses the option predicted by the WADD model specifically on 'reversal' trials—trials where the WADD model and the TTB model make opposing predictions. Under the TTB theory, subjects should almost never choose the WADD-predicted option on these trials (metric ≈ 0, modulo the lapse rate). Under the WADD theory, subjects should consistently choose the WADD-predicted option (metric ≈ 1). Restricting the analysis to these highly diagnostic reversal trials maximizes the expected difference between the two theories while minimizing within-subject noise.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.75, 0.9, 0.65, 0.8, 0.7])
    weights = np.log(validities / (1.0 - validities))
    order = np.argsort(validities)[::-1]
    
    wadd_matches_on_reversals = 0
    reversal_trials_total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_pred = -1
        for idx in order:
            if a[idx] > b[idx]:
                ttb_pred = 0
                break
            elif b[idx] > a[idx]:
                ttb_pred = 1
                break
        if ttb_pred == -1:
            ttb_pred = 0
            
        # WADD prediction
        score_a = np.sum(a * weights)
        score_b = np.sum(b * weights)
        wadd_pred = 0 if score_a > score_b else 1
        
        # Focus only on reversal trials where the two models disagree
        if ttb_pred != wadd_pred:
            reversal_trials_total += 1
            if row['response'] == wadd_pred:
                wadd_matches_on_reversals += 1
                
    if reversal_trials_total == 0:
        return 0.0
        
    return float(wadd_matches_on_reversals) / reversal_trials_total

### RESULTS
- Predicted under pi_3 (simulated): 0.1228 (var=0.0090)
- Predicted under pi_5 (simulated): 0.8419 (var=0.0137)
- Observed on real data: 0.1313 (var=0.0080)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    a_wadd = a_mat @ validities
    b_wadd = b_mat @ validities
    
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    wadd_prefers_a = a_wadd > b_wadd
    wadd_prefers_b = b_wadd > a_wadd
    
    disagree = (tally_prefers_a & wadd_prefers_b) | (tally_prefers_b & wadd_prefers_a)
    
    if not np.any(disagree):
        return 0.5
        
    tally_choice = np.where(tally_prefers_a, 0, 1)
    
    match = (data['response'].values[disagree] == tally_choice[disagree])
    return float(np.mean(match))
```

**Observed (real) value:** 0.1067 (var=0.0120)
**Predicted under pi_3:** 0.1250 (var=0.0106)
**Predicted under pi_5:** 0.1225 (var=0.0089)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_vals = np.stack(data['option_a_ratings'].values)
    b_vals = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where Tallying and WADD make strictly opposing predictions.
    # Trial 1: A=[0,0,1,1,1], B=[1,1,0,0,0] -> Tallying prefers A, WADD prefers B
    is_trial_1 = (a_vals[:, 0] == 0) & (a_vals[:, 2] == 1) & (b_vals[:, 0] == 1) & (b_vals[:, 2] == 0)
    
    # Trial 2: A=[1,1,0,0,0], B=[0,0,1,1,1] -> Tallying prefers B, WADD prefers A
    is_trial_2 = (a_vals[:, 0] == 1) & (a_vals[:, 2] == 0) & (b_vals[:, 0] == 0) & (b_vals[:, 2] == 1)
    
    mask = is_trial_1 | is_trial_2
    if not mask.any():
        return 0.5
        
    responses = data['response'].values
    wadd_aligned = (is_trial_1 & (responses == 1)) | (is_trial_2 & (responses == 0))
    
    return float(wadd_aligned[mask].mean())
```

**Observed (real) value:** 0.8649 (var=0.0063)
**Predicted under pi_3:** 0.8764 (var=0.0064)
**Predicted under pi_5:** 0.8684 (var=0.0084)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Approximate log-odds weights for WADD based on validities [0.88, 0.73, 0.73, 0.62, 0.62, 0.62]
    w = np.array([1.9924, 0.9946, 0.9946, 0.4895, 0.4895, 0.4895])
    
    tally_matches = []
    
    for _, row in data.iterrows():
        A = np.array(row['option_a_ratings'])
        B = np.array(row['option_b_ratings'])
        
        # Tallying counts strict wins
        A_tally = np.sum(A > B)
        B_tally = np.sum(B > A)
        
        # WADD uses weighted sums
        A_wadd = np.sum(A * w)
        B_wadd = np.sum(B * w)
        
        # 0 for A, 1 for B, -1 for tie
        tally_pref = 0 if A_tally > B_tally else (1 if B_tally > A_tally else -1)
        wadd_pref = 0 if A_wadd > B_wadd else (1 if B_wadd > A_wadd else -1)
        
        # We only care about trials where the two models make opposite strict predictions
        if tally_pref != -1 and wadd_pref != -1 and tally_pref != wadd_pref:
            if row['response'] == tally_pref:
                tally_matches.append(1.0)
            else:
                tally_matches.append(0.0)
                
    if len(tally_matches) == 0:
        return 0.5
        
    return float(np.mean(tally_matches))
```

**Observed (real) value:** 0.1467 (var=0.0156)
**Predicted under pi_3:** 0.1233 (var=0.0056)
**Predicted under pi_5:** 0.2737 (var=0.0121)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5333])
    weights = np.log(validities / (1.0 - validities))
    
    match_tally = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wadd = np.sum(a * weights)
        b_wadd = np.sum(b * weights)
        
        a_tally = np.sum(a > b)
        b_tally = np.sum(b > a)
        
        wadd_prefers_a = a_wadd > b_wadd + 1e-5
        wadd_prefers_b = b_wadd > a_wadd + 1e-5
        tally_prefers_a = a_tally > b_tally
        tally_prefers_b = b_tally > a_tally
        
        disagree = (wadd_prefers_a and tally_prefers_b) or (wadd_prefers_b and tally_prefers_a)
        
        if disagree:
            chose_a = (row['response'] == 0)
            chose_b = (row['response'] == 1)
            
            if tally_prefers_a and chose_a:
                match_tally.append(1.0)
            elif tally_prefers_b and chose_b:
                match_tally.append(1.0)
            else:
                match_tally.append(0.0)
                
    if len(match_tally) == 0:
        return 0.5
        
    return float(np.mean(match_tally))
```

**Observed (real) value:** 0.1133 (var=0.0069)
**Predicted under pi_3:** 0.1208 (var=0.0055)
**Predicted under pi_5:** 0.3104 (var=0.0089)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    mask = data['option_a_ratings'].apply(lambda x: sum(x)) == 1
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1780 (var=0.0177)
**Predicted under pi_3:** 0.1330 (var=0.0053)
**Predicted under pi_5:** 0.7910 (var=0.0136)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.85, 0.75, 0.7, 0.65, 0.6])
    weights = np.log(validities / (1.0 - validities))
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    score_a = A @ weights
    score_b = B @ weights
    
    wadd_pred = (score_b > score_a).astype(int)
    
    return float((data['response'].values == wadd_pred).mean())
```

**Observed (real) value:** 0.1713 (var=0.0123)
**Predicted under pi_3:** 0.1398 (var=0.0066)
**Predicted under pi_5:** 0.7900 (var=0.0105)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agree_aligned = 0
    agree_total = 0
    oppose_aligned = 0
    oppose_total = 0
    
    for _, row in data.iterrows():
        a_rat = np.array(row['option_a_ratings'])
        b_rat = np.array(row['option_b_ratings'])
        diff = a_rat - b_rat
        
        discr_idx = np.where(diff != 0)[0]
        if len(discr_idx) < 2:
            continue
            
        first_idx = discr_idx[0]
        second_idx = discr_idx[1]
        
        is_agree = (diff[first_idx] == diff[second_idx])
        
        first_favors_A = (diff[first_idx] > 0)
        chose_A = (row['response'] == 0)
        is_aligned = (first_favors_A == chose_A)
        
        if is_agree:
            agree_aligned += int(is_aligned)
            agree_total += 1
        else:
            oppose_aligned += int(is_aligned)
            oppose_total += 1
            
    if agree_total == 0 or oppose_total == 0:
        return 0.0
        
    # Haldane-Anscombe correction to handle 100% alignment and stabilize variance
    log_odds_agree = np.log((agree_aligned + 0.5) / (agree_total - agree_aligned + 0.5))
    log_odds_oppose = np.log((oppose_aligned + 0.5) / (oppose_total - oppose_aligned + 0.5))
    
    return float(log_odds_agree - log_odds_oppose)

```

**Observed (real) value:** 0.1341 (var=0.5478)
**Predicted under pi_3:** 0.0123 (var=1.0955)
**Predicted under pi_5:** 1.4131 (var=0.7275)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agree_match = []
    disagree_match = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        disc_cues = []
        for i in range(5):
            if a[i] > b[i]:
                disc_cues.append(0)
            elif b[i] > a[i]:
                disc_cues.append(1)
                
        if len(disc_cues) >= 2:
            first = disc_cues[0]
            second = disc_cues[1]
            
            chose_first = (resp == first)
            
            if first == second:
                agree_match.append(chose_first)
            else:
                disagree_match.append(chose_first)
                
    if len(agree_match) == 0 or len(disagree_match) == 0:
        return 0.0
        
    return float(np.mean(agree_match) - np.mean(disagree_match))

```

**Observed (real) value:** -0.0219 (var=0.0059)
**Predicted under pi_3:** -0.0087 (var=0.0042)
**Predicted under pi_5:** 0.5009 (var=0.0143)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 10 experiments, Theory 1 (Take The Best, pi_3) consistently and accurately predicts the observed data, often matching the point estimates very closely. Theory 2 (Weighted Additive, pi_5) fails spectacularly on almost all experiments, particularly on trials designed to pit non-compensatory against compensatory decision-making (e.g., Experiments 1, 2, 7, and 8, where WADD predicts values around 0.8 while the observed data is around 0.1-0.2, perfectly in line with TTB). Because WADD is systematically contradicted by the data, it is a degenerate theory in this domain.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Weighted Additive (WADD) theory with a brand-new theory. Given that pure compensatory integration (WADD) fails but subjects might not be perfectly deterministic TTB users, a good alternative could be a 'Strategy Mixture Model' (where a dominant fraction of subjects use TTB and a smaller fraction use a simple compensatory heuristic like Tallying), or 'Elimination by Aspects' (EBA), which provides a different non-compensatory mechanism. This will provide a much stronger and more realistic competitor to TTB than WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 9745,
  "candidates_token_count": 289,
  "total_token_count": 11317
}
```
