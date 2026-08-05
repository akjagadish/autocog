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
Subject-Specific Single-Cue Heuristic and Strong Spatial Bias: Subjects largely ignore the objective validities provided in the instructions. Instead of engaging in complex compensatory or non-compensatory integration of multiple cues, individuals adopt highly simplified, deterministic strategies. The population is split: some subjects rely on a strong spatial bias (e.g., always choosing the left or right option), while others adopt a single-cue heuristic, randomly selecting one feature to follow deterministically and ignoring all others. This extreme simplification explains the ~50% aggregate choice proportions on critical divergence trials, the failure to align with objective-consensus trials, and the extremely high between-subject variance, as choices are driven by idiosyncratic, deterministic biases rather than shared objective validities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    
    strategy_class = int(parameters["strategy_class"])
    spatial_dir = int(parameters["spatial_dir"])
    cue_weights = np.asarray(parameters["cue_weights"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    if strategy_class == 0:
        # Spatial Bias: deterministically choose Left (0) or Right (1)
        p_core = np.array([1.0, 0.0]) if spatial_dir == 0 else np.array([0.0, 1.0])
    else:
        # Single-Cue Heuristic: follow a single randomly preferred cue
        k = int(np.argmax(cue_weights))
        if a[k] > b[k]:
            p_core = np.array([1.0, 0.0])
        elif b[k] > a[k]:
            p_core = np.array([0.0, 1.0])
        else:
            # If the chosen cue is tied, guess randomly
            p_core = np.array([0.5, 0.5])
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_5
Heterogeneous Cue Hierarchies: Individuals all employ a lexicographic, non-compensatory decision strategy (Take-The-Best), but they differ in how they construct their cue hierarchies. Rather than perfectly internalizing the objective cue validities provided in instructions, subjects form subjective cue validities by combining the objective validities with idiosyncratic, subjective weights. The degree to which a subject relies on their idiosyncratic weights versus objective validities varies across the population. This results in some subjects strictly following the objective hierarchy, while others use idiosyncratic hierarchies (e.g., heavily overweighting a specific cue). This heterogeneity explains the high between-subject variance and why objectively dominant options in aggregate only receive partial choice shares.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    subjective_weights = np.asarray(parameters["subjective_weights"], dtype=float)
    noise_level = float(parameters["noise_level"])
    epsilon = float(parameters["epsilon"])
    
    # Construct subjective validities by adding individual-specific noise/preferences to objective validities
    # subjective_weights are in [0, 1], so we center them around 0
    subj_val = validities + noise_level * (subjective_weights - 0.5)
    
    # Lexicographic strategy based on subjective hierarchy
    cue_order = np.argsort(-subj_val, kind="stable").tolist()
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=4):** [0.95, 0.73, 0.56, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 3: A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 9: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 1, 0]  B=[1, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 13: A=[1, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 14: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 15: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Under the Advocated Theory, a significant portion of the population (those with a strategy_class of 0) adopts a strong spatial bias, meaning they will choose the same side (response = 0 or 1) on almost every single trial. This yields a subject-level mean response near 0.0 or 1.0, making their absolute deviation from 0.5 close to 0.5. The Competing Theory assumes all subjects use a lexicographic strategy based on subjective weights, which, given the balanced nature of the trial pairs, will result in an aggregate choice proportion closer to 0.5 for each subject. By computing the mean absolute deviation of each subject's overall response rate from 0.5, we expect a large, structural gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    subj_means = data.groupby('subject_id')['response'].mean()
    return float(np.mean(np.abs(subj_means - 0.5)))

### RESULTS
- Predicted under pi_6 (simulated): 0.3187 (var=0.0406)
- Predicted under pi_5 (simulated): 0.0704 (var=0.0023)
- Observed on real data: 0.5000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.95, 0.87, 0.82, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 4: A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 8: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 10: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 11: A=[1, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 13: A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 15: A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Measures the proportion of choices that align with the simple sum of features (Dawes' rule) on trials where one option has more positive features than the other. Under the Advocated Theory (Take-The-Best), choices will often align with the sum because the hierarchical non-compensatory strategy naturally correlates with the total number of positive cues. Under the Competing Theory, a large portion of subjects use a spatial bias (which matches the sum only 50% of the time) or a single-cue heuristic (which ignores the rest of the cues, reducing the overall alignment with the sum). Thus, the metric should be significantly higher for the Advocated Theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    diff_mask = sum_a != sum_b
    if not diff_mask.any():
        return 0.5
        
    resp = data.loc[diff_mask, 'response']
    sum_choice = (sum_b[diff_mask] > sum_a[diff_mask]).astype(int)
    
    match = (resp == sum_choice).mean()
    return float(match)

### RESULTS
- Predicted under pi_6 (simulated): 0.6321 (var=0.0264)
- Predicted under pi_5 (simulated): 0.7879 (var=0.0131)
- Observed on real data: 0.4982 (var=0.0021)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB winner
        val_order = [0, 2, 1, 3] # Indices in descending order of validity
        ttb_winner = None
        for j in val_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying winner
        a_wins = sum(1 for i in range(4) if a[i] > b[i])
        b_wins = sum(1 for i in range(4) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Check strict disagreement
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            total += 1
            if row['response'] == ttb_winner:
                ttb_match += 1
                
    if total == 0:
        return 0.5
    return ttb_match / total
```

**Observed (real) value:** 0.4667 (var=0.0267)
**Predicted under pi_6:** 0.3967 (var=0.0974)
**Predicted under pi_5:** 0.5289 (var=0.1492)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    mask = (a_tup == (0, 1, 1, 1)) & (b_tup == (1, 0, 1, 0))
    if not mask.any():
        return 0.5
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.3600 (var=0.2304)
**Predicted under pi_6:** 0.6000 (var=0.2156)
**Predicted under pi_5:** 0.5267 (var=0.1471)

### Experiment 5
**Design**
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trial 9, where Option A has ratings [0, 1, 1, 0] and Option B has [1, 0, 0, 0]
    def is_target(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        return a == [0, 1, 1, 0] and b == [1, 0, 0, 0]
    
    mask = data.apply(is_target, axis=1)
    target_trials = data[mask]
    
    if len(target_trials) == 0:
        return 0.5
        
    return float(target_trials['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Predicted under pi_6:** 0.5100 (var=0.2127)
**Predicted under pi_5:** 0.4333 (var=0.1644)

### Experiment 6
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 15 is the only trial where WADD and TTB diverge.
    # A = [0, 1, 1, 0], B = [1, 0, 0, 0]
    # WADD weights are [0.45, 0.27, 0.3, 0.05].
    # WADD score for A = 0.27 + 0.3 = 0.57
    # WADD score for B = 0.45
    # WADD prefers A (response = 0).
    # TTB looks at cue 0 first (validity 0.95), which favors B. 
    # TTB prefers B (response = 1).
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0)) & \
           data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0))
    if mask.sum() == 0:
        return 0.5
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Predicted under pi_6:** 0.4233 (var=0.1880)
**Predicted under pi_5:** 0.4667 (var=0.1400)

### Experiment 7
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask1 = (a_str == '1000') & (b_str == '0101')
    mask2 = (a_str == '0101') & (b_str == '1000')
    
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 0.5
        
    choices = 0
    if mask1.sum() > 0:
        choices += (data.loc[mask1, 'response'] == 0).sum()
    if mask2.sum() > 0:
        choices += (data.loc[mask2, 'response'] == 1).sum()
        
    return float(choices / total)
```

**Observed (real) value:** 0.4800 (var=0.2496)
**Predicted under pi_6:** 0.5067 (var=0.2100)
**Predicted under pi_5:** 0.5800 (var=0.1680)

### Experiment 8
**Design**
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_match = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0))
    b_match = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0))
    target_trials = data[a_match & b_match]
    if len(target_trials) == 0:
        return 0.5
    return float((target_trials['response'] == 0).mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Predicted under pi_6:** 0.5033 (var=0.2005)
**Predicted under pi_5:** 0.6000 (var=0.1856)

### Experiment 9
**Design**
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the sum of positive cues for each option to find Tally-tie trials
    a_sums = data['option_a_ratings'].apply(lambda x: sum(x))
    b_sums = data['option_b_ratings'].apply(lambda x: sum(x))
    tally_tie_mask = (a_sums == b_sums)
    
    if not tally_tie_mask.any():
        return 0.5
        
    tie_data = data[tally_tie_mask].copy()
    
    # Determine the objective Take-The-Best (TTB) choice
    # Objective validities are [0.95, 0.74, 0.81, 0.55], so cue order is 0, 2, 1, 3
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for cue in [0, 2, 1, 3]:
            if a[cue] > b[cue]:
                return 0
            elif b[cue] > a[cue]:
                return 1
        return -1
        
    tie_data['ttb_choice'] = tie_data.apply(get_ttb_choice, axis=1)
    
    valid_mask = tie_data['ttb_choice'] != -1
    if not valid_mask.any():
        return 0.5
        
    valid_tie_data = tie_data[valid_mask]
    matches = (valid_tie_data['response'] == valid_tie_data['ttb_choice']).astype(float)
    
    return float(matches.mean())
```

**Observed (real) value:** 0.6080 (var=0.0783)
**Predicted under pi_6:** 0.4707 (var=0.0804)
**Predicted under pi_5:** 0.5627 (var=0.1010)

### Experiment 10
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.8, 0.74, 0.55])
    cue_order = np.argsort(-validities)
    
    def get_ttb_pred(a, b):
        for j in cue_order:
            if a[j] > b[j]: return 0
            if b[j] > a[j]: return 1
        return 0.5
        
    def is_agreement_trial(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        sum_a = sum(a)
        sum_b = sum(b)
        
        if sum_a == sum_b:
            return False
            
        tally_pred = 0 if sum_a > sum_b else 1
        ttb_pred = get_ttb_pred(a, b)
        
        return tally_pred == ttb_pred

    # Calculate predictions and identify agreement trials
    data['ttb_pred'] = data.apply(lambda row: get_ttb_pred(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    data['is_agreement'] = data.apply(is_agreement_trial, axis=1)
    
    agreement_data = data[data['is_agreement']].copy()
    
    if len(agreement_data) == 0:
        return 0.0
        
    # Calculate match with the consensus prediction
    agreement_data['match'] = (agreement_data['response'] == agreement_data['ttb_pred']).astype(float)
    
    # Return the mean match proportion
    return float(agreement_data['match'].mean())
```

**Observed (real) value:** 0.5040 (var=0.0100)
**Predicted under pi_6:** 0.5850 (var=0.0119)
**Predicted under pi_5:** 0.8290 (var=0.0113)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The observed data across all experiments strongly points to a complete lack of cue integration or feature-based decision making. In Experiment 1, the observed metric is exactly 0.5000 (var=0.0000). Since the metric calculates the mean absolute deviation of each subject's response rate from 0.5, an observed value of 0.5 with 0 variance mathematically guarantees that every single subject exclusively chose Option A (0) or Option B (1) for the entire experiment. This deterministic spatial bias propagates through all other experiments, explaining why metrics measuring alignment with Dawes' rule (Exp 2: 0.4982), TTB (Exp 10: 0.5040), or specific feature patterns yield results hovering right around 0.50. Theory 1 (pi_6) partially captures this by including a spatial bias strategy, but dilutes it with a single-cue heuristic, leading to underpredictions of the spatial bias severity. Theory 2 (pi_5) completely fails because it assumes lexicographic feature integration (TTB), which strongly overpredicts rule alignment (e.g., predicting 0.829 in Exp 10 where 0.504 was observed).",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Pure Spatial Bias' theory. The data indicates that subjects completely ignore the objective validities and the feature values. Instead, 100% of the subjects adopt a deterministic spatial bias, choosing either exclusively Option A (Left) or exclusively Option B (Right) across all trials. The new model should assign a fixed spatial preference to each subject at the start of the experiment and predict that preference deterministically, with little to no epsilon noise. This will correctly capture the 0.500 MAD in Exp 1 and the ~0.50 baseline match rates for any feature-based rules in the other experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 10766,
  "candidates_token_count": 434,
  "total_token_count": 12645
}
```
