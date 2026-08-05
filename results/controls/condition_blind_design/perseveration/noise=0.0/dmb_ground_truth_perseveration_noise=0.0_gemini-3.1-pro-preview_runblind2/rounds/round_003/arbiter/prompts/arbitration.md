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
Strategy Mixture Theory: The population of decision-makers is heterogeneous, consisting of distinct subgroups that rely on fundamentally different decision rules. Rather than every individual using a noisy version of the same heuristic, about half the population employs a non-compensatory, frugal strategy (Take The Best), while the other half employs a compensatory, unweighted strategy (Tallying). This individual difference accounts for both the ~50% aggregate choice proportions on critical divergence trials and the high (~0.25) between-subject variance, which is characteristic of a Bernoulli distribution of highly deterministic strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    strategy = int(parameters["strategy"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if strategy == 0:
        # Take The Best (Non-compensatory)
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner is None:
            scores = np.array([0.5, 0.5])
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
            
    else:
        # Tallying (Compensatory)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


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

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.74, 0.81, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 2: A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 8: A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 9: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1]  B=[1, 0, 0, 1]
  trial 11: A=[1, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 13: A=[1, 0, 1, 0]  B=[1, 0, 0, 1]
  trial 14: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 16: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
We isolate trials where the Tallying strategy predicts a tie (the sum of positive cues is equal for options A and B). On these trials, Tallying users (half the population in SMT) will guess randomly, yielding a ~50% match rate with the objective Take-The-Best (TTB) prediction. TTB users (the other half in SMT) will match the objective TTB prediction ~100% of the time. Thus, SMT predicts an aggregate match rate of ~75% on these trials. In contrast, HCH posits that all users employ TTB but with idiosyncratic, noisy cue hierarchies. Because the Tally-tie trials in this design perfectly balance the number of cues favoring each option, a noisy cue hierarchy will favor the objective TTB choice only ~50% of the time. Thus, HCH predicts an aggregate match rate of ~50%. This metric computes the proportion of choices on Tally-tie trials that align with the objective TTB prediction, creating a robust 75% vs 50% contrast.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.7353 (var=0.0626)
- Predicted under pi_5 (simulated): 0.5627 (var=0.1010)
- Observed on real data: 0.6080 (var=0.0783)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.95, 0.8, 0.74, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  trial 4: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 11: A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 13: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 14: A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  trial 15: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 16: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
We focus on 'Agreement Trials' where both Take-The-Best (with objective validities) and Tallying predict the exact same choice. Under Strategy Mixture Theory, all subjects (whether they use TTB or Tallying) will consistently choose this consensus option, leading to a very high match proportion (~97%) and extremely low between-subject variance. In contrast, under Heterogeneous Cue Hierarchies, subjects use idiosyncratic subjective validities. Because some of these Agreement Trials contain conflicting minor cues, subjects who heavily overweight a minor cue will reliably deviate from the consensus. This results in a structurally lower mean match proportion and a higher between-subject variance for HCH. The combination of a clear mean difference and near-zero variance in the Strategy Mixture group ensures a highly significant discrimination.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.9800 (var=0.0006)
- Predicted under pi_5 (simulated): 0.8290 (var=0.0113)
- Observed on real data: 0.5040 (var=0.0100)

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
**Predicted under pi_4:** 0.5211 (var=0.2280)
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
**Predicted under pi_4:** 0.4633 (var=0.2237)
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
**Predicted under pi_4:** 0.5167 (var=0.2247)
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
**Predicted under pi_4:** 0.3933 (var=0.2220)
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
**Predicted under pi_4:** 0.5767 (var=0.2302)
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
**Predicted under pi_4:** 0.7567 (var=0.0669)
**Predicted under pi_5:** 0.6000 (var=0.1856)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both theories fail profoundly on Experiment 2. The metric for Experiment 2 isolates 'agreement trials' where both Take-The-Best (TTB) and Tallying predict the exact same option. Both Theory 1 (Strategy Mixture) and Theory 2 (Heterogeneous Cue Hierarchies) predict very high alignment (0.98 and 0.83, respectively) with this consensus option. However, the observed data shows an aggregate match of exactly ~0.50 with very low between-subject variance (0.01). This means subjects are effectively guessing on these trials, or their choices are entirely orthogonal to the objective validities. Furthermore, in Experiments 4-8, the observed mean is ~0.50 but the variance is ~0.25, indicating that subjects are making highly deterministic but idiosyncratic choices (e.g., always choosing A or always choosing B, or relying on a single idiosyncratic cue that happens to favor A or B consistently for that subject). Neither TTB, Tallying, nor simple noise on objective weights can explain this pattern of complete decoupling from objective validities.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Subject-Specific Single-Cue Heuristic' or 'Strong Spatial Bias' theory. The data suggests subjects largely ignore the objective validities provided in the instructions. Instead, they either adopt a strong spatial bias (e.g., always choosing the left or right option) or pick a single, random feature to follow deterministically, ignoring all other features. This explains the 50% aggregate means across most experiments, the extremely high variance (0.25) on specific single trials (where half the population deterministically picks A and the other half B based on their chosen cue/bias), and the failure to align with objective-consensus trials in Experiment 2."
}
```

## Usage

```json
{
  "prompt_token_count": 9534,
  "candidates_token_count": 409,
  "total_token_count": 11171
}
```
