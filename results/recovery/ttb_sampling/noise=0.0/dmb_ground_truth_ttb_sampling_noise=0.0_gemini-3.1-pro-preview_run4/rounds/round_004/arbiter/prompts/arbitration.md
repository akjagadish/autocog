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
People use a non-compensatory, lexicographic heuristic known as 'Take-The-Best' (TTB) to make decisions. Instead of integrating all available information (as in weighted additive models) or counting all positive features (as in tallying), decision-makers search through features sequentially in descending order of their validity. The search stops at the very first feature that discriminates between the two options (i.e., one option possesses the feature and the other does not). The option that wins on this single discriminating cue is chosen deterministically, and all remaining lower-validity features are completely ignored. If no features discriminate between the options, the decision-maker guesses randomly. Response noise is modeled purely as a lapse rate (epsilon) where the subject occasionally makes a random guess instead of executing the TTB strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Take-The-Best expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(-validities)
    
    # Default to guessing if all features tie
    p_core = np.array([0.5, 0.5])
    
    # Search through features in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_6
Decision-makers in multi-attribute choice are heterogeneous in their strategy use. A large majority of the population relies on the non-compensatory 'Take-The-Best' (TTB) heuristic, which sequentially searches cues by validity and stops at the first discriminating feature. However, a small subset of the population uses a compensatory 'Tallying' strategy, integrating information by simply counting the number of winning features for each option. This mixture preserves the predominantly non-compensatory nature of the population's choices (keeping conflict-vs-alignment effects near zero) while probabilistically accounting for the slight elevation in compensatory choices observed in specific trial types. Both strategies are subject to a uniform lapse rate (epsilon).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Prediction
    order = np.argsort(-validities)
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying Prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally_pred = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally_pred = np.array([0.0, 1.0])
    else:
        p_tally_pred = np.array([0.5, 0.5])
        
    # Probabilistic mixture to smooth the optimization landscape
    p_tally = float(parameters["p_tally"])
    p_core = (1.0 - p_tally) * p_ttb + p_tally * p_tally_pred
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.51]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 9: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]

**Rationale:** To quantitatively dissociate pure Take-The-Best (TTB) from a TTB-Tallying Mixture, we manipulate the agreement between the highest-validity discriminating cue and the overall count of positive cues. Pure TTB predicts a constant choice probability (driven solely by the uniform lapse rate epsilon) for the option favored by the highest-validity discriminating cue, regardless of how many lower-validity cues favor the other option. In contrast, the Strategy Mixture model predicts a systematic drop in choice probability for the TTB-favored option on 'conflict' trials (where Tallying favors the opposite option) compared to 'alignment' trials (where both TTB and Tallying favor the same option). By including trials with direct conflict, perfect alignment, and ties in the Tallying metric, we can precisely identify whether deviations from TTB are purely random noise or driven by a compensatory Tallying process.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
The metric computes the difference in the proportion of 'Take-The-Best'-compliant choices between 'alignment' trials (where Tallying and TTB predict the same choice) and 'conflict' trials (where Tallying predicts the opposite choice). Pure TTB predicts this difference to be roughly zero, as the lapse rate is uniform across trial types. In contrast, the Mixture model (TTB + Tallying) predicts a positive difference, as Tallying increases the choice probability of the TTB-favored option on alignment trials and decreases it on conflict trials.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_compliant = []
    trial_type = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction (cues are already in descending validity order 0 to 5)
        ttb_pred = None
        for idx in range(6):
            if a[idx] > b[idx]:
                ttb_pred = 0
                break
            elif b[idx] > a[idx]:
                ttb_pred = 1
                break
        
        # Tallying prediction
        a_wins = sum(1 for i in range(6) if a[i] > b[i])
        b_wins = sum(1 for i in range(6) if b[i] > a[i])
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            tally_pred = -1
            
        is_ttb = (row['response'] == ttb_pred)
        ttb_compliant.append(is_ttb)
        
        if tally_pred == ttb_pred:
            trial_type.append(1) # Alignment
        elif tally_pred == -1:
            trial_type.append(0) # Tie
        else:
            trial_type.append(-1) # Conflict
            
    df = pd.DataFrame({'ttb_compliant': ttb_compliant, 'trial_type': trial_type})
    
    align_rate = df[df['trial_type'] == 1]['ttb_compliant'].mean()
    conflict_rate = df[df['trial_type'] == -1]['ttb_compliant'].mean()
    
    # Handle cases where some trial types might be missing (e.g., small slices)
    if pd.isna(align_rate):
        align_rate = 0.0
    if pd.isna(conflict_rate):
        conflict_rate = 0.0
        
    return float(align_rate - conflict_rate)

### RESULTS
- Predicted under pi_3 (simulated): 0.0056 (var=0.0043)
- Predicted under pi_6 (simulated): 0.0606 (var=0.0054)
- Observed on real data: 0.0000 (var=0.0040)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the pure Take-The-Best (TTB) model from the TTB-Tallying Mixture model, we manipulate the agreement between the highest-validity discriminating cue and the overall count of positive cues. Pure TTB predicts a constant choice probability (driven solely by the uniform lapse rate epsilon) for the option favored by the first discriminating cue, regardless of how many lower-validity cues favor the alternative. In contrast, the Strategy Mixture model predicts a systematic drop in choice probability for the TTB-favored option on 'conflict' trials (where Tallying favors the opposite option) compared to 'alignment' trials (where both TTB and Tallying favor the same option). By including trials with direct conflict, perfect alignment, and ties in the Tallying metric, we can precisely identify whether deviations from TTB are purely random noise or driven by a compensatory Tallying process.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in adherence to the Take-The-Best (TTB) heuristic between 'alignment' trials (where TTB and Tallying strategies favor the same option) and 'conflict' trials (where Tallying favors the opposite option). Under the pure TTB model, choices are strictly determined by the first discriminating cue regardless of Tallying, meaning the TTB adherence rate should be identical in both trial types (difference = 0). Under the Strategy Mixture model, the compensatory Tallying strategy pulls choices away from TTB on conflict trials but reinforces TTB on alignment trials, leading to a consistently positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    align_matches = 0
    align_total = 0
    conflict_matches = 0
    conflict_total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        resp = row['response']
        
        # TTB prediction (validities are strictly descending index 0 to 4)
        ttb_pred = -1
        for idx in range(5):
            if a[idx] > b[idx]:
                ttb_pred = 0
                break
            elif b[idx] > a[idx]:
                ttb_pred = 1
                break
                
        # Tally prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            tally_pred = -1
            
        if ttb_pred == -1:
            continue
            
        is_ttb_match = 1 if resp == ttb_pred else 0
        
        if tally_pred == ttb_pred:
            align_matches += is_ttb_match
            align_total += 1
        elif tally_pred != -1 and tally_pred != ttb_pred:
            conflict_matches += is_ttb_match
            conflict_total += 1
            
    p_align = align_matches / align_total if align_total > 0 else 0.5
    p_conflict = conflict_matches / conflict_total if conflict_total > 0 else 0.5
    
    return float(p_align - p_conflict)

### RESULTS
- Predicted under pi_3 (simulated): 0.0007 (var=0.0029)
- Predicted under pi_6 (simulated): 0.0567 (var=0.0047)
- Observed on real data: -0.0153 (var=0.0130)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    t1_data = data[is_t1]
    t2_data = data[is_t2]
    
    t1_match = (t1_data['response'] == 0).sum()
    t2_match = (t2_data['response'] == 1).sum()
    
    total = len(t1_data) + len(t2_data)
    
    if total == 0:
        return 0.5
    return float(t1_match + t2_match) / total

```

**Observed (real) value:** 0.1062 (var=0.0063)
**Predicted under pi_3:** 0.1212 (var=0.0080)
**Predicted under pi_6:** 0.1562 (var=0.0073)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_choices = 0
    total_critical = 0
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        
        sum_a = sum(a)
        sum_b = sum(b)
        sum_a_top = sum(a[:3])
        sum_b_top = sum(b[:3])
        
        # Check if this is a critical trial: one option has exactly 2 high-validity features, 
        # while the other has exactly 3 low-validity features.
        if sum_a == 2 and sum_a_top == 2 and sum_b == 3 and sum_b_top == 0:
            total_critical += 1
            if row['response'] == 1:
                tally_choices += 1
        elif sum_b == 2 and sum_b_top == 2 and sum_a == 3 and sum_a_top == 0:
            total_critical += 1
            if row['response'] == 0:
                tally_choices += 1
                
    return tally_choices / total_critical if total_critical > 0 else 0.5
```

**Observed (real) value:** 0.2022 (var=0.0207)
**Predicted under pi_3:** 0.1183 (var=0.0055)
**Predicted under pi_6:** 0.1300 (var=0.0053)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
                
        if pred is not None:
            if resp == pred:
                matches += 1
            total += 1
            
    return matches / total if total > 0 else 0.5
```

**Observed (real) value:** 0.8579 (var=0.0072)
**Predicted under pi_3:** 0.8931 (var=0.0064)
**Predicted under pi_6:** 0.8712 (var=0.0037)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_more_than_a = data.apply(lambda row: sum(row['option_b_ratings']) > sum(row['option_a_ratings']), axis=1)
    if not b_more_than_a.any():
        return 0.0
    return float(data.loc[b_more_than_a, 'response'].mean())
```

**Observed (real) value:** 0.1383 (var=0.0137)
**Predicted under pi_3:** 0.1246 (var=0.0075)
**Predicted under pi_6:** 0.1458 (var=0.0050)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_choices(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
                
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally = 0
        elif b_wins > a_wins:
            tally = 1
        else:
            tally = -1
            
        return pd.Series({'ttb': ttb, 'tally': tally})
        
    choices = data.apply(get_choices, axis=1)
    ttb_choice = choices['ttb']
    tally_choice = choices['tally']
    resp = data['response']
    
    disagree_mask = (ttb_choice != -1) & (tally_choice != -1) & (ttb_choice != tally_choice)
    agree_mask = (ttb_choice != -1) & (tally_choice != -1) & (ttb_choice == tally_choice)
    
    if disagree_mask.sum() == 0 or agree_mask.sum() == 0:
        return 0.0
        
    disagree_error_rate = (resp[disagree_mask] != ttb_choice[disagree_mask]).mean()
    agree_error_rate = (resp[agree_mask] != ttb_choice[agree_mask]).mean()
    
    return float(disagree_error_rate - agree_error_rate)
```

**Observed (real) value:** -0.0300 (var=0.0162)
**Predicted under pi_3:** 0.0044 (var=0.0067)
**Predicted under pi_6:** 0.0486 (var=0.0091)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Agreement trial: Both TTB and Tallying favor Option A
    t2 = data[(data['A_str'] == '11100') & (data['B_str'] == '00011')]
    
    # Conflict trials: TTB favors Option A, but Tallying favors Option B
    conflict_trials = data[
        ((data['A_str'] == '10000') & (data['B_str'] == '01111')) |
        ((data['A_str'] == '11000') & (data['B_str'] == '10111')) |
        ((data['A_str'] == '00100') & (data['B_str'] == '00011'))
    ]
    
    if len(t2) == 0 or len(conflict_trials) == 0:
        return 0.0
        
    p_A_agree = (t2['response'] == 0).mean()
    p_A_conflict = (conflict_trials['response'] == 0).mean()
    
    return float(p_A_agree - p_A_conflict)
```

**Observed (real) value:** 0.0083 (var=0.0117)
**Predicted under pi_3:** 0.0008 (var=0.0100)
**Predicted under pi_6:** 0.0629 (var=0.0071)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_acc(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 1, 1, 1) and b == (0, 0, 0, 0, 0):
            return 1 if resp == 0 else 0
        elif a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 1, 1):
            return 1 if resp == 0 else 0
        elif a == (0, 1, 1, 1, 1) and b == (1, 0, 0, 0, 0):
            return 1 if resp == 1 else 0
        return None
        
    accs = data.apply(get_acc, axis=1)
    
    is_align = data.apply(lambda r: tuple(r['option_a_ratings']) == (1, 1, 1, 1, 1) and tuple(r['option_b_ratings']) == (0, 0, 0, 0, 0), axis=1)
    is_conflict = data.apply(lambda r: (tuple(r['option_a_ratings']) == (1, 0, 0, 0, 0) and tuple(r['option_b_ratings']) == (0, 1, 1, 1, 1)) or (tuple(r['option_a_ratings']) == (0, 1, 1, 1, 1) and tuple(r['option_b_ratings']) == (1, 0, 0, 0, 0)), axis=1)
    
    p_align = accs[is_align].mean() if is_align.sum() > 0 else 0.0
    p_conflict = accs[is_conflict].mean() if is_conflict.sum() > 0 else 0.0
    
    return float(p_align - p_conflict)
```

**Observed (real) value:** 0.0117 (var=0.0244)
**Predicted under pi_3:** -0.0167 (var=0.0178)
**Predicted under pi_6:** 0.0750 (var=0.0129)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = np.array(data['response'])
    
    # Find first discriminating cue index for each trial
    diff = a_ratings - b_ratings
    abs_diff = np.abs(diff)
    disc_idx = np.argmax(abs_diff, axis=1)
    
    # TTB prediction: if a > b (diff > 0), predict A (0); else predict B (1)
    first_diff = diff[np.arange(len(diff)), disc_idx]
    ttb_pred = np.where(first_diff > 0, 0, 1)
    
    correct = (responses == ttb_pred).astype(float)
    
    # Compare accuracy on trials where discrimination happens early vs late
    early_mask = (disc_idx == 0) | (disc_idx == 1)
    late_mask = (disc_idx == 3) | (disc_idx == 4)
    
    if np.sum(early_mask) == 0 or np.sum(late_mask) == 0:
        return 0.0
        
    early_acc = np.mean(correct[early_mask])
    late_acc = np.mean(correct[late_mask])
    
    return float(early_acc - late_acc)

```

**Observed (real) value:** -0.0400 (var=0.0058)
**Predicted under pi_3:** -0.0021 (var=0.0072)
**Predicted under pi_6:** -0.0017 (var=0.0037)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of experiments (especially Exps 1, 2, 7, 8, and 9), the data strongly supports pure Take-The-Best (Theory 1) over the TTB-Tallying Mixture (Theory 2). The critical metric in these experiments measures the difference in TTB compliance between 'alignment' trials (where Tallying agrees with TTB) and 'conflict' trials (where Tallying disagrees). Theory 2 consistently predicts a substantial positive difference (~0.05 to ~0.07) because the compensatory Tallying strategy pulls choices away from the TTB prediction on conflict trials. However, the observed data consistently shows this difference is near zero or slightly negative, perfectly matching the predictions of the pure TTB model with a uniform lapse rate. While Theory 1 is highly accurate for these metrics, it fails to capture some nuances, such as the negative difference in accuracy between early and late discriminating cues seen in Experiment 10, or the specific choice rates in Experiment 4. Theory 2 is fundamentally misaligned with the core behavioral pattern of the population and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory to replace the TTB-Tallying mixture. Since pure TTB (Theory 1) captures the lack of conflict/alignment differences but misses depth-dependent effects (like Exp 10, where early vs. late discrimination yields different accuracies), the new theory should maintain a strictly non-compensatory or sequential search architecture but introduce a more sophisticated noise or stopping rule. For example, propose a 'Depth-Dependent TTB' where the probability of a lapse or execution error scales with the number of cues searched (search depth), or a 'Probabilistic Cue' model where decision-makers use a softmax choice rule based on the validity of the first discriminating cue rather than a constant uniform lapse. This would explain why accuracy varies with which cue discriminates without introducing a compensatory tallying mechanism."
}
```

## Usage

```json
{
  "prompt_token_count": 10523,
  "candidates_token_count": 440,
  "total_token_count": 12309
}
```
