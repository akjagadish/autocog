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
Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_6
Evidence-Dependent Noise: Decision-makers evaluate options using a single compensatory mechanism (Weighted Additive) where features are weighted by a subjective transformation of their validities. However, the decision process is subject to evidence-dependent noise: the temperature of the softmax choice rule scales with the total conflict between the options (defined as the total weighted evidence of features that differ between the two options). This ensures that trials with higher evidence magnitudes or greater feature conflict naturally generate higher decision noise. This single-mechanism approach preserves log-odds linearity while explaining why extremeness flattens across trials with varying evidence magnitudes.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus expects shape (2, n_features); got {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective feature weighting
    weights = val ** gamma
    
    # Calculate weighted sums of evidence for each option
    scores = np.dot(stim, weights)
    
    # Calculate conflict: total weight of features where the options differ
    diff = np.abs(stim[0] - stim[1])
    conflict = np.dot(diff, weights)
    
    # Effective beta scales inversely with conflict (higher conflict = more noise)
    beta_eff = beta / (1.0 + theta * conflict)
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta_eff * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=8):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  trial 4: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  trial 5: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates pure Weighted Additive (WADD) from Evidence-Dependent Noise (EDN) by exploiting a 'magnitude effect' reversal. We construct a sequence of trial pairs where the net evidence favoring Option A strictly increases (by adding pairs of features where A wins on the slightly higher validity), but the total feature conflict increases at a much faster rate. Because pure WADD's choice probabilities depend monotonically on the net weighted evidence, it MUST predict that preference for Option A becomes stronger as more net evidence is added, regardless of the gamma or beta parameters. In contrast, EDN posits that decision noise scales with total feature conflict. Because the conflict grows massively relative to the small gains in net evidence, EDN predicts a preference reversal: the choice probability for Option A will regress toward 0.5 on the high-conflict trials, despite the larger net evidence in its favor. This structural divergence provides a definitive test between the two theories.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By pooling four trials with low net evidence and low/moderate conflict (Group 1) and contrasting them with two trials that have strictly higher net evidence but massively higher conflict (Group 2), we greatly reduce the per-subject variance while isolating the 'magnitude effect reversal'. Pure WADD predicts that Group 2 MUST have a higher probability of choosing Option A than Group 1 because its net evidence is structurally larger. Evidence-Dependent Noise predicts that the massive increase in total conflict in Group 2 will overwhelm the net evidence gain, drastically inflating decision noise and pulling the choice probability toward 0.5. This structural divergence provides a robust, low-variance discriminator between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A'] = data['option_a_ratings'].apply(tuple)
    data['B'] = data['option_b_ratings'].apply(tuple)
    data['pair'] = list(zip(data['A'], data['B']))
    
    # Group 1: Low Net Evidence, Low/Moderate Conflict
    # T1, T2, T5, T6
    group1 = {
        ((0,0,0,0,1,0,0,0), (0,0,0,0,0,1,0,0)),
        ((1,0,0,0,1,0,0,0), (0,1,0,0,0,1,0,0)),
        ((0,0,1,0,0,0,0,0), (0,0,0,1,0,0,0,0)),
        ((0,0,1,0,0,0,1,0), (0,0,0,1,0,0,0,1))
    }
    
    # Group 2: Higher Net Evidence, High Conflict
    # T3, T4
    group2 = {
        ((1,0,1,0,1,0,0,0), (0,1,0,1,0,1,0,0)),
        ((1,0,1,0,1,0,1,0), (0,1,0,1,0,1,0,1))
    }
    
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        g1_mask = subj_df['pair'].isin(group1)
        g2_mask = subj_df['pair'].isin(group2)
        
        if g1_mask.sum() > 0 and g2_mask.sum() > 0:
            p_g1 = subj_df.loc[g1_mask, 'chose_A'].mean()
            p_g2 = subj_df.loc[g2_mask, 'chose_A'].mean()
            diffs.append(p_g2 - p_g1)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))

### RESULTS
- Predicted under pi_3 (simulated): 0.0350 (var=0.0175)
- Predicted under pi_6 (simulated): -0.0017 (var=0.0182)
- Observed on real data: -0.0458 (var=0.0349)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.6]

**Trial pairs (n=18):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0]
  trial 10: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 11: A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 12: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 13: A=[0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 14: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 15: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 16: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 17: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 18: A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1]

**Rationale:** This design quantitatively dissociates pure WADD from Evidence-Dependent Noise (EDN) by testing for a 'Bifurcation of the Psychometric Function'. We construct a spectrum of trials ranging from strongly favoring Option B to strongly favoring Option A, using three primary evidence cues. We then duplicate this entire spectrum under two conditions: a Low Conflict condition (where two additional matched-validity cues are tied) and a High Conflict condition (where the matched-validity cues are split between the options). Because the split cues have identical validities, they contribute zero net evidence under any validity transformation (gamma). Consequently, pure WADD dictates that all trials must collapse onto a single, unified psychometric curve governed solely by net evidence. In contrast, EDN predicts a structural bifurcation: the High Conflict condition will produce a systematically flatter psychometric curve (lower slope) than the Low Conflict condition, as the persistent background conflict suppresses the effective decision temperature.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
This metric isolates the trials with the absolute highest net evidence (T8, T9, T17, T18). The Competing Theory (WADD) posits a high base lapse rate (epsilon up to 0.5) and relatively low sensitivity (beta up to 5.0), meaning that even on the most obvious trials, accuracy is heavily capped and choices remain noisy. In contrast, the Advocated Theory (EDN) posits a much lower base lapse rate (epsilon up to 0.1) and higher sensitivity (beta up to 20.0). Even with conflict penalties, the sheer magnitude of net evidence on these specific trials allows EDN to achieve near-perfect accuracy. By focusing exclusively on these 'easy' trials, we strip away the variance introduced by the 'hard' trials where both models predict performance near chance, resulting in a large and highly reliable accuracy gap between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # T8: A wins
    m_t8 = (data['A_str'] == '11100') & (data['B_str'] == '00000')
    # T9: B wins
    m_t9 = (data['A_str'] == '00000') & (data['B_str'] == '11100')
    # T17: A wins
    m_t17 = (data['A_str'] == '11110') & (data['B_str'] == '00001')
    # T18: B wins
    m_t18 = (data['A_str'] == '00010') & (data['B_str'] == '11101')
    
    acc_8 = (data[m_t8]['response'] == 0).mean()
    acc_9 = (data[m_t9]['response'] == 1).mean()
    acc_17 = (data[m_t17]['response'] == 0).mean()
    acc_18 = (data[m_t18]['response'] == 1).mean()
    
    accs = [a for a in [acc_8, acc_9, acc_17, acc_18] if not np.isnan(a)]
    if len(accs) == 0:
        return 0.5
        
    return float(np.mean(accs))

### RESULTS
- Predicted under pi_3 (simulated): 0.8460 (var=0.0108)
- Predicted under pi_6 (simulated): 0.7510 (var=0.0329)
- Observed on real data: 0.8720 (var=0.0112)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    cue_order = np.argsort(-validities)
    
    ttb_matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            if row['response'] == ttb_winner:
                ttb_matches += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_matches / total)

```

**Observed (real) value:** 0.3292 (var=0.0132)
**Predicted under pi_3:** 0.3406 (var=0.0096)
**Predicted under pi_6:** 0.4765 (var=0.0105)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        tally_a = sum(1 for x, y in zip(a, b) if x > y)
        tally_b = sum(1 for x, y in zip(a, b) if y > x)
        
        if tally_a > tally_b:
            tally_pred = 0
        elif tally_b > tally_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        ttb_pred = None
        for x, y in zip(a, b):
            if x > y:
                ttb_pred = 0
                break
            elif y > x:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            agreements.append(1 if resp == tally_pred else 0)
            
    if not agreements:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.7356 (var=0.0121)
**Predicted under pi_3:** 0.7931 (var=0.0137)
**Predicted under pi_6:** 0.5833 (var=0.0090)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify trials where Tallying sees an exact tie (2 wins vs 2 wins)
    t4_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 0))
    t5_mask = (a_tuples == (0, 0, 1, 1, 0)) & (b_tuples == (1, 1, 0, 0, 0))
    t6_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1))
    
    # Under WADD, these trials are NOT ties, because the validities are strictly ordered:
    # w0 > w2 and w1 > w3, etc. 
    # WADD strictly prefers A in T4, B in T5, and A in T6 regardless of the gamma parameter.
    
    c4 = np.sum((data['response'] == 0) & t4_mask)
    c5 = np.sum((data['response'] == 1) & t5_mask)
    c6 = np.sum((data['response'] == 0) & t6_mask)
    
    total = np.sum(t4_mask) + np.sum(t5_mask) + np.sum(t6_mask)
    
    if total == 0:
        return 0.5
        
    return float((c4 + c5 + c6) / total)
```

**Observed (real) value:** 0.6989 (var=0.0397)
**Predicted under pi_3:** 0.7222 (var=0.0223)
**Predicted under pi_6:** 0.7100 (var=0.0265)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    t1_optA = (0, 0, 1, 1, 1)
    t1_optB = (1, 1, 0, 0, 0)
    
    t5_optA = (0, 1, 1, 1, 0)
    t5_optB = (1, 0, 0, 0, 1)
    
    t1_choices = []
    t5_choices = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t1_optA and b == t1_optB:
            t1_choices.append(1 if resp == 0 else 0)
        elif a == t1_optB and b == t1_optA:
            t1_choices.append(1 if resp == 1 else 0)
            
        elif a == t5_optA and b == t5_optB:
            t5_choices.append(1 if resp == 0 else 0)
        elif a == t5_optB and b == t5_optA:
            t5_choices.append(1 if resp == 1 else 0)
            
    if not t1_choices or not t5_choices:
        return 0.0
        
    return float(np.mean(t5_choices) - np.mean(t1_choices))
```

**Observed (real) value:** 0.1850 (var=0.2183)
**Predicted under pi_3:** 0.2562 (var=0.0685)
**Predicted under pi_6:** 0.1800 (var=0.0437)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Project sequence column to string for hashable equality checks
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    def subj_score(df):
        # Identify the extreme trials (T7 and T8)
        m7 = df['a_str'] == '10110'
        m8 = df['a_str'] == '01001'
        
        # Identify the weak/tie trials (T5 and T6)
        m5 = df['a_str'] == '10001'
        m6 = df['a_str'] == '01100'
        
        # Calculate probability of choosing Option A on each trial
        p7 = float((df.loc[m7, 'response'] == 0).mean()) if m7.any() else 0.5
        p8 = float((df.loc[m8, 'response'] == 0).mean()) if m8.any() else 0.5
        p5 = float((df.loc[m5, 'response'] == 0).mean()) if m5.any() else 0.5
        p6 = float((df.loc[m6, 'response'] == 0).mean()) if m6.any() else 0.5
        
        # Calculate 'extremeness' (absolute deviation from indifference)
        ext7 = abs(p7 - 0.5)
        ext8 = abs(p8 - 0.5)
        ext5 = abs(p5 - 0.5)
        ext6 = abs(p6 - 0.5)
        
        # Contrast extreme trials vs weak trials
        return (ext7 + ext8) - (ext5 + ext6)
        
    data_copy = data.copy()
    data_copy['a_str'] = a_str
    
    if 'subject_id' in data_copy.columns:
        return float(data_copy.groupby('subject_id').apply(subj_score).mean())
    return subj_score(data_copy)
```

**Observed (real) value:** 0.0167 (var=0.1200)
**Predicted under pi_3:** 0.4017 (var=0.0430)
**Predicted under pi_6:** 0.2150 (var=0.0722)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_t56_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trials 5 and 6 pit {1, 2} against {3, 4, 5}
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return row['response'] == 0
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return row['response'] == 1
            
        return np.nan
        
    choices = data.apply(get_t56_choice, axis=1)
    
    df_temp = pd.DataFrame({
        'subject_id': data['subject_id'],
        'choice': choices
    }).dropna()
    
    if len(df_temp) == 0:
        return 0.0
        
    # Calculate each subject's proportion of choosing the {1, 2} option
    subj_means = df_temp.groupby('subject_id')['choice'].mean()
    
    # Calculate extremeness as the squared deviation from 0.5
    extremeness = (subj_means - 0.5) ** 2
    
    return float(extremeness.mean())
```

**Observed (real) value:** 0.0750 (var=0.0053)
**Predicted under pi_3:** 0.0397 (var=0.0026)
**Predicted under pi_6:** 0.0306 (var=0.0016)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # These 6 trials all share the exact same Tallying difference (+1 for Option A).
    # However, their WADD evidence differences vary widely (from +0.1 to +1.3 under linear weights).
    tally_plus_A = ['10011', '01110', '11100', '10110', '00111', '11010']
    
    df_filtered = data[data['A_str'].isin(tally_plus_A)]
    if df_filtered.empty:
        return 0.0
        
    p_A = df_filtered.groupby('A_str')['response'].apply(lambda x: (x == 0).mean())
    
    if len(p_A) < 2:
        return 0.0
        
    # We measure the standard deviation of the choice probabilities across these 6 trials.
    return float(p_A.std())
```

**Observed (real) value:** 0.1109 (var=0.0106)
**Predicted under pi_3:** 0.0867 (var=0.0027)
**Predicted under pi_6:** 0.0996 (var=0.0071)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify the three critical trials
    def is_t15(a, b): return tuple(a) == (1, 0, 0, 1, 1) and tuple(b) == (0, 1, 1, 0, 0)
    def is_t4(a, b):  return tuple(a) == (1, 0, 0, 0, 0) and tuple(b) == (0, 1, 1, 1, 1)
    def is_t2(a, b):  return tuple(a) == (1, 0, 0, 0, 0) and tuple(b) == (0, 1, 1, 0, 0)
    
    data['is_t15'] = data.apply(lambda row: is_t15(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    data['is_t4'] = data.apply(lambda row: is_t4(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    data['is_t2'] = data.apply(lambda row: is_t2(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    
    def subject_metric(subj_df):
        # Count choices for A (response == 0)
        y_15 = (1.0 - subj_df.loc[subj_df['is_t15'], 'response']).sum()
        y_4 = (1.0 - subj_df.loc[subj_df['is_t4'], 'response']).sum()
        y_2 = (1.0 - subj_df.loc[subj_df['is_t2'], 'response']).sum()
        
        n_15 = subj_df['is_t15'].sum()
        n_4 = subj_df['is_t4'].sum()
        n_2 = subj_df['is_t2'].sum()
        
        if n_15 == 0 or n_4 == 0 or n_2 == 0:
            return 0.0
            
        # Laplace smoothing to aggressively reduce variance from extreme proportions
        p_15 = (y_15 + 1.0) / (n_15 + 2.0)
        p_4 = (y_4 + 1.0) / (n_4 + 2.0)
        p_2 = (y_2 + 1.0) / (n_2 + 2.0)
        
        L_15 = np.log(p_15 / (1.0 - p_15))
        L_4 = np.log(p_4 / (1.0 - p_4))
        L_2 = np.log(p_2 / (1.0 - p_2))
        
        return L_15 + L_4 - 2 * L_2
        
    subject_scores = data.groupby('subject_id').apply(subject_metric)
    return float(subject_scores.mean())
```

**Observed (real) value:** 0.1182 (var=4.9895)
**Predicted under pi_3:** 0.9495 (var=4.2008)
**Predicted under pi_6:** 0.4633 (var=2.6682)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (pi_3, WADD) captures the high accuracy on easy trials (Exp 2) and general heuristic matches (Exps 3, 4) quite well, but completely fails to predict the magnitude effect reversal in Exp 1, predicting a positive shift instead of the observed negative shift. It also severely overestimates the log-odds metric in Exp 10. Theory 2 (pi_6, Evidence-Dependent Noise) correctly predicts the preference reversal in Exp 1 and performs better on Exp 6, 7, and 10, but it drastically underestimates baseline accuracy on easy trials (Exp 2) and fails to capture the behavioral patterns in Exps 3 and 4. Because neither the pure compensatory model nor the conflict-penalized noise model can unify the high baseline accuracy with the conflict-driven preference reversals, a structurally different approach is needed.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Dual-Strategy / Heuristic Selection' theory. Instead of a single compensatory mechanism with varying noise, assume decision-makers probabilistically mix between two distinct heuristics: Tallying (equal weighting of features) and Take-The-Best (lexicographic choice based on the highest validity cue). The probability of selecting between these strategies could depend on the trial's structural conflict or the difference in validity between the top features. This would naturally predict near-perfect accuracy on 'easy' trials where both heuristics agree (fixing Exp 2), while successfully capturing preference reversals on high-conflict trials where the dominant strategy shifts or where Tallying pulls choices toward indifference (fixing Exp 1 and Exp 10)."
}
```

## Usage

```json
{
  "prompt_token_count": 11852,
  "candidates_token_count": 370,
  "total_token_count": 13397
}
```
