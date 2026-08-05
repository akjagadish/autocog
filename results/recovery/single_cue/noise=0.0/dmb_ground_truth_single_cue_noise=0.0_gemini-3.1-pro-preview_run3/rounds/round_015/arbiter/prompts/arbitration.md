# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_17" and "pi_15") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_17" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_15" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_17
Decisiveness-Dependent Strategy Selection with Sharp Transition: Decision-makers probabilistically select between a compensatory Tallying strategy and a non-compensatory Take-The-Best (TTB) strategy on a trial-by-trial basis. The probability of using Tallying is a logistic function of the absolute difference in tally scores between the two options. By strictly constraining the sensitivity (theta) to be positive and the threshold to [0.1, 0.9], the model naturally transitions to a sharp step function where Tallying heavily dominates for decisive tally differences (delta >= 1), while TTB is strictly reserved as a tie-breaker for complex/tied stimuli (delta == 0).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    delta_tally = abs(a_wins - b_wins)
    
    if a_wins > b_wins:
        p_a_tally = 1.0
    elif b_wins > a_wins:
        p_a_tally = 0.0
    else:
        p_a_tally = 0.5
        
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    p_a_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_a_ttb = 1.0
            break
        elif b[idx] > a[idx]:
            p_a_ttb = 0.0
            break
            
    # Strategy selection probability
    # Probability of using Tallying depends on the decisiveness of the tally
    exponent = -theta * (delta_tally - threshold)
    exponent = np.clip(exponent, -500.0, 500.0) # Prevent overflow
    p_use_tally = 1.0 / (1.0 + np.exp(exponent))
    
    p_a_core = p_use_tally * p_a_tally + (1.0 - p_use_tally) * p_a_ttb
    p_b_core = 1.0 - p_a_core
    
    p_core = np.array([p_a_core, p_b_core])
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_15
Feature Cancellation then Tally/TTB with Cancellation-Scaled Determinism

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Feature cancellation: identify surviving distinct features
    mask = a != b
    
    # Tallying on distinct features
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    score_a = tally_a
    score_b = tally_b
    
    # If tallies are tied, break the tie using TTB on surviving features
    if tally_a == tally_b and np.sum(mask) > 0:
        order = np.argsort(val)[::-1]
        for idx in order:
            if mask[idx]:
                if a[idx] > b[idx]:
                    score_a += 1.0
                elif b[idx] > a[idx]:
                    score_b += 1.0
                break
            
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with cancellation-scaled determinism
    n_surviving = np.sum(mask)
    effective_beta = beta / n_surviving if n_surviving > 0 else beta
    
    z = effective_beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_17)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the advocated Decisiveness-Dependent Strategy Selection model from the competing Feature Cancellation model, we exploit the fact that while both models agree on the deterministic direction of choices (favoring the tally winner on unequal tallies, and the TTB winner on tied tallies), they fundamentally differ in how choice determinism is scaled. The advocated theory's determinism depends solely on the absolute difference in tallies (delta_tally), completely ignoring the number of shared features. In contrast, the competing theory scales its choice determinism (beta) inversely by the number of surviving (unshared) features. We designed tied-tally trials (delta_tally = 0) with either 2 or 4 surviving features, and unequal-tally trials (delta_tally = 1) with either 3 or 5 surviving features. The advocated theory predicts identical determinism across trials with the same tally difference, whereas the competing theory is structurally forced to predict significantly noisier choices on trials with more surviving features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
This metric leverages the distribution of 'errors' (choices of the universally disfavored Option B) across trial types to strongly dissociate the two theories. 

Under the Advocated theory, choice determinism depends strictly on the tally difference. Because T1 and T2 both have a tally difference of 0, and T3 and T4 both have a tally difference of 1, the theory predicts an equal expected number of errors for T1 vs T2, and for T3 vs T4. Thus, the proportion of total errors occurring on the high-surviving-feature trials (T2 and T4) should be tightly distributed around exactly 0.5.

Under the Competing theory, choice determinism is structurally scaled downwards (inversely) by the number of surviving features. Therefore, trials with more surviving features (T2 and T4) will have significantly lower determinism and generate a disproportionately large share of the errors compared to T1 and T3. This forces the metric to be significantly greater than 0.5. By formatting this as a within-subject proportion of total errors, we normalize away massive variance caused by subject-level differences in baseline noise (epsilon), yielding an extremely stable contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Uniquely identify the 4 trial types by the string representation of Option A
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 1: 2 surviving features, delta_tally = 0
    t1_str = '10111'
    # Trial 2: 4 surviving features, delta_tally = 0
    t2_str = '10101'
    # Trial 3: 3 surviving features, delta_tally = 1
    t3_str = '01100'
    # Trial 4: 5 surviving features, delta_tally = 1
    t4_str = '01110'
    
    subj_metrics = []
    for subj, subj_df in data.groupby('subject_id'):
        b1 = (subj_df[subj_df['A_str'] == t1_str]['response'] == 1).sum()
        b2 = (subj_df[subj_df['A_str'] == t2_str]['response'] == 1).sum()
        b3 = (subj_df[subj_df['A_str'] == t3_str]['response'] == 1).sum()
        b4 = (subj_df[subj_df['A_str'] == t4_str]['response'] == 1).sum()
        
        # Number of errors (B choices) in high-surviving-feature trials
        b_high_surv = b2 + b4
        # Total number of errors across all trials
        b_total = b1 + b2 + b3 + b4
        
        # Calculate the proportion of errors that occurred on high-surviving-feature trials
        if b_total > 0:
            subj_metrics.append(b_high_surv / b_total)
        else:
            # If subject made zero errors (perfect determinism), they provide no differential signal
            subj_metrics.append(0.5)
            
    return float(np.mean(subj_metrics))

### RESULTS
- Predicted under pi_17 (simulated): 0.4898 (var=0.0203)
- Predicted under pi_15 (simulated): 0.6198 (var=0.0317)
- Observed on real data: 0.5577 (var=0.0018)

## EXPERIMENT 2 (proposed by pi_15)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the advocated Feature Cancellation model from the competing Decisiveness-Dependent Strategy Selection model, we exploit the models' differing mechanisms for scaling choice determinism. Both models agree on the deterministic direction of choices (favoring the TTB winner on tied tallies, and the tally winner on unequal tallies). However, the competing theory scales determinism purely based on the absolute difference in tallies (delta_tally), completely ignoring the number of shared features. In contrast, the advocated Feature Cancellation model scales its choice determinism (beta) inversely by the number of surviving (unshared) features. We designed tied-tally trials (delta_tally = 0) with either 2 or 4 surviving features, and unequal-tally trials (delta_tally = 1) with either 3 or 5 surviving features. The competing theory is structurally forced to predict identical determinism across trials with the same tally difference, whereas the advocated theory correctly predicts significantly noisier choices on trials with more surviving features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
We contrast Trial 1 (tied tallies, 2 surviving features) and Trial 4 (decisive tallies, 5 surviving features). The Competing theory views Trial 4 as extremely decisive (both Tally and TTB favor Option A), yielding P(A) near 1.0, while Trial 1 has tied tallies (where Tally is indifferent), pulling P(A) lower. Thus, Competing predicts P(A|Trial 1) < P(A|Trial 4), yielding a negative metric. The Advocated theory scales choice determinism inversely by surviving features. Trial 1 has only 2 surviving features (high determinism), while Trial 4 has 5 (low determinism). Because the score difference is 1 in both cases, Advocated strictly predicts P(A|Trial 1) > P(A|Trial 4), yielding a positive metric. This opposite-sign prediction guarantees strong discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    mask1 = (a_sums == 4) & (b_sums == 4)
    mask4 = (a_sums == 3) & (b_sums == 2)
    
    p_a_1 = 1.0 - data.loc[mask1, 'response'].mean()
    p_a_4 = 1.0 - data.loc[mask4, 'response'].mean()
    
    return float(p_a_1 - p_a_4)


### RESULTS
- Predicted under pi_17 (simulated): -0.0575 (var=0.0111)
- Predicted under pi_15 (simulated): 0.0875 (var=0.0155)
- Observed on real data: -0.0283 (var=0.0154)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
        
        if ttb is not None:
            matches.append(1.0 if row['response'] == ttb else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1454 (var=0.0092)
**Predicted under pi_17:** 0.5010 (var=0.0016)
**Predicted under pi_15:** 0.4973 (var=0.0011)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    
    valid = a_wins != b_wins
    if not np.any(valid):
        return 0.5
    
    pred = (b_wins[valid] > a_wins[valid]).astype(int)
    resp = data['response'].values[valid]
    
    return float(np.mean(pred == resp))
```

**Observed (real) value:** 0.7971 (var=0.0103)
**Predicted under pi_17:** 0.8545 (var=0.0098)
**Predicted under pi_15:** 0.8188 (var=0.0098)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.65, 0.55, 0.5])
    
    wadd_consistent_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_a = np.sum(a * val)
        wadd_b = np.sum(b * val)
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        # Check if WADD and Tallying make strictly opposite predictions
        if (wadd_a > wadd_b and tally_a < tally_b) or (wadd_a < wadd_b and tally_a > tally_b):
            wadd_pref = 0 if wadd_a > wadd_b else 1
            if row['response'] == wadd_pref:
                wadd_consistent_choices.append(1)
            else:
                wadd_consistent_choices.append(0)
                
    if len(wadd_consistent_choices) == 0:
        return 0.5
    return float(np.mean(wadd_consistent_choices))
```

**Observed (real) value:** 0.1733 (var=0.0221)
**Predicted under pi_17:** 0.1867 (var=0.0271)
**Predicted under pi_15:** 0.2189 (var=0.0151)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying has a strict preference
    # (i.e., one option has more positive ratings than the other)
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    # Tallying predicts choosing the option with the higher sum.
    # Response is 0 for A, 1 for B.
    # If sum_a < sum_b, Tallying prefers B (1).
    # If sum_a > sum_b, Tallying prefers A (0).
    tallying_choice = (sum_a < sum_b).astype(int)
    
    # Calculate the proportion of responses matching the Tallying prediction
    matches = (data.loc[mask, 'response'] == tallying_choice.loc[mask])
    
    return float(matches.mean())
```

**Observed (real) value:** 0.8125 (var=0.0197)
**Predicted under pi_17:** 0.8350 (var=0.0116)
**Predicted under pi_15:** 0.7588 (var=0.0169)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_top5 = a[:5]
        b_top5 = b[:5]
        
        a_wins = np.sum(a_top5 > b_top5)
        b_wins = np.sum(b_top5 > a_top5)
        
        if a_wins > b_wins:
            if row['response'] == 0:
                match_count += 1
            total_count += 1
        elif b_wins > a_wins:
            if row['response'] == 1:
                match_count += 1
            total_count += 1
            
    if total_count == 0:
        return 0.5
    return float(match_count / total_count)
```

**Observed (real) value:** 0.1717 (var=0.0110)
**Predicted under pi_17:** 0.6296 (var=0.0026)
**Predicted under pi_15:** 0.5729 (var=0.0041)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    correct_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials with a strong tally difference (>= 2)
        if abs(a_wins - b_wins) >= 2:
            total_count += 1
            if a_wins > b_wins and row['response'] == 0:
                correct_count += 1
            elif b_wins > a_wins and row['response'] == 1:
                correct_count += 1
                
    if total_count == 0:
        return 0.0
    return float(correct_count / total_count)
```

**Observed (real) value:** 0.8554 (var=0.0133)
**Predicted under pi_17:** 0.8800 (var=0.0101)
**Predicted under pi_15:** 0.8646 (var=0.0100)

### Experiment 9
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    correct = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins == b_wins:
            ttb_pred = None
            for idx in range(len(a)):
                if a[idx] > b[idx]:
                    ttb_pred = 0
                    break
                elif b[idx] > a[idx]:
                    ttb_pred = 1
                    break
            if ttb_pred is not None:
                if row['response'] == ttb_pred:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.5
```

**Observed (real) value:** 0.6094 (var=0.0030)
**Predicted under pi_17:** 0.8567 (var=0.0085)
**Predicted under pi_15:** 0.7786 (var=0.0167)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    tied_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            tied_trials += 1
            ttb_choice = -1
            # Validities are monotonically decreasing with index
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_choice = 0
                    break
                elif b[i] > a[i]:
                    ttb_choice = 1
                    break
            
            if row['response'] == ttb_choice:
                matches += 1
                
    if tied_trials == 0:
        return 0.5
    return float(matches / tied_trials)
```

**Observed (real) value:** 0.6178 (var=0.0023)
**Predicted under pi_17:** 0.8750 (var=0.0060)
**Predicted under pi_15:** 0.7906 (var=0.0122)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tied_mask = (a_wins == b_wins)
    
    if np.sum(tied_mask) == 0:
        return 0.5
        
    # Response is 0 if subject chose A, 1 if subject chose B
    # We calculate the proportion of times A was chosen on tied trials
    return float(np.mean(data['response'].values[tied_mask] == 0))
```

**Observed (real) value:** 0.7361 (var=0.0113)
**Predicted under pi_17:** 0.8407 (var=0.0098)
**Predicted under pi_15:** 0.7621 (var=0.0153)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_advocated = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Advocated favors B (1), Competing favors A (0)
        if a == (1, 0, 0, 0, 1, 1) and b == (0, 1, 1, 1, 0, 0):
            if resp == 1:
                match_advocated += 1
            total += 1
        # Trial 2: Advocated favors A (0), Competing favors B (1)
        elif a == (0, 1, 1, 1, 0, 0) and b == (1, 0, 0, 0, 1, 1):
            if resp == 0:
                match_advocated += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(match_advocated / total)
```

**Observed (real) value:** 0.1525 (var=0.0073)
**Predicted under pi_17:** 0.1706 (var=0.0102)
**Predicted under pi_15:** 0.2431 (var=0.0150)

### Experiment 13
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert option_a_ratings to tuple for matching
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Favored responses for each trial type
    favored_map = {
        (1, 1, 1, 0, 0): 0,
        (1, 0, 0, 1, 0): 0,
        (1, 0, 1, 0, 0): 1,
        (1, 0, 0, 0, 0): 1,
        (0, 1, 1, 1, 0): 0
    }
    
    # Check if choice matches favored
    is_favored = data.apply(lambda row: 1 if row['response'] == favored_map.get(tuple(row['option_a_ratings']), -1) else 0, axis=1)
    
    unequal_trials = {(1, 1, 1, 0, 0), (1, 0, 0, 0, 0), (0, 1, 1, 1, 0)}
    tied_trials = {(1, 0, 0, 1, 0), (1, 0, 1, 0, 0)}
    
    mask_unequal = a_tuples.isin(unequal_trials)
    mask_tied = a_tuples.isin(tied_trials)
    
    if mask_unequal.sum() == 0 or mask_tied.sum() == 0:
        return 0.0
        
    acc_unequal = is_favored[mask_unequal].mean()
    acc_tied = is_favored[mask_tied].mean()
    
    return float(acc_unequal - acc_tied)
```

**Observed (real) value:** -0.2295 (var=0.0163)
**Predicted under pi_17:** -0.0039 (var=0.0089)
**Predicted under pi_15:** -0.0235 (var=0.0070)

### Experiment 14
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: TTB chooses the option with cue 0 (which is A here)
        if a == (1, 0, 0, 1, 1) and b == (0, 1, 1, 1, 0):
            if row['response'] == 0: ttb_consistent += 1
            total += 1
        elif a == (0, 1, 1, 1, 0) and b == (1, 0, 0, 1, 1):
            if row['response'] == 1: ttb_consistent += 1
            total += 1
            
        # Trial 2: TTB chooses the option with cue 0 (which is B here)
        elif a == (0, 1, 1, 0, 1) and b == (1, 0, 0, 1, 1):
            if row['response'] == 1: ttb_consistent += 1
            total += 1
        elif a == (1, 0, 0, 1, 1) and b == (0, 1, 1, 0, 1):
            if row['response'] == 0: ttb_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_consistent) / total
```

**Observed (real) value:** 0.6633 (var=0.0060)
**Predicted under pi_17:** 0.8717 (var=0.0082)
**Predicted under pi_15:** 0.7308 (var=0.0254)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Create a trial identifier
    data['trial_key'] = data.apply(lambda row: (tuple(row['option_a_ratings']), tuple(row['option_b_ratings'])), axis=1)

    t1_key = ((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))
    t2_key = ((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))

    # Calculate proportion of A choices (response == 0) for each trial
    t1_data = data[data['trial_key'] == t1_key]
    t2_data = data[data['trial_key'] == t2_key]

    p_a_t1 = (t1_data['response'] == 0).mean() if len(t1_data) > 0 else 0.5
    p_a_t2 = (t2_data['response'] == 0).mean() if len(t2_data) > 0 else 0.5

    return float(p_a_t1 - p_a_t2)
```

**Observed (real) value:** 0.6547 (var=0.0514)
**Predicted under pi_17:** 0.6884 (var=0.0455)
**Predicted under pi_15:** 0.5526 (var=0.0829)

### Experiment 16
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.5
    return float(data[is_trial_1]['response'].mean())
```

**Observed (real) value:** 0.8267 (var=0.0129)
**Predicted under pi_17:** 0.8192 (var=0.0198)
**Predicted under pi_15:** 0.7725 (var=0.0177)

### Experiment 17
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    is_tie = a_sums == b_sums
    tie_data = data[is_tie]
    if len(tie_data) == 0:
        return 0.5
    a_has_top = tie_data['option_a_ratings'].apply(lambda x: x[0] == 1)
    chose_a = tie_data['response'] == 0
    chose_ttb = a_has_top == chose_a
    return float(chose_ttb.mean())
```

**Observed (real) value:** 0.8492 (var=0.0107)
**Predicted under pi_17:** 0.8721 (var=0.0075)
**Predicted under pi_15:** 0.7275 (var=0.0170)

### Experiment 18
**Design**
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # The Tally-then-TTB model's predicted choice exactly matches the value of Option A's 2nd feature (index 1)
    # across all 4 trial types, whereas the Rank-Based model always predicts the opposite.
    a_feat1 = data['option_a_ratings'].apply(lambda x: x[1])
    return float(np.mean(data['response'] == a_feat1))
```

**Observed (real) value:** 0.5967 (var=0.0013)
**Predicted under pi_17:** 0.8279 (var=0.0098)
**Predicted under pi_15:** 0.7402 (var=0.0170)

### Experiment 19
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tied_matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        if np.sum(a) == np.sum(b):
            ttb_winner = None
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_winner = 0
                    break
                elif b[i] > a[i]:
                    ttb_winner = 1
                    break
            
            if ttb_winner is not None:
                tied_matches.append(1 if row['response'] == ttb_winner else 0)
                
    if not tied_matches:
        return 0.5
    return float(np.mean(tied_matches))
```

**Observed (real) value:** 0.3221 (var=0.0054)
**Predicted under pi_17:** 0.8384 (var=0.0095)
**Predicted under pi_15:** 0.7621 (var=0.0230)

### Experiment 20
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_disagrees_tally = 0
    ttb_chosen = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        # TTB winner (validities are strictly decreasing from index 0)
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        tally_winner = None
        if tally_a > tally_b:
            tally_winner = 0
        elif tally_b > tally_a:
            tally_winner = 1
            
        if tally_winner is not None and ttb_winner != tally_winner:
            ttb_disagrees_tally += 1
            if row['response'] == ttb_winner:
                ttb_chosen += 1
                
    if ttb_disagrees_tally == 0:
        return 0.0
    return float(ttb_chosen / ttb_disagrees_tally)
```

**Observed (real) value:** 0.4850 (var=0.0066)
**Predicted under pi_17:** 0.1313 (var=0.0093)
**Predicted under pi_15:** 0.1806 (var=0.0152)

### Experiment 21
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    t1_mask = a_str == '11000'
    t2_mask = a_str == '10001'
    
    t1_data = data[t1_mask]
    t2_data = data[t2_mask]
    
    if len(t1_data) == 0 or len(t2_data) == 0:
        return 0.0
        
    p_a_t1 = 1.0 - t1_data['response'].mean()
    p_a_t2 = 1.0 - t2_data['response'].mean()
    
    return float(p_a_t2 - p_a_t1)
```

**Observed (real) value:** 0.6000 (var=0.0708)
**Predicted under pi_17:** 0.6550 (var=0.0413)
**Predicted under pi_15:** 0.5525 (var=0.0664)

### Experiment 22
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert option_a_ratings to string for easy filtering
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # response == 0 means Option A was chosen
    is_a = (data['response'] == 0).astype(float)
    
    # Calculate proportion of Option A choices for each trial type
    p_t1 = is_a[a_str == '11000'].mean()
    p_t2 = is_a[a_str == '10001'].mean()
    p_t3 = is_a[a_str == '10000'].mean()
    p_t4 = is_a[a_str == '10010'].mean()
    
    # Handle potential NaNs safely
    p_t1 = p_t1 if pd.notna(p_t1) else 0.0
    p_t2 = p_t2 if pd.notna(p_t2) else 0.0
    p_t3 = p_t3 if pd.notna(p_t3) else 0.0
    p_t4 = p_t4 if pd.notna(p_t4) else 0.0
    
    # Tally-then-TTB heavily favors A in T2/T4 (tied tallies broken by cue 1)
    # but heavily favors B in T1/T3 (B wins the tally outright).
    # WADD either favors A in all (high gamma) or slightly prefers B in T1/T3 and is neutral in T2/T4 (low gamma).
    return (p_t2 + p_t4) - (p_t1 + p_t3)

```

**Observed (real) value:** 0.9417 (var=0.1401)
**Predicted under pi_17:** 1.3125 (var=0.0965)
**Predicted under pi_15:** 1.2383 (var=0.1430)

### Experiment 23
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    tied_ttb_match = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            # Ties are broken by TTB in the advocated model.
            # The validities are strictly decreasing, so the highest validity
            # cue is simply the first one where options differ.
            ttb_winner = -1
            for idx in range(len(a)):
                if a[idx] > b[idx]:
                    ttb_winner = 0
                    break
                elif b[idx] > a[idx]:
                    ttb_winner = 1
                    break
            
            if ttb_winner != -1:
                tied_ttb_match.append(1 if row['response'] == ttb_winner else 0)
                
    if not tied_ttb_match:
        return 0.5
    return float(np.mean(tied_ttb_match))
```

**Observed (real) value:** 0.6617 (var=0.0064)
**Predicted under pi_17:** 0.8658 (var=0.0050)
**Predicted under pi_15:** 0.7450 (var=0.0217)

### Experiment 24
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_unequal(row):
        return sum(row['option_a_ratings']) != sum(row['option_b_ratings'])
        
    def favored_choice(row):
        a_sum = sum(row['option_a_ratings'])
        b_sum = sum(row['option_b_ratings'])
        if a_sum > b_sum:
            return 0
        elif b_sum > a_sum:
            return 1
        else:
            # Equal tally: tie-breaker is the first cue (highest validity)
            if row['option_a_ratings'][0] > row['option_b_ratings'][0]:
                return 0
            else:
                return 1

    unequal_mask = data.apply(is_unequal, axis=1)
    favored = data.apply(favored_choice, axis=1)
    is_favored = (data['response'] == favored)
    
    p_unequal = is_favored[unequal_mask].mean()
    p_equal = is_favored[~unequal_mask].mean()
    
    return float(p_unequal - p_equal)

```

**Observed (real) value:** -0.3583 (var=0.0092)
**Predicted under pi_17:** 0.0083 (var=0.0074)
**Predicted under pi_15:** -0.0029 (var=0.0069)

### Experiment 25
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    t1_mask = data['A_tuple'] == (0, 0, 1, 1, 1)
    t2_mask = data['A_tuple'] == (1, 0, 0, 0, 1)
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1):
        p_a_t1 = 0.5
    if pd.isna(p_a_t2):
        p_a_t2 = 0.5
        
    return float(p_a_t1 + p_a_t2)
```

**Observed (real) value:** 1.7383 (var=0.0606)
**Predicted under pi_17:** 1.6417 (var=0.0416)
**Predicted under pi_15:** 1.5417 (var=0.0672)

### Experiment 26
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 and Trial 2. 
    # In both of these trials, Option A's 5th feature (index 4) is 1.
    # In Trials 3 and 4, Option A's 5th feature is 0.
    is_target_trial = data['option_a_ratings'].apply(lambda x: x[4] == 1)
    
    # Calculate the proportion of times Option B (response == 1) is chosen on these target trials.
    return float(data.loc[is_target_trial, 'response'].mean())
```

**Observed (real) value:** 0.1383 (var=0.0142)
**Predicted under pi_17:** 0.1617 (var=0.0099)
**Predicted under pi_15:** 0.2254 (var=0.0168)

### Experiment 27
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 1: Tie broken by 1st cue
    t1_mask = (a_str == '10100') & (b_str == '01010')
    # Trial 2: Tie broken by 2nd cue
    t2_mask = (a_str == '11001') & (b_str == '10110')
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    
    if p_a_t1 != p_a_t1:
        p_a_t1 = 0.0
    if p_a_t2 != p_a_t2:
        p_a_t2 = 0.0
        
    return float(p_a_t1 - p_a_t2)
```

**Observed (real) value:** -0.3583 (var=0.0167)
**Predicted under pi_17:** 0.0200 (var=0.0086)
**Predicted under pi_15:** 0.0158 (var=0.0169)

### Experiment 28
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_trial_1 = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[4] == 1 and x[1] == 0)
    is_trial_2 = data['option_a_ratings'].apply(lambda x: x[1] == 1 and x[4] == 1 and x[0] == 0)
    is_trial_3 = data['option_a_ratings'].apply(lambda x: x[2] == 1 and x[3] == 1 and x[4] == 1)
    mask = is_trial_1 | is_trial_2 | is_trial_3
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1644 (var=0.0120)
**Predicted under pi_17:** 0.1564 (var=0.0057)
**Predicted under pi_15:** 0.2297 (var=0.0154)

### Experiment 29
**Design**
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_n_surv(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return sum(x != y for x, y in zip(a, b))
        
    n_surv = data.apply(get_n_surv, axis=1)
    
    subj_metrics = []
    for subj, subj_df in data.groupby('subject_id'):
        subj_n_surv = n_surv.loc[subj_df.index]
        chose_a_subj = (subj_df['response'] == 0)
        
        lo = {}
        for n in [1, 2, 4, 5]:
            mask = (subj_n_surv == n)
            if mask.sum() == 0:
                lo[n] = 0.0
                continue
            n_a = chose_a_subj[mask].sum()
            n_b = mask.sum() - n_a
            # Smoothed empirical log odds
            lo[n] = np.log((n_a + 0.5) / (n_b + 0.5))
            
        # Contrast log odds of low-surviving vs high-surviving feature trials
        val = lo[1] + lo[2] - lo[4] - lo[5]
        subj_metrics.append(val)
        
    if not subj_metrics:
        return 0.0
        
    return float(np.mean(subj_metrics))
```

**Observed (real) value:** 1.6988 (var=1.4079)
**Predicted under pi_17:** 0.1873 (var=1.6174)
**Predicted under pi_15:** 0.8507 (var=2.2339)

### Experiment 30
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the number of surviving features (mask size) for each trial
    # Trial 1 has mask size 1, Trial 2 has mask size 5
    mask_sizes = data.apply(
        lambda row: sum(a != b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])),
        axis=1
    )
    
    is_trial1 = mask_sizes == 1
    is_trial2 = mask_sizes == 5
    
    # Calculate the proportion of times Option A (response == 0) was chosen
    p_a_t1 = np.mean(data.loc[is_trial1, 'response'] == 0)
    p_a_t2 = np.mean(data.loc[is_trial2, 'response'] == 0)
    
    if pd.isna(p_a_t1) or pd.isna(p_a_t2):
        return 0.0
        
    return float(p_a_t1 - p_a_t2)
```

**Observed (real) value:** 0.3433 (var=0.0217)
**Predicted under pi_17:** 0.0067 (var=0.0071)
**Predicted under pi_15:** 0.1092 (var=0.0169)

### Experiment 31
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tup == (1, 1, 0, 0, 0)) & (b_tup == (0, 0, 1, 1, 1))
    t2_mask = (a_tup == (1, 0, 0, 0, 1)) & (b_tup == (0, 1, 1, 0, 0))
    t3_mask = (a_tup == (1, 1, 1, 0, 0)) & (b_tup == (0, 1, 1, 1, 1))
    
    wadd_choices = 0.0
    wadd_choices += (data.loc[t1_mask, 'response'] == 0).sum()
    wadd_choices += (data.loc[t2_mask, 'response'] == 1).sum()
    wadd_choices += (data.loc[t3_mask, 'response'] == 0).sum()
    
    total_diagnostic = t1_mask.sum() + t2_mask.sum() + t3_mask.sum()
    
    if total_diagnostic == 0:
        return 0.5
        
    return float(wadd_choices / total_diagnostic)
```

**Observed (real) value:** 0.1270 (var=0.0060)
**Predicted under pi_17:** 0.1768 (var=0.0099)
**Predicted under pi_15:** 0.1940 (var=0.0117)

### Experiment 32
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_sig(lst):
        return "".join(str(int(x)) for x in lst)
        
    data['A_str'] = data['option_a_ratings'].apply(get_sig)
    
    t1_sig = "11000"
    t2_sig = "10001"
    t3_sig = "11100"
    
    score = 0.0
    
    t1_data = data[data['A_str'] == t1_sig]
    if len(t1_data) > 0:
        score += np.mean(t1_data['response'] == 1)
        
    t2_data = data[data['A_str'] == t2_sig]
    if len(t2_data) > 0:
        score += np.mean(t2_data['response'] == 0)
        
    t3_data = data[data['A_str'] == t3_sig]
    if len(t3_data) > 0:
        score += np.mean(t3_data['response'] == 1)
        
    return float(score)
```

**Observed (real) value:** 2.4433 (var=0.1344)
**Predicted under pi_17:** 2.4442 (var=0.0654)
**Predicted under pi_15:** 2.3525 (var=0.1602)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 1 (pi_17) generally predicts the observed data more accurately than Theory 2 (pi_15), particularly in Experiments 5, 6, 12, and an impressive exact match in Experiment 32. Experiment 1 (proposed by Theory 1) yielded an observed metric of 0.5577, which is somewhat intermediate but shows a slight feature cancellation effect (as it's > 0.5). However, Experiment 2 (proposed by Theory 2) yielded a negative metric (-0.0283), completely contradicting Theory 2's positive prediction (0.0875) and aligning directionally with Theory 1 (-0.0575). Despite Theory 1's relative success, both theories fail drastically on Experiments 3, 7, 13, and 29. Theory 2's core assumption that determinism scales inversely with surviving features is not strongly supported by Experiment 2, making it a good candidate for replacement.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory that moves away from scaling determinism purely by the number of surviving features. Instead, consider a theory where Feature Cancellation acts as a strict pre-processing step, followed by an evidence-accumulation or weighted-additive process on the surviving features. Alternatively, propose a model where the decision-maker uses a lexicographic heuristic (like TTB) but allows for 'trembling hand' errors that scale with the similarity of the options (e.g., distance in the feature space). This could better account for the failures in Experiments 3, 7, and 13 while retaining the strong performance on tally-dominant trials."
}
```

## Usage

```json
{
  "prompt_token_count": 21349,
  "candidates_token_count": 391,
  "total_token_count": 22745
}
```
