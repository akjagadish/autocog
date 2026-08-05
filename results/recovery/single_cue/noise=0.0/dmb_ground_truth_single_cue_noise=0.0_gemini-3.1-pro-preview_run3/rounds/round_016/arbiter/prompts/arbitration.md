# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_17" and "pi_18") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_17" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_18" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_18
Proportional Evidence Accumulation on Surviving Features with Exponentiated Tie-Breaker

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
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    mask = a != b
    n_surv = np.sum(mask)
    
    if n_surv == 0:
        return np.array([0.5, 0.5])
        
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    # Exponentiate validities to smoothly transition between WADD and TTB tie-breaking
    val_transformed = val ** theta
    tie_w = val_transformed / np.sum(val_transformed)
    
    # Score combines Tallying and a strictly bounded validity-based tie-breaker
    score_a = tally_a + gamma * np.sum(tie_w * a_wins)
    score_b = tally_b + gamma * np.sum(tie_w * b_wins)
    
    # The decision variable is the proportion of surviving evidence
    norm_score_a = score_a / n_surv
    norm_score_b = score_b / n_surv
    
    scores = np.array([norm_score_a, norm_score_b])
    
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
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_17)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** To quantitatively dissociate the advocated Decisiveness-Dependent Strategy Selection model from the competing Proportional Evidence Accumulation model, we exploit how each model resolves tied tallies. The advocated model probabilistically selects between Tallying and Take-The-Best (TTB) based solely on the absolute difference in tallies. On tied-tally trials (difference = 0), it strictly falls back to the TTB prediction. Consequently, for any two tied-tally trials where Option A wins the single highest-validity cue, the advocated model predicts exactly identical choice probabilities for A, regardless of how the remaining cues are distributed. In contrast, the competing model resolves ties using a compensatory weighted sum of exponentiated validities. We designed two tied-tally trials (Trial 1 and Trial 2). In both, Option A wins the highest-validity cue, ensuring the advocated model predicts identical determinism. However, in Trial 1, Option B wins the 2nd and 3rd best cues (providing a strong compensatory counter-weight), whereas in Trial 2, Option A wins the 2nd best cue. The competing model is structurally forced to predict that Option A is chosen with significantly higher probability in Trial 2 than in Trial 1, whereas the advocated model strictly predicts equal probabilities. Unequal-tally trials are included to constrain baseline determinism.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in choice determinism (absolute deviation of P(A) from 0.5) between decisive unequal-tally trials (Trials 4 and 5) and tied-tally trials (Trials 1 and 2). According to the advocated Decisiveness-Dependent Strategy Selection model, decisive trials are governed by Tallying and tied trials strictly by Take-The-Best. Because both strategies are deterministic and align with Option A in these specific trials, the model predicts identical, maximal determinism across all four trials, yielding a metric near zero. In contrast, the competing Proportional Evidence Accumulation model uses a compensatory mechanism where tied tallies rely entirely on a down-weighted validity tie-breaker, resulting in structurally smaller decision variables. Consequently, the competing model predicts significantly lower determinism for the tied trials compared to the decisive trials, yielding a robustly positive metric.

Source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    m1 = (a_tuples == (1, 0, 0, 0, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    m2 = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 0, 1))
    m4 = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    m5 = (a_tuples == (1, 1, 1, 0, 0)) & (b_tuples == (0, 0, 0, 0, 1))
    
    p1 = (data.loc[m1, 'response'] == 0).mean() if m1.any() else 0.5
    p2 = (data.loc[m2, 'response'] == 0).mean() if m2.any() else 0.5
    p4 = (data.loc[m4, 'response'] == 0).mean() if m4.any() else 0.5
    p5 = (data.loc[m5, 'response'] == 0).mean() if m5.any() else 0.5
    
    return float((abs(p4 - 0.5) + abs(p5 - 0.5)) - (abs(p1 - 0.5) + abs(p2 - 0.5)))

### RESULTS
- Predicted under pi_17 (simulated): 0.0117 (var=0.0389)
- Predicted under pi_18 (simulated): 0.3317 (var=0.0749)
- Observed on real data: -0.0133 (var=0.0439)

## EXPERIMENT 2 (proposed by pi_18)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the advocated Proportional Evidence Accumulation model from the competing Decisiveness-Dependent Strategy Selection model, we exploit how each model resolves tied tallies and scales determinism. The competing model resolves tied tallies by relying on a strictly non-compensatory Take-The-Best (TTB) mechanism, predicting identical choice probabilities for any tied-tally trials where the same option wins the highest-validity cue. In contrast, the advocated model uses a compensatory exponentiated-validity tie-breaker and normalizes the accumulated evidence by the number of surviving (unshared) features. We designed multiple tied-tally trials where Option A wins the single highest-validity cue, but the distribution of the remaining cues and the total number of surviving features varies (e.g., Option B winning the 2nd and 3rd best cues vs. Option A winning the 2nd best cue; 4 surviving features vs. 2 surviving features). The competing model is structurally forced to predict identical choice probabilities across all these tied-tally trials, whereas the advocated model correctly predicts varying determinism based on its compensatory tie-breaker and surviving-feature normalization.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
Calculates the difference in the proportion of A choices between Trial 2 and Trial 1. Both are tied-tally trials where Option A wins the highest validity cue. The competing model's TTB tie-breaker evaluates both trials identically (predicting A with the same probability). The advocated model evaluates the secondary cues, favoring Option A much more strongly in Trial 2 than in Trial 1. Thus, this difference should be zero for the competing model and positive for the advocated model.

Source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0))
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    p_a_t1 = (data[t1_mask]['response'] == 0).mean()
    p_a_t2 = (data[t2_mask]['response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t1)

### RESULTS
- Predicted under pi_17 (simulated): -0.3405 (var=0.0148)
- Predicted under pi_18 (simulated): -0.1789 (var=0.0284)
- Observed on real data: -0.3284 (var=0.0264)

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
**Predicted under pi_18:** 0.3767 (var=0.0032)

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
**Predicted under pi_18:** 0.8462 (var=0.0079)

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
**Predicted under pi_18:** 0.2144 (var=0.0251)

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
**Predicted under pi_18:** 0.7325 (var=0.0260)

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
**Predicted under pi_18:** 0.5492 (var=0.0054)

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
**Predicted under pi_18:** 0.8785 (var=0.0110)

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
**Predicted under pi_18:** 0.6181 (var=0.0120)

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
**Predicted under pi_18:** 0.5900 (var=0.0089)

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
**Predicted under pi_18:** 0.6049 (var=0.0130)

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
**Predicted under pi_18:** 0.3962 (var=0.0212)

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
**Predicted under pi_18:** 0.1842 (var=0.0206)

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
**Predicted under pi_18:** 0.5350 (var=0.0179)

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
**Predicted under pi_18:** 0.4232 (var=0.0668)

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
**Predicted under pi_18:** 0.7433 (var=0.0293)

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
**Predicted under pi_18:** 0.5954 (var=0.0214)

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
**Predicted under pi_18:** 0.6931 (var=0.0108)

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
**Predicted under pi_18:** 0.6668 (var=0.0234)

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
**Predicted under pi_18:** 0.1475 (var=0.0118)

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
**Predicted under pi_18:** 0.4417 (var=0.0551)

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
**Predicted under pi_18:** 0.9058 (var=0.1451)

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
**Predicted under pi_18:** 0.6500 (var=0.0264)

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
**Predicted under pi_18:** 0.1592 (var=0.0153)

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
**Predicted under pi_18:** 1.4192 (var=0.0505)

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
**Predicted under pi_18:** 0.2767 (var=0.0094)

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
**Predicted under pi_18:** 0.0983 (var=0.0213)

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
**Predicted under pi_18:** 0.3333 (var=0.0102)

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
**Predicted under pi_18:** 0.5626 (var=1.8073)

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
**Predicted under pi_18:** 0.0250 (var=0.0140)

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
**Predicted under pi_18:** 0.2625 (var=0.0115)

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
**Predicted under pi_18:** 2.2042 (var=0.1077)

### Experiment 33
**Design**
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.5577 (var=0.0018)
**Predicted under pi_17:** 0.4898 (var=0.0203)
**Predicted under pi_18:** 0.5484 (var=0.0086)

### Experiment 34
**Design**
  A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    mask1 = (a_sums == 4) & (b_sums == 4)
    mask4 = (a_sums == 3) & (b_sums == 2)
    
    p_a_1 = 1.0 - data.loc[mask1, 'response'].mean()
    p_a_4 = 1.0 - data.loc[mask4, 'response'].mean()
    
    return float(p_a_1 - p_a_4)

```

**Observed (real) value:** -0.0283 (var=0.0154)
**Predicted under pi_17:** -0.0575 (var=0.0111)
**Predicted under pi_18:** -0.1317 (var=0.0225)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (pi_17) performed exceptionally well on the new experiments (Exp 1 and 2), accurately capturing the determinism differences between decisive and tied trials. However, across the broader dataset (e.g., Exp 9, 10, 19, 23), Theory 1 consistently over-predicts determinism on tied-tally trials because it assumes a strict, deterministic fallback to Take-The-Best (yielding predictions ~0.85 when observed data is ~0.60). Theory 2 (pi_18) captures the lower determinism of tied trials better due to its proportional evidence accumulation but struggles on other structural tests where Tallying clearly dominates. A hybrid approach is needed.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory: 'Tallying with Graded Validity Tie-Breaker'. The decision-maker primarily relies on the difference in tallies (a simple, robust strategy). However, instead of strictly falling back to a non-compensatory Take-The-Best strategy on tied tallies (which overpredicts determinism), the decision-maker falls back to a compensatory weighted sum of validities (WADD) or a probabilistic cue-based evidence accumulation. This ensures that decisive tally differences drive strong determinism, while tied tallies are resolved with a softer, compensatory mechanism that correctly yields the empirically observed lower determinism (~0.60)."
}
```

## Usage

```json
{
  "prompt_token_count": 22566,
  "candidates_token_count": 327,
  "total_token_count": 25106
}
```
