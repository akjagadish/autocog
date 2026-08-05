# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_15") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_15" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Tally-then-TTB (Tally with Validity Tie-Breaker): Decision-makers first compare options by tallying the total number of winning features for each option, ignoring cue validities. If one option has strictly more wins, it is chosen. This captures the compensatory nature of decision-making when there is a clear majority of supporting cues. However, if the tally results in a tie, the decision-maker falls back to a non-compensatory 'Take-The-Best' strategy to break the tie, choosing the option that wins on the single most valid cue. This hybrid model preserves robust Tallying performance on unequal-tally trials while capturing deterministic validity-based tie-breaking on ambiguous trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying phase: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    score_a = a_wins
    score_b = b_wins
    
    # Tie-breaking phase: if tallies are equal, use TTB
    if a_wins == b_wins:
        cue_order = np.argsort(-val, kind="stable")
        for idx in cue_order:
            if a[idx] > b[idx]:
                score_a += 1.0
                break
            elif b[idx] > a[idx]:
                score_b += 1.0
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate Tally-then-TTB (advocated) from the Feature Cancellation model (competing), we exploit how each theory scales choice determinism. In Tally-then-TTB, choice determinism is driven purely by the final score difference. A 1-point tally advantage and a tie broken by the highest-validity cue both yield a score difference of exactly 1.0, leading to identical predicted determinism across such trials regardless of how many features the options share. In contrast, the competing model scales its determinism parameter (beta) inversely by the number of 'surviving' (non-canceling) features. We designed pairs of trials that have identical score differences under Tally-then-TTB (either a 1-point tally win or a tied tally broken by the top cue) but drastically different numbers of surviving features (e.g., 5 vs. 1, or 4 vs. 2). The advocated model predicts equal determinism across these pairs, whereas the competing model is structurally forced to predict much noisier choices on trials with more surviving features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pairs trials that have identical score differences under the advocated model but different numbers of surviving features. The advocated model predicts equal determinism across all trials, yielding an expected difference in log odds of 0. The competing model scales its determinism parameter inversely by the number of surviving features, predicting much higher log odds of choosing the favored option on trials with fewer surviving features. By computing the contrast in smoothed log odds of choosing Option A between low-surviving (1 and 2) and high-surviving (4 and 5) trials, this metric magnifies the differences between probabilities, avoiding the compression issues of raw probability differences and providing a highly discriminative signal.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): -0.1484 (var=1.1731)
- Predicted under pi_15 (simulated): 0.8507 (var=2.2339)
- Observed on real data: 1.6988 (var=1.4079)

## EXPERIMENT 2 (proposed by pi_15)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Feature Cancellation model (advocated) from Tally-then-TTB (competing), we exploit how each theory scales choice determinism. In Tally-then-TTB, choice determinism is driven purely by the final score difference. A 1-point tally advantage and a tie broken by the highest-validity cue both yield a score difference of exactly 1.0, leading to identical predicted determinism across such trials regardless of how many features the options share. In contrast, the advocated Feature Cancellation model scales its determinism parameter (beta) inversely by the number of 'surviving' (non-canceling) features. We designed pairs of trials that have identical score differences under both models (either a 1-point tally win or a tied tally broken by the top cue) but drastically different numbers of surviving features (e.g., 5 vs. 1, or 4 vs. 2). The competing model is structurally forced to predict equal determinism across these pairs, whereas the advocated model predicts much noisier choices on trials with more surviving features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
Under the competing Tally-then-TTB model, the score difference between Option A and Option B is exactly 1 for both Trial 1 and Trial 2, leading to identical predicted choice probabilities (the metric should be ~0). Under the advocated Feature Cancellation model, the determinism of choice is inversely scaled by the number of surviving features. Since Trial 1 has 1 surviving feature and Trial 2 has 5, the model predicts much more deterministic choices (higher probability of choosing A) on Trial 1 compared to Trial 2, leading to a large positive metric.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): 0.0050 (var=0.0093)
- Predicted under pi_15 (simulated): 0.1092 (var=0.0169)
- Observed on real data: 0.3433 (var=0.0217)

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
**Predicted under pi_5:** 0.4925 (var=0.0016)
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
**Predicted under pi_5:** 0.8621 (var=0.0047)
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
**Predicted under pi_5:** 0.1133 (var=0.0160)
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
**Predicted under pi_5:** 0.8444 (var=0.0134)
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
**Predicted under pi_5:** 0.6242 (var=0.0023)
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
**Predicted under pi_5:** 0.8738 (var=0.0118)
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
**Predicted under pi_5:** 0.8394 (var=0.0109)
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
**Predicted under pi_5:** 0.8386 (var=0.0099)
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
**Predicted under pi_5:** 0.8551 (var=0.0083)
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
**Predicted under pi_5:** 0.1394 (var=0.0095)
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
**Predicted under pi_5:** 0.0040 (var=0.0051)
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
**Predicted under pi_5:** 0.8442 (var=0.0128)
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
**Predicted under pi_5:** 0.6368 (var=0.0672)
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
**Predicted under pi_5:** 0.8808 (var=0.0082)
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
**Predicted under pi_5:** 0.8404 (var=0.0110)
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
**Predicted under pi_5:** 0.8594 (var=0.0106)
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
**Predicted under pi_5:** 0.8463 (var=0.0144)
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
**Predicted under pi_5:** 0.1412 (var=0.0087)
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
**Predicted under pi_5:** 0.7008 (var=0.0386)
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
**Predicted under pi_5:** 1.4517 (var=0.1365)
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
**Predicted under pi_5:** 0.8337 (var=0.0169)
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
**Predicted under pi_5:** 0.0121 (var=0.0047)
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
**Predicted under pi_5:** 1.6933 (var=0.0511)
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
**Predicted under pi_5:** 0.1396 (var=0.0090)
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
**Predicted under pi_5:** -0.0125 (var=0.0120)
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
**Predicted under pi_5:** 0.1678 (var=0.0192)
**Predicted under pi_15:** 0.2297 (var=0.0154)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both recent experiments (Exp 1 and Exp 2), the observed data strongly refutes the Tally-then-TTB model's assumption of constant determinism. Instead, it shows a massive scaling effect where choices are much more deterministic on trials with fewer differing features, aligning directionally with Theory 2 (Feature Cancellation). However, the observed magnitudes in Exp 1 (1.69) and Exp 2 (0.34) far exceed Theory 2's predictions (0.85 and 0.11, respectively). Furthermore, looking at the broader suite of experiments (e.g., Exps 13, 24, 27), both models routinely fail to capture significant negative metric values, predicting near-zero instead. This indicates that neither strict tallying nor simple tallying-with-cancellation adequately captures the subjects' compensatory behavior and noise scaling.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory based on a Weighted Additive (WADD) mechanism combined with Similarity-Scaled Determinism. Instead of non-compensatory tallying or TTB, subjects likely compute a weighted sum of cue differences, where cue weights are derived from their stated validities (e.g., via a non-linear transformation like log-odds). Crucially, to explain the massive variance in determinism (Exps 1 and 2), the decision noise (temperature) should scale proportionally to the total number of differing features (or total stimulus complexity). This fully compensatory approach with complexity-penalized determinism will better capture both the nuanced preference reversals in Exps 13/24/27 and the steep determinism gradients in Exps 1/2."
}
```

## Usage

```json
{
  "prompt_token_count": 18658,
  "candidates_token_count": 388,
  "total_token_count": 20037
}
```
