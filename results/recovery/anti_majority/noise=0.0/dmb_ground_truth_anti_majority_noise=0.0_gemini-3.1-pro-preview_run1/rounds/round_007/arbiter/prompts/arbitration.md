# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_8" and "pi_9") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_8" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_9" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_8
Variance Aversion (Configural Weighting): Decision-makers evaluate options by averaging the subjective validities of their present features, but they apply a configural penalty proportional to the variance of those validities. This mechanism reflects a preference for 'holistic consistency'—options with tightly clustered, moderately high features are preferred over options with a mix of very high and very low features, as the latter are penalized for their inconsistency.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    lambda_var = float(parameters["lambda_var"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        var_w = np.var(active_w) if len(active_w) > 1 else 0.0
        
        # Average evidence penalized by the variance of the active validities
        return mean_w - lambda_var * var_w
        
    score_a = get_score(a)
    score_b = get_score(b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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


## THEORY 2 — pi_9
Weakest Link (Worst-Case Focus): Decision-makers evaluate an option primarily by averaging the subjective validities of its present features, but they apply a disproportionate penalty based on the option's 'weakest link' (the lowest validity active feature). Specifically, the penalty is proportional to the gap between the option's best feature and its weakest feature. This cognitive mechanism reflects a worst-case focus where attention is drawn to the most inferior attribute, naturally predicting both the dilution effect (adding weak features reduces the mean and increases the weakest-link penalty) and variance aversion (options with tighter feature clusters have smaller weakest-link penalties).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    lambda_weak = float(parameters["lambda_weak"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        min_w = np.min(active_w)
        max_w = np.max(active_w)
        
        # Averaging baseline with a penalty based on the weakest link's distance from the best feature
        return mean_w - lambda_weak * (max_w - min_w)
        
    score_a = get_score(a)
    score_b = get_score(b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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


## EXPERIMENT 1 (proposed by pi_8)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.7, 0.6, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  trial 3: A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Variance Aversion (Configural Weighting) model from the Weakest Link (Worst-Case Focus) model, we exploit their differing penalties for feature dispersion. Both models compute the mean validity of active features but apply different penalties: Variance Aversion penalizes by the variance of the active validities, whereas Weakest Link penalizes by the range (max - min). We design pairs where Option A has a smaller range but higher variance than Option B, while maintaining the same mean validity. In such cases, the Weakest Link model strictly prefers Option A (due to its smaller range penalty), whereas the Variance Aversion model strictly prefers Option B (due to its smaller variance penalty). We also include trials where the models predict indifference versus a strict preference, providing a clear map of their divergent predictions.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
By combining Trial 1 and Trial 3, we create a composite score that measures the proportion of choices aligning with Variance Aversion's predictions. In Trial 1, Variance Aversion prefers B (lower variance) while Weakest Link prefers A (smaller range). In Trial 3, Variance Aversion strongly prefers A (lower variance), while Weakest Link often predicts indifference or prefers B (depending on the gamma parameter). Averaging the alignment on these two trials reduces the per-subject variance while amplifying the gap between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (0, 1, 0, 0, 1) and b == (1, 0, 1, 1, 1):
            return 1
        if a == (0, 1, 1, 0, 1) and b == (0, 1, 0, 0, 1):
            return 3
        return 0
    
    trial_types = data.apply(is_target_trial, axis=1)
    
    t1_mask = (trial_types == 1)
    t3_mask = (trial_types == 3)
    
    p_b_t1 = data[t1_mask]['response'].mean() if t1_mask.sum() > 0 else 0.5
    p_a_t3 = 1.0 - (data[t3_mask]['response'].mean() if t3_mask.sum() > 0 else 0.5)
    
    return float((p_b_t1 + p_a_t3) / 2.0)

### RESULTS
- Predicted under pi_8 (simulated): 0.4800 (var=0.0203)
- Predicted under pi_9 (simulated): 0.2421 (var=0.0098)
- Observed on real data: 0.2095 (var=0.0222)

## EXPERIMENT 2 (proposed by pi_9)

### DESIGN
**Validities (n_features=8):** [1.0, 0.95, 0.8, 0.8, 0.7, 0.7, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the Weakest Link (Worst-Case Focus) model from the Variance Aversion (Configural Weighting) model. Both models evaluate options based on the mean validity of their active features but apply different penalties: Weakest Link penalizes by the range of active validities (max - min), whereas Variance Aversion penalizes by their variance. By creating a set of features where an option can have a large range but a small variance (e.g., extreme max and min values but many features clustered tightly at the mean) compared to an option with a smaller range but larger variance (fewer features, moderately far from the mean), we can create trials where the models predict opposite preferences.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
Trial 1 pits Weakest Link against Variance Aversion directly. Option A has a large range but small variance (many features clustered around the mean), while Option B has a smaller range but larger variance (fewer features, moderately far from the mean). Weakest Link prefers B (smaller range), while Variance Aversion prefers A (smaller variance). The metric calculates the proportion of times B is chosen in this trial, which will clearly discriminate the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    mask = (a_str == '10111101') & (b_str == '01000010')
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_8 (simulated): 0.4325 (var=0.0853)
- Predicted under pi_9 (simulated): 0.8467 (var=0.0177)
- Observed on real data: 0.8400 (var=0.0141)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    ttb_winner = np.zeros(len(data))
    
    for i in range(len(data)):
        winner = -1
        for j in range(5):
            if diff[i, j] > 0:
                winner = 0
                break
            elif diff[i, j] < 0:
                winner = 1
                break
        ttb_winner[i] = winner
        
    return float(np.mean(data['response'].values == ttb_winner))
```

**Observed (real) value:** 0.7675 (var=0.0218)
**Predicted under pi_8:** 0.7269 (var=0.0090)
**Predicted under pi_9:** 0.7027 (var=0.0111)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    strict_mask = a_wins != b_wins
    if not np.any(strict_mask):
        return 0.5
        
    tally_preds = (b_wins > a_wins).astype(int)
    responses = np.array(data['response'].tolist())
    
    match = (tally_preds[strict_mask] == responses[strict_mask])
    return float(np.mean(match))
```

**Observed (real) value:** 0.2552 (var=0.0312)
**Predicted under pi_8:** 0.3031 (var=0.0102)
**Predicted under pi_9:** 0.3010 (var=0.0098)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_choice = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice != -1:
            matches.append(1 if resp == ttb_choice else 0)
            
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.6317 (var=0.0095)
**Predicted under pi_8:** 0.6606 (var=0.0122)
**Predicted under pi_9:** 0.6531 (var=0.0082)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Focus on diagnostic trials where one option has more positive features than the other.
    # In this specific design, these are exactly the trials where the single highest-validity
    # discriminating cue favors the option with FEWER total features.
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_data = data[mask]
    sum_a_filt = sum_a[mask]
    sum_b_filt = sum_b[mask]
    
    # Determine which option has the greater number of positive features
    # 0 for A, 1 for B
    compensatory_choice = (sum_b_filt > sum_a_filt).astype(int)
    
    # Calculate the proportion of choices aligning with the compensatory (WADD-like) option
    match = (filtered_data['response'] == compensatory_choice).mean()
    return float(match)

```

**Observed (real) value:** 0.2888 (var=0.0207)
**Predicted under pi_8:** 0.2447 (var=0.0135)
**Predicted under pi_9:** 0.2772 (var=0.0096)

### Experiment 7
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # The validities are strictly decreasing from cue 0 to cue 3.
    # So the Take-The-Best (TTB) choice is simply determined by the first cue 
    # (from index 0 to 3) where the two options differ.
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(4):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0
        
    data['ttb_choice'] = data.apply(get_ttb_choice, axis=1)
    data['is_ttb'] = (data['response'] == data['ttb_choice']).astype(float)
    
    # Create a hashable string representation of the trial pair to group by
    data['trial_str'] = data.apply(lambda x: ''.join(map(str, x['option_a_ratings'])) + '_' + ''.join(map(str, x['option_b_ratings'])), axis=1)
    
    # Calculate the proportion of TTB-consistent choices for each unique trial type
    trial_means = data.groupby('trial_str')['is_ttb'].mean()
    
    # Return the standard deviation of these proportions across the 10 trial types
    return float(trial_means.std())
```

**Observed (real) value:** 0.3309 (var=0.0110)
**Predicted under pi_8:** 0.2154 (var=0.0127)
**Predicted under pi_9:** 0.3350 (var=0.0072)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_cues'] = data['option_a_ratings'].apply(sum)
    data['b_cues'] = data['option_b_ratings'].apply(sum)
    
    def ttb_favors_a(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]: return True
            if b[i] > a[i]: return False
        return False

    data['ttb_a'] = data.apply(ttb_favors_a, axis=1)
    
    df_conflict = data[data['ttb_a']].copy()
    
    high_mask = df_conflict['b_cues'] >= 3
    low_mask = df_conflict['b_cues'] <= 2
    
    p_high = df_conflict.loc[high_mask, 'response'].mean()
    p_low = df_conflict.loc[low_mask, 'response'].mean()
    
    if pd.isna(p_high): p_high = 0.5
    if pd.isna(p_low): p_low = 0.5
    
    return float(p_high - p_low)
```

**Observed (real) value:** -0.1823 (var=0.0238)
**Predicted under pi_8:** 0.0471 (var=0.0153)
**Predicted under pi_9:** 0.0951 (var=0.0299)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where Option A's features are a strict subset of Option B's features
    is_subset = np.all(a_ratings <= b_ratings, axis=1) & (np.sum(a_ratings, axis=1) < np.sum(b_ratings, axis=1))
    
    subset_data = data[is_subset]
    if len(subset_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen
    return float(np.mean(subset_data['response'] == 0))
```

**Observed (real) value:** 0.8678 (var=0.0153)
**Predicted under pi_8:** 0.8183 (var=0.0062)
**Predicted under pi_9:** 0.8403 (var=0.0098)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    p_b_3 = data.loc[b_sum == 3, 'response'].mean()
    p_b_2 = data.loc[b_sum == 2, 'response'].mean()
    return float(p_b_3 - p_b_2)
```

**Observed (real) value:** -0.1200 (var=0.0258)
**Predicted under pi_8:** 0.0303 (var=0.0130)
**Predicted under pi_9:** 0.0600 (var=0.0183)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    # Determine TTB choice (first discriminating cue)
    diff = A - B
    idx = np.argmax(np.abs(diff), axis=1)
    ttb_choice = np.where(diff[np.arange(len(diff)), idx] == 1, 0, 1)
    
    # Determine Averaging choice
    sum_a = np.sum(A, axis=1)
    sum_b = np.sum(B, axis=1)
    
    avg_a = np.zeros(len(A))
    mask_a = sum_a > 0
    avg_a[mask_a] = np.sum(A[mask_a] * val, axis=1) / sum_a[mask_a]
    
    avg_b = np.zeros(len(B))
    mask_b = sum_b > 0
    avg_b[mask_b] = np.sum(B[mask_b] * val, axis=1) / sum_b[mask_b]
    
    avg_choice = np.where(avg_a > avg_b, 0, np.where(avg_b > avg_a, 1, -1))
    
    # Filter trials where TTB and Averaging disagree
    disagree_mask = (avg_choice != -1) & (ttb_choice != avg_choice)
    
    if not np.any(disagree_mask):
        return 0.5
        
    responses = data['response'].values[disagree_mask]
    ttb_choices = ttb_choice[disagree_mask]
    
    return float(np.mean(responses == ttb_choices))

```

**Observed (real) value:** 0.1572 (var=0.0102)
**Predicted under pi_8:** 0.1875 (var=0.0125)
**Predicted under pi_9:** 0.1233 (var=0.0071)

### Experiment 12
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.1454 (var=0.0162)
**Predicted under pi_8:** 0.1600 (var=0.0111)
**Predicted under pi_9:** 0.1227 (var=0.0068)

### Experiment 13
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has the highest-validity feature (index 0) and Option B does not.
    a_has_best = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_best = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    mask = a_has_best & ~b_has_best
    
    if mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times the subject chose Option B (response == 1)
    # despite Option A having the best possible feature.
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.7428 (var=0.0066)
**Predicted under pi_8:** 0.7783 (var=0.0258)
**Predicted under pi_9:** 0.8689 (var=0.0100)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    extreme_chosen = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        idx_a = a.index(1) if 1 in a else len(a)
        idx_b = b.index(1) if 1 in b else len(b)
        
        if idx_a < idx_b:
            if resp == 0:
                extreme_chosen += 1
            total += 1
        elif idx_b < idx_a:
            if resp == 1:
                extreme_chosen += 1
            total += 1
            
    return float(extreme_chosen / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.1758 (var=0.0096)
**Predicted under pi_8:** 0.1379 (var=0.0131)
**Predicted under pi_9:** 0.1619 (var=0.0184)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has the highest validity feature (index 0)
    # but also has at least one other feature, creating high variance.
    # Option B in the design always lacks the highest validity feature in these trials.
    is_target_trial = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) > 1)
    target_data = data[is_target_trial]
    
    if len(target_data) == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen (response == 0)
    return (1.0 - target_data['response']).mean()
```

**Observed (real) value:** 0.3307 (var=0.0208)
**Predicted under pi_8:** 0.2930 (var=0.0358)
**Predicted under pi_9:** 0.1910 (var=0.0142)

### Experiment 16
**Design**
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Filter for trials where both options have the same number of positive features
    # In the design, these are trials 1-6. In all of these, Option A has a higher 
    # maximum validity (anchor) but exactly the same remaining features as Option B.
    # Therefore, Option B's features are more clustered (lower variance).
    is_matched = data['option_a_ratings'].apply(sum) == data['option_b_ratings'].apply(sum)
    matched_data = data[is_matched]
    if len(matched_data) == 0:
        return 0.0
    # Return the proportion of times Option B was chosen
    return float(matched_data['response'].mean())
```

**Observed (real) value:** 0.8456 (var=0.0113)
**Predicted under pi_8:** 0.7581 (var=0.0347)
**Predicted under pi_9:** 0.8089 (var=0.0259)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 16 experiments, Theory 2 (Weakest Link) significantly outperforms Theory 1 (Variance Aversion), especially on the newly proposed Experiments 1 and 2 where it almost perfectly captures the observed data. Theory 2 also better tracks the data in Experiments 7, 9, 14, and 16. However, both theories fail to capture the negative metrics in Experiments 8 and 10, which measure choices under high vs. low conflict based on cue counts. Theory 1's variance penalty mechanism is overly sensitive to the dispersion of active validities, leading to wildly inaccurate predictions in Experiment 1 and 2. Because Theory 2 is a strong baseline but still misses some conflict-driven phenomena, Theory 1 should be completely replaced by a new theory that can address these gaps.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Rank-Based Heuristic' or 'Relative Cue Difference' theory. Instead of aggregating validities with a holistic penalty (variance or weakest link), this new theory should posit that decision-makers evaluate options by comparing them directly on a cue-by-cue basis, but the weight of each cue's difference is non-linearly scaled by its rank or validity. Alternatively, consider a 'Negative Evidence / Regret' theory where the absence of high-validity features is penalized more heavily than the presence of low-validity features. This could help capture the negative values in Experiments 8 and 10, where high-conflict situations cause subjects to systematically deviate from Take-The-Best, while still competing strongly against Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 12595,
  "candidates_token_count": 363,
  "total_token_count": 14195
}
```
