# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_8" and "pi_7") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_8" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_7" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_7
Anchor and Adjust: Decision-makers evaluate options by first identifying the highest-validity positive feature, which serves as an anchor. Subsequent positive features adjust this anchored evaluation downwards if their validities are lower than the anchor. This explains the 'less is more' dilution effect, as adding low-validity features drags the overall evaluation down, while naturally predicting that an option with only high-validity features will be preferred over one that adds additional low-validity features (strict subset preference).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # The highest validity positive feature is the anchor
        anchor = np.max(active_w)
        idx = np.argmax(active_w)
        
        # Remaining features adjust the evaluation
        other_w = np.delete(active_w, idx)
        if len(other_w) == 0:
            return anchor
            
        # Weighted average of anchor and adjusting features
        return (anchor + alpha * np.sum(other_w)) / (1.0 + alpha * len(other_w))
        
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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the Variance Aversion (Configural Weighting) model from the Anchor and Adjust model by exploiting their fundamentally different treatments of feature dispersion. The Anchor and Adjust model evaluates an option by anchoring on its highest-validity positive feature and adjusting based on the sum of the remaining features. In contrast, the Variance Aversion model computes the mean of the positive features and applies a penalty proportional to their variance. We construct trials that pit an option with a very high anchor but high variance (e.g., one very high and one very low validity feature) against an option with a lower anchor but tightly clustered, moderately high features (low variance). By varying the difference in means and the severity of the variance disparity across trials, we can distinguish the variance penalty (which strongly suppresses the high-dispersion option) from the anchoring effect (which elevates the high-dispersion option due to its superior maximum validity).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the choice probability of Option A on trials where Option A possesses the highest-validity feature but also has other features (thus having high feature variance). The 'Anchor and Adjust' model strongly favors Option A in these trials because it anchors on the highest-validity feature (0.95). In contrast, the 'Variance Aversion' model penalizes Option A due to the high variance among its active features, leading to a much lower preference for A compared to the anchoring model.

Source:
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

### RESULTS
- Predicted under pi_8 (simulated): 0.2930 (var=0.0358)
- Predicted under pi_7 (simulated): 0.4693 (var=0.0140)
- Observed on real data: 0.3307 (var=0.0208)

## EXPERIMENT 2 (proposed by pi_7)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Anchor and Adjust model from the Variance Aversion model, we systematically manipulate the single highest-validity feature (the anchor) while keeping the remaining features identical across the two options. The Anchor and Adjust model evaluates an option by taking its maximum validity feature and adjusting it based on the others; thus, if Option A and Option B share the exact same set of lower-validity features but Option A has a higher anchor, Anchor and Adjust will universally prefer Option A, as improving the anchor strictly increases the evaluation. In contrast, the Variance Aversion model computes the mean of the validities and applies a penalty proportional to their variance. By increasing the highest validity feature while holding the lower ones constant, the variance of the feature set increases. If the variance penalty parameter is sufficiently high, the Variance Aversion model will paradoxically prefer Option B (the option with the lower maximum validity) because its features are more tightly clustered, leading to a smaller configural penalty. This provides a sharp qualitative divergence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By isolating trials where both options have the same number of features (trials 1-6), we zero in on the core dissociation between the models. In these trials, Option A always possesses the highest-validity feature (the best anchor), but Option B's features are more tightly clustered (lower variance). The Anchor and Adjust model will strongly and universally prefer Option A because improving the anchor strictly increases the evaluation, driving the metric (proportion of Option B choices) close to 0. In contrast, the Variance Aversion model penalizes Option A for its higher feature variance, driving it to frequently prefer Option B and resulting in a significantly higher metric value.

Source:
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

### RESULTS
- Predicted under pi_8 (simulated): 0.7581 (var=0.0347)
- Predicted under pi_7 (simulated): 0.4275 (var=0.0161)
- Observed on real data: 0.8456 (var=0.0113)

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
**Predicted under pi_7:** 0.7963 (var=0.0147)

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
**Predicted under pi_7:** 0.2562 (var=0.0079)

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
**Predicted under pi_7:** 0.7623 (var=0.0122)

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
**Predicted under pi_7:** 0.2288 (var=0.0145)

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
**Predicted under pi_7:** 0.1707 (var=0.0032)

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
**Predicted under pi_7:** -0.0130 (var=0.0073)

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
**Predicted under pi_7:** 0.7853 (var=0.0078)

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
**Predicted under pi_7:** -0.0331 (var=0.0051)

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
**Predicted under pi_7:** 0.2328 (var=0.0172)

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
**Predicted under pi_7:** 0.1398 (var=0.0117)

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
**Predicted under pi_7:** 0.6978 (var=0.0229)

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
**Predicted under pi_7:** 0.2781 (var=0.0334)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both the newly proposed experiments and the historical ones, Theory 1 (Variance Aversion, pi_8) significantly outperforms Theory 2 (Anchor and Adjust, pi_7). In Experiment 1, subjects preferred the high-variance Option A much less than pi_7 predicted, aligning with pi_8's prediction. In Experiment 2, subjects overwhelmingly chose Option B (the lower-variance option), yielding a metric of 0.8456, which pi_8 closely anticipated (0.7581) while pi_7 completely missed (0.4275). Theory 1's variance penalty accurately captures the core phenomenon that subjects penalize options with a wide dispersion of cue validities. Theory 2's anchoring mechanism fails to capture this 'weakest link' or inconsistency penalty. Since Theory 1 is performing exceptionally well, Theory 2 should be replaced with a new theory that can offer a strong, distinct mechanistic alternative to Variance Aversion for explaining this penalty.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Weakest Link' or 'Worst-Case Focus' theory. Instead of computing a global variance penalty (like Theory 1), this theory posits that decision-makers evaluate an option through a baseline additive or averaging process, but apply a specific, disproportionate penalty based on the *lowest* validity active feature (the weakest link). This provides a distinct cognitive mechanism (attention drawn to the worst attribute) that naturally predicts the dilution and variance-aversion effects observed in the data, serving as a strong mechanistic competitor to Theory 1's holistic variance computation."
}
```

## Usage

```json
{
  "prompt_token_count": 11707,
  "candidates_token_count": 362,
  "total_token_count": 13050
}
```
