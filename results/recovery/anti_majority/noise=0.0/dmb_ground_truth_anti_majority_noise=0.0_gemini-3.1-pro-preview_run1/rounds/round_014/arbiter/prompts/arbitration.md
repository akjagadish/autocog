# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_16" and "pi_14") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_16" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_14" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_16
Variance Aversion and Feature Consistency

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_var = float(parameters["lambda_var"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Subjective utility: validities transformed and shifted by a threshold
    w = (val ** gamma) - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        base_score = np.sum(active_w)
        
        # Apply variance penalty for multiple unique features
        if len(active_w) > 1:
            # Use standard deviation scaled by the number of active features
            # so the penalty competes symmetrically with the additive base score.
            std_w = np.std(active_w)
            return base_score - lambda_var * std_w * len(active_w)
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
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


## THEORY 2 — pi_14
Thresholded Unique Features with Spread Penalty: Decision-makers simplify choices by cancelling out shared features, then evaluate the unique features relative to a subjective validity threshold. Features above the threshold provide positive evidence, while those below act as penalties. These values are integrated additively, but options with multiple unique features suffer a conflict penalty proportional to the spread (max - min) of their thresholded validities. This penalizes options with a wide variance in their unique features while strictly preserving shared-feature cancellation.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Transform validities and apply subjective threshold
    v_trans = val ** gamma
    w = v_trans - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Additive integration of thresholded evidence
        base_score = np.sum(active_w)
        
        # Spread penalty applied if there are multiple unique features
        if len(active_w) > 1:
            conflict_penalty = lambda_penalty * (np.max(active_w) - np.min(active_w))
            return base_score - conflict_penalty
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
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


## EXPERIMENT 1 (proposed by pi_16)

### DESIGN
**Validities (n_features=8):** [0.95, 0.95, 0.95, 0.75, 0.75, 0.55, 0.55, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1]
  trial 3: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Variance Aversion) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their differing penalties for feature dispersion. Both models compute the sum of thresholded validities for unique features but apply different penalties: the Competing Theory penalizes by the range (max - min) of the active validities, whereas the Advocated Theory penalizes by the standard deviation scaled by the number of active features. We design pairs where Option A and Option B have similar sums of validities and the exact same range, but different standard deviations (e.g., Option A has features clustered at the mean, while Option B has features clustered at the extremes). The Competing Theory applies the identical spread penalty to both options, whereas the Advocated Theory assigns a larger variance penalty to Option B (the extreme cluster) and thus strongly prefers Option A.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 1, Option A and Option B have the exact same number of unique features (4) and the exact same max-min spread of validities (0.95 to 0.55). Consequently, the Competing Theory applies the identical spread penalty to both options, leaving its preference to be determined solely by small differences in the base score (depending on gamma). In contrast, the Advocated Theory penalizes by the standard deviation of validities. Option B's validities are clustered at the extremes (two 0.95s, two 0.55s), resulting in a higher standard deviation than Option A (which has validities of 0.95, 0.75, 0.75, 0.55). Thus, the Advocated Theory uniquely predicts a strong preference for Option A due to its lower variance penalty. Isolating this specific trial provides a clean, highly discriminative signal.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify Trial 1 where Option A and Option B have the exact same range of validities
    # but different standard deviations.
    # Trial 1: A=[1, 0, 0, 1, 1, 1, 0, 0], B=[0, 1, 1, 0, 0, 0, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(tuple) == (1, 0, 0, 1, 1, 1, 0, 0)
    
    # Calculate the proportion of times Option A was chosen in Trial 1
    trial_1_data = data[is_trial_1]
    if len(trial_1_data) == 0:
        return 0.5
        
    return float((trial_1_data['response'] == 0).mean())

### RESULTS
- Predicted under pi_16 (simulated): 0.8638 (var=0.0229)
- Predicted under pi_14 (simulated): 0.2125 (var=0.0431)
- Observed on real data: 0.1675 (var=0.0112)

## EXPERIMENT 2 (proposed by pi_14)

### DESIGN
**Validities (n_features=12):** [0.9, 0.9, 0.9, 0.85, 0.85, 0.7, 0.7, 0.55, 0.55, 0.5, 0.5, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Thresholded Unique Features with Spread Penalty) from the Competing Theory (Variance Aversion and Feature Consistency), we exploit their differing penalties for feature dispersion. Both models compute the sum of thresholded validities for unique features but apply different penalties: the Advocated Theory penalizes by the range (max - min) of the active validities, whereas the Competing Theory penalizes by the standard deviation scaled by the number of active features. We design pairs where Option A and Option B have the exact same sum of validities and the exact same range, but different standard deviations (e.g., Option A has validities clustered at the mean, while Option B has validities clustered at the extremes). We also design full reversal trials where Option A has a larger range but a smaller standard deviation than Option B. The Advocated Theory will strictly prefer the option with the smaller range, while the Competing Theory will strictly prefer the option with the smaller standard deviation.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
Trials 2 and 4 act as strict reversal trials between the two theories. In these trials, Option A's unique features are [0.9, 0.7, 0.7, 0.5] and Option B's unique features are [0.85, 0.85, 0.55, 0.55]. The Advocated Theory measures spread using the range: A's range (0.4) is larger than B's range (0.3), so it penalizes A more and strictly prefers B (response -> 1). The Competing Theory measures spread using standard deviation: A's standard deviation (~0.141) is strictly smaller than B's standard deviation (0.15), so it penalizes B more and strictly prefers A (response -> 0). Because these inequalities hold across all possible values of the non-linear utility parameter (gamma), the mean response on these trials provides a highly robust, parameter-independent discriminator.

Source:
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    t2_a = (1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0)
    t2_b = (0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0)
    
    t4_a = (1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0)
    t4_b = (0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0)
    
    mask = ((a_keys == t2_a) & (b_keys == t2_b)) | ((a_keys == t4_a) & (b_keys == t4_b))
    return float(data[mask]['response'].mean())

### RESULTS
- Predicted under pi_16 (simulated): 0.5932 (var=0.1372)
- Predicted under pi_14 (simulated): 0.8189 (var=0.0197)
- Observed on real data: 0.8526 (var=0.0133)

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
**Predicted under pi_16:** 0.7569 (var=0.0178)
**Predicted under pi_14:** 0.7602 (var=0.0175)

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
**Predicted under pi_16:** 0.2876 (var=0.0055)
**Predicted under pi_14:** 0.3014 (var=0.0318)

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
**Predicted under pi_16:** 0.6648 (var=0.0097)
**Predicted under pi_14:** 0.7217 (var=0.0250)

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
**Predicted under pi_16:** 0.1944 (var=0.0125)
**Predicted under pi_14:** 0.1997 (var=0.0245)

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
**Predicted under pi_16:** 0.3234 (var=0.0060)
**Predicted under pi_14:** 0.2511 (var=0.0107)

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
**Predicted under pi_16:** 0.1379 (var=0.0413)
**Predicted under pi_14:** 0.1129 (var=0.0217)

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
**Predicted under pi_16:** 0.8339 (var=0.0092)
**Predicted under pi_14:** 0.7411 (var=0.0548)

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
**Predicted under pi_16:** 0.0237 (var=0.0282)
**Predicted under pi_14:** -0.0116 (var=0.0155)

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
**Predicted under pi_16:** 0.1142 (var=0.0051)
**Predicted under pi_14:** 0.1542 (var=0.0224)

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
**Predicted under pi_16:** 0.1171 (var=0.0053)
**Predicted under pi_14:** 0.1571 (var=0.0082)

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
**Predicted under pi_16:** 0.7206 (var=0.0050)
**Predicted under pi_14:** 0.7394 (var=0.0075)

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
**Predicted under pi_16:** 0.1333 (var=0.0115)
**Predicted under pi_14:** 0.1758 (var=0.0269)

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
**Predicted under pi_16:** 0.5820 (var=0.0018)
**Predicted under pi_14:** 0.5510 (var=0.0046)

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
**Predicted under pi_16:** 0.1794 (var=0.0119)
**Predicted under pi_14:** 0.1606 (var=0.0128)

### Experiment 17
**Design**
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.2095 (var=0.0222)
**Predicted under pi_16:** 0.2221 (var=0.0345)
**Predicted under pi_14:** 0.2295 (var=0.0326)

### Experiment 18
**Design**
  A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    mask = (a_str == '10111101') & (b_str == '01000010')
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8400 (var=0.0141)
**Predicted under pi_16:** 0.8675 (var=0.0228)
**Predicted under pi_14:** 0.7908 (var=0.0552)

### Experiment 19
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Project lists to strings for hashable grouping
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # response=0 means A was chosen, response=1 means B was chosen.
    # So 1 - mean(response) is the empirical probability of choosing A.
    p_A = 1.0 - data.groupby(['A_str', 'B_str'])['response'].mean()
    
    def get_p(a, b):
        if (a, b) in p_A.index:
            return p_A.loc[(a, b)]
        return 0.5
        
    # Trial pairs that differ only by a shared feature added to both options
    # Advocated theory predicts 0 difference; Weakest Link predicts large shifts.
    
    # Trial 2 vs Trial 1 (Shared feature 1 added)
    d1 = get_p('11000', '10110') - get_p('01000', '00110')
    
    # Trial 3 vs Trial 1 (Shared feature 5 added)
    d2 = get_p('01001', '00111') - get_p('01000', '00110')
    
    # Trial 5 vs Trial 4 (Shared feature 1 added)
    d3 = get_p('11100', '10011') - get_p('01100', '00011')
    
    # Trial 8 vs Trial 7 (Shared feature 1 added)
    d4 = get_p('10100', '10010') - get_p('00100', '00010')
    
    return float(np.abs(d1) + np.abs(d2) + np.abs(d3) + np.abs(d4))

```

**Observed (real) value:** 0.0567 (var=0.0378)
**Predicted under pi_16:** 0.0767 (var=0.0438)
**Predicted under pi_14:** 0.0600 (var=0.0527)

### Experiment 20
**Design**
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_a_single = data['option_a_ratings'].apply(sum) == 1
    is_b_single = data['option_b_ratings'].apply(sum) == 1
    
    target_trials = is_a_single | is_b_single
    if not target_trials.any():
        return 0.0
        
    df_target = data[target_trials]
    
    chose_multi = ((df_target['option_b_ratings'].apply(sum) == 1) & (df_target['response'] == 0)) | \
                  ((df_target['option_a_ratings'].apply(sum) == 1) & (df_target['response'] == 1))
                  
    return float(chose_multi.mean())

```

**Observed (real) value:** 0.2232 (var=0.0305)
**Predicted under pi_16:** 0.1363 (var=0.0142)
**Predicted under pi_14:** 0.1311 (var=0.0189)

### Experiment 21
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Response == 0 means Option A was chosen
    chose_a = (data['response'] == 0).astype(float)
    
    # Identify base trials (last feature is 0 for both options)
    # and derivative trials (last feature is 1 for both options)
    is_derivative = data['option_a_ratings'].apply(lambda x: x[-1] == 1) & data['option_b_ratings'].apply(lambda x: x[-1] == 1)
    is_base = data['option_a_ratings'].apply(lambda x: x[-1] == 0) & data['option_b_ratings'].apply(lambda x: x[-1] == 0)
    
    # Calculate the proportion of choosing Option A in both trial types
    p_a_base = chose_a[is_base].mean()
    p_a_deriv = chose_a[is_derivative].mean()
    
    # The metric is the difference in preference for A
    # Advocated theory: ~0 (shared feature is cancelled)
    # Competing theory: non-zero (shared weak feature increases weakest-link penalty asymmetrically)
    return float(p_a_base - p_a_deriv)
```

**Observed (real) value:** 0.0008 (var=0.0064)
**Predicted under pi_16:** -0.0042 (var=0.0033)
**Predicted under pi_14:** -0.0175 (var=0.0052)

### Experiment 22
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials by their option_a_ratings string representation
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[1, 0, 0, 0, 0]
    # Trial 2: A=[1, 0, 0, 0, 1]
    # Trial 5: A=[0, 0, 1, 0, 0]
    # Trial 6: A=[0, 0, 1, 0, 1]
    t1 = data[data['a_str'] == '10000']
    t2 = data[data['a_str'] == '10001']
    t5 = data[data['a_str'] == '00100']
    t6 = data[data['a_str'] == '00101']
    
    # Response is 0 for A, 1 for B, so (1 - mean(response)) is the proportion of A choices
    p_A_t1 = 1.0 - t1['response'].mean() if len(t1) > 0 else 0.5
    p_A_t2 = 1.0 - t2['response'].mean() if len(t2) > 0 else 0.5
    p_A_t5 = 1.0 - t5['response'].mean() if len(t5) > 0 else 0.5
    p_A_t6 = 1.0 - t6['response'].mean() if len(t6) > 0 else 0.5
    
    # Advocated theory predicts P(A) drops from T1 to T2, and P(A) rises from T5 to T6.
    # Competing theory predicts no change in either pair since the shared features cancel out.
    return (p_A_t1 - p_A_t2) + (p_A_t6 - p_A_t5)
```

**Observed (real) value:** 0.0767 (var=0.0438)
**Predicted under pi_16:** 0.0233 (var=0.0378)
**Predicted under pi_14:** 0.0183 (var=0.0434)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    t2_a, t2_b = (1, 0, 0, 1, 1), (0, 0, 1, 1, 0)
    t5_a, t5_b = (0, 1, 0, 1, 1), (0, 0, 1, 0, 1)
    
    b_chosen = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t2_a and b == t2_b:
            b_chosen.append(resp == 1)
        elif a == t2_b and b == t2_a:
            b_chosen.append(resp == 0)
        elif a == t5_a and b == t5_b:
            b_chosen.append(resp == 1)
        elif a == t5_b and b == t5_a:
            b_chosen.append(resp == 0)
            
    if not b_chosen:
        return 0.5
    return sum(b_chosen) / len(b_chosen)
```

**Observed (real) value:** 0.8175 (var=0.0099)
**Predicted under pi_16:** 0.8856 (var=0.0097)
**Predicted under pi_14:** 0.8400 (var=0.0165)

### Experiment 24
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_target(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # Identify trials where A has feature 2, B has features 1 and 3
        # and they share at least one low-validity feature (features 4 or 5)
        return (a[0] == 0 and a[1] == 1 and a[2] == 0 and
                b[0] == 1 and b[1] == 0 and b[2] == 1 and
                sum(a[3:]) > 0)
                
    mask = data.apply(is_target, axis=1)
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1744 (var=0.0135)
**Predicted under pi_16:** 0.1323 (var=0.0212)
**Predicted under pi_14:** 0.1892 (var=0.0363)

### Experiment 25
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def is_target_trial(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return a[4] == 0 and b[4] == 1 and a[:4] == b[:4]

    mask = data.apply(is_target_trial, axis=1)
    target_trials = data[mask]
    
    if len(target_trials) == 0:
        return 0.5
        
    return float(target_trials['response'].mean())
```

**Observed (real) value:** 0.1375 (var=0.0066)
**Predicted under pi_16:** 0.2562 (var=0.0543)
**Predicted under pi_14:** 0.2529 (var=0.0642)

### Experiment 26
**Design**
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Identify trials where B is an empty option and A contains only low-validity features
    mask = (b_str == '00000') & (a_str.isin(['00001', '00010', '00011']))
    
    if mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times the subject chose B (response == 1)
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8830 (var=0.0105)
**Predicted under pi_16:** 0.8052 (var=0.0356)
**Predicted under pi_14:** 0.7363 (var=0.0482)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_trials_a_chosen = 0
    target_trials_total = 0
    
    for _, row in data.iterrows():
        a_ratings = row['option_a_ratings']
        b_ratings = row['option_b_ratings']
        
        unique_a = [i for i, (a, b) in enumerate(zip(a_ratings, b_ratings)) if a == 1 and b == 0]
        unique_b = [i for i, (a, b) in enumerate(zip(a_ratings, b_ratings)) if b == 1 and a == 0]
        
        if len(unique_a) == 1:
            if len(unique_b) == 0 or unique_a[0] < min(unique_b):
                target_trials_total += 1
                if row['response'] == 0:
                    target_trials_a_chosen += 1
                    
    if target_trials_total == 0:
        return 0.5
    return target_trials_a_chosen / target_trials_total
```

**Observed (real) value:** 0.6378 (var=0.0198)
**Predicted under pi_16:** 0.7731 (var=0.0074)
**Predicted under pi_14:** 0.7378 (var=0.0098)

### Experiment 28
**Design**
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask = ((data['A_str'] == '00000') & (data['B_str'] == '00001')) | \
           ((data['A_str'] == '10000') & (data['B_str'] == '10001'))
    
    if mask.sum() == 0:
        return 0.0
    
    return float((data.loc[mask, 'response'] == 0).mean())

```

**Observed (real) value:** 0.8733 (var=0.0127)
**Predicted under pi_16:** 0.7625 (var=0.0536)
**Predicted under pi_14:** 0.6850 (var=0.0631)

### Experiment 29
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Trial 4: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0]
    mask_4 = (a_keys == (0, 1, 1, 0, 0, 0)) & (b_keys == (1, 0, 0, 0, 1, 0))
    # Trial 5: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
    mask_5 = (a_keys == (1, 0, 0, 0, 1, 0)) & (b_keys == (0, 1, 1, 0, 0, 0))
    # Trial 8: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
    mask_8 = (a_keys == (1, 0, 0, 0, 0, 1)) & (b_keys == (0, 1, 1, 0, 0, 0))
    
    choices_high_max = 0
    total = 0
    
    if mask_4.any():
        choices_high_max += (data.loc[mask_4, 'response'] == 1).sum()
        total += mask_4.sum()
        
    if mask_5.any():
        choices_high_max += (data.loc[mask_5, 'response'] == 0).sum()
        total += mask_5.sum()
        
    if mask_8.any():
        choices_high_max += (data.loc[mask_8, 'response'] == 0).sum()
        total += mask_8.sum()
        
    if total == 0:
        return 0.5
        
    return float(choices_high_max / total)
```

**Observed (real) value:** 0.3052 (var=0.0260)
**Predicted under pi_16:** 0.1504 (var=0.0107)
**Predicted under pi_14:** 0.1356 (var=0.0139)

### Experiment 30
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has a large spread of validities (features 1 and 5)
    mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0, 1])
    if mask.sum() == 0:
        return 0.0
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.8579 (var=0.0059)
**Predicted under pi_16:** 0.8789 (var=0.0160)
**Predicted under pi_14:** 0.8568 (var=0.0117)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both targeted experiments, Theory 2 (pi_14, Spread Penalty) dramatically outperforms Theory 1 (pi_16, Variance Aversion). In Experiment 1, proposed by Theory 1 to highlight its standard deviation penalty, the observed data (0.1675) strongly contradicted Theory 1's prediction (0.8638) and aligned almost perfectly with Theory 2's prediction (0.2125). In Experiment 2, proposed by Theory 2 to highlight its range-based penalty over standard deviation, the observed data (0.8526) again favored Theory 2 (0.8189) over Theory 1 (0.5932). Theory 1's core claim\u2014that subjects calculate and penalize the standard deviation of validities\u2014is empirically falsified. Theory 2's mechanism of range-based penalty (max - min) better captures the specific conflict penalties humans apply when integrating multiple unique features.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Theory 1 is degenerate because its core mechanism (penalizing standard deviation of feature validities) was falsified by its own proposed experiment, with subjects instead behaving in accordance with Theory 2's spread (max-min) penalty. Propose a brand-new theory to replace Theory 1. Instead of tweaking variance penalties, the new theory should explore a fundamentally different cognitive mechanism. For example, consider a Sequential Sampling or Lexicographic Semi-Order model where decision-makers compare features one by one in order of validity, but only stop if the difference in validities exceeds a subjective threshold (otherwise they tally the remaining features). Alternatively, explore a theory based on Diminishing Marginal Utility for additional features, where the first unique feature provides full evidence but subsequent unique features are heavily discounted, explaining why options with many weak features lose to options with one strong feature without requiring a 'conflict' penalty."
}
```

## Usage

```json
{
  "prompt_token_count": 22381,
  "candidates_token_count": 430,
  "total_token_count": 23432
}
```
