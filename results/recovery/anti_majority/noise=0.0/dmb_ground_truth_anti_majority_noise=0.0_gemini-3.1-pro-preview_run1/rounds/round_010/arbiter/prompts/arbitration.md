# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_11" and "pi_12") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_11" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_12" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_11
Unique Features Weakest-Link: Decision-makers simplify choices by first cancelling out features shared by both options. They then evaluate each option based solely on its unique features, computing the average subjective validity of these features but applying a disproportionate penalty based on the option's 'weakest link' (the gap between its best and worst unique features).

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
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    def get_score(unique_x):
        active_w = w[unique_x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        min_w = np.min(active_w)
        max_w = np.max(active_w)
        
        # Averaging baseline with a penalty based on the weakest link's distance from the best feature
        return mean_w - lambda_weak * (max_w - min_w)
        
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


## THEORY 2 — pi_12
Strict Averaging of Active Features: Decision-makers evaluate an option by computing the simple mean of the subjective validities of all features present in that option. This holistic averaging naturally produces the dilution effect (adding weak features to a strong subset lowers the average) while preserving relative ordinal preferences when shared features are added to both options, avoiding the catastrophic preference reversals caused by range-based penalties.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    # Calculate the mean subjective validity of active features for each option
    score_a = np.mean(w[a > 0]) if np.sum(a) > 0 else 0.0
    score_b = np.mean(w[b > 0]) if np.sum(b) > 0 else 0.0
    
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


## EXPERIMENT 1 (proposed by pi_11)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 1]
  trial 6: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Unique Features Weakest-Link) from the Competing Theory (Strict Averaging of Active Features), we exploit their fundamentally different treatments of shared features and feature dispersion. The Advocated Theory cancels out shared features and applies a weakest-link penalty (max - min) to the remaining unique features. The Competing Theory computes a simple average of all active features, regardless of whether they are shared. We design base trials where Option A has two unique features with a large range (but the same mean as Option B) and Option B has a single unique feature (zero range). The Competing Theory predicts indifference, while the Advocated Theory strongly prefers Option B due to the weakest-link penalty on A. We then create derivative trials by adding a shared low-validity feature. Because the Competing Theory averages all features, the shared low-validity feature dilutes Option B's average more than Option A's (since B has fewer features), causing the Competing Theory to prefer Option A. Meanwhile, the Advocated Theory cancels the shared feature and continues to strictly prefer Option B.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In Trials 2 and 5, Option A has two unique features with a large range of validities, while Option B has a single unique feature with an intermediate validity. Both options also share a low-validity feature. The Advocated Theory cancels the shared feature and penalizes Option A for its large range, leading to a strong preference for Option B. The Competing Theory simply averages all active features; because Option B has fewer features overall, the shared low-validity feature dilutes Option B's average more than Option A's, leading to a preference for Option A. This metric computes the proportion of times Option B is chosen in these critical trials, cleanly differentiating the two theories.

Source:
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

### RESULTS
- Predicted under pi_11 (simulated): 0.8562 (var=0.0161)
- Predicted under pi_12 (simulated): 0.3369 (var=0.0193)
- Observed on real data: 0.8175 (var=0.0099)

## EXPERIMENT 2 (proposed by pi_12)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  trial 3: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Strict Averaging of Active Features) from the Competing Theory (Unique Features Weakest-Link), we exploit their fundamentally different treatments of shared features and feature dispersion. The Competing Theory cancels out any shared features and applies a 'weakest-link' penalty based on the range of the remaining unique features. The Advocated Theory evaluates options holistically by computing the simple mean of all active features, regardless of whether they are shared. We design base trials where Option A has a single high-validity unique feature, and Option B has two unique features with the same mean but a larger range. The Competing Theory strongly prefers Option A because Option B suffers a weakest-link penalty. We then create derivative trials by adding shared low-validity features to both options. Because the Competing Theory cancels shared features, its preference for Option A remains unchanged. However, for the Advocated Theory, adding shared low-validity features dilutes Option A's average much more severely (since it has fewer features) than Option B's average, causing a clear preference reversal where Option B is now favored.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of times Option B is chosen in trials where Option A has a single high-validity unique feature and Option B has two unique features with the same mean but a larger range, AND where both options share at least one low-validity feature (Trials 2, 3, and 4). The Competing Theory cancels shared features and applies a weakest-link penalty to B's unique features, consistently preferring Option A. The Advocated Theory averages all active features; because Option A has fewer features initially, adding shared low-validity features dilutes its average much more than it dilutes Option B's average, causing a preference reversal where Option B is favored. Thus, the Advocated Theory predicts a much higher value for this metric (preference for B) compared to the Competing Theory.

Source:
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

### RESULTS
- Predicted under pi_11 (simulated): 0.1313 (var=0.0077)
- Predicted under pi_12 (simulated): 0.6749 (var=0.0174)
- Observed on real data: 0.1744 (var=0.0135)

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
**Predicted under pi_11:** 0.7431 (var=0.0078)
**Predicted under pi_12:** 0.7773 (var=0.0132)

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
**Predicted under pi_11:** 0.3002 (var=0.0074)
**Predicted under pi_12:** 0.2433 (var=0.0123)

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
**Predicted under pi_11:** 0.6800 (var=0.0148)
**Predicted under pi_12:** 0.7779 (var=0.0157)

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
**Predicted under pi_11:** 0.2238 (var=0.0163)
**Predicted under pi_12:** 0.2734 (var=0.0177)

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
**Predicted under pi_11:** 0.3516 (var=0.0081)
**Predicted under pi_12:** 0.0636 (var=0.0028)

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
**Predicted under pi_11:** 0.1365 (var=0.0288)
**Predicted under pi_12:** -0.0521 (var=0.0082)

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
**Predicted under pi_11:** 0.4553 (var=0.0389)
**Predicted under pi_12:** 0.7244 (var=0.0134)

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
**Predicted under pi_11:** 0.0881 (var=0.0173)
**Predicted under pi_12:** -0.0663 (var=0.0065)

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
**Predicted under pi_11:** 0.1367 (var=0.0067)
**Predicted under pi_12:** 0.4917 (var=0.0161)

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
**Predicted under pi_11:** 0.1165 (var=0.0071)
**Predicted under pi_12:** 0.2948 (var=0.0132)

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
**Predicted under pi_11:** 0.7500 (var=0.0037)
**Predicted under pi_12:** 0.4753 (var=0.0129)

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
**Predicted under pi_11:** 0.1454 (var=0.0176)
**Predicted under pi_12:** 0.7104 (var=0.0169)

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
**Predicted under pi_11:** 0.5660 (var=0.0031)
**Predicted under pi_12:** 0.6900 (var=0.0164)

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
**Predicted under pi_11:** 0.1847 (var=0.0131)
**Predicted under pi_12:** 0.3000 (var=0.0102)

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
**Predicted under pi_11:** 0.4658 (var=0.0058)
**Predicted under pi_12:** 0.4653 (var=0.0088)

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
**Predicted under pi_11:** 0.8783 (var=0.0112)
**Predicted under pi_12:** 0.6325 (var=0.0189)

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
**Predicted under pi_11:** 0.1117 (var=0.0651)
**Predicted under pi_12:** 0.2717 (var=0.0608)

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
**Predicted under pi_11:** 0.1016 (var=0.0080)
**Predicted under pi_12:** 0.1700 (var=0.0154)

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
**Predicted under pi_11:** -0.0037 (var=0.0080)
**Predicted under pi_12:** 0.0350 (var=0.0110)

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
**Predicted under pi_11:** -0.0100 (var=0.0418)
**Predicted under pi_12:** 0.1500 (var=0.0681)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (Unique Features Weakest-Link) significantly outperforms Theory 2 (Strict Averaging) across the majority of experiments, particularly those designed to test the cancellation of shared features (e.g., Experiments 1, 2, 11, 13, 14, 19, 21). Theory 2 consistently fails in these environments because its strict averaging applies to all features, leading to false predictions of preference reversals when shared features are added. However, Theory 1 struggles in certain edge cases (like Experiment 16 and 9) where its specific weakest-link penalty (max - min) on unique features either over-penalizes or fails to capture the true evaluation dynamic of the unique features left over. Theory 2 is fundamentally flawed because it ignores shared feature cancellation, which the data strongly supports.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory that retains the core, empirically supported mechanism of 'Shared Feature Cancellation' from Theory 1, but replaces the 'weakest-link' penalty with a different evaluation of the unique features. For example, a 'Unique Features Additive Utility' theory, where decision-makers cancel shared features and then sum the subjective values (potentially with diminishing returns or a compensatory tallying mechanism) of the remaining unique features. This would avoid the catastrophic failures of strict averaging while offering a stronger, more robust alternative to the weakest-link penalty for evaluating unique feature sets."
}
```

## Usage

```json
{
  "prompt_token_count": 16953,
  "candidates_token_count": 331,
  "total_token_count": 19324
}
```
