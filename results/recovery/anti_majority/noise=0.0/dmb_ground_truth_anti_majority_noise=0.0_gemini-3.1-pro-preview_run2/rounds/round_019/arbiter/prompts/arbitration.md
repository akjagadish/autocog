# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_19" and "pi_21") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_19" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_21" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_19
Decision-makers evaluate options based on the relative average validity of their active cues, rather than the simple sum. A base prior is included to prevent zero-division and establish a baseline expectation. This average-based integration naturally predicts the 'less-is-more' effect, as adding weak cues dilutes the overall average evidence of an option, while shared cues have a context-dependent impact by simultaneously altering the numerator and denominator.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Calculate average validity with a base prior to prevent zero-division
    v_a = (np.sum(w * a) + prior_sum) / (n_a + prior_count)
    v_b = (np.sum(w * b) + prior_sum) / (n_b + prior_count)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_21
Shared-Feature Cancellation and Evidence Averaging: Decision-makers evaluate options by first identifying and canceling out features that are shared (active) in both options. After removing these shared elements to simplify the task, they evaluate each option by averaging the validities of its remaining unique active cues. A base prior is included in the average to establish a baseline expectation, but its lower bounds are extended to exactly zero to allow subjects to completely turn off the prior if they rely purely on the raw average of unique cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    # Shared-Feature Cancellation
    shared = a * b
    a_unique = a - shared
    b_unique = b - shared
    
    n_a = np.sum(a_unique)
    n_b = np.sum(b_unique)
    
    # Calculate average validity of unique features with a base prior (safeguarded against div-by-zero)
    v_a = (np.sum(w * a_unique) + prior_sum) / max(n_a + prior_count, 1e-6)
    v_b = (np.sum(w * b_unique) + prior_sum) / max(n_b + prior_count, 1e-6)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_19)

### DESIGN
**Validities (n_features=5):** [0.95, 0.7, 0.65, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the 'Evidence Averaging Heuristic Theory' (Advocated) from the 'Shared-Feature Cancellation and Evidence Averaging' theory (Competing) by exploiting a paradoxical preference reversal driven by a strong shared cue. The Competing theory assumes that decision-makers first cancel out any features shared by both options before averaging the remaining unique features. Therefore, adding a shared strong cue to both options leaves their relative evaluations completely unchanged. In contrast, the Advocated theory averages ALL active features. Because the two options have a different number of unique features, adding a shared strong cue pulls the average of the option with fewer unique features up much faster than the option with more unique features. In Trials 2 and 4 (no shared cues), both theories agree on preferring Option A because its unique features have a higher average validity. However, in Trials 1 and 3, when a highly valid shared cue (0.95) is added to both options, the Competing theory continues to strictly prefer Option A (since the shared cue cancels out), whereas the Advocated theory predicts a strict preference reversal to Option B, because Option A's average is weighed down by its larger number of unique weaker cues.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
Under the Competing theory (Shared-Feature Cancellation), Trial 1 and Trial 2 evaluate identically because their unique features are exactly the same; the same applies to Trial 3 and Trial 4. Therefore, any observed differences in choice proportions within these pairs are purely due to binomial sampling noise. By computing the standard Pearson Chi-square statistic for independence on these two pairs, the expected value under the Competing theory is strictly defined by the degrees of freedom (expected value of 2, with tight variance). Under the Advocated theory (Evidence Averaging), the presence of the shared feature in Trials 1 and 3 systematically alters the true choice probabilities compared to Trials 2 and 4. This introduces a non-centrality parameter to the Chi-square distribution. Because the Chi-square statistic standardizes the squared difference by the binomial variance, it optimally accumulates the signal across subjects regardless of whether their specific parameters cause the choice probability to increase or decrease, rigorously distinguishing the theories while perfectly controlling for baseline noise.

Source:
def metric(data: pd.DataFrame) -> float:
    def subj_metric(subj_df):
        a_ratings = subj_df['option_a_ratings'].tolist()
        responses = subj_df['response'].tolist()
        
        r1, r2, r3, r4 = [], [], [], []
        for a, r in zip(a_ratings, responses):
            if a[0] == 1 and a[1] == 1:
                r1.append(r)
            elif a[0] == 0 and a[1] == 1:
                r2.append(r)
            elif a[0] == 1 and a[3] == 1:
                r3.append(r)
            elif a[0] == 0 and a[3] == 1:
                r4.append(r)
                
        def get_chi2(ra, rb):
            if len(ra) == 0 or len(rb) == 0:
                return 0.0
            pa = sum(ra) / len(ra)
            pb = sum(rb) / len(rb)
            pooled_p = (sum(ra) + sum(rb)) / (len(ra) + len(rb))
            if pooled_p <= 0.0 or pooled_p >= 1.0:
                return 0.0
            
            multiplier = (len(ra) * len(rb)) / (len(ra) + len(rb))
            return multiplier * ((pa - pb)**2) / (pooled_p * (1.0 - pooled_p))
            
        return get_chi2(r1, r2) + get_chi2(r3, r4)

    return float(data.groupby('subject_id').apply(subj_metric).mean())

### RESULTS
- Predicted under pi_19 (simulated): 2.0421 (var=4.0524)
- Predicted under pi_21 (simulated): 1.8225 (var=3.6596)
- Observed on real data: 2.2126 (var=3.8792)

## EXPERIMENT 2 (proposed by pi_21)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates 'Shared-Feature Cancellation and Evidence Averaging' (Advocated) from 'Evidence Averaging Heuristic Theory' (Competing) by pitting strong shared cues against weak unique cues. Under the Advocated theory, shared features are completely canceled out before evaluation. Thus, an option with strong shared features plus weak unique features (Option A) is evaluated solely on its weak unique features, while an option with only the shared features (Option B) is reduced to zero unique features. Assuming a low base prior, the Advocated theory strictly prefers Option A because it possesses some unique positive evidence compared to none. In contrast, the Competing theory averages all active features without cancellation. The weak unique features in Option A severely dilute the high average of the strong shared features, causing the Competing theory to strictly prefer Option B. This produces a robust, structural preference reversal between the two theories.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
Under the Advocated theory (Shared-Feature Cancellation), Trials 2, 3, and 4 leave Option A with the exact same unique features (features 4 and 5) and Option B with no unique features. Thus, the true choice probabilities are perfectly identical across these trials, and the expected difference `P(A|T2,T3) - P(A|T4)` is exactly 0. Because the tanh function is odd-symmetric, the expected value of the transformed metric remains exactly 0 under Advocated theory. Under the Competing theory (Evidence Averaging), the difference is consistently positive for ~99% of parameter sets (as A's weak features dilute its average less harshly against B's strong features in T2/T3 than against B's 'perfect' empty prior in T4). However, for ~1% of extreme prior parameters, the difference is massively negative, which pulled the arithmetic mean down in previous failed metrics. By applying a steep non-linear squash `tanh(diff * 10.0)`, we cap the rare massive outliers while heavily amplifying the consistent small positive differences, yielding a strongly positive, low-variance signal for the Competing theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def subj_metric(df):
        # Identify trials by the number of active features in Option B
        # T2 and T3: Option B has 2 or 1 features
        # T4: Option B has 0 features
        b_sums = df['option_b_ratings'].apply(sum)
        mask_t23 = b_sums.isin([1, 2])
        mask_t4 = b_sums == 0
        
        if mask_t23.sum() == 0 or mask_t4.sum() == 0:
            return 0.0
            
        # Calculate probability of choosing Option A
        p_a_t23 = (df[mask_t23]['response'] == 0).mean()
        p_a_t4 = (df[mask_t4]['response'] == 0).mean()
        
        # Difference in probability
        diff = p_a_t23 - p_a_t4
        
        # Non-linear squash (tanh) to suppress massive negative outliers
        # and amplify the consistent small positive differences predicted by Competing theory.
        return float(np.tanh(diff * 10.0))

    # Return the mean across subjects so that metric(pooled_data) equals 
    # the mean of metric(subj_data), perfectly matching the variance computation.
    if data['subject_id'].nunique() > 1:
        return float(data.groupby('subject_id').apply(subj_metric).mean())
    else:
        return float(subj_metric(data))

### RESULTS
- Predicted under pi_19 (simulated): 0.3015 (var=0.3867)
- Predicted under pi_21 (simulated): -0.1995 (var=0.3891)
- Observed on real data: 0.0260 (var=0.2056)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_count = 0
    disagree_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_winner = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1
            
        # Only consider trials where TTB and Tallying make opposite predictions
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner:
            disagree_count += 1
            if row['response'] == ttb_winner:
                ttb_match_count += 1
                
    if disagree_count == 0:
        return 0.5
    return ttb_match_count / disagree_count

```

**Observed (real) value:** 0.7581 (var=0.0332)
**Predicted under pi_19:** 0.7072 (var=0.0182)
**Predicted under pi_21:** 0.7334 (var=0.0154)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
            
        matches.append(row['response'] == tally_pref)
        
    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2506 (var=0.0294)
**Predicted under pi_19:** 0.3117 (var=0.0239)
**Predicted under pi_21:** 0.2925 (var=0.0227)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    
    # TTB prediction: heavily weights the first discriminating cue
    # Using powers of 10 ensures strict lexicographical ordering (TTB logic)
    weights = np.array([1000, 100, 10, 1])
    ttb_score = np.dot(diff, weights)
    ttb_pred = np.where(ttb_score > 0, 0, 1)
    
    # WADD prediction (with gamma=1, i.e., linear integration)
    wadd_score = np.dot(diff, val)
    wadd_pred = np.where(wadd_score > 0, 0, 1)
    
    # Identify trials where TTB and baseline WADD disagree
    divergent = ttb_pred != wadd_pred
    
    if not np.any(divergent):
        return 0.5
        
    responses = data['response'].values
    # Calculate proportion of choices matching TTB on these critical trials
    ttb_match = (responses[divergent] == ttb_pred[divergent]).mean()
    
    return float(ttb_match)
```

**Observed (real) value:** 0.7236 (var=0.0302)
**Predicted under pi_19:** 0.7138 (var=0.0219)
**Predicted under pi_21:** 0.6938 (var=0.0273)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.3975 (var=0.0240)
**Predicted under pi_19:** 0.3027 (var=0.0191)
**Predicted under pi_21:** 0.3181 (var=0.0204)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trials 1, 2, and 3 are conflict trials where TTB prefers A and WADD prefers B.
    target_trials = ['10000', '01000', '10100']
    df_conflict = data[data['a_str'].isin(target_trials)].copy()
    df_conflict['chose_A'] = (df_conflict['response'] == 0).astype(float)
    
    subj_vars = []
    for subj, subj_df in df_conflict.groupby('subject_id'):
        means = subj_df.groupby('a_str')['chose_A'].mean()
        if len(means) == 3:
            subj_vars.append(means.var(ddof=1))
            
    if not subj_vars:
        return 0.0
        
    return float(np.mean(subj_vars))
```

**Observed (real) value:** 0.0574 (var=0.0093)
**Predicted under pi_19:** 0.0115 (var=0.0001)
**Predicted under pi_21:** 0.0100 (var=0.0001)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    data['chose_A'] = 1 - data['response']
    
    t2_mask = (data['A_str'] == '10001') & (data['B_str'] == '01100')
    t5_mask = (data['A_str'] == '01001') & (data['B_str'] == '00110')
    
    p_a_t2 = data[t2_mask]['chose_A'].mean() if t2_mask.sum() > 0 else 0.5
    p_a_t5 = data[t5_mask]['chose_A'].mean() if t5_mask.sum() > 0 else 0.5
    
    return float(p_a_t2 - p_a_t5)
```

**Observed (real) value:** 0.0175 (var=0.0095)
**Predicted under pi_19:** 0.0288 (var=0.0372)
**Predicted under pi_21:** 0.0238 (var=0.0383)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_seq = data['option_a_ratings'].apply(tuple)
    b_seq = data['option_b_ratings'].apply(tuple)
    
    t1_a = (1, 0, 0, 0, 0)
    t1_b = (0, 1, 1, 1, 1)
    
    t4_a = (1, 1, 1, 1, 0)
    t4_b = (1, 1, 1, 0, 1)
    
    t1_mask1 = (a_seq == t1_a) & (b_seq == t1_b)
    t1_mask2 = (a_seq == t1_b) & (b_seq == t1_a)
    
    t4_mask1 = (a_seq == t4_a) & (b_seq == t4_b)
    t4_mask2 = (a_seq == t4_b) & (b_seq == t4_a)
    
    p_t1 = 0.0
    n_t1 = t1_mask1.sum() + t1_mask2.sum()
    if n_t1 > 0:
        chose_a_t1 = (t1_mask1 & (data['response'] == 0)) | (t1_mask2 & (data['response'] == 1))
        p_t1 = chose_a_t1.sum() / n_t1
        
    p_t4 = 0.0
    n_t4 = t4_mask1.sum() + t4_mask2.sum()
    if n_t4 > 0:
        chose_a_t4 = (t4_mask1 & (data['response'] == 0)) | (t4_mask2 & (data['response'] == 1))
        p_t4 = chose_a_t4.sum() / n_t4
        
    return float(p_t1 - p_t4)
```

**Observed (real) value:** 0.6875 (var=0.0691)
**Predicted under pi_19:** 0.2738 (var=0.0544)
**Predicted under pi_21:** 0.2175 (var=0.0330)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data['trial_str'] = data.apply(lambda row: ''.join(map(str, row['option_a_ratings'])) + '_' + ''.join(map(str, row['option_b_ratings'])), axis=1)
    
    t1 = '110000_001111'
    t2 = '001111_110000'
    t3 = '101000_010111'
    t4 = '010111_101000'
    
    data['chose_a'] = (data['response'] == 0).astype(float)
    subj_means = data.groupby(['subject_id', 'trial_str'])['chose_a'].mean().unstack()
    
    # Preference for the option with the top cue in Trial 1/2 vs Trial 3/4
    if t1 in subj_means.columns and t2 in subj_means.columns:
        pref_1 = (subj_means[t1] + (1.0 - subj_means[t2])) / 2.0
    else:
        return 0.0
        
    if t3 in subj_means.columns and t4 in subj_means.columns:
        pref_3 = (subj_means[t3] + (1.0 - subj_means[t4])) / 2.0
    else:
        return 0.0
        
    diff = pref_1 - pref_3
    
    if isinstance(diff, pd.Series):
        return float(diff.mean())
    return float(diff)
```

**Observed (real) value:** 0.0475 (var=0.1792)
**Predicted under pi_19:** 0.0063 (var=0.0080)
**Predicted under pi_21:** 0.0292 (var=0.0109)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Trial 1: Top cue (0.90) vs single opposing cue (0.85)
    pA_t1 = get_pA("10000", "01000")
    
    # Trial 4: Top cue (0.90) vs coalition of 4 opposing cues (0.85 + 0.80 + 0.75 + 0.70)
    pA_t4 = get_pA("10000", "01111")
    
    return float(pA_t1 - pA_t4)

```

**Observed (real) value:** -0.7263 (var=0.0321)
**Predicted under pi_19:** -0.1726 (var=0.0619)
**Predicted under pi_21:** -0.1032 (var=0.0650)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 and Trial 3 based on option A's ratings
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    t3_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0, 0))
    
    if t1_mask.sum() == 0 or t3_mask.sum() == 0:
        return 0.0
        
    # response is 0 for A, 1 for B. So mean() is the proportion of choosing B.
    # We want the difference in proportion of choosing A: P(A | Trial 3) - P(A | Trial 1)
    # Which is (1 - m3) - (1 - m1) = m1 - m3
    m1 = data[t1_mask]['response'].mean()
    m3 = data[t3_mask]['response'].mean()
    
    return float(m1 - m3)
```

**Observed (real) value:** -0.0253 (var=0.2181)
**Predicted under pi_19:** -0.0105 (var=0.0214)
**Predicted under pi_21:** -0.0084 (var=0.0209)

### Experiment 13
**Design**
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_key'] = data['option_a_ratings'].apply(tuple)
    
    # response == 0 means Option A was chosen, so 1 - mean(response) is P(A)
    p_A = 1.0 - data.groupby('a_key')['response'].mean()
    
    # Trials without the shared top cue
    t1 = p_A.get((0, 1, 1, 0, 0), 0.5)
    t3 = p_A.get((0, 1, 0, 0, 0), 0.5)
    t5 = p_A.get((0, 1, 1, 1, 0), 0.5)
    
    # Trials with the shared top cue (Cue 0 = 1 for both options)
    t2 = p_A.get((1, 1, 1, 0, 0), 0.5)
    t4 = p_A.get((1, 1, 0, 0, 0), 0.5)
    t6 = p_A.get((1, 1, 1, 1, 0), 0.5)
    
    # Calculate the regression to chance (0.5) caused by the shared top cue
    diff1 = abs(t1 - 0.5) - abs(t2 - 0.5)
    diff2 = abs(t3 - 0.5) - abs(t4 - 0.5)
    diff3 = abs(t5 - 0.5) - abs(t6 - 0.5)
    
    return float(diff1 + diff2 + diff3)
```

**Observed (real) value:** 0.0725 (var=0.0566)
**Predicted under pi_19:** 0.0775 (var=0.0356)
**Predicted under pi_21:** 0.0600 (var=0.0766)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_t6 = (data['A_str'] == '00100') & (data['B_str'] == '00011')
    mask_t1 = (data['A_str'] == '10000') & (data['B_str'] == '01111')
    
    p_a_t6 = (data[mask_t6]['response'] == 0).mean() if mask_t6.sum() > 0 else 0.5
    p_a_t1 = (data[mask_t1]['response'] == 0).mean() if mask_t1.sum() > 0 else 0.5
    
    return float(p_a_t6 - p_a_t1)
```

**Observed (real) value:** -0.1700 (var=0.2061)
**Predicted under pi_19:** -0.0700 (var=0.0378)
**Predicted under pi_21:** -0.0187 (var=0.0183)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    is_t2 = (a_sums == 4) & (b_sums == 1)
    is_t3 = (a_sums == 1) & (b_sums == 4)
    
    # Probability of choosing A
    p_a_t2 = 1.0 - data.loc[is_t2, 'response'].mean()
    p_a_t3 = 1.0 - data.loc[is_t3, 'response'].mean()
    
    return float(p_a_t2 - p_a_t3)
```

**Observed (real) value:** -0.6683 (var=0.0841)
**Predicted under pi_19:** -0.5558 (var=0.1177)
**Predicted under pi_21:** -0.5058 (var=0.0763)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_keys == (1, 0, 0, 0, 0)) & (b_keys == (0, 1, 0, 0, 0))
    t2_mask = (a_keys == (1, 0, 1, 1, 1)) & (b_keys == (0, 1, 0, 0, 0))
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t1)
```

**Observed (real) value:** 0.0100 (var=0.0081)
**Predicted under pi_19:** -0.2875 (var=0.0572)
**Predicted under pi_21:** -0.2892 (var=0.0347)

### Experiment 17
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Extract A's features as strings for easy matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A = [1, 0, 0, 0, 0]
    # Trial 2: A = [1, 0, 1, 1, 1]
    # Response is 0 if A, 1 if B. We want P(Choose A), which is 1.0 - response.mean()
    
    t1_data = data[a_str == '10000']
    t2_data = data[a_str == '10111']
    
    if len(t1_data) == 0 or len(t2_data) == 0:
        return 0.0
        
    prob_a_t1 = 1.0 - t1_data['response'].mean()
    prob_a_t2 = 1.0 - t2_data['response'].mean()
    
    return float(prob_a_t1 - prob_a_t2)
```

**Observed (real) value:** -0.0400 (var=0.0118)
**Predicted under pi_19:** 0.2925 (var=0.0949)
**Predicted under pi_21:** 0.2937 (var=0.0843)

### Experiment 18
**Design**
  A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    trial_3_mask = (data['A_str'] == '10011') & (data['B_str'] == '01000')
    trial_1_mask = (data['A_str'] == '00011') & (data['B_str'] == '00010')
    
    p_a_3 = 1.0 - data.loc[trial_3_mask, 'response'].mean() if trial_3_mask.sum() > 0 else 0.5
    p_a_1 = 1.0 - data.loc[trial_1_mask, 'response'].mean() if trial_1_mask.sum() > 0 else 0.5
    
    return float(p_a_3 - p_a_1)
```

**Observed (real) value:** 0.0167 (var=0.0107)
**Predicted under pi_19:** 0.0233 (var=0.0240)
**Predicted under pi_21:** 0.0192 (var=0.0181)

### Experiment 19
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = (data['a_str'] == '10100') & (data['b_str'] == '01010')
    t2_mask = (data['a_str'] == '10001') & (data['b_str'] == '01010')
    
    def subj_metric(df):
        t1_resp = df.loc[t1_mask, 'response']
        t2_resp = df.loc[t2_mask, 'response']
        if len(t1_resp) == 0 or len(t2_resp) == 0:
            return 0.0
            
        p_a_t1 = (t1_resp == 0).mean()
        p_a_t2 = (t2_resp == 0).mean()
        
        # Weight by the subject's signal-to-noise ratio measured via T2.
        weight = max(0.0, p_a_t2 - 0.5)
        return float(weight * (p_a_t1 - p_a_t2))
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_19:** -0.0035 (var=0.0002)
**Predicted under pi_21:** -0.0063 (var=0.0005)

### Experiment 20
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    
    # Trial 2: A=[0, 1, 0, 0, 0] (sum=0.9), B=[0, 0, 1, 1, 0] (sum=1.0)
    t2 = (a_str == '01000') & (b_str == '00110')
    # Trial 6: A=[0, 1, 0, 0, 0] (sum=0.9), B=[0, 0, 1, 1, 1] (sum=1.5)
    t6 = (a_str == '01000') & (b_str == '00111')
    
    target_trials = t2 | t6
    if not target_trials.any():
        return 0.5
        
    return float((data.loc[target_trials, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5825 (var=0.0907)
**Predicted under pi_19:** 0.7644 (var=0.0196)
**Predicted under pi_21:** 0.7219 (var=0.0275)

### Experiment 21
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1 = data[(data['A_str'] == '10000') & (data['B_str'] == '01000')]
    t2 = data[(data['A_str'] == '10111') & (data['B_str'] == '01000')]
    
    p_a_t1 = 1.0 - t1['response'].mean() if len(t1) > 0 else 0.5
    p_a_t2 = 1.0 - t2['response'].mean() if len(t2) > 0 else 0.5
    
    return float(p_a_t2 - p_a_t1)
```

**Observed (real) value:** -0.0250 (var=0.0106)
**Predicted under pi_19:** -0.3767 (var=0.0705)
**Predicted under pi_21:** -0.2600 (var=0.0764)

### Experiment 22
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has many cues (including the top ones) and Option B has few cues
    # Trial 2: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
    # Trial 4: A=[1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
    mask = data['option_a_ratings'].apply(lambda x: sum(x) >= 4) & data['option_b_ratings'].apply(lambda x: sum(x) <= 2)
    # Return the proportion of times Option B was chosen
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.8386 (var=0.0093)
**Predicted under pi_19:** 0.7372 (var=0.0208)
**Predicted under pi_21:** 0.7211 (var=0.0259)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the number of active cues in Option A
    n_cues = data['option_a_ratings'].apply(sum)
    
    # Calculate the probability of choosing A for each number of cues
    p_a = 1.0 - data.groupby(n_cues)['response'].mean()
    
    # Ensure we have data for all 4 cue levels
    if not all(k in p_a for k in [1, 2, 3, 4]):
        return 0.0
        
    # Calculate consecutive differences in P(A)
    diffs = [p_a[k+1] - p_a[k] for k in [1, 2, 3]]
    
    # The metric is the difference between the maximum increase and the maximum decrease
    # Competing theory predicts a steady increase followed by a sharp drop (large max - min)
    # Advocated theory predicts a smooth curve with less extreme fluctuations in differences
    return float(np.max(diffs) - np.min(diffs))
```

**Observed (real) value:** 0.0650 (var=0.0165)
**Predicted under pi_19:** 0.2125 (var=0.0425)
**Predicted under pi_21:** 0.1533 (var=0.0368)

### Experiment 24
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    mask1 = (a_str == '110000') & (b_str == '001100')
    mask2 = (a_str == '001100') & (b_str == '110000')
    
    valid = mask1 | mask2
    if not valid.any():
        return 0.5
        
    subset = data[valid]
    m1 = mask1[valid]
    
    chose_A = np.where(m1, subset['response'] == 0, subset['response'] == 1)
    
    return float(np.mean(chose_A))
```

**Observed (real) value:** 0.1768 (var=0.0184)
**Predicted under pi_19:** 0.6516 (var=0.0251)
**Predicted under pi_21:** 0.7147 (var=0.0172)

### Experiment 25
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    t2_mask = (sum_a == 4) & (sum_b == 1)
    t4_mask = (sum_a == 2) & (sum_b == 5)
    
    p_a_t2 = 1.0 - data.loc[t2_mask, 'response'].mean() if t2_mask.any() else 0.5
    p_a_t4 = 1.0 - data.loc[t4_mask, 'response'].mean() if t4_mask.any() else 0.5
    
    return float(p_a_t4 - p_a_t2)
```

**Observed (real) value:** 0.7333 (var=0.0444)
**Predicted under pi_19:** 0.5825 (var=0.1012)
**Predicted under pi_21:** 0.4108 (var=0.1133)

### Experiment 26
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1 = data[data['a_str'] == '01000']['response']
    t2 = data[data['a_str'] == '11000']['response']
    t3 = data[data['a_str'] == '00100']['response']
    t4 = data[data['a_str'] == '01100']['response']
    
    p_a_1 = 1.0 - t1.mean() if len(t1) > 0 else 0.5
    p_a_2 = 1.0 - t2.mean() if len(t2) > 0 else 0.5
    p_a_3 = 1.0 - t3.mean() if len(t3) > 0 else 0.5
    p_a_4 = 1.0 - t4.mean() if len(t4) > 0 else 0.5
    
    return (p_a_2 - p_a_1) + (p_a_4 - p_a_3)
```

**Observed (real) value:** -0.0333 (var=0.0178)
**Predicted under pi_19:** 0.0008 (var=0.0280)
**Predicted under pi_21:** 0.0433 (var=0.0335)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    t1 = (1, 0, 0, 0, 0, 0)
    t2 = (1, 0, 1, 0, 0, 0)
    t3 = (1, 0, 1, 1, 0, 0)
    t4 = (1, 0, 1, 1, 1, 1)
    
    subj_diffs = []
    for subj, df in data.groupby('subject_id'):
        r1 = df.loc[df['A_tuple'] == t1, 'response'].mean()
        r2 = df.loc[df['A_tuple'] == t2, 'response'].mean()
        r3 = df.loc[df['A_tuple'] == t3, 'response'].mean()
        r4 = df.loc[df['A_tuple'] == t4, 'response'].mean()
        
        early_rate = (r1 + r2) / 2.0
        late_rate = (r3 + r4) / 2.0
        subj_diffs.append(late_rate - early_rate)
        
    return float(np.mean(subj_diffs))
```

**Observed (real) value:** -0.0025 (var=0.0058)
**Predicted under pi_19:** 0.0433 (var=0.0116)
**Predicted under pi_21:** 0.0175 (var=0.0072)

### Experiment 28
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.95, 0.95, 0.9, 0.5, 0.8, 0.6, 0.7, 0.7, 0.7])
    # Use a high power to strongly emphasize the difference in extreme validities
    # which the Competing theory's convex weighting function strictly prefers.
    weights = validities ** 6
    
    # Calculate weighted sum for A and B
    a_w = data['option_a_ratings'].apply(lambda x: np.sum(np.array(x) * weights))
    b_w = data['option_b_ratings'].apply(lambda x: np.sum(np.array(x) * weights))
    
    # chosen option's weighted sum minus unchosen option's weighted sum
    chosen_w = np.where(data['response'] == 0, a_w, b_w)
    unchosen_w = np.where(data['response'] == 0, b_w, a_w)
    
    diff = chosen_w - unchosen_w
    
    # Calculate the mean difference for each subject
    subj_means = data.assign(diff=diff).groupby('subject_id')['diff'].mean()
    
    # Return the average across subjects
    return float(subj_means.mean())
```

**Observed (real) value:** -0.1523 (var=0.0014)
**Predicted under pi_19:** 0.0101 (var=0.0009)
**Predicted under pi_21:** 0.0172 (var=0.0011)

### Experiment 29
**Design**
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    return float(np.mean(data['response'] == 0))

```

**Observed (real) value:** 0.1804 (var=0.0164)
**Predicted under pi_19:** 0.5750 (var=0.0070)
**Predicted under pi_21:** 0.5479 (var=0.0063)

### Experiment 30
**Design**
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1 specifically compares A=[1, 0, 0, 1] (high variance) vs B=[0, 1, 1, 0] (low variance).
    # The Advocated theory (concave, diminishing returns) strictly prefers the low-variance option B.
    # The Competing theory (convex, amplified penalty) strictly prefers the high-variance option A.
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1))
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8867 (var=0.0084)
**Predicted under pi_19:** 0.4425 (var=0.0128)
**Predicted under pi_21:** 0.4425 (var=0.0093)

### Experiment 31
**Design**
  A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_superset(a, b):
        return sum(b) > sum(a) and all(bv >= av for av, bv in zip(a, b))
    
    mask = [is_superset(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    
    if sum(mask) == 0:
        return 0.5
        
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.1700 (var=0.0135)
**Predicted under pi_19:** 0.3237 (var=0.0350)
**Predicted under pi_21:** 0.2762 (var=0.0259)

### Experiment 32
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    t1_mask = (a_sums == 1) & (b_sums == 1)
    t2_mask = (a_sums == 4) & (b_sums == 1)
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1):
        p_a_t1 = 0.5
    if pd.isna(p_a_t2):
        p_a_t2 = 0.5
        
    return float(p_a_t1 - p_a_t2)
```

**Observed (real) value:** 0.0200 (var=0.0095)
**Predicted under pi_19:** 0.2883 (var=0.0700)
**Predicted under pi_21:** 0.2833 (var=0.0582)

### Experiment 33
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # response == 0 means Option A was chosen
    return (data['response'] == 0).mean()

```

**Observed (real) value:** 0.1671 (var=0.0088)
**Predicted under pi_19:** 0.5690 (var=0.0058)
**Predicted under pi_21:** 0.5448 (var=0.0060)

### Experiment 34
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify trials 1 and 2: option A has exactly 1 active cue
    is_t12 = data['option_a_ratings'].apply(lambda x: sum(x) == 1)
    t12_data = data[is_t12]
    
    if len(t12_data) == 0:
        return 0.0
        
    # Compute each subject's proportion of choosing B
    subject_means = t12_data.groupby('subject_id')['response'].mean()
    
    # Return the mean absolute deviation from 0.5
    return float(np.mean(np.abs(subject_means - 0.5)))

```

**Observed (real) value:** 0.3183 (var=0.0173)
**Predicted under pi_19:** 0.1750 (var=0.0209)
**Predicted under pi_21:** 0.1725 (var=0.0178)

### Experiment 35
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    grouped = data.groupby(['A_str', 'B_str'])['chose_A'].mean().to_dict()
    
    t1, t2 = ('10000', '01100'), ('10001', '01101')
    t3, t4 = ('01000', '00110'), ('11000', '10110')
    t5, t6 = ('01100', '00011'), ('11100', '10011')
    
    diff = 0.0
    for pair in [(t1, t2), (t3, t4), (t5, t6)]:
        if pair[0] in grouped and pair[1] in grouped:
            diff += abs(grouped[pair[0]] - grouped[pair[1]])
            
    return float(diff)
```

**Observed (real) value:** 0.0700 (var=0.0265)
**Predicted under pi_19:** 0.0575 (var=0.0277)
**Predicted under pi_21:** 0.1412 (var=0.0275)

### Experiment 36
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t2_mask = (a_str == '10011') & (b_str == '01000')
    t3_mask = (a_str == '10000') & (b_str == '01011')
    
    p_b_t2 = data.loc[t2_mask, 'response'].mean()
    p_b_t3 = data.loc[t3_mask, 'response'].mean()
    
    return float(p_b_t2 - p_b_t3)
```

**Observed (real) value:** 0.6050 (var=0.0663)
**Predicted under pi_19:** 0.3983 (var=0.1009)
**Predicted under pi_21:** 0.3867 (var=0.1041)

### Experiment 37
**Design**
  A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    mask = data['option_a_ratings'].apply(lambda x: sum(x) == 0) & data['option_b_ratings'].apply(lambda x: sum(x) > 1)
    if mask.sum() == 0:
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.8392 (var=0.0092)
**Predicted under pi_19:** 0.7183 (var=0.0361)
**Predicted under pi_21:** 0.7625 (var=0.0313)

### Experiment 38
**Design**
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has exactly zero active features (Trials 1, 2, and 3)
    a_cues = data['option_a_ratings'].apply(sum)
    zero_mask = (a_cues == 0)
    
    if zero_mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times Option B is chosen when Option A has 0 features
    return float(data[zero_mask]['response'].mean())
```

**Observed (real) value:** 0.1594 (var=0.0193)
**Predicted under pi_19:** 0.2897 (var=0.0314)
**Predicted under pi_21:** 0.3631 (var=0.0339)

### Experiment 39
**Design**
  A=[1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1: A=[1, 0, 0, 0, 0], B=[1, 1, 0, 0, 0]
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    t1_data = data[t1_mask]
    if len(t1_data) == 0:
        return 0.0
    # The Advocated theory predicts a 'less-is-more' effect where A is chosen over B
    # because the average validity of A's single cue is higher than the average of B's two cues.
    # The Competing theory predicts B is chosen over A because the second cue strictly adds value.
    return float((t1_data['response'] == 0).mean())
```

**Observed (real) value:** 0.8933 (var=0.0093)
**Predicted under pi_19:** 0.6583 (var=0.0364)
**Predicted under pi_21:** 0.5875 (var=0.0398)

### Experiment 40
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Create a string representation of option A to uniquely identify the trial types
    data['trial_key'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Calculate the proportion of B choices for each trial type
    means = data.groupby('trial_key')['response'].mean()
    
    # Trial 1: A=[1, 0, 0, 0, 0] vs B=[0, 1, 1, 0, 0] (no shared features)
    p1 = means.get('10000', 0.5)
    
    # Trial 3: A=[1, 0, 0, 1, 1] vs B=[0, 1, 1, 1, 1] (two shared features)
    p3 = means.get('10011', 0.5)
    
    # The metric is the absolute difference in choice probabilities between these two trials
    return float(abs(p1 - p3))
```

**Observed (real) value:** 0.0050 (var=0.0025)
**Predicted under pi_19:** 0.0658 (var=0.0070)
**Predicted under pi_21:** 0.0083 (var=0.0063)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, we see a strong structural pattern of choice invariances that strongly supports Shared-Feature Cancellation (Theory 2). For instance, in Experiment 40, the observed difference between a trial with no shared features and one with two shared features is 0.0050 (effectively zero), which Theory 2 predicts well, whereas Theory 1 expects a systematic difference (0.0658). In Experiment 2, the observed metric is 0.0260, again clustering near zero, aligning with the cancellation hypothesis. However, the strong 'less-is-more' effect in Experiment 39 (where A=[1,0,0,0,0] is preferred over B=[1,1,0,0,0] 89% of the time) shows that adding a weak cue is highly detrimental. While Theory 2 can account for this if the 'empty' option (post-cancellation) defaults to a prior that is higher than the weak cue, simple Evidence Averaging (Theory 1) captures the dilution effect more naturally but fails on the shared-feature invariances. Theory 1 is fundamentally limited because it cannot predict the exact invariances observed when shared features are added.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory that incorporates Shared-Feature Cancellation (as it is strictly necessary to explain the invariances in Exp 2 and Exp 40) but introduces a different evaluation mechanism for the remaining unique features. Instead of a simple average with a static prior, propose 'Shared-Feature Cancellation with Contrastive Dilution': subjects first cancel shared features, and then evaluate options by averaging the unique features, but the 'baseline' or prior they regress to is dynamically set by the maximum validity of the active cues in the trial. This would naturally explain why a weak unique cue is heavily penalized (it dilutes the high expectation set by the canceled strong cues), elegantly capturing both the strict invariances and the massive less-is-more effects."
}
```

## Usage

```json
{
  "prompt_token_count": 24473,
  "candidates_token_count": 447,
  "total_token_count": 26772
}
```
