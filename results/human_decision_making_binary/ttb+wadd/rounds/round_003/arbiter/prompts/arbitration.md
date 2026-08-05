# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Rank-Based Weighting Model with Exponential Decay (Average Ties): Subjects evaluate options by computing a weighted sum of their features. They rank features based on validities and assign subjective weights using an exponential decay function. Ties in validities are assigned their average rank, providing a smoother spacing of decay weights. A decay parameter lambda smoothly interpolates between Tallying (lambda=1) and Take The Best (lambda=0).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    lambda_param = float(parameters["lambda_param"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Rank features by validity (highest validity = rank 1)
    # Using 'average' method to handle ties smoothly
    ranks = rankdata(-validities, method='average')
    
    # Exponential rank decay: w = lambda_param ^ (rank - 1)
    w = lambda_param ** (ranks - 1.0)
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.6, 0.58, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** This design quantitatively dissociates the Rank-Based Weighting Model (Advocated) from the Non-linear Subjective Weighting Model (Competing) by exploiting the difference between rank-based exponential decay and value-based non-linear scaling. The validities are chosen to have a large gap between rank 1 and rank 2, but very small gaps between ranks 2, 3, and 4. The Rank-Based model enforces a constant ratio of weights between adjacent ranks (w_i / w_{i+1} = 1/lambda), regardless of the actual validity values. Consequently, the Rank-Based model strictly constrains the relative trade-offs among lower-ranked cues. By contrasting combinations like 'Cue 2' vs 'Cue 3 + Cue 4', the Non-linear model (which is sensitive to the tightly clustered values of cues 2, 3, and 4) will strongly favor the sum of cues 3 and 4 across a wide range of its gamma parameter. Conversely, the Rank-Based model, depending on lambda, can easily predict Cue 2 to outweigh Cues 3 and 4 combined, due to its fixed exponential rank decay.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric evaluates the difference in preference for a single higher-validity cue when pitted against two lower-validity cues, comparing Trial 3 (Cue 1 vs Cues 2+3) and Trial 1 (Cue 2 vs Cues 3+4). The Non-linear model leverages the actual validity values: since Cue 1 (0.95) is vastly superior to Cues 2 and 3 (0.6, 0.58), it often strongly prefers Cue 1. However, since Cue 2 (0.6) is barely better than Cues 3 and 4 (0.58, 0.55), it rarely prefers Cue 2 over their sum. This results in a large positive difference. By contrast, the Rank-Based model enforces a constant exponential ratio between adjacent ranks, making the relative trade-off (Rank 1 vs Ranks 2+3) structurally identical to (Rank 2 vs Ranks 3+4), scaling only by the decay factor. Thus, the Rank-Based model will predict a much smaller or even negative difference in these choice probabilities.

Source:
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Trial 3: Cue 2+3 (A) vs Cue 1 (B)
    is_t3 = (a_keys == (0, 1, 1, 0, 0)) & (b_keys == (1, 0, 0, 0, 0))
    # Trial 1: Cue 2 (A) vs Cue 3+4 (B)
    is_t1 = (a_keys == (0, 1, 0, 0, 0)) & (b_keys == (0, 0, 1, 1, 0))
    
    # Probability of choosing the single higher-ranked cue in each trial
    p_cue1_t3 = (data.loc[is_t3, 'response'] == 1).mean()
    p_cue2_t1 = (data.loc[is_t1, 'response'] == 0).mean()
    
    if pd.isna(p_cue1_t3) or pd.isna(p_cue2_t1):
        return 0.0
        
    return float(p_cue1_t3 - p_cue2_t1)

### RESULTS
- Predicted under pi_5 (simulated): 0.0625 (var=0.0309)
- Predicted under pi_4 (simulated): 0.3100 (var=0.0608)
- Observed on real data: 0.1325 (var=0.1076)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.98, 0.96, 0.94, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** This design quantitatively dissociates the Non-linear Subjective Weighting (NLSW) model from the Rank-Based (RB) model by exploiting the difference between value-based scaling and rank-based exponential decay. The validities [0.98, 0.96, 0.94, 0.60, 0.55] feature a tiny numerical gap between the top three cues, but a massive gap between the third and fourth cues. Under the NLSW model, the weights for the top three cues will be very similar, while the weights for the bottom two will be heavily suppressed for moderate to large gamma. Consequently, NLSW predicts that Cues 2 + 3 will easily outweigh Cue 1 (Trial 1), but Cue 3 will easily outweigh Cues 4 + 5 (Trial 2). In stark contrast, the RB model assigns weights based solely on rank, meaning the relative weight ratio of Rank 1 vs (Rank 2 + Rank 3) is mathematically identical to Rank 3 vs (Rank 4 + Rank 5). Thus, the RB model strictly predicts that a subject's preference direction (Option A vs Option B) must be identical across these two trials, whereas the NLSW model naturally predicts a preference reversal.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design explicitly dissociates the NLSW and RB models using Trial 1 (Cue 1 vs Cues 2+3) and Trial 2 (Cue 3 vs Cues 4+5). Under the Rank-Based model with exponential decay, the relative weight of Rank 1 vs Ranks 2+3 is mathematically identical to Rank 3 vs Ranks 4+5. Thus, the RB model predicts that the probability of choosing Option B in Trial 1 exactly equals the probability of choosing Option B in Trial 2. By contrast, the NLSW model squashes the weights of the bottom two validities heavily compared to the top three, leading to a strong preference for Option B in Trial 1 but a strong preference for Option A in Trial 2. This metric calculates the within-subject difference in the probability of choosing Option B between Trial 1 and Trial 2. It will be near 0 for the RB model and significantly greater than 0 for the NLSW model.

Source:
def metric(data: pd.DataFrame) -> float:
    def subj_diff(df):
        t1_mask = df['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
        t2_mask = df['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0, 0))
        p_b_t1 = df[t1_mask]['response'].mean()
        p_b_t2 = df[t2_mask]['response'].mean()
        return p_b_t1 - p_b_t2
        
    diffs = data.groupby('subject_id').apply(subj_diff).dropna()
    return float(diffs.mean())

### RESULTS
- Predicted under pi_5 (simulated): -0.1633 (var=0.0508)
- Predicted under pi_4 (simulated): 0.5233 (var=0.0964)
- Observed on real data: 0.5243 (var=0.1374)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        sum_a = sum(row['option_a_ratings'])
        sum_b = sum(row['option_b_ratings'])
        
        # In the compensatory trials designed for this experiment,
        # the option favored by TTB always has fewer total positive cues
        # than the option favored by WADD.
        if sum_a != sum_b:
            ttb_choice = 0 if sum_a < sum_b else 1
            ttb_match.append(1 if row['response'] == ttb_choice else 0)
            
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.4175 (var=0.0635)
**Predicted under pi_5:** 0.4354 (var=0.0557)
**Predicted under pi_4:** 0.5029 (var=0.0626)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        response = row['response']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        if ttb_choice is not None and response == ttb_choice:
            ttb_matches += 1
        total += 1
        
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4317 (var=0.0770)
**Predicted under pi_5:** 0.5202 (var=0.0558)
**Predicted under pi_4:** 0.4823 (var=0.0320)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Define the critical trials where PHS and WADD strongly diverge.
    # In Trial 1, Option A has the single highest validity cue and more total cues,
    # so PHS (TTB + Tallying) strongly predicts Option A (response 0).
    # Option B has multiple high-validity cues that outweigh A's cues in a weighted sum,
    # so WADD strongly predicts Option B (response 1).
    # Trial 2 is the exact symmetric swap.
    t1_A = (1, 0, 0, 0, 1, 1, 1)
    t2_A = (0, 1, 1, 1, 0, 0, 0)
    
    is_t1 = data['option_a_ratings'].apply(tuple) == t1_A
    is_t2 = data['option_a_ratings'].apply(tuple) == t2_A
    
    match_phs = 0
    match_phs += (data.loc[is_t1, 'response'] == 0).sum()
    match_phs += (data.loc[is_t2, 'response'] == 1).sum()
    
    total = is_t1.sum() + is_t2.sum()
    if total == 0:
        return 0.5
    return float(match_phs / total)

```

**Observed (real) value:** 0.2317 (var=0.0651)
**Predicted under pi_5:** 0.6146 (var=0.0222)
**Predicted under pi_4:** 0.2029 (var=0.0250)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_1 = data['option_b_ratings'].apply(lambda x: x[1])
    critical_mask = (a_0 == 1) & (b_1 == 1)
    if critical_mask.sum() == 0:
        return 0.5
    return float(data.loc[critical_mask, 'response'].mean())
```

**Observed (real) value:** 0.8158 (var=0.0450)
**Predicted under pi_5:** 0.3379 (var=0.0256)
**Predicted under pi_4:** 0.7850 (var=0.0505)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def get_trial_type(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a[0] == 1:
            other = b
        else:
            other = a
            
        if other == (0, 1, 1, 1, 0):
            return 'strong'
        elif other == (0, 0, 0, 1, 1):
            return 'weak'
        return 'other'

    data['trial_type'] = data.apply(get_trial_type, axis=1)
    
    def is_ttb_choice(row):
        a_has_cue = row['option_a_ratings'][0] == 1
        return 1 if (row['response'] == 0 and a_has_cue) or (row['response'] == 1 and not a_has_cue) else 0

    data['ttb_choice'] = data.apply(is_ttb_choice, axis=1)
    
    subj_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        strong_rate = subj_df[subj_df['trial_type'] == 'strong']['ttb_choice'].mean()
        weak_rate = subj_df[subj_df['trial_type'] == 'weak']['ttb_choice'].mean()
        if pd.notna(strong_rate) and pd.notna(weak_rate):
            subj_diffs.append(weak_rate - strong_rate)
            
    return float(np.mean(subj_diffs)) if subj_diffs else 0.0
```

**Observed (real) value:** 0.4767 (var=0.1713)
**Predicted under pi_5:** 0.2217 (var=0.0652)
**Predicted under pi_4:** 0.2833 (var=0.0876)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    overrides = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        sum_a = sum(a)
        sum_b = sum(b)
        tally_winner = 0 if sum_a > sum_b else (1 if sum_b > sum_a else None)
        
        if ttb_winner is not None and tally_winner is not None and ttb_winner == tally_winner:
            chosen = row['response']
            if chosen != ttb_winner:
                overrides.append(1)
            else:
                overrides.append(0)
                
    if not overrides:
        return 0.0
    return float(np.mean(overrides))
```

**Observed (real) value:** 0.4729 (var=0.0463)
**Predicted under pi_5:** 0.2985 (var=0.0113)
**Predicted under pi_4:** 0.3604 (var=0.0143)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 8 experiments, Theory 2 (Non-linear Subjective Weighting, pi_4) consistently outperforms Theory 1 (Rank-Based Weighting, pi_5). In Experiment 2, 5, and 6, Theory 2's predictions almost perfectly match the observed data, while Theory 1 completely fails to capture the direction or magnitude of the effects (e.g., predicting -0.16 vs observed 0.52 in Exp 2, and 0.61 vs observed 0.23 in Exp 5). Theory 1's strict reliance on rank ignores the actual numerical gaps between validities, which the data shows are critical to subjects' decision-making (as subjects heavily discount cues when there is a massive drop-off in validity). However, Theory 2 still slightly underestimates the override rates in Experiments 7 and 8. Theory 1 is degenerate because its core assumption of equidistant rank decay is strongly falsified.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace the Rank-Based Weighting Model with a brand-new theory, such as a 'Threshold-Gated Compensatory Model' or a 'Semi-Lexicographic Heuristic'. This new theory should posit that subjects evaluate the numerical gaps between cue validities: if a gap between adjacent cues exceeds a certain subjective threshold, all lower-ranked cues are heavily discounted or outright ignored (acting non-compensatory). If the validities are clustered closely together, subjects integrate them additively. This differs from Theory 2's continuous power-law scaling by introducing discrete strategy shifts or step-function discounting based on validity gaps, which might better capture the high override rates seen in Experiments 7 and 8."
}
```

## Usage

```json
{
  "prompt_token_count": 8262,
  "candidates_token_count": 386,
  "total_token_count": 9503
}
```
