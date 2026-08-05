# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3_1" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3_1
Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weights for all features, ignoring validities.
    # The score for each option is simply the sum of its positive features.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_5
Cancellation and Focus (Ratio Rule)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Cancellation: eliminate features shared by both options.
    # Focus only on unique positive features.
    unique_a = np.sum((stim[0] > stim[1]).astype(float))
    unique_b = np.sum((stim[1] > stim[0]).astype(float))
    
    scores = np.array([unique_a, unique_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if unique_a == 0 and unique_b == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Ratio rule over unique features to provide a distinct alternative to Tallying's softmax.
        s_beta = np.zeros_like(scores)
        for i in range(len(scores)):
            if scores[i] > 0:
                s_beta[i] = scores[i] ** beta
            else:
                s_beta[i] = 0.0 if beta > 0 else 1.0
        p_core = s_beta / np.sum(s_beta)
        
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate Tallying from Cancellation and Focus, we manipulate the number of shared features while keeping the difference in total positive features constant. Tallying computes the unweighted sum of positive features and applies a softmax rule based on the difference in these sums, meaning it predicts the same choice probabilities for any trial pairs with the same difference in total features. Cancellation and Focus, however, eliminates shared features and applies a ratio rule to the remaining unique positive features. We design trials where Option A always has exactly 1 more positive feature than Option B. In some trials (e.g., A has 4, B has 3, with 3 shared), B has no unique positive features, so Cancellation predicts a deterministic choice for A (1 vs 0 unique), while Tallying predicts a soft preference. In other trials (e.g., A has 3, B has 2, with 0 shared), Cancellation applies a ratio rule to a mixture of unique features (3 vs 2), predicting a much weaker preference for A, whereas Tallying's prediction remains identical to the first case.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
To overcome the high between-subject variance caused by uniformly sampled parameters (beta and epsilon), this metric computes a within-subject Welch's t-statistic for the difference in choice probabilities. We compare Trial 1 (where Option B has unique features) against the pooled Deterministic trials (T2, T4, T5, T6, where Option B has no unique features). Tallying predicts identical choice probabilities for both groups because Option A always has exactly 1 more total positive feature than Option B; therefore, the t-statistic for Tallying will follow a standard normal distribution (mean ~0, variance ~1). Cancellation and Focus predicts a higher probability of choosing B in Trial 1 than in the Deterministic trials, yielding a strongly positive t-statistic. By standardizing the difference by its standard error within each subject, we heavily stabilize the variance for the null theory (Tallying) while amplifying the signal for Cancellation.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    # Focus on trials where Option A has exactly 1 more positive feature than Option B
    diff_1 = (a_sum - b_sum) == 1
    
    def unique_b_count(row):
        return sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b == 1 and a == 0)
    
    unique_b = data.apply(unique_b_count, axis=1)
    
    # T1: Option B has unique features (soft choice under Cancellation)
    is_t1 = diff_1 & (unique_b > 0)
    # T2, T4, T5, T6: Option B has NO unique features (deterministic choice under Cancellation)
    is_det = diff_1 & (unique_b == 0)
    
    n_t1 = is_t1.sum()
    n_det = is_det.sum()
    
    if n_t1 == 0 or n_det == 0:
        return 0.0
        
    k_t1 = data.loc[is_t1, 'response'].sum()  # Number of times B was chosen
    k_det = data.loc[is_det, 'response'].sum()
    
    # Laplace smoothing to prevent zero variance
    p_t1 = (k_t1 + 1.0) / (n_t1 + 2.0)
    p_det = (k_det + 1.0) / (n_det + 2.0)
    
    var_t1 = p_t1 * (1.0 - p_t1) / (n_t1 + 2.0)
    var_det = p_det * (1.0 - p_det) / (n_det + 2.0)
    
    se = np.sqrt(var_t1 + var_det)
    
    # Return the within-subject t-statistic of the difference
    return float((p_t1 - p_det) / se)


### RESULTS
- Predicted under pi_3_1 (simulated): 0.5709 (var=1.4069)
- Predicted under pi_5 (simulated): 7.0416 (var=1.6042)
- Observed on real data: 0.1650 (var=1.3651)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate Cancellation and Focus from Tallying, we exploit how each theory handles shared versus unique features. Tallying computes the unweighted sum of positive features and applies a softmax rule based on the difference in these sums. Therefore, Tallying predicts identical choice probabilities for any pair of options that have the same difference in total positive features. Cancellation and Focus, however, first eliminates shared positive features and then applies a ratio rule to the remaining unique positive features. We design a set of trials where the difference in total positive features is held constant at +1 (Option A always has exactly 1 more positive feature than Option B), but the number of shared features varies. In Trial 1, Option A has 4 positive features and Option B has 3, with 3 shared. Option B thus has 0 unique positive features, and Cancellation predicts a deterministic choice for Option A. In Trial 2, Option A has 2 positive features and Option B has 1, with 0 shared. Cancellation applies a ratio rule to 2 vs 1 unique features, predicting a much softer preference for A. Tallying predicts the exact same preference strength in both trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the probability of choosing Option A between Trial 1 and Trial 2. In both trials, Option A has exactly 1 more positive feature than Option B. According to Tallying, the choice probability depends only on the difference in the total sum of positive features. Therefore, Tallying predicts identical choice probabilities for Option A in both trials, meaning the metric should be close to 0. In contrast, Cancellation and Focus eliminates shared features. In Trial 1, after cancellation, Option A has 1 unique positive feature and Option B has 0, leading to a near-deterministic choice for A. In Trial 2, Option A has 2 unique features and Option B has 1, leading to a softer preference (ratio rule). Thus, Cancellation and Focus predicts a significantly higher probability of choosing A in Trial 1 compared to Trial 2, yielding a positive metric.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: A has 4 positive features, B has 3. Shared = 3.
    # Unique A = 1, Unique B = 0.
    t1_mask = (a_tup == (1, 1, 1, 1, 0)) & (b_tup == (1, 1, 1, 0, 0))
    
    # Trial 2: A has 2 positive features, B has 1. Shared = 0.
    # Unique A = 2, Unique B = 1.
    t2_mask = (a_tup == (1, 1, 0, 0, 0)) & (b_tup == (0, 0, 1, 0, 0))
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1): p_a_t1 = 0.5
    if pd.isna(p_a_t2): p_a_t2 = 0.5
    
    return float(p_a_t1 - p_a_t2)


### RESULTS
- Predicted under pi_3_1 (simulated): -0.0250 (var=0.0230)
- Predicted under pi_5 (simulated): 0.1025 (var=0.0295)
- Observed on real data: 0.0325 (var=0.0278)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    diff = a_ratings - b_ratings
    # By using exponentially decreasing weights, the sign of the dot product 
    # is completely determined by the first non-zero difference, 
    # exactly mimicking Take The Best's lexicographic choice rule.
    weights = np.array([10000, 1000, 100, 10, 1])
    ttb_scores = diff @ weights
    ttb_choices = np.where(ttb_scores > 0, 0, 1)
    valid_trials = ttb_scores != 0
    if not np.any(valid_trials):
        return 0.5
    matches = (data['response'].values[valid_trials] == ttb_choices[valid_trials])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3458 (var=0.0045)
**Predicted under pi_3_1:** 0.3287 (var=0.0195)
**Predicted under pi_5:** 0.3252 (var=0.0160)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    count = 0
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner is not None:
            sum_a = sum(a)
            sum_b = sum(b)
            # Isolate compensatory trials: the TTB winner has a strictly smaller sum of features
            if (ttb_winner == 0 and sum_a < sum_b) or (ttb_winner == 1 and sum_b < sum_a):
                if resp == ttb_winner:
                    matches += 1
                count += 1
                
    return matches / count if count > 0 else 0.5
```

**Observed (real) value:** 0.3217 (var=0.0048)
**Predicted under pi_3_1:** 0.3088 (var=0.0206)
**Predicted under pi_5:** 0.3187 (var=0.0158)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_a = (1, 1, 0, 0, 0)
    t1_b = (0, 0, 1, 1, 1)
    
    mask1 = (a_tuples == t1_a) & (b_tuples == t1_b)
    mask2 = (a_tuples == t1_b) & (b_tuples == t1_a)
    
    chosen_t1_b = (mask1 & (data['response'] == 1)).sum() + (mask2 & (data['response'] == 0)).sum()
    total = mask1.sum() + mask2.sum()
    
    if total == 0:
        return 0.5
    return float(chosen_t1_b / total)
```

**Observed (real) value:** 0.6333 (var=0.0139)
**Predicted under pi_3_1:** 0.6467 (var=0.0382)
**Predicted under pi_5:** 0.6250 (var=0.0281)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # We focus on the 'opposition' trial where the two models make opposite predictions.
    # The target option is (1, 1, 0, 0, 0), which has fewer positive features (2 vs 3) 
    # but higher total validity (0.95+0.85=1.80 vs 0.60+0.55+0.50=1.65).
    # WADD prefers the target option; Tallying prefers the alternative (0, 0, 1, 1, 1).
    
    a_is_target = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (1, 1, 0, 0, 0))
    b_is_target = data['option_b_ratings'].apply(lambda x: tuple(int(v) for v in x) == (1, 1, 0, 0, 0))
    
    # Response 0 means A was chosen, 1 means B was chosen.
    chose_target = (a_is_target & (data['response'] == 0)) | (b_is_target & (data['response'] == 1))
    relevant = a_is_target | b_is_target
    
    if relevant.sum() == 0:
        return 0.5
        
    return float(chose_target.sum() / relevant.sum())
```

**Observed (real) value:** 0.3550 (var=0.0190)
**Predicted under pi_3_1:** 0.3425 (var=0.0269)
**Predicted under pi_5:** 0.4325 (var=0.0212)

### Experiment 7
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_aligned = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Focus on 'indifference' trials for Tallying
        if sum(a) == sum(b):
            # Cue 0 is the most valid cue in this design
            if a[0] > b[0]:
                ttb_choice = 0
            elif b[0] > a[0]:
                ttb_choice = 1
            else:
                continue
                
            if row['response'] == ttb_choice:
                ttb_aligned.append(1)
            else:
                ttb_aligned.append(0)
                
    if len(ttb_aligned) == 0:
        return 0.5
        
    return float(np.mean(ttb_aligned))
```

**Observed (real) value:** 0.5100 (var=0.0037)
**Predicted under pi_3_1:** 0.5012 (var=0.0062)
**Predicted under pi_5:** 0.4963 (var=0.0078)

### Experiment 8
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying predicts indifference (equal sums)
    indiff_mask = sum_a == sum_b
    if not indiff_mask.any():
        return 0.5
        
    indiff_data = data[indiff_mask]
    
    # Compute the proportion of times Option A was chosen on these trials
    # Response == 0 means Option A was chosen
    p_a = (indiff_data['response'] == 0).mean()
    
    return float(p_a)
```

**Observed (real) value:** 0.5000 (var=0.0051)
**Predicted under pi_3_1:** 0.4929 (var=0.0044)
**Predicted under pi_5:** 0.4917 (var=0.0054)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for idx, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        if sum(a) == sum(b):
            ttb_winner = None
            for j in range(len(a)):
                if a[j] > b[j]:
                    ttb_winner = 0
                    break
                elif b[j] > a[j]:
                    ttb_winner = 1
                    break
                    
            if ttb_winner is not None:
                ttb_match.append(1 if row['response'] == ttb_winner else 0)
                
    if len(ttb_match) == 0:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.5167 (var=0.0043)
**Predicted under pi_3_1:** 0.5083 (var=0.0088)
**Predicted under pi_5:** 0.5089 (var=0.0055)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Isolate 'indifference' trials where both options have the same number of positive features
    indiff_mask = sum_a == sum_b
    if not indiff_mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen on these trials
    # Option A is favored by the Take-The-Best heuristic on all these trials.
    return float((data.loc[indiff_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5044 (var=0.0081)
**Predicted under pi_3_1:** 0.4950 (var=0.0090)
**Predicted under pi_5:** 0.5083 (var=0.0076)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (Tallying) consistently outperforms Theory 2 (Cancellation and Focus) across almost all experiments. In Experiment 1, Theory 2's prediction (7.04) wildly overshoots the observed metric (0.16), while Theory 1 is much closer (0.57). In Experiment 2, Theory 1 again closely matches the observed data (-0.025 vs 0.032) compared to Theory 2 (0.10). Across experiments 3-10, Tallying continues to be a very strong predictor of subject behavior, as subjects seem to largely ignore cue validities and simply count positive features. Theory 2's assumption that shared features are perfectly cancelled out and decisions are made on a ratio of unique features is not supported by the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Cancellation and Focus with a new theory that serves as a stronger baseline for compensatory decision making, such as the Weighted Additive (WADD) model. WADD assumes subjects integrate all features weighted by their given cue validities. While Tallying assumes equal weights, introducing WADD will allow us to test if there are any subtle effects of cue validities that a pure equal-weighting tallying model misses, providing a classic rational baseline to contrast with the heuristic Tallying model."
}
```

## Usage

```json
{
  "prompt_token_count": 9157,
  "candidates_token_count": 310,
  "total_token_count": 10567
}
```
