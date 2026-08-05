# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_18" and "pi_17") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_18" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_17" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_18
Parallel Similarity and Feature-Difference Model: Decision-makers evaluate options in parallel by computing a similarity-weighted score. Instead of a discrete stopping rule, the strategy smoothly transitions between Tallying (equal weighting) and a heavily weighted linear model (WADD) based on the overall similarity of the options. A Jaccard-like similarity metric is used. To ensure stability, the feature weights for both strategies are normalized to sum to 1.0, and the softmax temperature parameter is expanded to accommodate the normalized scale, allowing for highly deterministic choice behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate similarity using Jaccard index (matching presences)
    a_bool = a > 0.5
    b_bool = b > 0.5
    intersection = np.sum(a_bool & b_bool)
    union = np.sum(a_bool | b_bool)
    sim = float(intersection) / float(union) if union > 0 else 1.0
    
    threshold = float(parameters["sim_threshold"])
    slope = float(parameters["slope"])
    
    # Logistic transition function:
    # With slope in [-50, 50], optimization can determine if high similarity 
    # leads to Tallying (alpha->1) or WADD (alpha->0).
    z_alpha = -slope * (sim - threshold)
    z_alpha = np.clip(z_alpha, -100, 100)
    alpha = 1.0 / (1.0 + np.exp(z_alpha))
    
    gamma = float(parameters["gamma"])
    # Non-linear scaling of validities for the WADD component, normalized to sum to 1
    w_wadd_raw = val ** gamma
    w_wadd = w_wadd_raw / np.sum(w_wadd_raw)
    
    # Tallying weights, normalized to sum to 1
    w_tally = np.ones_like(val) / len(val)
    
    # Interpolate feature weights
    weights = alpha * w_tally + (1.0 - alpha) * w_wadd
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_17
Strategy Selection based on Environment Classification

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Environment classification based on validity dispersion
    dispersion = np.std(val)
    threshold = float(parameters["dispersion_threshold"])
    slope = float(parameters["slope"])
    
    # Probability of selecting Take-The-Best over Tallying
    w_ttb = 1.0 / (1.0 + np.exp(-slope * (dispersion - threshold)))
    
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    diff = a - b
    ttb_a, ttb_b = 0.0, 0.0
    for idx in order:
        if diff[idx] > 0:
            ttb_a = 1.0
            break
        elif diff[idx] < 0:
            ttb_b = 1.0
            break
            
    # Tallying prediction
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    
    z_ttb = beta * np.array([ttb_a, ttb_b])
    p_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb /= np.sum(p_ttb)
    
    z_tally = beta * np.array([tally_a, tally_b])
    p_tally = np.exp(z_tally - np.max(z_tally))
    p_tally /= np.sum(p_tally)
    
    # Mixture of strategies
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_18)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 8: A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Parallel Similarity and Feature-Difference Model) from the Competing Theory (Environment Classification Strategy Selection) by exploiting the trial-by-trial similarity metric unique to the Advocated Theory. The Competing Theory evaluates the environment's validity dispersion once, establishing a fixed mixture weight between Take-The-Best (TTB) and Tallying for all trials. Consequently, if two trials have identical feature differences (Option A - Option B), the Competing Theory must predict the exact same choice probability for both, regardless of the options' baseline features. In contrast, the Advocated Theory computes the Jaccard similarity of the options' positive features on every trial, using this similarity to dynamically interpolate between Tallying and a Weighted Additive (WADD) strategy. By constructing pairs of trials with perfectly identical feature differences but vastly different numbers of shared positive features, we create a stark double dissociation: the Competing Theory predicts a flat, identical response profile across these matched pairs, whereas the Advocated Theory predicts significant shifts in choice probability as the varying similarity triggers a trial-by-trial transition between compensatory and non-compensatory processing.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the probability of choosing Option B on Trials 5 and 6. In these trials, Option A has features {1, 4} (validities 0.95, 0.6) and Option B has {2, 3} (validities 0.9, 0.85). Both options have exactly 2 positive features, meaning the Tallying strategy evaluates them as a perfect tie. 

For the Competing Theory, the Take-The-Best (TTB) strategy strictly favors Option A because it possesses the most valid differing feature (v1 = 0.95). Since Tallying is a tie, any mixture of TTB and Tallying will systematically favor Option A, meaning P(Choose B) is strictly less than 0.5.

For the Advocated Theory, the Weighted Additive (WADD) strategy evaluates the sum of non-linearly scaled validities. For almost all valid parameter ranges, WADD favors Option B because the combined weight of v2 and v3 outweighs v1 and v4 (e.g., 0.9 + 0.85 = 1.75 > 0.95 + 0.6 = 1.55). Since Tallying is a tie, the Advocated Theory systematically favors Option B, meaning P(Choose B) is strictly greater than 0.5. This structural opposition creates a massive, highly reliable mean gap between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    def is_t5_or_t6(a, b):
        a_tup = tuple(int(x) for x in a)
        b_tup = tuple(int(x) for x in b)
        if a_tup == (1, 0, 0, 1, 0, 0) and b_tup == (0, 1, 1, 0, 0, 0):
            return True
        if a_tup == (1, 0, 0, 1, 1, 1) and b_tup == (0, 1, 1, 0, 1, 1):
            return True
        return False
        
    mask = data.apply(lambda row: is_t5_or_t6(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    return float(target_data['response'].mean())

### RESULTS
- Predicted under pi_18 (simulated): 0.6358 (var=0.0193)
- Predicted under pi_17 (simulated): 0.3217 (var=0.0284)
- Observed on real data: 0.7733 (var=0.0252)

## EXPERIMENT 2 (proposed by pi_17)

### DESIGN
**Validities (n_features=9):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 1, 1, 1]
  trial 5: A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Environment Classification Strategy Selection) from the Competing Theory (Parallel Similarity and Feature-Difference Model), we exploit the fact that the Advocated Theory fixes its strategy mixture (TTB vs Tallying) based on the overall experiment's validity dispersion, and evaluates options based strictly on their feature differences. Thus, for any two trials with identical feature differences (Option A - Option B), the Advocated Theory must predict the exact same choice probabilities. In contrast, the Competing Theory computes the Jaccard similarity of the options on every trial to dynamically interpolate between Tallying and a Weighted Additive (WADD) strategy. By constructing matched pairs of trials with perfectly identical feature differences but vastly different numbers of shared positive features, we create a sharp dissociation: the Advocated Theory predicts a flat, identical response profile across these matched pairs, whereas the Competing Theory predicts significant shifts in choice probability as varying similarity triggers a trial-by-trial transition between compensatory and equal-weighting processing.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
By computing the sum of squared differences (SSD) in choice probabilities between matched trials with identical feature differences but varying similarity, we amplify the signal of the Competing Theory's dynamic strategy shifts. The Advocated Theory, which strictly bases decisions on feature differences, predicts an SSD near zero (driven only by binomial noise). The Competing Theory, which transitions between Tallying and WADD based on similarity, will often produce significantly larger squared differences, leading to a robust dissociation.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    diffs = []
    unions = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        diffs.append(tuple(int(x) - int(y) for x, y in zip(a, b)))
        unions.append(sum(max(int(x), int(y)) for x, y in zip(a, b)))
        
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'diff': diffs,
        'union': unions,
        'response': data['response']
    })
    
    grouped = df.groupby(['subject_id', 'diff', 'union'])['response'].mean().reset_index()
    
    subj_ssds = []
    for subj, subj_df in grouped.groupby('subject_id'):
        ssd = 0.0
        for d, grp in subj_df.groupby('diff'):
            if len(grp) == 2:
                u_min = grp['union'].min()
                u_max = grp['union'].max()
                v1 = grp.loc[grp['union'] == u_min, 'response'].values[0]
                v2 = grp.loc[grp['union'] == u_max, 'response'].values[0]
                ssd += (v1 - v2) ** 2
        subj_ssds.append(ssd)
            
    return float(np.mean(subj_ssds)) if subj_ssds else 0.0

### RESULTS
- Predicted under pi_18 (simulated): 0.2720 (var=0.1248)
- Predicted under pi_17 (simulated): 0.0723 (var=0.0029)
- Observed on real data: 0.0423 (var=0.0018)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            matches.append(1.0 if resp == ttb_winner else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5711 (var=0.0014)
**Predicted under pi_18:** 0.4067 (var=0.0686)
**Predicted under pi_17:** 0.5098 (var=0.0740)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_mask = a_wins != b_wins
    if not np.any(valid_mask):
        return 0.5
        
    tally_preds = np.where(a_wins > b_wins, 0, 1)
    responses = np.array(data['response'].tolist())
    
    matches = (responses[valid_mask] == tally_preds[valid_mask])
    return float(np.mean(matches))

```

**Observed (real) value:** 0.6890 (var=0.0042)
**Predicted under pi_18:** 0.5487 (var=0.0831)
**Predicted under pi_17:** 0.4542 (var=0.0708)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.7, 0.65, 0.6, 0.55])
    w = val - 0.5
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Isolate trials where Tallying sees a tie
        if a_wins == b_wins:
            score_a = np.sum(a * w)
            score_b = np.sum(b * w)
            
            if score_a > score_b:
                target = 0
            elif score_b > score_a:
                target = 1
            else:
                continue
                
            if row['response'] == target:
                match_count += 1
            total += 1
            
    if total == 0:
        return 0.5
        
    return match_count / total
```

**Observed (real) value:** 0.6200 (var=0.0045)
**Predicted under pi_18:** 0.6494 (var=0.0410)
**Predicted under pi_17:** 0.6622 (var=0.0267)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t3 = (a_str == '10001') & (b_str == '01110')
    t5 = (a_str == '01011') & (b_str == '10100')
    t6 = (a_str == '10010') & (b_str == '01101')
    
    score = 0.0
    count = 0
    
    if t3.any():
        score += data.loc[t3, 'response'].mean()
        count += 1
    if t5.any():
        score += (1 - data.loc[t5, 'response']).mean()
        count += 1
    if t6.any():
        score += data.loc[t6, 'response'].mean()
        count += 1
        
    return score / count if count > 0 else 0.5
```

**Observed (real) value:** 0.8542 (var=0.0086)
**Predicted under pi_18:** 0.5187 (var=0.1074)
**Predicted under pi_17:** 0.4892 (var=0.0737)

### Experiment 7
**Design**
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    correct = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1
        if a == (1, 0, 1, 0) and b == (0, 1, 0, 1):
            correct.append(1 if row['response'] == 0 else 0)
        elif a == (0, 1, 0, 1) and b == (1, 0, 1, 0):
            correct.append(1 if row['response'] == 1 else 0)
            
        # Trial 3
        elif a == (1, 1, 0, 0) and b == (0, 0, 1, 1):
            correct.append(1 if row['response'] == 0 else 0)
        elif a == (0, 0, 1, 1) and b == (1, 1, 0, 0):
            correct.append(1 if row['response'] == 1 else 0)
            
        # Trial 4
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 0):
            correct.append(1 if row['response'] == 0 else 0)
        elif a == (0, 0, 1, 0) and b == (0, 1, 0, 0):
            correct.append(1 if row['response'] == 1 else 0)
            
    if not correct:
        return 0.5
    return float(np.mean(correct))
```

**Observed (real) value:** 0.3850 (var=0.0061)
**Predicted under pi_18:** 0.6742 (var=0.0388)
**Predicted under pi_17:** 0.6646 (var=0.0295)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tied_mask = a_wins == b_wins
    
    if not np.any(tied_mask):
        return 0.5
        
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_val = np.sum(a_ratings * val, axis=1)
    b_val = np.sum(b_ratings * val, axis=1)
    
    preferred = np.where(a_val > b_val, 0, np.where(b_val > a_val, 1, -1))
    
    valid_mask = tied_mask & (preferred != -1)
    
    if not np.any(valid_mask):
        return 0.5
        
    responses = data['response'].values
    
    alignment = responses[valid_mask] == preferred[valid_mask]
    
    return float(np.mean(alignment))

```

**Observed (real) value:** 0.3250 (var=0.0053)
**Predicted under pi_18:** 0.6388 (var=0.0119)
**Predicted under pi_17:** 0.5887 (var=0.0103)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def is_tie(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a > b) == np.sum(b > a)
        
    ties = data[data.apply(is_tie, axis=1)]
    if len(ties) == 0:
        return 0.0
        
    devs = []
    for subj, subj_df in ties.groupby('subject_id'):
        p_A = np.mean(subj_df['response'] == 0)
        devs.append(np.abs(p_A - 0.5))
        
    return float(np.mean(devs))
```

**Observed (real) value:** 0.0617 (var=0.0029)
**Predicted under pi_18:** 0.2092 (var=0.0278)
**Predicted under pi_17:** 0.2250 (var=0.0226)

### Experiment 10
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    df_zero = data[a_wins == b_wins]
    if len(df_zero) == 0:
        return 0.0
        
    subj_means = df_zero.groupby('subject_id')['response'].mean()
    return float(np.mean(np.abs(subj_means - 0.5)))

```

**Observed (real) value:** 0.3450 (var=0.0185)
**Predicted under pi_18:** 0.2850 (var=0.0244)
**Predicted under pi_17:** 0.1925 (var=0.0172)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    f1_chosen = 0
    total_diff = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        if a[0] > b[0]:
            f1_chosen += (1 if resp == 0 else 0)
            total_diff += 1
        elif b[0] > a[0]:
            f1_chosen += (1 if resp == 1 else 0)
            total_diff += 1
            
    if total_diff == 0:
        return 0.5
    return float(f1_chosen / total_diff)

```

**Observed (real) value:** 0.4933 (var=0.0023)
**Predicted under pi_18:** 0.5908 (var=0.0740)
**Predicted under pi_17:** 0.5919 (var=0.0621)

### Experiment 12
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: A has more wins, but B wins on the most valid features
    t1 = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    # Trial 2 & 4: Tied wins, but B wins on the most valid features
    t2 = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 0, 1))
    t4 = (a_tuples == (0, 1, 0, 0, 1)) & (b_tuples == (1, 0, 1, 0, 0))
    
    mask = t1 | t2 | t4
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen in these trials
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.3858 (var=0.0034)
**Predicted under pi_18:** 0.4008 (var=0.0380)
**Predicted under pi_17:** 0.3992 (var=0.0471)

### Experiment 13
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_A_01111 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    is_B_10000 = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    is_A_10000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    is_B_01111 = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    
    mask1 = is_A_01111 & is_B_10000
    mask2 = is_A_10000 & is_B_01111
    
    target_trials = data[mask1 | mask2]
    if len(target_trials) == 0:
        return 0.0
        
    chose_10000 = ((target_trials['response'] == 1) & mask1) | ((target_trials['response'] == 0) & mask2)
    return float(chose_10000.mean())
```

**Observed (real) value:** 0.1396 (var=0.0004)
**Predicted under pi_18:** 0.0500 (var=0.0021)
**Predicted under pi_17:** 0.0838 (var=0.0023)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    p_a_t8 = (data.loc[a_str == '00111', 'response'] == 0).mean()
    p_a_t1 = (data.loc[a_str == '10000', 'response'] == 0).mean()
    if pd.isna(p_a_t8) or pd.isna(p_a_t1):
        return 0.0
    return float(p_a_t8 - p_a_t1)
```

**Observed (real) value:** 0.0233 (var=0.0075)
**Predicted under pi_18:** 0.2250 (var=0.3517)
**Predicted under pi_17:** 0.1283 (var=0.2928)

### Experiment 15
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    is_trial_1 = ((a_str == "01100") & (b_str == "10000")) | ((a_str == "10000") & (b_str == "01100"))
    is_trial_3 = ((a_str == "10011") & (b_str == "01100")) | ((a_str == "01100") & (b_str == "10011"))
    
    t1_data = data[is_trial_1]
    t3_data = data[is_trial_3]
    
    if len(t1_data) == 0 or len(t3_data) == 0:
        return 0.0
        
    a_is_23_t1 = t1_data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) == "01100"
    chose_23_t1 = (a_is_23_t1 & (t1_data['response'] == 0)) | (~a_is_23_t1 & (t1_data['response'] == 1))
    
    a_is_23_t3 = t3_data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) == "01100"
    chose_23_t3 = (a_is_23_t3 & (t3_data['response'] == 0)) | (~a_is_23_t3 & (t3_data['response'] == 1))
    
    return float(chose_23_t1.mean() + chose_23_t3.mean())
```

**Observed (real) value:** 0.2611 (var=0.0265)
**Predicted under pi_18:** 1.3305 (var=0.1346)
**Predicted under pi_17:** 0.5884 (var=0.1169)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    f1_chosen = ((a_f1 == 1) & chose_a) | ((b_f1 == 1) & chose_b)
    return float(f1_chosen.mean())
```

**Observed (real) value:** 0.3254 (var=0.0027)
**Predicted under pi_18:** 0.2894 (var=0.0179)
**Predicted under pi_17:** 0.5942 (var=0.0599)

### Experiment 17
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 3: A=[1, 0, 0, 0, 1], B=[0, 1, 1, 0, 0]
    t3_mask = (data['a_str'] == '10001') & (data['b_str'] == '01100')
    # Trial 4: A=[0, 1, 0, 1, 1], B=[1, 0, 0, 0, 0]
    t4_mask = (data['a_str'] == '01011') & (data['b_str'] == '10000')
    
    p_a_t3 = (data[t3_mask]['response'] == 0).mean()
    p_a_t4 = (data[t4_mask]['response'] == 0).mean()
    
    def safe_mean(val):
        return 0.5 if pd.isna(val) else float(val)
        
    return 2.0 * safe_mean(p_a_t3) + safe_mean(p_a_t4)
```

**Observed (real) value:** 1.1875 (var=0.0375)
**Predicted under pi_18:** 1.8687 (var=0.0939)
**Predicted under pi_17:** 1.8375 (var=0.0598)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t2 = data['option_a_ratings'].apply(tuple) == (0, 1, 1, 1, 0)
    is_t3 = data['option_a_ratings'].apply(tuple) == (1, 0, 1, 0, 0)
    
    p_a_t2 = (data.loc[is_t2, 'response'] == 0).mean()
    p_a_t3 = (data.loc[is_t3, 'response'] == 0).mean()
    
    return float(p_a_t2 + p_a_t3)
```

**Observed (real) value:** 0.9950 (var=0.0117)
**Predicted under pi_18:** 1.3050 (var=0.1375)
**Predicted under pi_17:** 0.9925 (var=0.0137)

### Experiment 19
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    mask1 = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    mask2 = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    wadd_choices = 0
    wadd_choices += (data.loc[mask1, 'response'] == 1).sum()
    wadd_choices += (data.loc[mask2, 'response'] == 0).sum()
    
    total = mask1.sum() + mask2.sum()
    
    if total == 0:
        return 0.0
    return float(wadd_choices / total)
```

**Observed (real) value:** 0.1523 (var=0.0164)
**Predicted under pi_18:** 0.4015 (var=0.0478)
**Predicted under pi_17:** 0.1508 (var=0.0080)

### Experiment 20
**Design**
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Project list columns to tuples to make them hashable/comparable
    data['opt_a_tup'] = data['option_a_ratings'].apply(tuple)
    
    # Define the Option A and Option B rating patterns for the target trials
    t1_a = (0, 1, 1, 1, 0, 0, 0)
    t1_b = (1, 0, 0, 0, 1, 1, 1)
    
    t3_a = (0, 0, 1, 1, 1, 1, 0)
    t3_b = (1, 1, 0, 0, 0, 0, 1)
    
    t7_a = (0, 0, 0, 1, 1, 1, 1)
    t7_b = (1, 1, 1, 0, 0, 0, 0)
    
    def get_rate(t_a, t_b):
        mask_f = data['opt_a_tup'] == t_a
        mask_b = data['opt_a_tup'] == t_b
        
        rates = []
        if mask_f.sum() > 0:
            rates.append((data.loc[mask_f, 'response'] == 0).mean())
        if mask_b.sum() > 0:
            rates.append((data.loc[mask_b, 'response'] == 1).mean())
            
        return float(np.mean(rates)) if rates else 0.0

    r1 = get_rate(t1_a, t1_b)
    r3 = get_rate(t3_a, t3_b)
    r7 = get_rate(t7_a, t7_b)
    
    return float(r1 + r3 - r7)
```

**Observed (real) value:** 0.8083 (var=0.0226)
**Predicted under pi_18:** 0.4142 (var=0.0752)
**Predicted under pi_17:** 0.1775 (var=0.0374)

### Experiment 21
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    target_opt = (0, 1, 1, 0, 0)
    
    is_target_a = (a_tuples == target_opt)
    is_target_b = (b_tuples == target_opt)
    
    mask = is_target_a | is_target_b
    if not mask.any():
        return 0.0
        
    subset = data[mask]
    is_a_subset = is_target_a[mask]
    is_b_subset = is_target_b[mask]
    
    chose_target = (is_a_subset & (subset['response'] == 0)) | (is_b_subset & (subset['response'] == 1))
    
    return float(chose_target.mean())
```

**Observed (real) value:** 0.1283 (var=0.0127)
**Predicted under pi_18:** 0.3300 (var=0.0576)
**Predicted under pi_17:** 0.1833 (var=0.0155)

### Experiment 22
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_vals = data['option_a_ratings'].apply(tuple)
    t1 = a_vals == (0, 1, 1, 0, 0)
    t2 = a_vals == (1, 0, 0, 1, 1)
    t3 = a_vals == (0, 1, 0, 1, 1)
    t4 = a_vals == (1, 0, 1, 0, 0)
    
    m1 = data.loc[t1, 'response'].mean() if t1.any() else 0.5
    m2 = data.loc[t2, 'response'].mean() if t2.any() else 0.5
    m3 = data.loc[t3, 'response'].mean() if t3.any() else 0.5
    m4 = data.loc[t4, 'response'].mean() if t4.any() else 0.5
    
    return float((m3 - m4) - (m1 - m2))
```

**Observed (real) value:** -1.4933 (var=0.1750)
**Predicted under pi_18:** -0.3842 (var=1.4798)
**Predicted under pi_17:** -0.4650 (var=0.3649)

### Experiment 23
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate sum of features for A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Determine if subject chose the option with the higher sum
    # response == 0 means A, response == 1 means B
    chose_higher_sum = ((sum_a > sum_b) & (data['response'] == 0)) | \
                       ((sum_b > sum_a) & (data['response'] == 1))
                       
    # Determine if the most valid feature (feature 0) is tied
    feat0_a = data['option_a_ratings'].apply(lambda x: x[0])
    feat0_b = data['option_b_ratings'].apply(lambda x: x[0])
    feat0_tied = (feat0_a == feat0_b)
    
    # Calculate mean of chose_higher_sum for tied and untied trials
    mean_tied = chose_higher_sum[feat0_tied].mean()
    mean_untied = chose_higher_sum[~feat0_tied].mean()
    
    return float(mean_tied - mean_untied)
```

**Observed (real) value:** 0.1719 (var=0.0043)
**Predicted under pi_18:** -0.0516 (var=0.0672)
**Predicted under pi_17:** -0.0306 (var=0.0070)

### Experiment 24
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    df = data.copy()
    # Convert response to +1 for choosing A, -1 for choosing B
    df['C'] = 1 - 2 * df['response']
    
    # Hashable representations
    df['a_tup'] = df['option_a_ratings'].apply(tuple)
    df['b_tup'] = df['option_b_ratings'].apply(tuple)
    
    def get_c(a_target, b_target):
        mask1 = (df['a_tup'] == a_target) & (df['b_tup'] == b_target)
        mask2 = (df['a_tup'] == b_target) & (df['b_tup'] == a_target)
        
        c_sum = 0.0
        if mask1.sum() > 0:
            c_sum += df.loc[mask1, 'C'].sum()
        if mask2.sum() > 0:
            c_sum -= df.loc[mask2, 'C'].sum()
            
        total = mask1.sum() + mask2.sum()
        return c_sum / total if total > 0 else 0.0

    # Trial 1: F0 discriminates (A is better), rest favor B
    c1 = get_c((1,0,0,0,0), (0,1,1,1,1))
    # Trial 2: F0 tied, F1 favors A, rest favor B
    c2 = get_c((1,1,0,0,0), (1,0,1,1,1))
    # Trial 3: F0 tied, F1 favors B, rest favor A
    c3 = get_c((0,0,1,1,1), (0,1,0,0,0))
    # Trial 4: F0 discriminates (B is better), rest favor A
    c4 = get_c((0,1,1,1,0), (1,0,0,0,1))
    
    return float(c1 * c2 + c3 * c4)
```

**Observed (real) value:** 0.9954 (var=0.3116)
**Predicted under pi_18:** 0.2829 (var=0.3776)
**Predicted under pi_17:** 0.0014 (var=0.3506)

### Experiment 25
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    is_m1 = (a_str == '01100') & (b_str == '10011')
    is_m2 = (a_str == '10011') & (b_str == '01100')
    
    relevant = is_m1 | is_m2
    if not relevant.any():
        return 0.5
        
    chose_x = (is_m1 & (data['response'] == 0)) | (is_m2 & (data['response'] == 1))
    
    return float(chose_x.sum() / relevant.sum())
```

**Observed (real) value:** 0.1333 (var=0.0161)
**Predicted under pi_18:** 0.4917 (var=0.0845)
**Predicted under pi_17:** 0.1783 (var=0.0219)

### Experiment 26
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify the critical trial where theories strongly diverge
    # Trial 1: Option A wins on features 2 and 3, Option B wins on 1, 4, and 5
    a_target = (0, 1, 1, 0, 0)
    b_target = (1, 0, 0, 1, 1)
    
    a_match = data['option_a_ratings'].apply(tuple) == a_target
    b_match = data['option_b_ratings'].apply(tuple) == b_target
    mask = a_match & b_match
    
    if mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1579 (var=0.0122)
**Predicted under pi_18:** 0.4400 (var=0.0963)
**Predicted under pi_17:** 0.2063 (var=0.0231)

### Experiment 27
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Create hashable string representations of the ratings
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Identify Trial 4 and Trial 5
    t4_mask = (a_str == '11000') & (b_str == '00111')
    t5_mask = (a_str == '01100') & (b_str == '10011') # Wait, T5 in experiment is A=[0,1,1,1,0] B=[1,0,0,0,1]
    t5_mask = (a_str == '01110') & (b_str == '10001')
    
    # Calculate the product of B choice rates on T4 and T5 per subject
    # For Competing theory, subjects never reliably choose B on both T4 and T5
    # because T4 B requires threshold > 2, while T5 B requires threshold <= 1.
    subj_products = []
    for subj, df in data.groupby('subject_id'):
        t4_df = df[t4_mask[df.index]]
        t5_df = df[t5_mask[df.index]]
        
        if len(t4_df) == 0 or len(t5_df) == 0:
            continue
            
        t4_b_rate = (t4_df['response'] == 1).mean()
        t5_b_rate = (t5_df['response'] == 1).mean()
        
        subj_products.append(t4_b_rate * t5_b_rate)
        
    if not subj_products:
        return 0.0
        
    return float(np.mean(subj_products))
```

**Observed (real) value:** 0.1258 (var=0.0107)
**Predicted under pi_18:** 0.1038 (var=0.0080)
**Predicted under pi_17:** 0.1951 (var=0.0100)

### Experiment 28
**Design**
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # In this design, all trials consist of Option X vs Option Y
    # Option X: [0, 1, 1, 1, 0, 0, 0] (Feature 1 is 0)
    # Option Y: [1, 0, 0, 0, 1, 1, 1] (Feature 1 is 1)
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    
    # Check if the subject chose Option X
    x_chosen = ((a_f1 == 0) & (data['response'] == 0)) | ((a_f1 == 1) & (data['response'] == 1))
    
    return float(x_chosen.mean())
```

**Observed (real) value:** 0.8029 (var=0.0127)
**Predicted under pi_18:** 0.1867 (var=0.0186)
**Predicted under pi_17:** 0.1658 (var=0.0183)

### Experiment 29
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify Trial 2 where A = [1, 1, 0, 0, 0]
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    # Identify Trial 6 where A = [1, 0, 1, 0, 0]
    is_t6 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_t2 = float(np.mean(data.loc[is_t2, 'response'] == 0)) if is_t2.sum() > 0 else 0.0
    p_a_t6 = float(np.mean(data.loc[is_t6, 'response'] == 0)) if is_t6.sum() > 0 else 0.0
    
    # Return an indicator for the specific choice pattern (T2=A and T6=B)
    return 1.0 if (p_a_t2 > 0.5 and p_a_t6 < 0.5) else 0.0
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_18:** 0.0000 (var=0.0564)
**Predicted under pi_17:** 0.0000 (var=0.0384)

### Experiment 30
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Extract features to isolate Trials 1 and 2
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    a_f2 = data['option_a_ratings'].apply(lambda x: x[1])
    a_f4 = data['option_a_ratings'].apply(lambda x: x[3])
    
    # Trial 1: a=[1, 0, 0, 1, 1, 1], b=[0, 1, 1, 0, 0, 0]
    # Trial 2: a=[0, 1, 1, 0, 0, 0], b=[1, 0, 0, 1, 1, 1]
    # In both cases, feature 1 matches feature 4 for option A, and differs from feature 2.
    mask = (a_f1 == a_f4) & (a_f1 != a_f2) & (a_f1 != b_f1)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    # Calculate if the choice aligns with the first (most valid) feature
    a_is_better_on_f1 = subset['option_a_ratings'].apply(lambda x: x[0]) > subset['option_b_ratings'].apply(lambda x: x[0])
    b_is_better_on_f1 = subset['option_a_ratings'].apply(lambda x: x[0]) < subset['option_b_ratings'].apply(lambda x: x[0])
    
    chose_a = subset['response'] == 0
    chose_b = subset['response'] == 1
    
    aligned = (a_is_better_on_f1 & chose_a) | (b_is_better_on_f1 & chose_b)
    
    return float(aligned.mean())
```

**Observed (real) value:** 0.4938 (var=0.0019)
**Predicted under pi_18:** 0.5950 (var=0.1057)
**Predicted under pi_17:** 0.8500 (var=0.0141)

### Experiment 31
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify trial 1 where A=[0, 1, 1, 0, 0] and B=[1, 0, 0, 1, 1]
    # We convert the ratings to tuples to allow for robust comparison
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    if not is_trial_1.any():
        return 0.0
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    # The Competing Theory structurally forces a choice of B (response == 1)
    # either by early stopping at feature 1 or by negative final tally.
    # The Advocated Theory allows Option A to win due to magnitude weighting.
    return float((data.loc[is_trial_1, 'response'] == 0).mean())

```

**Observed (real) value:** 0.1317 (var=0.0105)
**Predicted under pi_18:** 0.3742 (var=0.1054)
**Predicted under pi_17:** 0.1450 (var=0.0083)

### Experiment 32
**Design**
  A=[0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_a = (0, 1, 1, 0, 1, 0, 0)
    t1_b = (1, 0, 0, 1, 0, 1, 1)
    t2_a = (1, 0, 0, 1, 0, 1, 1)
    t2_b = (0, 1, 1, 0, 1, 0, 0)
    
    is_t1 = (a_tuples == t1_a) & (b_tuples == t1_b)
    is_t2 = (a_tuples == t2_a) & (b_tuples == t2_b)
    
    mask = is_t1 | is_t2
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    is_t1_sub = is_t1[mask]
    
    # Advocated choice: B (1) on T1, A (0) on T2
    advocated_choice = is_t1_sub.astype(int)
    
    return float((subset['response'] == advocated_choice).mean())
```

**Observed (real) value:** 0.1700 (var=0.0106)
**Predicted under pi_18:** 0.5938 (var=0.0498)
**Predicted under pi_17:** 0.8456 (var=0.0117)

### Experiment 33
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1))
    if not is_trial_1.any():
        return 0.0
    return float(data.loc[is_trial_1, 'response'].mean())
```

**Observed (real) value:** 0.1411 (var=0.0141)
**Predicted under pi_18:** 0.5463 (var=0.0944)
**Predicted under pi_17:** 0.1526 (var=0.0181)

### Experiment 34
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.9, 0.85, 0.8, 0.75])
    
    val_diffs = []
    chose_ttbs = []
    
    for idx, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        diff = a - b
        ttb_winner = -1
        for i in range(len(diff)):
            if diff[i] > 0:
                ttb_winner = 0
                break
            elif diff[i] < 0:
                ttb_winner = 1
                break
                
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        tally_diff = sum_a - sum_b
        tally_winner = 0 if tally_diff > 0 else (1 if tally_diff < 0 else -1)
        
        # Filter for critical trials where TTB and Tally conflict by exactly 1
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner and abs(tally_diff) == 1:
            chose_ttb = 1 if row['response'] == ttb_winner else 0
            
            v_a = np.sum(val * a)
            v_b = np.sum(val * b)
            
            v_diff = (v_a - v_b) if ttb_winner == 0 else (v_b - v_a)
                
            val_diffs.append(v_diff)
            chose_ttbs.append(chose_ttb)
            
    if len(val_diffs) < 2:
        return 0.0
        
    cov_matrix = np.cov(val_diffs, chose_ttbs)
    return float(cov_matrix[0, 1] * 100)
```

**Observed (real) value:** -1.2915 (var=0.2048)
**Predicted under pi_18:** 0.8139 (var=1.0233)
**Predicted under pi_17:** 0.0133 (var=0.1724)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both theories demonstrate strengths in their self-proposed experiments but fail to generalize across the broader dataset. THEORY 1 (pi_18) accurately captures the shift in choice probabilities in Experiment 1 (observed 0.77, predicted 0.63) but drastically overestimates the variance in Experiment 2 (observed 0.04, predicted 0.27). Conversely, THEORY 2 (pi_17) captures the stability in Experiment 2 (observed 0.04, predicted 0.07) but completely misses the dynamic shifts in Experiment 1 (predicted 0.32). Furthermore, neither theory accurately predicts the outcomes of several other experiments (e.g., Exp 6, 7, 15, 20, 24). This suggests that while trial-by-trial strategy adaptation is occurring, it is likely not driven strictly by Jaccard similarity (Theory 1) nor is it fixed at the environment level based on validity dispersion (Theory 2).",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory that relies on an evidence accumulation framework (e.g., a discrete Sequential Sampling or Decision Field Theory model) or a trial-level difficulty/conflict metric. Instead of switching strategies based on overall environment dispersion or simple feature overlap, the new theory should evaluate the strength of evidence for each option dynamically. For example, if the most valid features provide highly conflicting evidence, the decision process might naturally integrate more features (mimicking a shift towards WADD or Tallying), whereas decisive early evidence leads to a TTB-like quick termination. This approach can explain both the sensitivity to feature differences in Exp 1 and the stability in Exp 2 without relying on rigid heuristic mixtures."
}
```

## Usage

```json
{
  "prompt_token_count": 25366,
  "candidates_token_count": 396,
  "total_token_count": 26668
}
```
