# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Pure Tallying Heuristic: Decision-makers evaluate options by completely ignoring the provided validities or weights of the cues. Instead, they simply count the number of features on which one option is strictly better than the other. The option with the higher total count of winning features is chosen. This unweighted counting mechanism explains the robust preference for options with more winning features, even if those features are of lower validity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins; completely ignore validities and magnitudes
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    scores = np.array([a_wins, b_wins], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Weighted Tallying heuristic: People evaluate options by comparing them feature-by-feature, but instead of just counting the number of winning features (as in regular Tallying), they weight each win by the feature's subjective validity or importance. This heuristic discards cardinal magnitudes (the size of the difference between feature values is ignored), making it robust to extreme outlier values that would skew an additive model. However, unlike unweighted Tallying, it incorporates the known validities of the cues, allowing more important features to break ties or even override a larger count of less important features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Weighted Tallying expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate strict wins for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Weight the wins by the validities
    score_a = np.sum(a_wins * v)
    score_b = np.sum(b_wins * v)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.7, 0.6, 0.5]

**rating_max:** 1

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** To cleanly dissociate Pure Tallying from Weighted Tallying, we use binary features to eliminate any magnitude-based effects, isolating the core difference: whether feature wins are counted equally or weighted by validity. We carefully select validities [0.95, 0.85, 0.70, 0.60, 0.50] such that the sum of the top two validities exactly equals the sum of the bottom three. This creates a critical trial where one option wins on the two most valid features and the other wins on the three least valid features. Pure Tallying strongly prefers the option with three wins, whereas Weighted Tallying is perfectly indifferent. We also include trials where both heuristics tie in terms of feature counts but Weighted Tallying has a strong preference due to validity differences.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric contrasts behavior on Trial 1 vs. Trial 2. In Trial 1, Option A has 2 wins on the most valid features, while Option B has 3 wins on the least valid features. Pure Tallying strictly prefers B (more wins), while Weighted Tallying is perfectly indifferent (validity sums are equal: 1.8 vs 1.8). In Trial 2, Option A has 2 wins on the most valid features, and Option B has 2 wins on the least valid features. Here, Pure Tallying is indifferent (equal wins), while Weighted Tallying strictly prefers A (higher validity sum). The metric computes P(Choose B in Trial 1) - P(Choose A in Trial 2). For Pure Tallying, this is roughly 1.0 - 0.5 = 0.5. For Weighted Tallying, this is roughly 0.5 - 1.0 = -0.5. This yields a large, stable difference in means between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def match_t1(row):
        a, b = row['option_a_ratings'], row['option_b_ratings']
        return sum(a) == 2 and sum(b) == 3
        
    def match_t2(row):
        a, b = row['option_a_ratings'], row['option_b_ratings']
        return sum(a) == 2 and sum(b) == 2 and a[0] == 1 and b[2] == 1
        
    is_t1 = data.apply(match_t1, axis=1)
    is_t2 = data.apply(match_t2, axis=1)
    
    t1_data = data[is_t1]
    t2_data = data[is_t2]
    
    if len(t1_data) == 0 or len(t2_data) == 0:
        return 0.0
        
    p_b_t1 = t1_data['response'].mean()
    p_a_t2 = 1.0 - t2_data['response'].mean()
    
    return float(p_b_t1 - p_a_t2)


### RESULTS
- Predicted under pi_6 (simulated): 0.3695 (var=0.0253)
- Predicted under pi_4 (simulated): -0.3305 (var=0.0274)
- Observed on real data: 0.0400 (var=0.9584)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.9, 0.85, 0.8, 0.6, 0.55, 0.5]

**rating_max:** 1

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]

**Rationale:** To cleanly dissociate Weighted Tallying from Pure Tallying, we use 6 binary features (rating_max=1) to eliminate any magnitude-based effects and focus entirely on how feature wins are counted. Pure Tallying counts all winning features equally, whereas Weighted Tallying weights each win by its validity. We select a set of validities [0.90, 0.85, 0.80, 0.60, 0.55, 0.50] that creates a clear separation between high-validity and low-validity features. The trials are designed to pit these mechanisms against each other: in some trials, one option wins on two high-validity features while the other wins on three low-validity features (Pure Tallying prefers the latter, Weighted Tallying prefers the former). We also include trials where both options have an equal number of winning features (a tie for Pure Tallying) but differ in the validities of those features, leading to a strong preference in Weighted Tallying.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option with exactly two winning features of highest validity (0.90 and 0.85) over an option with three winning features of lower validity (0.60, 0.55, and 0.50). Under Pure Tallying, subjects simply count the winning features (2 vs 3) and thus strongly prefer the latter option, yielding a metric close to 0. Under Weighted Tallying, subjects weight the wins by validity (1.75 vs 1.65), reversing the preference and yielding a metric close to 1. This provides a massive, stable discrimination between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where one of the options has the two highest-validity features (index 0 and 1) 
    # and nothing else. This corresponds to Trial 1: [1, 1, 0, 0, 0, 0] vs [0, 0, 0, 1, 1, 1].
    a_target = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1)
    b_target = data['option_b_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1)
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    # Check if the subject chose the target option
    target_chosen = (a_target & chose_a) | (b_target & chose_b)
    valid_trials = a_target | b_target
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float(target_chosen.sum() / valid_trials.sum())

### RESULTS
- Predicted under pi_6 (simulated): 0.1500 (var=0.0158)
- Predicted under pi_4 (simulated): 0.6350 (var=0.0226)
- Observed on real data: 0.5400 (var=0.2484)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 4, 4, 4]  B=[5, 3, 3, 3]
  A=[5, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[2, 5, 2, 2]  B=[3, 1, 3, 3]
  A=[4, 4, 1, 5]  B=[5, 5, 2, 0]
  A=[0, 0, 5, 5]  B=[1, 1, 4, 4]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        if a[0] == 1:
            matches.append(r == 0)
        elif a[0] == 5:
            matches.append(r == 1)
        elif a[0] == 2:
            matches.append(r == 1)
        elif a[0] == 4:
            matches.append(r == 1)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7250 (var=0.0281)
**Predicted under pi_6:** 0.8526 (var=0.0098)
**Predicted under pi_4:** 0.8592 (var=0.0096)

### Experiment 4
**Design**
  A=[5, 5, 5, 5]  B=[0, 6, 6, 6]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 5, 5, 4]
  A=[8, 4, 4, 4]  B=[5, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 2: A=[0, 4, 4, 4], B=[10, 0, 0, 0]
    # WADD prefers B (response=1) because 10*0.9 = 9.0 > 4*(0.8+0.6+0.5) = 7.6
    # Tallying prefers A (response=0) because A wins on 3 features vs B's 1
    
    # Identify trial 2 by checking the first element of option A
    is_trial_2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 4, 4, 4))
    trial_data = data[is_trial_2]
    
    if len(trial_data) == 0:
        return 0.5
        
    return float(trial_data['response'].mean())
```

**Observed (real) value:** 0.2600 (var=0.1924)
**Predicted under pi_6:** 0.1600 (var=0.0177)
**Predicted under pi_4:** 0.1537 (var=0.0208)

### Experiment 5
**Design**
  A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  A=[10, 10, 0, 0]  B=[0, 0, 1, 1]
  A=[5, 5, 5, 5]  B=[4, 4, 4, 8]
  A=[9, 9, 0, 0]  B=[10, 0, 1, 1]

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
        
        if a_wins != b_wins:
            tally_pref = 0 if a_wins > b_wins else 1
            matches.append(1 if row['response'] == tally_pref else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7440 (var=0.0257)
**Predicted under pi_6:** 0.8423 (var=0.0117)
**Predicted under pi_4:** 0.8658 (var=0.0049)

### Experiment 6
**Design**
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[8, 8, 8, 0]  B=[7, 7, 7, 10]
  A=[10, 0, 0, 0]  B=[7, 1, 1, 1]
  A=[10, 10, 0, 0]  B=[0, 0, 2, 2]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  A=[2, 2, 0, 0]  B=[0, 0, 10, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    eqw_predicts_a = a_sums > b_sums
    eqw_predicts_b = b_sums > a_sums
    
    match_a = eqw_predicts_a & (data['response'] == 0)
    match_b = eqw_predicts_b & (data['response'] == 1)
    
    valid_trials = eqw_predicts_a | eqw_predicts_b
    if not valid_trials.any():
        return 0.5
        
    return float((match_a | match_b).sum() / valid_trials.sum())
```

**Observed (real) value:** 0.3680 (var=0.0278)
**Predicted under pi_6:** 0.2895 (var=0.0063)
**Predicted under pi_4:** 0.2812 (var=0.0030)

### Experiment 7
**Design**
  A=[1, 1, 4, 4, 4]  B=[4, 4, 1, 1, 1]
  A=[5, 5, 2, 2, 2]  B=[2, 2, 4, 4, 4]
  A=[4, 2, 4, 2, 3]  B=[2, 4, 2, 4, 3]
  A=[2, 3, 4, 4, 2]  B=[4, 3, 2, 2, 4]
  A=[4, 2, 3, 3, 3]  B=[2, 4, 3, 3, 3]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Trial 1: A=[1, 1, 4, 4, 4], B=[4, 4, 1, 1, 1]
    # Tallying prefers A (3 vs 2). Weighted Tallying prefers B (1.7 vs 1.5).
    t1_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1 and x[2] == 4)
    
    # Trial 2: A=[5, 5, 2, 2, 2], B=[2, 2, 4, 4, 4]
    # Tallying prefers B (3 vs 2). Weighted Tallying prefers A (1.7 vs 1.5).
    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 5 and x[1] == 5 and x[2] == 2)
    
    # Proportion of times choosing B (response == 1) in Trial 1
    t1_data = data.loc[t1_mask, 'response']
    val1 = t1_data.mean() if len(t1_data) > 0 else 0.5
    
    # Proportion of times choosing A (response == 0) in Trial 2
    t2_data = data.loc[t2_mask, 'response']
    val2 = (1.0 - t2_data.mean()) if len(t2_data) > 0 else 0.5
    
    return float((val1 + val2) / 2.0)
```

**Observed (real) value:** 0.4200 (var=0.2436)
**Predicted under pi_6:** 0.1900 (var=0.0167)
**Predicted under pi_4:** 0.7405 (var=0.0148)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify Trial 1 where A=[1, 1, 0, 0, 0] and B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.5
    trial_1_data = data[is_trial_1]
    # Calculate the proportion of times Option A was chosen (response == 0)
    prob_a = (trial_1_data['response'] == 0).mean()
    return float(prob_a)
```

**Observed (real) value:** 0.2800 (var=0.2016)
**Predicted under pi_6:** 0.1926 (var=0.0315)
**Predicted under pi_4:** 0.7411 (var=0.0266)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # In all designed trials, Option A has fewer wins but on higher-ranked cues,
    # whereas Option B has more wins but on slightly lower-validity cues.
    # Weighted Tallying treats the clustered validities as nearly linear,
    # so the sum of multiple lower-ranked wins (Option B) easily outweighs Option A.
    # Soft TTB uses a power-law decay on rank, allowing the top-ranked cue(s) of Option A to dominate.
    # Thus, Soft TTB predicts choosing A (response == 0) much more frequently than Weighted Tallying.
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.3400 (var=0.0733)
**Predicted under pi_6:** 0.3008 (var=0.0070)
**Predicted under pi_4:** 0.1760 (var=0.0121)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Extract the first feature (highest validity) for options A and B
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Weighted Tallying favors the combination of multiple lower-ranked features
    # over the single highest-ranked feature in this specific experimental design.
    # Therefore, WT predicts choosing the option that LOSES on feature 1.
    # If A > B on feature 1, B loses on feature 1, so WT predicts 1 (Option B).
    # If A < B on feature 1, A loses on feature 1, so WT predicts 0 (Option A).
    wt_predicted_response = (a_f1 > b_f1).astype(int)
    
    # Calculate the proportion of trials where the subject's choice aligns with WT
    match = (data['response'] == wt_predicted_response)
    return float(match.mean())
```

**Observed (real) value:** 0.6080 (var=0.0943)
**Predicted under pi_6:** 0.7491 (var=0.0121)
**Predicted under pi_4:** 0.7714 (var=0.0087)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across multiple experiments, the observed metrics consistently fall between the predictions of Pure Tallying (pi_6) and Weighted Tallying (pi_4). For instance, in Experiment 1, pi_6 predicts 0.36, pi_4 predicts -0.33, and the observed value is 0.04. In Experiment 2, observed is 0.54 (between 0.15 and 0.63). In Experiment 7, observed is 0.42 (between 0.19 and 0.74). This systematic intermediacy strongly suggests that neither pure unweighted tallying nor linearly validity-weighted tallying captures the population behavior. Instead, the population likely consists of a mixture of strategies (e.g., some subjects using Take-The-Best and others using Pure Tallying, or individuals probabilistically switching between them), or a single strategy with non-linear/exponential weighting on cue rank (like Soft Take-The-Best) that produces intermediate choices.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Heuristic Strategy Mixture' or 'Soft Take-The-Best' theory. A mixture theory would posit that decision-makers probabilistically sample between a Pure Tallying strategy and a strict Take-The-Best (lexicographic) strategy on each trial (or across subjects). Alternatively, a Soft TTB theory would use rank-based exponential weights rather than the explicit validities, allowing the most valid cue to have strong but not absolute dominance. This will naturally capture the intermediate means observed across experiments better than Weighted Tallying, which uses the linear validities."
}
```

## Usage

```json
{
  "prompt_token_count": 8032,
  "candidates_token_count": 374,
  "total_token_count": 9595
}
```
