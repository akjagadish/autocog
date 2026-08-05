# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take-The-Best (TTB) heuristic: People make decisions between multi-attribute options by ranking features according to their validities and comparing the options lexicographically. The decision-maker examines the feature with the highest validity first; if one option is strictly better on that feature, it is chosen immediately and no further features are considered. If there is a tie, they move to the second most valid feature, and so on. This non-compensatory, one-reason decision making process implies that a single highly valid cue can completely determine the choice, overriding any number of smaller advantages on less valid cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    choice = None
    for idx in order:
        if a[idx] > b[idx]:
            choice = 0
            break
        elif b[idx] > a[idx]:
            choice = 1
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = 2
    
    if choice is None:
        p_core = np.ones(n_opts) / n_opts
    else:
        p_core = np.zeros(n_opts)
        p_core[choice] = 1.0
        
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## THEORY 2 — pi_6
Mixture of Strategies (TTB and WADD): Decision makers are cognitively flexible and probabilistically alternate between a fast-and-frugal non-compensatory heuristic (Take-The-Best) and a compensatory, optimal evidence-integration strategy (Weighted Additive). On any given decision, an individual either relies solely on the most valid discriminating cue (TTB) or integrates all available cues weighted by their Bayesian log-odds (WADD). This hybrid approach explains both the strict adherence to high-validity cues on some trials and the compensatory influence of multiple weaker cues on others.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Strategy ---
    order = np.argsort(validities)[::-1]
    ttb_choice = None
    for idx in order:
        if a[idx] > b[idx]:
            ttb_choice = 0
            break
        elif b[idx] > a[idx]:
            ttb_choice = 1
            break
            
    if ttb_choice is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        p_ttb = np.zeros(2)
        p_ttb[ttb_choice] = 1.0
        
    # --- Weighted Additive (WADD) Strategy ---
    v_clipped = np.clip(validities, 1e-5, 1.0 - 1e-5)
    w = np.log(v_clipped / (1.0 - v_clipped))
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    scores = np.array([score_a, score_b])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # --- Probabilistic Mixture ---
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_wadd
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.64, 0.73, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 5: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 6: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 8: A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 9: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 12: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 13: A=[0, 1, 1, 1]  B=[1, 0, 0, 1]
  trial 14: A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 15: A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 16: A=[0, 0, 1, 1]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Given the specific validities, the Weighted Additive (WADD) strategy is actually strictly lexicographic and agrees with the Take-The-Best (TTB) strategy's choice on all trials. However, the two theories differ in their predicted choice probabilities. TTB predicts a constant lapse rate across all trials. The Mixture model (which includes WADD) predicts higher confidence (and thus fewer 'errors' relative to TTB) on trials where the most valid feature (Feature 0) discriminates, compared to trials where Feature 0 is tied and the decision relies on weaker features. This metric calculates the difference in the proportion of TTB-consistent choices between trials where Feature 0 discriminates and trials where it is tied. TTB predicts this difference to be 0, while the Mixture model predicts it to be strictly positive.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    order = [0, 2, 1, 3]
    
    def ttb_correct(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for idx in order:
            if a[idx] > b[idx]:
                return 1.0 if row['response'] == 0 else 0.0
            elif b[idx] > a[idx]:
                return 1.0 if row['response'] == 1 else 0.0
        return np.nan

    data['ttb_correct'] = data.apply(ttb_correct, axis=1)
    data['f0_diff'] = data.apply(lambda row: row['option_a_ratings'][0] != row['option_b_ratings'][0], axis=1)
    
    high_diff = data[data['f0_diff'] == True]['ttb_correct'].mean()
    low_diff = data[data['f0_diff'] == False]['ttb_correct'].mean()
    
    if pd.isna(high_diff): high_diff = 0.0
    if pd.isna(low_diff): low_diff = 0.0
    
    return float(high_diff - low_diff)

### RESULTS
- Predicted under pi_3 (simulated): 0.0088 (var=0.0048)
- Predicted under pi_6 (simulated): 0.0204 (var=0.0070)
- Observed on real data: 0.0041 (var=0.0037)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=4):** [0.95, 0.73, 0.56, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 3: A=[1, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 9: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 1, 0]  B=[1, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 13: A=[1, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 14: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 15: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Under the pure Take-The-Best (TTB) theory, the probability of choosing the TTB-favored option is constant across all trials where TTB discriminates, regardless of the compensatory evidence. Under the Mixture of TTB and Weighted Additive (WADD) theory, the compensatory WADD strategy will have varying confidence depending on the absolute difference in log-odds scores between the two options. By splitting trials into 'High WADD difference' and 'Low WADD difference' and comparing the proportion of TTB-consistent choices, pure TTB predicts a difference of near 0, whereas the Mixture theory predicts a strictly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.95, 0.73, 0.56, 0.55])
    v_clipped = np.clip(validities, 1e-5, 1.0 - 1e-5)
    w = np.log(v_clipped / (1.0 - v_clipped))
    
    high_diff_correct = []
    low_diff_correct = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_choice = None
        for idx in range(4):
            if a[idx] > b[idx]:
                ttb_choice = 0
                break
            elif b[idx] > a[idx]:
                ttb_choice = 1
                break
        
        if ttb_choice is None:
            continue
            
        score_a = np.sum(a * w)
        score_b = np.sum(b * w)
        diff = abs(score_a - score_b)
        
        is_correct = (row['response'] == ttb_choice)
        
        if diff > 2.0:
            high_diff_correct.append(is_correct)
        elif diff < 1.0:
            low_diff_correct.append(is_correct)
            
    if len(high_diff_correct) == 0 or len(low_diff_correct) == 0:
        return 0.0
        
    return float(np.mean(high_diff_correct) - np.mean(low_diff_correct))

### RESULTS
- Predicted under pi_3 (simulated): -0.0074 (var=0.0062)
- Predicted under pi_6 (simulated): 0.0502 (var=0.0094)
- Observed on real data: 0.0153 (var=0.0022)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.82, 0.84, 0.55])
    
    match_count = 0
    tie_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Identify trials where Tallying predicts a tie
        if a_wins == b_wins and a_wins > 0:
            tie_count += 1
            score_a = np.sum(a * val)
            score_b = np.sum(b * val)
            
            # Check if response aligns with WADD's strict preference
            if score_a > score_b and row['response'] == 0:
                match_count += 1
            elif score_b > score_a and row['response'] == 1:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
        
    return float(match_count / tie_count)
```

**Observed (real) value:** 0.8422 (var=0.0120)
**Predicted under pi_3:** 0.8789 (var=0.0091)
**Predicted under pi_6:** 0.8467 (var=0.0148)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 4: A=[0, 0, 1, 1], B=[0, 1, 1, 0]
    # A wins on feature 4 (validity 0.55), B wins on feature 2 (validity 0.78)
    mask4 = (a_str == '0011') & (b_str == '0110')
    
    # Trial 9: A=[1, 0, 1, 0], B=[0, 0, 1, 1]
    # A wins on feature 1 (validity 0.95), B wins on feature 4 (validity 0.55)
    mask9 = (a_str == '1010') & (b_str == '0011')
    
    p_b_4 = data.loc[mask4, 'response'].mean()
    p_b_9 = data.loc[mask9, 'response'].mean()
    
    if pd.isna(p_b_4):
        p_b_4 = 0.5
    if pd.isna(p_b_9):
        p_b_9 = 0.5
        
    return float(p_b_4 - p_b_9)
```

**Observed (real) value:** 0.7000 (var=0.0889)
**Predicted under pi_3:** 0.7567 (var=0.0413)
**Predicted under pi_6:** 0.7367 (var=0.0701)

### Experiment 5
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.95, 0.77, 0.8, 0.55])
    order = np.argsort(validities)[::-1]

    match_count = 0
    total = 0

    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']

        ttb_choice = None
        for idx in order:
            if a[idx] > b[idx]:
                ttb_choice = 0
                break
            elif b[idx] > a[idx]:
                ttb_choice = 1
                break
        
        if ttb_choice is not None:
            if resp == ttb_choice:
                match_count += 1
            total += 1

    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.8583 (var=0.0121)
**Predicted under pi_3:** 0.8779 (var=0.0049)
**Predicted under pi_6:** 0.8727 (var=0.0069)

### Experiment 6
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 11 is the critical compensatory trial:
    # A = [1, 1, 0, 0], B = [1, 0, 1, 1]
    # Here, TTB looks at feature 0 (tied) and then feature 1, choosing the option with feature 1.
    # WADD weighs feature 1 (validity 0.84) against features 2 and 3 (validities 0.64 + 0.55 = 1.19),
    # typically favoring the option with features 2 and 3.
    mask_straight = (a_tuples == (1, 1, 0, 0)) & (b_tuples == (1, 0, 1, 1))
    mask_swapped = (a_tuples == (1, 0, 1, 1)) & (b_tuples == (1, 1, 0, 0))
    
    mask = mask_straight | mask_swapped
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    is_straight = mask_straight[mask]
    
    # In straight, TTB choice is A (response 0). In swapped, TTB choice is B (response 1).
    ttb_choice = np.where(is_straight, 0, 1)
    
    return float(np.mean(subset['response'] == ttb_choice))

```

**Observed (real) value:** 0.8133 (var=0.0229)
**Predicted under pi_3:** 0.8900 (var=0.0229)
**Predicted under pi_6:** 0.8833 (var=0.0292)

### Experiment 7
**Design**
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def analyze_row(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb = None
        for idx in [0, 1, 2, 3]:
            if a[idx] > b[idx]:
                ttb = 0
                break
            elif b[idx] > a[idx]:
                ttb = 1
                break
        if ttb is None:
            ttb = 0
            
        diff = [a[i] - b[i] for i in range(4)]
        signs = [d for d in diff if d != 0]
        unanimous = all(s > 0 for s in signs) or all(s < 0 for s in signs)
        
        return pd.Series({'match': resp == ttb, 'unanimous': unanimous})
        
    res = data.apply(analyze_row, axis=1)
    
    mean_unanimous = res[res['unanimous']]['match'].mean()
    mean_conflict = res[~res['unanimous']]['match'].mean()
    
    if pd.isna(mean_unanimous) or pd.isna(mean_conflict):
        return 0.0
        
    return float(mean_unanimous - mean_conflict)
```

**Observed (real) value:** 0.0176 (var=0.0070)
**Predicted under pi_3:** 0.0162 (var=0.0047)
**Predicted under pi_6:** -0.0267 (var=0.0037)

### Experiment 8
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # Cue 0 has the highest validity (0.95), Cue 2 has the second highest (0.88)
    cue0_diff = a_ratings[:, 0] - b_ratings[:, 0]
    cue2_diff = a_ratings[:, 2] - b_ratings[:, 2]
    
    # Find trials where Cue 0 and Cue 2 disagree
    disagree = (cue0_diff != 0) & (cue2_diff != 0) & (cue0_diff != cue2_diff)
    
    if np.sum(disagree) == 0:
        return 0.5
        
    # TTB predicts the option favored by Cue 0 (the highest validity cue)
    ttb_pred = (cue0_diff[disagree] < 0).astype(int)
    resp = data['response'].values[disagree]
    
    return float(np.mean(ttb_pred == resp))
```

**Observed (real) value:** 0.8307 (var=0.0176)
**Predicted under pi_3:** 0.8720 (var=0.0069)
**Predicted under pi_6:** 0.8647 (var=0.0107)

### Experiment 9
**Design**
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    small_margins = [
        ((0, 1, 0, 0), (0, 1, 0, 1)),
        ((0, 0, 0, 0), (0, 1, 0, 1)),
        ((0, 0, 0, 0), (0, 1, 0, 0))
    ]
    
    large_margins = [
        ((1, 1, 1, 1), (0, 0, 0, 0)),
        ((1, 0, 1, 1), (0, 0, 0, 1)),
        ((1, 1, 1, 0), (0, 0, 1, 1))
    ]
    
    small_correct = 0
    small_total = 0
    large_correct = 0
    large_total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Determine TTB choice based on validity order: F0, F2, F1, F3
        ttb_choice = None
        for idx in [0, 2, 1, 3]:
            if a[idx] > b[idx]:
                ttb_choice = 0
                break
            elif b[idx] > a[idx]:
                ttb_choice = 1
                break
        
        if ttb_choice is None:
            continue
            
        pair = (a, b)
        pair_rev = (b, a)
        
        if pair in small_margins or pair_rev in small_margins:
            if resp == ttb_choice:
                small_correct += 1
            small_total += 1
        elif pair in large_margins or pair_rev in large_margins:
            if resp == ttb_choice:
                large_correct += 1
            large_total += 1
            
    prop_small = small_correct / small_total if small_total > 0 else 0.5
    prop_large = large_correct / large_total if large_total > 0 else 0.5
    
    return prop_large - prop_small
```

**Observed (real) value:** -0.0244 (var=0.0136)
**Predicted under pi_3:** -0.0078 (var=0.0081)
**Predicted under pi_6:** 0.0133 (var=0.0113)

### Experiment 10
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trials where the difference in WADD log-odds scores is very small (<= 0.85)
    low_diff = {
        ('0110', '1000'): 1,
        ('0110', '1001'): 1,
        ('0010', '0001'): 0,
        ('1101', '1110'): 1,
        ('0011', '0010'): 0,
        ('0101', '0100'): 0,
        ('1001', '1000'): 0
    }
    
    # Trials where the difference in WADD log-odds scores is very large (>= 2.94)
    high_diff = {
        ('0101', '1110'): 1,
        ('0010', '1010'): 1,
        ('0110', '1110'): 1
    }
    
    acc_low = []
    acc_high = []
    
    for _, row in data.iterrows():
        pair = (row['A_str'], row['B_str'])
        if pair in low_diff:
            acc_low.append(1 if row['response'] == low_diff[pair] else 0)
        elif pair in high_diff:
            acc_high.append(1 if row['response'] == high_diff[pair] else 0)
            
    mean_low = sum(acc_low) / len(acc_low) if acc_low else 0.5
    mean_high = sum(acc_high) / len(acc_high) if acc_high else 0.5
    
    return mean_high - mean_low

```

**Observed (real) value:** 0.0216 (var=0.0046)
**Predicted under pi_3:** 0.0181 (var=0.0101)
**Predicted under pi_6:** 0.0083 (var=0.0085)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 1 (Take-The-Best) consistently predicts the observed data quite well, particularly on metrics that isolate the effect of compensatory evidence (e.g., Exps 1, 7, 9, 10). The observed data shows very weak or near-zero compensatory effects, aligning closely with pure TTB's predictions. Theory 2 (Mixture of TTB and WADD) consistently overpredicts the influence of compensatory evidence, producing larger differences than observed (Exp 2) or even predicting effects in the wrong direction (Exp 7). Since the mixture model struggles to capture the strictly non-compensatory nature of the behavior without reducing to TTB, it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory to replace the Mixture model. A strong classic alternative in this domain is 'Tallying' (Unit-Weight Linear Model), where decision-makers ignore cue validities entirely and simply count the number of positive features favoring each option, choosing the one with the higher tally (and guessing in case of a tie). Alternatively, consider a 'Threshold Lexicographic' theory where decision makers only consider features whose validity exceeds a certain threshold. This will provide a structurally distinct competitor to pure TTB that doesn't rely on complex Bayesian log-odds integration."
}
```

## Usage

```json
{
  "prompt_token_count": 12144,
  "candidates_token_count": 309,
  "total_token_count": 13769
}
```
