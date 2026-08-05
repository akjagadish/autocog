# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take-The-Best (TTB) heuristic: People make decisions by considering features one at a time, ordered by their validity. They stop at the first feature that discriminates between the options and choose the option with the higher value on that feature. If no features discriminate, they guess. This is a non-compensatory, lexicographic process.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## THEORY 2 — pi_5
Mixture of Take-The-Best (TTB) and Tallying: Decision-makers primarily rely on the deterministic, non-compensatory Take-The-Best heuristic, sequentially checking features ordered by validity and stopping at the first discriminating feature. A small subset of decisions relies on a compensatory Tallying strategy. By restricting the tallying mixture weight to a low range, the model maintains overwhelming TTB dominance, capturing high TTB adherence and the lack of a strong compensatory effect from the number of favoring features, while still allowing structured deviations from pure TTB.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    epsilon = float(parameters["epsilon"])
    
    # TTB Strategy
    order = np.argsort(validities)[::-1]
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying Strategy
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tal = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tal = np.array([0.0, 1.0])
    else:
        p_tal = np.array([0.5, 0.5])
        
    # Mixture of TTB and Tallying
    p_core = (1.0 - w_tally) * p_ttb + w_tally * p_tal
    
    # Uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Rationale:** To strongly dissociate pure Take-The-Best (TTB) from a mixture of TTB and Tallying, we use a 5-feature design and manipulate the alignment between the single best discriminating feature and the total number of favoring features. Pure TTB predicts a constant choice probability for the option favored by the highest-validity discriminating feature, regardless of how many total features favor it. The competing mixture model, however, predicts a slightly lower choice probability when Tallying contradicts TTB, and a slightly higher probability when Tallying aligns with TTB. By including trials where TTB and Tallying strongly conflict (e.g., one option wins on the best feature, the other wins on the remaining four), trials where they agree, and trials where Tallying is tied, we can detect the subtle graded choice probabilities predicted by the mixture model that pure TTB cannot capture.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the difference in the rate of choosing the Take-The-Best (TTB) favored option between 'Agreement' trials (where TTB and Tallying favor the same option) and 'Conflict' trials (where TTB and Tallying favor opposite options). Under the pure TTB model, the rate of choosing the TTB option should be identical across both trial types (any deviations are purely due to a uniform lapse rate epsilon), so the expected difference is 0. Under the Mixture model, the presence of the Tallying strategy increases the choice probability of the TTB option in Agreement trials and decreases it in Conflict trials, leading to a strictly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_agreement = []
    ttb_match_conflict = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # TTB winner
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        # Tallying winner
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a > sum_b:
            tal_winner = 0
        elif sum_b > sum_a:
            tal_winner = 1
        else:
            tal_winner = -1
            
        if ttb_winner is not None:
            chose_ttb = 1 if resp == ttb_winner else 0
            if tal_winner == ttb_winner:
                ttb_match_agreement.append(chose_ttb)
            elif tal_winner != -1 and tal_winner != ttb_winner:
                ttb_match_conflict.append(chose_ttb)
                
    if len(ttb_match_agreement) == 0 or len(ttb_match_conflict) == 0:
        return 0.0
        
    return float(np.mean(ttb_match_agreement) - np.mean(ttb_match_conflict))

### RESULTS
- Predicted under pi_3 (simulated): -0.0046 (var=0.0060)
- Predicted under pi_5 (simulated): 0.0692 (var=0.0043)
- Observed on real data: 0.0367 (var=0.0120)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate pure Take-The-Best (TTB) from a Mixture of TTB and Tallying, we exploit the fact that pure TTB's predicted choice probability depends exclusively on the single highest-validity discriminating feature, remaining completely invariant to the total number of features favoring each option. The advocated mixture model, however, predicts that the choice probability for the TTB-favored option will systematically vary depending on whether Tallying (the unweighted sum of features) opposes, ties, or aligns with the TTB choice. We construct a 5-feature design with a clear validity hierarchy. We include subsets of trials where Option A always wins on the most valid discriminating feature (e.g., Feature 1 or Feature 2 when Feature 1 is tied), but the total number of features favoring Option A ranges from a minority (strong Tallying opposition) to a majority (strong Tallying alignment). Pure TTB predicts a flat choice probability across these subsets, whereas the Mixture model predicts a graded, monotonically increasing probability for Option A as Tallying support increases.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
By design, Take-The-Best (TTB) favors Option A on every single trial. Therefore, a pure TTB model predicts that the probability of choosing Option A should be constant across all trials (subject only to a uniform lapse rate). However, the Mixture of TTB and Tallying model predicts that the probability of choosing Option A will vary systematically depending on whether the Tallying strategy (unweighted sum of features) agrees with or contradicts TTB. This metric calculates the difference in the rate of choosing Option A between trials where Tallying favors A and trials where Tallying opposes A. A pure TTB model will yield a value around 0, whereas the Mixture model will yield a robustly positive value.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying favors Option A vs. opposes Option A
    # (Note: TTB always favors Option A in this specific experimental design)
    tally_favors_a = sum_a > sum_b
    tally_opposes_a = sum_a < sum_b
    
    # Choice A is represented by response == 0
    choice_a = (data['response'] == 0).astype(float)
    
    # Calculate the proportion of times Option A was chosen in each subset
    mean_favors = choice_a[tally_favors_a].mean()
    mean_opposes = choice_a[tally_opposes_a].mean()
    
    if pd.isna(mean_favors) or pd.isna(mean_opposes):
        return 0.0
        
    return float(mean_favors - mean_opposes)

### RESULTS
- Predicted under pi_3 (simulated): 0.0050 (var=0.0056)
- Predicted under pi_5 (simulated): 0.0817 (var=0.0059)
- Observed on real data: -0.0056 (var=0.0086)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    a_wadd = np.sum(a_ratings * validities, axis=1)
    b_wadd = np.sum(b_ratings * validities, axis=1)
    
    conflict_mask = (a_wins > b_wins) & (b_wadd > a_wadd)
    
    if np.sum(conflict_mask) == 0:
        return 0.5
        
    return float(np.mean(data['response'].values[conflict_mask]))
```

**Observed (real) value:** 0.8844 (var=0.0081)
**Predicted under pi_3:** 0.8728 (var=0.0081)
**Predicted under pi_5:** 0.8761 (var=0.0034)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    is_a_heavy = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_b_heavy = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    is_a_many = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    is_b_many = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    trial_1 = is_a_heavy & is_b_many
    trial_2 = is_a_many & is_b_heavy
    
    critical_trials = trial_1 | trial_2
    
    if not critical_trials.any():
        return 0.5
        
    heavy_chosen = (trial_1 & (data['response'] == 0)) | (trial_2 & (data['response'] == 1))
    
    return float(heavy_chosen[critical_trials].mean())
```

**Observed (real) value:** 0.8533 (var=0.0160)
**Predicted under pi_3:** 0.8750 (var=0.0083)
**Predicted under pi_5:** 0.8858 (var=0.0058)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    v = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    a_v = np.sum(a_ratings * v, axis=1)
    b_v = np.sum(b_ratings * v, axis=1)
    
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    
    wadd_prefers_a = a_v > b_v
    wadd_prefers_b = b_v > a_v
    
    conflict_a = tally_prefers_a & wadd_prefers_b
    conflict_b = tally_prefers_b & wadd_prefers_a
    
    conflict_mask = conflict_a | conflict_b
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    
    tally_aligned = np.zeros(len(data), dtype=bool)
    tally_aligned[conflict_a & (responses == 0)] = True
    tally_aligned[conflict_b & (responses == 1)] = True
    
    return float(np.mean(tally_aligned[conflict_mask]))
```

**Observed (real) value:** 0.1500 (var=0.0150)
**Predicted under pi_3:** 0.1625 (var=0.0258)
**Predicted under pi_5:** 0.0800 (var=0.0092)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def f0_chosen(row):
        a0 = row['option_a_ratings'][0]
        b0 = row['option_b_ratings'][0]
        if a0 == b0:
            return None
        return a0 if row['response'] == 0 else b0
        
    f0_vals = data.apply(f0_chosen, axis=1).dropna()
    if len(f0_vals) == 0:
        return 0.5
    return float(f0_vals.mean())
```

**Observed (real) value:** 0.9033 (var=0.0065)
**Predicted under pi_3:** 0.8964 (var=0.0042)
**Predicted under pi_5:** 0.9244 (var=0.0021)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        if pred is not None:
            matches.append(1 if r == pred else 0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.8671 (var=0.0115)
**Predicted under pi_3:** 0.8735 (var=0.0058)
**Predicted under pi_5:** 0.9029 (var=0.0020)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # TTB winner: based on the first feature that discriminates
    diff = a_ratings - b_ratings
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    has_diff = np.any(diff != 0, axis=1)
    
    # Tally winner: based on simple sum of features
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    tally_winner = np.where(sum_a > sum_b, 0, np.where(sum_b > sum_a, 1, -1))
    
    # Identify conflict trials where TTB and Tallying disagree
    conflict = has_diff & (tally_winner != -1) & (ttb_winner != tally_winner)
    
    if not np.any(conflict):
        return 0.0
        
    responses = data['response'].values
    tally_chosen = (responses[conflict] == tally_winner[conflict]).sum()
    
    return float(tally_chosen / np.sum(conflict))
```

**Observed (real) value:** 0.1378 (var=0.0101)
**Predicted under pi_3:** 0.1311 (var=0.0080)
**Predicted under pi_5:** 0.1097 (var=0.0029)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # TTB strictly predicts Option A for Trials 1-10, and Option B for Trials 11-12.
    # We define adherence to TTB as choosing the option TTB predicts.
    ttb_preds = {
        '1000001111': 0, # T1: A
        '1100010111': 0, # T2: A
        '0100000111': 0, # T3: A
        '0110001011': 0, # T4: A
        '0010000011': 0, # T5: A
        '1010001011': 0, # T6: A
        '1001001101': 0, # T7: A
        '1110000011': 0, # T8: A
        '1011001001': 0, # T9: A
        '0111000001': 0, # T10: A
        '0011111000': 1, # T11: B
        '0110110010': 1  # T12: B
    }
    
    data['trial_id'] = data.apply(
        lambda row: ''.join([str(int(v)) for v in row['option_a_ratings']]) + 
                    ''.join([str(int(v)) for v in row['option_b_ratings']]), 
        axis=1
    )
    
    data['ttb_match'] = (data['response'] == data['trial_id'].map(ttb_preds)).astype(float)
    
    # Calculate overall TTB adherence for the subject across all 96 trials
    overall_adherence = data['ttb_match'].mean()
    
    # Return 1.0 if adherence is above 85%, else 0.0. 
    # This heavily penalizes the high uniform lapse rate of TTB (epsilon up to 0.5),
    # while capturing the fact that >95% of PTTB subjects (who typically have high gamma 
    # and epsilon <= 0.25) will easily clear this threshold.
    return 1.0 if overall_adherence > 0.85 else 0.0

```

**Observed (real) value:** 1.0000 (var=0.2400)
**Predicted under pi_3:** 1.0000 (var=0.1924)
**Predicted under pi_5:** 1.0000 (var=0.0736)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    # TTB decides based on the first feature where A and B differ
    diff = A != B
    first_diff_idx = np.argmax(diff, axis=1)
    
    row_indices = np.arange(len(data))
    a_vals = A[row_indices, first_diff_idx]
    b_vals = B[row_indices, first_diff_idx]
    
    # TTB choice: 0 for A, 1 for B
    ttb_choices = (b_vals > a_vals).astype(int)
    
    # Match between subject response and TTB choice
    responses = data['response'].values
    ttb_match = (responses == ttb_choices).astype(float)
    
    # Count how many features favor the TTB choice
    d_a = (A > B).astype(int)
    d_b = (B > A).astype(int)
    
    ttb_features_count = np.where(ttb_choices == 0, d_a.sum(axis=1), d_b.sum(axis=1))
    
    # Compare trials where TTB is supported by 3 features vs 1 feature
    mask_3 = (ttb_features_count == 3)
    mask_1 = (ttb_features_count == 1)
    
    if np.sum(mask_3) == 0 or np.sum(mask_1) == 0:
        return 0.0
        
    return float(np.mean(ttb_match[mask_3]) - np.mean(ttb_match[mask_1]))
```

**Observed (real) value:** -0.0033 (var=0.0076)
**Predicted under pi_3:** 0.0017 (var=0.0075)
**Predicted under pi_5:** 0.0794 (var=0.0045)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of experiments (e.g., Experiments 2, 5, 6, 8, 10), the observed data strongly aligns with the predictions of Theory 1 (pure Take-The-Best). Theory 1 correctly predicts that choice probabilities are driven entirely by the highest-validity discriminating feature and remain invariant to the total number of favoring features (Tallying). Theory 2 (Mixture of TTB and Tallying) consistently overpredicts the compensatory influence of Tallying, yielding expected metric values that systematically deviate from the near-zero values observed in reality. The minor positive effect in Experiment 1 is not consistently replicated in other designs, suggesting that any deviations from pure TTB are likely not due to a compensatory Tallying mixture.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory that maintains a strictly non-compensatory, lexicographic core but introduces structured noise into the *ordering* of the features. For example, a 'Probabilistic Take-The-Best' model where decision-makers occasionally misorder the feature validities (e.g., due to memory constraints or subjective weighting) and thus apply TTB based on a suboptimal feature hierarchy. This can capture slight deviations from pure TTB without incorrectly predicting the systematic compensatory effects of Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 10651,
  "candidates_token_count": 300,
  "total_token_count": 12050
}
```
