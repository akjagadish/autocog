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
Subjects use a Tallying (Equal Weights) heuristic, where they ignore the explicit cue validities and simply count the total number of positive features for each option. They choose the option with the highest total number of positive features, and guess randomly if there is a tie. A highly restricted softmax temperature and lapse rate account for the overall noisy, near-chance behavior observed in the experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Count total positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Noisy Take The Best: Decision-makers use a lexicographic heuristic (Take The Best), checking features in descending order of validity and stopping at the first feature that discriminates between options. However, human application of this rule is highly noisy. A heavily restricted inverse temperature parameter accounts for the empirical observation that subjects' choices on these trials hover near chance level (0.50-0.55), rather than the highly deterministic choices (0.85+) predicted by standard TTB.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity descending; stable sort preserves original order on ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.95, 0.59, 0.81, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 12: A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 13: A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 14: A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 15: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 16: A=[1, 1, 1, 0]  B=[1, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
We isolate trials where the Tallying heuristic predicts a tie (the sum of positive features is equal for both options A and B). On these specific trials (4, 5, and 16 in the design), Tallying evaluates the options as exactly equal and predicts a 50% chance of choosing A. In contrast, Take The Best systematically favors option A on all three of these trials because option A has a positive value on the most valid discriminating cue. By calculating the proportion of times subjects choose option A on these 'Tally-tie' trials, we expect Tallying to yield exactly 0.5, whereas Take The Best will yield a value significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    is_tie = data.apply(lambda row: sum(row['option_a_ratings']) == sum(row['option_b_ratings']), axis=1)
    tie_trials = data[is_tie]
    if len(tie_trials) == 0:
        return 0.5
    return (tie_trials['response'] == 0).mean()

### RESULTS
- Predicted under pi_5 (simulated): 0.5300 (var=0.0121)
- Predicted under pi_4 (simulated): 0.5478 (var=0.0110)
- Observed on real data: 0.4800 (var=0.2496)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.56, 0.78, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  trial 8: A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 9: A=[1, 1, 0, 0]  B=[0, 1, 0, 1]
  trial 10: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 11: A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  trial 12: A=[1, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 13: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 14: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 15: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 16: A=[1, 1, 0, 0]  B=[0, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Noisy Take The Best (TTB) heuristic exclusively on trials where TTB and Tallying produce conflicting predictions (either a strict disagreement or where Tallying predicts a tie). On these targeted trials, TTB-simulated data will yield an alignment rate above 0.5, while Tallying-simulated data will yield a rate below 0.5 (due to strict disagreements pulling the average down from chance level). Filtering for these specific trials maximizes the discriminative power between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # The validities are [0.95, 0.56, 0.78, 0.55]
    # The descending order of validities corresponds to indices: 0, 2, 1, 3
    cue_order = [0, 2, 1, 3]
    
    ttb_match_count = 0
    disagreement_trials_count = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Determine TTB winner
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Determine Tally winner
        score_a = sum(a)
        score_b = sum(b)
        if score_a > score_b:
            tally_winner = 0
        elif score_b > score_a:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Focus only on trials where TTB and Tally do not perfectly agree
        # (e.g. Tally predicts a tie but TTB has a preference, or they strictly disagree)
        if ttb_winner is not None and ttb_winner != tally_winner:
            if row['response'] == ttb_winner:
                ttb_match_count += 1
            disagreement_trials_count += 1
            
    if disagreement_trials_count == 0:
        return 0.5
        
    return float(ttb_match_count) / disagreement_trials_count

### RESULTS
- Predicted under pi_5 (simulated): 0.4800 (var=0.0083)
- Predicted under pi_4 (simulated): 0.5389 (var=0.0082)
- Observed on real data: 0.4933 (var=0.0277)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # The validities are [0.95, 0.72, 0.73, 0.55]
    # Thus, the descending order of cue validities for Take The Best is:
    # Feature 0 (0.95), Feature 2 (0.73), Feature 1 (0.72), Feature 3 (0.55)
    cue_order = [0, 2, 1, 3]
    matches = []
    
    for a_vals, b_vals, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = np.array(a_vals)
        b = np.array(b_vals)
        
        # Isolate trials where Tallying predicts a tie (equal number of feature wins)
        if np.sum(a > b) == np.sum(b > a):
            # Take The Best will still make a deterministic prediction based on the first discriminating cue
            for j in cue_order:
                if a[j] > b[j]:
                    matches.append(1.0 if resp == 0 else 0.0)
                    break
                elif b[j] > a[j]:
                    matches.append(1.0 if resp == 1 else 0.0)
                    break
                    
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.5500 (var=0.0600)
**Predicted under pi_5:** 0.4883 (var=0.0099)
**Predicted under pi_4:** 0.5433 (var=0.0108)

### Experiment 4
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score = 0.0
    count = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_list = list(a)
        b_list = list(b)
        # Trial 1: Tallying ties, TTB predicts A (0)
        if a_list == [1, 1, 0, 1] and b_list == [0, 1, 1, 1]:
            score += (1 if r == 0 else 0)
            count += 1
        # Trial 2: Tallying ties, TTB predicts B (1)
        elif a_list == [0, 1, 1, 0] and b_list == [1, 0, 1, 0]:
            score += (1 if r == 1 else 0)
            count += 1
        # Trial 9: Tallying predicts A, TTB predicts B (1)
        elif a_list == [0, 1, 0, 1] and b_list == [0, 0, 1, 0]:
            score += (1 if r == 1 else 0)
            count += 1
            
    if count == 0:
        return 0.5
    return float(score / count)
```

**Observed (real) value:** 0.5333 (var=0.0267)
**Predicted under pi_5:** 0.4678 (var=0.0142)
**Predicted under pi_4:** 0.5456 (var=0.0123)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.6, 0.87, 0.55])
    
    wadd_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        if tally_a == tally_b:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            if wadd_a > wadd_b:
                wadd_match.append(1 if row['response'] == 0 else 0)
            elif wadd_b > wadd_a:
                wadd_match.append(1 if row['response'] == 1 else 0)
                
    return float(np.mean(wadd_match)) if len(wadd_match) > 0 else 0.5
```

**Observed (real) value:** 0.4900 (var=0.0624)
**Predicted under pi_5:** 0.4783 (var=0.0119)
**Predicted under pi_4:** 0.5417 (var=0.0140)

### Experiment 6
**Design**
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    is_t2 = (a_str == '0110') & (b_str == '1001')
    is_t16 = (a_str == '1011') & (b_str == '0111')
    
    t2_wadd_choices = (data.loc[is_t2, 'response'] == 1).sum()
    t16_wadd_choices = (data.loc[is_t16, 'response'] == 0).sum()
    
    total_trials = is_t2.sum() + is_t16.sum()
    
    if total_trials == 0:
        return 0.5
        
    return float((t2_wadd_choices + t16_wadd_choices) / total_trials)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_5:** 0.5183 (var=0.0181)
**Predicted under pi_4:** 0.5167 (var=0.0244)

### Experiment 7
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    val = np.array([0.95, 0.55, 0.65, 0.55])
    
    # Extract options as 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate WADD scores
    scores_a = np.dot(a_ratings, val)
    scores_b = np.dot(b_ratings, val)
    
    diffs = np.abs(scores_a - scores_b)
    
    # Determine if the subject chose the option with the higher WADD score
    # (Note: For this specific design, TTB and WADD agree on the winner for all 16 trials)
    chose_a = (data['response'] == 0).values
    correct = ((scores_a > scores_b) & chose_a) | ((scores_b > scores_a) & ~chose_a)
    
    # Contrast trials with a large difference in WADD scores vs a small difference
    high_diff = diffs >= 0.8
    low_diff = diffs <= 0.4
    
    if np.sum(high_diff) == 0 or np.sum(low_diff) == 0:
        return 0.0
        
    return float(np.mean(correct[high_diff]) - np.mean(correct[low_diff]))
```

**Observed (real) value:** -0.0933 (var=0.1024)
**Predicted under pi_5:** 0.0336 (var=0.0121)
**Predicted under pi_4:** 0.0102 (var=0.0104)

### Experiment 8
**Design**
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.87, 0.89, 0.55])
    
    acc_large = []
    acc_small = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        score_a = np.sum(a * val)
        score_b = np.sum(b * val)
        
        diff = abs(score_a - score_b)
        
        if score_a > score_b:
            correct = 1 if row['response'] == 0 else 0
        else:
            correct = 1 if row['response'] == 1 else 0
            
        if diff > 1.0:
            acc_large.append(correct)
        elif diff < 0.5:
            acc_small.append(correct)
            
    if not acc_large or not acc_small:
        return 0.0
        
    return float(np.mean(acc_large) - np.mean(acc_small))
```

**Observed (real) value:** -0.0240 (var=0.3594)
**Predicted under pi_5:** 0.0897 (var=0.0213)
**Predicted under pi_4:** 0.0112 (var=0.0189)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither Tallying (THEORY 1) nor Noisy Take The Best (THEORY 2) consistently captures the observed data. In Experiments 1 and 2, the observed data closely aligns with the predictions of Tallying (values near 0.48-0.49). However, in Experiments 3 and 4, the observed data aligns much better with Noisy Take The Best (values near 0.53-0.55). In later experiments like 7 and 8, both models struggle to capture the negative or near-zero differences observed. This inconsistency suggests that human decision-makers might not be using a strict lexicographic rule or a strict equal-weights tallying rule across all contexts. Instead, they likely integrate all cue validities in a weighted compensatory manner, but with high levels of decision noise.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a Weighted Additive (WADD) heuristic theory. Unlike Tallying (which ignores validities) and Take The Best (which uses a strict non-compensatory lexicographic rule), WADD assumes subjects compute a weighted sum of the positive features for each option, where the weights are proportional to the explicit cue validities provided in the instructions. To account for the near-chance performance, incorporate a noisy choice rule (e.g., softmax temperature and a lapse rate)."
}
```

## Usage

```json
{
  "prompt_token_count": 9231,
  "candidates_token_count": 316,
  "total_token_count": 10321
}
```
