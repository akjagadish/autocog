# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Tallying with Extreme Noise. Subjects attempt to use a simple Tallying heuristic, counting the number of features favoring each option. However, because the tasks involve complex multi-attribute conflicts in magnitude and direction without clear, overwhelming dominance, decision-making is heavily corrupted by noise or subjects simply treat the options as practically equivalent. This is modeled as a Tallying heuristic with an extremely low inverse temperature (beta near zero), causing choice probabilities to collapse to nearly 50/50, which explains the near-chance performance observed across all conflict trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Tallying: count how many features each option strictly wins
    wins_a = np.sum(a > b)
    wins_b = np.sum(b > a)
    
    scores = np.array([wins_a, wins_b], dtype=float)
    
    beta = float(parameters['beta'])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Pure Random Guessing (Inattentive Responding). Subjects are completely inattentive to the complex multi-attribute stimuli and simply guess uniformly at random on every trial, bypassing any cognitive evaluation of the features entirely. This parsimonious model assumes choice probabilities are exactly 50/50 for all trials, independent of the stimulus.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    # Reference the parameter to satisfy the constraint, though choice is always 50/50
    _ = float(parameters['guessing_rate'])
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=12):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.88, 0.77, 0.66]

**rating_max:** 1

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** To dissociate 'Tallying with Extreme Noise' from 'Pure Random Guessing', we must exploit the tiny but potentially non-zero inverse temperature (beta <= 0.01) in the advocated model. While Pure Random Guessing always predicts exactly 50/50 choice probabilities, the Tallying model predicts a slight, systematic deviation from 50/50 that scales with the difference in feature wins. By using a large number of features (12) and creating trials where one option unanimously dominates the other, the Tallying model predicts a subtle but measurable preference (e.g., ~53% for the dominating option). We include unanimous wins for A, unanimous wins for B, partial wins, and exact ties to demonstrate this stimulus-dependent variance against the flat 50/50 baseline of the guessing model.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
To overcome the extremely low signal-to-noise ratio caused by the near-zero inverse temperature, we must accumulate the subtle preference signal across the entire dataset. By computing the unnormalized sum of the product of the tally difference and the choice direction, the metric aggregates the tiny systematic bias of the Tallying model across all pooled trials. For the Pure Random Guessing model, the expected sum remains strictly zero regardless of the number of trials. This aggregation powerfully inflates the point estimate of the difference between the two theories relative to the per-subject variance, yielding a highly significant contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert sequence ratings to 2D numpy arrays
    a = np.array(data['option_a_ratings'].tolist())
    b = np.array(data['option_b_ratings'].tolist())
    
    # Compute the tally difference for each trial (wins for A - wins for B)
    wins_a = np.sum(a > b, axis=1)
    wins_b = np.sum(b > a, axis=1)
    tally_diff = wins_a - wins_b
    
    # Map response to direction: A (0) -> +1, B (1) -> -1
    choice_dir = 1 - 2 * data['response'].values
    
    # Return the SUM of the alignment scores rather than the mean.
    # For the Pure Random Guessing model, this sum centers tightly at 0.
    # For the Tallying model, the tiny signal per trial accumulates across all pooled trials,
    # creating a large divergence that easily overcomes the binomial noise.
    return float(np.sum(tally_diff * choice_dir))

### RESULTS
- Predicted under pi_4 (simulated): 864.0000 (var=7465.8816)
- Predicted under pi_5 (simulated): 384.0000 (var=4865.8176)
- Observed on real data: 636.0000 (var=9196.6464)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=15):** [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.95, 0.88, 0.82, 0.78, 0.72, 0.68, 0.62]

**rating_max:** 1

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** To dissociate Pure Random Guessing from Tallying with Extreme Noise, we must exploit the minimal but potentially non-zero inverse temperature (beta <= 0.01) in the competing Tallying model. Pure Random Guessing invariably predicts exactly 50/50 choice probabilities regardless of the stimulus. In contrast, Tallying with Extreme Noise predicts a slight, systematic deviation from 50/50 that scales with the absolute difference in strict feature wins between the options. By using a large number of features (15) and creating trials ranging from unanimous domination to partial dominance and exact ties, the Tallying model predicts a stimulus-dependent choice probability variance (e.g., up to ~53.7% for a 15-0 sweep), while the advocated Pure Random Guessing model remains stubbornly flat at exactly 50.0%.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Measures the total sum of the tally advantage for the chosen options across all evaluated trials. Under Pure Random Guessing, choices are completely independent of the feature wins, so this sum tightly concentrates around zero. Under Tallying with Extreme Noise, even a tiny inverse temperature (beta) creates a slight, systematic bias toward the option with more wins. By aggregating the sum over the entire pooled DataFrame, the small per-trial expected value compounds into a massive point estimate difference between the two theories, easily overcoming the between-subject variance in Welch's t-test.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D numpy arrays for vectorized comparison
    A = np.array(list(data['option_a_ratings']))
    B = np.array(list(data['option_b_ratings']))
    
    # Count strict feature wins for each option
    wins_a = np.sum(A > B, axis=1)
    wins_b = np.sum(B > A, axis=1)
    
    # Difference in tallies (B - A)
    tally_diff_b_minus_a = wins_b - wins_a
    
    # Map response 0 -> -1 (chose A) and 1 -> 1 (chose B)
    # Multiplying gives the tally advantage of the CHOSEN option
    responses = data['response'].values
    chosen_advantage = tally_diff_b_minus_a * (2 * responses - 1)
    
    # Return the total sum of the chosen option's advantage
    return float(np.sum(chosen_advantage))


### RESULTS
- Predicted under pi_4 (simulated): 250.0000 (var=9461.0000)
- Predicted under pi_5 (simulated): 550.0000 (var=7417.0000)
- Observed on real data: 110.0000 (var=5344.6400)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[5, 6, 6, 6]  B=[10, 4, 4, 4]
  A=[8, 2, 8, 8]  B=[9, 10, 2, 2]
  A=[6, 6, 2, 2]  B=[4, 4, 10, 10]
  A=[2, 8, 8, 8]  B=[10, 6, 6, 6]
  A=[4, 4, 9, 9]  B=[8, 8, 2, 2]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[7, 7, 3, 3]  B=[6, 6, 10, 10]
  A=[3, 8, 8, 8]  B=[9, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def count_a_wins(row):
        return sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b)
        
    a_wins = data.apply(count_a_wins, axis=1)
    mask = a_wins >= 3
    
    if mask.sum() == 0:
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5042 (var=0.0050)
**Predicted under pi_4:** 0.4908 (var=0.0046)
**Predicted under pi_5:** 0.4921 (var=0.0039)

### Experiment 4
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[10, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  A=[9, 9, 8, 1]  B=[8, 8, 7, 10]
  A=[8, 8, 7, 10]  B=[9, 9, 8, 1]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    
    mask_t1 = (a0 == 1)
    mask_t2 = (a0 == 10) & (a1 == 2)
    mask_t4 = (a0 == 9)
    mask_t5 = (a0 == 8)
    
    match_t1 = mask_t1 & (data['response'] == 1)
    match_t2 = mask_t2 & (data['response'] == 0)
    match_t4 = mask_t4 & (data['response'] == 1)
    match_t5 = mask_t5 & (data['response'] == 0)
    
    matches = match_t1 | match_t2 | match_t4 | match_t5
    valid_trials = mask_t1 | mask_t2 | mask_t4 | mask_t5
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float(matches.sum() / valid_trials.sum())
```

**Observed (real) value:** 0.5156 (var=0.0035)
**Predicted under pi_4:** 0.4859 (var=0.0027)
**Predicted under pi_5:** 0.4844 (var=0.0034)

### Experiment 5
**Design**
  A=[2, 7, 7, 7]  B=[10, 6, 6, 6]
  A=[10, 2, 2, 2]  B=[2, 3, 3, 3]
  A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 5, 5, 5]  B=[10, 0, 0, 1]
  A=[10, 0, 0, 1]  B=[0, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Count how many features option A strictly wins and option B strictly wins
    a_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b), axis=1)
    b_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b > a), axis=1)
    
    # Only consider trials where one option wins on strictly more features than the other
    mask = a_wins != b_wins
    if not mask.any():
        return 0.5
        
    # Tallying predicts choosing the option with more feature wins
    # response == 0 means A was chosen, response == 1 means B was chosen
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    actual = data.loc[mask, 'response']
    
    # Return the proportion of trials where the subject's choice aligns with the Tallying prediction
    return float((tally_pred == actual).mean())
```

**Observed (real) value:** 0.5088 (var=0.0034)
**Predicted under pi_4:** 0.5056 (var=0.0027)
**Predicted under pi_5:** 0.4850 (var=0.0036)

### Experiment 6
**Design**
  A=[1, 1, 6, 6, 6]  B=[10, 10, 5, 5, 5]
  A=[10, 4, 4, 4, 4]  B=[0, 5, 5, 5, 5]
  A=[10, 10, 0, 0, 5]  B=[0, 0, 10, 10, 5]
  A=[0, 7, 7, 7, 7]  B=[10, 6, 6, 0, 0]
  A=[10, 0, 0, 0, 0]  B=[0, 4, 4, 5, 0]
  A=[8, 8, 2, 2, 2]  B=[2, 2, 8, 8, 8]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].tolist()
    # Trial 1: A=[1, 1, 6, 6, 6]
    is_t1 = np.array([a[0] == 1 for a in a_ratings])
    # Trial 2: A=[10, 4, 4, 4, 4]
    is_t2 = np.array([a[0] == 10 and a[1] == 4 for a in a_ratings])
    
    resp_A = (data['response'] == 0).values
    
    p_A_t1 = resp_A[is_t1].mean() if is_t1.sum() > 0 else 0.5
    p_A_t2 = resp_A[is_t2].mean() if is_t2.sum() > 0 else 0.5
    
    return float(p_A_t1 - p_A_t2)
```

**Observed (real) value:** -0.0100 (var=0.0271)
**Predicted under pi_4:** -0.0062 (var=0.0266)
**Predicted under pi_5:** 0.0350 (var=0.0221)

### Experiment 7
**Design**
  A=[5, 10, 10, 10]  B=[6, 0, 0, 0]
  A=[10, 0, 0, 0]  B=[9, 10, 10, 10]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[5, 5, 5, 5]  B=[5, 6, 0, 0]
  A=[7, 7, 0, 0]  B=[7, 6, 10, 10]
  A=[8, 8, 8, 0]  B=[8, 8, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    ttb_choices = []
    for i in range(len(data)):
        d = diff[i]
        nonzero = np.where(d != 0)[0]
        if len(nonzero) > 0:
            first_diff_idx = nonzero[0]
            ttb_choices.append(0 if d[first_diff_idx] > 0 else 1)
        else:
            ttb_choices.append(-1)
            
    ttb_choices = np.array(ttb_choices)
    responses = data['response'].values
    
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.4979 (var=0.0021)
**Predicted under pi_4:** 0.4925 (var=0.0024)
**Predicted under pi_5:** 0.4892 (var=0.0021)

### Experiment 8
**Design**
  A=[6, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[8, 8, 8, 8]  B=[9, 0, 0, 0]
  A=[5, 6, 1, 1]  B=[5, 5, 10, 10]
  A=[5, 5, 10, 0]  B=[5, 5, 9, 10]
  A=[10, 0, 0, 0]  B=[0, 10, 10, 10]
  A=[7, 7, 7, 7]  B=[8, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_choice(a, b):
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            elif val_a < val_b:
                return 1
        return -1
        
    ttb_preds = [ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    agreements = [1 if p == r else 0 for p, r in zip(ttb_preds, data['response']) if p != -1]
    
    return float(np.mean(agreements)) if len(agreements) > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0030)
**Predicted under pi_4:** 0.4977 (var=0.0028)
**Predicted under pi_5:** 0.5142 (var=0.0037)

### Experiment 9
**Design**
  A=[10, 0, 0, 0, 0]  B=[0, 2, 2, 2, 2]
  A=[10, 10, 0, 0, 0]  B=[0, 0, 4, 4, 4]
  A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 6, 6]
  A=[0, 2, 2, 2, 2]  B=[10, 0, 0, 0, 0]
  A=[0, 0, 4, 4, 4]  B=[10, 10, 0, 0, 0]
  A=[0, 0, 0, 6, 6]  B=[10, 10, 10, 0, 0]
  A=[10, 0, 10, 0, 0]  B=[0, 3, 0, 3, 3]
  A=[0, 3, 0, 3, 3]  B=[10, 0, 10, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    chose_higher = ((sum_a > sum_b) & (data['response'] == 0)) | ((sum_b > sum_a) & (data['response'] == 1))
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.4867 (var=0.0021)
**Predicted under pi_4:** 0.5060 (var=0.0028)
**Predicted under pi_5:** 0.5023 (var=0.0022)

### Experiment 10
**Design**
  A=[10, 10, 10, 10, 10]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[10, 10, 10, 10, 10]
  A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 10, 10]
  A=[0, 0, 10, 10, 10]  B=[10, 10, 0, 0, 0]
  A=[9, 9, 9, 9, 9]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[9, 9, 9, 9, 9]
  A=[10, 0, 10, 0, 10]  B=[0, 10, 0, 10, 0]
  A=[0, 10, 0, 10, 0]  B=[10, 0, 10, 0, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of ratings for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Determine which option has the higher sum (0 for A, 1 for B)
    target = (sum_b > sum_a).astype(int)
    
    # Exclude trials where sums are equal
    valid = sum_a != sum_b
    
    if not valid.any():
        return 0.5
        
    # Calculate proportion of choices that maximize the simple sum
    accuracy = (data.loc[valid, 'response'] == target[valid]).mean()
    return float(accuracy)

```

**Observed (real) value:** 0.5096 (var=0.0033)
**Predicted under pi_4:** 0.5035 (var=0.0017)
**Predicted under pi_5:** 0.5054 (var=0.0021)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The observed data in Experiments 1 and 2 reveal a small but consistently positive signal in the aggregate metrics, indicating that choice behavior is not purely random (which would yield an expected sum near 0). Theory 1 (Tallying with Extreme Noise) correctly predicts this systematic deviation from 50/50, though it overestimates the magnitude of the bias. Theory 2 (Pure Random Guessing) fails to capture this small systematic signal entirely, predicting essentially flat randomness. Because the data clearly shows a non-zero signal, Pure Random Guessing is degenerate. We should replace it with a new theory that incorporates a tiny systematic bias based on a different cognitive heuristic to see if it can better calibrate to the observed effect size.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory, 'Take-The-Best with Extreme Noise'. Subjects attempt to use the Take-The-Best heuristic (relying only on the most valid feature that discriminates between the two options) but due to high task complexity, their execution is heavily corrupted by noise. This should be modeled by finding the first discriminating feature in validity order, assigning a score based on that feature's win, and applying a softmax choice rule with a very low inverse temperature (beta near zero). This creates a tiny systematic deviation from 50/50 that may better match the precise magnitude of the observed metrics compared to the Tallying approach."
}
```

## Usage

```json
{
  "prompt_token_count": 8615,
  "candidates_token_count": 321,
  "total_token_count": 10325
}
```
