# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_10") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_10" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_10
Decision-makers do not integrate all cues on every trial. Instead, they maintain a repertoire of distinct heuristics—specifically Take The Best (TTB) and Tallying—and select one to strictly apply on each decision. On any given trial, an individual either completely relies on TTB (basing their choice solely on the single most valid discriminating cue), falls back to Tallying (counting the number of positive features for each option while ignoring validities), or simply guesses randomly due to lapses in attention. This discrete strategy selection captures bimodal response patterns without predicting the systematic compensatory shifts that cue-integration models enforce.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Take The Best (TTB) prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_winner = -1
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner == 0:
        ttb_probs = np.array([1.0, 0.0])
    elif ttb_winner == 1:
        ttb_probs = np.array([0.0, 1.0])
    else:
        ttb_probs = np.array([0.5, 0.5])
        
    # 2. Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_probs = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_probs = np.array([0.0, 1.0])
    else:
        tally_probs = np.array([0.5, 0.5])
        
    # 3. Strategy mixture weights
    w_ttb = float(parameters["w_ttb"])
    w_tally_rel = float(parameters["w_tally_rel"])
    
    w_tally = (1.0 - w_ttb) * w_tally_rel
    w_rand = (1.0 - w_ttb) * (1.0 - w_tally_rel)
    
    # Final probability is a discrete mixture of the strategies
    p = w_ttb * ttb_probs + w_tally * tally_probs + w_rand * np.array([0.5, 0.5])
    
    return p


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the TTB-Tallying mixture model (Competing Theory), we systematically vary the support from Tallying while holding the highest-validity discriminating cue constant. Across the first four trials, the highest-validity cue (Feature 1) favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Competing Theory incorporates Tallying (which counts the total number of positive features). As we parametrically increase the number of lower-validity cues favoring Option A (from 0 to 3), the Tallying component shifts from strongly favoring Option B to strongly favoring Option A. Consequently, the mixture model predicts a monotonic increase in the probability of choosing Option A across these trials, directly contradicting the flat line predicted by pure TTB. Trials 5-7 repeat this logic with Feature 1 tied, testing the same principle on the second-highest validity cue.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the proportion of choosing Option A between Trial 4 and Trial 1. In both trials, the most valid cue (Feature 1) favors Option A. Pure Take The Best (TTB) uses only this first cue, ignoring the rest, and thus predicts an identical probability of choosing A across both trials (difference around 0). However, the Competing Theory's mixture model incorporates a Tallying component. In Trial 1, Tallying strongly favors B (4 features to 1), whereas in Trial 4, Tallying strongly favors A (4 features to 1). Therefore, the mixture model predicts a substantially higher rate of choosing A in Trial 4 compared to Trial 1, resulting in a large positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    # Calculate sum of features for A and B to identify trials
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify Trial 4 (A has 4 features, B has 1) and Trial 1 (A has 1 feature, B has 4)
    trial_4 = (sum_a == 4) & (sum_b == 1)
    trial_1 = (sum_a == 1) & (sum_b == 4)
    
    # Proportion of choosing Option A (response == 0)
    p_a_trial4 = (data.loc[trial_4, 'response'] == 0).mean()
    p_a_trial1 = (data.loc[trial_1, 'response'] == 0).mean()
    
    # Return the difference
    return float(p_a_trial4 - p_a_trial1)

### RESULTS
- Predicted under pi_1 (simulated): -0.0108 (var=0.0159)
- Predicted under pi_10 (simulated): 0.2077 (var=0.0711)
- Observed on real data: -0.7538 (var=0.0360)

## EXPERIMENT 2 (proposed by pi_10)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the TTB-Tallying mixture model (Advocated Theory) from the pure Take The Best model (Competing Theory), we systematically vary the support from the Tallying strategy while holding the pure TTB prediction constant. Across the first four trials, the highest-validity discriminating cue (Feature 1) always favors Option A. In pure TTB, choice probability depends strictly on the first cue that discriminates between the options, ignoring both the magnitude of that cue's validity and any subsequent cues. Thus, pure TTB predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Advocated Theory incorporates Tallying, which counts the total number of positive features. By parametrically shifting the Tallying score from strongly favoring Option B to strongly favoring Option A, the mixture model predicts a graded, monotonic shift in choice probabilities, directly contradicting the flat line predicted by pure TTB. The final two trials test this same principle when the highest-validity cue is tied.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The pure Take The Best (TTB) model predicts that choice probability is determined entirely by the first discriminating cue. In both Trial 1 and Trial 4, the highest-validity cue (Feature 1) strongly favors Option A, and thus pure TTB predicts an identical rate of choosing Option A across these two trials (a difference of 0). Conversely, the Advocated Theory (TTB-Tallying mixture) predicts that subjects will sometimes use a Tallying strategy. In Trial 1, Tallying strongly favors Option B (1 vs 4 positive features), while in Trial 4, Tallying strongly favors Option A (5 vs 0 positive features). Therefore, the mixture model predicts a substantially higher rate of choosing Option A in Trial 4 compared to Trial 1. This metric calculates exactly this difference: P(Choose A | Trial 4) - P(Choose A | Trial 1).

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert lists to tuples to make them hashable and comparable
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Identify Trial 4 and Trial 1 by Option A's feature vector
    t4_mask = a_tuples == (1, 1, 1, 1, 1)
    t1_mask = a_tuples == (1, 0, 0, 0, 0)
    
    t4_data = data[t4_mask]
    t1_data = data[t1_mask]
    
    if len(t4_data) == 0 or len(t1_data) == 0:
        return 0.0
        
    # response == 0 means subject chose Option A
    p_a_t4 = (t4_data['response'] == 0).mean()
    p_a_t1 = (t1_data['response'] == 0).mean()
    
    return float(p_a_t4 - p_a_t1)

### RESULTS
- Predicted under pi_1 (simulated): 0.0275 (var=0.0108)
- Predicted under pi_10 (simulated): 0.2063 (var=0.0604)
- Observed on real data: -0.0300 (var=0.0138)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Features are pre-sorted by validity due to the design
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred != -1:
            matches.append(1.0 if row['response'] == ttb_pred else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.6369 (var=0.0028)
**Predicted under pi_1:** 0.8289 (var=0.0104)
**Predicted under pi_10:** 0.6564 (var=0.0518)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def tally_predict(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_w = np.sum(a > b)
        b_w = np.sum(b > a)
        if a_w > b_w:
            return 0
        elif b_w > a_w:
            return 1
        else:
            return -1
            
    preds = data.apply(tally_predict, axis=1)
    valid_trials = preds != -1
    
    if valid_trials.sum() == 0:
        return 0.5
        
    match = (preds[valid_trials] == data.loc[valid_trials, 'response']).mean()
    return float(match)

```

**Observed (real) value:** 0.5855 (var=0.0012)
**Predicted under pi_1:** 0.3718 (var=0.0030)
**Predicted under pi_10:** 0.5553 (var=0.0286)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_target_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        # Trial 1: A has the best cue, B has all the rest
        if a == [1, 0, 0, 0] and b == [0, 1, 1, 1]:
            return True
        # Trial 2: A has the second best cue, B has the rest
        if a == [0, 1, 0, 0] and b == [0, 0, 1, 1]:
            return True
        return False

    mask = data.apply(is_target_trial, axis=1)
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
    
    # Return the proportion of times Option A was chosen (response == 0)
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.8287 (var=0.0147)
**Predicted under pi_1:** 0.8350 (var=0.0172)
**Predicted under pi_10:** 0.6494 (var=0.0476)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where A = [1, 0, 0, 0, 0] and B = [0, 1, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    if not is_trial_1.any():
        return 0.0
    return float(data.loc[is_trial_1, 'response'].mean())
```

**Observed (real) value:** 0.1811 (var=0.0113)
**Predicted under pi_1:** 0.1126 (var=0.0110)
**Predicted under pi_10:** 0.4116 (var=0.0702)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify the Take The Best (TTB) winner for each trial based on the highest validity cue (index 0)
    ttb_winner = [0 if a[0] > b[0] else 1 for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    ttb_consistent = (data['response'] == ttb_winner)
    
    # Calculate the Tallying score difference to identify trial types
    sum_a = [sum(a) for a in data['option_a_ratings']]
    sum_b = [sum(b) for b in data['option_b_ratings']]
    abs_diff = [abs(a - b) for a, b in zip(sum_a, sum_b)]
    
    df = pd.DataFrame({'ttb_consistent': ttb_consistent, 'abs_diff': abs_diff})
    
    # Trial 3: Tallying is perfectly neutral (difference of 0)
    df_tie = df[df['abs_diff'] == 0]
    k1 = df_tie['ttb_consistent'].sum()
    n1 = len(df_tie)
    
    # Trials 1 and 2: Tallying strongly opposes TTB (difference of 3)
    df_extreme = df[df['abs_diff'] == 3]
    k2 = df_extreme['ttb_consistent'].sum()
    n2 = len(df_extreme)
    
    if n1 == 0 or n2 == 0:
        return 0.0
        
    # Apply Laplace smoothing (Beta(0.5, 0.5) prior) to stabilize variance for deterministic subjects
    k1_prime = k1 + 0.5
    n1_prime = n1 + 1.0
    p1 = k1_prime / n1_prime
    
    k2_prime = k2 + 0.5
    n2_prime = n2 + 1.0
    p2 = k2_prime / n2_prime
    
    # Pooled proportion for the standard error
    p_pool = (k1_prime + k2_prime) / (n1_prime + n2_prime)
    variance = p_pool * (1.0 - p_pool) * (1.0 / n1_prime + 1.0 / n2_prime)
    
    if variance <= 0:
        return 0.0
        
    # Z-score for the difference in proportions
    z = (p1 - p2) / np.sqrt(variance)
    
    return float(z)
```

**Observed (real) value:** -26.7284 (var=5.1964)
**Predicted under pi_1:** 2.0570 (var=0.8462)
**Predicted under pi_10:** 6.0438 (var=1.5430)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the sum of Option A's features to identify the trial type
    # Trial 1 has sum(A) == 1, Trial 5 has sum(A) == 5
    sum_a = data['option_a_ratings'].apply(sum)
    
    # response == 0 means Option A was chosen
    choose_a = 1 - data['response']
    
    # Calculate the mean probability of choosing A for Trial 5 and Trial 1
    mean_a_5 = choose_a[sum_a == 5].mean()
    mean_a_1 = choose_a[sum_a == 1].mean()
    
    if pd.isna(mean_a_5) or pd.isna(mean_a_1):
        return 0.0
        
    return float(mean_a_5 - mean_a_1)
```

**Observed (real) value:** 0.0253 (var=0.0144)
**Predicted under pi_1:** 0.0116 (var=0.0088)
**Predicted under pi_10:** 0.2158 (var=0.0809)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where TTB and WADD strongly disagree.
    # TTB chooses the option favored by the highest-validity discriminating cue.
    # WADD integrates all cues, so it will favor the option with more lower-validity cues.
    
    def is_disagreement_trial(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Find the first discriminating cue (highest validity)
        ttb_favors_a = False
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_favors_a = True
                break
            elif b[i] > a[i]:
                ttb_favors_a = False
                break
                
        # A simple proxy for WADD favoring B is if B has strictly more positive cues than A
        wadd_favors_b = sum(b) > sum(a)
        
        return ttb_favors_a and wadd_favors_b

    mask = data.apply(is_disagreement_trial, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.0
        
    # Return the proportion of times the subject chose Option A (TTB's choice) on these trials
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.8256 (var=0.0128)
**Predicted under pi_1:** 0.8306 (var=0.0090)
**Predicted under pi_10:** 0.5941 (var=0.0609)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = {
        (1, 0, 0, 0, 0): 1,
        (0, 1, 0, 0, 0): 1,
        (1, 0, 1, 0, 0): 1,
        (0, 1, 1, 0, 0): 0,
        (1, 0, 0, 0, 1): 1
    }
    
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a_tuple = tuple(row['option_a_ratings'])
        if a_tuple in wadd_choices:
            if row['response'] == wadd_choices[a_tuple]:
                matches += 1
            total += 1
            
    if total == 0:
        return 0.0
    return float(matches) / total
```

**Observed (real) value:** 0.4341 (var=0.0007)
**Predicted under pi_1:** 0.1347 (var=0.0084)
**Predicted under pi_10:** 0.3558 (var=0.0523)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify trials where WADD favors Option B.
    # In the experimental design, these correspond to trials 1, 2, and 4.
    b_favored_by_wadd = {
        (0, 1, 1, 1, 1),
        (0, 1, 1, 1, 0),
        (0, 0, 1, 1, 1)
    }
    
    # Convert lists to tuples for hashing
    is_wadd_b = data['option_b_ratings'].apply(lambda x: tuple(x) in b_favored_by_wadd)
    
    # Calculate the proportion of times Option B was chosen (response == 1)
    # when WADD favors B vs when WADD favors A.
    p_b_when_wadd_b = data[is_wadd_b]['response'].mean()
    p_b_when_wadd_a = data[~is_wadd_b]['response'].mean()
    
    # Return the difference. 
    # TTB always favors A in all 5 trials, so it predicts ~0 difference.
    # The mixture model (which uses WADD) predicts > 0.
    return float(p_b_when_wadd_b - p_b_when_wadd_a)
```

**Observed (real) value:** 0.1958 (var=0.0111)
**Predicted under pi_1:** 0.0077 (var=0.0052)
**Predicted under pi_10:** 0.2239 (var=0.0500)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_match = []
    wadd_diff = []
    
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = -1
        for i in range(5):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        is_ttb_choice = (row['response'] == ttb_winner)
        ttb_match.append(is_ttb_choice)
        
        if ttb_winner == 0:
            wd = np.sum(val * a) - np.sum(val * b)
        else:
            wd = np.sum(val * b) - np.sum(val * a)
            
        wadd_diff.append(wd)
        
    data_copy = data.copy()
    data_copy['ttb_match'] = ttb_match
    data_copy['wadd_diff'] = wadd_diff
    
    pos_wadd = data_copy[data_copy['wadd_diff'] > 0]['ttb_match'].mean()
    neg_wadd = data_copy[data_copy['wadd_diff'] < 0]['ttb_match'].mean()
    
    if pd.isna(pos_wadd) or pd.isna(neg_wadd):
        return 0.0
        
    return float(pos_wadd - neg_wadd)
```

**Observed (real) value:** 0.0883 (var=0.0061)
**Predicted under pi_1:** -0.0081 (var=0.0044)
**Predicted under pi_10:** 0.1994 (var=0.0299)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_3 = (a_str == '11111') & (b_str == '00000')
    mask_1 = (a_str == '10000') & (b_str == '01111')
    
    p_a_3 = (data.loc[mask_3, 'response'] == 0).mean()
    p_a_1 = (data.loc[mask_1, 'response'] == 0).mean()
    
    return float(p_a_3 - p_a_1)
```

**Observed (real) value:** -0.0585 (var=0.0243)
**Predicted under pi_1:** 0.0031 (var=0.0218)
**Predicted under pi_10:** 0.2538 (var=0.0812)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10000') & (b_str == '01111')
    t3_mask = (a_str == '11100') & (b_str == '00011')
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1):
        p_a_t1 = 0.0
    if pd.isna(p_a_t3):
        p_a_t3 = 0.0
        
    return float(p_a_t3 - p_a_t1)
```

**Observed (real) value:** -0.0150 (var=0.0195)
**Predicted under pi_1:** 0.0337 (var=0.0078)
**Predicted under pi_10:** 0.2363 (var=0.0558)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_sums = data['option_a_ratings'].apply(sum).values
    choose_a = (data['response'] == 0).astype(float).values
    return float(np.cov(a_sums, choose_a)[0, 1])
```

**Observed (real) value:** -0.0152 (var=0.0024)
**Predicted under pi_1:** 0.0038 (var=0.0027)
**Predicted under pi_10:** 0.1295 (var=0.0171)

### Experiment 16
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate the sum of features for Option A and Option B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify Trial 3 (sum_A == 4, sum_B == 2) and Trial 1 (sum_A == 2, sum_B == 4)
    mask_3 = (sum_a == 4) & (sum_b == 2)
    mask_1 = (sum_a == 2) & (sum_b == 4)
    
    # Calculate the proportion of choosing Option A (response == 0)
    p_a_3 = (data.loc[mask_3, 'response'] == 0).mean()
    p_a_1 = (data.loc[mask_1, 'response'] == 0).mean()
    
    if pd.isna(p_a_3) or pd.isna(p_a_1):
        return 0.0
        
    return float(p_a_3 - p_a_1)
```

**Observed (real) value:** -0.0316 (var=0.0171)
**Predicted under pi_1:** 0.0042 (var=0.0092)
**Predicted under pi_10:** 0.2105 (var=0.0554)

### Experiment 17
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    b_sums = data['option_b_ratings'].apply(sum)
    high_b = data[b_sums >= 3]['response'].mean()
    low_b = data[b_sums <= 1]['response'].mean()
    if pd.isna(high_b) or pd.isna(low_b):
        return 0.0
    return float(high_b - low_b)
```

**Observed (real) value:** -0.0017 (var=0.0095)
**Predicted under pi_1:** 0.0002 (var=0.0084)
**Predicted under pi_10:** 0.2098 (var=0.0402)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    x = sum_a - sum_b
    chose_a = (data['response'] == 0).astype(float)
    
    group_high = chose_a[x > 0]
    group_low = chose_a[x < 0]
    
    if len(group_high) == 0 or len(group_low) == 0:
        return 0.0
        
    return float(group_high.mean() - group_low.mean())
```

**Observed (real) value:** -0.1026 (var=0.0137)
**Predicted under pi_1:** -0.0060 (var=0.0037)
**Predicted under pi_10:** 0.2096 (var=0.0376)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the suite of experiments, Theory 1 (pure Take The Best) consistently outperforms Theory 2 (the TTB-Tallying mixture model). In experiments designed to parametrically vary the support of lower-validity cues while holding the highest-validity cue constant (e.g., Exp 2, 8, 13, 14, 15, 16, 17), the observed data show flat choice probabilities that strongly align with pure TTB's predictions (differences near 0). Theory 2 consistently overestimates the use of Tallying, predicting large monotonic shifts (differences of ~0.20) that do not materialize in the real data. While there are isolated instances where some compensatory behavior occurs (e.g., Exp 10, 11), the rigid discrete mixture in Theory 2 fails to capture the overwhelming non-compensatory nature of the subjects' choices in the vast majority of the tasks.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a Sequential Evidence Accumulation theory (e.g., a simplified Sequential Sampling Model for heuristics). Instead of a discrete mixture between TTB and Tallying, assume decision-makers sample cues in descending order of validity and accumulate evidence. Crucially, the decision to stop and choose is probabilistic: strong evidence from the first cue usually triggers an immediate decision (mimicking TTB), but there is a small, continuous probability of continuing to sample lower-validity cues (leading to occasional compensatory effects). This would naturally capture the near-zero differences in the flat-line experiments while allowing for the slight deviations seen in other experiments, without forcing a bimodal TTB-vs-Tallying split."
}
```

## Usage

```json
{
  "prompt_token_count": 14117,
  "candidates_token_count": 384,
  "total_token_count": 16608
}
```
