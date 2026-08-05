# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_21") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_21" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_21
Strategy Mixture Model with Normalized Compensatory Evidence: The population consists of individuals who use different decision strategies. A proportion of the population uses a fast-and-frugal Take-The-Best (TTB) heuristic, stopping at the first discriminating cue and making a deterministic choice. The rest of the population uses a compensatory Weighted Additive (WADD) strategy, integrating all available cues weighted by their validities. To ensure the compensatory strategy is properly calibrated across different experimental environments, the WADD evidence is normalized by the total sum of validities, bounding it to [-1, 1] and allowing a single inverse temperature parameter to apply consistent stochasticity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # TTB Probability
    ttb_prob = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_prob = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_prob = np.array([0.0, 1.0])
            break
            
    # WADD Probability
    evidence = np.sum(val * (a - b)) / np.sum(val)
    scores = np.array([evidence, -evidence]) / 2.0
    z = beta * scores
    z -= np.max(z)  # Numerical stability
    wadd_prob = np.exp(z) / np.sum(np.exp(z))
    
    # Mixture
    p_core = w_ttb * ttb_prob + (1.0 - w_ttb) * wadd_prob
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design exploits a subtle but profound structural limitation in the Competing Theory's Strategy Mixture Model to strongly advocate for the pure Take The Best (TTB) model. In the pure TTB model, stochasticity is governed by a softmax inverse temperature (beta) applied directly to the TTB score, allowing it to predict a flat, invariant choice probability at any level (e.g., a constant 65% preference for Option A) across trials where the same cue discriminates. In contrast, the Competing Theory's mixture model lacks a temperature parameter for its TTB component, treating TTB as perfectly deterministic and relying entirely on the lapse rate (epsilon, bounded at 0.5) for TTB noise. Consequently, the mixture model cannot predict a flat choice probability below 75% without reducing the TTB mixture weight (w_ttb < 1.0). If w_ttb is reduced, the WADD component becomes active. By systematically varying the WADD evidence from strongly opposing Option A to strongly supporting Option A across six trials while keeping the highest-validity cue constant, pure TTB predicts a perfectly flat line (at any noise level). The mixture model, forced to use WADD to account for moderate noise, will erroneously predict a steep monotonic increase across these trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric measures the difference in the probability of choosing Option A between Trial 6 (where lower-validity cues strongly support A) and Trial 1 (where lower-validity cues strongly oppose A). Because the highest validity cue favors A in both trials, pure Take The Best ignores the lower-validity cues and predicts a difference of 0. The Strategy Mixture Model, however, relies on its compensatory WADD component to produce stochasticity, which integrates all cues. As WADD evidence shifts from strongly favoring B in Trial 1 to strongly favoring A in Trial 6, the mixture model predicts a substantial positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the number of positive features for Option A
    sum_a = data['option_a_ratings'].apply(sum)
    
    # Trial 6 uniquely has 4 positive features for Option A
    # Trial 1 uniquely has 1 positive feature for Option A
    mask_6 = sum_a == 4
    mask_1 = sum_a == 1
    
    # Calculate the proportion of times Option A (response == 0) was chosen
    p_a_6 = (data.loc[mask_6, 'response'] == 0).mean()
    p_a_1 = (data.loc[mask_1, 'response'] == 0).mean()
    
    return float(p_a_6 - p_a_1)

### RESULTS
- Predicted under pi_1 (simulated): 0.0075 (var=0.0148)
- Predicted under pi_21 (simulated): 0.2875 (var=0.0655)
- Observed on real data: -0.6950 (var=0.0398)

## EXPERIMENT 2 (proposed by pi_21)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Strategy Mixture Model with Normalized Compensatory Evidence (Advocated Theory) from the pure Take The Best model (Competing Theory), we systematically vary the normalized WADD evidence while holding the TTB prediction constant. Across the first five trials, the highest-validity cue (Feature 1) always favors Option A. Pure TTB bases its decision solely on this first discriminating cue, predicting an identical, flat probability of choosing Option A across these trials. In contrast, the Advocated Theory mixes TTB with a normalized WADD strategy. As we parametrically shift the lower-validity cues from strongly favoring Option B to strongly favoring Option A, the normalized WADD evidence shifts from negative to positive. Consequently, the mixture model predicts a monotonic increase in the probability of choosing Option A, directly contradicting the flat line predicted by pure TTB. The final trials replicate this logic when the first cue is tied, forcing TTB to rely on the second cue while WADD continues to integrate all normalized evidence.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
Calculates the difference in the probability of choosing Option A between Trial 5 and Trial 1. Pure TTB predicts a flat probability (difference = 0) because the first cue always favors A in both trials. The Advocated Theory (mixture with WADD) predicts a positive difference because the lower-validity cues shift from strongly favoring B in Trial 1 to strongly favoring A in Trial 5, increasing the WADD evidence for Option A.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_keys == (1, 0, 0, 0, 0)) & (b_keys == (0, 1, 1, 1, 0))
    t5_mask = (a_keys == (1, 1, 1, 0, 0)) & (b_keys == (0, 0, 0, 1, 1))
    
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_a_t5 = 1.0 - data[t5_mask]['response'].mean()
    
    if pd.isna(p_a_t1) or pd.isna(p_a_t5):
        return 0.0
        
    return float(p_a_t5 - p_a_t1)

### RESULTS
- Predicted under pi_1 (simulated): 0.0400 (var=0.0190)
- Predicted under pi_21 (simulated): 0.2554 (var=0.0668)
- Observed on real data: -0.0092 (var=0.0229)

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
**Predicted under pi_21:** 0.5976 (var=0.0405)

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
**Predicted under pi_21:** 0.5477 (var=0.0182)

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
**Predicted under pi_21:** 0.5056 (var=0.0450)

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
**Predicted under pi_21:** 0.4642 (var=0.0700)

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
**Predicted under pi_21:** 12.5365 (var=2.5598)

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
**Predicted under pi_21:** 0.3674 (var=0.0617)

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
**Predicted under pi_21:** 0.5850 (var=0.0375)

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
**Predicted under pi_21:** 0.4467 (var=0.0247)

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
**Predicted under pi_21:** 0.2340 (var=0.0358)

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
**Predicted under pi_21:** 0.1892 (var=0.0308)

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
**Predicted under pi_21:** 0.3215 (var=0.0849)

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
**Predicted under pi_21:** 0.2988 (var=0.0683)

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
**Predicted under pi_21:** 0.1817 (var=0.0124)

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
**Predicted under pi_21:** 0.2768 (var=0.0844)

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
**Predicted under pi_21:** 0.2035 (var=0.0289)

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
**Predicted under pi_21:** 0.2922 (var=0.0454)

### Experiment 19
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** -0.7538 (var=0.0360)
**Predicted under pi_1:** -0.0108 (var=0.0159)
**Predicted under pi_21:** 0.3862 (var=0.0747)

### Experiment 20
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** -0.0300 (var=0.0138)
**Predicted under pi_1:** 0.0275 (var=0.0108)
**Predicted under pi_21:** 0.3800 (var=0.0722)

### Experiment 21
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    A_tuples = data['option_a_ratings'].apply(tuple)
    B_tuples = data['option_b_ratings'].apply(tuple)
    
    mask_1 = (A_tuples == (1, 0, 0, 0, 0)) & (B_tuples == (0, 1, 1, 1, 1))
    mask_5 = (A_tuples == (1, 1, 1, 1, 1)) & (B_tuples == (0, 0, 0, 0, 0))
    
    p_a_1 = (data.loc[mask_1, 'response'] == 0).mean()
    p_a_5 = (data.loc[mask_5, 'response'] == 0).mean()
    
    if pd.isna(p_a_1):
        p_a_1 = 0.0
    if pd.isna(p_a_5):
        p_a_5 = 0.0
        
    return float(p_a_5 - p_a_1)
```

**Observed (real) value:** 0.0062 (var=0.0090)
**Predicted under pi_1:** 0.0138 (var=0.0200)
**Predicted under pi_21:** 0.3200 (var=0.0765)

### Experiment 22
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the sum of features for options A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify Trial 1 (A has 1 feature, B has 4 features)
    mask_1 = (sum_a == 1) & (sum_b == 4)
    # Identify Trial 7 (A has 5 features, B has 0 features)
    mask_7 = (sum_a == 5) & (sum_b == 0)
    
    # Calculate the probability of choosing Option A (response == 0)
    p_a_1 = 1.0 - data.loc[mask_1, 'response'].mean()
    p_a_7 = 1.0 - data.loc[mask_7, 'response'].mean()
    
    if pd.isna(p_a_1) or pd.isna(p_a_7):
        return 0.0
        
    # Return the difference in probability of choosing A between Trial 7 and Trial 1
    return float(p_a_7 - p_a_1)
```

**Observed (real) value:** -0.0123 (var=0.0202)
**Predicted under pi_1:** -0.0446 (var=0.0164)
**Predicted under pi_21:** 0.3015 (var=0.0731)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the number of positive features for Option A
    a_sums = data['option_a_ratings'].apply(sum)
    chose_a = (data['response'] == 0).astype(float)
    
    # Isolate Trial 4: Option A has 5 positive features. WADD strongly favors A.
    mask_t4 = a_sums == 5
    
    # Isolate Trial 1: Option A has 1 positive feature. WADD strongly favors B.
    mask_t1 = a_sums == 1
    
    if mask_t4.sum() == 0 or mask_t1.sum() == 0:
        return 0.0
        
    # Under pure TTB, the first cue always favors A in both Trial 4 and Trial 1,
    # so the probability of choosing A is identical (expected difference = 0).
    # Under the Strategy Mixture Model, the WADD component shifts from strongly
    # favoring B in Trial 1 to strongly favoring A in Trial 4, yielding a large
    # positive difference.
    return float(chose_a[mask_t4].mean() - chose_a[mask_t1].mean())
```

**Observed (real) value:** -0.0246 (var=0.0084)
**Predicted under pi_1:** -0.0154 (var=0.0149)
**Predicted under pi_21:** 0.2815 (var=0.0721)

### Experiment 24
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the sum of positive ratings for Option A and Option B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify Trial 5 (A strongly favored by all cues) and Trial 1 (A favored only by the top cue)
    mask_t5 = (sum_a == 5) & (sum_b == 0)
    mask_t1 = (sum_a == 1) & (sum_b == 4)
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_t5 = (data.loc[mask_t5, 'response'] == 0).mean()
    p_a_t1 = (data.loc[mask_t1, 'response'] == 0).mean()
    
    # Handle edge cases where a subject might not have these trials (though unlikely given the design)
    if pd.isna(p_a_t5):
        p_a_t5 = 0.5
    if pd.isna(p_a_t1):
        p_a_t1 = 0.5
        
    return float(p_a_t5 - p_a_t1)

```

**Observed (real) value:** -0.0133 (var=0.0237)
**Predicted under pi_1:** -0.0283 (var=0.0135)
**Predicted under pi_21:** 0.3350 (var=0.0876)

### Experiment 25
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    low_mask = (sum_a == 1) & (sum_b <= 2)
    high_mask = (sum_a == 1) & (sum_b >= 6)
    
    p_a_low = (data.loc[low_mask, 'response'] == 0).mean()
    p_a_high = (data.loc[high_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_low) or pd.isna(p_a_high):
        return 0.0
        
    return float(p_a_low - p_a_high)
```

**Observed (real) value:** 0.0133 (var=0.0101)
**Predicted under pi_1:** -0.0008 (var=0.0098)
**Predicted under pi_21:** 0.1875 (var=0.0429)

### Experiment 26
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_net_c(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        net = 0
        for j in range(1, len(a)):
            if b[j] > a[j]:
                net += 1
            elif a[j] > b[j]:
                net -= 1
        return net
        
    net_c = data.apply(get_net_c, axis=1)
    
    # The maximum threshold in the Advocated Theory is 6.
    # Therefore, trials with net contradiction >= 6 will ALWAYS trigger the confirmatory check drop.
    # Trials with net contradiction == 0 will NEVER trigger the drop.
    high_mask = net_c >= 6
    low_mask = net_c == 0
    
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0
        
    return float(data['response'][high_mask].mean() - data['response'][low_mask].mean())
```

**Observed (real) value:** 0.0140 (var=0.0181)
**Predicted under pi_1:** 0.0270 (var=0.0151)
**Predicted under pi_21:** 0.2490 (var=0.0519)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    preds = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        pred = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        preds.append(pred)
        
    preds = np.array(preds)
    responses = data['response'].values
    
    valid = preds != 0.5
    if not np.any(valid):
        return 0.5
        
    match = (responses[valid] == preds[valid])
    return float(np.mean(match))
```

**Observed (real) value:** 0.5596 (var=0.0014)
**Predicted under pi_1:** 0.8587 (var=0.0122)
**Predicted under pi_21:** 0.5848 (var=0.0299)

### Experiment 28
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate sum of cues for each option
    sum_a = data['option_a_ratings'].apply(lambda x: sum(x))
    sum_b = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Filter for trials where Tallying has a strict preference
    mask = sum_a != sum_b
    if mask.sum() == 0:
        return 0.5
        
    df = data[mask]
    s_a = sum_a[mask]
    s_b = sum_b[mask]
    
    # Tallying predicts 0 if A has more positive cues, 1 if B has more
    tally_pred = (s_b > s_a).astype(int)
    
    # Calculate proportion of responses matching Tallying
    accuracy = (df['response'] == tally_pred).mean()
    
    return float(accuracy)
```

**Observed (real) value:** 0.1325 (var=0.0067)
**Predicted under pi_1:** 0.1250 (var=0.0124)
**Predicted under pi_21:** 0.4479 (var=0.0462)

### Experiment 29
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate the difference in the number of positive features between A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff = sum_a - sum_b
    
    # In all 7 trials, the highest-validity discriminating cue favors Option A.
    # Thus, pure TTB always predicts Option A with the exact same probability,
    # regardless of the lower-validity cues (i.e., regardless of 'diff').
    # The Competing Theory (mixture with WADD) is sensitive to the total feature
    # difference, predicting a higher probability of choosing B when 'diff' < 0
    # compared to when 'diff' > 0.
    
    # Mean probability of choosing B when the total feature count favors B
    p_b_neg = data.loc[diff < 0, 'response'].mean()
    
    # Mean probability of choosing B when the total feature count favors A
    p_b_pos = data.loc[diff > 0, 'response'].mean()
    
    return float(p_b_neg - p_b_pos)

```

**Observed (real) value:** 0.0295 (var=0.0071)
**Predicted under pi_1:** -0.0010 (var=0.0072)
**Predicted under pi_21:** 0.2479 (var=0.0409)

### Experiment 30
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Filter for trials where Feature 1 discriminates in favor of A (Trials 1 to 5)
    mask = data['option_a_ratings'].apply(lambda x: x[0] == 1) & \
           data['option_b_ratings'].apply(lambda x: x[0] == 0)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.0
        
    # Calculate sum of A's features to distinguish trials 1 to 5
    a_sum = subset['option_a_ratings'].apply(lambda x: sum(x))
    
    # Probability of choosing A for trial 5 (a_sum == 5) vs trial 1 (a_sum == 1)
    t5 = subset[a_sum == 5]
    t1 = subset[a_sum == 1]
    
    if len(t5) == 0 or len(t1) == 0:
        return 0.0
        
    p5 = (t5['response'] == 0).mean()
    p1 = (t1['response'] == 0).mean()
    
    return float(p5 - p1)
```

**Observed (real) value:** 0.0277 (var=0.0198)
**Predicted under pi_1:** -0.0015 (var=0.0205)
**Predicted under pi_21:** 0.4246 (var=0.0559)

### Experiment 31
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['pair'] = data['A_str'] + '_' + data['B_str']
    
    # Trials where lower-validity cues strongly oppose the Take-The-Best winner (Option A)
    opposed = ['10000_01111', '10001_01110', '01000_00111', '00100_00011']
    # Trials where lower-validity cues support the Take-The-Best winner (Option A)
    supported = ['10011_01100', '10111_01000', '01011_00100', '00111_00000']
    
    opposed_data = data[data['pair'].isin(opposed)]
    supported_data = data[data['pair'].isin(supported)]
    
    if len(opposed_data) == 0 or len(supported_data) == 0:
        return 0.0
        
    # response == 0 means A, response == 1 means B
    p_A_opposed = 1.0 - opposed_data['response'].mean()
    p_A_supported = 1.0 - supported_data['response'].mean()
    
    diff = p_A_supported - p_A_opposed
    
    # Clip the difference to robustly handle the extreme between-subject variance 
    # generated by the Additive model's skewed decay_rate distribution.
    return float(np.clip(diff, -0.1, 0.1))

```

**Observed (real) value:** 0.0130 (var=0.0043)
**Predicted under pi_1:** 0.0060 (var=0.0029)
**Predicted under pi_21:** 0.1000 (var=0.0019)

### Experiment 32
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trials where the Advocated model predicts a very strong preference for Option A
    # Trial 3: A has top 3 cues, B has bottom 2
    # Trial 4: A has all cues, B has none
    t3_4_mask = ((a_tuples == (1, 1, 1, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1))) | \
                ((a_tuples == (1, 1, 1, 1, 1)) & (b_tuples == (0, 0, 0, 0, 0)))
                
    # Trials where the Advocated model predicts a much weaker preference for Option A
    # Trial 5: Cue 1 ties, Cue 2 favors A, but Cues 3,4,5 favor B (strong compensatory pressure)
    # Trial 6: Cue 1 ties, Cue 2,3,4,5 favor A (but score difference is still small due to decay)
    t5_6_mask = ((a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (1, 0, 1, 1, 1))) | \
                ((a_tuples == (1, 1, 1, 1, 1)) & (b_tuples == (1, 0, 0, 0, 0)))
    
    if t3_4_mask.sum() == 0 or t5_6_mask.sum() == 0:
        return 0.0
        
    # Mean probability of choosing Option B (response = 1)
    resp_t56 = data.loc[t5_6_mask, 'response'].mean()
    resp_t34 = data.loc[t3_4_mask, 'response'].mean()
    
    return float(resp_t56 - resp_t34)
```

**Observed (real) value:** 0.3650 (var=0.0157)
**Predicted under pi_1:** 0.0038 (var=0.0066)
**Predicted under pi_21:** 0.1625 (var=0.0219)

### Experiment 33
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_rank_and_correct(a, b, resp):
        for i in range(5):
            if a[i] > b[i]:
                return i, int(resp == 0)
            elif b[i] > a[i]:
                return i, int(resp == 1)
        return -1, 0

    ranks = []
    corrects = []
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        r, c = get_rank_and_correct(a, b, resp)
        ranks.append(r)
        corrects.append(c)
        
    df = pd.DataFrame({'rank': ranks, 'correct': corrects})
    
    early = df[df['rank'].isin([0, 1])]['correct'].mean()
    late = df[df['rank'].isin([3, 4])]['correct'].mean()
    
    if pd.isna(early) or pd.isna(late):
        return 0.0
        
    return float(early - late)
```

**Observed (real) value:** 0.0311 (var=0.0073)
**Predicted under pi_1:** 0.0111 (var=0.0088)
**Predicted under pi_21:** -0.2167 (var=0.0385)

### Experiment 34
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    
    diff = a_mat - b_mat
    disc_idx = np.argmax(np.abs(diff), axis=1)
    favored = np.where(diff[np.arange(len(diff)), disc_idx] > 0, 0, 1)
    
    is_favored = (data['response'] == favored).astype(float)
    
    idx_0_mask = (disc_idx == 0)
    idx_4_mask = (disc_idx == 4)
    
    mean_0 = is_favored[idx_0_mask].mean() if idx_0_mask.sum() > 0 else 0.5
    mean_4 = is_favored[idx_4_mask].mean() if idx_4_mask.sum() > 0 else 0.5
    
    return float(mean_0 - mean_4)
```

**Observed (real) value:** 0.0000 (var=0.0096)
**Predicted under pi_1:** -0.0062 (var=0.0067)
**Predicted under pi_21:** 0.0323 (var=0.0131)

### Experiment 35
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trial 1 and trial 5 based on option A's features
    data_copy = data.copy()
    data_copy['A_tuple'] = data_copy['option_a_ratings'].apply(tuple)
    
    t1_mask = data_copy['A_tuple'] == (1, 0, 0, 0, 0, 0)
    t5_mask = data_copy['A_tuple'] == (0, 0, 0, 0, 1, 0)
    
    # response == 0 means the subject chose Option A
    # Calculate the proportion of times Option A was chosen in each trial type
    p_a_t1 = 1.0 - data_copy[t1_mask]['response'].mean()
    p_a_t5 = 1.0 - data_copy[t5_mask]['response'].mean()
    
    # Return the difference in preference for Option A between trial 5 and trial 1
    return float(p_a_t5 - p_a_t1)
```

**Observed (real) value:** 0.0062 (var=0.0198)
**Predicted under pi_1:** -0.0169 (var=0.0121)
**Predicted under pi_21:** 0.1800 (var=0.0354)

### Experiment 36
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['chose_A'] = 1 - data['response']
    
    means = data.groupby('A_str')['chose_A'].mean()
    
    def get_mean(s):
        val = means.get(s)
        return val if pd.notna(val) else 0.5
        
    t1 = get_mean('10000')
    t2 = get_mean('11000')
    t3 = get_mean('11100')
    t4 = get_mean('11110')
    t5 = get_mean('01000')
    t6 = get_mean('01110')
    
    return float((t4 + t3) - (t1 + t2) + (t6 - t5))
```

**Observed (real) value:** -0.0250 (var=0.0584)
**Predicted under pi_1:** -0.0962 (var=0.0514)
**Predicted under pi_21:** 0.8550 (var=0.4598)

### Experiment 37
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data_copy = data.copy()
    data_copy['a_str'] = data_copy['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Trial 4: A=[0, 0, 0, 1, 0, 0], B=[0, 0, 0, 0, 1, 1]
    t4_mask = data_copy['a_str'] == '000100'
    # Trial 5: A=[0, 0, 0, 1, 1, 0], B=[0, 0, 0, 0, 0, 1]
    t5_mask = data_copy['a_str'] == '000110'
    
    if t4_mask.sum() == 0 or t5_mask.sum() == 0:
        return 0.0
        
    p_b_t4 = (data_copy.loc[t4_mask, 'response'] == 1).mean()
    p_b_t5 = (data_copy.loc[t5_mask, 'response'] == 1).mean()
    
    # Pure TTB predicts the exact same probability of choosing B (driven purely by noise) 
    # for both trials, as the top discriminating cue is identical (validity 0.65 favoring A).
    # The Competing Theory will fall back to WADD if the threshold is > 0.65.
    # In Trial 4, WADD strongly favors B, while in Trial 5, WADD favors A.
    # This difference isolates the WADD fallback effect on the lower-validity trials.
    return float(p_b_t4 - p_b_t5)
```

**Observed (real) value:** -0.7621 (var=0.0200)
**Predicted under pi_1:** -0.0168 (var=0.0120)
**Predicted under pi_21:** 0.1874 (var=0.0359)

### Experiment 38
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Option B's sum of ratings uniquely identifies the trial in this design
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 5 and 6 correspond to sum_b <= 1.
    # In these trials, both Take The Best (TTB) and the Weighted Additive (WADD) 
    # fallback strategy favor Option A. Thus, the Advocated Theory predicts a 
    # consistently high probability of choosing A for all subjects.
    mask_high = sum_b <= 1
    
    # Trials 1, 2, 3, and 4 correspond to sum_b >= 2.
    # In all these trials, TTB favors Option A, but WADD favors Option B.
    # Depending on the subject's validity threshold, the WADD fallback will be 
    # triggered in at least some of these trials (especially Trials 3 and 4 
    # where the top cue validity is lower), lowering the overall rate of choosing A.
    mask_low = sum_b >= 2
    
    # Pure TTB predicts a flat probability of choosing A across all trials because 
    # the top discriminating cue always favors A. Thus, TTB expects a difference of 0.
    # By pooling 32 trials for mask_high and 64 trials for mask_low, we drastically 
    # minimize the binomial response noise, reducing the per-subject variance for TTB 
    # and maximizing the t-statistic for the contrast.
    p_a_high = (data.loc[mask_high, 'response'] == 0).mean()
    p_a_low = (data.loc[mask_low, 'response'] == 0).mean()
    
    return float(p_a_high - p_a_low)
```

**Observed (real) value:** -0.0087 (var=0.0043)
**Predicted under pi_1:** 0.0088 (var=0.0043)
**Predicted under pi_21:** 0.1825 (var=0.0167)

### Experiment 39
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            elif val_b > val_a:
                return 1
        return 0.5
    
    ttb_choices = data.apply(get_ttb_choice, axis=1)
    return float((data['response'] == ttb_choices).mean())
```

**Observed (real) value:** 0.8742 (var=0.0064)
**Predicted under pi_1:** 0.8258 (var=0.0167)
**Predicted under pi_21:** 0.7242 (var=0.0163)

### Experiment 40
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate the sum of 1s for each option
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Determine if the subject chose the option with more 1s
    chose_more_1s = ((a_sums > b_sums) & (data['response'] == 0)) | ((b_sums > a_sums) & (data['response'] == 1))
    
    # Calculate the proportion of times each subject chose the option with more 1s
    subject_props = chose_more_1s.groupby(data['subject_id']).mean()
    
    # Calculate the absolute difference from 0.5 for each subject, then average
    return float(np.mean(np.abs(subject_props - 0.5)))

```

**Observed (real) value:** 0.1083 (var=0.0022)
**Predicted under pi_1:** 0.1210 (var=0.0016)
**Predicted under pi_21:** 0.1110 (var=0.0070)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The observed data across these experiments reveals a profound anomaly that neither pure Take The Best (pi_1) nor the Strategy Mixture Model (pi_21) can capture. In trials where Option A is supported only by the highest-validity cue and Option B by all lower-validity cues (e.g., A=[1,0,0,0,0], B=[0,1,1,1,1]), subjects overwhelmingly choose Option A, confirming that the highest-validity cue strongly dictates choice. However, as Option A accumulates more lower-validity cues (e.g., A=[1,1,1,1,0], B=[0,0,0,0,1]), the probability of choosing A paradoxically *plummets*, leading to highly negative metric values (e.g., -0.6950 in Exp 1, -0.7538 in Exp 19). Both current theories predict either a flat line (TTB) or an increase (WADD mixture) in the probability of choosing A as its features improve. The real data suggests a 'suspicious coincidence' or 'too-good-to-be-true' penalty: when an option dominates on almost all cues, subjects might infer that the cues are redundant, the task is a trick, or they apply a contrarian strategy. Alternatively, there is a strict feature-matching heuristic where subjects prefer options that represent a 'trade-off' rather than absolute dominance.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory that accounts for the 'dominance penalty' or 'anti-tallying' effect observed in the data. The new theory should retain a lexicographic or validity-weighted core (to explain why A is chosen when it only has the top cue) but introduce a mechanism where overwhelming support on lower-validity cues paradoxically reduces the probability of choosing that option. This could be modeled as a 'suspicion' parameter, a negative weight on cue redundancy, or a heuristic that actively penalizes options with a high sum of positive cues when a trade-off is expected."
}
```

## Usage

```json
{
  "prompt_token_count": 28178,
  "candidates_token_count": 473,
  "total_token_count": 32778
}
```
