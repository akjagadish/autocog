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
People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_6
Noisy Encoding Tallying: Decision-makers rely on the unweighted Tallying heuristic but suffer from noisy perception or encoding of the environment. Each binary feature has an independent probability of being misperceived (a 1 flipped to a 0, or a 0 flipped to a 1). Subjects then compute the tally of these perceived features and deterministically choose the option with the higher tally, breaking ties randomly. This naturally predicts that decision errors scale with the total number of features (capturing non-linear log-odds in certain experiments) because more features provide more opportunities for bit-flips to alter the tally difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    import math
    
    stim = np.asarray(state, dtype=float)
    n_features = stim.shape[1]
    
    p = float(parameters["flip_prob"])
    epsilon = float(parameters["epsilon"])
    
    def binom_pmf(k, n, prob):
        if n == 0:
            return 1.0 if k == 0 else 0.0
        if prob == 0.0:
            return 1.0 if k == 0 else 0.0
        if prob == 1.0:
            return 1.0 if k == n else 0.0
        return math.comb(n, k) * (prob ** k) * ((1 - prob) ** (n - k))
        
    def get_tally_dist(N1, N0, p):
        dist = np.zeros(N1 + N0 + 1)
        for x in range(N1 + 1):
            px = binom_pmf(x, N1, 1 - p)
            if px == 0.0:
                continue
            for y in range(N0 + 1):
                py = binom_pmf(y, N0, p)
                if py > 0.0:
                    dist[x + y] += px * py
        return dist
        
    N1_A = int(np.sum(stim[0]))
    N0_A = n_features - N1_A
    dist_A = get_tally_dist(N1_A, N0_A, p)
    
    N1_B = int(np.sum(stim[1]))
    N0_B = n_features - N1_B
    dist_B = get_tally_dist(N1_B, N0_B, p)
    
    prob_A_wins = 0.0
    prob_tie = 0.0
    for vA in range(len(dist_A)):
        if dist_A[vA] == 0.0:
            continue
        for vB in range(len(dist_B)):
            if dist_B[vB] == 0.0:
                continue
            if vA > vB:
                prob_A_wins += dist_A[vA] * dist_B[vB]
            elif vA == vB:
                prob_tie += dist_A[vA] * dist_B[vB]
                
    p_A_core = prob_A_wins + 0.5 * prob_tie
    p_B_core = 1.0 - p_A_core
    
    p_core = np.array([p_A_core, p_B_core])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.9, 0.85, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=11):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 10: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Softmax Tallying from Noisy Encoding Tallying by manipulating the total number of positive features while keeping the tally difference constant. Softmax Tallying predicts that choice probability depends exclusively on the difference in tallies (e.g., a 1 vs 0 trial yields the exact same choice probability as a 6 vs 5 trial, as the difference is 1 in both). In contrast, Noisy Encoding Tallying assumes each feature has an independent probability of being misperceived. Because features are bounded between 0 and 1, the binomial noise distributions become highly skewed near the boundaries (e.g., an option with 0 positive features can only gain perceived features, while an option with 6 positive features can only lose them). This causes Noisy Encoding Tallying to predict systematically different choice probabilities for 1v0 compared to 6v5, despite the constant difference. By including multiple levels of constant differences (diff=1, diff=2, diff=3) at varying absolute feature counts, we can clearly separate the exact-difference logistic predictions of Softmax Tallying from the boundary-skewed predictions of Noisy Encoding Tallying.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



### METRIC
Rationale:
Softmax Tallying's parameter distribution (beta in [0.1, 20.0]) heavily favors large values where the choice probability hits the ceiling (P~1.0) even for a minimal tally difference of 1. Consequently, it predicts almost no marginal increase in accuracy when the tally difference increases from 1 to 3. In contrast, Noisy Encoding Tallying's noise parameter (flip_prob up to 0.2) keeps the accuracy for a difference of 1 well below the ceiling (around 0.78 on average), allowing for a substantial and measurable increase in accuracy when the difference is 3 (around 0.99). Therefore, the difference in choice probability between diff=3 and diff=1 trials will be near zero for Softmax Tallying but large and positive for Noisy Encoding Tallying, providing a highly discriminative contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the unweighted sum of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Calculate the difference in tallies (A always has more or equal positive features in this design)
    diff = sum_a - sum_b
    
    # Calculate the proportion of times Option A (the better option) was chosen
    # response == 0 indicates Option A was chosen
    p_a_diff_1 = 1.0 - data.loc[diff == 1, 'response'].mean()
    p_a_diff_3 = 1.0 - data.loc[diff == 3, 'response'].mean()
    
    # Handle edge cases where a subject might have missing data
    if np.isnan(p_a_diff_1) or np.isnan(p_a_diff_3):
        return 0.0
        
    # Return the increase in accuracy when the tally difference grows from 1 to 3
    return float(p_a_diff_3 - p_a_diff_1)

### RESULTS
- Predicted under pi_3 (simulated): 0.0400 (var=0.0147)
- Predicted under pi_6 (simulated): 0.1787 (var=0.0107)
- Observed on real data: 0.0225 (var=0.0135)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=7):** [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Noisy Encoding Tallying from Softmax Tallying by simultaneously testing two properties: the linearity of log-odds and the invariance of constant differences. Softmax Tallying predicts that choice log-odds scale perfectly linearly with the tally difference, and that a constant difference (e.g., 2v1 vs 6v5) yields identical choice probabilities. In contrast, Noisy Encoding Tallying relies on binomial bit-flips, producing non-linear log-odds across increasing differences (e.g., 1v0 to 6v0) and varying choice probabilities for constant differences depending on the absolute number of positive features (due to boundary effects). By systematically varying both the tally difference and the absolute feature count in a 7-feature environment, we can definitively separate the two models.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Under Softmax Tallying, choice probability is a logistic function of the tally difference scaled by beta. Because beta is drawn uniformly from [0.1, 20.0], the average subject has a high beta, meaning their choice probability is already at ceiling (1 - epsilon/2) even for a tally difference of 1. Thus, there is very little difference in accuracy between trials with a difference of 1 and trials with a difference of 4 or more. Under Noisy Encoding Tallying, perceptual noise scales such that the variance of the perceived tally difference is constant across trials. For a tally difference of 1, the expected difference is small relative to the noise, resulting in accuracy significantly below ceiling. For a difference of 4 or more, the signal overcomes the noise, reaching ceiling accuracy. Therefore, the difference in accuracy between high-difference (>= 4) and low-difference (== 1) trials will be substantially larger under Noisy Encoding Tallying than under Softmax Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff = sum_a - sum_b
    
    mask_high = diff >= 4
    mask_low = diff == 1
    
    if mask_high.sum() == 0 or mask_low.sum() == 0:
        return 0.0
        
    pA_high = 1.0 - data.loc[mask_high, 'response'].mean()
    pA_low = 1.0 - data.loc[mask_low, 'response'].mean()
    
    return float(pA_high - pA_low)

### RESULTS
- Predicted under pi_3 (simulated): 0.0158 (var=0.0100)
- Predicted under pi_6 (simulated): 0.2026 (var=0.0093)
- Observed on real data: 0.0255 (var=0.0044)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    is_comp = []
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        if ttb_choice is None:
            match = 0.5
        else:
            match = 1.0 if resp == ttb_choice else 0.0
            
        ttb_match.append(match)
        # Trials where the sum of features differ are compensatory in this design
        is_comp.append(sum(a) != sum(b))
        
    ttb_match = np.array(ttb_match)
    is_comp = np.array(is_comp)
    
    if is_comp.sum() == 0 or (~is_comp).sum() == 0:
        return 0.0
        
    comp_mean = ttb_match[is_comp].mean()
    non_comp_mean = ttb_match[~is_comp].mean()
    
    return float(non_comp_mean - comp_mean)
```

**Observed (real) value:** 0.3500 (var=0.0124)
**Predicted under pi_3:** 0.3367 (var=0.0130)
**Predicted under pi_6:** 0.3375 (var=0.0158)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0.5
        
    ttb_preds = data.apply(ttb_choice, axis=1)
    match = (data['response'] == ttb_preds).mean()
    return float(match)
```

**Observed (real) value:** 0.1074 (var=0.0059)
**Predicted under pi_3:** 0.1324 (var=0.0111)
**Predicted under pi_6:** 0.1808 (var=0.0077)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Convert rating lists to tuples for exact matching
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: A has the 2 most valid cues, B has the 3 less valid cues
    t1_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    # Trial 2: B has the 2 most valid cues, A has the 3 less valid cues
    t2_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    # For these trials, Tallying strictly prefers the option with 3 positive features.
    # WADD typically prefers the option with the 2 most valid cues.
    # We calculate the proportion of times the subject chose the 3-feature option.
    t1_tally_choices = (data.loc[t1_mask, 'response'] == 1).sum()
    t2_tally_choices = (data.loc[t2_mask, 'response'] == 0).sum()
    
    total_tally_choices = t1_tally_choices + t2_tally_choices
    total_critical_trials = t1_mask.sum() + t2_mask.sum()
    
    if total_critical_trials == 0:
        return 0.5
        
    return float(total_tally_choices / total_critical_trials)
```

**Observed (real) value:** 0.8213 (var=0.0136)
**Predicted under pi_3:** 0.8387 (var=0.0187)
**Predicted under pi_6:** 0.8225 (var=0.0090)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Identify critical dissociation trials where WADD and Tallying make opposite predictions
    a_is_11000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    b_is_00111 = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    t1 = a_is_11000 & b_is_00111
    
    a_is_00111 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    b_is_11000 = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    t7 = a_is_00111 & b_is_11000
    
    # WADD prefers the option with fewer, but higher-validity features ([1, 1, 0, 0, 0])
    wadd_t1 = t1 & (data['response'] == 0)
    wadd_t7 = t7 & (data['response'] == 1)
    
    wadd_choices = wadd_t1.sum() + wadd_t7.sum()
    total_trials = t1.sum() + t7.sum()
    
    if total_trials == 0:
        return 0.5
    return float(wadd_choices / total_trials)
```

**Observed (real) value:** 0.1383 (var=0.0112)
**Predicted under pi_3:** 0.1467 (var=0.0120)
**Predicted under pi_6:** 0.2117 (var=0.0089)

### Experiment 7
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    top3_a = a_ratings[:, :3].sum(axis=1)
    top3_b = b_ratings[:, :3].sum(axis=1)
    
    # Isolate trials where total features are tied but top-3 features differ
    valid_mask = (sum_a == sum_b) & (top3_a != top3_b)
    
    if not np.any(valid_mask):
        return 0.5
        
    responses = data['response'].values
    
    # Did the subject choose the option with more top-3 features?
    chose_high_val = np.zeros(len(data), dtype=bool)
    chose_high_val[(top3_a > top3_b) & (responses == 0)] = True
    chose_high_val[(top3_a < top3_b) & (responses == 1)] = True
    
    return float(np.mean(chose_high_val[valid_mask]))
```

**Observed (real) value:** 0.5133 (var=0.0104)
**Predicted under pi_3:** 0.5058 (var=0.0098)
**Predicted under pi_6:** 0.5158 (var=0.0099)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # Pure Tallying relies only on the unweighted sum of features.
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    # Identify "tie" trials where both options have the exact same number of total positive features.
    # In the provided design, these are trials 5, 6, 7, and 9.
    tie_mask = (sum_a == sum_b)
    
    if not np.any(tie_mask):
        return 0.5
        
    responses = data['response'].values
    
    # In all of these tie trials, Option A was adversarially designed to contain strictly more 
    # high-validity features than Option B. 
    # We calculate the proportion of times the subject chose Option A on these trials.
    p_a = np.mean(responses[tie_mask] == 0)
    
    return float(p_a)
```

**Observed (real) value:** 0.5144 (var=0.0066)
**Predicted under pi_3:** 0.4917 (var=0.0053)
**Predicted under pi_6:** 0.5089 (var=0.0080)

### Experiment 9
**Design**
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score_A = data['option_a_ratings'].apply(sum)
    score_B = data['option_b_ratings'].apply(sum)
    
    diff = score_A - score_B
    mask = diff >= 2
    
    if not mask.any():
        return 0.0
        
    error_rate = (data.loc[mask, 'response'] == 1).mean()
    return float(error_rate)
```

**Observed (real) value:** 0.1190 (var=0.0080)
**Predicted under pi_3:** 0.0920 (var=0.0050)
**Predicted under pi_6:** 0.1055 (var=0.0071)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate unweighted tallies for each option
    tally_A = data['option_a_ratings'].apply(sum)
    tally_B = data['option_b_ratings'].apply(sum)
    
    # Group trials into two pools with matched tally differences (diff = 1, 2, 3)
    # Pool 0: Option B has 0 features (1v0, 2v0, 3v0)
    mask_v0 = (tally_B == 0) & (tally_A.isin([1, 2, 3]))
    
    # Pool 1: Option B has 1 feature (2v1, 3v1, 4v1)
    mask_v1 = (tally_B == 1) & (tally_A.isin([2, 3, 4]))
    
    # Count the number of times Option B (the inferior option) was chosen
    B0 = (data.loc[mask_v0, 'response'] == 1).sum()
    B1 = (data.loc[mask_v1, 'response'] == 1).sum()
    
    # Return the smoothed log-ratio of B choices
    # Laplace smoothing (+2) stabilizes the variance for subjects with few errors
    return float(np.log((B1 + 2.0) / (B0 + 2.0)))
```

**Observed (real) value:** 0.1285 (var=0.2416)
**Predicted under pi_3:** -0.1155 (var=0.1470)
**Predicted under pi_6:** 0.0800 (var=0.1275)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1 and 2, the observed data show only a marginal increase in accuracy (around 0.02) when the tally difference increases from 1 to 3 or 4. Theory 1 (Softmax Tallying, pi_3) captures this perfectly, as its high-beta parameters push choice probability to the ceiling even for a difference of 1. Theory 2 (Noisy Encoding Tallying, pi_6) predicts a massive accuracy gain (~0.18-0.20) because its bit-flip noise keeps accuracy for a difference of 1 artificially low, which is strongly contradicted by the data. However, Theory 2 performs better on Experiment 10, capturing boundary effects (more errors on 2v1 than 1v0) that Theory 1 misses. Because Theory 2 fails the discriminative experiments so significantly, it should be replaced with a new theory that can explain both the rapid ceiling effect and the boundary-dependent error rates.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory that hybridizes tallying with a different noise mechanism. For instance, a 'Thresholded Tallying' or 'Sequential Feature Sampling' model. Alternatively, a model that uses Softmax Tallying but incorporates a small, stimulus-dependent lapse rate (e.g., lapse increases when both options have non-zero features) could capture the boundary effects in Experiment 10 without destroying the rapid accuracy ceiling observed in Experiments 1 and 2."
}
```

## Usage

```json
{
  "prompt_token_count": 10571,
  "candidates_token_count": 343,
  "total_token_count": 12329
}
```
