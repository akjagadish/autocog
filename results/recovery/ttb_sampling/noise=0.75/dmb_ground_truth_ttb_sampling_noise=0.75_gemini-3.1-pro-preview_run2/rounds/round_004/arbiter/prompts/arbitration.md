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
Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_6
Dual-Process Hybrid Model (Take-The-Best + Weighted Additive) with soft compensatory noise: Human decision-making is predominantly non-compensatory (Take-The-Best), but subjects sometimes fall back on a compensatory strategy (Weighted Additive). Because WADD operates on sums of validities, its response noise is decoupled and constrained to extremely low values to reflect the highly soft, almost guessing-like nature of human compensatory choice. A high potential lapse rate captures general response noise and further scales down overconfident predictions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Hybrid expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) evaluation
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in ranked_features:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    # Weighted Additive (WADD) evaluation
    wadd_a = np.sum(a * validities)
    wadd_b = np.sum(b * validities)
    wadd_scores = np.array([wadd_a, wadd_b])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    
    # Softmax probabilities for TTB
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        p_ttb_vec = np.array([0.5, 0.5])
    else:
        z_ttb = beta_ttb * ttb_scores
        z_ttb = z_ttb - np.max(z_ttb)
        e_ttb = np.exp(z_ttb)
        p_ttb_vec = e_ttb / np.sum(e_ttb)
        
    # Softmax probabilities for WADD
    if wadd_a == wadd_b:
        p_wadd_vec = np.array([0.5, 0.5])
    else:
        z_wadd = beta_wadd * wadd_scores
        z_wadd = z_wadd - np.max(z_wadd)
        e_wadd = np.exp(z_wadd)
        p_wadd_vec = e_wadd / np.sum(e_wadd)
        
    # Mix the two processes
    p_ttb_weight = float(parameters["p_ttb"])
    p_core = p_ttb_weight * p_ttb_vec + (1.0 - p_ttb_weight) * p_wadd_vec
    
    # Apply uniform lapse
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate pure Take-The-Best (TTB) from the Dual-Process Hybrid Model (TTB + WADD), we exploit a unique invariance in the pure TTB model: its predicted choice probability depends only on the fact that a cue is the highest-ranking discriminator, and is completely insensitive to the absolute validity of that cue. By systematically varying WHICH cue is the highest discriminating cue (e.g., the 1st, 2nd, 3rd, or 4th most valid cue) while holding a subordinate opposing cue constant, pure TTB predicts an identical, flat choice probability across these conditions because the TTB score is always 1.0 vs 0.0. In contrast, the Hybrid model's WADD component is sensitive to the absolute validities of the features. As the best discriminating cue's validity decreases, the WADD support decreases, leading the Hybrid model to predict a graded reduction in choice probability.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the Welch's t-statistic comparing the probability of choosing the TTB winner when the compensatory WADD model supports TTB versus when it opposes TTB. Under the pure TTB model, the choice probability depends only on the best discriminating cue, making it identical across these conditions; thus, the expected t-statistic is 0. Under the Dual-Process Hybrid model, the compensatory WADD evidence systematically modulates choices, making the t-statistic robustly positive. By directly returning the t-statistic, the metric naturally aggregates statistical power (scaling with sqrt(N) on the pooled dataset) while maintaining a tightly bounded variance of ~1.0 at the single-subject level. This structural advantage reliably pushes the contrast past the significance threshold.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    support_vals = []
    oppose_vals = []
    
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
                
        if ttb_winner == -1:
            continue
            
        is_ttb = 1.0 if row['response'] == ttb_winner else 0.0
        
        wadd_a = np.sum(a * validities)
        wadd_b = np.sum(b * validities)
        
        if ttb_winner == 0:
            wadd_diff = wadd_a - wadd_b
        else:
            wadd_diff = wadd_b - wadd_a
            
        if wadd_diff > 0:
            support_vals.append(is_ttb)
        elif wadd_diff < 0:
            oppose_vals.append(is_ttb)
            
    if len(support_vals) < 2 or len(oppose_vals) < 2:
        return 0.0
        
    m_sup = float(np.mean(support_vals))
    m_opp = float(np.mean(oppose_vals))
    v_sup = float(np.var(support_vals, ddof=1))
    v_opp = float(np.var(oppose_vals, ddof=1))
    
    n_sup = len(support_vals)
    n_opp = len(oppose_vals)
    
    denom = np.sqrt(v_sup / n_sup + v_opp / n_opp)
    
    if denom == 0:
        return 0.0
        
    return float((m_sup - m_opp) / denom)

### RESULTS
- Predicted under pi_3 (simulated): -1.9152 (var=0.8559)
- Predicted under pi_6 (simulated): 1.6202 (var=1.0695)
- Observed on real data: -1.3285 (var=0.9669)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=14):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 13: A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate pure Take-The-Best (TTB) from the Dual-Process Hybrid Model (TTB + WADD), we exploit a fundamental symmetry required by pure TTB. Pure TTB's choice probabilities depend entirely on the single most valid discriminating feature. We construct a perfectly balanced set of trials where TTB favors Option A in exactly half the trials and Option B in the other half, with the exact same distribution of best discriminating features across the two halves. Pure TTB strictly predicts that the grand mean probability of choosing Option A across all trials will be exactly 0.5, as its response noise is symmetric. However, the trial pairs are designed such that the compensatory Weighted Additive (WADD) score strictly favors Option A in EVERY single trial. Because the Hybrid model mixes TTB with a continuous WADD component, it predicts a systematic symmetry violation: the constant underlying WADD support for Option A will push the grand mean choice probability for Option A strictly above 0.5. This provides a crisp, non-parametric test for the presence of the soft compensatory process advocated by the Hybrid model.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



### METRIC
Rationale:
Through rigorous mathematical analysis of the parameter distributions, both the Pure TTB model and the Dual-Process Hybrid model produce an identical expected mean TTB consistency of ~0.61 across subjects. However, they achieve this mean through fundamentally different parameter geometries. In the pure TTB model, the true probability of choosing the TTB winner is bounded by a maximum of ~0.88 (due to beta capping at 2.0), which compresses the distribution of subject consistencies. In the Hybrid model, the true probability is driven by the product of two uniformly distributed variables (p_ttb and epsilon) and can reach a full 1.0. This structural difference causes the Hybrid model's theoretical variance of true subject consistencies to be significantly higher than that of the Pure TTB model (~0.0121 vs ~0.0080). By directly computing the sample variance of the subjects' overall TTB consistencies, we capture this robust distributional difference. The metric provides a stable, highly significant contrast because it relies on the entire distribution's spread rather than noisy extreme tail events.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    subject_variances = []
    for subj, subj_df in data.groupby('subject_id'):
        # Calculate the subject's consistency with the Take-The-Best (TTB) winner
        ttb_consistent = 0
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            # Since validities are strictly decreasing, the first differing feature determines the TTB winner
            winner = -1
            for i in range(5):
                if a[i] > b[i]:
                    winner = 0
                    break
                elif b[i] > a[i]:
                    winner = 1
                    break
                    
            if winner != -1 and row['response'] == winner:
                ttb_consistent += 1
                
        p = ttb_consistent / len(subj_df)
        subject_variances.append(p)
        
    # The variance of subject consistencies is theoretically much higher for the Hybrid model
    if len(subject_variances) > 1:
        return float(np.var(subject_variances, ddof=1))
    else:
        return 0.0

### RESULTS
- Predicted under pi_3 (simulated): 0.0176 (var=0.0000)
- Predicted under pi_6 (simulated): 0.0150 (var=0.0000)
- Observed on real data: 0.0017 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_choices = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: A has 3 wins on weak features, B has 2 wins on strong features
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            # Tallying prefers A
            tally_choices.append(1.0 if row['response'] == 0 else 0.0)
        # Trial 2: A has 2 wins on strong features, B has 3 wins on weak features
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            # Tallying prefers B
            tally_choices.append(1.0 if row['response'] == 1 else 0.0)
            
    if not tally_choices:
        return 0.5
    return float(np.mean(tally_choices))
```

**Observed (real) value:** 0.3975 (var=0.0118)
**Predicted under pi_3:** 0.3862 (var=0.0217)
**Predicted under pi_6:** 0.3887 (var=0.0175)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the options are [1, 1, 0, 0, 0] and [0, 0, 1, 1, 1]
    is_11000_A = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_11000_B = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    mask = is_11000_A | is_11000_B
    if mask.sum() == 0:
        return 0.5
    
    subset = data[mask]
    is_11000_A_sub = is_11000_A[mask]
    
    # Check if the subject chose the option with [1, 1, 0, 0, 0]
    chose_11000 = (is_11000_A_sub & (subset['response'] == 0)) | (~is_11000_A_sub & (subset['response'] == 1))
    return float(chose_11000.mean())
```

**Observed (real) value:** 0.5833 (var=0.0110)
**Predicted under pi_3:** 0.5883 (var=0.0171)
**Predicted under pi_6:** 0.5858 (var=0.0150)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_count = 0
    total_incongruent = 0
    
    validities = [0.95, 0.85, 0.8, 0.75, 0.7]
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_choice = None
        for i in range(5):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        wadd_score_a = sum(a[i] * v for i, v in enumerate(validities))
        wadd_score_b = sum(b[i] * v for i, v in enumerate(validities))
        wadd_choice = 0 if wadd_score_a > wadd_score_b else 1
        
        if ttb_choice is not None and ttb_choice != wadd_choice:
            total_incongruent += 1
            if resp == ttb_choice:
                ttb_match_count += 1
                
    if total_incongruent == 0:
        return 0.5
    return ttb_match_count / total_incongruent

```

**Observed (real) value:** 0.5876 (var=0.0025)
**Predicted under pi_3:** 0.6240 (var=0.0143)
**Predicted under pi_6:** 0.6074 (var=0.0121)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
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
    responses = data['response'].values
    
    diff = a_ratings - b_ratings
    
    ttb_choices = []
    for i in range(len(diff)):
        choice = -1
        for j in range(diff.shape[1]):
            if diff[i, j] > 0:
                choice = 0
                break
            elif diff[i, j] < 0:
                choice = 1
                break
        ttb_choices.append(choice)
        
    ttb_choices = np.array(ttb_choices)
    valid = ttb_choices != -1
    
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.5893 (var=0.0033)
**Predicted under pi_3:** 0.6031 (var=0.0082)
**Predicted under pi_6:** 0.6178 (var=0.0126)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    high_support_trials = []
    low_support_trials = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Feature 0 has the highest validity in this design
        a0 = a[0]
        b0 = b[0]
        
        if a0 > b0:
            ttb_winner = 0
            ttb_support = sum(a)
        elif b0 > a0:
            ttb_winner = 1
            ttb_support = sum(b)
        else:
            continue
            
        chose_ttb = 1 if row['response'] == ttb_winner else 0
        
        if ttb_support == 4:
            high_support_trials.append(chose_ttb)
        elif ttb_support == 1:
            low_support_trials.append(chose_ttb)
            
    m_high = np.mean(high_support_trials) if len(high_support_trials) > 0 else 0.5
    m_low = np.mean(low_support_trials) if len(low_support_trials) > 0 else 0.5
    
    return float(m_high - m_low)
```

**Observed (real) value:** -0.0133 (var=0.0336)
**Predicted under pi_3:** -0.0100 (var=0.0295)
**Predicted under pi_6:** 0.0189 (var=0.0172)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Determine which option is the Take-The-Best (TTB) winner.
    # Feature 0 has the highest validity (0.95), so the option with a 1 in feature 0 is the TTB winner.
    ttb_winner = data['option_a_ratings'].apply(lambda x: 0 if x[0] == 1 else 1)
    chose_ttb = (data['response'] == ttb_winner).astype(float)
    
    # Calculate how many features the opposing (TTB-losing) option has.
    def get_opposing_cues(row):
        if row['option_a_ratings'][0] == 1:
            return sum(row['option_b_ratings'])
        else:
            return sum(row['option_a_ratings'])
            
    opposing_cues = data.apply(get_opposing_cues, axis=1)
    
    # Compare the probability of choosing the TTB winner when there are few opposing cues (1-2)
    # versus when there are many opposing cues (5-6).
    p_few = chose_ttb[opposing_cues <= 2].mean()
    p_many = chose_ttb[opposing_cues >= 5].mean()
    
    if np.isnan(p_few) or np.isnan(p_many):
        return 0.0
        
    return float(p_few - p_many)
```

**Observed (real) value:** 0.0162 (var=0.0214)
**Predicted under pi_3:** -0.0275 (var=0.0130)
**Predicted under pi_6:** 0.0075 (var=0.0161)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert lists of binary ratings into 2D numpy arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Find trials with exactly 1 discriminating feature (Hamming distance == 1)
    hamming_dist = np.sum(a_ratings != b_ratings, axis=1)
    mask = hamming_dist == 1
    
    if not np.any(mask):
        return 0.5
        
    # Determine the favored option on these trials.
    # Since they differ by exactly one feature, the option with the '1' 
    # on that feature will have a strictly greater sum of ratings.
    a_sum = np.sum(a_ratings[mask], axis=1)
    b_sum = np.sum(b_ratings[mask], axis=1)
    
    # favored_option is 0 (Option A) if A has the higher sum, else 1 (Option B)
    favored_option = (b_sum > a_sum).astype(int)
    responses = data.loc[mask, 'response'].values
    
    # Return the absolute choice rate of the favored option on these specific trials
    return float(np.mean(responses == favored_option))
```

**Observed (real) value:** 0.5844 (var=0.0223)
**Predicted under pi_3:** 0.6011 (var=0.0181)
**Predicted under pi_6:** 0.6000 (var=0.0227)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def subject_metric(sub_df):
        a_ratings = np.array(sub_df['option_a_ratings'].tolist())
        b_ratings = np.array(sub_df['option_b_ratings'].tolist())
        diff = a_ratings - b_ratings
        
        n_trials = len(diff)
        n_support = np.zeros(n_trials, dtype=int)
        for i in range(n_trials):
            d = diff[i]
            nz = np.where(d != 0)[0]
            if len(nz) > 0:
                fd = d[nz[0]]
                n_support[i] = np.sum((d * fd) > 0)
        
        trial_keys = sub_df['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x])) + '_' + \
                     sub_df['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
        
        df_temp = pd.DataFrame({
            'trial_key': trial_keys,
            'response': sub_df['response'].values,
            'n_support': n_support
        })
        
        grouped = df_temp.groupby('trial_key')
        p_j = grouped['response'].mean()
        n_supp_j = grouped['n_support'].first()
        
        # D_j measures the determinism of the choices for trial type j
        D_j = (p_j - 0.5) ** 2
        
        strong_mask = n_supp_j >= 3
        weak_mask = n_supp_j == 1
        
        if np.sum(strong_mask) == 0 or np.sum(weak_mask) == 0:
            return 0.0
            
        # Return the difference in determinism between Strong and Weak support trials
        return float(D_j[strong_mask].mean() - D_j[weak_mask].mean())

    if 'subject_id' in data.columns and data['subject_id'].nunique() > 1:
        return float(data.groupby('subject_id').apply(subject_metric).mean())
    else:
        return float(subject_metric(data))
```

**Observed (real) value:** -0.0080 (var=0.0019)
**Predicted under pi_3:** -0.0024 (var=0.0019)
**Predicted under pi_6:** -0.0001 (var=0.0021)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 1 (Pure TTB) consistently outperforms Theory 2 (Dual-Process Hybrid). The Hybrid model predicts robust compensatory effects from its Weighted Additive (WADD) component, leading to expected positive shifts in metrics where WADD supports TTB (e.g., Experiment 1 and Experiment 7). However, the observed data consistently fails to show this positive compensatory effect, yielding negative or near-zero values that align much closer with Pure TTB's predictions. Furthermore, in Experiment 2, both models overpredict the variance in subject consistency, but the Hybrid model's fundamental assumption of a soft compensatory process is strongly contradicted by the directional metrics in Experiments 1 and 7.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Dual-Process Hybrid theory with a completely different non-compensatory or semi-compensatory alternative, such as a 'Tallying' (Equal Weights) heuristic. In a Tallying theory, subjects simply count the number of positive features for each option, ignoring their specific validities, and choose the option with the most positive features. This provides a stark, structurally distinct contrast to TTB's strict lexicographic ranking, without relying on the validity-weighted sums (WADD) that the data has empirically rejected."
}
```

## Usage

```json
{
  "prompt_token_count": 12078,
  "candidates_token_count": 292,
  "total_token_count": 13901
}
```
