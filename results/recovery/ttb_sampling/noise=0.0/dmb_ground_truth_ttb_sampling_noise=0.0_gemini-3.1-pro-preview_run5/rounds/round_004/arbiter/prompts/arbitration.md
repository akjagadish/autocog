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
Take The Best (TTB) posits that decision-makers do not integrate all available information. Instead, they rank features by their validity (informativeness) and search through them sequentially. The first feature that discriminates between the two options completely determines the choice, and all subsequent features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory heuristic allows a single highly valid cue to decisively outweigh any number of less valid cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to maintain original order in case of ties.
    order = np.argsort(-validities, kind='stable')
    
    score_a = 0.0
    score_b = 0.0
    
    # Sequential search: find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_6
Probabilistic Cue Retrieval TTB: Decision-makers follow the non-compensatory, sequential search architecture of Take The Best (TTB), but execution is noisy at the cue level. When a discriminating cue is evaluated, there is a probability of skipping it or failing to retrieve its value. This skip probability is a function of a base skip rate and is inversely proportional to the cue's validity (operationalized as proportional to 1 - validity). If a cue is skipped, the decision-maker continues to the next most valid cue. This produces slightly higher deviation rates on trials where the best cue is noisy, without defaulting to a fully compensatory linear integration.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    order = np.argsort(-validities, kind='stable')
    
    p_skip_base = float(parameters["p_skip"])
    
    p_A = 0.0
    p_B = 0.0
    p_continue = 1.0
    
    # Sequential search with probabilistic cue retrieval
    for idx in order:
        if a[idx] != b[idx]:
            # Probability of skipping this cue is proportional to its invalidity
            p_skip_cue = p_skip_base * (1.0 - validities[idx])
            p_eval = 1.0 - p_skip_cue
            
            if a[idx] > b[idx]:
                p_A += p_continue * p_eval
            else:
                p_B += p_continue * p_eval
                
            p_continue *= p_skip_cue
            
    # If all discriminating cues are skipped, or no cues discriminate, guess randomly
    p_A += p_continue * 0.5
    p_B += p_continue * 0.5
    
    p_core = np.array([p_A, p_B])
    
    # Global response noise
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** To cleanly dissociate pure Take The Best (TTB) from Probabilistic Cue Retrieval TTB, we use a 5-feature environment with a linear decline in validities and manipulate both the absolute validity of the highest discriminating cue and the alignment of the subsequent cues. Pure TTB strictly stops at the first discriminating cue, predicting a constant choice probability for the favored option regardless of the cue's validity or the alignment of subsequent cues. In contrast, Probabilistic Cue Retrieval TTB assumes a skip probability that increases as cue validity decreases. By creating trials where the first discriminating cue favors Option A but all subsequent cues favor Option B, and progressively tying the higher-validity cues (forcing reliance on less valid cues), Probabilistic TTB predicts a sharp drop in Option A choices. We also include control trials where subsequent cues align with the first discriminating cue, where Probabilistic TTB predicts a recovery in Option A choices, whereas pure TTB predicts no difference.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
To cleanly discriminate Pure TTB from Probabilistic Cue Retrieval TTB, we must perfectly cancel out the baseline epsilon noise while maximizing the signal of cue-skipping. Probabilistic TTB predicts deviations strictly when a cue is skipped AND subsequent cues oppose it. The skip rate is proportional to (1 - validity), giving theoretical signals of 0.05, 0.15, 0.25, and 0.35 for the 4 'Oppose' trial types, and 0 for the 3 'Align' trial types. By applying mathematically optimal matched-filter weights [-9, 5, 19, 33, -16, -16, -16] to the deviation rates of these exactly 7 trial types, the weights sum to precisely 0. This completely eliminates the expected value of epsilon noise for Pure TTB (yielding exactly 0), while perfectly extracting the linear skip-rate signal for Probabilistic TTB, maximizing the t-statistic.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Mathematically optimal matched-filter weights to maximize the signal-to-noise ratio.
    # Signal is proportional to (1 - validity) for Oppose trials, and 0 for Align trials.
    # The validities are [0.95, 0.85, 0.75, 0.65, 0.55], so invalidities are [0.05, 0.15, 0.25, 0.35].
    # The weights are perfectly mean-centered to sum to exactly 0, which perfectly cancels 
    # out the baseline epsilon noise for Pure TTB.
    weights = {
        1: -9.0,   # Oppose, disc cue 0 (Signal ~ 0.05)
        2: 5.0,    # Oppose, disc cue 1 (Signal ~ 0.15)
        3: 19.0,   # Oppose, disc cue 2 (Signal ~ 0.25)
        4: 33.0,   # Oppose, disc cue 3 (Signal ~ 0.35)
        5: -16.0,  # Align, disc cue 1 (Signal = 0)
        6: -16.0,  # Align, disc cue 2 (Signal = 0)
        7: -16.0   # Align, disc cue 3 (Signal = 0)
    }
    
    subject_scores = []
    
    for subj, subj_df in data.groupby('subject_id'):
        dev_counts = {k: 0.0 for k in weights}
        trial_counts = {k: 0.0 for k in weights}
        
        for _, row in subj_df.iterrows():
            a = row['option_a_ratings']
            b = row['option_b_ratings']
            resp = row['response']
            
            disc_cue = -1
            ttb_choice = -1
            for i in range(len(a)):
                if a[i] != b[i]:
                    disc_cue = i
                    ttb_choice = 0 if a[i] > b[i] else 1
                    break
                    
            if disc_cue == -1:
                continue
                
            score_a = sum(1 for i in range(disc_cue+1, len(a)) if a[i] > b[i])
            score_b = sum(1 for i in range(disc_cue+1, len(a)) if b[i] > a[i])
            
            if a[disc_cue] > b[disc_cue]:
                is_oppose = score_b > score_a
            else:
                is_oppose = score_a > score_b
                
            trial_type = -1
            if is_oppose:
                if disc_cue == 0: trial_type = 1
                elif disc_cue == 1: trial_type = 2
                elif disc_cue == 2: trial_type = 3
                elif disc_cue == 3: trial_type = 4
            else:
                if disc_cue == 1: trial_type = 5
                elif disc_cue == 2: trial_type = 6
                elif disc_cue == 3: trial_type = 7
                
            if trial_type != -1:
                trial_counts[trial_type] += 1
                if resp != ttb_choice:
                    dev_counts[trial_type] += 1
                    
        score = 0.0
        for k in weights:
            if trial_counts[k] > 0:
                score += weights[k] * (dev_counts[k] / trial_counts[k])
                
        subject_scores.append(score)
        
    return float(np.mean(subject_scores)) if subject_scores else 0.0

### RESULTS
- Predicted under pi_3 (simulated): 0.0615 (var=20.8215)
- Predicted under pi_6 (simulated): 3.7892 (var=29.6875)
- Observed on real data: 0.1723 (var=32.5213)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=6):** [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 6: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Rationale:** To cleanly dissociate pure TTB from Probabilistic Cue Retrieval TTB, we use a 6-feature environment with linearly decreasing validities from 1.0 to 0.5. We manipulate the validity of the first discriminating cue by systematically tying the most valid cues. In the 'oppose' condition, the first discriminating cue favors Option A, but all subsequent cues favor Option B. Pure TTB strictly stops at the first discriminating cue, predicting a constant, high probability of choosing Option A across all these trials regardless of which cue discriminates. In contrast, Probabilistic Cue Retrieval TTB predicts that as the validity of the first discriminating cue decreases, the probability of skipping it increases. Consequently, for the 'oppose' trials, Probabilistic TTB predicts a monotonically decreasing probability of choosing Option A. We also include 'agree' trials where subsequent cues also favor Option A, serving as a control where both models predict high choice probabilities, confirming the divergence is driven by cue retrieval failures falling back on opposing cues.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Previous metrics failed to reach significance because they either had too much per-subject variance or diluted the signal with trials where the cue skip probability was near zero. By perfectly isolating the three 'Agree' trials and the three 'Oppose' trials that have the highest theoretical skip probabilities (lowest validities), we maximize the mean difference between Pure TTB and PCR-TTB. Expanding the set from 2 to 3 trials per condition (27 repetitions per subject per condition instead of 18) significantly reduces the binomial noise and between-subject variance, pushing the t-statistic past the significance threshold.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the sum of features to cleanly identify trial types
    data['A_sum'] = data['option_a_ratings'].apply(sum)
    data['B_sum'] = data['option_b_ratings'].apply(sum)
    
    # 'Agree' trials (Trials 8, 9, 10): 
    # The first discriminating cue has a high skip probability.
    # However, if skipped, the downstream cues STILL strongly favor Option A.
    # Both Pure TTB and PCR-TTB predict high adherence to A.
    mask_agree = (data['A_sum'] == 6) & (data['B_sum'].isin([2, 3, 4]))
    
    # 'Oppose' trials (Trials 3, 4, 5):
    # The first discriminating cue has the same high skip probabilities as above.
    # But if skipped, the downstream cues strongly OPPOSE it (favoring Option B).
    # Pure TTB predicts identical adherence to A as in the Agree trials.
    # PCR-TTB predicts a significant drop in choosing A.
    mask_oppose = (data['B_sum'] == 5) & (data['A_sum'].isin([3, 4, 5]))
    
    p_agree = (data[mask_agree]['response'] == 0).mean()
    p_oppose = (data[mask_oppose]['response'] == 0).mean()
    
    if pd.isna(p_agree): p_agree = 0.5
    if pd.isna(p_oppose): p_oppose = 0.5
        
    # The difference will be ~0 for Pure TTB, and strongly positive for PCR-TTB.
    return float(p_agree - p_oppose)

### RESULTS
- Predicted under pi_3 (simulated): -0.0222 (var=0.0074)
- Predicted under pi_6 (simulated): 0.0548 (var=0.0099)
- Observed on real data: -0.0148 (var=0.0143)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    X = A - B
    y = np.where(data['response'] == 0, 1.0, -1.0)
    
    # Fit OLS to estimate the subjective weight of each feature
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # We apply a carefully constructed contrast vector k to the estimated weights.
    # The design matrix X has rank 4, with a null space spanned by [1, 0, 0, 1, 0].
    # For Tallying, all features are treated equally (true weights = [1, 1, 1, 1, 1]).
    # The projection of Tallying's weights onto the row space of X is [0, 1, 1, 0, 1].
    # Our contrast vector k is orthogonal to the null space AND satisfies k @ [0, 1, 1, 0, 1] = 0.
    # This mathematically guarantees that Tallying evaluates to exactly 0 in expectation.
    # For WADD, features are weighted by validities. The expected projected weights are 
    # [0.1, 0.45, 0.3, -0.1, 0.25]. 
    # k @ expected_WADD = 0.88(0.1) + 1(0.45) - 0.88(-0.1) - 1(0.25) = 0.376 > 0.
    k = np.array([0.88, 1.0, 0.0, -0.88, -1.0])
    
    return float(np.dot(k, beta_hat))

```

**Observed (real) value:** 0.7537 (var=0.0444)
**Predicted under pi_3:** 0.7379 (var=0.0518)
**Predicted under pi_6:** 0.7213 (var=0.0670)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1 = (1, 1, 0, 0, 0)
    t2 = (0, 0, 1, 1, 1)
    
    is_t1 = (a_tuples == t1) & (b_tuples == t2)
    is_t2 = (a_tuples == t2) & (b_tuples == t1)
    
    wadd_consistent = 0
    wadd_consistent += (data.loc[is_t1, 'response'] == 0).sum()
    wadd_consistent += (data.loc[is_t2, 'response'] == 1).sum()
    
    total = is_t1.sum() + is_t2.sum()
    
    if total == 0:
        return 0.5
    return float(wadd_consistent / total)
```

**Observed (real) value:** 0.9117 (var=0.0062)
**Predicted under pi_3:** 0.8692 (var=0.0105)
**Predicted under pi_6:** 0.9008 (var=0.0096)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    ttb_match = 0
    total = 0
    
    for a, b, resp in zip(a_tuples, b_tuples, data['response']):
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (1, 1, 0, 0) and b == (1, 0, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (0, 1, 0, 1) and b == (1, 0, 0, 0):
            ttb_match += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_match / total)
```

**Observed (real) value:** 0.8492 (var=0.0120)
**Predicted under pi_3:** 0.8883 (var=0.0044)
**Predicted under pi_6:** 0.8796 (var=0.0056)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Features are sorted by validity in the design (index 0 is highest)
        for i in range(len(a)):
            if a[i] > b[i]:
                if resp == 0:
                    matches += 1
                break
            elif b[i] > a[i]:
                if resp == 1:
                    matches += 1
                break
                
    return float(matches) / len(data)
```

**Observed (real) value:** 0.8279 (var=0.0105)
**Predicted under pi_3:** 0.8777 (var=0.0063)
**Predicted under pi_6:** 0.8452 (var=0.0034)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_0 = data['option_b_ratings'].apply(lambda x: x[0])
    mask = a_0 != b_0
    if not mask.any():
        return 0.5
    
    chose_feat0 = ((data['response'] == 0) & (a_0 == 1)) | ((data['response'] == 1) & (b_0 == 1))
    return float(chose_feat0[mask].mean())
```

**Observed (real) value:** 0.8489 (var=0.0084)
**Predicted under pi_3:** 0.8806 (var=0.0065)
**Predicted under pi_6:** 0.8942 (var=0.0044)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    non_ttb_choices = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_pred = 0 if a[i] > b[i] else 1
                break
        if row['response'] != ttb_pred:
            non_ttb_choices += 1
            
    rate = non_ttb_choices / len(data) if len(data) > 0 else 0.0
    
    # Variance stabilization for single-subject slices
    n_subj = data['subject_id'].nunique()
    if n_subj == 1:
        return rate / 10000.0
        
    return rate
```

**Observed (real) value:** 0.1693 (var=0.0000)
**Predicted under pi_3:** 0.1113 (var=0.0000)
**Predicted under pi_6:** 0.1447 (var=0.0000)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    def wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(b * validities) - np.sum(a * validities)
        
    diffs = data.apply(wadd_diff, axis=1)
    mask = diffs > 0
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1483 (var=0.0067)
**Predicted under pi_3:** 0.1125 (var=0.0055)
**Predicted under pi_6:** 0.1225 (var=0.0033)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    mask = b_sums > a_sums
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1660 (var=0.0073)
**Predicted under pi_3:** 0.1245 (var=0.0049)
**Predicted under pi_6:** 0.1108 (var=0.0038)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1 and 2, the observed data strongly contradict the predictions of Probabilistic Cue Retrieval TTB (pi_6). In Experiment 1, pi_6 predicted a massive signal for cue-skipping (3.78), but the observed metric was near zero (0.17), closely aligning with the pure TTB model (pi_3, 0.06). In Experiment 2, pi_6 predicted a positive difference due to skip probabilities, whereas the observed data showed a slight negative difference (-0.0148), again aligning with pure TTB. While pi_6 performs marginally better on some older experiments by injecting extra noise, its core mechanistic assumption (failing to retrieve cues based on invalidity) is overwhelmingly rejected by the targeted designs. Pure TTB (pi_3) provides a much more robust baseline.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Probabilistic Cue Retrieval TTB with a formal 'Weighted Additive (WADD) with Softmax' theory. WADD is the classic compensatory alternative to the non-compensatory TTB. It posits that decision-makers integrate all available features, weighting each by its validity, to compute an overall evidence score for each option. The choice is then made probabilistically using a softmax function over the evidence scores. This will provide a strong compensatory benchmark against the non-compensatory TTB, allowing future experiments to directly test whether subjects integrate all cues or stop at the first discriminating one."
}
```

## Usage

```json
{
  "prompt_token_count": 10056,
  "candidates_token_count": 334,
  "total_token_count": 11484
}
```
