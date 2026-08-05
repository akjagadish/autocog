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
Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_6
Mixture of Simple Heuristics (Take-The-Best and Tallying) with Softened Determinism

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # TTB Heuristic: find the first discriminating cue in descending order of validity
    order = np.argsort(validities)[::-1]
    ttb_diff = 0.0
    for idx in order:
        if stim[0, idx] != stim[1, idx]:
            ttb_diff = stim[0, idx] - stim[1, idx]
            break

    # Tallying Heuristic: unweighted count of winning features
    a_wins = np.sum(stim[0] > stim[1])
    b_wins = np.sum(stim[1] > stim[0])
    tally_diff = a_wins - b_wins

    # Translate differences into probabilities using softmax (decision noise)
    # Bound the differences to prevent overflow
    z_ttb = np.clip(beta * ttb_diff, -100, 100)
    p_ttb_A = 1.0 / (1.0 + np.exp(-z_ttb))
    
    z_tally = np.clip(beta * tally_diff, -100, 100)
    p_tally_A = 1.0 / (1.0 + np.exp(-z_tally))

    # Mixture of the two heuristics
    p_core_A = w_ttb * p_ttb_A + (1.0 - w_ttb) * p_tally_A
    p_core_B = 1.0 - p_core_A

    # Apply uniform lapse rate
    p_final = (1.0 - epsilon) * np.array([p_core_A, p_core_B]) + epsilon * np.array([0.5, 0.5])

    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** This design contrasts Probabilistic Cue Sampling (PCS) with a Mixture of Take-The-Best and Tallying (MTTB) by exploiting their divergent handling of tied feature counts and evidence accumulation. We use five cues where the sum of the second and third validities exceeds the sum of the first and fourth (or fifth). In these trials, Option A possesses the single highest validity cue and one low validity cue, while Option B possesses the second and third highest validity cues, resulting in an equal number of winning cues for both options. Because the feature count is tied, Tallying is indifferent; however, TTB strongly favors Option A due to the top cue. Consequently, any mixture of TTB and Tallying must predict a preference for Option A (P(A) >= 0.5). In contrast, PCS samples cues proportionally to a softmax over validities. Because the combined sampling probability of the second and third cues exceeds that of the first and lower cues, PCS systematically predicts a preference for Option B (P(A) < 0.5). This provides a strict parameter-free qualitative dissociation between the two theories.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the proportion of trials where the subject chooses the option possessing the highest validity cue (Cue 1). According to the experimental design, Tallying is always indifferent on these trials due to tied feature counts, while Take-The-Best strongly favors the option with Cue 1. Thus, the Mixture of TTB and Tallying predicts a choice proportion for the Cue 1 option strictly >= 0.5. Conversely, Probabilistic Cue Sampling favors the opposing option because the combined sampling probability of the second and third cues (which the opposing option holds) outweighs the first cue and any remaining lower-validity cues, predicting a proportion < 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    a_c1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_c1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    chose_cue1 = ((a_c1 > b_c1) & chose_a) | ((b_c1 > a_c1) & chose_b)
    
    return float(chose_cue1.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.4908 (var=0.0025)
- Predicted under pi_6 (simulated): 0.5667 (var=0.0087)
- Observed on real data: 0.4942 (var=0.0012)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.98, 0.92, 0.86, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design contrasts a Mixture of Take-The-Best and Tallying against Probabilistic Cue Sampling (PCS) by exploiting trials where Tallying is indifferent but TTB and PCS strongly disagree. We use a 5-cue design where the sum of the sampling probabilities for the second and third cues exceeds the sum for the first and fourth (or fifth) cues under PCS's softmax, even at maximum gamma. In these critical trials, Option A possesses the highest-validity cue and one low-validity cue, while Option B possesses the second and third highest-validity cues. Because the number of winning cues is tied, Tallying is perfectly indifferent. TTB, however, deterministically favors Option A due to the top cue. Consequently, any mixture of TTB and Tallying must predict a preference for Option A (P(A) >= 0.5). In contrast, PCS samples cues proportionally to a softmax distribution; because the combined probability of the second and third cues is greater, PCS systematically predicts a preference for Option B (P(A) < 0.5). This provides a strict, parameter-free qualitative dissociation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By restricting the analysis exclusively to trials where the unweighted sum of features is tied (trials 1-6), we isolate the condition where the Tallying heuristic is perfectly indifferent. Under the Advocated Theory (Mixture of TTB and Tallying), TTB deterministically breaks the tie in favor of the option with the highest validity cue (Cue 0), so the mixture reliably predicts choosing the Cue 0 option (probability > 0.5). In contrast, Probabilistic Cue Sampling samples cues proportionally to their softmax validities; because the combined sampling probability for Cues 1 and 2 exceeds that of Cue 0 and any remaining low-validity cue, PCS systematically favors the opposing option (probability < 0.5). This creates a stark, parameter-free contrast in the subset of trials where Tallying noise is eliminated.

Source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    tally_tied = a_sums == b_sums
    
    tied_data = data[tally_tied]
    if len(tied_data) == 0:
        return 0.5
        
    a_has_cue0 = tied_data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_cue0 = tied_data['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    chose_a = tied_data['response'] == 0
    chose_b = tied_data['response'] == 1
    
    chose_cue0 = (chose_a & a_has_cue0) | (chose_b & b_has_cue0)
    return float(chose_cue0.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.4839 (var=0.0039)
- Predicted under pi_6 (simulated): 0.5458 (var=0.0075)
- Observed on real data: 0.5111 (var=0.0050)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_consistent = 0
    total_incongruent = 0
    
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        resp = row['response']
        
        # Incongruent trial 1: Option A has fewer but higher-validity features
        if a == [1, 1, 0, 0, 0] and b == [0, 0, 1, 1, 1]:
            total_incongruent += 1
            if resp == 1:  # Tallying prefers B (3 features > 2 features)
                tally_consistent += 1
        
        # Incongruent trial 4: Option B has fewer but higher-validity features
        elif a == [0, 0, 1, 1, 1] and b == [1, 1, 0, 0, 0]:
            total_incongruent += 1
            if resp == 0:  # Tallying prefers A (3 features > 2 features)
                tally_consistent += 1
                
    if total_incongruent == 0:
        return 0.5
    return tally_consistent / total_incongruent
```

**Observed (real) value:** 0.5067 (var=0.0125)
**Predicted under pi_3:** 0.4892 (var=0.0113)
**Predicted under pi_6:** 0.5175 (var=0.0167)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where one option has the two highest validity features [1, 1, 0, 0, 0]
    # and the other has the three lowest validity features [0, 0, 1, 1, 1].
    is_t1 = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)))
    is_t2 = (data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)))
    
    wadd_choices = 0
    total = 0
    
    if is_t1.any():
        wadd_choices += (data.loc[is_t1, 'response'] == 0).sum()
        total += is_t1.sum()
        
    if is_t2.any():
        wadd_choices += (data.loc[is_t2, 'response'] == 1).sum()
        total += is_t2.sum()
        
    return float(wadd_choices / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.4800 (var=0.0099)
**Predicted under pi_3:** 0.4967 (var=0.0119)
**Predicted under pi_6:** 0.5267 (var=0.0159)

### Experiment 5
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert the option ratings lists into strings for safe hashing and comparison
    a_strs = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_strs = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Create an order-independent trial type identifier. 
    # Sorting ensures that A vs B and B vs A are mapped to the same trial type.
    trial_types = [a + '_' + b if a < b else b + '_' + a for a, b in zip(a_strs, b_strs)]
    df = data.assign(trial_type=trial_types)
    
    # Calculate the empirical choice proportion (p) for each subject and trial type.
    # Since variance p*(1-p) is symmetric, it doesn't matter which option's proportion we measure.
    p = df.groupby(['subject_id', 'trial_type'])['response'].mean()
    
    # Calculate the intra-subject variance of choices for each trial type
    var = p * (1.0 - p)
    
    # Return the mean intra-subject variance across all subjects and trial types
    return float(var.mean())
```

**Observed (real) value:** 0.2357 (var=0.0001)
**Predicted under pi_3:** 0.2280 (var=0.0001)
**Predicted under pi_6:** 0.2292 (var=0.0002)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    df = data.copy()
    # Create hashable trial identifiers safely using list comprehensions
    df['trial_id'] = df['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) + \
                     "_" + \
                     df['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    subj_vars = []
    for subj, subj_df in df.groupby('subject_id'):
        # Proportion of choosing A (response == 0) for each trial type
        p_A = subj_df.groupby('trial_id')['response'].apply(lambda x: np.mean(x == 0))
        # Variance of these proportions across the 12 trial types
        subj_vars.append(p_A.var(ddof=1))
        
    return float(np.mean(subj_vars))
```

**Observed (real) value:** 0.0341 (var=0.0001)
**Predicted under pi_3:** 0.0334 (var=0.0002)
**Predicted under pi_6:** 0.0416 (var=0.0004)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # The experimental design is such that in EVERY trial, the first discriminating
    # cue (highest validity) favors option A, while the majority of lower-validity
    # cues favor option B.
    # STTB will systematically choose A (response = 0) because it terminates search
    # at the first discriminating cue.
    # PCS will sample lower-validity cues proportionally and therefore choose B (response = 1)
    # much more frequently.
    # Thus, the overall proportion of B choices directly discriminates the two theories.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5054 (var=0.0024)
**Predicted under pi_3:** 0.5713 (var=0.0028)
**Predicted under pi_6:** 0.4963 (var=0.0062)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    target_trials = [(1, 1, 0, 0, 0), (0, 1, 0, 0, 0), (1, 1, 1, 0, 0)]
    mask = data['A_tuple'].isin(target_trials)
    if mask.sum() == 0:
        return 0.5
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4892 (var=0.0034)
**Predicted under pi_3:** 0.3842 (var=0.0043)
**Predicted under pi_6:** 0.4900 (var=0.0161)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))

    chose_canonical = np.where(
        a_str < b_str,
        (data['response'] == 0).astype(float),
        (data['response'] == 1).astype(float)
    )
    
    pair_id = np.where(
        a_str < b_str,
        a_str + "_" + b_str,
        b_str + "_" + a_str
    )
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'pair_id': pair_id,
        'chose_canonical': chose_canonical
    })
    
    grouped = df.groupby(['subject_id', 'pair_id'])['chose_canonical'].mean().reset_index()
    subj_vars = grouped.groupby('subject_id')['chose_canonical'].var(ddof=0)
    
    return float(subj_vars.mean())
```

**Observed (real) value:** 0.0179 (var=0.0001)
**Predicted under pi_3:** 0.0222 (var=0.0001)
**Predicted under pi_6:** 0.0194 (var=0.0001)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Project lists to tuples for hashability
    data['a_tuple'] = data['option_a_ratings'].apply(tuple)
    
    # Trial 6: A has top cue + cues 1,2. B has cues 3,4.
    t6 = (1, 1, 1, 0, 0)
    # Trial 7: A has cues 1,2. B has top cue + cues 3,4.
    t7 = (0, 1, 1, 0, 0)
    
    # Filter data to only these two trial types
    mask = data['a_tuple'].isin([t6, t7])
    df = data[mask].copy()
    
    # Calculate choice for A (0 means chose A)
    df['chose_a'] = (df['response'] == 0).astype(float)
    
    # Compute P(A|T6) - P(A|T7) per subject
    def subject_diff(subj_df):
        means = subj_df.groupby('a_tuple')['chose_a'].mean()
        p6 = means.get(t6, 0.5)
        p7 = means.get(t7, 0.5)
        return float(p6 - p7)
        
    diffs = df.groupby('subject_id').apply(subject_diff)
    return float(diffs.mean())

```

**Observed (real) value:** 0.0133 (var=0.0298)
**Predicted under pi_3:** 0.2400 (var=0.0355)
**Predicted under pi_6:** 0.2533 (var=0.0742)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Neither theory adequately explains the observed data across all experiments. While Theory 1 (PCS) captures the near-indifference in Experiment 1 better than Theory 2 (MTTB), it fails significantly in Experiments 7 and 8 where Theory 2 performs well. Furthermore, both theories drastically fail on Experiment 10, predicting a difference of ~0.25 when the observed difference is nearly zero (0.0133). This systematic failure suggests that subjects are not cleanly using either a single sampled cue (PCS) or a mixture of heuristics (MTTB). Instead, subjects appear to exhibit much softer, heavily regularized integration of cues that heavily discounts marginal differences in cue counts or validities.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory based on Parallel Evidence Accumulation with Leaky Integration (e.g., a simplified Leaky Competing Accumulator). In this theory, all features are evaluated in parallel, and evidence for each option is accumulated based on cue validities. However, strong lateral inhibition or evidence leakage causes the system to be highly insensitive to small net differences in evidence (explaining the near-zero difference in Exp 10 and the near-0.5 choice proportions in Exps 1, 2, 7, and 8). This differs from PCS (which samples single cues) and MTTB (which relies on strict heuristic rules), providing a continuous, compensatory mechanism that naturally dampens overconfidence."
}
```

## Usage

```json
{
  "prompt_token_count": 9361,
  "candidates_token_count": 329,
  "total_token_count": 10685
}
```
