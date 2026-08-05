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
Take The Best (TTB) Heuristic: Decision makers use a non-compensatory, lexicographic strategy to choose between options. They search through cues in descending order of subjective validity (or informativeness). The first cue that discriminates between the two options strictly determines the choice, and all remaining lower-validity cues are ignored. If no cues discriminate, the decision maker guesses. Response noise is modeled as a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity. 
    # We use a stable sort so that if validities are tied, left-to-right order is preserved.
    order = np.argsort(-validities, kind='stable')
    
    # Default to guessing if no cues discriminate
    p_core = np.array([0.5, 0.5])
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_6
Strategy Selection: Decision-makers predominantly rely on a non-compensatory Take The Best (TTB) heuristic but probabilistically mix in a simple compensatory Tallying strategy. Unlike validity-dependent lapse models, this mixture maintains flat adherence across different validities of discriminating cues, while allowing small but consistent deviations in choice probabilities when TTB and Tallying conflict. The probability of tallying is kept very small to align with the near-zero deviations observed in experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(-validities, kind='stable')
    p_ttb_choice = np.array([0.5, 0.5])
    for i in order:
        if a[i] > b[i]:
            p_ttb_choice = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_ttb_choice = np.array([0.0, 1.0])
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally_choice = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally_choice = np.array([0.0, 1.0])
    else:
        p_tally_choice = np.array([0.5, 0.5])
        
    p_tally = float(parameters["p_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Mixture of strategies
    p_core = (1.0 - p_tally) * p_ttb_choice + p_tally * p_tally_choice
    
    # Apply uniform response noise (lapse)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 11: A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the pure Take The Best (TTB) model from the Strategy Selection (TTB + Tallying) model, we exploit the compensatory nature of Tallying. Pure TTB relies exclusively on the highest-validity discriminating feature and predicts a constant choice probability for the favored option regardless of how many lower-validity features favor the alternative. In contrast, the Strategy Selection model assumes occasional lapses into Tallying. By keeping the TTB prediction constant but varying the Tallying prediction (Agreement, Conflict, or Tie), we can test whether the choice probability for the TTB-favored option drops systematically when Tallying favors the alternative. The trials vary the highest discriminating cue to ensure the effect is robust across different cue validities.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The Strategy Selection mixture model introduces a subtle but consistent bias: subjects make more 'errors' (deviations from TTB) on Conflict trials than on Agree trials. Attempt 1 failed because the standard error was computed using noisy sample variances from small trial counts, which artificially inflated the between-subject variance of the metric. To fix this, we compute a Z-statistic using a stable pooled variance, ensuring the single-subject metric has a tight variance of ~1.0 under pure TTB. We then scale the metric by the square root of the number of subjects; this allows the pooled point estimate to correctly reflect the amplified signal of the full sample, overcoming the heavy binomial noise that masks the effect in individual subjects.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # TTB choice (lexicographic)
    diff = a_ratings - b_ratings
    weights = 2 ** np.arange(diff.shape[1])[::-1]
    ttb_vals = diff.dot(weights)
    ttb_choice = np.where(ttb_vals > 0, 0, 1)
    
    # Tallying choice (compensatory)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_choice = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    responses = data['response'].values
    errors = (responses != ttb_choice).astype(float)
    
    agree_mask = (ttb_choice == tally_choice) & (ttb_choice != -1)
    conflict_mask = (ttb_choice != tally_choice) & (tally_choice != -1) & (ttb_choice != -1)
    
    if np.sum(agree_mask) == 0 or np.sum(conflict_mask) == 0:
        return 0.0
        
    agree_err = errors[agree_mask]
    conflict_err = errors[conflict_mask]
    
    mean_diff = np.mean(conflict_err) - np.mean(agree_err)
    
    # Compute stable standard error using the pooled overall error rate
    p_pool = np.mean(errors)
    if p_pool <= 0.0 or p_pool >= 1.0:
        return 0.0
        
    var_pool = p_pool * (1.0 - p_pool)
    se = np.sqrt(var_pool * (1.0 / len(agree_err) + 1.0 / len(conflict_err)))
    
    z_stat = mean_diff / se
    
    # Scale by sqrt(N) to project the pooled statistical power
    n_subj = data['subject_id'].nunique()
    
    return float(z_stat * np.sqrt(n_subj))

### RESULTS
- Predicted under pi_3 (simulated): -0.4917 (var=0.9874)
- Predicted under pi_6 (simulated): 4.7706 (var=1.1655)
- Observed on real data: -0.1642 (var=0.6574)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Strategy Selection model (TTB + Tallying) from pure TTB, we exploit the compensatory nature of Tallying. Pure TTB relies exclusively on the highest-validity discriminating feature, predicting a constant choice probability for the favored option regardless of how many lower-validity features favor the alternative. The Strategy Selection model, however, assumes occasional lapses into a Tallying strategy. By keeping the TTB prediction constant but varying whether Tallying agrees with, opposes, or ties with TTB, we can observe whether the choice probability for the TTB-favored option systematically drops when Tallying strongly favors the alternative. We also vary the highest discriminating cue to ensure the effect is robust.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
To reliably discriminate the two theories despite the extremely small mixture probability of Tallying (p_tally <= 0.05) and high binomial noise, we compute a cumulative difference score rather than an average. We count the number of TTB-consistent choices on trials where TTB and Tallying agree, scale it by 2 (since the design has exactly 32 agree trials and 64 conflict trials per subject), and subtract the TTB-consistent choices on conflict trials. For Pure TTB, the expected score is exactly 0 because uniform lapses affect both trial types equally. For Strategy Selection, the occasional use of Tallying selectively reduces TTB adherence on conflict trials, yielding a strictly positive expected score. By returning the sum over all rows, the metric acts as an extensive test statistic: its point estimate on the pooled data aggregates the signal across all subjects, overcoming the low per-subject effect size.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agree_ttb_match = 0
    conflict_ttb_match = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Take The Best (TTB) prediction (cues are ordered by validity)
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            tally_pred = -1
            
        if ttb_pred == -1:
            continue
            
        ttb_match = 1 if row['response'] == ttb_pred else 0
        
        if ttb_pred == tally_pred:
            agree_ttb_match += ttb_match
        else:
            conflict_ttb_match += ttb_match
            
    # Weight agree matches by 2 because there are exactly twice as many conflict trials as agree trials.
    # This cumulative score scales with the dataset size, amplifying the signal.
    return float(agree_ttb_match * 2 - conflict_ttb_match)

### RESULTS
- Predicted under pi_3 (simulated): 1.0000 (var=21.4596)
- Predicted under pi_6 (simulated): 27.0000 (var=15.5684)
- Observed on real data: 9.0000 (var=17.2704)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score = 0.0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying strongly prefers A, WADD generally prefers B
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            score += (1 - resp)
            count += 1
        # Trial 2: Tallying strongly prefers B, WADD generally prefers A
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            score += resp
            count += 1
        # Trial 3: Tallying is indifferent (50/50), WADD generally prefers B
        elif a == (0, 1, 0, 0, 1) and b == (1, 0, 1, 0, 0):
            score += (1 - resp)
            count += 1
        # Trial 4: Tallying is indifferent (50/50), WADD generally prefers A
        elif a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 0, 1):
            score += resp
            count += 1
            
    return float(score / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.1575 (var=0.0090)
**Predicted under pi_3:** 0.1037 (var=0.0066)
**Predicted under pi_6:** 0.1092 (var=0.0053)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert option A ratings to a string for easy matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Identify Trial 1: A = [0, 0, 1, 1, 1], B = [1, 1, 0, 0, 0]
    t1_mask = a_str == '00111'
    
    if t1_mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times the subject chose Option A (response == 0)
    return float((data.loc[t1_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1400 (var=0.0213)
**Predicted under pi_3:** 0.1263 (var=0.0110)
**Predicted under pi_6:** 0.1275 (var=0.0087)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_choice(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1
        
    ttb_choices = [get_ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    matches = (np.array(data['response']) == np.array(ttb_choices))
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8538 (var=0.0067)
**Predicted under pi_3:** 0.8667 (var=0.0052)
**Predicted under pi_6:** 0.8827 (var=0.0037)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred is not None:
            matches.append(1 if resp == ttb_pred else 0)
            
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.8292 (var=0.0119)
**Predicted under pi_3:** 0.8700 (var=0.0069)
**Predicted under pi_6:** 0.8992 (var=0.0044)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Agreement trials (where TTB and Tallying agree)
    # Trial 2: A='11100', B='00010' -> Both choose A (response 0)
    # Trial 5: A='00010', B='11100' -> Both choose B (response 1)
    t2_mask = (a_str == '11100') & (b_str == '00010')
    t5_mask = (a_str == '00010') & (b_str == '11100')
    
    errors = 0
    total_agree = 0
    if t2_mask.sum() > 0:
        errors += (data.loc[t2_mask, 'response'] == 1).sum()
        total_agree += t2_mask.sum()
    if t5_mask.sum() > 0:
        errors += (data.loc[t5_mask, 'response'] == 0).sum()
        total_agree += t5_mask.sum()
        
    e_agree = errors / total_agree if total_agree > 0 else 0
    
    # Conflict trials (where TTB and Tallying completely disagree)
    # Trial 1: A='10000', B='01110' -> TTB chooses A (0), Tallying chooses B (1)
    # Trial 4: A='01110', B='10000' -> TTB chooses B (1), Tallying chooses A (0)
    t1_mask = (a_str == '10000') & (b_str == '01110')
    t4_mask = (a_str == '01110') & (b_str == '10000')
    
    tally_choices = 0
    total_conflict = 0
    if t1_mask.sum() > 0:
        tally_choices += (data.loc[t1_mask, 'response'] == 1).sum()
        total_conflict += t1_mask.sum()
    if t4_mask.sum() > 0:
        tally_choices += (data.loc[t4_mask, 'response'] == 0).sum()
        total_conflict += t4_mask.sum()
        
    p_tally = tally_choices / total_conflict if total_conflict > 0 else 0
    
    # Subtracting agreement error rate controls for the lapse rate (epsilon)
    return float(p_tally - e_agree)
```

**Observed (real) value:** -0.0062 (var=0.0059)
**Predicted under pi_3:** 0.0256 (var=0.0083)
**Predicted under pi_6:** 0.0250 (var=0.0057)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements_match = []
    conflicts_match = []
    
    for _, row in data.iterrows():
        A = row['option_a_ratings']
        B = row['option_b_ratings']
        resp = row['response']
        
        # TTB prediction (cues are already ordered by validity 0 to 4)
        ttb_choice = -1
        for i in range(5):
            if A[i] > B[i]:
                ttb_choice = 0
                break
            elif B[i] > A[i]:
                ttb_choice = 1
                break
                
        # Tallying prediction
        a_wins = sum(1 for a, b in zip(A, B) if a > b)
        b_wins = sum(1 for a, b in zip(A, B) if b > a)
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            tally_choice = -1
            
        if ttb_choice == -1:
            continue
            
        is_match = 1 if resp == ttb_choice else 0
        
        if ttb_choice == tally_choice:
            agreements_match.append(is_match)
        elif tally_choice != -1 and ttb_choice != tally_choice:
            conflicts_match.append(is_match)
            
    if not agreements_match or not conflicts_match:
        return 0.0
        
    return float(np.mean(agreements_match) - np.mean(conflicts_match))
```

**Observed (real) value:** 0.0070 (var=0.0102)
**Predicted under pi_3:** -0.0076 (var=0.0133)
**Predicted under pi_6:** 0.0343 (var=0.0102)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_disc_cue(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                return i
        return -1
        
    def get_ttb_favored(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                return 0 if a[i] > b[i] else 1
        return -1

    disc_cues = data.apply(get_disc_cue, axis=1)
    ttb_favored = data.apply(get_ttb_favored, axis=1)
    
    adherence = (data['response'] == ttb_favored).astype(float)
    
    high_val_mask = disc_cues.isin([0, 1])
    low_val_mask = disc_cues.isin([3, 4])
    
    if high_val_mask.sum() == 0 or low_val_mask.sum() == 0:
        return 0.0
        
    high_val_adherence = adherence[high_val_mask].mean()
    low_val_adherence = adherence[low_val_mask].mean()
    
    return float(high_val_adherence - low_val_adherence)
```

**Observed (real) value:** -0.0133 (var=0.0058)
**Predicted under pi_3:** -0.0028 (var=0.0057)
**Predicted under pi_6:** -0.0017 (var=0.0052)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_disc_cue(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                return i
        return -1

    cues = data.apply(get_disc_cue, axis=1)
    # In all designed trials, option A is favored by the highest discriminating cue.
    # Thus, '1 - response' computes the TTB accuracy (1 if chose A, 0 if chose B).
    acc = 1.0 - data['response']
    
    high_val = acc[cues.isin([0, 1])]
    low_val = acc[cues.isin([3, 4])]
    
    if len(high_val) == 0 or len(low_val) == 0:
        return 0.0
        
    return float(high_val.mean() - low_val.mean())
```

**Observed (real) value:** -0.0079 (var=0.0084)
**Predicted under pi_3:** 0.0090 (var=0.0053)
**Predicted under pi_6:** -0.0117 (var=0.0056)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both new experiments and the historical ones, the observed data consistently shows little to no evidence of a compensatory Tallying strategy. In Experiment 1, the observed Z-statistic comparing conflict and agree trials was near zero (-0.16), aligning closely with pure Take The Best (Theory 1) and firmly rejecting the TTB + Tallying mixture (Theory 2) which predicted a large positive value. In Experiment 2, the cumulative score was 9.0, which is closer to Theory 1's prediction of 1.0 than Theory 2's 27.0. Experiments 7 and 8 further confirm that deviations from TTB do not systematically align with Tallying predictions. Theory 1 (pure TTB with a uniform lapse) is a very strong baseline, while Theory 2's assumption of a Tallying mixture is degenerate given the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory to replace the Strategy Selection (TTB + Tallying) mixture. Since the data strongly supports a non-compensatory lexicographic process with no significant Tallying influence, the new theory should explore variations of cue-based processing. For example, consider a Probabilistic Cue Search model where decision-makers do not always search in strict descending order of validity, but instead sample cues probabilistically weighted by their validities; or a Sequential Evidence Accumulation model where cues are processed one by one but the threshold for a decision depends on the validity of the cue. This would provide a more nuanced alternative to pure TTB without relying on compensatory tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 10570,
  "candidates_token_count": 363,
  "total_token_count": 11877
}
```
