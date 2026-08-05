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
People compare two options using the Take-The-Best (TTB) heuristic. TTB is a fast-and-frugal lexicographic strategy that searches through cues in descending order of their validity. The first cue that discriminates between the two options completely determines the choice, and all remaining lower-validity cues are ignored. If no cue discriminates, the decision maker guesses randomly. Response noise enters through a softmax over the binary TTB outcome with inverse temperature beta, plus an independent lapse rate epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_5
Two-Stage Confidence-Threshold Strategy Selection: Decision-makers default to the fast and frugal Take-The-Best (TTB) heuristic, evaluating options based solely on the most valid discriminating cue. However, if the validity of this top discriminating cue falls below a subjective confidence threshold, the decision-maker deems the single-cue evidence insufficient and falls back to a compensatory Weighted Additive (WADD) strategy that integrates all available cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    confidence_threshold = float(parameters["confidence_threshold"])
    
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    scores = np.zeros(2)
    if np.any(discrim_mask):
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        if max_v >= confidence_threshold:
            # Strategy 1: Take-The-Best (TTB)
            top_idx = np.where((discrim_mask) & (validities == max_v))[0][0]
            if stim[0, top_idx] > stim[1, top_idx]:
                scores[0] = 1.0
            else:
                scores[1] = 1.0
        else:
            # Strategy 2: Weighted Additive (WADD) fallback
            wadd_scores = stim @ validities
            if wadd_scores[0] > wadd_scores[1]:
                scores[0] = 1.0
            elif wadd_scores[1] > wadd_scores[0]:
                scores[1] = 1.0
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.9, 0.75, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 8: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 9: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 11: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Rationale:** To maximally distinguish Theory 1 (pure Take-The-Best) from Theory 2 (Two-Stage Confidence-Threshold Strategy Selection), we must exploit the confidence threshold mechanism in Theory 2. Theory 2 defaults to TTB but falls back to a compensatory Weighted Additive (WADD) rule if the highest discriminating cue's validity is below a subjective confidence threshold (which lies between 0.5 and 0.8). We use 6 features with validities spanning from 0.90 down to 0.50. We design critical trials where the highest discriminating cue has a validity of 0.90, 0.75, 0.65, or 0.60. In these trials, the top discriminating cue favors one option (which TTB will pick), while the sum of the remaining lower-validity cues strongly favors the other option (which WADD will pick). For the 0.90 cue, both theories will confidently use TTB and agree. However, for the 0.75, 0.65, and 0.60 cues, Theory 2 may fall below its confidence threshold and switch to WADD, predicting the opposite choice of Theory 1. By varying the validity of the top discriminating cue across trials, we can precisely identify if and where a subject's strategy shifts from non-compensatory (TTB) to compensatory (WADD).

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses the option predicted by Take-The-Best (TTB) specifically on trials where TTB and the Weighted Additive (WADD) strategy disagree, and the top discriminating cue's validity is strictly less than 0.9. According to Theory 1, subjects will consistently use TTB, so this metric should be close to 1.0 (subject to decision noise). According to Theory 2, subjects will fall back to WADD if the top cue's validity is below their subjective confidence threshold (which ranges from 0.5 to 0.8). By filtering for max_v < 0.9, we isolate the trials where Theory 2 subjects are most likely to switch to WADD, resulting in a substantially lower TTB-adherence rate compared to Theory 1.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.9, 0.75, 0.65, 0.6, 0.55, 0.5])
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    ttb_match = 0
    valid_count = 0
    
    for i in range(len(responses)):
        d = a_ratings[i] - b_ratings[i]
        nonzero = np.where(d != 0)[0]
        if len(nonzero) == 0:
            continue
            
        top_idx = nonzero[0]
        ttb_pred = 0 if d[top_idx] > 0 else 1
        max_v = validities[top_idx]
        
        wadd_a = np.sum(a_ratings[i] * validities)
        wadd_b = np.sum(b_ratings[i] * validities)
        if wadd_a == wadd_b:
            continue
        wadd_pred = 0 if wadd_a > wadd_b else 1
        
        # Focus on trials where TTB and WADD disagree, and top cue validity < 0.9
        if ttb_pred != wadd_pred and max_v < 0.9:
            valid_count += 1
            if responses[i] == ttb_pred:
                ttb_match += 1
                
    if valid_count == 0:
        return 0.5
        
    return float(ttb_match / valid_count)

### RESULTS
- Predicted under pi_3 (simulated): 0.8503 (var=0.0087)
- Predicted under pi_5 (simulated): 0.5619 (var=0.0643)
- Observed on real data: 0.8250 (var=0.0092)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.95, 0.75, 0.65, 0.58, 0.54, 0.51]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 7: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Rationale:** To maximally distinguish Theory 1 (Two-Stage Confidence-Threshold Strategy Selection) from Theory 2 (pure Take-The-Best), we exploit the confidence threshold mechanism in Theory 1. Theory 1 uses TTB if the maximum discriminating validity is above a subjective threshold (between 0.5 and 0.8), but falls back to a compensatory Weighted Additive (WADD) rule if it falls below. Theory 2 always uses TTB. We use 6 features with validities ranging from 0.95 to 0.51. We design critical dissociation trials where the highest discriminating validity is systematically varied (0.95, 0.75, 0.65, 0.58). In these trials, the top discriminating cue favors Option A (which TTB will pick), while the sum of the remaining lower-validity cues favors Option B (which WADD will pick). For the 0.95 cue, both theories will confidently use TTB and agree on Option A. However, for the 0.75, 0.65, and 0.58 cues, Theory 1 may fall below its confidence threshold and switch to WADD (picking Option B), while Theory 2 will continue to pick Option A. By observing at which validity level the choice switches from A to B, we can precisely identify the presence and value of the confidence threshold.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Option A is supported by exactly one cue (which is not the highest validity cue of 0.95), while Option B is supported by multiple lower-validity cues such that the WADD score favors B. Theory 2 (pure TTB) will consistently choose Option A (response = 0). Theory 1, however, will often evaluate the top cue's validity (<= 0.75) as falling below its confidence threshold and switch to WADD, thereby choosing Option B (response = 1). Thus, the mean response on these target trials will be significantly higher for Theory 1 than for Theory 2.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where option A is favored only by a single cue (which is not the most valid cue overall)
    # These are trials where TTB picks A, but WADD picks B, and the max validity is <= 0.75.
    is_target = data['option_a_ratings'].apply(lambda x: sum(x) == 1 and x[0] == 0)
    if not is_target.any():
        return 0.0
    return float(data.loc[is_target, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1631 (var=0.0121)
- Predicted under pi_5 (simulated): 0.4102 (var=0.0669)
- Observed on real data: 0.1573 (var=0.0116)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    score = 0
    count = 0
    for _, row in data.iterrows():
        A = tuple(row['option_a_ratings'])
        B = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers B (3 to 2 wins), WADD prefers A (1.85 to 1.65)
        if A == (1, 1, 0, 0, 0) and B == (0, 0, 1, 1, 1):
            score += (resp == 1)
            count += 1
        # Trial 8: Tallying prefers A (2 to 1 wins), WADD prefers B (0.95 to 1.15)
        elif A == (0, 0, 1, 1, 0) and B == (1, 0, 0, 0, 0):
            score += (resp == 0)
            count += 1
            
    if count == 0:
        return 0.5
    return float(score / count)
```

**Observed (real) value:** 0.1625 (var=0.0238)
**Predicted under pi_3:** 0.1525 (var=0.0155)
**Predicted under pi_5:** 0.1812 (var=0.0183)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    trial_1 = (a_tup == (1, 1, 0, 0, 0)) & (b_tup == (0, 0, 1, 1, 1))
    trial_2 = (a_tup == (0, 0, 1, 1, 1)) & (b_tup == (1, 1, 0, 0, 0))
    
    t1_wadd_aligned = (data['response'] == 0) & trial_1
    t2_wadd_aligned = (data['response'] == 1) & trial_2
    
    wadd_choices = t1_wadd_aligned.sum() + t2_wadd_aligned.sum()
    total_dissociation = trial_1.sum() + trial_2.sum()
    
    return float(wadd_choices / total_dissociation) if total_dissociation > 0 else 0.5
```

**Observed (real) value:** 0.8217 (var=0.0115)
**Predicted under pi_3:** 0.8517 (var=0.0129)
**Predicted under pi_5:** 0.8517 (var=0.0092)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_choices = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        a_str = ''.join(map(str, a))
        b_str = ''.join(map(str, b))
        
        is_t1 = (a_str == '11000' and b_str == '00111')
        is_t2 = (a_str == '00111' and b_str == '11000')
        
        if is_t1:
            tally_choices.append(1 if row['response'] == 1 else 0)
        elif is_t2:
            tally_choices.append(1 if row['response'] == 0 else 0)
            
    if not tally_choices:
        return 0.5
        
    return float(np.mean(tally_choices))
```

**Observed (real) value:** 0.1200 (var=0.0109)
**Predicted under pi_3:** 0.1575 (var=0.0131)
**Predicted under pi_5:** 0.1775 (var=0.0165)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target = [1, 1, 0, 0, 0, 0]
    competitor = [0, 0, 1, 1, 1, 0]
    
    def is_target_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        return (a == target and b == competitor) or (a == competitor and b == target)
        
    mask = data.apply(is_target_trial, axis=1)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    def chose_target(row):
        a = list(row['option_a_ratings'])
        chose_a = (row['response'] == 0)
        return 1.0 if (a == target) == chose_a else 0.0
        
    return float(subset.apply(chose_target, axis=1).mean())
```

**Observed (real) value:** 0.8567 (var=0.0164)
**Predicted under pi_3:** 0.8517 (var=0.0167)
**Predicted under pi_5:** 0.8250 (var=0.0226)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    ttb_match = 0
    disagree_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'], dtype=float)
        b = np.array(row['option_b_ratings'], dtype=float)
        
        # TTB choice
        ttb_choice = -1
        for i in range(5):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice == -1:
            continue
            
        # WADD choice
        wadd_a = np.dot(a, validities)
        wadd_b = np.dot(b, validities)
        if wadd_a == wadd_b:
            continue
        wadd_choice = 0 if wadd_a > wadd_b else 1
        
        # Only consider trials where the two models fundamentally disagree
        if ttb_choice != wadd_choice:
            disagree_count += 1
            if row['response'] == ttb_choice:
                ttb_match += 1
                
    if disagree_count == 0:
        return 0.5
    return float(ttb_match / disagree_count)
```

**Observed (real) value:** 0.8833 (var=0.0156)
**Predicted under pi_3:** 0.8257 (var=0.0098)
**Predicted under pi_5:** 0.8263 (var=0.0127)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.65, 0.95, 0.55, 0.85, 0.75])
    
    def is_wadd_choice(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        score_a = np.dot(a, validities)
        score_b = np.dot(b, validities)
        pred = 0 if score_a > score_b else 1
        return row['response'] == pred

    return float(data.apply(is_wadd_choice, axis=1).mean())
```

**Observed (real) value:** 0.3138 (var=0.0086)
**Predicted under pi_3:** 0.2778 (var=0.0048)
**Predicted under pi_5:** 0.3253 (var=0.0072)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    pair_str = a_str + "_" + b_str
    
    ttb_choices = {
        '10000_01000': 0,
        '10000_01100': 0,
        '10000_01110': 0,
        '10000_01111': 0,
        '11000_10100': 0,
        '11000_10111': 0,
        '11100_11010': 0,
        '11100_11011': 0,
        '11111_00000': 0,
        '01111_10000': 1
    }
    
    low_opposing = {
        '10000_01000',
        '11000_10100',
        '11100_11010',
        '11111_00000'
    }
    
    high_opposing = {
        '10000_01110',
        '10000_01111',
        '11000_10111',
        '01111_10000'
    }
    
    data['ttb_choice'] = pair_str.map(ttb_choices)
    data['is_ttb_match'] = (data['response'] == data['ttb_choice']).astype(float)
    
    is_low = pair_str.isin(low_opposing)
    is_high = pair_str.isin(high_opposing)
    
    low_match = data.loc[is_low, 'is_ttb_match'].mean()
    high_match = data.loc[is_high, 'is_ttb_match'].mean()
    
    if pd.isna(low_match) or pd.isna(high_match):
        return 0.0
        
    return float(low_match - high_match)
```

**Observed (real) value:** -0.0033 (var=0.0067)
**Predicted under pi_3:** -0.0256 (var=0.0049)
**Predicted under pi_5:** -0.0139 (var=0.0059)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    compensatory_choices = []
    
    for _, row in data.iterrows():
        A = row['option_a_ratings']
        B = row['option_b_ratings']
        
        diff = np.array(A) - np.array(B)
        discrim_mask = diff != 0
        if not np.any(discrim_mask):
            continue
            
        # Top cue is the first non-zero in diff (since validities are monotonically decreasing)
        top_idx = np.where(discrim_mask)[0][0]
        top_favors_A = (diff[top_idx] > 0)
        
        count_A = np.sum(diff > 0)
        count_B = np.sum(diff < 0)
        
        # Identify trials where the top cue opposes the simple majority of discriminating cues
        if top_favors_A and count_B > count_A:
            chosen_majority = (row['response'] == 1)
            compensatory_choices.append(chosen_majority)
        elif (not top_favors_A) and count_A > count_B:
            chosen_majority = (row['response'] == 0)
            compensatory_choices.append(chosen_majority)
            
    if not compensatory_choices:
        return 0.0
        
    return float(np.mean(compensatory_choices))
```

**Observed (real) value:** 0.1543 (var=0.0099)
**Predicted under pi_3:** 0.1414 (var=0.0113)
**Predicted under pi_5:** 0.1714 (var=0.0121)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (pure Take-The-Best) significantly outperforms Theory 2 (Two-Stage Confidence-Threshold). In Experiments 1 and 2, which were specifically designed to test the confidence threshold mechanism, the observed data strongly aligns with Theory 1's predictions. Subjects do not seem to fall back to a compensatory Weighted Additive (WADD) strategy when the top cue's validity is low; instead, they continue to adhere to the non-compensatory TTB heuristic. Theory 2 predicts far too much compensatory behavior. While Theory 1 is an excellent baseline, it slightly underpredicts deviations in some experiments (like Exp 7 and 8), suggesting that while the core TTB mechanism is correct, the noise or error process might be more nuanced than a simple lapse rate.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Probabilistic Cue Search' (or Probabilistic TTB) theory. Instead of a deterministic search order based strictly on validities, assume that the order in which cues are evaluated is probabilistic, weighted by their validities (e.g., a softmax over validities to pick the first cue to evaluate). Once a cue is selected, if it discriminates, it determines the choice. This maintains the fast-and-frugal, non-compensatory nature of TTB that fits the data so well, but allows for systematic deviations that depend on how close the cue validities are to one another, providing a more principled source of variability than a flat lapse rate."
}
```

## Usage

```json
{
  "prompt_token_count": 10987,
  "candidates_token_count": 346,
  "total_token_count": 12455
}
```
