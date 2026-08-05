# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_4
Probabilistic Cue Search (PCS): Decision makers evaluate options by searching through cues sequentially. However, unlike the strict deterministic order of Take-The-Best, the search order is probabilistic. The probability of retrieving and evaluating a cue is proportional to an exponential function of its validity (a Plackett-Luce choice process). Because non-discriminating cues simply prompt the decision maker to continue searching, the probability that the decision is ultimately determined by a specific discriminating cue is exactly proportional to its weight relative to all other discriminating cues. This model elegantly bridges random search (Tallying) and deterministic search (TTB) through a single search-temperature parameter.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Find discriminating cues
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    if not np.any(discrim_mask):
        p_core = np.array([0.5, 0.5])
    else:
        # Due to the Plackett-Luce property of sampling without replacement,
        # the probability that a particular discriminating cue is encountered 
        # before any other discriminating cue is simply its proportional weight
        # among the set of discriminating cues. Non-discriminating cues drop out.
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        # Calculate weights for discriminating cues with numerical stability
        weights = np.zeros_like(validities)
        weights[discrim_mask] = np.exp(gamma * (validities[discrim_mask] - max_v))
        
        w_A = np.sum(weights[diff > 0])
        w_B = np.sum(weights[diff < 0])
        
        total_w = w_A + w_B
        if total_w == 0:
            p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([w_A / total_w, w_B / total_w])
            
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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 9: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To maximally distinguish Take-The-Best (TTB) from Probabilistic Cue Search (PCS), we exploit how the two models handle lower-validity discriminating cues. TTB is strictly non-compensatory: its choice probabilities depend only on the single highest-validity discriminating cue, meaning the number and strength of lower-validity cues favoring the alternative option have zero effect. In contrast, PCS computes choice probabilities based on the relative weights of all discriminating cues (a Plackett-Luce choice rule). By holding the highest-validity discriminating cue constant while systematically increasing the number of lower-validity cues favoring the alternative option, we create a strong quantitative dissociation: TTB predicts flat choice probabilities across these trials, whereas PCS predicts a decreasing preference for the option favored by the top cue as the competing cues accumulate.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
By partitioning trials into 'low opposing' (0-1 cues favoring the alternative) and 'high opposing' (3-4 cues favoring the alternative), we isolate the core divergence between the models. TTB ignores all lower-validity cues, predicting an identical match rate across both partitions (difference = 0). PCS dynamically accumulates evidence, predicting a substantial drop in match rate for the 'high opposing' partition. Pooling across multiple trials dramatically reduces binomial variance compared to single-trial contrasts.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): -0.0256 (var=0.0049)
- Predicted under pi_4 (simulated): 0.0683 (var=0.0122)
- Observed on real data: -0.0033 (var=0.0067)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 8: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 9: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To maximally distinguish Take-The-Best (TTB) from Probabilistic Cue Search (PCS), we systematically vary the number and validity of lower-validity cues that oppose the highest-validity discriminating cue. TTB is strictly non-compensatory: its choice is determined entirely by the top discriminating cue, meaning any lower-validity cues favoring the alternative option are ignored and have no effect on choice probabilities (beyond fixed lapse/softmax noise). In contrast, PCS uses a Plackett-Luce choice rule over all discriminating cues. Thus, as more lower-validity cues stack up against the top cue, PCS predicts a graded decrease in the probability of choosing the option favored by the top cue. By including trials where the top cue is opposed by one, two, three, or four lower-validity cues, we create a strong quantitative dissociation between the step-function prediction of TTB and the graded-integration prediction of PCS.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses the option favored by the simple majority of discriminating cues, specifically on trials where this majority is opposed by the single most valid discriminating cue. Take-The-Best (TTB) is strictly non-compensatory and will virtually never choose the majority option in these trials (only doing so via random noise). Probabilistic Cue Search (PCS), however, evaluates cues probabilistically and will integrate the multiple opposing cues, leading to a significantly higher rate of choosing the majority option.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.1414 (var=0.0113)
- Predicted under pi_4 (simulated): 0.2632 (var=0.0287)
- Observed on real data: 0.1543 (var=0.0099)

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
**Predicted under pi_4:** 0.1512 (var=0.0213)

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
**Predicted under pi_4:** 0.8258 (var=0.0142)

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
**Predicted under pi_4:** 0.1750 (var=0.0157)

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
**Predicted under pi_4:** 0.8383 (var=0.0134)

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
**Predicted under pi_4:** 0.7540 (var=0.0179)

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
**Predicted under pi_4:** 0.3367 (var=0.0090)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 1 (Take-The-Best, pi_3) consistently outperforms Theory 2 (Probabilistic Cue Search, pi_4). In Experiment 1, the observed difference between low- and high-opposing trials is effectively zero (-0.0033), which closely matches the non-compensatory prediction of TTB and contradicts the graded integration prediction of PCS (0.0683). Experiment 2 similarly shows a low rate of compensatory choices (0.1543), closely aligning with TTB (0.1414) and rejecting PCS (0.2632). Experiment 7 further confirms that when TTB and WADD disagree, subjects overwhelmingly choose the TTB option (0.8833), which TTB captures much better than PCS. While PCS fits slightly better on Experiment 8, TTB provides a more robust and parsimonious account across the full suite of experiments. Because TTB is already highly successful, Theory 2 should be replaced with a brand-new theory that acts as a stronger competitor to TTB.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory based on a 'Two-Stage' or 'Strategy Selection' framework. Instead of assuming a probabilistic search over all cues, this theory should propose that subjects first attempt a fast Tallying (unit-weight) strategy. If Tallying yields a clear winner (e.g., a difference of 2 or more cues), they choose it; if it yields a tie or a weak difference, they fall back to a strict Take-The-Best strategy to break the tie. Alternatively, propose a model where individuals use TTB by default but switch to a compensatory strategy only when the top discriminating cue's validity falls below a certain confidence threshold. This maintains the largely non-compensatory nature of the observed data while offering a distinct mechanistic alternative to pure TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 9020,
  "candidates_token_count": 429,
  "total_token_count": 10286
}
```
