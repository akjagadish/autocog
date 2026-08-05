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
Take-The-Best (TTB) heuristic: People make decisions between multi-attribute options by ranking features according to their validities and comparing the options lexicographically. The decision-maker examines the feature with the highest validity first; if one option is strictly better on that feature, it is chosen immediately and no further features are considered. If there is a tie, they move to the second most valid feature, and so on. This non-compensatory, one-reason decision making process implies that a single highly valid cue can completely determine the choice, overriding any number of smaller advantages on less valid cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    choice = None
    for idx in order:
        if a[idx] > b[idx]:
            choice = 0
            break
        elif b[idx] > a[idx]:
            choice = 1
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = 2
    
    if choice is None:
        p_core = np.ones(n_opts) / n_opts
    else:
        p_core = np.zeros(n_opts)
        p_core[choice] = 1.0
        
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## THEORY 2 — pi_4
Probabilistic Take-The-Best (PTTB): People evaluate cues sequentially to make decisions, but the sequence is stochastic rather than strictly deterministic. The probability of sampling a cue is proportional to an exponential function of its subjective validity. When sampling without replacement, the probability that the first discriminating cue encountered is cue $i$ is exactly proportional to its weight among all discriminating cues. This model bridges Tallying (when temperature is high/gamma is 0, all cues are equally likely to be the deciding one) and deterministic Take-The-Best (when gamma is large, the highest validity cue almost always determines the choice), providing a nuanced non-compensatory heuristic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("PTTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify cues that discriminate between the two options
    discrim = (a != b)
    
    if not np.any(discrim):
        # No discriminating cues, guess uniformly
        p_core = np.array([0.5, 0.5])
    else:
        # The probability of a discriminating cue being the first one found
        # is proportional to exp(gamma * validity).
        val_discrim = validities[discrim]
        # Max-subtraction for numerical stability
        z = gamma * (val_discrim - np.max(val_discrim))
        w = np.exp(z)
        w = w / np.sum(w)
        
        # The probability of choosing an option is the sum of the probabilities
        # of the discriminating cues that favor it.
        p_a = np.sum(w[a[discrim] > b[discrim]])
        p_b = np.sum(w[b[discrim] > a[discrim]])
        
        p_core = np.array([p_a, p_b])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.75, 0.57, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  trial 4: A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  trial 5: A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 1]  B=[0, 0, 1, 1]
  trial 7: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 8: A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  trial 9: A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  trial 10: A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  trial 11: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 12: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 13: A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  trial 14: A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  trial 15: A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the rate at which subjects choose the option favored by the Take-The-Best (TTB) heuristic between two types of trials: 'unanimous' trials (where all discriminating cues point to the same option) and 'conflict' trials (where some lesser cues point to the opposite option). For the TTB model, the presence of conflicting lesser cues is entirely ignored, meaning the probability of choosing the TTB option is identical across both trial types (yielding an expected difference of 0, governed only by the constant lapse rate epsilon). For the Probabilistic TTB (PTTB) model, conflicting cues attract choice probability proportional to their validities, meaning the model will choose the TTB option less frequently in conflict trials than in unanimous trials (yielding an expected difference strictly greater than 0).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def analyze_row(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb = None
        for idx in [0, 1, 2, 3]:
            if a[idx] > b[idx]:
                ttb = 0
                break
            elif b[idx] > a[idx]:
                ttb = 1
                break
        if ttb is None:
            ttb = 0
            
        diff = [a[i] - b[i] for i in range(4)]
        signs = [d for d in diff if d != 0]
        unanimous = all(s > 0 for s in signs) or all(s < 0 for s in signs)
        
        return pd.Series({'match': resp == ttb, 'unanimous': unanimous})
        
    res = data.apply(analyze_row, axis=1)
    
    mean_unanimous = res[res['unanimous']]['match'].mean()
    mean_conflict = res[~res['unanimous']]['match'].mean()
    
    if pd.isna(mean_unanimous) or pd.isna(mean_conflict):
        return 0.0
        
    return float(mean_unanimous - mean_conflict)

### RESULTS
- Predicted under pi_3 (simulated): 0.0162 (var=0.0047)
- Predicted under pi_4 (simulated): 0.0428 (var=0.0074)
- Observed on real data: 0.0176 (var=0.0070)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.68, 0.88, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 9: A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 11: A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  trial 13: A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  trial 14: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 15: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 16: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject's choice aligns with the highest validity cue (Cue 0, validity 0.95) specifically on trials where it is opposed by the second highest validity cue (Cue 2, validity 0.88). TTB strictly follows the highest validity cue, so it will almost always choose the option favored by Cue 0 on these trials. PTTB, however, samples cues probabilistically based on their validities. Because 0.95 and 0.88 are very close, PTTB has a substantial probability of sampling Cue 2 first and thus choosing the option favored by Cue 2, leading to a visibly lower alignment with Cue 0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # Cue 0 has the highest validity (0.95), Cue 2 has the second highest (0.88)
    cue0_diff = a_ratings[:, 0] - b_ratings[:, 0]
    cue2_diff = a_ratings[:, 2] - b_ratings[:, 2]
    
    # Find trials where Cue 0 and Cue 2 disagree
    disagree = (cue0_diff != 0) & (cue2_diff != 0) & (cue0_diff != cue2_diff)
    
    if np.sum(disagree) == 0:
        return 0.5
        
    # TTB predicts the option favored by Cue 0 (the highest validity cue)
    ttb_pred = (cue0_diff[disagree] < 0).astype(int)
    resp = data['response'].values[disagree]
    
    return float(np.mean(ttb_pred == resp))

### RESULTS
- Predicted under pi_3 (simulated): 0.8720 (var=0.0069)
- Predicted under pi_4 (simulated): 0.7027 (var=0.0111)
- Observed on real data: 0.8307 (var=0.0176)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.82, 0.84, 0.55])
    
    match_count = 0
    tie_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Identify trials where Tallying predicts a tie
        if a_wins == b_wins and a_wins > 0:
            tie_count += 1
            score_a = np.sum(a * val)
            score_b = np.sum(b * val)
            
            # Check if response aligns with WADD's strict preference
            if score_a > score_b and row['response'] == 0:
                match_count += 1
            elif score_b > score_a and row['response'] == 1:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
        
    return float(match_count / tie_count)
```

**Observed (real) value:** 0.8422 (var=0.0120)
**Predicted under pi_3:** 0.8789 (var=0.0091)
**Predicted under pi_4:** 0.7300 (var=0.0142)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 4: A=[0, 0, 1, 1], B=[0, 1, 1, 0]
    # A wins on feature 4 (validity 0.55), B wins on feature 2 (validity 0.78)
    mask4 = (a_str == '0011') & (b_str == '0110')
    
    # Trial 9: A=[1, 0, 1, 0], B=[0, 0, 1, 1]
    # A wins on feature 1 (validity 0.95), B wins on feature 4 (validity 0.55)
    mask9 = (a_str == '1010') & (b_str == '0011')
    
    p_b_4 = data.loc[mask4, 'response'].mean()
    p_b_9 = data.loc[mask9, 'response'].mean()
    
    if pd.isna(p_b_4):
        p_b_4 = 0.5
    if pd.isna(p_b_9):
        p_b_9 = 0.5
        
    return float(p_b_4 - p_b_9)
```

**Observed (real) value:** 0.7000 (var=0.0889)
**Predicted under pi_3:** 0.7567 (var=0.0413)
**Predicted under pi_4:** 0.7000 (var=0.0844)

### Experiment 5
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.95, 0.77, 0.8, 0.55])
    order = np.argsort(validities)[::-1]

    match_count = 0
    total = 0

    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']

        ttb_choice = None
        for idx in order:
            if a[idx] > b[idx]:
                ttb_choice = 0
                break
            elif b[idx] > a[idx]:
                ttb_choice = 1
                break
        
        if ttb_choice is not None:
            if resp == ttb_choice:
                match_count += 1
            total += 1

    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.8583 (var=0.0121)
**Predicted under pi_3:** 0.8779 (var=0.0049)
**Predicted under pi_4:** 0.8962 (var=0.0031)

### Experiment 6
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 11 is the critical compensatory trial:
    # A = [1, 1, 0, 0], B = [1, 0, 1, 1]
    # Here, TTB looks at feature 0 (tied) and then feature 1, choosing the option with feature 1.
    # WADD weighs feature 1 (validity 0.84) against features 2 and 3 (validities 0.64 + 0.55 = 1.19),
    # typically favoring the option with features 2 and 3.
    mask_straight = (a_tuples == (1, 1, 0, 0)) & (b_tuples == (1, 0, 1, 1))
    mask_swapped = (a_tuples == (1, 0, 1, 1)) & (b_tuples == (1, 1, 0, 0))
    
    mask = mask_straight | mask_swapped
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    is_straight = mask_straight[mask]
    
    # In straight, TTB choice is A (response 0). In swapped, TTB choice is B (response 1).
    ttb_choice = np.where(is_straight, 0, 1)
    
    return float(np.mean(subset['response'] == ttb_choice))

```

**Observed (real) value:** 0.8133 (var=0.0229)
**Predicted under pi_3:** 0.8900 (var=0.0229)
**Predicted under pi_4:** 0.7633 (var=0.0534)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments, Theory 1 (Take-The-Best, pi_3) consistently outperforms Theory 2 (Probabilistic TTB, pi_4) on most metrics, particularly in Experiments 1, 2, 3, and 5. The observed data demonstrates a strong reliance on the highest validity cue, which TTB captures well. However, TTB slightly overestimates this reliance (e.g., predicting ~0.89 when observed is ~0.81 in Exp 6), and PTTB tends to underestimate it significantly due to being too stochastic. Neither theory perfectly captures the nuance in the data where subjects might occasionally integrate multiple cues or exhibit varying degrees of decision noise depending on the cue weight differences.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Probabilistic TTB theory with a new Compensatory / Weighted Additive (WADD) theory. In this theory, the decision maker calculates an overall subjective value for each option by summing the validities (or a transformation of the validities) of the cues that favor it. A softmax function is then used to translate the difference in these overall values into a choice probability. This provides a soft compensatory mechanism that can mimic TTB-like behavior when one validity dominates, but allows for cue integration when multiple lesser cues oppose the best cue, differentiating it from both strict lexicographic TTB and single-cue stochastic sampling (PTTB)."
}
```

## Usage

```json
{
  "prompt_token_count": 7991,
  "candidates_token_count": 319,
  "total_token_count": 9189
}
```
