# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Rank-Weighted Additive Theory: Individuals integrate all cues but weight them according to an exponential decay based solely on their rank-order of validity. This creates a 'soft' lexicographic rule that acts primarily like Take-The-Best, but allows multiple secondary cues to exert a small, non-zero compensatory pull on the decision. Response variability is captured via a softmax choice rule and a lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Ranks: 0 is highest validity
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(val))
    
    # Exponential decay based on rank
    decay = float(parameters["decay"])
    weights = decay ** ranks
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_3
Weighted Additive (WADD) Theory with Subjective Validity Scaling: People compare options by computing a comprehensive value for each option, integrating all available cues by weighting each feature. However, instead of using raw objective validities, individuals apply a non-linear subjective scaling to these validities. This scaling allows a single highly valid cue to be disproportionately weighted to balance out multiple less valid cues, gracefully explaining why individuals exhibit near-indifference (choice proportions around 0.5) when faced with adversarial choices where one option excels on the best cue but the other excels on numerous lesser cues. Response noise is captured via a softmax over the options' weighted sums, along with an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Apply non-linear subjective scaling to validities
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Compute weighted additive values for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.95, 0.94, 0.93, 0.85, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]

**Rationale:** This design strictly dissociates Rank-Weighted Additive Theory from WADD with Subjective Validity Scaling by exploiting their fundamentally different weighting mechanisms. Rank-Weighted Additive determines weights strictly by the ordinal rank of validities (e.g., weight = decay^rank), meaning the weight ratio between rank 0 and 1 is identical to the ratio between rank 3 and 4. In contrast, WADD scales the cardinal validity values. By clustering the top validities closely together (e.g., 0.95, 0.94, 0.93) and placing the middle validities much higher than the bottom ones (e.g., 0.85 vs 0.55), we create scenarios where Option A possesses a slightly higher-ranked top cue but a much lower-ranked bottom cue compared to Option B. Rank-Weighted Additive's mathematical structure guarantees it will always prefer Option A (since the rank advantage at the top mathematically outweighs the rank disadvantage at the bottom for any decay < 1). Meanwhile, WADD will strongly prefer Option B because the cardinal penalty of losing the top cue is negligible compared to the massive cardinal gain from the middle cues over the bottom cues, holding true across a very wide range of non-linear gamma scalings.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In trials where Option A has exactly 2 positive cues, Option A always possesses the single best cue available in that trial, but Option B possesses the second and third best cues. Rank-Weighted Additive Theory relies on lexicographic-like rank weighting, meaning the highest valid cue dominates, leading to a strong preference for Option A (response = 0). Conversely, WADD with Subjective Validity Scaling evaluates the cardinal validities. Because the top three cues (0.95, 0.94, 0.93) are very close in validity and the middle cue (0.85) is much higher than the bottom cues (0.55, 0.5), the combined value of Option B's cues significantly outweighs Option A's under any gamma scaling, leading to a strong preference for Option B (response = 1). The metric calculates the proportion of choices for Option B in these specific trials, expecting ~0 for Rank-Weighted and ~1 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where option A has exactly 2 positive cues (Trials 1-4)
    mask = data['option_a_ratings'].apply(lambda x: sum(x) == 2)
    
    if mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times Option B was chosen in these trials
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.3812 (var=0.0124)
- Predicted under pi_3 (simulated): 0.7278 (var=0.0104)
- Observed on real data: 0.5100 (var=0.0039)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.65, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  trial 5: A=[0, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Rank-Weighted Additive Theory from WADD by testing the strict exponential decay assumption of the former. By isolating comparisons between adjacent ranks (Rank 0 vs 1, Rank 1 vs 2, Rank 2 vs 3) using validities that have a massive cardinal drop between the first and second cue but small drops thereafter (e.g., 0.95, 0.65, 0.60, 0.55), we expose the rigidity of the Rank-Weighted model. Rank-Weighted Theory dictates that the ratio of choice logit differences between successive adjacent rank comparisons must be exactly equal to the constant decay parameter 'd'. Therefore, it mathematically cannot accommodate a large difference in choice proportions for Rank 0 vs 1 alongside equal choice proportions for Rank 1 vs 2 and Rank 2 vs 3. WADD, relying on cardinal validities, naturally predicts these varying choice logit differences because it scales the actual objective validities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the simple difference in choice proportions between the Rank 0 vs 1 tradeoff (Trial 1) and the Rank 1 vs 2 tradeoff (Trial 2). WADD predicts a substantial drop in choice probability between these two trials due to the large cardinal difference in validities (0.95 to 0.65 is a 0.30 gap, while 0.65 to 0.60 is only a 0.05 gap). Consequently, WADD expects p1 to be significantly higher than p2. Rank-Weighted Theory, constrained by exponential decay, predicts a more proportional scaling of logit differences across consecutive ranks, making p1 and p2 closer in value. By using a simple first difference rather than a second derivative, we avoid combining the noise of three separate trial types, keeping the between-subject variance tightly bounded while still capturing the core structural divergence between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1 = data[(data['a_str'] == '1000') & (data['b_str'] == '0100')]
    t2 = data[(data['a_str'] == '0100') & (data['b_str'] == '0010')]
    
    p1 = 1.0 - t1['response'].mean() if len(t1) > 0 else 0.5
    p2 = 1.0 - t2['response'].mean() if len(t2) > 0 else 0.5
    
    return float(p1 - p2)

### RESULTS
- Predicted under pi_5 (simulated): 0.1050 (var=0.0369)
- Predicted under pi_3 (simulated): 0.3017 (var=0.0355)
- Observed on real data: 0.0967 (var=0.0243)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(np.mean(data['response'] == 0))
```

**Observed (real) value:** 0.4554 (var=0.0025)
**Predicted under pi_5:** 0.5046 (var=0.0299)
**Predicted under pi_3:** 0.4731 (var=0.0503)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    tally_pred = (sum_b > sum_a).astype(int)
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.5387 (var=0.0030)
**Predicted under pi_5:** 0.4998 (var=0.0304)
**Predicted under pi_3:** 0.5256 (var=0.0576)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_chose_winner = []
    opposing_cues = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        winner_ttb = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner_ttb = 0
                break
            elif b[j] > a[j]:
                winner_ttb = 1
                break
                
        if winner_ttb is None:
            continue
            
        opp = 0
        for j in range(len(a)):
            if winner_ttb == 0 and b[j] > a[j]:
                opp += 1
            elif winner_ttb == 1 and a[j] > b[j]:
                opp += 1
                
        opposing_cues.append(opp)
        ttb_chose_winner.append(1 if row['response'] == winner_ttb else 0)
        
    df = pd.DataFrame({'opp': opposing_cues, 'chose_ttb': ttb_chose_winner})
    
    p_1 = df[df['opp'] == 1]['chose_ttb'].mean()
    p_3 = df[df['opp'] >= 3]['chose_ttb'].mean()
    
    if pd.isna(p_1) or pd.isna(p_3):
        return 0.0
        
    return float(p_1 - p_3)
```

**Observed (real) value:** 0.0456 (var=0.0198)
**Predicted under pi_5:** 0.1547 (var=0.0639)
**Predicted under pi_3:** 0.2686 (var=0.0732)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify which option is favored by Take The Best (the one with 1 on the top cue)
    ttb_favored_is_A = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    ttb_match = ((data['response'] == 0) == ttb_favored_is_A).astype(float)
    
    # Extract the features of the opposing (non-TTB-favored) option
    def get_opponent(row):
        if row['option_a_ratings'][0] == 1:
            return tuple(row['option_b_ratings'])
        else:
            return tuple(row['option_a_ratings'])
            
    opp = data.apply(get_opponent, axis=1)
    
    # Trials where the opposing option is very weak (only has the 5th best cue)
    weak_opp = opp == (0, 0, 0, 0, 1)
    # Trials where the opposing option is very strong (has both the 2nd and 3rd best cues)
    strong_opp = opp == (0, 1, 1, 0, 0)
    
    val_weak = ttb_match[weak_opp].mean() if weak_opp.any() else 0.5
    val_strong = ttb_match[strong_opp].mean() if strong_opp.any() else 0.5
    
    # Return the difference in choice probability for the TTB-favored option
    return float(val_weak - val_strong)
```

**Observed (real) value:** 0.0600 (var=0.0436)
**Predicted under pi_5:** 0.2017 (var=0.0650)
**Predicted under pi_3:** 0.2983 (var=0.0789)

### Experiment 7
**Design**
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Create string representations of the stimuli to identify trial types
    data['trial_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + '_' + data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of times Option B was chosen for each trial type
    p_b = data.groupby('trial_str')['response'].mean()
    
    # Matched pair 1
    t1 = '101000_010100'
    t2 = '100010_010001'
    
    # Matched pair 2
    t3 = '100000_011000'
    t4 = '100000_000110'
    
    # Matched pair 3
    t5 = '100000_011100'
    t6 = '100000_001110'
    
    diff = 0.0
    
    # WADD predicts P(B) is higher in t2 than t1, t3 than t4, and t5 than t6.
    # The Dual Process Mixture predicts identical probabilities within these pairs.
    if t1 in p_b and t2 in p_b:
        diff += (p_b[t2] - p_b[t1])
    if t3 in p_b and t4 in p_b:
        diff += (p_b[t3] - p_b[t4])
    if t5 in p_b and t6 in p_b:
        diff += (p_b[t5] - p_b[t6])
        
    return float(diff)
```

**Observed (real) value:** 0.1200 (var=0.1834)
**Predicted under pi_5:** 0.1417 (var=0.1309)
**Predicted under pi_3:** 0.4050 (var=0.2225)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has >= 3 positive cues and Option B has exactly 2
    a_sums = data['option_a_ratings'].apply(lambda x: sum(x))
    b_sums = data['option_b_ratings'].apply(lambda x: sum(x))
    
    mask = (b_sums == 2) & (a_sums >= 3)
    
    if mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times Option B was chosen
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4508 (var=0.0118)
**Predicted under pi_5:** 0.3396 (var=0.0124)
**Predicted under pi_3:** 0.6525 (var=0.0730)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 8 experiments, Theory 1 (Rank-Weighted Additive, pi_5) consistently outperforms Theory 2 (Subjective Validity Scaling WADD, pi_3). Theory 2 consistently overpredicts the magnitude of choice shifts driven by cardinal validity differences (e.g., Exps 2, 5, 6, and 7), whereas Theory 1's rank-based decay naturally dampens these effects, better matching the observed data. However, Theory 1 still struggles slightly with experiments where choices approach indifference (like Exp 1 and Exp 8), suggesting that a purely rank-based exponential decay might not perfectly capture the compensatory mechanisms at play. Because Theory 2's core assumption of subjective cardinal scaling fundamentally fails to capture the rigidity of human choices in this domain, it should be replaced entirely.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a Dual Process Mixture theory. Instead of integrating all cues with either rank-based or scaled-cardinal weights, assume that individuals probabilistically switch between two distinct heuristics: Take-The-Best (a strict lexicographic rule relying only on the highest valid discriminating cue) and Tallying (a simple unit-weight additive rule that counts the number of positive cues). The probability of using either strategy could be a free parameter (or dependent on the structural clarity of the choice). This mixture naturally predicts intermediate choice proportions without relying on hypersensitive cardinal scaling, and structurally accommodates both the strong top-cue dominance and the occasional multi-cue compensatory pull seen in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 8351,
  "candidates_token_count": 346,
  "total_token_count": 9834
}
```
