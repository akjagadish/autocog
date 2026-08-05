# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Strategy Mixture Theory (TTB + WADD): Decision makers do not universally adopt a single monolithic strategy. Instead, choices are generated from a probabilistic mixture of decision rules. On any given trial, an individual uses a non-compensatory heuristic (Take The Best) with probability 'alpha', and a compensatory strategy (Weighted Additive - WADD) with probability '1 - alpha'. Mixing these strategies captures intermediate rates of compensatory and non-compensatory choices, while WADD leverages cue validities for a more nuanced compensatory evaluation.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take The Best (TTB)
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - scores_ttb.max())
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / e_ttb.sum()
        
    # Strategy 2: WADD (Weighted Additive)
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of the two strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Apply lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_6
Rank-Dependent Tallying: Decision-makers primarily evaluate options based on the sheer number of positive features (a tallying-like process), but the weight of each feature is subject to rank-based discounting. Rather than scaling exponentially with log-odds, a feature's weight decays as a power law of its validity rank. This ensures a strong compensatory mechanism where multiple moderate cues can easily overpower a single high-validity cue, and tallying differences dominate choice probabilities unless the validity rank differences are extreme.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Assign ranks to features based on validity (1 = highest validity)
    n_features = len(val)
    order = np.argsort(-val, kind="stable")
    ranks = np.zeros(n_features)
    ranks[order] = np.arange(1, n_features + 1)
    
    # Rank-based discounting: weight decays as an inverse power of rank
    w = 1.0 / (ranks ** gamma)
    
    # Compute weighted tally for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 9: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** Theory 1 predicts choices as a probabilistic mixture of Take-The-Best (TTB) and a Linear Weighted Additive rule (WADD). Because Linear WADD uses raw validities as weights, it is strongly compensatory: the sum of several moderate validities easily exceeds a single high validity. Theory 2, on the other hand, uses rank-dependent tallying where the weight of a feature decays as an inverse power of its rank (gamma >= 2.0). Mathematically, for any gamma >= 2.0, the weight of the highest-ranked feature (1.0) is strictly greater than the sum of all subsequent feature weights combined (which maxes out at ~0.645). Thus, Theory 2 acts as a deterministic, strictly non-compensatory lexicographic model (essentially a pure TTB rule). By pitting the highest-validity discriminating cue against multiple lower-validity cues, we can cleanly separate the two theories: Theory 2 will uniformly choose the option favored by the single best cue, while Theory 1 will exhibit a mixed response profile reflecting its compensatory WADD component.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
In Theory 1 (Strategy Mixture), choices on disagree trials are a probabilistic mixture of TTB and WADD. Because both strategies are locally highly deterministic on these trials, the probability of choosing the TTB-favored option is simply the mixture parameter 'alpha', regardless of whether the first or second cue discriminates. Thus, Theory 1 predicts the TTB match rate will be identical across Group 1 and Group 2, making the difference ~0. 

In Theory 2 (Rank-Dependent Tallying), the decision relies on the difference in weighted tallies. When Cue 0 discriminates (Group 1), the score difference is massive (e.g. ~0.65), leading to a high TTB match rate. However, when Cue 0 is tied and Cue 1 discriminates (Group 2), the weight of Cue 1 (0.25) barely exceeds the sum of the remaining cues (0.2135), resulting in a tiny score difference (0.0365). Because Theory 2 uses a softmax rule, this tiny difference pushes the choice probability very close to 50%. Thus, Theory 2 predicts a drastically lower TTB match rate in Group 2 compared to Group 1, yielding a strongly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Identify TTB predictions
    diff = A - B
    diff_bool = diff != 0
    first_diff_idx = diff_bool.argmax(axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_winner = np.where(first_diff_val > 0, 0, 1)
    
    # Identify WADD predictions
    validities = np.array([0.95, 0.85, 0.8, 0.75, 0.7])
    wadd_diff = (A - B).dot(validities)
    wadd_winner = np.where(wadd_diff > 0, 0, 1)
    
    # Restrict to trials where TTB and WADD make opposite predictions
    disagree = (ttb_winner != wadd_winner) & diff_bool.any(axis=1)
    
    # Group 1: Disagree trials where the highest valid cue (Cue 0) discriminates
    group1 = disagree & (first_diff_idx == 0)
    
    # Group 2: Disagree trials where Cue 0 is tied, so Cue 1 discriminates
    group2 = disagree & (first_diff_idx == 1)
    
    if np.sum(group1) == 0 or np.sum(group2) == 0:
        return 0.0
        
    # Calculate TTB match rate in both groups
    match1 = np.mean(resp[group1] == ttb_winner[group1])
    match2 = np.mean(resp[group2] == ttb_winner[group2])
    
    # The contrast is the difference in TTB match rate
    return float(match1 - match2)

### RESULTS
- Predicted under pi_4 (simulated): -0.0039 (var=0.0149)
- Predicted under pi_6 (simulated): 0.3156 (var=0.0219)
- Observed on real data: -0.0956 (var=0.1293)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** Theory 1 is described as a rank-dependent tallying model, but its mathematical parameterization (gamma >= 2.0) ensures that the weight of the highest-ranked feature (1.0) is strictly greater than the sum of all subsequent feature weights combined (which maxes out at ~0.645 for gamma=2.0). Thus, Theory 1 acts as a deterministic, non-compensatory lexicographic model (pure TTB-like behavior). Theory 2, however, is a probabilistic mixture of TTB and a Linear Weighted Additive rule (WADD). Because Linear WADD uses raw validities as weights, it is strongly compensatory: the sum of several moderate validities easily exceeds a single high validity. By pitting the highest-validity discriminating cue against multiple lower-validity cues, we can cleanly separate the two theories: Theory 1 will uniformly choose the option favored by the single best cue, while Theory 2 will exhibit a mixed response profile reflecting its compensatory WADD component.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Theory 1 operates as a strict non-compensatory (TTB-like) model due to its high gamma parameter, meaning it will only deviate from TTB choices due to random lapse (epsilon). Theory 2 mixes TTB with a compensatory WADD strategy, actively preferring the WADD-favored option when the two strategies disagree. This metric computes the probability of choosing the WADD-favored option on disagreement trials (Trials 1-6) and subtracts the empirical baseline lapse rate computed from agreement trials (Trials 7-8). For Theory 1, this difference will be ~0, while for Theory 2 it will be significantly positive (reflecting the WADD mixture component).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    wadd_b_ttb_a = []
    lapse_rate_trials = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        resp = row['response']
        
        diff = a - b
        ttb_winner = None
        for i in range(len(diff)):
            if diff[i] > 0:
                ttb_winner = 0
                break
            elif diff[i] < 0:
                ttb_winner = 1
                break
                
        wadd_a = np.sum(a * val)
        wadd_b = np.sum(b * val)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        if ttb_winner == 0 and wadd_winner == 1:
            wadd_b_ttb_a.append(1 if resp == 1 else 0)
        elif ttb_winner == wadd_winner:
            lapse_rate_trials.append(1 if resp != ttb_winner else 0)
            
    p_comp = np.mean(wadd_b_ttb_a) if len(wadd_b_ttb_a) > 0 else 0.0
    p_lapse = np.mean(lapse_rate_trials) if len(lapse_rate_trials) > 0 else 0.0
    
    return float(p_comp - p_lapse)

### RESULTS
- Predicted under pi_4 (simulated): 0.2419 (var=0.0202)
- Predicted under pi_6 (simulated): 0.0569 (var=0.0109)
- Observed on real data: -0.5456 (var=0.0763)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction
        winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            elif b[j] > a[j]:
                winner = 1
                break
        
        if winner is not None:
            if row['response'] == winner:
                ttb_consistent += 1
            total += 1
            
    return float(ttb_consistent / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.6583 (var=0.0166)
**Predicted under pi_4:** 0.7150 (var=0.0215)
**Predicted under pi_6:** 0.7777 (var=0.0066)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pred = np.where(a_wins > b_wins, 0, 1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    matches = (data['response'].values[mask] == tally_pred[mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2750 (var=0.0193)
**Predicted under pi_4:** 0.4446 (var=0.0098)
**Predicted under pi_6:** 0.3960 (var=0.0026)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    comp_choices = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: A has best cue, B has all others (compensatory is B)
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            comp_choices += (resp == 1)
            total += 1
        # Trial 2: B has best cue, A has next two (compensatory is A)
        elif a == (0, 1, 1, 0) and b == (1, 0, 0, 0):
            comp_choices += (resp == 0)
            total += 1
        # Trial 4: A has best available cue, B has next two (compensatory is B)
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            comp_choices += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.0
    return float(comp_choices / total)
```

**Observed (real) value:** 0.2278 (var=0.0190)
**Predicted under pi_4:** 0.2911 (var=0.0192)
**Predicted under pi_6:** 0.2494 (var=0.0082)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has only 1 positive feature (the most valid one) 
    # and Option B has 3 or 4 positive features (all lower validity).
    # This corresponds exactly to Trial 1 and Trial 3 in the design.
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = (a_sum == 1) & (b_sum >= 3)
    if not mask.any():
        return 0.0
    # Return the proportion of times Option B was chosen (response == 1)
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.2333 (var=0.0272)
**Predicted under pi_4:** 0.2808 (var=0.0246)
**Predicted under pi_6:** 0.2933 (var=0.0151)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    def get_p_a(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        
        chose_a = 0
        total = 0
        
        if mask1.any():
            chose_a += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.any():
            chose_a += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
            
        if total == 0:
            return 0.5
        return chose_a / total

    # Trial 8: Concordant for Strategy Mixture, weak advantage for NL-WADD
    p_a_t8 = get_p_a('10011', '01100')
    
    # Trial 7: Discordant for Strategy Mixture, massive advantage for NL-WADD
    p_a_t7 = get_p_a('11000', '00111')
    
    return float(p_a_t8 - p_a_t7)
```

**Observed (real) value:** -0.3900 (var=0.1891)
**Predicted under pi_4:** 0.1625 (var=0.0470)
**Predicted under pi_6:** -0.0175 (var=0.0300)

### Experiment 8
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t7_a = (a_str == '10100') & (b_str == '01011')
    t7_b = (a_str == '01011') & (b_str == '10100')
    
    t36_a = (a_str == '10000') & (b_str == '01111')
    t36_b = (a_str == '01111') & (b_str == '10000')
    
    t7_ttb = pd.concat([
        data.loc[t7_a, 'response'] == 0,
        data.loc[t7_b, 'response'] == 1
    ])
    
    t36_ttb = pd.concat([
        data.loc[t36_a, 'response'] == 0,
        data.loc[t36_b, 'response'] == 1
    ])
    
    t7_rate = t7_ttb.mean() if len(t7_ttb) > 0 else 0.5
    t36_rate = t36_ttb.mean() if len(t36_ttb) > 0 else 0.5
    
    return float(t7_rate - t36_rate)
```

**Observed (real) value:** -0.1983 (var=0.1423)
**Predicted under pi_4:** 0.0000 (var=0.0209)
**Predicted under pi_6:** 0.0167 (var=0.0176)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    val = np.array([0.99, 0.85, 0.8, 0.75, 0.6])
    diff = a_ratings - b_ratings
    
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    
    val_clipped = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    lo_score_a = a_ratings @ log_odds
    lo_score_b = b_ratings @ log_odds
    lo_winner = np.where(lo_score_a > lo_score_b, 0, 1)
    
    lin_score_a = a_ratings @ val
    lin_score_b = b_ratings @ val
    lin_winner = np.where(lin_score_a > lin_score_b, 0, 1)
    
    group1_mask = (ttb_winner == lo_winner) & (ttb_winner != lin_winner)
    group2_mask = (ttb_winner != lo_winner) & (ttb_winner != lin_winner)
    
    if np.sum(group1_mask) == 0 or np.sum(group2_mask) == 0:
        return 0.0
        
    rate1 = np.mean(responses[group1_mask] == ttb_winner[group1_mask])
    rate2 = np.mean(responses[group2_mask] == ttb_winner[group2_mask])
    
    return float(rate1 - rate2)
```

**Observed (real) value:** -0.0698 (var=0.0405)
**Predicted under pi_4:** 0.0048 (var=0.0095)
**Predicted under pi_6:** 0.2168 (var=0.0088)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    group_x_pairs = {
        frozenset(['10000', '00111']),
        frozenset(['01000', '00110']),
        frozenset(['00100', '00011'])
    }
    
    group_y_pairs = {
        frozenset(['10000', '01111']),
        frozenset(['01000', '00111'])
    }
    
    def get_single_cue_chosen(row):
        a_is_single = sum(row['option_a_ratings']) == 1
        if a_is_single:
            return 1.0 if row['response'] == 0 else 0.0
        else:
            return 1.0 if row['response'] == 1 else 0.0

    is_x = data.apply(lambda row: frozenset([row['A_str'], row['B_str']]) in group_x_pairs, axis=1)
    is_y = data.apply(lambda row: frozenset([row['A_str'], row['B_str']]) in group_y_pairs, axis=1)
    
    x_trials = data[is_x]
    y_trials = data[is_y]
    
    if len(x_trials) == 0 or len(y_trials) == 0:
        return 0.0
        
    p_x = x_trials.apply(get_single_cue_chosen, axis=1).mean()
    p_y = y_trials.apply(get_single_cue_chosen, axis=1).mean()
    
    return float(p_x - p_y)
```

**Observed (real) value:** -0.1433 (var=0.0662)
**Predicted under pi_4:** 0.0372 (var=0.0113)
**Predicted under pi_6:** -0.0608 (var=0.0135)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the ten experiments, both Theory 1 (Strategy Mixture) and Theory 2 (Rank-Dependent Tallying) fail to capture the strongly negative observed metrics in Experiments 2, 7, and 8. For instance, in Experiment 2, the observed value is -0.5456, while both theories predict positive values. This indicates that subjects are making choices that are not just a mixture of TTB and WADD, nor simply rank-discounted tallies, but likely involve a different underlying cognitive mechanism\u2014perhaps one where compensatory cues are actively discounted if they conflict with the most valid cue, or an evidence accumulation process with a strict threshold. Given the systematic failures of both models on these specific conflict/disagreement trials, replacing Theory 1 with a fundamentally new theory is the best path forward.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Sequential Evidence Accumulation with Probabilistic Stopping' theory. In this theory, decision-makers process cues sequentially in order of their validities. After each cue, they accumulate evidence proportional to the cue's log-odds validity. Crucially, after evaluating each cue, there is a probability of stopping the search and making a decision based on the current evidence tally, which depends on the current evidence gap (confidence). This naturally bridges TTB (early stopping when the first cue provides enough evidence) and compensatory strategies (continuing search when the first cue's evidence is weak or the threshold is high), but it avoids the rigid deterministic predictions of a pure mixture model and can naturally produce the negative contrast effects observed in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 10998,
  "candidates_token_count": 357,
  "total_token_count": 12423
}
```
