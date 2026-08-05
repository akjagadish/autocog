# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Strategy Mixture of TTB and WADD: Decision-makers draw from a repertoire of strategies, specifically mixing a non-compensatory heuristic (Take The Best) and a fully compensatory one (Weighted Additive). By probabilistically blending TTB (which selects the option with the highest valid cue) and WADD (which computes subjective value via log-odds weighted sums of all cues), the model naturally predicts the intermediate choice proportions observed on conflict trials where the two strategies make opposite predictions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    p_ttb_weight = float(parameters["p_ttb"])
    
    # --- Take The Best (TTB) Prediction ---
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
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Weighted Additive (WADD) Prediction ---
    # Transform validities to log-odds to represent additive evidence
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores_wadd = np.array([v_a, v_b])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- Mixture ---
    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_wadd
    
    n_opts = p_mixed.shape[0]
    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_6
Rank-Weighted Additive Model: Decision-makers assign subjective weights to cues based on their ordinal rank of validity rather than transforming raw validities via log-odds. The weight of each cue decays as an inverse power function of its rank (e.g., 1 / rank^k). This compensatory approach preserves monotonicity and naturally dampens the extreme predictions of pure log-odds WADD, successfully explaining intermediate choice proportions and avoiding the ordinal violations typical of non-compensatory heuristics like Take The Best.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    decay_rate = float(parameters["decay_rate"])
    epsilon = float(parameters["epsilon"])
    
    # Determine ranks of validities (highest validity gets rank 1)
    order = np.argsort(-val, kind="stable")
    ranks = np.zeros_like(val)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Calculate subjective weights based on rank
    w = 1.0 / (ranks ** decay_rate)
    
    # Compute weighted additive values for both options
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores = np.array([v_a, v_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.88, 0.6, 0.58, 0.56]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** This design quantitatively and qualitatively dissociates the Strategy Mixture (TTB + WADD) from the Rank-Weighted Additive model by exploiting a mathematical rigidity in the Rank-Weighted model's inverse power law. In the Rank-Weighted model, the weight of cue `i` is `1 / i^k`. This strictly enforces that the drop in subjective weight from Rank 1 to Rank 2 is proportionally larger than the drop from Rank 2 to Rank 3. However, the Strategy Mixture model (via its WADD component) uses the log-odds of validities, which can violate this rank-based constraint. We set validities such that the log-odds drop from Rank 1 to Rank 2 is tiny (0.90 vs 0.88), but the drop from Rank 2 to Rank 3 is massive (0.88 vs 0.60). In Trial 1 (Cue 1 vs. Cues 2,3,4,5), WADD strongly prefers Option B because Cue 2 alone almost offsets Cue 1. In Trial 2 (Cue 2 vs. Cues 3,4,5), WADD strongly prefers Option A because the lower cues are too weak to offset Cue 2. It is mathematically impossible for the Rank-Weighted model to simultaneously prefer Option B in Trial 1 and Option A in Trial 2 for any value of `k`, providing a definitive signature of log-odds weighting over rank-based weighting.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the sum of the probability of choosing Option B in Trial 1 (where Cue 1 competes with Cues 2,3,4,5) and the probability of choosing Option A in Trial 2 (where Cue 2 competes with Cues 3,4,5). The Rank-Weighted model's inverse power law strictly couples the weights across ranks, making it mathematically impossible to simultaneously prefer B in Trial 1 and A in Trial 2, bounding this sum around 1.0 or lower. In contrast, the Strategy Mixture model (via the log-odds transformation in its WADD component) easily accommodates both preferences simultaneously, resulting in a significantly higher sum.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def get_t1_b_resp(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 1, 1):
            return row['response'] == 1
        elif b == (1, 0, 0, 0, 0) and a == (0, 1, 1, 1, 1):
            return row['response'] == 0
        return np.nan

    def get_t2_a_resp(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return row['response'] == 0
        elif b == (0, 1, 0, 0, 0) and a == (0, 0, 1, 1, 1):
            return row['response'] == 1
        return np.nan

    t1_b = data.apply(get_t1_b_resp, axis=1).dropna()
    t2_a = data.apply(get_t2_a_resp, axis=1).dropna()
    
    val1 = t1_b.mean() if len(t1_b) > 0 else 0.0
    val2 = t2_a.mean() if len(t2_a) > 0 else 0.0
    
    return float(val1 + val2)

### RESULTS
- Predicted under pi_5 (simulated): 1.3675 (var=0.0615)
- Predicted under pi_6 (simulated): 0.9688 (var=0.0429)
- Observed on real data: 1.0350 (var=0.0428)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.98, 0.95, 0.55, 0.53, 0.51]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** This design provides a strict qualitative dissociation between the Rank-Weighted Additive Model and the Strategy Mixture (TTB + WADD) theory. We use a 5-feature design with two extremely high validities (0.98, 0.95) and three very low validities (0.55, 0.53, 0.51). Because WADD relies on log-odds transformations, the log-odds of the top two cues (~3.89 and ~2.94) overwhelmingly dominate the bottom three cues (~0.20, ~0.12, ~0.04). In our critical trials, Option A possesses the highest-validity cue(s) while Option B possesses a greater number of low-validity cues. Consequently, both TTB (which only looks at the top discriminating cue) and WADD (which sums the massive log-odds) will strictly and strongly prefer Option A. Thus, any mixture of TTB and WADD must predict a strong preference for Option A. However, the Rank-Weighted Additive Model completely ignores the magnitude of the log-odds gap, assigning weights based solely on ordinal rank (e.g., 1, 1/2^k, 1/3^k, etc.). For low-to-moderate values of the decay parameter 'k', the sum of the ranks for Option B's multiple cues will exceed the rank weight of Option A's top cue(s), allowing the Rank-Weighted model to predict a preference for Option B. This constitutes a definitive ordinal violation that the Mixture model cannot accommodate under any parameterization.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
In Trials 2 and 3, Option A possesses only a single cue (either the 1st or 2nd most valid), while Option B possesses multiple lower-validity cues. Both TTB and WADD overwhelmingly prefer Option A in these trials due to the massive gap in validities and log-odds. Consequently, the Strategy Mixture model must predict a near-zero probability of choosing Option B. However, the Rank-Weighted Additive Model uses ordinal ranks, allowing the sum of Option B's multiple lower-ranked cues to exceed Option A's single high-ranked cue for moderate decay rates. Thus, the proportion of 'Option B' choices on these specific trials will be significantly higher under the Rank-Weighted model compared to the Mixture model.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify target trials where Option A has exactly 1 positive rating (Trials 2 and 3)
    is_target = data['option_a_ratings'].apply(lambda x: sum(x) == 1)
    if not is_target.any():
        return 0.0
    # Return the proportion of times Option B was chosen in these trials
    return float(data.loc[is_target, 'response'].mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.0996 (var=0.0049)
- Predicted under pi_6 (simulated): 0.6346 (var=0.0297)
- Observed on real data: 0.5225 (var=0.0035)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction (assumes features are sorted by descending validity)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Tallying prediction
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a > sum_b:
            tally_pred = 0
        elif sum_b > sum_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        # Focus strictly on conflict trials where the theories make opposite predictions
        if ttb_pred is not None and tally_pred is not None and ttb_pred != tally_pred:
            matches.append(1 if row['response'] == ttb_pred else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

```

**Observed (real) value:** 0.4775 (var=0.0054)
**Predicted under pi_5:** 0.5858 (var=0.0213)
**Predicted under pi_6:** 0.4142 (var=0.0251)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        tally_pred = 0 if a_wins > b_wins else (1 if b_wins > a_wins else None)
        
        val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
        cue_order = np.argsort(-val)
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            total += 1
            
    return tally_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5340 (var=0.0037)
**Predicted under pi_5:** 0.2730 (var=0.0113)
**Predicted under pi_6:** 0.5279 (var=0.0279)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    w = np.array([2.19722458, 1.38629436, 1.09861229, 0.40546511])
    
    wadd_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        wadd_a = np.sum(a * w)
        wadd_b = np.sum(b * w)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        if ttb_winner is not None and ttb_winner != wadd_winner:
            subject_choice = row['response']
            wadd_choices.append(1 if subject_choice == wadd_winner else 0)
            
    if len(wadd_choices) == 0:
        return 0.5
        
    return float(np.mean(wadd_choices))
```

**Observed (real) value:** 0.5275 (var=0.0041)
**Predicted under pi_5:** 0.4975 (var=0.0513)
**Predicted under pi_6:** 0.5204 (var=0.0216)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5571 (var=0.0045)
**Predicted under pi_5:** 0.4419 (var=0.0604)
**Predicted under pi_6:** 0.5573 (var=0.0203)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify critical trials where TTB and Tallying both prefer Option A, but WADD prefers Option B.
    # These are trials where Option A has the highest validity cue (A[0] == 1), Option B has the next two highest (B[1] == 1, B[2] == 1),
    # and Option A has more positive cues overall (sum(A) > sum(B)).
    is_critical = data.apply(
        lambda row: sum(row['option_a_ratings']) > sum(row['option_b_ratings']) 
                    and row['option_a_ratings'][0] == 1 
                    and row['option_b_ratings'][1] == 1,
        axis=1
    )
    if not is_critical.any():
        return 0.5
    
    # Return the proportion of times Option B was chosen on these critical trials.
    # Strategy Mixture will be close to 0 (since both TTB and Tallying prefer A).
    # WADD will be close to 1 (since the log-odds of cues 2 and 3 outweigh cue 1 and the minor cues).
    return float(data.loc[is_critical, 'response'].mean())
```

**Observed (real) value:** 0.4738 (var=0.0085)
**Predicted under pi_5:** 0.5610 (var=0.0440)
**Predicted under pi_6:** 0.2728 (var=0.0135)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_profile = (0, 1, 1, 0, 0)
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == target_profile:
            total += 1
            if row['response'] == 0:
                matches += 1
        elif b == target_profile:
            total += 1
            if row['response'] == 1:
                matches += 1
    if total == 0:
        return 0.5
    return float(matches / total)
```

**Observed (real) value:** 0.4550 (var=0.0058)
**Predicted under pi_5:** 0.5221 (var=0.0563)
**Predicted under pi_6:** 0.3308 (var=0.0088)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.92, 0.85, 0.75, 0.65, 0.55])
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    
    conflict_ttb_match = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_choice = None
        for i in range(len(val)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        wadd_a = np.sum(a * w)
        wadd_b = np.sum(b * w)
        wadd_choice = 0 if wadd_a > wadd_b else 1
        
        if ttb_choice is not None and ttb_choice != wadd_choice:
            conflict_ttb_match.append(1 if row['response'] == ttb_choice else 0)
            
    if len(conflict_ttb_match) == 0:
        return 0.0
    return float(np.mean(conflict_ttb_match))
```

**Observed (real) value:** 0.4289 (var=0.0093)
**Predicted under pi_5:** 0.4956 (var=0.0566)
**Predicted under pi_6:** 0.4706 (var=0.0296)

### Experiment 10
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1_wadd_chosen = []
    t2_wadd_chosen = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        opt_set = {a, b}
        if opt_set == {(0, 1, 1, 1), (1, 0, 0, 0)}:
            wadd_opt = (0, 1, 1, 1)
            chose_wadd = (a == wadd_opt and resp == 0) or (b == wadd_opt and resp == 1)
            t1_wadd_chosen.append(1.0 if chose_wadd else 0.0)
        elif opt_set == {(1, 0, 0, 0), (0, 1, 0, 1)}:
            wadd_opt = (1, 0, 0, 0)
            chose_wadd = (a == wadd_opt and resp == 0) or (b == wadd_opt and resp == 1)
            t2_wadd_chosen.append(1.0 if chose_wadd else 0.0)
            
    if not t1_wadd_chosen or not t2_wadd_chosen:
        return 0.0
        
    return float(np.mean(t1_wadd_chosen) - np.mean(t2_wadd_chosen))
```

**Observed (real) value:** 0.1137 (var=0.0198)
**Predicted under pi_5:** -0.3269 (var=0.0730)
**Predicted under pi_6:** 0.0269 (var=0.1224)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 2 (Rank-Weighted Additive Model) consistently outperforms Theory 1 (Strategy Mixture of TTB and WADD). In Experiment 1, despite being proposed by Theory 1 to demonstrate its superiority, the observed data (1.0350) was much closer to Theory 2's prediction (0.9688) than Theory 1's (1.3675). In Experiment 2, Theory 1 drastically underestimated the choice proportion of Option B (predicting 0.0996 vs observed 0.5225), whereas Theory 2 was reasonably close (0.6346). Theory 2 also dominates on Experiments 4, 5, 6, 9, and 10. However, Theory 2 struggles on Experiments 7 and 8. Theory 1's rigid reliance on the log-odds transformation for its compensatory component creates extreme predictions that are not borne out by human behavior. Therefore, Theory 1 is degenerate and should be replaced with a new theory.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Sequential Evidence Accumulation' (or Drift Diffusion) theory for binary features. Instead of computing a static weighted sum or mixing heuristics, decision-makers sequentially sample cues proportional to their validities (or salience). The accumulated evidence drifts toward a decision boundary. This introduces natural noise and accounts for intermediate choice probabilities without relying on extreme log-odds transformations (like WADD) or rigid inverse-power rank weights (like Theory 2). The parameters should include a drift scaling factor and a decision boundary."
}
```

## Usage

```json
{
  "prompt_token_count": 9641,
  "candidates_token_count": 367,
  "total_token_count": 11088
}
```
