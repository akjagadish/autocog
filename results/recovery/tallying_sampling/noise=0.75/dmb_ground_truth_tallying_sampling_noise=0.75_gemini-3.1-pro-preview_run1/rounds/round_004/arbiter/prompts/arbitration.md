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
Dual-Process Strategy Mixture: Decision-making is driven by a probabilistic mixture of two distinct strategies. With a certain probability (mixture_p), subjects employ a non-compensatory Take-The-Best (TTB) heuristic, making a choice based solely on the most valid discriminating cue. Otherwise, they use a compensatory Weighted Additive (WADD) strategy, integrating all available features weighted by their validities into a comprehensive utility score. This blend captures both the strict, flat sensitivity of heuristic processing and the graded, trade-off sensitivity of compensatory processing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    mixture_p = float(parameters['mixture_p'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # --- TTB Component ---
    cue_order = np.argsort(-val, kind='stable')
    a, b = stim[0], stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- WADD Component ---
    # WADD uses validities as weights
    scores = stim @ val
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd = e / e.sum()
    
    # --- Mixture ---
    p_core = mixture_p * p_ttb + (1.0 - mixture_p) * p_wadd
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_6
Sequential Evidence Accumulation: Decision-making is driven by a sequential sampling process where features are evaluated in order of their subjective validity. As each feature is processed, the validity-weighted difference between the options is added to a running accumulator. If this accumulated evidence reaches a predefined threshold at any point, search is immediately terminated and a choice is made (mimicking non-compensatory heuristics like Take-The-Best). If all features are exhausted without the evidence hitting the boundary, the subject makes a probabilistic choice based on the final accumulated tally (mimicking compensatory strategies like WADD). This single-process model naturally unifies fast-and-frugal heuristics and exhaustive compensatory integration depending on the height of the evidence threshold.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    threshold = float(parameters['threshold'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Sort cues by descending validity; stable sort to preserve original order on ties
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    
    A_val = 0.0
    decided = False
    p_a = 0.5
    
    # Sequentially accumulate evidence
    for j in cue_order:
        A_val += val[j] * (a[j] - b[j])
        if A_val >= threshold:
            p_a = 1.0
            decided = True
            break
        elif A_val <= -threshold:
            p_a = 0.0
            decided = True
            break
            
    # If boundary not reached, make a soft choice based on the final tally
    if not decided:
        z = beta * A_val
        # Numerically stable logistic function for P(A)
        if z > 100:
            p_a = 1.0
        elif z < -100:
            p_a = 0.0
        else:
            p_a = 1.0 / (1.0 + np.exp(-z))
            
    p_b = 1.0 - p_a
    p_core = np.array([p_a, p_b])
    
    # Blend with standard uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.8, 0.75]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design dissociates the Sequential Evidence Accumulation model from the Dual-Process Strategy Mixture model by contrasting cue cancellation with lexicographic priority. In the Sequential model, early cues with similar validities (e.g., Cue 1 and Cue 2) that favor opposite options will largely cancel each other out in the accumulator, allowing the remaining lower-validity cues to dictate the final choice. In contrast, the Mixture model's TTB component exhibits absolute lexicographic priority: it is wholly determined by the first discriminating cue and completely ignores the opposing second cue. By presenting pairs where Cues 1 and 2 oppose each other but Cues 3-5 strongly favor one option, the Sequential model predicts choices aligned with the sum of Cues 3-5, while the Mixture model predicts a substantial proportion of choices anchored strictly to Cue 1, resulting in a massive divergence in predicted choice probabilities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the difference in the probability of choosing the TTB-favored option between two sets of trials: T1/T6 (where the TTB option is also strongly favored by the WADD component) and T2/T5 (where the TTB option is strongly opposed by the WADD component). Crucially, taking this difference perfectly cancels out the uniform lapse rate (`epsilon * 0.5`) present in both models, which drastically reduces between-subject variance and yields a highly stable metric. For the Dual-Process Mixture model, the heuristic TTB component strictly follows the first cue in both trial sets, so the difference is driven entirely by the WADD component scaled by `(1-p)`, yielding a relatively small expected difference (~0.25). For the Sequential Evidence Accumulation model, the difference is much larger (~0.46): in T2/T5, the accumulated evidence for the TTB option reaches only +0.95, which frequently fails to clear the decision threshold. This forces the model to fall back on the compensatory soft choice that strongly opposes the TTB option. In T1/T6, the soft choice strongly favors the TTB option. Thus, the Sequential model predicts a much steeper drop in heuristic choices when subsequent cues oppose the first cue, creating a massive and highly significant gap.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Extract cues as 2D arrays safely
    a_cues = np.array(data['option_a_ratings'].tolist())
    b_cues = np.array(data['option_b_ratings'].tolist())
    
    # Identify TTB option (the one with Cue 1 = 1)
    ttb_is_a = (a_cues[:, 0] == 1)
    
    # Get Cue 2 for TTB option
    ttb_cue2 = np.where(ttb_is_a, a_cues[:, 1], b_cues[:, 1])
    
    # Get sum of Cues 3, 4, 5 for TTB option
    ttb_cues_345_sum = np.where(ttb_is_a, a_cues[:, 2:].sum(axis=1), b_cues[:, 2:].sum(axis=1))
    
    # Identify T1/T6: TTB option has Cue 2 == 0 and Cues 3,4,5 sum == 3
    mask_t16 = (ttb_cue2 == 0) & (ttb_cues_345_sum == 3)
    
    # Identify T2/T5: TTB option has Cue 2 == 0 and Cues 3,4,5 sum == 0
    mask_t25 = (ttb_cue2 == 0) & (ttb_cues_345_sum == 0)
    
    # Calculate P(TTB chosen) for T1/T6
    if mask_t16.sum() == 0:
        p_t16 = 0.5
    else:
        chose_ttb_t16 = np.where(ttb_is_a[mask_t16], 
                                 data['response'].values[mask_t16] == 0, 
                                 data['response'].values[mask_t16] == 1)
        p_t16 = chose_ttb_t16.mean()
        
    # Calculate P(TTB chosen) for T2/T5
    if mask_t25.sum() == 0:
        p_t25 = 0.5
    else:
        chose_ttb_t25 = np.where(ttb_is_a[mask_t25], 
                                 data['response'].values[mask_t25] == 0, 
                                 data['response'].values[mask_t25] == 1)
        p_t25 = chose_ttb_t25.mean()
        
    return float(p_t16 - p_t25)

### RESULTS
- Predicted under pi_5 (simulated): 0.2483 (var=0.0897)
- Predicted under pi_6 (simulated): 0.5250 (var=0.1077)
- Observed on real data: 0.1917 (var=0.0208)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.85, 0.65, 0.6, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 2: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the Sequential Evidence Accumulation model from the Dual-Process Strategy Mixture (TTB + WADD) by exploiting a fundamental difference in their stopping rules: absolute evidence vs. relative rank. In the Sequential model, the decision to stop and choose depends on the absolute validity of the cue hitting a fixed threshold. A high-validity cue at the top of the hierarchy might exceed the threshold (acting non-compensatorily), but if top cues are tied, a lower-validity cue might fail to reach the threshold, forcing the model to integrate remaining cues (acting compensatorily). In contrast, the TTB component of the Mixture model is scale-invariant: it always stops at the first discriminating cue, regardless of whether it is the 1st or the 3rd cue, and regardless of its absolute validity. By creating matched pairs of trials where the conflict occurs either at the top of the cue hierarchy (Cue 1 vs rest) or at the bottom (Cue 3 vs rest) while keeping the WADD log-odds perfectly identical, the Mixture model predicts the exact same choice probabilities for both trial types. The Sequential model, however, predicts a massive shift in preference (e.g., from 100% Option A to favoring Option B) because the absolute evidence threshold is crossed in the top-conflict case but not in the bottom-conflict case.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric measures the difference in the rate of 'Take-The-Best' (TTB) consistent choices between trivial Catch trials (Trials 9 & 10) and high-conflict trials (Trials 1, 2, 7, & 8). In the Catch trials, both models predict a near-100% core probability of choosing the dominant option. In the conflict trials, the Sequential Evidence Accumulation model often fails to reach its absolute stopping threshold and falls back to a compensatory WADD strategy that strongly favors the non-TTB option, leading to a massive drop in TTB-consistent choices. In contrast, the Dual-Process Mixture model explicitly selects the TTB heuristic at a constant baseline rate across all trials, keeping the TTB choice rate relatively high even in conflict. Furthermore, the broader lapse rate (epsilon) prior in the Mixture model generally compresses its dynamic range. Consequently, the Sequential model predicts a much larger drop in accuracy between Catch and Conflict trials than the Mixture model.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    is_t9 = (a_str == '11111') & (b_str == '00000')
    is_t10 = (a_str == '00000') & (b_str == '11111')
    
    is_t1 = (a_str == '10100') & (b_str == '01011')
    is_t2 = (a_str == '11100') & (b_str == '11011')
    is_t7 = (a_str == '01011') & (b_str == '10100')
    is_t8 = (a_str == '11011') & (b_str == '11100')
    
    catch_hits = 0
    n_catch = 0
    if is_t9.any():
        catch_hits += (data.loc[is_t9, 'response'] == 0).sum()
        n_catch += is_t9.sum()
    if is_t10.any():
        catch_hits += (data.loc[is_t10, 'response'] == 1).sum()
        n_catch += is_t10.sum()
        
    catch_rate = catch_hits / n_catch if n_catch > 0 else 0.0
    
    conf_hits = 0
    n_conf = 0
    if is_t1.any():
        conf_hits += (data.loc[is_t1, 'response'] == 0).sum()
        n_conf += is_t1.sum()
    if is_t2.any():
        conf_hits += (data.loc[is_t2, 'response'] == 0).sum()
        n_conf += is_t2.sum()
    if is_t7.any():
        conf_hits += (data.loc[is_t7, 'response'] == 1).sum()
        n_conf += is_t7.sum()
    if is_t8.any():
        conf_hits += (data.loc[is_t8, 'response'] == 1).sum()
        n_conf += is_t8.sum()
        
    conf_rate = conf_hits / n_conf if n_conf > 0 else 0.0
    
    return float(catch_rate - conf_rate)

### RESULTS
- Predicted under pi_5 (simulated): 0.1700 (var=0.0374)
- Predicted under pi_6 (simulated): 0.3389 (var=0.0569)
- Observed on real data: 0.1733 (var=0.0149)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice is not None and row['response'] == ttb_choice:
            ttb_matches += 1
            
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4071 (var=0.0029)
**Predicted under pi_5:** 0.5783 (var=0.0144)
**Predicted under pi_6:** 0.4890 (var=0.0280)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_resp = data.loc[mask, 'response']
    a_higher = sum_a[mask] > sum_b[mask]
    
    chose_higher = (a_higher & (filtered_resp == 0)) | (~a_higher & (filtered_resp == 1))
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.6008 (var=0.0065)
**Predicted under pi_5:** 0.4596 (var=0.0249)
**Predicted under pi_6:** 0.5200 (var=0.0342)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    match_count = 0
    total_count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        if a == (1, 0, 0, 0, 1):
            match_count += (resp == 1)
            total_count += 1
        elif a == (1, 0, 0, 1, 1):
            match_count += (resp == 1)
            total_count += 1
        elif a == (1, 0, 1, 0, 0):
            match_count += (resp == 0)
            total_count += 1
    if total_count == 0:
        return 0.5
    return match_count / total_count
```

**Observed (real) value:** 0.4467 (var=0.0059)
**Predicted under pi_5:** 0.4429 (var=0.0043)
**Predicted under pi_6:** 0.4275 (var=0.0059)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # We want to identify trials with the critical pair:
    # [1, 0, 0, 0, 1] vs [0, 1, 1, 0, 0]
    
    def is_target(a, b):
        return (tuple(a) == (1, 0, 0, 0, 1) and tuple(b) == (0, 1, 1, 0, 0))
    
    def is_target_swapped(a, b):
        return (tuple(a) == (0, 1, 1, 0, 0) and tuple(b) == (1, 0, 0, 0, 1))

    target_chosen = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if is_target(a, b):
            # Option B is [0, 1, 1, 0, 0]
            target_chosen.append(1 if resp == 1 else 0)
        elif is_target_swapped(a, b):
            # Option A is [0, 1, 1, 0, 0]
            target_chosen.append(1 if resp == 0 else 0)
            
    if not target_chosen:
        return 0.5
    
    return float(np.mean(target_chosen))
```

**Observed (real) value:** 0.5000 (var=0.0194)
**Predicted under pi_5:** 0.3675 (var=0.0249)
**Predicted under pi_6:** 0.3563 (var=0.0441)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0
        
    ttb_choices = data.apply(get_ttb_choice, axis=1)
    return float((data['response'] == ttb_choices).mean())
```

**Observed (real) value:** 0.4429 (var=0.0044)
**Predicted under pi_5:** 0.5685 (var=0.0147)
**Predicted under pi_6:** 0.4990 (var=0.0252)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    is_1v1 = (sum_a == 1) & (sum_b == 1)
    is_1v2 = (sum_a == 1) & (sum_b == 2)
    
    chose_a = (data['response'] == 0).values.astype(float)
    
    p_1v1 = np.mean(chose_a[is_1v1]) if np.sum(is_1v1) > 0 else 0.5
    p_1v2 = np.mean(chose_a[is_1v2]) if np.sum(is_1v2) > 0 else 0.5
    
    return float(p_1v1 - p_1v2)
```

**Observed (real) value:** 0.0907 (var=0.0122)
**Predicted under pi_5:** 0.0685 (var=0.0162)
**Predicted under pi_6:** 0.0613 (var=0.0157)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract top cue for A and B
    a_top = data['option_a_ratings'].apply(lambda x: x[0])
    b_top = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where the top cue is discriminating in favor of A
    is_discrim = (a_top == 1) & (b_top == 0)
    discrim_data = data[is_discrim]
    
    if len(discrim_data) == 0:
        return 0.0
        
    # Calculate the advantage of Option A over Option B on the lower cues
    a_lower_sum = discrim_data['option_a_ratings'].apply(lambda x: sum(x[1:]))
    b_lower_sum = discrim_data['option_b_ratings'].apply(lambda x: sum(x[1:]))
    lower_adv = a_lower_sum - b_lower_sum
    
    # Split into trials where lower cues favor A vs favor B
    high_mask = lower_adv > 0
    low_mask = lower_adv < 0
    
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0
        
    # Calculate the probability of choosing A in both subsets (response == 0 means A chosen)
    p_a_high = 1.0 - discrim_data.loc[high_mask, 'response'].mean()
    p_a_low = 1.0 - discrim_data.loc[low_mask, 'response'].mean()
    
    # Return the difference in P(Choose A)
    return float(p_a_high - p_a_low)
```

**Observed (real) value:** 0.1417 (var=0.0236)
**Predicted under pi_5:** 0.0938 (var=0.0263)
**Predicted under pi_6:** 0.2008 (var=0.0400)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = (sum_b - sum_a) >= 2
    if not mask.any():
        return 0.0
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.5550 (var=0.0115)
**Predicted under pi_5:** 0.4275 (var=0.0267)
**Predicted under pi_6:** 0.6550 (var=0.0438)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 1 (Dual-Process Strategy Mixture) performs significantly better on the newly proposed Experiments 1 and 2, accurately predicting the relatively small shifts in choice probability. Theory 2 (Sequential Evidence Accumulation) drastically overpredicts the effect of cue opposition and threshold failures in these experiments. However, Theory 1 struggles on Experiments 3, 4, 6, and 7, where a more nuanced compensatory integration or different stopping mechanism seems necessary. Neither theory provides a complete account of the data across all 10 experiments. Theory 2's strict absolute evidence threshold is clearly falsified by Exps 1 and 2, making it a good candidate for replacement.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Rank-Dependent Cue Weighting' or 'Probabilistic Stopping' theory. Instead of a strict absolute evidence threshold (which failed in Exps 1 and 2), decision-making could be modeled as a single compensatory process where the subjective weight of a cue drops off non-linearly (e.g., exponentially) based on its rank, or where the probability of terminating search is a soft function of both cue rank and accumulated evidence. This would naturally capture both the heuristic-like dominance of top cues and the graded integration of lower cues without the brittle threshold mechanics of Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 10759,
  "candidates_token_count": 305,
  "total_token_count": 11939
}
```
