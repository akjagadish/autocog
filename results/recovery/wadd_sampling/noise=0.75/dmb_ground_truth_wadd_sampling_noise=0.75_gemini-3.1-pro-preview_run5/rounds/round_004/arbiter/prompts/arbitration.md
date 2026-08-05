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


## THEORY 2 — pi_6
Dynamic Dual Process Strategy Mixture: Individuals probabilistically switch between a non-compensatory 'Take-The-Best' (TTB) heuristic and a compensatory 'Tallying' strategy. Crucially, the probability of deploying TTB is not fixed but depends on the structural clarity of the choice—specifically, the validity of the best discriminating cue. When the best discriminating cue is highly valid, individuals are more likely to rely on TTB; when it is weaker, they shift towards Tallying (which integrates all positive cues with equal weight). To account for trials where choice behavior strongly diverges from both heuristics, the model allows for a wide range of decision noise (lapse rate) and potentially inverted or very soft Tallying temperatures.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the best discriminating cue for Take-The-Best (TTB)
    cue_order = np.argsort(-val, kind="stable")
    winner_ttb = None
    v_disc = 0.5  # default if no cues discriminate
    
    for j in cue_order:
        if a[j] != b[j]:
            winner_ttb = 0 if a[j] > b[j] else 1
            v_disc = val[j]
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Strategy 2: Tallying (unit-weight additive)
    scores_tally = np.array([np.sum(a), np.sum(b)])
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * scores_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Dynamic Mixture Weight
    # w_ttb depends on a base rate and scales with the validity of the discriminating cue
    w_base = float(parameters["w_base"])
    alpha = float(parameters["alpha"])
    
    w_ttb = w_base + alpha * (v_disc - 0.5)
    w_ttb = np.clip(w_ttb, 0.0, 1.0)
    
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=13):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 9: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 12: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 13: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Rationale:** This design isolates the effect of the specific ranks of secondary cues while holding constant the identity of the best discriminating cue and the difference in total cue counts between options. The Dynamic Dual Process Strategy Mixture theory computes its predictions based solely on the validity of the top discriminating cue (which determines the mixture weight) and the unweighted sum of all cues (Tallying). Consequently, it predicts identical choice probabilities across trials within each matched set. In contrast, the Rank-Weighted Additive theory integrates all cues using a rank-based exponential decay, making it sensitive to exactly which secondary cues are present. It therefore predicts robust, systematic differences in choice probabilities within the matched sets as the secondary cues shift from higher to lower ranks.

**Computed schedule:** 13 unique pairs × 7 reps = 91 trials per subject.



### METRIC
Rationale:
To maximize discrimination while minimizing subject-level variance, we aggregate trials into two groups ('high' and 'low') based on the Rank-Weighted Additive theory's predicted difference in option values, while perfectly controlling for the Dynamic Mixture theory's components. In the 'high' pairs, Option B's positive cues are shifted to lower validity ranks compared to the 'low' pairs, heavily penalizing B under Rank-Weighted Additive integration and thus increasing the probability of choosing A. However, for the Dynamic Mixture model, each 'high' pair is structurally identical to its corresponding 'low' pair: the best discriminating cue is exactly the same, and the unweighted sum of cues for both options is identical. Thus, the Dynamic Mixture model predicts a difference of exactly zero between these two groups, whereas the Rank-Weighted Additive model predicts a robust positive difference. Averaging over 21 trials per group stabilizes the metric against decision noise.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    high_pairs = {
        ((1, 0, 0, 0, 0), (0, 0, 1, 1, 1)),
        ((1, 0, 1, 0, 0), (0, 0, 0, 1, 1)),
        ((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))
    }
    
    low_pairs = {
        ((1, 0, 0, 0, 0), (0, 1, 1, 1, 0)),
        ((1, 0, 0, 0, 1), (0, 1, 1, 0, 0)),
        ((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))
    }
    
    data['pair'] = list(zip(data['A_tuple'], data['B_tuple']))
    
    is_high = data['pair'].isin(high_pairs)
    is_low = data['pair'].isin(low_pairs)
    
    p_high = (data.loc[is_high, 'response'] == 0).mean() if is_high.sum() > 0 else 0.5
    p_low = (data.loc[is_low, 'response'] == 0).mean() if is_low.sum() > 0 else 0.5
    
    return float(p_high - p_low)

### RESULTS
- Predicted under pi_5 (simulated): 0.1057 (var=0.0268)
- Predicted under pi_6 (simulated): -0.0057 (var=0.0136)
- Observed on real data: 0.0400 (var=0.0339)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.98, 0.95, 0.6, 0.57, 0.54]

**Trial pairs (n=9):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design tests the fundamental weighting mechanism of the Dynamic Dual Process Strategy Mixture (Advocated) against the Rank-Weighted Additive Theory (Competing). The validities are chosen to have a highly non-linear drop in cardinal values: a small drop between Rank 0 and Rank 1 (0.98 to 0.95), a massive drop between Rank 1 and Rank 2 (0.95 to 0.60), and another small drop between Rank 2 and Rank 3. Across trials, we shift which cue is the best discriminator (Rank 0, 1, or 2) while keeping the Tallying advantage for Option B constant. The Advocated theory predicts that the mixture weight on TTB will remain high and roughly constant for Rank 0 and Rank 1 discriminators, but will fall off a cliff for Rank 2, leading to a highly non-linear shift towards Tallying (Option B). In contrast, the Competing theory assumes weights decay exponentially by rank alone, totally ignoring the cardinal validity gap. It strictly dictates that the choice logits will decay by a constant proportional factor across these rank shifts, mathematically unable to produce the 'cliff' predicted by the Mixture theory.

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



### METRIC
Rationale:
This metric exploits a structural limitation of the Dynamic Dual Process Strategy Mixture (Advocated theory). In the Advocated theory, secondary cues are integrated via a perfectly unit-weighted Tallying mechanism. Therefore, shifting the opposing cues in Option B to lower ranks (e.g., from ranks 1 and 2 in Trial 1 to ranks 2 and 3 in Trial 6, or from ranks 2 and 3 in Trial 2 to ranks 3 and 4 in Trial 7) does not change the Tallying score, nor does it change the Take-The-Best choice. Thus, the Advocated theory predicts exactly zero change in true choice probabilities between these matched pairs. The Rank-Weighted Additive Theory (Competing), however, strictly penalizes lower-ranked cues via exponential decay. Shifting opposing cues to lower ranks substantially reduces their subtractive weight, predicting a systematic and large increase in the probability of choosing Option A. By summing the within-subject increase in P(Choose A) across these pairs, we isolate this exact divergence.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 0, 0):
            return 'T1'
        if a == (1, 0, 0, 0, 0) and b == (0, 0, 1, 1, 0):
            return 'T6'
        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 0):
            return 'T2'
        if a == (0, 1, 0, 0, 0) and b == (0, 0, 0, 1, 1):
            return 'T7'
        return 'Other'
        
    data['trial'] = data.apply(get_trial, axis=1)
    df = data[data['trial'] != 'Other'].copy()
    
    if df.empty:
        return 0.0
        
    # P(Choose A) = 1 - response
    df['p_A'] = 1.0 - df['response']
    
    subj_means = df.groupby(['subject_id', 'trial'])['p_A'].mean().unstack(fill_value=0.5)
    
    for t in ['T1', 'T6', 'T2', 'T7']:
        if t not in subj_means.columns:
            subj_means[t] = 0.5
            
    # The Competing theory predicts a positive shift in P(A) when opposing cues are downgraded in rank.
    # The Advocated theory predicts exactly zero shift.
    diff1 = subj_means['T6'] - subj_means['T1']
    diff2 = subj_means['T7'] - subj_means['T2']
    
    return float((diff1 + diff2).mean())


### RESULTS
- Predicted under pi_5 (simulated): 0.2720 (var=0.1208)
- Predicted under pi_6 (simulated): -0.0060 (var=0.0702)
- Observed on real data: 0.0040 (var=0.0660)

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
**Predicted under pi_6:** 0.5183 (var=0.0368)

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
**Predicted under pi_6:** 0.4806 (var=0.0540)

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
**Predicted under pi_6:** 0.1250 (var=0.0223)

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
**Predicted under pi_6:** 0.0817 (var=0.0568)

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
**Predicted under pi_6:** -0.1250 (var=0.1140)

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
**Predicted under pi_6:** 0.2667 (var=0.0245)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where option A has exactly 2 positive cues (Trials 1-4)
    mask = data['option_a_ratings'].apply(lambda x: sum(x) == 2)
    
    if mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times Option B was chosen in these trials
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5100 (var=0.0039)
**Predicted under pi_5:** 0.3812 (var=0.0124)
**Predicted under pi_6:** 0.3853 (var=0.0275)

### Experiment 10
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1 = data[(data['a_str'] == '1000') & (data['b_str'] == '0100')]
    t2 = data[(data['a_str'] == '0100') & (data['b_str'] == '0010')]
    
    p1 = 1.0 - t1['response'].mean() if len(t1) > 0 else 0.5
    p2 = 1.0 - t2['response'].mean() if len(t2) > 0 else 0.5
    
    return float(p1 - p2)
```

**Observed (real) value:** 0.0967 (var=0.0243)
**Predicted under pi_5:** 0.1050 (var=0.0369)
**Predicted under pi_6:** -0.0200 (var=0.0404)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 1 (Rank-Weighted Additive) captures the qualitative effects in Experiments 1, 7, and 10 much better than Theory 2 (Dynamic Mixture). Theory 2 consistently predicts no difference (or negative difference) in these experiments because its Tallying component unit-weights all secondary cues, ignoring their ranks. However, Theory 1 fails dramatically in Experiment 2, where it predicts a huge difference (0.27) while the observed difference is essentially zero (0.004). Experiment 2 manipulated the cardinal validities to have a cliff (e.g., 0.95 to 0.60). Theory 1 ignores this cardinal cliff because it only uses ordinal ranks to exponentially decay weights. Theory 2 performed better on Experiment 2 because its mixture weight reacted to the validity of the best cue. Neither theory perfectly captures all the data, but the failures point to a clear missing mechanism: individuals are integrating all cues (unlike pure TTB or Tallying), but they are doing so using weights derived from the *cardinal* validities, not just their ordinal ranks.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Cardinal Validity-Weighted Additive Theory' (e.g., a Log-Odds Additive model). Instead of relying on a heuristic mixture (like Theory 2) or a strict rank-based exponential decay (like Theory 1), this theory should posit that subjects integrate all cues by weighting them according to a non-linear transformation of their actual cardinal validities (such as log-odds: log(v / (1 - v)), or a parameterized scaling of the validities). This naturally accounts for the 'cliff' in Experiment 2 (where a massive drop in cardinal validity translates to a massive drop in weight, mimicking a heuristic shift) while preserving the fully compensatory, cue-specific integration needed to explain the systematic shifts in Experiments 1, 7, and 10."
}
```

## Usage

```json
{
  "prompt_token_count": 10482,
  "candidates_token_count": 442,
  "total_token_count": 11853
}
```
