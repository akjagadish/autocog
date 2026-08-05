# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Rank-based Weighting Theory posits that decision-makers do not use complex mathematical transformations like log-odds to weigh evidence. Instead, they rely on the simple ordinal ranking of cue validities. Cues are weighted according to an inverse function of their rank (proportional to 1/rank^rho). By restricting the decay parameter rho to a moderate range, the theory maintains a highly compensatory mechanism that prevents over-reliance on the most valid cue, capturing the softer probability matching observed in human multi-attribute decision making.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute ranks (1 is the highest validity)
    # Using stable sort for consistent tie-breaking if validities are equal
    order = np.argsort(-val, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Apply rank-based weighting
    rho = float(parameters["rho"])
    w = 1.0 / (ranks ** rho)
    
    # Compute weighted sum of features for each option
    scores = np.dot(stim, w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Strategy Mixture Theory with Independent Scaling: Individuals use a probabilistic mixture of distinct heuristics (WADD, Tallying, and Take-The-Best), but because the internal evidence scales of these heuristics vary dramatically (log-odds sums vs. integer counts vs. binary indicators), each heuristic applies its own independent temperature parameter to properly calibrate its choice probabilities before mixing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, w)
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
    if winner is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    beta_wadd = float(parameters["beta_wadd"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    p_wadd = get_probs(scores_wadd, beta_wadd)
    p_tally = get_probs(scores_tally, beta_tally)
    p_ttb = get_probs(scores_ttb, beta_ttb)
    
    w1 = float(parameters["w_wadd"])
    w2 = float(parameters["w_tally"])
    w3 = float(parameters["w_ttb"])
    w_sum = w1 + w2 + w3 + 1e-9
    
    p_mix = (w1 * p_wadd + w2 * p_tally + w3 * p_ttb) / w_sum
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.88, 0.86, 0.55, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate Rank-based Weighting Theory from the Strategy Mixture Theory, we use a set of validities where the top three cues have high and closely clustered objective validities (e.g., 0.90, 0.88, 0.86), while the remaining cues have low validities (~0.55, 0.51). For a log-odds based mechanism like WADD (a core component of the Strategy Mixture), the top three cues have very similar, large evidence weights, meaning the 2nd and 3rd cues combined will easily outweigh the 1st cue. However, Rank-based Weighting relies strictly on the ordinal rank of the cues, enforcing a steep decay (e.g., proportional to 1/rank^rho) regardless of how close the actual validities are. By pitting the 1st-ranked cue (sometimes with weak cues) against the 2nd and 3rd-ranked cues combined, Rank-based Weighting predicts the 1st cue dominates (since 1 > 1/2 + 1/3), whereas WADD strongly favors the combination of the 2nd and 3rd cues. We include variations in the number of weak cues to manipulate Tallying, creating profiles where Strategy Mixture and Rank-based Weighting make opposite predictions.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the overall probability of choosing the option containing the highest-validity cue (Cue 1) across all trials. In this experimental design, every single trial pits an option containing Cue 1 against an option containing Cues 2 and 3. 

Under Strategy Mixture Theory, the WADD heuristic computes log-odds. Because the top three validities are closely clustered (0.90, 0.88, 0.86), the combined log-odds evidence for Cues 2 and 3 (~3.8) massively outweighs Cue 1 alone (~2.2). Tallying also generally aligns against Cue 1 since the opposing option typically has more total cues. Consequently, Strategy Mixture strongly predicts subjects will prefer the option with Cues 2 and 3, resulting in a low overall rate of choosing Cue 1.

In stark contrast, Rank-based Weighting Theory ignores log-odds and relies on ordinal rank decay (1/rank^rho). Because 1 > 1/2^rho + 1/3^rho for a large portion of the parameter space (e.g., rho > 0.75), Rank-based Weighting predicts that Cue 1 alone is often sufficient to dominate Cues 2 and 3. Thus, it predicts a significantly higher overall probability of choosing the option with Cue 1. 

By pooling across all 96 trials per subject, this metric directly captures the core structural dissociation between log-odds accumulation and ordinal rank decay while driving within-subject variance to near-zero, ensuring a highly robust statistical contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings as 2D numpy arrays
    a_cues = np.vstack(data['option_a_ratings'].values)
    b_cues = np.vstack(data['option_b_ratings'].values)
    
    # Identify which option possesses the highest-validity cue (Cue 1, index 0)
    a_has_cue1 = a_cues[:, 0] == 1
    b_has_cue1 = b_cues[:, 0] == 1
    
    # Determine the subject's choice
    chose_a = data['response'].values == 0
    chose_b = data['response'].values == 1
    
    # Boolean array indicating if the chosen option contained Cue 1
    chose_cue1 = (chose_a & a_has_cue1) | (chose_b & b_has_cue1)
    
    # Return the overall proportion of trials where the Cue 1 option was chosen
    return float(chose_cue1.mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.6829 (var=0.0424)
- Predicted under pi_4 (simulated): 0.4321 (var=0.0139)
- Observed on real data: 0.4550 (var=0.0032)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.88, 0.85, 0.82, 0.7, 0.6]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 2: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Strategy Mixture Theory from the Rank-based Weighting Theory, we exploit the non-linear transformation of validities into log-odds (used by WADD in the Mixture Theory) versus the strict ordinal rank decay (used by Rank-based Weighting). We select a top-heavy validity distribution where the first three cues are highly valid and closely spaced, while the remaining cues are weaker. In critical trials, Option A is supported by the 2nd and 3rd ranked cues, while Option B is supported by the 1st and 5th ranked cues. For WADD, the sum of the log-odds for cues 2 and 3 heavily outweighs the sum for cues 1 and 5, strongly favoring Option A. Tallying results in a tie (2 cues each), leaving the Strategy Mixture Theory to predominantly favor Option A (unless the TTB weight is overwhelmingly high). Conversely, the Rank-based Weighting Theory applies a steep decay based strictly on rank (e.g., 1/rank), meaning the 1st ranked cue combined with the 5th easily outweighs the 2nd and 3rd (e.g., 1 + 1/5 > 1/2 + 1/3), leading to a strict preference for Option B. This creates a sharp, qualitative divergence in predicted choice probabilities.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the probability of choosing the option that possesses the most valid cue (cue 1), specifically restricted to trials where both options have an EQUAL total number of cues (Trials 1 and 5). Under the Strategy Mixture Theory, the Tallying heuristic yields a tie on these trials, while the WADD heuristic strongly favors the option WITHOUT cue 1 (because cues 2 and 3 combined heavily outweigh cues 1 and 5). Thus, only the Take-The-Best (TTB) heuristic favors the option with cue 1, leading to a moderate choice probability (~0.5) depending on the subject's TTB weight. In stark contrast, the Rank-based Weighting Theory assigns immense ordinal value to the 1st cue. Because the cue counts are equal, the compensatory mechanism is neutralized, and the option with the 1st cue strictly dominates for almost all values of the decay parameter rho (>0). This leads RBW to predict a choice rate very close to 1.0 with extremely low between-subject variance. The massive mean gap (~0.5 vs ~0.95) and tight variance guarantee strong discriminability.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue1_chosen = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Only consider trials where both options have an EQUAL number of cues
        if sum(a) == sum(b):
            # Identify which option possesses the 1st (most valid) cue
            # We assume the first element (index 0) is the most valid cue based on the design
            if a[0] == 1 and b[0] == 0:
                cue1_chosen.append(1 if resp == 0 else 0)
            elif b[0] == 1 and a[0] == 0:
                cue1_chosen.append(1 if resp == 1 else 0)
                
    if len(cue1_chosen) == 0:
        return 0.5
        
    return float(np.mean(cue1_chosen))

### RESULTS
- Predicted under pi_5 (simulated): 0.7531 (var=0.0227)
- Predicted under pi_4 (simulated): 0.5325 (var=0.0177)
- Observed on real data: 0.4913 (var=0.0096)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    # TTB: winner based on the first discriminating cue
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    
    # Tallying: winner based on total number of winning cues
    a_wins = np.sum(diff > 0, axis=1)
    b_wins = np.sum(diff < 0, axis=1)
    
    # Identify conflict trials where Tallying has a strict winner that disagrees with TTB
    tally_winner = np.where(a_wins > b_wins, 0, 1)
    conflict_mask = (a_wins != b_wins) & (ttb_winner != tally_winner)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    ttb_consistent = (responses[conflict_mask] == ttb_winner[conflict_mask])
    
    return float(np.mean(ttb_consistent))
```

**Observed (real) value:** 0.4619 (var=0.0040)
**Predicted under pi_5:** 0.3494 (var=0.0478)
**Predicted under pi_4:** 0.5231 (var=0.0144)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    return float(np.mean(tally_pred == data['response'].values[mask]))
```

**Observed (real) value:** 0.5637 (var=0.0028)
**Predicted under pi_5:** 0.5731 (var=0.0472)
**Predicted under pi_4:** 0.6109 (var=0.0083)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Log-odds weights for validities: [0.9, 0.8, 0.6, 0.55, 0.51]
    w = np.array([2.19722458, 1.38629436, 0.40546511, 0.2006707 , 0.04000533])
    
    def wadd_predicts_A(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * w) > np.sum(b * w)
    
    wadd_pred_A = data.apply(wadd_predicts_A, axis=1)
    
    # Subject chose A if response == 0, B if response == 1
    match = (wadd_pred_A & (data['response'] == 0)) | (~wadd_pred_A & (data['response'] == 1))
    
    return float(match.mean())
```

**Observed (real) value:** 0.4629 (var=0.0047)
**Predicted under pi_5:** 0.4960 (var=0.0554)
**Predicted under pi_4:** 0.5973 (var=0.0174)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert lists to tuples to allow for equality comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify the critical trials where Tallying and WADD make opposing predictions
    t1_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 0, 0, 0, 0))
    t5_mask = (a_tuples == (0, 1, 1, 1, 0)) & (b_tuples == (1, 0, 0, 0, 1))
    t3_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    
    mask_all = t1_mask | t5_mask | t3_mask
    if not mask_all.any():
        return 0.5
        
    # Tallying predictions: 
    # T1: A wins on 3 features, B on 1 -> prefers A (0)
    # T5: A wins on 3 features, B on 2 -> prefers A (0)
    # T3: A wins on 2 features, B on 3 -> prefers B (1)
    preds = pd.Series(index=data.index, data=np.nan)
    preds.loc[t1_mask] = 0
    preds.loc[t5_mask] = 0
    preds.loc[t3_mask] = 1
    
    # Calculate the proportion of choices matching Tallying's predictions
    match = (data.loc[mask_all, 'response'] == preds.loc[mask_all])
    return float(match.mean())
```

**Observed (real) value:** 0.5211 (var=0.0132)
**Predicted under pi_5:** 0.5006 (var=0.0790)
**Predicted under pi_4:** 0.3950 (var=0.0238)

### Experiment 7
**Design**
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: WADD prefers A, Tallying and TTB prefer B
    is_t1 = a_tuples == (0, 1, 1, 0, 0, 0)
    
    # Trial 2: WADD prefers B, Tallying and TTB prefer A
    is_t2 = a_tuples == (1, 0, 0, 1, 1, 1)
    
    wadd_choice_t1 = (data.loc[is_t1, 'response'] == 0).astype(float)
    wadd_choice_t2 = (data.loc[is_t2, 'response'] == 1).astype(float)
    
    combined = np.concatenate([wadd_choice_t1.values, wadd_choice_t2.values])
    if len(combined) == 0:
        return 0.5
    return float(np.mean(combined))
```

**Observed (real) value:** 0.4568 (var=0.0095)
**Predicted under pi_5:** 0.1663 (var=0.0144)
**Predicted under pi_4:** 0.3784 (var=0.0173)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.8, 0.75, 0.7, 0.6])
    w = np.log(val / (1.0 - val))
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    score_a = a_ratings.dot(w)
    score_b = b_ratings.dot(w)
    
    # TTB prefers A, but WADD prefers B
    cond1 = (a_ratings[:, 0] > b_ratings[:, 0]) & (score_a < score_b)
    # TTB prefers B, but WADD prefers A
    cond2 = (b_ratings[:, 0] > a_ratings[:, 0]) & (score_b < score_a)
    
    responses = data['response'].values
    
    ttb_chose_a = cond1 & (responses == 0)
    ttb_chose_b = cond2 & (responses == 1)
    
    ttb_choices = np.sum(ttb_chose_a) + np.sum(ttb_chose_b)
    total_disagreements = np.sum(cond1) + np.sum(cond2)
    
    if total_disagreements == 0:
        return 0.0
        
    return float(ttb_choices / total_disagreements)

```

**Observed (real) value:** 0.3875 (var=0.0057)
**Predicted under pi_5:** 0.4775 (var=0.0936)
**Predicted under pi_4:** 0.3856 (var=0.0216)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 8 experiments, Theory 2 (Strategy Mixture Theory, pi_4) generally outperforms Theory 1 (Rank-based Weighting, pi_5), especially on Experiments 1, 2, 3, 7, and 8 where pi_5's predictions are highly discrepant from the observed data. Theory 1's strict ordinal rank decay leads to overly deterministic predictions in favor of the highest-ranked cue, failing to capture the nuanced compensatory behavior seen in the data. However, Theory 2 also struggles with Experiments 5 and 6, suggesting that a simple static mixture of heuristics might not fully capture the underlying cognitive process. Given that Theory 1 is fundamentally misaligned with the empirical choice probabilities in most critical dissociations, it should be replaced with a completely new theory that integrates the sequential nature of evidence processing without relying on rigid rank-based decay or static heuristic mixtures.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Probabilistic Sequential Search' or 'Soft Lexicographic' theory. Instead of assigning static weights based on rank or mixing whole heuristics, this theory posits that decision-makers evaluate cues sequentially in order of validity, but with a probability of stopping search and deciding at each step that depends on the cue's validity and the current evidence difference. If a cue discriminates, it heavily biases the choice, but there is a baseline probability of continuing search (a 'slip' or 'leak' rate). This introduces a dynamic, cue-by-cue evidence accumulation process that naturally blends lexicographic and compensatory behaviors, better capturing the softer choice probabilities and context-dependent trade-offs observed across the experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 8814,
  "candidates_token_count": 368,
  "total_token_count": 9917
}
```
