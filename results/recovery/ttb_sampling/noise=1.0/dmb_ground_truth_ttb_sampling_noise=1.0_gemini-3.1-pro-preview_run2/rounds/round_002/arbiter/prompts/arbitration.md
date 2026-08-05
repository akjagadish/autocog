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
People employ a probabilistic 'Take The Best' (TTB) heuristic to make binary choices. They order features by their subjective validity, which is informed by objective validities but subject to random fluctuations (noise). They compare the options sequentially based on this subjective ordering. The first feature that discriminates between the two options determines the choice, and all subsequent features are ignored. This introduces stochasticity into the cue hierarchy, explaining why choices sometimes deviate from strict reliance on the single highest-validity cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    
    # Add Gaussian noise to validities to model subjective fluctuations in cue hierarchy
    noisy_validities = validities + np.random.normal(0, sigma, size=validities.shape)
    
    # Order features by noisy validity in descending order
    order = np.argsort(noisy_validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Iterate through features in order of validity
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    # If no features discriminate, both options are tied
    if scores[0] == 0.0 and scores[1] == 0.0:
        scores = np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Take The First (Left-to-Right Lexicographic) with high noise capacity

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Take The First expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    n_features = stim.shape[1]
    scores = np.array([0.0, 0.0])
    
    # Scan features from left to right (index 0 to n_features - 1)
    for idx in range(n_features):
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    # If no features discriminate, both options are tied
    if scores[0] == 0.0 and scores[1] == 0.0:
        scores = np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
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
**Validities (n_features=5):** [0.51, 0.6, 0.7, 0.8, 0.99]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Take The First (Left-to-Right Lexicographic), we arrange the feature validities in ascending order from left to right. The leftmost feature has the lowest validity, while the rightmost feature has the highest validity. Take The First strictly evaluates features from left to right, thus heavily relying on the lowest-validity features if they discriminate. In contrast, Take The Best reorders features by their subjective validities (with noise) and will predominantly evaluate the rightmost, highest-validity feature first. By pitting the leftmost features against the rightmost features, the two models make completely opposite predictions.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In all 8 trial pairs of this design, the highest-validity feature (the rightmost one) perfectly contradicts the leftmost discriminating feature. Take The Best (TTB) prioritizes features by validity (right-to-left in this design), leading to choices that align with the rightmost feature. Take The First (TTF) evaluates features strictly left-to-right, leading to choices that align with the leftmost feature. By calculating the proportion of trials where the subject's response aligns with the rightmost feature, TTB will score highly (close to 1.0, modulo noise), while TTF will score very lowly (close to 0.0, modulo noise). This creates an extremely stark contrast between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    # The rightmost feature (index 4) has the highest validity (0.99).
    # Take The Best (TTB) will rely heavily on this feature.
    # Take The First (TTF) will scan left-to-right and rely on the leftmost discriminating features.
    # The experimental design is set up so that the leftmost discriminating feature 
    # ALWAYS predicts the exact opposite of the rightmost feature.
    # Therefore, TTB predicts the subject will choose the option with a 1 on the rightmost feature,
    # whereas TTF predicts the subject will choose the option with a 0 on the rightmost feature.
    
    # We extract the rightmost feature value for option B
    b_rightmost = data['option_b_ratings'].apply(lambda x: x[-1])
    
    # TTB predicts choosing B (response=1) when B has 1 on the rightmost feature,
    # and choosing A (response=0) when B has 0 (meaning A has 1).
    # Thus, TTB predicts response == b_rightmost.
    # We return the proportion of trials where the choice aligns with TTB.
    return float((data['response'] == b_rightmost).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.6038 (var=0.0059)
- Predicted under pi_4 (simulated): 0.2458 (var=0.0203)
- Observed on real data: 0.4946 (var=0.0022)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.65, 0.6, 0.95, 0.9, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate Take The First (Left-to-Right Lexicographic) from Take The Best (TTB), we place the highest validity features in the middle of the array, with lower validities on the left and right edges. Take The First strictly evaluates features from left to right, meaning it will base its decisions on the leftmost features (indices 0 and 1) whenever they discriminate, regardless of their low validity. Take The Best, however, sorts features by their subjective validities and will jump straight to the middle features (indices 2 and 3). By designing trials where the leftmost features favor one option and the middle features favor the other, we can cleanly separate the two models. This avoids the confound of a strictly ascending validity array where left-to-right could be mistaken for reverse-validity search.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the proportion of trials where the subject's choice aligns with the prediction of the Take The First (Left-to-Right) heuristic. Since the experimental design specifically pits the left-to-right reading order against the objective validities (where the highest validities are in the middle), the Take The First model will score very high on this metric. Conversely, the Take The Best model, which prioritizes the middle features, will systematically make choices opposed to the left-to-right prediction on key trials, resulting in a significantly lower score.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = np.array(data['response'].tolist())
    
    n_features = a_ratings.shape[1]
    
    ttf_preds = np.full(len(data), -1)
    
    for i in range(n_features):
        mask = (ttf_preds == -1) & (a_ratings[:, i] != b_ratings[:, i])
        ttf_preds[mask] = np.where(a_ratings[mask, i] > b_ratings[mask, i], 0, 1)
        
    valid_mask = ttf_preds != -1
    if not np.any(valid_mask):
        return 0.5
        
    return float(np.mean(responses[valid_mask] == ttf_preds[valid_mask]))

### RESULTS
- Predicted under pi_3 (simulated): 0.5342 (var=0.0034)
- Predicted under pi_4 (simulated): 0.7290 (var=0.0199)
- Observed on real data: 0.4983 (var=0.0015)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    match_count = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
        if row['response'] == tally_pref:
            match_count += 1
        total += 1
    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.4832 (var=0.0024)
**Predicted under pi_3:** 0.5784 (var=0.0045)
**Predicted under pi_4:** 0.4800 (var=0.0032)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A has top 2 features, B has bottom 3
    t1 = (a_str == '11000') & (b_str == '00111')
    # Trial 2: A has bottom 3 features, B has top 2
    t2 = (a_str == '00111') & (b_str == '11000')
    
    critical = t1 | t2
    if not critical.any():
        return 0.5
        
    # WADD prefers the option with the top 2 features (A in t1, B in t2)
    # Tallying prefers the option with the bottom 3 features (since 3 > 2)
    wadd_choices = (t1 & (data['response'] == 0)) | (t2 & (data['response'] == 1))
    return float(wadd_choices.sum() / critical.sum())
```

**Observed (real) value:** 0.4750 (var=0.0061)
**Predicted under pi_3:** 0.4975 (var=0.0116)
**Predicted under pi_4:** 0.7325 (var=0.0285)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    two_feature_chosen = []
    for subj, subj_df in data.groupby('subject_id'):
        subj_vals = []
        for _, row in subj_df.iterrows():
            a_ratings = row['option_a_ratings']
            b_ratings = row['option_b_ratings']
            resp = row['response']
            
            sum_a = sum(a_ratings)
            sum_b = sum(b_ratings)
            
            # Focus on trials where one option has exactly 2 features and the other has 4
            if sum_a == 2 and sum_b == 4:
                subj_vals.append(1.0 if resp == 0 else 0.0)
            elif sum_b == 2 and sum_a == 4:
                subj_vals.append(1.0 if resp == 1 else 0.0)
                
        if subj_vals:
            two_feature_chosen.append(np.mean(subj_vals))
            
    return float(np.mean(two_feature_chosen)) if two_feature_chosen else 0.5
```

**Observed (real) value:** 0.4913 (var=0.0041)
**Predicted under pi_3:** 0.4190 (var=0.0068)
**Predicted under pi_4:** 0.5420 (var=0.0045)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f0 = data['option_b_ratings'].apply(lambda x: x[0])
    a_sum = data['option_a_ratings'].apply(lambda x: sum(x))
    
    mask = (a_f0 == 0) & (b_f0 == 1) & (a_sum >= 3)
    
    if mask.sum() == 0:
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.0061)
**Predicted under pi_3:** 0.4308 (var=0.0146)
**Predicted under pi_4:** 0.7521 (var=0.0234)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed metric values hover remarkably close to 0.5 (ranging from 0.475 to 0.520). This includes metrics specifically designed to capture Take The Best (Exp 1), Take The First (Exp 2), Tallying (Exp 3), and WADD (Exp 4). Neither Theory 1 (TTB) nor Theory 2 (TTF) successfully captures this consistent ~0.5 baseline, as both models predict strong directional preferences (e.g., TTF predicts 0.75 in Exp 6 and 0.24 in Exp 1; TTB predicts 0.60 in Exp 1). The lack of alignment with any systematic heuristic suggests that participants may be exhibiting near-random behavior, perhaps due to task confusion, lack of motivation, or a completely different, unmodeled choice mechanism that effectively cancels out across subjects.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the 'Take The First' theory with a 'Random Choice / Minimal Effort' theory. The new theory should posit that participants, lacking trial-by-trial feedback and facing complex binary arrays, largely resort to random guessing or are dominated by a high lapse rate. The model should predict choice probabilities very close to 0.5 for all options, regardless of the stimulus features or validities. This will better capture the empirical reality that observed metrics consistently fail to show any strong heuristic-driven deviations from 0.5."
}
```

## Usage

```json
{
  "prompt_token_count": 6583,
  "candidates_token_count": 338,
  "total_token_count": 8190
}
```
