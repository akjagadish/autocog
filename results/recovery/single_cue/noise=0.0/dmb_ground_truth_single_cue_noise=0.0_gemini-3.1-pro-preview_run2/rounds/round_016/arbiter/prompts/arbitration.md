# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_16" and "pi_18") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_16" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_18" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_16
Recency-Biased Cue Overweighting: Decision-makers evaluate options by attempting to integrate all available features, but due to visual recency and short-term memory effects, the final feature in the sequence is disproportionately salient. While the first N-1 features are weighted according to their stated validities (subject to non-linear scaling), the final feature is assigned an independent, often much larger weight. This mechanism explains boundary cases where subjects' choices are driven by the nominally least valid cue, effectively overriding both compensatory tallying and the expected Take-The-Best hierarchy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    # Overweight the final feature due to recency
    w[-1] = recency_weight
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_18
Serial Position Dual-Overweighting: Decision-makers evaluate options by integrating features, but due to memory and attention constraints at the sequence boundaries, both the first (primacy) and the last (recency) cues are assigned independent, disproportionately large weights. Unlike models that normalize attention or weights, these boundary weights are unnormalized, allowing them to independently dominate choice when necessary. Middle cues are weighted by their stated validities, scaled non-linearly. This mechanism captures both extreme primacy and extreme recency effects without the dampening effect of weight normalization.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    primacy_weight = float(parameters["primacy_weight"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    
    # Overweight the first and final features due to primacy and recency
    if len(w) > 1:
        w[0] = primacy_weight
        w[-1] = recency_weight
    elif len(w) == 1:
        w[0] = primacy_weight + recency_weight
        
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_16)

### DESIGN
**Validities (n_features=5):** [0.55, 0.9, 0.85, 0.8, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1]

**Rationale:** To quantitatively dissociate Recency-Biased Cue Overweighting (Advocated Theory) from Serial Position Dual-Overweighting (Competing Theory), we exploit the critical difference in their treatment of the first cue. The Advocated Theory treats the first cue exactly like any middle cue, weighting it strictly according to its stated validity (scaled by gamma). The Competing Theory, however, posits an independent, unnormalized 'primacy_weight' for the first cue, allowing it to dominate choices regardless of its stated validity. We use a 5-feature design where the first and last cues have low validities, while the middle cues have high validities. The trial set includes 'primacy conflicts' where the first cue favors one option while the high-validity middle cues favor the alternative (Advocated predicts middle cues win; Competing allows the first cue to win). We also include direct conflicts between the first and last cues to contrast the independent primacy and recency weights in the Competing Theory against the isolated recency weight in the Advocated Theory.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the choice proportion of an option supported solely by the first cue (Option A in Trials 1 and 3) when pitted against options supported by high-validity middle cues or the final cue. Under the Advocated Theory (Recency-Biased Cue Overweighting), the first cue is evaluated using its low stated validity, meaning Option A will almost never be chosen. Under the Competing Theory (Serial Position Dual-Overweighting), the first cue is assigned an independent, unnormalized 'primacy_weight' which allows Option A to be chosen at a significantly higher rate.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A is supported ONLY by the first cue
    # and Option B is supported either by strong middle cues (Trial 1) 
    # or the final cue (Trial 3).
    is_A_10000 = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0, 0])
    is_B_01100 = data['option_b_ratings'].apply(lambda x: list(x) == [0, 1, 1, 0, 0])
    is_B_00001 = data['option_b_ratings'].apply(lambda x: list(x) == [0, 0, 0, 0, 1])
    
    mask = is_A_10000 & (is_B_01100 | is_B_00001)
    
    if mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())


### RESULTS
- Predicted under pi_16 (simulated): 0.1219 (var=0.0063)
- Predicted under pi_18 (simulated): 0.6050 (var=0.0766)
- Observed on real data: 0.8650 (var=0.0113)

## EXPERIMENT 2 (proposed by pi_18)

### DESIGN
**Validities (n_features=5):** [0.55, 0.95, 0.9, 0.85, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Serial Position Dual-Overweighting (Advocated Theory) from Recency-Biased Cue Overweighting (Competing Theory), we exploit their critical difference in handling the first cue. The Advocated Theory assigns an independent, unnormalized 'primacy_weight' to the first cue, allowing it to dominate choices regardless of its stated validity. The Competing Theory treats the first cue like any middle cue, weighting it strictly by its stated validity (scaled by gamma). We use a 5-feature design where the first and last cues have low stated validities, while the middle cues have high validities. The trial set includes 'primacy conflicts' where the first cue favors one option while the high-validity middle cues favor the alternative (the Advocated Theory predicts the first cue can win; the Competing Theory predicts the middle cues will win). We also include direct conflicts between the first and last cues to contrast the dual independent weights in the Advocated Theory against the single recency weight in the Competing Theory.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject chooses the option favored by the first cue, specifically in 'primacy conflict' trials where the first cue contradicts the majority of the high-validity middle cues. The Advocated Theory (Serial Position Dual-Overweighting) predicts a higher choice rate for the first cue due to its unnormalized primacy weight, whereas the Competing Theory (Recency-Biased Cue Overweighting) predicts a low choice rate because the first cue is weighted only by its low stated validity.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    first_cue_choices = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        response = row['response']
        
        a_mid = sum(a[1:4])
        b_mid = sum(b[1:4])
        
        if a[0] > b[0] and a_mid < b_mid:
            first_cue_choices.append(1 if response == 0 else 0)
        elif b[0] > a[0] and b_mid < a_mid:
            first_cue_choices.append(1 if response == 1 else 0)
            
    if not first_cue_choices:
        return 0.0
    return float(np.mean(first_cue_choices))

### RESULTS
- Predicted under pi_16 (simulated): 0.3000 (var=0.0066)
- Predicted under pi_18 (simulated): 0.6843 (var=0.0383)
- Observed on real data: 0.8380 (var=0.0080)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    ttb_aligned = 0
    total = len(data)
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        # The features are already ordered by validity in the design (0 is highest)
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred == resp:
            ttb_aligned += 1
            
    return float(ttb_aligned / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.2467 (var=0.0072)
**Predicted under pi_16:** 0.1632 (var=0.0069)
**Predicted under pi_18:** 0.2983 (var=0.0121)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    tally_choices = np.where(a_wins > b_wins, 0, 1)
    matches = (data['response'].values[mask] == tally_choices[mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8444 (var=0.0148)
**Predicted under pi_16:** 0.8489 (var=0.0096)
**Predicted under pi_18:** 0.7650 (var=0.0237)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify critical trials where WADD and Tallying make strictly opposite predictions.
    # Trial 1: A has fewer but higher-validity features, B has more but lower-validity features.
    # WADD prefers A, Tallying prefers B.
    is_t1 = (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1))
    
    # Trial 5: The reversed version of Trial 1.
    # WADD prefers B, Tallying prefers A.
    is_t5 = (data['option_a_ratings'].apply(tuple) == (0, 0, 1, 1, 1)) & (data['option_b_ratings'].apply(tuple) == (1, 1, 0, 0, 0))
    
    # Count choices that align with the WADD model's predictions
    wadd_aligned_t1 = (data.loc[is_t1, 'response'] == 0).sum()
    wadd_aligned_t5 = (data.loc[is_t5, 'response'] == 1).sum()
    
    total_critical = is_t1.sum() + is_t5.sum()
    
    if total_critical == 0:
        return 0.5
        
    return float((wadd_aligned_t1 + wadd_aligned_t5) / total_critical)
```

**Observed (real) value:** 0.1317 (var=0.0093)
**Predicted under pi_16:** 0.1425 (var=0.0274)
**Predicted under pi_18:** 0.5375 (var=0.1269)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trial 1: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Identify trial 6: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    is_trial_6 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    # Calculate the proportion of choosing option B on these trials
    p_b_trial_1 = data.loc[is_trial_1, 'response'].mean()
    p_b_trial_6 = data.loc[is_trial_6, 'response'].mean()
    
    # Handle cases where a subject might not have these trials (though with 12 reps it's very unlikely)
    if pd.isna(p_b_trial_1) or pd.isna(p_b_trial_6):
        return 0.0
        
    # Return the difference in preference for B between Trial 1 and Trial 6
    return float(p_b_trial_1 - p_b_trial_6)

```

**Observed (real) value:** 0.6933 (var=0.0487)
**Predicted under pi_16:** 0.5017 (var=0.3071)
**Predicted under pi_18:** 0.0817 (var=0.4985)

### Experiment 7
**Design**
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    top_cue_chosen = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where the tally is tied and the top cue (index 0) breaks the tie
        if a_wins == b_wins and a[0] != b[0]:
            if a[0] > b[0]:
                top_cue_chosen.append(1 if row['response'] == 0 else 0)
            else:
                top_cue_chosen.append(1 if row['response'] == 1 else 0)
                
    if len(top_cue_chosen) == 0:
        return 0.5
    return float(np.mean(top_cue_chosen))
```

**Observed (real) value:** 0.4850 (var=0.0026)
**Predicted under pi_16:** 0.5121 (var=0.0038)
**Predicted under pi_18:** 0.6283 (var=0.0367)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    b_wins = (b_ratings > a_ratings).sum(axis=1)
    
    a_top = a_ratings[:, 0] > b_ratings[:, 0]
    b_top = b_ratings[:, 0] > a_ratings[:, 0]
    
    is_tie = (a_wins == b_wins)
    
    target_trials = is_tie & (a_top | b_top)
    
    if not np.any(target_trials):
        return 0.5
        
    responses = data['response'].values[target_trials]
    a_top_target = a_top[target_trials]
    b_top_target = b_top[target_trials]
    
    match = ( (responses == 0) & a_top_target ) | ( (responses == 1) & b_top_target )
    
    return float(np.mean(match))
```

**Observed (real) value:** 0.5283 (var=0.0043)
**Predicted under pi_16:** 0.8067 (var=0.0192)
**Predicted under pi_18:** 0.8567 (var=0.0286)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        ttb_pred = None
        for i in range(5):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        if ttb_pred is not None:
            matches.append(1 if resp == ttb_pred else 0)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3475 (var=0.0033)
**Predicted under pi_16:** 0.3650 (var=0.0024)
**Predicted under pi_18:** 0.4760 (var=0.0348)

### Experiment 10
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D numpy arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Tallying predictions: count features where one option strictly beats the other
    tally_a = np.sum(a_ratings > b_ratings, axis=1)
    tally_b = np.sum(b_ratings > a_ratings, axis=1)
    tally_c = np.where(tally_a > tally_b, 0, np.where(tally_b > tally_a, 1, -1))
    
    # Take-The-Best predictions: purely determined by the highest-validity feature (index 0)
    ttb_c = np.where(a_ratings[:, 0] > b_ratings[:, 0], 0, 1)
    
    # Isolate trials where the two heuristics make deterministic, opposite predictions
    mask = (tally_c != -1) & (tally_c != ttb_c)
    
    if not np.any(mask):
        return 0.5
        
    # Calculate the proportion of choices that align with the Tallying heuristic
    responses = data['response'].values[mask]
    tally_choices = tally_c[mask]
    
    return float(np.mean(responses == tally_choices))
```

**Observed (real) value:** 0.4975 (var=0.0028)
**Predicted under pi_16:** 0.5850 (var=0.0062)
**Predicted under pi_18:** 0.3513 (var=0.0311)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    conflict_trials = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_tup = tuple(a)
        b_tup = tuple(b)
        if a_tup == (1, 1, 0, 0, 0) and b_tup == (0, 0, 1, 1, 1):
            conflict_trials += 1
            if resp == 0:
                wadd_choices += 1
        elif a_tup == (0, 0, 1, 1, 1) and b_tup == (1, 1, 0, 0, 0):
            conflict_trials += 1
            if resp == 1:
                wadd_choices += 1
    return wadd_choices / conflict_trials if conflict_trials > 0 else 0.5
```

**Observed (real) value:** 0.1163 (var=0.0129)
**Predicted under pi_16:** 0.1869 (var=0.0485)
**Predicted under pi_18:** 0.5863 (var=0.1192)

### Experiment 12
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target_chosen = []
    for _, row in data.iterrows():
        a = tuple(int(x) for x in row['option_a_ratings'])
        b = tuple(int(x) for x in row['option_b_ratings'])
        
        # Identify the strict conflict trial
        is_A_target = (a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1))
        is_B_target = (b == (1, 1, 0, 0, 0) and a == (0, 0, 1, 1, 1))
        
        if is_A_target or is_B_target:
            chose_A = (row['response'] == 0)
            if (is_A_target and chose_A) or (is_B_target and not chose_A):
                target_chosen.append(1)
            else:
                target_chosen.append(0)
                
    if len(target_chosen) == 0:
        return 0.5
    return float(np.mean(target_chosen))
```

**Observed (real) value:** 0.1495 (var=0.0219)
**Predicted under pi_16:** 0.1505 (var=0.0126)
**Predicted under pi_18:** 0.5326 (var=0.1525)

### Experiment 13
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_A_ttb_A_choices = []
    tally_A_ttb_B_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 'A'
                break
            elif b[i] > a[i]:
                ttb_winner = 'B'
                break
                
        if a_wins == 3 and b_wins == 2:
            is_A = 1 if row['response'] == 0 else 0
            if ttb_winner == 'A':
                tally_A_ttb_A_choices.append(is_A)
            elif ttb_winner == 'B':
                tally_A_ttb_B_choices.append(is_A)
                
    mean_A_ttb_A = np.mean(tally_A_ttb_A_choices) if len(tally_A_ttb_A_choices) > 0 else 0.5
    mean_A_ttb_B = np.mean(tally_A_ttb_B_choices) if len(tally_A_ttb_B_choices) > 0 else 0.5
    
    return float(mean_A_ttb_A - mean_A_ttb_B)
```

**Observed (real) value:** 0.8075 (var=0.0287)
**Predicted under pi_16:** 0.7050 (var=0.0569)
**Predicted under pi_18:** 0.7612 (var=0.0310)

### Experiment 14
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    
    # Calculate tally scores
    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    
    # Identify tally tie trials
    ties = (a_wins == b_wins)
    if not np.any(ties):
        return 0.5
        
    # For tie trials, determine the TTB prediction
    # Feature 0 has the highest validity in this design
    a_f0 = a[ties, 0]
    b_f0 = b[ties, 0]
    
    responses = data['response'].values[ties]
    
    ttb_choices = np.where(a_f0 > b_f0, 0, np.where(b_f0 > a_f0, 1, -1))
    
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.5208 (var=0.0051)
**Predicted under pi_16:** 0.8037 (var=0.0190)
**Predicted under pi_18:** 0.8654 (var=0.0112)

### Experiment 15
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_A = (1, 1, 1, 0, 0, 0)
    target_B = (0, 0, 0, 1, 1, 1)
    
    a_match = data['option_a_ratings'].apply(lambda x: tuple(x) == target_A)
    b_match = data['option_b_ratings'].apply(lambda x: tuple(x) == target_B)
    idx1 = a_match & b_match
    
    a_match_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == target_B)
    b_match_rev = data['option_b_ratings'].apply(lambda x: tuple(x) == target_A)
    idx2 = a_match_rev & b_match_rev
    
    chose_target = 0
    total = 0
    
    if idx1.any():
        chose_target += (data.loc[idx1, 'response'] == 0).sum()
        total += idx1.sum()
        
    if idx2.any():
        chose_target += (data.loc[idx2, 'response'] == 1).sum()
        total += idx2.sum()
        
    if total == 0:
        return 0.5
        
    return float(chose_target / total)
```

**Observed (real) value:** 0.1832 (var=0.0124)
**Predicted under pi_16:** 0.2442 (var=0.0694)
**Predicted under pi_18:** 0.4095 (var=0.1307)

### Experiment 16
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    t5_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0, 0))
    
    chose_first_half_winner_t1 = (data[t1_mask]['response'] == 0).sum()
    chose_first_half_winner_t5 = (data[t5_mask]['response'] == 1).sum()
    
    total_relevant_trials = t1_mask.sum() + t5_mask.sum()
    if total_relevant_trials == 0:
        return 0.5
        
    return float((chose_first_half_winner_t1 + chose_first_half_winner_t5) / total_relevant_trials)
```

**Observed (real) value:** 0.1762 (var=0.0166)
**Predicted under pi_16:** 0.2981 (var=0.1138)
**Predicted under pi_18:** 0.5819 (var=0.1411)

### Experiment 17
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Extract option ratings as numpy arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate tallies for each option
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    # Identify trials where the tally is tied
    tie_mask = (a_wins == b_wins)
    tie_data = data[tie_mask].copy()
    
    if len(tie_data) == 0:
        return 0.0
    
    # Create a hashable trial identifier
    tie_data['trial_id'] = tie_data.apply(lambda r: tuple(r['option_a_ratings']) + tuple(r['option_b_ratings']), axis=1)
    
    # Calculate the proportion of times each subject chose Option A (response == 0) for each tally-tie trial type
    p_a = tie_data.groupby(['subject_id', 'trial_id'])['response'].apply(lambda x: (x == 0).mean())
    
    # Calculate the mean squared deviation from 0.5 (random guessing)
    sq_dev = (p_a - 0.5) ** 2
    
    return float(sq_dev.mean())
```

**Observed (real) value:** 0.1591 (var=0.0033)
**Predicted under pi_16:** 0.1528 (var=0.0041)
**Predicted under pi_18:** 0.1522 (var=0.0034)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t3_mask = data['a_str'] == '00111'
    t4_mask = data['a_str'] == '11100'
    
    t3_data = data[t3_mask]
    t4_data = data[t4_mask]
    
    if len(t3_data) == 0 or len(t4_data) == 0:
        return 0.0
        
    p_a_t3 = 1.0 - t3_data.groupby('subject_id')['response'].mean()
    p_a_t4 = 1.0 - t4_data.groupby('subject_id')['response'].mean()
    
    df = pd.DataFrame({'t3': p_a_t3, 't4': p_a_t4}).dropna()
    if len(df) == 0:
        return 0.0
        
    return float(np.mean((df['t4'] - df['t3'])**2))
```

**Observed (real) value:** 0.4773 (var=0.0539)
**Predicted under pi_16:** 0.5161 (var=0.0811)
**Predicted under pi_18:** 0.4818 (var=0.0945)

### Experiment 19
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tie_mask = a_wins == b_wins
    
    if not np.any(tie_mask):
        return 0.5
        
    a_tie = a_ratings[tie_mask]
    b_tie = b_ratings[tie_mask]
    responses = data['response'].values[tie_mask]
    
    ttb_preds = np.zeros(len(a_tie))
    for i in range(len(a_tie)):
        for j in range(a_tie.shape[1]):
            if a_tie[i, j] > b_tie[i, j]:
                ttb_preds[i] = 0
                break
            elif b_tie[i, j] > a_tie[i, j]:
                ttb_preds[i] = 1
                break
                
    return float(np.mean(responses == ttb_preds))
```

**Observed (real) value:** 0.5411 (var=0.0079)
**Predicted under pi_16:** 0.7722 (var=0.0165)
**Predicted under pi_18:** 0.8094 (var=0.0260)

### Experiment 20
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_ttb = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus only on trials where Tallying predicts a tie
        if a_wins == b_wins:
            # Determine Take-The-Best (TTB) prediction
            ttb_choice = -1
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_choice = 0
                    break
                elif b[i] > a[i]:
                    ttb_choice = 1
                    break
            
            if ttb_choice != -1:
                match_ttb.append(1.0 if row['response'] == ttb_choice else 0.0)
                
    if len(match_ttb) == 0:
        return 0.5
    return float(np.mean(match_ttb))
```

**Observed (real) value:** 0.6822 (var=0.0059)
**Predicted under pi_16:** 0.7967 (var=0.0193)
**Predicted under pi_18:** 0.8589 (var=0.0154)

### Experiment 21
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_predictions = {
        ((1, 1, 0, 0, 0), (0, 0, 1, 1, 1)): 0,
        ((0, 0, 1, 1, 1), (1, 1, 0, 0, 0)): 1,
        ((1, 0, 0, 0, 0), (0, 0, 0, 1, 1)): 0,
        ((0, 1, 0, 0, 0), (0, 0, 0, 1, 1)): 0
    }
    
    match_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if (a, b) in wadd_predictions:
            if row['response'] == wadd_predictions[(a, b)]:
                match_count += 1
            total_count += 1
            
    if total_count == 0:
        return 0.5
        
    return match_count / total_count

```

**Observed (real) value:** 0.1150 (var=0.0062)
**Predicted under pi_16:** 0.1733 (var=0.0254)
**Predicted under pi_18:** 0.3137 (var=0.0700)

### Experiment 22
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Identify trials where the total number of positive features is equal for A and B
    # In the experimental design, this corresponds exactly to trials 1 and 2.
    tie_trials = data[a_sums == b_sums]
    
    if len(tie_trials) == 0:
        return 0.5
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    # Tallying predicts exactly 0.5 (random guessing) because the feature counts are tied.
    # WADD predicts > 0.5 because Option A possesses the higher-validity features.
    return float((tie_trials['response'] == 0).mean())
```

**Observed (real) value:** 0.3400 (var=0.0140)
**Predicted under pi_16:** 0.4558 (var=0.0102)
**Predicted under pi_18:** 0.7392 (var=0.0560)

### Experiment 23
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Focus only on Tally-tie trials
        if np.sum(a > b) == np.sum(b > a):
            # Find the Take-The-Best (TTB) prediction
            # Validities are monotonically decreasing with index, so cue 0 is best
            for i in range(len(a)):
                if a[i] > b[i]:
                    matches.append(1 if row['response'] == 0 else 0)
                    break
                elif b[i] > a[i]:
                    matches.append(1 if row['response'] == 1 else 0)
                    break

    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.6178 (var=0.0052)
**Predicted under pi_16:** 0.7972 (var=0.0176)
**Predicted under pi_18:** 0.8328 (var=0.0162)

### Experiment 24
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tie_mask = (a_wins == b_wins)
    
    if not np.any(tie_mask):
        return 0.5
        
    a_tie = a_mat[tie_mask]
    b_tie = b_mat[tie_mask]
    resp_tie = data['response'].values[tie_mask]
    
    ttb_preds = []
    for i in range(len(a_tie)):
        a = a_tie[i]
        b = b_tie[i]
        pred = 0
        for j in range(len(a)):
            if a[j] > b[j]:
                pred = 0
                break
            elif b[j] > a[j]:
                pred = 1
                break
        ttb_preds.append(pred)
        
    ttb_preds = np.array(ttb_preds)
    matches = (resp_tie == ttb_preds)
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5033 (var=0.0079)
**Predicted under pi_16:** 0.6483 (var=0.0196)
**Predicted under pi_18:** 0.7554 (var=0.0153)

### Experiment 25
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_3 = (data['A_str'] == '11000') & (data['B_str'] == '00111')
    mask_5 = (data['A_str'] == '00111') & (data['B_str'] == '11000')
    
    chose_high_val = 0
    total = 0
    
    if mask_3.sum() > 0:
        chose_high_val += (data.loc[mask_3, 'response'] == 0).sum()
        total += mask_3.sum()
        
    if mask_5.sum() > 0:
        chose_high_val += (data.loc[mask_5, 'response'] == 1).sum()
        total += mask_5.sum()
        
    if total == 0:
        return 0.5
        
    return float(chose_high_val / total)

```

**Observed (real) value:** 0.1633 (var=0.0175)
**Predicted under pi_16:** 0.1775 (var=0.0319)
**Predicted under pi_18:** 0.5342 (var=0.1449)

### Experiment 26
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def target_chosen(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        target = (1, 1, 0, 0, 0, 0)
        alt = (0, 0, 1, 1, 1, 0)
        
        if a == target and b == alt:
            return 1.0 if row['response'] == 0 else 0.0
        elif b == target and a == alt:
            return 1.0 if row['response'] == 1 else 0.0
        return np.nan

    choices = data.apply(target_chosen, axis=1)
    val = np.nanmean(choices)
    if np.isnan(val):
        return 0.5
    return float(val)
```

**Observed (real) value:** 0.1333 (var=0.0172)
**Predicted under pi_16:** 0.7467 (var=0.0847)
**Predicted under pi_18:** 0.8333 (var=0.0294)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Check if it is a Tally tie
        if np.sum(a > b) == np.sum(b > a):
            # Take-The-Best prediction based on the highest validity feature (index 0)
            if a[0] > b[0]:
                matches.append(row['response'] == 0)
            elif b[0] > a[0]:
                matches.append(row['response'] == 1)
                
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5126 (var=0.0074)
**Predicted under pi_16:** 0.7622 (var=0.0156)
**Predicted under pi_18:** 0.8422 (var=0.0220)

### Experiment 28
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tie_mask = (a_wins == b_wins)
    
    if not np.any(tie_mask):
        return 0.5
        
    a_tie = a_ratings[tie_mask]
    b_tie = b_ratings[tie_mask]
    resp_tie = responses[tie_mask]
    
    ttb_winners = []
    for i in range(len(a_tie)):
        winner = -1
        for j in range(5):
            if a_tie[i, j] > b_tie[i, j]:
                winner = 0
                break
            elif b_tie[i, j] > a_tie[i, j]:
                winner = 1
                break
        ttb_winners.append(winner)
        
    ttb_winners = np.array(ttb_winners)
    valid_mask = (ttb_winners != -1)
    
    if not np.any(valid_mask):
        return 0.5
        
    match = (resp_tie[valid_mask] == ttb_winners[valid_mask])
    return float(np.mean(match))
```

**Observed (real) value:** 0.5867 (var=0.0101)
**Predicted under pi_16:** 0.7617 (var=0.0137)
**Predicted under pi_18:** 0.8450 (var=0.0168)

### Experiment 29
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score = 0
    count = 0
    
    for _, row in data.iterrows():
        a = tuple(int(x) for x in row['option_a_ratings'])
        b = tuple(int(x) for x in row['option_b_ratings'])
        resp = int(row['response'])
        
        # T1: Pure Tally ties (2-2). Drop 5th -> B wins (1-2). Target: B
        if a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 0, 0):
            if resp == 1: score += 1
            count += 1
        # T2: Pure Tally ties (2-2). Drop 5th -> A wins (2-1). Target: A
        elif a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 0, 1):
            if resp == 0: score += 1
            count += 1
        # T3: Pure Tally A wins (2-1). Drop 5th -> Tie (1-1). Target: B (attenuated advantage)
        elif a == (1, 0, 0, 0, 1) and b == (0, 0, 0, 1, 0):
            if resp == 1: score += 1
            count += 1
        # T4: Pure Tally A wins (2-1). Drop 5th -> A wins (2-0). Target: A (amplified advantage)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 0, 0, 1):
            if resp == 0: score += 1
            count += 1
        # T5: Pure Tally B wins (2-1). Drop 5th -> Tie (1-1). Target: A (attenuated advantage)
        elif a == (0, 0, 0, 1, 0) and b == (1, 0, 0, 0, 1):
            if resp == 0: score += 1
            count += 1
        # T6: Pure Tally B wins (2-1). Drop 5th -> B wins (2-0). Target: B (amplified advantage)
        elif a == (0, 0, 0, 0, 1) and b == (1, 1, 0, 0, 0):
            if resp == 1: score += 1
            count += 1
            
    if count == 0:
        return 0.5
    return float(score) / count
```

**Observed (real) value:** 0.1528 (var=0.0126)
**Predicted under pi_16:** 0.1769 (var=0.0170)
**Predicted under pi_18:** 0.2694 (var=0.0213)

### Experiment 30
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert option_a_ratings to string for easy matching
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 0, 1]
    t1_mask = data['A_str'] == '11000'
    # Trial 2: A=[0, 1, 0, 0, 1], B=[1, 0, 1, 0, 0]
    t2_mask = data['A_str'] == '01001'
    
    # Calculate probability of choosing A (response == 0)
    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_a_t2 = 1.0 - data.loc[t2_mask, 'response'].mean()
    
    # Handle edge cases where a subject might miss a trial type
    if pd.isna(p_a_t1):
        p_a_t1 = 0.5
    if pd.isna(p_a_t2):
        p_a_t2 = 0.5
        
    return p_a_t1 - p_a_t2

```

**Observed (real) value:** -0.7100 (var=0.0550)
**Predicted under pi_16:** -0.6562 (var=0.1524)
**Predicted under pi_18:** -0.0925 (var=0.5110)

### Experiment 31
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Calculate tally scores for each option
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    # Extract the final feature values
    final_a = a_mat[:, -1]
    final_b = b_mat[:, -1]
    
    # Identify "conflict" trials where Tallying predicts one option 
    # but the final feature favors the other.
    conflict_mask = ((a_wins > b_wins) & (final_b > final_a)) | ((b_wins > a_wins) & (final_a > final_b))
    
    if not np.any(conflict_mask):
        return 0.5
        
    resp = data['response'].values
    
    # Determine which option the final feature favors (0 for A, 1 for B)
    final_choice = np.where(final_a > final_b, 0, 1)
    
    # Calculate the proportion of choices on conflict trials that align with the final feature
    aligned = (resp[conflict_mask] == final_choice[conflict_mask])
    return float(np.mean(aligned))
```

**Observed (real) value:** 0.8422 (var=0.0217)
**Predicted under pi_16:** 0.7528 (var=0.0677)
**Predicted under pi_18:** 0.4917 (var=0.1207)

### Experiment 32
**Design**
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate tallies for each trial
    def tally_a(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return sum(1 for x, y in zip(a, b) if x > y)
        
    def tally_b(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return sum(1 for x, y in zip(a, b) if y > x)
        
    def last_feat_diff(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return a[-1] - b[-1]

    t_a = data.apply(tally_a, axis=1)
    t_b = data.apply(tally_b, axis=1)
    lf_diff = data.apply(last_feat_diff, axis=1)
    
    # Filter trials where tally(A) <= tally(B) and A's last feature > B's last feature
    mask = (t_a <= t_b) & (lf_diff > 0)
    
    if mask.sum() == 0:
        return 0.0
        
    # Proportion of choosing A (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.8200 (var=0.0146)
**Predicted under pi_16:** 0.7889 (var=0.0378)
**Predicted under pi_18:** 0.7356 (var=0.0192)

### Experiment 33
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify Trial 3 where A=[1, 0, 0, 0, 0] and B=[0, 1, 1, 1, 0]
    is_t3 = (
        (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) &
        (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 1, 0))
    )
    if not is_t3.any():
        return 0.0
    # Return the proportion of times option A was chosen (response == 0)
    return float((data.loc[is_t3, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5156 (var=0.0364)
**Predicted under pi_16:** 0.1911 (var=0.0376)
**Predicted under pi_18:** 0.6867 (var=0.1159)

### Experiment 34
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A wins solely on the first cue (primacy)
    # and Option B wins on a higher-validity middle cue.
    # In the design, this corresponds to Trial 1 and Trial 2 where A=[1, 0, 0, 0, 0].
    is_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    subset = data[is_target]
    if len(subset) == 0:
        return 0.0
    # Return the proportion of times Option A was chosen (response == 0)
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.8950 (var=0.0103)
**Predicted under pi_16:** 0.2658 (var=0.0147)
**Predicted under pi_18:** 0.8625 (var=0.0124)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The observed data across both experiments decisively demonstrates a massive primacy effect. Subjects consistently chose the option favored by the first cue even when it was opposed by multiple highly valid middle cues or the final cue (observed choice proportions > 0.83 in both Exp 1 and Exp 2). Theory 2 (pi_18) correctly anticipates this primacy effect via its dual-overweighting mechanism but significantly underestimates its magnitude. Theory 1 (pi_16) completely fails to capture this behavior, as it treats the first cue strictly according to its stated validity, resulting in predictions that are far too low. Because Theory 1 entirely misses the fundamental primacy-driven nature of these decisions, it is degenerate.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Primacy-Dominant Anchoring' theory. This theory should posit that decision-makers use the first cue as a powerful anchor that overwhelmingly dominates the evaluation process. While the final cue may receive a secondary recency boost, the primacy weight should be structurally much larger than both the recency weight and the middle cue validities. This differs from Theory 2 by explicitly enforcing a hierarchy where primacy is the primary driver of choice, rather than treating primacy and recency as roughly symmetric boundary effects."
}
```

## Usage

```json
{
  "prompt_token_count": 25330,
  "candidates_token_count": 286,
  "total_token_count": 26674
}
```
