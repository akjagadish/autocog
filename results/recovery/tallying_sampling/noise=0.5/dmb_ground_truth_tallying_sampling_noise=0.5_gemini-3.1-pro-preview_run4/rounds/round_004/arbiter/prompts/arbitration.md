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
Strategy Mixture Theory: Decision-makers are heterogeneous in their strategies. A proportion of choices are made using a compensatory Weighted Additive (WADD) strategy that integrates cue validities and subjective weights, while the remainder rely on a simpler Tallying (Equal Weight) heuristic that merely counts the number of positive features. Behavior on any given trial is a probabilistic mixture of these two strategies, with the mixture proportion varying across individuals. This naturally explains why aggregate behavior falls between the pure predictions of WADD and Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_mix = float(parameters["w_mix"])
    
    # WADD strategy component
    wadd_scores = stim @ (validities * w)
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Tallying strategy component
    tally_scores = stim.sum(axis=1)
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_mix * p_wadd + (1.0 - w_mix) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_6
Heuristic Toolbox (TTB + Tallying Mixture): Decision-makers are equipped with a repertoire of fast-and-frugal heuristics rather than complex compensatory algorithms. Specifically, individuals probabilistically switch between a non-compensatory Take-The-Best (TTB) heuristic and an equal-weight Tallying heuristic. TTB sequentially searches cues in order of validity and decides based on the first discriminating cue, capturing lexicographic decision-making. Tallying simply counts the number of positive features, capturing instances where cues are integrated equally. This 2-way mixture accounts for choices where subjects either rely on a single dominant cue or fall back to simple feature counting, without the excessive noise introduced by subjective compensatory weighting.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_ttb = float(parameters["w_ttb"])
    
    # 1. Take-The-Best (TTB) strategy component
    p_ttb = np.array([0.5, 0.5])
    # Sort indices by validity descending
    order = np.argsort(validities)[::-1]
    for idx in order:
        if stim[0, idx] != stim[1, idx]:
            if stim[0, idx] > stim[1, idx]:
                p_ttb = np.array([1.0, 0.0])
            else:
                p_ttb = np.array([0.0, 1.0])
            break
            
    # 2. Tallying strategy component
    tally_scores = stim.sum(axis=1)
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of TTB and Tallying
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.8, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate Strategy Mixture Theory (WADD + Tallying) from the Heuristic Toolbox (TTB + Tallying), we must separate the compensatory cue integration of WADD from the non-compensatory lexicographic stopping rule of Take-The-Best (TTB), while controlling for the influence of Tallying (equal weighting). We use 5 features with a steep drop-off in validities. In 'Tally-Tied' trials, both options have the same number of positive features, neutralizing the Tallying component for both models. However, Option A possesses the single highest validity cue (favored by TTB), while Option B possesses a coalition of slightly less valid cues whose sum exceeds the highest validity cue (favored by WADD). Strategy Mixture Theory predicts a strong preference for Option B on these trials (driven by WADD, with Tallying indifferent), whereas the Heuristic Toolbox predicts a strong preference for Option A (driven by TTB, with Tallying indifferent). We also include trials where Tallying aligns with WADD but opposes TTB, further widening the predicted choice probabilities between the two theories.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In this experimental design, the predictions of Take-The-Best (TTB) and the Weighted Additive (WADD) strategy are diametrically opposed on 100% of the trials. By measuring the proportion of choices that align with the TTB prediction (i.e., choosing the option with the highest validity cue), we can strongly distinguish the Heuristic Toolbox theory (which incorporates TTB) from the Strategy Mixture Theory (which relies on WADD). The Strategy Mixture Theory will consistently produce values below 0.5 (as both WADD and Tallying either oppose TTB or are indifferent), whereas the Heuristic Toolbox will produce significantly higher values due to the presence of the TTB heuristic.

Source:
def metric(data: pd.DataFrame) -> float:
    # Take-The-Best (TTB) relies on the most valid cue, which is feature 0.
    # In this specific experimental design, Option A has feature 0 = 1 on trials 1, 3, 5
    # and Option B has feature 0 = 1 on trials 2, 4, 6.
    # Thus, TTB always predicts choosing the option with feature 0 = 1.
    # We calculate the proportion of choices aligning with TTB's prediction.
    ttb_predictions = data['option_a_ratings'].apply(lambda x: 0 if x[0] == 1 else 1)
    return float((data['response'] == ttb_predictions).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.3165 (var=0.0250)
- Predicted under pi_6 (simulated): 0.5481 (var=0.0433)
- Observed on real data: 0.3879 (var=0.0036)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Heuristic Toolbox (TTB + Tallying) from Strategy Mixture Theory (WADD + Tallying), we must separate the non-compensatory lexicographic nature of TTB from the compensatory integration of WADD, while controlling for the Tallying (equal weight) component present in both theories. We use 5 features with a linear drop-off in validities. The critical trials are 'Tally-Tied' where both options have the same number of positive features, rendering the Tallying component indifferent. In these trials, Option A possesses the single highest-validity cue (driving TTB to choose A), while Option B possesses a coalition of slightly less valid cues whose sum exceeds the highest-validity cue (driving WADD to choose B). The Heuristic Toolbox predicts a strong preference for Option A on these trials, whereas Strategy Mixture Theory predicts a strong preference for Option B. We also include trials where WADD is indifferent but TTB has a strict preference, and trials where Tallying opposes TTB but aligns with WADD, providing a full spectrum of dissociative predictions.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the 'Tally-Tied' trials where both options have an equal number of positive features, effectively neutralizing the Tallying component present in both theories. In these specific trials, Option A possesses the single highest-validity cue, leading the Heuristic Toolbox (TTB) to strongly prefer Option A. Conversely, Option B has a coalition of cues whose combined validity often outweighs Option A's, leading the Strategy Mixture Theory (WADD) to prefer Option B or remain indifferent. Therefore, the proportion of choices for Option A on these trials will be significantly higher under the Heuristic Toolbox than under Strategy Mixture Theory.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Tallying is indifferent (equal number of positive features)
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    tied_mask = a_sums == b_sums
    
    if tied_mask.sum() == 0:
        return 0.5
        
    # In these 'Tally-Tied' trials, TTB consistently favors Option A due to the highest validity cue,
    # whereas WADD tends to favor Option B (or is indifferent) due to compensatory integration.
    # We calculate the proportion of times Option A was chosen (response == 0).
    return float((data.loc[tied_mask, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.4917 (var=0.0154)
- Predicted under pi_6 (simulated): 0.7104 (var=0.0205)
- Observed on real data: 0.5133 (var=0.0060)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for j in range(len(a)):
            if a[j] != b[j]:
                ttb_pred = 0 if a[j] > b[j] else 1
                if row['response'] == ttb_pred:
                    ttb_matches += 1
                total += 1
                break
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.3511 (var=0.0061)
**Predicted under pi_4:** 0.2527 (var=0.0083)
**Predicted under pi_6:** 0.5433 (var=0.0448)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.6604 (var=0.0042)
**Predicted under pi_4:** 0.7085 (var=0.0149)
**Predicted under pi_6:** 0.4725 (var=0.0397)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify the critical dissociation trial (Trial 1 in the design)
    # Option A has more positive features (3 vs 2), favoring Tallying.
    # Option B has the two most valid features, favoring WADD.
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
                 
    if is_trial_1.sum() == 0:
        return 0.5
        
    # Return the proportion of times the subject chose Option A (response == 0)
    return float(np.mean(data.loc[is_trial_1, 'response'] == 0))
```

**Observed (real) value:** 0.7067 (var=0.0162)
**Predicted under pi_4:** 0.6433 (var=0.0614)
**Predicted under pi_6:** 0.5400 (var=0.0792)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_wadd_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        # Trial 1: WADD favors the option with fewer, but higher-validity features.
        # Tallying strictly favors the option with more features.
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return 0 == row['response']
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return 1 == row['response']
        return np.nan

    choices = data.apply(get_wadd_choice, axis=1).dropna()
    
    if len(choices) == 0:
        return 0.5
        
    return float(choices.mean())
```

**Observed (real) value:** 0.3050 (var=0.0157)
**Predicted under pi_4:** 0.3400 (var=0.0725)
**Predicted under pi_6:** 0.5025 (var=0.0659)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1: WADD prefers A (1.8 vs 1.7), but Tallying prefers B (3 vs 2).
    t1_mask = (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1))
    
    # Trial 7: WADD prefers A (2.05 vs 1.45), and Tallying prefers A (3 vs 2).
    t7_mask = (data['option_a_ratings'].apply(tuple) == (0, 1, 1, 1, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (1, 0, 0, 0, 1))
    
    if t1_mask.sum() == 0 or t7_mask.sum() == 0:
        return 0.0
        
    # Difference in choice rate for B between the conflict trial and the agreement trial.
    # Subtracting the baseline noise/lapse rate controls for subject-specific epsilon variance.
    return float(data.loc[t1_mask, 'response'].mean() - data.loc[t7_mask, 'response'].mean())
```

**Observed (real) value:** 0.4267 (var=0.0718)
**Predicted under pi_4:** 0.4117 (var=0.1343)
**Predicted under pi_6:** -0.0417 (var=0.1762)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Create a unique string identifier for each trial type based on the feature vectors
    t_a = data['option_a_ratings'].apply(tuple).astype(str)
    t_b = data['option_b_ratings'].apply(tuple).astype(str)
    df = data.assign(trial_id=t_a + "_" + t_b)
    
    def subj_metric(subj_df):
        # Calculate the mean response (proportion of B choices) for each of the 8 trial types
        t_means = subj_df.groupby('trial_id')['response'].mean()
        overall_mean = t_means.mean()
        # Calculate Mean Absolute Deviation (MAD) across the trial types
        mad = (t_means - overall_mean).abs().mean()
        # The metric combines the overall bias towards B and the consistency across trial types
        return float(overall_mean - mad)
        
    if df['subject_id'].nunique() > 1:
        return float(df.groupby('subject_id').apply(subj_metric).mean())
    else:
        return float(subj_metric(df))
```

**Observed (real) value:** 0.5993 (var=0.0045)
**Predicted under pi_4:** 0.5469 (var=0.0305)
**Predicted under pi_6:** 0.3872 (var=0.0441)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t2_mask = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1, 0))
    t3_mask = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 0, 0))
    p_a_t2 = 1.0 - data[t2_mask]['response'].mean()
    p_a_t3 = 1.0 - data[t3_mask]['response'].mean()
    return float(p_a_t3 - p_a_t2)
```

**Observed (real) value:** 0.1467 (var=0.0502)
**Predicted under pi_4:** 0.2033 (var=0.0412)
**Predicted under pi_6:** 0.1383 (var=0.0582)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Trial 7: A=[0, 1, 1, 0, 0], B=[0, 0, 0, 1, 1]
    # Trial 8: A=[1, 1, 0, 1, 0], B=[0, 0, 1, 1, 1]
    # In these trials, Tally is tied (2v2 and 3v3). 
    # Non-linear scaling strictly preserves the ordinality of validities: A's transformed validities 
    # always sum to a higher value than B's, regardless of gamma (except at exactly gamma=0 where they tie).
    # Thus, Non-linear scaling almost invariably chooses A (response=0).
    # Strategy Mixture uses WADD with uniformly random subjective weights per feature. Because the weights 
    # are random, WADD will sometimes incorrectly prefer B, inflating the choice proportion of B.
    
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        return (a == (0, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1)) or \
               (a == (1, 1, 0, 1, 0) and b == (0, 0, 1, 1, 1))
               
    mask = data.apply(is_target_trial, axis=1)
    if not mask.any():
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())

```

**Observed (real) value:** 0.5383 (var=0.0087)
**Predicted under pi_4:** 0.3942 (var=0.0220)
**Predicted under pi_6:** 0.2883 (var=0.0223)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across almost all experiments (1 through 10), Theory 1 (Strategy Mixture Theory: WADD + Tallying) consistently outperforms Theory 2 (Heuristic Toolbox: TTB + Tallying). Theory 2 predicts extreme, lexicographic-driven choice patterns that do not align with the observed data, which shows much more graded, compensatory behavior. While Theory 1 is generally strong, it occasionally under-predicts or over-predicts certain choice proportions, suggesting room for an alternative compensatory account. Since the non-compensatory TTB assumption of Theory 2 is fundamentally contradicted by the data, Theory 2 should be discarded in favor of a brand-new theory.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Non-linear Validity Scaling Theory'. Instead of a mixture of distinct strategies (like WADD and Tallying) or strict lexicographic rules (like TTB), this theory posits a single integrated compensatory mechanism where the subjective weight of each cue is a non-linear transformation of its objective validity (e.g., w_i = v_i^gamma). This allows the model to smoothly interpolate between equal weighting (gamma -> 0), objective WADD (gamma = 1), and hyper-focus on the most valid cues approximating TTB (gamma -> infinity). This provides a more elegant, unified explanation for the graded choice probabilities observed across the experiments without relying on probabilistic strategy switching."
}
```

## Usage

```json
{
  "prompt_token_count": 9083,
  "candidates_token_count": 321,
  "total_token_count": 10390
}
```
