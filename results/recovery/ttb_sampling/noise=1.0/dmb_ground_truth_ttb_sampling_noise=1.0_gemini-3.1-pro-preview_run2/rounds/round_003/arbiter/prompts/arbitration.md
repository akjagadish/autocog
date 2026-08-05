# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_5
Random Choice / Minimal Effort: In the absence of trial-by-trial feedback and when faced with complex multi-attribute binary arrays, participants largely abandon systematic cue-based strategies. Instead, they resort to minimal effort processing, which manifests as random guessing or behavior dominated by an extremely high lapse rate. Any residual systematicity is extremely weak, resulting in choice probabilities that are consistently very close to 0.5 across all experimental conditions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Minimal effort evaluation (e.g., simple tallying of 1s)
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Dominated by a extremely high lapse rate (epsilon near 1.0)
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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To dissociate Take The Best (TTB) from the Random Choice / Minimal Effort model, we employ a 5-feature design with a steep validity hierarchy (0.95 to 0.55). The critical trials pit the single most valid feature against all remaining features. TTB predicts a strong preference for the option with the highest-validity feature, as it sequentially evaluates and stops at the first discriminating cue. In contrast, the Random Choice model predicts choices that are either perfectly random (due to a near 1.0 lapse rate) or slightly leaning toward the option with more positive features (due to minimal effort tallying). The stark difference in predicted choice probabilities (strong systematic preference vs. ~0.5) will clearly adjudicate between the models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In all experimental trials, exactly one option possesses the most valid feature (Feature 0, validity 0.95). The Take The Best (TTB) model predicts that subjects will predominantly choose the option with this highest-validity feature, resulting in a proportion significantly greater than 0.5. In contrast, the Random Choice / Minimal Effort model predicts that choices will be heavily influenced by a near-total lapse rate, pulling the proportion of choices for the Feature 0 option very close to 0.5. Measuring the overall proportion of trials where the subject chooses the option with Feature 0 = 1 will effectively discriminate between the systematic preference of TTB and the random guessing of the competing theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_feat0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_feat0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    chose_feat0 = ((a_feat0 == 1) & (b_feat0 == 0) & (data['response'] == 0)) | \
                  ((a_feat0 == 0) & (b_feat0 == 1) & (data['response'] == 1))
                  
    return float(chose_feat0.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.4260 (var=0.0061)
- Predicted under pi_5 (simulated): 0.5044 (var=0.0033)
- Observed on real data: 0.5025 (var=0.0023)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To cleanly dissociate the Random Choice / Minimal Effort theory from Take The Best (TTB), we use a 5-feature design with a clear validity hierarchy. TTB predicts strong, systematic preferences based on the most valid discriminating cue, regardless of the sheer number of cues favoring the other option. In contrast, the Random Choice theory predicts that participants will exhibit a massive lapse rate and rely on minimal effort tallying, leading to choice probabilities near 0.5 across all trials. By pitting one or two high-validity cues against multiple lower-validity cues, and including trials where the number of positive cues is equal but validities differ, we can expose the lack of systematic cue-based processing if the Random Choice theory holds true.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The Random Choice theory predicts that subjects will guess with probabilities very close to 0.5 on all trials, meaning the empirical proportion of choosing Option B will tightly cluster around 0.5 (subject only to binomial noise). The Take The Best (TTB) theory predicts systematic preferences on most trials driven by the highest-validity discriminating cue, pushing choice proportions towards 0 or 1. By calculating the mean absolute deviation of choice proportions from 0.5 across the different trial types for each subject, we can cleanly separate the two theories. Random Choice will yield a value near 0.147 (the expected deviation for 12 trials at p=0.5), while TTB will yield a significantly higher value.

Source:
def metric(data: pd.DataFrame) -> float:
    # Create a hashable key for each unique trial type
    trial_keys = data['option_a_ratings'].apply(tuple) + data['option_b_ratings'].apply(tuple)
    
    # Calculate the proportion of 'Option B' choices (response == 1) for each subject and trial type
    props = data.groupby([data['subject_id'], trial_keys])['response'].mean()
    
    # Calculate the mean absolute deviation from 0.5 for each subject across all trial types
    # Random choice predicts this will be close to the binomial expectation for p=0.5 (approx 0.147 for n=12)
    # TTB predicts much higher deviations as choices are driven by cue validities
    mad_per_subject = (props - 0.5).abs().groupby('subject_id').mean()
    
    return float(mad_per_subject.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1727 (var=0.0022)
- Predicted under pi_5 (simulated): 0.1177 (var=0.0009)
- Observed on real data: 0.1138 (var=0.0011)

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
**Predicted under pi_5:** 0.5041 (var=0.0026)

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
**Predicted under pi_5:** 0.4850 (var=0.0090)

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
**Predicted under pi_5:** 0.4827 (var=0.0034)

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
**Predicted under pi_5:** 0.4913 (var=0.0069)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.4946 (var=0.0022)
**Predicted under pi_3:** 0.6038 (var=0.0059)
**Predicted under pi_5:** 0.5031 (var=0.0030)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.4983 (var=0.0015)
**Predicted under pi_3:** 0.5342 (var=0.0034)
**Predicted under pi_5:** 0.5058 (var=0.0020)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 8 experiments, the observed metrics consistently hover around 0.5 (or the expected value for random guessing). Theory 2 (pi_5: Random Choice / Minimal Effort) predicts this near-random behavior perfectly, capturing the lack of systematic cue-based strategies in the absence of trial-by-trial feedback. Theory 1 (pi_3: Probabilistic Take The Best) repeatedly predicts strong systematic deviations from 0.5 (e.g., relying on the highest-validity cue), which are entirely absent in the real data. Since pi_5 effectively explains the data as largely random guessing, pi_3 is degenerate and should be replaced with a new theory that explores whether there is any faint, alternative systematicity (like a weak tallying or position bias) rather than a strict validity hierarchy.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory, such as 'Weak Tallying with Position Bias' or 'Satisficing with High Lapse'. Since TTB's strict hierarchy completely fails, the new theory should assume that participants mostly guess but might possess a very weak, noisy preference for options with simply a higher total number of positive features (tallying) or features on the extreme left (reading order bias). This will serve as a more realistic competitor to the pure minimal effort/random choice model of pi_5, testing if there's any residual non-random signal in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 7618,
  "candidates_token_count": 326,
  "total_token_count": 9403
}
```
