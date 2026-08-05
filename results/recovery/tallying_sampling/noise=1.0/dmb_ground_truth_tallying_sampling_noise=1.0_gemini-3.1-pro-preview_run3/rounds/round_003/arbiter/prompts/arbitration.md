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
Weighted Additive Model (WADD) with Subjective Validities: Decision-makers integrate all available features by computing a weighted sum of cue values for each option. The weights are subjective validities, modeled as a power transformation of the objective validities. Choices are made probabilistically using a softmax function over the options' weighted sums, along with an independent lapse rate. Limiting the softmax inverse temperature ensures higher decision noise, which tempers overconfidence when cue validities conflict.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform objective validities into subjective weights
    subj_weights = validities ** gamma
    
    # Compute WADD scores
    scores = stim @ subj_weights
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Equal Weights (Tallying)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    
    # Count the number of positive features for each option
    tally_a = np.sum(stim[0])
    tally_b = np.sum(stim[1])
    
    if tally_a > tally_b:
        scores = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        scores = np.array([0.0, 1.0])
    else:
        # Guess randomly if there is a tie
        return np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary tally score
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Independent lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive Model (WADD) from Equal Weights (Tallying), we exploit Tallying's strict reliance on the sheer count of positive features, completely ignoring cue validities. We use a 5-feature environment with highly skewed validities (e.g., two highly valid cues and three less valid cues). This allows us to create 'conflict' trials where one option has fewer positive cues but a higher total validity sum (e.g., Option A has the 2 most valid cues, Option B has the 3 least valid cues). Tallying deterministically prefers the option with more cues (Option B), whereas WADD integrates the validities and prefers the option with higher total validity (Option A). Furthermore, we include 'tie' trials where both options have the same number of positive cues but drastically different validity sums. Tallying predicts random guessing (50/50) on these trials, while WADD predicts strong preferences based on the validities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Weighted Additive Model (WADD) on strictly conflicting trials. On these specific trials, one option has fewer positive features but higher total validity, while the other option has more positive features but lower total validity. WADD predicts a preference for the option with higher total validity (resulting in a metric value > 0.5), whereas Tallying predicts a deterministic preference for the option with more features (resulting in a metric value < 0.5).

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data_a = data['option_a_ratings'].apply(tuple)
    data_b = data['option_b_ratings'].apply(tuple)
    
    mask1 = (data_a == (1, 1, 0, 0, 0)) & (data_b == (0, 0, 1, 1, 1))
    mask2 = (data_a == (0, 0, 1, 1, 1)) & (data_b == (1, 1, 0, 0, 0))
    
    wadd_aligned = 0
    total = 0
    
    if mask1.sum() > 0:
        wadd_aligned += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.sum() > 0:
        wadd_aligned += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(wadd_aligned / total)

### RESULTS
- Predicted under pi_5 (simulated): 0.5417 (var=0.0137)
- Predicted under pi_4 (simulated): 0.3633 (var=0.0202)
- Observed on real data: 0.4967 (var=0.0110)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate Equal Weights (Tallying) from the Weighted Additive Model (WADD), we exploit Tallying's strict reliance on the count of positive features and its complete disregard for cue validities. We use a 5-feature environment with highly skewed validities (e.g., two highly valid cues and three less valid cues). This allows us to create 'conflict' trials where one option has fewer positive cues but a higher total validity sum (e.g., Option A has the 2 most valid cues, Option B has the 3 least valid cues). Tallying deterministically prefers the option with more cues (Option B), whereas WADD integrates the validities and prefers the option with higher total validity (Option A). Furthermore, we include 'tie' trials where both options have the same number of positive cues but drastically different validity sums. Tallying predicts random guessing (50/50) on these trials, while WADD predicts strong preferences based on the validities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of choices for Option B on the specific conflict trial (Trial 1). On this trial, Option B has 3 positive cues (the least valid ones) while Option A has 2 positive cues (the most valid ones). Tallying strictly counts the number of cues and therefore deterministically prefers Option B. The Weighted Additive Model (WADD), however, integrates cue validities and generally finds the sum of Option A's highly valid cues to outweigh Option B's less valid cues, preferring Option A. Thus, Tallying predicts a choice proportion for B significantly above 0.5, while WADD predicts a choice proportion below 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify the specific conflict trial where Option A has 2 highly valid cues
    # and Option B has 3 less valid cues. We can uniquely identify this trial
    # because it is the only one where Option B has exactly 3 positive cues.
    b_sums = data['option_b_ratings'].apply(sum)
    trial_1_mask = b_sums == 3
    
    if trial_1_mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times the subject chose Option B (response == 1)
    return float(data.loc[trial_1_mask, 'response'].mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.4667 (var=0.0197)
- Predicted under pi_4 (simulated): 0.6283 (var=0.0381)
- Observed on real data: 0.4867 (var=0.0226)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_cue1_winner(row):
        if row['option_a_ratings'][0] > row['option_b_ratings'][0]:
            return 0
        elif row['option_b_ratings'][0] > row['option_a_ratings'][0]:
            return 1
        return -1
    
    cue1_winners = data.apply(get_cue1_winner, axis=1)
    valid_trials = cue1_winners != -1
    if not valid_trials.any():
        return 0.5
    
    return np.mean(data.loc[valid_trials, 'response'] == cue1_winners[valid_trials])
```

**Observed (real) value:** 0.5071 (var=0.0028)
**Predicted under pi_5:** 0.5120 (var=0.0043)
**Predicted under pi_4:** 0.4778 (var=0.0027)

### Experiment 4
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    conflict_trials = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            if resp == 0: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 1, 1, 1) and b == (1, 0, 0, 0):
            if resp == 1: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            if resp == 0: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 0, 1, 1) and b == (0, 1, 0, 0):
            if resp == 1: ttb_matches += 1
            conflict_trials += 1
            
    if conflict_trials == 0:
        return 0.5
    return float(ttb_matches / conflict_trials)
```

**Observed (real) value:** 0.4822 (var=0.0067)
**Predicted under pi_5:** 0.5056 (var=0.0124)
**Predicted under pi_4:** 0.3261 (var=0.0239)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['chose_a'] = (data['response'] == 0).astype(float)
    
    # We focus on the '3v1' trials where WADD predicts near-deterministic choices
    target_trials = ['0111', '1110', '1011']
    df_target = data[data['a_str'].isin(target_trials)]
    
    if df_target.empty:
        return 0.5
        
    # Calculate the proportion of A choices per subject and trial type
    grouped = df_target.groupby(['subject_id', 'a_str'])['chose_a'].mean()
    
    # Consistency is how far the choice proportion is from 0.5 (i.e., max(P(A), P(B)))
    consistency = np.maximum(grouped, 1.0 - grouped)
    
    return float(consistency.mean())
```

**Observed (real) value:** 0.6489 (var=0.0031)
**Predicted under pi_5:** 0.6422 (var=0.0035)
**Predicted under pi_4:** 0.6970 (var=0.0079)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Objective validities used in the experiment
    v = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    # Reconstruct the feature matrices
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate objective scores and differences
    score_a = a_ratings @ v
    score_b = b_ratings @ v
    diff = score_a - score_b
    
    # Select "easy" trials with a large score difference (abs(diff) >= 0.65)
    # In this design, this cleanly isolates Trials 7, 9, and 10.
    mask = np.abs(diff) >= 0.65
    
    if not np.any(mask):
        return 0.5
        
    # Determine if the subject chose the option with the higher objective score
    is_correct = ((diff > 0) & (data['response'] == 0)) | ((diff < 0) & (data['response'] == 1))
    
    # Return the simple accuracy on these easy trials
    return float(is_correct[mask].mean())
```

**Observed (real) value:** 0.5233 (var=0.0124)
**Predicted under pi_5:** 0.5592 (var=0.0141)
**Predicted under pi_4:** 0.6708 (var=0.0215)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    return float(np.mean(data['response'] == 0))

```

**Observed (real) value:** 0.5075 (var=0.0019)
**Predicted under pi_5:** 0.4852 (var=0.0039)
**Predicted under pi_4:** 0.3387 (var=0.0148)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    m1 = a_str == '11000'
    m2 = a_str == '10100'
    m4 = a_str == '01111'
    m7 = a_str == '00111'
    
    def get_p_a(mask):
        subset = data.loc[mask, 'response']
        if len(subset) == 0:
            return 0.5
        return float((subset == 0).mean())
        
    p1 = get_p_a(m1)
    p2 = get_p_a(m2)
    p4 = get_p_a(m4)
    p7 = get_p_a(m7)
    
    return float((p4 - p7) + (p1 - p2))
```

**Observed (real) value:** 0.0133 (var=0.1054)
**Predicted under pi_5:** 0.1900 (var=0.0939)
**Predicted under pi_4:** -0.0233 (var=0.0692)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across almost all experiments, Theory 1 (pi_5 - WADD with Subjective Validities) significantly outperforms Theory 2 (pi_4 - Tallying). Tallying frequently predicts extreme choice proportions (e.g., in Exp 1, Exp 2, Exp 4, and Exp 7) because it deterministically counts features while ignoring their validities. The observed data, however, often hovers near 0.5, indicating either random guessing on difficult conflict trials or a compensatory mechanism heavily tempered by decision noise. WADD handles this much better by integrating validities and using a softmax temperature. Since Tallying is a poor fit for the aggregate behavior, it should be replaced with a stronger competitor from the heuristics literature.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Tallying model with a 'Take-The-Best' (TTB) heuristic model. TTB is a classic lexicographic strategy where decision-makers inspect cues in descending order of their validities and base their choice entirely on the first cue that discriminates between the two options. To account for the high degree of randomness observed in the data (with many metrics near 0.5), the new model must incorporate significant decision noise, either through a high lapse rate or a probabilistic application of the stopping rule. This will provide a strong, non-compensatory alternative to WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 7483,
  "candidates_token_count": 313,
  "total_token_count": 8987
}
```
