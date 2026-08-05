# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5_1" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5_1
High-Temperature Weighted Additive (WADD) Theory: Subjects attempt to integrate all available features by computing a weighted sum of feature differences, using subjective validities as weights. However, their decision process is subject to extremely high cognitive noise (a very low inverse temperature, beta, in the softmax choice rule). This results in behavior that appears nearly random, capturing the high lapse rates observed across experiments, while still retaining a slight sensitivity to large aggregate differences in option quality.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compute scores as weighted sum of features
    scores = stim @ validities
    
    # Extremely high cognitive noise (low beta)
    beta = float(parameters["beta"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Random Guessing / High-Lapse Baseline: When confronted with complex multi-attribute choices involving cardinal feature magnitudes that do not easily yield to simple heuristics, decision makers become overwhelmed and resort to random guessing. Their choices reflect a near-uniform probability distribution over the available options, occasionally influenced by a negligible spatial or option-order bias.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    bias = float(parameters["bias"])
    return np.array([bias, 1.0 - bias])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5_1)

### DESIGN
**Validities (n_features=4):** [1.0, 0.9, 0.6, 0.5]

**rating_max:** 10000

**Trial pairs (n=8):**
  trial 1: A=[10000, 10000, 0, 0]  B=[0, 0, 10000, 10000]
  trial 2: A=[0, 0, 10000, 10000]  B=[10000, 10000, 0, 0]
  trial 3: A=[10000, 0, 10000, 0]  B=[0, 10000, 0, 10000]
  trial 4: A=[0, 10000, 0, 10000]  B=[10000, 0, 10000, 0]
  trial 5: A=[10000, 10000, 10000, 10000]  B=[0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0]  B=[10000, 10000, 10000, 10000]
  trial 7: A=[10000, 5000, 0, 0]  B=[0, 0, 5000, 10000]
  trial 8: A=[0, 0, 5000, 10000]  B=[10000, 5000, 0, 0]

**Rationale:** To dissociate High-Temperature WADD from Random Guessing, we must overcome the extremely low beta parameter (high cognitive noise) that pushes WADD's predictions toward 50%. By employing an exceptionally large rating scale (rating_max = 10000) and pitting high-validity features against low-validity features, we generate massive absolute differences in the validity-weighted scores (up to 30,000 points). For High-Temperature WADD, these extreme differences scale up through the low beta (e.g., 30,000 * 0.0001 = 3.0), producing a robust and graded deviation from 50% choice probability that tracks the weighted sum. The Random Guessing model, completely blind to stimulus magnitudes, will predict a flat ~50% probability across all these extreme variations, creating a stark quantitative dissociation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice aligns with the option favored by the Weighted Additive (WADD) model. Random Guessing predicts a uniform ~0.50 accuracy on this metric, regardless of the stimulus magnitudes. In contrast, High-Temperature WADD leverages the massive rating differences to overcome its high cognitive noise, predicting a significantly higher proportion of WADD-consistent choices (>0.60). The large mean gap and low per-subject variance will robustly separate the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([1.0, 0.9, 0.6, 0.5])
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    score_a = A @ validities
    score_b = B @ validities
    
    # 0 if A > B, 1 if B > A
    wadd_choice = (score_b > score_a).astype(int)
    
    match = (data['response'] == wadd_choice)
    
    return float(match.mean())

### RESULTS
- Predicted under pi_5_1 (simulated): 0.6392 (var=0.0065)
- Predicted under pi_4 (simulated): 0.5019 (var=0.0025)
- Observed on real data: 0.5200 (var=0.0019)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**rating_max:** 100000

**Trial pairs (n=6):**
  trial 1: A=[100000, 0, 0, 0]  B=[0, 100000, 100000, 100000]
  trial 2: A=[0, 100000, 100000, 100000]  B=[100000, 0, 0, 0]
  trial 3: A=[100000, 100000, 0, 0]  B=[0, 0, 100000, 100000]
  trial 4: A=[0, 0, 100000, 100000]  B=[100000, 100000, 0, 0]
  trial 5: A=[100000, 0, 100000, 0]  B=[0, 100000, 0, 100000]
  trial 6: A=[0, 100000, 0, 100000]  B=[100000, 0, 100000, 0]

**Rationale:** To demonstrate that subjects resort to Random Guessing when faced with overwhelming cognitive complexity, we use 4 features with massive, abstract cardinal magnitudes (rating_max = 100000) that conflict across options. The Competing High-Temperature WADD theory assumes subjects still compute weighted sums, meaning the extreme magnitudes will easily overcome the high cognitive noise (low beta) and produce deterministic preferences (e.g., >99% choice probability for the option with the higher weighted sum). However, the Advocated Random Guessing theory predicts that subjects will simply be overwhelmed by the large numbers and complex tradeoffs, abandoning any integration attempts and defaulting to a flat ~50% choice probability across all trials. This creates a stark, qualitative and quantitative dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice matches the prediction of a deterministic Weighted Additive (WADD) model. The Advocated Theory (Random Guessing) predicts a match proportion of approximately 0.5 because subjects ignore the complex cardinal values and choose randomly. The Competing Theory (High-Temperature WADD) predicts a match proportion significantly above 0.5, because even with high cognitive noise, the massive differences in weighted sums (driven by rating_max=100000) will systematically bias choices toward the WADD-favored option.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    v = np.array([0.95, 0.85, 0.75, 0.65])
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    score_a = a_ratings.dot(v)
    score_b = b_ratings.dot(v)
    
    wadd_pred = (score_b > score_a).astype(int)
    
    return float((data['response'] == wadd_pred).mean())

### RESULTS
- Predicted under pi_5_1 (simulated): 0.8569 (var=0.0059)
- Predicted under pi_4 (simulated): 0.5017 (var=0.0025)
- Observed on real data: 0.5125 (var=0.0021)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[5, 5, 5, 5]  B=[4, 4, 4, 10]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 10, 0, 0]  B=[1, 0, 1, 1]
  A=[5, 5, 6, 6]  B=[6, 6, 0, 0]
  A=[2, 2, 2, 2]  B=[0, 0, 10, 10]
  A=[4, 4, 4, 4]  B=[3, 3, 5, 5]
  A=[0, 10, 10, 10]  B=[10, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    relevant_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        
        # Trial 1: A=[1, 6, 6, 6], B=[10, 5, 5, 5]
        if a[0] == 1 and a[1] == 6 and a[2] == 6:
            tallying_consistent += (1 if r == 0 else 0)
            relevant_trials += 1
            
        # Trial 3: A=[10, 0, 0, 0], B=[0, 1, 1, 1]
        elif a[0] == 10 and a[1] == 0 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
        # Trial 4: A=[0, 10, 0, 0], B=[1, 0, 1, 1]
        elif a[0] == 0 and a[1] == 10 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
    return float(tallying_consistent / relevant_trials) if relevant_trials > 0 else 0.5
```

**Observed (real) value:** 0.5056 (var=0.0050)
**Predicted under pi_5_1:** 0.5261 (var=0.0072)
**Predicted under pi_4:** 0.5250 (var=0.0050)

### Experiment 4
**Design**
  A=[2, 6, 6]  B=[10, 5, 5]
  A=[8, 4, 3]  B=[2, 5, 4]
  A=[10, 0, 5]  B=[0, 10, 5]
  A=[0, 10, 5]  B=[10, 0, 5]
  A=[5, 5, 5]  B=[1, 6, 6]
  A=[8, 2, 2]  B=[10, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    validities = np.array([0.9, 0.6, 0.5])
    
    score_a = a_ratings @ validities
    score_b = b_ratings @ validities
    
    wadd_pred = (score_b > score_a).astype(int)
    
    responses = data['response'].values
    return float(np.mean(responses == wadd_pred))

```

**Observed (real) value:** 0.4963 (var=0.0025)
**Predicted under pi_5_1:** 0.5102 (var=0.0034)
**Predicted under pi_4:** 0.5065 (var=0.0019)

### Experiment 5
**Design**
  A=[7, 2, 2]  B=[5, 9, 9]
  A=[4, 8, 8]  B=[6, 1, 1]
  A=[8, 3, 3]  B=[7, 9, 8]
  A=[5, 10, 10]  B=[8, 0, 0]
  A=[9, 1, 1]  B=[7, 8, 8]
  A=[3, 9, 9]  B=[6, 2, 2]
  A=[6, 5, 5]  B=[5, 10, 10]
  A=[2, 7, 7]  B=[4, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue1_a = np.array([x[0] for x in data['option_a_ratings']])
    cue1_b = np.array([x[0] for x in data['option_b_ratings']])
    
    a_better = (cue1_a > cue1_b)
    b_better = (cue1_b > cue1_a)
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    aligned = (a_better & chose_a) | (b_better & chose_b)
    
    return float(aligned.mean())
```

**Observed (real) value:** 0.5138 (var=0.0022)
**Predicted under pi_5_1:** 0.4992 (var=0.0024)
**Predicted under pi_4:** 0.4971 (var=0.0025)

### Experiment 6
**Design**
  A=[6, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[7, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[8, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[9, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[10, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[10, 2, 2, 2]  B=[0, 10, 10, 10]
  A=[3, 10, 10, 10]  B=[8, 1, 1, 1]
  A=[10, 5, 5, 5]  B=[0, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Trial 1: A=[6, 1, 1, 1], B=[5, 10, 10, 10]
    t1_mask = data['option_a_ratings'].apply(lambda x: list(x) == [6, 1, 1, 1])
    # Trial 7: A=[3, 10, 10, 10], B=[8, 1, 1, 1]
    t7_mask = data['option_a_ratings'].apply(lambda x: list(x) == [3, 10, 10, 10])
    
    t1_resp = data[t1_mask]['response'].mean()
    t7_resp = data[t7_mask]['response'].mean()
    
    if pd.isna(t1_resp):
        t1_resp = 0.5
    if pd.isna(t7_resp):
        t7_resp = 0.5
        
    return float(t1_resp - t7_resp)
```

**Observed (real) value:** 0.0033 (var=0.0308)
**Predicted under pi_5_1:** 0.0183 (var=0.0412)
**Predicted under pi_4:** -0.0050 (var=0.0312)

### Experiment 7
**Design**
  A=[6, 2, 2]  B=[5, 9, 9]
  A=[5, 9, 9]  B=[6, 2, 2]
  A=[7, 2, 2]  B=[5, 9, 9]
  A=[5, 9, 9]  B=[7, 2, 2]
  A=[8, 2, 2]  B=[5, 9, 9]
  A=[5, 9, 9]  B=[8, 2, 2]
  A=[9, 2, 2]  B=[5, 9, 9]
  A=[5, 9, 9]  B=[9, 2, 2]
  A=[10, 2, 2]  B=[5, 9, 9]
  A=[5, 9, 9]  B=[10, 2, 2]
  A=[5, 6, 2]  B=[5, 5, 9]
  A=[5, 5, 9]  B=[5, 6, 2]
  A=[5, 8, 2]  B=[5, 5, 9]
  A=[5, 5, 9]  B=[5, 8, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_cue0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_cue0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    chose_a = (data['response'] == 0)
    
    mask_a = a_cue0 > b_cue0
    mask_b = b_cue0 > a_cue0
    
    sum_a = chose_a[mask_a].sum()
    sum_b = chose_a[mask_b].sum()
    
    return float(sum_a - sum_b)
```

**Observed (real) value:** -5.0000 (var=20.9600)
**Predicted under pi_5_1:** 22.0000 (var=14.0864)
**Predicted under pi_4:** 9.0000 (var=18.9076)

### Experiment 8
**Design**
  A=[100, 0, 0]  B=[20, 100, 100]
  A=[0, 100, 100]  B=[90, 0, 0]
  A=[80, 10, 10]  B=[10, 90, 90]
  A=[10, 80, 80]  B=[80, 10, 10]
  A=[90, 20, 20]  B=[20, 80, 80]
  A=[20, 90, 90]  B=[90, 20, 20]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_cues = np.array(data['option_a_ratings'].tolist())
    b_cues = np.array(data['option_b_ratings'].tolist())
    
    a_favored = a_cues[:, 0] > b_cues[:, 0]
    b_favored = a_cues[:, 0] < b_cues[:, 0]
    
    chose_a = (data['response'] == 0).values
    
    rate_a_favored = np.mean(chose_a[a_favored]) if np.any(a_favored) else 0.0
    rate_b_favored = np.mean(chose_a[b_favored]) if np.any(b_favored) else 0.0
    
    return float(rate_a_favored - rate_b_favored)
```

**Observed (real) value:** -0.0058 (var=0.0035)
**Predicted under pi_5_1:** -0.0087 (var=0.0083)
**Predicted under pi_4:** -0.0013 (var=0.0102)

### Experiment 9
**Design**
  A=[100, 100, 100, 100]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[100, 100, 100, 100]
  A=[100, 100, 0, 0]  B=[0, 0, 100, 100]
  A=[0, 0, 100, 100]  B=[100, 100, 0, 0]
  A=[100, 0, 0, 0]  B=[0, 100, 100, 100]
  A=[0, 100, 100, 100]  B=[100, 0, 0, 0]
  A=[50, 50, 50, 50]  B=[50, 50, 50, 50]
  A=[100, 50, 0, 0]  B=[0, 50, 100, 100]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sums = data['option_a_ratings'].apply(sum)
    
    t1_responses = data[a_sums == 400]['response']
    t2_responses = data[a_sums == 0]['response']
    
    p_a_t1 = 1.0 - t1_responses.mean() if len(t1_responses) > 0 else 0.5
    p_a_t2 = 1.0 - t2_responses.mean() if len(t2_responses) > 0 else 0.5
    
    return float(p_a_t1 - p_a_t2)

```

**Observed (real) value:** -0.0267 (var=0.0310)
**Predicted under pi_5_1:** 0.0017 (var=0.0454)
**Predicted under pi_4:** 0.0083 (var=0.0390)

### Experiment 10
**Design**
  A=[100, 100, 0, 0]  B=[0, 0, 100, 100]
  A=[0, 0, 100, 100]  B=[100, 100, 0, 0]
  A=[100, 0, 100, 0]  B=[0, 100, 0, 100]
  A=[0, 100, 0, 100]  B=[100, 0, 100, 0]
  A=[100, 50, 50, 0]  B=[0, 50, 50, 100]
  A=[0, 50, 50, 100]  B=[100, 50, 50, 0]
  A=[50, 50, 50, 50]  B=[50, 50, 50, 50]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([1.0, 0.9, 0.6, 0.5])
    
    def get_score_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.dot(b - a, validities)
        
    score_diffs = data.apply(get_score_diff, axis=1)
    
    # Calculate the total accumulated weighted difference across all trials.
    return float(np.sum((data['response'] - 0.5) * score_diffs))
```

**Observed (real) value:** 1600.0000 (var=42184.0000)
**Predicted under pi_5_1:** 880.0000 (var=47258.2400)
**Predicted under pi_4:** -220.0000 (var=44712.6400)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both new experiments (1 and 2), which employ massive feature magnitudes (up to 100,000) to distinguish the theories, the observed choice probabilities remain remarkably close to 50% (0.5200 and 0.5125). Theory 1 (High-Temperature WADD) completely fails here: because it computes a weighted sum, the extreme magnitudes overpower the low beta parameter, leading to predictions of 63% and 85% alignment, which are entirely absent in the real data. Theory 2 (Random Guessing) correctly predicts the ~50% behavior, as it assumes subjects are overwhelmed and guess. Looking at the historical experiments, observed metrics consistently hover near random guessing or zero-difference baselines. Theory 1's reliance on linear integration of cardinal values makes it highly vulnerable to scaling effects that real subjects simply do not exhibit.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace the High-Temperature WADD theory with a 'Magnitude-Insensitive Tallying' or 'Sign-based Heuristic' theory. Instead of integrating the cardinal values of the features (which breaks down when magnitudes are multiplied by 10,000), subjects might simply compare features ordinally: tallying how many features favor Option A vs Option B, completely ignoring the size of the difference. When the tally is tied or close, they guess randomly. This theory would explain why massive rating scales do not produce deterministic choices, keeping predictions close to 50% in complex or balanced designs, while still allowing for slight deviations from pure randomness when one option clearly dominates on the number of winning features."
}
```

## Usage

```json
{
  "prompt_token_count": 7856,
  "candidates_token_count": 370,
  "total_token_count": 9169
}
```
