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
Random Guessing: Due to task complexity, lack of trial-by-trial correctness feedback, or low motivation, subjects do not systematically evaluate the options using the provided validities or features. Instead, they make uniformly random choices on every trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_6
Spatial Bias / Option Preference: Subjects completely ignore the complex feature information provided on each trial. Instead, their choices are driven by an idiosyncratic, subject-level baseline preference for Option A versus Option B (e.g., a left/right spatial bias or an order effect). Each subject has a fixed probability of choosing Option A on any given trial, which varies between subjects but remains constant across trials for a given subject. This explains why choices are invariant to feature differences while accounting for individual differences in baseline choice rates.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    p_a = float(parameters["bias_A"])
    return np.array([p_a, 1.0 - p_a])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design aims to distinguish Random Guessing from a Spatial Bias / Option Preference model. Both models predict that feature information is ignored, but they differ in the baseline choice rate: Random Guessing predicts exactly 50% choices for Option A across all trials, whereas Spatial Bias predicts a consistent idiosyncratic preference (e.g., 70% for A). To test this, we construct a perfectly symmetric set of trials where Option A dominates, Option B dominates, or they have mixed advantages. Because the design is perfectly balanced across features, any overall deviation from a 50% choice rate for Option A would strongly support Spatial Bias over pure Random Guessing. Conversely, an exact 50% choice rate across all trials, regardless of whether Option A is overwhelmingly superior or inferior, would support Random Guessing.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Under the Random Guessing theory, each subject chooses Option A with exactly 50% probability on every trial, so their overall choice proportion will be very close to 0.5 (with only small binomial variance). Under the Spatial Bias theory, subjects have an idiosyncratic baseline preference between 0.3 and 0.7, meaning their choice proportions will systematically deviate from 0.5. By calculating the squared deviation of each subject's choice proportion from 0.5 and averaging this across subjects, we obtain a metric that is near zero for Random Guessing but substantially higher for Spatial Bias. Computing this at the subject level before averaging ensures the metric correctly captures the between-subject variance in bias.

Source:
def metric(data: pd.DataFrame) -> float:
    subject_means = data.groupby('subject_id')['response'].mean()
    sq_devs = (subject_means - 0.5) ** 2
    return float(sq_devs.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.0027 (var=0.0000)
- Predicted under pi_6 (simulated): 0.0180 (var=0.0002)
- Observed on real data: 0.0018 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=6):** [0.9, 0.8, 0.75, 0.7, 0.6, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 3: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 8: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Rationale:** To dissociate Spatial Bias from Random Guessing, we present a series of trials where the objective quality of Option A and Option B varies significantly. Random Guessing predicts a choice proportion of exactly 0.5 for Option A across all trials and all subjects. Spatial Bias predicts that subjects will have an idiosyncratic preference for Option A (or B) that remains constant across all trials, regardless of whether A is objectively superior or inferior. By examining the variance in overall choice proportions across subjects, we can distinguish the two: Random Guessing predicts a tight distribution around 0.5, while Spatial Bias predicts a wider distribution reflecting individual differences in baseline preference. The trial set includes pairs where A is clearly superior, B is clearly superior, and where they are mixed.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Spatial Bias predicts that subjects have an idiosyncratic preference for one option over the other, meaning their overall choice proportions will deviate significantly from 0.5 (reflecting their individual bias). Random Guessing predicts choice proportions will be tightly clustered around 0.5, as choices are essentially coin flips. By calculating the squared deviation of each subject's choice proportion from 0.5, we can clearly distinguish the wider distribution of Spatial Bias from the narrow distribution of Random Guessing.

Source:
def metric(data: pd.DataFrame) -> float:
    subject_means = data.groupby('subject_id')['response'].mean()
    return float(((subject_means - 0.5) ** 2).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.0025 (var=0.0000)
- Predicted under pi_6 (simulated): 0.0195 (var=0.0003)
- Observed on real data: 0.0047 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].apply(tuple)
    b_ratings = data['option_b_ratings'].apply(tuple)
    
    mask1 = (a_ratings == (0, 0, 0, 1, 1, 1)) & (b_ratings == (1, 1, 0, 0, 0, 0))
    mask2 = (a_ratings == (1, 1, 0, 0, 0, 0)) & (b_ratings == (0, 0, 0, 1, 1, 1))
    
    resp = []
    if mask1.any():
        resp.extend(data.loc[mask1, 'response'].tolist())
    if mask2.any():
        resp.extend((1 - data.loc[mask2, 'response']).tolist())
        
    if len(resp) == 0:
        return 0.5
    return float(np.mean(resp))
```

**Observed (real) value:** 0.4844 (var=0.0167)
**Predicted under pi_4:** 0.4867 (var=0.0202)
**Predicted under pi_6:** 0.5200 (var=0.0565)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    is_t1 = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    is_t2 = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    wadd_consistent = (is_t1 & (data['response'] == 0)) | (is_t2 & (data['response'] == 1))
    mask = is_t1 | is_t2
    
    if mask.sum() == 0:
        return 0.5
    return float(wadd_consistent.sum() / mask.sum())
```

**Observed (real) value:** 0.4800 (var=0.0103)
**Predicted under pi_4:** 0.5150 (var=0.0123)
**Predicted under pi_6:** 0.5000 (var=0.0078)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

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
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            matches.append(1.0 if ttb_pred == resp else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.4954 (var=0.0018)
**Predicted under pi_4:** 0.5096 (var=0.0020)
**Predicted under pi_6:** 0.4898 (var=0.0029)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(np.sum)
    b_sums = data['option_b_ratings'].apply(np.sum)
    wadd_choice = (b_sums > a_sums).astype(int)
    return float((data['response'] == wadd_choice).mean())
```

**Observed (real) value:** 0.5046 (var=0.0028)
**Predicted under pi_4:** 0.5193 (var=0.0019)
**Predicted under pi_6:** 0.5035 (var=0.0032)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    keys = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + '_' + data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    means = data.groupby(keys)['response'].mean()
    return float(np.var(means))
```

**Observed (real) value:** 0.0015 (var=0.0001)
**Predicted under pi_4:** 0.0003 (var=0.0001)
**Predicted under pi_6:** 0.0005 (var=0.0001)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.8, 0.75, 0.6, 0.5])
    
    matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a_ratings = np.array(row['option_a_ratings'])
        b_ratings = np.array(row['option_b_ratings'])
        
        score_a = np.dot(a_ratings, validities)
        score_b = np.dot(b_ratings, validities)
        
        if abs(score_a - score_b) > 1e-5:
            predicted_response = 0 if score_a > score_b else 1
            if row['response'] == predicted_response:
                matches += 1
            total += 1
            
    return matches / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5040 (var=0.0025)
**Predicted under pi_4:** 0.5040 (var=0.0025)
**Predicted under pi_6:** 0.4943 (var=0.0029)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    responses = data.loc[mask, 'response']
    a_more = sum_a[mask] > sum_b[mask]
    
    chose_max = ((responses == 0) & a_more) | ((responses == 1) & ~a_more)
    return float(chose_max.mean())
```

**Observed (real) value:** 0.5006 (var=0.0047)
**Predicted under pi_4:** 0.4986 (var=0.0024)
**Predicted under pi_6:** 0.4978 (var=0.0036)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff = sum_a - sum_b
    
    chose_a = 1 - data['response']
    
    pos_mean = chose_a[diff > 0].mean()
    neg_mean = chose_a[diff < 0].mean()
    
    if pd.isna(pos_mean) or pd.isna(neg_mean):
        return 0.0
        
    return float(pos_mean - neg_mean)
```

**Observed (real) value:** 0.0112 (var=0.0080)
**Predicted under pi_4:** 0.0206 (var=0.0141)
**Predicted under pi_6:** -0.0106 (var=0.0084)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the data strongly points to subjects largely ignoring the feature information, as accuracy or feature-consistency metrics hover very tightly around 0.50 (e.g., Exp 8, Exp 9, Exp 6). Theory 1 (Random Guessing) predicts these outcomes extremely well, predicting virtually no variance in choice proportions beyond pure binomial noise. However, Experiment 2 shows a mean squared deviation of 0.0047, which is slightly higher than the pure binomial variance predicted by Theory 1 (0.0025), but significantly lower than the massive variance predicted by Theory 2 (0.0195). This suggests that while subjects are mostly guessing, there is a very small, weak idiosyncratic spatial/baseline bias in the population, rather than the strong bias posited by the current parameterization of Theory 2.",
  "verdict": "new_model",
  "target_theory_idx": 2,
  "recommendation": "Preserve the theory description but update the parameter ranges for `bias_A`. The current implementation likely samples `bias_A` from a wide uniform distribution (e.g., [0.0, 1.0] or [0.2, 0.8]), which overestimates the between-subject variance in choice proportions. Constrain the parameter range to a tightly bounded distribution around 0.5 (e.g., Uniform(0.4, 0.6) or a Beta distribution with high concentration around 0.5). This will allow the model to capture the slight overdispersion in subject choice proportions (around 0.0047) without overpredicting it."
}
```

## Usage

```json
{
  "prompt_token_count": 7974,
  "candidates_token_count": 362,
  "total_token_count": 9770
}
```
