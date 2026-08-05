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
Position Bias / Task Disengagement: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects often disengage from effortful feature integration or search. Instead, they fall back on stimulus-independent response strategies, such as a strong position bias (e.g., a tendency to select the first option, Option A) or random guessing. Decisions are driven entirely by these spatial/temporal biases, and the actual feature validities and values are ignored.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the stimulus and history entirely.
    # It predicts based solely on a spatial position bias for Option A.
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


## THEORY 2 — pi_6
Strict Random Guessing: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects completely disengage from the task. They ignore the stimulus features, spatial positions, and their own choice history, instead falling back on a pure uniform random guessing strategy (flipping a coin) on every single trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # Predict exactly 0.5 for both options regardless of stimulus or history
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 2: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 7: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  trial 9: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** Both Theory 1 (Position Bias) and Theory 2 (Strict Random Guessing) assume that subjects ignore the stimulus features entirely. Theory 1 allows for an asymmetric preference for one spatial position over the other (e.g., choosing Option A 80% of the time), whereas Theory 2 strictly predicts a 50/50 split on every trial. To ensure that any observed position bias is purely a spatial/response bias and not an artifact of the stimulus, we provide a perfectly balanced set of trials. For every trial where Option A has a given set of features and Option B has another, there is a mirror-image trial where the features are swapped. This guarantees that the objective quality of Option A and Option B is identical over the course of the experiment.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
To distinguish between Position Bias (which allows for any probability of choosing Option A) and Strict Random Guessing (which enforces exactly a 50% probability), we calculate each subject's absolute deviation from a 0.5 choice proportion for Option A. Under Strict Random Guessing, this value will be close to 0 (driven only by binomial sampling noise). Under Position Bias, subjects will exhibit varying degrees of bias, resulting in a substantially higher average absolute deviation.

Source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the proportion of times each subject chose Option A (response == 0)
    subject_means = data.groupby('subject_id')['response'].apply(lambda x: (x == 0).mean())
    # Calculate the absolute deviation from 0.5 for each subject
    abs_deviations = (subject_means - 0.5).abs()
    # Return the mean of these absolute deviations across the population
    return float(abs_deviations.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.2556 (var=0.0198)
- Predicted under pi_6 (simulated): 0.0400 (var=0.0009)
- Observed on real data: 0.5000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=3):** [0.9, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0]  B=[1, 0, 0]
  trial 2: A=[0, 1, 0]  B=[0, 1, 0]
  trial 3: A=[0, 0, 1]  B=[0, 0, 1]
  trial 4: A=[1, 1, 0]  B=[1, 1, 0]
  trial 5: A=[1, 0, 1]  B=[1, 0, 1]
  trial 6: A=[0, 1, 1]  B=[0, 1, 1]
  trial 7: A=[1, 1, 1]  B=[1, 1, 1]
  trial 8: A=[0, 0, 0]  B=[0, 0, 0]

**Rationale:** To definitively distinguish Strict Random Guessing from Position Bias without confounding spatial preference with stimulus features, we present a series of 'Buridan's ass' trials where Option A and Option B are objectively identical in all features. Because there is no stimulus-based reason to prefer one option over the other on any trial, any consistent overall preference for Option A must be driven by a spatial position bias (Theory 2). Theory 1 predicts a strict 50/50 split across these trials, whereas Theory 2 can accommodate an asymmetric preference for one side when facing identical choices.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Strict Random Guessing predicts that every subject will choose Option A and Option B with a 50/50 probability, resulting in a proportion of B choices very close to 0.5 for all subjects (mean absolute deviation from 0.5 will be close to 0). Position Bias, however, predicts that subjects will have varying idiosyncratic spatial biases (some strongly preferring A, others strongly preferring B). This will lead to individual choice proportions that systematically deviate from 0.5, resulting in a significantly higher mean absolute deviation.

Source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the proportion of B choices (response == 1) for each subject
    subject_means = data.groupby('subject_id')['response'].mean()
    # Return the mean absolute deviation from 0.5 across all subjects
    return float((subject_means - 0.5).abs().mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.2471 (var=0.0244)
- Predicted under pi_6 (simulated): 0.0481 (var=0.0011)
- Observed on real data: 0.5000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    preds = []
    for i in range(len(data)):
        a_vec = data['option_a_ratings'].iloc[i]
        b_vec = data['option_b_ratings'].iloc[i]
        pred = -1
        for j in range(len(a_vec)):
            if a_vec[j] > b_vec[j]:
                pred = 0
                break
            elif b_vec[j] > a_vec[j]:
                pred = 1
                break
        preds.append(pred)
        
    return float(np.mean(data['response'].values == np.array(preds)))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.4969 (var=0.0020)
**Predicted under pi_6:** 0.5062 (var=0.0024)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            pred = 0
        elif b_wins > a_wins:
            pred = 1
        else:
            continue
            
        if row['response'] == pred:
            match_count += 1
        total += 1
        
    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5031 (var=0.0019)
**Predicted under pi_6:** 0.5046 (var=0.0027)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_ttb_aligned(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_pred = 0 if a[i] > b[i] else 1
                return row['response'] == ttb_pred
        return False
        
    return float(data.apply(is_ttb_aligned, axis=1).mean())
```

**Observed (real) value:** 0.4250 (var=0.1350)
**Predicted under pi_4:** 0.5221 (var=0.0492)
**Predicted under pi_6:** 0.5050 (var=0.0029)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                break
    return float(matches / len(data))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.4985 (var=0.0017)
**Predicted under pi_6:** 0.5121 (var=0.0035)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    df = data.copy()
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join(map(str, x))) + "_" + df['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    trial_means = df.groupby('trial_type')['response'].mean()
    return float(np.var(trial_means))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_4:** 0.0000 (var=0.0001)
**Predicted under pi_6:** 0.0004 (var=0.0001)

### Experiment 8
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    val = np.array([0.9, 0.8, 0.7, 0.6])
    score_a = a_ratings.dot(val)
    score_b = b_ratings.dot(val)
    diff = score_a - score_b
    
    choose_a = 1.0 - data['response'].values
    
    mask_a = diff > 0.01
    mask_b = diff < -0.01
    
    p_a = np.mean(choose_a[mask_a]) if np.sum(mask_a) > 0 else 0.0
    p_b = np.mean(choose_a[mask_b]) if np.sum(mask_b) > 0 else 0.0
    
    return float(p_a - p_b)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_4:** 0.0088 (var=0.0068)
**Predicted under pi_6:** 0.0096 (var=0.0108)

### Experiment 9
**Design**
  A=[1, 0, 0]  B=[0, 1, 1]
  A=[0, 1, 1]  B=[1, 0, 0]
  A=[1, 1, 0]  B=[0, 0, 1]
  A=[0, 0, 1]  B=[1, 1, 0]
  A=[1, 0, 1]  B=[0, 1, 0]
  A=[0, 1, 0]  B=[1, 0, 1]
  A=[1, 1, 1]  B=[0, 0, 0]
  A=[0, 0, 0]  B=[1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def subject_metric(df):
        resp = df['response'].values
        if len(resp) < 2:
            return 0.0
        
        idx_0 = np.where(resp[:-1] == 0)[0]
        idx_1 = np.where(resp[:-1] == 1)[0]
        
        if len(idx_0) == 0 or len(idx_1) == 0:
            # If a subject only ever gives one response, there is no variance
            # in transitions. For Theory 1 (Position Bias), this corresponds to 
            # extreme bias and a true difference of 0. 
            return 0.0
            
        p0_given_0 = np.mean(resp[idx_0 + 1] == 0)
        p0_given_1 = np.mean(resp[idx_1 + 1] == 0)
        
        return float((p0_given_0 - p0_given_1)**2)

    return float(data.groupby('subject_id').apply(subject_metric).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_4:** 0.0067 (var=0.0001)
**Predicted under pi_6:** 0.0103 (var=0.0002)

### Experiment 10
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def subj_metric(df):
        responses = df['response'].values
        if len(responses) < 2:
            return 0.0
        r_t = responses[1:]
        r_t_minus_1 = responses[:-1]
        
        n00 = np.sum((r_t_minus_1 == 0) & (r_t == 0))
        n01 = np.sum((r_t_minus_1 == 0) & (r_t == 1))
        n10 = np.sum((r_t_minus_1 == 1) & (r_t == 0))
        n11 = np.sum((r_t_minus_1 == 1) & (r_t == 1))
        
        p_0_given_0 = n00 / (n00 + n01) if (n00 + n01) > 0 else np.nan
        p_0_given_1 = n10 / (n10 + n11) if (n10 + n11) > 0 else np.nan
        
        if np.isnan(p_0_given_0) or np.isnan(p_0_given_1):
            return 0.0
        
        return abs(p_0_given_0 - p_0_given_1)

    return float(data.groupby('subject_id').apply(subj_metric).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_4:** 0.0890 (var=0.0033)
**Predicted under pi_6:** 0.0780 (var=0.0037)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metrics consistently point to an extreme behavioral pattern: subjects are completely ignoring the stimuli and choosing the exact same option (either 100% Option A or 100% Option B) for the entire duration of the experiment. In Experiments 1 and 2, the mean absolute deviation of choice proportions from 0.5 is exactly 0.5000 with 0 variance, meaning every single subject had a choice proportion of either 1.0 or 0.0. Furthermore, transition-based metrics in Experiments 9 and 10 evaluate to 0.0000, which happens when subjects never switch their responses. Theory 1 (Position Bias) captures the essence of this spatial preference but its uniform parameter distribution predicts intermediate biases (e.g., 75% A), leading to an expected deviation of ~0.25. Theory 2 (Strict Random Guessing) predicts exactly 50/50 choices, completely failing to capture the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with an 'Extreme Spatial Strategy' or 'Extreme Position Bias' theory. The prose should explicitly state that subjects completely disengage from the task and adopt a degenerate strategy of pressing exactly one button (either always A or always B) for the entirety of the experiment. The model should sample a parameter (e.g., preferred_side) that is strictly 0 or 1, and the predict function should return [1.0, 0.0] or [0.0, 1.0] accordingly."
}
```

## Usage

```json
{
  "prompt_token_count": 7395,
  "candidates_token_count": 360,
  "total_token_count": 8940
}
```
