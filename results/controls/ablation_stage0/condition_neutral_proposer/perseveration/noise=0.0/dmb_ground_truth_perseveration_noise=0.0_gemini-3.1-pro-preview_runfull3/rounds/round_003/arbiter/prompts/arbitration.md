# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_5
Sequential Dependency / Alternation: In the absence of correctness feedback and when facing complex multi-cue choices, subjects often disengage from evaluating the actual features of the options. Instead of relying on a static spatial position bias, subjects exhibit sequential dependencies in their choices, such as a tendency to repeat their previous choice (inertia) or to systematically alternate between Option A and Option B. This history-dependent strategy completely ignores the stimulus validities and values, making decisions solely based on the temporal sequence of past actions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    p_repeat = float(parameters["p_repeat"])
    
    if not history["response"]:
        # First trial: no history, predict uniformly
        return np.array([0.5, 0.5])
    
    last_resp = history["response"][-1]
    
    if last_resp == 0:
        # Last response was Option A
        prob_a = p_repeat
        prob_b = 1.0 - p_repeat
    else:
        # Last response was Option B
        prob_a = 1.0 - p_repeat
        prob_b = p_repeat
        
    return np.array([prob_a, prob_b])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=3):** [0.9, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0]  B=[0, 1, 1]
  trial 2: A=[0, 1, 1]  B=[1, 0, 0]
  trial 3: A=[1, 1, 0]  B=[0, 0, 1]
  trial 4: A=[0, 0, 1]  B=[1, 1, 0]
  trial 5: A=[1, 0, 1]  B=[0, 1, 0]
  trial 6: A=[0, 1, 0]  B=[1, 0, 1]
  trial 7: A=[1, 1, 1]  B=[0, 0, 0]
  trial 8: A=[0, 0, 0]  B=[1, 1, 1]

**Rationale:** Both Theory 1 (Position Bias) and Theory 2 (Sequential Dependency) assume that subjects ignore the stimulus features entirely. Theory 1 predicts a constant probability of choosing Option A across all trials, independent of past choices. Theory 2 predicts that the choice on the current trial depends strictly on the choice made on the previous trial (e.g., a tendency to repeat or alternate). Because both theories ignore the stimulus, the specific feature values do not drive the dissociation. Instead, the models are distinguished by analyzing the sequential dependencies in the response data. We provide a standard set of multi-attribute choice trials to serve as a plausible task environment, ensuring subjects have a reason to make choices while we measure their disengagement strategies.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric computes the squared difference between two conditional probabilities: P(Response_t = 0 | Response_{t-1} = 0) and P(Response_t = 0 | Response_{t-1} = 1) for each subject. Under Theory 1 (Position Bias), choices are independent across trials, so these conditional probabilities both converge to the subject's baseline bias for Option A, making their squared difference approximately 0 (subject only to small sampling noise). Under Theory 2 (Sequential Dependency), choices are strongly dependent on the previous trial, so P(0|0) = p_repeat and P(0|1) = 1 - p_repeat. Their squared difference is (2*p_repeat - 1)^2, which averages to 1/3 across a uniformly distributed p_repeat parameter. This creates a massive, highly significant mean gap between the theories.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.0067 (var=0.0001)
- Predicted under pi_5 (simulated): 0.3092 (var=0.0846)
- Observed on real data: 0.0000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Rationale:** Both Theory 1 (Sequential Dependency) and Theory 2 (Position Bias) assume that subjects completely ignore the stimulus features. Theory 1 predicts that the probability of choosing Option A depends on the previous choice (e.g., repeating or alternating), whereas Theory 2 predicts a constant probability of choosing Option A across all trials, regardless of history. Since the stimulus features themselves do not drive the dissociation, we provide a standard, varied set of multi-attribute choice trials to serve as a plausible task environment. The actual dissociation will come from analyzing the sequential dependence of the subjects' responses.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the absolute difference in the probability of choosing Option A given that the previous choice was Option A versus Option B. Under the Sequential Dependency theory, choices depend heavily on the previous choice (either repeating or alternating), leading to a large absolute difference in these conditional probabilities. Under the Position Bias theory, choices are independent of the previous choice, so the conditional probabilities should be roughly equal (modulo sampling noise), resulting in a metric value close to zero.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.0890 (var=0.0033)
- Predicted under pi_5 (simulated): 0.5241 (var=0.0691)
- Observed on real data: 0.0000 (var=0.0000)

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
**Predicted under pi_5:** 0.4921 (var=0.0019)

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
**Predicted under pi_5:** 0.5050 (var=0.0019)

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
**Predicted under pi_5:** 0.5083 (var=0.0148)

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
**Predicted under pi_5:** 0.4885 (var=0.0022)

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
**Predicted under pi_5:** 0.0004 (var=0.0001)

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
**Predicted under pi_5:** -0.0104 (var=0.0072)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data strongly rejects Theory 2 (Sequential Dependency). Metrics measuring sequential patterns (Exp 1 and Exp 2) show an observed value of exactly 0.0000, meaning choices are completely independent of previous trials. Furthermore, stimulus-based metrics (Exp 3, 4, 6, 8) yield observed values of exactly 0.5 or 0.0 with 0 variance, indicating that subjects are not using the features either. Theory 1 (Position Bias) captures this well, as a spatial bias (or a specific case of it, random guessing) produces no sequential dependency and no stimulus correlation. Theory 2 predicts large sequential dependencies that simply do not exist in the data. Thus, Theory 2 is degenerate and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Strict Random Guessing' theory. While Theory 1 allows for varying spatial position biases across subjects, the zero variance in many observed metrics suggests subjects might be uniformly guessing with exactly p=0.5. The new theory should posit that in the absence of feedback, subjects completely disengage and simply flip a coin on every trial (predicting exactly [0.5, 0.5] with no free parameters), ignoring stimulus, history, and spatial position alike."
}
```

## Usage

```json
{
  "prompt_token_count": 6744,
  "candidates_token_count": 301,
  "total_token_count": 8262
}
```
