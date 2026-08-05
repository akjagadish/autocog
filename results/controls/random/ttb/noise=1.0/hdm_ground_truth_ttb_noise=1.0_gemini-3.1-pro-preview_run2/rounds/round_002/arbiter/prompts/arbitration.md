# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take The Best (TTB) Heuristic: Decision makers process information sequentially, searching through features in descending order of their subjective validity. They stop at the first feature that discriminates between the options (i.e., the absolute difference in feature values exceeds a certain threshold) and choose the option favored by that single cue. All other lower-validity features are ignored. If no cue discriminates, they guess. This represents a classic fast-and-frugal one-reason decision-making benchmark.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        diff = a[idx] - b[idx]
        if abs(diff) > threshold:
            if diff > 0:
                p_core = np.array([1.0, 0.0])
            else:
                p_core = np.array([0.0, 1.0])
            break
            
    # Incorporate lapse rate (epsilon)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=3):** [0.95, 0.75, 0.55]

**rating_max:** 10

**Trial pairs (n=14):**
  trial 1: A=[6, 2, 2]  B=[5, 9, 9]
  trial 2: A=[5, 9, 9]  B=[6, 2, 2]
  trial 3: A=[7, 2, 2]  B=[5, 9, 9]
  trial 4: A=[5, 9, 9]  B=[7, 2, 2]
  trial 5: A=[8, 2, 2]  B=[5, 9, 9]
  trial 6: A=[5, 9, 9]  B=[8, 2, 2]
  trial 7: A=[9, 2, 2]  B=[5, 9, 9]
  trial 8: A=[5, 9, 9]  B=[9, 2, 2]
  trial 9: A=[10, 2, 2]  B=[5, 9, 9]
  trial 10: A=[5, 9, 9]  B=[10, 2, 2]
  trial 11: A=[5, 6, 2]  B=[5, 5, 9]
  trial 12: A=[5, 5, 9]  B=[5, 6, 2]
  trial 13: A=[5, 8, 2]  B=[5, 5, 9]
  trial 14: A=[5, 5, 9]  B=[5, 8, 2]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Random Guessing, we exploit TTB's deterministic, lexicographic search and its sensitivity to the cue difference threshold. We use 3 cardinal features and systematically vary the difference between Option A and Option B on the most valid cue (from 1 to 5 points), while the less valid cues strongly favor the opposite option. For TTB, if the difference on the top cue exceeds its internal threshold, it will decisively choose the option favored by that cue (nearly 100% probability, minus lapse rate). If the difference falls below the threshold, it will move to the second cue and decisively choose the opposite option. This creates a sharp step-function in choice probabilities across trials. The Random Guessing model, in stark contrast, predicts a flat ~50% choice probability across all trials regardless of feature differences.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



### METRIC
Rationale:
Take The Best (TTB) makes deterministic choices based on the most valid cue when the difference between options exceeds an internal threshold. Although high thresholds and lapse rates reduce TTB to random guessing for many subjects, a critical subset will reliably choose the option favored by the top cue. By computing the unnormalized difference in the total number of times Option A is chosen when favored by Cue 0 versus when Option B is favored, we capture this directional signal. Because this metric computes a sum rather than a mean, it aggregates the signal across all trials in the pooled dataset, producing a massive point estimate for TTB. Meanwhile, the Random Guessing baseline expects a value of exactly zero regardless of pooling, ensuring a highly significant statistical contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    a_cue0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_cue0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    chose_a = (data['response'] == 0)
    
    mask_a = a_cue0 > b_cue0
    mask_b = b_cue0 > a_cue0
    
    sum_a = chose_a[mask_a].sum()
    sum_b = chose_a[mask_b].sum()
    
    return float(sum_a - sum_b)

### RESULTS
- Predicted under pi_3 (simulated): -41.0000 (var=16.1876)
- Predicted under pi_4 (simulated): 9.0000 (var=18.9076)
- Observed on real data: -5.0000 (var=20.9600)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=3):** [0.95, 0.75, 0.55]

**rating_max:** 100

**Trial pairs (n=6):**
  trial 1: A=[100, 0, 0]  B=[20, 100, 100]
  trial 2: A=[0, 100, 100]  B=[90, 0, 0]
  trial 3: A=[80, 10, 10]  B=[10, 90, 90]
  trial 4: A=[10, 80, 80]  B=[80, 10, 10]
  trial 5: A=[90, 20, 20]  B=[20, 80, 80]
  trial 6: A=[20, 90, 90]  B=[90, 20, 20]

**Rationale:** To quantitatively dissociate Random Guessing from Take The Best (TTB), we exploit TTB's lexicographic stopping rule and its threshold parameter. TTB searches through features by validity and stops at the first feature where the absolute difference exceeds its threshold (which is bounded between 0 and 50). By using a large rating scale (rating_max=100) and ensuring the difference on the highest-validity feature is always greater than 50, we force TTB to deterministically choose the option favored by the most valid cue on every trial, completely ignoring all other features. The Random Guessing model, on the other hand, predicts a flat ~50% choice probability regardless of the feature values. This creates a stark, qualitative and quantitative dissociation: TTB predicts strong alternation of choices (A, B, A, B...) tracking the highest-validity cue, while Random Guessing predicts uniform indifference.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the difference in the probability of choosing Option A when Option A is favored by the highest-validity cue versus when Option B is favored by it. Random Guessing predicts this difference to be near zero, as choices are independent of cue values. Take The Best strongly relies on the first cue, so it will choose A significantly more often when A is favored by it, leading to a substantial positive difference.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.4996 (var=0.0748)
- Predicted under pi_4 (simulated): -0.0013 (var=0.0102)
- Observed on real data: -0.0058 (var=0.0035)

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
**Predicted under pi_3:** 0.4572 (var=0.0238)
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
**Predicted under pi_3:** 0.5246 (var=0.0059)
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
**Predicted under pi_3:** 0.4906 (var=0.0131)
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
**Predicted under pi_3:** 0.0450 (var=0.0420)
**Predicted under pi_4:** -0.0050 (var=0.0312)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed data consistently shows near-random choice behavior, with metric values hovering around 0.5 for proportions or 0 for differences. Theory 1 (Take The Best) predicts strong, deterministic deviations from 0.5 based on lexicographic cue processing, which drastically fail to match the observed real data. Theory 2 (Random Guessing) captures the data extremely well because the subjects' aggregate choices are indistinguishable from random noise or uniform guessing. Since TTB fundamentally mischaracterizes this near-random behavior, it needs to be completely replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Noisy Evidence Accumulation' or 'High-Temperature Weighted Additive (WADD)' theory. Unlike TTB's deterministic, single-reason stopping rule, this new theory should assume that subjects attempt to integrate all available features (e.g., computing a weighted sum of feature differences), but their decision process is subject to extremely high cognitive noise (a very high temperature parameter in a softmax choice rule). This would allow the model to predict the near-random behavior observed across all experiments while still retaining a formal mechanism of multi-attribute integration, distinguishing it from pure random guessing by allowing for slight sensitivities to large aggregate differences."
}
```

## Usage

```json
{
  "prompt_token_count": 5712,
  "candidates_token_count": 286,
  "total_token_count": 6987
}
```
