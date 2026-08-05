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
When faced with complex multi-attribute choices without trial-by-trial feedback, subjects experience cognitive overload. Instead of systematically integrating cue validities and feature vectors, they abandon structured decision strategies and resort to random guessing. Choice behavior is driven entirely by this stochasticity, with only a potential slight bias toward one spatial position (e.g., Option A or Option B) over the other.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # Under cognitive overload, subjects ignore the state (features) and just guess.
    # The choice probability is determined only by an intrinsic side bias.
    p_b = float(parameters.get('side_bias', 0.5))
    p_a = 1.0 - p_b
    return np.array([p_a, p_b])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## THEORY 2 — pi_5
Tallying under Overload (Equal Weights): Under cognitive overload without trial-by-trial feedback, subjects abandon complex integration of cue validities. Instead, they fall back on a highly simplified Equal Weights heuristic, merely tallying the total number of positive features (1s) for each option. Even with this simplification, the high cognitive demand leads to near-random choice behavior, which is captured by extreme softmax noise and a very high lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    # Equal Weights / Tallying: count the number of positive features (1s) for each option
    a, b = stim[0], stim[1]
    a_score = np.sum(a)
    b_score = np.sum(b)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Apply high lapse rate
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=11):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0]
  trial 11: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Cognitive Overload theory (which predicts completely flat choice probabilities driven only by a spatial side bias) from the Tallying under Overload theory (which predicts that choice probabilities, despite being extremely noisy, will systematically track the difference in the unweighted sum of features). We present trials with maximal variation in the tally difference between Option A and Option B (ranging from +5 to -5). The Tallying theory predicts a monotonic relationship between the tally difference and the probability of choosing A, whereas the Cognitive Overload theory predicts a completely flat choice rate across all trials.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



### METRIC
Rationale:
The Cognitive Overload theory models choice as a random guess with a potential spatial side bias, predicting that subjects will systematically deviate from a 50/50 choice proportion across the experiment. The Tallying under Overload theory, however, predicts symmetric choice probabilities centered exactly at 50% because the trial design is perfectly balanced in terms of tally differences (ranging from +5 to -5). By computing the mean absolute deviation of each subject's overall choice rate from 0.5, this metric captures the presence of the side bias in the Cognitive Overload theory and its absence in the Tallying theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    subject_means = data.groupby('subject_id')['response'].apply(lambda x: np.mean(x == 0))
    return float(np.mean(np.abs(subject_means - 0.5)))

### RESULTS
- Predicted under pi_4 (simulated): 0.1050 (var=0.0035)
- Predicted under pi_5 (simulated): 0.0427 (var=0.0010)
- Observed on real data: 0.0436 (var=0.0007)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=8):** [0.95, 0.85, 0.75, 0.65, 0.6, 0.55, 0.52, 0.5]

**Trial pairs (n=9):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Tallying under Overload theory from the pure Cognitive Overload (random guessing) theory, we must test whether choice probabilities remain completely flat across trials or whether they track the unweighted feature tally differences, even if heavily dampened by extreme noise and lapse rates. We use 8 features to create a wide spectrum of tally differences between Option A and Option B, ranging from +8 to -8. The Tallying under Overload theory predicts a slight but systematic monotonic relationship between the tally difference and the probability of choosing A, whereas the pure Cognitive Overload theory predicts a completely flat choice rate driven only by a constant spatial side bias.

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



### METRIC
Rationale:
The Competing theory (Cognitive Overload) posits that subjects fall back on a random guessing strategy driven by a constant side bias, which varies uniformly between 0.3 and 0.7 across subjects. This results in an expected absolute deviation from 0.5 of roughly 0.10. The Advocated theory (Tallying under Overload) assumes subjects use equal weights to tally features. Since the experimental design is perfectly symmetric (for every trial favoring A, there is a mirror trial favoring B), the true overall choice rate for any subject under the Advocated theory is exactly 0.5. Any deviation from 0.5 is solely due to binomial noise, yielding a much smaller expected absolute deviation (around 0.04). Therefore, the mean absolute deviation of the subject's overall choice rate from 0.5 strongly discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Group by subject and calculate each subject's overall choice rate for Option B
    subject_means = data.groupby('subject_id')['response'].mean()
    # Calculate the absolute deviation from 0.5 (which represents no side bias)
    # and return the mean deviation across subjects.
    return float(subject_means.apply(lambda x: abs(x - 0.5)).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.0962 (var=0.0048)
- Predicted under pi_5 (simulated): 0.0442 (var=0.0009)
- Observed on real data: 0.0333 (var=0.0008)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    def ttb_predict(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif a[i] < b[i]:
                return 1
        return 0.5
        
    ttb_choices = data.apply(ttb_predict, axis=1)
    matches = (data['response'] == ttb_choices)
    return float(matches.mean())
```

**Observed (real) value:** 0.5050 (var=0.0019)
**Predicted under pi_4:** 0.4925 (var=0.0032)
**Predicted under pi_5:** 0.4908 (var=0.0022)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Only consider trials where Tallying has a strict preference
        if a_wins > b_wins:
            tally_choice = 0
            matches.append(int(row['response'] == tally_choice))
        elif b_wins > a_wins:
            tally_choice = 1
            matches.append(int(row['response'] == tally_choice))
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

```

**Observed (real) value:** 0.5107 (var=0.0040)
**Predicted under pi_4:** 0.5063 (var=0.0043)
**Predicted under pi_5:** 0.4903 (var=0.0035)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    
    # TTB prediction: sign of the first non-zero difference
    abs_diff = np.abs(diff)
    first_diff_idx = np.argmax(abs_diff, axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # WADD prediction: based on weighted sum
    a_score = np.dot(a_mat, val)
    b_score = np.dot(b_mat, val)
    wadd_pred = np.where(a_score > b_score, 0, 1)
    
    # Identify conflict trials where TTB and WADD make opposite predictions
    conflict = (ttb_pred != wadd_pred) & (first_diff_val != 0)
    
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    # Calculate the proportion of choices that align with TTB on conflict trials
    matches = (responses[conflict] == ttb_pred[conflict])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5038 (var=0.0039)
**Predicted under pi_4:** 0.5231 (var=0.0205)
**Predicted under pi_5:** 0.4844 (var=0.0034)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    score_A = A.dot(val)
    score_B = B.dot(val)
    
    wadd_choice = (score_B > score_A).astype(int)
    
    return float(np.mean(data['response'].values == wadd_choice))
```

**Observed (real) value:** 0.5018 (var=0.0029)
**Predicted under pi_4:** 0.4891 (var=0.0030)
**Predicted under pi_5:** 0.4760 (var=0.0027)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.9, 0.8, 0.7, 0.6])
    
    def score_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(b * val) - np.sum(a * val)
        
    diffs = data.apply(score_diff, axis=1)
    
    b_better = data['response'][diffs > 0.5]
    a_better = data['response'][diffs < -0.5]
    
    m_b = b_better.mean() if len(b_better) > 0 else 0.5
    m_a = a_better.mean() if len(a_better) > 0 else 0.5
    
    return float(m_b - m_a)
```

**Observed (real) value:** -0.0500 (var=0.0218)
**Predicted under pi_4:** 0.0039 (var=0.0327)
**Predicted under pi_5:** 0.0139 (var=0.0334)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    # Calculate the difference in WADD scores between Option A and Option B
    score_diff = A.dot(validities) - B.dot(validities)
    
    # 1 if A was chosen, 0 if B was chosen
    chose_a = 1.0 - data['response'].values
    
    if np.var(score_diff) == 0:
        return 0.0
    
    # Calculate the linear slope of choosing A as a function of the score difference
    slope, _ = np.polyfit(score_diff, chose_a, 1)
    return float(slope)
```

**Observed (real) value:** 0.0014 (var=0.0003)
**Predicted under pi_4:** 0.0010 (var=0.0005)
**Predicted under pi_5:** 0.0035 (var=0.0004)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1 and 2, the observed mean absolute deviation of subjects' choice rates from 0.5 is ~0.033 to ~0.043. This is exactly what we would expect from pure binomial noise on a true underlying choice probability of 0.5 given the number of trials. Theory 1 (pi_4) predicted a much larger deviation (~0.10) because it assumed a significant spatial side bias. Theory 2 (pi_5) predicted the deviation well (~0.04) because its symmetric tallying design resulted in an expected overall choice rate of 0.5, with deviations driven only by noise. However, on other experiments (e.g., Exp 8), Theory 2's tallying mechanism predicts a slightly higher sensitivity to feature differences than is observed, whereas Theory 1's random guessing captures the near-zero slopes better. The data strongly suggest that subjects are engaging in pure random guessing with virtually no side bias.",
  "verdict": "new_model",
  "target_theory_idx": 1,
  "recommendation": "Preserve the Cognitive Overload (random guessing) theory, but heavily restrict the `side_bias` parameter range. The current range allows for too much systematic side bias, leading to an overestimation of the absolute deviation from 0.5. Adjust the parameter range for `side_bias` to be tightly centered around 0.5 (e.g., [0.48, 0.52] or fixed at 0.5) to reflect that subjects guess almost perfectly randomly without a strong spatial preference."
}
```

## Usage

```json
{
  "prompt_token_count": 7303,
  "candidates_token_count": 353,
  "total_token_count": 8815
}
```
