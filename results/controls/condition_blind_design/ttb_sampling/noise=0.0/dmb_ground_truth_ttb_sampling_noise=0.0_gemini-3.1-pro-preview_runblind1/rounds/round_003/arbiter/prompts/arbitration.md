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
Take-The-Best (TTB) assumes that decision-makers do not integrate all information or simply count features. Instead, they rank features by their subjective or instructed validity and compare options lexicographically. They stop at the first feature that discriminates between the two options and choose the one with the higher value on that feature. If all features tie, they guess.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(-validities, kind='mergesort')
    
    score_a = 0.0
    score_b = 0.0
    
    # Lexicographic comparison
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    epsilon = float(parameters["epsilon"])
    
    if score_a > score_b:
        p_core = np.array([1.0, 0.0])
    elif score_b > score_a:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_5
Logistic Choice Model (Weighted Additive with Non-linear Validity Scaling): Decision-makers integrate all available information by computing a weighted sum of the features for each option. However, the subjective weights placed on these features are a non-linear scaling of their objective validities, governed by an exponent parameter (gamma). This allows the compensatory process to approximate non-compensatory heuristics when gamma is large, as the highest-validity features will dominate the weighted sum. The choice is made probabilistically by passing these weighted sums through a softmax function, modulated by an inverse temperature parameter (beta) to capture choice determinism, along with a baseline lapse rate (epsilon) for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities to compute subjective weights
    weights = validities ** gamma
    
    # Compute weighted sum of features using subjective weights
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    p_core = np.exp(z)
    p_core /= p_core.sum()
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.71, 0.84, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 1]  B=[0, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 12: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 13: A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 14: A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  trial 15: A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 16: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in adherence to the Take-The-Best (TTB) prediction between trials where the Logistic Choice Model (WADD) has a large difference in weighted sums vs. a small difference. TTB predicts equal choice certainty across both sets of trials because it relies on a single discriminating cue in both cases (yielding a metric near 0). WADD predicts much lower choice certainty on the 'low difference' trials, resulting in a large positive metric.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_tup'] = data['option_a_ratings'].apply(tuple)
    data['b_tup'] = data['option_b_ratings'].apply(tuple)
    
    # Trials where WADD has a very small difference in weighted sums
    low_diff_trials = {
        ((0, 1, 0, 0), (0, 0, 1, 0)): 1,  # TTB chooses B (response 1)
        ((0, 1, 1, 0), (0, 0, 1, 1)): 0   # TTB chooses A (response 0)
    }
    
    # Trials where WADD has a very large difference in weighted sums
    high_diff_trials = {
        ((1, 0, 0, 0), (1, 1, 1, 0)): 1,  # TTB chooses B
        ((1, 1, 1, 1), (0, 1, 1, 0)): 0,  # TTB chooses A
        ((1, 0, 1, 0), (1, 1, 1, 1)): 1,  # TTB chooses B
        ((1, 1, 1, 0), (0, 0, 1, 0)): 0   # TTB chooses A
    }
    
    def get_group(row):
        pair = (row['a_tup'], row['b_tup'])
        if pair in low_diff_trials:
            return 'low', low_diff_trials[pair]
        if pair in high_diff_trials:
            return 'high', high_diff_trials[pair]
        return 'none', -1
        
    groups = data.apply(get_group, axis=1)
    data['group'] = [g[0] for g in groups]
    data['ttb'] = [g[1] for g in groups]
    
    low_data = data[data['group'] == 'low']
    high_data = data[data['group'] == 'high']
    
    if len(low_data) == 0 or len(high_data) == 0:
        return 0.0
        
    low_acc = (low_data['response'] == low_data['ttb']).mean()
    high_acc = (high_data['response'] == high_data['ttb']).mean()
    
    return float(high_acc - low_acc)

### RESULTS
- Predicted under pi_3 (simulated): 0.0100 (var=0.0104)
- Predicted under pi_5 (simulated): 0.0570 (var=0.0153)
- Observed on real data: -0.0073 (var=0.0102)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.95, 0.59, 0.81, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 11: A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 12: A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 13: A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 14: A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  trial 15: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 16: A=[1, 1, 1, 0]  B=[1, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric exploits the variance in choice determinism predicted by the Logistic Choice Model (WADD). We partition trials into 'Strong' (where the WADD value difference between options is large) and 'Weak' (where the WADD value difference is very small). Under TTB, all these trials are resolved decisively by a single discriminating feature, predicting equal accuracy (TTB-consistency) across both sets (difference ~ 0). WADD predicts choices will be much more deterministic on Strong trials than on Weak trials, leading to a strongly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    # Convert lists to strings for fast comparison
    data_a = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    data_b = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Strong trials: large difference in weighted sums under WADD
    strong_0 = (data_a == "1011") & (data_b == "0000")
    strong_1 = ((data_a == "0100") & (data_b == "1111")) | ((data_a == "0010") & (data_b == "1111"))
    strong_2 = (data_a == "1111") & (data_b == "0001")
    
    # Weak trials: very small difference in weighted sums under WADD
    weak_0 = ((data_a == "0110") & (data_b == "0101")) | ((data_a == "1110") & (data_b == "1011"))
    
    strong_correct = 0
    strong_total = 0
    
    if strong_0.any():
        strong_correct += (data.loc[strong_0, 'response'] == 0).sum()
        strong_total += strong_0.sum()
    if strong_1.any():
        strong_correct += (data.loc[strong_1, 'response'] == 1).sum()
        strong_total += strong_1.sum()
    if strong_2.any():
        strong_correct += (data.loc[strong_2, 'response'] == 0).sum()
        strong_total += strong_2.sum()
        
    weak_correct = 0
    weak_total = 0
    if weak_0.any():
        weak_correct += (data.loc[weak_0, 'response'] == 0).sum()
        weak_total += weak_0.sum()
        
    strong_acc = strong_correct / strong_total if strong_total > 0 else 0.0
    weak_acc = weak_correct / weak_total if weak_total > 0 else 0.0
    
    return float(strong_acc - weak_acc)

### RESULTS
- Predicted under pi_3 (simulated): -0.0175 (var=0.0144)
- Predicted under pi_5 (simulated): 0.1925 (var=0.0170)
- Observed on real data: -0.0300 (var=0.0180)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_favored_choices = 0
    tied_trials = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a == (0, 1, 0, 1) and b == (1, 1, 0, 0):
            wadd_favored = 1
        elif a == (1, 0, 1, 1) and b == (1, 1, 1, 0):
            wadd_favored = 1
        elif a == (1, 1, 1, 0) and b == (0, 1, 1, 1):
            wadd_favored = 0
        elif a == (1, 1, 0, 1) and b == (1, 1, 1, 0):
            wadd_favored = 1
        else:
            continue
            
        tied_trials += 1
        if row['response'] == wadd_favored:
            wadd_favored_choices += 1
            
    if tied_trials == 0:
        return 0.5
    return float(wadd_favored_choices / tied_trials)
```

**Observed (real) value:** 0.8667 (var=0.0063)
**Predicted under pi_3:** 0.8450 (var=0.0131)
**Predicted under pi_5:** 0.8667 (var=0.0102)

### Experiment 4
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.8, 0.85, 0.55])
    wadd_consistent = 0
    tally_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where Tallying predicts a tie
        if a_wins == b_wins:
            score_a = np.sum(a * validities)
            score_b = np.sum(b * validities)
            
            if score_a > score_b:
                wadd_choice = 0
            elif score_b > score_a:
                wadd_choice = 1
            else:
                continue
                
            if row['response'] == wadd_choice:
                wadd_consistent += 1
            tally_ties += 1
            
    if tally_ties == 0:
        return 0.5
    return float(wadd_consistent / tally_ties)
```

**Observed (real) value:** 0.8767 (var=0.0106)
**Predicted under pi_3:** 0.8683 (var=0.0128)
**Predicted under pi_5:** 0.9017 (var=0.0119)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 12: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
    # TTB chooses A (f0 discriminates). WADD might choose B if w2 is high and w0/w3 are low.
    mask_t12 = (a_tuples == (1, 0, 0, 1)) & (b_tuples == (0, 0, 1, 0))
    
    # Trial 15: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
    # TTB chooses A (f2 discriminates). WADD might choose B if w1 > w2.
    mask_t15 = (a_tuples == (1, 0, 1, 0)) & (b_tuples == (1, 1, 0, 0))
    
    # Trial 4: A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
    # TTB chooses B (f0 discriminates). WADD might choose A if w2 is high and w0 is low.
    mask_t4 = (a_tuples == (0, 1, 1, 0)) & (b_tuples == (1, 1, 0, 0))
    
    ttb_aligned = 0
    total = 0
    
    if mask_t12.sum() > 0:
        ttb_aligned += (data.loc[mask_t12, 'response'] == 0).sum()
        total += mask_t12.sum()
        
    if mask_t15.sum() > 0:
        ttb_aligned += (data.loc[mask_t15, 'response'] == 0).sum()
        total += mask_t15.sum()
        
    if mask_t4.sum() > 0:
        ttb_aligned += (data.loc[mask_t4, 'response'] == 1).sum()
        total += mask_t4.sum()
        
    if total == 0:
        return 0.5
        
    return float(ttb_aligned) / total
```

**Observed (real) value:** 0.8644 (var=0.0213)
**Predicted under pi_3:** 0.8889 (var=0.0114)
**Predicted under pi_5:** 0.9111 (var=0.0094)

### Experiment 6
**Design**
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 11 is the most discriminative trial:
    # Option A: [1, 1, 0, 0] vs Option B: [0, 1, 1, 1]
    # TTB looks at the highest validity feature (F0: 0.95), where A=1 and B=0, so TTB strictly prefers A (response 0).
    # WADD computes a weighted sum. Since F1 cancels out, WADD compares w0*0.95 (for A) against w2*0.58 + w3*0.55 (for B).
    # Because the weights w0, w2, w3 are independently drawn from U(0,1), the sum w2*0.58 + w3*0.55 is frequently greater than w0*0.95.
    # Thus, WADD will choose B (response 1) more than half the time, whereas TTB will choose B almost never (only on lapses).
    
    a_target = [1, 1, 0, 0]
    b_target = [0, 1, 1, 1]
    
    mask = (data['option_a_ratings'].apply(lambda x: list(x) == a_target) & 
            data['option_b_ratings'].apply(lambda x: list(x) == b_target))
            
    if not mask.any():
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1800 (var=0.0309)
**Predicted under pi_3:** 0.1633 (var=0.0283)
**Predicted under pi_5:** 0.1167 (var=0.0381)

### Experiment 7
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.55, 0.65, 0.55])
    order = np.argsort(-validities, kind='mergesort')
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_choice = -1
        for idx in order:
            if a[idx] > b[idx]:
                ttb_choice = 0
                break
            elif b[idx] > a[idx]:
                ttb_choice = 1
                break
                
        if ttb_choice != -1:
            if row['response'] == ttb_choice:
                match_count += 1
            total += 1
            
    if total == 0:
        return 0.5
    return match_count / total
```

**Observed (real) value:** 0.8700 (var=0.0128)
**Predicted under pi_3:** 0.8781 (var=0.0057)
**Predicted under pi_5:** 0.8700 (var=0.0059)

### Experiment 8
**Design**
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def choice_matches_ttb(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        response = row['response']
        for idx in [0, 2, 1, 3]:
            if a[idx] > b[idx]:
                return 1 if response == 0 else 0
            elif b[idx] > a[idx]:
                return 1 if response == 1 else 0
        return 0.5
    
    matches = data.apply(choice_matches_ttb, axis=1)
    return float(matches.mean())
```

**Observed (real) value:** 0.8250 (var=0.0157)
**Predicted under pi_3:** 0.8694 (var=0.0069)
**Predicted under pi_5:** 0.9017 (var=0.0039)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 1 (Take-The-Best, pi_3) consistently outperforms Theory 2 (Logistic Choice Model / WADD, pi_5). This is especially evident in Experiments 1 and 2, which were specifically designed to test the compensatory nature of pi_5. In those experiments, pi_5 predicted a large difference in choice determinism between trials with large vs. small weighted-sum differences, but the observed data showed almost no difference (values near zero or slightly negative), tightly matching the predictions of pi_3. While pi_5 matches the observed data well in some baseline accuracy metrics (e.g., Exp 3 and Exp 7), it fundamentally fails on the adversarial tests because it assumes compensatory integration of all features, which the subjects do not appear to be doing. Therefore, pi_3 is the much stronger core explanation.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Logistic Choice Model with a 'Strategy Selection / Adaptive Toolbox' theory. Instead of assuming a single compensatory mechanism with non-linear weights, this new theory should posit that decision-makers probabilistically select between different simple heuristics on a trial-by-trial or subject-by-subject basis. For example, a mixture model where agents predominantly use Take-The-Best but have a certain probability of falling back to 'Tallying' (simply counting the number of positive features without weighting them by validity) when confidence is low. This preserves the non-compensatory nature of the data while allowing for slight deviations from strict TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 10038,
  "candidates_token_count": 346,
  "total_token_count": 11267
}
```
