# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People use a non-compensatory, lexicographic heuristic called 'Take The Best' (TTB) to choose between options. They search through features in descending order of their subjective validity and stop at the first feature that discriminates between the two options (i.e., one option has a positive rating and the other does not). The choice is based entirely on this single discriminating cue, completely ignoring all remaining features, regardless of how many lower-validity cues might favor the alternative.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    chosen = -1
    for idx in order:
        if a[idx] > b[idx]:
            chosen = 0
            break
        elif b[idx] > a[idx]:
            chosen = 1
            break
            
    # Deterministic choice based on the first discriminating cue
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # If all features tie, guess randomly
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse rate for noise
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Probabilistic Take-The-Best (PTTB) with Softmax Smoothing: Subjects use a non-compensatory lexicographic search to choose between options, but their subjective representation of cue validities is subject to slight trial-by-trial noise. The cue hierarchy is determined by sorting the noisy validities. This causes occasional inversions in the search order (especially for cues with similar validities). Instead of a uniform lapse rate, the resulting deterministic choices are smoothed via a softmax function with a temperature parameter, allowing for a consistent baseline deviation from strict TTB without over-penalizing high-confidence predictions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"PTTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    temperature = float(parameters["temperature"])
    
    n_samples = 100
    p_core = np.zeros(2)
    
    for _ in range(n_samples):
        # Add Gaussian noise to the validities to simulate subjective trial-by-trial hierarchy
        noisy_v = validities + np.random.normal(0, sigma, size=len(validities))
        order = np.argsort(noisy_v)[::-1]
        
        chosen = -1
        for idx in order:
            if a[idx] > b[idx]:
                chosen = 0
                break
            elif b[idx] > a[idx]:
                chosen = 1
                break
                
        if chosen == 0:
            p_core[0] += 1.0
        elif chosen == 1:
            p_core[1] += 1.0
        else:
            p_core += 0.5
            
    p_core /= n_samples
    
    # Apply softmax with temperature to the core probabilities for smoother deviation
    z = p_core / temperature
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.83, 0.81, 0.7]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design specifically targets the dissociation between standard Take-The-Best (TTB) with a uniform lapse rate and Probabilistic Take-The-Best (PTTB) with validity noise. The advocated theory (TTB) relies solely on the ordinal ranking of validities: it will always check the highest-ranked cue first, regardless of how close it is to the second-best cue. The competing theory (PTTB) adds Gaussian noise to the validities before sorting, meaning that cues with similar validities will frequently swap ranks in the subject's mind. By using a set of validities with one large gap (e.g., 0.95 vs 0.85) and several small gaps (e.g., 0.85, 0.83, 0.81), we can create trials where the two models make identical predictions (when the discriminating cue is separated by a large gap) and trials where they diverge significantly (when the discriminating cue is only marginally better than the opposing cues, leading PTTB to frequently sample the opposing cues first).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In the chosen experimental design, the first discriminating cue (in descending order of true validities) always favors Option A. Therefore, the strict Take-The-Best (TTB) model will predict Option A on 100% of the trials, with deviations solely due to a uniform lapse rate (epsilon). In contrast, Probabilistic TTB (PTTB) introduces trial-by-trial noise to the validities, causing frequent rank inversions among cues with similar validities. Since Option B frequently possesses more positive cues overall, these rank inversions will lead PTTB to frequently sample an Option B-favoring cue first. Combined with softmax smoothing, PTTB will exhibit a significantly higher overall rate of choosing Option B. The overall proportion of Option B choices (mean of the `response` column) provides a robust, direct measure of this divergence.

Source:
def metric(data: pd.DataFrame) -> float:
    # Strict TTB always predicts option A (response 0) for all trials in this design
    # because the first discriminating cue always favors A.
    # Therefore, any response of 1 (choosing B) is due to noise.
    # PTTB will have a much higher rate of choosing B due to validity swaps
    # (especially when the gap between validities is small) and softmax smoothing.
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1260 (var=0.0054)
- Predicted under pi_4 (simulated): 0.3148 (var=0.0217)
- Observed on real data: 0.1229 (var=0.0065)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.92, 0.89, 0.86, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** This design specifically targets the dissociation between standard Take-The-Best (TTB) and Probabilistic Take-The-Best (PTTB) by manipulating the validity distance between the best discriminating cue and the next best cues. Standard TTB predicts a constant choice probability (1 - epsilon) for the option favored by the highest-validity discriminating cue, regardless of how close the next-best cue's validity is. In contrast, PTTB adds trial-by-trial noise to validities before sorting, meaning rank inversions are highly likely for cues with similar validities but exceedingly rare for cues with disparate validities. By pairing validities that are tightly clustered (e.g., 0.92, 0.89, 0.86) with those that are distant (0.65, 0.55), we create 'small gap' trials where PTTB predicts a much softer preference than TTB, and 'large gap' trials where PTTB and TTB both predict strong preferences.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Under standard Take-The-Best (TTB), the choice on both Trial 1 and Trial 2 is completely determined by the first cue (validity 0.92), which favors option A in both cases. Therefore, TTB predicts an identical probability of choosing Option B on both trials (just the uniform lapse rate), making the difference 0. Under Probabilistic Take-The-Best (PTTB), the noise in validities frequently causes the second and third cues (validities 0.89, 0.86) to temporarily outrank the first. In Trial 1, these cues strongly favor B, leading to a high rate of choosing B. In Trial 2, these cues do not favor B, so A is still chosen. Thus, PTTB predicts a significantly higher rate of choosing B in Trial 1 compared to Trial 2, resulting in a large positive metric.

Source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1))
    
    # Response is 0 for A, 1 for B. 
    # mean(response) is the rate of choosing option B.
    mean_t1 = data[t1_mask]['response'].mean()
    mean_t2 = data[t2_mask]['response'].mean()
    
    return float(mean_t1 - mean_t2)

### RESULTS
- Predicted under pi_3 (simulated): 0.0033 (var=0.0180)
- Predicted under pi_4 (simulated): 0.2050 (var=0.0648)
- Observed on real data: 0.0033 (var=0.0153)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers A (0), WADD prefers B (1)
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        # Trial 2: Tallying prefers B (1), WADD prefers A (0)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total
```

**Observed (real) value:** 0.1311 (var=0.0202)
**Predicted under pi_3:** 0.1222 (var=0.0114)
**Predicted under pi_4:** 0.1367 (var=0.0146)

### Experiment 4
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: Tallying prefers A (3 vs 2 wins), WADD prefers B (higher validity sum)
    t1_a = (0, 0, 0, 1, 1, 1)
    
    # Trial 5: Tallying prefers B (3 vs 2 wins), WADD prefers A (higher validity sum)
    t5_a = (1, 1, 0, 0, 0, 0)
    
    t1_mask = a_tuples == t1_a
    t5_mask = a_tuples == t5_a
    
    t1_resp = data.loc[t1_mask, 'response']
    t5_resp = data.loc[t5_mask, 'response']
    
    if len(t1_resp) == 0 or len(t5_resp) == 0:
        return 0.0
        
    p_a_t1 = (t1_resp == 0).mean()
    p_a_t5 = (t5_resp == 0).mean()
    
    # Tallying: P(A|T1) is high, P(A|T5) is low -> Positive difference
    # WADD: P(A|T1) is low, P(A|T5) is high -> Negative difference
    return float(p_a_t1 - p_a_t5)
```

**Observed (real) value:** -0.6650 (var=0.0405)
**Predicted under pi_3:** -0.7075 (var=0.0449)
**Predicted under pi_4:** -0.7492 (var=0.0451)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1 pits an option A with 3 low-validity features against an option B with 2 high-validity features.
    # Tallying prefers A (3 wins vs 2 wins), whereas WADD prefers B (score 1.7 vs 1.8).
    is_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if is_target.sum() == 0:
        return 0.5
    return float((data.loc[is_target, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1600 (var=0.0219)
**Predicted under pi_3:** 0.1425 (var=0.0219)
**Predicted under pi_4:** 0.1725 (var=0.0174)

### Experiment 6
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify critical trials (Trial 1 and Trial 2) where A is [0, 0, 0, 1, 1, 1]
    # and B has the first feature as 1 (either [1, 1, 0, 0, 0, 0] or [1, 0, 1, 0, 0, 0]).
    is_A_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    is_B_starts_1 = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    mask = is_A_target & is_B_starts_1
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1333 (var=0.0092)
**Predicted under pi_3:** 0.1267 (var=0.0090)
**Predicted under pi_4:** 0.1358 (var=0.0136)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_choices = 0
    total = 0
    
    for a_vals, b_vals, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = np.array(a_vals)
        b = np.array(b_vals)
        diff = a - b
        non_zero = np.where(diff != 0)[0]
        if len(non_zero) > 0:
            first_idx = non_zero[0]
            if diff[first_idx] > 0 and np.sum(b) > np.sum(a):
                total += 1
                if resp == 0:
                    ttb_choices += 1
            elif diff[first_idx] < 0 and np.sum(a) > np.sum(b):
                total += 1
                if resp == 1:
                    ttb_choices += 1
                    
    return float(ttb_choices / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.8375 (var=0.0070)
**Predicted under pi_3:** 0.8883 (var=0.0058)
**Predicted under pi_4:** 0.7529 (var=0.0208)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = sum_b > sum_a
    if not mask.any():
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1500 (var=0.0087)
**Predicted under pi_3:** 0.1217 (var=0.0073)
**Predicted under pi_4:** 0.1875 (var=0.0156)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Strict Take-The-Best, pi_3) consistently provides highly accurate predictions, particularly in Experiments 1 and 2 where it correctly predicts the lack of sensitivity to small validity gaps. Theory 2 (Probabilistic TTB, pi_4) consistently overestimates the variance and noise in choices by predicting that subjects will frequently sample lower-validity cues first when validity gaps are small. This leads to large errors in Exp 1 and Exp 2. Since Theory 1 is a very strong baseline, Theory 2 should be discarded and replaced with a fundamentally different mechanism to see if an alternative cognitive process can match TTB's predictive success.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a completely new theory that uses a different mechanism from TTB. For example, a noisy Weighted Additive (WADD) model where subjects integrate all cues weighted by their validities but with late-stage decision noise, or a two-stage heuristic (e.g., Tallying first, and using TTB only to break ties). This will test if a compensatory or hybrid approach can rival the strict non-compensatory TTB model."
}
```

## Usage

```json
{
  "prompt_token_count": 8130,
  "candidates_token_count": 271,
  "total_token_count": 9177
}
```
