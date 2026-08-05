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
Weighted Additive Model with Power-Scaled Log-Odds (WADD-Gamma). Decision-makers compute a weighted sum of features for each option. The weights are derived from the log-odds of the cue validities, raised to a power gamma. This parameterization allows the model to smoothly interpolate between Tallying/Equal-Weighting (gamma = 0) and standard log-odds WADD (gamma = 1). Choices are then made via a softmax over the weighted sums, incorporating an independent lapse rate for noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to ensure log-odds are strictly positive and well-defined
    v_clipped = np.clip(validities, 0.5001, 0.9999)
    log_odds = np.log(v_clipped / (1.0 - v_clipped))
    
    gamma = float(parameters["gamma"])
    weights = log_odds ** gamma
    
    # Weighted sum for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Probabilistic Search Take-The-Best (PS-TTB)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.92, 0.82, 0.72, 0.62, 0.52, 0.52]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate WADD-Gamma from Probabilistic Search Take-The-Best (PS-TTB), we use a 6-feature design with a carefully chosen validity gradient. WADD-Gamma integrates all features using a weighted sum where weights are log-odds raised to a power (gamma). PS-TTB, on the other hand, probabilistically samples cue orders based on their validities and stops at the first discriminating cue. We include 'compensatory' trials where the single highest-validity cue points to Option B, but the sum of the remaining lower-validity cues strongly points to Option A. Because PS-TTB stops at the first discriminating cue, it will frequently choose Option B on these trials. WADD-Gamma will integrate the multiple lower-validity cues and, depending on the gamma parameter, can strongly favor Option A. We also include trials where the top cue is tied (0 vs 0 or 1 vs 1) to test how the models handle secondary cues: PS-TTB's probabilistic search will distribute choices among the remaining cues, whereas WADD-Gamma deterministically sums them.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric contrasts the probability of choosing Option B on Trial 1 versus Trial 2. On Trial 2, both WADD-Gamma and PS-TTB strongly agree that Option A is superior across their entire parameter spaces, making P(B|T2) an excellent estimate of a subject's baseline noise/lapse rate. On Trial 1, WADD-Gamma consistently favors Option A (due to its compensatory nature integrating the 5 lower-validity cues), keeping P(B|T1) near the baseline noise level. Thus, WADD-Gamma predicts the difference P(B|T1) - P(B|T2) to be approximately zero. In contrast, PS-TTB frequently evaluates the most valid cue first and chooses Option B on Trial 1, leading to a significantly higher P(B|T1). By subtracting the baseline noise (P(B|T2)), we isolate the lexicographic search mechanism while minimizing within-subject variance caused by different epsilon (noise) parameters.

Source:
def metric(data: pd.DataFrame) -> float:
    # Trial 1: A=[0, 1, 1, 1, 1, 1], B=[1, 0, 0, 0, 0, 0]
    # Trial 2: A=[1, 0, 0, 1, 1, 1], B=[0, 1, 1, 0, 0, 0]
    def is_t1(row):
        return tuple(int(v) for v in row['option_a_ratings']) == (0, 1, 1, 1, 1, 1) and \
               tuple(int(v) for v in row['option_b_ratings']) == (1, 0, 0, 0, 0, 0)
    def is_t2(row):
        return tuple(int(v) for v in row['option_a_ratings']) == (1, 0, 0, 1, 1, 1) and \
               tuple(int(v) for v in row['option_b_ratings']) == (0, 1, 1, 0, 0, 0)
    
    mask_t1 = data.apply(is_t1, axis=1)
    mask_t2 = data.apply(is_t2, axis=1)
    
    p_b_t1 = float(data[mask_t1]['response'].mean()) if mask_t1.any() else 0.5
    p_b_t2 = float(data[mask_t2]['response'].mean()) if mask_t2.any() else 0.5
    
    return p_b_t1 - p_b_t2

### RESULTS
- Predicted under pi_5 (simulated): -0.0583 (var=0.0362)
- Predicted under pi_4 (simulated): -0.0550 (var=0.0418)
- Observed on real data: -0.0067 (var=0.0494)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To quantitatively dissociate Probabilistic Search Take-The-Best (PS-TTB) from WADD-Gamma, we exploit WADD-Gamma's parameterization, which uses a power parameter (gamma <= 0.75) to compress the log-odds of validities. This compression makes WADD-Gamma highly compensatory, reliably favoring options with a larger number of lower-validity cues. In contrast, PS-TTB probabilistically searches through cues based on their validities and stops at the first discriminating cue, maintaining a strong non-compensatory characteristic. We use a 5-feature design with a steep validity gradient. We include 'conflict' trials where the highest-validity cue points to one option, but the remaining cues point to the other (WADD-Gamma favors the latter, PS-TTB favors the former). We also include tied highest-cue trials to test secondary cue reliance and agreement trials to accurately estimate baseline noise and search parameters.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric measures the change in preference for the option possessing the highest-validity cue (Cue 1) when it is supported by additional lower-validity cues, compared to when it stands alone against a strong compensatory alternative. In Trial 1, Option A has only Cue 1, while Option B has Cues 2, 3, and 4. WADD-Gamma, being compensatory, reliably favors Option B because the sum of the lower validities outweighs Cue 1. PS-TTB, being non-compensatory, searches sequentially by validity and frequently stops at Cue 1, thus favoring Option A. In Trial 7, Option A has Cues 1, 3, and 4, while Option B has Cues 2 and 5. Here, WADD-Gamma's weighted sum shifts massively in favor of Option A. PS-TTB also favors Option A, as it still possesses the highest validity cue. By taking the difference in the probability of choosing Option A between Trial 7 and Trial 1, we isolate the models' structural differences. For PS-TTB, the probability of choosing Option A is high in both trials (especially for small/moderate tau), leading to a tight difference near 0 (ranging up to 0.35 for fully random large tau). For WADD-Gamma, the probability swings from strongly favoring B in Trial 1 to strongly favoring A in Trial 7, yielding a large positive difference (typically 0.6 to 1.0). This provides a large, consistent mean gap with minimal overlap and low between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    def p_target(target_a, target_b):
        m_straight = (a_str == target_a) & (b_str == target_b)
        m_swap = (a_str == target_b) & (b_str == target_a)
        
        n_straight = m_straight.sum()
        n_swap = m_swap.sum()
        
        if n_straight + n_swap == 0:
            return 0.5
            
        wins = 0
        if n_straight > 0:
            wins += (data.loc[m_straight, 'response'] == 0).sum()
        if n_swap > 0:
            wins += (data.loc[m_swap, 'response'] == 1).sum()
            
        return float(wins / (n_straight + n_swap))
        
    # Trial 1: Target A = 10000, Target B = 01110
    p_t1 = p_target("10000", "01110")
    
    # Trial 7: Target A = 10110, Target B = 01001
    p_t7 = p_target("10110", "01001")
    
    return float(p_t7 - p_t1)

### RESULTS
- Predicted under pi_5 (simulated): 0.4050 (var=0.1042)
- Predicted under pi_4 (simulated): 0.1567 (var=0.0657)
- Observed on real data: 0.1933 (var=0.0443)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_0 = data['option_b_ratings'].apply(lambda x: x[0])
    ttb_choice = np.where(b_0 > a_0, 1, 0)
    return float(np.mean(data['response'] == ttb_choice))
```

**Observed (real) value:** 0.4550 (var=0.0021)
**Predicted under pi_5:** 0.4558 (var=0.0046)
**Predicted under pi_4:** 0.4452 (var=0.0026)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.75, 0.65, 0.55])
    
    def is_ttb_match(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_winner = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        # WADD expected prediction (assuming uniform weights)
        wadd_a = np.sum(a * validities)
        wadd_b = np.sum(b * validities)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        # Only consider compensatory trials where the models disagree
        if ttb_winner is not None and ttb_winner != wadd_winner:
            return 1.0 if row['response'] == ttb_winner else 0.0
        return np.nan

    matches = data.apply(is_ttb_match, axis=1)
    return float(matches.mean())
```

**Observed (real) value:** 0.4225 (var=0.0057)
**Predicted under pi_5:** 0.3683 (var=0.0210)
**Predicted under pi_4:** 0.3950 (var=0.0059)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def is_11000(x):
        return tuple(x) == (1, 1, 0, 0, 0)
    
    def is_00111(x):
        return tuple(x) == (0, 0, 1, 1, 1)
        
    a_11000 = data['option_a_ratings'].apply(is_11000)
    b_00111 = data['option_b_ratings'].apply(is_00111)
    
    a_00111 = data['option_a_ratings'].apply(is_00111)
    b_11000 = data['option_b_ratings'].apply(is_11000)
    
    t1 = a_11000 & b_00111
    t3 = a_00111 & b_11000
    
    chose_11000 = (t1 & (data['response'] == 0)) | (t3 & (data['response'] == 1))
    
    relevant = t1 | t3
    if relevant.sum() == 0:
        return 0.5
    return float(chose_11000.sum() / relevant.sum())
```

**Observed (real) value:** 0.4183 (var=0.0241)
**Predicted under pi_5:** 0.6350 (var=0.0375)
**Predicted under pi_4:** 0.4542 (var=0.0118)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        # Target trials where WADD and Tallying strictly disagree
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return True
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return True
        return False

    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    sum_a = target_data['option_a_ratings'].apply(sum)
    sum_b = target_data['option_b_ratings'].apply(sum)
    
    # Calculate how often the subject chose the option with MORE positive features (Tallying's preference)
    chose_more = ((target_data['response'] == 0) & (sum_a > sum_b)) | \
                 ((target_data['response'] == 1) & (sum_b > sum_a))
                 
    return float(chose_more.mean())
```

**Observed (real) value:** 0.5867 (var=0.0125)
**Predicted under pi_5:** 0.3608 (var=0.0418)
**Predicted under pi_4:** 0.5675 (var=0.0095)

### Experiment 7
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_ratings = np.stack(data['option_a_ratings'].values)
    B_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_A = A_ratings.sum(axis=1)
    sum_B = B_ratings.sum(axis=1)
    
    # Identify all trials where the difference in total positive features is exactly 1.
    # These are T1, T5, T6, and T7.
    diff_1_mask = np.abs(sum_A - sum_B) == 1
    
    if not np.any(diff_1_mask):
        return 0.5
        
    data_diff1 = data[diff_1_mask]
    sum_A_diff1 = sum_A[diff_1_mask]
    sum_B_diff1 = sum_B[diff_1_mask]
    responses = data_diff1['response'].values
    
    # 1 if the subject chose the option with the higher total number of positive features, 0 otherwise
    chose_higher = np.where(sum_A_diff1 > sum_B_diff1, responses == 0, responses == 1)
    
    return float(np.mean(chose_higher))
```

**Observed (real) value:** 0.6117 (var=0.0051)
**Predicted under pi_5:** 0.6625 (var=0.0205)
**Predicted under pi_4:** 0.5617 (var=0.0039)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the sum of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Isolate trials where the feature sums differ (diff > 0)
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    valid_data = data[mask].copy()
    
    # Determine the 'majority' option (0 for A, 1 for B)
    maj_choice = (sum_a[mask] < sum_b[mask]).astype(int)
    
    # Check if the subject chose the majority option
    valid_data['is_maj'] = (valid_data['response'] == maj_choice).astype(float)
    
    # Calculate the overall accuracy relative to the feature sum per subject
    subj_acc = valid_data.groupby('subject_id')['is_maj'].mean()
    
    # Apply a smooth sigmoid centered at 0.75.
    # Tallying uses a logistic link function, routinely producing choice accuracies > 0.80 
    # when beta is moderate/high and epsilon is low. 
    # PS-TTB with tau > 1 (99% of its parameter space) behaves identically to Random Search,
    # whose theoretical maximum accuracy on these trials is capped exactly at ~0.716.
    # A smooth slope (12.0) prevents the variance from exploding (unlike steep thresholds or high powers),
    # while cleanly separating the dense right tail of Tallying from the bounded distribution of PS-TTB.
    score = 1.0 / (1.0 + np.exp(-12.0 * (subj_acc - 0.75)))
    
    return float(score.mean())
```

**Observed (real) value:** 0.1432 (var=0.0027)
**Predicted under pi_5:** 0.2567 (var=0.0565)
**Predicted under pi_4:** 0.1988 (var=0.0196)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the eight experiments, Theory 2 (PS-TTB) consistently outperforms Theory 1 (WADD-Gamma). In Experiments 2, 5, and 6, WADD-Gamma's predictions diverge drastically from the observed data, largely because its compensatory nature heavily overestimates the influence of multiple lower-validity cues. PS-TTB's non-compensatory probabilistic search mechanism aligns much better with the observed metric values, suggesting that subjects frequently rely on a single discriminating cue rather than integrating all available information. However, PS-TTB isn't perfect (e.g., in Exp 7, WADD-Gamma is slightly closer), indicating that there may be a subset of trials or subjects where compensatory integration occurs. Theory 1 (WADD-Gamma) is fundamentally limited by its strict integration mechanism and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Selection Model' or a 'Two-Stage Heuristic'. In this new theory, decision-makers probabilistically choose between a compensatory strategy (like Tallying or WADD) and a non-compensatory strategy (like TTB) on a trial-by-trial or subject-by-subject basis, depending on the dispersion of cue validities or a latent strategy-preference parameter. Alternatively, propose a Sequential Sampling Model (like a multi-attribute Drift Diffusion Model) where cue validities drive the drift rate, naturally naturally producing TTB-like behavior for steep validity gradients and compensatory behavior for flat ones. This would provide a stronger, more flexible competitor to PS-TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 8827,
  "candidates_token_count": 357,
  "total_token_count": 10163
}
```
