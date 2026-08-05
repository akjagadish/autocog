# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Parallel Constraint Satisfaction (PCS) Decision Network with Softmax Cue Weighting. Choices emerge from a recurrent neural network where options and features bidirectionally interact. Options accumulate evidence from features and mutually inhibit each other. Options also send feedback to features, amplifying features that support the leading option (coherence shift). To allow the model to flexibly transition between compensatory and non-compensatory (lexicographic) behavior without destabilizing the network, the initial cue validities are transformed via a softmax function controlled by a temperature parameter (tau). This ensures the highest-validity cue can decisively dominate the parallel accumulation when necessary, while bounded inhibition and feedback terms prevent runaway dynamics.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    tau = float(parameters["tau"])
    lam = float(parameters["lam"])
    gamma = float(parameters["gamma"])
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax transformation of validities to allow lexicographic dominance
    z_v = tau * validities
    z_v = z_v - np.max(z_v)
    weights = np.exp(z_v) / np.sum(np.exp(z_v))
    
    # Initialize activations: Options at 0, Cues at their resting weights
    O = np.array([0.0, 0.0])
    C = weights.copy()
    
    w_A = a - b
    w_B = b - a
    
    alpha = 0.1
    n_iter = 50
    
    for _ in range(n_iter):
        # Net input to options: evidence from cues minus lateral inhibition
        net_O_A = np.sum(w_A * C) - lam * O[1]
        net_O_B = np.sum(w_B * C) - lam * O[0]
        
        # Net input to cues: feedback from options
        net_C = gamma * w_A * O[0] + gamma * w_B * O[1]
        
        # Update options (McClelland & Rumelhart rule)
        net_O = np.array([net_O_A, net_O_B])
        delta_O = np.where(net_O > 0,
                           alpha * net_O * (1.0 - O) - kappa * O,
                           alpha * net_O * O - kappa * O)
        O = np.clip(O + delta_O, 0.0, 1.0)
        
        # Update cues (decay towards resting weights)
        delta_C = np.where(net_C > 0,
                           alpha * net_C * (1.0 - C) - kappa * (C - weights),
                           alpha * net_C * C - kappa * (C - weights))
        C = np.clip(C + delta_C, 0.0, 1.0)
        
    scores = O
    
    # Softmax choice rule
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_5
Sequential Evidence Accumulation. Decision-makers inspect features sequentially in decreasing order of validity. At each step, the validity-weighted difference between the options' features is added to an accumulated evidence tally. If the absolute value of this accumulated evidence exceeds a critical threshold, search stops immediately and a decision is made based on the current tally (mimicking Take-The-Best when the threshold is low). If all features are exhausted without crossing the threshold, a choice is made probabilistically based on the final accumulated tally (mimicking compensatory strategies like Weighted Additive when the threshold is high).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order (stable sort for ties)
    order = np.argsort(-validities, kind='stable')
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    
    # Sequential search and evidence accumulation
    for idx in order:
        diff = a[idx] - b[idx]
        evidence += validities[idx] * diff
        
        # Stop search if accumulated evidence exceeds the threshold
        if abs(evidence) >= threshold:
            break
            
    # Convert the final evidence into choice probabilities.
    # Evidence represents the accumulated advantage of Option A over Option B.
    scores = np.array([evidence, 0.0])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To strictly dissociate the Parallel Constraint Satisfaction (PCS) Decision Network from the Sequential Evidence Accumulation model, we exploit the PCS model's softmax transformation of cue validities. When the temperature parameter (tau) is low, PCS approaches an equal-weighting (Tallying) strategy, regardless of the original validity magnitudes. The Sequential Accumulator, however, searches in strict validity order and its predictions are mathematically bounded by Take-The-Best (at low thresholds) and Weighted Additive (at high thresholds). In the critical dissociation trials (e.g., Trial 1), Option B wins on the two highest-validity features, while Option A wins on the three lowest-validity features. Because the sum of the top two validities exceeds the sum of the bottom three, both TTB and WADD strongly favor Option B. Consequently, the Sequential Accumulator MUST predict Option B for any threshold. However, PCS with a low tau will predict Option A because it counts more supporting features (3 vs 2). This creates a strict qualitative dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with a simple Tallying strategy (choosing the option with the most positive features, regardless of feature validities). The PCS model, particularly at lower temperature (tau) values, naturally tends toward this equal-weighting behavior because the softmax transformation compresses the validity differences. In contrast, the Sequential Evidence Accumulation model is strictly bounded between Take-The-Best and Weighted Additive strategies. In the critical dissociation trials (e.g., Trial 1 and 2), the option with fewer features has the two highest-validity cues, making it the mathematically guaranteed choice for the Sequential Accumulator under any threshold. Therefore, the PCS model will produce a significantly higher proportion of Tallying-consistent choices overall compared to the Sequential Accumulator.

Source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Tallying prediction: choose the option with more positive features
    # Returns 0 if A has more, 1 if B has more
    tally_prediction = (sum_b > sum_a).astype(int)
    
    # Calculate the proportion of trials where the subject's response matches the Tallying prediction
    return float((data['response'] == tally_prediction).mean())

### RESULTS
- Predicted under pi_6 (simulated): 0.4465 (var=0.0207)
- Predicted under pi_5 (simulated): 0.2800 (var=0.0471)
- Observed on real data: 0.4158 (var=0.0019)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=9):** [0.95, 0.85, 0.8, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0]

**Rationale:** To strictly dissociate Sequential Evidence Accumulation from the Parallel Constraint Satisfaction (PCS) network, we exploit a non-monotonic stopping behavior unique to the Sequential model. In Trial 1, Option B wins on the single highest-validity cue (Cue 1) and on five low-validity cues (Cues 5-9). Option A wins on three intermediate-validity cues (Cues 2-4). Because Option B is favored by Take-The-Best (Cue 1), Tallying (6 wins vs 3), and WADD (sum of validities is higher), the PCS model—whose predictions are bounded by these compensatory and non-compensatory extremes—must strongly predict Option B across all possible parameterizations. However, the Sequential Accumulator exhibits a unique intermediate-threshold behavior: it survives Cue 1 without stopping, then accumulates three successive wins for Option A, pushing the evidence tally across the threshold to trigger an early stop and a choice for Option A. This creates a strict qualitative dissociation where the advocated model predicts a choice (A) that is mathematically impossible for the competing model.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
In Trial 3, Option B holds the top two highest-validity cues (+0.95, +0.85), while Option A holds all seven remaining cues. Under Sequential Evidence Accumulation, the maximum possible stopping threshold is 1.5. After evaluating the first two cues, the accumulated evidence for Option B is 1.80, which strictly exceeds the maximum threshold. Thus, the Sequential model will ALWAYS stop by the second cue and choose B, regardless of its parameters, yielding P(A|T3) near 0 (driven only by the epsilon lapse rate) with extremely low between-subject variance. Conversely, the PCS model integrates all cues simultaneously. For roughly half of its parameter space (tau < 4.5), the sheer number of cues favoring A (7 vs 2) overwhelms the validity advantage of B, causing PCS to strongly prefer A. This creates a massive mean difference between the two models (near 0.0 for Sequential vs ~0.45 for PCS) that will easily reach statistical significance.

Source:
def metric(data: pd.DataFrame) -> float:
    # Mask for Trial 3: A=[0, 0, 1, 1, 1, 1, 1, 1, 1], B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
    t3_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1, 1, 1, 1, 1)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0, 0, 0, 0))
    
    if t3_mask.sum() == 0:
        return 0.0
        
    # Return the probability of choosing Option A in Trial 3
    p_a_t3 = 1.0 - data[t3_mask]['response'].mean()
    return float(p_a_t3)

### RESULTS
- Predicted under pi_6 (simulated): 0.4232 (var=0.0422)
- Predicted under pi_5 (simulated): 0.1789 (var=0.0139)
- Observed on real data: 0.4463 (var=0.0140)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tallying_consistent = 0
    total_dissociation_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Identify dissociation trials: one option has more features, but lacks the first (highest validity) feature
        if sum_a > sum_b and a[0] == 0:
            total_dissociation_trials += 1
            if row['response'] == 0:
                tallying_consistent += 1
        elif sum_b > sum_a and b[0] == 0:
            total_dissociation_trials += 1
            if row['response'] == 1:
                tallying_consistent += 1
                
    if total_dissociation_trials == 0:
        return 0.5
        
    return float(tallying_consistent / total_dissociation_trials)
```

**Observed (real) value:** 0.4113 (var=0.0093)
**Predicted under pi_6:** 0.3375 (var=0.0246)
**Predicted under pi_5:** 0.1256 (var=0.0087)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_f1 = np.array([a[0] for a in data['option_a_ratings']])
    b_f1 = np.array([b[0] for b in data['option_b_ratings']])
    resp = data['response'].values
    chosen_f1 = np.where(resp == 0, a_f1, b_f1)
    return float(np.mean(chosen_f1))
```

**Observed (real) value:** 0.5900 (var=0.0027)
**Predicted under pi_6:** 0.6069 (var=0.0138)
**Predicted under pi_5:** 0.7700 (var=0.0166)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def ttb_match(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return row['response'] == 0
            elif b[i] > a[i]:
                return row['response'] == 1
        return False

    return float(data.apply(ttb_match, axis=1).mean())
```

**Observed (real) value:** 0.6004 (var=0.0022)
**Predicted under pi_6:** 0.4473 (var=0.0141)
**Predicted under pi_5:** 0.5810 (var=0.0826)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Select trials where Option B has more positive features than Option A
    # In the experimental design (Trials 1, 2, 3, 5), Option A is endorsed by the 
    # single highest-validity discriminating cue, but Option B has a higher sheer 
    # number of lower-validity cues.
    mask = sum_b > sum_a
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option B was chosen (response == 1)
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4131 (var=0.0041)
**Predicted under pi_6:** 0.5228 (var=0.0230)
**Predicted under pi_5:** 0.4400 (var=0.1377)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    data = data.copy()
    data['a_tup'] = data['option_a_ratings'].apply(tuple)
    data['b_tup'] = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 0, 0]
    m1_norm = (data['a_tup'] == (1, 0, 0, 0, 0)) & (data['b_tup'] == (0, 1, 1, 0, 0))
    m1_swap = (data['a_tup'] == (0, 1, 1, 0, 0)) & (data['b_tup'] == (1, 0, 0, 0, 0))
    p_a_t1_norm = (data.loc[m1_norm, 'response'] == 0).mean() if m1_norm.sum() > 0 else np.nan
    p_a_t1_swap = (data.loc[m1_swap, 'response'] == 1).mean() if m1_swap.sum() > 0 else np.nan
    t1_vals = [x for x in [p_a_t1_norm, p_a_t1_swap] if not np.isnan(x)]
    p_a_t1 = np.mean(t1_vals) if t1_vals else 0.5
    
    # Trial 3: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 1, 1]
    m3_norm = (data['a_tup'] == (1, 0, 0, 0, 0)) & (data['b_tup'] == (0, 1, 1, 1, 1))
    m3_swap = (data['a_tup'] == (0, 1, 1, 1, 1)) & (data['b_tup'] == (1, 0, 0, 0, 0))
    p_a_t3_norm = (data.loc[m3_norm, 'response'] == 0).mean() if m3_norm.sum() > 0 else np.nan
    p_a_t3_swap = (data.loc[m3_swap, 'response'] == 1).mean() if m3_swap.sum() > 0 else np.nan
    t3_vals = [x for x in [p_a_t3_norm, p_a_t3_swap] if not np.isnan(x)]
    p_a_t3 = np.mean(t3_vals) if t3_vals else 0.5
    
    return float(p_a_t1 - p_a_t3)
```

**Observed (real) value:** -0.0400 (var=0.0356)
**Predicted under pi_6:** 0.1133 (var=0.0422)
**Predicted under pi_5:** 0.0033 (var=0.0219)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.8, 0.75, 0.6, 0.5])
    a_mat = np.vstack(data['option_a_ratings'].values)
    b_mat = np.vstack(data['option_b_ratings'].values)
    
    a_scores = a_mat @ validities
    b_scores = b_mat @ validities
    
    # Identify critical trials where the sum of validities is higher for Option B
    critical_trials = b_scores > a_scores
    
    if not np.any(critical_trials):
        return 0.5
        
    # Return the proportion of times Option B was chosen in these critical trials
    return float(data.loc[critical_trials, 'response'].mean())
```

**Observed (real) value:** 0.4238 (var=0.0036)
**Predicted under pi_6:** 0.4866 (var=0.0089)
**Predicted under pi_5:** 0.3606 (var=0.0862)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert option A ratings to tuples for easy matching
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: TTB chooses A, Tallying chooses A. 
    # Accumulator chooses A (low threshold) or B (high threshold).
    t1_mask = a_tuples == (1, 0, 0, 0, 1, 1, 1)
    
    # Trial 3: TTB chooses A, Tallying chooses B.
    # Accumulator ALWAYS chooses A (evidence is strictly positive at every step).
    t3_mask = a_tuples == (1, 0, 1, 1, 0, 0, 0)
    
    # Calculate probability of choosing Option A (response == 0)
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1): p_a_t1 = 0.5
    if pd.isna(p_a_t3): p_a_t3 = 0.5
    
    # Metric: Difference in probability of choosing A between Trial 1 and Trial 3
    return float(p_a_t1 - p_a_t3)
```

**Observed (real) value:** -0.0175 (var=0.0255)
**Predicted under pi_6:** -0.3225 (var=0.1029)
**Predicted under pi_5:** -0.1987 (var=0.1125)

### Experiment 10
**Design**
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    t1_a = (0, 1, 1, 1, 0, 0, 0)
    t2_a = (1, 0, 0, 0, 1, 1, 1)
    
    is_t1 = a_tuples == t1_a
    is_t2 = a_tuples == t2_a
    
    t1_score = (data.loc[is_t1, 'response'] == 0).mean() if is_t1.any() else 0.0
    t2_score = (data.loc[is_t2, 'response'] == 1).mean() if is_t2.any() else 0.0
    
    return float(t1_score + t2_score)

```

**Observed (real) value:** 0.8050 (var=0.0264)
**Predicted under pi_6:** 1.1000 (var=0.1319)
**Predicted under pi_5:** 0.9350 (var=0.5647)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither theory demonstrates comprehensive task-invariance. Theory 1 (PCS, pi_6) performs remarkably well on Experiments 1-4, capturing the observed metrics where compensatory or tallying-like behavior is prevalent. However, Theory 2 (Sequential Evidence Accumulation, pi_5) provides better predictions for Experiments 5-10, where non-compensatory, sequential stopping dynamics seem to play a stronger role. The fact that both theories succeed in different subsets of the experimental domain suggests that human decision-makers might be employing a mechanism that blends these dynamics\u2014neither strictly parallel with complex recurrence (PCS) nor strictly sequential and deterministic in search order (Sequential Accumulator). Therefore, a new theory is warranted to bridge this gap.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Probabilistic Leaky Competing Accumulator' theory. Instead of a strict deterministic sequential search (like Theory 2) or a highly parameterized recurrent network (like Theory 1), this theory should posit that decision-makers probabilistically sample cues proportional to their validities. Evidence from sampled cues accumulates in competing accumulators (one for each option) subject to leakage and lateral inhibition. This allows the model to exhibit fast, non-compensatory choices when high-validity cues are sampled early and push the accumulator past a threshold, while also naturally producing compensatory, tally-like behavior when evidence accumulation is prolonged due to leakage or high thresholds. This stochastic sampling approach can better capture the variance and intermediate metrics observed across all experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 9483,
  "candidates_token_count": 341,
  "total_token_count": 10734
}
```
