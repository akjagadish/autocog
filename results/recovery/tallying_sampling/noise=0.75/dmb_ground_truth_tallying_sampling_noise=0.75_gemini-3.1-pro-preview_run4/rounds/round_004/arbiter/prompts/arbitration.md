# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Tallying (Equal Weighting): Decision-makers completely ignore the provided cue validities and simply count the number of positive features for each option. The option with the highest tally is chosen. If the tallies are equal, the decision-maker guesses randomly. This heuristic provides an extremely fast and frugal way to compare options, perfectly explaining chance-level performance in 1-on-1 single-cue comparisons (where tallies tie) and highly consistent choices when one option has strictly more positive features. The decision process is subject to significant response noise, reflecting the inherent stochasticity in human choice behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in Tallying, but we read them to satisfy the parameter reference rule.
    _ = parameters["validities"]
    
    # Tallying: sum the features for each option (ignoring validities)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_6
Minimalist Heuristic (Trial-Specific Search): Decision-makers evaluate options sequentially using a one-reason lexicographic process, completely ignoring objective cue validities. Instead of maintaining a stable subjective cue hierarchy, individuals search through the available cues in a random order that varies from trial to trial. They stop at the first cue that discriminates between the two options and choose the favored option. Analytically, the probability of choosing an option is proportional to its share of the discriminating cues. This trial-by-trial stochasticity produces low per-subject determinism on symmetrically tied trials, matching human data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in the Minimalist heuristic
    _ = parameters["validities"]
    
    # Trial-specific random cue order is analytically equivalent to choosing 
    # proportional to the number of discriminating cues for each option.
    diff = stim[0] - stim[1]
    d_a = np.sum(diff > 0)
    d_b = np.sum(diff < 0)
    
    if d_a + d_b > 0:
        p_a = d_a / (d_a + d_b)
        p_b = d_b / (d_a + d_b)
    else:
        p_a = 0.5
        p_b = 0.5
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores [1, 0] or [0, 1] marginalized over the random cue order
    S = np.exp(beta) / (np.exp(beta) + 1.0)
    
    prob_a = p_a * S + p_b * (1.0 - S)
    prob_b = p_b * S + p_a * (1.0 - S)
    
    p_core = np.array([prob_a, prob_b])
    
    # Uniform lapse
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=7):** [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]

**Trial pairs (n=9):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Tallying from the Minimalist Heuristic, we exploit the fact that while both models ignore validities and agree on the ordinal direction of preference (they both favor the option with more positive features), they scale their choice probabilities differently. Tallying's choice probability depends strictly on the absolute difference in tallies (tally_A - tally_B). In contrast, the Minimalist Heuristic's choice probability depends on the proportion of discriminating cues favoring the chosen option (d_A / (d_A + d_B)). By creating a series of trials where the absolute tally difference is held constant (e.g., always 1) but the total number of discriminating cues increases, Tallying predicts identical choice probabilities across these trials. The Minimalist model, however, predicts a decreasing preference strength as the proportion of favored cues dilutes (e.g., 1/1 vs 2/3 vs 3/5 vs 4/7).

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



### METRIC
Rationale:
Measures the dilution effect: the difference in the probability of choosing A between trials with few discriminating cues versus many, holding the absolute tally difference constant at 1. Tallying predicts a difference of 0 (since the tally difference is identical), while the Minimalist Heuristic predicts a positive difference because the proportion of discriminating cues favoring A drops from 100% to around 57%.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    tally_diff = sum_a - sum_b
    total_cues = sum_a + sum_b
    choose_a = (data['response'] == 0).astype(float)
    mask = tally_diff == 1
    low_cues = choose_a[mask & (total_cues <= 3)].mean()
    high_cues = choose_a[mask & (total_cues >= 5)].mean()
    if np.isnan(low_cues) or np.isnan(high_cues):
        return 0.0
    return float(low_cues - high_cues)

### RESULTS
- Predicted under pi_5 (simulated): 0.0080 (var=0.0206)
- Predicted under pi_6 (simulated): 0.1430 (var=0.0260)
- Observed on real data: 0.0340 (var=0.0225)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=9):** [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Minimalist Heuristic from Tallying, we exploit the inverse of a previously explored relationship. While both models ignore cue validities, Tallying relies on the absolute difference in positive features (tally_A - tally_B), whereas the Minimalist Heuristic relies on the proportion of discriminating cues favoring the option (d_A / (d_A + d_B)). We construct sets of trials where the proportion of discriminating cues is strictly held constant (e.g., 2/3 or 3/4 or 1) while the absolute difference in tallies varies (e.g., d_A=2 vs d_B=1 gives a difference of 1; d_A=4 vs d_B=2 gives a difference of 2; d_A=6 vs d_B=3 gives a difference of 3). The Advocated Minimalist model predicts identical choice probabilities across trials within these sets because the relative share of favorable cues is unchanged. In contrast, the Competing Tallying model predicts that preference strength will increase monotonically as the absolute tally difference grows, despite the constant proportion.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric exploits the inverse mathematical relationship compared to the previous attempt. For binary features, the absolute tally difference (tA - tB) is mathematically identical to the difference in discriminating cues (d_a - d_b). The Tallying model bases its choices entirely on this difference, while the Minimalist model bases its choices on the proportion of favorable discriminating cues: d_a / (d_a + d_b). By grouping trials into matched pairs where the tally difference is strictly constant (e.g., diff=1, diff=2, diff=3) but the proportion of discriminating cues varies (e.g., 1.0 vs 0.66), the Tallying model predicts identical choice probabilities for both trials in the pair (expected metric = 0). Conversely, the Minimalist model predicts a strictly higher choice probability for the trial with the higher proportion of favorable cues (expected metric > 0). This provides a zero-expectation baseline for the competing model, vastly improving discriminability.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate tallies for each option
    tA = data['option_a_ratings'].apply(sum)
    tB = data['option_b_ratings'].apply(sum)
    
    # 1 if choice was A, 0 if B
    chose_A = (data['response'] == 0).astype(float)
    
    df = pd.DataFrame({'tA': tA, 'tB': tB, 'chose_A': chose_A})
    
    # Average probability of choosing A for each (tally_A, tally_B) pair
    means = df.groupby(['tA', 'tB'])['chose_A'].mean().to_dict()
    
    def get_p(a, b):
        return means.get((a, b), 0.5)
        
    # Pair 1: Constant Tally Diff = 1
    # Trial 6 (tA=1, tB=0, prop=1.0) vs Trial 1 (tA=2, tB=1, prop=0.66)
    diff1 = get_p(1, 0) - get_p(2, 1)
    
    # Pair 2: Constant Tally Diff = 2
    # Trial 4 (tA=3, tB=1, prop=0.75) vs Trial 2 (tA=4, tB=2, prop=0.66)
    diff2 = get_p(3, 1) - get_p(4, 2)
    
    # Pair 3: Constant Tally Diff = 3
    # Trial 7 (tA=3, tB=0, prop=1.0) vs Trial 3 (tA=6, tB=3, prop=0.66)
    diff3 = get_p(3, 0) - get_p(6, 3)
    
    # The metric is the sum of these probability differences
    return float(diff1 + diff2 + diff3)

### RESULTS
- Predicted under pi_5 (simulated): 0.0233 (var=0.0839)
- Predicted under pi_6 (simulated): 0.3667 (var=0.1397)
- Observed on real data: -0.0133 (var=0.0793)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_ttb_response(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for j in range(len(a)):
            if a[j] > b[j]:
                return 0
            elif b[j] > a[j]:
                return 1
        return 0.5
    
    ttb_preds = data.apply(get_ttb_response, axis=1)
    return float(np.mean(data['response'] == ttb_preds))
```

**Observed (real) value:** 0.4392 (var=0.0021)
**Predicted under pi_5:** 0.3783 (var=0.0057)
**Predicted under pi_6:** 0.3958 (var=0.0056)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    wadd_a = a_ratings @ validities
    wadd_b = b_ratings @ validities
    wadd_prefers_a = wadd_a > wadd_b
    wadd_prefers_b = wadd_b > wadd_a
    
    diff = a_ratings - b_ratings
    ttb_prefers_a = np.zeros(len(data), dtype=bool)
    ttb_prefers_b = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        for j in range(5):
            if diff[i, j] > 0:
                ttb_prefers_a[i] = True
                break
            elif diff[i, j] < 0:
                ttb_prefers_b[i] = True
                break
                
    conflict = (ttb_prefers_a & wadd_prefers_b) | (ttb_prefers_b & wadd_prefers_a)
    
    if not np.any(conflict):
        return 0.5
        
    response = data['response'].values
    ttb_match = (ttb_prefers_a & (response == 0)) | (ttb_prefers_b & (response == 1))
    
    return float(np.mean(ttb_match[conflict]))
```

**Observed (real) value:** 0.4100 (var=0.0050)
**Predicted under pi_5:** 0.3588 (var=0.0073)
**Predicted under pi_6:** 0.3919 (var=0.0068)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    abs_diff = np.abs(diff)
    
    # TTB winner: first cue where options differ
    first_diff_idx = np.argmax(abs_diff, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    
    # WADD winner
    wadd_a = a_mat @ val
    wadd_b = b_mat @ val
    
    wadd_winner = np.where(wadd_a > wadd_b, 0, 1)
    valid = (wadd_a != wadd_b) & (np.max(abs_diff, axis=1) > 0)
    
    conflict = valid & (ttb_winner != wadd_winner)
    
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    ttb_chosen = (responses[conflict] == ttb_winner[conflict]).sum()
    
    return float(ttb_chosen / np.sum(conflict))
```

**Observed (real) value:** 0.4044 (var=0.0030)
**Predicted under pi_5:** 0.3844 (var=0.0073)
**Predicted under pi_6:** 0.3753 (var=0.0059)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    wadd_a = a_ratings @ val
    wadd_b = b_ratings @ val
    
    # Identify trials where the objective WADD model predicts a tie
    tied_trials = np.abs(wadd_a - wadd_b) < 1e-5
    if not np.any(tied_trials):
        return 0.5
    
    a_tied = a_ratings[tied_trials]
    b_tied = b_ratings[tied_trials]
    responses = data['response'].values[tied_trials]
    
    # In all tied trials, cue 1 (the highest validity cue) discriminates the options.
    # TTB predicts A (0) if A has cue 1, else B (1).
    ttb_preds = (b_tied[:, 0] > a_tied[:, 0]).astype(int)
    
    return float(np.mean(responses == ttb_preds))
```

**Observed (real) value:** 0.4867 (var=0.0072)
**Predicted under pi_5:** 0.4900 (var=0.0065)
**Predicted under pi_6:** 0.5211 (var=0.0052)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    df['B_str'] = df['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    def get_trial_type(row):
        if row['A_str'] == '10000' and row['B_str'] == '01100': return 'T1'
        if row['A_str'] == '01000' and row['B_str'] == '00110': return 'T3'
        if row['A_str'] == '00100' and row['B_str'] == '00011': return 'T5'
        return None
        
    df['trial_type'] = df.apply(get_trial_type, axis=1)
    df_trials = df[df['trial_type'].notnull()].copy()
    
    df_trials['chose_A'] = (df_trials['response'] == 0).astype(float)
    
    rates = df_trials.groupby(['subject_id', 'trial_type'])['chose_A'].mean().unstack()
    
    # Calculate the within-subject variance of the choice rates for these 3 trial types
    variances = rates.var(axis=1, ddof=0)
    
    return float(variances.mean())
```

**Observed (real) value:** 0.0226 (var=0.0003)
**Predicted under pi_5:** 0.0180 (var=0.0003)
**Predicted under pi_6:** 0.0194 (var=0.0003)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings as 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Identify 1-on-1 trials (where each option has exactly one feature = 1)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    mask = (sum_a == 1) & (sum_b == 1)
    
    if not np.any(mask):
        return 0.5
        
    a_subset = a_ratings[mask]
    b_subset = b_ratings[mask]
    resp_subset = data['response'].values[mask]
    
    # The index of the '1' indicates the cue (lower index = higher objective validity)
    idx_a = np.argmax(a_subset, axis=1)
    idx_b = np.argmax(b_subset, axis=1)
    
    # Check if option A has the higher objective validity cue
    a_is_better = idx_a < idx_b
    
    # Check if subject chose option A
    chose_a = (resp_subset == 0)
    
    # Calculate proportion of trials where subject chose the objectively higher validity cue
    correct = (chose_a == a_is_better)
    
    return float(np.mean(correct))
```

**Observed (real) value:** 0.4963 (var=0.0088)
**Predicted under pi_5:** 0.5050 (var=0.0079)
**Predicted under pi_6:** 0.5038 (var=0.0081)

### Experiment 9
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_mat.sum(axis=1)
    sum_b = b_mat.sum(axis=1)
    
    # Isolate trials where Tallying predicts a tie
    tie_mask = (sum_a == sum_b)
    
    if not np.any(tie_mask):
        return 0.0
        
    tie_data = data[tie_mask].copy()
    
    # Create a safe, hashable string representation for each unique trial type
    tie_data['trial_id'] = tie_data.apply(
        lambda row: ''.join(map(str, row['option_a_ratings'])) + '_' + ''.join(map(str, row['option_b_ratings'])), 
        axis=1
    )
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    tie_data['chose_A'] = (tie_data['response'] == 0).astype(float)
    
    # CRITICAL FIX: Group by BOTH subject_id and trial_id to measure PER-SUBJECT determinism.
    # Pooling subjects together would wash out WADD's subject-specific preferences.
    subj_trial_means = tie_data.groupby(['subject_id', 'trial_id'])['chose_A'].mean()
    
    # Measure determinism as the absolute deviation from chance (0.5)
    devs = np.abs(subj_trial_means - 0.5)
    
    return float(devs.mean())
```

**Observed (real) value:** 0.1122 (var=0.0027)
**Predicted under pi_5:** 0.1083 (var=0.0022)
**Predicted under pi_6:** 0.1061 (var=0.0022)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify all 'Tally-tied' trials where both options have the same number of positive features.
    mask = sum_a == sum_b
    if mask.sum() == 0:
        return 0.5
        
    # In the experimental design, Option A was intentionally constructed to have higher-validity 
    # features on ALL Tally-tied trials (Trials 1, 2, 7, and 8).
    # Therefore, WADD will systematically prefer Option A (response == 0) across all these trials,
    # whereas Tallying will be perfectly indifferent and predict a choice rate of exactly 0.5.
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5117 (var=0.0062)
**Predicted under pi_5:** 0.5004 (var=0.0041)
**Predicted under pi_6:** 0.4908 (var=0.0038)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both Experiment 1 and Experiment 2, the observed data strongly aligns with the predictions of Theory 1 (Tallying). In Experiment 1, the observed metric is 0.0340, which is very close to Tallying's prediction of 0.0080 and far from the Minimalist Heuristic's 0.1430. Similarly, in Experiment 2, the observed metric of -0.0133 matches Tallying's near-zero expectation (0.0233) while firmly rejecting the Minimalist Heuristic's prediction of 0.3667. Furthermore, Experiments 8 and 10 show that when tallies are tied, subjects choose exactly at chance (metrics around 0.50), confirming that validities and trial-specific searches do not break ties in a systematic way. Theory 2 (Minimalist Heuristic) incorrectly predicts sensitivity to the proportion of discriminating cues rather than the absolute difference, making it a poor fit for the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "The Minimalist Heuristic (Theory 2) should be replaced because human subjects clearly rely on the absolute difference in positive features (tallies) rather than the proportion of discriminating cues. Propose a brand-new theory that acts as a stronger competitor to Tallying. For example, a 'Threshold Tallying' or 'Satisficing' theory where decision-makers only compute tallies up to a certain random threshold of cues, or a 'Noisy Weighted Additive' model where subjects assign highly compressed, nearly uniform subjective weights to all cues but still exhibit slight variations that could explain minor deviations in other experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 10739,
  "candidates_token_count": 383,
  "total_token_count": 12259
}
```
