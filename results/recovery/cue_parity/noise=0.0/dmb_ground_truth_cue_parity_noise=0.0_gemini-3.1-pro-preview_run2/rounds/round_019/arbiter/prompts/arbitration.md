# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_16" and "pi_21") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_16" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_21" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_16
Leaky Competing Accumulator with Non-linear Configural Weighting and Evidence Transduction: Decision-makers evaluate cues simultaneously, with evidence for each option dynamically inhibiting the other in a leaky competing accumulator (LCA). Cues have a configural impact, scaled non-linearly by the total number of supporting cues. Additionally, the integrated evidence for each option is passed through a non-linear transducer (alpha) before entering the accumulation process, allowing the model to amplify the differences driven by high-validity cues and capture strong non-compensatory reversals.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Configural weighting: effective validity of a cue is non-linearly modulated by the total number of cues
    # Using max(1e-6, sum) to avoid 0^negative_gamma undefined errors
    sum_a_safe = max(1e-6, sum_a)
    sum_b_safe = max(1e-6, sum_b)
    
    w_a = val * (sum_a_safe ** gamma)
    w_b = val * (sum_b_safe ** gamma)
    
    # Make sure inputs are non-negative and apply non-linear transducer alpha
    I_A = max(0.0, np.sum(w_a * a)) ** alpha
    I_B = max(0.0, np.sum(w_b * b)) ** alpha
    
    # Leaky Competing Accumulator (LCA) simulation
    x_a, x_b = 0.0, 0.0
    dt = 0.1
    steps = 100
    
    for _ in range(steps):
        dx_a = (I_A - leak * x_a - inhibition * x_b) * dt
        dx_b = (I_B - leak * x_b - inhibition * x_a) * dt
        
        x_a = max(0.0, x_a + dx_a)
        x_b = max(0.0, x_b + dx_b)
        
    # Softmax choice based on final activations
    z = theta * np.array([x_a, x_b])
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_21
Sequential Cue Sampling with Conflict-Modulated Threshold: Decision-makers sample cues sequentially in order of their validities. Instead of integrating all cues simultaneously, they accumulate evidence until it reaches a decision threshold. Crucially, the level of conflict between the options (inversely related to the absolute tally difference) dynamically modulates this threshold. Under low conflict, the threshold is low, leading to fast, non-compensatory decisions (like Take-The-Best). Under high conflict, the threshold increases, enforcing extensive sampling and yielding compensatory behavior. This provides a natural speed-accuracy tradeoff without mathematical artifacts.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta_0 = float(parameters["theta_0"])
    theta_1 = float(parameters["theta_1"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate conflict based on the absolute tally difference of discriminating cues
    N_a = np.sum((a > b).astype(int))
    N_b = np.sum((b > a).astype(int))
    tally_diff = abs(N_a - N_b)
    
    # High tally difference means low conflict; zero tally difference means maximal conflict
    conflict = 1.0 / (1.0 + tally_diff)
    
    # Decision threshold is modulated by the level of conflict
    theta = theta_0 + theta_1 * conflict
    
    # Sequential sampling of cues prioritized by validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        E += (a[j] - b[j]) * val[j]
        if abs(E) >= theta:
            break
            
    # Softmax choice probability based on accumulated evidence
    z = beta * np.array([E, -E])
    z = z - np.max(z)
    p = np.exp(z) / np.sum(np.exp(z))
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_16)

### DESIGN
**Validities (n_features=8):** [0.95, 0.85, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 1]
  trial 8: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  trial 9: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  trial 12: A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]

**Rationale:** This design quantitatively dissociates the advocated Leaky Competing Accumulator (LCA) theory with Configural Weighting from the competing Sequential Cue Sampling theory by exploiting the latter's invariance to disjoint, equal-validity cues. The competing theory evaluates cues sequentially and stops when a conflict-modulated threshold is reached. If Option A and Option B are given additional disjoint cues of identical validity, the sequential accumulator's evidence will temporarily increase and then immediately decrease by the exact same amount. As long as this intermediate fluctuation does not exceed the maximum evidence already reached by the highest-validity cue, the probability of early stopping and the final accumulated evidence remain mathematically identical. Furthermore, because the added cues are disjoint and equal in number, the tally difference remains constant, keeping the conflict-modulated threshold unchanged. Thus, the competing theory predicts perfectly invariant choice probabilities across these trials. The advocated LCA theory, however, applies a configural weight to all cues scaled by the total number of cues favoring each option (sum ** gamma). Adding disjoint cues increases this total sum for both options, non-linearly scaling the effective evidence and predicting systematic, graded shifts in choice probabilities.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
For trials 1 through 8, the highest-validity cue always favors Option A, and the second-highest always favors Option B. The remaining cues are disjoint but perfectly matched in validity. According to the competing Sequential Cue Sampling theory, the identical peak evidence and identical conflict (tally difference is always 0) mean the decision threshold and stopping probabilities are mathematically identical across all these trials. Therefore, the choice probability P(Choose A) should be perfectly invariant, meaning the extremity of preference |P - 0.5| is constant, yielding a standard deviation of zero (plus minimal binomial noise). In contrast, the advocated LCA theory with Configural Weighting predicts that the total number of cues non-linearly scales the evidence. This causes the extremity of preference to systematically increase or decrease (depending on the sign of gamma) as the number of cues grows from 1 to 4. By taking the standard deviation of the extremity across cue levels per subject, we capture this systematic shift while naturally folding the symmetric variance caused by gamma, yielding a tight, highly discriminative metric.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    # Identify trials 1-8: Option A's first cue is 1
    data['A_0'] = data['option_a_ratings'].apply(lambda x: x[0])
    trials_1_8 = data[data['A_0'] == 1].copy()
    
    # Number of cues present for Option A (which is equal to Option B in these trials)
    trials_1_8['num_cues'] = trials_1_8['option_a_ratings'].apply(sum)
    
    def subj_metric(df):
        # Calculate the extremity of preference (absolute deviation from 0.5) for each cue level
        means = df.groupby('num_cues')['response'].mean()
        extremities = means.apply(lambda x: abs(x - 0.5))
        # Return the standard deviation of these extremities
        return float(extremities.std())
        
    subj_stds = trials_1_8.groupby('subject_id').apply(subj_metric)
    return float(subj_stds.mean())

### RESULTS
- Predicted under pi_16 (simulated): 0.1338 (var=0.0040)
- Predicted under pi_21 (simulated): 0.0901 (var=0.0012)
- Observed on real data: 0.0772 (var=0.0021)

## EXPERIMENT 2 (proposed by pi_21)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=11):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 11: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Sequential Cue Sampling theory from the competing Leaky Competing Accumulator (LCA) theory by exploiting their fundamentally divergent treatment of tied cues. The advocated theory calculates the conflict-modulated threshold based strictly on the tally difference of *discriminating* cues, and evaluates evidence sequentially as the difference between options. Consequently, adding tied cues (where both options have a 1) neither changes the threshold nor the accumulated evidence, leading the advocated theory to predict absolutely identical choice probabilities across sets of trials where tied cues are systematically added. The competing LCA theory, however, applies a configural weight to all validities based on the *total* number of cues favoring each option (sum ** gamma). Adding tied cues increases this total sum for both options, which non-linearly scales the effective weights of the discriminating cues. Because this scaling is non-linear, it disproportionately affects the option with fewer initial cues, predicting a systematic, graded shift in choice probabilities as tied cues are added. By presenting base trials with fixed discriminating cues and systematically adding tied cues, we can cleanly test for configural tied-cue integration (LCA) versus tied-cue invariance (Sequential Sampling).

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



### METRIC
Rationale:
This metric isolates the core divergence between the advocated Sequential Cue Sampling (SCS) theory and the competing Leaky Competing Accumulator (LCA) theory: the effect of tied cues. SCS evaluates evidence based on the tally difference of *discriminating* cues, ignoring tied cues entirely. Thus, the choice probabilities for a given set of discriminating cues should remain identical regardless of how many tied cues are added. LCA, on the other hand, configuraly scales cue validities based on the total number of cues supporting each option, meaning that adding tied cues systematically alters the choice probabilities. By grouping trials into 'base pairs' (identical discriminating cues) and calculating the absolute difference in the probability of choosing Option A between the trial with the maximum tied cues and the trial with the minimum tied cues, we directly quantify this shift. SCS predicts a difference of 0 (subject only to binomial sampling noise), whereas LCA predicts a large, systematic shift.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_base_pair(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        disc_a = tuple(1 if a[i]==1 and b[i]==0 else 0 for i in range(len(a)))
        disc_b = tuple(1 if b[i]==1 and a[i]==0 else 0 for i in range(len(b)))
        return str((disc_a, disc_b))
        
    data['base_pair'] = data.apply(get_base_pair, axis=1)
    data['tied_count'] = data.apply(lambda row: sum(1 for i in range(len(row['option_a_ratings'])) if row['option_a_ratings'][i]==1 and row['option_b_ratings'][i]==1), axis=1)
    
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    grouped = data.groupby(['subject_id', 'base_pair', 'tied_count'])['chose_A'].mean().reset_index()
    
    diffs = []
    for (subj, bp), group in grouped.groupby(['subject_id', 'base_pair']):
        if len(group) > 1:
            min_tied = group['tied_count'].min()
            max_tied = group['tied_count'].max()
            p_min = group.loc[group['tied_count'] == min_tied, 'chose_A'].values[0]
            p_max = group.loc[group['tied_count'] == max_tied, 'chose_A'].values[0]
            diffs.append(abs(p_max - p_min))
            
    return float(np.mean(diffs)) if diffs else 0.0

### RESULTS
- Predicted under pi_16 (simulated): 0.2433 (var=0.0266)
- Predicted under pi_21 (simulated): 0.1050 (var=0.0054)
- Observed on real data: 0.1350 (var=0.0064)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner is not None:
            if resp == ttb_winner:
                ttb_matches += 1
            total += 1
            
    return float(ttb_matches / total) if total > 0 else 0.5

```

**Observed (real) value:** 0.7417 (var=0.0032)
**Predicted under pi_16:** 0.4573 (var=0.0351)
**Predicted under pi_21:** 0.3156 (var=0.0137)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

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
        
        if a_wins > b_wins:
            pred = 0
        elif b_wins > a_wins:
            pred = 1
        else:
            continue
            
        matches.append(row['response'] == pred)
        
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.2881 (var=0.0100)
**Predicted under pi_16:** 0.5750 (var=0.1191)
**Predicted under pi_21:** 0.8252 (var=0.0180)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_winner(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1

    extreme_comp = []
    extreme_noncomp = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_win = get_ttb_winner(a, b)
        if ttb_win == -1:
            continue
            
        ttb_chosen = 1 if resp == ttb_win else 0
        
        # Tallying margin for the TTB winner
        margin = sum(a) - sum(b) if ttb_win == 0 else sum(b) - sum(a)
            
        # Extreme compensatory trials (TTB winner is opposed by many cues)
        if margin <= -2:
            extreme_comp.append(ttb_chosen)
        # Non-compensatory trials (TTB winner is supported by overall cue sum)
        elif margin >= 1:
            extreme_noncomp.append(ttb_chosen)
            
    if not extreme_comp or not extreme_noncomp:
        return 0.0
        
    return float(np.mean(extreme_noncomp) - np.mean(extreme_comp))
```

**Observed (real) value:** 0.1389 (var=0.0317)
**Predicted under pi_16:** 0.3928 (var=0.2999)
**Predicted under pi_21:** 0.6739 (var=0.0521)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_p_a(a_list, b_list):
        a_tup = tuple(a_list)
        b_tup = tuple(b_list)
        mask = (data['option_a_ratings'].apply(tuple) == a_tup) & (data['option_b_ratings'].apply(tuple) == b_tup)
        if mask.sum() == 0:
            return 0.5
        return 1.0 - data.loc[mask, 'response'].mean()

    # Trial 1: highest validity cue favors A, lowest validity cue favors B
    p_a_t1 = get_p_a([1, 0, 0, 0, 0], [0, 0, 0, 0, 1])
    
    # Trial 4: 4th validity cue favors A, lowest validity cue favors B
    p_a_t4 = get_p_a([0, 0, 0, 1, 0], [0, 0, 0, 0, 1])
    
    return float(p_a_t1 - p_a_t4)
```

**Observed (real) value:** 0.0000 (var=0.0233)
**Predicted under pi_16:** 0.0600 (var=0.0228)
**Predicted under pi_21:** 0.1150 (var=0.0283)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_first_diff(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i, (val_a, val_b) in enumerate(zip(a, b)):
            if val_a != val_b:
                return i
        return -1

    # Calculate which cue is the highest valid discriminating cue
    first_diffs = data.apply(get_first_diff, axis=1)
    
    # Probability of choosing Option A
    chose_a = 1 - data['response']
    
    # Mean P(Choose A) when Cue 1 (index 0) is the highest discriminating cue
    mask_cue1 = (first_diffs == 0)
    p_a_cue1 = chose_a[mask_cue1].mean() if mask_cue1.sum() > 0 else 0.0
    
    # Mean P(Choose A) when Cue 3 (index 2) is the highest discriminating cue
    mask_cue3 = (first_diffs == 2)
    p_a_cue3 = chose_a[mask_cue3].mean() if mask_cue3.sum() > 0 else 0.0
    
    # The metric is the difference in P(Choose A) between Cue 1 and Cue 3 trials.
    return float(p_a_cue1 - p_a_cue3)
```

**Observed (real) value:** -0.4333 (var=0.0244)
**Predicted under pi_16:** -0.1993 (var=0.0126)
**Predicted under pi_21:** -0.0873 (var=0.0119)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify if option A has the highest validity cue (feature 0)
    # Feature 0 has validity 0.95, so Take The Best (TTB) always favors the option with a 1 here.
    a_has_f0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    
    # Identify if the subject chose A or B
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    # Check if the chosen option is the one with feature 0 (the TTB winner)
    ttb_chosen = (a_has_f0 & chose_a) | (~a_has_f0 & chose_b)
    
    # Return the overall proportion of choosing the TTB winner
    return float(ttb_chosen.mean())
```

**Observed (real) value:** 0.4946 (var=0.0018)
**Predicted under pi_16:** 0.6073 (var=0.0105)
**Predicted under pi_21:** 0.6742 (var=0.0061)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['a_tup'] = data['option_a_ratings'].apply(tuple)
    data['b_tup'] = data['option_b_ratings'].apply(tuple)
    
    target_trials = [
        ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
        ((1, 0, 0, 1, 0), (0, 1, 1, 0, 0)),
        ((1, 0, 1, 0, 0), (0, 1, 0, 1, 0)),
        ((1, 0, 0, 0, 1), (0, 1, 0, 1, 0)),
        ((0, 1, 1, 0, 0), (1, 0, 0, 0, 1)),
        ((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))
    ]
    
    mad_sum = 0.0
    count = 0
    for a_tup, b_tup in target_trials:
        mask = (data['a_tup'] == a_tup) & (data['b_tup'] == b_tup)
        if mask.sum() > 0:
            p_a = 1.0 - data[mask]['response'].mean()
            mad_sum += abs(p_a - 0.5)
            count += 1
            
    return float(mad_sum / count) if count > 0 else 0.0
```

**Observed (real) value:** 0.3633 (var=0.0090)
**Predicted under pi_16:** 0.1622 (var=0.0062)
**Predicted under pi_21:** 0.1547 (var=0.0076)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Convert ratings to hashable tuples
    t_keys = data['option_a_ratings'].apply(tuple)
    
    # Focus on the highly conflicting trials where SCA is likely to fail to reach its evidence threshold
    t6 = (1, 0, 1, 1, 0, 0)
    t7 = (1, 0, 1, 0, 0, 0)
    t8 = (1, 0, 0, 0, 0, 0)
    
    mask = t_keys.isin([t6, t7, t8])
    if not mask.any():
        return 0.0
        
    # Calculate the mean response for each of these trial types
    subset = data[mask]
    t_keys_subset = t_keys[mask]
    means = subset.groupby(t_keys_subset)['response'].mean()
    
    # Calculate 'determinism': how far the mean response is from 0.5 (random guessing)
    determinism = (means - 0.5).abs().mean()
    
    return float(determinism)
```

**Observed (real) value:** 0.3389 (var=0.0140)
**Predicted under pi_16:** 0.1267 (var=0.0151)
**Predicted under pi_21:** 0.3256 (var=0.0067)

### Experiment 11
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Calculate tallies (sum of positive cues) for each option
    a_tallies = data['option_a_ratings'].apply(lambda x: sum(x))
    b_tallies = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Filter for trials where the tallies are unequal (trials 1, 2, 6, 7)
    unequal_mask = a_tallies != b_tallies
    if not unequal_mask.any():
        return 0.5
        
    filtered_data = data[unequal_mask]
    a_tally_filtered = a_tallies[unequal_mask]
    b_tally_filtered = b_tallies[unequal_mask]
    
    # Determine which option has the higher tally (0 for A, 1 for B)
    higher_tally_option = (b_tally_filtered > a_tally_filtered).astype(int)
    
    # Calculate the proportion of choices that match the higher tally option
    match = (filtered_data['response'] == higher_tally_option).mean()
    
    return float(match)
```

**Observed (real) value:** 0.8892 (var=0.0052)
**Predicted under pi_16:** 0.6842 (var=0.0493)
**Predicted under pi_21:** 0.7738 (var=0.0407)

### Experiment 12
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    # Trial 4: A=[1, 0, 0, 0, 1], B=[0, 1, 1, 0, 0]
    mask = (
        ((a_str == '00111') & (b_str == '11000')) |
        ((a_str == '10001') & (b_str == '01100'))
    )
    
    if mask.sum() == 0:
        return 0.0
    
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4900 (var=0.0032)
**Predicted under pi_16:** 0.5425 (var=0.0144)
**Predicted under pi_21:** 0.5308 (var=0.0221)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_key(row):
        a = tuple(int(x) for x in row['option_a_ratings'])
        b = tuple(int(x) for x in row['option_b_ratings'])
        return str(a) + str(b)
    
    data['trial_str'] = data.apply(get_key, axis=1)
    
    t5 = str((1, 0, 1, 0, 1, 0)) + str((0, 1, 0, 0, 0, 0))
    t6 = str((0, 0, 1, 1, 1, 0)) + str((0, 0, 0, 0, 0, 1))
    
    t7 = str((1, 0, 1, 1, 1, 0)) + str((0, 1, 0, 0, 0, 0))
    t8 = str((0, 0, 1, 1, 1, 1)) + str((1, 0, 0, 0, 0, 0))
    
    t9 = str((1, 0, 1, 1, 1, 1)) + str((0, 1, 0, 0, 0, 0))
    t10 = str((0, 1, 1, 1, 1, 1)) + str((1, 0, 0, 0, 0, 0))
    
    data['chose_A'] = (data['response'] == 0).astype(float)
    subj_trial = data.groupby(['subject_id', 'trial_str'])['chose_A'].mean().unstack()
    
    diff2 = (subj_trial[t5] - subj_trial[t6]).abs().mean() if t5 in subj_trial and t6 in subj_trial else 0
    diff3 = (subj_trial[t7] - subj_trial[t8]).abs().mean() if t7 in subj_trial and t8 in subj_trial else 0
    diff4 = (subj_trial[t9] - subj_trial[t10]).abs().mean() if t9 in subj_trial and t10 in subj_trial else 0
    
    return float(diff2 + diff3 + diff4)
```

**Observed (real) value:** 0.3556 (var=0.0405)
**Predicted under pi_16:** 0.3667 (var=0.0426)
**Predicted under pi_21:** 0.5511 (var=0.2118)

### Experiment 14
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract whether Option B has the most valid feature (index 0)
    b_has_best_cue = data['option_b_ratings'].apply(lambda x: x[0])
    
    # response == 0 means Option A was chosen
    chose_a = 1.0 - data['response']
    
    # Calculate the choice probability for A when B does NOT have the best cue
    p_a_when_b_lacks_best = chose_a[b_has_best_cue == 0].mean()
    
    # Calculate the choice probability for A when B HAS the best cue
    p_a_when_b_has_best = chose_a[b_has_best_cue == 1].mean()
    
    # The difference reflects sensitivity to cue validities
    return float(p_a_when_b_lacks_best - p_a_when_b_has_best)
```

**Observed (real) value:** 0.0067 (var=0.0065)
**Predicted under pi_16:** 0.1050 (var=0.0357)
**Predicted under pi_21:** 0.1204 (var=0.0404)

### Experiment 15
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify Trial 1 where Option A has lower tally (2 vs 3) but higher validity under both TTB and WADD
    is_t1 = (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1))
    if not is_t1.any():
        return 0.0
    # response == 0 means Option A was chosen
    return float((data.loc[is_t1, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1240 (var=0.0162)
**Predicted under pi_16:** 0.3700 (var=0.0849)
**Predicted under pi_21:** 0.8200 (var=0.0228)

### Experiment 16
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    # Focus purely on trials where the tally difference is exactly 1
    mask = abs(tally_a - tally_b) == 1
    if not mask.any():
        return 0.5
        
    df = data[mask]
    tally_a_filtered = tally_a[mask]
    tally_b_filtered = tally_b[mask]
    
    # Check if the subject chose the option with the higher tally
    chose_a = df['response'] == 0
    higher_tally_a = tally_a_filtered > tally_b_filtered
    
    chose_higher = (chose_a == higher_tally_a)
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.2000 (var=0.0131)
**Predicted under pi_16:** 0.5883 (var=0.1241)
**Predicted under pi_21:** 0.8375 (var=0.0386)

### Experiment 17
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    mask = (tally_a == 2) & (tally_b == 3)
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8633 (var=0.0110)
**Predicted under pi_16:** 0.6267 (var=0.0842)
**Predicted under pi_21:** 0.6900 (var=0.0886)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tallies = data['option_a_ratings'].apply(sum)
    b_tallies = data['option_b_ratings'].apply(sum)
    
    unequal_mask = a_tallies != b_tallies
    if not unequal_mask.any():
        return 0.5
        
    unequal_data = data[unequal_mask]
    a_higher = a_tallies[unequal_mask] > b_tallies[unequal_mask]
    b_higher = b_tallies[unequal_mask] > a_tallies[unequal_mask]
    
    chose_a = unequal_data['response'] == 0
    chose_b = unequal_data['response'] == 1
    
    matched = (a_higher & chose_a) | (b_higher & chose_b)
    
    return float(matched.mean())
```

**Observed (real) value:** 0.6222 (var=0.0019)
**Predicted under pi_16:** 0.6447 (var=0.0651)
**Predicted under pi_21:** 0.7300 (var=0.0418)

### Experiment 19
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    mask = abs(tally_a - tally_b) == 1
    if not mask.any():
        return 0.0
        
    lower_tally_is_b = (tally_b[mask] < tally_a[mask]).astype(int)
    responses = data.loc[mask, 'response']
    
    chose_lower_tally = (responses == lower_tally_is_b).mean()
    return float(chose_lower_tally)
```

**Observed (real) value:** 0.5092 (var=0.0035)
**Predicted under pi_16:** 0.3846 (var=0.0860)
**Predicted under pi_21:** 0.1596 (var=0.0193)

### Experiment 20
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    diff_1_mask = abs(tally_a - tally_b) == 1
    subset = data[diff_1_mask]
    
    if len(subset) == 0:
        return 0.5
        
    chose_higher_tally = ((tally_a[diff_1_mask] > tally_b[diff_1_mask]) & (subset['response'] == 0)) | ((tally_b[diff_1_mask] > tally_a[diff_1_mask]) & (subset['response'] == 1))
    
    return float(chose_higher_tally.mean())
```

**Observed (real) value:** 0.6858 (var=0.0032)
**Predicted under pi_16:** 0.5667 (var=0.0670)
**Predicted under pi_21:** 0.7321 (var=0.0457)

### Experiment 21
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    diff = (tally_a - tally_b).abs()
    
    mask = diff >= 2
    if not mask.any():
        return 0.5
    
    subset = data[mask]
    ta = tally_a[mask]
    tb = tally_b[mask]
    resp = subset['response']
    
    chose_higher_tally = ((ta > tb) & (resp == 0)) | ((tb > ta) & (resp == 1))
    
    return float(chose_higher_tally.mean())
```

**Observed (real) value:** 0.3050 (var=0.0041)
**Predicted under pi_16:** 0.4083 (var=0.1237)
**Predicted under pi_21:** 0.7754 (var=0.0613)

### Experiment 22
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t2_mask = (a_str == '10000') & (b_str == '01110')
    t3_mask = (a_str == '10000') & (b_str == '01111')
    t7_mask = (a_str == '01110') & (b_str == '10000')
    
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_a_t7 = (data.loc[t7_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t2): p_a_t2 = 0.5
    if pd.isna(p_a_t3): p_a_t3 = 0.5
    if pd.isna(p_a_t7): p_a_t7 = 0.5
    
    return float(p_a_t2 + p_a_t3 - p_a_t7)
```

**Observed (real) value:** 0.8700 (var=0.0323)
**Predicted under pi_16:** 0.6450 (var=1.2247)
**Predicted under pi_21:** -0.3333 (var=0.7522)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    t10_mask = (a_keys == (0, 1, 1, 0, 0)) & (b_keys == (1, 0, 0, 1, 1))
    t5_mask = (a_keys == (1, 0, 0, 0, 0)) & (b_keys == (0, 1, 0, 0, 0))
    
    p_a_t10 = (data.loc[t10_mask, 'response'] == 0).mean()
    p_a_t5 = (data.loc[t5_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t10): p_a_t10 = 0.5
    if pd.isna(p_a_t5): p_a_t5 = 0.5
    
    return float(p_a_t10 - p_a_t5)
```

**Observed (real) value:** -0.6578 (var=0.0661)
**Predicted under pi_16:** -0.4867 (var=0.1140)
**Predicted under pi_21:** -0.5889 (var=0.0540)

### Experiment 24
**Design**
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where cue 1 and cue 2 conflict
    conflict = (a_ratings[:, 0] != b_ratings[:, 0]) & \
               (a_ratings[:, 1] != b_ratings[:, 1]) & \
               (a_ratings[:, 0] != a_ratings[:, 1])
               
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    
    a_has_cue1 = a_ratings[:, 0] > b_ratings[:, 0]
    b_has_cue1 = b_ratings[:, 0] > a_ratings[:, 0]
    
    # Check if the chosen option is the one favored by cue 1
    chose_cue1 = (a_has_cue1 & (responses == 0)) | (b_has_cue1 & (responses == 1))
    
    return float(np.mean(chose_cue1[conflict]))
```

**Observed (real) value:** 0.6108 (var=0.0044)
**Predicted under pi_16:** 0.5713 (var=0.0096)
**Predicted under pi_21:** 0.6500 (var=0.0111)

### Experiment 25
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 5 where Option A has the single best cue and Option B has all other 5 cues.
    t5_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 1, 1, 1))
    
    if t5_mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen in this specific trial
    return float((data.loc[t5_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.8600 (var=0.0198)
**Predicted under pi_16:** 0.4250 (var=0.1701)
**Predicted under pi_21:** 0.2033 (var=0.0600)

### Experiment 26
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.array(data['option_a_ratings'].tolist())
    b = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    sum_a = a.sum(axis=1)
    sum_b = b.sum(axis=1)
    
    # HVO (High Validity Option) is the one with the best cue (index 0)
    # We filter for trials where the HVO is outnumbered in simple tally
    cond_a = (a[:, 0] == 1) & (b[:, 0] == 0) & (sum_a < sum_b)
    cond_b = (b[:, 0] == 1) & (a[:, 0] == 0) & (sum_b < sum_a)
    
    hvo_chosen = np.sum((cond_a & (resp == 0)) | (cond_b & (resp == 1)))
    total = np.sum(cond_a | cond_b)
    
    return float(hvo_chosen / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5813 (var=0.0017)
**Predicted under pi_16:** 0.4517 (var=0.1186)
**Predicted under pi_21:** 0.1667 (var=0.0268)

### Experiment 27
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Check where high-validity cues (first two) are tied
    tied_high = (a_ratings[:, 0] == b_ratings[:, 0]) & (a_ratings[:, 1] == b_ratings[:, 1])
    
    # Sum the 0.5-validity cues (last three)
    sum_a_low = a_ratings[:, 2:].sum(axis=1)
    sum_b_low = b_ratings[:, 2:].sum(axis=1)
    
    a_more = sum_a_low > sum_b_low
    b_more = sum_b_low > sum_a_low
    
    # Filter to trials where high-validity cues are tied AND low-validity cues differ
    valid_trials = tied_high & (a_more | b_more)
    if not np.any(valid_trials):
        return 0.5
        
    responses = data['response'].values
    
    # Check if subject chose the option with more 0.5-validity cues
    chose_a_more = a_more & (responses == 0)
    chose_b_more = b_more & (responses == 1)
    
    consistent = chose_a_more | chose_b_more
    
    return float(np.sum(consistent[valid_trials]) / np.sum(valid_trials))
```

**Observed (real) value:** 0.6217 (var=0.0020)
**Predicted under pi_16:** 0.5864 (var=0.0792)
**Predicted under pi_21:** 0.8406 (var=0.0059)

### Experiment 28
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1 vs 7
    is_t1 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (b_tuples == (0, 0, 0, 0, 0, 0, 0, 0))
    is_t1_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (a_tuples == (0, 0, 0, 0, 0, 0, 0, 0))
    is_t7 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (b_tuples == (1, 1, 1, 1, 1, 1, 0, 0))
    is_t7_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (a_tuples == (1, 1, 1, 1, 1, 1, 0, 0))
    
    # Trial 8 vs 12
    is_t8 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 0)) & (b_tuples == (0, 0, 0, 0, 0, 0, 0, 1))
    is_t8_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 0)) & (a_tuples == (0, 0, 0, 0, 0, 0, 0, 1))
    is_t12 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 0)) & (b_tuples == (1, 1, 1, 1, 0, 0, 0, 1))
    is_t12_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 0)) & (a_tuples == (1, 1, 1, 1, 0, 0, 0, 1))
    
    # Trial 13 vs 15
    is_t13 = (a_tuples == (1, 0, 1, 1, 1, 1, 1, 1)) & (b_tuples == (0, 1, 0, 0, 0, 0, 0, 0))
    is_t13_rev = (b_tuples == (1, 0, 1, 1, 1, 1, 1, 1)) & (a_tuples == (0, 1, 0, 0, 0, 0, 0, 0))
    is_t15 = (a_tuples == (1, 1, 1, 0, 1, 1, 1, 1)) & (b_tuples == (1, 1, 0, 1, 0, 0, 0, 0))
    is_t15_rev = (b_tuples == (1, 1, 1, 0, 1, 1, 1, 1)) & (a_tuples == (1, 1, 0, 1, 0, 0, 0, 0))
    
    untied_chosen = []
    tied_chosen = []
    
    for idx, row in data.iterrows():
        # Untied trials
        if is_t1[idx] or is_t8[idx] or is_t13[idx]:
            untied_chosen.append(1.0 if row['response'] == 0 else 0.0)
        elif is_t1_rev[idx] or is_t8_rev[idx] or is_t13_rev[idx]:
            untied_chosen.append(1.0 if row['response'] == 1 else 0.0)
            
        # Tied trials
        if is_t7[idx] or is_t12[idx] or is_t15[idx]:
            tied_chosen.append(1.0 if row['response'] == 0 else 0.0)
        elif is_t7_rev[idx] or is_t12_rev[idx] or is_t15_rev[idx]:
            tied_chosen.append(1.0 if row['response'] == 1 else 0.0)
            
    p_untied = sum(untied_chosen) / len(untied_chosen) if untied_chosen else 0.5
    p_tied = sum(tied_chosen) / len(tied_chosen) if tied_chosen else 0.5
    
    return float(p_untied - p_tied)
```

**Observed (real) value:** -0.0111 (var=0.0096)
**Predicted under pi_16:** -0.0178 (var=0.0459)
**Predicted under pi_21:** 0.0167 (var=0.0126)

### Experiment 29
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t4_mask = (a_str == '11000') & (b_str == '00111')
    t5_mask = (a_str == '11100') & (b_str == '00011')
    t6_mask = (a_str == '01000') & (b_str == '00111')
    t7_mask = (a_str == '01000') & (b_str == '00011')
    
    p_a = {}
    for i, mask in enumerate([t4_mask, t5_mask, t6_mask, t7_mask]):
        if mask.sum() > 0:
            p_a[i] = (data.loc[mask, 'response'] == 0).mean()
        else:
            p_a[i] = 0.5
            
    return float((p_a[1] - p_a[0]) + (p_a[3] - p_a[2]))
```

**Observed (real) value:** 0.7800 (var=0.0444)
**Predicted under pi_16:** 0.4083 (var=0.1940)
**Predicted under pi_21:** 0.4883 (var=0.1544)

### Experiment 30
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # High Tally pool: Tally differences of 0 and +1
    high_mask = (
        ((data['A_str'] == '10000') & (data['B_str'] == '01000')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00100')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00010')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00001')) | 
        ((data['A_str'] == '11000') & (data['B_str'] == '00001')) | 
        ((data['A_str'] == '10100') & (data['B_str'] == '00001')) | 
        ((data['A_str'] == '10010') & (data['B_str'] == '00001'))   
    )
    
    # Low Tally pool: Tally differences of -1 and -2
    low_mask = (
        ((data['A_str'] == '10000') & (data['B_str'] == '01100')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00110')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00011')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '01110')) | 
        ((data['A_str'] == '10000') & (data['B_str'] == '00111'))   
    )
    
    high_mean = data[high_mask]['chose_A'].mean()
    low_mean = data[low_mask]['chose_A'].mean()
    
    if pd.isna(high_mean): high_mean = 0.5
    if pd.isna(low_mean): low_mean = 0.5
        
    return float(high_mean - low_mean)
```

**Observed (real) value:** -0.3346 (var=0.0085)
**Predicted under pi_16:** 0.3668 (var=0.2179)
**Predicted under pi_21:** 0.6527 (var=0.0696)

### Experiment 31
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    t3_mask = (data['A_tuple'] == (1, 0, 0, 1, 0)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t4_mask = (data['A_tuple'] == (1, 0, 0, 0, 1)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t5_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t7_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 1))
    
    p_a_t3 = data.loc[t3_mask, 'chose_A'].mean() if t3_mask.sum() > 0 else 0.0
    p_a_t4 = data.loc[t4_mask, 'chose_A'].mean() if t4_mask.sum() > 0 else 0.0
    p_a_t5 = data.loc[t5_mask, 'chose_A'].mean() if t5_mask.sum() > 0 else 0.0
    p_a_t7 = data.loc[t7_mask, 'chose_A'].mean() if t7_mask.sum() > 0 else 0.0
    
    return (p_a_t3 - p_a_t4) + (p_a_t7 - p_a_t5)
```

**Observed (real) value:** 0.0578 (var=0.0307)
**Predicted under pi_16:** 0.2356 (var=0.0941)
**Predicted under pi_21:** 0.2578 (var=0.0773)

### Experiment 32
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_has_cue_1 = data['option_a_ratings'].apply(lambda x: x[1] == 1)
    a_has_cue_4 = data['option_a_ratings'].apply(lambda x: x[4] == 1)
    
    p_a_given_cue_1 = (data[a_has_cue_1]['response'] == 0).mean()
    p_a_given_cue_4 = (data[a_has_cue_4]['response'] == 0).mean()
    
    return float(p_a_given_cue_1 - p_a_given_cue_4)
```

**Observed (real) value:** -0.0033 (var=0.0073)
**Predicted under pi_16:** 0.1850 (var=0.0634)
**Predicted under pi_21:** 0.3508 (var=0.0385)

### Experiment 33
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    scores = []
    
    for subj, subj_df in data.groupby('subject_id'):
        high_choices = []
        mid_choices = []
        
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            sum_a = np.sum(a)
            sum_b = np.sum(b)
            
            # Only look at trials where both options have the same total number of cues.
            # This perfectly neutralizes the variance caused by LCA's configural gamma parameter.
            if sum_a == sum_b:
                val_a = np.sum(a * val)
                val_b = np.sum(b * val)
                
                chose_a = 1.0 if row['response'] == 0 else 0.0
                
                # Trials 1-4 and 9-12: The linear sum of validities strongly favors A.
                if val_a > val_b + 1e-4:
                    high_choices.append(chose_a)
                # Trials 13-14: The linear sum of validities is EXACTLY equal for A and B.
                elif abs(val_a - val_b) <= 1e-4:
                    mid_choices.append(chose_a)
                    
        if len(high_choices) > 0 and len(mid_choices) > 0:
            scores.append(np.mean(high_choices) - np.mean(mid_choices))
            
    if not scores:
        return 0.0
        
    return float(np.mean(scores))
```

**Observed (real) value:** 0.7617 (var=0.0238)
**Predicted under pi_16:** 0.2429 (var=0.0438)
**Predicted under pi_21:** 0.2183 (var=0.0358)

### Experiment 34
**Design**
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_target_trial = data['option_a_ratings'].apply(lambda x: x[0] == 1) & data['option_b_ratings'].apply(lambda x: x[1] == 1 and x[2] == 1)
    target_data = data[is_target_trial]
    if len(target_data) == 0:
        return 0.5
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.3800 (var=0.0041)
**Predicted under pi_16:** 0.3042 (var=0.0229)
**Predicted under pi_21:** 0.3583 (var=0.0181)

### Experiment 35
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Focus on trials where the tallies are perfectly matched
    mask = sum_a == sum_b
    
    # Compare choice rates between trials with high background tally vs low background tally
    high_mask = mask & (sum_a >= 4)
    low_mask = mask & (sum_a <= 2)
    
    # P(Choose A) = 1.0 - mean(response)
    p_a_high = 1.0 - data.loc[high_mask, 'response'].mean()
    p_a_low = 1.0 - data.loc[low_mask, 'response'].mean()
    
    if pd.isna(p_a_high) or pd.isna(p_a_low):
        return 0.0
        
    return float(p_a_high - p_a_low)
```

**Observed (real) value:** 0.0000 (var=0.0126)
**Predicted under pi_16:** -0.0900 (var=0.0201)
**Predicted under pi_21:** -0.0281 (var=0.0206)

### Experiment 36
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the tally (number of positive cues) for options A and B
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    # Determine if the subject's response matched the option with the higher tally
    # response == 0 means A was chosen, response == 1 means B was chosen
    chose_higher_tally = ((tally_a > tally_b) & (data['response'] == 0)) | \
                         ((tally_b > tally_a) & (data['response'] == 1))
                         
    # Return the overall proportion of trials where the higher tally option was chosen
    return float(chose_higher_tally.mean())
```

**Observed (real) value:** 0.3504 (var=0.0038)
**Predicted under pi_16:** 0.6140 (var=0.0735)
**Predicted under pi_21:** 0.8296 (var=0.0244)

### Experiment 37
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Extract the first 4 cues to identify the base trial type
    data['a_base'] = data['option_a_ratings'].apply(lambda x: tuple(x[:4]))
    
    # Base 2 trials: Option A has cues 0 and 3 (validities 0.95 + 0.65 = 1.60)
    # Option B has cues 1 and 2 (validities 0.85 + 0.75 = 1.60)
    base2_mask = data['a_base'] == (1, 0, 0, 1)
    
    # Base 4 trials: Option A has cues 1 and 2 (validities 0.85 + 0.75 = 1.60)
    # Option B has cues 0 and 3 (validities 0.95 + 0.65 = 1.60)
    base4_mask = data['a_base'] == (0, 1, 1, 0)
    
    if base2_mask.sum() == 0 or base4_mask.sum() == 0:
        return 0.0
        
    # Calculate the probability of choosing Option A (response == 0) for each base
    p_base2 = (data.loc[base2_mask, 'response'] == 0).mean()
    p_base4 = (data.loc[base4_mask, 'response'] == 0).mean()
    
    # Return the difference in choice probability
    return float(p_base2 - p_base4)
```

**Observed (real) value:** 0.0083 (var=0.0107)
**Predicted under pi_16:** -0.0083 (var=0.0292)
**Predicted under pi_21:** 0.0258 (var=0.0466)

### Experiment 38
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data = data.copy()
    data['a_tup'] = data['option_a_ratings'].apply(tuple)
    data['b_tup'] = data['option_b_ratings'].apply(tuple)
    
    # Critical trials where cue sums and counts are strictly matched
    # The target is the option with the highest-validity cue
    targets = {
        ((1, 0, 0, 1, 0), (0, 1, 1, 0, 0)): 0,
        ((1, 0, 0, 0, 1), (0, 1, 0, 1, 0)): 0,
        ((0, 1, 0, 0, 1), (0, 0, 1, 1, 0)): 0,
        ((0, 1, 1, 0, 0), (1, 0, 0, 1, 0)): 1,
        ((0, 1, 0, 1, 0), (1, 0, 0, 0, 1)): 1,
        ((0, 0, 1, 1, 0), (0, 1, 0, 0, 1)): 1
    }
    
    scores = []
    for subj, subj_df in data.groupby('subject_id'):
        correct = 0
        total = 0
        for _, row in subj_df.iterrows():
            pair = (row['a_tup'], row['b_tup'])
            if pair in targets:
                if row['response'] == targets[pair]:
                    correct += 1
                total += 1
        if total > 0:
            scores.append(correct / total)
            
    if not scores:
        return 0.5
        
    return float(sum(scores) / len(scores))
```

**Observed (real) value:** 0.4883 (var=0.0028)
**Predicted under pi_16:** 0.4900 (var=0.0075)
**Predicted under pi_21:** 0.5025 (var=0.0052)

### Experiment 39
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    if 'subject_id' not in data.columns:
        data = data.copy()
        data['subject_id'] = 'dummy'

    scores = []
    for subj, subj_df in data.groupby('subject_id'):
        subj_df = subj_df.copy()
        
        # Calculate sum of cues
        subj_df['sum_a'] = subj_df['option_a_ratings'].apply(sum)
        subj_df['sum_b'] = subj_df['option_b_ratings'].apply(sum)
        
        # Identify if A or B is exactly the top cue [1, 0, 0, 0, 0, 0]
        subj_df['a_is_top'] = subj_df['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 1)
        subj_df['b_is_top'] = subj_df['option_b_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 1)
        
        p_top = {}
        for k in [2, 3, 4, 5]:
            # A is top, B has k cues
            t_A = subj_df[subj_df['a_is_top'] & (subj_df['sum_b'] == k)]
            p_A = 1.0 - t_A['response'].mean() if len(t_A) > 0 else 0.5
            
            # B is top, A has k cues
            t_B = subj_df[subj_df['b_is_top'] & (subj_df['sum_a'] == k)]
            p_B = t_B['response'].mean() if len(t_B) > 0 else 0.5
            
            p_top[k] = (p_A + p_B) / 2.0
            
        # Metric: P(Top Cue | vs 4 or 5) - P(Top Cue | vs 2 or 3)
        score = (p_top[4] + p_top[5]) - (p_top[2] + p_top[3])
        scores.append(score)
        
    return float(np.mean(scores))
```

**Observed (real) value:** -0.0100 (var=0.0374)
**Predicted under pi_16:** 0.0258 (var=0.0304)
**Predicted under pi_21:** 0.0142 (var=0.1760)

### Experiment 40
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Identify the critical trials where LCA predicts exact 50/50 symmetry
    # due to identical linear sums of validities and equal cue counts.
    def is_target(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        t1 = a == (1, 0, 0, 0, 0, 1) and b == (0, 0, 1, 1, 0, 0)
        t2 = a == (1, 0, 0, 0, 0, 1) and b == (0, 1, 0, 0, 1, 0)
        t3 = a == (0, 1, 0, 0, 1, 0) and b == (0, 0, 1, 1, 0, 0)
        t4 = a == (1, 0, 1, 0, 0, 1) and b == (0, 1, 0, 1, 1, 0)
        return t1 or t2 or t3 or t4

    mask = data.apply(is_target, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(1 - target_data['response']))
```

**Observed (real) value:** 0.3354 (var=0.0033)
**Predicted under pi_16:** 0.4885 (var=0.0065)
**Predicted under pi_21:** 0.5023 (var=0.0058)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the large suite of experiments, both theories show significant predictive gaps, but Theory 1 (pi_16 - LCA with Configural Weighting) generally tracks the direction of the observed data better than Theory 2 (pi_21 - Sequential Sampling). Theory 2's strict sequential sampling and thresholding mechanism causes it to predict extreme probabilities or complete invariance in scenarios where human subjects show graded, compensatory trade-offs (e.g., Experiments 4, 16, 22, 26). While Theory 2 performed slightly better on the specific metrics in Experiments 1 and 2, it systematically fails to capture the nuanced integration of tied and conflicting cues seen across the broader dataset. Therefore, Theory 2's core assumption of strict sequential sampling with a conflict-modulated threshold appears too rigid.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Sequential Cue Sampling theory with a Parallel Constraint Satisfaction or Bayesian Evidence Accumulation theory. The new theory should allow for all cues to be evaluated simultaneously (parallel integration) but with dynamically adjusted weights based on cue consistency and conflict. Instead of a strict stopping rule, evidence could be integrated in a network where cues excite or inhibit each other based on their alignment, producing graded choice probabilities that naturally account for both compensatory behavior in high-conflict trials and non-compensatory behavior when a single strong cue dominates."
}
```

## Usage

```json
{
  "prompt_token_count": 34581,
  "candidates_token_count": 308,
  "total_token_count": 37219
}
```
