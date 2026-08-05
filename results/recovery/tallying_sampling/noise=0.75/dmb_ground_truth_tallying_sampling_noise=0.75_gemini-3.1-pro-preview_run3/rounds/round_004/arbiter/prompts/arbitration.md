# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Decision-makers use a 'Tallying' (Equal-Weight) heuristic, a compensatory strategy that ignores cue validities. They simply count the number of positive features (or advantages) each option has and choose the option with the highest total count. If the counts are equal, they guess. Because pure Tallying makes choices that strongly oppose Take The Best on compensatory trials, high levels of choice stochasticity (noise) are needed to pull the predicted consistency up toward the observed ~0.40-0.42 range, reflecting uncertainty or lapses in applying the heuristic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: sum the unweighted feature values for each option.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
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


## THEORY 2 — pi_6
Decision-makers probabilistically sample between a compensatory Equal-Weight (Tallying) heuristic and a non-compensatory Lexicographic (Take-The-Best) heuristic on a trial-by-trial basis, but they heavily favor Tallying. Furthermore, when they do use Take-The-Best, the application of the rule is subject to decision noise (modeled via a softmax temperature), reflecting uncertainty or stochasticity in identifying the most valid cue rather than a perfectly deterministic lexicographic choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying: sum the unweighted feature values for each option
    scores_tally = stim.sum(axis=1)
    beta = float(parameters["beta"])
    z = beta * (scores_tally - scores_tally.max())
    e = np.exp(z)
    p_tally = e / e.sum()
    
    # Take-The-Best (TTB): find the first discriminating cue ordered by validity
    order = np.argsort(validities)[::-1]
    diff = stim[0, order] - stim[1, order]
    non_zero = np.where(diff != 0)[0]
    
    p_ttb = np.array([0.5, 0.5])
    if len(non_zero) > 0:
        first_diff = diff[non_zero[0]]
        score_ttb = np.array([1.0, 0.0]) if first_diff > 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * score_ttb
        e_ttb = np.exp(z_ttb - np.max(z_ttb))
        p_ttb = e_ttb / e_ttb.sum()
            
    # Mixture of Tallying and TTB
    w_tally = float(parameters["w_tally"])
    p_mix = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Global lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_mix)
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=7):** [0.9, 0.8, 0.7, 0.6, 0.55, 0.55, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]

**Rationale:** To robustly dissociate a pure Tallying model from a Tallying/Take-The-Best (TTB) mixture model, we use a 7-feature design. This allows for a wider range of feature sum differences (Δ). The pure Tallying model's choice probabilities depend exclusively on Δ. In contrast, the mixture model's probabilities are systematically perturbed by the TTB component. We include 'Tie' trials (Δ = 0) where pure Tallying predicts exactly 50/50 guessing, but the mixture model predicts a bias toward the TTB-favored option. We also include trials with varying Δ where TTB either aligns with or opposes Tallying, testing whether the choice probabilities exhibit the TTB-induced shifts predicted by the mixture model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric contrasts the proportion of Tally-aligned choices on 'Agreement' trials (where Take-The-Best agrees with Tallying) versus 'Disagreement' trials. Specifically, it pools T4 (diff=2) and T6 (diff=1) for Agreement, and T3 (diff=2) and T8 (diff=3) for Disagreement. For the Advocated theory (Pure Tallying), the average score difference on Disagreement trials (mean diff = 2.5) is strictly larger than on Agreement trials (mean diff = 1.5). Because Pure Tallying choices depend entirely on this difference, it predicts a higher Tally-alignment on the Disagreement trials, making the metric strictly NEGATIVE. Conversely, the Competing theory (Mixture model) predicts that the intermittent use of TTB will heavily boost Tally-aligned choices on Agreement trials and severely depress them on Disagreement trials, overpowering the underlying score differences and driving the metric to be POSITIVE. This creates a strong, unambiguous mean gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    A_sum = A.sum(axis=1)
    B_sum = B.sum(axis=1)
    A_0 = A[:, 0]
    
    # Identify specific trials by their Tallying sums and Cue 0 values
    # T4: Agreement. Tally favors A (4 vs 2), TTB favors A (A_0=1)
    mask_T4 = (A_sum == 4) & (B_sum == 2) & (A_0 == 1)
    # T6: Agreement. Tally favors B (2 vs 3), TTB favors B (A_0=0)
    mask_T6 = (A_sum == 2) & (B_sum == 3) & (A_0 == 0)
    
    # T3: Disagreement. Tally favors A (4 vs 2), TTB favors B (A_0=0)
    mask_T3 = (A_sum == 4) & (B_sum == 2) & (A_0 == 0)
    # T8: Disagreement. Tally favors A (5 vs 2), TTB favors B (A_0=0)
    mask_T8 = (A_sum == 5) & (B_sum == 2) & (A_0 == 0)
    
    # Compute proportion of Tally-aligned choices
    p_T4 = np.mean(resp[mask_T4] == 0) if np.sum(mask_T4) > 0 else 0.5
    p_T6 = np.mean(resp[mask_T6] == 1) if np.sum(mask_T6) > 0 else 0.5
    
    p_T3 = np.mean(resp[mask_T3] == 0) if np.sum(mask_T3) > 0 else 0.5
    p_T8 = np.mean(resp[mask_T8] == 0) if np.sum(mask_T8) > 0 else 0.5
    
    # Average Tally-alignment for Agreement vs Disagreement trials
    p_agree = (p_T4 + p_T6) / 2.0
    p_disagree = (p_T3 + p_T8) / 2.0
    
    # Metric: Difference in Tally-alignment (Agreement - Disagreement)
    return float(p_agree - p_disagree)

### RESULTS
- Predicted under pi_3 (simulated): -0.0208 (var=0.0129)
- Predicted under pi_6 (simulated): 0.0983 (var=0.0241)
- Observed on real data: -0.0200 (var=0.0235)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Rationale:** To robustly dissociate a pure Tallying model from a Tallying/Take-The-Best (TTB) mixture model, we utilize a 5-feature design and manipulate the alignment between the two heuristics. Pure Tallying choices depend strictly on the difference in the total count of positive features between the options. Crucially, on trials where the feature counts are tied, pure Tallying predicts exactly 50/50 guessing. In contrast, the mixture model predicts a systematic deviation from 50/50 towards the option favored by TTB (the one with the highest-validity discriminating cue). Furthermore, on trials where Tallying favors one option but TTB favors the other, pure Tallying predicts the same choice probability as it would if TTB agreed, whereas the mixture model predicts an attenuated choice probability due to the conflicting TTB component.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the Tally-match rate between 'Oppose' trials (where Tallying and TTB disagree) and 'Agree' trials (where they agree). Crucially, in this experimental design, the Oppose trials have a higher average Tallying difference (mean diff = 1.5) than the Agree trials (mean diff = 1.0). A pure Tallying model, which only cares about the sum difference, will therefore predict a HIGHER Tally-match rate on Oppose trials than on Agree trials (yielding a positive metric). In stark contrast, the Mixture model incorporates the TTB heuristic, which pulls choices away from Tallying on Oppose trials and towards Tallying on Agree trials, yielding a strongly NEGATIVE metric. This structural sign reversal between the two theories ensures robust discrimination with high statistical power.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_oppose = []
    tally_agree = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Skip tie trials
        if sum_a == sum_b:
            continue
            
        diff = a - b
        non_zero = np.where(diff != 0)[0]
        if len(non_zero) == 0:
            continue
            
        ttb_favors_a = (diff[non_zero[0]] > 0)
        tally_favors_a = (sum_a > sum_b)
        
        chose_a = (row['response'] == 0)
        chose_tally = (chose_a == tally_favors_a)
        
        if tally_favors_a != ttb_favors_a:
            tally_oppose.append(1.0 if chose_tally else 0.0)
        else:
            tally_agree.append(1.0 if chose_tally else 0.0)
            
    if not tally_oppose or not tally_agree:
        return 0.0
        
    return float(np.mean(tally_oppose) - np.mean(tally_agree))

### RESULTS
- Predicted under pi_3 (simulated): 0.0292 (var=0.0122)
- Predicted under pi_6 (simulated): -0.0833 (var=0.0232)
- Observed on real data: 0.0133 (var=0.0119)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Features are already sorted by validity in descending order.
        # Find the first discriminating cue.
        ttb_choice = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_choice = 0
                break
            elif b[j] > a[j]:
                ttb_choice = 1
                break
                
        if ttb_choice != -1:
            if resp == ttb_choice:
                ttb_consistent += 1
            total += 1
            
    return ttb_consistent / total if total > 0 else 0.5
```

**Observed (real) value:** 0.4249 (var=0.0029)
**Predicted under pi_3:** 0.4247 (var=0.0036)
**Predicted under pi_6:** 0.4442 (var=0.0103)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = [0.9, 0.8, 0.7, 0.6]
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB choice
        ttb = -1
        for i in range(4):
            if a[i] > b[i]: 
                ttb = 0
                break
            elif b[i] > a[i]: 
                ttb = 1
                break
                
        # Expected WADD choice (with equal subjective weights)
        score_a = sum(a[i] * validities[i] for i in range(4))
        score_b = sum(b[i] * validities[i] for i in range(4))
        wadd = 0 if score_a > score_b else (1 if score_b > score_a else -1)
        
        # Only consider trials where TTB and WADD diverge
        if ttb != -1 and wadd != -1 and ttb != wadd:
            matches.append(1.0 if row['response'] == ttb else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.4027 (var=0.0071)
**Predicted under pi_3:** 0.4300 (var=0.0059)
**Predicted under pi_6:** 0.4067 (var=0.0180)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    def get_mask(a_target, b_target):
        m1 = (a_str == a_target) & (b_str == b_target)
        m2 = (a_str == b_target) & (b_str == a_target)
        return m1, m2
    
    t1_m1, t1_m2 = get_mask('00111', '11000')
    t5_m1, t5_m2 = get_mask('00110', '00001')
    t7_m1, t7_m2 = get_mask('01110', '10001')
    t8_m1, t8_m2 = get_mask('00111', '10100')
    
    def get_p_a(m1, m2):
        mask = m1 | m2
        if not mask.any():
            return 0.5
        chose_a = np.where(m1, data['response'] == 0, data['response'] == 1)
        return float(chose_a[mask].mean())
    
    p_a_t1 = get_p_a(t1_m1, t1_m2)
    p_a_t5 = get_p_a(t5_m1, t5_m2)
    p_a_t7 = get_p_a(t7_m1, t7_m2)
    p_a_t8 = get_p_a(t8_m1, t8_m2)
    
    group1_p = (p_a_t5 + p_a_t7) / 2.0
    group2_p = (p_a_t1 + p_a_t8) / 2.0
    
    return float(group1_p - group2_p)
```

**Observed (real) value:** -0.0383 (var=0.0218)
**Predicted under pi_3:** 0.0183 (var=0.0170)
**Predicted under pi_6:** 0.0583 (var=0.0138)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    t6_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    t1_wadd_match = (data['response'] == 0) & t1_mask
    t6_wadd_match = (data['response'] == 1) & t6_mask
    
    relevant_trials = t1_mask | t6_mask
    if not relevant_trials.any():
        return 0.5
        
    return float((t1_wadd_match.sum() + t6_wadd_match.sum()) / relevant_trials.sum())
```

**Observed (real) value:** 0.4012 (var=0.0049)
**Predicted under pi_3:** 0.4512 (var=0.0076)
**Predicted under pi_6:** 0.3819 (var=0.0171)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_mat.sum(axis=1)
    sum_b = b_mat.sum(axis=1)
    
    tally_choice = np.zeros(len(data)) - 1
    tally_choice[sum_a > sum_b] = 0
    tally_choice[sum_b > sum_a] = 1
    
    diff = a_mat - b_mat
    non_zero = diff != 0
    first_diff_idx = np.argmax(non_zero, axis=1)
    has_diff = np.any(non_zero, axis=1)
    
    ttb_choice = np.zeros(len(data)) - 1
    row_indices = np.arange(len(data))
    first_diff_vals = diff[row_indices, first_diff_idx]
    
    ttb_choice[(has_diff) & (first_diff_vals > 0)] = 0
    ttb_choice[(has_diff) & (first_diff_vals < 0)] = 1
    
    conflict = (tally_choice != -1) & (ttb_choice != -1) & (tally_choice != ttb_choice)
    
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[conflict] == ttb_choice[conflict])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4294 (var=0.0047)
**Predicted under pi_3:** 0.4267 (var=0.0064)
**Predicted under pi_6:** 0.4086 (var=0.0113)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    mask = sum_b > sum_a
    if mask.sum() == 0:
        return np.nan
        
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.4069 (var=0.0041)
**Predicted under pi_3:** 0.3997 (var=0.0059)
**Predicted under pi_6:** 0.3925 (var=0.0134)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Convert lists to tuples for hashable comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Identify Trial 1 and Trial 2 where Option A has the highest-validity feature 
    # but fewer total features than Option B.
    t1_a = (1, 1, 0, 0, 0)
    t2_a = (1, 0, 0, 0, 0)
    
    target_trials = data[a_tuples.isin([t1_a, t2_a])]
    
    if len(target_trials) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return (target_trials['response'] == 0).mean()

```

**Observed (real) value:** 0.4333 (var=0.0072)
**Predicted under pi_3:** 0.4575 (var=0.0125)
**Predicted under pi_6:** 0.4250 (var=0.0174)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t3_a = (1, 1, 1, 0, 0, 0)
    t3_b = (0, 0, 0, 1, 1, 1)
    
    t4_a = (1, 0, 0, 0, 0, 0)
    t4_b = (0, 0, 0, 0, 0, 1)
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t3_a and b == t3_b:
            total += 1
            if resp == 0: match_count += 1
        elif a == t3_b and b == t3_a:
            total += 1
            if resp == 1: match_count += 1
            
        elif a == t4_a and b == t4_b:
            total += 1
            if resp == 0: match_count += 1
        elif a == t4_b and b == t4_a:
            total += 1
            if resp == 1: match_count += 1
            
    if total == 0:
        return 0.5
        
    return float(match_count / total)
```

**Observed (real) value:** 0.4888 (var=0.0062)
**Predicted under pi_3:** 0.5081 (var=0.0059)
**Predicted under pi_6:** 0.5269 (var=0.0102)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both newly proposed experiments, Theory 1 (Pure Tallying) vastly outperforms Theory 2 (Tallying/TTB Mixture). In Experiment 1, the observed metric (-0.0200) perfectly matches the Pure Tallying prediction (-0.0208) while completely rejecting the Mixture model's prediction (0.0983). Similarly, in Experiment 2, the observed metric (0.0133) aligns with the positive value predicted by Pure Tallying (0.0292) and strongly contradicts the negative value predicted by the Mixture model (-0.0833). The data clearly show that subjects do not exhibit the systematic lexicographic (TTB) deviations predicted by Theory 2 on these critical trials. While Theory 1 is generally very strong across the older experiments as well, there are minor deviations (e.g., in Exp 4, 6, and 9) where some mild cue-weighting might be happening. Therefore, Theory 2 is degenerate because its TTB component is too strong and non-compensatory.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory based on a Weighted Additive (WADD) heuristic rather than a Tallying/TTB mixture. In this new theory, subjects are compensatory but do not use strictly equal weights (as in Theory 1) nor do they use a strict non-compensatory rule (as in Theory 2). Instead, they weight each feature proportionally to its stated validity (or a transformed version of it, like subjective validities that decay less steeply than TTB requires), and then sum these weighted features subject to decision noise. This allows for slight preferences toward higher-validity cues without predicting the massive, discrete shifts on tie-trials that the TTB mixture model erroneously predicted."
}
```

## Usage

```json
{
  "prompt_token_count": 10159,
  "candidates_token_count": 411,
  "total_token_count": 11464
}
```
