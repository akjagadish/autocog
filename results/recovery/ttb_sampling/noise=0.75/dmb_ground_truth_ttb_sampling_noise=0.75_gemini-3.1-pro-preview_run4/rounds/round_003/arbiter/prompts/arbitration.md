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
Take The Best (TTB) is a non-compensatory lexicographic heuristic. Decision makers rank features according to their validity. To choose between two options, they compare them on the most valid feature. If one option has a higher value on this feature, it is chosen immediately, and all remaining features are ignored. If the options are tied on this feature, the decision maker moves to the next most valid feature, and so on. If the options tie on all features, the decision maker guesses randomly. Response noise is modeled via a simple lapse rate (epsilon) where the subject makes a random choice instead of following the TTB rule. The lapse rate can be high, reflecting significant guessing in the empirical data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Order features by validity, descending
    order = np.argsort(validities)[::-1]
    
    # Find the first discriminating feature
    chosen = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen = 1
            break
            
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # Tie on all features
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_5
Subjective Take The Best (Noisy-Validity TTB). Decision makers strictly follow the non-compensatory Take The Best (TTB) heuristic on any given trial, meaning they base their choice solely on the first discriminating cue they consider and ignore all others. However, their internal ranking of cue validities is noisy. This is modeled by sampling the primary discriminating cue via a softmax distribution over the objective validities of the cues that differ between the options. The inverse temperature parameter beta controls the noise in the validity ranking: as beta increases, the model converges to deterministic TTB, whereas lower beta values allow sub-optimal cues to occasionally be evaluated first. Because decisions rely on a single cue, the model captures the flat sensitivity curves to supporting cue quantities (Exps 1, 2, 6). Meanwhile, the stochastic cue selection suppresses the overall agreement with the objective TTB predictions, matching the lower empirical performance in Exps 3, 4, and 5. Response noise is included via a simple lapse rate (epsilon).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify discriminating cues
    diff = stim[0] - stim[1]
    discrim_idx = np.where(diff != 0)[0]
    
    if len(discrim_idx) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Softmax over validities of discriminating cues
        v_discrim = validities[discrim_idx]
        # Numerically stable softmax
        z = beta * v_discrim
        z -= np.max(z)
        w = np.exp(z)
        p_discrim = w / np.sum(w)
        
        p_a = 0.0
        p_b = 0.0
        for i, idx in enumerate(discrim_idx):
            if diff[idx] > 0:
                p_a += p_discrim[i]
            else:
                p_b += p_discrim[i]
                
        p_core = np.array([p_a, p_b])
        
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate Pure Take The Best (TTB) from Subjective (Noisy-Validity) TTB, we exploit their different treatments of subordinate discriminating cues. Pure TTB strictly evaluates cues in descending order of objective validity; therefore, as long as Option A wins on the most valid discriminating cue, Pure TTB predicts a constant, flat probability of choosing A, completely ignoring all other cues. In contrast, Subjective TTB selects the primary cue via a softmax distribution over all cues that discriminate between the options. Consequently, Subjective TTB predicts that the probability of choosing Option A will decrease as we add more subordinate cues favoring Option B, and increase as we add more subordinate cues favoring Option A. We parametrically vary the number of supporting and opposing subordinate cues while keeping the highest validity cue constantly in favor of Option A.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Pure TTB bases its decision entirely on the single most valid discriminating cue (which favors Option A in all these trials) and is insensitive to the number of subordinate cues. Therefore, the probability of choosing Option A should be identical whether the subordinate cues support Option A or Option B. Subjective TTB, however, samples the cue to use based on all discriminating cues. By pooling the three trials where subordinate cues heavily support Option A and comparing them to the three trials where subordinate cues heavily support Option B, we maximize the divergence between the two models while dramatically reducing single-trial binomial noise (each proportion is based on 36 trials per subject rather than 12). This will yield a near-zero mean with low variance for Pure TTB, and a consistently positive difference for Subjective TTB.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    A_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    B_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trials where subordinate cues support Option A
    a_support_mask = (A_str.isin(['10100', '10110', '10111'])) & (B_str == '01000')
    # Trials where subordinate cues support Option B
    b_support_mask = (A_str == '10000') & (B_str.isin(['01100', '01110', '01111']))
    
    p_a_support = (data.loc[a_support_mask, 'response'] == 0).mean()
    p_b_support = (data.loc[b_support_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_support) or pd.isna(p_b_support):
        return 0.0
        
    return float(p_a_support - p_b_support)

### RESULTS
- Predicted under pi_3 (simulated): 0.0206 (var=0.0089)
- Predicted under pi_5 (simulated): 0.1600 (var=0.0250)
- Observed on real data: -0.0278 (var=0.0121)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=20):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 9: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 11: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 18: A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 19: A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 20: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate Subjective (Noisy-Validity) TTB from Pure TTB, we restrict the discriminating cues to exactly two per trial: one favoring Option A and one favoring Option B. Across trials, we fix the winning cue for the favored option but parametrically decrease the validity of the opposing cue. Because Pure TTB evaluates cues strictly in descending order of objective validity, it predicts a completely flat choice probability across these trials: the favored option is always chosen with the same probability as long as its winning cue has a higher validity than the opposing cue. In contrast, Subjective TTB samples the primary cue via a softmax distribution over the validities of all discriminating cues. Consequently, Subjective TTB predicts that the probability of choosing the favored option will monotonically increase as the validity of the opposing cue decreases, because the lower-validity opposing cue becomes less likely to win the softmax competition.

**Computed schedule:** 20 unique pairs × 4 reps = 80 trials per subject.



### METRIC
Rationale:
By design, each trial pits exactly two discriminating cues against each other. Pure Take The Best (TTB) evaluates cues strictly in order of validity, so the probability of choosing the option supported by the higher-validity cue is constant regardless of the opposing cue's validity. Subjective TTB, however, samples the primary cue via a softmax distribution over the validities of the discriminating cues. Therefore, as the validity gap between the two cues increases, the probability of sampling the higher-validity cue increases. This metric computes the difference in the proportion of choices favoring the higher-validity cue between trials with a large validity gap (index difference >= 3) and trials with a small validity gap (index difference == 1). Pure TTB predicts a difference near zero, while Subjective TTB predicts a significantly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    diffs = []
    favored = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        idx_a = -1
        idx_b = -1
        for i, (va, vb) in enumerate(zip(a, b)):
            if va > vb:
                idx_a = i
            elif vb > va:
                idx_b = i
                
        if idx_a != -1 and idx_b != -1:
            if idx_a < idx_b:
                diff = idx_b - idx_a
                fav = 0
            else:
                diff = idx_a - idx_b
                fav = 1
            diffs.append(diff)
            favored.append(1 if row['response'] == fav else 0)
            
    df = pd.DataFrame({'diff': diffs, 'fav': favored})
    high = df[df['diff'] >= 3]['fav'].mean()
    low = df[df['diff'] == 1]['fav'].mean()
    return float(high - low)

### RESULTS
- Predicted under pi_3 (simulated): -0.0042 (var=0.0142)
- Predicted under pi_5 (simulated): 0.1460 (var=0.0122)
- Observed on real data: -0.0179 (var=0.0321)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Calculate the sum of features for options A and B
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Isolate trials 1 and 2 where Tallying and WADD make opposite predictions.
    # These are uniquely identified by the total number of features across both options being 5.
    mask = (a_sums + b_sums) == 5
    if not mask.any():
        return 0.5
    
    # Tallying predicts the option with more features (lower sum means response 1, i.e., B)
    tally_predictions = (a_sums < b_sums).astype(int)
    
    # Calculate the proportion of choices on these trials that align with Tallying
    matches = (data.loc[mask, 'response'] == tally_predictions.loc[mask]).mean()
    return float(matches)

```

**Observed (real) value:** 0.3883 (var=0.0126)
**Predicted under pi_3:** 0.3117 (var=0.0269)
**Predicted under pi_5:** 0.1892 (var=0.0222)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    mask = (a_str == '11000') & (b_str == '00111')
    if not mask.any():
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5633 (var=0.0246)
**Predicted under pi_3:** 0.7400 (var=0.0299)
**Predicted under pi_5:** 0.8150 (var=0.0251)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0.5
        
    preds = [ttb_predict(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    resps = data['response'].values
    
    agreements = []
    for p, r in zip(preds, resps):
        if p == 0.5:
            agreements.append(0.5)
        else:
            agreements.append(1.0 if p == r else 0.0)
            
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.5971 (var=0.0020)
**Predicted under pi_3:** 0.7942 (var=0.0242)
**Predicted under pi_5:** 0.6404 (var=0.0178)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    a_sum = data['option_a_ratings'].apply(sum)
    diff = b_sum - a_sum
    
    high_diff = data[diff >= 2]['response']
    low_diff = data[diff <= 0]['response']
    
    if len(high_diff) == 0 or len(low_diff) == 0:
        return 0.0
        
    return float(high_diff.mean() - low_diff.mean())
```

**Observed (real) value:** -0.0133 (var=0.0090)
**Predicted under pi_3:** -0.0025 (var=0.0123)
**Predicted under pi_5:** 0.1212 (var=0.0126)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_favored = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    ttb_chosen = ((a_favored & (data['response'] == 0)) | (~a_favored & (data['response'] == 1))).astype(float)
    
    opposing_cues = data.apply(lambda row: sum(row['option_b_ratings']) if row['option_a_ratings'][0] == 1 else sum(row['option_a_ratings']), axis=1)
    
    mask_1 = opposing_cues == 1
    mask_4 = opposing_cues == 4
    
    if mask_1.sum() == 0 or mask_4.sum() == 0:
        return 0.0
        
    return float(ttb_chosen[mask_1].mean() - ttb_chosen[mask_4].mean())
```

**Observed (real) value:** -0.0017 (var=0.0224)
**Predicted under pi_3:** -0.0033 (var=0.0108)
**Predicted under pi_5:** 0.0942 (var=0.0254)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    supports = []
    chose_ttbs = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # The most valid cue is always the first one (index 0)
        if a[0] > b[0]:
            ttb_fav = 0
            sup = sum(a)
        else:
            ttb_fav = 1
            sup = sum(b)
            
        supports.append(sup)
        chose_ttbs.append(1 if row['response'] == ttb_fav else 0)
        
    df = pd.DataFrame({'support': supports, 'chose_ttb': chose_ttbs})
    
    m4 = df[df['support'] == 4]['chose_ttb'].mean()
    m1 = df[df['support'] == 1]['chose_ttb'].mean()
    
    if pd.isna(m4) or pd.isna(m1):
        return 0.0
        
    return float(m4 - m1)

```

**Observed (real) value:** -0.0217 (var=0.0199)
**Predicted under pi_3:** -0.0150 (var=0.0147)
**Predicted under pi_5:** 0.0217 (var=0.0109)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the data strongly supports a strict non-compensatory decision rule. In Experiments 1, 2, 6, 7, and 8, the observed metrics evaluating sensitivity to subordinate or opposing cues consistently cluster around zero. Pure Take The Best (THEORY 1) captures this beautifully because it bases its decision solely on the highest-validity discriminating cue, ignoring the rest. Subjective TTB (THEORY 2), however, predicts a significant positive sensitivity to subordinate and opposing cues because of its softmax sampling over all discriminating cues, which is resoundingly falsified by the data. While THEORY 1 gets the structural predictions right, it overpredicts the overall agreement with TTB in Experiments 3, 4, and 5, suggesting its simple constant lapse rate is insufficient to capture the performance drop in those specific designs. THEORY 2 is fundamentally flawed because its compensatory noise mechanism contradicts the core empirical finding of flat sensitivity.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace THEORY 2 with 'Confidence-scaled Take The Best'. In this theory, decision makers strictly follow the non-compensatory TTB search rule: they find the most valid discriminating cue and completely ignore all subordinate cues (preserving the near-zero sensitivity seen in Exps 1, 2, 6, 7, 8). However, instead of a constant lapse rate, the probability of successfully executing the TTB choice versus guessing randomly is a function of the primary discriminating cue's validity. If the best discriminating cue has lower validity, the subject is less confident and guesses more often. This will naturally lower the overall TTB agreement in experiments where decisions rely on weaker cues (Exps 3, 4, 5), without introducing any compensatory leakage from subordinate cues."
}
```

## Usage

```json
{
  "prompt_token_count": 8396,
  "candidates_token_count": 394,
  "total_token_count": 9949
}
```
