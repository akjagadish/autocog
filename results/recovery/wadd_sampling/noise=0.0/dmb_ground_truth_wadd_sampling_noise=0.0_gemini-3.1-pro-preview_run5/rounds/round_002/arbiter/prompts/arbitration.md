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
Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Validity-Thresholded Tallying: Decision-makers simplify choices by ignoring cues with low validities and applying a simple tallying heuristic to the remaining high-validity cues. A subjective threshold determines which cues are considered reliable enough to use. For the included cues, the decision-maker counts how many times each option has a higher feature value than the other. The option with more wins among the thresholded cues is chosen. This boundedly rational strategy combines the frugality and robustness of tallying with the validity-sensitivity of weighted additive models, effectively breaking ties in favor of options that excel on more valid cues without requiring complex mental arithmetic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Thresholded Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues meet the subjective validity threshold
    valid_mask = val >= tau
    
    if not np.any(valid_mask):
        # If no cues are reliable enough, the decision-maker has no preference
        scores = np.zeros(2)
    else:
        # Tally wins only on the included cues
        a_filtered = stim[0, valid_mask]
        b_filtered = stim[1, valid_mask]
        
        a_wins = float(np.sum(a_filtered > b_filtered))
        b_wins = float(np.sum(b_filtered > a_filtered))
        
        scores = np.array([a_wins, b_wins])
        
    # Softmax over the tally scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 7: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]

**Rationale:** This design dissociates Weighted Additive (WADD) from Validity-Thresholded Tallying by exploiting the non-monotonic preference reversals predicted by the thresholding mechanism. We use five cues with linearly decreasing validities. In key trials, Option A is supported by the single most valid cue and the lowest valid cues, while Option B is supported by the intermediate valid cues. Because Thresholded Tallying applies a hard validity cutoff and then tallies the remaining cues equally, varying the threshold parameter tau leads to wild, non-monotonic swings in preference (e.g., A wins -> Tie -> B wins -> Tie -> A wins). In contrast, WADD integrates all cues with graded weights, predicting a consistent preference for the option with the higher weighted sum. This structural difference makes it impossible for WADD's smooth weighting (via gamma) to mimic the discrete, threshold-driven choice patterns of Thresholded Tallying.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric evaluates the difference in choice extremeness (absolute deviation from 0.5) between the strongest utility trials (Trials 7 and 8) and the weakest utility trials (Trials 5 and 6). For the Weighted Additive (WADD) model, utility differences are smooth and graded: Trials 7 and 8 have massive utility differences (±0.95), pushing choice probabilities far from 0.5, while Trials 5 and 6 have near-zero utility differences (-0.10 and 0.00), keeping probabilities very close to 0.5. This structural contrast ensures WADD predicts a large, highly positive value for this metric. In stark contrast, Validity-Thresholded Tallying applies a hard threshold and tallies wins discretely. Because Tallying drops lower-validity cues, it frequently evaluates the 'weak' Trials 5 and 6 using only 1 or 2 high-validity cues, leading to unambiguous wins that are just as extreme as Trials 7 and 8. For 80% of the threshold parameter space, Tallying predicts the exact same extremeness for these two sets of trials, anchoring its expected difference tightly near 0.0 with very low between-subject variance. This non-linear contrast cleanly and robustly separates the continuous integration of WADD from the discrete thresholding of Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Project sequence column to string for hashable equality checks
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    def subj_score(df):
        # Identify the extreme trials (T7 and T8)
        m7 = df['a_str'] == '10110'
        m8 = df['a_str'] == '01001'
        
        # Identify the weak/tie trials (T5 and T6)
        m5 = df['a_str'] == '10001'
        m6 = df['a_str'] == '01100'
        
        # Calculate probability of choosing Option A on each trial
        p7 = float((df.loc[m7, 'response'] == 0).mean()) if m7.any() else 0.5
        p8 = float((df.loc[m8, 'response'] == 0).mean()) if m8.any() else 0.5
        p5 = float((df.loc[m5, 'response'] == 0).mean()) if m5.any() else 0.5
        p6 = float((df.loc[m6, 'response'] == 0).mean()) if m6.any() else 0.5
        
        # Calculate 'extremeness' (absolute deviation from indifference)
        ext7 = abs(p7 - 0.5)
        ext8 = abs(p8 - 0.5)
        ext5 = abs(p5 - 0.5)
        ext6 = abs(p6 - 0.5)
        
        # Contrast extreme trials vs weak trials
        return (ext7 + ext8) - (ext5 + ext6)
        
    data_copy = data.copy()
    data_copy['a_str'] = a_str
    
    if 'subject_id' in data_copy.columns:
        return float(data_copy.groupby('subject_id').apply(subj_score).mean())
    return subj_score(data_copy)

### RESULTS
- Predicted under pi_3 (simulated): 0.4017 (var=0.0430)
- Predicted under pi_4 (simulated): 0.2633 (var=0.1040)
- Observed on real data: 0.0167 (var=0.1200)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 8: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]

**Rationale:** This design dissociates Validity-Thresholded Tallying from Weighted Additive (WADD) by leveraging the discrete, non-monotonic choice patterns produced by a hard validity threshold compared to smooth exponential weighting. By using five cues with a linear spread of validities, we construct trials where the highest valid cue is pitted against various combinations of lower valid cues. For example, when Option A is supported by cue 1 and 4, and Option B by cues 2 and 3, Thresholded Tallying predicts a non-monotonic sequence of preferences (e.g., A -> Tie -> B -> Tie) as the validity threshold is lowered. WADD, in contrast, applies a continuous convex/concave transformation to the validities via the gamma parameter, which cannot capture these discrete step-like preference reversals. The chosen trial pairs maximize the divergence between a step-function tallying approach and a continuous weighted-sum approach.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the 'extremeness' of preferences on trials 5 and 6, where the option with cues {1, 2} is pitted against the option with cues {3, 4, 5}. Under the continuous Weighted Additive (WADD) model, the sum of the weights for cues 1 and 2 is nearly identical to the sum of the weights for cues 3, 4, and 5 across all valid gamma parameters. Consequently, WADD predicts high choice uncertainty (probabilities near 0.5) on these trials, resulting in a low squared deviation from 0.5. In contrast, the Validity-Thresholded Tallying model relies on a hard validity threshold. For 85% of possible thresholds, the tally results in a strict, deterministic win for one of the options (e.g., if tau > 0.65, cues 4 and 5 are ignored, and {1, 2} strictly beats {3}). This discreteness produces extreme choice probabilities (near 1.0 or 0.0). By computing the squared deviation of each subject's choice proportion from 0.5, this metric captures the signature discreteness of the Tallying heuristic, predicting a significantly higher extremeness score than WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_t56_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trials 5 and 6 pit {1, 2} against {3, 4, 5}
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return row['response'] == 0
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return row['response'] == 1
            
        return np.nan
        
    choices = data.apply(get_t56_choice, axis=1)
    
    df_temp = pd.DataFrame({
        'subject_id': data['subject_id'],
        'choice': choices
    }).dropna()
    
    if len(df_temp) == 0:
        return 0.0
        
    # Calculate each subject's proportion of choosing the {1, 2} option
    subj_means = df_temp.groupby('subject_id')['choice'].mean()
    
    # Calculate extremeness as the squared deviation from 0.5
    extremeness = (subj_means - 0.5) ** 2
    
    return float(extremeness.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.0397 (var=0.0026)
- Predicted under pi_4 (simulated): 0.1146 (var=0.0067)
- Observed on real data: 0.0750 (var=0.0053)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    cue_order = np.argsort(-validities)
    
    ttb_matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            if row['response'] == ttb_winner:
                ttb_matches += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_matches / total)

```

**Observed (real) value:** 0.3292 (var=0.0132)
**Predicted under pi_3:** 0.3406 (var=0.0096)
**Predicted under pi_4:** 0.3800 (var=0.0403)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        tally_a = sum(1 for x, y in zip(a, b) if x > y)
        tally_b = sum(1 for x, y in zip(a, b) if y > x)
        
        if tally_a > tally_b:
            tally_pred = 0
        elif tally_b > tally_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        ttb_pred = None
        for x, y in zip(a, b):
            if x > y:
                ttb_pred = 0
                break
            elif y > x:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            agreements.append(1 if resp == tally_pred else 0)
            
    if not agreements:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.7356 (var=0.0121)
**Predicted under pi_3:** 0.7931 (var=0.0137)
**Predicted under pi_4:** 0.6919 (var=0.0412)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify trials where Tallying sees an exact tie (2 wins vs 2 wins)
    t4_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 0))
    t5_mask = (a_tuples == (0, 0, 1, 1, 0)) & (b_tuples == (1, 1, 0, 0, 0))
    t6_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1))
    
    # Under WADD, these trials are NOT ties, because the validities are strictly ordered:
    # w0 > w2 and w1 > w3, etc. 
    # WADD strictly prefers A in T4, B in T5, and A in T6 regardless of the gamma parameter.
    
    c4 = np.sum((data['response'] == 0) & t4_mask)
    c5 = np.sum((data['response'] == 1) & t5_mask)
    c6 = np.sum((data['response'] == 0) & t6_mask)
    
    total = np.sum(t4_mask) + np.sum(t5_mask) + np.sum(t6_mask)
    
    if total == 0:
        return 0.5
        
    return float((c4 + c5 + c6) / total)
```

**Observed (real) value:** 0.6989 (var=0.0397)
**Predicted under pi_3:** 0.7222 (var=0.0223)
**Predicted under pi_4:** 0.6472 (var=0.0365)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    t1_optA = (0, 0, 1, 1, 1)
    t1_optB = (1, 1, 0, 0, 0)
    
    t5_optA = (0, 1, 1, 1, 0)
    t5_optB = (1, 0, 0, 0, 1)
    
    t1_choices = []
    t5_choices = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t1_optA and b == t1_optB:
            t1_choices.append(1 if resp == 0 else 0)
        elif a == t1_optB and b == t1_optA:
            t1_choices.append(1 if resp == 1 else 0)
            
        elif a == t5_optA and b == t5_optB:
            t5_choices.append(1 if resp == 0 else 0)
        elif a == t5_optB and b == t5_optA:
            t5_choices.append(1 if resp == 1 else 0)
            
    if not t1_choices or not t5_choices:
        return 0.0
        
    return float(np.mean(t5_choices) - np.mean(t1_choices))
```

**Observed (real) value:** 0.1850 (var=0.2183)
**Predicted under pi_3:** 0.2562 (var=0.0685)
**Predicted under pi_4:** 0.1887 (var=0.0688)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments, both theories show strengths and weaknesses, but neither consistently captures the observed data. THEORY 1 (WADD, pi_3) performs well on Experiments 3, 4, and 5, accurately predicting the continuous integration of cues. However, it completely fails on Experiment 1, where the observed extremeness difference is near zero (0.0167), contradicting WADD's prediction of a large difference (0.4017). THEORY 2 (Thresholded Tallying, pi_4) captures Experiment 6 perfectly and is closer to the observed data in Experiment 2, but it underperforms on Experiments 3 and 5. Furthermore, while THEORY 2 was designed to predict a near-zero metric in Experiment 1, its actual simulated prediction (0.2633) missed the mark significantly, suggesting that a strict, deterministic validity threshold combined with simple tallying is too rigid. Because human behavior seems to exhibit both the smooth integration of WADD in some contexts and the discrete, threshold-like behavior in others, a fundamentally new approach is needed that can seamlessly bridge these mechanisms without relying on a hard, deterministic cutoff.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Probabilistic Cue-Dropping' or 'Strategy Selection' theory. Instead of a hard validity threshold, assume that each cue has a probability of being included in the decision process that scales with its validity (e.g., via a logistic function). Once a subset of cues is probabilistically sampled on a given trial, the decision-maker tallies the wins among the sampled cues. Alternatively, propose a mixture model where individuals probabilistically choose between a purely compensatory WADD strategy and a non-compensatory heuristic (like Take-The-Best or Tallying) on a trial-by-trial basis. This would allow the model to produce the near-zero extremeness differences in Experiment 1 (via heuristic trials) while maintaining the graded sensitivity required for Experiments 3-5."
}
```

## Usage

```json
{
  "prompt_token_count": 7709,
  "candidates_token_count": 445,
  "total_token_count": 8891
}
```
