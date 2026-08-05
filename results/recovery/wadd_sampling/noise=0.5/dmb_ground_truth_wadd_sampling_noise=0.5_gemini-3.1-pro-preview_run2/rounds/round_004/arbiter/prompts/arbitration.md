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
Probabilistic Strategy Selection Theory: Decision-makers maintain a repertoire of distinct decision heuristics (Take-The-Best, Tallying, and Weighted Additive/WADD). On each trial, they probabilistically sample one strategy from this repertoire according to base-rate strategy weights, and execute it purely. This discrete mixture of strategies accounts for intermediate aggregate choice proportions while naturally predicting exact ties when the sampled strategy itself encounters a tie.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB)
    diff = a - b
    order = np.argsort(val)[::-1]
    ttb_pred = np.array([0.5, 0.5])
    for idx in order:
        if diff[idx] > 0:
            ttb_pred = np.array([1.0, 0.0])
            break
        elif diff[idx] < 0:
            ttb_pred = np.array([0.0, 1.0])
            break
            
    # Tallying
    tal_diff = np.sum(a) - np.sum(b)
    if tal_diff > 0:
        tal_pred = np.array([1.0, 0.0])
    elif tal_diff < 0:
        tal_pred = np.array([0.0, 1.0])
    else:
        tal_pred = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD)
    wadd_diff = np.sum(val * a) - np.sum(val * b)
    if wadd_diff > 0:
        wadd_pred = np.array([1.0, 0.0])
    elif wadd_diff < 0:
        wadd_pred = np.array([0.0, 1.0])
    else:
        wadd_pred = np.array([0.5, 0.5])
        
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    w_arr = np.array([w_ttb, w_tal, w_wadd])
    sum_w = np.sum(w_arr)
    if sum_w == 0:
        p_strat = np.array([1/3, 1/3, 1/3])
    else:
        p_strat = w_arr / sum_w
        
    p_core = p_strat[0] * ttb_pred + p_strat[1] * tal_pred + p_strat[2] * wadd_pred
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_6
Softened Integrated Strategy Value Theory: Decision-makers evaluate options by integrating evidence from multiple heuristics (Take-The-Best, Tallying, and Weighted Additive) into a single continuous subjective value for each option. To prevent non-compensatory heuristics from disproportionately dominating the integrated value, the TTB signal is softened by scaling it according to the normalized validity of the discriminating cue. These integrated values are then compared via a stochastic decision process (softmax) to produce a choice. This allows the model to gracefully capture indifference in delicately balanced trials while still reflecting heuristic-aligned preferences.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    sum_val = np.sum(val)
    
    # Take-The-Best (TTB) Score (softened)
    diff = a - b
    order = np.argsort(val)[::-1]
    score_ttb_a, score_ttb_b = 0.0, 0.0
    for idx in order:
        if diff[idx] > 0:
            score_ttb_a, score_ttb_b = val[idx] / sum_val, 0.0
            break
        elif diff[idx] < 0:
            score_ttb_a, score_ttb_b = 0.0, val[idx] / sum_val
            break
            
    # Tallying Score
    score_tal_a = np.sum(a) / n_features
    score_tal_b = np.sum(b) / n_features
    
    # Weighted Additive (WADD) Score
    score_wadd_a = np.sum(val * a) / sum_val
    score_wadd_b = np.sum(val * b) / sum_val
    
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize weights
    w_sum = w_ttb + w_tal + w_wadd
    if w_sum > 0:
        w_ttb /= w_sum
        w_tal /= w_sum
        w_wadd /= w_sum
    else:
        w_ttb = w_tal = w_wadd = 1.0 / 3.0
        
    # Integrated Option Values
    score_a = w_ttb * score_ttb_a + w_tal * score_tal_a + w_wadd * score_wadd_a
    score_b = w_ttb * score_ttb_b + w_tal * score_tal_b + w_wadd * score_wadd_b
    
    scores = np.array([score_a, score_b])
    
    # Single Softmax Decision Process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    # Lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** This design sharply dissociates the Advocated Probabilistic Strategy Selection Theory from the Competing Softened Integrated Strategy Value Theory by exploiting their fundamentally different treatments of the Take-The-Best (TTB) heuristic. In the Advocated theory, TTB is an all-or-nothing categorical rule: if TTB selects an option, it contributes its full base-rate probability to that option, regardless of which cue in the hierarchy broke the tie. In contrast, the Competing theory scales the TTB score by the normalized validity of the specific discriminating cue. We design a sequence of 'TTB-dilution' trials where the cue that breaks the TTB tie shifts progressively from the most valid feature down to the third most valid feature. In all these trials, both Tallying and WADD strongly oppose the TTB choice. The Competing theory predicts a progressively weakening preference for the TTB-favored option as the discriminating cue's validity drops, whereas the Advocated theory predicts a constant, undiluted TTB contribution across the sequence.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric contrasts the TTB-favored choice probability between Trial 4 and Trial 1. In both trials, TTB is perfectly opposed by Tallying and WADD. Under the Advocated probabilistic model, strategy outcomes are categorical: P(A|T1) and P(B|T4) both exactly equal the base-rate probability of sampling TTB, making their difference zero regardless of strategy weights. Under the Competing integrated value model, the TTB signal is identical in both trials (cue 0 breaks the tie). However, the magnitude of the opposing Tallying and WADD scores is much stronger in Trial 1 than in Trial 4. Because the TTB component exactly cancels out when taking the difference, the integrated value penalty against the TTB-favored option is systematically and reliably larger in Trial 1, pulling P(A|T1) strictly lower than P(B|T4). This guarantees a robustly positive difference in the Competing model, with extremely low parameter-induced variance since the w_ttb parameter mathematically cancels out.

Source:
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: A=(1,0,0,0,0) vs B=(0,1,1,1,0)
    # TTB favors A. Tal and WADD favor B.
    t1_dir = (a_keys == (1, 0, 0, 0, 0)) & (b_keys == (0, 1, 1, 1, 0))
    t1_rev = (a_keys == (0, 1, 1, 1, 0)) & (b_keys == (1, 0, 0, 0, 0))
    
    t1_a_choices = 0
    t1_total = 0
    if t1_dir.sum() > 0:
        t1_a_choices += (data.loc[t1_dir, 'response'] == 0).sum()
        t1_total += t1_dir.sum()
    if t1_rev.sum() > 0:
        t1_a_choices += (data.loc[t1_rev, 'response'] == 1).sum()
        t1_total += t1_rev.sum()
        
    p_a_t1 = t1_a_choices / t1_total if t1_total > 0 else 0.5
    
    # Trial 4: A=(0,0,1,1,1) vs B=(1,1,0,0,0)
    # TTB favors B. Tal and WADD favor A.
    t4_dir = (a_keys == (0, 0, 1, 1, 1)) & (b_keys == (1, 1, 0, 0, 0))
    t4_rev = (a_keys == (1, 1, 0, 0, 0)) & (b_keys == (0, 0, 1, 1, 1))
    
    t4_b_choices = 0
    t4_total = 0
    if t4_dir.sum() > 0:
        t4_b_choices += (data.loc[t4_dir, 'response'] == 1).sum()
        t4_total += t4_dir.sum()
    if t4_rev.sum() > 0:
        t4_b_choices += (data.loc[t4_rev, 'response'] == 0).sum()
        t4_total += t4_rev.sum()
        
    p_b_t4 = t4_b_choices / t4_total if t4_total > 0 else 0.5
    
    return float(p_b_t4 - p_a_t1)

### RESULTS
- Predicted under pi_5 (simulated): -0.0063 (var=0.0271)
- Predicted under pi_6 (simulated): 0.1512 (var=0.0372)
- Observed on real data: 0.1250 (var=0.0447)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 2: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Rationale:** This design sharply dissociates the Advocated Softened Integrated Strategy Value Theory from the Competing Probabilistic Strategy Selection Theory by exploiting their fundamentally different treatments of the Weighted Additive (WADD) score magnitude. The Competing theory assumes decision-makers sample pure, discrete strategies; thus, if WADD favors Option A, the WADD heuristic outputs a categorical preference for A regardless of whether the WADD score difference is 0.05 or 0.85. In contrast, the Advocated theory integrates the continuous WADD score into a softmax decision rule, meaning the probability of choosing A scales directly with the magnitude of the WADD difference. We employ 'Tallying-Tied' trials where Take-The-Best (TTB) consistently points to Option A and Tallying is perfectly tied, but the magnitude of the WADD difference favoring Option A varies from marginal to massive. The Competing theory is mathematically constrained to predict the exact same choice probabilities across these trials, while the Advocated theory predicts a smoothly increasing preference for Option A.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric exploits a rigid algebraic constraint in the Competing (Probabilistic Strategy Selection) theory. In the Competing theory, the probability of choosing the TTB-favored option when Tally is tied is P_tied = (1-eps)*(w_ttb + w_wadd + 0.5*w_tal) + 0.5*eps. When Tally strictly opposes TTB, the probability is P_opposes = (1-eps)*(w_ttb + w_wadd) + 0.5*eps. Substituting these into the contrast `1 - 2*P_tied + P_opposes` algebraically collapses to exactly `0.5 * eps`, which is entirely invariant to the subject's strategy weights (w_ttb, w_tal, w_wadd). Because it cancels out the strategy weights, the between-subject variance for the Competing theory is strictly bounded to the variance of epsilon plus binomial noise.

In contrast, the Advocated (Softened Integrated Strategy Value) theory integrates the Tallying penalty (-1/6) directly into the continuous value score before applying a non-linear softmax. This causes P_opposes to drop significantly lower than what a linear mixture would predict, making the value of `1 - 2*P_tied + P_opposes` substantially larger than 0.5 * eps. By isolating this exact linear combination, we drastically reduce the Competing theory's variance while maintaining a large mean gap, yielding a highly significant Welch's t-statistic.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.5])
    
    # Convert lists to numpy arrays for vectorized operations
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # The first cue (index 0) has the highest validity (0.95) and is always the
    # discriminating cue for Take-The-Best (TTB) in this specific experimental design.
    # Favored option is 0 (A) if A has the cue (a_ratings[:, 0] == 1), else 1 (B).
    favored_option = 1 - a_ratings[:, 0]
    chose_favored = (data['response'] == favored_option).astype(float)
    
    # Identify Tally-tied vs Tally-opposes trials
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    tally_tied = (sum_a == sum_b)
    
    # Calculate WADD difference magnitude
    wadd_a = (a_ratings * val).sum(axis=1)
    wadd_b = (b_ratings * val).sum(axis=1)
    wadd_diff_abs = np.abs(wadd_a - wadd_b)
    
    # Isolate Trial types 1 & 4 (Tally tied, marginal WADD diff ~ 0.05)
    is_t1_t4 = tally_tied & (wadd_diff_abs < 0.2)
    
    # Isolate Trial types 7 & 8 (Tally strictly opposes TTB)
    is_t7_t8 = ~tally_tied
    
    if is_t1_t4.sum() == 0 or is_t7_t8.sum() == 0:
        return 0.0
        
    p_tied_low = chose_favored[is_t1_t4].mean()
    p_opposes = chose_favored[is_t7_t8].mean()
    
    # Theoretical contrast metric
    return float(1.0 - 2.0 * p_tied_low + p_opposes)


### RESULTS
- Predicted under pi_5 (simulated): 0.1258 (var=0.0459)
- Predicted under pi_6 (simulated): 0.4025 (var=0.0480)
- Observed on real data: 0.5217 (var=0.1884)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    # The validities are monotonically decreasing from left to right.
    # Thus, Take The Best (TTB) will choose based on the first feature
    # (from index 0 to 4) that discriminates between A and B.
    diff = a_ratings - b_ratings
    
    # Find the index of the first discriminating cue (where difference is non-zero)
    first_diff_idx = (diff != 0).argmax(axis=1)
    
    # Extract the difference value at that first discriminating cue
    first_diff_val = np.take_along_axis(diff, first_diff_idx[:, None], axis=1).squeeze()
    
    # If first_diff_val > 0 (A has the feature, B does not), TTB predicts A (response 0).
    # If first_diff_val < 0 (B has the feature, A does not), TTB predicts B (response 1).
    ttb_predictions = (first_diff_val < 0).astype(int)
    
    # Return the proportion of choices that match the TTB prediction.
    return float(np.mean(responses == ttb_predictions))
```

**Observed (real) value:** 0.4167 (var=0.0087)
**Predicted under pi_5:** 0.3848 (var=0.0142)
**Predicted under pi_6:** 0.4444 (var=0.0069)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    diff = a_sums - b_sums
    mask = diff != 0
    
    if not mask.any():
        return 0.5
        
    tallying_pred = (diff < 0).astype(int)
    accuracy = (data.loc[mask, 'response'] == tallying_pred[mask]).mean()
    
    return float(accuracy)
```

**Observed (real) value:** 0.6044 (var=0.0126)
**Predicted under pi_5:** 0.6494 (var=0.0221)
**Predicted under pi_6:** 0.6067 (var=0.0138)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying sees a tie (equal number of features)
    tied_trials = sum_a == sum_b
    
    if tied_trials.sum() == 0:
        return 0.5
        
    # Calculate the proportion of times Option A was chosen in these tied trials
    # (Response 0 means Option A was chosen)
    prob_a_chosen = (data.loc[tied_trials, 'response'] == 0).mean()
    
    return float(prob_a_chosen)
```

**Observed (real) value:** 0.5611 (var=0.0131)
**Predicted under pi_5:** 0.7311 (var=0.0098)
**Predicted under pi_6:** 0.5839 (var=0.0084)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Create a string representation of option A's features to identify trials safely
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Trial 4: A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    p_a_t4 = 1.0 - data.loc[data['A_str'] == '11000', 'response'].mean()
    
    # Trial 5: A=[1, 0, 0, 1, 1] vs B=[0, 1, 1, 0, 0]
    p_a_t5 = 1.0 - data.loc[data['A_str'] == '10011', 'response'].mean()
    
    # Return the sum of probabilities of choosing A in these two trials
    return float(p_a_t4 + p_a_t5)
```

**Observed (real) value:** 1.0350 (var=0.0694)
**Predicted under pi_5:** 1.2200 (var=0.0368)
**Predicted under pi_6:** 1.1587 (var=0.0375)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # a_cue_0 is 1 if option A has the most valid cue, 0 otherwise.
    a_cue_0 = data['option_a_ratings'].apply(lambda x: x[0])
    # response is 0 if A was chosen, 1 if B was chosen.
    # If response == a_cue_0, the subject chose the option that LACKS the most valid cue.
    # For example, if A has it (1) and they chose B (1), they chose the option without it.
    # If A lacks it (0) and they chose A (0), they chose the option without it.
    return float((data['response'] == a_cue_0).mean())
```

**Observed (real) value:** 0.5792 (var=0.0095)
**Predicted under pi_5:** 0.5790 (var=0.0152)
**Predicted under pi_6:** 0.5390 (var=0.0081)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert rating lists to string representations to identify specific trials
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Identify the 'linear-tie' trials
    # Trial 1: Cues 1 & 4 vs Cues 2 & 3 (sum of validities = 1.5 for both)
    mask_t1 = (data['A_str'] == '10010') & (data['B_str'] == '01100')
    # Trial 5: Cues 2 & 4 vs Cues 1 & 5 (sum of validities = 1.4 for both)
    mask_t5 = (data['A_str'] == '01010') & (data['B_str'] == '10001')
    
    dev = 0.0
    if mask_t1.sum() > 0:
        p_A_t1 = (data.loc[mask_t1, 'response'] == 0).mean()
        dev += abs(p_A_t1 - 0.5)
        
    if mask_t5.sum() > 0:
        p_A_t5 = (data.loc[mask_t5, 'response'] == 0).mean()
        dev += abs(p_A_t5 - 0.5)
        
    return float(dev)
```

**Observed (real) value:** 0.0167 (var=0.0257)
**Predicted under pi_5:** 0.2633 (var=0.0436)
**Predicted under pi_6:** 0.1950 (var=0.0292)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    target_mask = a_tuples.isin([(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (1, 0, 1, 0, 0)])
    target_data = data[target_mask]
    if len(target_data) == 0:
        return 0.0
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.4058 (var=0.0101)
**Predicted under pi_5:** 0.4204 (var=0.0242)
**Predicted under pi_6:** 0.4692 (var=0.0128)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.65, 0.55, 0.5])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    
    has_nonzero = np.any(diff != 0, axis=1)
    first_nonzero_idx = np.argmax(diff != 0, axis=1)
    
    ttb_sign = diff[np.arange(len(diff)), first_nonzero_idx]
    ttb_choice = np.where(ttb_sign > 0, 0, 1)
    ttb_choice = np.where(has_nonzero, ttb_choice, -1)
    
    tal_diff = np.sum(a_mat, axis=1) - np.sum(b_mat, axis=1)
    tal_choice = np.where(tal_diff > 0, 0, np.where(tal_diff < 0, 1, -1))
    
    wadd_diff = np.sum(val * a_mat, axis=1) - np.sum(val * b_mat, axis=1)
    wadd_choice = np.where(wadd_diff > 0, 0, np.where(wadd_diff < 0, 1, -1))
    
    mask = (tal_choice == wadd_choice) & (tal_choice != -1) & (ttb_choice != -1) & (ttb_choice != tal_choice)
    
    if not np.any(mask):
        return 0.0
        
    responses = data['response'].values[mask]
    ttb_choices = ttb_choice[mask]
    
    return float(np.mean(responses == ttb_choices))
```

**Observed (real) value:** 0.3950 (var=0.0100)
**Predicted under pi_5:** 0.3450 (var=0.0207)
**Predicted under pi_6:** 0.4325 (var=0.0112)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of experiments, Theory 2 (Softened Integrated Strategy Value Theory) significantly outperforms Theory 1 (Probabilistic Strategy Selection Theory). For instance, in Experiment 1, Theory 2 correctly predicts a positive difference (0.1512 vs observed 0.1250) while Theory 1 predicts zero (-0.0063). In Experiment 2, Theory 2 captures the larger contrast metric much better than Theory 1. Theory 2 also provides much closer estimates for Experiments 3, 5, 7, and 10. However, both models struggle with Experiment 8, where the observed metric is near zero (0.0167), but both models predict substantial deviations (0.2633 and 0.1950). Because Theory 1 is fundamentally misaligned with the data (predicting rigid strategy execution), it should be replaced entirely with a new theory that can compete with the integrated value approach while possibly addressing the shortcomings in Experiment 8.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Sequential Evidence Accumulation Theory'. Instead of selecting a single heuristic (Theory 1) or statically integrating all heuristics (Theory 2), assume decision-makers evaluate features sequentially in order of validity. They maintain a running tally of evidence weighted by validity. If the accumulated evidence exceeds an internal confidence threshold at any point, they stop and make a choice (resembling a softened TTB). If all features are exhausted without crossing the threshold, they choose based on the final accumulated sum (resembling WADD/Tallying). This provides a mechanistic, process-based explanation for why choices look like a blend of TTB and compensatory strategies, naturally capturing threshold effects that might explain the near-zero deviations in certain balanced trial designs like Experiment 8."
}
```

## Usage

```json
{
  "prompt_token_count": 10772,
  "candidates_token_count": 405,
  "total_token_count": 12021
}
```
