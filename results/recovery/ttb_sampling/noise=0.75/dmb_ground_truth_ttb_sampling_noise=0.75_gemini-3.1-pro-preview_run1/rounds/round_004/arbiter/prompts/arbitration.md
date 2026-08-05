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
Take The Best (TTB) with Probabilistic Stopping: Decision-makers use a lexicographic heuristic, ranking features by subjective validity and stopping at the first discriminating feature. However, rather than making a strictly deterministic choice based on this feature, the decision is probabilistic. The probability of choosing the winning option scales with the validity of that discriminating feature via a softmax function with a highly regularized inverse temperature (beta). This allows confidence to vary depending on how valid the deciding feature is, capturing empirical noise without relying entirely on a global random lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    beta = float(parameters["beta"])
    
    a, b = stim[0], stim[1]
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            scores = np.array([0.0, validities[f]])
            break
            
    # If no feature discriminates, default to uniform guessing
    if scores[0] == scores[1]:
        p_core = np.array([0.5, 0.5])
    else:
        # Probabilistic choice scaling with the validity of the discriminating feature
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    # Apply lapse rate
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_6
Strict Take-The-Best with Uniform Lapse: Decision-makers rely on a strict lexicographic heuristic, ranking cues by subjective validity and making a deterministic choice based solely on the highest-validity discriminating cue. To account for empirical noise and inattention, decisions are subject to a uniform lapse rate, where the decision-maker simply guesses randomly on a fixed proportion of trials rather than scaling their confidence by the cue's validity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[f] > a[f]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate TTB with Probabilistic Stopping from Strict TTB with Uniform Lapse, this design varies the validity of the first discriminating feature across trials. Under the competing Strict TTB model, the choice probability for the TTB-favored option is determined by a uniform lapse rate and thus remains constant across all trials, regardless of whether the deciding feature is highly valid or barely better than chance. In contrast, the advocated Probabilistic Stopping model scales the choice probability via a softmax function based on the validity of the discriminating feature. By creating a set of trials where the highest-validity discriminating feature systematically shifts from the most valid cue down to the least valid cue, we can observe whether subjects' choice consistency degrades as the deciding cue's validity decreases, which is uniquely predicted by the advocated model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Strict TTB makes a deterministic core choice (100% confidence) based on the highest-validity discriminating cue, relying entirely on a uniform lapse rate to generate noise. In contrast, Probabilistic Stopping scales its core confidence by the validity of the deciding cue using a softmax function. Because the deciding cue's validity is always less than 1.0, Probabilistic Stopping's internal confidence is structurally diluted compared to Strict TTB. When subjected to the same uniform lapse rate distribution, this dilution results in a significantly lower overall proportion of TTB-consistent choices for Probabilistic Stopping. Thus, the overall TTB-consistency rate serves as a highly robust discriminator with minimal binomial variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert ratings to 2D numpy arrays
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    # Find the first discriminating feature for each trial
    # Validities are sorted descending, so the first feature with a difference is the deciding one
    diff = A - B
    is_diff = diff != 0
    first_diff_idx = np.argmax(is_diff, axis=1)
    has_diff = np.any(is_diff, axis=1)
    
    # Extract the sign of the first difference to determine TTB prediction
    row_indices = np.arange(len(diff))
    first_diff_sign = diff[row_indices, first_diff_idx]
    
    # TTB favors Option A (0) if A > B (sign == 1)
    # TTB favors Option B (1) if B > A (sign == -1)
    ttb_favors_b = (first_diff_sign == -1).astype(int)
    
    # Calculate consistency
    consistent = (resp == ttb_favors_b)
    
    if not np.any(has_diff):
        return 0.5
        
    return float(np.mean(consistent[has_diff]))

### RESULTS
- Predicted under pi_3 (simulated): 0.6004 (var=0.0090)
- Predicted under pi_6 (simulated): 0.7540 (var=0.0223)
- Observed on real data: 0.5854 (var=0.0039)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 3: A=[0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 10: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate Strict TTB with Uniform Lapse (advocated) from TTB with Probabilistic Stopping (competing), this design exploits the parameter bounds of the competing model. The competing model scales choice probability via a softmax function based on the validity of the discriminating cue, with an inverse temperature (beta) capped at 2.5. By including trials where the only discriminating cue has a very low validity (e.g., 0.55), the maximum choice probability the competing model can predict for the TTB-favored option is strictly bounded (around 0.80). In contrast, the advocated Strict TTB model uses a global uniform lapse rate, allowing it to predict high choice consistency (e.g., 0.95) even when the deciding cue has low validity. Varying the highest discriminating cue from high to low validity across trials will reveal whether choice consistency remains flat (uniquely predicted by the advocated model) or degrades (required by the competing model).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the Chi-square statistic for the independence between the deciding cue's validity and the choice made. Under the advocated theory (Strict TTB with Uniform Lapse), the probability of choosing the TTB-favored option is entirely driven by a global uniform lapse rate and is independent of the cue's validity. Thus, the Chi-square statistic will be close to its degrees of freedom (approx. 4) on both per-subject and pooled data. Under the competing theory (TTB with Probabilistic Stopping), the choice probability scales with the deciding cue's validity. Because Chi-square is an extensive statistic (it scales with sample size), the dependence in the competing theory will produce a massive Chi-square value on the pooled dataset (~2250 trials), while remaining moderately small on the per-subject slices. This guarantees a huge point-estimate gap relative to the tight between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # The validities are fixed in this experimental design
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    def get_deciding_validity(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(5):
            if a[i] != b[i]:
                return validities[i]
        return 0.0

    val = data.apply(get_deciding_validity, axis=1)
    chose_a = (data['response'] == 0).astype(int)
    
    df = pd.DataFrame({'val': val, 'chose_a': chose_a})
    
    # Calculate observed counts for the contingency table
    crosstab = pd.crosstab(df['val'], df['chose_a'])
    
    # If the subject always chose A (or B), variables are perfectly independent (zero variance)
    if crosstab.shape[1] < 2 or crosstab.shape[0] < 2:
        return 0.0
        
    observed = crosstab.values
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total = observed.sum()
    
    expected = np.outer(row_sums, col_sums) / total
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.sum((observed - expected)**2 / expected)
        
    if np.isnan(chi2) or np.isinf(chi2):
        return 0.0
        
    return float(chi2)

### RESULTS
- Predicted under pi_3 (simulated): 8.0794 (var=8.5181)
- Predicted under pi_6 (simulated): 0.5861 (var=11.3097)
- Observed on real data: 4.5718 (var=6.8918)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_dev(df):
        m3 = df['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x)) == (1, 0, 0, 0, 0)
        m4 = df['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x)) == (0, 1, 1, 0, 0)
        
        dev = 0.0
        count = 0
        if m3.sum() > 0:
            dev += abs((df.loc[m3, 'response'] == 0).mean() - 0.5)
            count += 1
        if m4.sum() > 0:
            dev += abs((df.loc[m4, 'response'] == 0).mean() - 0.5)
            count += 1
            
        return dev / count if count > 0 else 0.0

    return float(data.groupby('subject_id').apply(get_dev).mean())
```

**Observed (real) value:** 0.1217 (var=0.0056)
**Predicted under pi_3:** 0.1783 (var=0.0133)
**Predicted under pi_6:** 0.2712 (var=0.0199)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    count = 0
    match_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        sum_a = sum(a)
        sum_b = sum(b)
        
        # Isolate conflict trials where one option has fewer features (sum=2 vs sum=3)
        # but the fewer features include the most predictive one (index 0 is 1).
        if sum_a == 2 and sum_b == 3 and a[0] == 1:
            match_trials += 1
            if resp == 0:  # Chose the option with fewer but more valid features
                count += 1
        elif sum_b == 2 and sum_a == 3 and b[0] == 1:
            match_trials += 1
            if resp == 1:  # Chose the option with fewer but more valid features
                count += 1
                
    if match_trials == 0:
        return 0.0
    return count / match_trials
```

**Observed (real) value:** 0.6062 (var=0.0088)
**Predicted under pi_3:** 0.6200 (var=0.0141)
**Predicted under pi_6:** 0.7544 (var=0.0186)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    trial_count = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        sa = sum(a)
        sb = sum(b)
        
        if sa == sb:
            continue
            
        ttb = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
                
        # Target "compensatory" trials where TTB chooses the option with fewer positive features
        if (ttb == 0 and sa < sb) or (ttb == 1 and sb < sa):
            trial_count += 1
            if row['response'] == ttb:
                match_count += 1
                
    if trial_count == 0:
        return 0.5
        
    return float(match_count / trial_count)
```

**Observed (real) value:** 0.5920 (var=0.0046)
**Predicted under pi_3:** 0.6173 (var=0.0123)
**Predicted under pi_6:** 0.7140 (var=0.0253)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    v = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    wadd_matches = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        resp = row['response']
        
        wadd_a = np.dot(a, v)
        wadd_b = np.dot(b, v)
        wadd_pred = 0 if wadd_a > wadd_b else 1
        
        diff = a - b
        ttb_pred = None
        for i in range(len(v)):
            if diff[i] == 1:
                ttb_pred = 0
                break
            elif diff[i] == -1:
                ttb_pred = 1
                break
                
        if ttb_pred is not None and wadd_pred != ttb_pred:
            wadd_matches.append(1 if resp == wadd_pred else 0)
            
    if not wadd_matches:
        return 0.5
        
    return float(np.mean(wadd_matches))
```

**Observed (real) value:** 0.4011 (var=0.0034)
**Predicted under pi_3:** 0.3628 (var=0.0139)
**Predicted under pi_6:** 0.2458 (var=0.0247)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_first_disc(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]: return i
        return -1
        
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1
        
    def get_tally_diff(row):
        a = sum(row['option_a_ratings'])
        b = sum(row['option_b_ratings'])
        ttb = get_ttb_choice(row)
        if ttb == 0:
            return a - b
        elif ttb == 1:
            return b - a
        return 0

    first_disc = data.apply(get_first_disc, axis=1)
    tally_diffs = data.apply(get_tally_diff, axis=1)
    ttb_choices = data.apply(get_ttb_choice, axis=1)
    
    is_ttb_chosen = (data['response'] == ttb_choices)
    
    # Only look at trials where the first discriminating feature is feature 0
    mask_0 = first_disc == 0
    
    # Trials where Tallying agrees with TTB (Tally diff > 0)
    mask_agree = mask_0 & (tally_diffs > 0)
    # Trials where Tallying strongly disagrees with TTB (Tally diff < -1)
    mask_disagree = mask_0 & (tally_diffs < -1)
    
    if mask_agree.sum() == 0 or mask_disagree.sum() == 0:
        return 0.0
        
    return float(is_ttb_chosen[mask_agree].mean() - is_ttb_chosen[mask_disagree].mean())
```

**Observed (real) value:** 0.0333 (var=0.0361)
**Predicted under pi_3:** 0.0167 (var=0.0303)
**Predicted under pi_6:** 0.0050 (var=0.0248)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of features for A and B
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Conflict trials: Tallying prefers B (sum B > sum A)
    # Congruent trials: Tallying prefers A (sum A > sum B)
    conflict_mask = b_sums > a_sums
    congruent_mask = a_sums > b_sums
    
    # response == 1 means choosing B
    p_b_conflict = data.loc[conflict_mask, 'response'].mean()
    p_b_congruent = data.loc[congruent_mask, 'response'].mean()
    
    if np.isnan(p_b_conflict):
        p_b_conflict = 0.0
    if np.isnan(p_b_congruent):
        p_b_congruent = 0.0
        
    return float(p_b_conflict - p_b_congruent)
```

**Observed (real) value:** -0.0124 (var=0.0079)
**Predicted under pi_3:** -0.0009 (var=0.0098)
**Predicted under pi_6:** -0.0107 (var=0.0079)

### Experiment 9
**Design**
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    t1_a, t1_b = (1, 0, 1, 1, 1), (0, 1, 1, 1, 1)
    t3_a, t3_b = (1, 0, 0, 0, 0), (0, 1, 1, 1, 1)
    
    t5_a, t5_b = (1, 1, 0, 1, 1), (1, 0, 1, 1, 1)
    t6_a, t6_b = (1, 1, 0, 0, 0), (1, 0, 1, 1, 1)
    
    t8_a, t8_b = (1, 1, 1, 0, 1), (1, 1, 0, 1, 1)
    t9_a, t9_b = (1, 1, 1, 0, 0), (1, 1, 0, 1, 1)
    
    def get_pa(a, b):
        mask = (data['A_tuple'] == a) & (data['B_tuple'] == b)
        if mask.sum() == 0:
            return 0.5
        return data.loc[mask, 'chose_A'].mean()
        
    diff1 = get_pa(t1_a, t1_b) - get_pa(t3_a, t3_b)
    diff2 = get_pa(t5_a, t5_b) - get_pa(t6_a, t6_b)
    diff3 = get_pa(t8_a, t8_b) - get_pa(t9_a, t9_b)
    
    return float(diff1 + diff2 + diff3)
```

**Observed (real) value:** -0.0680 (var=0.1734)
**Predicted under pi_3:** 0.0020 (var=0.1186)
**Predicted under pi_6:** 0.0180 (var=0.0931)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    top_cue_chosen = []
    opposing_cues = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        top_cue_option = -1
        opp = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                if top_cue_option == -1:
                    top_cue_option = 0 if a[i] == 1 else 1
                else:
                    if top_cue_option == 0 and b[i] == 1:
                        opp += 1
                    elif top_cue_option == 1 and a[i] == 1:
                        opp += 1
                        
        top_cue_chosen.append(1 if resp == top_cue_option else 0)
        opposing_cues.append(opp)
        
    df = pd.DataFrame({'top_chosen': top_cue_chosen, 'opp': opposing_cues})
    
    mean_low = df[df['opp'] == 1]['top_chosen'].mean()
    mean_high = df[df['opp'] >= 3]['top_chosen'].mean()
    
    if pd.isna(mean_low) or pd.isna(mean_high):
        return 0.0
        
    return float(mean_low - mean_high)
```

**Observed (real) value:** 0.0013 (var=0.0233)
**Predicted under pi_3:** -0.0192 (var=0.0085)
**Predicted under pi_6:** -0.0125 (var=0.0141)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of experiments, Theory 1 (pi_3) significantly outperforms Theory 2 (pi_6). Theory 1's mechanism of scaling choice probability by the validity of the discriminating cue via a softmax function closely matches the empirical choice proportions, whereas Theory 2's strict TTB with a uniform lapse rate systematically overpredicts TTB consistency (as seen in Experiments 1, 4, and 5). However, both theories struggle to fully capture the compensatory effects observed in Experiments 9 and 10, where the number of opposing cues influences choices, leading to slight deviations from pure lexicographic predictions. Given that Theory 1 is a strong baseline for lexicographic strategies, Theory 2 should be replaced with a brand-new theory that can account for compensatory mechanisms.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new compensatory theory, such as a Weighted Additive (WADD) model with probabilistic choice, or a mixture model of TTB and WADD. Unlike the strict or probabilistic lexicographic stopping in the current theories, this new theory should compute a global utility for each option by summing all feature values weighted by their subjective validities, followed by a softmax choice rule. This will naturally account for the compensatory effects and the influence of opposing cues observed in Experiments 6, 9, and 10, providing a fundamentally distinct competitor to Theory 1."
}
```

## Usage

```json
{
  "prompt_token_count": 10509,
  "candidates_token_count": 314,
  "total_token_count": 11897
}
```
