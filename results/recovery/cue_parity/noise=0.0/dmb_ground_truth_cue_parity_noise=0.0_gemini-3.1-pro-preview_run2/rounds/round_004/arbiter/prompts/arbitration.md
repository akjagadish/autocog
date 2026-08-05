# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Tallying with Validity Tie-Breaking: Decision-makers primarily rely on a compensatory but simplified strategy where they count the total number of positive cues for each option (Tallying), ignoring the explicit validities. The option with the higher tally is chosen. Only if both options have an equal number of positive cues does the decision-maker fall back to the provided validities to break the tie, typically by consulting the most valid discriminating cue (Take The Best). This boundedly rational approach explains why options supported by a highly valid cue are often rejected if they have fewer total positive cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    a_wins = np.sum(a)
    b_wins = np.sum(b)
    
    if a_wins > b_wins:
        scores = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        scores = np.array([0.0, 1.0])
    else:
        # Tie-breaker using Take The Best
        val = np.asarray(parameters["validities"], dtype=float)
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner == 0:
            scores = np.array([1.0, 0.0])
        elif winner == 1:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.5, 0.5])
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
Weighted Additive (WADD) theory with non-linear scaling posits that decision-makers integrate all available information by weighting each feature by its validity, but subjectively scale these validities to adjust the degree of compensation. By exponentiating validities by a parameter gamma and then normalizing them, the model flexibly interpolates between equal weighting (Tallying), proportional weighting, and highly non-compensatory weighting (Take The Best) while keeping the integrated scores on a stable scale. Choices are made probabilistically using a softmax over the options' scores.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities with normalization to prevent score shrinkage
    weights = val ** gamma
    weights = weights / np.sum(weights)
    
    # Calculate overall scores by weighting features by their scaled validity
    a_score = np.dot(a, weights)
    b_score = np.dot(b, weights)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 5: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 8: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** This design quantitatively dissociates 'Tallying with Validity Tie-Breaking' from Weighted Additive (WADD) theory by contrasting trials where Tallying and WADD conflict, and by probing the nature of tie-breaking. Under the advocated theory, decision-makers strictly choose the option with the most positive cues (Tallying) regardless of validities; validities are only consulted via Take-The-Best (TTB) when the tallies are tied. Therefore, the advocated model predicts that choice probabilities will be identical across all tally-tie trials as long as TTB selects a winner, regardless of the validity of the tie-breaking cue. WADD, by contrast, integrates all scaled validities. It predicts graded choice probabilities that systematically decrease as the highest discriminating cue shifts to lower validities. Furthermore, in trials where one option has more total cues but the other has fewer, higher-validity cues, the advocated model predicts a categorical preference for the high-tally option, whereas WADD predicts a preference for the high-validity option.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option with the higher total number of positive cues on trials where the tallies are unequal (trials 1, 2, 6, and 7). In the experimental design, these trials are structured such that the option with the *fewer* total cues always possesses the single most valid cue (validity = 0.95). The advocated theory (Tallying) predicts that subjects will predominantly choose the option with the higher tally, yielding a metric value close to 1.0. In contrast, the competing WADD theory, which scales validities non-linearly (with gamma typically > 1), will overwhelmingly favor the most valid cue, thus predicting that subjects will choose the option with the lower tally, yielding a metric value close to 0.0.

Source:
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

### RESULTS
- Predicted under pi_6 (simulated): 0.8267 (var=0.0160)
- Predicted under pi_3 (simulated): 0.2567 (var=0.0343)
- Observed on real data: 0.8892 (var=0.0052)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Weighted Additive (WADD) theory from the competing 'Tallying with Validity Tie-Breaking' model by contrasting scenarios where the two models fundamentally disagree on the integration of evidence. The competing theory posits a strict lexical process: it first compares the sheer number of positive cues (Tallying), and only if these are equal does it consult validities via Take-The-Best (TTB). Thus, it predicts categorical, invariant choice probabilities across all trials where the tally difference is the same (e.g., all trials where A has more cues than B, or all tally-tie trials where TTB favors A). WADD, in contrast, integrates all exponentially scaled validities. It predicts graded choice probabilities that systematically shift as the validities of the specific cues favoring each option change, regardless of the raw tally. By presenting pairs where Option A wins the tally but loses on high-validity cues, and tally-tie pairs where the TTB-determining cue shifts down the validity hierarchy, we can cleanly test whether choices are categorically driven by tallies/ties or continuously driven by integrated validities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times Option B is chosen across Trial 1 and Trial 4. The Competing theory (Tallying with Tie-Breaking) strictly predicts Option A in both trials (Trial 1 because A has more total positive cues, and Trial 4 because despite a tally tie, Option A wins on the most valid cue). In contrast, the Advocated theory (WADD) predicts Option B in at least one of these trials for any value of gamma. For low to intermediate gamma, WADD predicts B on Trial 4 because the combined weight of the 2nd and 3rd cues outweighs the 1st and 5th. For intermediate to high gamma, WADD predicts B on Trial 1 because the 1st and 2nd cues strongly dominate the 3rd, 4th, and 5th. Thus, the mean response across these two trials will be near 0 for the Competing theory, but significantly higher for WADD.

Source:
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

### RESULTS
- Predicted under pi_6 (simulated): 0.1433 (var=0.0117)
- Predicted under pi_3 (simulated): 0.5283 (var=0.0126)
- Observed on real data: 0.4900 (var=0.0032)

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
**Predicted under pi_6:** 0.3781 (var=0.0027)
**Predicted under pi_3:** 0.6323 (var=0.0207)

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
**Predicted under pi_6:** 0.8555 (var=0.0079)
**Predicted under pi_3:** 0.3557 (var=0.0136)

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
**Predicted under pi_6:** 0.6733 (var=0.0393)
**Predicted under pi_3:** 0.1228 (var=0.0284)

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
**Predicted under pi_6:** -0.0117 (var=0.0161)
**Predicted under pi_3:** 0.2517 (var=0.0479)

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
**Predicted under pi_6:** 0.0027 (var=0.0082)
**Predicted under pi_3:** 0.2487 (var=0.0254)

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
**Predicted under pi_6:** 0.8752 (var=0.0073)
**Predicted under pi_3:** 0.7946 (var=0.0205)

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
**Predicted under pi_6:** 0.3372 (var=0.0077)
**Predicted under pi_3:** 0.2997 (var=0.0134)

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
**Predicted under pi_6:** 0.3189 (var=0.0133)
**Predicted under pi_3:** 0.2617 (var=0.0149)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, neither theory consistently captures the observed data. Theory 1 (Tallying, pi_6) correctly predicts the strong preference for the higher-tally option in Experiment 1 (obs=0.889, pi_6=0.826) where WADD completely fails (pi_3=0.256). However, Theory 1 drastically fails on Experiments 2, 4, and 5, where choices show clear sensitivity to cue validities that a strict tallying model misses. Theory 2 (WADD, pi_3) captures the graded, validity-dependent choices in Experiments 2, 3, 4, and 5 much better, but completely misses the categorical tallying behavior in Experiment 1 and the null effect in Experiment 6. Because human decision-makers appear to exhibit strong tallying behavior in some contexts and weighted integration in others, a purely monolithic strategy (either strict Tallying or strict WADD) is degenerate.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Selection' or 'Dual-Process' theory. In this new theory, decision-makers first evaluate the raw tally difference between the two options. If the tally difference is sufficiently large (e.g., >= 1 or 2), they rely on the fast, compensatory Tallying heuristic, ignoring explicit cue validities. However, if the tallies are tied or very close (low discriminability by tally), they switch to a more effortful Weighted Additive (WADD) or Take-The-Best strategy to integrate the validities of the conflicting cues. This hybrid approach will naturally capture the dominant tallying behavior seen in Experiment 1 while allowing for the graded, validity-sensitive responses observed in Experiments 2, 4, and 5."
}
```

## Usage

```json
{
  "prompt_token_count": 9722,
  "candidates_token_count": 408,
  "total_token_count": 11219
}
```
