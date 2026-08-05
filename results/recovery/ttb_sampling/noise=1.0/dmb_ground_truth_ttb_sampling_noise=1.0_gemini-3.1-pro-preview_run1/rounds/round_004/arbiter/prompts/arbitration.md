# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Strategy Mixture Theory: Instead of relying on a single heuristic, decision-makers draw from a repertoire of strategies on a trial-by-trial basis. Specifically, individuals mix between a non-compensatory lexicographic rule (Take-The-Best) and a compensatory rule (Tallying). On any given trial, a subject employs TTB with probability `p_ttb` and Tallying with probability `1 - p_ttb`. This intra-individual strategy variation naturally accounts for the aggregate ~0.50 choice proportions observed in conflict trials where the two heuristics prescribe different options, while a relatively stable mixture proportion across the population explains the low between-subject variance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (TTB)
    order = np.argsort(validities)[::-1]
    score_ttb = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            score_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            score_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    score_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Softmax for TTB
    z_ttb = beta * score_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    prob_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * score_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    prob_tally = e_tally / np.sum(e_tally)
    
    # Mix the two strategies
    p_core = p_ttb * prob_ttb + (1.0 - p_ttb) * prob_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_6
Non-linear Weighted Additive (WADD) Theory with Power Weights and Max-Normalized Validities: Decision-makers integrate all available cues simultaneously in a compensatory manner. The subjective weight assigned to each cue is determined by a power transformation of its objective validity, scaled by a parameter gamma. To ensure gamma operates consistently across experiments with varying validity scales, the objective validities are first normalized by their maximum value. This stable, bounded non-linear mapping ensures that cues with zero validity receive zero subjective weight, while allowing the model to naturally balance the top cue against the remaining cues on conflict trials. A temperature-parameterized softmax choice rule is used to smoothly control stochasticity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Max-normalize validities to make gamma scale-invariant across experiments
    v_norm = validities / np.max(validities)
    
    # Power transformation of normalized validities to subjective weights
    weights = v_norm ** gamma
    weights = weights / np.sum(weights)
    
    # Compute weighted sum of features for each option
    scores = stim @ weights
    
    # Standard softmax choice rule with temperature
    z = scores / tau
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates Strategy Mixture Theory from the Non-linear Weighted Additive (WADD) Theory by varying the compensatory majority while keeping the TTB/Tallying conflict constant. Strategy Mixture assumes individuals mix Take-The-Best (TTB) and Tallying trial-by-trial with a stable probability. Thus, whenever TTB and Tallying conflict, Strategy Mixture predicts a choice probability of ~0.50, regardless of the size of the Tallying majority (e.g., whether Tallying wins by 1 cue or 3 cues). In contrast, WADD integrates all cues. By presenting a series of conflict trials where the highest validity cue is pitted against an increasing number of lower validity cues, WADD is forced to predict varying choice probabilities across these trials (as the sum of weights for the compensatory option grows). Strategy Mixture uniquely predicts a flat ~0.50 choice proportion across all these conflict trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Strategy Mixture Theory predicts that on all conflict trials, the decision-maker mixes between Take-The-Best and Tallying with a stable probability. Since TTB and Tallying consistently conflict across all 8 trial types in this design, the true probability of choosing the TTB-favored option remains constant. Thus, the variance in the proportion of TTB choices across the 8 trial types within a subject should be purely due to binomial sampling noise (relatively low). In contrast, WADD integrates all cues. As the number of compensatory cues pitted against the top cue varies across the 8 trial types, WADD predicts systematic shifts in the choice probabilities. This true variation in probabilities will result in a significantly higher variance of TTB choice proportions across trial types within each subject.

Source:
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    
    def get_ttb_option(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0
        
    df['ttb_pref'] = df.apply(get_ttb_option, axis=1)
    df['chose_ttb'] = (df['response'] == df['ttb_pref']).astype(float)
    df['trial_key'] = df.apply(lambda r: (tuple(r['option_a_ratings']), tuple(r['option_b_ratings'])), axis=1)
    
    props = df.groupby(['subject_id', 'trial_key'])['chose_ttb'].mean().reset_index()
    subj_vars = props.groupby('subject_id')['chose_ttb'].var(ddof=0)
    
    return float(subj_vars.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.0401 (var=0.0003)
- Predicted under pi_6 (simulated): 0.0183 (var=0.0001)
- Observed on real data: 0.0172 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.92, 0.9, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Non-linear Weighted Additive (WADD) Theory from the Strategy Mixture Theory by exploiting WADD's power-transformed, max-normalized validities. By clustering validities into a high group (e.g., 0.95, 0.92, 0.90) and a low group (0.55, 0.50), we can construct trials where Take-The-Best (TTB) and Tallying both agree on one option, yet WADD predicts the exact opposite choice. For instance, if Option A wins on the highest-validity cue and the two lowest-validity cues, it wins on 3 out of 5 cues (favored by Tallying) and wins the best cue (favored by TTB), so Strategy Mixture strongly predicts A. However, WADD with a moderate-to-high gamma parameter will heavily penalize the low-validity cues and strongly prefer Option B, which wins on the second and third highest cues. Additional trials pit TTB against Tallying where Strategy Mixture predicts a rigid 0.5 choice proportion, while WADD makes deterministic predictions that vary across the trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the proportion of Option B choices between Trial 1 and Trial 3. In Trial 1, WADD's non-linear weighting of validities leads it to strongly prefer Option B, whereas Strategy Mixture (relying on TTB and Tallying) strongly prefers Option A. In Trial 3, the preferences are inverted: WADD prefers Option A, while Strategy Mixture prefers Option B. Thus, for WADD, P(B|Trial 1) - P(B|Trial 3) will be highly positive, whereas for Strategy Mixture it will be highly negative. This creates a maximal, stable dissociation between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: A=[1, 0, 0, 1, 1], B=[0, 1, 1, 0, 0]
    # WADD favors B (relies on cues 2 and 3 > cue 1 due to non-linear weighting)
    # Strategy Mixture favors A (TTB favors A, Tallying favors A)
    t1_mask = data['A_key'] == (1, 0, 0, 1, 1)
    
    # Trial 3: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 0]
    # WADD favors A (cues 2 and 3 > cue 1)
    # Strategy Mixture favors B (TTB favors B, Tallying ties)
    t3_mask = data['A_key'] == (0, 1, 1, 0, 0)
    
    t1_resp = data.loc[t1_mask, 'response'].mean()
    t3_resp = data.loc[t3_mask, 'response'].mean()
    
    if pd.isna(t1_resp): t1_resp = 0.5
    if pd.isna(t3_resp): t3_resp = 0.5
    
    return float(t1_resp - t3_resp)

### RESULTS
- Predicted under pi_4 (simulated): -0.5325 (var=0.0410)
- Predicted under pi_6 (simulated): 0.0687 (var=0.0289)
- Observed on real data: -0.0300 (var=0.0316)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f0 = np.array([x[0] for x in data['option_a_ratings']])
    a_f1 = np.array([x[1] for x in data['option_a_ratings']])
    b_f0 = np.array([x[0] for x in data['option_b_ratings']])
    b_f1 = np.array([x[1] for x in data['option_b_ratings']])
    
    a_sum2 = a_f0 + a_f1
    b_sum2 = b_f0 + b_f1
    
    # Identify critical trials (trials 1 and 2) where one option has the two highest 
    # validity features (sum=2) and the other has none of them (sum=0) but wins on the rest.
    mask = ((a_sum2 == 0) & (b_sum2 == 2)) | ((a_sum2 == 2) & (b_sum2 == 0))
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    a_sum2_rel = a_sum2[mask]
    
    # Tallying prefers the option with more features (which here means the one with 0 on the first two features)
    tally_chose_a = (a_sum2_rel == 0) & (responses == 0)
    tally_chose_b = (a_sum2_rel == 2) & (responses == 1)
    
    return float(np.mean(tally_chose_a | tally_chose_b))

```

**Observed (real) value:** 0.4867 (var=0.0082)
**Predicted under pi_4:** 0.4908 (var=0.0117)
**Predicted under pi_6:** 0.4275 (var=0.0242)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if a[0] == 1 and a[1] == 1 and b[0] == 0 and b[1] == 0:
            if resp == 0:
                wadd_choices += 1
            total += 1
        elif a[0] == 0 and a[1] == 0 and b[0] == 1 and b[1] == 1:
            if resp == 1:
                wadd_choices += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(wadd_choices / total)

```

**Observed (real) value:** 0.5089 (var=0.0125)
**Predicted under pi_4:** 0.5067 (var=0.0155)
**Predicted under pi_6:** 0.5344 (var=0.0115)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    return float(np.mean(data['response'] == 0))
```

**Observed (real) value:** 0.5121 (var=0.0035)
**Predicted under pi_4:** 0.5231 (var=0.0036)
**Predicted under pi_6:** 0.5342 (var=0.0057)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        if pred != -1:
            matches.append(1 if r == pred else 0)
    return float(np.mean(matches)) if matches else 0.0
```

**Observed (real) value:** 0.5188 (var=0.0033)
**Predicted under pi_4:** 0.5460 (var=0.0047)
**Predicted under pi_6:** 0.5587 (var=0.0073)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Create a hashable trial identifier
    data['trial_id'] = data.apply(
        lambda row: (tuple(row['option_a_ratings']), tuple(row['option_b_ratings'])), 
        axis=1
    )
    
    # Identify conflict trials where TTB and Tallying prescribe different options.
    # TTB relies on the first cue (index 0). Tallying relies on the sum of cues.
    def is_conflict(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_a = a[0] > b[0]
        ttb_b = b[0] > a[0]
        tally_a = sum(a) > sum(b)
        tally_b = sum(b) > sum(a)
        
        if ttb_a and tally_b:
            return True
        if ttb_b and tally_a:
            return True
        return False

    data['is_conflict'] = data.apply(is_conflict, axis=1)
    conflict_data = data[data['is_conflict']]
    
    if len(conflict_data) == 0:
        return 0.0
        
    # For each subject and each unique conflict trial, compute the choice proportion.
    # response == 0 means A, response == 1 means B. 
    # The mean of response is the proportion of B choices (p_B).
    # The absolute difference from 0.5 measures how deterministic the subject's choice is.
    grouped = conflict_data.groupby(['subject_id', 'trial_id'])['response'].mean().reset_index()
    grouped['extremity'] = (grouped['response'] - 0.5).abs()
    
    # Average the extremity of preferences per subject across all conflict trials,
    # then return the grand mean.
    subj_extremity = grouped.groupby('subject_id')['extremity'].mean()
    return float(subj_extremity.mean())

```

**Observed (real) value:** 0.1010 (var=0.0012)
**Predicted under pi_4:** 0.1108 (var=0.0011)
**Predicted under pi_6:** 0.1458 (var=0.0063)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 2: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t2_mask = a_tuples == (1, 1, 0, 0, 0)
    # Trial 4: A=[0, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t4_mask = a_tuples == (0, 1, 0, 0, 0)
    
    if not t2_mask.any() or not t4_mask.any():
        return 0.0
        
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t4)
```

**Observed (real) value:** -0.0084 (var=0.0161)
**Predicted under pi_4:** 0.0032 (var=0.0287)
**Predicted under pi_6:** 0.0642 (var=0.0186)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Identify conflict trials where TTB and Tallying point to different options
    conflict_a_strs = ['10000', '01110', '11000', '01011', '00111']
    df_conflict = data[data['a_str'].isin(conflict_a_strs)]
    
    if len(df_conflict) == 0:
        return 0.0
        
    # Calculate the mean response for each subject and each trial type
    grouped = df_conflict.groupby(['subject_id', 'a_str'])['response'].mean()
    
    # Calculate within-subject consistency: 4 * (p - 0.5)^2
    # This maps p=0.5 to 0.0 (coin flip) and p=0.0 or 1.0 to 1.0 (deterministic)
    consistency = 4.0 * ((grouped - 0.5) ** 2)
    
    return float(consistency.mean())
```

**Observed (real) value:** 0.0673 (var=0.0007)
**Predicted under pi_4:** 0.0774 (var=0.0020)
**Predicted under pi_6:** 0.1089 (var=0.0153)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: TTB option is '10000', Tally option is '01110'
    t1_ttb_chosen = ((a_str == '10000') & (b_str == '01110') & (data['response'] == 0)) | \
                    ((a_str == '01110') & (b_str == '10000') & (data['response'] == 1))
    t1_trials = ((a_str == '10000') & (b_str == '01110')) | ((a_str == '01110') & (b_str == '10000'))
    
    # Trial 2: TTB option is '11000', Tally option is '00111'
    t2_ttb_chosen = ((a_str == '11000') & (b_str == '00111') & (data['response'] == 0)) | \
                    ((a_str == '00111') & (b_str == '11000') & (data['response'] == 1))
    t2_trials = ((a_str == '11000') & (b_str == '00111')) | ((a_str == '00111') & (b_str == '11000'))
    
    p1 = t1_ttb_chosen.sum() / t1_trials.sum() if t1_trials.sum() > 0 else 0.0
    p2 = t2_ttb_chosen.sum() / t2_trials.sum() if t2_trials.sum() > 0 else 0.0
    
    return float(p2 - p1)
```

**Observed (real) value:** 0.0075 (var=0.0276)
**Predicted under pi_4:** 0.0113 (var=0.0353)
**Predicted under pi_6:** 0.0887 (var=0.0389)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 1 (Strategy Mixture) performs exceptionally well on Experiments 3 through 10, predicting the observed metrics with high precision. However, it fails significantly on Experiment 2, predicting a strong negative difference (-0.53) where the observed data shows nearly zero (-0.03). Theory 2 (Non-linear WADD) performs better on Experiments 1 and 2 but systematically misses the mark on the remaining 8 experiments, often overestimating the extremity or systematicity of the choices. The rigid power-transformation in Theory 2 forces it into overly deterministic predictions on certain cue configurations, while Theory 1's strict reliance on TTB and Tallying fails when cue validities are clustered (as in Exp 2). Because neither theory captures the full spectrum, but Theory 1 is much closer overall, we should replace Theory 2 with a brand-new theory that can bridge this gap.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Sequential Evidence Accumulation' or 'Probabilistic Cue Integration' theory. Instead of a deterministic non-linear weighting (like the current WADD) or a strict mixture of two rigid heuristics (like Strategy Mixture), this theory should assume that decision-makers accumulate evidence by sampling cues probabilistically based on their validities. If a threshold is reached, a decision is made; otherwise, it falls back to a noisy integration of all sampled cues. This would naturally produce the ~0.50 choice proportions on conflict trials (mimicking Strategy Mixture) while allowing sensitivity to cue clustering (fixing the failure on Exp 2)."
}
```

## Usage

```json
{
  "prompt_token_count": 9724,
  "candidates_token_count": 362,
  "total_token_count": 11194
}
```
