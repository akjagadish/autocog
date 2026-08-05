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


## THEORY 2 — pi_4
Strategy Mixture Model (TTB and Tallying): Decision-makers exhibit heterogeneous strategy use, with the population consisting of a mix of Take-The-Best (TTB) users and Tallying users. Rather than a pure one-reason heuristic or a fully compensatory process, aggregate behavior reflects a probabilistic mixture. On any given trial, a subject's choice is a weighted blend of a lexicographic TTB process (which chooses based on the single most valid discriminating cue) and a Tallying process (which counts the number of feature-wise wins for each option). Allowing the mixture weight to vary freely between 0 and 1 across individuals captures the empirical finding that aggregate behavior is predominantly non-compensatory but softened by a subset of subjects who rely more heavily on compensatory tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    w_tally = float(parameters["w_tally"])
    epsilon = float(parameters["epsilon"])
    
    # --- TTB Prediction ---
    order = np.argsort(validities)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for f in order:
        if a[f] > b[f]:
            ttb_scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            ttb_scores = np.array([0.0, validities[f]])
            break
            
    if ttb_scores[0] == ttb_scores[1]:
        p_ttb = np.array([0.5, 0.5])
    else:
        z = beta_ttb * (ttb_scores - ttb_scores.max())
        e = np.exp(z)
        p_ttb = e / e.sum()
        
    # --- Tallying Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    
    if tally_scores[0] == tally_scores[1]:
        p_tally = np.array([0.5, 0.5])
    else:
        z = beta_tally * (tally_scores - tally_scores.max())
        e = np.exp(z)
        p_tally = e / e.sum()
        
    # --- Mixture ---
    p_core = (1.0 - w_tally) * p_ttb + w_tally * p_tally
    
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
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate TTB with Probabilistic Stopping from the Strategy Mixture Model (TTB + Tallying), this design varies the number of features favoring the non-TTB option while keeping the highest valid discriminating feature constant across sets of trials. Under the advocated TTB model, the probability of choosing the TTB-favored option depends solely on the validity of the single best discriminating feature; thus, choice probabilities should remain constant across trials that share the same top discriminator. In contrast, the Mixture model predicts that the choice probability will shift systematically as the Tallying component pulls the decision toward the option with the greater total number of winning features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By restricting the analysis to trials where the first discriminating feature is identical (feature 0), the pure Take-The-Best (TTB) model with probabilistic stopping predicts an identical probability of choosing the TTB-favored option across these trials, yielding a difference of 0. In contrast, the Strategy Mixture Model predicts that subjects will be systematically influenced by the Tallying component. Thus, the probability of choosing the TTB option will be higher when Tallying agrees with TTB than when Tallying strongly disagrees, yielding a strictly positive difference.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.0167 (var=0.0303)
- Predicted under pi_4 (simulated): 0.3350 (var=0.0737)
- Observed on real data: 0.0333 (var=0.0361)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Strategy Mixture Model (TTB + Tallying) from TTB with Probabilistic Stopping, this design varies the total number of features favoring each option while holding the highest-validity discriminating feature constant across subsets of trials. Under the competing TTB model, the choice probability depends solely on the validity of the first discriminating feature, so trials with the same top discriminator will yield identical choice probabilities. In contrast, the advocated Mixture model integrates a Tallying component, meaning choice probabilities will systematically shift depending on how many total features favor the TTB-winning option versus the TTB-losing option.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
By design, TTB always prefers option A across all trials, and the distribution of the first discriminating feature (and its validity) is perfectly matched between two subsets of trials: those where Tallying prefers B (Conflict) and those where Tallying prefers A (Congruent). Under TTB with Probabilistic Stopping, the probability of choosing B depends only on the discriminating feature's validity, yielding identical expected choice rates for B on both subsets (difference ≈ 0). Under the Strategy Mixture Model, the presence of Tallying systematically boosts the probability of choosing B on Conflict trials and suppresses it on Congruent trials, resulting in a strictly positive difference.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): -0.0009 (var=0.0098)
- Predicted under pi_4 (simulated): 0.2804 (var=0.0500)
- Observed on real data: -0.0124 (var=0.0079)

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
**Predicted under pi_4:** 0.1600 (var=0.0120)

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
**Predicted under pi_4:** 0.4756 (var=0.0509)

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
**Predicted under pi_4:** 0.5113 (var=0.0262)

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
**Predicted under pi_4:** 0.4450 (var=0.0276)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (pi_3: TTB with Probabilistic Stopping) consistently outperforms Theory 2 (pi_4: Strategy Mixture Model). In Experiments 1 and 2, pi_4 predicts a massive influence of Tallying (differences of ~0.30), whereas the observed data shows near-zero differences, indicating that subjects are not systematically relying on a compensatory Tallying strategy. Theory 1 captures this non-compensatory, near-lexicographic behavior much better, while accurately modeling the noise as a function of the validity of the discriminating cue. Theory 2's fundamental assumption that a significant proportion of choices are driven by Tallying is strongly contradicted by the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Probabilistic Cue Search' theory. Instead of a deterministic search order with probabilistic execution (like Theory 1) or a mixture with compensatory tallying (like Theory 2), this theory posits that decision-makers always use a strict one-reason heuristic but search through cues probabilistically. The probability of inspecting a cue next is proportional to its validity (e.g., a softmax over validities of uninspected cues). Once a discriminating cue is found, it deterministically dictates the choice. This maintains the purely non-compensatory nature of the decision process (aligning with the observed near-zero tallying effects) but introduces noise through variations in attention or memory retrieval, offering a strong, cognitively plausible competitor to Theory 1."
}
```

## Usage

```json
{
  "prompt_token_count": 7309,
  "candidates_token_count": 340,
  "total_token_count": 8726
}
```
