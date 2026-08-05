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
People integrate information from multiple cues by computing a weighted sum of the feature values, where the weights are a non-linear transformation of the cues' validities. This Weighted Additive (WADD) strategy is compensatory but incorporates a scaling parameter (gamma) that can exponentiate the validities. This allows the decision-maker to dynamically re-balance attention—either steepening the weights to behave more like Take The Best, or flattening them to behave more like Tallying—to match the empirical balance of conflict resolution.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    # Weighted Additive (WADD) heuristic with exponentiated validities.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Exponentiate validities to allow non-linear scaling of weights
    weights = val ** gamma
    
    # Compute the weighted sum of features for each option
    # using the scaled cue validities as weights.
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_5
Bayesian Cue Integration with Belief Dampening: Decision-makers process cues as conditionally independent pieces of evidence, translating each cue's validity into a log-odds weight. However, instead of taking provided probabilities at face value, humans dampen extreme probabilities, contracting them toward 0.5 (ignorance) before converting them to log-odds. The overall evidence for an option is the sum of the log-odds of its positive features.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Dampen validities toward 0.5 based on gamma
    gamma = float(parameters["gamma"])
    val_adj = 0.5 + (val - 0.5) * gamma
    
    # Clip validities to avoid division by zero or log of zero
    val_adj = np.clip(val_adj, 0.5001, 0.9999)
    
    # Calculate log-odds weights for each cue
    weights = np.log(val_adj / (1.0 - val_adj))
    
    # Compute the evidence for each option as the sum of log-odds
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.75, 0.75]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates WADD (with exponentiated validities) from Bayesian Cue Integration (with belief dampening) by exploiting their divergent limiting behaviors. As Bayesian dampening increases (gamma -> 0), the model does NOT collapse to Equal Weight; instead, it becomes a linear model on the raw probability excess (val - 0.5). In contrast, WADD with low gamma flattens validities and collapses to Tallying (Equal Weight). We construct trials where Bayesian Integration ALWAYS prefers the option with higher sum of (val - 0.5) and log-odds, while WADD with a low gamma reverses preference due to Tallying. Conversely, we include trials where Bayesian Integration ALWAYS prefers the option with more cues, but WADD with a high gamma reverses preference due to the exponentiation of a single dominant cue.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric evaluates the within-subject difference in the probability of choosing the option with the strongest cue(s) between Trial 1 and Trial 2. For Bayesian Cue Integration, the evidence gap between the target and opposing options shrinks significantly from T1 to T2 (mathematically, the difference in scores drops by a large, strictly positive margin across all dampening values), leading to a massive drop in the choice probability for the target option (a large positive P(T1) - P(T2)). In contrast, the Weighted Additive (WADD) model's non-linear exponentiation creates a much smaller gap between these two trials across its parameter space, often resulting in similar choice rates for both trials. The within-subject subtraction cancels out individual baseline noise (such as overall temperature and lapse rates), providing a stable signal with low between-subject variance that cleanly discriminates the models.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1: Target has cues [1, 1, 0, 0, 0], Opposing has [0, 0, 1, 1, 1]
    is_t1_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Identify Trial 2: Target has cue [1, 0, 0, 0, 0], Opposing has [0, 1, 1, 0, 0]
    is_t2_fwd = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    is_t2_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    
    # Calculate P(Target | T1)
    t1_target_choices = (is_t1_fwd & (data['response'] == 0)).sum() + (is_t1_rev & (data['response'] == 1)).sum()
    t1_total = is_t1_fwd.sum() + is_t1_rev.sum()
    p_t1 = float(t1_target_choices / t1_total) if t1_total > 0 else 0.5
    
    # Calculate P(Target | T2)
    t2_target_choices = (is_t2_fwd & (data['response'] == 0)).sum() + (is_t2_rev & (data['response'] == 1)).sum()
    t2_total = is_t2_fwd.sum() + is_t2_rev.sum()
    p_t2 = float(t2_target_choices / t2_total) if t2_total > 0 else 0.5
    
    # Return the simple difference between the two choice probabilities
    return p_t1 - p_t2


### RESULTS
- Predicted under pi_3 (simulated): 0.2816 (var=0.0715)
- Predicted under pi_5 (simulated): 0.4916 (var=0.0813)
- Observed on real data: 0.2200 (var=0.1117)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** To quantitatively dissociate Bayesian Cue Integration (with belief dampening) from Weighted Additive (WADD) with exponentiated validities, we exploit the divergent ways these models handle cue weighting. Bayesian Integration converts validities to log-odds, which heavily penalizes lower validities and creates a convex weighting function. As belief dampening increases (gamma -> 0), the log-odds weights become strictly linear with respect to the raw probability excess (validity - 0.5). In contrast, WADD uses an exponential transformation (validity ^ gamma), which can be concave (gamma < 1), linear (gamma = 1), or convex (gamma > 1). We design specific trials where Bayesian Integration ALWAYS prefers Option A across all valid dampening parameters, whereas WADD flips its preference to Option B for certain gamma values (e.g., when gamma < 1, WADD flattens the weights and behaves more like Tallying). Additionally, we include trials where Bayesian Integration flips its preference depending on the dampening parameter, but WADD predicts the opposite flip, ensuring robust parameter identifiability and model dissociation.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the probability of choosing Option B between Trial 4 and Trial 2. Under the Advocated theory (Bayesian Cue Integration with Belief Dampening), Option A is consistently preferred in Trial 2 (due to the high log-odds of the first cue) and Option B is consistently preferred in Trial 4, leading to a difference approaching 1.0. Under the Competing theory (WADD with exponentiated validities), the preference in both trials flips depending on the gamma parameter, resulting in a much smaller average difference (close to 0.1) across the population.

Source:
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    # Trial 2: Option A is [1, 0, 0, 0, 0], Option B is [0, 1, 0, 0, 1]
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    # Trial 4: Option A is [0, 1, 0, 1, 1], Option B is [1, 0, 1, 0, 0]
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 1))
    
    t2_resp = data.loc[t2_mask, 'response'].mean()
    t4_resp = data.loc[t4_mask, 'response'].mean()
    
    if pd.isna(t2_resp) or pd.isna(t4_resp):
        return 0.0
        
    # Response is 1 if Option B is chosen, 0 if Option A is chosen.
    return float(t4_resp - t2_resp)

### RESULTS
- Predicted under pi_3 (simulated): 0.0463 (var=0.2994)
- Predicted under pi_5 (simulated): 0.5779 (var=0.0844)
- Observed on real data: 0.0211 (var=0.2810)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        a_wins = sum(1 for i in range(4) if a[i] > b[i])
        b_wins = sum(1 for i in range(4) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if resp == ttb_winner:
                ttb_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_consistent / total)

```

**Observed (real) value:** 0.3520 (var=0.0355)
**Predicted under pi_3:** 0.4170 (var=0.0576)
**Predicted under pi_5:** 0.5613 (var=0.0028)

### Experiment 4
**Design**
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Fixed validities from the experimental design
    val = np.array([0.65, 0.95, 0.55, 0.75, 0.85])
    cue_order = np.argsort(-val, kind='stable').tolist()
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue  # Tallying predicts a tie, skip
            
        # TTB prediction
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        # Only consider trials where the two heuristics make STRICTLY OPPOSITE predictions
        if ttb_pred is not None and tally_pred != ttb_pred:
            matches.append(1.0 if row['response'] == tally_pred else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.6600 (var=0.0377)
**Predicted under pi_3:** 0.6763 (var=0.0203)
**Predicted under pi_5:** 0.6290 (var=0.0061)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_has_f0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_f0 = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    chose_f0 = (a_has_f0 & chose_a) | (b_has_f0 & chose_b)
    return float(chose_f0.mean())
```

**Observed (real) value:** 0.3862 (var=0.0372)
**Predicted under pi_3:** 0.4656 (var=0.0213)
**Predicted under pi_5:** 0.4963 (var=0.0037)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 3 where A=[1, 0, 1, 0, 0] and B=[0, 1, 0, 1, 0]
    # This is the only trial where the sum of features is 2 for both options.
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    mask = (a_sums == 2) & (b_sums == 2)
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.6225 (var=0.0889)
**Predicted under pi_3:** 0.8100 (var=0.0256)
**Predicted under pi_5:** 0.7863 (var=0.0247)

### Experiment 7
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify the tie-breaker trials (Trials 1 and 6) where Option A is [0, 1, 1, 0, 0]
    is_target = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (0, 1, 1, 0, 0))
    if not is_target.any():
        return 0.5
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[is_target, 'response'] == 0))
```

**Observed (real) value:** 0.6217 (var=0.0568)
**Predicted under pi_3:** 0.5975 (var=0.0257)
**Predicted under pi_5:** 0.7117 (var=0.0234)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10011') & (b_str == '01100')
    t2_mask = (a_str == '01100') & (b_str == '10011')
    
    t1_choices = data.loc[t1_mask, 'response'] == 0
    t2_choices = data.loc[t2_mask, 'response'] == 1
    
    total_target_choices = t1_choices.sum() + t2_choices.sum()
    total_trials = t1_mask.sum() + t2_mask.sum()
    
    if total_trials == 0:
        return 0.5
    return float(total_target_choices / total_trials)
```

**Observed (real) value:** 0.6025 (var=0.0898)
**Predicted under pi_3:** 0.4706 (var=0.0862)
**Predicted under pi_5:** 0.2062 (var=0.0213)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of experiments, Theory 1 (pi_3: WADD with exponentiated validities) significantly outperforms Theory 2 (pi_5: Bayesian Cue Integration with Belief Dampening). This is most striking in Experiments 1, 2, and 8, where Theory 2 predicts massive effects (or extreme values) that are completely absent in the observed data. Theory 1 accurately captures the more muted empirical effects. However, Theory 1 still shows some miscalibration in certain edge cases (e.g., Experiment 6). Because Theory 2's log-odds mechanism fundamentally mischaracterizes how human subjects aggregate cue validities in this paradigm, it is degenerate and should be replaced with a structurally different approach.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Rank-Based Weighting' theory. Instead of scaling raw validities or computing log-odds, humans might ignore the exact numerical validities and instead rely on their ordinal rank. The theory should posit that decision-makers sort the cues by validity and assign weights that decay based on rank (e.g., w_i = alpha ** rank_i). This ordinal approach captures bounded rationality\u2014reducing cognitive load by discarding precise probabilities\u2014while still strongly weighting the most valid cues, offering a robust alternative to the continuous transformations of WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 7868,
  "candidates_token_count": 301,
  "total_token_count": 9079
}
```
