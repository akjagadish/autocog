# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Evidence Accumulation with Bounded Threshold: People sample cues sequentially in order of validity. Each discriminating cue adds its validity to the evidence for the favored option. If the absolute difference in evidence between the two options reaches a threshold `theta`, sampling stops and a decision is made based on the accumulated evidence (mimicking Take The Best). If all cues are exhausted without the threshold being reached, the decision is based on the final integrated evidence (mimicking Weighted Additive). This allows for early stopping when initial evidence is strong, but continued sampling when evidence is weak or tied.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    theta = float(parameters["theta"])
    
    ev_a = 0.0
    ev_b = 0.0
    
    # Sequential evidence accumulation
    for j in cue_order:
        if a[j] > b[j]:
            ev_a += val[j]
        elif b[j] > a[j]:
            ev_b += val[j]
            
        # Early stopping if threshold is reached
        if abs(ev_a - ev_b) >= theta:
            break
            
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_3
People make decisions by computing a weighted sum of the features for each option, using the cue validities scaled by an exponent as the weights. Unlike Take The Best, which is strictly non-compensatory, the Weighted Additive (WADD) strategy integrates all available information. However, by scaling validities with an exponent (gamma), the model can flexibly capture varying degrees of compensatoriness, ranging from equal-weighting (Tallying) to highly skewed weighting that approximates one-reason decision making (TTB). Choice probabilities are generated via a softmax over the weighted sums, combined with a uniform lapse rate to account for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Rationale:** To distinguish the Evidence Accumulation with Bounded Threshold model (THEORY 1) from the exponent-scaled Weighted Additive model (THEORY 2), we exploit the threshold mechanism directly. In THEORY 1, if the threshold is reached early, sampling stops and subsequent cues are completely ignored. If the threshold is not reached, it integrates all cues linearly (sum of validities). In THEORY 2, all cues are always integrated, but their weights are non-linearly scaled by an exponent (gamma). We use a set of 4 features with specific validities such that the linear sum of validities is perfectly tied for some pairs (e.g., 0.95 + 0.65 = 1.60 vs 0.85 + 0.75 = 1.60). For these tied pairs, THEORY 1 predicts either a choice for the top cue (if threshold is low) or a perfect tie (if threshold is high). THEORY 2, however, applies an exponent to the validities: if gamma > 1, the option with the more extreme validities (0.95 and 0.65) wins; if gamma < 1, the option with the moderate validities (0.85 and 0.75) wins. By varying the dispersion and sums of the validities across trials, we can cleanly separate the early-stopping/linear-integration dynamics of THEORY 1 from the non-linear full-integration dynamics of THEORY 2.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Previous metrics failed because the parameter distributions (gamma for WADD, theta for Bounded Threshold) cause both models to behave identically to Take-The-Best for ~90% of subjects, while the remaining ~10% exhibit opposing behavior on conflict trials. This creates massive bimodal variance that destroys statistical significance. To bypass this, we measure TTB adherence strictly on the 'consensus' trials (T6 and T4) where BOTH models theoretically predict a 100% win rate for the TTB option across all parameter values (for WADD, 0.95^g + 0.85^g > 0.75^g + 0.65^g for all g>0). This completely eliminates the bimodal strategy variance, leaving only the noise parameters. Since WADD is parametrized with a much higher lapse rate (epsilon up to 0.5) than Bounded Threshold (epsilon up to 0.1), WADD will exhibit a significantly lower adherence on these consensus trials, and the drastically reduced variance will make this gap highly significant.

Source:
def metric(data: pd.DataFrame) -> float:
    match_count = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Focus exclusively on consensus trials where both models 
        # theoretically predict the TTB option wins for almost all parameters.
        if a == (1, 1, 0, 0) and b == (0, 0, 1, 1):
            if row['response'] == 0:
                match_count += 1
            total += 1
        elif a == (1, 0, 1, 0) and b == (0, 1, 0, 1):
            if row['response'] == 0:
                match_count += 1
            total += 1
            
    return match_count / total if total > 0 else 0.5

### RESULTS
- Predicted under pi_4 (simulated): 0.9531 (var=0.0057)
- Predicted under pi_3 (simulated): 0.8662 (var=0.0111)
- Observed on real data: 0.1175 (var=0.0070)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To distinguish Evidence Accumulation with Bounded Threshold (THEORY 2) from exponent-scaled Weighted Additive (THEORY 1), we use pairs where the linear sum of validities is perfectly tied (e.g., 0.95 + 0.65 = 1.60 vs 0.85 + 0.75 = 1.60). For these, THEORY 2 either chooses the option favored by the highest validity cue (if early stopping is triggered) or predicts a perfect tie (if all cues are integrated). THEORY 2 can never predict the option with moderate validities. In contrast, THEORY 1 uses an exponent on validities: a parameter gamma < 1 will cause the moderate validities to sum to a higher value than the extreme validities, allowing THEORY 1 to strongly predict the moderate option. This creates a qualitative dissociation where THEORY 1 can predict choices that are impossible under THEORY 2. We also include standard TTB vs Tallying pairs to ensure rich parameter estimation.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Measures the proportion of times the subject chooses the 'moderate' option (the one lacking the single highest validity cue) on trials where the linear sum of validities is perfectly tied. Theory 2 (Evidence Accumulation) either stops early and picks the extreme option, or integrates all cues and predicts a tie (50% max for the moderate option). Theory 1 (WADD with exponent) can systematically prefer the moderate option if gamma < 1, leading to a higher choice rate for the moderate option.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    moderate_chosen = 0
    tied_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(val * a)
        sum_b = np.sum(val * b)
        
        # Check if it's a tied pair (linear sum of validities is equal)
        if abs(sum_a - sum_b) < 1e-4:
            tied_trials += 1
            
            # Find the best discriminating cue
            for i in range(5):
                if a[i] != b[i]:
                    best_cue_favors_a = (a[i] > b[i])
                    break
            
            # The 'moderate' option is the one NOT favored by the best cue
            chose_a = (row['response'] == 0)
            
            if best_cue_favors_a and not chose_a:
                moderate_chosen += 1
            elif not best_cue_favors_a and chose_a:
                moderate_chosen += 1
                
    if tied_trials == 0:
        return 0.0
    return float(moderate_chosen / tied_trials)

### RESULTS
- Predicted under pi_4 (simulated): 0.0852 (var=0.0127)
- Predicted under pi_3 (simulated): 0.2067 (var=0.0154)
- Observed on real data: 0.6593 (var=0.0108)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd

    def get_predictions(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
                
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(a)) if b[i] > a[i])
        if a_wins > b_wins:
            tally = 0
        elif b_wins > a_wins:
            tally = 1
        else:
            tally = -1
            
        return pd.Series({'ttb': ttb, 'tally': tally})
        
    preds = data.apply(get_predictions, axis=1)
    
    # Filter for trials where TTB and Tallying make strictly opposite predictions
    mask = (preds['ttb'] != -1) & (preds['tally'] != -1) & (preds['ttb'] != preds['tally'])
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of choices that align with Take The Best on these critical trials
    ttb_choices = (data.loc[mask, 'response'] == preds.loc[mask, 'ttb']).mean()
    return float(ttb_choices)

```

**Observed (real) value:** 0.7858 (var=0.0260)
**Predicted under pi_4:** 0.8446 (var=0.0633)
**Predicted under pi_3:** 0.7021 (var=0.0525)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_match = 0
    count = 0
    
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
            continue  # Tallying predicts a tie
            
        # TTB prediction (cues are ordered by descending validity based on the design)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred is None:
            continue  # TTB predicts a tie
            
        # Only consider trials where the two theories make strictly opposite predictions
        if tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            count += 1
            
    if count == 0:
        return 0.5
        
    return float(tally_match / count)

```

**Observed (real) value:** 0.2306 (var=0.0195)
**Predicted under pi_4:** 0.1831 (var=0.0521)
**Predicted under pi_3:** 0.3344 (var=0.0398)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    diff = a_mat - b_mat
    
    match = 0
    total = 0
    for i in range(len(diff)):
        row_diff = diff[i]
        nonzero_idx = np.nonzero(row_diff)[0]
        if len(nonzero_idx) > 0:
            first_idx = nonzero_idx[0]
            ttb_choice = 0 if row_diff[first_idx] > 0 else 1
            if resp[i] == ttb_choice:
                match += 1
            total += 1
            
    return float(match / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.6521 (var=0.0118)
**Predicted under pi_4:** 0.7771 (var=0.0930)
**Predicted under pi_3:** 0.6856 (var=0.0224)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    mask = (sum_a == 1) & (sum_b >= 3)
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.2233 (var=0.0237)
**Predicted under pi_4:** 0.1308 (var=0.0399)
**Predicted under pi_3:** 0.2500 (var=0.0332)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both theories fail to capture the observed behavior in Experiments 1 and 2. In Experiment 1, subjects overwhelmingly chose the option contrary to what both the Bounded Threshold and Exponent-Scaled WADD models predict on 'consensus' trials (observed 0.1175 vs. predicted ~0.86-0.95). In Experiment 2, subjects chose the 'moderate' option on tied-sum trials at a much higher rate (0.6593) than either theory could accommodate (0.0852 and 0.2067). While Theory 2 (pi_3) performs slightly better on Experiments 4, 5, and 6, neither theory provides a unified account of the anomalous choices in the first two experiments. The rigid reliance on validity-based cue ordering or linear/exponential integration seems fundamentally misaligned with how subjects are evaluating the options when there are trade-offs between cue rank and cue quantity or configuration.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a fundamentally different theory of multi-attribute choice, such as a Configural Weighting model or a Heuristic-Switching model based on the variance of the cues. The new theory should account for why subjects might systematically prefer options with 'moderate' validities or even systematically violate validity-based dominance (as seen in Experiment 1). Consider mechanisms where the subjective weight of a cue is context-dependent (e.g., depends on the presence of other cues) or where subjects use a two-stage process: first comparing the total number of positive features (tallying), and only using validities to break ties or when the difference in counts is small."
}
```

## Usage

```json
{
  "prompt_token_count": 6996,
  "candidates_token_count": 377,
  "total_token_count": 8301
}
```
