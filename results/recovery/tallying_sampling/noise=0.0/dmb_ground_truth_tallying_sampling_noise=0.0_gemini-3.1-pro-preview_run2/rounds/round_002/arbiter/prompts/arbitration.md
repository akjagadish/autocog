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
People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Thresholded Tallying (Soft Subset Tallying): Decision makers use an equal-weight heuristic, but first filter the available information by cue validity. Instead of a hard threshold, they use a soft, probabilistic inclusion criterion where features with validity near or above a subjective threshold are more likely to be considered. These softly filtered features are then tallied to form a preference. This differentiable thresholding allows smooth transitions between pure Tallying (all cues included) and stricter subset tallying, reflecting boundedly rational cue selection that is easier to optimize during learning or fitting.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Tallying expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    
    # Filter features: soft threshold using a logistic sigmoid
    mask = 1.0 / (1.0 + np.exp(-k * (validities - theta)))
    
    # Tallying: count the number of positive features among the softly kept ones
    scores = np.sum(stim * mask, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.55, 0.52, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Rationale:** This design dissociates pure Tallying from Thresholded Tallying by exploiting a sharp difference in cue validities. The 6 features are split into a high-validity group (cues 1-3) and a low-validity group (cues 4-6). Pure Tallying ignores these validities and simply counts the total number of positive features. Thresholded Tallying, however, applies a soft filter that down-weights or eliminates the low-validity cues. By pitting options with many low-validity endorsements against options with fewer high-validity endorsements, pure Tallying will choose the option with the higher total count, whereas Thresholded Tallying will choose the option with the higher count among the high-validity cues. Ties in Tallying are also broken by Thresholded Tallying based on the distribution of high-validity cues.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Pure Tallying strictly counts the total number of positive features. On trials where both options have the exact same number of positive features (Trials 2 and 4), Tallying is completely indifferent and will choose either option with 50% probability, regardless of the subject's noise parameters. Thresholded Tallying, however, applies a soft filter that down-weights lower-validity features. Even when its threshold is low and it mimics Tallying, the logistic filter strictly preserves the order of validities, meaning it will always assign a higher score to the option with more high-validity features. By measuring the proportion of choices for the option with more high-validity features specifically on trials where the total feature counts are tied, we isolate a condition where Tallying predicts exactly 0.5 (with minimal variance) while Thresholded Tallying predicts a value strictly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    top3_a = a_ratings[:, :3].sum(axis=1)
    top3_b = b_ratings[:, :3].sum(axis=1)
    
    # Isolate trials where total features are tied but top-3 features differ
    valid_mask = (sum_a == sum_b) & (top3_a != top3_b)
    
    if not np.any(valid_mask):
        return 0.5
        
    responses = data['response'].values
    
    # Did the subject choose the option with more top-3 features?
    chose_high_val = np.zeros(len(data), dtype=bool)
    chose_high_val[(top3_a > top3_b) & (responses == 0)] = True
    chose_high_val[(top3_a < top3_b) & (responses == 1)] = True
    
    return float(np.mean(chose_high_val[valid_mask]))

### RESULTS
- Predicted under pi_3 (simulated): 0.5058 (var=0.0098)
- Predicted under pi_4 (simulated): 0.6250 (var=0.0323)
- Observed on real data: 0.5133 (var=0.0104)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 9: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  trial 10: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate Thresholded Tallying from Pure Tallying (Equal Weight), we use a 6-feature design with a clear bimodal distribution of validities: three high-validity features (0.85-0.95) and three low-validity features (0.50-0.60). Pure Tallying ignores validities and counts all positive features equally. Thresholded Tallying applies a soft threshold that filters out low-validity features, effectively tallying only the high-validity ones. By pitting options with fewer high-validity features against options with a greater total number of features (which are predominantly low-validity), we create stark dissociations. For example, an option with 2 high-validity and 0 low-validity features will be chosen by Thresholded Tallying over an option with 0 high-validity and 3 low-validity features, while Pure Tallying predicts the opposite. We also include tie-breaking trials where Pure Tallying predicts equal preference but Thresholded Tallying strongly favors the option with high-validity cues.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
For the Pure Tallying model, options A and B in the tie trials (trials 5, 6, 7, and 9) have the exact same unweighted sum of features. Consequently, Pure Tallying predicts a choice probability of exactly 0.5 for Option A on every single one of these trials, regardless of the subject's noise (beta) or lapse (epsilon) parameters. The variance of this metric for Pure Tallying subjects will be extremely low, driven purely by binomial sampling noise over the 36 tie trials per subject. Conversely, Thresholded Tallying softly filters out the low-validity features. Because Option A is heavily loaded with high-validity features in all these tie trials, Thresholded Tallying will strongly prefer Option A whenever its threshold parameter is sufficiently high to engage the filtering. This creates a large, directional mean shift for Thresholded Tallying (> 0.5) against a mathematically locked baseline (0.5) with minimal variance for Pure Tallying, guaranteeing a highly significant t-statistic.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # Pure Tallying relies only on the unweighted sum of features.
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    # Identify "tie" trials where both options have the exact same number of total positive features.
    # In the provided design, these are trials 5, 6, 7, and 9.
    tie_mask = (sum_a == sum_b)
    
    if not np.any(tie_mask):
        return 0.5
        
    responses = data['response'].values
    
    # In all of these tie trials, Option A was adversarially designed to contain strictly more 
    # high-validity features than Option B. 
    # We calculate the proportion of times the subject chose Option A on these trials.
    p_a = np.mean(responses[tie_mask] == 0)
    
    return float(p_a)

### RESULTS
- Predicted under pi_3 (simulated): 0.4917 (var=0.0053)
- Predicted under pi_4 (simulated): 0.6483 (var=0.0285)
- Observed on real data: 0.5144 (var=0.0066)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    is_comp = []
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        if ttb_choice is None:
            match = 0.5
        else:
            match = 1.0 if resp == ttb_choice else 0.0
            
        ttb_match.append(match)
        # Trials where the sum of features differ are compensatory in this design
        is_comp.append(sum(a) != sum(b))
        
    ttb_match = np.array(ttb_match)
    is_comp = np.array(is_comp)
    
    if is_comp.sum() == 0 or (~is_comp).sum() == 0:
        return 0.0
        
    comp_mean = ttb_match[is_comp].mean()
    non_comp_mean = ttb_match[~is_comp].mean()
    
    return float(non_comp_mean - comp_mean)
```

**Observed (real) value:** 0.3500 (var=0.0124)
**Predicted under pi_3:** 0.3367 (var=0.0130)
**Predicted under pi_4:** 0.3817 (var=0.0453)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0.5
        
    ttb_preds = data.apply(ttb_choice, axis=1)
    match = (data['response'] == ttb_preds).mean()
    return float(match)
```

**Observed (real) value:** 0.1074 (var=0.0059)
**Predicted under pi_3:** 0.1324 (var=0.0111)
**Predicted under pi_4:** 0.2116 (var=0.0309)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Convert rating lists to tuples for exact matching
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: A has the 2 most valid cues, B has the 3 less valid cues
    t1_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    # Trial 2: B has the 2 most valid cues, A has the 3 less valid cues
    t2_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    # For these trials, Tallying strictly prefers the option with 3 positive features.
    # WADD typically prefers the option with the 2 most valid cues.
    # We calculate the proportion of times the subject chose the 3-feature option.
    t1_tally_choices = (data.loc[t1_mask, 'response'] == 1).sum()
    t2_tally_choices = (data.loc[t2_mask, 'response'] == 0).sum()
    
    total_tally_choices = t1_tally_choices + t2_tally_choices
    total_critical_trials = t1_mask.sum() + t2_mask.sum()
    
    if total_critical_trials == 0:
        return 0.5
        
    return float(total_tally_choices / total_critical_trials)
```

**Observed (real) value:** 0.8213 (var=0.0136)
**Predicted under pi_3:** 0.8387 (var=0.0187)
**Predicted under pi_4:** 0.6025 (var=0.1203)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Identify critical dissociation trials where WADD and Tallying make opposite predictions
    a_is_11000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    b_is_00111 = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    t1 = a_is_11000 & b_is_00111
    
    a_is_00111 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    b_is_11000 = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    t7 = a_is_00111 & b_is_11000
    
    # WADD prefers the option with fewer, but higher-validity features ([1, 1, 0, 0, 0])
    wadd_t1 = t1 & (data['response'] == 0)
    wadd_t7 = t7 & (data['response'] == 1)
    
    wadd_choices = wadd_t1.sum() + wadd_t7.sum()
    total_trials = t1.sum() + t7.sum()
    
    if total_trials == 0:
        return 0.5
    return float(wadd_choices / total_trials)
```

**Observed (real) value:** 0.1383 (var=0.0112)
**Predicted under pi_3:** 0.1467 (var=0.0120)
**Predicted under pi_4:** 0.3342 (var=0.1021)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, Theory 1 (Pure Tallying) provides an incredibly accurate fit to the observed data. In Experiments 1 and 2, the metrics specifically target trials where the unweighted sums of features are tied but the options differ in high-validity cues. The observed metrics are consistently around 0.51, indicating almost perfect indifference. This aligns with Theory 1's prediction of pure unweighted tallying. Theory 2 (Thresholded Tallying) predicts a significant shift in preference toward options with higher-validity features, which is strongly contradicted by the observed data. Because the data shows no evidence of validity-based thresholding, Theory 2 is degenerate for this domain.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with a brand-new theory that competes with Pure Tallying without relying on cue validities. A strong candidate would be 'Stochastic Cue Sampling' or 'Random Subset Tallying'. In this theory, decision-makers are bounded by working memory and cannot process all features simultaneously. Instead of applying a post-tallying softmax noise, they randomly sample a subset of available cues on each trial (e.g., exactly k cues) and perform pure tallying on that subset. This provides a mechanistic, cognitive explanation for the observed choice noise while preserving the validity-agnostic nature of the Equal Weight heuristic that so perfectly matches the data."
}
```

## Usage

```json
{
  "prompt_token_count": 7103,
  "candidates_token_count": 321,
  "total_token_count": 8802
}
```
