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
Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weights for all features, ignoring validities.
    # The score for each option is simply the sum of its positive features.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Mixture of Heuristics: Subjects probabilistically switch between a compensatory, equal-weighting strategy (Tallying) and a frugal, lexicographic strategy (Take-The-Best). By mixing these two heuristics with a prior bias towards Tallying, the model captures the dominant tendency to count features while allowing for occasional validity-maximizing behavior, avoiding the unrealistic assumption of full compensatory validity weighting (WADD).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Tallying prediction
    scores_tally = np.sum(stim, axis=1)
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Take-The-Best prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_ttb_core = np.ones(2) / 2.0
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # Mixture
    p_tally_weight = float(parameters["p_tally"])
    p_combined = p_tally_weight * p_tally_core + (1.0 - p_tally_weight) * p_ttb_core
    
    # Lapse
    epsilon = float(parameters["epsilon"])
    n_opts = p_combined.shape[0]
    
    return (1.0 - epsilon) * p_combined + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate pure Tallying from a Mixture of Heuristics (Tallying + Take-The-Best), we design trials where the two strategies diverge. Tallying relies solely on the unweighted sum of positive features, while Take-The-Best (TTB) relies exclusively on the single highest-validity discriminating feature. By creating 'opposition' trials where one option has more positive features but lacks the highest-validity feature, Tallying will strongly prefer the feature-rich option, whereas the Mixture model will show a dampened preference (or even a reversal) due to the TTB component pulling in the opposite direction. We also include 'indifference' trials where both options have an equal number of positive features; here pure Tallying predicts a 50/50 split, but the Mixture model predicts a systematic preference for the option possessing the highest-validity feature.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Take-The-Best (TTB) heuristic specifically on trials where the Tallying heuristic is indifferent (i.e., both options have the same total number of positive features). Under the pure Tallying theory, subjects should guess randomly on these trials, yielding a metric value around 0.5. Under the Mixture of Heuristics theory, the TTB component will systematically drive choices toward the option with the highest-validity feature, resulting in a metric value significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_aligned = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Focus on 'indifference' trials for Tallying
        if sum(a) == sum(b):
            # Cue 0 is the most valid cue in this design
            if a[0] > b[0]:
                ttb_choice = 0
            elif b[0] > a[0]:
                ttb_choice = 1
            else:
                continue
                
            if row['response'] == ttb_choice:
                ttb_aligned.append(1)
            else:
                ttb_aligned.append(0)
                
    if len(ttb_aligned) == 0:
        return 0.5
        
    return float(np.mean(ttb_aligned))

### RESULTS
- Predicted under pi_3 (simulated): 0.4988 (var=0.0071)
- Predicted under pi_4 (simulated): 0.6475 (var=0.0132)
- Observed on real data: 0.5100 (var=0.0037)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Mixture of Heuristics (Tallying + Take-The-Best) from pure Tallying, we exploit trials where the two strategies make divergent predictions. Pure Tallying relies exclusively on the unweighted sum of positive features, whereas the Mixture model is influenced by the single highest-validity discriminating feature (the TTB component). We use 'indifference' trials where both options have an equal number of positive features but differ on the highest-validity cue; pure Tallying predicts a 50/50 split, but the Mixture model predicts a systematic preference. We also use 'opposition' trials where one option has more positive features (favored by Tallying) but the other option has the highest-validity feature (favored by TTB); here, pure Tallying strongly prefers the feature-rich option, while the Mixture model shows a dampened preference or reversal.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In the experimental design, there are specific 'indifference' trials where both options have the exact same number of positive features (sum_a == sum_b). On these trials, the pure Tallying model is completely indifferent and will choose Option A roughly 50% of the time. However, the Mixture model incorporates a Take-The-Best (TTB) component. The design is structured such that on all indifference trials, Option A possesses the highest-validity discriminating cue. Therefore, the Mixture model will systematically choose Option A more than 50% of the time. By measuring the proportion of Option A choices on these specific trials, we can cleanly separate the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying predicts indifference (equal sums)
    indiff_mask = sum_a == sum_b
    if not indiff_mask.any():
        return 0.5
        
    indiff_data = data[indiff_mask]
    
    # Compute the proportion of times Option A was chosen on these trials
    # Response == 0 means Option A was chosen
    p_a = (indiff_data['response'] == 0).mean()
    
    return float(p_a)

### RESULTS
- Predicted under pi_3 (simulated): 0.5058 (var=0.0050)
- Predicted under pi_4 (simulated): 0.6758 (var=0.0087)
- Observed on real data: 0.5000 (var=0.0051)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    diff = a_ratings - b_ratings
    # By using exponentially decreasing weights, the sign of the dot product 
    # is completely determined by the first non-zero difference, 
    # exactly mimicking Take The Best's lexicographic choice rule.
    weights = np.array([10000, 1000, 100, 10, 1])
    ttb_scores = diff @ weights
    ttb_choices = np.where(ttb_scores > 0, 0, 1)
    valid_trials = ttb_scores != 0
    if not np.any(valid_trials):
        return 0.5
    matches = (data['response'].values[valid_trials] == ttb_choices[valid_trials])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3458 (var=0.0045)
**Predicted under pi_3:** 0.2477 (var=0.0261)
**Predicted under pi_4:** 0.3965 (var=0.0186)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    count = 0
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner is not None:
            sum_a = sum(a)
            sum_b = sum(b)
            # Isolate compensatory trials: the TTB winner has a strictly smaller sum of features
            if (ttb_winner == 0 and sum_a < sum_b) or (ttb_winner == 1 and sum_b < sum_a):
                if resp == ttb_winner:
                    matches += 1
                count += 1
                
    return matches / count if count > 0 else 0.5
```

**Observed (real) value:** 0.3217 (var=0.0048)
**Predicted under pi_3:** 0.2842 (var=0.0239)
**Predicted under pi_4:** 0.4108 (var=0.0268)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_a = (1, 1, 0, 0, 0)
    t1_b = (0, 0, 1, 1, 1)
    
    mask1 = (a_tuples == t1_a) & (b_tuples == t1_b)
    mask2 = (a_tuples == t1_b) & (b_tuples == t1_a)
    
    chosen_t1_b = (mask1 & (data['response'] == 1)).sum() + (mask2 & (data['response'] == 0)).sum()
    total = mask1.sum() + mask2.sum()
    
    if total == 0:
        return 0.5
    return float(chosen_t1_b / total)
```

**Observed (real) value:** 0.6333 (var=0.0139)
**Predicted under pi_3:** 0.7700 (var=0.0377)
**Predicted under pi_4:** 0.6000 (var=0.0361)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # We focus on the 'opposition' trial where the two models make opposite predictions.
    # The target option is (1, 1, 0, 0, 0), which has fewer positive features (2 vs 3) 
    # but higher total validity (0.95+0.85=1.80 vs 0.60+0.55+0.50=1.65).
    # WADD prefers the target option; Tallying prefers the alternative (0, 0, 1, 1, 1).
    
    a_is_target = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (1, 1, 0, 0, 0))
    b_is_target = data['option_b_ratings'].apply(lambda x: tuple(int(v) for v in x) == (1, 1, 0, 0, 0))
    
    # Response 0 means A was chosen, 1 means B was chosen.
    chose_target = (a_is_target & (data['response'] == 0)) | (b_is_target & (data['response'] == 1))
    relevant = a_is_target | b_is_target
    
    if relevant.sum() == 0:
        return 0.5
        
    return float(chose_target.sum() / relevant.sum())
```

**Observed (real) value:** 0.3550 (var=0.0190)
**Predicted under pi_3:** 0.2975 (var=0.0360)
**Predicted under pi_4:** 0.3588 (var=0.0270)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Tallying) captures the core decision-making mechanism much better than Theory 2 (Mixture of Heuristics). The most definitive evidence comes from Experiments 1 and 2, where the models' predictions diverge on 'indifference' trials (where both options have an equal number of positive features but differ in cue validities). On these trials, the Mixture model expects subjects to fall back on Take-The-Best, predicting a strong preference (~0.65-0.67) for the option with the highest validity cue. However, the observed data in both experiments is exactly 0.50, indicating that subjects are perfectly indifferent and guessing at random. This strongly refutes the presence of a Take-The-Best or validity-weighted component and confirms that subjects rely purely on equal-weighted feature counting (Tallying). The reason Tallying slightly underperformed in Experiments 3, 4, 5, and 6 is simply that its simulated parameters (beta/epsilon) were too deterministic. The observed data in those experiments is consistently closer to 0.5 than Theory 1's predictions, meaning subjects exhibit more noise or lapsing than the current parameter ranges simulate.",
  "verdict": "new_model",
  "target_theory_idx": 1,
  "recommendation": "Keep the pure Tallying theory description verbatim, as the mechanism is correct. However, regenerate the parameter ranges for `beta` and `epsilon` to allow for more stochasticity. Specifically, expand the lower bound of the `beta` (softmax inverse temperature) range to include smaller values, and potentially increase the upper bound of the `epsilon` (lapse rate) range. This will dampen the model's predictions on unequal-sum trials (bringing them closer to the observed ~0.65 rather than the simulated ~0.77), while perfectly maintaining the 0.50 prediction on equal-sum indifference trials."
}
```

## Usage

```json
{
  "prompt_token_count": 6514,
  "candidates_token_count": 422,
  "total_token_count": 9304
}
```
