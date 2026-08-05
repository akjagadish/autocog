# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_2_1") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2_1" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Tallying with Extreme Noise. Subjects attempt to use a simple Tallying heuristic, counting the number of features favoring each option. However, because the tasks involve complex multi-attribute conflicts in magnitude and direction without clear, overwhelming dominance, decision-making is heavily corrupted by noise or subjects simply treat the options as practically equivalent. This is modeled as a Tallying heuristic with an extremely low inverse temperature (beta near zero), causing choice probabilities to collapse to nearly 50/50, which explains the near-chance performance observed across all conflict trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Tallying: count how many features each option strictly wins
    wins_a = np.sum(a > b)
    wins_b = np.sum(b > a)
    
    scores = np.array([wins_a, wins_b], dtype=float)
    
    beta = float(parameters['beta'])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_2_1
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature subjective weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[10, 0, 0, 0, 0]  B=[0, 2, 2, 2, 2]
  trial 2: A=[10, 10, 0, 0, 0]  B=[0, 0, 4, 4, 4]
  trial 3: A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 6, 6]
  trial 4: A=[0, 2, 2, 2, 2]  B=[10, 0, 0, 0, 0]
  trial 5: A=[0, 0, 4, 4, 4]  B=[10, 10, 0, 0, 0]
  trial 6: A=[0, 0, 0, 6, 6]  B=[10, 10, 10, 0, 0]
  trial 7: A=[10, 0, 10, 0, 0]  B=[0, 3, 0, 3, 3]
  trial 8: A=[0, 3, 0, 3, 3]  B=[10, 0, 10, 0, 0]

**Rationale:** This design tests the advocated 'Tallying with Extreme Noise' theory, which predicts near-chance (50/50) responding across all conflict trials due to a near-zero inverse temperature, against the competing WADD theory, which predicts strong deterministic preferences driven by validity-weighted cardinal differences. We construct trials where Option A has a massive magnitude advantage on a minority of high-validity features, while Option B has small advantages on a majority of low-validity features. WADD predicts a strong preference for the option with the large magnitude advantage, whereas the advocated theory predicts choices will remain uniformly random regardless of the magnitude or validity differences.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pits an option with a few extreme values (10s) on high-validity features against an option with many moderate values on lower-validity features. The option with the 10s always has a strictly greater sum of feature values. The advocated theory (Tallying with Extreme Noise) posits that subjects are overwhelmed and guess, predicting a choice proportion of ~0.5 for the higher-sum option. The competing theory (WADD) computes validity-weighted sums, which will strongly favor the option with the extreme values on the most important features. Thus, WADD predicts a choice proportion significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    chose_higher = ((sum_a > sum_b) & (data['response'] == 0)) | ((sum_b > sum_a) & (data['response'] == 1))
    return float(chose_higher.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.5060 (var=0.0028)
- Predicted under pi_2_1 (simulated): 0.6094 (var=0.0162)
- Observed on real data: 0.4867 (var=0.0021)

## EXPERIMENT 2 (proposed by pi_2_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[10, 10, 10, 10, 10]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[10, 10, 10, 10, 10]
  trial 3: A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 10, 10]
  trial 4: A=[0, 0, 10, 10, 10]  B=[10, 10, 0, 0, 0]
  trial 5: A=[9, 9, 9, 9, 9]  B=[1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1]  B=[9, 9, 9, 9, 9]
  trial 7: A=[10, 0, 10, 0, 10]  B=[0, 10, 0, 10, 0]
  trial 8: A=[0, 10, 0, 10, 0]  B=[10, 0, 10, 0, 10]

**Rationale:** To conclusively dissociate the advocated Weighted Additive (WADD) theory from the competing 'Tallying with Extreme Noise' theory, we exploit the fundamental difference in their predicted choice certainty. The competing theory posits that choices are driven by a Tallying heuristic heavily corrupted by extreme noise (beta near zero), which strictly bounds the maximum possible log-odds difference to the number of features and guarantees near-50/50 choice probabilities on all trials. By using a high rating maximum (10) and generating trials with massive cardinal magnitude differences, WADD predicts highly deterministic choices (approaching 100% certainty for the superior option) due to the large differences in weighted sums, whereas the competing theory remains constrained to predict near-chance behavior.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The Weighted Additive (WADD) theory predicts that choices will reliably favor the option with the higher validity-weighted sum of attributes. Since the trials are designed with massive differences in cardinal ratings, WADD predicts a high choice probability for the sum-maximizing option (limited only by the uniform lapse rate epsilon). In contrast, the 'Tallying with Extreme Noise' theory limits the maximum log-odds difference to the number of features (e.g., 5) and applies a near-zero beta, guaranteeing choice probabilities remain extremely close to 50/50 on all trials. By computing the proportion of trials where the subject chooses the option with the higher sum of ratings, we expect WADD to yield a value significantly above 0.5, while Tallying with Extreme Noise will produce a tight distribution centered just above 0.50.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of ratings for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Determine which option has the higher sum (0 for A, 1 for B)
    target = (sum_b > sum_a).astype(int)
    
    # Exclude trials where sums are equal
    valid = sum_a != sum_b
    
    if not valid.any():
        return 0.5
        
    # Calculate proportion of choices that maximize the simple sum
    accuracy = (data.loc[valid, 'response'] == target[valid]).mean()
    return float(accuracy)


### RESULTS
- Predicted under pi_4 (simulated): 0.5035 (var=0.0017)
- Predicted under pi_2_1 (simulated): 0.6550 (var=0.0165)
- Observed on real data: 0.5096 (var=0.0033)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[5, 6, 6, 6]  B=[10, 4, 4, 4]
  A=[8, 2, 8, 8]  B=[9, 10, 2, 2]
  A=[6, 6, 2, 2]  B=[4, 4, 10, 10]
  A=[2, 8, 8, 8]  B=[10, 6, 6, 6]
  A=[4, 4, 9, 9]  B=[8, 8, 2, 2]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[7, 7, 3, 3]  B=[6, 6, 10, 10]
  A=[3, 8, 8, 8]  B=[9, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def count_a_wins(row):
        return sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b)
        
    a_wins = data.apply(count_a_wins, axis=1)
    mask = a_wins >= 3
    
    if mask.sum() == 0:
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5042 (var=0.0050)
**Predicted under pi_4:** 0.4908 (var=0.0046)
**Predicted under pi_2_1:** 0.5188 (var=0.0101)

### Experiment 4
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[10, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  A=[9, 9, 8, 1]  B=[8, 8, 7, 10]
  A=[8, 8, 7, 10]  B=[9, 9, 8, 1]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    
    mask_t1 = (a0 == 1)
    mask_t2 = (a0 == 10) & (a1 == 2)
    mask_t4 = (a0 == 9)
    mask_t5 = (a0 == 8)
    
    match_t1 = mask_t1 & (data['response'] == 1)
    match_t2 = mask_t2 & (data['response'] == 0)
    match_t4 = mask_t4 & (data['response'] == 1)
    match_t5 = mask_t5 & (data['response'] == 0)
    
    matches = match_t1 | match_t2 | match_t4 | match_t5
    valid_trials = mask_t1 | mask_t2 | mask_t4 | mask_t5
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float(matches.sum() / valid_trials.sum())
```

**Observed (real) value:** 0.5156 (var=0.0035)
**Predicted under pi_4:** 0.4859 (var=0.0027)
**Predicted under pi_2_1:** 0.5500 (var=0.0079)

### Experiment 5
**Design**
  A=[2, 7, 7, 7]  B=[10, 6, 6, 6]
  A=[10, 2, 2, 2]  B=[2, 3, 3, 3]
  A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 5, 5, 5]  B=[10, 0, 0, 1]
  A=[10, 0, 0, 1]  B=[0, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Count how many features option A strictly wins and option B strictly wins
    a_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b), axis=1)
    b_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b > a), axis=1)
    
    # Only consider trials where one option wins on strictly more features than the other
    mask = a_wins != b_wins
    if not mask.any():
        return 0.5
        
    # Tallying predicts choosing the option with more feature wins
    # response == 0 means A was chosen, response == 1 means B was chosen
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    actual = data.loc[mask, 'response']
    
    # Return the proportion of trials where the subject's choice aligns with the Tallying prediction
    return float((tally_pred == actual).mean())
```

**Observed (real) value:** 0.5088 (var=0.0034)
**Predicted under pi_4:** 0.5056 (var=0.0027)
**Predicted under pi_2_1:** 0.5081 (var=0.0067)

### Experiment 6
**Design**
  A=[1, 1, 6, 6, 6]  B=[10, 10, 5, 5, 5]
  A=[10, 4, 4, 4, 4]  B=[0, 5, 5, 5, 5]
  A=[10, 10, 0, 0, 5]  B=[0, 0, 10, 10, 5]
  A=[0, 7, 7, 7, 7]  B=[10, 6, 6, 0, 0]
  A=[10, 0, 0, 0, 0]  B=[0, 4, 4, 5, 0]
  A=[8, 8, 2, 2, 2]  B=[2, 2, 8, 8, 8]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].tolist()
    # Trial 1: A=[1, 1, 6, 6, 6]
    is_t1 = np.array([a[0] == 1 for a in a_ratings])
    # Trial 2: A=[10, 4, 4, 4, 4]
    is_t2 = np.array([a[0] == 10 and a[1] == 4 for a in a_ratings])
    
    resp_A = (data['response'] == 0).values
    
    p_A_t1 = resp_A[is_t1].mean() if is_t1.sum() > 0 else 0.5
    p_A_t2 = resp_A[is_t2].mean() if is_t2.sum() > 0 else 0.5
    
    return float(p_A_t1 - p_A_t2)
```

**Observed (real) value:** -0.0100 (var=0.0271)
**Predicted under pi_4:** -0.0062 (var=0.0266)
**Predicted under pi_2_1:** -0.2162 (var=0.0574)

### Experiment 7
**Design**
  A=[5, 10, 10, 10]  B=[6, 0, 0, 0]
  A=[10, 0, 0, 0]  B=[9, 10, 10, 10]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[5, 5, 5, 5]  B=[5, 6, 0, 0]
  A=[7, 7, 0, 0]  B=[7, 6, 10, 10]
  A=[8, 8, 8, 0]  B=[8, 8, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    ttb_choices = []
    for i in range(len(data)):
        d = diff[i]
        nonzero = np.where(d != 0)[0]
        if len(nonzero) > 0:
            first_diff_idx = nonzero[0]
            ttb_choices.append(0 if d[first_diff_idx] > 0 else 1)
        else:
            ttb_choices.append(-1)
            
    ttb_choices = np.array(ttb_choices)
    responses = data['response'].values
    
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.4979 (var=0.0021)
**Predicted under pi_4:** 0.4925 (var=0.0024)
**Predicted under pi_2_1:** 0.3977 (var=0.0132)

### Experiment 8
**Design**
  A=[6, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[8, 8, 8, 8]  B=[9, 0, 0, 0]
  A=[5, 6, 1, 1]  B=[5, 5, 10, 10]
  A=[5, 5, 10, 0]  B=[5, 5, 9, 10]
  A=[10, 0, 0, 0]  B=[0, 10, 10, 10]
  A=[7, 7, 7, 7]  B=[8, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_choice(a, b):
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            elif val_a < val_b:
                return 1
        return -1
        
    ttb_preds = [ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    agreements = [1 if p == r else 0 for p, r in zip(ttb_preds, data['response']) if p != -1]
    
    return float(np.mean(agreements)) if len(agreements) > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0030)
**Predicted under pi_4:** 0.4977 (var=0.0028)
**Predicted under pi_2_1:** 0.3810 (var=0.0140)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments, the observed data consistently shows choice proportions extremely close to 0.5 (or differences close to 0), regardless of the magnitude of cardinal feature differences, cue validities, or tally counts. Theory 1 (Tallying with Extreme Noise) successfully captures this by applying a near-zero inverse temperature, effectively collapsing all predictions to chance. Theory 2 (WADD) consistently fails because it predicts strong, systematic preferences based on weighted cardinal values, which are entirely absent in the empirical data. The data strongly suggests that subjects are not systematically integrating information.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the WADD theory with a 'Pure Random Guessing' (or Inattentive Responding) theory. While Theory 1 fits the data by computing Tallying scores and then applying extreme noise to wash them out, a much more parsimonious explanation for the universal 50/50 choice behavior is that subjects are completely inattentive and simply guess uniformly at random on every trial, bypassing any cognitive evaluation of the features entirely. The new theory should model choice probabilities as exactly 0.5 for all trials, independent of the stimulus."
}
```

## Usage

```json
{
  "prompt_token_count": 7104,
  "candidates_token_count": 270,
  "total_token_count": 8616
}
```
