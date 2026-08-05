# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_2
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
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

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates Take The Best (TTB) from the Weighted Additive rule (WADD) by systematically pitting a single high-validity feature against multiple lower-validity features. TTB stops at the first discriminating feature, meaning it will favor an option that excels on the highest-validity cue regardless of its values on all other cues. In contrast, WADD integrates all cues weighted by their validities, meaning that a deficit on the best cue can be compensated by advantages on several lesser cues. The trials include extreme compensations (where WADD strongly disagrees with TTB), partial compensations (where WADD predicts a tie but TTB strongly prefers one option), and baseline trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Take The Best (TTB) heuristic specifically on 'compensatory' trials. Compensatory trials are defined here as trials where the option favored by TTB has fewer overall positive features (1s) than the alternative. On these trials, TTB strongly prefers the option with the single best discriminating feature, whereas the Weighted Additive (WADD) rule integrates all features and strongly prefers the alternative because it has a higher quantity of lesser-validity features. Consequently, TTB will score high on this metric (near 1.0, modulo noise), while WADD will score low (near 0.0), producing a massive and stable between-theory gap.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.6173 (var=0.0123)
- Predicted under pi_2 (simulated): 0.3450 (var=0.0137)
- Observed on real data: 0.5920 (var=0.0046)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the compensatory Weighted Additive (WADD) rule from the non-compensatory Take The Best (TTB) heuristic. By using 5 features with linearly decreasing validities, we can construct trials where the single best discriminating feature strongly favors one option (driving TTB's choice), while the sum of multiple lower-validity features favors the other option (driving WADD's choice). We also include trials where the highest validity feature is tied, forcing TTB to drop to the next feature, while WADD integrates all features. This allows us to observe whether subjects compensate for a deficit on the most valid cue by accumulating evidence from lesser cues.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the trials where the Weighted Additive (WADD) rule and the Take The Best (TTB) heuristic make strictly opposing predictions. It calculates the proportion of these distinguishing trials where the subject's choice aligns with the WADD prediction. Under the WADD theory, this metric will be high (approaching 1.0 for low noise), whereas under the TTB theory, the subject's choices will systematically align with the TTB prediction, driving this metric low (approaching 0.0). By restricting only to trials with opposing predictions, it maximizes the discriminatory power between the compensatory and non-compensatory theories.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.3628 (var=0.0139)
- Predicted under pi_2 (simulated): 0.6883 (var=0.0243)
- Observed on real data: 0.4011 (var=0.0034)

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
**Predicted under pi_2:** 0.2325 (var=0.0158)

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
**Predicted under pi_2:** 0.5181 (var=0.0768)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (pi_3 - Take The Best with Probabilistic Stopping) consistently outperforms Theory 2 (pi_2 - Weighted Additive). In Experiment 1, the observed metric is 0.5920, closely matching pi_3's 0.6173 but far from pi_2's 0.3450. In Experiment 2, the observed metric is 0.4011, which aligns with pi_3's 0.3628 and strongly rejects pi_2's 0.6883 prediction. Experiments 3 and 4 further confirm this pattern: pi_3 closely tracks the empirical data, while pi_2 fails to capture the non-compensatory behavior observed in subjects. The data clearly shows that subjects rely on lexicographic, non-compensatory heuristics rather than fully integrating all feature validities. Since Theory 2 (WADD) is fundamentally mismatched to the data, it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory to replace the Weighted Additive (WADD) model. Since subjects exhibit predominantly non-compensatory behavior but with some noise (as captured well by Theory 1), a strong competitor could be a 'Strategy Selection' or 'Mixture' theory. This new theory would posit that a population consists of a majority of TTB users and a minority of WADD (or Tallying) users, capturing the fact that aggregate behavior is slightly softer than pure TTB. Alternatively, propose a 'Two-Stage Heuristic' where subjects first attempt a fast tally of features and only fall back to rigorous validity-based lexicographic search if the tally is too close."
}
```

## Usage

```json
{
  "prompt_token_count": 5660,
  "candidates_token_count": 393,
  "total_token_count": 6718
}
```
