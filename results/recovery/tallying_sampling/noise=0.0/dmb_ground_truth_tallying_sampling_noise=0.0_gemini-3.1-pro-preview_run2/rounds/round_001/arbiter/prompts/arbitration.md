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
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate Tallying (Equal Weight) from Weighted Additive (WADD), we utilize a 5-feature design where the sum of the validities of the two most valid cues exceeds the sum of the remaining three less valid cues. In the critical dissociation trials, one option is endorsed by the two highly valid cues, while the competing option is endorsed by the three lower-validity cues. Tallying ignores cue validities and simply counts the positive features, thus strictly preferring the option with three endorsements. WADD, however, computes a weighted sum and will prefer the option with the two highly valid cues. We also include trials where both models agree, and trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strong preference, allowing for a precise estimation of the models' parameters.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices favoring the Tallying model's prediction on the critical dissociation trials (Trials 1 and 2). In these trials, one option is endorsed by the two most valid cues (sum of validities = 1.80) and the other by the three least valid cues (sum of validities = 1.65). Tallying simply counts the features and robustly prefers the option with three positive features. WADD, evaluating the weighted validities, will typically favor the option with the two highly valid features. The metric will return values close to 1.0 for data simulated under Tallying, and substantially lower values (often below 0.5) for data simulated under WADD.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.8387 (var=0.0187)
- Predicted under pi_2 (simulated): 0.4719 (var=0.1069)
- Observed on real data: 0.8213 (var=0.0136)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) model from the Tallying (Equal Weight) model. By using a 5-feature design with a steep drop-off in validities between the top two features and the bottom three, we create critical dissociation trials where one option is supported by the two highly valid features (WADD preference) while the other option is supported by the three lower-validity features (Tallying preference). We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference due to validity weighting, and baseline trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of WADD-compliant choices on the critical dissociation trials (Trials 1 and 7). On these trials, one option has two high-validity features ([1, 1, 0, 0, 0]) while the other has three low-validity features ([0, 0, 1, 1, 1]). The Weighted Additive (WADD) model strongly prefers the former, whereas the Tallying (Equal Weight) model strictly prefers the latter because it counts more positive features. The metric will yield a high value for WADD and a low value for Tallying, perfectly discriminating the two models.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.1467 (var=0.0120)
- Predicted under pi_2 (simulated): 0.5417 (var=0.0901)
- Observed on real data: 0.1383 (var=0.0112)

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
**Predicted under pi_2:** 0.2608 (var=0.0530)

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
**Predicted under pi_2:** 0.3596 (var=0.0246)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Tallying / Equal Weight) consistently matches the observed data remarkably well. The observed data shows strong preference for options with a higher raw count of positive features, largely ignoring the specific validities of those features. Theory 2 (Weighted Additive / WADD) consistently fails to capture this behavior, predicting a much stronger sensitivity to cue validities than is actually observed in the human data. In Experiment 1, the observed proportion of Tallying choices was 0.82, perfectly aligning with Theory 1's prediction (0.84) and far from Theory 2's prediction (0.47). Similar patterns hold for Experiment 2 and the other historical experiments. Because Tallying is such a strong baseline, WADD is effectively falsified here.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory to replace WADD that acts as a stronger competitor to pure Tallying. Since pure Tallying captures the data well but ignores validities entirely, a compelling alternative would be 'Thresholded Tallying' (or 'Subset Tallying'): subjects still use an equal-weight tallying heuristic, but they only count features whose validities exceed a certain subjective threshold (e.g., only tallying 'important' features and ignoring the rest). This introduces a parameter for the validity threshold, blending the simplicity of Tallying with a coarse sensitivity to cue validities."
}
```

## Usage

```json
{
  "prompt_token_count": 5543,
  "candidates_token_count": 329,
  "total_token_count": 6930
}
```
