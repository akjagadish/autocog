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
Tallying (Equal Weighting) posits that decision-makers simply count the total number of positive features for each option, ignoring the differential validities of the cues. The option with the higher count of positive features is preferred. This represents a compensatory but highly frugal heuristic, where evidence is accumulated equally across all available cues. If the counts are equal, the decision-maker guesses. Response noise is modeled via a softmax over these counts with an independent lapse rate. To account for empirical choices that often deviate from pure tallying on conflict trials, the decision process incorporates substantial choice noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    return np.random.choice(len(probabilities), p=probabilities)


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

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Tallying (Equal Weighting) from Weighted Additive (WADD), we must construct trials where the two models make divergent predictions based on their core mechanisms. Tallying assumes decision-makers count the number of positive features for each option, ignoring the validities (weights) of the cues. In contrast, WADD computes a weighted sum of features, heavily prioritizing cues with higher validities. By using a 5-feature design with a steep drop-off in validities, we can create 'conflict' trials where one option has fewer positive features (disfavored by Tallying) but those features have much higher validities (favored by WADD). We also include 'tie' trials where both options have the same number of positive features (Tallying predicts guessing) but differ in the validities of those features (WADD predicts a clear preference), as well as 'agreement' trials where both models favor the same option. This design provides maximum leverage to distinguish uniform evidence accumulation from validity-weighted compensatory integration.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option [1, 1, 0, 0, 0] when it is paired with [0, 0, 1, 1, 1]. The Tallying model predicts a preference for [0, 0, 1, 1, 1] because it has 3 positive features compared to 2. However, the Weighted Additive (WADD) model predicts a preference for [1, 1, 0, 0, 0] because the combined validities of the first two features (0.95 + 0.85 = 1.8) outweigh the combined validities of the last three (0.6 + 0.55 + 0.5 = 1.65). Therefore, WADD will score highly on this metric, while Tallying will score very low.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def is_11000(x):
        return tuple(x) == (1, 1, 0, 0, 0)
    
    def is_00111(x):
        return tuple(x) == (0, 0, 1, 1, 1)
        
    a_11000 = data['option_a_ratings'].apply(is_11000)
    b_00111 = data['option_b_ratings'].apply(is_00111)
    
    a_00111 = data['option_a_ratings'].apply(is_00111)
    b_11000 = data['option_b_ratings'].apply(is_11000)
    
    t1 = a_11000 & b_00111
    t3 = a_00111 & b_11000
    
    chose_11000 = (t1 & (data['response'] == 0)) | (t3 & (data['response'] == 1))
    
    relevant = t1 | t3
    if relevant.sum() == 0:
        return 0.5
    return float(chose_11000.sum() / relevant.sum())

### RESULTS
- Predicted under pi_3 (simulated): 0.3817 (var=0.0195)
- Predicted under pi_2 (simulated): 0.5667 (var=0.0891)
- Observed on real data: 0.4183 (var=0.0241)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) model from Tallying (Equal Weighting) by exploiting trials where the models diverge due to feature validity integration. Tallying simply counts the number of positive features, whereas WADD integrates both the presence of features and their respective validities. By using a 5-feature design with a steep drop-off in validities, we create strong dissociation trials: for instance, Option A has fewer positive features but they are highly valid (WADD favors A), while Option B has more positive features with lower validities (Tallying favors B). We also include 'tie' trials where Tallying predicts guessing but WADD predicts a clear preference, and agreement trials to properly estimate baseline noise and specific parameter distributions.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Tallying model on the specific dissociation trials (trials 1 and 6). On these trials, one option has fewer positive features but higher validities (favored by WADD), while the other has more positive features but lower validities (favored by Tallying). Tallying strongly predicts choosing the option with more features, leading to a high expected value for this metric. Conversely, WADD predicts a preference for the option with fewer features, leading to an expected value below 0.5. By isolating only these strong dissociation trials, the metric maximizes the discriminative gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        # Target trials where WADD and Tallying strictly disagree
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return True
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return True
        return False

    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    sum_a = target_data['option_a_ratings'].apply(sum)
    sum_b = target_data['option_b_ratings'].apply(sum)
    
    # Calculate how often the subject chose the option with MORE positive features (Tallying's preference)
    chose_more = ((target_data['response'] == 0) & (sum_a > sum_b)) | \
                 ((target_data['response'] == 1) & (sum_b > sum_a))
                 
    return float(chose_more.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.6075 (var=0.0135)
- Predicted under pi_2 (simulated): 0.4008 (var=0.1003)
- Observed on real data: 0.5867 (var=0.0125)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_0 = data['option_b_ratings'].apply(lambda x: x[0])
    ttb_choice = np.where(b_0 > a_0, 1, 0)
    return float(np.mean(data['response'] == ttb_choice))
```

**Observed (real) value:** 0.4550 (var=0.0021)
**Predicted under pi_3:** 0.4154 (var=0.0064)
**Predicted under pi_2:** 0.4006 (var=0.0278)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.75, 0.65, 0.55])
    
    def is_ttb_match(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_winner = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        # WADD expected prediction (assuming uniform weights)
        wadd_a = np.sum(a * validities)
        wadd_b = np.sum(b * validities)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        # Only consider compensatory trials where the models disagree
        if ttb_winner is not None and ttb_winner != wadd_winner:
            return 1.0 if row['response'] == ttb_winner else 0.0
        return np.nan

    matches = data.apply(is_ttb_match, axis=1)
    return float(matches.mean())
```

**Observed (real) value:** 0.4225 (var=0.0057)
**Predicted under pi_3:** 0.3733 (var=0.0167)
**Predicted under pi_2:** 0.3225 (var=0.0383)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Tallying) consistently outperforms Theory 2 (WADD). In Experiment 1, the observed metric is 0.418, which is much closer to Tallying's prediction (0.381) than WADD's (0.566). In Experiment 2, Tallying's prediction (0.607) closely matches the observed data (0.586), whereas WADD predicts the opposite preference (0.400). Experiments 3 and 4 further confirm that subjects are not using compensatory validity weighting as WADD predicts. Since Tallying is already capturing the data quite well, WADD is clearly degenerate and should be replaced with a stronger alternative.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace WADD with a fundamentally different non-compensatory theory, such as 'Take-The-Best' (TTB) or a Lexicographic heuristic. Instead of computing a weighted sum across all features, TTB posits that decision-makers search through cues in descending order of validity and stop at the first cue that discriminates between the options. This will provide a strong non-compensatory contrast to the equally-weighted compensatory accumulation of Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 5551,
  "candidates_token_count": 291,
  "total_token_count": 6666
}
```
