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
Tallying (Equal-Weight) Heuristic: People evaluate options by simply counting the number of positive features (or cues favoring each option) and choosing the option with the higher total count. This compensatory strategy ignores the differential validities or subjective importance of different cues, treating all pieces of evidence equally. The choice probability is determined by a softmax over the total feature tallies for each option, combined with a uniform lapse rate. Crucially, the softmax temperature is constrained to produce softer choice probabilities, reflecting that humans do not apply the tallying rule completely deterministically.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tally the number of positive cues for each option
    score_a = np.sum(stim[0])
    score_b = np.sum(stim[1])
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.55, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 3: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Rationale:** This design dissociates the Tallying (Equal-Weight) heuristic from the Weighted Additive (WADD) model. Tallying simply counts the number of positive features and chooses the option with the higher count, completely ignoring cue validities. WADD, on the other hand, computes a weighted sum of the features based on their validities. By using a 6-feature design with a steep drop-off in validities between the top three and bottom three cues, we can pit quantity against quality. In several critical trials, Option A possesses a greater number of positive features (all from the low-validity tier), making it the clear choice for Tallying. However, Option B possesses fewer positive features but from the high-validity tier, such that their weighted sum exceeds that of Option A, making Option B the choice for WADD. Control trials where both models agree are included to balance the design and allow for accurate parameter estimation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In Trials 1, 2, and 3, Option A has exactly 3 positive features (all from the low-validity tier) and Option B has exactly 2 positive features (from the high-validity tier). Tallying evaluates options purely by counting features, so it will strongly prefer Option A (3 vs 2). WADD, conversely, weighs features by their validity, so the weighted sum for Option B will typically exceed that of Option A, causing it to prefer Option B. By measuring the proportion of Option A choices on these specific trials, we expect Tallying to yield a value significantly above 0.5, while WADD will yield a value below 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify critical trials where Option A has 3 positive features and Option B has 2.
    # In the provided design, these correspond to Trials 1, 2, and 3.
    # Tallying will consistently prefer A (3 > 2), whereas WADD will tend to prefer B
    # because B's features are drawn from the high-validity tier.
    is_target = data.apply(lambda row: sum(row['option_a_ratings']) == 3 and sum(row['option_b_ratings']) == 2, axis=1)
    
    target_data = data[is_target]
    
    if len(target_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(target_data['response'] == 0))

### RESULTS
- Predicted under pi_3 (simulated): 0.5928 (var=0.0078)
- Predicted under pi_2 (simulated): 0.4311 (var=0.0612)
- Observed on real data: 0.5678 (var=0.0020)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.99, 0.95, 0.55, 0.51, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the advocated Weighted Additive (WADD) model from the competing Tallying (Equal-Weight) heuristic. Tallying assumes subjects simply count the number of positive features and choose the option with the higher tally, ignoring the differential validities of the features. WADD, in contrast, computes a weighted sum of the features based on their validities. We use a 5-feature design with a steep drop-off in validities between the top two cues and the bottom three cues. In critical trials, one option possesses fewer positive features but from the high-validity tier (favored by WADD), while the other option possesses a greater number of positive features but from the low-validity tier (favored by Tallying). The design also includes trials where both options have an equal number of positive features, but one is favored by WADD due to higher validities, causing Tallying to predict a tie (guessing) while WADD predicts a deterministic choice.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design includes 'tie trials' where both options have the exact same number of positive features (Trials 5, 6, 7, 8). Tallying, which ignores feature validities and only counts positive features, evaluates these options as exactly equal. Consequently, Tallying always predicts a 50% chance of choosing either option on these trials, regardless of its temperature or lapse rate parameters. The empirical choice proportions for Tallying subjects will therefore tightly cluster around 0.5, with deviations solely due to binomial noise (yielding a very tight expected absolute deviation of ~0.111 for 12 repetitions). WADD, on the other hand, weights features by their subjective validities. Even when the number of positive features is equal, the weighted sums will almost never be exactly equal. Thus, WADD predicts a systematic preference for one option over the other on these tie trials. By measuring the absolute deviation of the empirical choice proportions from 0.5 on these trials, we can cleanly separate the two models: Tallying will exhibit low absolute deviations (pure binomial noise), while WADD will exhibit significantly higher absolute deviations reflecting its deterministic underlying preferences.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Find 'tie trials' where the number of positive features is equal for both options
    def is_tie(row):
        return sum(row['option_a_ratings']) == sum(row['option_b_ratings'])
        
    tie_mask = data.apply(is_tie, axis=1)
    tie_data = data[tie_mask].copy()
    
    if tie_data.empty:
        return 0.0
        
    # Create a hashable ID for each trial type to group by
    def make_id(row):
        a_str = ''.join([str(int(x)) for x in row['option_a_ratings']])
        b_str = ''.join([str(int(x)) for x in row['option_b_ratings']])
        return a_str + "_" + b_str
        
    tie_data['trial_id'] = tie_data.apply(make_id, axis=1)
    
    # Calculate the choice proportion for each subject and trial type
    means = tie_data.groupby(['subject_id', 'trial_id'])['response'].mean()
    
    # Compute absolute deviation from random guessing (0.5)
    abs_devs = (means - 0.5).abs()
    
    return float(abs_devs.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1133 (var=0.0021)
- Predicted under pi_2 (simulated): 0.2367 (var=0.0151)
- Observed on real data: 0.1133 (var=0.0017)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f1 = np.array([x[0] for x in data['option_a_ratings']])
    b_f1 = np.array([x[0] for x in data['option_b_ratings']])
    responses = data['response'].values
    
    favors_A = a_f1 > b_f1
    favors_B = b_f1 > a_f1
    
    aligned = (favors_A & (responses == 0)) | (favors_B & (responses == 1))
    return float(np.mean(aligned))
```

**Observed (real) value:** 0.4171 (var=0.0031)
**Predicted under pi_3:** 0.3875 (var=0.0065)
**Predicted under pi_2:** 0.3740 (var=0.0225)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_prediction(a, b):
        # Cues are already in descending order of validity
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            elif val_b > val_a:
                return 1
        return -1
        
    ttb_preds = np.array([get_ttb_prediction(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])])
    valid_mask = (ttb_preds != -1)
    
    if not np.any(valid_mask):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[valid_mask] == ttb_preds[valid_mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4083 (var=0.0021)
**Predicted under pi_3:** 0.3448 (var=0.0115)
**Predicted under pi_2:** 0.2510 (var=0.0145)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Tallying / Equal-Weight) substantially outperforms Theory 2 (Weighted Additive). In Experiment 1, subjects preferred the option with more positive features even when those features had lower validities, aligning with Tallying's prediction (>0.5) rather than WADD's (<0.5). In Experiment 2's tie trials (equal number of positive features but different validities), subjects showed no systematic preference (absolute deviation from 0.5 was ~0.11, perfectly matching Tallying's binomial noise prediction), completely rejecting WADD's prediction of a deterministic preference. Theory 1 also better captures the metrics in Experiments 3 and 4. Since WADD is systematically failing to capture the equal-weighting behavior of subjects, it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Take-The-Best' (TTB) heuristic or a 'Mixture of Tallying and TTB' model. As Tallying (a compensatory, equal-weight strategy) is currently the best fit, its natural cognitive competitor is a non-compensatory, one-reason decision making strategy like TTB. TTB evaluates cues sequentially in order of validity and stops at the first cue that discriminates between the options. Alternatively, since some subjects might use Tallying while others use TTB, a mixture model of these two heuristics could provide an even stronger competitor that captures individual differences better than a pure WADD model."
}
```

## Usage

```json
{
  "prompt_token_count": 5632,
  "candidates_token_count": 344,
  "total_token_count": 7271
}
```
