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
Tallying (Equal Weight) Theory: People make choices by simply counting the number of positive features for each option, completely ignoring cue validities. The option with the highest total count of positive features is chosen. This compensatory heuristic trades off the cognitive effort of weighting cues against accuracy, assuming all cues are equally important. Noise is introduced via a softmax over the feature counts and a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: Simply sum the features for each option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** The design contrasts Tallying (Equal Weight) with WADD by setting up a stark opposition between feature count and feature validity. By using a skewed validity distribution (two highly valid cues and three less valid cues), we can create 'dissociation trials' where one option has more positive features (favored by Tallying) but the other option has a higher validity-weighted sum (favored by WADD). Including trials where the two models agree ensures a robust behavioral baseline.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the probability of choosing Option A on the specific 'dissociation' trial where the two theories make strictly opposing predictions. In this trial, Option A has a higher tally of positive features (3 vs 2), so the Tallying theory predicts a strong preference for A. However, Option B possesses the two most valid features (validities 0.95 and 0.9), so the Weighted Additive (WADD) theory assigns a higher expected weighted sum to B, predicting a preference for B. Because the theories are separated by a large margin in their predictions for this specific trial, the difference in the metric means will be substantial, and the within-theory variance will be tightly clustered.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify the critical dissociation trial (Trial 1 in the design)
    # Option A has more positive features (3 vs 2), favoring Tallying.
    # Option B has the two most valid features, favoring WADD.
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
                 
    if is_trial_1.sum() == 0:
        return 0.5
        
    # Return the proportion of times the subject chose Option A (response == 0)
    return float(np.mean(data.loc[is_trial_1, 'response'] == 0))

### RESULTS
- Predicted under pi_3 (simulated): 0.8667 (var=0.0111)
- Predicted under pi_2 (simulated): 0.4200 (var=0.0778)
- Observed on real data: 0.7067 (var=0.0162)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) theory from the Tallying (Equal Weight) theory, we manipulate the trade-off between feature count and feature validity. Tallying predicts choices strictly based on the unweighted sum of positive features. WADD predicts choices based on the validity-weighted sum. By utilizing a skewed validity distribution (two highly predictive cues and three less predictive cues), we create strong dissociation trials where one option has fewer positive features but higher overall validity (favored by WADD), while the other has more positive features but lower overall validity (favored by Tallying). We also include trials where Tallying predicts a tie but WADD has a strict preference, and baseline trials where both models agree, ensuring a robust test of whether participants integrate cue validities or simply count features.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on Trial 1, which provides a strong dissociation between the two models. In this trial, one option has fewer positive features but they are the most predictive ones (validities 0.95 and 0.85), while the other option has more positive features but they are less predictive. Tallying will almost deterministically choose the option with more features (ignoring validity), whereas WADD will tend to choose the option with fewer, high-validity features. The metric calculates the proportion of times the WADD-favored option is chosen.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_wadd_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        # Trial 1: WADD favors the option with fewer, but higher-validity features.
        # Tallying strictly favors the option with more features.
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return 0 == row['response']
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return 1 == row['response']
        return np.nan

    choices = data.apply(get_wadd_choice, axis=1).dropna()
    
    if len(choices) == 0:
        return 0.5
        
    return float(choices.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1338 (var=0.0122)
- Predicted under pi_2 (simulated): 0.5613 (var=0.1096)
- Observed on real data: 0.3050 (var=0.0157)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for j in range(len(a)):
            if a[j] != b[j]:
                ttb_pred = 0 if a[j] > b[j] else 1
                if row['response'] == ttb_pred:
                    ttb_matches += 1
                total += 1
                break
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.3511 (var=0.0061)
**Predicted under pi_3:** 0.1813 (var=0.0063)
**Predicted under pi_2:** 0.3147 (var=0.0155)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.6604 (var=0.0042)
**Predicted under pi_3:** 0.8125 (var=0.0084)
**Predicted under pi_2:** 0.6256 (var=0.0201)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, neither Tallying (THEORY 1) nor WADD (THEORY 2) perfectly captures the observed behavior. In Experiments 1 and 2, the observed metric falls between the predictions of Tallying and WADD, though slightly closer to Tallying in Exp 1 and Tallying in Exp 2. In Experiments 3 and 4, WADD provides a substantially better fit to the observed data than Tallying. The fact that the observed data consistently falls between the two extremes of equal-weighting (Tallying) and fully compensatory validity-weighting (WADD) suggests that either the population is a mixture of strategy users, or individuals are using a different boundedly rational heuristic, such as Take-The-Best (TTB), which relies heavily on the most valid cues but ignores the rest.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace the Tallying theory with a 'Take-The-Best' (TTB) theory or a Strategy Mixture theory. TTB posits that decision-makers search through features in descending order of their cue validity, stopping at the very first feature that discriminates between the two options and choosing the option favored by that cue. Alternatively, a Strategy Mixture theory would suggest that a proportion of the population uses WADD while the rest uses Tallying, which would naturally explain why aggregate observed metrics consistently fall between the pure predictions of THEORY 1 and THEORY 2."
}
```

## Usage

```json
{
  "prompt_token_count": 4980,
  "candidates_token_count": 330,
  "total_token_count": 6110
}
```
