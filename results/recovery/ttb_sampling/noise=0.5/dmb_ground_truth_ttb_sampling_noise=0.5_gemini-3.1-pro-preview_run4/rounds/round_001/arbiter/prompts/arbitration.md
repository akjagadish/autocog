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
People use a non-compensatory heuristic called 'Take The Best' (TTB) to choose between options. They search through features in descending order of subjective validity, stopping at the first feature that discriminates between the two options, and choose the option with the higher value on that feature. If no features discriminate, they guess randomly.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = 0.0
    b_wins = 0.0
    
    # Search for the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            a_wins = 1.0
            break
        elif b[idx] > a[idx]:
            b_wins = 1.0
            break
            
    # If no feature discriminates, baseline preference is uniform
    if a_wins == 0.0 and b_wins == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([a_wins, b_wins])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates Take The Best (TTB) from Weighted Additive (WADD) by contrasting a single highly-valid feature against multiple lower-validity features. In each trial, Option A is superior on the highest-validity feature that discriminates between the two options. TTB, being a non-compensatory lexicographic heuristic, will invariably select Option A because it stops searching as soon as it finds this first discriminating feature. In contrast, Option B is designed to be superior on a larger number of lower-validity features such that their sum outweighs the single highest-validity feature. WADD, being a compensatory model, will integrate all features and therefore consistently prefer Option B.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In this design, every trial pits a single high-validity feature (Option A) against a larger number of lower-validity features (Option B) such that the sum of the latter exceeds the former. Take The Best (TTB) uses a non-compensatory lexicographic rule, stopping at the first discriminating feature, and thus always chooses Option A. Weighted Additive (WADD) integrates all features, and since Option B has a higher weighted sum, it consistently chooses Option B. The proportion of B choices (mean of the response column) will be near 0 for TTB and near 1 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    # The experimental design guarantees that Option A always wins on the highest-validity discriminating feature.
    # TTB will therefore consistently choose Option A (response = 0).
    # Option B always has more lower-validity features whose sum outweighs Option A's single advantage.
    # WADD will therefore consistently choose Option B (response = 1).
    # The mean of the response column perfectly separates the two models: TTB is near 0, WADD is near 1.
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2602 (var=0.0220)
- Predicted under pi_2 (simulated): 0.6813 (var=0.0244)
- Observed on real data: 0.3167 (var=0.0037)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic by pitting a single, highly-valid feature against multiple lower-validity features. In the critical trials, Option A is superior on the single most valid feature that discriminates between the two options. TTB, which stops searching at the first discriminating feature, will invariably choose Option A. However, Option B is superior on several lower-validity features whose combined validities exceed that of the single feature favored by Option A. WADD integrates all features and will therefore consistently prefer Option B. We vary the specific validities involved (e.g., the top feature vs. features 2 and 3; the second feature vs. features 3 and 4) to ensure the dissociation is robust across the validity hierarchy.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
By design, every trial in this experiment pits a single highly-valid feature (favoring Option A) against multiple lower-validity features (favoring Option B). The non-compensatory Take The Best (TTB) model stops at the single best discriminating feature and will invariably choose Option A (response = 0). The compensatory Weighted Additive (WADD) model integrates all features; because the sum of the lower validities exceeds the single highest validity, WADD will predominantly choose Option B (response = 1). Therefore, the overall proportion of choosing Option B (the mean of the `response` column) will be near 0 for TTB (modulo lapse rate) and near 1 for WADD (modulo choice noise), providing a massive and stable quantitative dissociation between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2858 (var=0.0237)
- Predicted under pi_2 (simulated): 0.6787 (var=0.0150)
- Observed on real data: 0.3229 (var=0.0034)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    
    agreements = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        a_wadd = np.sum(a * validities)
        b_wadd = np.sum(b * validities)
        
        # Check if both models have strict preferences
        if a_wins != b_wins and a_wadd != b_wadd:
            tally_pref = 0 if a_wins > b_wins else 1
            wadd_pref = 0 if a_wadd > b_wadd else 1
            
            # Focus only on trials where Tallying and WADD disagree
            if tally_pref != wadd_pref:
                agreements.append(1.0 if row['response'] == tally_pref else 0.0)
                
    if len(agreements) == 0:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.3133 (var=0.0068)
**Predicted under pi_3:** 0.2700 (var=0.0226)
**Predicted under pi_2:** 0.4407 (var=0.0546)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option A as a 2D array
    A = np.vstack(data['option_a_ratings'].values)
    
    # Identify Trial 1: A = [0, 0, 1, 1, 1, 0]
    # WADD prefers B (response 1), Tallying prefers A (response 0)
    is_t1 = (A[:, 0] == 0) & (A[:, 1] == 0) & (A[:, 2] == 1) & (A[:, 3] == 1)
    
    # Identify Trial 3: A = [1, 1, 0, 0, 0, 0]
    # WADD prefers A (response 0), Tallying prefers B (response 1)
    is_t3 = (A[:, 0] == 1) & (A[:, 1] == 1) & (A[:, 2] == 0) & (A[:, 3] == 0)
    
    responses = data['response'].values
    
    wadd_match = 0
    total = 0
    
    wadd_match += np.sum(responses[is_t1] == 1)
    total += np.sum(is_t1)
    
    wadd_match += np.sum(responses[is_t3] == 0)
    total += np.sum(is_t3)
    
    return float(wadd_match / total) if total > 0 else 0.5

```

**Observed (real) value:** 0.7188 (var=0.0044)
**Predicted under pi_3:** 0.7688 (var=0.0284)
**Predicted under pi_2:** 0.5719 (var=0.0907)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Take The Best) significantly outperforms Theory 2 (Weighted Additive). The observed metrics in Experiments 1 and 2 (0.3167 and 0.3229) closely align with TTB's predictions (~0.26 and ~0.28) and are far from WADD's predictions (~0.68). Similarly, in Experiments 3 and 4, TTB provides much more accurate predictions of the observed metrics than WADD. Because TTB captures the data well, it should be retained. WADD completely fails to capture the human behavior in these specific binary-feature decision tasks and should be entirely replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Weighted Additive (WADD) theory with a 'Tallying' (Equal Weights) heuristic theory. Tallying assumes that decision-makers ignore the specific validities of the features and simply count the number of positive features (or the number of features favoring each option), choosing the option with the highest count. This will provide another bounded-rationality heuristic to contrast against the successful Take The Best theory."
}
```

## Usage

```json
{
  "prompt_token_count": 5360,
  "candidates_token_count": 268,
  "total_token_count": 6260
}
```
