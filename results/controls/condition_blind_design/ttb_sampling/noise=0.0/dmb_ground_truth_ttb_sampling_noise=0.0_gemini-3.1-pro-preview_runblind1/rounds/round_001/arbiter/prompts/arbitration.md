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
Take-The-Best (TTB) assumes that decision-makers do not integrate all information or simply count features. Instead, they rank features by their subjective or instructed validity and compare options lexicographically. They stop at the first feature that discriminates between the two options and choose the one with the higher value on that feature. If all features tie, they guess.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(-validities, kind='mergesort')
    
    score_a = 0.0
    score_b = 0.0
    
    # Lexicographic comparison
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    epsilon = float(parameters["epsilon"])
    
    if score_a > score_b:
        p_core = np.array([1.0, 0.0])
    elif score_b > score_a:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = probabilities / probabilities.sum()
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
**Validities (n_features=4):** [0.95, 0.6, 0.87, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 2: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  trial 8: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 10: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 11: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 12: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 13: A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 15: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  trial 16: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Measures the proportion of choices that align with Take-The-Best (TTB) on a subset of 'compensatory' trials where WADD's subjective feature weighting could lead to the opposite choice. TTB is strictly non-compensatory and lexicographic, so it will show near-100% alignment on these trials, whereas WADD will show lower alignment due to variance in subjective weights across subjects.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 12: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
    # TTB chooses A (f0 discriminates). WADD might choose B if w2 is high and w0/w3 are low.
    mask_t12 = (a_tuples == (1, 0, 0, 1)) & (b_tuples == (0, 0, 1, 0))
    
    # Trial 15: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
    # TTB chooses A (f2 discriminates). WADD might choose B if w1 > w2.
    mask_t15 = (a_tuples == (1, 0, 1, 0)) & (b_tuples == (1, 1, 0, 0))
    
    # Trial 4: A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
    # TTB chooses B (f0 discriminates). WADD might choose A if w2 is high and w0 is low.
    mask_t4 = (a_tuples == (0, 1, 1, 0)) & (b_tuples == (1, 1, 0, 0))
    
    ttb_aligned = 0
    total = 0
    
    if mask_t12.sum() > 0:
        ttb_aligned += (data.loc[mask_t12, 'response'] == 0).sum()
        total += mask_t12.sum()
        
    if mask_t15.sum() > 0:
        ttb_aligned += (data.loc[mask_t15, 'response'] == 0).sum()
        total += mask_t15.sum()
        
    if mask_t4.sum() > 0:
        ttb_aligned += (data.loc[mask_t4, 'response'] == 1).sum()
        total += mask_t4.sum()
        
    if total == 0:
        return 0.5
        
    return float(ttb_aligned) / total

### RESULTS
- Predicted under pi_3 (simulated): 0.8889 (var=0.0114)
- Predicted under pi_2 (simulated): 0.5856 (var=0.0243)
- Observed on real data: 0.8644 (var=0.0213)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.57, 0.58, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 3: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 11: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 14: A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 15: A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates a single, highly discriminative 'compensatory' trial (Trial 11). On this trial, the single most valid feature (validity 0.95) strongly favors Option A. Take-The-Best (TTB) uses a lexicographic rule and will reliably choose Option A, resulting in a mean response near 0 (modulo lapse rate). In contrast, the Weighted Additive rule (WADD) integrates all features. The combined validity of the features favoring Option B (0.58 + 0.55 = 1.13) outweighs the single feature favoring Option A (0.95). Because WADD's random feature weights are independently drawn, the sum of the weighted evidence for B will frequently exceed the weighted evidence for A, causing WADD to choose Option B (response 1) on average more than 50% of the time. This creates a massive gap in expected choice proportion between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    # Trial 11 is the most discriminative trial:
    # Option A: [1, 1, 0, 0] vs Option B: [0, 1, 1, 1]
    # TTB looks at the highest validity feature (F0: 0.95), where A=1 and B=0, so TTB strictly prefers A (response 0).
    # WADD computes a weighted sum. Since F1 cancels out, WADD compares w0*0.95 (for A) against w2*0.58 + w3*0.55 (for B).
    # Because the weights w0, w2, w3 are independently drawn from U(0,1), the sum w2*0.58 + w3*0.55 is frequently greater than w0*0.95.
    # Thus, WADD will choose B (response 1) more than half the time, whereas TTB will choose B almost never (only on lapses).
    
    a_target = [1, 1, 0, 0]
    b_target = [0, 1, 1, 1]
    
    mask = (data['option_a_ratings'].apply(lambda x: list(x) == a_target) & 
            data['option_b_ratings'].apply(lambda x: list(x) == b_target))
            
    if not mask.any():
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1633 (var=0.0283)
- Predicted under pi_2 (simulated): 0.6033 (var=0.1021)
- Observed on real data: 0.1800 (var=0.0309)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_favored_choices = 0
    tied_trials = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a == (0, 1, 0, 1) and b == (1, 1, 0, 0):
            wadd_favored = 1
        elif a == (1, 0, 1, 1) and b == (1, 1, 1, 0):
            wadd_favored = 1
        elif a == (1, 1, 1, 0) and b == (0, 1, 1, 1):
            wadd_favored = 0
        elif a == (1, 1, 0, 1) and b == (1, 1, 1, 0):
            wadd_favored = 1
        else:
            continue
            
        tied_trials += 1
        if row['response'] == wadd_favored:
            wadd_favored_choices += 1
            
    if tied_trials == 0:
        return 0.5
    return float(wadd_favored_choices / tied_trials)
```

**Observed (real) value:** 0.8667 (var=0.0063)
**Predicted under pi_3:** 0.8450 (var=0.0131)
**Predicted under pi_2:** 0.6717 (var=0.0328)

### Experiment 4
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.8, 0.85, 0.55])
    wadd_consistent = 0
    tally_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where Tallying predicts a tie
        if a_wins == b_wins:
            score_a = np.sum(a * validities)
            score_b = np.sum(b * validities)
            
            if score_a > score_b:
                wadd_choice = 0
            elif score_b > score_a:
                wadd_choice = 1
            else:
                continue
                
            if row['response'] == wadd_choice:
                wadd_consistent += 1
            tally_ties += 1
            
    if tally_ties == 0:
        return 0.5
    return float(wadd_consistent / tally_ties)
```

**Observed (real) value:** 0.8767 (var=0.0106)
**Predicted under pi_3:** 0.8683 (var=0.0128)
**Predicted under pi_2:** 0.5183 (var=0.0651)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Take-The-Best) consistently and accurately predicts the observed data, whereas Theory 2 (Weighted Additive) fails to capture the empirical patterns. The metrics specifically targeted compensatory versus non-compensatory choice behavior, and the human subjects overwhelmingly followed the non-compensatory, lexicographic pattern predicted by TTB. Since WADD's predictions are far off the mark, it is not a competitive baseline in its current form.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Tallying' (Equal Weight) theory as a new competitor to Take-The-Best. Tallying ignores cue validities entirely and simply counts the number of positive features for each option, choosing the one with the higher count (and guessing on a tie). This provides a distinct, cognitively plausible heuristic alternative that is compensatory with respect to features but non-compensatory with respect to validities, serving as a stronger and simpler benchmark than WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 6607,
  "candidates_token_count": 231,
  "total_token_count": 7453
}
```
