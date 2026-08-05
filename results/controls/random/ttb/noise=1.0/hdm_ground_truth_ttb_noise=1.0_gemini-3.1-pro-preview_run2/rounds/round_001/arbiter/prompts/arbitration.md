# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take The Best (TTB) Heuristic: Decision makers process information sequentially, searching through features in descending order of their subjective validity. They stop at the first feature that discriminates between the options (i.e., the absolute difference in feature values exceeds a certain threshold) and choose the option favored by that single cue. All other lower-validity features are ignored. If no cue discriminates, they guess. This represents a classic fast-and-frugal one-reason decision-making benchmark.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        diff = a[idx] - b[idx]
        if abs(diff) > threshold:
            if diff > 0:
                p_core = np.array([1.0, 0.0])
            else:
                p_core = np.array([0.0, 1.0])
            break
            
    # Incorporate lapse rate (epsilon)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=3):** [0.9, 0.75, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[7, 2, 2]  B=[5, 9, 9]
  trial 2: A=[4, 8, 8]  B=[6, 1, 1]
  trial 3: A=[8, 3, 3]  B=[7, 9, 8]
  trial 4: A=[5, 10, 10]  B=[8, 0, 0]
  trial 5: A=[9, 1, 1]  B=[7, 8, 8]
  trial 6: A=[3, 9, 9]  B=[6, 2, 2]
  trial 7: A=[6, 5, 5]  B=[5, 10, 10]
  trial 8: A=[2, 7, 7]  B=[4, 1, 1]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit the non-compensatory nature of TTB versus the compensatory nature of WADD. We use 3 features with varying validities (0.9, 0.75, 0.6) and a large rating scale (rating_max=10). In the designed trials, one option always has a small advantage on the most valid feature (cue 1), which TTB uses to exclusively determine its choice (assuming a low threshold). However, the other option has massive advantages on the remaining, slightly less valid features. Because WADD integrates all feature values weighted by their validities, it will strongly prefer the option that compensates for a small loss on the most valid cue with large gains on the others. This creates diametrically opposed predictions between the two models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experiment is designed so that the option with a small advantage on the most valid cue (Cue 1) suffers a massive disadvantage on the remaining cues. Because WADD is compensatory, it will almost always favor the option that is worse on Cue 1 but better on Cues 2 and 3. In contrast, TTB will either choose the option that is better on Cue 1 (if its threshold is small enough) or guess (if its threshold is large, which is very common given the parameter distribution up to 50). Therefore, the proportion of choices aligning with the best option on Cue 1 will be close to 0 for WADD, but much closer to 0.5 for TTB.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue1_a = np.array([x[0] for x in data['option_a_ratings']])
    cue1_b = np.array([x[0] for x in data['option_b_ratings']])
    
    a_better = (cue1_a > cue1_b)
    b_better = (cue1_b > cue1_a)
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    aligned = (a_better & chose_a) | (b_better & chose_b)
    
    return float(aligned.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.4906 (var=0.0131)
- Predicted under pi_2 (simulated): 0.0250 (var=0.0206)
- Observed on real data: 0.5138 (var=0.0022)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[6, 1, 1, 1]  B=[5, 10, 10, 10]
  trial 2: A=[7, 1, 1, 1]  B=[5, 10, 10, 10]
  trial 3: A=[8, 1, 1, 1]  B=[5, 10, 10, 10]
  trial 4: A=[9, 1, 1, 1]  B=[5, 10, 10, 10]
  trial 5: A=[10, 1, 1, 1]  B=[5, 10, 10, 10]
  trial 6: A=[10, 2, 2, 2]  B=[0, 10, 10, 10]
  trial 7: A=[3, 10, 10, 10]  B=[8, 1, 1, 1]
  trial 8: A=[10, 5, 5, 5]  B=[0, 6, 6, 6]

**Rationale:** To quantitatively dissociate Weighted Additive (WADD) from Take The Best (TTB), we must target the core difference in how they process feature magnitudes and validities. TTB acts lexicographically: it searches for the first feature (ordered by validity) where the difference between options exceeds a threshold, ignoring all lower-validity cues. WADD is compensatory, integrating all features weighted by their validities. In this design, we use 4 cardinal features (rating_max=10). We present trials where Option A holds a varying advantage on the single highest-validity feature, while Option B holds a massive advantage on all remaining lower-validity features. For WADD, the large advantages on multiple lower-validity features easily compensate for a small-to-moderate loss on the highest-validity feature, leading to a strong preference for Option B. For TTB, if the difference on the highest-validity feature exceeds its threshold, it will decisively choose Option A, completely ignoring Option B's overwhelming advantage elsewhere. By systematically varying the difference on the highest-validity feature across trials, we can precisely identify TTB's threshold parameter while observing WADD's smooth, compensatory choice probabilities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the difference in choice probability for Option B between Trial 1 and Trial 7. In Trial 1, Option A has a tiny advantage (+1) on the most valid feature, while Option B has a massive advantage (+9) on all other features. WADD's compensatory nature easily overrides the small deficit on feature 1, strongly preferring B (response ~ 1). In Trial 7, Option B has a +5 advantage on the most valid feature, but Option A has a +9 advantage on all others. WADD again compensates and strongly prefers A (response ~ 0). Thus, WADD's metric score is roughly 1.0 - 0.0 = 1.0. TTB, on the other hand, evaluates features lexicographically. Because TTB's threshold parameter is drawn uniformly from [0, 50], 82% of simulated TTB subjects will have a threshold >= 9, meaning they will guess on both Trial 1 and Trial 7, yielding a metric score of 0.5 - 0.5 = 0.0. This creates a massive mean difference between the theories (~1.0 vs ~0.06) with very low between-subject variance, cleanly discriminating them.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Trial 1: A=[6, 1, 1, 1], B=[5, 10, 10, 10]
    t1_mask = data['option_a_ratings'].apply(lambda x: list(x) == [6, 1, 1, 1])
    # Trial 7: A=[3, 10, 10, 10], B=[8, 1, 1, 1]
    t7_mask = data['option_a_ratings'].apply(lambda x: list(x) == [3, 10, 10, 10])
    
    t1_resp = data[t1_mask]['response'].mean()
    t7_resp = data[t7_mask]['response'].mean()
    
    if pd.isna(t1_resp):
        t1_resp = 0.5
    if pd.isna(t7_resp):
        t7_resp = 0.5
        
    return float(t1_resp - t7_resp)

### RESULTS
- Predicted under pi_3 (simulated): 0.0450 (var=0.0420)
- Predicted under pi_2 (simulated): 1.0000 (var=0.0000)
- Observed on real data: 0.0033 (var=0.0308)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[5, 5, 5, 5]  B=[4, 4, 4, 10]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 10, 0, 0]  B=[1, 0, 1, 1]
  A=[5, 5, 6, 6]  B=[6, 6, 0, 0]
  A=[2, 2, 2, 2]  B=[0, 0, 10, 10]
  A=[4, 4, 4, 4]  B=[3, 3, 5, 5]
  A=[0, 10, 10, 10]  B=[10, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    relevant_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        
        # Trial 1: A=[1, 6, 6, 6], B=[10, 5, 5, 5]
        if a[0] == 1 and a[1] == 6 and a[2] == 6:
            tallying_consistent += (1 if r == 0 else 0)
            relevant_trials += 1
            
        # Trial 3: A=[10, 0, 0, 0], B=[0, 1, 1, 1]
        elif a[0] == 10 and a[1] == 0 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
        # Trial 4: A=[0, 10, 0, 0], B=[1, 0, 1, 1]
        elif a[0] == 0 and a[1] == 10 and a[2] == 0:
            tallying_consistent += (1 if r == 1 else 0)
            relevant_trials += 1
            
    return float(tallying_consistent / relevant_trials) if relevant_trials > 0 else 0.5
```

**Observed (real) value:** 0.5056 (var=0.0050)
**Predicted under pi_3:** 0.4572 (var=0.0238)
**Predicted under pi_2:** 0.1333 (var=0.0578)

### Experiment 4
**Design**
  A=[2, 6, 6]  B=[10, 5, 5]
  A=[8, 4, 3]  B=[2, 5, 4]
  A=[10, 0, 5]  B=[0, 10, 5]
  A=[0, 10, 5]  B=[10, 0, 5]
  A=[5, 5, 5]  B=[1, 6, 6]
  A=[8, 2, 2]  B=[10, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    validities = np.array([0.9, 0.6, 0.5])
    
    score_a = a_ratings @ validities
    score_b = b_ratings @ validities
    
    wadd_pred = (score_b > score_a).astype(int)
    
    responses = data['response'].values
    return float(np.mean(responses == wadd_pred))

```

**Observed (real) value:** 0.4963 (var=0.0025)
**Predicted under pi_3:** 0.5246 (var=0.0059)
**Predicted under pi_2:** 0.8433 (var=0.0738)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the observed metric values hover remarkably close to 0.5, suggesting that participants are effectively guessing or that their aggregate behavior is indistinguishable from random choice. THEORY 1 (pi_3, Take The Best) captures this well primarily because its threshold parameter is allowed to be very large (up to 50, exceeding the maximum rating of 10), which forces the model to find no discriminating features and default to random guessing. THEORY 2 (pi_2, WADD) completely fails to capture the data, predicting extreme compensatory behavior (e.g., ~1.0 in Exp 2 and ~0.02 in Exp 1) that is not borne out by the ~0.5 observed values.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "The current WADD theory (THEORY 2) is degenerate for this dataset as participants do not exhibit strong compensatory behavior. Given that aggregate choices are consistently near 0.5, replace THEORY 2 with a 'Random Guessing / High-Lapse Baseline' theory, or a 'Tallying' heuristic with a very high baseline noise/lapse rate. A pure Random Guessing model would explicitly instantiate the hypothesis that subjects are overwhelmed by the cardinal feature magnitudes and simply guess, perfectly capturing the ~0.5 metric values across all experiments without relying on a mismatched mechanism (like TTB with an artificially inflated threshold)."
}
```

## Usage

```json
{
  "prompt_token_count": 5421,
  "candidates_token_count": 318,
  "total_token_count": 6491
}
```
