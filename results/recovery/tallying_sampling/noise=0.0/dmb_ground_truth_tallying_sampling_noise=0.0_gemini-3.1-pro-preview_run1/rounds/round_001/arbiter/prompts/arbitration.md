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
People make decisions using a 'Tallying' or 'Equal Weight' heuristic. Instead of weighting features by their validity or importance, decision-makers simply count the total number of positive features for each option. They choose the option with the highest tally, effectively ignoring the provided cue validities. This explains why participants consistently choose options with more positive cues even when a single higher-validity cue points in the opposite direction.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: sum the positive features for each option, ignoring validities.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=15):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 9: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 11: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 12: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 13: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 14: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 15: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate Tallying (Equal Weight) from the Weighted Additive (WADD) rule, we use a 5-feature environment with a steep drop-off in cue validities: two highly valid cues (0.95, 0.90) and three lower-validity cues (0.60, 0.55, 0.50). Tallying ignores validities and simply counts the number of positive features. WADD weights each feature by its validity. By pitting an option with fewer, but highly valid cues (e.g., the top two cues) against an option with more, but lower-validity cues (e.g., the bottom three cues), we create strong dissociable trials. For example, an option with the top 2 cues (Tally = 2, WADD = 1.85) will be chosen by WADD over an option with the bottom 3 cues (Tally = 3, WADD = 1.65), whereas Tallying predicts the exact opposite. We include a mix of such dissociation trials, agreement trials (where the higher tally also has the higher weighted sum), and tally-tie trials (where Tallying is indifferent but WADD has a strict preference) to thoroughly map the strategy space.

**Computed schedule:** 15 unique pairs × 6 reps = 90 trials per subject.



### METRIC
Rationale:
To maximally discriminate Tallying from WADD, we isolate the specific dissociation trials where the two strategies prefer opposite options (e.g., one option has fewer but highly valid cues, while the other has more but less valid cues). The metric calculates the proportion of subject choices on these specific trials that align with the Tallying prediction. Tallying will yield a score close to 1.0 (subject to lapse noise), whereas WADD will yield a score close to 0.0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    
    tally_a = a_mat.sum(axis=1)
    tally_b = b_mat.sum(axis=1)
    
    wadd_a = a_mat @ validities
    wadd_b = b_mat @ validities
    
    tally_diff = tally_a - tally_b
    wadd_diff = wadd_a - wadd_b
    
    # Identify trials where Tallying and WADD strictly disagree on the preferred option
    disagree_mask = (tally_diff * wadd_diff) < 0
    
    if not np.any(disagree_mask):
        return 0.5
        
    responses = data['response'].values
    # Tally predicts 0 (Option A) if tally_a > tally_b, else 1 (Option B)
    tally_preds = (tally_b > tally_a).astype(int)
    
    agreements = (responses[disagree_mask] == tally_preds[disagree_mask])
    return float(np.mean(agreements))

### RESULTS
- Predicted under pi_3 (simulated): 0.8550 (var=0.0108)
- Predicted under pi_2 (simulated): 0.5300 (var=0.1022)
- Observed on real data: 0.8567 (var=0.0103)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=7):** [0.95, 0.9, 0.85, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) rule from Tallying (Equal Weight) without repeating prior 5-feature designs, we introduce a 7-feature environment with a gradual linear decline in validities. Tallying simply counts the number of positive features, completely ignoring their validities, whereas WADD computes a validity-weighted sum. We construct trials where an option with fewer but higher-validity features is pitted against an option with more but lower-validity features. By tuning the number of features on each side, we create combinations where WADD strongly prefers the option with fewer features, trials where WADD is indifferent but Tallying has a strict preference, and agreement trials where both models align. This 7-feature structure provides a wider dynamic range of sums to distinguish between the models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where option A has fewer positive features overall than option B, but option A possesses the two highest-validity features (specifically, Trial 1 and Trial 3 in the design). The Tallying (Equal Weight) model predicts that subjects will consistently choose option B because it has a higher simple sum of features. In contrast, the Weighted Additive (WADD) model predicts a preference for option A, because the high validities of the first two features more than compensate for the lower total count. By measuring the proportion of times option A is chosen in these specific trials, we expect a stark contrast: WADD will yield a value > 0.5, whereas Tallying will yield a value < 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_target(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # Select trials where A has fewer positive features than B,
        # but A has the top two most valid features.
        return sum(a) < sum(b) and sum(a[:2]) == 2
        
    mask = data.apply(is_target, axis=1)
    target_data = data[mask]
    if len(target_data) == 0:
        return 0.5
        
    # Return the proportion of times option A (response == 0) was chosen
    return float((target_data['response'] == 0).mean())


### RESULTS
- Predicted under pi_3 (simulated): 0.1708 (var=0.0211)
- Predicted under pi_2 (simulated): 0.5058 (var=0.0656)
- Observed on real data: 0.1333 (var=0.0206)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    valid_trials = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        # Features are pre-sorted by validity (0.95, 0.65, 0.60, 0.55)
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                if resp == 0:
                    matches += 1
                valid_trials += 1
                break
            elif val_b > val_a:
                if resp == 1:
                    matches += 1
                valid_trials += 1
                break
    return float(matches / valid_trials) if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.1825 (var=0.0124)
**Predicted under pi_3:** 0.1410 (var=0.0104)
**Predicted under pi_2:** 0.3519 (var=0.0370)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        choice = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                choice = 0
                break
            elif b[j] > a[j]:
                choice = 1
                break
        if choice == r:
            matches += 1
        total += 1
    return float(matches) / total if total > 0 else 0.0
```

**Observed (real) value:** 0.1233 (var=0.0089)
**Predicted under pi_3:** 0.1200 (var=0.0051)
**Predicted under pi_2:** 0.2406 (var=0.0128)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Tallying / Equal Weight) provides an exceptionally accurate fit to the observed data, whereas Theory 2 (Weighted Additive / WADD) consistently diverges from human behavior. The observed metrics tightly track the Tallying predictions (e.g., matching ~0.85 in Exp 1, ~0.13 in Exp 2, ~0.18 in Exp 3, and ~0.12 in Exp 4). The data strongly suggests that participants are ignoring cue validities and simply counting the number of positive features. Because Tallying is overwhelmingly supported and WADD is consistently failing, WADD should be replaced with a different classic heuristic to serve as a stronger structural competitor to Tallying.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose the 'Take-The-Best' (TTB) heuristic as a brand-new theory to replace WADD. TTB posits a non-compensatory, one-reason decision making process: decision-makers search through cues in descending order of validity and stop at the very first cue that discriminates between the two options, basing their choice entirely on that single cue. Introducing TTB will provide a theoretically distinct, lexicographic benchmark to contrast against the equal-weight compensatory mechanism of Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 5624,
  "candidates_token_count": 299,
  "total_token_count": 6813
}
```
