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
People employ a probabilistic 'Take The Best' (TTB) heuristic to make binary choices. They order features by their subjective validity, which is informed by objective validities but subject to random fluctuations (noise). They compare the options sequentially based on this subjective ordering. The first feature that discriminates between the two options determines the choice, and all subsequent features are ignored. This introduces stochasticity into the cue hierarchy, explaining why choices sometimes deviate from strict reliance on the single highest-validity cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    
    # Add Gaussian noise to validities to model subjective fluctuations in cue hierarchy
    noisy_validities = validities + np.random.normal(0, sigma, size=validities.shape)
    
    # Order features by noisy validity in descending order
    order = np.argsort(noisy_validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Iterate through features in order of validity
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    # If no features discriminate, both options are tied
    if scores[0] == 0.0 and scores[1] == 0.0:
        scores = np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
**Validities (n_features=6):** [0.9, 0.88, 0.86, 0.6, 0.58, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) while distinguishing from previously explored designs, this experiment uses a 6-feature environment with a distinct cluster of high validities and a cluster of low validities. By making the top three validities very close (0.90, 0.88, 0.86), TTB's subjective cue hierarchy (which includes noise) will frequently swap their order, whereas WADD integrates them stably. We design trials where the highest-validity feature points to one option, but a combination of the slightly-lower high-validity features and multiple low-validity features points to the other. This setup not only tests compensatory vs. non-compensatory choice but also probes the stochastic nature of TTB's cue hierarchy.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In this 6-feature experimental design, Trials 2, 4, 5, 6, and 7 all pit an option with exactly 2 positive features against an option with exactly 4 positive features. Crucially, the 2-feature option always consists of two of the top 3 highest-validity features, while the 4-feature option consists of the remaining one top-tier feature plus all three of the bottom-tier features. The Take The Best (TTB) heuristic, driven by a noisy cue hierarchy, will choose the 2-feature option whenever either of its two top-tier features happens to be ranked first (approximately 2/3 of the time). In stark contrast, the Weighted Additive (WADD) model integrates all features; because the sum of validities for the 4-feature option (~2.6) substantially outweighs the sum for the 2-feature option (~1.7), WADD will systematically prefer the 4-feature option. Measuring the proportion of times the subject chooses the 2-feature option in these specific trials creates a powerful, low-variance discriminator: it will be high (~0.66) for TTB and very low (~0.0) for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    two_feature_chosen = []
    for subj, subj_df in data.groupby('subject_id'):
        subj_vals = []
        for _, row in subj_df.iterrows():
            a_ratings = row['option_a_ratings']
            b_ratings = row['option_b_ratings']
            resp = row['response']
            
            sum_a = sum(a_ratings)
            sum_b = sum(b_ratings)
            
            # Focus on trials where one option has exactly 2 features and the other has 4
            if sum_a == 2 and sum_b == 4:
                subj_vals.append(1.0 if resp == 0 else 0.0)
            elif sum_b == 2 and sum_a == 4:
                subj_vals.append(1.0 if resp == 1 else 0.0)
                
        if subj_vals:
            two_feature_chosen.append(np.mean(subj_vals))
            
    return float(np.mean(two_feature_chosen)) if two_feature_chosen else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.4190 (var=0.0068)
- Predicted under pi_2 (simulated): 0.3123 (var=0.0287)
- Observed on real data: 0.4913 (var=0.0041)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic by pitting a single highly valid feature against multiple moderately valid features. We use five features where the highest validity is 0.95 and the others range from 0.80 to 0.65. In critical trials, Option B excels on the single most predictive feature, while Option A is supported by a combination of the remaining features. Because the sum of the lower validities far exceeds the highest validity, WADD strongly prefers Option A. In contrast, TTB (even with validity noise) will primarily rely on the highest-validity feature and thus prefer Option B. By varying which subset of lower-validity features Option A possesses, we can map out the precise compensatory threshold predicted by WADD but absent in TTB.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses Option B in trials where Option B has the highest-validity feature (feature 0) and Option A lacks it, but Option A has a compensatory advantage in the number of lower-validity features (sum >= 3). The Take The Best (TTB) heuristic will almost always choose Option B because it relies solely on the first discriminating feature (feature 0). In contrast, the Weighted Additive (WADD) model integrates all features, allowing Option A's multiple lower-validity features to compensate for its deficit on the highest-validity feature, leading to a much lower rate of Option B choices.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f0 = data['option_b_ratings'].apply(lambda x: x[0])
    a_sum = data['option_a_ratings'].apply(lambda x: sum(x))
    
    mask = (a_f0 == 0) & (b_f0 == 1) & (a_sum >= 3)
    
    if mask.sum() == 0:
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.4308 (var=0.0146)
- Predicted under pi_2 (simulated): 0.3183 (var=0.0230)
- Observed on real data: 0.5200 (var=0.0061)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    match_count = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
        if row['response'] == tally_pref:
            match_count += 1
        total += 1
    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.4832 (var=0.0024)
**Predicted under pi_3:** 0.5784 (var=0.0045)
**Predicted under pi_2:** 0.6618 (var=0.0121)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A has top 2 features, B has bottom 3
    t1 = (a_str == '11000') & (b_str == '00111')
    # Trial 2: A has bottom 3 features, B has top 2
    t2 = (a_str == '00111') & (b_str == '11000')
    
    critical = t1 | t2
    if not critical.any():
        return 0.5
        
    # WADD prefers the option with the top 2 features (A in t1, B in t2)
    # Tallying prefers the option with the bottom 3 features (since 3 > 2)
    wadd_choices = (t1 & (data['response'] == 0)) | (t2 & (data['response'] == 1))
    return float(wadd_choices.sum() / critical.sum())
```

**Observed (real) value:** 0.4750 (var=0.0061)
**Predicted under pi_3:** 0.4975 (var=0.0116)
**Predicted under pi_2:** 0.4917 (var=0.0953)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metric values consistently hover around 0.5 (chance level). For instance, in Experiment 1, subjects choose the 2-feature option 49% of the time, and in Experiment 2, they choose the single-best-feature option 52% of the time. Neither pi_3 (Take The Best) nor pi_2 (Weighted Additive) accurately captures this near-random behavior, as both expect subjects to strongly leverage the provided validities (leading to predictions further from 0.5). pi_2 is particularly far off, predicting ~0.31 when the observed is ~0.50. The data strongly suggests that subjects are either choosing randomly or employing a heuristic that ignores the provided validities entirely, such as a presentation-order heuristic (e.g., scanning features from left to right and picking the first one that discriminates).",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Take The First' (or Left-to-Right Lexicographic) theory. This theory posits that subjects ignore the numerically stated validities because of cognitive overload. Instead, they scan the features from left to right (based on presentation order) and choose the option that has a '1' on the first feature that discriminates between the two options. This theory will naturally result in ~50% aggregate preferences in designs where the discriminating features are uniformly distributed across the array, perfectly capturing the ~0.5 observed metrics across all experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 5849,
  "candidates_token_count": 340,
  "total_token_count": 8159
}
```
